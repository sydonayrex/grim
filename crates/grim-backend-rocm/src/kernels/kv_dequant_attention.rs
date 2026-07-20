//! Fused KV-dequantized attention HIP kernel (WI-R5).
//!
//! Extends standard QKV-attention to support on-the-fly dequantization of Key and Value
//! caches from 4-bit or 8-bit representations, minimizing LDS traffic and high-latency DRAM reads.

pub const KERNEL_SOURCE: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void grim_kv_dequant_attention(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_tensor,
    const float* __restrict__ k_scales,
    const unsigned char* __restrict__ v_tensor,
    const float* __restrict__ v_scales,
    float* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_seq_len,
    int cache_offset,
    float inv_sqrt_d,
    int quant_bits
) {
    // grid_x enumerates seq_len * num_heads (one block per (seq, head)),
    // grid_y == 1. Derive both indices from blockIdx.x.
    const int flat = blockIdx.x;
    const int i = flat / num_heads; // query position (0..seq_len)
    const int h = flat % num_heads; // head index
    if (i >= seq_len || h >= num_heads) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = (i * num_heads + h) * head_dim;
    const int abs_i = cache_offset + i;

    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    const int num_waves = blockDim.x / wave_size;

    const int d = lane_id;
    const bool thread_active = d < head_dim;

    if (head_dim > 256) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            int dd = lane_id + chunk * wave_size;
            if (dd < head_dim) out[q_offset + dd] = nanf("");
        }
        return;
    }

    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][256];

    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int range_len = hi;
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    for (int j = j_start; j < j_end; ++j) {
        float score = 0.0f;
        if (quant_bits == 8) {
            const int k_row_offset = (j * num_kv_heads + kv_head) * head_dim;
            const float scale = k_scales[j * num_kv_heads + kv_head];
            for (int dim = 0; dim < 256; ++dim) {
                if (dim < head_dim) {
                    float k_val = (((float)((int)k_tensor[k_row_offset + dim]) - 128.0f) / 127.0f) * scale;
                    score += q[q_offset + dim] * k_val;
                }
            }
        } else {
            const int k_row_offset = ((j * num_kv_heads + kv_head) * head_dim) / 2;
            const float scale = k_scales[j * num_kv_heads + kv_head];
            for (int dim = 0; dim < 256; ++dim) {
                if (dim < head_dim) {
                    unsigned char byte = k_tensor[k_row_offset + dim / 2];
                    float nib = (float)((dim % 2 == 0) ? (byte & 0xF) : (byte >> 4));
                    float k_val = (nib - 8.0f) / 7.0f * scale;
                    score += q[q_offset + dim] * k_val;
                }
            }
        }
        score *= inv_sqrt_d;

        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);
        running_sum = running_sum * scale_old + scale_new;

        if (quant_bits == 8) {
            const int v_row_offset = (j * num_kv_heads + kv_head) * head_dim;
            const float scale = v_scales[j * num_kv_heads + kv_head];
            for (int chunk = 0; chunk < 4; ++chunk) {
                int dd = lane_id + chunk * wave_size;
                if (dd < head_dim) {
                    float v_val = (((float)((int)v_tensor[v_row_offset + dd]) - 128.0f) / 127.0f) * scale;
                    out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_val;
                }
            }
        } else {
            const int v_row_offset = ((j * num_kv_heads + kv_head) * head_dim) / 2;
            const float scale = v_scales[j * num_kv_heads + kv_head];
            for (int chunk = 0; chunk < 4; ++chunk) {
                int dd = lane_id + chunk * wave_size;
                if (dd < head_dim) {
                    unsigned char byte = v_tensor[v_row_offset + dd / 2];
                    float nib = (float)((dd % 2 == 0) ? (byte & 0xF) : (byte >> 4));
                    float v_val = (nib - 8.0f) / 7.0f * scale;
                    out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_val;
                }
            }
        }
    }

    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 4; ++chunk) {
        int dd = lane_id + chunk * wave_size;
        if (dd < head_dim) {
            s_acc[wave_id][dd] = out_acc[chunk];
        } else if (dd < 256) {
            s_acc[wave_id][dd] = 0.0f;
        }
    }
    __syncthreads();

    if (wave_id != 0) return;

    float m_final = s_max[0];
    float sum_final = s_sum[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) {
        if (w >= num_waves) break;
        const float mw = s_max[w];
        const float uw = s_sum[w];
        const float m_new = fmaxf(m_final, mw);
        const float scale_a = expf(m_final - m_new);
        const float scale_b = expf(mw - m_new);
        sum_final = sum_final * scale_a + uw * scale_b;
        m_final = m_new;
    }

    for (int chunk = 0; chunk < 4; ++chunk) {
        int dd = lane_id + chunk * wave_size;
        if (dd < head_dim) {
            float acc_final = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w >= num_waves) break;
                acc_final += s_acc[w][dd] * expf(s_max[w] - m_final);
            }
            const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;
            out[q_offset + dd] = acc_final * inv_sum;
        }
    }
}
"#;
