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
    const int i = blockIdx.x; // query position (0..seq_len)
    const int h = blockIdx.y; // head index
    if (i >= seq_len || h >= num_heads) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = (i * num_heads + h) * head_dim;

    // Local registers for accumulation and softmax tracking
    extern __shared__ float s_mem[];
    float* s_acc = s_mem; // Dynamic LDS allocation
    
    // Wavefront index setup
    const int tid = threadIdx.x;
    const int wave_id = tid / 64;
    const int lane_id = tid % 64;

    // Initialize online softmax registers
    float r_max = -1e20f;
    float r_sum = 0.0f;
    
    // Allocate local accumulators for the head dimension owned by this thread
    // Assuming max head_dim = 128
    float r_acc[128] = {0.0f};

    // Parallelize the KV sequence loop across the 4 wavefronts
    const int step = 256;
    for (int j = tid; j < kv_seq_len; j += step) {
        // Causal masking
        if (j > cache_offset + i) continue;

        // Compute dot product q * K_j
        float score = 0.0f;
        
        // Dequantize and compute dot product
        if (quant_bits == 8) {
            const int k_row_offset = (j * num_kv_heads + kv_head) * head_dim;
            const float scale = k_scales[j * num_kv_heads + kv_head];
            for (int d = 0; d < head_dim; ++d) {
                float k_val = (float)k_tensor[k_row_offset + d] * scale;
                score += q[q_offset + d] * k_val;
            }
        } else { // 4-bit
            const int k_row_offset = ((j * num_kv_heads + kv_head) * head_dim) / 2;
            const float scale = k_scales[j * num_kv_heads + kv_head];
            for (int d = 0; d < head_dim; ++d) {
                unsigned char byte = k_tensor[k_row_offset + d / 2];
                float k_val = (float)((d % 2 == 0) ? (byte & 0xF) : (byte >> 4)) * scale;
                score += q[q_offset + d] * k_val;
            }
        }

        score *= inv_sqrt_d;

        // Softmax update
        float old_max = r_max;
        if (score > r_max) {
            r_max = score;
            float scale = __expf(old_max - r_max);
            r_sum = r_sum * scale + 1.0f;
            for (int d = 0; d < head_dim; ++d) {
                r_acc[d] *= scale;
            }
        } else {
            r_sum += __expf(score - r_max);
        }

        float exp_score = __expf(score - r_max);
        
        // Accumulate V_j
        if (quant_bits == 8) {
            const int v_row_offset = (j * num_kv_heads + kv_head) * head_dim;
            const float scale = v_scales[j * num_kv_heads + kv_head];
            for (int d = 0; d < head_dim; ++d) {
                float v_val = (float)v_tensor[v_row_offset + d] * scale;
                r_acc[d] += exp_score * v_val;
            }
        } else { // 4-bit
            const int v_row_offset = ((j * num_kv_heads + kv_head) * head_dim) / 2;
            const float scale = v_scales[j * num_kv_heads + kv_head];
            for (int d = 0; d < head_dim; ++d) {
                unsigned char byte = v_tensor[v_row_offset + d / 2];
                float v_val = (float)((d % 2 == 0) ? (byte & 0xF) : (byte >> 4)) * scale;
                r_acc[d] += exp_score * v_val;
            }
        }
    }

    // Accumulate results from different threads using LDS
    // In this basic version, we serialize block reduction through shared memory
    for (int d = 0; d < head_dim; ++d) {
        s_acc[tid] = r_acc[d];
        __syncthreads();

        // Tree reduction within the block
        for (int offset = 128; offset > 0; offset /= 2) {
            if (tid < offset) {
                s_acc[tid] += s_acc[tid + offset];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[(i * num_heads + h) * head_dim + d] = s_acc[0] / (r_sum + 1e-6f);
        }
        __syncthreads();
    }
}
"#;
