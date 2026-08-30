//! Fused KV-dequantized attention CUDA kernel.
//!
//! Ported from grim-backend-rocm `kernels/kv_dequant_attention.rs` (WRECK-5).
//!
//! Supports three quantization formats for KV cache:
//!   0 = FP16 (row-major half, row_bytes = head_dim * 2)
//!   1 = Q8_0 (block: 2-byte fp16 delta + 32 int8 codes = 34 bytes per 32 elements)
//!   2 = Q4_K (super-block: 144 bytes per 256 elements — d/dmin/scales/nibbles)
//!
//! Legacy quant_bits paths (4=nibble, 8=int8) preserved for ABI stability.
//! warpSize → 32 for CUDA (no wavefront assumption).

pub const KV_DEQUANT_ATTENTION_SOURCE: &str = r#"
#include <math.h>

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
    int quant_bits,
    int quant_format
) {
    const int flat     = blockIdx.x;
    const int i        = flat / num_heads;
    const int h        = flat % num_heads;
    if (i >= seq_len || h >= num_heads) return;

    const int q_per_kv  = num_heads / num_kv_heads;
    const int kv_head   = h / q_per_kv;
    const int q_offset  = (i * num_heads + h) * head_dim;
    const int abs_i     = cache_offset + i;

    const int tid       = threadIdx.x;
    const int warp_size = 32;  // CUDA warp is always 32
    const int wave_id   = tid / warp_size;
    const int lane_id   = tid % warp_size;
    const int num_waves = blockDim.x / warp_size;

    if (head_dim > 256) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            int dd = lane_id + chunk * warp_size;
            if (dd < head_dim) out[q_offset + dd] = __int_as_float(0x7FC00000); // NaN
        }
        return;
    }

    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][256];

    const int row_bytes_fp16 = head_dim * 2;
    const int row_bytes_q8_0 = ((head_dim + 31) / 32) * 34;
    const int row_bytes_q4k  = ((head_dim + 255) / 256) * 144;

    const int hi       = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int base_j   = hi / num_waves;
    const int rem_j    = hi % num_waves;
    int j_start = wave_id * base_j + (wave_id < rem_j ? wave_id : rem_j);
    int j_end   = j_start + base_j + (wave_id < rem_j ? 1 : 0);

    // Inline fp16→float (software, no cuda_fp16.h required in all JIT targets)
    #define GRIM_FP16_TO_F32(bits_val, out_var) do { \
        unsigned short _b = (unsigned short)(bits_val); \
        unsigned _e = (_b >> 10) & 0x1F, _m = _b & 0x3FF; \
        if (_e == 0)  { out_var = (_m == 0) ? 0.0f : ((float)_m * 5.9604644775390625e-8f); } \
        else if (_e == 31) { out_var = ((_b >> 15) & 1) ? -1e30f : 1e30f; } \
        else { float _s = ((_b >> 15) & 1) ? -1.0f : 1.0f; \
               out_var = _s * (1.0f + (float)_m / 1024.0f) * powf(2.0f, (float)(_e - 15)); } \
    } while (0)

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f, running_sum = 0.0f;

    for (int j = j_start; j < j_end; ++j) {
        float score = 0.0f;

        if (quant_format == 0) {
            // FP16 K
            const unsigned char* kr = k_tensor + (j * num_kv_heads + kv_head) * row_bytes_fp16;
            for (int dim = 0; dim < head_dim; ++dim) {
                float kv; unsigned short kb = ((const unsigned short*)kr)[dim];
                GRIM_FP16_TO_F32(kb, kv);
                score += q[q_offset + dim] * kv;
            }
        } else if (quant_format == 1) {
            // Q8_0 K: 34-byte blocks (2-byte fp16 delta + 32 int8)
            const unsigned char* kr = k_tensor + (j * num_kv_heads + kv_head) * row_bytes_q8_0;
            for (int dim = 0; dim < head_dim; ++dim) {
                const unsigned char* blk = kr + (dim / 32) * 34;
                float delta; GRIM_FP16_TO_F32(((const unsigned short*)blk)[0], delta);
                float kv = delta * (float)((const signed char*)(blk + 2))[dim % 32];
                score += q[q_offset + dim] * kv;
            }
        } else if (quant_format == 2) {
            // Q4_K super-block K
            const unsigned char* kr = k_tensor + (j * num_kv_heads + kv_head) * row_bytes_q4k;
            for (int dim = 0; dim < head_dim; ++dim) {
                const unsigned char* sb = kr + (dim / 256) * 144;
                float d_val, dmin_val;
                GRIM_FP16_TO_F32(((const unsigned short*)sb)[0], d_val);
                GRIM_FP16_TO_F32(((const unsigned short*)sb)[1], dmin_val);
                const unsigned char* scales = sb + 4;
                const unsigned char* qs     = sb + 16;
                int elem = dim % 256;
                int k4   = elem / 64, off = elem % 64;
                int s    = 2 * k4 + (off >= 32 ? 1 : 0), jj = off & 31;
                unsigned char sc, m;
                if (s < 4) { sc = scales[s] & 63; m = scales[s + 4] & 63; }
                else { sc = (scales[s+4]&0xF)|((scales[s-4]>>6)<<4); m = (scales[s+4]>>4)|((scales[s]>>6)<<4); }
                int qb = 32 * k4 + jj;
                unsigned char nib = (off < 32) ? (qs[qb] & 0xF) : (qs[qb] >> 4);
                score += q[q_offset + dim] * (d_val * (float)sc * (float)nib - dmin_val * (float)m);
            }
        } else if (quant_bits == 8) {
            const int kr = (j * num_kv_heads + kv_head) * head_dim;
            float sc = k_scales[j * num_kv_heads + kv_head];
            for (int dim = 0; dim < head_dim; ++dim)
                score += q[q_offset + dim] * ((((float)(int)k_tensor[kr + dim] - 128.0f) / 127.0f) * sc);
        } else {
            // Legacy nibble (quant_bits == 4)
            const int kr = ((j * num_kv_heads + kv_head) * head_dim) / 2;
            float sc = k_scales[j * num_kv_heads + kv_head];
            for (int dim = 0; dim < head_dim; ++dim) {
                unsigned char byte = k_tensor[kr + dim / 2];
                float nib = (float)((dim % 2 == 0) ? (byte & 0xF) : (byte >> 4));
                score += q[q_offset + dim] * ((nib - 8.0f) / 7.0f * sc);
            }
        }
        score *= inv_sqrt_d;

        const float old_max  = running_max;
        running_max          = fmaxf(running_max, score);
        const float sc_old   = expf(old_max - running_max);
        const float sc_new   = expf(score   - running_max);
        running_sum          = running_sum * sc_old + sc_new;

        // V accumulate (same quant_format dispatch)
        for (int chunk = 0; chunk < 4; ++chunk) {
            int dd = lane_id + chunk * warp_size;
            if (dd >= head_dim) continue;
            float v_val = 0.0f;
            if (quant_format == 0) {
                const unsigned char* vr = v_tensor + (j * num_kv_heads + kv_head) * row_bytes_fp16;
                GRIM_FP16_TO_F32(((const unsigned short*)vr)[dd], v_val);
            } else if (quant_format == 1) {
                const unsigned char* vr = v_tensor + (j * num_kv_heads + kv_head) * row_bytes_q8_0;
                const unsigned char* blk = vr + (dd / 32) * 34;
                float delta; GRIM_FP16_TO_F32(((const unsigned short*)blk)[0], delta);
                v_val = delta * (float)((const signed char*)(blk + 2))[dd % 32];
            } else if (quant_format == 2) {
                const unsigned char* vr = v_tensor + (j * num_kv_heads + kv_head) * row_bytes_q4k;
                const unsigned char* sb = vr + (dd / 256) * 144;
                float d_val, dmin_val;
                GRIM_FP16_TO_F32(((const unsigned short*)sb)[0], d_val);
                GRIM_FP16_TO_F32(((const unsigned short*)sb)[1], dmin_val);
                const unsigned char* scales = sb + 4;
                const unsigned char* qs     = sb + 16;
                int elem = dd % 256, k4 = elem / 64, off = elem % 64;
                int s = 2 * k4 + (off >= 32 ? 1 : 0), jj = off & 31;
                unsigned char sc, m;
                if (s < 4) { sc = scales[s] & 63; m = scales[s + 4] & 63; }
                else { sc = (scales[s+4]&0xF)|((scales[s-4]>>6)<<4); m = (scales[s+4]>>4)|((scales[s]>>6)<<4); }
                int qb = 32 * k4 + jj;
                unsigned char nib = (off < 32) ? (qs[qb] & 0xF) : (qs[qb] >> 4);
                v_val = d_val * (float)sc * (float)nib - dmin_val * (float)m;
            } else if (quant_bits == 8) {
                const int vr = (j * num_kv_heads + kv_head) * head_dim;
                float sc = v_scales[j * num_kv_heads + kv_head];
                v_val = ((((float)(int)v_tensor[vr + dd] - 128.0f) / 127.0f)) * sc;
            } else {
                const int vr = ((j * num_kv_heads + kv_head) * head_dim) / 2;
                float sc = v_scales[j * num_kv_heads + kv_head];
                unsigned char byte = v_tensor[vr + dd / 2];
                float nib = (float)((dd % 2 == 0) ? (byte & 0xF) : (byte >> 4));
                v_val = (nib - 8.0f) / 7.0f * sc;
            }
            out_acc[chunk] = out_acc[chunk] * sc_old + sc_new * v_val;
        }
    }

    if (lane_id == 0) { s_max[wave_id] = running_max; s_sum[wave_id] = running_sum; }
    for (int chunk = 0; chunk < 4; ++chunk) {
        int dd = lane_id + chunk * warp_size;
        s_acc[wave_id][dd < 256 ? dd : 0] = (dd < head_dim) ? out_acc[chunk] : 0.0f;
    }
    __syncthreads();
    if (wave_id != 0) return;

    float m_final = s_max[0], sum_final = s_sum[0];
    for (int w = 1; w < num_waves && w < 8; ++w) {
        float mw = s_max[w], uw = s_sum[w];
        float mn = fmaxf(m_final, mw);
        sum_final = sum_final * expf(m_final - mn) + uw * expf(mw - mn);
        m_final = mn;
    }
    for (int chunk = 0; chunk < 4; ++chunk) {
        int dd = lane_id + chunk * warp_size;
        if (dd < head_dim) {
            float acc = 0.0f;
            for (int w = 0; w < num_waves && w < 8; ++w)
                acc += s_acc[w][dd] * expf(s_max[w] - m_final);
            out[q_offset + dd] = (sum_final > 0.0f) ? acc / sum_final : 0.0f;
        }
    }
    #undef GRIM_FP16_TO_F32
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_quant_format_and_q8_path() {
        assert!(KV_DEQUANT_ATTENTION_SOURCE.contains("int quant_format"));
        assert!(KV_DEQUANT_ATTENTION_SOURCE.contains("Q8_0 K"));
        assert!(KV_DEQUANT_ATTENTION_SOURCE.contains("Q4_K super-block K"));
        assert!(KV_DEQUANT_ATTENTION_SOURCE.contains("grim_kv_dequant_attention"));
    }
}
