//! Fused RMSNorm + MXFP4 GEMM + RoPE and Tiled MXFP4 GEMM HIP kernels for ROCm.
//!
//! Implements high-throughput Microscaling FP4 (OCP MXFP4: 4-bit E2M1 codes +
//! 8-bit E8M0 shared exponents per 32-element micro-block) matrix multiplication:
//!
//! 1. `grim_fused_rmsnorm_mxfp4_gemm_rope_kv`: End-to-end fused attention projection
//!    (RMSNorm -> MXFP4 GEMM -> RoPE -> direct VRAM write / KV-cache update).
//! 2. `grim_fused_rmsnorm_mxfp4_gemm`: Fused MLP projection (RMSNorm -> MXFP4 GEMM).
//! 3. `grim_mxfp4_gemm_tiled`: Tiled 2D batched GEMM for standalone MXFP4 linear layers.
//! 4. `grim_mxfp4_backward_gemm`: Backward GEMM (dX = dY @ B^T) dequantizing B on-the-fly.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// Fast lookup table for 4-bit E2M1 code to base float value
__device__ __constant__ float MXFP4_E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float mxfp4_decode_fast(unsigned char code, unsigned char shared_exp) {
    float base = MXFP4_E2M1_LUT[code & 0x0F];
    // scale = 2^(shared_exp - 127). ldexpf is exact and hipRTC-portable
    // (`__exp2f` is not declared by hiprtc_runtime.h on gfx12 targets).
    return base * ldexpf(1.0f, (int)shared_exp - 127);
}

// ---------------------------------------------------------------------------
// 1. Fused RMSNorm + MXFP4 QKV GEMM + RoPE + Direct KV Cache Scatter
// ---------------------------------------------------------------------------
//
// Fuses:
//   1. In-LDS RMSNorm: x_norm = (x / rms(x)) * gamma
//   2. On-the-fly MXFP4 GEMM for Q, K, V projections
//   3. In-register Rotary Position Embedding (RoPE) for Q and K
//   4. Direct scatter to VRAM Q buffer and persistent VRAM KV cache (K, V)
//
// Grid: (M, (N_q + N_k + N_v + 31) / 32)
// Block: (32, 1) or (64, 1) or (256, 1)
// ---------------------------------------------------------------------------
__global__ void grim_fused_rmsnorm_mxfp4_gemm_rope_kv(
    const float* __restrict__ x,                // [M, K]
    const float* __restrict__ gamma,            // [K]
    const unsigned char* __restrict__ w_codes,  // Packed 4-bit codes for [N_total, K]
    const unsigned char* __restrict__ w_exps,   // E8M0 exponents for [N_total, K/32]
    float* __restrict__ q_out,                  // [M, num_q_heads, head_dim]
    float* __restrict__ k_cache,                // [max_seq, num_kv_heads, head_dim] (optional)
    float* __restrict__ v_cache,                // [max_seq, num_kv_heads, head_dim] (optional)
    float* __restrict__ out_all,                // [M, N_total] fallback if q/k/v split not used
    const unsigned int* __restrict__ positions, // [M] sequence positions for RoPE
    int M,
    int K,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float rope_theta,
    const float* __restrict__ inv_freq,  // optional YaRN inv_freq[rotary_half]; null => plain theta
    float mscale,                         // YaRN attention_factor applied to sin/cos; 1.0 = none
    float eps,
    int max_seq_len
) {
    const int row = blockIdx.x; // token index in batch/seq (0..M-1)
    const int col = blockIdx.y * blockDim.x + threadIdx.x; // output feature index
    const int N_q = num_q_heads * head_dim;
    const int N_k = num_kv_heads * head_dim;
    const int N_v = num_kv_heads * head_dim;
    const int N_total = N_q + N_k + N_v;

    if (row >= M) return;

    // Phase 1: Compute RMS for row in shared memory LDS
    // Each thread in the block computes a partial sum of squares
    extern __shared__ float s_mem[];
    float* s_sum = s_mem; // [blockDim.x]

    float local_sq = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = x[row * K + k];
        local_sq += val * val;
    }
    s_sum[threadIdx.x] = local_sq;
    __syncthreads();

    // Reduction across block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float rms = rsqrtf(s_sum[0] / (float)K + eps);
    __syncthreads();

    if (col >= N_total) return;

    // Phase 2: MXFP4 GEMM dot product: col-th row of W dot (x * rms * gamma)
    float acc = 0.0f;
    const int exps_per_row = K / 32;
    const int row_exp_offset = col * exps_per_row;
    const int row_codes_offset = col * (K / 2);

    for (int block_k = 0; block_k < exps_per_row; ++block_k) {
        unsigned char exp_val = w_exps[row_exp_offset + block_k];
        int k_base = block_k * 32;
        int code_byte_base = row_codes_offset + block_k * 16;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned char packed_byte = w_codes[code_byte_base + i];
            unsigned char c0 = packed_byte & 0x0F;
            unsigned char c1 = (packed_byte >> 4) & 0x0F;

            int k0 = k_base + i * 2;
            int k1 = k0 + 1;

            float x0_norm = x[row * K + k0] * rms * gamma[k0];
            float x1_norm = x[row * K + k1] * rms * gamma[k1];

            float w0 = mxfp4_decode_fast(c0, exp_val);
            float w1 = mxfp4_decode_fast(c1, exp_val);

            acc += x0_norm * w0 + x1_norm * w1;
        }
    }

    // Phase 3 & 4: RoPE rotation and direct scatter to Q / KV Cache
    if (out_all != nullptr) {
        out_all[row * N_total + col] = acc;
    }

    unsigned int pos = positions ? positions[row] : (unsigned int)row;

    if (col < N_q) {
        // Query projection
        int h = col / head_dim;
        int d = col % head_dim;

        float q_val = acc;
        if (d < rotary_dim) {
            int pair_idx = d / 2;
            int is_odd = d % 2;
            float freq = (inv_freq != nullptr)
                ? inv_freq[pair_idx]
                : 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle) * mscale;
            float sin_a = sinf(angle) * mscale;

            // Reconstruct pair using neighbour element
            float partner_acc = 0.0f;
            int partner_col = is_odd ? (col - 1) : (col + 1);
            int partner_row_exp = partner_col * exps_per_row;
            int partner_row_codes = partner_col * (K / 2);

            for (int bk = 0; bk < exps_per_row; ++bk) {
                unsigned char e_val = w_exps[partner_row_exp + bk];
                int kb = bk * 32;
                int cb_base = partner_row_codes + bk * 16;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    unsigned char p_byte = w_codes[cb_base + i];
                    float w0 = mxfp4_decode_fast(p_byte & 0x0F, e_val);
                    float w1 = mxfp4_decode_fast((p_byte >> 4) & 0x0F, e_val);
                    int k0 = kb + i * 2;
                    float x0_norm = x[row * K + k0] * rms * gamma[k0];
                    float x1_norm = x[row * K + (k0 + 1)] * rms * gamma[k0 + 1];
                    partner_acc += x0_norm * w0 + x1_norm * w1;
                }
            }

            if (is_odd) {
                q_val = partner_acc * sin_a + acc * cos_a;
            } else {
                q_val = acc * cos_a - partner_acc * sin_a;
            }
        }
        if (q_out != nullptr) {
            q_out[row * N_q + col] = q_val;
        }
    } else if (col < N_q + N_k) {
        // Key projection
        int k_col = col - N_q;
        int h = k_col / head_dim;
        int d = k_col % head_dim;

        float k_val = acc;
        if (d < rotary_dim) {
            int pair_idx = d / 2;
            int is_odd = d % 2;
            float freq = (inv_freq != nullptr)
                ? inv_freq[pair_idx]
                : 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle) * mscale;
            float sin_a = sinf(angle) * mscale;

            float partner_acc = 0.0f;
            int partner_col = is_odd ? (col - 1) : (col + 1);
            int partner_row_exp = partner_col * exps_per_row;
            int partner_row_codes = partner_col * (K / 2);

            for (int bk = 0; bk < exps_per_row; ++bk) {
                unsigned char e_val = w_exps[partner_row_exp + bk];
                int kb = bk * 32;
                int cb_base = partner_row_codes + bk * 16;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    unsigned char p_byte = w_codes[cb_base + i];
                    float w0 = mxfp4_decode_fast(p_byte & 0x0F, e_val);
                    float w1 = mxfp4_decode_fast((p_byte >> 4) & 0x0F, e_val);
                    int k0 = kb + i * 2;
                    float x0_norm = x[row * K + k0] * rms * gamma[k0];
                    float x1_norm = x[row * K + (k0 + 1)] * rms * gamma[k0 + 1];
                    partner_acc += x0_norm * w0 + x1_norm * w1;
                }
            }

            if (is_odd) {
                k_val = partner_acc * sin_a + acc * cos_a;
            } else {
                k_val = acc * cos_a - partner_acc * sin_a;
            }
        }
        if (k_cache != nullptr && (int)pos < max_seq_len) {
            k_cache[pos * N_k + k_col] = k_val;
        }
    } else {
        // Value projection (raw, unrotated)
        int v_col = col - N_q - N_k;
        if (v_cache != nullptr && (int)pos < max_seq_len) {
            v_cache[pos * N_v + v_col] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Fused RMSNorm + MXFP4 GEMM (e.g. for MLP gate/up/down projections)
// ---------------------------------------------------------------------------
__global__ void grim_fused_rmsnorm_mxfp4_gemm(
    const float* __restrict__ x,                // [M, K]
    const float* __restrict__ gamma,            // [K]
    const unsigned char* __restrict__ w_codes,  // Packed 4-bit codes for [N, K]
    const unsigned char* __restrict__ w_exps,   // E8M0 exponents for [N, K/32]
    float* __restrict__ out,                    // [M, N]
    int M,
    int N,
    int K,
    float eps
) {
    const int row = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= M) return;

    // Shared memory reduction for RMSNorm
    extern __shared__ float s_mem_mlp[];
    float* s_sum = s_mem_mlp;

    float local_sq = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = x[row * K + k];
        local_sq += val * val;
    }
    s_sum[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float rms = rsqrtf(s_sum[0] / (float)K + eps);
    __syncthreads();

    if (col >= N) return;

    float acc = 0.0f;
    const int exps_per_row = K / 32;
    const int row_exp_offset = col * exps_per_row;
    const int row_codes_offset = col * (K / 2);

    for (int block_k = 0; block_k < exps_per_row; ++block_k) {
        unsigned char exp_val = w_exps[row_exp_offset + block_k];
        int k_base = block_k * 32;
        int code_byte_base = row_codes_offset + block_k * 16;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned char packed_byte = w_codes[code_byte_base + i];
            unsigned char c0 = packed_byte & 0x0F;
            unsigned char c1 = (packed_byte >> 4) & 0x0F;

            int k0 = k_base + i * 2;
            int k1 = k0 + 1;

            float x0_norm = x[row * K + k0] * rms * gamma[k0];
            float x1_norm = x[row * K + k1] * rms * gamma[k1];

            float w0 = mxfp4_decode_fast(c0, exp_val);
            float w1 = mxfp4_decode_fast(c1, exp_val);

            acc += x0_norm * w0 + x1_norm * w1;
        }
    }

    out[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 3. Tiled 2D MXFP4 GEMM (Standalone Linear Matmul C = A @ B)
// ---------------------------------------------------------------------------
__global__ void grim_mxfp4_gemm_tiled(
    const float* __restrict__ A,                // [M, K]
    const unsigned char* __restrict__ B_codes,  // [N, K/2]
    const unsigned char* __restrict__ B_exps,   // [N, K/32]
    float* __restrict__ C,                      // [M, N]
    int M,
    int N,
    int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    const int exps_per_row = K / 32;
    const int row_exp_offset = col * exps_per_row;
    const int row_codes_offset = col * (K / 2);

    for (int block_k = 0; block_k < exps_per_row; ++block_k) {
        unsigned char exp_val = B_exps[row_exp_offset + block_k];
        int k_base = block_k * 32;
        int code_byte_base = row_codes_offset + block_k * 16;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned char packed_byte = B_codes[code_byte_base + i];
            unsigned char c0 = packed_byte & 0x0F;
            unsigned char c1 = (packed_byte >> 4) & 0x0F;

            int k0 = k_base + i * 2;
            int k1 = k0 + 1;

            float a0 = A[row * K + k0];
            float a1 = A[row * K + k1];

            float w0 = mxfp4_decode_fast(c0, exp_val);
            float w1 = mxfp4_decode_fast(c1, exp_val);

            acc += a0 * w0 + a1 * w1;
        }
    }

    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// LFM2-style: per-head QK-Norm + RoPE (consumes a raw QKV GEMM output)
// ---------------------------------------------------------------------------
// Fuses the post-projection normalization used by QK-norm models (e.g. LFM2):
//   1. Per-head RMSNorm over `head_dim` applied to Q and K (using gamma_qk)
//   2. Rotary Position Embedding (RoPE), YaRN-aware via inv_freq + mscale
//   3. Scatter to q_out / k_cache / v_cache (V is copied raw, unnormalized)
// The raw QKV `gemm_out = x @ W_qkv` is produced by a separate GEMM launch
// (see `grim_mxfp4_gemm_tiled` + the backend launcher). This split keeps the
// per-head reduction (which needs the whole head vector) out of the GEMM grid.
// Grid: (M * (num_q_heads + 2*num_kv_heads) + 63) / 64 threads, block 64.
// ---------------------------------------------------------------------------
__global__ void grim_qk_norm_rope(
    const float* __restrict__ gemm_out,   // [M, N_total] raw QKV (N_total = N_q + 2*N_k)
    const float* __restrict__ gamma_q,     // [head_dim] Q-norm weight
    const float* __restrict__ gamma_k,     // [head_dim] K-norm weight
    const unsigned int* __restrict__ positions, // [M]
    float* __restrict__ q_out,             // [M, num_q_heads, head_dim] (optional)
    float* __restrict__ k_cache,           // [max_seq, num_kv_heads, head_dim] (optional)
    float* __restrict__ v_cache,           // [max_seq, num_kv_heads, head_dim] (optional)
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int rotary_dim, float rope_theta,
    const float* __restrict__ inv_freq,    // optional YaRN inv_freq[rotary_half]
    float mscale, float eps, int max_seq_len
) {
    int N_k = num_kv_heads * head_dim;
    int N_q = num_q_heads * head_dim;
    int N_total = N_q + 2 * N_k;
    int H_total = num_q_heads + 2 * num_kv_heads;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * H_total) return;

    int row = tid / H_total;
    int local = tid % H_total;
    unsigned int pos = (positions != nullptr) ? positions[row] : (unsigned int)row;

    if (local < num_q_heads) {
        int h = local;
        int base = row * N_total + h * head_dim;
        float ss = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float v = gemm_out[base + d];
            ss += v * v;
        }
        float rms = rsqrtf(ss / (float)head_dim + eps);
        for (int i = 0; i < (rotary_dim / 2); ++i) {
            int d0 = 2 * i, d1 = 2 * i + 1;
            float v0 = gemm_out[base + d0] * rms * gamma_q[d0];
            float v1 = gemm_out[base + d1] * rms * gamma_q[d1];
            float freq = (inv_freq != nullptr)
                ? inv_freq[i]
                : 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle) * mscale;
            float s = sinf(angle) * mscale;
            if (q_out != nullptr) {
                int o = row * N_q + h * head_dim;
                q_out[o + d0] = v0 * c - v1 * s;
                q_out[o + d1] = v0 * s + v1 * c;
            }
        }
        for (int d = rotary_dim; d < head_dim; ++d) {
            float v = gemm_out[base + d] * rms * gamma_q[d];
            if (q_out != nullptr) q_out[row * N_q + h * head_dim + d] = v;
        }
    } else if (local < num_q_heads + num_kv_heads) {
        int h = local - num_q_heads;
        int base = row * N_total + N_q + h * head_dim;
        float ss = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float v = gemm_out[base + d];
            ss += v * v;
        }
        float rms = rsqrtf(ss / (float)head_dim + eps);
        for (int i = 0; i < (rotary_dim / 2); ++i) {
            int d0 = 2 * i, d1 = 2 * i + 1;
            float v0 = gemm_out[base + d0] * rms * gamma_k[d0];
            float v1 = gemm_out[base + d1] * rms * gamma_k[d1];
            float freq = (inv_freq != nullptr)
                ? inv_freq[i]
                : 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle) * mscale;
            float s = sinf(angle) * mscale;
            if (k_cache != nullptr && (int)pos < max_seq_len) {
                int o = (int)pos * N_k + h * head_dim;
                k_cache[o + d0] = v0 * c - v1 * s;
                k_cache[o + d1] = v0 * s + v1 * c;
            }
        }
        for (int d = rotary_dim; d < head_dim; ++d) {
            float v = gemm_out[base + d] * rms * gamma_k[d];
            if (k_cache != nullptr && (int)pos < max_seq_len)
                k_cache[(int)pos * N_k + h * head_dim + d] = v;
        }
    } else {
        int h = local - num_q_heads - num_kv_heads;
        int base = row * N_total + N_q + N_k + h * head_dim;
        if (v_cache != nullptr && (int)pos < max_seq_len) {
            int o = (int)pos * N_k + h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                v_cache[o + d] = gemm_out[base + d];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Backward MXFP4 GEMM (dA = dY @ B^T)
// ---------------------------------------------------------------------------
__global__ void grim_mxfp4_backward_gemm(
    const float* __restrict__ dY,               // [M, N]
    const unsigned char* __restrict__ B_codes,  // [N, K/2]
    const unsigned char* __restrict__ B_exps,   // [N, K/32]
    float* __restrict__ dA,                     // [M, K]
    int M,
    int N,
    int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || k_idx >= K) return;

    const int block_k = k_idx / 32;
    const int in_block = k_idx % 32;
    const int byte_in_block = in_block / 2;
    const int is_high = in_block % 2;
    const int exps_per_row = K / 32;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dy = dY[row * N + n];

        unsigned char exp_val = B_exps[n * exps_per_row + block_k];
        unsigned char packed_byte = B_codes[n * (K / 2) + block_k * 16 + byte_in_block];
        unsigned char code = is_high ? ((packed_byte >> 4) & 0x0F) : (packed_byte & 0x0F);
        float w = mxfp4_decode_fast(code, exp_val);

        acc += dy * w;
    }

    dA[row * K + k_idx] = acc;
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_all_fused_and_tiled_entries() {
        assert!(KERNEL_SOURCE.contains("grim_fused_rmsnorm_mxfp4_gemm_rope_kv"));
        assert!(KERNEL_SOURCE.contains("grim_fused_rmsnorm_mxfp4_gemm"));
        assert!(KERNEL_SOURCE.contains("grim_mxfp4_gemm_tiled"));
        assert!(KERNEL_SOURCE.contains("grim_mxfp4_backward_gemm"));
        assert!(KERNEL_SOURCE.contains("mxfp4_decode_fast"));
    }
}
