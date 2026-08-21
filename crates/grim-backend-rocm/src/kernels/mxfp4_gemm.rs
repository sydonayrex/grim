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
//! 5. `grim_mxfp4_gemm_splitk` + `grim_mxfp4_splitk_reduce`: split-K pair for
//!    skinny-M decode GEMMs (M <= 8), where a plain (M, N/threads) grid leaves
//!    most CUs idle.
//!
//! Perf notes (decode-path review):
//! - Weight codes are read as `uint4` (16 B = 32 codes = exactly one MXFP4
//!   micro-block) instead of 16 scalar byte loads.
//! - Activations are read as `float4` (K is a multiple of 32, so offsets stay
//!   16B-aligned; the launcher validates this).
//! - The E8M0 block scale is computed once per 32-element block
//!   (`exp2f(e - 127)`) instead of calling `ldexpf` per element.
//! - RoPE pairs: the partner column's GEMM result is fetched from the adjacent
//!   lane with one `__shfl_xor_sync` instead of recomputing the entire K-long
//!   dot product (which used to double the QK GEMM cost for rotary dims).
//!   Warp bases are multiples of 32 (blockDim.x is a power of two >= 32) and
//!   RoPE pairs are adjacent (2i, 2i+1), so both lanes of a pair always live
//!   in the same warp.

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

// E8M0 -> float scale for a whole 32-element micro-block. Callers multiply
// LUT bases by this once per block instead of paying ldexpf per element.
__device__ __forceinline__ float mxfp4_block_scale(unsigned char shared_exp) {
    return exp2f((float)(int)shared_exp - 127.0f);
}

// Accumulate one 32-element MXFP4 micro-block into `acc`, vectorizing the
// code stream as one uint4 (16 B) and the activation stream as float4s.
// `a_row` points at A[row*K + block_k*32] (as float4*); `gamma4` optionally
// points at gamma[block_k*32] (as float4*) and is applied elementwise (null
// for the un-normalized kernels).
__device__ __forceinline__ float mxfp4_dot_block(
    const float4* __restrict__ a_row,
    const float4* __restrict__ gamma4,        // may be null
    const unsigned char* __restrict__ codes,  // 16 bytes = 32 codes
    float block_scale,
    float acc)
{
    const uint4 packed = *reinterpret_cast<const uint4*>(codes);
    float4 x[8];
    #pragma unroll
    for (int v = 0; v < 8; ++v) x[v] = a_row[v];
    if (gamma4 != nullptr) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float4 g = gamma4[v];
            x[v].x *= g.x; x[v].y *= g.y;
            x[v].z *= g.z; x[v].w *= g.w;
        }
    }
    const float* xf = reinterpret_cast<const float*>(x);
    const unsigned int* pc = reinterpret_cast<const unsigned int*>(&packed);
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        unsigned int word = pc[w];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            unsigned int byte_val = (word >> (8 * b)) & 0xFFu;
            float w0 = MXFP4_E2M1_LUT[byte_val & 0x0F] * block_scale;
            float w1 = MXFP4_E2M1_LUT[(byte_val >> 4) & 0x0F] * block_scale;
            int k0 = w * 8 + b * 2;
            acc += xf[k0 + 0] * w0 + xf[k0 + 1] * w1;
        }
    }
    return acc;
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
__global__ void __launch_bounds__(256)
grim_fused_rmsnorm_mxfp4_gemm_rope_kv(
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
    const float4* x_row = reinterpret_cast<const float4*>(x + (size_t)row * K);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);

    for (int block_k = 0; block_k < exps_per_row; ++block_k) {
        float scale = mxfp4_block_scale(w_exps[row_exp_offset + block_k]);
        acc = mxfp4_dot_block(
            x_row + block_k * 8,
            gamma4 + block_k * 8,
            w_codes + row_codes_offset + block_k * 16,
            scale,
            acc);
    }
    // Apply the RMSNorm scale post-accumulation (mathematically identical to
    // normalizing x per element: rms factors out of the dot product; gamma
    // does NOT factor out and is applied inside the block helper).
    acc *= rms;

    // Phase 3 & 4: RoPE rotation and direct scatter to Q / KV Cache
    if (out_all != nullptr) {
        out_all[row * N_total + col] = acc;
    }

    unsigned int pos = positions ? positions[row] : (unsigned int)row;

    if (col < N_q) {
        // Query projection
        int d = col % head_dim;

        float q_val = acc;
        if (d < rotary_dim) {
            int pair_idx = d / 2;
            int is_odd = d % 2;
            float freq = (inv_freq != nullptr)
                ? inv_freq[pair_idx]
                : 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)rotary_dim);
            float angle = (float)pos * freq;
            float cos_a, sin_a;
            sincosf(angle, &sin_a, &cos_a);
            cos_a *= mscale;
            sin_a *= mscale;

            // Partner GEMM result lives in the adjacent lane (col ^ 1); both
            // lanes of a RoPE pair are in the same warp, so one shuffle
            // replaces the full partner-column dot-product recompute.
            float partner_acc = __shfl_xor(acc, 1, warpSize);

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
        int d = k_col % head_dim;

        float k_val = acc;
        if (d < rotary_dim) {
            int pair_idx = d / 2;
            int is_odd = d % 2;
            float freq = (inv_freq != nullptr)
                ? inv_freq[pair_idx]
                : 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)rotary_dim);
            float angle = (float)pos * freq;
            float cos_a, sin_a;
            sincosf(angle, &sin_a, &cos_a);
            cos_a *= mscale;
            sin_a *= mscale;

            float partner_acc = __shfl_xor(acc, 1, warpSize);

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
__global__ void __launch_bounds__(256)
grim_fused_rmsnorm_mxfp4_gemm(
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
    const float4* x_row = reinterpret_cast<const float4*>(x + (size_t)row * K);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);

    for (int block_k = 0; block_k < exps_per_row; ++block_k) {
        float scale = mxfp4_block_scale(w_exps[row_exp_offset + block_k]);
        acc = mxfp4_dot_block(
            x_row + block_k * 8,
            gamma4 + block_k * 8,
            w_codes + row_codes_offset + block_k * 16,
            scale,
            acc);
    }
    acc *= rms;

    out[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 3. Tiled 2D MXFP4 GEMM (Standalone Linear Matmul C = A @ B)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
grim_mxfp4_gemm_tiled(
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
    const float4* a_row = reinterpret_cast<const float4*>(A + (size_t)row * K);

    for (int block_k = 0; block_k < exps_per_row; ++block_k) {
        float scale = mxfp4_block_scale(B_exps[row_exp_offset + block_k]);
        acc = mxfp4_dot_block(
            a_row + block_k * 8,
            (const float4*)nullptr,
            B_codes + row_codes_offset + block_k * 16,
            scale,
            acc);
    }

    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 3b. Split-K MXFP4 GEMM for skinny-M decode (M <= 8)
// ---------------------------------------------------------------------------
// With M=1 a plain (N/threads) grid occupies only a handful of CUs. Slice K
// across `gridDim.z` splits, write per-split partials, then reduce. The
// reduction kernel below sums the splits in a fixed order, keeping results
// deterministic across launches.
__global__ void __launch_bounds__(64)
grim_mxfp4_gemm_splitk(
    const float* __restrict__ A,                // [M, K]
    const unsigned char* __restrict__ B_codes,  // [N, K/2]
    const unsigned char* __restrict__ B_exps,   // [N, K/32]
    float* __restrict__ partials,               // [num_splits, M, N]
    int M,
    int N,
    int K,
    int num_splits
) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int split = blockIdx.z;

    if (row >= M || col >= N) return;

    const int exps_per_row = K / 32;
    const int k_per_split = (exps_per_row + num_splits - 1) / num_splits;
    const int bk_begin = split * k_per_split;
    const int bk_end = min(bk_begin + k_per_split, exps_per_row);

    float acc = 0.0f;
    const int row_exp_offset = col * exps_per_row;
    const int row_codes_offset = col * (K / 2);
    const float4* a_row = reinterpret_cast<const float4*>(A + (size_t)row * K);

    for (int block_k = bk_begin; block_k < bk_end; ++block_k) {
        float scale = mxfp4_block_scale(B_exps[row_exp_offset + block_k]);
        acc = mxfp4_dot_block(
            a_row + block_k * 8,
            (const float4*)nullptr,
            B_codes + row_codes_offset + block_k * 16,
            scale,
            acc);
    }

    partials[((size_t)split * M + row) * N + col] = acc;
}

__global__ void __launch_bounds__(256)
grim_mxfp4_splitk_reduce(
    const float* __restrict__ partials,         // [num_splits, M, N]
    float* __restrict__ C,                      // [M, N]
    int M,
    int N,
    int num_splits
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        acc += partials[((size_t)s * M) * N + idx];
    }
    C[idx] = acc;
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
__global__ void __launch_bounds__(256)
grim_qk_norm_rope(
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
            float c, s;
            sincosf(angle, &s, &c);
            c *= mscale;
            s *= mscale;
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
            float c, s;
            sincosf(angle, &s, &c);
            c *= mscale;
            s *= mscale;
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
// Coalescing fix: the naive one-thread-per-(row,k) form strides through
// B_codes[n][block_k] with a K/2-byte stride per n — the worst possible
// access pattern. This form has each thread own up to 32 consecutive k
// (one micro-block row of dA) and walk N in the inner loop: dY[row, n] is
// contiguous in n (coalesced across the warp's rows share), and each
// 16-byte code group / exponent byte is loaded once per thread instead of
// once per element.
__global__ void __launch_bounds__(256)
grim_mxfp4_backward_gemm(
    const float* __restrict__ dY,               // [M, N]
    const unsigned char* __restrict__ B_codes,  // [N, K/2]
    const unsigned char* __restrict__ B_exps,   // [N, K/32]
    float* __restrict__ dA,                     // [M, K]
    int M,
    int N,
    int K
) {
    // One thread per (row, 32-element k micro-block).
    const int row = blockIdx.y;
    const int block_k = blockIdx.x * blockDim.x + threadIdx.x;
    const int exps_per_row = K / 32;

    if (row >= M || block_k >= exps_per_row) return;

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) acc[i] = 0.0f;

    const int k_base = block_k * 32;
    const float* dy_row = dY + (size_t)row * N;

    for (int n = 0; n < N; ++n) {
        float dy = dy_row[n];
        float scale = mxfp4_block_scale(B_exps[n * exps_per_row + block_k]);
        const uint4 packed = *reinterpret_cast<const uint4*>(
            B_codes + (size_t)n * (K / 2) + block_k * 16);
        const unsigned int* pc = reinterpret_cast<const unsigned int*>(&packed);
        float f = dy * scale;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            unsigned int word = pc[w];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                unsigned int byte_val = (word >> (8 * b)) & 0xFFu;
                int k0 = w * 8 + b * 2;
                acc[k0 + 0] += f * MXFP4_E2M1_LUT[byte_val & 0x0F];
                acc[k0 + 1] += f * MXFP4_E2M1_LUT[(byte_val >> 4) & 0x0F];
            }
        }
    }

    float* out = dA + (size_t)row * K + k_base;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        out[i] = acc[i];
    }
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
        assert!(KERNEL_SOURCE.contains("grim_mxfp4_gemm_splitk"));
        assert!(KERNEL_SOURCE.contains("grim_mxfp4_splitk_reduce"));
        assert!(KERNEL_SOURCE.contains("__shfl_xor"));
        assert!(KERNEL_SOURCE.contains("mxfp4_block_scale"));
    }
}
