//! Fused RMSNorm + MXFP4 GEMM + RoPE and Tiled MXFP4 GEMM HIP kernels for ROCm.
//! Implements high-throughput Microscaling FP4 (OCP MXFP4: 4-bit E2M1 codes + 8-bit E8M0 shared exponents per.

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

// NVFP4: 9 bytes per 16 values, scale byte is E4M3, zero code is a real zero.
__device__ __forceinline__ float nvfp4_block_scale(unsigned char e4m3) {
    return fp8_e4m3_to_float_hip(e4m3);
}

__device__ __forceinline__ float nvfp4_decode_code(unsigned char code, float scale) {
    return mxfp4_to_float_hip(code, 127) * scale;
}

// Nutcracker: same 9-byte layout, scale byte is [exp:6 | sel:2]; the low 2
// bits select a special value that the repurposed zero code emits.
__device__ __forceinline__ float nutcracker_block_scale(unsigned char scale_byte, int* sel_out) {
    int sel = scale_byte & 0x3;
    *sel_out = sel;
    return exp2f((float)(int)(scale_byte >> 2) - 31.0f);
}

__device__ __forceinline__ float nutcracker_decode_code(unsigned char code, float scale, int sel) {
    if ((code & 0x7) == 0) {
        float mag = (sel & 0x1) ? 2.5f : 5.0f;
        return ((sel & 0x2) ? -mag : mag) * scale;
    }
    return mxfp4_to_float_hip(code, 127) * scale;
}

// v7/v9 word-wide VALU E2M1 lookup:
// The 8 E2M1 magnitudes are {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
// Their doubled integer values are {0, 1, 2, 3, 4, 6, 8, 12} — 8 bytes that fit
// in two 32-bit constants (`0x03020100u` and `0x0c080604u`)!
// Given 4 E2M1 codes packed in the 4 bytes of `codes4`, one `__builtin_amdgcn_perm`
// instruction looks up all 4 doubled magnitudes in registers with zero constant-cache
// reads. Multiplying by `half_scale = 0.5f * block_scale` is 100% IEEE-754 bit-exact
// because `block_scale = 2^(shared_exp - 127)` is a pure power of two.
__device__ __forceinline__ unsigned int mxfp4_e2m1x4_valu(unsigned int codes4) {
    return __builtin_amdgcn_perm(0x0c080604u, 0x03020100u, codes4 & 0x07070707u);
}

// Accumulate one 32-element MXFP4 micro-block into `acc`, vectorizing the code stream as one uint4 (16 B) and the activation stream as float4s.
// `a_row` points at A[row*K + block_k*32] (as float4*); `gamma4` optionally points at gamma[block_k*32] (as float4*).
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
    const unsigned int pc[4] = { packed.x, packed.y, packed.z, packed.w };
    const float half_scale = 0.5f * block_scale;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        const unsigned int word = pc[w];
        const unsigned int lo_c4 = word & 0x0F0F0F0Fu;
        const unsigned int hi_c4 = (word >> 4) & 0x0F0F0F0Fu;
        const unsigned int lo_m4 = mxfp4_e2m1x4_valu(lo_c4);
        const unsigned int hi_m4 = mxfp4_e2m1x4_valu(hi_c4);
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            float m0 = (float)((lo_m4 >> (8 * b)) & 0xFFu);
            float m1 = (float)((hi_m4 >> (8 * b)) & 0xFFu);
            if (lo_c4 & (0x08u << (8 * b))) m0 = -m0;
            if (hi_c4 & (0x08u << (8 * b))) m1 = -m1;
            float w0 = m0 * half_scale;
            float w1 = m1 * half_scale;
            int k0 = w * 8 + b * 2;
            acc += xf[k0 + 0] * w0 + xf[k0 + 1] * w1;
        }
    }
    return acc;
}

// 1. Fused RMSNorm + MXFP4 QKV GEMM +
// RoPE + Direct KV Cache Scatter Fuses: 1.
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
    int max_seq_len,
    int rope_interleaved
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
    // Apply the RMSNorm scale post-accumulation (mathematically identical to normalizing x per element: rms factors out
    // of the dot product; gamma does NOT factor out and is applied inside the block helper).
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
            int rot_half = rotary_dim / 2;
            int pair_idx = rope_interleaved ? (d / 2) : (d % rot_half);
            int is_second = rope_interleaved ? (d % 2) : (d >= rot_half);
            float freq = (inv_freq != nullptr)
                ? inv_freq[pair_idx]
                : 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)rotary_dim);
            float angle = (float)pos * freq;
            float cos_a, sin_a;
            sincosf(angle, &sin_a, &cos_a);
            cos_a *= mscale;
            sin_a *= mscale;

            float partner_acc;
            if (rope_interleaved) {
                // Partner GEMM result lives in the adjacent lane (col ^ 1); both lanes of a
                // RoPE pair are in the same warp, so one shuffle replaces the full partner-column dot-product recompute.
                partner_acc = __shfl_xor(acc, 1, warpSize);
            } else {
                // Half-split (NeoX): partner column is +/- rot_half within the same head.
                // blockDim.x == head_dim == 64, so the partner lane is inside this block:
                // exchange the accumulators through shared memory instead of a shuffle.
                int partner_d = (d < rot_half) ? (d + rot_half) : (d - rot_half);
                s_sum[threadIdx.x] = acc;
                __syncthreads();
                partner_acc = s_sum[(threadIdx.x - d) + partner_d];
            }

            if (is_second) {
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
            int rot_half = rotary_dim / 2;
            int pair_idx = rope_interleaved ? (d / 2) : (d % rot_half);
            int is_second = rope_interleaved ? (d % 2) : (d >= rot_half);
            float freq = (inv_freq != nullptr)
                ? inv_freq[pair_idx]
                : 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)rotary_dim);
            float angle = (float)pos * freq;
            float cos_a, sin_a;
            sincosf(angle, &sin_a, &cos_a);
            cos_a *= mscale;
            sin_a *= mscale;

            float partner_acc;
            if (rope_interleaved) {
                partner_acc = __shfl_xor(acc, 1, warpSize);
            } else {
                int partner_d = (d < rot_half) ? (d + rot_half) : (d - rot_half);
                s_sum[threadIdx.x] = acc;
                __syncthreads();
                partner_acc = s_sum[(threadIdx.x - d) + partner_d];
            }

            if (is_second) {
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

// 2. Fused RMSNorm + MXFP4 GEMM (e.g.
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

// 3. Tiled 2D MXFP4 GEMM (Standalone Linear Matmul C = A @ B)
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

// 3b. Split-K MXFP4 GEMM for skinny-M decode (M <= 8) With
// M=1 a plain (N/threads) grid occupies only a handful of CUs.
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

// 3c. Rook (MXFP4) v8/v9 wave-coalesced 4-column GEMV (`M == 1`):
// - 4 waves per CTA (`__launch_bounds__(128)`, 16 columns per CTA, 4 columns per wave),
//   eliminating the 2-launch split-K + `partials` allocation overhead at M=1.
// - Coalesced K-loads across the 32 lanes of each wave (`block_k = lane; block_k < exps_per_row; block_k += 32`):
//   adjacent lanes read contiguous `uint4` 16-byte code chunks (`512` contiguous bytes per wave per column)
//   and contiguous `float4` activation chunks, instead of striding across `N` by `K/2` bytes!
// - B-first load ordering (`packed_b[4]` + `exp_b[4]` issued before `a_row` loads) +
//   VALU `__builtin_amdgcn_perm` E2M1 decode (`mxfp4_e2m1x4_valu`).
__global__ void __launch_bounds__(128)
grim_mxfp4_gemv_v2(
    const float* __restrict__ A,                // [1, K]
    const unsigned char* __restrict__ B_codes,  // [N, K/2]
    const unsigned char* __restrict__ B_exps,   // [N, K/32]
    float* __restrict__ C,                      // [1, N]
    int M,
    int N,
    int K
) {
    (void)M;
    const int wave     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int col_base = blockIdx.x * 16 + wave * 4;
    if (col_base >= N) return;

    const int active_cols  = (col_base + 4 <= N) ? 4 : (N - col_base);
    const int exps_per_row = K / 32;
    const float4* a_v4     = reinterpret_cast<const float4*>(A);

    const uint4* b_codes_v4[4];
    const unsigned char* b_exps_col[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + ((c < active_cols) ? c : 0);
        b_codes_v4[c] = reinterpret_cast<const uint4*>(B_codes + (size_t)col * (K / 2));
        b_exps_col[c] = B_exps + (size_t)col * exps_per_row;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int block_k = lane; block_k < exps_per_row; block_k += 32) {
        // v8 B-first load ordering: issue all 4 columns' B loads before reading A.
        uint4 packed_b[4];
        unsigned char exp_b[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            packed_b[c] = b_codes_v4[c][block_k];
            exp_b[c]    = b_exps_col[c][block_k];
        }

        float4 x[8];
        const float4* a_blk = a_v4 + (size_t)block_k * 8;
        #pragma unroll
        for (int v = 0; v < 8; ++v) x[v] = a_blk[v];
        const float* xf = reinterpret_cast<const float*>(x);

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (c >= active_cols) break;
            const float half_scale = 0.5f * mxfp4_block_scale(exp_b[c]);
            const unsigned int pc[4] = {
                packed_b[c].x, packed_b[c].y, packed_b[c].z, packed_b[c].w
            };
            float bacc = 0.0f;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const unsigned int word = pc[w];
                const unsigned int lo_c4 = word & 0x0F0F0F0Fu;
                const unsigned int hi_c4 = (word >> 4) & 0x0F0F0F0Fu;
                const unsigned int lo_m4 = mxfp4_e2m1x4_valu(lo_c4);
                const unsigned int hi_m4 = mxfp4_e2m1x4_valu(hi_c4);
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float m0 = (float)((lo_m4 >> (8 * b)) & 0xFFu);
                    float m1 = (float)((hi_m4 >> (8 * b)) & 0xFFu);
                    if (lo_c4 & (0x08u << (8 * b))) m0 = -m0;
                    if (hi_c4 & (0x08u << (8 * b))) m1 = -m1;
                    int k0 = w * 8 + b * 2;
                    bacc += (xf[k0 + 0] * m0 + xf[k0 + 1] * m1) * half_scale;
                }
            }
            facc[c] += bacc;
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[c] += __shfl_xor(facc[c], off);
    }
    if (lane == 0) {
        #pragma unroll
        for (int c = 0; c < active_cols; ++c)
            C[col_base + c] = facc[c];
    }
}

#define ROOK_M_TILE 8

// 3d. Rook (MXFP4) v9 multi-row 4-column hoisted-B GEMM (`ROOK_M_TILE = 8` rows x 4 cols per wave):
// - grid = (ceil(N / 4), ceil(M / ROOK_M_TILE)), block = (32, 1, 1).
// - Coalesced K-loads across the 32-lane wave (`block_k = lane; block_k < exps_per_row; block_k += 32`).
// - Decodes 4 B columns ONCE per 32-element micro-block in registers, then dots against
//   up to `ROOK_M_TILE = 8` rows of `A`, amortizing B DRAM reads and B VALU decode by 8x!
__global__ void __launch_bounds__(32)
grim_mxfp4_gemm_rows_r8(
    const float* __restrict__ A,                // [M, K]
    const unsigned char* __restrict__ B_codes,  // [N, K/2]
    const unsigned char* __restrict__ B_exps,   // [N, K/32]
    float* __restrict__ C,                      // [M, N]
    int M,
    int N,
    int K
) {
    const int col_base = blockIdx.x * 4;
    const int row_base = blockIdx.y * ROOK_M_TILE;
    const int lane     = threadIdx.x;
    if (row_base >= M || col_base >= N) return;

    const int active_cols  = (col_base + 4 <= N) ? 4 : (N - col_base);
    const int active_rows  = (row_base + ROOK_M_TILE <= M) ? ROOK_M_TILE : (M - row_base);
    const int exps_per_row = K / 32;

    const uint4* b_codes_v4[4];
    const unsigned char* b_exps_col[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + ((c < active_cols) ? c : 0);
        b_codes_v4[c] = reinterpret_cast<const uint4*>(B_codes + (size_t)col * (K / 2));
        b_exps_col[c] = B_exps + (size_t)col * exps_per_row;
    }

    float facc[ROOK_M_TILE][4];
    #pragma unroll
    for (int r = 0; r < ROOK_M_TILE; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            facc[r][c] = 0.0f;
        }
    }

    for (int block_k = lane; block_k < exps_per_row; block_k += 32) {
        uint4 packed_b[4];
        float half_scale[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            packed_b[c]   = b_codes_v4[c][block_k];
            half_scale[c] = 0.5f * mxfp4_block_scale(b_exps_col[c][block_k]);
        }

        unsigned int lo_c4[4][4], hi_c4[4][4], lo_m4[4][4], hi_m4[4][4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const unsigned int pc[4] = {
                packed_b[c].x, packed_b[c].y, packed_b[c].z, packed_b[c].w
            };
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                lo_c4[c][w] = pc[w] & 0x0F0F0F0Fu;
                hi_c4[c][w] = (pc[w] >> 4) & 0x0F0F0F0Fu;
                lo_m4[c][w] = mxfp4_e2m1x4_valu(lo_c4[c][w]);
                hi_m4[c][w] = mxfp4_e2m1x4_valu(hi_c4[c][w]);
            }
        }

        #pragma unroll
        for (int r = 0; r < ROOK_M_TILE; ++r) {
            if (r < active_rows) {
                const float4* a_blk = reinterpret_cast<const float4*>(
                    A + (size_t)(row_base + r) * K + (size_t)block_k * 32);
                float4 x[8];
                #pragma unroll
                for (int v = 0; v < 8; ++v) x[v] = a_blk[v];
                const float* xf = reinterpret_cast<const float*>(x);

                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    if (c >= active_cols) break;
                    float bacc = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const unsigned int lc = lo_c4[c][w];
                        const unsigned int hc = hi_c4[c][w];
                        const unsigned int lm = lo_m4[c][w];
                        const unsigned int hm = hi_m4[c][w];
                        #pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            float m0 = (float)((lm >> (8 * b)) & 0xFFu);
                            float m1 = (float)((hm >> (8 * b)) & 0xFFu);
                            if (lc & (0x08u << (8 * b))) m0 = -m0;
                            if (hc & (0x08u << (8 * b))) m1 = -m1;
                            int k0 = w * 8 + b * 2;
                            bacc += (xf[k0 + 0] * m0 + xf[k0 + 1] * m1) * half_scale[c];
                        }
                    }
                    facc[r][c] += bacc;
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROOK_M_TILE; ++r) {
        if (r < active_rows) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float v = facc[r][c];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_xor(v, off);
                if (lane == 0 && c < active_cols)
                    C[(size_t)(row_base + r) * N + col_base + c] = v;
            }
        }
    }
}

// LFM2-style: per-head QK-Norm + RoPE (consumes a raw QKV GEMM output) Fuses the post-projection normalization used by QK-norm models (e.g.
// LFM2): 1.
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
    float mscale, float eps, int max_seq_len, int rope_interleaved
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
            int d0 = rope_interleaved ? 2 * i : i;
            int d1 = rope_interleaved ? 2 * i + 1 : i + (rotary_dim / 2);
            float v0 = gemm_out[base + d0] * rms * gamma_q[d0];
            float v1 = gemm_out[base + d1] * rms * gamma_q[d1];
            float freq = (inv_freq != nullptr)
                ? inv_freq[i]
                : 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
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
            int d0 = rope_interleaved ? 2 * i : i;
            int d1 = rope_interleaved ? 2 * i + 1 : i + (rotary_dim / 2);
            float v0 = gemm_out[base + d0] * rms * gamma_k[d0];
            float v1 = gemm_out[base + d1] * rms * gamma_k[d1];
            float freq = (inv_freq != nullptr)
                ? inv_freq[i]
                : 1.0f / powf(rope_theta, (float)(2 * i) / (float)rotary_dim);
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

// 4. Backward MXFP4 GEMM (dA = dY @ B^T) Coalescing fix: the naive one-thread-per-(row,k) form
// strides through B_codes[n][block_k] with a K/2-byte stride per n - the worst possible access pattern.
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

// 5. Fused Nutcracker GEMV with LDS caching and cooperative Wave reduction (M=1..4)
// Layout per 16 weights: 9 bytes (1 byte [exp:6|sel:2] scale + 8 bytes 4-bit codes).
__global__ void __launch_bounds__(256)
grim_nutcracker_gemv(
    const float* __restrict__ A,                // [M, K]
    const unsigned char* __restrict__ B_packed, // [N, (K/16)*9]
    float* __restrict__ C,                      // [M, N]
    int M,
    int N,
    int K
) {
    // 1 CTA handles one (row, col) output dot-product or 1 col with 256 threads cooperating.
    const int col = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= M || col >= N) return;

    __shared__ float s_red[256];
    const int tid = threadIdx.x;
    const int blocks_per_col = K / 16;
    const size_t col_bytes_stride = (size_t)blocks_per_col * 9;
    const unsigned char* b_col = B_packed + (size_t)col * col_bytes_stride;
    const float* a_row = A + (size_t)row * K;

    float acc = 0.0f;

    // Grid-stride over 16-element sub-blocks
    for (int b = tid; b < blocks_per_col; b += blockDim.x) {
        const unsigned char* blk = b_col + (size_t)b * 9;
        int sel; float scale = nutcracker_block_scale(blk[0], &sel);
        const unsigned char* codes = blk + 1;
        const float* a_ptr = a_row + b * 16;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned char c_byte = codes[i];
            float w0 = nutcracker_decode_code(c_byte & 0x0F, scale, sel);
            float w1 = nutcracker_decode_code((c_byte >> 4) & 0x0F, scale, sel);
            acc += a_ptr[i * 2 + 0] * w0;
            acc += a_ptr[i * 2 + 1] * w1;
        }
    }

    // Cooperative reduction using RDNA Wave shuffle (Wave32 or Wave64)
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        acc += __shfl_down(acc, offset, warpSize);
    }

    const int lane = tid % warpSize;
    const int wid = tid / warpSize;
    if (lane == 0) {
        s_red[wid] = acc;
    }
    __syncthreads();

    // Final inter-wave reduction in LDS by first wave
    if (wid == 0) {
        int num_warps = blockDim.x / warpSize;
        float wave_sum = (lane < num_warps) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            wave_sum += __shfl_down(wave_sum, offset, warpSize);
        }
        if (lane == 0) {
            C[(size_t)row * N + col] = wave_sum;
        }
    }
}

// 6. Tiled Nutcracker GEMM (M > 4 batch GEMM)
__global__ void __launch_bounds__(256)
grim_nutcracker_gemm_tiled(
    const float* __restrict__ A,                // [M, K]
    const unsigned char* __restrict__ B_packed, // [N, (K/16)*9]
    float* __restrict__ C,                      // [M, N]
    int M,
    int N,
    int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    const int blocks_per_col = K / 16;
    const size_t col_bytes_stride = (size_t)blocks_per_col * 9;
    const unsigned char* b_col = B_packed + (size_t)col * col_bytes_stride;
    const float* a_row = A + (size_t)row * K;

    float acc = 0.0f;
    for (int b = 0; b < blocks_per_col; ++b) {
        const unsigned char* blk = b_col + (size_t)b * 9;
        int sel; float scale = nutcracker_block_scale(blk[0], &sel);
        const unsigned char* codes = blk + 1;
        const float* a_ptr = a_row + b * 16;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned char c_byte = codes[i];
            float w0 = nutcracker_decode_code(c_byte & 0x0F, scale, sel);
            float w1 = nutcracker_decode_code((c_byte >> 4) & 0x0F, scale, sel);
            acc += a_ptr[i * 2 + 0] * w0 + a_ptr[i * 2 + 1] * w1;
        }
    }

    C[(size_t)row * N + col] = acc;
}
// 7. Fused NVFP4 GEMV with LDS caching and cooperative Wave reduction (M=1..4)
// Layout per 16 weights: 9 bytes (1 byte E4M3 scale + 8 bytes 4-bit codes).
__global__ void __launch_bounds__(256)
grim_nvfp4_gemv(
    const float* __restrict__ A,                // [M, K]
    const unsigned char* __restrict__ B_packed, // [N, (K/16)*9]
    float* __restrict__ C,                      // [M, N]
    int M,
    int N,
    int K
) {
    // 1 CTA handles one (row, col) output dot-product or 1 col with 256 threads cooperating.
    const int col = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= M || col >= N) return;

    __shared__ float s_red[256];
    const int tid = threadIdx.x;
    const int blocks_per_col = K / 16;
    const size_t col_bytes_stride = (size_t)blocks_per_col * 9;
    const unsigned char* b_col = B_packed + (size_t)col * col_bytes_stride;
    const float* a_row = A + (size_t)row * K;

    float acc = 0.0f;

    // Grid-stride over 16-element sub-blocks
    for (int b = tid; b < blocks_per_col; b += blockDim.x) {
        const unsigned char* blk = b_col + (size_t)b * 9;
        float scale = nvfp4_block_scale(blk[0]);
        const unsigned char* codes = blk + 1;
        const float* a_ptr = a_row + b * 16;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned char c_byte = codes[i];
            float w0 = nvfp4_decode_code(c_byte & 0x0F, scale);
            float w1 = nvfp4_decode_code((c_byte >> 4) & 0x0F, scale);
            acc += a_ptr[i * 2 + 0] * w0;
            acc += a_ptr[i * 2 + 1] * w1;
        }
    }

    // Cooperative reduction using RDNA Wave shuffle (Wave32 or Wave64)
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        acc += __shfl_down(acc, offset, warpSize);
    }

    const int lane = tid % warpSize;
    const int wid = tid / warpSize;
    if (lane == 0) {
        s_red[wid] = acc;
    }
    __syncthreads();

    // Final inter-wave reduction in LDS by first wave
    if (wid == 0) {
        int num_warps = blockDim.x / warpSize;
        float wave_sum = (lane < num_warps) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            wave_sum += __shfl_down(wave_sum, offset, warpSize);
        }
        if (lane == 0) {
            C[(size_t)row * N + col] = wave_sum;
        }
    }
}

// 8. Tiled NVFP4 GEMM (M > 4 batch GEMM)
__global__ void __launch_bounds__(256)
grim_nvfp4_gemm_tiled(
    const float* __restrict__ A,                // [M, K]
    const unsigned char* __restrict__ B_packed, // [N, (K/16)*9]
    float* __restrict__ C,                      // [M, N]
    int M,
    int N,
    int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    const int blocks_per_col = K / 16;
    const size_t col_bytes_stride = (size_t)blocks_per_col * 9;
    const unsigned char* b_col = B_packed + (size_t)col * col_bytes_stride;
    const float* a_row = A + (size_t)row * K;

    float acc = 0.0f;
    for (int b = 0; b < blocks_per_col; ++b) {
        const unsigned char* blk = b_col + (size_t)b * 9;
        float scale = nvfp4_block_scale(blk[0]);
        const unsigned char* codes = blk + 1;
        const float* a_ptr = a_row + b * 16;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned char c_byte = codes[i];
            float w0 = nvfp4_decode_code(c_byte & 0x0F, scale);
            float w1 = nvfp4_decode_code((c_byte >> 4) & 0x0F, scale);
            acc += a_ptr[i * 2 + 0] * w0 + a_ptr[i * 2 + 1] * w1;
        }
    }

    C[(size_t)row * N + col] = acc;
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
        assert!(KERNEL_SOURCE.contains("grim_nutcracker_gemv"));
        assert!(KERNEL_SOURCE.contains("grim_nutcracker_gemm_tiled"));
        assert!(KERNEL_SOURCE.contains("grim_nvfp4_gemv"));
        assert!(KERNEL_SOURCE.contains("grim_nvfp4_gemm_tiled"));
        assert!(KERNEL_SOURCE.contains("nvfp4_block_scale"));
        assert!(KERNEL_SOURCE.contains("nutcracker_block_scale"));
        assert!(KERNEL_SOURCE.contains("mxfp4_decode_fast"));
        assert!(KERNEL_SOURCE.contains("grim_mxfp4_gemm_splitk"));
        assert!(KERNEL_SOURCE.contains("grim_mxfp4_splitk_reduce"));
        assert!(KERNEL_SOURCE.contains("__shfl_xor"));
        assert!(KERNEL_SOURCE.contains("__shfl_down"));
        assert!(KERNEL_SOURCE.contains("mxfp4_block_scale"));
    }
}
