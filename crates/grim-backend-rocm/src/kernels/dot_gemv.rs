//! SPEED-DOT: vector dot-product GEMV kernels for M=1..4 decode on RDNA3/4.
//!
//! Uses __builtin_amdgcn_sudot4 (V_DOT4_I32_IU8, signed x signed) to execute
//! fast vec_dot_q8_0_q8_1 mat-vec multiplication with Q8_1 activation quantization.

/// Full JIT source for Q8_0 x Q8_1 dot-product GEMV kernels.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)

__device__ __forceinline__ int grim_sdot4(int a, int b, int c) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1035__) || defined(__gfx1036__)
    // RDNA2: dot1-insts only. V_DOT4_I32_I8 (signed x signed, i32 acc).
    return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
    // RDNA3/4: V_DOT4_I32_IU8 with both inputs signed (signA=true, signB=true).
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    return (int)((char)(a & 0xff)) * ((int)(char)(b & 0xff))
         + (int)((char)((a >> 8) & 0xff)) * ((int)(char)((b >> 8) & 0xff))
         + (int)((char)((a >> 16) & 0xff)) * ((int)(char)((b >> 16) & 0xff))
         + (int)((char)((a >> 24) & 0xff)) * ((int)(char)((b >> 24) & 0xff))
         + c;
#endif
}

#define GRIM_Q8_1_BLOCK_SIZE 32
#define GRIM_Q8_1_BYTES      36
#define GRIM_Q8_0_BLOCK_SIZE 32
#define GRIM_Q8_0_BYTES      34

// Upstream GGUF Q2_0 (dtype tag 42): ggml_half d + qs[16], 64 weights per
// 18-byte block, codebook {-1, 0, +1, +2} (llama.cpp ggml-common.h:187).
#define GRIM_Q2_0_BLOCK_SIZE 64
#define GRIM_Q2_0_BYTES      18

// Q2_0 x Q8_0 GEMV, ported from llama.cpp `ggml_vec_dot_q2_0_q8_0_generic`
// (ggml/src/ggml-cpu/quants.c:177).
//
// The block pairing is the load-bearing constraint, and it is not symmetric:
// ONE Q2_0 block of 64 weights maps to TWO Q8_0 blocks of 32, so each Q2_0
// block issues two half-block dots against consecutive activation blocks. The
// same pairing shows up on the GPU side as `get_dm(uint ib) { return
// data_a[ib / 2].d; }` in the reference's `mul_mat_vecq_funcs.glsl`.
//
// Reference accumulation, transcribed verbatim:
//     sumi_block  = sum over the 32 codes of ((code - 1) * qy)
//     sumi       += d1_k * sumi_block          (d1 = the Q8_0 block's scale)
//     sumf       += d0 * sumi                 (d0 = the Q2_0 block's scale)
// The inner dot stays in integer: codes are in [-1, 2] and activations in
// [-127, 127], so a 64-wide block accumulates to at most 2*127*64 = 16256,
// comfortably inside int32. That is what keeps the weight packed -- no f32
// expansion of the 14.6 GB expert banks.
extern "C" __global__ void grim_dot4_q2_0_q80_gemv(
    const unsigned char* __restrict__ A_q80,
    const unsigned char* __restrict__ B_q20,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_x_blocks = K / GRIM_Q2_0_BLOCK_SIZE;   // 64-weight weight blocks
    const int n_y_blocks = K / GRIM_Q8_0_BLOCK_SIZE;   // 32-weight activation blocks

    const unsigned char* a_row =
        A_q80 + (long long)row * n_y_blocks * GRIM_Q8_0_BYTES;

    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const unsigned char* b_col[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q20 + (long long)(col_base + j) * n_x_blocks * GRIM_Q2_0_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_x_blocks; blk += 32) {
        // The two Q8_0 activation blocks that pair with this Q2_0 block.
        const unsigned char* a_blk0 = a_row + (long long)(blk * 2)     * GRIM_Q8_0_BYTES;
        const unsigned char* a_blk1 = a_row + (long long)(blk * 2 + 1) * GRIM_Q8_0_BYTES;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q2_0_BYTES;
            float d0 = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const unsigned char* qs = b_blk + 2;   // 16 bytes = 64 two-bit codes

            float sumi = 0.0f;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const unsigned char* a_blk = half == 0 ? a_blk0 : a_blk1;
                float d1 = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                const signed char* qy = (const signed char*)(a_blk + 2);

                int sumi_block = 0;
                const unsigned char* qs_h = qs + half * 8;   // 8 bytes = 32 codes
                for (int b = 0; b < 8; ++b) {
                    const unsigned char byte = qs_h[b];
                    // Unpack four 2-bit codes and map {0,1,2,3} -> {-1,0,+1,+2}.
                    sumi_block += ((int)((byte >> 0) & 3) - 1) * (int)qy[b*4 + 0];
                    sumi_block += ((int)((byte >> 2) & 3) - 1) * (int)qy[b*4 + 1];
                    sumi_block += ((int)((byte >> 4) & 3) - 1) * (int)qy[b*4 + 2];
                    sumi_block += ((int)((byte >> 6) & 3) - 1) * (int)qy[b*4 + 3];
                }
                sumi += d1 * (float)sumi_block;
            }
            facc[j] += d0 * sumi;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

// PLAN-kernel-launch-reduction Phase B: QKV GEMV — one launch covers the three
// projections (each col-base 4-group lies wholly inside one section, since the
// launcher requires Nq % 4 == 0 and Nkv % 4 == 0). Same math as
// grim_dot4_q80_q81_gemv, with the section picked per 4-col group.
extern "C" __global__ void grim_dot4_qkv_q80_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ Wq,
    const unsigned char* __restrict__ Wk,
    const unsigned char* __restrict__ Wv,
    float* __restrict__ Q,
    float* __restrict__ Ko,
    float* __restrict__ V,
    int M, int Nq, int Nkv, int KD)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    const int n_total  = Nq + 2 * Nkv;
    if (row >= M || col_base >= n_total) return;

    const int n_q_blocks = KD / GRIM_Q8_0_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* B;
    float* C;
    int N;
    int local_base;
    if (col_base < Nq) {
        B = Wq; C = Q; N = Nq; local_base = col_base;
    } else if (col_base < Nq + Nkv) {
        B = Wk; C = Ko; N = Nkv; local_base = col_base - Nq;
    } else {
        B = Wv; C = V; N = Nkv; local_base = col_base - Nq - Nkv;
    }

    const unsigned char* b_col[4];
    const int active_cols = (local_base + 4 <= N) ? 4 : (N - local_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B + (long long)(local_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const unsigned char* a_blk = a_row + blk * GRIM_Q8_1_BYTES;
        float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        const signed char* a_codes = (const signed char*)(a_blk + 4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);
            int iacc = 0;
            #pragma unroll
            for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e += 4) {
                int a4; __builtin_memcpy(&a4, a_codes + e, 4);
                int b4; __builtin_memcpy(&b4, b_codes + e, 4);
                iacc = grim_sdot4(a4, b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + local_base + j] = facc[j];
        }
    }
}

// Phase D.2: Fused activation quantize + QKV GEMV.
// Reads f32 activation input directly, quantizes per-block in registers,
// computes Q, K, V dot4 projections in one launch without separate quantize kernel.
extern "C" __global__ void grim_dot4_qkv_q80_f32act_gemv(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ Wq,
    const unsigned char* __restrict__ Wk,
    const unsigned char* __restrict__ Wv,
    float* __restrict__ Q,
    float* __restrict__ Ko,
    float* __restrict__ V,
    int M, int Nq, int Nkv, int KD)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    const int n_total  = Nq + 2 * Nkv;
    if (row >= M || col_base >= n_total) return;

    const int n_q_blocks = KD / GRIM_Q8_0_BLOCK_SIZE;
    const float* a_row = act_f32 + (long long)row * KD;

    const unsigned char* B;
    float* C;
    int N;
    int local_base;
    if (col_base < Nq) {
        B = Wq; C = Q; N = Nq; local_base = col_base;
    } else if (col_base < Nq + Nkv) {
        B = Wk; C = Ko; N = Nkv; local_base = col_base - Nq;
    } else {
        B = Wv; C = V; N = Nkv; local_base = col_base - Nq - Nkv;
    }

    const unsigned char* b_col[4];
    const int active_cols = (local_base + 4 <= N) ? 4 : (N - local_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B + (long long)(local_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const float* blk_src = a_row + (long long)blk * GRIM_Q8_0_BLOCK_SIZE;

        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            amax = fmaxf(amax, __builtin_fabsf(blk_src[e]));
        }

        const float d     = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        _Float16 hd = (_Float16)d;
        float d_a = (float)hd;

        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(blk_src[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(blk_src[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(blk_src[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(blk_src[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0) |
                        ((int)((unsigned char)q1) << 8) |
                        ((int)((unsigned char)q2) << 16) |
                        ((int)((unsigned char)q3) << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);
            int iacc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int b4; __builtin_memcpy(&b4, b_codes + p * 4, 4);
                iacc = grim_sdot4(a4_reg[p], b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + local_base + j] = facc[j];
        }
    }
}

// PLAN 4 (DukeNukem): norm-fused QKV — rms_norm prologue + per-block
// activation quantize + QKV dot4 GEMV in one launch. The prologue replays
// grim_rms_norm's exact traversal (col = lane; col < KD; col += 32) and
// formula (sqrtf(ss/KD + eps), out = x*w/rms) so results are bit-identical
// to separate norm + f32act launches; the redundant per-block sumsq reads
// one L2-resident 4KB row and is dwarfed by the eliminated launch.
extern "C" __global__ void grim_dot4_qkv_q80_norm_f32act_gemv(
    const float* __restrict__ res,
    const float* __restrict__ gamma,
    float eps,
    const unsigned char* __restrict__ Wq,
    const unsigned char* __restrict__ Wk,
    const unsigned char* __restrict__ Wv,
    float* __restrict__ Q,
    float* __restrict__ Ko,
    float* __restrict__ V,
    int M, int Nq, int Nkv, int KD)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    const int n_total  = Nq + 2 * Nkv;
    if (row >= M || col_base >= n_total) return;

    const int n_q_blocks = KD / GRIM_Q8_0_BLOCK_SIZE;
    const float* r_row = res + (long long)row * KD;

    const unsigned char* B;
    float* C;
    int N;
    int local_base;
    if (col_base < Nq) {
        B = Wq; C = Q; N = Nq; local_base = col_base;
    } else if (col_base < Nq + Nkv) {
        B = Wk; C = Ko; N = Nkv; local_base = col_base - Nq;
    } else {
        B = Wv; C = V; N = Nkv; local_base = col_base - Nq - Nkv;
    }

    const unsigned char* b_col[4];
    const int active_cols = (local_base + 4 <= N) ? 4 : (N - local_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B + (long long)(local_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    // Prologue: row sumsq with grim_rms_norm-identical traversal and formula.
    float ss = 0.0f;
    for (int col = lane; col < KD; col += 32) {
        float v = r_row[col];
        ss += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor(ss, off);
    float rms = sqrtf(ss / (float)KD + eps);

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        // Norm on the fly (grim_rms_norm-identical: x*w/rms), then quantize
        // exactly like the f32act body (fp max is order-exact, fp16 scale
        // round-trip, roundf codes).
        float nrm[GRIM_Q8_0_BLOCK_SIZE];
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            nrm[e] = r_row[(long long)blk * GRIM_Q8_0_BLOCK_SIZE + e]
                   * gamma[(long long)blk * GRIM_Q8_0_BLOCK_SIZE + e] / rms;
        }

        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            amax = fmaxf(amax, __builtin_fabsf(nrm[e]));
        }

        const float d     = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        _Float16 hd = (_Float16)d;
        float d_a = (float)hd;

        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(nrm[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(nrm[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(nrm[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(nrm[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0) |
                        ((int)((unsigned char)q1) << 8) |
                        ((int)((unsigned char)q2) << 16) |
                        ((int)((unsigned char)q3) << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);
            int iacc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int b4; __builtin_memcpy(&b4, b_codes + p * 4, 4);
                iacc = grim_sdot4(a4_reg[p], b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + local_base + j] = facc[j];
        }
    }
}
// ONE launch replaces (gate GEMV, up GEMV, silu_mul). Each 4-col group does
// two independent dot4 passes (gate blob row, up blob row) over the same
// quantized activation, then writes silu(g) * u. No cross-col reduction.
extern "C" __global__ void grim_dot4_gate_up_silu_q80_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ Wg,
    const unsigned char* __restrict__ Wu,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* g_col[4];
    const unsigned char* u_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        g_col[j] = (j < active_cols)
                 ? Wg + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
        u_col[j] = (j < active_cols)
                 ? Wu + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const unsigned char* a_blk = a_row + blk * GRIM_Q8_1_BYTES;
        float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        const signed char* a_codes = (const signed char*)(a_blk + 4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            int iacc_g = 0;
            int iacc_u = 0;
            {
                const unsigned char* b_blk = g_col[j] + blk * GRIM_Q8_0_BYTES;
                float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
                const signed char* b_codes = (const signed char*)(b_blk + 2);
                #pragma unroll
                for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e += 4) {
                    int a4; __builtin_memcpy(&a4, a_codes + e, 4);
                    int b4; __builtin_memcpy(&b4, b_codes + e, 4);
                    iacc_g = grim_sdot4(a4, b4, iacc_g);
                }
                gacc[j] += (float)iacc_g * d_a * d_b;
            }
            {
                const unsigned char* b_blk = u_col[j] + blk * GRIM_Q8_0_BYTES;
                float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
                const signed char* b_codes = (const signed char*)(b_blk + 2);
                #pragma unroll
                for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e += 4) {
                    int a4; __builtin_memcpy(&a4, a_codes + e, 4);
                    int b4; __builtin_memcpy(&b4, b_codes + e, 4);
                    iacc_u = grim_sdot4(a4, b4, iacc_u);
                }
                uacc[j] += (float)iacc_u * d_a * d_b;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gacc[j] += __shfl_xor(gacc[j], off);
            uacc[j] += __shfl_xor(uacc[j], off);
        }
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++) {
            float g = gacc[j];
            float u = uacc[j];
            C[(long long)row * N + col_base + j] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

// Phase D.2: Fused activation quantize + gate/up GEMV + SiLU epilogue.
// Takes f32 activation input directly, quantizes per-block in registers,
// computes gate & up dot4 projections and writes silu(g)*u directly to C.
extern "C" __global__ void grim_dot4_gate_up_silu_q80_f32act_gemv(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ Wg,
    const unsigned char* __restrict__ Wu,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const float* a_row = act_f32 + (long long)row * K;

    const unsigned char* g_col[4];
    const unsigned char* u_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        g_col[j] = (j < active_cols)
                 ? Wg + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
        u_col[j] = (j < active_cols)
                 ? Wu + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const float* blk_src = a_row + (long long)blk * GRIM_Q8_0_BLOCK_SIZE;

        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            amax = fmaxf(amax, __builtin_fabsf(blk_src[e]));
        }

        const float d     = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        _Float16 hd = (_Float16)d;
        float d_a = (float)hd;

        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(blk_src[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(blk_src[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(blk_src[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(blk_src[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0) |
                        ((int)((unsigned char)q1) << 8) |
                        ((int)((unsigned char)q2) << 16) |
                        ((int)((unsigned char)q3) << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            int iacc_g = 0;
            int iacc_u = 0;
            {
                const unsigned char* b_blk = g_col[j] + blk * GRIM_Q8_0_BYTES;
                float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
                const signed char* b_codes = (const signed char*)(b_blk + 2);
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int b4; __builtin_memcpy(&b4, b_codes + p * 4, 4);
                    iacc_g = grim_sdot4(a4_reg[p], b4, iacc_g);
                }
                gacc[j] += (float)iacc_g * d_a * d_b;
            }
            {
                const unsigned char* b_blk = u_col[j] + blk * GRIM_Q8_0_BYTES;
                float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
                const signed char* b_codes = (const signed char*)(b_blk + 2);
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int b4; __builtin_memcpy(&b4, b_codes + p * 4, 4);
                    iacc_u = grim_sdot4(a4_reg[p], b4, iacc_u);
                }
                uacc[j] += (float)iacc_u * d_a * d_b;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gacc[j] += __shfl_xor(gacc[j], off);
            uacc[j] += __shfl_xor(uacc[j], off);
        }
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++) {
            float g = gacc[j];
            float u = uacc[j];
            C[(long long)row * N + col_base + j] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

// Ollama-inspired RDNA4 experiment: eight Wave32 groups per workgroup.
// The K loop is partitioned across waves, then four output accumulators are
// reduced through LDS. This is opt-in via GRIM_DOT4_256=1; the existing
// one-wave path remains the production control until measured on gfx1200/1201.
extern "C" __global__ void grim_dot4_gate_up_silu_q80_f32act_gemv_256(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ Wg,
    const unsigned char* __restrict__ Wu,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int wave     = threadIdx.x / 32;
    const int lane     = threadIdx.x % 32;
    const int group    = lane / 8;
    const int sub_lane = lane % 8;
    const int n_waves  = blockDim.x / 32;
    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const int blocks_per_wave = (n_q_blocks + n_waves - 1) / n_waves;
    if (row >= M || col_base >= N) return;

    const float* a_row = act_f32 + (long long)row * K;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);

    const unsigned char* g_col[4];
    const unsigned char* u_col[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        g_col[j] = (j < active_cols)
            ? Wg + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
            : nullptr;
        u_col[j] = (j < active_cols)
            ? Wu + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
            : nullptr;
    }

    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = 0; b < blocks_per_wave; b++) {
        const int blk = wave * blocks_per_wave + b;
        const bool active = group < blocks_per_wave && blk < n_q_blocks;
        float block_g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float block_u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float d_a = 0.0f;
        int a4_reg = 0;
        if (active) {
            const float* blk_src = a_row + (long long)blk * GRIM_Q8_0_BLOCK_SIZE;
            float amax = 0.0f;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                amax = fmaxf(amax, __builtin_fabsf(blk_src[sub_lane * 4 + e]));
            }
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor(amax, off));
            }
            const float d = amax / 127.0f;
            const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
            d_a = (float)(_Float16)d;
            signed char q0 = (signed char)__builtin_roundf(blk_src[sub_lane * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(blk_src[sub_lane * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(blk_src[sub_lane * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(blk_src[sub_lane * 4 + 3] * inv_d);
            a4_reg = (int)((unsigned char)q0)
                | ((int)((unsigned char)q1) << 8)
                | ((int)((unsigned char)q2) << 16)
                | ((int)((unsigned char)q3) << 24);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            if (active) {
                const unsigned char* g_blk = g_col[j] + blk * GRIM_Q8_0_BYTES;
                const unsigned char* u_blk = u_col[j] + blk * GRIM_Q8_0_BYTES;
                float d_g = fp16_to_float_device(((const unsigned short*)g_blk)[0]);
                float d_u = fp16_to_float_device(((const unsigned short*)u_blk)[0]);
                int g4;
                int u4;
                __builtin_memcpy(&g4, (const signed char*)(g_blk + 2) + sub_lane * 4, 4);
                __builtin_memcpy(&u4, (const signed char*)(u_blk + 2) + sub_lane * 4, 4);
                block_g[j] += (float)grim_sdot4(a4_reg, g4, 0) * d_a * d_g;
                block_u[j] += (float)grim_sdot4(a4_reg, u4, 0) * d_a * d_u;
            }
        }
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                block_g[j] += __shfl_xor(block_g[j], off);
                block_u[j] += __shfl_xor(block_u[j], off);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            gacc[j] += block_g[j];
            uacc[j] += block_u[j];
        }
    }

    __shared__ float s_g[8][4];
    __shared__ float s_u[8][4];
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            s_g[wave][j] = gacc[j];
            s_u[wave][j] = uacc[j];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 0; j < active_cols; j++) {
            float g = 0.0f;
            float u = 0.0f;
            for (int w = 0; w < 8; w++) {
                g += s_g[w][j];
                u += s_u[w][j];
            }
            C[(long long)row * N + col_base + j] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

// Graph fusion: residual add + RMSNorm + Q8_0 Gate/Up GEMV + SiLU.
// This keeps the residual stream in the graph workspace while avoiding the
// separate add_rms_norm and norm-to-f32act passes used by the split path.
extern "C" __global__ void grim_dot4_add_rms_norm_gate_up_silu_q80_gemv(
    const float* __restrict__ base,
    const float* __restrict__ attn,
    const float* __restrict__ gamma,
    float eps,
    const unsigned char* __restrict__ Wg,
    const unsigned char* __restrict__ Wu,
    float* __restrict__ residual_out,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const float* base_row = base + (long long)row * K;
    const float* attn_row = attn + (long long)row * K;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);

    float ss = 0.0f;
    for (int col = lane; col < K; col += 32) {
        float v = base_row[col] + attn_row[col];
        ss += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor(ss, off);
    }
    const float rms = sqrtf(ss / (float)K + eps);
    if (col_base == 0) {
        for (int col = lane; col < K; col += 32) {
            residual_out[(long long)row * K + col] = base_row[col] + attn_row[col];
        }
    }

    const unsigned char* g_col[4];
    const unsigned char* u_col[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        g_col[j] = (j < active_cols)
            ? Wg + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
            : nullptr;
        u_col[j] = (j < active_cols)
            ? Wu + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
            : nullptr;
    }

    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const int base_offset = blk * GRIM_Q8_0_BLOCK_SIZE;
        float nrm[GRIM_Q8_0_BLOCK_SIZE];
        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            nrm[e] = (base_row[base_offset + e] + attn_row[base_offset + e])
                * gamma[base_offset + e] / rms;
            amax = fmaxf(amax, __builtin_fabsf(nrm[e]));
        }
        float d = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        d = fmaxf(d, 1e-12f);
        const float d_a = (float)(_Float16)d;
        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(nrm[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(nrm[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(nrm[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(nrm[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0)
                | ((int)((unsigned char)q1) << 8)
                | ((int)((unsigned char)q2) << 16)
                | ((int)((unsigned char)q3) << 24);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* g_blk = g_col[j] + blk * GRIM_Q8_0_BYTES;
            const unsigned char* u_blk = u_col[j] + blk * GRIM_Q8_0_BYTES;
            const float d_g = fp16_to_float_device(((const unsigned short*)g_blk)[0]);
            const float d_u = fp16_to_float_device(((const unsigned short*)u_blk)[0]);
            const signed char* g_codes = (const signed char*)(g_blk + 2);
            const signed char* u_codes = (const signed char*)(u_blk + 2);
            int iacc_g = 0;
            int iacc_u = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int g4;
                int u4;
                __builtin_memcpy(&g4, g_codes + p * 4, 4);
                __builtin_memcpy(&u4, u_codes + p * 4, 4);
                iacc_g = grim_sdot4(a4_reg[p], g4, iacc_g);
                iacc_u = grim_sdot4(a4_reg[p], u4, iacc_u);
            }
            gacc[j] += (float)iacc_g * d_a * d_g;
            uacc[j] += (float)iacc_u * d_a * d_u;
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gacc[j] += __shfl_xor(gacc[j], off);
            uacc[j] += __shfl_xor(uacc[j], off);
        }
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++) {
            float g = gacc[j];
            float u = uacc[j];
            C[(long long)row * N + col_base + j] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

// One-wave 16-output tile experiment: quantize the activation once per tile,
// then reuse it across 16 output columns instead of repeating quantization for
// every four-column group.
extern "C" __global__ void grim_dot4_q80_f32act_add_gemv_tile16(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ B_q80,
    const float* __restrict__ residual,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int tile_base = blockIdx.x * 16;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= M || tile_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const int ncols = (tile_base + 16 <= N) ? 16 : (N - tile_base);
    const float* a_row = act_f32 + (long long)row * K;
    float facc[16] = {0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const float* blk_src = a_row + (long long)blk * GRIM_Q8_0_BLOCK_SIZE;
        float nrm[GRIM_Q8_0_BLOCK_SIZE];
        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            nrm[e] = blk_src[e];
            amax = fmaxf(amax, __builtin_fabsf(nrm[e]));
        }
        const float d = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        const float d_a = (float)(_Float16)d;
        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(nrm[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(nrm[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(nrm[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(nrm[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0)
                | ((int)((unsigned char)q1) << 8)
                | ((int)((unsigned char)q2) << 16)
                | ((int)((unsigned char)q3) << 24);
        }
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            if (j >= ncols) break;
            const unsigned char* b_blk = B_q80
                + (long long)(tile_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                + blk * GRIM_Q8_0_BYTES;
            const float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);
            int iacc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int b4;
                __builtin_memcpy(&b4, b_codes + p * 4, 4);
                iacc = grim_sdot4(a4_reg[p], b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j >= ncols) break;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            facc[j] += __shfl_xor(facc[j], off);
        }
    }
    if (lane == 0) {
        for (int j = 0; j < ncols; j++) {
            const long long out_idx = (long long)row * N + tile_base + j;
            const float res = (residual != nullptr) ? residual[out_idx] : 0.0f;
            C[out_idx] = res + facc[j];
        }
    }
}

// One-wave 8-output activation-reuse experiment: quantize each activation
// block once for eight output columns. This is deliberately opt-in because
// the extra accumulators reduce the number of independent output groups.
extern "C" __global__ void grim_dot4_q80_f32act_add_gemv_tile8(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ B_q80,
    const float* __restrict__ residual,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int tile_base = blockIdx.x * 8;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= M || tile_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const int ncols = (tile_base + 8 <= N) ? 8 : (N - tile_base);
    const float* a_row = act_f32 + (long long)row * K;
    float facc[8] = {0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const float* blk_src = a_row + (long long)blk * GRIM_Q8_0_BLOCK_SIZE;
        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            amax = fmaxf(amax, __builtin_fabsf(blk_src[e]));
        }
        const float d = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        const float d_a = (float)(_Float16)d;
        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(blk_src[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(blk_src[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(blk_src[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(blk_src[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0)
                | ((int)((unsigned char)q1) << 8)
                | ((int)((unsigned char)q2) << 16)
                | ((int)((unsigned char)q3) << 24);
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j >= ncols) break;
            const unsigned char* b_blk = B_q80
                + (long long)(tile_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                + blk * GRIM_Q8_0_BYTES;
            const float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);
            int iacc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int b4;
                __builtin_memcpy(&b4, b_codes + p * 4, 4);
                iacc = grim_sdot4(a4_reg[p], b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j >= ncols) break;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            facc[j] += __shfl_xor(facc[j], off);
        }
    }
    if (lane == 0) {
        for (int j = 0; j < ncols; j++) {
            const long long out_idx = (long long)row * N + tile_base + j;
            const float res = (residual != nullptr) ? residual[out_idx] : 0.0f;
            C[out_idx] = res + facc[j];
        }
    }
}

// Phase D.2: Fused activation quantize + GEMV + optional residual add epilogue.
// Takes f32 activation input directly, quantizes per-block in registers,
// computes dot4 GEMV projection, and writes (residual ? residual[...] : 0.0f) + facc[j] directly to C.
// When residual == nullptr, acts as standard fused GEMV.
// Supports C == residual for safe in-place addition.
extern "C" __global__ void grim_dot4_q80_f32act_add_gemv(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ B_q80,
    const float* __restrict__ residual,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const float* a_row = act_f32 + (long long)row * K;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q80 + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const float* blk_src = a_row + (long long)blk * GRIM_Q8_0_BLOCK_SIZE;

        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            amax = fmaxf(amax, __builtin_fabsf(blk_src[e]));
        }

        const float d     = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        _Float16 hd = (_Float16)d;
        float d_a = (float)hd;

        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(blk_src[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(blk_src[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(blk_src[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(blk_src[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0) |
                        ((int)((unsigned char)q1) << 8) |
                        ((int)((unsigned char)q2) << 16) |
                        ((int)((unsigned char)q3) << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);
            int iacc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int b4; __builtin_memcpy(&b4, b_codes + p * 4, 4);
                iacc = grim_sdot4(a4_reg[p], b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++) {
            long long out_idx = (long long)row * N + col_base + j;
            float res = (residual != nullptr) ? residual[out_idx] : 0.0f;
            C[out_idx] = res + facc[j];
        }
    }
}

__launch_bounds__(32)
extern "C" __global__ void grim_quantize_q8_1(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,
    int K,
    int n_rows)
{
    const int global_blk = blockIdx.x;
    const int n_q_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int row = global_blk / n_q_blocks;
    const int blk = global_blk % n_q_blocks;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const float* row_src = src + (long long)row * K + blk * GRIM_Q8_1_BLOCK_SIZE;
    unsigned char* row_dst = dst + (long long)row * n_q_blocks * GRIM_Q8_1_BYTES
                                 + blk * GRIM_Q8_1_BYTES;

    float val = (tid < GRIM_Q8_1_BLOCK_SIZE) ? row_src[tid] : 0.0f;

    float amax = __builtin_fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, mask));

    const float d     = amax / 127.0f;
    const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
    signed char qi = (signed char)__builtin_roundf(val * inv_d);

    float fsum = (tid < GRIM_Q8_1_BLOCK_SIZE) ? (float)qi : 0.0f;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        fsum += __shfl_xor(fsum, mask);

    if (tid == 0) {
        _Float16 hd = (_Float16)d;
        _Float16 hs = (_Float16)(fsum * d);
        unsigned short wd, ws;
        __builtin_memcpy(&wd, &hd, 2);
        __builtin_memcpy(&ws, &hs, 2);
        row_dst[0] = (unsigned char)(wd & 0xFF);
        row_dst[1] = (unsigned char)(wd >> 8);
        row_dst[2] = (unsigned char)(ws & 0xFF);
        row_dst[3] = (unsigned char)(ws >> 8);
    }

    if (tid < GRIM_Q8_1_BLOCK_SIZE) {
        ((signed char*)(row_dst + 4))[tid] = qi;
    }
}

// SPEED-FP32-GEMV: custom FP32 matrix-vector multiply for the output projection
// (lm_head). rocBLAS is pathologically slow at M=1 GEMV (1.45 ms for N=65536,
// K=1024) because it dispatches a full GEMM kernel with workspace allocation and
// teardown per call. This kernel matches the proven dot4 pattern (1 wave per
// block, 4 columns per block) for reliable JIT on RDNA4.
extern "C" __global__ void grim_fp32_gemv(
    const float* __restrict__ act,   // [M, K]
    const float* __restrict__ weight, // [N, K]
    float* __restrict__ out,         // [M, N]
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const float* a_row = act + (long long)row * K;
    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
            ? (const unsigned char*)(weight + (long long)(col_base + j) * K)
            : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int blk = lane; blk < K; blk += 32) {
        float a = a_row[blk];
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            float b = ((const float*)b_col[j])[blk];
            facc[j] += a * b;
        }
    }

    // Shuffle reduction within the wave.
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            out[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

// PLAN 2 (Phase D launch reduction): Fused activation quantize + dot4 GEMV.
// Cooperatively quantizes 32-element blocks of act_f32 into LDS, then computes
// dot4 against Q8_0 weights with zero separate quantize launch.
// The Q8_0 x f32-activation dot4 GEMV BODY — the SECOND of the format's two
// launcher variants (the first is the prequantized-q8_1 one above).
//
// Same discipline as that body: the wave's arm calls THIS function, so the
// ring's answer is bit-identical to the launcher's. The models' plain Q8_0
// paths are the f32-activation ones, so this is the body a Q8_0 dot4 node
// would actually name.
//
// One call is one 32-lane GROUP's share of the work: the group owns the
// output row `row` and the four output columns starting at `col_base`, and
// its 32 lanes split the K-blocks. The 32-lane granularity is load-bearing
// (the reduction is a 32-lane shuffle tree).
__device__ __forceinline__ void grim_dot4_q80_f32act_gemv_body(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ B_q80,
    float* __restrict__ C,
    int M, int N, int K,
    int row, int col_base, int lane)
{
    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const float* a_row = act_f32 + (long long)row * K;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q80 + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const float* blk_src = a_row + (long long)blk * GRIM_Q8_0_BLOCK_SIZE;

        // 1. Quantize lane's 32-element block into local registers
        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            amax = fmaxf(amax, __builtin_fabsf(blk_src[e]));
        }

        const float d     = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        _Float16 hd = (_Float16)d;
        float d_a = (float)hd;

        // Pack 32 signed chars into 8 int32s in registers
        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(blk_src[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(blk_src[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(blk_src[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(blk_src[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0) |
                        ((int)((unsigned char)q1) << 8) |
                        ((int)((unsigned char)q2) << 16) |
                        ((int)((unsigned char)q3) << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);

            int iacc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int b4;
                __builtin_memcpy(&b4, b_codes + p * 4, 4);
                iacc = grim_sdot4(a4_reg[p], b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

extern "C" __global__ void grim_dot4_q80_f32act_gemv(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ B_q80,
    float* __restrict__ C,
    int M, int N, int K)
{
    grim_dot4_q80_f32act_gemv_body(
        act_f32, B_q80, C, M, N, K, blockIdx.y, blockIdx.x * 4, threadIdx.x);
}

// PLAN 4 (DukeNukem) Task 3: generic norm-fused single-projection GEMV.
// rms_norm prologue (grim_rms_norm-identical traversal + formula) + per-block
// quantize + dot4 GEMV in one launch. Feeds shortconv in_proj, lm_head, and
// any Q8_0 dot4 projection whose input is rms-normed. Bit-identical to
// separate norm + f32act launches.
extern "C" __global__ void grim_dot4_q80_norm_f32act_gemv(
    const float* __restrict__ res,
    const float* __restrict__ gamma,
    float eps,
    const unsigned char* __restrict__ B_q80,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const float* r_row = res + (long long)row * K;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q80 + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    // Prologue: row sumsq with grim_rms_norm-identical traversal and formula.
    float ss = 0.0f;
    for (int col = lane; col < K; col += 32) {
        float v = r_row[col];
        ss += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor(ss, off);
    float rms = sqrtf(ss / (float)K + eps);

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        float nrm[GRIM_Q8_0_BLOCK_SIZE];
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            nrm[e] = r_row[(long long)blk * GRIM_Q8_0_BLOCK_SIZE + e]
                   * gamma[(long long)blk * GRIM_Q8_0_BLOCK_SIZE + e] / rms;
        }

        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e++) {
            amax = fmaxf(amax, __builtin_fabsf(nrm[e]));
        }

        const float d     = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        _Float16 hd = (_Float16)d;
        float d_a = (float)hd;

        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(nrm[p * 4 + 0] * inv_d);
            signed char q1 = (signed char)__builtin_roundf(nrm[p * 4 + 1] * inv_d);
            signed char q2 = (signed char)__builtin_roundf(nrm[p * 4 + 2] * inv_d);
            signed char q3 = (signed char)__builtin_roundf(nrm[p * 4 + 3] * inv_d);
            a4_reg[p] = (int)((unsigned char)q0) |
                        ((int)((unsigned char)q1) << 8) |
                        ((int)((unsigned char)q2) << 16) |
                        ((int)((unsigned char)q3) << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);
            int iacc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int b4; __builtin_memcpy(&b4, b_codes + p * 4, 4);
                iacc = grim_sdot4(a4_reg[p], b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}


// The Q8_0 x Q8_1 GEMV BODY, shared by the standalone kernel and the ring's
// `OP_Q8_0_DOT4` arm (plan PM-6 / condition 4).
//
// One call is one 32-lane GROUP's share of the work: the group owns the
// output row `row` and the four output columns starting at `col_base`, and
// its 32 lanes split the K-blocks between them. The final reduction is a
// 32-lane shuffle tree (offsets 16..1), so the group MUST be 32 lanes: on a
// wave64 card a 32-lane group is half a wave, and XOR with 16 never crosses
// that half, so the tree still reduces exactly those 32 lanes.
//
// The standalone kernel launches this with blockDim=32 (one group per block);
// the wave's arm calls it once per 32-lane group of its 128 threads. SAME
// source, SAME per-lane block loop, SAME reduction order — so the arm's
// result is BIT-IDENTICAL to the launcher's, which is what the gate asserts.
// Two copies of this arithmetic would drift (and a quantized GEMM's rounding
// is exactly the kind of thing that hides a drift until the logits move).
__device__ __forceinline__ void grim_dot4_q80_q81_gemv_body(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q80,
    float* __restrict__ C,
    int M, int N, int K,
    int row, int col_base, int lane)
{
    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q80 + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const unsigned char* a_blk = a_row + blk * GRIM_Q8_1_BYTES;
        float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        const signed char* a_codes = (const signed char*)(a_blk + 4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);

            int iacc = 0;
            #pragma unroll
            for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e += 4) {
                int a4;
                __builtin_memcpy(&a4, a_codes + e, 4);
                int b4;
                __builtin_memcpy(&b4, b_codes + e, 4);
                iacc = grim_sdot4(a4, b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

extern "C" __global__ void grim_dot4_q80_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q80,
    float* __restrict__ C,
    int M, int N, int K)
{
    grim_dot4_q80_q81_gemv_body(A_q81, B_q80, C, M, N, K,
                                blockIdx.y, blockIdx.x * 4, threadIdx.x);
}

// Down-projection activation-reuse experiment: consume a prequantized Q8.1
// activation row and add the residual in the same Wave32 dot4 launch. This
// trades the default one-launch F32 quantize+dot4 path for two stable graph
// nodes, so it is opt-in and must be measured end-to-end.
extern "C" __global__ void grim_dot4_q80_q81_add_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q80,
    const float* __restrict__ residual,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q_blocks * GRIM_Q8_1_BYTES;
    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
            ? B_q80 + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
            : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const unsigned char* a_blk = a_row + blk * GRIM_Q8_1_BYTES;
        const float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        const signed char* a_codes = (const signed char*)(a_blk + 4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            const float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const signed char* b_codes = (const signed char*)(b_blk + 2);
            int iacc = 0;
            #pragma unroll
            for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e += 4) {
                int a4;
                int b4;
                __builtin_memcpy(&a4, a_codes + e, 4);
                __builtin_memcpy(&b4, b_codes + e, 4);
                iacc = grim_sdot4(a4, b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            facc[j] += __shfl_xor(facc[j], off);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            const long long out_idx = (long long)row * N + col_base + j;
            const float res = (residual != nullptr) ? residual[out_idx] : 0.0f;
            C[out_idx] = res + facc[j];
        }
    }
}

__device__ __forceinline__ unsigned grim_pk_f16(float lo, float hi) {
    _Float16 a = (_Float16)lo;
    _Float16 b = (_Float16)hi;
    unsigned short ba, bb;
    __builtin_memcpy(&ba, &a, 2);
    __builtin_memcpy(&bb, &b, 2);
    return (unsigned)ba | ((unsigned)bb << 16);
}

__device__ __forceinline__ float grim_fdot2_f32_f16(unsigned a, unsigned b, float c) {
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_fdot2)
    return __builtin_amdgcn_fdot2((__attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16)a,
                                  (__attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16)b,
                                  c, false);
#else
    float res = c;
    asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(res) : "v"(a), "v"(b));
    return res;
#endif
}

__device__ __forceinline__ float grim_fdot2_f32_bf16(unsigned a, unsigned b, float c) {
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_fdot2_f32_bf16)
    return __builtin_amdgcn_fdot2_f32_bf16(
        (__attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short)a,
        (__attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short)b,
        c, false);
#else
    // Software fallback for architectures without dot12-insts
    unsigned int a0_bits = ((unsigned int)(a & 0xFFFF)) << 16;
    unsigned int a1_bits = ((unsigned int)(a >> 16)) << 16;
    unsigned int b0_bits = ((unsigned int)(b & 0xFFFF)) << 16;
    unsigned int b1_bits = ((unsigned int)(b >> 16)) << 16;
    float a0, a1, b0, b1;
    __builtin_memcpy(&a0, &a0_bits, 4);
    __builtin_memcpy(&a1, &a1_bits, 4);
    __builtin_memcpy(&b0, &b0_bits, 4);
    __builtin_memcpy(&b1, &b1_bits, 4);
    return fmaf(a0, b0, fmaf(a1, b1, c));
#endif
}

__device__ __forceinline__ unsigned short grim_float_to_bf16_rn(float f) {
    unsigned int x;
    __builtin_memcpy(&x, &f, 4);
    unsigned int lsb = (x >> 16) & 1;
    unsigned int rounding = 0x7fff + lsb;
    x += rounding;
    return (unsigned short)(x >> 16);
}

// ─── Phase 4.5d: BF16 × BF16 dot2 GEMV (V_DOT2_F32_BF16, dot12-insts) ─────
// B is row-major/transposed natural weights [N, K] as BF16 (2 bytes/weight).
// A is activation [M, K] as BF16 (2 bytes/act).
// Accumulates natively in FP32 with 2 BF16 products per instruction, outputs BF16.
extern "C" __global__ void grim_dot2_bf16_gemv(
    const unsigned short* __restrict__ A,
    const unsigned short* __restrict__ B,
    unsigned short* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_pairs = K / 2;
    const unsigned* a_row = (const unsigned*)(A + (long long)row * K);

    const unsigned* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? (const unsigned*)(B + (long long)(col_base + j) * K)
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int p = lane; p < n_pairs; p += 32) {
        unsigned a2 = a_row[p];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            unsigned b2 = b_col[j][p];
            facc[j] = grim_fdot2_f32_bf16(a2, b2, facc[j]);
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = grim_float_to_bf16_rn(facc[j]);
        }
    }
}

extern "C" __global__ void grim_dot2_q80_gemv(
    const unsigned short* __restrict__ act_f16,
    const unsigned char*  __restrict__ B_q80,
    float* __restrict__ C,
    int N, int K)
{
    const int col  = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= N) return;
    const int blocks_per_row = K / 32;
    const int row_bytes = blocks_per_row * 34;
    const unsigned char* row_ptr = B_q80 + (long long)col * row_bytes;
    const unsigned* act2 = (const unsigned*)act_f16;
    float facc = 0.0f;
    for (int blk = lane; blk < blocks_per_row; blk += 32) {
        float d = fp16_to_float_device(((const unsigned short*)(row_ptr + blk * 34))[0]);
        const unsigned char* cp = row_ptr + blk * 34 + 2;
        float bacc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            unsigned b4 = (unsigned)cp[j] | ((unsigned)cp[j+1] << 8)
                        | ((unsigned)cp[j+2] << 16) | ((unsigned)cp[j+3] << 24);
            float c0 = (float)(int)(signed char)(b4 & 0xFF);
            float c1 = (float)(int)(signed char)((b4 >>  8) & 0xFF);
            float c2 = (float)(int)(signed char)((b4 >> 16) & 0xFF);
            float c3 = (float)(int)(signed char)((b4 >> 24) & 0xFF);
            unsigned a01 = act2[(blk * 32 + j) >> 1];
            unsigned w01 = grim_pk_f16(c0, c1);
            bacc = grim_fdot2_f32_f16(a01, w01, bacc);
            unsigned a23 = act2[(blk * 32 + j + 2) >> 1];
            unsigned w23 = grim_pk_f16(c2, c3);
            bacc = grim_fdot2_f32_f16(a23, w23, bacc);
        }
        facc += bacc * d;
    }
    for (int off = 16; off > 0; off >>= 1)
        facc += __shfl_xor(facc, off);
    if (lane == 0)
        C[col] = facc;
}

#define GRIM_Q4K_SUPERBLOCK_SIZE 256
#define GRIM_Q4K_BYTES           144

// ─── Phase 4.5c: FP8 E4M3 × FP8 E4M3 dot4 GEMV (dot11-insts, RDNA4) ───────
// B is column-major E4M3 [N, K] (one byte per weight); A is f32 [M, K] and is
// quantized to E4M3 in registers (RNE) before packing. Four products per
// V_DOT4_F32_FP8_FP8 with f32 accumulation.

__device__ __forceinline__ float grim_fdot4_fp8(float c, int a, int b) {
// Builtin name is `dot4_*`, NOT `fdot4_*` — there is no `fdot4` builtin in any
// ROCm clang. The old spelling made __has_builtin false, so this helper silently
// ran the #else scalar branch on gfx1201 (which has the native instruction) and
// paid a powf() per E4M3 element for four products.
//
// ARGUMENT ORDER is (packed_a, packed_b, f32_addend) -> f32, i.e. the two packed
// FP8 operands come FIRST and the accumulator LAST. BuiltinsAMDGPU.def records
// this as "fUiUif", which reads like (f, u, u) but lists the RETURN type first:
// the real params are (u, i, u, i, f) = a, a_neg, b, b_neg, acc. Passing
// (c, a, b) compiles cleanly, emits a correct-looking
// `v_dot4_f32_fp8_fp8 v, 0, b, c` with `a` constant-folded to literal 0, and
// silently discards every A contribution — parity diverges by ~1e10. Verified
// numerically on gfx1201 before and after.
//
// Guarded by builtin_name_oracle_tests (names) and
// tests/jit_instruction_emit.rs (emission) plus dot_gemv_parity (numerics).
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_dot4_f32_fp8_fp8)
    return __builtin_amdgcn_dot4_f32_fp8_fp8((unsigned int)a, (unsigned int)b, c);
#else
    // Portable fallback for gfx10/gfx11, which lack dot11-insts. Unpack 4 E4M3
    // codes per operand and fma in f32. Correct but far slower than the
    // instruction it stands in for — do not "optimise" this into being the
    // gfx12 path.
    float acc = c;
    for (int e = 0; e < 4; ++e) {
        float av = fp8_e4m3_to_float_hip((unsigned char)((a >> (8 * e)) & 0xFF));
        float bv = fp8_e4m3_to_float_hip((unsigned char)((b >> (8 * e)) & 0xFF));
        acc = fmaf(av, bv, acc);
    }
    return acc;
#endif
}

// f32 -> E4M3 (RNE). Mirrored in Rust by the GPU parity test.
__device__ __forceinline__ unsigned char grim_f32_to_fp8_e4m3(float f) {
    if (__builtin_isnan(f)) return 0x7F;
    unsigned sign = __builtin_signbit(f) ? 0x80u : 0x00u;
    float a = __builtin_fabsf(f);
    // E4M3 has no infinities: exp=15,mant=7 (0x7F) is NaN and the largest
    // finite value is exp=15,mant=6 (0x7E) = 1.75*2^8 = 448. Saturate at 448.
    //
    // This guard used to be 480.0, which is off by exactly one mantissa step.
    // Any f32 in [448,480) has exponent field E=15, so its 3-bit mantissa q
    // runs 0..7 -- and q==7 lands on 0x7F, the NaN slot. f32 in [464.01,480)
    // therefore encoded to NaN, which then propagated through the f32 dot as
    // NaN rather than as a saturated value. 448 is the correct threshold: the
    // first f32 that would carry q past 6 is 464, well inside it.
    if (__builtin_isinf(a) || a >= 448.0f) return (unsigned char)(sign | 0x7E); // saturate to 448
    unsigned bits;
    __builtin_memcpy(&bits, &a, 4);
    unsigned m = bits & 0x7FFFFFu;
    int e = (int)((bits >> 23) & 0xFFu);
    if (e == 0) return (unsigned char)sign; // f32 subnormals are below E4M3 min
    int E = e - 120; // E4M3 exponent (bias 7), 1.m23 * 2^(e-127) = 1.m3 * 2^(E-7)
    if (E >= 1) {
        // Normal: round 23-bit fraction to 3 bits, RNE.
        unsigned q = m >> 20;
        unsigned r = m & 0xFFFFFu;
        if (r > 0x80000u || (r == 0x80000u && (q & 1u))) q++;
        if (q == 8u) { q = 0; E++; }
        if (E > 15) return (unsigned char)(sign | 0x7E); // overflow -> 448
        return (unsigned char)(sign | ((unsigned)E << 3) | q);
    }
    // Subnormal: value * 512 = 1.m23 * 2^(E+2); round to 3-bit integer code.
    int sh = 21 - E;
    unsigned mant = 0x800000u | m;
    unsigned q = mant >> sh;
    unsigned r = mant & ((1u << sh) - 1u);
    unsigned half = 1u << (sh - 1);
    if (r > half || (r == half && (q & 1u))) q++;
    if (q == 0) return (unsigned char)sign;
    return (unsigned char)(sign | q);
}

__device__ __forceinline__ int grim_pack4_fp8(
    const float* __restrict__ src) {
    unsigned b0 = grim_f32_to_fp8_e4m3(src[0]);
    unsigned b1 = grim_f32_to_fp8_e4m3(src[1]);
    unsigned b2 = grim_f32_to_fp8_e4m3(src[2]);
    unsigned b3 = grim_f32_to_fp8_e4m3(src[3]);
    return (int)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}


extern "C" __global__ void grim_dot4_fp8_gemv(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_chunks = K / 32;
    const float* a_row = A + (long long)row * K;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_fp8 + (long long)(col_base + j) * K
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each lane processes one 32-element chunk per stride (same tiling as the
    // Q8_0 dot4 GEMV): 8 fp8 dot4s per chunk.
    for (int chunk = lane; chunk < n_chunks; chunk += 32) {
        const float* a_chunk = a_row + (long long)chunk * 32;

        // WS-C1: pack the 32 activations ONCE per chunk. This used to sit inside
        // the `j` loop below, so the f32->E4M3 ladder ran 4x per group of four —
        // sixteen conversions per chunk to produce four identical words. The
        // words depend only on `chunk`, never on `j`. Costs 8 VGPRs.
        int a4_words[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            a4_words[e] = grim_pack4_fp8(a_chunk + e * 4);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_chunk = b_col[j] + (long long)chunk * 32;
            float acc = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int b4;
                __builtin_memcpy(&b4, b_chunk + e * 4, 4);
                acc = grim_fdot4_fp8(acc, a4_words[e], b4);
            }
            facc[j] += acc;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}


// ─── Phase 4.5f: Q2_K dot4 GEMV (two-dot decomposition) ───────────────────
// llama.cpp Q2_K layout (matches grim_quant::dequant_q2k, ggml-quants.c:959):
// 84 bytes per 256 weights: 16 scale bytes (4-bit sc + 4-bit m per 16-elem
// sub-block), 64 code bytes INTERLEAVED — byte qs[j/4 + l] holds one 2-bit
// code for each of the four 32-weight runs (fields 0/2/4/6 = runs j=0..3),
// f16 d @80, f16 dmin @82. value = d*sc*q - dmin*m. Two-dot:
// dot = d*sc*Σ(a_i*q_i) - dmin*m*Σa_i. Activations are Q8_1.
// (Was a sequential 4-bytes-per-sub-block layout that exists in no released
// GGUF; fixed 2026-10-04 alongside the new quant_q2k encoder.)

__device__ __forceinline__ int grim_expand2_q2k(unsigned char b) {
    return (int)((b & 3u) | ((b >> 2 & 3u) << 8) | ((b >> 4 & 3u) << 16)
               | ((b >> 6 & 3u) << 24));
}

// Extract the 2-bit code at field `shift` from 4 consecutive INTERLEAVED
// code bytes into the packed 4-code form grim_sdot4 consumes.
__device__ __forceinline__ int grim_extract4_q2k_ilv(const unsigned char* p, int shift) {
    return (int)(((p[0] >> shift) & 3u) | (((p[1] >> shift) & 3u) << 8)
               | (((p[2] >> shift) & 3u) << 16) | (((p[3] >> shift) & 3u) << 24));
}

extern "C" __global__ void grim_dot4_q2k_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q2k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q2k + (long long)(col_base + j) * n_sb * 84
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane; sb < n_sb; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * 84;
            float d    = fp16_to_float_device(((const unsigned short*)b_sb)[40]); // byte 80
            float dmin = fp16_to_float_device(((const unsigned short*)b_sb)[41]); // byte 82

            // 16 sub-blocks of 16 elems = 8 q8_1 blocks' worth, pairwise.
            for (int half_pair = 0; half_pair < 8; half_pair++) {
                const unsigned char* a_blk = a_sb + half_pair * GRIM_Q8_1_BYTES;
                float d_a   = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                for (int half = 0; half < 2; half++) {
                    const int sub = half_pair * 2 + half;
                    const float sc = (float)(b_sb[sub] & 0x0F);
                    const float mi = (float)(b_sb[sub] >> 4);

                    // Interleaved codes: sub-block `sub` covers weights
                    // n = (sub/8)*128, run j = (sub%8)/2, half-plane
                    // sub%2; code for weight l=i lives at field 2*j of
                    // byte qs[(sub/8)*32 + i + (sub%2)*16].
                    const int nh = sub / 8;
                    const int run = (sub % 8) / 2;
                    const int hplane = sub % 2;
                    const unsigned char* qs_ilv = b_sb + 16 + nh * 32 + hplane * 16;
                    const int shift = 2 * run;
                    int pos = 0;
                    int a_sum = 0;
                    #pragma unroll
                    for (int i = 0; i < 16; i += 4) {
                        int a4;
                        __builtin_memcpy(&a4, a_codes + half * 16 + i, 4);
                        int q4 = grim_extract4_q2k_ilv(qs_ilv + i, shift);
                        pos = grim_sdot4(a4, q4, pos);
                        // The dmin correction needs the sum over THIS half's
                        // 16 codes: each sub-block covers half a q8_1 block,
                        // and the stored f16 sum spans all 32 codes. Using it
                        // double-counted (once per half) and mixed halves.
                        a_sum += a_codes[half * 16 + i];
                        a_sum += a_codes[half * 16 + i + 1];
                        a_sum += a_codes[half * 16 + i + 2];
                        a_sum += a_codes[half * 16 + i + 3];
                    }
                    // Both terms scale the raw codes by d_a: the q8_1 f16
                    // sum_a is in code units, so using it here (besides
                    // spanning 32 codes) left the dmin term unscaled.
                    facc[j] += d * sc * ((float)pos * d_a)
                             - dmin * mi * ((float)a_sum * d_a);
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}



// ─── Q2_K dot4 GEMV: word-wide-mask code unpack (res8 P1) ────────────────
//
// The stock Q2_K kernel above fetched the four 2-bit codes of a 4-weight
// group with FOUR SEPARATE 1-BYTE LOADS (grim_extract4_q2k_ilv) and then
// re-assembled them into a 32-bit word with four shifts and three ORs. The
// Q2_K block stores one code per BYTE (84 B / 256 weights, qs[64] at
// offset 16), so the four codes of a group are four CONSECUTIVE bytes and
// one 32-bit load delivers all of them.
//
// The mask is exact: the codes are the 2-bit field at bit `shift` of each
// byte, shift = 2*run in {0,2,4,6}. After `w >> shift`, each byte's low
// 2 bits still come from its own byte (the bits shifted in from the NEXT
// byte land at bit 8-shift >= 2 and are masked off by 0x03030303), so
//
//     (w >> shift) & 0x03030303  ==  the stock per-byte extraction
//
// bit for bit, for every shift in {0,2,4,6}. This is the SAME mask the
// landed group hoist uses for the Q5_K `qh` field.
//
// ALIASING: the dword is 4-byte aligned by construction. The offset of a
// group from the tensor base is col*n_sb*84 + sb*84 + 16 + nh*32 +
// hplane*16 + i, every term a multiple of 4 (84 = 4*21, and i steps by 4),
// and the tensor base is a device allocation (256-byte aligned). The load is
// a __builtin_memcpy of 4 bytes, which is alignment-safe regardless.
//
// The grid mapping (4 output columns per workgroup, 32 lanes per block),
// the accumulation order and the arithmetic are IDENTICAL to the stock
// kernel, so the only difference is how the codes are read. Parity:
// tests/dot4_kquant_grouped_parity.rs (q2k row), byte-identity against the
// retained stock kernel AND against the llama.cpp-faithful dequant_q2k
// dotted with the same q8_1 codes.
extern "C" __global__ void grim_dot4_q2k_q81_gemv_fast(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q2k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q2k + (long long)(col_base + j) * n_sb * 84
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane; sb < n_sb; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * 84;
            float d    = fp16_to_float_device(((const unsigned short*)b_sb)[40]); // byte 80
            float dmin = fp16_to_float_device(((const unsigned short*)b_sb)[41]); // byte 82

            for (int half_pair = 0; half_pair < 8; half_pair++) {
                const unsigned char* a_blk = a_sb + half_pair * GRIM_Q8_1_BYTES;
                float d_a   = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                for (int half = 0; half < 2; half++) {
                    const int sub = half_pair * 2 + half;
                    const float sc = (float)(b_sb[sub] & 0x0F);
                    const float mi = (float)(b_sb[sub] >> 4);

                    const int nh = sub / 8;
                    const int run = (sub % 8) / 2;
                    const int hplane = sub % 2;
                    const unsigned char* qs_ilv = b_sb + 16 + nh * 32 + hplane * 16;
                const int shift = 2 * run;
                int pos = 0;
                int a_sum = 0;
                #pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    int a4;
                    __builtin_memcpy(&a4, a_codes + half * 16 + i, 4);
                    // ONE 32-bit load for the four codes, then one whole-word
                    // mask, instead of four 1-byte loads and four
                    // select/shift/pack steps. Bit-identical by
                    // construction (see the note above the kernel).
                    unsigned int w4;
                    __builtin_memcpy(&w4, qs_ilv + i, 4);
                    int q4 = (int)((w4 >> shift) & 0x03030303u);
                    pos = grim_sdot4(a4, q4, pos);
                    a_sum += a_codes[half * 16 + i];
                    a_sum += a_codes[half * 16 + i + 1];
                    a_sum += a_codes[half * 16 + i + 2];
                    a_sum += a_codes[half * 16 + i + 3];
                }
                facc[j] += d * sc * ((float)pos * d_a)
                         - dmin * mi * ((float)a_sum * d_a);
            }
        }
    }
}

#pragma unroll
for (int j = 0; j < 4; j++) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        facc[j] += __shfl_xor(facc[j], off);
}

if (lane == 0) {
    #pragma unroll
    for (int j = 0; j < active_cols; j++) {
        C[(long long)row * N + col_base + j] = facc[j];
    }
}
}

// Step 2.1: Q2_K sub-block-per-lane GEMV (lane>>3 = super-block slot,
// lane&7 = 32-weight sub-block). 84 B super-block stride (4 B aligned),
// qs at +16. Each lane owns one 32-weight sub-block (two 16-weight halves
// `2*sub` and `2*sub + 1`). Tolerance-gated vs the stock kernel.
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q2k_q81_gemv_sub(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q2k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub   = lane & 7;
    const int slot  = lane >> 3;
    const int nh    = sub >> 2;
    const int run   = sub & 3;
    const int shift = 2 * run;
    const int sub16 = 2 * sub;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * 84;
    const unsigned char* b0 = B_q2k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = slot; sb < n_sb; sb += 4) {
        const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
        const float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        int a4[8];
        __builtin_memcpy(a4, a_blk + 4, 32);
        int asum0 = 0, asum1 = 0;
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            asum0 = grim_sdot4(a4[w], 0x01010101, asum0);
            asum1 = grim_sdot4(a4[w + 4], 0x01010101, asum1);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * 84;
            const float d    = fp16_to_float_device(((const unsigned short*)b_sb)[40]);
            const float dmin = fp16_to_float_device(((const unsigned short*)b_sb)[41]);

            const unsigned char b_sc0 = b_sb[sub16];
            const unsigned char b_sc1 = b_sb[sub16 + 1];
            const float sc0 = (float)(b_sc0 & 0x0F);
            const float mi0 = (float)(b_sc0 >> 4);
            const float sc1 = (float)(b_sc1 & 0x0F);
            const float mi1 = (float)(b_sc1 >> 4);

            const unsigned* qsw = (const unsigned*)(b_sb + 16 + nh * 32);
            int pos0 = 0, pos1 = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const int q4 = (int)((qsw[w] >> shift) & 0x03030303u);
                if (w < 4) pos0 = grim_sdot4(a4[w], q4, pos0);
                else       pos1 = grim_sdot4(a4[w], q4, pos1);
            }
            facc[j] += d_a * (d * (sc0 * (float)pos0 + sc1 * (float)pos1)
                            - dmin * (mi0 * (float)asum0 + mi1 * (float)asum1));
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

// Step 2.2 (#1 + #2): Q2_K sub-block-per-lane GEMV v2:
//   (#1) 4-column load batching before the sudot4 loop.
//   (#2) De-duplicated 16 B qs loads across even/odd sub-lanes (`half = sub&1`
//        loads `qsw[4*half .. 4*half+3]` and exchanges with `lane ^ 1`),
//        plus 32-bit (d, dmin) and 16-bit scale pair loads.
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q2k_q81_gemv_sub_v2(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q2k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub   = lane & 7;
    const int slot  = lane >> 3;
    const int nh    = sub >> 2;
    const int run   = sub & 3;
    const int half  = sub & 1;
    const int shift = 2 * run;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * 84;
    const unsigned char* b0 = B_q2k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb_base = 0; sb_base < n_sb; sb_base += 4) {
        const int sb = sb_base + slot;
        const bool sb_valid = (sb < n_sb);
        float d_a = 0.0f;
        int a4[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int asum0 = 0, asum1 = 0;
        if (sb_valid) {
            const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
            d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
            __builtin_memcpy(a4, a_blk + 4, 32);
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                asum0 = grim_sdot4(a4[w], 0x01010101, asum0);
                asum1 = grim_sdot4(a4[w + 4], 0x01010101, asum1);
            }
        }

        unsigned hdr_v[4];
        unsigned short sc_v[4];
        unsigned q_own_v[4][4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (sb_valid && j < active_cols) {
                const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * 84;
                hdr_v[j] = ((const unsigned*)b_sb)[20];
                sc_v[j]  = ((const unsigned short*)b_sb)[sub];
                const unsigned* qsw = (const unsigned*)(b_sb + 16 + nh * 32 + half * 16);
                q_own_v[j][0] = qsw[0];
                q_own_v[j][1] = qsw[1];
                q_own_v[j][2] = qsw[2];
                q_own_v[j][3] = qsw[3];
            } else {
                hdr_v[j] = 0u;
                sc_v[j]  = 0u;
                q_own_v[j][0] = 0u;
                q_own_v[j][1] = 0u;
                q_own_v[j][2] = 0u;
                q_own_v[j][3] = 0u;
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const float d    = fp16_to_float_device((unsigned short)(hdr_v[j] & 0xFFFFu));
            const float dmin = fp16_to_float_device((unsigned short)(hdr_v[j] >> 16));
            const unsigned sc_pair = (unsigned)sc_v[j];
            const float sc0 = (float)(sc_pair & 0x0Fu);
            const float mi0 = (float)((sc_pair >> 4) & 0x0Fu);
            const float sc1 = (float)((sc_pair >> 8) & 0x0Fu);
            const float mi1 = (float)((sc_pair >> 12) & 0x0Fu);

            unsigned q_part[4];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                q_part[u] = (unsigned)__shfl_xor((int)q_own_v[j][u], 1);
            }
            const unsigned* q0 = half ? q_part : q_own_v[j];
            const unsigned* q1 = half ? q_own_v[j] : q_part;
            const unsigned qw[8] = {q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]};

            int pos0 = 0, pos1 = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const int q4 = (int)((qw[w] >> shift) & 0x03030303u);
                if (w < 4) pos0 = grim_sdot4(a4[w], q4, pos0);
                else       pos1 = grim_sdot4(a4[w], q4, pos1);
            }
            if (sb_valid) {
                facc[j] += d_a * (d * (sc0 * (float)pos0 + sc1 * (float)pos1)
                                - dmin * (mi0 * (float)asum0 + mi1 * (float)asum1));
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}


// ─── Phase 4.5f: Q3_K dot4 GEMV (two-dot + sign-plane correction) ────────
// Grim Q3_K layout (matches grim_quant::dequant_q3k, llama.cpp spec):
// 110 bytes per 256: hmask[32] (sign bits), qs[64] (2-bit magnitudes),
// scales[12] (ggml 6-bit shuffle -> 16 i8, used as sc-32), f16 d @108.
// value_i = dl * (q_i - hm_i*4), dl = d*(sc-32). Two-dot:
//   dot = dl*Σ(a_i*q_i) - dl*4*Σ_{hm applies} a_i.
// Activations are Q8_1: a_i = code_i*d_a.

extern "C" __global__ void grim_dot4_q3k_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q3k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q3k + (long long)(col_base + j) * n_sb * 110
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int ones4 = 0x01010101;

    for (int sb = lane; sb < n_sb; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * 110;
            const unsigned char* hmask = b_sb;
            const unsigned char* qs    = b_sb + 32;
            const unsigned char* scales = b_sb + 96;
            float d = fp16_to_float_device(((const unsigned short*)b_sb)[54]); // byte 108

            // ggml 12-byte scale shuffle -> 16 i8 (matches dequant_q3k exactly).
            int sc[16];
            {
                unsigned aux0 = (unsigned)scales[0] | ((unsigned)scales[1] << 8)
                              | ((unsigned)scales[2] << 16) | ((unsigned)scales[3] << 24);
                unsigned aux1 = (unsigned)scales[4] | ((unsigned)scales[5] << 8)
                              | ((unsigned)scales[6] << 16) | ((unsigned)scales[7] << 24);
                unsigned tmp  = (unsigned)scales[8] | ((unsigned)scales[9] << 8)
                              | ((unsigned)scales[10] << 16) | ((unsigned)scales[11] << 24);
                unsigned kmask1 = 0x03030303u;
                unsigned kmask2 = 0x0F0F0F0Fu;
                unsigned a0 = (aux0 & kmask2) | ((tmp & kmask1) << 4);
                unsigned a1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
                unsigned a2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                unsigned a3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                unsigned auxs[4] = { a0, a1, a2, a3 };
                for (int t = 0; t < 4; ++t)
                    for (int b2 = 0; b2 < 4; ++b2) {
                        unsigned v = (auxs[t] >> (8 * b2)) & 0xFF;
                        sc[t * 4 + b2] = (int)(v >= 128u ? (int)v - 256 : (int)v);
                    }
            }

            unsigned char m_bit = 1;
            int _is = 0;
            for (int half = 0; half < 2; half++) {
                int shift = 0;
                for (int j4 = 0; j4 < 4; j4++) {
                    // One 32-elem q8_1 activation block per (half, j4).
                    const unsigned char* a_blk = a_sb + (half * 4 + j4) * GRIM_Q8_1_BYTES;
                    float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                    const signed char* codes = (const signed char*)(a_blk + 4);

                    for (int h2 = 0; h2 < 2; h2++) {
                        const float dl = d * (float)(sc[_is] - 32);
                        _is += 1;
                        int pos = 0;
                        int corr = 0;
                        #pragma unroll
                        for (int i = 0; i < 16; i += 4) {
                            int a4;
                            __builtin_memcpy(&a4, codes + h2 * 16 + i, 4);
                            int q4 = 0;
                            int mask4 = 0;
                            #pragma unroll
                            for (int e2 = 0; e2 < 4; ++e2) {
                                int l = h2 * 16 + i + e2;
                                int q_off = half * 32;
                                int code = (qs[q_off + l] >> shift) & 3;
                                q4 |= code << (8 * e2);
                                mask4 |= ((hmask[l] & m_bit) != 0) ? 0 : (0xFF << (8 * e2));
                            }
                            pos = grim_sdot4(a4, q4, pos);
                            corr = grim_sdot4(a4 & mask4, ones4, corr);
                        }
                        facc[j] += dl * d_a * ((float)pos - 4.0f * (float)corr);
                    }
                    shift += 2;
                    m_bit = (unsigned char)((m_bit << 1) & 0xFF);
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

// ─── Q3_K dot4 GEMV: word-wide-mask magnitude plane (res8 P5) ─────────────
// The stock Q3_K kernel above fetches each 2-bit magnitude code with its
// own 1-byte load. Q3_K's qs[] is ONE code per BYTE (hmask[32] | qs[64] at
// offset 32), so the four codes of a 4-weight group are four consecutive
// bytes: one 32-bit load plus `(w >> shift) & 0x03030303` delivers all
// four, bit-identically, for every shift in {0,2,4,6} — the same mask and
// the same argument as the Q2_K fast kernel. The SIGN plane (hmask) is one
// bit spread across bytes at byte-granular positions and cannot be masked
// word-wide; it keeps the per-byte loads. Grid mapping, accumulation order
// and arithmetic are identical to the stock kernel. Parity:
// tests/dot4_kquant_grouped_parity.rs (q3k row).
extern "C" __global__ void grim_dot4_q3k_q81_gemv_fast(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q3k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;
    if (row >= M || col_base >= N) return;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q3k + (long long)(col_base + j) * n_sb * 110
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int ones4 = 0x01010101;

    for (int sb = lane; sb < n_sb; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * 110;
            const unsigned char* hmask = b_sb;
            const unsigned char* qs    = b_sb + 32;
            const unsigned char* scales = b_sb + 96;
            float d = fp16_to_float_device(((const unsigned short*)b_sb)[54]); // byte 108

            int sc[16];
            {
                unsigned aux0 = (unsigned)scales[0] | ((unsigned)scales[1] << 8)
                              | ((unsigned)scales[2] << 16) | ((unsigned)scales[3] << 24);
                unsigned aux1 = (unsigned)scales[4] | ((unsigned)scales[5] << 8)
                              | ((unsigned)scales[6] << 16) | ((unsigned)scales[7] << 24);
                unsigned tmp  = (unsigned)scales[8] | ((unsigned)scales[9] << 8)
                              | ((unsigned)scales[10] << 16) | ((unsigned)scales[11] << 24);
                unsigned kmask1 = 0x03030303u;
                unsigned kmask2 = 0x0F0F0F0Fu;
                unsigned a0 = (aux0 & kmask2) | ((tmp & kmask1) << 4);
                unsigned a1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
                unsigned a2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                unsigned a3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                unsigned auxs[4] = { a0, a1, a2, a3 };
                for (int t = 0; t < 4; ++t)
                    for (int b2 = 0; b2 < 4; ++b2) {
                        unsigned v = (auxs[t] >> (8 * b2)) & 0xFF;
                        sc[t * 4 + b2] = (int)(v >= 128u ? (int)v - 256 : (int)v);
                    }
            }

            unsigned char m_bit = 1;
            int _is = 0;
            for (int half = 0; half < 2; half++) {
                int shift = 0;
                for (int j4 = 0; j4 < 4; j4++) {
                    const unsigned char* a_blk = a_sb + (half * 4 + j4) * GRIM_Q8_1_BYTES;
                    float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                    const signed char* codes = (const signed char*)(a_blk + 4);

                    for (int h2 = 0; h2 < 2; h2++) {
                        const float dl = d * (float)(sc[_is] - 32);
                        _is += 1;
                        int pos = 0;
                int corr = 0;
                #pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    int a4;
                    __builtin_memcpy(&a4, codes + h2 * 16 + i, 4);
                    // ONE 32-bit load for the four 2-bit magnitudes,
                    // then one whole-word mask, instead of four
                    // 1-byte loads and four select/shift/steps.
                    // Bit-identical for every shift in {0,2,4,6}
                    // (see the note above the kernel).
                    unsigned int w4;
                    __builtin_memcpy(&w4, qs + half * 32 + h2 * 16 + i, 4);
                    int q4 = (int)((w4 >> shift) & 0x03030303u);
                    int mask4 = 0;
                    #pragma unroll
                    for (int e2 = 0; e2 < 4; ++e2) {
                        int l = h2 * 16 + i + e2;
                        mask4 |= ((hmask[l] & m_bit) != 0) ? 0 : (0xFF << (8 * e2));
                    }
                    pos = grim_sdot4(a4, q4, pos);
                    corr = grim_sdot4(a4 & mask4, ones4, corr);
                }
                facc[j] += dl * d_a * ((float)pos - 4.0f * (float)corr);
            }
            shift += 2;
            m_bit = (unsigned char)((m_bit << 1) & 0xFF);
        }
    }
}
}

#pragma unroll
for (int j = 0; j < 4; j++) {
#pragma unroll
for (int off = 16; off > 0; off >>= 1)
    facc[j] += __shfl_xor(facc[j], off);
}

if (lane == 0) {
#pragma unroll
for (int j = 0; j < active_cols; j++) {
    C[(long long)row * N + col_base + j] = facc[j];
}
}
}

// Step 2.1: Q3_K sub-block-per-lane GEMV (lane>>3 = super-block slot,
// lane&7 = 32-weight sub-block). 110 B super-block stride -> 2 B aligned
// `grim_u32_a2` loads. Both magnitude (qs) and sign bit 2 (hmask) are
// unpacked word-wide into `q4` in [0..7], then the -4 offset is folded in
// int32 via `pos - 4*asum`, cutting `sudot4` calls in half vs `_fast`.
typedef unsigned int __attribute__((aligned(2))) grim_u32_a2;
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q3k_q81_gemv_sub(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q3k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub   = lane & 7;
    const int slot  = lane >> 3;
    const int half  = sub >> 2;
    const int j4    = sub & 3;
    const int shift = 2 * j4;
    const int is0   = 2 * sub;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * 110;
    const unsigned char* b0 = B_q3k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = slot; sb < n_sb; sb += 4) {
        const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
        const float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        int a4[8];
        __builtin_memcpy(a4, a_blk + 4, 32);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * 110;
            const grim_u32_a2* hmw = (const grim_u32_a2*)b_sb;
            const grim_u32_a2* qsw = (const grim_u32_a2*)(b_sb + 32 + half * 32);
            const unsigned char* scales = b_sb + 96;
            const float d = fp16_to_float_device(((const unsigned short*)b_sb)[54]);

            const unsigned s0_lo = scales[is0 & 7];
            const unsigned s1_lo = scales[(is0 + 1) & 7];
            const unsigned s0_hi = (scales[8 + (is0 & 3)] >> (4 * half + 2 * ((is0 >> 2) & 1))) & 0x03u;
            const unsigned s1_hi = (scales[8 + ((is0 + 1) & 3)] >> (4 * half + 2 * (((is0 + 1) >> 2) & 1))) & 0x03u;
            const unsigned v0 = (half ? ((s0_lo >> 4) & 0x0Fu) : (s0_lo & 0x0Fu)) | (s0_hi << 4);
            const unsigned v1 = (half ? ((s1_lo >> 4) & 0x0Fu) : (s1_lo & 0x0Fu)) | (s1_hi << 4);
            const int sc0 = (int)(v0 >= 128u ? (int)v0 - 256 : (int)v0) - 32;
            const int sc1 = (int)(v1 >= 128u ? (int)v1 - 256 : (int)v1) - 32;

            int pos0 = 0, pos1 = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const unsigned qw = qsw[w];
                const unsigned hw = hmw[w];
                const unsigned mag = (qw >> shift) & 0x03030303u;
                const unsigned hbit = (((~hw) >> sub) & 0x01010101u) * 0xFCu;
                const int q4 = (int)(mag | hbit);
                if (w < 4) pos0 = grim_sdot4(a4[w], q4, pos0);
                else       pos1 = grim_sdot4(a4[w], q4, pos1);
            }
            facc[j] += d * d_a * ((float)sc0 * (float)pos0 + (float)sc1 * (float)pos1);
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

// Step 2.2 (#1 + #2): Q3_K sub-block-per-lane GEMV v2:
//   (#1) 4-column load batching before the sudot4 loop.
//   (#2) De-duplicated 16 B qs (`qsw + (sub&1)*4`) and 16 B hmask (`hmw + (sub&1)*4`)
//        loads exchanged between even/odd sub-lanes via `__shfl_xor(..., 1)`.
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q3k_q81_gemv_sub_v2(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q3k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub   = lane & 7;
    const int slot  = lane >> 3;
    const int half  = sub >> 2;
    const int j4    = sub & 3;
    const int odd   = sub & 1;
    const int shift = 2 * j4;
    const int is0   = 2 * sub;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * 110;
    const unsigned char* b0 = B_q3k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb_base = 0; sb_base < n_sb; sb_base += 4) {
        const int sb = sb_base + slot;
        const bool sb_valid = (sb < n_sb);
        float d_a = 0.0f;
        int a4[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (sb_valid) {
            const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
            d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
            __builtin_memcpy(a4, a_blk + 4, 32);
        }

        unsigned short d_v[4];
        unsigned short sc_lo_v[4];
        unsigned short sc_hi_v[4];
        unsigned q_own_v[4][4];
        unsigned h_own_v[4][4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (sb_valid && j < active_cols) {
                const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * 110;
                d_v[j]     = ((const unsigned short*)b_sb)[54];
                sc_lo_v[j] = *(const unsigned short*)(b_sb + 96 + (is0 & 6));
                sc_hi_v[j] = *(const unsigned short*)(b_sb + 104 + (is0 & 2));
                const grim_u32_a2* hmw = (const grim_u32_a2*)(b_sb + odd * 16);
                const grim_u32_a2* qsw = (const grim_u32_a2*)(b_sb + 32 + half * 32 + odd * 16);
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    q_own_v[j][u] = qsw[u];
                    h_own_v[j][u] = hmw[u];
                }
            } else {
                d_v[j]     = 0u;
                sc_lo_v[j] = 0u;
                sc_hi_v[j] = 0u;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    q_own_v[j][u] = 0u;
                    h_own_v[j][u] = 0u;
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const float d = fp16_to_float_device(d_v[j]);
            const unsigned s0_lo = (unsigned)sc_lo_v[j] & 0xFFu;
            const unsigned s1_lo = (unsigned)sc_lo_v[j] >> 8;
            const unsigned s0_hi_b = (unsigned)sc_hi_v[j] & 0xFFu;
            const unsigned s1_hi_b = (unsigned)sc_hi_v[j] >> 8;
            const unsigned s0_hi = (s0_hi_b >> (4 * half + 2 * ((is0 >> 2) & 1))) & 0x03u;
            const unsigned s1_hi = (s1_hi_b >> (4 * half + 2 * (((is0 + 1) >> 2) & 1))) & 0x03u;
            const unsigned v0 = (half ? ((s0_lo >> 4) & 0x0Fu) : (s0_lo & 0x0Fu)) | (s0_hi << 4);
            const unsigned v1 = (half ? ((s1_lo >> 4) & 0x0Fu) : (s1_lo & 0x0Fu)) | (s1_hi << 4);
            const int sc0 = (int)(v0 >= 128u ? (int)v0 - 256 : (int)v0) - 32;
            const int sc1 = (int)(v1 >= 128u ? (int)v1 - 256 : (int)v1) - 32;

            unsigned q_part[4], h_part[4];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                q_part[u] = (unsigned)__shfl_xor((int)q_own_v[j][u], 1);
                h_part[u] = (unsigned)__shfl_xor((int)h_own_v[j][u], 1);
            }
            const unsigned* q0 = odd ? q_part : q_own_v[j];
            const unsigned* q1 = odd ? q_own_v[j] : q_part;
            const unsigned* h0 = odd ? h_part : h_own_v[j];
            const unsigned* h1 = odd ? h_own_v[j] : h_part;
            const unsigned qw[8] = {q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]};
            const unsigned hw[8] = {h0[0], h0[1], h0[2], h0[3], h1[0], h1[1], h1[2], h1[3]};

            int pos0 = 0, pos1 = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const unsigned mag = (qw[w] >> shift) & 0x03030303u;
                const unsigned hbit = (((~hw[w]) >> sub) & 0x01010101u) * 0xFCu;
                const int q4 = (int)(mag | hbit);
                if (w < 4) pos0 = grim_sdot4(a4[w], q4, pos0);
                else       pos1 = grim_sdot4(a4[w], q4, pos1);
            }
            if (sb_valid) {
                facc[j] += d * d_a * ((float)sc0 * (float)pos0 + (float)sc1 * (float)pos1);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

extern "C" __global__ void grim_dot4_q4k_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q4k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int n_q4k_superblocks = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q4k + (long long)(col_base + j) * n_q4k_superblocks * GRIM_Q4K_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each thread processes superblocks in strided fashion across the wave (lane, lane + 32, ...)
    for (int sb = lane; sb < n_q4k_superblocks; sb += 32) {
        // One Q4_K superblock corresponds to 8 Q8_1 activation blocks (256 elements)
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q4K_BYTES;

            const unsigned short* h_ptr = (const unsigned short*)b_sb;
            float d    = fp16_to_float_device(h_ptr[0]);
            float dmin = fp16_to_float_device(h_ptr[1]);

            const unsigned char* scales = b_sb + 4;
            const unsigned char* qs     = b_sb + 16;

            // 8 sub-blocks of 32 elements
            // REGISTER PRESSURE (2026-10-22). The full unroll of
            // 4 columns x 8 sub-blocks x 8 e-iterations compiled this
            // kernel to 192 VGPRs with 180 B of SCRATCH SPILL per thread
            // (rocprofv3 kernel-symbol dump). The Q4_K `_fast` sibling
            // reaches 72 VGPRs and 0 B of scratch, and the Q8_0 sibling
            // 64 VGPRs and 0 B. The spill puts a local-memory round trip
            // in the inner loop of a kernel that is NOT bandwidth-bound
            // (m1_decode_regime_probe: Q5_K cold ~= warm at every
            // working set), so the spill is the thing to remove.
            //
            // BIT-IDENTICAL by construction: the only thing that changes is
            // how far the compiler unrolls. The int32 accumulation
            // (`pos_dot = grim_sdot4(...)`) is an exact integer chain
            // whose ORDER is unchanged, and the float accumulation
            // (`facc[j] += sub_res`) still runs once per `is` in the same
            // order. Bounding the unroll cannot change either value.
            #pragma unroll
            for (int is = 0; is < 8; ++is) {
                const unsigned char* a_blk = a_sb + is * GRIM_Q8_1_BYTES;
                float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                unsigned char sc, m;
                if (is < 4) {
                    sc = scales[is] & 63;
                    m  = scales[is + 4] & 63;
                } else {
                    sc = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                    m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
                }

                // Sub-block is: group = is / 2 (0..3), half = is % 2 (0: low nibble, 1: high nibble)
                int group = is / 2;
                int half  = is % 2;
                const unsigned char* qs_sub = qs + group * 32;

                int pos_dot = 0;
                #pragma unroll
                for (int e = 0; e < 32; e += 4) {
                    int a4;
                    __builtin_memcpy(&a4, a_codes + e, 4);

                    // M7.2c (the IQ group-hoisting recipe): the four codes of a
                    // 4-weight group are four nibbles of ONE dword, not four
                    // 1-byte loads. `qs` is 4-byte aligned at every e here
                    // (qs at +16, 144-byte super-block stride), so the four
                    // bytes are one load. The 0x0F0F0F0F mask keeps each byte
                    // independent, so the word result is identical to the
                    // per-byte arithmetic it replaces.
                    unsigned qw4;
                    __builtin_memcpy(&qw4, qs_sub + e, 4);
                    int q4 = (int)(half ? ((qw4 >> 4) & 0x0F0F0F0Fu)
                                        : (qw4 & 0x0F0F0F0Fu));

                    pos_dot = grim_sdot4(a4, q4, pos_dot);
                }

                // Two-dot decomposition: value_i = d * sc * q_i - dmin * m
                // dot = sum(a_i * value_i) = d * sc * sum(a_i * q_i) - dmin * m * sum(a_i)
                float sub_res = d * (float)sc * ((float)pos_dot * d_a) - dmin * (float)m * sum_a;
                facc[j] += sub_res;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

// Step 2.1: Q4_K sub-block-per-lane GEMV. Maps 8 lanes to one 256-weight
// super-block (lane>>3 = super-block slot, lane&7 = 32-weight sub-block).
// Each lane loads 32 packed nibble bytes as two 16 B vectors (`uint4`) from
// `b_sb + 16 + (sub>>1)*32` (144 B stride, qs at +16 -> 16 B aligned).
// Tolerance-gated vs the stock kernel.
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q4k_q81_gemv_sub(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q4k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub   = lane & 7;
    const int slot  = lane >> 3;
    const int group = sub >> 1;
    const int half  = sub & 1;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * GRIM_Q4K_BYTES;
    const unsigned char* b0 = B_q4k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = slot; sb < n_sb; sb += 4) {
        const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
        const float d_a   = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        const float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
        int a4[8];
        __builtin_memcpy(a4, a_blk + 4, 32);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * GRIM_Q4K_BYTES;
            const unsigned short* h_ptr = (const unsigned short*)b_sb;
            const float d    = fp16_to_float_device(h_ptr[0]);
            const float dmin = fp16_to_float_device(h_ptr[1]);
            const unsigned char* scales = b_sb + 4;
            int sc, m;
            if (sub < 4) {
                sc = scales[sub] & 63;
                m  = scales[sub + 4] & 63;
            } else {
                sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
                m  = (scales[sub + 4] >> 4)   | ((scales[sub] >> 6) << 4);
            }
            const uint4* qs4 = (const uint4*)(b_sb + 16 + group * 32);
            const uint4 q0 = qs4[0], q1 = qs4[1];
            const unsigned qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            int pos = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const unsigned nib = half ? ((qw[w] >> 4) & 0x0F0F0F0Fu) : (qw[w] & 0x0F0F0F0Fu);
                pos = grim_sdot4(a4[w], (int)nib, pos);
            }
            facc[j] += d * (float)sc * ((float)pos * d_a) - dmin * (float)m * sum_a;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

// Step 2.2 (#1 + #2): Q4_K sub-block-per-lane GEMV v2 with:
//   (#1) 4-column load batching (all 4 columns' headers and 128 B qs lines are
//        issued in flight before the sudot4 compute loop, hiding DRAM latency).
//   (#2) De-duplicated 16 B qs loads across even/odd sub-lanes (`sub*16`
//        instead of `(sub>>1)*32`), cutting qs L0 cache traffic in half
//        (100% unique 128 B line per 8-lane sub-group) + 4-byte aligned scale
//        word unpacking instead of 1-byte loads.
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q4k_q81_gemv_sub_v2(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q4k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub  = lane & 7;
    const int slot = lane >> 3;
    const int half = sub & 1;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * GRIM_Q4K_BYTES;
    const unsigned char* b0 = B_q4k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb_base = 0; sb_base < n_sb; sb_base += 4) {
        const int sb = sb_base + slot;
        const bool sb_valid = (sb < n_sb);
        float d_a = 0.0f, sum_a = 0.0f;
        int a4[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (sb_valid) {
            const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
            d_a   = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
            sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
            __builtin_memcpy(a4, a_blk + 4, 32);
        }

        // (#1 + #2): Batch all 4 columns' 16 B header+scale loads and 16 B
        // unique qs loads (`b_sb + 16 + sub*16`) up front before compute.
        uint4 hdr_v[4];
        uint4 q_own_v[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (sb_valid && j < active_cols) {
                const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * GRIM_Q4K_BYTES;
                hdr_v[j]   = ((const uint4*)b_sb)[0];
                q_own_v[j] = *(const uint4*)(b_sb + 16 + sub * 16);
            } else {
                hdr_v[j]   = make_uint4(0u, 0u, 0u, 0u);
                q_own_v[j] = make_uint4(0u, 0u, 0u, 0u);
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned h0  = hdr_v[j].x;
            const unsigned sc0 = hdr_v[j].y;
            const unsigned sc1 = hdr_v[j].z;
            const unsigned sc2 = hdr_v[j].w;
            const float d    = fp16_to_float_device((unsigned short)(h0 & 0xFFFFu));
            const float dmin = fp16_to_float_device((unsigned short)(h0 >> 16));

            int sc, m;
            if (sub < 4) {
                const int sh = 8 * sub;
                sc = (int)((sc0 >> sh) & 63u);
                m  = (int)((sc1 >> sh) & 63u);
            } else {
                const int sh = 8 * (sub - 4);
                const unsigned b_hi = (sc2 >> sh) & 0xFFu;
                const unsigned b_lo = (sc0 >> sh) & 0xFFu;
                const unsigned b_mi = (sc1 >> sh) & 0xFFu;
                sc = (int)((b_hi & 0x0Fu) | ((b_lo >> 6) << 4));
                m  = (int)((b_hi >> 4)    | ((b_mi >> 6) << 4));
            }

            // Exchange the partner half's 16 B of qs via __shfl_xor(..., 1).
            const uint4 q_own = q_own_v[j];
            uint4 q_part;
            q_part.x = (unsigned)__shfl_xor((int)q_own.x, 1);
            q_part.y = (unsigned)__shfl_xor((int)q_own.y, 1);
            q_part.z = (unsigned)__shfl_xor((int)q_own.z, 1);
            q_part.w = (unsigned)__shfl_xor((int)q_own.w, 1);

            const uint4 q0 = half ? q_part : q_own;
            const uint4 q1 = half ? q_own  : q_part;
            const unsigned qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};

            int pos = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const unsigned nib = half ? ((qw[w] >> 4) & 0x0F0F0F0Fu) : (qw[w] & 0x0F0F0F0Fu);
                pos = grim_sdot4(a4[w], (int)nib, pos);
            }
            if (sb_valid) {
                facc[j] += d * (float)sc * ((float)pos * d_a) - dmin * (float)m * sum_a;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

// M7.2c: the RETAINED ORACLE for the grim_dot4_q4k_q81_gemv group hoist — the pre-change
// kernel body, verbatim, so the hoist can be gated on BYTE-IDENTICAL
// output instead of a hand oracle. Not on any dispatch path; only
// tests/dot4_kquant_grouped_parity.rs launches it.
extern "C" __global__ void grim_dot4_q4k_q81_gemv_ref(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q4k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int n_q4k_superblocks = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q4k + (long long)(col_base + j) * n_q4k_superblocks * GRIM_Q4K_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each thread processes superblocks in strided fashion across the wave (lane, lane + 32, ...)
    for (int sb = lane; sb < n_q4k_superblocks; sb += 32) {
        // One Q4_K superblock corresponds to 8 Q8_1 activation blocks (256 elements)
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q4K_BYTES;

            const unsigned short* h_ptr = (const unsigned short*)b_sb;
            float d    = fp16_to_float_device(h_ptr[0]);
            float dmin = fp16_to_float_device(h_ptr[1]);

            const unsigned char* scales = b_sb + 4;
            const unsigned char* qs     = b_sb + 16;

            // 8 sub-blocks of 32 elements
            #pragma unroll
            for (int is = 0; is < 8; ++is) {
                const unsigned char* a_blk = a_sb + is * GRIM_Q8_1_BYTES;
                float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                unsigned char sc, m;
                if (is < 4) {
                    sc = scales[is] & 63;
                    m  = scales[is + 4] & 63;
                } else {
                    sc = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                    m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
                }

                // Sub-block is: group = is / 2 (0..3), half = is % 2 (0: low nibble, 1: high nibble)
                int group = is / 2;
                int half  = is % 2;
                const unsigned char* qs_sub = qs + group * 32;

                int pos_dot = 0;
                #pragma unroll
                for (int e = 0; e < 32; e += 4) {
                    int a4;
                    __builtin_memcpy(&a4, a_codes + e, 4);

                    // Unpack 4 nibbles into 4 i8 values packed in one i32
                    unsigned char b0 = qs_sub[e + 0];
                    unsigned char b1 = qs_sub[e + 1];
                    unsigned char b2 = qs_sub[e + 2];
                    unsigned char b3 = qs_sub[e + 3];

                    int q0 = half ? (b0 >> 4) : (b0 & 0x0F);
                    int q1 = half ? (b1 >> 4) : (b1 & 0x0F);
                    int q2 = half ? (b2 >> 4) : (b2 & 0x0F);
                    int q3 = half ? (b3 >> 4) : (b3 & 0x0F);

                    int q4 = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);

                    pos_dot = grim_sdot4(a4, q4, pos_dot);
                }

                // Two-dot decomposition: value_i = d * sc * q_i - dmin * m
                // dot = sum(a_i * value_i) = d * sc * sum(a_i * q_i) - dmin * m * sum(a_i)
                float sub_res = d * (float)sc * ((float)pos_dot * d_a) - dmin * (float)m * sum_a;
                facc[j] += sub_res;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

#define GRIM_Q5K_SUPERBLOCK_SIZE 256
#define GRIM_Q5K_BYTES           176

// SPEED-DOT (Phase 4.5): Q5_K x Q8_1 GEMV via V_DOT4_I32_IU8 (__builtin_amdgcn_sudot4).
// 256-element super-block (176 bytes) processed across 8 sub-blocks of 32 elements.
extern "C" __global__ void grim_dot4_q5k_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q5k,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    const int lane = threadIdx.x; // 0..31
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int n_q5k_superblocks = K / GRIM_Q5K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q5k + (long long)(col_base + j) * n_q5k_superblocks * GRIM_Q5K_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane; sb < n_q5k_superblocks; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q5K_BYTES;

            const unsigned short* h_ptr = (const unsigned short*)b_sb;
            float d    = fp16_to_float_device(h_ptr[0]);
            float dmin = fp16_to_float_device(h_ptr[1]);

            const unsigned char* scales = b_sb + 4;
            const unsigned char* qh     = b_sb + 16;
            const unsigned char* qs     = b_sb + 48;

            // REGISTER PRESSURE (2026-10-22). Fully unrolling
            // 4 columns x 8 sub-blocks x 8 e-iterations compiled this
            // kernel to 192 VGPRs with 180 B of SCRATCH SPILL per thread
            // (rocprofv3 kernel-symbol dump) — the same shape as the
            // Q6_K sibling, and the Q4_K `_fast` variant's own history
            // ("the stock 4-col kernel compiles to 192 VGPRs + 560 B
            // scratch -> 1 wave/SIMD plus local-memory spills; the 88-VGPR
            // q8_0 sibling sustains 300 GB/s"). The spill puts a
            // local-memory round trip inside a kernel that is NOT
            // bandwidth-bound (m1_decode_regime_probe: Q5_K cold ~= warm
            // at every working set), so the spill is what to remove.
            //
            // BIT-IDENTICAL by construction: only the unroll factor
            // changes. The int32 accumulation (`pos_dot = grim_sdot4(...)`)
            // is an exact integer chain whose ORDER is unchanged, and the
            // float accumulation (`facc[j] += sub_res`) still runs once
            // per `is` in the same order.
            #pragma unroll 2
            for (int is = 0; is < 8; ++is) {
                const unsigned char* a_blk = a_sb + is * GRIM_Q8_1_BYTES;
                float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                unsigned char sc, m;
                if (is < 4) {
                    sc = scales[is] & 63;
                    m  = scales[is + 4] & 63;
                } else {
                    sc = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                    m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
                }

                int group = is / 2;
                int half  = is % 2;
                const unsigned char* qs_sub = qs + group * 32;

                int pos_dot = 0;
                #pragma unroll 2
                for (int e = 0; e < 32; e += 4) {
                    int a4;
                    __builtin_memcpy(&a4, a_codes + e, 4);

                    // M7.2b (the IQ group-hoisting recipe, applied to Q5_K):
                    // the four 4-bit codes and the four `qh` high bits were
                    // fetched with EIGHT separate 1-byte loads per 4 weights.
                    // Q5_K spends 2 bytes per weight (one code byte + one qh
                    // byte) against Q4_K's 0.75, and the measured decode gap
                    // tracks that byte ratio: Q5_K ran at 8.7 GB/s where Q4_K
                    // ran at 24.5 at the 27B's dominant shape.
                    //
                    // Both fields are 4-byte aligned for every e in this loop
                    // (qs sits at +48, qh at +16, the super-block stride is
                    // 176 - all multiples of 4), so the four bytes are one
                    // dword each. The per-byte nibble/bit extraction then
                    // becomes two whole-word masks instead of eight
                    // shift-mask-select triples.
                    unsigned qw4, qh4;
                    __builtin_memcpy(&qw4, qs_sub + e, 4);
                    __builtin_memcpy(&qh4, qh + e, 4);

                    // Byte i of `nib` is code i in its low nibble; byte i of
                    // `msb` is that code's qh bit at bit 4. The masks keep
                    // each byte independent, so the word-level result is
                    // identical to the per-byte arithmetic it replaces.
                    unsigned nib = half ? ((qw4 >> 4) & 0x0F0F0F0Fu)
                                        : (qw4 & 0x0F0F0F0Fu);
                    unsigned msb = ((qh4 >> (2 * group + half)) & 0x01010101u) << 4;

                    int q4 = (int)(nib | msb);

                    pos_dot = grim_sdot4(a4, q4, pos_dot);
                }

                float sub_res = d * (float)sc * ((float)pos_dot * d_a) - dmin * (float)m * sum_a;
                facc[j] += sub_res;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

// Step 2.1: Q5_K sub-block-per-lane GEMV. The kernel above maps one lane to
// one 256-weight super-block, so at K=5120 (20 super-blocks) 12 of 32 lanes
// idle and each active lane walks 8 sub-blocks x 4 columns serially. Here
// lane = (sb_slot = lane>>3, sub = lane&7): 8 lanes share a super-block, a
// wave covers 4 super-blocks per step, every lane owns one whole 32-weight
// sub-block (so the dmin*m*sum_a term is applied exactly once per sub-block).
// qs/qh are fetched as 16 B vectors (super-block stride 176 = 11*16, qh at
// +16, qs at +48: all 16 B aligned). Float summation order differs from the
// stock kernel -> tolerance-gated, not byte-identical.
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q5k_q81_gemv_sub(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q5k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub   = lane & 7;
    const int slot  = lane >> 3;
    const int group = sub >> 1;
    const int half  = sub & 1;
    const int hshift = 2 * group + half;

    const int n_sb = K / GRIM_Q5K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * GRIM_Q5K_BYTES;
    const unsigned char* b0 = B_q5k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = slot; sb < n_sb; sb += 4) {
        const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
        const float d_a   = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        const float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
        int a4[8];
        __builtin_memcpy(a4, a_blk + 4, 32);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * GRIM_Q5K_BYTES;
            const unsigned short* h_ptr = (const unsigned short*)b_sb;
            const float d    = fp16_to_float_device(h_ptr[0]);
            const float dmin = fp16_to_float_device(h_ptr[1]);
            const unsigned char* scales = b_sb + 4;
            int sc, m;
            if (sub < 4) {
                sc = scales[sub] & 63;
                m  = scales[sub + 4] & 63;
            } else {
                sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
                m  = (scales[sub + 4] >> 4)   | ((scales[sub] >> 6) << 4);
            }
            const uint4* qs4 = (const uint4*)(b_sb + 48 + group * 32);
            const uint4* qh4 = (const uint4*)(b_sb + 16);
            const uint4 q0 = qs4[0], q1 = qs4[1];
            const uint4 h0 = qh4[0], h1 = qh4[1];
            const unsigned qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            const unsigned hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            int pos = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const unsigned nib = half ? ((qw[w] >> 4) & 0x0F0F0F0Fu) : (qw[w] & 0x0F0F0F0Fu);
                const unsigned msb = ((hw[w] >> hshift) & 0x01010101u) << 4;
                pos = grim_sdot4(a4[w], (int)(nib | msb), pos);
            }
            facc[j] += d * (float)sc * ((float)pos * d_a) - dmin * (float)m * sum_a;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

// Step 2.2 (#1 + #2): Q5_K sub-block-per-lane GEMV v2:
//   (#1) 4-column load batching before the sudot4 loop.
//   (#2) De-duplicated 16 B qs (`b_sb + 48 + sub*16`) and 16 B qh (`b_sb + 16 + half*16`)
//        loads exchanged between even/odd sub-lanes via `__shfl_xor(..., 1)`,
//        cutting vector loads per lane from four uint4s (64 B) to two uint4s (32 B)
//        per column, plus 16 B uint4 header+scale unpacking.
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q5k_q81_gemv_sub_v2(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q5k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub    = lane & 7;
    const int slot   = lane >> 3;
    const int group  = sub >> 1;
    const int half   = sub & 1;
    const int hshift = 2 * group + half;

    const int n_sb = K / GRIM_Q5K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * GRIM_Q5K_BYTES;
    const unsigned char* b0 = B_q5k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb_base = 0; sb_base < n_sb; sb_base += 4) {
        const int sb = sb_base + slot;
        const bool sb_valid = (sb < n_sb);
        float d_a = 0.0f, sum_a = 0.0f;
        int a4[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (sb_valid) {
            const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
            d_a   = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
            sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
            __builtin_memcpy(a4, a_blk + 4, 32);
        }

        uint4 hdr_v[4];
        uint4 q_own_v[4];
        uint4 h_own_v[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (sb_valid && j < active_cols) {
                const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * GRIM_Q5K_BYTES;
                hdr_v[j]   = ((const uint4*)b_sb)[0];
                h_own_v[j] = *(const uint4*)(b_sb + 16 + half * 16);
                q_own_v[j] = *(const uint4*)(b_sb + 48 + sub * 16);
            } else {
                hdr_v[j]   = make_uint4(0u, 0u, 0u, 0u);
                h_own_v[j] = make_uint4(0u, 0u, 0u, 0u);
                q_own_v[j] = make_uint4(0u, 0u, 0u, 0u);
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned h0  = hdr_v[j].x;
            const unsigned sc0 = hdr_v[j].y;
            const unsigned sc1 = hdr_v[j].z;
            const unsigned sc2 = hdr_v[j].w;
            const float d    = fp16_to_float_device((unsigned short)(h0 & 0xFFFFu));
            const float dmin = fp16_to_float_device((unsigned short)(h0 >> 16));

            int sc, m;
            if (sub < 4) {
                const int sh = 8 * sub;
                sc = (int)((sc0 >> sh) & 63u);
                m  = (int)((sc1 >> sh) & 63u);
            } else {
                const int sh = 8 * (sub - 4);
                const unsigned b_hi = (sc2 >> sh) & 0xFFu;
                const unsigned b_lo = (sc0 >> sh) & 0xFFu;
                const unsigned b_mi = (sc1 >> sh) & 0xFFu;
                sc = (int)((b_hi & 0x0Fu) | ((b_lo >> 6) << 4));
                m  = (int)((b_hi >> 4)    | ((b_mi >> 6) << 4));
            }

            const uint4 q_own = q_own_v[j];
            const uint4 h_own = h_own_v[j];
            uint4 q_part, h_part;
            q_part.x = (unsigned)__shfl_xor((int)q_own.x, 1);
            q_part.y = (unsigned)__shfl_xor((int)q_own.y, 1);
            q_part.z = (unsigned)__shfl_xor((int)q_own.z, 1);
            q_part.w = (unsigned)__shfl_xor((int)q_own.w, 1);
            h_part.x = (unsigned)__shfl_xor((int)h_own.x, 1);
            h_part.y = (unsigned)__shfl_xor((int)h_own.y, 1);
            h_part.z = (unsigned)__shfl_xor((int)h_own.z, 1);
            h_part.w = (unsigned)__shfl_xor((int)h_own.w, 1);

            const uint4 q0 = half ? q_part : q_own;
            const uint4 q1 = half ? q_own  : q_part;
            const uint4 h0_v = half ? h_part : h_own;
            const uint4 h1_v = half ? h_own  : h_part;
            const unsigned qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            const unsigned hw[8] = {h0_v.x, h0_v.y, h0_v.z, h0_v.w, h1_v.x, h1_v.y, h1_v.z, h1_v.w};

            int pos = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const unsigned nib = half ? ((qw[w] >> 4) & 0x0F0F0F0Fu) : (qw[w] & 0x0F0F0F0Fu);
                const unsigned msb = ((hw[w] >> hshift) & 0x01010101u) << 4;
                pos = grim_sdot4(a4[w], (int)(nib | msb), pos);
            }
            if (sb_valid) {
                facc[j] += d * (float)sc * ((float)pos * d_a) - dmin * (float)m * sum_a;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

// M7.2b: the RETAINED ORACLE for the Q5_K group-hoisting change above.
//
// This is the pre-change `grim_dot4_q5k_q81_gemv` body, verbatim — eight
// 1-byte loads per 4 weights — kept alive under a second entry point so the
// hoist can be gated on BYTE-IDENTICAL output instead of a hand oracle. The
// parity gate is tests/dot4_q5k_grouped_parity.rs.
//
// It is NOT on any dispatch path: only the parity gate launches it.
extern "C" __global__ void grim_dot4_q5k_q81_gemv_ref(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q5k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x; // 0..31
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int n_q5k_superblocks = K / GRIM_Q5K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q5k + (long long)(col_base + j) * n_q5k_superblocks * GRIM_Q5K_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane; sb < n_q5k_superblocks; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q5K_BYTES;

            const unsigned short* h_ptr = (const unsigned short*)b_sb;
            float d    = fp16_to_float_device(h_ptr[0]);
            float dmin = fp16_to_float_device(h_ptr[1]);

            const unsigned char* scales = b_sb + 4;
            const unsigned char* qh     = b_sb + 16;
            const unsigned char* qs     = b_sb + 48;

            #pragma unroll
            for (int is = 0; is < 8; ++is) {
                const unsigned char* a_blk = a_sb + is * GRIM_Q8_1_BYTES;
                float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                unsigned char sc, m;
                if (is < 4) {
                    sc = scales[is] & 63;
                    m  = scales[is + 4] & 63;
                } else {
                    sc = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                    m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
                }

                int group = is / 2;
                int half  = is % 2;
                const unsigned char* qs_sub = qs + group * 32;

                int pos_dot = 0;
                #pragma unroll
                for (int e = 0; e < 32; e += 4) {
                    int a4;
                    __builtin_memcpy(&a4, a_codes + e, 4);

                    unsigned char b0 = qs_sub[e + 0];
                    unsigned char b1 = qs_sub[e + 1];
                    unsigned char b2 = qs_sub[e + 2];
                    unsigned char b3 = qs_sub[e + 3];

                    int q0 = half ? (b0 >> 4) : (b0 & 0x0F);
                    int q1 = half ? (b1 >> 4) : (b1 & 0x0F);
                    int q2 = half ? (b2 >> 4) : (b2 & 0x0F);
                    int q3 = half ? (b3 >> 4) : (b3 & 0x0F);

                    int msb0 = (qh[e + 0] >> (2 * group + half)) & 1;
                    int msb1 = (qh[e + 1] >> (2 * group + half)) & 1;
                    int msb2 = (qh[e + 2] >> (2 * group + half)) & 1;
                    int msb3 = (qh[e + 3] >> (2 * group + half)) & 1;

                    q0 |= (msb0 << 4);
                    q1 |= (msb1 << 4);
                    q2 |= (msb2 << 4);
                    q3 |= (msb3 << 4);

                    int q4 = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);

                    pos_dot = grim_sdot4(a4, q4, pos_dot);
                }

                float sub_res = d * (float)sc * ((float)pos_dot * d_a) - dmin * (float)m * sum_a;
                facc[j] += sub_res;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

#define GRIM_Q6K_SUPERBLOCK_SIZE 256
#define GRIM_Q6K_BYTES           210

// SPEED-DOT (Phase 4.5): Q6_K x Q8_1 GEMV via V_DOT4_I32_IU8 (__builtin_amdgcn_sudot4).
// 256-element super-block (210 bytes) processed across 8 sub-blocks of 32 elements.
extern "C" __global__ void grim_dot4_q6k_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q6k,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    const int lane = threadIdx.x; // 0..31
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int n_q6k_superblocks = K / GRIM_Q6K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q6k + (long long)(col_base + j) * n_q6k_superblocks * GRIM_Q6K_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane; sb < n_q6k_superblocks; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q6K_BYTES;

            const unsigned char* ql     = b_sb;
            const unsigned char* qh     = b_sb + 128;
            const signed char*   scales = (const signed char*)(b_sb + 192);
            const unsigned short* d_ptr = (const unsigned short*)(b_sb + 208);
            float d = fp16_to_float_device(d_ptr[0]);

            // REGISTER PRESSURE (2026-10-22): 192 VGPRs + 88 B of scratch
            // spill per thread. Same reasoning and the same bit-identity
            // argument as the Q5_K sibling above: bounding the unroll
            // changes neither the int32 accumulation order nor the float
            // accumulation order.
            #pragma unroll 2
            for (int is_sub = 0; is_sub < 8; ++is_sub) {
                const unsigned char* a_blk = a_sb + is_sub * GRIM_Q8_1_BYTES;
                float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                int n = is_sub / 4;
                int quarter = is_sub % 4;
                // is_sub 0..7: for is_sub < 4 (n=0): quarter 0..3
                // scales indexing: sc_idx = n * 8 + (quarter / 2) + 2 * (quarter % 2)... wait,
                // let's use exact formula:
                // quarter = is_sub % 4; n = is_sub / 4;
                // In each 32-element sub-block, l goes from 0..31: is = l/16, so elements 0..15 have is=0, elements 16..31 have is=1.
                // We process two halves of 16 elements:
                #pragma unroll
                for (int half16 = 0; half16 < 2; ++half16) {
                    int sc_idx = n * 8 + half16 + 2 * quarter;
                    signed char sc = scales[sc_idx];

                    int pos_dot = 0;
                    float fsum16 = 0.0f;

                    #pragma unroll 2
                    for (int e16 = 0; e16 < 16; e16 += 4) {
                        int e = half16 * 16 + e16;
                        int a4;
                        __builtin_memcpy(&a4, a_codes + e, 4);

                        // M7.2c (the IQ group-hoisting recipe): the four
                        // activation codes are already in `a4` — re-reading
                        // them as four 1-byte loads was pure waste.
                        fsum16 += (float)((int)(signed char)(a4 & 0xFF)
                                        + (int)(signed char)((a4 >> 8) & 0xFF)
                                        + (int)(signed char)((a4 >> 16) & 0xFF)
                                        + (int)(signed char)((a4 >> 24) & 0xFF));

                        int l0 = e + 0;
                        int l1 = e + 1;
                        int l2 = e + 2;
                        int l3 = e + 3;

                        int ql_off0 = n * 64 + l0 + ((quarter & 1) ? 32 : 0);
                        int ql_off1 = n * 64 + l1 + ((quarter & 1) ? 32 : 0);
                        int ql_off2 = n * 64 + l2 + ((quarter & 1) ? 32 : 0);
                        int ql_off3 = n * 64 + l3 + ((quarter & 1) ? 32 : 0);

                        // The four ql codes and the four qh high bits are
                        // four CONSECUTIVE bytes each. Q6_K's super-block
                        // stride is 210 B, so the block base is only 2-byte
                        // aligned, and a dword read there is not guaranteed
                        // aligned -- but a ushort read always is, and two
                        // ushorts plus a shift rebuild the same dword.
                        unsigned short ql01, ql23, qh01, qh23;
                        __builtin_memcpy(&ql01, ql + ql_off0, 2);
                        __builtin_memcpy(&ql23, ql + ql_off2, 2);
                        __builtin_memcpy(&qh01, qh + n * 32 + l0, 2);
                        __builtin_memcpy(&qh23, qh + n * 32 + l2, 2);
                        unsigned qlw = (unsigned)ql01 | ((unsigned)ql23 << 16);
                        unsigned qhw = (unsigned)qh01 | ((unsigned)qh23 << 16);

                        // Byte i of `nibw` is code i in its low nibble; byte i
                        // of `qhw` shifted is the 2-bit qh field of that code.
                        unsigned nibw = (quarter & 2) ? ((qlw >> 4) & 0x0F0F0F0Fu)
                                                     : (qlw & 0x0F0F0F0Fu);
                        unsigned qhb = (qhw >> (2 * quarter)) & 0x03030303u;

                        int q4 = (int)(nibw | (qhb << 4));
                        pos_dot = grim_sdot4(a4, q4, pos_dot);
                    }

                    // formula: value_i = d * sc * (q_code - 32)
                    // sum = d * sc * sum(a_i * q_i) - d * sc * 32 * sum(a_i)
                    float sub_res = d * (float)sc * ((float)pos_dot * d_a - 32.0f * (fsum16 * d_a));
                    facc[j] += sub_res;
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

// Step 2.1: Q6_K sub-block-per-lane GEMV (same lane map as the Q5_K `_sub`
// kernel: lane>>3 = super-block slot, lane&7 = 32-weight sub-block). The
// 210 B super-block stride only guarantees 2 B alignment, so ql/qh words are
// read through a 2-byte-aligned u32 type and the compiler picks the widest
// legal access. The q-32 offset term is folded exactly in int32:
// sum a*(q-32) = pos - 32*asum. Tolerance-gated vs the stock kernel.
typedef unsigned int __attribute__((aligned(2))) grim_u32_a2;
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q6k_q81_gemv_sub(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q6k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub = lane & 7;
    const int slot = lane >> 3;
    const int nh = sub >> 2;          // 128-weight half of the super-block
    const int quarter = sub & 3;
    const int ql_off = nh * 64 + ((quarter & 1) ? 32 : 0);
    const int qh_off = 128 + nh * 32;
    const int qh_shift = 2 * quarter;
    const bool hi_nib = (quarter & 2) != 0;
    const int sc0 = nh * 8 + 2 * quarter;

    const int n_sb = K / GRIM_Q6K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * GRIM_Q6K_BYTES;
    const unsigned char* b0 = B_q6k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = slot; sb < n_sb; sb += 4) {
        const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
        const float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        int a4[8];
        __builtin_memcpy(a4, a_blk + 4, 32);
        int asum0 = 0, asum1 = 0;
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            asum0 = grim_sdot4(a4[w], 0x01010101, asum0);
            asum1 = grim_sdot4(a4[w + 4], 0x01010101, asum1);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * GRIM_Q6K_BYTES;
            const grim_u32_a2* qlw = (const grim_u32_a2*)(b_sb + ql_off);
            const grim_u32_a2* qhw = (const grim_u32_a2*)(b_sb + qh_off);
            const signed char* scales = (const signed char*)(b_sb + 192);
            const float d = fp16_to_float_device(*(const unsigned short*)(b_sb + 208));
            int pos0 = 0, pos1 = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const unsigned ql = qlw[w];
                const unsigned qh = qhw[w];
                const unsigned nib = hi_nib ? ((ql >> 4) & 0x0F0F0F0Fu) : (ql & 0x0F0F0F0Fu);
                const unsigned hb = (qh >> qh_shift) & 0x03030303u;
                const int q4 = (int)(nib | (hb << 4));
                if (w < 4) pos0 = grim_sdot4(a4[w], q4, pos0);
                else       pos1 = grim_sdot4(a4[w], q4, pos1);
            }
            const float s0 = (float)scales[sc0];
            const float s1 = (float)scales[sc0 + 1];
            facc[j] += d * d_a * (s0 * (float)(pos0 - 32 * asum0) + s1 * (float)(pos1 - 32 * asum1));
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

// Step 2.2 (#1 + #2): Q6_K sub-block-per-lane GEMV v2:
//   (#1) 4-column load batching before the sudot4 loop.
//   (#2) De-duplicated 16 B ql (`ql_off + (hi_nib?16:0)`, exchanged with `lane ^ 2`)
//        and 16 B qh (`qh_off + ((quarter&1)?16:0)`, exchanged with `lane ^ 1`),
//        cutting ql+qh memory loads per lane in half (64 B/col -> 32 B/col).
extern "C" __global__ void __launch_bounds__(32) grim_dot4_q6k_q81_gemv_sub_v2(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q6k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub      = lane & 7;
    const int slot     = lane >> 3;
    const int nh       = sub >> 2;
    const int quarter  = sub & 3;
    const int q_odd    = quarter & 1;
    const bool hi_nib  = (quarter & 2) != 0;
    const int ql_off   = nh * 64 + (q_odd ? 32 : 0) + (hi_nib ? 16 : 0);
    const int qh_off   = 128 + nh * 32 + (q_odd ? 16 : 0);
    const int qh_shift = 2 * quarter;
    const int sc0      = nh * 8 + 2 * quarter;

    const int n_sb = K / GRIM_Q6K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * GRIM_Q6K_BYTES;
    const unsigned char* b0 = B_q6k + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb_base = 0; sb_base < n_sb; sb_base += 4) {
        const int sb = sb_base + slot;
        const bool sb_valid = (sb < n_sb);
        float d_a = 0.0f;
        int a4[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int asum0 = 0, asum1 = 0;
        if (sb_valid) {
            const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
            d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
            __builtin_memcpy(a4, a_blk + 4, 32);
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                asum0 = grim_sdot4(a4[w], 0x01010101, asum0);
                asum1 = grim_sdot4(a4[w + 4], 0x01010101, asum1);
            }
        }

        unsigned short d_v[4];
        unsigned short sc_v[4];
        unsigned ql_own_v[4][4];
        unsigned qh_own_v[4][4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (sb_valid && j < active_cols) {
                const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * GRIM_Q6K_BYTES;
                d_v[j]  = *(const unsigned short*)(b_sb + 208);
                sc_v[j] = *(const unsigned short*)(b_sb + 192 + sc0);
                const grim_u32_a2* qlw = (const grim_u32_a2*)(b_sb + ql_off);
                const grim_u32_a2* qhw = (const grim_u32_a2*)(b_sb + qh_off);
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    ql_own_v[j][u] = qlw[u];
                    qh_own_v[j][u] = qhw[u];
                }
            } else {
                d_v[j]  = 0u;
                sc_v[j] = 0u;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    ql_own_v[j][u] = 0u;
                    qh_own_v[j][u] = 0u;
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const float d  = fp16_to_float_device(d_v[j]);
            const float s0 = (float)(signed char)(sc_v[j] & 0xFFu);
            const float s1 = (float)(signed char)(sc_v[j] >> 8);

            unsigned ql_part[4], qh_part[4];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                ql_part[u] = (unsigned)__shfl_xor((int)ql_own_v[j][u], 2);
                qh_part[u] = (unsigned)__shfl_xor((int)qh_own_v[j][u], 1);
            }
            const unsigned* ql0 = hi_nib ? ql_part : ql_own_v[j];
            const unsigned* ql1 = hi_nib ? ql_own_v[j] : ql_part;
            const unsigned* qh0 = q_odd ? qh_part : qh_own_v[j];
            const unsigned* qh1 = q_odd ? qh_own_v[j] : qh_part;
            const unsigned ql_all[8] = {ql0[0], ql0[1], ql0[2], ql0[3], ql1[0], ql1[1], ql1[2], ql1[3]};
            const unsigned qh_all[8] = {qh0[0], qh0[1], qh0[2], qh0[3], qh1[0], qh1[1], qh1[2], qh1[3]};

            int pos0 = 0, pos1 = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                const unsigned ql = ql_all[w];
                const unsigned qh = qh_all[w];
                const unsigned nib = hi_nib ? ((ql >> 4) & 0x0F0F0F0Fu) : (ql & 0x0F0F0F0Fu);
                const unsigned hb = (qh >> qh_shift) & 0x03030303u;
                const int q4 = (int)(nib | (hb << 4));
                if (w < 4) pos0 = grim_sdot4(a4[w], q4, pos0);
                else       pos1 = grim_sdot4(a4[w], q4, pos1);
            }
            if (sb_valid) {
                facc[j] += d * d_a * (s0 * (float)(pos0 - 32 * asum0) + s1 * (float)(pos1 - 32 * asum1));
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}
// M7.2h: Q6_K dot4 GEMV with f32 activations (exact path for LM head).
//
// Like grim_dot4_q6k_q81_gemv but reads f32 activations directly instead of
// Q8_1 packed. No activation quantization — act_f32[i] is used raw, so the
// only error source is the Q6_K weight dequantization (which is exact, no
// dmin term, so no rounding of the stored f16 sum). This gives near-dot4
// speed with f32 activation exactness for the vocab-scale LM head GEMV.
//
// Scalar MACs replace grim_sdot4 since activations are f32, not int8 — we
// lose the dot4 instruction but keep the packed weight dequant structure.
// Measured on the real 27B output.weight (Q6_K, n=248320, k=5120):
//   exact fused-dequant: 4.9 GB/s (~213 ms/token on the head)
//   dot4 q81:            78.5 GB/s (~13 ms/token, but Q8_1 act quant error)
//   dot4 f32act:         ~20-30 GB/s (~45-70 ms/token, exact activations)
//
// Not on any dispatch path; only tests/q6k_f32act_dot4_parity.rs launches it.
// The dispatch wires it in via launch_dot4_q6k_f32act_gemv_for_ab.
extern "C" __global__ void grim_dot4_q6k_f32act_gemv(
    const float* __restrict__ act_f32,
    const unsigned char* __restrict__ B_q6k,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    const int lane = threadIdx.x; // 0..31
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int n_q6k_superblocks = K / GRIM_Q6K_SUPERBLOCK_SIZE;
    const float* a_row = act_f32 + (long long)row * K;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q6k + (long long)(col_base + j) * n_q6k_superblocks * GRIM_Q6K_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane; sb < n_q6k_superblocks; sb += 32) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q6K_BYTES;

            const unsigned char* ql     = b_sb;
            const unsigned char* qh     = b_sb + 128;
            const signed char*   scales = (const signed char*)(b_sb + 192);
            const unsigned short* d_ptr = (const unsigned short*)(b_sb + 208);
            float d = fp16_to_float_device(d_ptr[0]);

            #pragma unroll 2
            for (int is_sub = 0; is_sub < 8; ++is_sub) {
                const float* a_blk = a_row + sb * 256 + is_sub * 32;
                int n = is_sub / 4;
                int quarter = is_sub % 4;
                #pragma unroll
                for (int half16 = 0; half16 < 2; ++half16) {
                    int sc_idx = n * 8 + half16 + 2 * quarter;
                    signed char sc = scales[sc_idx];

                    float pos_dot = 0;

                    #pragma unroll 2
                    for (int e16 = 0; e16 < 16; e16 += 4) {
                        int e = half16 * 16 + e16;
                        float a0 = a_blk[e + 0];
                        float a1 = a_blk[e + 1];
                        float a2 = a_blk[e + 2];
                        float a3 = a_blk[e + 3];

                        int l0 = e + 0;
                        int l1 = e + 1;
                        int l2 = e + 2;
                        int l3 = e + 3;

                        int ql_off0 = n * 64 + l0 + ((quarter & 1) ? 32 : 0);
                        int ql_off1 = n * 64 + l1 + ((quarter & 1) ? 32 : 0);
                        int ql_off2 = n * 64 + l2 + ((quarter & 1) ? 32 : 0);
                        int ql_off3 = n * 64 + l3 + ((quarter & 1) ? 32 : 0);

                        unsigned short ql01, ql23, qh01, qh23;
                        __builtin_memcpy(&ql01, ql + ql_off0, 2);
                        __builtin_memcpy(&ql23, ql + ql_off2, 2);
                        __builtin_memcpy(&qh01, qh + n * 32 + l0, 2);
                        __builtin_memcpy(&qh23, qh + n * 32 + l2, 2);
                        unsigned qlw = (unsigned)ql01 | ((unsigned)ql23 << 16);
                        unsigned qhw = (unsigned)qh01 | ((unsigned)qh23 << 16);

                        unsigned nibw = (quarter & 2) ? ((qlw >> 4) & 0x0F0F0F0Fu)
                                                     : (qlw & 0x0F0F0F0Fu);
                        unsigned qhb = (qhw >> (2 * quarter)) & 0x03030303u;

                        int q4 = (int)(nibw | (qhb << 4));

                        // Scalar MACs with f32 activations (grim_sdot4 needs
                        // int8 operands, which we don't have here).
                        int q0 = (q4 >> 0) & 0xFF;
                        int q1 = (q4 >> 8) & 0xFF;
                        int q2 = (q4 >> 16) & 0xFF;
                        int q3 = (q4 >> 24) & 0xFF;

                        pos_dot += a0 * (float)(q0 - 32);
                        pos_dot += a1 * (float)(q1 - 32);
                        pos_dot += a2 * (float)(q2 - 32);
                        pos_dot += a3 * (float)(q3 - 32);
                    }

                    // formula: value_i = d * sc * (q_code - 32)
                    // dot = d * sc * sum(a_i * (q_i - 32))
                    facc[j] += d * (float)sc * pos_dot;
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}


// M7.2c: the RETAINED ORACLE for the grim_dot4_q6k_q81_gemv group hoist — the pre-change
// kernel body, verbatim, so the hoist can be gated on BYTE-IDENTICAL
// output instead of a hand oracle. Not on any dispatch path; only
// tests/dot4_kquant_grouped_parity.rs launches it.
extern "C" __global__ void grim_dot4_q6k_q81_gemv_ref(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q6k,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    const int lane = threadIdx.x; // 0..31
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int n_q6k_superblocks = K / GRIM_Q6K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q6k + (long long)(col_base + j) * n_q6k_superblocks * GRIM_Q6K_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane; sb < n_q6k_superblocks; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q6K_BYTES;

            const unsigned char* ql     = b_sb;
            const unsigned char* qh     = b_sb + 128;
            const signed char*   scales = (const signed char*)(b_sb + 192);
            const unsigned short* d_ptr = (const unsigned short*)(b_sb + 208);
            float d = fp16_to_float_device(d_ptr[0]);

            #pragma unroll
            for (int is_sub = 0; is_sub < 8; ++is_sub) {
                const unsigned char* a_blk = a_sb + is_sub * GRIM_Q8_1_BYTES;
                float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                int n = is_sub / 4;
                int quarter = is_sub % 4;
                // is_sub 0..7: for is_sub < 4 (n=0): quarter 0..3
                // scales indexing: sc_idx = n * 8 + (quarter / 2) + 2 * (quarter % 2)... wait,
                // let's use exact formula:
                // quarter = is_sub % 4; n = is_sub / 4;
                // In each 32-element sub-block, l goes from 0..31: is = l/16, so elements 0..15 have is=0, elements 16..31 have is=1.
                // We process two halves of 16 elements:
                #pragma unroll
                for (int half16 = 0; half16 < 2; ++half16) {
                    int sc_idx = n * 8 + half16 + 2 * quarter;
                    signed char sc = scales[sc_idx];

                    int pos_dot = 0;
                    float fsum16 = 0.0f;

                    #pragma unroll
                    for (int e16 = 0; e16 < 16; e16 += 4) {
                        int e = half16 * 16 + e16;
                        int a4;
                        __builtin_memcpy(&a4, a_codes + e, 4);

                        fsum16 += (float)a_codes[e + 0] + (float)a_codes[e + 1] + (float)a_codes[e + 2] + (float)a_codes[e + 3];

                        int l0 = e + 0;
                        int l1 = e + 1;
                        int l2 = e + 2;
                        int l3 = e + 3;

                        int ql_off0 = n * 64 + l0 + ((quarter & 1) ? 32 : 0);
                        int ql_off1 = n * 64 + l1 + ((quarter & 1) ? 32 : 0);
                        int ql_off2 = n * 64 + l2 + ((quarter & 1) ? 32 : 0);
                        int ql_off3 = n * 64 + l3 + ((quarter & 1) ? 32 : 0);

                        unsigned char ql0 = ql[ql_off0];
                        unsigned char ql1 = ql[ql_off1];
                        unsigned char ql2 = ql[ql_off2];
                        unsigned char ql3 = ql[ql_off3];

                        int nib0 = (quarter & 2) ? (ql0 >> 4) : (ql0 & 0x0F);
                        int nib1 = (quarter & 2) ? (ql1 >> 4) : (ql1 & 0x0F);
                        int nib2 = (quarter & 2) ? (ql2 >> 4) : (ql2 & 0x0F);
                        int nib3 = (quarter & 2) ? (ql3 >> 4) : (ql3 & 0x0F);

                        int qh0 = (qh[n * 32 + l0] >> (2 * quarter)) & 0x03;
                        int qh1 = (qh[n * 32 + l1] >> (2 * quarter)) & 0x03;
                        int qh2 = (qh[n * 32 + l2] >> (2 * quarter)) & 0x03;
                        int qh3 = (qh[n * 32 + l3] >> (2 * quarter)) & 0x03;

                        int q0 = nib0 | (qh0 << 4);
                        int q1 = nib1 | (qh1 << 4);
                        int q2 = nib2 | (qh2 << 4);
                        int q3 = nib3 | (qh3 << 4);

                        int q4 = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
                        pos_dot = grim_sdot4(a4, q4, pos_dot);
                    }

                    // formula: value_i = d * sc * (q_code - 32)
                    // sum = d * sc * sum(a_i * q_i) - d * sc * 32 * sum(a_i)
                    float sub_res = d * (float)sc * ((float)pos_dot * d_a - 32.0f * (fsum16 * d_a));
                    facc[j] += sub_res;
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

// ─── Phase 4.5b: W4A4 sudot8 GEMV (dot8-insts, RDNA4 gfx1200/gfx1201) ────────
#if defined(__gfx1200__) || defined(__gfx1201__)

__device__ __forceinline__ float grim_bf16_to_float(unsigned short h) {
    unsigned int bits = ((unsigned int)h) << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

#define GRIM_U4_GROUP_SIZE 128

// Quantize A [M, K] -> packed SIGNED-symmetric 4-bit nibbles + scale + sum of
// codes per 128-element group.
// 128 floats -> 16 uint32 words (each word packs 8 signed 4-bit nibbles
// little-endian, two's complement; the sudot8 consumer reads A signed).
extern "C" __global__ void grim_quantize_u4_group128(
    const float* __restrict__ src,
    unsigned int* __restrict__ dst_codes,
    float* __restrict__ dst_scales,
    int* __restrict__ dst_sums,
    int K,
    int n_rows)
{
    const int global_blk = blockIdx.x;
    const int n_groups = K / GRIM_U4_GROUP_SIZE;
    const int row = global_blk / n_groups;
    const int grp = global_blk % n_groups;
    if (row >= n_rows) return;

    const int tid = threadIdx.x; // Block size 32: each thread handles 4 floats
    const float* grp_src = src + (long long)row * K + grp * GRIM_U4_GROUP_SIZE;
    unsigned int* grp_dst_codes = dst_codes + ((long long)row * n_groups + grp) * 16;

    float v0 = grp_src[tid * 4 + 0];
    float v1 = grp_src[tid * 4 + 1];
    float v2 = grp_src[tid * 4 + 2];
    float v3 = grp_src[tid * 4 + 3];

    float local_max = fmaxf(fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fabsf(v2)), fabsf(v3));
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, mask));
    }

    const float d = local_max / 7.0f;
    const float inv_d = (local_max > 1e-9f) ? (7.0f / local_max) : 0.0f;

    // Signed-symmetric codes in [-8, 7]: negative activations survive as
    // negative nibbles (two's complement) instead of clamping to 0. The
    // packing below is bit-identical either way -- (unsigned char) of a
    // negative signed char is modulo 256 -- so only the range and the scale
    // change, not the layout. -8 is never emitted (symmetric range); the
    // sudot8 consumer reads A signed (see grim_dot8_w4a4_gemv).
    signed char q0 = (signed char)fminf(fmaxf(__builtin_roundf(v0 * inv_d), -8.0f), 7.0f);
    signed char q1 = (signed char)fminf(fmaxf(__builtin_roundf(v1 * inv_d), -8.0f), 7.0f);
    signed char q2 = (signed char)fminf(fmaxf(__builtin_roundf(v2 * inv_d), -8.0f), 7.0f);
    signed char q3 = (signed char)fminf(fmaxf(__builtin_roundf(v3 * inv_d), -8.0f), 7.0f);

    int thread_sum = (int)q0 + (int)q1 + (int)q2 + (int)q3;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_sum += __shfl_xor(thread_sum, mask);
    }

    if (tid == 0) {
        dst_scales[(long long)row * n_groups + grp] = d;
        dst_sums[(long long)row * n_groups + grp] = thread_sum;
    }

    // Pack 8 nibbles into 16 words across the 32 threads.
    // Thread 2*i supplies bits 0..15, thread 2*i+1 supplies bits 16..31.
    // Mask to nibbles explicitly: signed-char promotion sign-extends, so
    // shifting a negative q without masking would corrupt upper bits.
    unsigned int half_word = (((unsigned int)(unsigned char)q0) & 0xFu)
        | ((((unsigned int)(unsigned char)q1) & 0xFu) << 4)
        | ((((unsigned int)(unsigned char)q2) & 0xFu) << 8)
        | ((((unsigned int)(unsigned char)q3) & 0xFu) << 12);
    unsigned int other_half = __shfl_xor(half_word, 1);
    if ((tid & 1) == 0) {
        unsigned int word = half_word | (other_half << 16);
        grp_dst_codes[tid / 2] = word;
    }
}


extern "C" __global__ void grim_dot8_w4a4_gemv(
    const unsigned int* __restrict__ A_codes,
    const float* __restrict__ A_scales,
    const int* __restrict__ A_sums,
    const unsigned int* __restrict__ B_qweight,
    const unsigned short* __restrict__ B_scales,
    const unsigned char* __restrict__ B_zeros,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_groups = K / 128;
    const unsigned int* a_codes_row = A_codes + (long long)row * n_groups * 16;
    const float* a_scales_row = A_scales + (long long)row * n_groups;
    const int* a_sums_row = A_sums + (long long)row * n_groups;

    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const int words_per_col = K / 8;

    const unsigned int* b_cols[4];
    const unsigned short* b_scales_col[4];
    const unsigned char* b_zeros_col[4];

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j < active_cols) {
            int col = col_base + j;
            b_cols[j] = B_qweight + (long long)col * words_per_col;
            b_scales_col[j] = B_scales + (long long)col * n_groups;
            b_zeros_col[j] = B_zeros + (long long)col * n_groups;
        } else {
            b_cols[j] = nullptr;
            b_scales_col[j] = nullptr;
            b_zeros_col[j] = nullptr;
        }
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = lane; g < n_groups; g += 32) {
        float d_a = a_scales_row[g];
        int sum_qa = a_sums_row[g];
        const unsigned int* a_grp = a_codes_row + g * 16;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            float d_b = grim_bf16_to_float(b_scales_col[j][g]);
            int z_b = (int)b_zeros_col[j][g];
            const unsigned int* b_grp = b_cols[j] + g * 16;

            int iacc = 0;
            #pragma unroll
            for (int w = 0; w < 16; w++) {
                unsigned int a_word = a_grp[w];
                unsigned int b_word = b_grp[w];
                // A signed (symmetric [-8, 7] nibbles), B unsigned (0-15 with
                // zero point): the mixed-sign dot the W4A4 format requires.
                iacc = __builtin_amdgcn_sudot8(true, a_word, false, b_word, iacc, false);
            }

            // Two-dot zero-point algebraic formulation:
            // sum_i (A_i * W_i) = d_a * d_b * (iacc - z_b * sum_qa)
            float grp_val = d_a * d_b * (float)(iacc - z_b * sum_qa);
            facc[j] += grp_val;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            facc[j] += __shfl_xor(facc[j], off);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

#endif // __gfx1200__ || __gfx1201__

// ForestRaven decode/prefill GEMV: F32 activations quantized in-register
// (per-32-block absmax, full-fp32 scale) against per-row absmax INT8 weights
// in the framed blob ([u64 codes_len][codes N*K][u64 scales_len][scales N]).
//
// One scale multiply per output (d_a[block] * s_b[row]) instead of Q8_0's
// per-32-block B scales: the B scale stream is N fp32, not N*K/32 fp16.
// Integer accumulation via grim_sdot4 (V_DOT4_I32_IU8 on RDNA3/4), so the
// dots are exact and the only rounding is the final float scaling -- the
// host reference mirrors the in-register A quantization bit-for-bit and the
// residual is pure float-association noise.
//
// Caller contract (checked host-side in the launcher, not here): K % 32 == 0
// (block loop strides 32) and blob total == 16 + N*K + 4*N. A tail K would
// read past the codes segment into the scales stream -- numerically silent,
// so the gate lives on the host where it can refuse. N tails are handled
// via active_cols; M rows via blockIdx.y like the Q8_0 GEMV above.
//
// Measured (RX 9070 XT, gfx1200, n=k=4096, `forest_dispatch_bench`):
//   m=1 :  31.2 us, 538 GB/s -- 2.04x over the same-class Q8_0 dot4 GEMV
//         (63.5 us, 264 GB/s). The win is structural: one fp32 scale per row
//         instead of per-32-block fp16 scales, so the inner loop carries no
//         scale traffic and no fp16->fp32 conversions.
//   m=16: 191.8 us,  87.5 GB/s -- 2.60x over Q8_0 (497.7 us, 33.7 GB/s).
// Graph-capture safe (caller-owned buffers only, single launch, no scratch).
extern "C" __global__ void grim_dot4_forest_gemv(
    const float* __restrict__ A,
    const unsigned char* __restrict__ Bblob,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    // Frame parse by arithmetic: the host validated the total size, so these
    // offsets are exact without reading the length prefixes.
    const signed char* Bcodes = (const signed char*)(Bblob + 8);
    const float* Bscales = (const float*)(Bblob + 8 + (long long)N * K + 8);

    const float* a_row = A + (long long)row * K;

    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    float s_b[4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        s_b[j] = (j < active_cols) ? Bscales[col_base + j] : 0.0f;

    const int n_blocks = K / 32;
    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_blocks; blk += 32) {
        const float* blk_src = a_row + (long long)blk * 32;

        // 1. Per-32-block activation absmax. Full-fp32 scale, deliberately
        // NOT fp16-rounded: unlike Q8_0/Q8_1 the A scale is ephemeral (never
        // stored), so rounding it would lose precision for nothing. The host
        // reference uses the same unrounded scale.
        float amax = 0.0f;
        #pragma unroll
        for (int e = 0; e < 32; e++)
            amax = fmaxf(amax, __builtin_fabsf(blk_src[e]));
        const float d_a   = amax / 127.0f;
        // Reciprocal-multiply, not divide-per-element: matches the host
        // reference's formula bit-for-bit (same ops, same order), and the
        // Q8_0 f32act kernel above does the same. Zero block -> all-zero
        // codes and zero scale: contributes exactly 0, no NaN.
        const float inv_a = (amax == 0.0f) ? 0.0f : (127.0f / amax);

        int a4_reg[8];
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            signed char q0 = (signed char)__builtin_roundf(blk_src[p * 4 + 0] * inv_a);
            signed char q1 = (signed char)__builtin_roundf(blk_src[p * 4 + 1] * inv_a);
            signed char q2 = (signed char)__builtin_roundf(blk_src[p * 4 + 2] * inv_a);
            signed char q3 = (signed char)__builtin_roundf(blk_src[p * 4 + 3] * inv_a);
            a4_reg[p] = (int)((unsigned char)q0) |
                        ((int)((unsigned char)q1) << 8) |
                        ((int)((unsigned char)q2) << 16) |
                        ((int)((unsigned char)q3) << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            // B codes are dense row-major: row (col_base+j) starts at
            // (col_base+j)*K, block blk at +blk*32. No per-block scale to
            // load -- one s_b[j] for the whole row, hoisted above.
            const signed char* b_blk =
                Bcodes + (long long)(col_base + j) * K + (long long)blk * 32;
            int iacc = 0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int b4;
                __builtin_memcpy(&b4, b_blk + p * 4, 4);
                iacc = grim_sdot4(a4_reg[p], b4, iacc);
            }
            facc[j] += (float)iacc * d_a * s_b[j];
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

#endif // RDNA3/RDNA4

// RDNA2 ISA probe: v_dot4_i32_i8 semantics on known packed operands.
extern "C" __global__ void grim_sdot4_probe(
    const int* __restrict__ a_packed,
    const int* __restrict__ b_packed,
    float* __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1035__) || defined(__gfx1036__)
    int acc = __builtin_amdgcn_sdot4(a_packed[i], b_packed[i], 0, false);
    // Chained + negative intermediate (mimics GEMV accumulation pattern).
    acc = __builtin_amdgcn_sdot4(a_packed[i], b_packed[i], acc, false);
    acc = __builtin_amdgcn_sdot4(b_packed[i], a_packed[i], -acc, false);
    out[i] = (float)acc;
#else
    int acc = __builtin_amdgcn_sudot4(true, a_packed[i], true, b_packed[i], 0, false);
    acc = __builtin_amdgcn_sudot4(true, a_packed[i], true, b_packed[i], acc, false);
    acc = __builtin_amdgcn_sudot4(true, b_packed[i], true, a_packed[i], -acc, false);
    out[i] = (float)acc;
#endif
    if (i == 0) out[n] = (float)warpSize; // wave-size report (buffer sized n+1)
}

// SPEED-DOT Phase 2 (9B perf plan Step 2): word-wide-mask variant of
// grim_dot4_q4k_q81_gemv. The floor probe (tests/dot4_gemv_floor_probe.rs)
// measured the stock kernel at ~23 GB/s vs ~304 GB/s for the q8_0 kernel
// with the SAME sdot4 loop and grid mapping: the per-byte
// select/shift/pack nibble unpack is an instruction-throughput bottleneck,
// not DRAM. Here one 32-bit load of the packed nibbles yields BOTH
// sub-block dot operands of a group via two masks
//   lo = w & 0x0F0F0F0F   (sub-block 2g, low nibbles)
//   hi = (w >> 4) & 0x0F0F0F0F (sub-block 2g+1, high nibbles)
// fed straight to sdot4: ~10x fewer integer ops per element. Grid mapping,
// block size and accumulation order per sub-block are identical to the
// stock kernel; parity is gated in tests/dot_gemv_parity.rs
// (dot4_q4k_gemv_fast_parity).
__launch_bounds__(32, 3)
extern "C" __global__ void grim_dot4_q4k_q81_gemv_fast(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q4k,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Register-pressure variant: 2 columns per workgroup, pair-wise nibble
    // masks, bounded unroll. The stock 4-col kernel compiles to 192 VGPRs +
    // 560 B scratch (rocprof kernel-symbol dump) -> 1 wave/SIMD plus
    // local-memory spills; the 88-VGPR q8_0 sibling sustains 300 GB/s.
    // Grid mapping changes (N/2 blocks), accumulation order per column does
    // not. Parity: tests/dot_gemv_parity.rs dot4_q4k_gemv_fast_parity.
    const int col_base = blockIdx.x * 2;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int n_q4k_superblocks = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[2];
    const int active_cols = (col_base + 2 <= N) ? 2 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q4k + (long long)(col_base + j) * n_q4k_superblocks * GRIM_Q4K_BYTES
                 : nullptr;
    }

    float facc[2] = {0.0f, 0.0f};

    for (int sb = lane; sb < n_q4k_superblocks; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q4K_BYTES;

            const unsigned short* h_ptr = (const unsigned short*)b_sb;
            float d    = fp16_to_float_device(h_ptr[0]);
            float dmin = fp16_to_float_device(h_ptr[1]);

            const unsigned char* scales = b_sb + 4;
            const unsigned char* qs     = b_sb + 16;

            // Pairs of sub-blocks (2g, 2g+1) share the same 32 packed bytes:
            // low nibbles of word w are sub-block 2g's codes, high nibbles
            // are 2g+1's. Two masks per word replace the stock kernel's
            // per-byte select/shift/pack chain. Unroll by 2: full unroll
            // bloated live ranges past the wave-residency cliff.
            #pragma unroll 2
            for (int g = 0; g < 4; ++g) {
                int is0 = 2 * g;
                int is1 = 2 * g + 1;
                unsigned char sc0, m0, sc1, m1;
                if (is0 < 4) {
                    sc0 = scales[is0] & 63;
                    m0  = scales[is0 + 4] & 63;
                    sc1 = scales[is1] & 63;
                    m1  = scales[is1 + 4] & 63;
                } else {
                    sc0 = (scales[is0 + 4] & 0x0F) | ((scales[is0 - 4] >> 6) << 4);
                    m0  = (scales[is0 + 4] >> 4)  | ((scales[is0] >> 6) << 4);
                    sc1 = (scales[is1 + 4] & 0x0F) | ((scales[is1 - 4] >> 6) << 4);
                    m1  = (scales[is1 + 4] >> 4)  | ((scales[is1] >> 6) << 4);
                }

                const unsigned char* a_blk0 = a_sb + is0 * GRIM_Q8_1_BYTES;
                const unsigned char* a_blk1 = a_sb + is1 * GRIM_Q8_1_BYTES;
                float d_a0 = fp16_to_float_device(((const unsigned short*)a_blk0)[0]);
                float sum_a0 = fp16_to_float_device(((const unsigned short*)a_blk0)[1]);
                float d_a1 = fp16_to_float_device(((const unsigned short*)a_blk1)[0]);
                float sum_a1 = fp16_to_float_device(((const unsigned short*)a_blk1)[1]);
                const signed char* a_codes0 = (const signed char*)(a_blk0 + 4);
                const signed char* a_codes1 = (const signed char*)(a_blk1 + 4);

                const unsigned char* qs_sub = qs + g * 32;

                int pos_lo = 0;
                int pos_hi = 0;
                #pragma unroll
                for (int w8 = 0; w8 < 8; ++w8) {
                    unsigned int word;
                    __builtin_memcpy(&word, qs_sub + w8 * 4, 4);
                    unsigned int lo = word & 0x0F0F0F0Fu;
                    unsigned int hi = (word >> 4) & 0x0F0F0F0Fu;

                    int a4lo, a4hi;
                    __builtin_memcpy(&a4lo, a_codes0 + w8 * 4, 4);
                    __builtin_memcpy(&a4hi, a_codes1 + w8 * 4, 4);

                    pos_lo = grim_sdot4(a4lo, (int)lo, pos_lo);
                    pos_hi = grim_sdot4(a4hi, (int)hi, pos_hi);
                }

                facc[j] += d * (float)sc0 * ((float)pos_lo * d_a0) - dmin * (float)m0 * sum_a0;
                facc[j] += d * (float)sc1 * ((float)pos_hi * d_a1) - dmin * (float)m1 * sum_a1;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 2; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

// ─── Phase res0-T1-4: Q4_K DMMV (num_rows batching, zolotukhin lever) ─────
// The chunked-prefill lever from the RDNA4 DMMV work (zolotukhin 2026-04-22,
// "why RDNA4 prefill wants a 32-column DMMV before a GEMM"): at M rows of
// 2..8, the single-row dot4 kernel re-reads each weight byte M times (grid.y
// = M). This variant assigns ONE workgroup to 2 weight columns x ALL M rows
// (caller guarantees M <= 8, launcher-enforced): the Q4_K super-block
// header, the scale shuffle and the packed code words are read/decoded ONCE
// and reused across the row loop. Weight traffic divides by M.
//
// Byte-identity: row r's accumulation order (sb lane stride, g loop, w8
// loop, the two-dot terms) is IDENTICAL to grim_dot4_q4k_q81_gemv_fast for
// that row, so dmmv(M=8) must equal 8 single-row launches bit for bit —
// gate: tests/dot_gemv_parity.rs dot4_q4k_dmmv_parity.
extern "C" __global__ void grim_dot4_q4k_q81_gemv_dmmv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q4k,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 2;
    const int lane     = threadIdx.x;
    if (col_base >= N) return;
    if (M > 8) return; // launcher contract; refuse silently-wrong shapes

    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int n_q4k_superblocks = K / GRIM_Q4K_SUPERBLOCK_SIZE;

    const unsigned char* b_col[2];
    const int active_cols = (col_base + 2 <= N) ? 2 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q4k + (long long)(col_base + j) * n_q4k_superblocks * GRIM_Q4K_BYTES
                 : nullptr;
    }

    float facc[8][2];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        facc[r][0] = 0.0f;
        facc[r][1] = 0.0f;
    }

    for (int sb = lane; sb < n_q4k_superblocks; sb += 32) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * GRIM_Q4K_BYTES;

            const unsigned short* h_ptr = (const unsigned short*)b_sb;
            float d    = fp16_to_float_device(h_ptr[0]);
            float dmin = fp16_to_float_device(h_ptr[1]);

            const unsigned char* scales = b_sb + 4;
            const unsigned char* qs     = b_sb + 16;

            #pragma unroll 2
            for (int g = 0; g < 4; ++g) {
                int is0 = 2 * g;
                int is1 = 2 * g + 1;
                unsigned char sc0, m0, sc1, m1;
                if (is0 < 4) {
                    sc0 = scales[is0] & 63;
                    m0  = scales[is0 + 4] & 63;
                    sc1 = scales[is1] & 63;
                    m1  = scales[is1 + 4] & 63;
                } else {
                    sc0 = (scales[is0 + 4] & 0x0F) | ((scales[is0 - 4] >> 6) << 4);
                    m0  = (scales[is0 + 4] >> 4)  | ((scales[is0] >> 6) << 4);
                    sc1 = (scales[is1 + 4] & 0x0F) | ((scales[is1 - 4] >> 6) << 4);
                    m1  = (scales[is1 + 4] >> 4)  | ((scales[is1] >> 6) << 4);
                }

                const unsigned char* qs_sub = qs + g * 32;

                // The packed code words are row-INDEPENDENT: decode once.
                unsigned int words[8];
                #pragma unroll
                for (int w8 = 0; w8 < 8; ++w8) {
                    unsigned int word;
                    __builtin_memcpy(&word, qs_sub + w8 * 4, 4);
                    words[w8] = word;
                }
                unsigned int lo[8];
                unsigned int hi[8];
                #pragma unroll
                for (int w8 = 0; w8 < 8; ++w8) {
                    lo[w8] = words[w8] & 0x0F0F0F0Fu;
                    hi[w8] = (words[w8] >> 4) & 0x0F0F0F0Fu;
                }

                for (int r = 0; r < M; ++r) {
                    // Row pointer computed inline (a prebuilt a_row[8] array
                    // under `#pragma unroll` miscompiled on gfx1200: the
                    // dmmv gate measured facc[r] == row r-1's data).
                    const unsigned char* a_sb =
                        A_q81 + (long long)r * n_q81_blocks * GRIM_Q8_1_BYTES
                              + sb * 8 * GRIM_Q8_1_BYTES;
                    const unsigned char* a_blk0 = a_sb + is0 * GRIM_Q8_1_BYTES;
                    const unsigned char* a_blk1 = a_sb + is1 * GRIM_Q8_1_BYTES;
                    float d_a0 = fp16_to_float_device(((const unsigned short*)a_blk0)[0]);
                    float sum_a0 = fp16_to_float_device(((const unsigned short*)a_blk0)[1]);
                    float d_a1 = fp16_to_float_device(((const unsigned short*)a_blk1)[0]);
                    float sum_a1 = fp16_to_float_device(((const unsigned short*)a_blk1)[1]);
                    const signed char* a_codes0 = (const signed char*)(a_blk0 + 4);
                    const signed char* a_codes1 = (const signed char*)(a_blk1 + 4);

                    int pos_lo = 0;
                    int pos_hi = 0;
                    #pragma unroll
                    for (int w8 = 0; w8 < 8; ++w8) {
                        int a4lo, a4hi;
                        __builtin_memcpy(&a4lo, a_codes0 + w8 * 4, 4);
                        __builtin_memcpy(&a4hi, a_codes1 + w8 * 4, 4);
                        pos_lo = grim_sdot4(a4lo, (int)lo[w8], pos_lo);
                        pos_hi = grim_sdot4(a4hi, (int)hi[w8], pos_hi);
                    }

                    facc[r][j] += d * (float)sc0 * ((float)pos_lo * d_a0) - dmin * (float)m0 * sum_a0;
                    facc[r][j] += d * (float)sc1 * ((float)pos_hi * d_a1) - dmin * (float)m1 * sum_a1;
                }
            }
        }
    }

    for (int r = 0; r < M; ++r) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                facc[r][j] += __shfl_xor(facc[r][j], off);
        }
    }

    if (lane == 0) {
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < 2; j++) {
                if (j < active_cols) {
                    C[(long long)r * N + col_base + j] = facc[r][j];
                }
            }
        }
    }
}

// ─── Phase res8-T2: IQ4_XS dot4 GEMV pair (stock + word-nibble fast) ──────
// IQ4_XS layout (matches grim_quant::dequant_iq4xs): 136 bytes per
// 256-weight super-block — f16 d @0, scales_h[2] @2, scales_l[8] @4,
// qs[128] @8. Eight 32-weight groups; group ib has
//   ls = ((sc_l[ib>>1] >> (4*(ib&1))) & 0xF) | (((scales_h >> (2*ib)) & 3) << 4)
//   dl = d * (ls - 32)
// and its 32 codes are NIBBLES of qs[0..15]: weights 0..15 take the LOW
// nibble of bytes 0..15, weights 16..31 the HIGH nibble of the SAME bytes.
// Values go through kvalues_iq4nl (an arbitrary i8 table, NOT linear in the
// code), so unlike Q2_K/Q3_K the dot4 operand is kvalues[code] packed into
// an i8x4 word — a 16-entry LDS table, not a mask-and-dot of the raw codes.
// One 32-weight group spans exactly one q8_1 activation block, so the
// per-group accumulation is pos = Σ sdot4(a4, kv4) over the group's 32
// codes (low half then high half) and facc += dl * pos * d_a.
//
// STOCK: one 1-byte load per qs byte, nibble picked per weight.
// FAST: one 32-bit load per 4 qs bytes, `& 0x0F0F0F0F` / `>>4 & 0x0F0F0F0F`
// — the nibble selector is fixed per half and the byte index is carried by
// the byte lanes of the word, so the extracted codes are identical. The two
// kernels share the grid mapping (4 output columns per 32-lane block,
// lanes striding super-blocks), the accumulation order and the arithmetic,
// so they are byte-identical by construction; parity:
// tests/dot4_kquant_grouped_parity.rs (iq4xs row).
__device__ __forceinline__ int grim_kv4_iq4xs(const int* tbl, unsigned int q4) {
    return (tbl[q4 & 0xFF] & 0xFF)
         | ((tbl[(q4 >> 8) & 0xFF] & 0xFF) << 8)
         | ((tbl[(q4 >> 16) & 0xFF] & 0xFF) << 16)
         | ((tbl[(q4 >> 24) & 0xFF] & 0xFF) << 24);
}

__device__ __forceinline__ void grim_iq4xs_dot4_body(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_iq4xs,
    float* __restrict__ C,
    int M, int N, int K,
    int col_base, int row, int lane, int word_unpack)
{
    if (row >= M || col_base >= N) return;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const int n_q81_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q81_blocks * GRIM_Q8_1_BYTES;

    __shared__ int kv_tbl[16];
    if (lane < 16) {
        kv_tbl[lane] = (lane == 0) ? -127 : (lane == 1) ? -104 : (lane == 2) ? -83
                     : (lane == 3) ? -65  : (lane == 4) ? -49  : (lane == 5) ? -35
                     : (lane == 6) ? -22  : (lane == 7) ? -10  : (lane == 8) ? 1
                     : (lane == 9) ? 13   : (lane == 10) ? 25  : (lane == 11) ? 38
                     : (lane == 12) ? 53  : (lane == 13) ? 69  : (lane == 14) ? 89 : 113;
    }
    __syncthreads();

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_iq4xs + (long long)(col_base + j) * n_sb * 136
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane; sb < n_sb; sb += 32) {
        const unsigned char* a_sb = a_row + sb * 8 * GRIM_Q8_1_BYTES;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_sb = b_col[j] + sb * 136;
            const float d = fp16_to_float_device(((const unsigned short*)b_sb)[0]);
            const unsigned int scales_h =
                (unsigned int)b_sb[2] | ((unsigned int)b_sb[3] << 8);
            const unsigned char* sc_l = b_sb + 4;
            const unsigned char* qs = b_sb + 8;

            #pragma unroll
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = (int)((sc_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F)
                             | (((int)((scales_h >> (2 * ib)) & 3)) << 4);
                const float dl = d * (float)(ls - 32);
                const unsigned char* a_blk = a_sb + ib * GRIM_Q8_1_BYTES;
                const float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);
                // group ib owns qs[ib*16 .. ib*16+16]; both nibble halves
                // live in these same 16 bytes.
                const unsigned char* qs_g = qs + ib * 16;

                int pos = 0;
                if (word_unpack) {
                    #pragma unroll
                    for (int w8 = 0; w8 < 4; ++w8) {
                        unsigned int word;
                        __builtin_memcpy(&word, qs_g + w8 * 4, 4);
                        int a4;
                        __builtin_memcpy(&a4, a_codes + w8 * 4, 4);
                        pos = grim_sdot4(a4, grim_kv4_iq4xs(kv_tbl, word & 0x0F0F0F0Fu), pos);
                    }
                    #pragma unroll
                    for (int w8 = 0; w8 < 4; ++w8) {
                        unsigned int word;
                        __builtin_memcpy(&word, qs_g + w8 * 4, 4);
                        int a4;
                        __builtin_memcpy(&a4, a_codes + 16 + w8 * 4, 4);
                        pos = grim_sdot4(a4, grim_kv4_iq4xs(kv_tbl, (word >> 4) & 0x0F0F0F0Fu), pos);
                    }
                } else {
                    #pragma unroll
                    for (int w8 = 0; w8 < 4; ++w8) {
                        unsigned int q4 = (unsigned int)(qs_g[w8 * 4] & 0x0F)
                                        | ((unsigned int)(qs_g[w8 * 4 + 1] & 0x0F) << 8)
                                        | ((unsigned int)(qs_g[w8 * 4 + 2] & 0x0F) << 16)
                                        | ((unsigned int)(qs_g[w8 * 4 + 3] & 0x0F) << 24);
                        int a4;
                        __builtin_memcpy(&a4, a_codes + w8 * 4, 4);
                        pos = grim_sdot4(a4, grim_kv4_iq4xs(kv_tbl, q4), pos);
                    }
                    #pragma unroll
                    for (int w8 = 0; w8 < 4; ++w8) {
                        unsigned int q4 = (unsigned int)(qs_g[w8 * 4] >> 4)
                                        | ((unsigned int)(qs_g[w8 * 4 + 1] >> 4) << 8)
                                        | ((unsigned int)(qs_g[w8 * 4 + 2] >> 4) << 16)
                                        | ((unsigned int)(qs_g[w8 * 4 + 3] >> 4) << 24);
                        int a4;
                        __builtin_memcpy(&a4, a_codes + 16 + w8 * 4, 4);
                        pos = grim_sdot4(a4, grim_kv4_iq4xs(kv_tbl, q4), pos);
                    }
                }
                facc[j] += dl * ((float)pos * d_a);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

extern "C" __global__ void grim_dot4_iq4xs_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_iq4xs,
    float* __restrict__ C,
    int M, int N, int K)
{
    grim_iq4xs_dot4_body(A_q81, B_iq4xs, C, M, N, K,
                         blockIdx.x * 4, blockIdx.y, threadIdx.x, 0);
}

extern "C" __global__ void grim_dot4_iq4xs_q81_gemv_fast(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_iq4xs,
    float* __restrict__ C,
    int M, int N, int K)
{
    grim_iq4xs_dot4_body(A_q81, B_iq4xs, C, M, N, K,
                         blockIdx.x * 4, blockIdx.y, threadIdx.x, 1);
}

// Step 2.1: IQ4_XS sub-block-per-lane GEMV (lane>>3 = super-block slot,
// lane&7 = 32-weight sub-block). 136 B super-block stride, qs at +8 + sub*16
// -> 8-byte aligned (`uint2` load). Uses wave-wide `__shfl` over the 16-entry
// `kvalues_iq4nl` register table (in lanes 0..15) so zero LDS bank conflicts
// occur. Tolerance-gated vs the stock kernel.
__device__ __forceinline__ int grim_kv4_iq4xs_shfl(int my_kv, unsigned int q4) {
    int c0 = __shfl(my_kv, q4 & 0xFFu);
    int c1 = __shfl(my_kv, (q4 >> 8) & 0xFFu);
    int c2 = __shfl(my_kv, (q4 >> 16) & 0xFFu);
    int c3 = __shfl(my_kv, (q4 >> 24) & 0xFFu);
    return (c0 & 0xFF) | ((c1 & 0xFF) << 8) | ((c2 & 0xFF) << 16) | ((c3 & 0xFF) << 24);
}

extern "C" __global__ void __launch_bounds__(32) grim_dot4_iq4xs_q81_gemv_sub(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_iq4xs,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub  = lane & 7;
    const int slot = lane >> 3;
    const int kidx = lane & 15;
    const int my_kv = (kidx == 0) ? -127 : (kidx == 1) ? -104 : (kidx == 2) ? -83
                    : (kidx == 3) ? -65  : (kidx == 4) ? -49  : (kidx == 5) ? -35
                    : (kidx == 6) ? -22  : (kidx == 7) ? -10  : (kidx == 8) ? 1
                    : (kidx == 9) ? 13   : (kidx == 10) ? 25  : (kidx == 11) ? 38
                    : (kidx == 12) ? 53  : (kidx == 13) ? 69  : (kidx == 14) ? 89 : 113;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * 136;
    const unsigned char* b0 = B_iq4xs + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb_base = 0; sb_base < n_sb; sb_base += 4) {
        const int sb = sb_base + slot;
        const bool sb_valid = (sb < n_sb);
        float d_a = 0.0f;
        int a4[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (sb_valid) {
            const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
            d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
            __builtin_memcpy(a4, a_blk + 4, 32);
        }

        // Preload active_cols headers + 16B qs per column into registers so
        // the wave-wide __shfl LUT below runs under a uniform 32-lane exec mask.
        float dl_col[4];
        uint2 qv0_col[4];
        uint2 qv1_col[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (sb_valid && j < active_cols) {
                const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * 136;
                const unsigned hdr = ((const unsigned*)b_sb)[0];
                const float d = fp16_to_float_device((unsigned short)(hdr & 0xFFFFu));
                const unsigned scales_h = hdr >> 16;
                const unsigned sc_l4 = ((const unsigned*)b_sb)[1];
                const int ls = (int)((sc_l4 >> (4 * sub)) & 0x0Fu)
                             | (((int)((scales_h >> (2 * sub)) & 3u)) << 4);
                dl_col[j] = d * (float)(ls - 32);
                const uint2* qs2 = (const uint2*)(b_sb + 8 + sub * 16);
                qv0_col[j] = qs2[0];
                qv1_col[j] = qs2[1];
            } else {
                dl_col[j] = 0.0f;
                qv0_col[j] = make_uint2(0u, 0u);
                qv1_col[j] = make_uint2(0u, 0u);
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned qw[4] = {qv0_col[j].x, qv0_col[j].y, qv1_col[j].x, qv1_col[j].y};
            int pos = 0;
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                pos = grim_sdot4(a4[w],     grim_kv4_iq4xs_shfl(my_kv, qw[w] & 0x0F0F0F0Fu),        pos);
                pos = grim_sdot4(a4[w + 4], grim_kv4_iq4xs_shfl(my_kv, (qw[w] >> 4) & 0x0F0F0F0Fu), pos);
            }
            facc[j] += dl_col[j] * d_a * (float)pos;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

// Step 2.2 (#1 + #2): IQ4_XS sub-block-per-lane GEMV v2:
//   (#1) 4-column batched 64-bit `uint2` header+scale loads (`((const uint2*)b_sb)[0]`)
//        and 128-bit `qs` loads before the sudot4 loop.
//   (#2) Pure VALU register lookup for `kvalues_iq4nl` via `__builtin_amdgcn_perm`,
//        replacing 128 `ds_bpermute_b32` (`__shfl`) instructions per super-block
//        with single-cycle VALU byte permutes so the DS unit is zero-contended.
__device__ __forceinline__ unsigned grim_kv1_iq4xs_valu(unsigned c) {
    const unsigned lo8 = __builtin_amdgcn_perm(0xF6EADDCFu, 0xBFAD9881u, c & 7u);
    const unsigned hi8 = __builtin_amdgcn_perm(0x71594535u, 0x26190D01u, c & 7u);
    return ((c & 8u) ? hi8 : lo8) & 0xFFu;
}

__device__ __forceinline__ int grim_kv4_iq4xs_valu(unsigned int q4) {
    const unsigned b0 = grim_kv1_iq4xs_valu(q4 & 0xFFu);
    const unsigned b1 = grim_kv1_iq4xs_valu((q4 >> 8) & 0xFFu);
    const unsigned b2 = grim_kv1_iq4xs_valu((q4 >> 16) & 0xFFu);
    const unsigned b3 = grim_kv1_iq4xs_valu((q4 >> 24) & 0xFFu);
    return (int)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

extern "C" __global__ void __launch_bounds__(32) grim_dot4_iq4xs_q81_gemv_sub_v2(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_iq4xs,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x;
    const int col_base = blockIdx.x * 4;
    const int row = blockIdx.y;
    if (row >= M || col_base >= N) return;

    const int sub  = lane & 7;
    const int slot = lane >> 3;

    const int n_sb = K / GRIM_Q4K_SUPERBLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * (K / GRIM_Q8_1_BLOCK_SIZE) * GRIM_Q8_1_BYTES;
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const long long col_stride = (long long)n_sb * 136;
    const unsigned char* b0 = B_iq4xs + (long long)col_base * col_stride;

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = slot; sb < n_sb; sb += 4) {
        const unsigned char* a_blk = a_row + (sb * 8 + sub) * GRIM_Q8_1_BYTES;
        const float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        int a4[8];
        __builtin_memcpy(a4, a_blk + 4, 32);

        uint2 hdr_col[4];
        uint2 qv0_col[4];
        uint2 qv1_col[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j < active_cols) {
                const unsigned char* b_sb = b0 + j * col_stride + (long long)sb * 136;
                hdr_col[j] = ((const uint2*)b_sb)[0];
                const uint2* qs2 = (const uint2*)(b_sb + 8 + sub * 16);
                qv0_col[j] = qs2[0];
                qv1_col[j] = qs2[1];
            } else {
                hdr_col[j] = make_uint2(0u, 0u);
                qv0_col[j] = make_uint2(0u, 0u);
                qv1_col[j] = make_uint2(0u, 0u);
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned hdr = hdr_col[j].x;
            const unsigned sc_l4 = hdr_col[j].y;
            const float d = fp16_to_float_device((unsigned short)(hdr & 0xFFFFu));
            const unsigned scales_h = hdr >> 16;
            const int ls = (int)((sc_l4 >> (4 * sub)) & 0x0Fu)
                         | (((int)((scales_h >> (2 * sub)) & 3u)) << 4);
            const float dl = d * (float)(ls - 32);

            const unsigned qw[4] = {qv0_col[j].x, qv0_col[j].y, qv1_col[j].x, qv1_col[j].y};
            int pos = 0;
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                pos = grim_sdot4(a4[w],     grim_kv4_iq4xs_valu(qw[w] & 0x0F0F0F0Fu),        pos);
                pos = grim_sdot4(a4[w + 4], grim_kv4_iq4xs_valu((qw[w] >> 4) & 0x0F0F0F0Fu), pos);
            }
            facc[j] += dl * d_a * (float)pos;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }
    if (lane == 0) {
        for (int j = 0; j < active_cols; j++)
            C[(long long)row * N + col_base + j] = facc[j];
    }
}

"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The Q2_0 kernel must be present in the JIT source under the exact name
    /// the launcher passes to `launch_compute_kernel`. A typo here would not
    /// fail at build time -- the kernel is resolved by name at first launch.
    #[test]
    fn source_contains_q2_0_gemv_kernel() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot4_q2_0_q80_gemv"),
            "Q2_0 GEMV kernel missing from the JIT source"
        );
        // Geometry constants must match grim-tensor's `expected_bytes`, or the
        // launcher would read rows with the wrong stride.
        assert!(KERNEL_SOURCE.contains("#define GRIM_Q2_0_BLOCK_SIZE 64"));
        assert!(KERNEL_SOURCE.contains("#define GRIM_Q2_0_BYTES      18"));
        // The codebook offset is the whole reason Q2_0 is not Q2_K or GSQRCO.
        assert!(
            KERNEL_SOURCE.contains("((int)((byte >> 0) & 3) - 1)"),
            "Q2_0 decode must map {{0,1,2,3}} -> {{-1,0,+1,+2}} (bias 1), not GSQRCO's bias 2"
        );
        // One weight block must consume exactly two activation blocks.
        assert!(KERNEL_SOURCE.contains("blk * 2 + 1"));
    }

    /// The Q2_0 kernel must be compiled on the same target set as the rest of
    /// this wave32-only source, and the dispatch must gate on it. A kernel
    /// defined outside the `#if` would compile everywhere but be wrong on
    /// wave64, where a 32-lane block's `__shfl_xor` reduction spans the full
    /// 64-wide wave with lanes 32..63 outside the block.
    #[test]
    fn q2_0_kernel_sits_inside_the_wave32_guard() {
        let start = KERNEL_SOURCE
            .find("#if defined(__gfx1030__)")
            .expect("wave32 guard header");
        let kern = KERNEL_SOURCE
            .find("__global__ void grim_dot4_q2_0_q80_gemv")
            .expect("Q2_0 kernel");
        // Walk the guard to its MATCHING `#endif`. A plain `find("#endif")`
        // lands on the one closing `grim_sdot4`'s nested RDNA2-vs-RDNA3/4
        // `#if`, which ends long before this kernel.
        let mut depth = 0usize;
        let mut end = None;
        let mut byte_offset = 0usize;
        for line in KERNEL_SOURCE[start..].lines() {
            let t = line.trim_start();
            if t.starts_with("#if") {
                depth += 1;
            } else if t.starts_with("#endif") {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + byte_offset + line.len());
                    break;
                }
            }
            byte_offset += line.len() + 1; // +1 for newline
        }
        let end = end.expect("wave32 guard footer");
        assert!(
            kern > start && kern < end,
            "Q2_0 kernel must live inside the wave32 #if guard (kernel at {kern}, \
             guard spans {start}..{end})"
        );
    }

    #[test]
    fn source_contains_dot4_gemv_kernel() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot4_q80_q81_gemv"),
            "missing dot4 Q8_0xQ8_1 GEMV kernel"
        );
    }

    #[test]
    fn source_contains_quantize_q8_1() {
        assert!(
            KERNEL_SOURCE.contains("grim_quantize_q8_1"),
            "missing Q8_1 activation quantizer"
        );
    }

    #[test]
    fn source_contains_q81_residual_add_experiment() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot4_q80_q81_add_gemv"),
            "missing Q8.1 down-projection residual-add experiment"
        );
    }

    #[test]
    fn source_uses_signed_dot4() {
        assert!(
            KERNEL_SOURCE.contains("amdgcn_sudot4"),
            "must use __builtin_amdgcn_sudot4 for signed int8 dot (V_DOT4_I32_IU8)"
        );
    }

    #[test]
    fn source_retains_legacy_dot2_gemv() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot2_q80_gemv"),
            "legacy dot2 GEMV retained for A/B testing"
        );
    }

    #[test]
    fn source_contains_256_thread_gate_up_experiment() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot4_gate_up_silu_q80_f32act_gemv_256"),
            "missing 256-thread eight-wave gate/up experiment"
        );
    }

    #[test]
    fn source_contains_16_output_add_tile_experiment() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot4_q80_f32act_add_gemv_tile16"),
            "missing one-wave 16-output residual GEMV experiment"
        );
    }

    #[test]
    fn source_contains_8_output_add_tile_experiment() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot4_q80_f32act_add_gemv_tile8"),
            "missing one-wave 8-output residual GEMV experiment"
        );
    }

    #[test]
    fn source_contains_dot2_bf16_gemv() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot2_bf16_gemv"),
            "missing dot2 BF16 GEMV kernel"
        );
        assert!(
            KERNEL_SOURCE.contains("grim_fdot2_f32_bf16"),
            "missing fdot2 BF16 dot helper"
        );
    }
}

#[cfg(test)]
mod sdot4_probe {
    use crate::RocmDevice;
    use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};

    /// RED-GREEN bisect for RDNA2 dot4 divergence: stage 1 — GPU
    /// `grim_quantize_q8_1` output must equal the host reference bit-for-bit.
    #[test]
    fn quantize_q81_stage_matches_host_on_rdna2() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        if !dev.gpu_target.starts_with("gfx103") {
            eprintln!("[quantize-probe] skip: target {} not RDNA2", dev.gpu_target);
            return;
        }

        let (m, k) = (1usize, 256usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.08).collect();

        let f32ty = DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        };
        let u8ty = DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        };
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, &a_bytes, &Shape::new(vec![m * k]), f32ty.clone())
                .unwrap();
        let n_q_blocks = k / 32;
        let dst_len = m * n_q_blocks * 36;
        let d_st = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; dst_len],
            &Shape::new(vec![dst_len]),
            u8ty.clone(),
        )
        .unwrap();

        let mut src_ptr = a_st.device_ptr().unwrap() as *mut std::ffi::c_void;
        let mut dst_ptr = d_st.device_ptr().unwrap() as *mut std::ffi::c_void;
        let mut kk = k as i32;
        let mut mm = m as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &mut src_ptr as *mut _ as *mut std::ffi::c_void,
            &mut dst_ptr as *mut _ as *mut std::ffi::c_void,
            &mut kk as *mut _ as *mut std::ffi::c_void,
            &mut mm as *mut _ as *mut std::ffi::c_void,
        ];
        let total_blocks = (n_q_blocks * m) as u32;
        dev.launch_compute_kernel(
            "grim_quantize_q8_1",
            crate::HipDim3::new(total_blocks, 1, 1),
            crate::HipDim3::new(32, 1, 1),
            &mut args,
        )
        .expect("quantize probe launch");
        dev.synchronize();

        let got_bytes: Vec<u8> = d_st
            .as_any()
            .downcast_ref::<crate::RocmStorage>()
            .unwrap()
            .copy_to_host()
            .unwrap();
        // Host reference (same code as tests/dot_gemv_parity.rs::host_quantize_q8_1).
        let mut want = vec![0u8; dst_len];
        for blk in 0..n_q_blocks {
            let base = blk * 32;
            let mut amax = 0.0f32;
            for j in 0..32 {
                amax = amax.max(a[base + j].abs());
            }
            let inv_d = if amax > 1e-9 { 127.0 / amax } else { 0.0 };
            let d = (amax / 127.0).max(1e-30);
            let d_bits = half::f16::from_f32(d).to_bits();
            let off = blk * 36;
            want[off] = (d_bits & 0xFF) as u8;
            want[off + 1] = (d_bits >> 8) as u8;
            let _ = inv_d;
            for j in 0..32 {
                let qi = (a[base + j] * (1.0 / d)).round();
                want[off + 4 + j] = (qi as i8) as u8;
            }
        }
        let mut mismatches = 0;
        for i in 0..dst_len {
            // Kernel also writes the f16 sum at [2..4] per 36-byte block;
            // the host reference leaves it zero. Skip those documented bytes.
            if i % 36 >= 2 && i % 36 < 4 {
                continue;
            }
            if got_bytes[i] != want[i] {
                if mismatches < 8 {
                    eprintln!(
                        "[quantize-probe] byte {i}: gpu={} host={}",
                        got_bytes[i], want[i]
                    );
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "quantize stage diverges on {} ({mismatches} bytes)",
            dev.gpu_target
        );
        eprintln!(
            "[quantize-probe] quantize stage MATCHES host on {}",
            dev.gpu_target
        );

        // ---- Stage 2: the q4k GEMV kernel with HOST-quantized activations,
        // vs CPU two-dot using grim_quant::dequant_q4k. Isolates the GEMV
        // kernel from the (now-green) quantize stage.
        let n = 4usize;
        let b_f32: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let b_packed = grim_quant::quant_q4k(&b_f32).expect("quant_q4k");
        let bty = DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(grim_tensor::KQuantScheme::Q4K),
        };
        let b_st = MemoryOps::from_cpu_bytes(
            &dev,
            &b_packed,
            &Shape::new(vec![b_packed.len()]),
            bty.clone(),
        )
        .unwrap();
        let o_st = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; (n + 1) * 4],
            &Shape::new(vec![n + 1]),
            f32ty.clone(),
        )
        .unwrap();

        // Host q8_1 for row 0 (from the stage-1 reference).
        let mut act_q81 = vec![0u8; n_q_blocks * 36];
        for blk in 0..n_q_blocks {
            let base = blk * 32;
            let mut amax = 0.0f32;
            for j in 0..32 {
                amax = amax.max(a[base + j].abs());
            }
            let d = (amax / 127.0).max(1e-30);
            let d_bits = half::f16::from_f32(d).to_bits();
            let off = blk * 36;
            act_q81[off] = (d_bits & 0xFF) as u8;
            act_q81[off + 1] = (d_bits >> 8) as u8;
            let mut fsum = 0.0f32;
            for j in 0..32 {
                let qi = (a[base + j] * (1.0 / d)).round() as i8;
                act_q81[off + 4 + j] = qi as u8;
                fsum += qi as f32;
            }
            // Kernel writes sum_a = fsum * d as f16 at [2..4] — the two-dot
            // GEMV consumes it for the dmin correction.
            let s_bits = half::f16::from_f32(fsum * d).to_bits();
            act_q81[off + 2] = (s_bits & 0xFF) as u8;
            act_q81[off + 3] = (s_bits >> 8) as u8;
        }
        let act_st = MemoryOps::from_cpu_bytes(
            &dev,
            &act_q81,
            &Shape::new(vec![act_q81.len()]),
            u8ty.clone(),
        )
        .unwrap();

        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut args2: Vec<*mut std::ffi::c_void> = vec![
            &mut (act_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _
                as *mut std::ffi::c_void,
            &mut (b_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _
                as *mut std::ffi::c_void,
            &mut (o_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _
                as *mut std::ffi::c_void,
            &mut mm as *mut _ as *mut std::ffi::c_void,
            &mut nn as *mut _ as *mut std::ffi::c_void,
            &mut kk as *mut _ as *mut std::ffi::c_void,
        ];
        dev.launch_compute_kernel(
            "grim_dot4_q4k_q81_gemv",
            crate::HipDim3::new(n as u32, 1, 1),
            crate::HipDim3::new(32, 1, 1),
            &mut args2,
        )
        .expect("gemv probe launch");
        dev.synchronize();

        let got_f32 = o_st
            .as_any()
            .downcast_ref::<crate::RocmStorage>()
            .unwrap()
            .copy_to_host()
            .unwrap();
        let got_f32: Vec<f32> = got_f32
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!("[probe] warpSize on {}: {}", dev.gpu_target, got_f32[n]);

        // CPU reference: dequant_q4k weights dot host-quantized activations.
        let row_len = (k / 256) * 144;
        let mut worst = 0.0f32;
        for col in 0..n {
            let b_deq =
                grim_quant::dequant_q4k(&b_packed[col * row_len..(col + 1) * row_len], k).unwrap();
            let mut acc = 0.0f32;
            for blk in 0..n_q_blocks {
                let d = half::f16::from_bits(u16::from_le_bytes([
                    act_q81[blk * 36],
                    act_q81[blk * 36 + 1],
                ]))
                .to_f32();
                for j in 0..32 {
                    let code = act_q81[blk * 36 + 4 + j] as i8;
                    acc += code as f32 * d * b_deq[blk * 32 + j];
                }
            }
            let diff = (acc - got_f32[col]).abs();
            worst = worst.max(diff);
            eprintln!(
                "[gemv-probe] col {col}: gpu={} cpu={acc} diff={diff}",
                got_f32[col]
            );
        }
        assert!(
            worst < 1e-3,
            "q4k GEMV diverges with host q8_1 on {}: {worst}",
            dev.gpu_target
        );
        eprintln!(
            "[gemv-probe] q4k GEMV MATCHES host two-dot on {}",
            dev.gpu_target
        );

        // ---- Stage 3: replicate the FAILING parity test exactly (n=128,
        // its activation values, GPU-quantized activations through the same
        // launch the dispatch uses) and bisect per-superblock.
        let n3 = 128usize;
        // EXACT data from the failing parity test (seed 0x1337 LCG).
        let mut seed = 0x1337u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let a3: Vec<f32> = (0..m * k).map(|_| rand()).collect();
        let b3_f32: Vec<f32> = (0..n3 * k).map(|_| rand()).collect();
        let b3_packed = grim_quant::quant_q4k(&b3_f32).expect("quant_q4k");
        let b3_st = MemoryOps::from_cpu_bytes(
            &dev,
            &b3_packed,
            &Shape::new(vec![b3_packed.len()]),
            bty.clone(),
        )
        .unwrap();
        let a3_bytes: Vec<u8> = a3.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a3_st =
            MemoryOps::from_cpu_bytes(&dev, &a3_bytes, &Shape::new(vec![m * k]), f32ty.clone())
                .unwrap();
        // GPU-quantize into a fresh buffer (same launch as the dispatch).
        let q81_len = n_q_blocks * 36;
        let q81_st = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; q81_len],
            &Shape::new(vec![q81_len]),
            u8ty.clone(),
        )
        .unwrap();
        let mut src_ptr = a3_st.device_ptr().unwrap() as *mut std::ffi::c_void;
        let mut dst_ptr = q81_st.device_ptr().unwrap() as *mut std::ffi::c_void;
        let mut kk = k as i32;
        let mut mm = m as i32;
        let mut qargs: Vec<*mut std::ffi::c_void> = vec![
            &mut src_ptr as *mut _ as *mut std::ffi::c_void,
            &mut dst_ptr as *mut _ as *mut std::ffi::c_void,
            &mut kk as *mut _ as *mut std::ffi::c_void,
            &mut mm as *mut _ as *mut std::ffi::c_void,
        ];
        dev.launch_compute_kernel(
            "grim_quantize_q8_1",
            crate::HipDim3::new((n_q_blocks * m) as u32, 1, 1),
            crate::HipDim3::new(32, 1, 1),
            &mut qargs,
        )
        .expect("stage3 quantize launch");
        dev.synchronize();

        let o3_st = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; n3 * 4],
            &Shape::new(vec![n3]),
            f32ty.clone(),
        )
        .unwrap();
        let mut mm3 = m as i32;
        let mut nn3 = n3 as i32;
        let mut args3: Vec<*mut std::ffi::c_void> = vec![
            &mut (q81_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _
                as *mut std::ffi::c_void,
            &mut (b3_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _
                as *mut std::ffi::c_void,
            &mut (o3_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _
                as *mut std::ffi::c_void,
            &mut mm3 as *mut _ as *mut std::ffi::c_void,
            &mut nn3 as *mut _ as *mut std::ffi::c_void,
            &mut kk as *mut _ as *mut std::ffi::c_void,
        ];
        dev.launch_compute_kernel(
            "grim_dot4_q4k_q81_gemv",
            crate::HipDim3::new((n3 as u32).div_ceil(4), 1, 1),
            crate::HipDim3::new(32, 1, 1),
            &mut args3,
        )
        .expect("stage3 gemv launch");
        dev.synchronize();

        let got3: Vec<f32> = o3_st
            .as_any()
            .downcast_ref::<crate::RocmStorage>()
            .unwrap()
            .copy_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Per-column CPU ref + first-diverging column's per-superblock partials.
        let row_len = (k / 256) * 144;
        // GPU q8_1 dump for the activation side.
        let gpu_q81: Vec<u8> = q81_st
            .as_any()
            .downcast_ref::<crate::RocmStorage>()
            .unwrap()
            .copy_to_host()
            .unwrap();
        let mut red_cols = 0;
        for col in 0..n3 {
            let b_deq =
                grim_quant::dequant_q4k(&b3_packed[col * row_len..(col + 1) * row_len], k).unwrap();
            let mut acc = 0.0f32;
            for blk in 0..n_q_blocks {
                let d = half::f16::from_bits(u16::from_le_bytes([
                    gpu_q81[blk * 36],
                    gpu_q81[blk * 36 + 1],
                ]))
                .to_f32();
                for j in 0..32 {
                    let code = gpu_q81[blk * 36 + 4 + j] as i8;
                    acc += code as f32 * d * b_deq[blk * 32 + j];
                }
            }
            let diff = (acc - got3[col]).abs();
            if diff > 1e-3 && red_cols < 3 {
                red_cols += 1;
                eprintln!(
                    "[stage3] RED col {col}: gpu={} cpu={acc} diff={diff}",
                    got3[col]
                );
            }
        }
        if red_cols == 0 {
            eprintln!(
                "[stage3] ALL {n3} columns match CPU within 1e-3 — divergence is NOT in the kernels"
            );
        } else {
            eprintln!(
                "[stage3] {red_cols}+ columns diverge — kernels differ from dequant reference"
            );
        }

        // ---- Stage 4: the SAME data through quantized_matmul (the dispatch
        // the parity test uses). Isolates dispatch plumbing vs kernels.
        unsafe {
            std::env::set_var("GRIM_DOT_GEMV", "1");
        }
        let o4_shape = Shape::new(vec![m, n3]);
        use grim_tensor::QuantOps;
        let (o4_st, o4_h) = QuantOps::quantized_matmul(
            &dev,
            a3_st.as_ref(),
            b3_st.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q8_0,
            &o4_shape,
        )
        .expect("stage4 quantized_matmul");
        o4_h.synchronize().unwrap();
        let got4: Vec<f32> = o4_st
            .as_any()
            .downcast_ref::<crate::RocmStorage>()
            .unwrap()
            .copy_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Repeat the dispatch call 3x: if only the FIRST call (with the
        // capability-profiler sweep interleaved) diverges, the bug is
        // first-call profiler interference, not steady-state dispatch.
        for rep_i in 0..3usize {
            let (rep_st, rep_h) = QuantOps::quantized_matmul(
                &dev,
                a3_st.as_ref(),
                b3_st.as_ref(),
                &[],
                grim_tensor::QuantFormat::Q8_0,
                &o4_shape,
            )
            .expect("repeat quantized_matmul");
            rep_h.synchronize().unwrap();
            let rep: Vec<f32> = rep_st
                .as_any()
                .downcast_ref::<crate::RocmStorage>()
                .unwrap()
                .copy_to_host()
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut worst = 0.0f32;
            for col in 0..n3 {
                let b_deq =
                    grim_quant::dequant_q4k(&b3_packed[col * row_len..(col + 1) * row_len], k)
                        .unwrap();
                let mut acc = 0.0f32;
                for blk in 0..n_q_blocks {
                    let d = half::f16::from_bits(u16::from_le_bytes([
                        gpu_q81[blk * 36],
                        gpu_q81[blk * 36 + 1],
                    ]))
                    .to_f32();
                    for j in 0..32 {
                        let code = gpu_q81[blk * 36 + 4 + j] as i8;
                        acc += code as f32 * d * b_deq[blk * 32 + j];
                    }
                }
                worst = worst.max((acc - rep[col]).abs());
            }
            eprintln!(
                "[stage5] dispatch call {}: worst diff = {}",
                rep_i + 1,
                worst
            );
        }
        let mut red4 = 0;
        for col in 0..n3 {
            let b_deq =
                grim_quant::dequant_q4k(&b3_packed[col * row_len..(col + 1) * row_len], k).unwrap();
            let mut acc = 0.0f32;
            for blk in 0..n_q_blocks {
                let d = half::f16::from_bits(u16::from_le_bytes([
                    gpu_q81[blk * 36],
                    gpu_q81[blk * 36 + 1],
                ]))
                .to_f32();
                for j in 0..32 {
                    let code = gpu_q81[blk * 36 + 4 + j] as i8;
                    acc += code as f32 * d * b_deq[blk * 32 + j];
                }
            }
            let diff = (acc - got4[col]).abs();
            if diff > 1e-3 && red4 < 3 {
                red4 += 1;
                eprintln!(
                    "[stage4] RED col {col}: gpu={} cpu={acc} diff={diff}",
                    got4[col]
                );
            }
        }
        eprintln!(
            "[stage4] {} divergent columns via dispatch (GRIM_DOT_GEMV=1)",
            red4
        );
        unsafe {
            std::env::set_var("GRIM_DOT_GEMV", "0");
        }
    }
}
