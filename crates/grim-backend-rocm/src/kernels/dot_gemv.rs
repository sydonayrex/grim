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
    // All B operands in these kernels are < 128, so signed B is equivalent.
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    // RDNA3/4: dot8-insts (sdot4 removed on RDNA4).
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#endif
}

#define GRIM_Q8_1_BLOCK_SIZE 32
#define GRIM_Q8_1_BYTES      36
#define GRIM_Q8_0_BLOCK_SIZE 32
#define GRIM_Q8_0_BYTES      34

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

extern "C" __global__ void grim_dot4_q80_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q80,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

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
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_fdot4_f32_fp8_fp8)
    return __builtin_amdgcn_fdot4_f32_fp8_fp8(c, a, b);
#else
    // Portable fallback: unpack 4 E4M3 codes per operand and fma in f32.
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
    if (__builtin_isinf(a) || a >= 480.0f) return (unsigned char)(sign | 0x7E); // saturate to 448
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
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_chunk = b_col[j] + (long long)chunk * 32;
            float acc = 0.0f;
            #pragma unroll
            for (int e = 0; e < 32; e += 4) {
                int a4 = grim_pack4_fp8(a_chunk + e);
                int b4;
                __builtin_memcpy(&b4, b_chunk + e, 4);
                acc = grim_fdot4_fp8(acc, a4, b4);
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
// Grim Q2_K layout (matches grim_quant::dequant_q2k): 84 bytes per 256 weights:
// 16 scale bytes (4-bit sc + 4-bit m per 16-elem sub-block), 64 code bytes
// (4x 2-bit codes per byte), f16 d @80, f16 dmin @82.
// value_i = d*sc*q_i - dmin*m. Two-dot: dot = d*sc*Σ(a_i*q_i) - dmin*m*Σa_i.
// Activations are Q8_1: a_i = code_i * d_a, Σa_i = sum_a (stored in block).

__device__ __forceinline__ int grim_expand2_q2k(unsigned char b) {
    return (int)((b & 3u) | ((b >> 2 & 3u) << 8) | ((b >> 4 & 3u) << 16)
               | ((b >> 6 & 3u) << 24));
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
                float sum_a = fp16_to_float_device(((const unsigned short*)a_blk)[1]);
                const signed char* a_codes = (const signed char*)(a_blk + 4);

                for (int half = 0; half < 2; half++) {
                    const int sub = half_pair * 2 + half;
                    const float sc = (float)(b_sb[sub] & 0x0F);
                    const float mi = (float)(b_sb[sub] >> 4);

                    int pos = 0;
                    #pragma unroll
                    for (int i = 0; i < 16; i += 4) {
                        int a4;
                        __builtin_memcpy(&a4, a_codes + half * 16 + i, 4);
                        int q4 = grim_expand2_q2k(b_sb[16 + sub * 4 + i / 4]);
                        pos = grim_sdot4(a4, q4, pos);
                    }
                    facc[j] += d * sc * ((float)pos * d_a) - dmin * mi * sum_a;
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

// Quantize A [M, K] -> packed unsigned 4-bit nibbles + scale + sum of codes per 128-element group.
// 128 floats -> 16 uint32 words (each word packs 8 unsigned 4-bit nibbles, little-endian).
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

    const float d = local_max / 15.0f;
    const float inv_d = (local_max > 1e-9f) ? (15.0f / local_max) : 0.0f;

    unsigned int q0 = (unsigned int)fminf(fmaxf(__builtin_roundf(v0 * inv_d), 0.0f), 15.0f);
    unsigned int q1 = (unsigned int)fminf(fmaxf(__builtin_roundf(v1 * inv_d), 0.0f), 15.0f);
    unsigned int q2 = (unsigned int)fminf(fmaxf(__builtin_roundf(v2 * inv_d), 0.0f), 15.0f);
    unsigned int q3 = (unsigned int)fminf(fmaxf(__builtin_roundf(v3 * inv_d), 0.0f), 15.0f);

    int thread_sum = (int)(q0 + q1 + q2 + q3);
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
    unsigned int half_word = q0 | (q1 << 4) | (q2 << 8) | (q3 << 12);
    unsigned int other_half = __shfl_xor(half_word, 1);
    if ((tid & 1) == 0) {
        unsigned int word = half_word | (other_half << 16);
        grp_dst_codes[tid / 2] = word;
    }
}

// Native W4A4 GEMV kernel using sudot8 (v_dot8_i32_iu4)
// OSTQuant layout: B_qweight is [N, K/8] u32, B_scales is [N, K/128] bf16, B_zeros is [N, K/128] u8.
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
                iacc = __builtin_amdgcn_sudot8(false, a_word, false, b_word, iacc, false);
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
"#;

#[cfg(test)]
mod tests {
    use super::*;

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

        let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
        let u8ty = DType { arith: ArithType::U8, storage: Storage::Native };
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a_st = MemoryOps::from_cpu_bytes(&dev, &a_bytes, &Shape::new(vec![m * k]), f32ty.clone()).unwrap();
        let n_q_blocks = k / 32;
        let dst_len = m * n_q_blocks * 36;
        let d_st = MemoryOps::from_cpu_bytes(&dev, &vec![0u8; dst_len], &Shape::new(vec![dst_len]), u8ty.clone()).unwrap();

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
                    eprintln!("[quantize-probe] byte {i}: gpu={} host={}", got_bytes[i], want[i]);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "quantize stage diverges on {} ({mismatches} bytes)", dev.gpu_target);
        eprintln!("[quantize-probe] quantize stage MATCHES host on {}", dev.gpu_target);

        // ---- Stage 2: the q4k GEMV kernel with HOST-quantized activations,
        // vs CPU two-dot using grim_quant::dequant_q4k. Isolates the GEMV
        // kernel from the (now-green) quantize stage.
        let n = 4usize;
        let b_f32: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let b_packed = grim_quant::quant_q4k(&b_f32).expect("quant_q4k");
        let bty = DType { arith: ArithType::F32, storage: Storage::KQuant(grim_tensor::KQuantScheme::Q4K) };
        let b_st = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), bty.clone()).unwrap();
        let o_st = MemoryOps::from_cpu_bytes(&dev, &vec![0u8; (n + 1) * 4], &Shape::new(vec![n + 1]), f32ty.clone()).unwrap();

        // Host q8_1 for row 0 (from the stage-1 reference).
        let mut act_q81 = vec![0u8; n_q_blocks * 36];
        for blk in 0..n_q_blocks {
            let base = blk * 32;
            let mut amax = 0.0f32;
            for j in 0..32 { amax = amax.max(a[base + j].abs()); }
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
        let act_st = MemoryOps::from_cpu_bytes(&dev, &act_q81, &Shape::new(vec![act_q81.len()]), u8ty.clone()).unwrap();

        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut args2: Vec<*mut std::ffi::c_void> = vec![
            &mut (act_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _ as *mut std::ffi::c_void,
            &mut (b_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _ as *mut std::ffi::c_void,
            &mut (o_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _ as *mut std::ffi::c_void,
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
        let got_f32: Vec<f32> = got_f32.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        eprintln!("[probe] warpSize on {}: {}", dev.gpu_target, got_f32[n]);

        // CPU reference: dequant_q4k weights dot host-quantized activations.
        let row_len = (k / 256) * 144;
        let mut worst = 0.0f32;
        for col in 0..n {
            let b_deq = grim_quant::dequant_q4k(&b_packed[col * row_len..(col + 1) * row_len], k).unwrap();
            let mut acc = 0.0f32;
            for blk in 0..n_q_blocks {
                let d = half::f16::from_bits(u16::from_le_bytes([act_q81[blk * 36], act_q81[blk * 36 + 1]])).to_f32();
                for j in 0..32 {
                    let code = act_q81[blk * 36 + 4 + j] as i8;
                    acc += code as f32 * d * b_deq[blk * 32 + j];
                }
            }
            let diff = (acc - got_f32[col]).abs();
            worst = worst.max(diff);
            eprintln!("[gemv-probe] col {col}: gpu={} cpu={acc} diff={diff}", got_f32[col]);
        }
        assert!(worst < 1e-3, "q4k GEMV diverges with host q8_1 on {}: {worst}", dev.gpu_target);
        eprintln!("[gemv-probe] q4k GEMV MATCHES host two-dot on {}", dev.gpu_target);

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
        let b3_st = MemoryOps::from_cpu_bytes(&dev, &b3_packed, &Shape::new(vec![b3_packed.len()]), bty.clone()).unwrap();
        let a3_bytes: Vec<u8> = a3.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a3_st = MemoryOps::from_cpu_bytes(&dev, &a3_bytes, &Shape::new(vec![m * k]), f32ty.clone()).unwrap();
        // GPU-quantize into a fresh buffer (same launch as the dispatch).
        let q81_len = n_q_blocks * 36;
        let q81_st = MemoryOps::from_cpu_bytes(&dev, &vec![0u8; q81_len], &Shape::new(vec![q81_len]), u8ty.clone()).unwrap();
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
            crate::HipDim3::new(256, 1, 1),
            &mut qargs,
        )
        .expect("stage3 quantize launch");
        dev.synchronize();

        let o3_st = MemoryOps::from_cpu_bytes(&dev, &vec![0u8; n3 * 4], &Shape::new(vec![n3]), f32ty.clone()).unwrap();
        let mut mm3 = m as i32;
        let mut nn3 = n3 as i32;
        let mut args3: Vec<*mut std::ffi::c_void> = vec![
            &mut (q81_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _ as *mut std::ffi::c_void,
            &mut (b3_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _ as *mut std::ffi::c_void,
            &mut (o3_st.device_ptr().unwrap() as *mut std::ffi::c_void) as *mut _ as *mut std::ffi::c_void,
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
            let b_deq = grim_quant::dequant_q4k(&b3_packed[col * row_len..(col + 1) * row_len], k).unwrap();
            let mut acc = 0.0f32;
            for blk in 0..n_q_blocks {
                let d = half::f16::from_bits(u16::from_le_bytes([gpu_q81[blk * 36], gpu_q81[blk * 36 + 1]])).to_f32();
                for j in 0..32 {
                    let code = gpu_q81[blk * 36 + 4 + j] as i8;
                    acc += code as f32 * d * b_deq[blk * 32 + j];
                }
            }
            let diff = (acc - got3[col]).abs();
            if diff > 1e-3 && red_cols < 3 {
                red_cols += 1;
                eprintln!("[stage3] RED col {col}: gpu={} cpu={acc} diff={diff}", got3[col]);
            }
        }
        if red_cols == 0 {
            eprintln!("[stage3] ALL {n3} columns match CPU within 1e-3 — divergence is NOT in the kernels");
        } else {
            eprintln!("[stage3] {red_cols}+ columns diverge — kernels differ from dequant reference");
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
                let b_deq = grim_quant::dequant_q4k(&b3_packed[col * row_len..(col + 1) * row_len], k).unwrap();
                let mut acc = 0.0f32;
                for blk in 0..n_q_blocks {
                    let d = half::f16::from_bits(u16::from_le_bytes([gpu_q81[blk * 36], gpu_q81[blk * 36 + 1]])).to_f32();
                    for j in 0..32 {
                        let code = gpu_q81[blk * 36 + 4 + j] as i8;
                        acc += code as f32 * d * b_deq[blk * 32 + j];
                    }
                }
                worst = worst.max((acc - rep[col]).abs());
            }
            eprintln!("[stage5] dispatch call {}: worst diff = {}", rep_i + 1, worst);
        }
        let mut red4 = 0;
        for col in 0..n3 {
            let b_deq = grim_quant::dequant_q4k(&b3_packed[col * row_len..(col + 1) * row_len], k).unwrap();
            let mut acc = 0.0f32;
            for blk in 0..n_q_blocks {
                let d = half::f16::from_bits(u16::from_le_bytes([gpu_q81[blk * 36], gpu_q81[blk * 36 + 1]])).to_f32();
                for j in 0..32 {
                    let code = gpu_q81[blk * 36 + 4 + j] as i8;
                    acc += code as f32 * d * b_deq[blk * 32 + j];
                }
            }
            let diff = (acc - got4[col]).abs();
            if diff > 1e-3 && red4 < 3 {
                red4 += 1;
                eprintln!("[stage4] RED col {col}: gpu={} cpu={acc} diff={diff}", got4[col]);
            }
        }
        eprintln!("[stage4] {} divergent columns via dispatch (GRIM_DOT_GEMV=1)", red4);
        unsafe {
            std::env::set_var("GRIM_DOT_GEMV", "0");
        }
    }
}
