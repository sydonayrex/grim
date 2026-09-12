//! SPEED-DOT: vector dot-product GEMV kernels for M=1..4 decode on RDNA3/4.
//!
//! Uses __builtin_amdgcn_sudot4 (V_DOT4_I32_IU8, signed x signed) to execute
//! fast vec_dot_q8_0_q8_1 mat-vec multiplication with Q8_1 activation quantization.

/// Full JIT source for Q8_0 x Q8_1 dot-product GEMV kernels.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)

__device__ __forceinline__ int grim_sdot4(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
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
    int8_t qi = (int8_t)__builtin_roundf(val * inv_d);

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
        ((int8_t*)(row_dst + 4))[tid] = qi;
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
        const int8_t* a_codes = (const int8_t*)(a_blk + 4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const int8_t* b_codes = (const int8_t*)(b_blk + 2);

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
                const int8_t* a_codes = (const int8_t*)(a_blk + 4);

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
                    const int8_t* codes = (const int8_t*)(a_blk + 4);

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
                const int8_t* a_codes = (const int8_t*)(a_blk + 4);

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
                const int8_t* a_codes = (const int8_t*)(a_blk + 4);

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
                const int8_t* a_codes = (const int8_t*)(a_blk + 4);

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

#endif // RDNA3/RDNA4
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

