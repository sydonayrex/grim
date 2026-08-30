//! Q2_K/Q3_K/Q4_K/Q5_K/Q6_K fused dequant-GEMM CUDA kernels.
//!
//! All five GGUF K-quant formats with forward and backward passes,
//! organized with shared dequant device helpers and a macro to avoid
//! code duplication per quant format.

pub const Q_GEMM_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

// ---------------------------------------------------------------------------
// FP16 → float helper
// ---------------------------------------------------------------------------
__device__ __forceinline__ float q_fp16_to_float(unsigned short h) {
    unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int e = ((unsigned int)(h & 0x7C00u)) << 13;
    unsigned int m = ((unsigned int)(h & 0x03FFu)) << 13;
    if ((h & 0x7FFFu) == 0) return __uint_as_float(s);
    if ((h & 0x7C00u) == 0x7C00u) return __uint_as_float(s | 0x7F800000u | (m ? 0x00400000u : 0u));
    if ((h & 0x7C00u) == 0) {
        float v = (float)(h & 0x03FFu) * (1.0f / 16777216.0f);
        return (h & 0x8000u) ? -v : v;
    }
    return __uint_as_float(s | e | m | 0x38000000u);
}

// ---------------------------------------------------------------------------
// Q2_K — 84 bytes per 256-element super-block.
// Layout: d(f16) + dmin(f16) + sc(8) + m(8) + qs(64)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_q2k(const unsigned char* blk, int in_sb) {
    float d    = q_fp16_to_float(((const unsigned short*)blk)[0]);
    float dmin = q_fp16_to_float(((const unsigned short*)blk)[1]);
    const unsigned char* sc = blk + 4;
    const unsigned char* m  = blk + 12;
    const unsigned char* qs = blk + 20;
    int sub = in_sb / 32, in_sub = in_sb % 32;
    float sub_sc = (float)(sc[sub] & 3);
    float sub_m  = (float)(m[sub]  & 3);
    int q_byte = in_sub / 4, q_shift = (in_sub % 4) * 2;
    unsigned char q_code = (qs[q_byte] >> q_shift) & 0x03;
    return d * sub_sc * (float)q_code - dmin * sub_m;
}

// ---------------------------------------------------------------------------
// Q3_K — 110 bytes per 256-element super-block.
// Layout: hmask(32) + qs(64) + scales(12) + d(f16) at offset 108
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_q3k(const unsigned char* blk, int in_sb) {
    const unsigned char* hmask  = blk + 0;
    const unsigned char* qs     = blk + 32;
    const unsigned char* scales = blk + 96;
    float d = q_fp16_to_float(((const unsigned short*)(blk + 108))[0]);
    int sub = in_sb / 32, is = (in_sb % 32) / 16, sc_idx = 2 * sub + is;
    int j = sc_idx & 7;
    signed char sc_byte;
    if (sc_idx < 8) {
        sc_byte = (signed char)((scales[j] & 0x0F) | ((scales[j + 8] & 0x03) << 4));
    } else {
        sc_byte = (signed char)((scales[j] >> 4) | ((scales[j + 8] & 0x0C) << 2));
    }
    float sc = (float)sc_byte - 32.0f;
    int in_sub = in_sb % 32;
    unsigned char hm_bit = (hmask[in_sub / 8] >> (in_sub % 8)) & 0x01;
    int col = in_sb / 32, byte_off = (col & 1) * 32, shift = (col & 6) >> 1;
    unsigned char qbits = (qs[in_sub + byte_off] >> (2 * shift)) & 0x03;
    int q_with_high = (int)qbits | (hm_bit ? 0 : 4);
    return d * sc * ((float)q_with_high - 4.0f);
}

// ---------------------------------------------------------------------------
// Q4_K — 144 bytes per 256-element super-block.
// Layout: d(f16) + dmin(f16) + scales(12) + qs(128)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_q4k(const unsigned char* blk, int in_sb) {
    float d    = q_fp16_to_float(((const unsigned short*)blk)[0]);
    float dmin = q_fp16_to_float(((const unsigned short*)blk)[1]);
    const unsigned char* scales = blk + 4;
    const unsigned char* qs     = blk + 16;
    int is = in_sb / 32;
    unsigned char sc, mn;
    if (is < 4) {
        sc = scales[is] & 63; mn = scales[is + 4] & 63;
    } else {
        sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
        mn = (scales[is + 4] >> 4)  | ((scales[is]     >> 6) << 4);
    }
    int q_idx = in_sb / 2;
    unsigned char packed = qs[q_idx];
    unsigned char q_code = (in_sb % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    return d * (float)sc * (float)q_code - dmin * (float)mn;
}

// ---------------------------------------------------------------------------
// Q5_K — 176 bytes per 256-element super-block.
// Layout: d(f16) + dmin(f16) + scales(12) + qs(128) + qh(32)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_q5k(const unsigned char* blk, int in_sb) {
    float d    = q_fp16_to_float(((const unsigned short*)blk)[0]);
    float dmin = q_fp16_to_float(((const unsigned short*)blk)[1]);
    const unsigned char* scales = blk + 4;
    const unsigned char* qs     = blk + 16;
    const unsigned char* qh     = blk + 144;
    int is = in_sb / 32;
    unsigned char sc, mn;
    if (is < 4) {
        sc = scales[is] & 63; mn = scales[is + 4] & 63;
    } else {
        sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
        mn = (scales[is + 4] >> 4)  | ((scales[is]     >> 6) << 4);
    }
    int q_idx = in_sb / 2;
    unsigned char q_low = (in_sb % 2 == 0) ? (qs[q_idx] & 0x0F) : ((qs[q_idx] >> 4) & 0x0F);
    unsigned char msb = (qh[in_sb / 8] >> (in_sb % 8)) & 1;
    int q_code = (int)q_low | ((int)msb << 4);
    return d * (float)sc * (float)q_code - dmin * (float)mn;
}

// ---------------------------------------------------------------------------
// Q6_K — 210 bytes per 256-element super-block.
// Layout: ql(128) + qh(64) + scales(16) + d(f16) at offset 208
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_q6k(const unsigned char* blk, int in_sb) {
    const unsigned char* ql     = blk + 0;
    const unsigned char* qh     = blk + 128;
    const unsigned char* scales = blk + 192;
    float d = q_fp16_to_float(((const unsigned short*)(blk + 208))[0]);
    int s = in_sb / 16, sc_val = (signed char)scales[s];
    int q_idx = in_sb % 128;
    int q_byte = q_idx / 2;
    unsigned char q_low = (in_sb % 2 == 0) ? (ql[q_byte] & 0x0F) : ((ql[q_byte] >> 4) & 0x0F);
    unsigned char q_hi_byte = qh[in_sb / 4];
    unsigned char q_high = (q_hi_byte >> ((in_sb % 4) * 2)) & 0x03;
    int q_code = (int)q_low | ((int)q_high << 4);
    return d * (float)sc_val * ((float)q_code - 32.0f);
}

// ---------------------------------------------------------------------------
// Macro to emit fwd + bwd kernels per K-quant format
// ---------------------------------------------------------------------------

#define GRIM_KQUANT_FWD(NAME, FN, BLOCK_BYTES) \
__global__ void grim_fused_dequant_gemm_##NAME( \
    const float* __restrict__ A, \
    const unsigned char* __restrict__ B, \
    float* __restrict__ C, \
    int M, int N, int K) \
{ \
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= (unsigned long long)M * N) return; \
    int row = (int)(idx / N), col = (int)(idx % N); \
    int bpr = K / 256; \
    const unsigned char* rowB = B + col * bpr * BLOCK_BYTES; \
    float acc = 0.0f; \
    for (int k = 0; k < K; ++k) { \
        int sb = k / 256, isb = k % 256; \
        acc += A[row * K + k] * FN(rowB + sb * BLOCK_BYTES, isb); \
    } \
    C[row * N + col] = acc; \
}

#define GRIM_KQUANT_BWD(NAME, FN, BLOCK_BYTES) \
__global__ void grim_fused_dequant_backward_gemm_##NAME( \
    const float* __restrict__ dY, \
    const unsigned char* __restrict__ B, \
    float* __restrict__ dX, \
    int M, int N, int K) \
{ \
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= (unsigned long long)M * K) return; \
    int row = (int)(idx / K), k_idx = (int)(idx % K); \
    int bpr = K / 256, sb = k_idx / 256, isb = k_idx % 256; \
    float acc = 0.0f; \
    for (int n = 0; n < N; ++n) { \
        acc += dY[row * N + n] * FN(B + n * bpr * BLOCK_BYTES + sb * BLOCK_BYTES, isb); \
    } \
    dX[row * K + k_idx] = acc; \
}

GRIM_KQUANT_FWD(q2k, dequant_q2k,  84)
GRIM_KQUANT_BWD(q2k, dequant_q2k,  84)

GRIM_KQUANT_FWD(q3k, dequant_q3k, 110)
GRIM_KQUANT_BWD(q3k, dequant_q3k, 110)

GRIM_KQUANT_FWD(q4k, dequant_q4k, 144)
GRIM_KQUANT_BWD(q4k, dequant_q4k, 144)

GRIM_KQUANT_FWD(q5k, dequant_q5k, 176)
GRIM_KQUANT_BWD(q5k, dequant_q5k, 176)

GRIM_KQUANT_FWD(q6k, dequant_q6k, 210)
GRIM_KQUANT_BWD(q6k, dequant_q6k, 210)

}
"#;
