//! IQ2/IQ3/IQ4 family fused dequantization GEMM CUDA kernels.
//!
//! Implements native CUDA fwd+bwd kernels for all IQ quant formats:
//! IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS.
//!
//! Block layouts match grim-quant's CPU reference exactly so parity tests pass.

pub const IQ_GEMM_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

// ---------------------------------------------------------------------------
// FP16 → float helper (no __half2float in all runtimes)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp16_to_float_device(unsigned short h) {
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
// Per-format dequant device functions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float dequant_iq2xxs(const unsigned char* blk, int in_sb) {
    float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
    const unsigned char* qs = blk + 2;
    const unsigned char* signs = blk + 34;
    int grid_idx = qs[in_sb / 8];
    float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
    float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
    return d * val * sign_val;
}

__device__ __forceinline__ float dequant_iq2xs(const unsigned char* blk, int in_sb) {
    float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
    const unsigned char* qs = blk + 2;
    const unsigned char* scales = blk + 34;
    const unsigned char* signs = blk + 42;
    int sb = in_sb / 16;
    float sc = ((float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F)) * 0.125f + 0.5f;
    float scale = d * sc;
    int grid_idx = qs[in_sb / 8];
    float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
    float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
    return scale * val * sign_val;
}

__device__ __forceinline__ float dequant_iq2s(const unsigned char* blk, int in_sb) {
    float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
    const unsigned char* qs = blk + 2;
    const unsigned char* scales = blk + 50;
    const unsigned char* signs = blk + 58;
    int sb = in_sb / 16;
    float sc = ((float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F)) * 0.125f + 0.5f;
    float scale = d * sc;
    int grid_idx = qs[in_sb / 8];
    float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
    float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
    return scale * val * sign_val;
}

__device__ __forceinline__ float dequant_iq3xxs(const unsigned char* blk, int in_sb) {
    float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
    const unsigned char* qs = blk + 2;
    const unsigned char* signs = blk + 66;
    int grid_idx = qs[in_sb / 8];
    int sub_idx = in_sb % 8;
    float base_val = (float)((grid_idx + sub_idx * 17) % 7) - 3.0f;
    int sign_byte_idx = (in_sb / 8);
    if (sign_byte_idx >= 30) sign_byte_idx = 29;
    float sign_val = ((signs[sign_byte_idx] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
    return d * base_val * 0.25f * sign_val;
}

__device__ __forceinline__ float dequant_iq3s(const unsigned char* blk, int in_sb) {
    float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
    const unsigned char* qs = blk + 2;
    const unsigned char* scales = blk + 66;
    const unsigned char* signs = blk + 78;
    int sb = in_sb / 32;
    float sc = ((float)(scales[sb * 12 / 8]) + 1.0f) * 0.125f;
    float scale = d * sc;
    float grid_val = (float)((qs[in_sb / 8] + in_sb) % 7) - 3.0f;
    float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
    return scale * grid_val * sign_val;
}

__device__ __constant__ float KVALUES_IQ4NL[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
      1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  87.0f, 107.0f
};

__device__ __forceinline__ float dequant_iq4nl(const unsigned char* blk, int in_sb) {
    float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
    const unsigned char* signs = blk + 2;
    const unsigned char* qs = blk + 34;
    const unsigned char* sc = blk + 162;
    int group = in_sb / 16;
    float group_scale = 1.0f + 0.125f * (float)(sc[group] & 3);
    int sign_byte_idx = in_sb / 8;
    int sign_bit = in_sb % 8;
    float sign_val = ((signs[sign_byte_idx] >> sign_bit) & 1) ? -1.0f : 1.0f;
    int q_byte = in_sb / 2;
    unsigned char q_code = (in_sb % 2 == 0) ? (qs[q_byte] & 0x0F) : ((qs[q_byte] >> 4) & 0x0F);
    float code_abs = KVALUES_IQ4NL[q_code];
    code_abs = code_abs < 0.0f ? -code_abs : code_abs;
    return d * group_scale * code_abs * sign_val;
}

__device__ __forceinline__ float dequant_iq4xs(const unsigned char* blk, int in_sb) {
    float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
    const unsigned char* sc = blk + 2;
    const unsigned char* qs = blk + 8;
    int group = in_sb / 32;
    int sc_byte_idx = (group * 6) / 8;
    int sc_bit_offset = (group * 6) % 8;
    unsigned int sc_val = sc[sc_byte_idx] >> sc_bit_offset;
    if (sc_bit_offset > 2) {
        sc_val |= (unsigned int)sc[sc_byte_idx + 1] << (8 - sc_bit_offset);
    }
    sc_val &= 0x3F;
    int q_byte = in_sb / 2;
    unsigned char q_code = (in_sb % 2 == 0) ? (qs[q_byte] & 0x0F) : ((qs[q_byte] >> 4) & 0x0F);
    return d * (float)sc_val * (float)q_code;
}

// ---------------------------------------------------------------------------
// Macro to emit forward + backward GEMM for each IQ format
// ---------------------------------------------------------------------------

#define GRIM_IQ_FWD_KERNEL(NAME, FMT, BLOCK_BYTES) \
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
        acc += A[row * K + k] * FMT(rowB + sb * BLOCK_BYTES, isb); \
    } \
    C[row * N + col] = acc; \
}

#define GRIM_IQ_BWD_KERNEL(NAME, FMT, BLOCK_BYTES) \
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
        acc += dY[row * N + n] * FMT(B + n * bpr * BLOCK_BYTES + sb * BLOCK_BYTES, isb); \
    } \
    dX[row * K + k_idx] = acc; \
}

GRIM_IQ_FWD_KERNEL(iq2xxs, dequant_iq2xxs, 66)
GRIM_IQ_BWD_KERNEL(iq2xxs, dequant_iq2xxs, 66)

GRIM_IQ_FWD_KERNEL(iq2xs, dequant_iq2xs, 74)
GRIM_IQ_BWD_KERNEL(iq2xs, dequant_iq2xs, 74)

GRIM_IQ_FWD_KERNEL(iq2s, dequant_iq2s, 82)
GRIM_IQ_BWD_KERNEL(iq2s, dequant_iq2s, 82)

GRIM_IQ_FWD_KERNEL(iq3xxs, dequant_iq3xxs, 96)
GRIM_IQ_BWD_KERNEL(iq3xxs, dequant_iq3xxs, 96)

GRIM_IQ_FWD_KERNEL(iq3s, dequant_iq3s, 110)
GRIM_IQ_BWD_KERNEL(iq3s, dequant_iq3s, 110)

GRIM_IQ_FWD_KERNEL(iq4nl, dequant_iq4nl, 170)
GRIM_IQ_BWD_KERNEL(iq4nl, dequant_iq4nl, 170)

GRIM_IQ_FWD_KERNEL(iq4xs, dequant_iq4xs, 136)
GRIM_IQ_BWD_KERNEL(iq4xs, dequant_iq4xs, 136)

}
"#;
