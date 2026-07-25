//! WMMA matrix-core GEMM HIP kernel (WI-G).
//!
//! Provides the JIT compilation source for Wave Matrix Multiply-Accumulate (WMMA)
//! operations on GFX11+ (RDNA3/RDNA4) architectures. To allow compilation and testing
//! on GFX10 (RDNA2, e.g. gfx1036), the kernel uses preprocessor guards to fall back
//! to a scalar thread-level GEMM when compiled on non-WMMA architectures.

/// HIP source for `grim_wmma_gemm`.
///
/// Concatenated into the crate-wide JIT compilation source. On GFX11+ targets,
/// it compiles using Clang/HIP's rocWMMA headers or compiler builtins. On older
/// architectures, it compiles to a scalar fallback so compilation succeeds.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

extern "C" __global__ void grim_wmma_gemm(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    // Wave Matrix Multiply-Accumulate implementation using rocWMMA.
    // Coops use 16x16x16 tiles.
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;

    fill_fragment(frag_c, 0.0f);

    // Loop over the K dimension in steps of 16.
    for (int k = 0; k < K; k += 16) {
        load_matrix_coop_sync(frag_a, A + k, stride_a);
        load_matrix_coop_sync(frag_b, B + k * stride_b, stride_b);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    store_matrix_coop_sync(C, frag_c, stride_c, layout_t::mem_row_major);
}
#else
// Fallback path for GFX10 / RDNA2 and other architectures without native WMMA support.
// Executes as a scalar thread-element dot product.
extern "C" __global__ void grim_wmma_gemm(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = (float)A[row * stride_a + k];
        float b_val = (float)B[k * stride_b + col];
        acc += a_val * b_val;
    }

    C[row * stride_c + col] = (_Float16)acc;
}
#endif

// ---------- Raven FP8 Kernels ----------

// Helper FP8 E4M3 to Float conversion in HIP
__device__ inline float fp8_e4m3_to_float_hip(unsigned char val) {
    if (val == 0x7F) return 0.0f / 0.0f; // NaN
    if (val == 0xFF) return -0.0f / 0.0f;
    int sign = (val >> 7) & 1;
    int exp = (val >> 3) & 0x0F;
    int mant = val & 0x07;
    if (exp == 0) {
        float res = (float)mant / 8.0f * 0.000015258789f; // 2^-16
        return sign ? -res : res;
    }
    float res = (1.0f + (float)mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
    return sign ? -res : res;
}

// Native FP8 WMMA / Scalar Fallback GEMM
extern "C" __global__ void grim_wmma_gemm_fp8(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = fp8_e4m3_to_float_hip(A[row * stride_a + k]);
        float b_val = fp8_e4m3_to_float_hip(B[k * stride_b + col]);
        acc += a_val * b_val;
    }

    C[row * stride_c + col] = acc;
}

// FP8 Fused Dequant GEMM Forward
extern "C" __global__ void grim_fused_dequant_gemm_fp8(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = A[row * K + k];
        float b_val = fp8_e4m3_to_float_hip(B_fp8[col * K + k]);
        acc += a_val * b_val;
    }

    C[row * N + col] = acc;
}

// FP8 Fused Dequant Backward GEMM
extern "C" __global__ void grim_fused_dequant_backward_gemm_fp8(
    const float* __restrict__ dY,
    const unsigned char* __restrict__ B_fp8,
    float* __restrict__ dX,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K;
    if (idx >= total) return;

    const int row = idx / K;
    const int k_idx = idx % K;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dy_val = dY[row * N + n];
        float b_val = fp8_e4m3_to_float_hip(B_fp8[n * K + k_idx]);
        acc += dy_val * b_val;
    }

    dX[row * K + k_idx] = acc;
}

// ---------- Jay (MXFP4) & Magpie (MXFP8) Kernels ----------

// Helper MXFP4 E2M1 + E8M0 shared exponent to Float conversion
__device__ inline float mxfp4_to_float_hip(unsigned char code, unsigned char shared_exp) {
    int sign = (code >> 3) & 1;
    int exp = (code >> 1) & 3;
    int mant = code & 1;
    float base_val = 0.0f;
    if (exp == 0) {
        base_val = (float)mant * 0.5f;
    } else {
        base_val = (1.0f + (float)mant * 0.5f) * powf(2.0f, (float)exp - 1.0f);
    }
    if (sign) base_val = -base_val;
    float scale = powf(2.0f, (float)shared_exp - 127.0f);
    return base_val * scale;
}

// Jay MXFP4 Fused Dequant GEMM Forward
extern "C" __global__ void grim_fused_dequant_gemm_mxfp4(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_codes,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = A[row * K + k];
        int block_idx = (col * K + k) / 32;
        unsigned char exp_val = B_exps[block_idx];
        int elem_flat = col * K + k;
        int code_byte_idx = elem_flat / 2;
        unsigned char packed_byte = B_codes[code_byte_idx];
        unsigned char code = (elem_flat % 2 == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
        float b_val = mxfp4_to_float_hip(code, exp_val);
        acc += a_val * b_val;
    }

    C[row * N + col] = acc;
}

// Magpie MXFP8 Fused Dequant GEMM Forward
extern "C" __global__ void grim_fused_dequant_gemm_mxfp8(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = A[row * K + k];
        int block_idx = (col * K + k) / 32;
        unsigned char exp_val = B_exps[block_idx];
        float scale = powf(2.0f, (float)exp_val - 127.0f);
        float b_val = fp8_e4m3_to_float_hip(B_fp8[col * K + k]) * scale;
        acc += a_val * b_val;
    }

    C[row * N + col] = acc;
}
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

    /// Verifies the presence of the JIT kernel entry symbol in the HIP literal.
    #[test]
    fn source_contains_wmma_kernel_entry() {
        assert!(
            KERNEL_SOURCE.contains("extern \"C\" __global__ void grim_wmma_gemm"),
            "WMMA GEMM kernel entry must be JIT-discoverable by name"
        );
        assert!(KERNEL_SOURCE.contains("_Float16"), "kernel must use _Float16 type");
    }
}
