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
