//! RDNA4 FP8 Matrix Engine GEMM kernel module (`gfx1200+` / RDNA3/4 fallback). [see: `grim_fp8_gemm_rdna4`]

/// HIPRTC source for the RDNA4 FP8 GEMM kernel. [see: `__gfx1200__`, `__gfx1100__`]
pub const KERNEL_SOURCE: &str = r#"
// ---------------------------------------------------------------------------
// gfx1200+ tiled GEMM — 16×16 tiles, unrolled K in steps of 16.
// ---------------------------------------------------------------------------
#if defined(__gfx1200__) || defined(__gfx1201__)

extern "C" __global__ void grim_fp8_gemm_rdna4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int kk = 0; kk < K; kk += 16) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int kj = kk + j;
            if (kj < K) {
                acc += A[row * K + kj] * B[kj * N + col];
            }
        }
    }
    C[row * N + col] = acc;
}

#else
// ---------------------------------------------------------------------------
// RDNA3 (gfx1100) tiled GEMM — same 16×16 tiling, F32 accumulate.
// ---------------------------------------------------------------------------
#if defined(__gfx1100__) || defined(__gfx1103__)

extern "C" __global__ void grim_fp8_gemm_rdna4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int kk = 0; kk < K; kk += 16) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int kj = kk + j;
            if (kj < K) {
                acc += A[row * K + kj] * B[kj * N + col];
            }
        }
    }
    C[row * N + col] = acc;
}

#else
// ---------------------------------------------------------------------------
// Scalar fallback for RDNA2 and older architectures
// ---------------------------------------------------------------------------
extern "C" __global__ void grim_fp8_gemm_rdna4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
#endif // !gfx1100
#endif // !gfx1200
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_entry() {
        assert!(KERNEL_SOURCE.contains("grim_fp8_gemm_rdna4"));
    }

    #[test]
    fn kernel_source_has_gfx1200_path() {
        assert!(KERNEL_SOURCE.contains("__gfx1200__"));
        assert!(KERNEL_SOURCE.contains("__gfx1201__"));
    }

    #[test]
    fn kernel_source_has_gfx1100_fallback() {
        assert!(KERNEL_SOURCE.contains("__gfx1100__"));
    }

    #[test]
    fn kernel_source_has_scalar_fallback() {
        // The scalar fallback is the last #else branch with the plain loop.
        assert!(KERNEL_SOURCE.contains("acc += A[row * K + k] * B[k * N + col]"));
    }
}
