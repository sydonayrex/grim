//! Decode-shaped FP16 GEMM CUDA kernel.
//!
//! Ported from grim-backend-rocm `kernels/decode_gemm.rs`.
//! Small-M (decode batch) FP16 GEMM: C[M,N] = A[M,K] @ B[K,N], f32 accumulate, FP16 out.
//! Uses `__half` (CUDA) in place of `_Float16` (HIP).

pub const DECODE_GEMM_SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void grim_decode_gemm_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    // Decode-shape FP16 GEMM: C[M,N] = A[M,K] @ B[K,N], f32 accumulate, FP16 out.
    // One thread per output element. F32 accumulation avoids catastrophic
    // cancellation at small M. Validated parity with cuBLAS at M=1..8.
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    const __half* a_row = A + row * stride_a;
    float acc = 0.0f;

    for (int k = 0; k < K; ++k)
        acc += __half2float(a_row[k]) * __half2float(B[k * stride_b + col]);

    C[row * stride_c + col] = __float2half(acc);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_decode_gemm_entry() {
        assert!(DECODE_GEMM_SOURCE.contains("grim_decode_gemm_f16"));
        assert!(DECODE_GEMM_SOURCE.contains("__half"));
        assert!(DECODE_GEMM_SOURCE.contains("float acc = 0.0f"));
    }
}
