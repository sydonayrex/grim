//! Q4_K Fused Dequantization GEMM HIP kernel (Crow Tier).
//!
//! Dequantizes llama.cpp `block_q4_K` super-blocks (256 weights, 6-bit scales, 4-bit codes)
//! on-the-fly inside HIP GEMM loops for forward and backward passes.

/// HIP source for `grim_fused_dequant_gemm_q4k` and `grim_fused_dequant_backward_gemm_q4k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __global__ void grim_fused_dequant_gemm_q4k(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q4k,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;

        const int row = (int)(idx / N);
        const int col = (int)(idx % N);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 144;
        const unsigned char* row_b_ptr = B_q4k + col * row_bytes;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            const unsigned char* block_ptr = row_b_ptr + sb_idx * 144;
            float w_val = dequant_q4k_element(block_ptr, in_sb);
            acc += a_val * w_val;
        }

        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_q4k(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_q4k,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 144;

        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q4k + n * row_bytes + sb_idx * 144;
            float w_val = dequant_q4k_element(block_ptr, in_sb);
            acc += dy_val * w_val;
        }

        dX[row * K + k_idx] = acc;
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_kernel_source_non_empty() {
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_gemm_q4k"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_q4k"));
    }
}
