//! BitNet b1.58 (1.58-bit Ternary W1.58A8) GEMM for ROCm.
//!
//! Implements 2-bit packed ternary matrix multiplication (weights in {-1, 0, +1})
//! with zero-multiplication integer additions and subtractions.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// BitNet b1.58 Ternary GEMM Kernel (W1.58A8)
// ---------------------------------------------------------------------------
//
// Grid: ((N + 15) / 16, (M + 15) / 16)
// Block: (16, 16)
// ---------------------------------------------------------------------------
__global__ void grim_bitnet_gemm_w158a8(
    const float* __restrict__ A,                // [M, K] activations
    const unsigned char* __restrict__ B_ternary,// [N, K/4] 2-bit packed ternary weights
    const float* __restrict__ scale_b,          // [N] per-channel weight scales
    float* __restrict__ C,                      // [M, N] output
    int M,
    int N,
    int K,
    float scale_a                               // per-tensor activation scale
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    const int bytes_per_row = K / 4;
    const int row_byte_offset = col * bytes_per_row;
    const float w_scale = scale_b[col];
    const float total_scale = scale_a * w_scale;

    float acc = 0.0f;

    for (int b = 0; b < bytes_per_row; ++b) {
        unsigned char packed = B_ternary[row_byte_offset + b];
        int k_base = b * 4;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned char code = (packed >> (i * 2)) & 0x03;
            float a_val = A[row * K + k_base + i];

            // 00 -> 0, 01 -> +1, 10 -> -1
            if (code == 1) {
                acc += a_val;
            } else if (code == 2) {
                acc -= a_val;
            }
        }
    }

    C[row * N + col] = acc * total_scale;
}

// ---------------------------------------------------------------------------
// INT8 Activation Variant for Peak Integer Throughput
// ---------------------------------------------------------------------------
__global__ void grim_bitnet_gemm_w158a8_int8(
    const signed char* __restrict__ A_int8,     // [M, K] INT8 activations
    const unsigned char* __restrict__ B_ternary,// [N, K/4] 2-bit packed weights
    const float* __restrict__ scale_b,          // [N]
    float* __restrict__ C,                      // [M, N]
    int M,
    int N,
    int K,
    float scale_a
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    const int bytes_per_row = K / 4;
    const int row_byte_offset = col * bytes_per_row;
    const float w_scale = scale_b[col];
    const float total_scale = scale_a * w_scale;

    int int_acc = 0;

    for (int b = 0; b < bytes_per_row; ++b) {
        unsigned char packed = B_ternary[row_byte_offset + b];
        int k_base = b * 4;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned char code = (packed >> (i * 2)) & 0x03;
            int a_val = (int)A_int8[row * K + k_base + i];

            if (code == 1) {
                int_acc += a_val;
            } else if (code == 2) {
                int_acc -= a_val;
            }
        }
    }

    C[row * N + col] = (float)int_acc * total_scale;
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_bitnet_gemm() {
        assert!(KERNEL_SOURCE.contains("grim_bitnet_gemm_w158a8"));
        assert!(KERNEL_SOURCE.contains("grim_bitnet_gemm_w158a8_int8"));
    }
}
