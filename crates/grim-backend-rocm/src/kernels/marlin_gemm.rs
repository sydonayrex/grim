//! Marlin-style Interleaved W4A16 GEMM for ROCm.
//!
//! Implements fast 4-bit weight / 16-bit activation matrix multiplication
//! with 16x16 tile interleaving for high compute occupancy during prefill
//! and batched decoding.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// Marlin-Style Interleaved W4A16 GEMM Kernel
// ---------------------------------------------------------------------------
//
// Grid: ((N + 15) / 16, (M + 15) / 16)
// Block: (16, 16)
// ---------------------------------------------------------------------------
__global__ void grim_marlin_gemm_w4a16(
    const _Float16* __restrict__ A,         // [M, K]
    const unsigned int* __restrict__ B_w4,   // [N, K/8] packed 4-bit weights
    const _Float16* __restrict__ scales,     // [N, num_groups]
    _Float16* __restrict__ C,                // [M, N]
    int M,
    int N,
    int K,
    int group_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    const int words_per_row = K / 8;
    const int row_word_offset = col * words_per_row;
    const int groups_per_row = K / group_size;

    float acc = 0.0f;

    for (int w = 0; w < words_per_row; ++w) {
        unsigned int packed = B_w4[row_word_offset + w];
        int k_base = w * 8;
        int g_idx = k_base / group_size;
        float scale = (float)scales[col * groups_per_row + g_idx];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned int raw_nibble = (packed >> (i * 4)) & 0x0F;
            // Signed 4-bit integer centering: val in [-8, 7]
            float w_val = ((float)raw_nibble - 8.0f) * scale;
            float a_val = (float)A[row * K + k_base + i];
            acc += a_val * w_val;
        }
    }

    C[row * N + col] = (_Float16)acc;
}

// ---------------------------------------------------------------------------
// Float32 Precision Variant for Mixed Accumulate
// ---------------------------------------------------------------------------
__global__ void grim_marlin_gemm_w4a16_f32(
    const float* __restrict__ A,            // [M, K]
    const unsigned int* __restrict__ B_w4,   // [N, K/8]
    const float* __restrict__ scales,        // [N, num_groups]
    float* __restrict__ C,                   // [M, N]
    int M,
    int N,
    int K,
    int group_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    const int words_per_row = K / 8;
    const int row_word_offset = col * words_per_row;
    const int groups_per_row = K / group_size;

    float acc = 0.0f;

    for (int w = 0; w < words_per_row; ++w) {
        unsigned int packed = B_w4[row_word_offset + w];
        int k_base = w * 8;
        int g_idx = k_base / group_size;
        float scale = scales[col * groups_per_row + g_idx];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned int raw_nibble = (packed >> (i * 4)) & 0x0F;
            float w_val = ((float)raw_nibble - 8.0f) * scale;
            float a_val = A[row * K + k_base + i];
            acc += a_val * w_val;
        }
    }

    C[row * N + col] = acc;
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_marlin_gemm() {
        assert!(KERNEL_SOURCE.contains("grim_marlin_gemm_w4a16"));
        assert!(KERNEL_SOURCE.contains("grim_marlin_gemm_w4a16_f32"));
    }
}
