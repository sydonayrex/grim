//! Marlin W4A16 and AWQ GroupInt fused dequant-GEMM kernels for CUDA.

pub const COMPRESSED_GEMM_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__global__ void grim_marlin_gemm_w4a16(
    const half* __restrict__ A,
    const unsigned int* __restrict__ B_w4,
    const half* __restrict__ scales,
    half* __restrict__ C,
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
        float scale = __half2float(scales[col * groups_per_row + g_idx]);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned int raw_nibble = (packed >> (i * 4)) & 0x0F;
            float w_val = ((float)raw_nibble - 8.0f) * scale;
            float a_val = __half2float(A[row * K + k_base + i]);
            acc += a_val * w_val;
        }
    }

    C[row * N + col] = __float2half(acc);
}

__global__ void grim_awq_dequant_gemm(
    const float* __restrict__ A,
    const unsigned int* __restrict__ qweight,
    const unsigned int* __restrict__ qzeros,
    const unsigned short* __restrict__ scales,
    float* __restrict__ C,
    int M, int N, int K,
    int bits, int group_size,
    int values_per_word, int zeros_words_per_row
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    int groups_per_row = (K + group_size - 1) / group_size;

    for (int k = 0; k < K; ++k) {
        int group = k / group_size;
        long long word_idx = (long long)(k / values_per_word) * N + col;
        unsigned int w_word = qweight[word_idx];
        unsigned int code = (w_word >> ((k % values_per_word) * bits)) & ((1u << bits) - 1u);

        long long z_word_idx = (long long)group * zeros_words_per_row + col / values_per_word;
        unsigned int z_word = qzeros[z_word_idx];
        float zero = (float)((z_word >> ((col % values_per_word) * bits)) & ((1u << bits) - 1u));

        unsigned short h = scales[col * groups_per_row + group];
        float scale = __half2float(*((const half*)&h));

        float w_val = (float)((float)code - zero) * scale;
        acc += A[row * K + k] * w_val;
    }

    C[row * N + col] = acc;
}

}
"#;
