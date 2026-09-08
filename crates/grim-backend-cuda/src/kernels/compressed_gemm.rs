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

// OCP FP8 E4M3 dequantization to float
__device__ __forceinline__ float fp8_e4m3_to_f32_dev(unsigned char byte) {
    bool sign = (byte & 0x80) != 0;
    int exp = (byte >> 3) & 0x0F;
    int mant = byte & 0x07;

    if (exp == 15 && mant == 7) {
        return 0.0f; // NaN
    }
    float val;
    if (exp == 0) {
        val = (float)mant / 512.0f;
    } else {
        val = (1.0f + (float)mant / 8.0f) * exp2f((float)(exp - 7));
    }
    return sign ? -val : val;
}

// CompressedTensors W8A8 FP8 GEMM:
// Layout in B_bytes: [u64 scale_len (8 bytes)][scales F32 (scale_len bytes)][FP8 codes (K*N bytes)]
// Or unpadded scales: scale_len = N * 4 for per-channel scales.
__global__ void grim_w8a8_fp8_dequant_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_bytes,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    // Read scale_len prefix
    unsigned long long scale_len = *((const unsigned long long*)B_bytes);
    const float* scales = (const float*)(B_bytes + 8);
    const unsigned char* codes = B_bytes + 8 + scale_len;

    float scale = (scale_len >= (unsigned long long)(N * 4)) ? scales[col] : scales[0];

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        unsigned char code = codes[col * K + k];
        float w_val = fp8_e4m3_to_f32_dev(code) * scale;
        acc += A[row * K + k] * w_val;
    }

    C[row * N + col] = acc;
}

// CompressedTensors W8A8 INT8 GEMM:
// Layout in B_bytes: [u64 scale_len (8 bytes)][scales F32 (scale_len bytes)][INT8 codes (K*N bytes)]
__global__ void grim_w8a8_int8_dequant_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_bytes,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    unsigned long long scale_len = *((const unsigned long long*)B_bytes);
    const float* scales = (const float*)(B_bytes + 8);
    const signed char* codes = (const signed char*)(B_bytes + 8 + scale_len);

    float scale = (scale_len >= (unsigned long long)(N * 4)) ? scales[col] : scales[0];

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        signed char code = codes[col * K + k];
        float w_val = (float)code * scale;
        acc += A[row * K + k] * w_val;
    }

    C[row * N + col] = acc;
}

}
"#;
