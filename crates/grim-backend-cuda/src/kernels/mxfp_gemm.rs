//! Microscaling FP4 / FP8 and K-quant fused GEMM kernels for CUDA.

pub const MXFP_GEMM_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__device__ __constant__ float MXFP4_E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float mxfp4_block_scale(unsigned char shared_exp) {
    return exp2f((float)(int)shared_exp - 127.0f);
}

__global__ void grim_mxfp4_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_codes,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ C,
    int M, int N, int K)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)M * N;
    if (idx >= total) return;

    int row = (int)(idx / N);
    int col = (int)(idx % N);

    int num_blocks = K / 32;
    int codes_per_row = K / 2;
    const unsigned char* row_codes = B_codes + col * codes_per_row;
    const unsigned char* row_exps = B_exps + col * num_blocks;

    float acc = 0.0f;
    for (int blk = 0; blk < num_blocks; ++blk) {
        float scale = mxfp4_block_scale(row_exps[blk]);
        const unsigned char* blk_codes = row_codes + blk * 16;
        for (int i = 0; i < 16; ++i) {
            unsigned char b = blk_codes[i];
            float w0 = MXFP4_E2M1_LUT[b & 0x0F] * scale;
            float w1 = MXFP4_E2M1_LUT[(b >> 4) & 0x0F] * scale;
            int k0 = blk * 32 + i * 2;
            acc += A[row * K + k0] * w0 + A[row * K + k0 + 1] * w1;
        }
    }
    C[row * N + col] = acc;
}
}
"#;
