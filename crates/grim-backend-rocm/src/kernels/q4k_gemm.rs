//! Q4_K Fused Dequantization GEMM HIP kernel (Crow Tier).
//!
//! Dequantizes llama.cpp `block_q4_K` super-blocks (256 weights, 6-bit scales, 4-bit codes)
//! on-the-fly inside HIP GEMM loops for forward and backward passes.

/// HIP source for `grim_fused_dequant_gemm_q4k` and `grim_fused_dequant_backward_gemm_q4k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __device__ inline float fp16_to_float_device(unsigned short h) {
        unsigned int sign = (h >> 15) & 1;
        unsigned int exp  = (h >> 10) & 0x1f;
        unsigned int mant = h & 0x3ff;
        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            float res = (float)mant / 1024.0f * 0.00006103515625f;
            return sign ? -res : res;
        } else if (exp == 31) {
            return sign ? -1.0f/0.0f : 1.0f/0.0f;
        }
        float res = (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
        return sign ? -res : res;
    }

    __device__ inline float dequant_q4k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs = block_ptr + 16;

        int is = in_sb / 32; // 0..7
        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        int q_idx = in_sb / 2;
        unsigned char packed = qs[q_idx];
        unsigned char q_code = (in_sb % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

        return d * (float)sc * (float)q_code - dmin * (float)m;
    }

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
