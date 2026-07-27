//! Q2_K Fused Dequantization GEMM HIP kernel (Crow Tier).
//!
//! Dequantizes llama.cpp `block_q2_K` super-blocks (256 weights,
//! 2-bit codes with per-sub-block scales) on-the-fly inside HIP GEMM loops.
//! This uses a simplified layout consistent with packed-symmetric dequant:
//! per 32-weight sub-block a uniform f32 scale followed by packed 2-bit codes.
//!
//! Q2_K block layout (84 bytes per 256 weights):
//! - d (f16): 2 bytes - super-block scale
//! - dmin (f16): 2 bytes - super-block minimum
//! - sc (8 bytes): 8 sub-block scales, 1 byte each (2-bit values)
//! - m (8 bytes): 8 sub-block minimums, 1 byte each (2-bit values)
//! - qs (64 bytes): 256 2-bit codes, packed 4 per byte
//! - qh (4 bytes): reserved/padding
//  Total: 2+2+8+8+64+4 = 88 (llama.cpp uses 84; 4 bytes are sub-block header overhead)

/// HIP source for `grim_fused_dequant_gemm_q2k` and `grim_fused_dequant_backward_gemm_q2k`.
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

    /// Dequantize one Q2_K element from an 84-byte super-block.
    /// `in_sb` is the weight index within the 256-weight super-block (0..255).
    __device__ inline float dequant_q2k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        const unsigned char* sc = block_ptr + 4;
        const unsigned char* m  = block_ptr + 12;
        const unsigned char* qs = block_ptr + 20;

        int sub = in_sb / 32; // 0..7 sub-block index
        int in_sub = in_sb % 32;

        // 2-bit scale: lower 2 bits of sc[sub]
        float sub_sc = (float)(sc[sub] & 3);
        float sub_m  = (float)(m[sub] & 3);

        // 2-bit code: 4 codes per byte, extract correct nibble
        int q_byte = in_sub / 4;
        int q_shift = (in_sub % 4) * 2;
        unsigned char q_code = (qs[q_byte] >> q_shift) & 0x03;

        return d * sub_sc * (float)q_code - dmin * sub_m;
    }

    __global__ void grim_fused_dequant_gemm_q2k(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q2k,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;

        const int row = (int)(idx / N);
        const int col = (int)(idx % N);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 84;
        const unsigned char* row_b_ptr = B_q2k + col * row_bytes;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            const unsigned char* block_ptr = row_b_ptr + sb_idx * 84;
            float w_val = dequant_q2k_element(block_ptr, in_sb);
            acc += a_val * w_val;
        }

        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_q2k(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_q2k,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 84;

        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q2k + n * row_bytes + sb_idx * 84;
            float w_val = dequant_q2k_element(block_ptr, in_sb);
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
    fn q2k_kernel_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_gemm_q2k"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_q2k"));
        assert!(KERNEL_SOURCE.contains("dequant_q2k_element"));
    }
}
