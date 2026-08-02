//! Q2_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q2_K`]
//  Total: 2+2+8+8+64+4 = 88 (llama.cpp uses 84; 4 bytes are sub-block header overhead)

/// HIP source for `grim_fused_dequant_gemm_q2k` and `grim_fused_dequant_backward_gemm_q2k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

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
