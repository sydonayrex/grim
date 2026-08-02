//! Q5_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q5_K`]

/// HIP source for `grim_fused_dequant_gemm_q5k` and `grim_fused_dequant_backward_gemm_q5k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize one Q5_K element from a 176-byte super-block.
    /// `in_sb` is the weight index within the 256-weight super-block (0..255).
    __device__ inline float dequant_q5k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs = block_ptr + 16;
        const unsigned char* qh = block_ptr + 144;

        int is = in_sb / 32; // 0..7 sub-block index

        // 6-bit scale unpacking (same as Q4_K): sc and m each 3 bits, packed 4 per byte
        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        // Low 4 bits from qs (2 codes per byte, alternating nibble placement)
        int q_idx = in_sb / 2;
        unsigned char packed = qs[q_idx];
        unsigned char q_low = (in_sb % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

        // MSB (5th bit) from qh: one bit per weight packed sequentially,
        // stored as one byte per 8 weights (32 bytes / 256 weights → 1 bit per weight)
        int qh_byte = in_sb / 8;
        int qh_bit  = in_sb % 8;
        unsigned char msb = (qh[qh_byte] >> qh_bit) & 1;

        // Full 5-bit code: low 4 bits + msb shifted to bit 4
        int q_code = (int)q_low | ((int)msb << 4);

        return d * (float)sc * (float)q_code - dmin * (float)m;
    }

    __global__ void grim_fused_dequant_gemm_q5k(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q5k,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;

        const int row = (int)(idx / N);
        const int col = (int)(idx % N);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 176;
        const unsigned char* row_b_ptr = B_q5k + col * row_bytes;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            const unsigned char* block_ptr = row_b_ptr + sb_idx * 176;
            float w_val = dequant_q5k_element(block_ptr, in_sb);
            acc += a_val * w_val;
        }

        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_q5k(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_q5k,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 176;

        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q5k + n * row_bytes + sb_idx * 176;
            float w_val = dequant_q5k_element(block_ptr, in_sb);
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
    fn q5k_kernel_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("dequant_q5k_element"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_gemm_q5k"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_q5k"));
    }
}
