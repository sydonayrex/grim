//! Q3_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q3_K`]

/// HIP source for `grim_fused_dequant_gemm_q3k` and `grim_fused_dequant_backward_gemm_q3k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize one Q3_K element from a super-block.
    /// `in_sb` is the weight index within the 256-weight super-block (0..255).
    __device__ inline float dequant_q3k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        // Q3_K format: 8 sub-blocks of 32 weights
        // Sub-block scales are stored in sc bytes with upper bits in qh bytes
        const unsigned char* sc = block_ptr + 4;
        const unsigned char* qh = block_ptr + 12;
        const unsigned char* qs = block_ptr + 14;
        const unsigned char* m = block_ptr + 78; // 14 + 64 = 78

        int sub = in_sb / 32;
        int in_sub = in_sb % 32;

        // 3-bit minimum scale value (lower 3 bits of sc byte for this sub-block)
        // plus 2 extra bits from qh
        unsigned char sc_byte = sc[sub];
        float sub_sc = (float)(sc_byte & 7); // 3 bits

        // Upper 3 bits of scale for this sub-block come from qh
        // qh packs 8 × 3 = 24 bits → 3 bytes
        int qh_byte = sub / 8;
        int qh_bit  = (sub % 8) * 3;
        float sc_upper = (float)((qh[qh_byte] >> qh_bit) & 7);
        float scale_total = sub_sc + sc_upper * 8.0f; // combine: 3+3 = 6 bits → 0..63

        // Sub-block minimum: 3 bits from m byte, same upper bits from qh
        unsigned char m_byte = m[sub];
        float sub_m = (float)(m_byte & 7);
        float m_upper = (float)((qh[qh_byte + 3] >> qh_bit) & 7);
        float m_total = sub_m + m_upper * 8.0f;

        // 3-bit code: 3 bits per weight, packed 8 per byte (but actually 3 bits:
        // byte_index = (in_sub * 3) / 8, bit_offset = (in_sub * 3) % 8 → cross-byte unpack)
        int bit_pos = in_sub * 3;
        int byte_idx = bit_pos / 8;
        int bit_idx  = bit_pos % 8;

        unsigned int q_value;
        if (bit_idx <= 5) {
            // All 3 bits are within one byte
            q_value = (qs[byte_idx] >> bit_idx) & 0x07;
        } else {
            // Bits span two bytes
            int bits_in_first = 8 - bit_idx; // 1 or 2 bits
            int bits_in_second = 3 - bits_in_first;
            q_value = (qs[byte_idx] >> bit_idx) & ((1 << bits_in_first) - 1);
            q_value |= ((qs[byte_idx + 1] & ((1 << bits_in_second) - 1)) << bits_in_first);
        }

        return d * scale_total * (float)q_value - dmin * m_total;
    }

    __global__ void grim_fused_dequant_gemm_q3k(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q3k,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;

        const int row = (int)(idx / N);
        const int col = (int)(idx % N);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 108;
        const unsigned char* row_b_ptr = B_q3k + col * row_bytes;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            const unsigned char* block_ptr = row_b_ptr + sb_idx * 108;
            float w_val = dequant_q3k_element(block_ptr, in_sb);
            acc += a_val * w_val;
        }

        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_q3k(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_q3k,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 108;

        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q3k + n * row_bytes + sb_idx * 108;
            float w_val = dequant_q3k_element(block_ptr, in_sb);
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
    fn q3k_kernel_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_gemm_q3k"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_q3k"));
        assert!(KERNEL_SOURCE.contains("dequant_q3k_element"));
    }
}
