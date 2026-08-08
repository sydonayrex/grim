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
        const unsigned char* qh = block_ptr + 16;   // 32 bytes, 2 high bits/weight
        const unsigned char* qs = block_ptr + 48;   // 128 bytes, 4 low bits/weight

        // ggml layout: four 64-weight groups. Within group n, the first 32
        // weights take the low nibble of qs[n*32 + l] with high bit
        // qh[l] & (1 << 2n) and scale sub-block 2n; the next 32 take the high
        // nibble with bit qh[l] & (1 << (2n+1)) and scale sub-block 2n+1.
        int n = in_sb / 64;      // 0..3 group
        int j = in_sb % 64;      // 0..63 within group
        int l = j & 31;          // qs/qh byte index within the group
        int hi = j >> 5;         // 0 = low nibble, 1 = high nibble
        int is = 2 * n + hi;     // 0..7 scale sub-block

        // 6-bit scale unpacking (same as Q4_K): sc and m each 6 bits
        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        unsigned char packed = qs[n * 32 + l];
        unsigned char q_low = hi ? (packed >> 4) : (packed & 0x0F);
        unsigned char msb = (qh[l] >> (2 * n + hi)) & 1;

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
