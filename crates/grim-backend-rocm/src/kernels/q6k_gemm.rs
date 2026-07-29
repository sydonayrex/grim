//! Q6_K Fused Dequantization GEMM HIP kernel (Crow Tier).
//!
//! Dequantizes llama.cpp `block_q6_K` super-blocks (256 weights,
//! 6-bit scales, 6-bit codes with two extra MSB planes) on-the-fly
//! inside HIP GEMM loops for forward and backward passes.
//!
//! Layout matches llama.cpp's `block_q6_K` (see ggml-common.h):
//! ```c
//! typedef struct {
//!     ggml_half d;          // 2 bytes - super-block scale
//!     ggml_half dmin;       // 2 bytes - super-block minimum
//!     unsigned char sc[12]; // 12 bytes  - packed 6-bit sub-block scales
//!     unsigned char qs[128];// 128 bytes - 4-bit low-nibble codes
//!     unsigned char qh[64]; // 64 bytes  - upper 2 bits per weight (2 bits per weight × 256 = 512 bits = 64 bytes)
//! } block_q6_K;   // 210 bytes total per 256 weights
//! ```

/// HIP source for `grim_fused_dequant_gemm_q6k` and `grim_fused_dequant_backward_gemm_q6k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize one Q6_K element from a 210-byte super-block.
    /// `in_sb` is the weight index within the 256-weight super-block (0..255).
    __device__ inline float dequant_q6k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs = block_ptr + 16;
        const unsigned char* qh = block_ptr + 144;

        int is = in_sb / 32; // 0..7 sub-block index
        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        // Low 4 bits from qs (same as Q4K/Q5K)
        int q_idx = in_sb / 2;
        unsigned char packed = qs[q_idx];
        unsigned char q_low = (in_sb % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

        // Upper 2 bits from qh: 4 weights per byte (2 bits each),
        // stored as bits [7:6] of each byte → (qh_byte * 4 + qh_group) gives weight index offset
        // qh layout: 2 bits per weight, 4 weights per byte → byte_index = in_sb / 4, bit_offset = (in_sb % 4) * 2
        int qh_byte_idx = in_sb / 4;
        int qh_bit_offset = (in_sb % 4) * 2;
        unsigned char qh_bits = (qh[qh_byte_idx] >> qh_bit_offset) & 0x03;

        // Full 6-bit code: bits [4:0] from low nibble + msb + qh_bits
        // Low nibble is bits [3:0], MSB of 5th bit is bit 4, upper 2 bits are bits [5:4] from qh
        int q_code = (int)q_low | ((int)qh_bits << 4);

        return d * (float)sc * (float)q_code - dmin * (float)m;
    }

    __global__ void grim_fused_dequant_gemm_q6k(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q6k,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;

        const int row = (int)(idx / N);
        const int col = (int)(idx % N);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 210;
        const unsigned char* row_b_ptr = B_q6k + col * row_bytes;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            const unsigned char* block_ptr = row_b_ptr + sb_idx * 210;
            float w_val = dequant_q6k_element(block_ptr, in_sb);
            acc += a_val * w_val;
        }

        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_q6k(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_q6k,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 210;

        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q6k + n * row_bytes + sb_idx * 210;
            float w_val = dequant_q6k_element(block_ptr, in_sb);
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
    fn q6k_kernel_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("dequant_q6k_element"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_gemm_q6k"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_q6k"));
    }
}
