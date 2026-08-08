//! Q6_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q6_K`]

/// HIP source for `grim_fused_dequant_gemm_q6k` and `grim_fused_dequant_backward_gemm_q6k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize one Q6_K element from a 210-byte super-block.
    /// `in_sb` is the weight index within the 256-weight super-block (0..255).
    ///
    /// Q6_K block layout (ggml `block_q6_K`, 210 bytes / 256 weights):
    ///   `ql`     128 bytes @ offset   0  — low 4 bits of each weight
    ///   `qh`      64 bytes @ offset 128  — high 2 bits of each weight
    ///   `scales`  16 bytes @ offset 192  — **signed** 8-bit per scale, no `dmin`
    ///   `d`        2 bytes @ offset 208  — single f16 scale
    /// Formula: `d * sc * (q - 32)` (note the `q - 32` centering; Q6_K has no
    /// `dmin`/min term, unlike Q4_K/Q5_K which use `d*sc*q - dmin*m`).
    ///
    /// This mirrors `grim_quant::dequant_q6k` (CPU reference, verified correct
    /// against ggml `dequantize_row_q6_K`) element-by-element. The CPU
    /// reference writes 128 outputs per outer stride `n in 0..1` in the order
    /// `[l, l+32, l+64, l+96]` for `l in 0..32`, i.e. `quarter = pos/32`
    /// selects which nibble/qh-bit-group applies. Inverting that mapping:
    ///   n        = in_sb / 128
    ///   pos      = in_sb % 128
    ///   quarter  = pos / 32        (0..3 → q1..q4 in the CPU code)
    ///   l        = pos % 32
    ///   is       = l / 16          (0 or 1, scale sub-block column)
    ///   sc_idx   = n*8 + is + 2*quarter   (the CPU's scales[sc_idx + {0,2,4,6}])
    ///   ql_offset= n*64 + l + (quarter & 1 ? 32 : 0)
    ///   nibble   = (quarter & 2) ? (ql_byte >> 4) : (ql_byte & 0x0F)
    ///   qh_byte  = qh[n*32 + l]
    ///   qh_bits  = (qh_byte >> (2*quarter)) & 0x03
    ///   q_code   = nibble | (qh_bits << 4)
    __device__ inline float dequant_q6k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned char* ql     = block_ptr;                    // 128 bytes
        const unsigned char* qh     = block_ptr + 128;              // 64 bytes
        const signed char*   scales = (const signed char*)(block_ptr + 192); // 16 bytes, signed
        const unsigned short* d_ptr = (const unsigned short*)(block_ptr + 208);
        float d = fp16_to_float_device(d_ptr[0]);

        int n       = in_sb / 128;
        int pos     = in_sb % 128;
        int quarter = pos / 32;        // 0..3 (q1..q4 in the CPU reference)
        int l       = pos % 32;
        int is      = l / 16;          // 0 or 1
        int sc_idx  = n * 8 + is + 2 * quarter;

        signed char sc = scales[sc_idx];

        // ql advances 64 bytes per outer stride; q2/q4 (odd quarter) read
        // the +32 byte partner of the same pair.
        int ql_offset = n * 64 + l + ((quarter & 1) ? 32 : 0);
        unsigned char ql_byte = ql[ql_offset];
        int nibble = (quarter & 2) ? (ql_byte >> 4) : (ql_byte & 0x0F);

        // qh packs 4 weights per byte (2 bits each); the quarter selects
        // which of the four 2-bit groups within qh[n*32 + l] is ours.
        unsigned char qh_byte = qh[n * 32 + l];
        int qh_bits = (qh_byte >> (2 * quarter)) & 0x03;

        int q_code = nibble | (qh_bits << 4);

        return d * (float)sc * ((float)q_code - 32.0f);
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
