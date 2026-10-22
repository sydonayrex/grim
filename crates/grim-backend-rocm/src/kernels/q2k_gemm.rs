//! Q2_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q2_K`]
//  Total: 2+2+8+8+64+4 = 88 (llama.cpp uses 84; 4 bytes are sub-block header overhead)

/// HIP source for `grim_fused_dequant_gemm_q2k` and `grim_fused_dequant_backward_gemm_q2k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize one Q2_K element from an 84-byte super-block.
    /// Layout: scales[16] @0..16, qs[64] @16..80, d (f16) @80, dmin (f16) @82.
    /// `in_sb` is the weight index within the 256-weight super-block (0..255).
        /// M7.2f: the GROUP HEADER of a Q2_K 16-weight group, decoded once.
    ///
    /// Returns the two 4-bit scales of the group's sub-block as VALUES, not
    /// as an array: a per-thread array indexed at runtime lands in scratch
    /// memory, and every weight would then pay a scratch load the wave
    /// cannot coalesce.
    __device__ __forceinline__ void dequant_q2k_header(
        const unsigned char* block_ptr,
        int in_sb,
        float* d_out,
        float* dmin_out,
        float* s0_out,
        float* s1_out)
    {
        *d_out    = fp16_to_float_device(((const unsigned short*)(block_ptr + 80))[0]);
        *dmin_out = fp16_to_float_device(((const unsigned short*)(block_ptr + 82))[0]);
        const int sub = in_sb / 16;
        const unsigned char s = block_ptr[sub];
        *s0_out = (float)(s & 0x0F);
        *s1_out = (float)(s >> 4);
    }

    /// M7.2f: FOUR Q2_K weights from one already-decoded group header.
    ///
    /// The four codes are four 2-bit fields of four CONSECUTIVE bytes (the
    /// interleaved llama.cpp layout), so one dword carries all four. Q2_K's
    /// 84-byte super-block stride is a multiple of 4, so the wide read is
    /// 4-byte aligned.
    __device__ __forceinline__ void dequant_q2k_quad(
        const unsigned char* block_ptr,
        int in_sb,
        float d,
        float dmin,
        float sc,
        float m,
        float* o0,
        float* o1,
        float* o2,
        float* o3)
    {
        const unsigned char* qs = block_ptr + 16;

        int sub = in_sb / 16;
        int w   = in_sb % 16;

        int q_byte  = (sub / 8) * 32 + (w % 16) + (sub % 2) * 16;
        int q_shift = 2 * ((sub % 8) / 2);
        unsigned qw;
        __builtin_memcpy(&qw, qs + q_byte, 4);
        {
            int code = (int)(((qw >> 0) & 0xFFu) >> q_shift) & 0x03;
            *o0 = d * sc * (float)code - dmin * m;
        }
        {
            int code = (int)(((qw >> 8) & 0xFFu) >> q_shift) & 0x03;
            *o1 = d * sc * (float)code - dmin * m;
        }
        {
            int code = (int)(((qw >> 16) & 0xFFu) >> q_shift) & 0x03;
            *o2 = d * sc * (float)code - dmin * m;
        }
        {
            int code = (int)(((qw >> 24) & 0xFFu) >> q_shift) & 0x03;
            *o3 = d * sc * (float)code - dmin * m;
        }
    }

__device__ inline float dequant_q2k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned char* scales = block_ptr;
        const unsigned char* qs = block_ptr + 16;
        float d    = fp16_to_float_device(((const unsigned short*)(block_ptr + 80))[0]);
        float dmin = fp16_to_float_device(((const unsigned short*)(block_ptr + 82))[0]);

        int sub = in_sb / 16;       // 16 sub-blocks of 16 weights
        int w   = in_sb % 16;

        float sc = (float)(scales[sub] & 0x0F);
        float m  = (float)(scales[sub] >> 4);

        // llama.cpp interleaved codes (ggml-quants.c:959): weight w of
        // sub-block `sub` lives at field 2*((sub%8)/2) of byte
        // qs[(sub/8)*32 + (w%16) + (sub%2)*16]. (Was a sequential
        // 4-bytes-per-sub-block layout that exists in no released GGUF;
        // fixed 2026-10-04 alongside the new quant_q2k encoder.)
        int q_byte = (sub / 8) * 32 + (w % 16) + (sub % 2) * 16;
        int q_shift = 2 * ((sub % 8) / 2);
        unsigned char q_code = (qs[q_byte] >> q_shift) & 0x03;

        return d * sc * (float)q_code - dmin * m;
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
