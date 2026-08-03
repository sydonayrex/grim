//! Q3_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q3_K`]

/// HIP source for `grim_fused_dequant_gemm_q3k` and `grim_fused_dequant_backward_gemm_q3k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

/// Dequantize one Q3K element from a super-block.
    /// `in_sb` is the weight index within the 256-weight super-block (0..255).
    ///
    /// Faithful HIP translation of the authoritative CPU reference
    /// `grim_quant::dequant_q3k` (crates/grim-quant/src/lib.rs), which itself
    /// matches llama.cpp `dequantize_row_q3_K` bit-for-bit.
    ///
    /// The true ggml `block_q3_K` layout is 110 bytes / 256 weights:
    ///   - 32 bytes hmask (one high-bit per weight)
    ///   - 64 bytes qs (2-bit quant codes, packed 4 per byte)
    ///   - 12 bytes scales (16 6-bit sub-block scales, aux-shuffled)
    ///   - 2 bytes d (f16 super-block scale)
    /// There is no `dmin` field and no `m` array.
    __device__ inline float dequant_q3k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned char* hmask = block_ptr + 0;
        const unsigned char* qs    = block_ptr + 32;
        const unsigned char* scales_ptr = block_ptr + 96;
        float d = fp16_to_float_device(((const unsigned short*)(block_ptr + 108))[0]);

        // ── Decode the 12-byte scales field into 16 signed bytes ──────────
        // Mirrors the ggml aux‑shuffle (dequantize_row_q3_K):
        //   memcpy(aux, x->scales, 12);
        //   tmp = aux[2];
        //   aux[2] = ((aux[0]>>4) & 0x0F0F0F0F) | (((tmp>>4) & 0x03030303) << 4);
        //   aux[3] = ((aux[1]>>4) & 0x0F0F0F0F) | (((tmp>>6) & 0x03030303) << 4);
        //   aux[0] = ( aux[0]     & 0x0F0F0F0F) | (((tmp>>0) & 0x03030303) << 4);
        //   aux[1] = ( aux[1]     & 0x0F0F0F0F) | (((tmp>>2) & 0x03030303) << 4);
        const unsigned int kmask1 = 0x03030303u;
        const unsigned int kmask2 = 0x0F0F0F0Fu;
        unsigned int aux0 = ((unsigned int)scales_ptr[0]) | (((unsigned int)scales_ptr[1]) << 8)
                          | (((unsigned int)scales_ptr[2]) << 16) | (((unsigned int)scales_ptr[3]) << 24);
        unsigned int aux1 = ((unsigned int)scales_ptr[4]) | (((unsigned int)scales_ptr[5]) << 8)
                          | (((unsigned int)scales_ptr[6]) << 16) | (((unsigned int)scales_ptr[7]) << 24);
        unsigned int tmp  = ((unsigned int)scales_ptr[8]) | (((unsigned int)scales_ptr[9]) << 8)
                          | (((unsigned int)scales_ptr[10]) << 16) | (((unsigned int)scales_ptr[11]) << 24);
        // In ggml the final two bytes of the 16-byte aux[3] are set to zero
        // because memcpy only copied 12 bytes and aux is a 4×uint32 array.
        // Therefore aux[3]~(0) is harmless.
        unsigned int qw0 = (aux0 & kmask2) | (((tmp >>  0) & kmask1) << 4);
        unsigned int qw1 = (aux1 & kmask2) | (((tmp >>  2) & kmask1) << 4);
        unsigned int qw2 = ((aux0 >> 4) & kmask2) | (((tmp >>  4) & kmask1) << 4);
        unsigned int qw3 = ((aux1 >> 4) & kmask2) | (((tmp >>  6) & kmask1) << 4);
        // Load all back into signed byte scale array; idx = in_sb / 32 selects one of 8
        // slots used by the reference.
        signed char sc[16];
        sc[ 0] = (signed char)(qw0 & 0xFF);
        sc[ 1] = (signed char)((qw0 >> 8)  & 0xFF);
        sc[ 2] = (signed char)((qw0 >> 16) & 0xFF);
        sc[ 3] = (signed char)((qw0 >> 24) & 0xFF);
        sc[ 4] = (signed char)(qw1 & 0xFF);
        sc[ 5] = (signed char)((qw1 >> 8)  & 0xFF);
        sc[ 6] = (signed char)((qw1 >> 16) & 0xFF);
        sc[ 7] = (signed char)((qw1 >> 24) & 0xFF);
        sc[ 8] = (signed char)(qw2 & 0xFF);
        sc[ 9] = (signed char)((qw2 >> 8)  & 0xFF);
        sc[10] = (signed char)((qw2 >> 16) & 0xFF);
        sc[11] = (signed char)((qw2 >> 24) & 0xFF);
        sc[12] = (signed char)(qw3 & 0xFF);
        sc[13] = (signed char)((qw3 >> 8)  & 0xFF);
        sc[14] = (signed char)((qw3 >> 16) & 0xFF);
        sc[15] = (signed char)((qw3 >> 24) & 0xFF);

        int n     = in_sb / 128;  // 0 or 1 → outer-loop stride selector
        int _j    = (in_sb % 128) / 32;
        int lo_hi = (in_sb % 32) / 16;  // 0 = first 16, 1 = second 16
        int l     = in_sb % 16;
        int sc_idx      = n * 8 + _j * 2 + lo_hi;  // maps to ref _is++ order
        float scale_val = (float)((int)sc[sc_idx] - 32);
        float dl = d * scale_val;

        int shift    = _j * 2;
        int q_off    = n * 32 + l + lo_hi * 16;   // q_off advances by 32 per n-pass (qs is 64 bytes)
        int q_val    = (qs[q_off] >> shift) & 3;
        int hm_bit   = (hmask[l + lo_hi * 16] >> (_j + n * 4)) & 1;
        // ggml: if hm_bit == 0 → -4, else → 0
        int q_biased = q_val - (hm_bit ? 0 : 4);

        return dl * (float)q_biased;
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
        const int row_bytes = blocks_per_row * 110;   // block_q3_K = 110 bytes
        const unsigned char* row_b_ptr = B_q3k + col * row_bytes;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            const unsigned char* block_ptr = row_b_ptr + sb_idx * 110;
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
        const int row_bytes = blocks_per_row * 110;   // block_q3_K = 110 bytes

        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q3k + n * row_bytes + sb_idx * 110;
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
