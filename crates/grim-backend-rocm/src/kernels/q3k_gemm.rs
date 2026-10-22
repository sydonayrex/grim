//! Q3_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q3_K`]

/// HIP source for `grim_fused_dequant_gemm_q3k` and `grim_fused_dequant_backward_gemm_q3k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

/// Dequantize one Q3K element from a super-block.
/// `in_sb` is the weight index within the 256-weight super-block (0..255).
    __device__ inline float dequant_q3k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned char* hmask = block_ptr + 0;
        const unsigned char* qs    = block_ptr + 32;
        const unsigned char* scales_ptr = block_ptr + 96;
        float d = fp16_to_float_device(((const unsigned short*)(block_ptr + 108))[0]);

        // ── Decode the 12-byte scales field into 16 signed bytes ────────── Mirrors the ggml aux‑shuffle (dequantize_row_q3_K): memcpy(aux, x->scales, 12); tmp = aux[2]; aux[2] = ((aux[0]>>4) & 0x0F0F0F0F) | (((tmp>>4) & 0x03030303) << 4); aux[3]
        // = ((aux[1]>>4) & 0x0F0F0F0F) | (((tmp>>6) & 0x03030303) << 4); aux[0] = ( aux[0] & 0x0F0F0F0F) | (((tmp>>0) & 0x03030303) << 4); aux[1] = ( aux[1] & 0x0F0F0F0F) | (((tmp>>2) & 0x03030303) << 4);
        const unsigned int kmask1 = 0x03030303u;
        const unsigned int kmask2 = 0x0F0F0F0Fu;
        unsigned int aux0 = ((unsigned int)scales_ptr[0]) | (((unsigned int)scales_ptr[1]) << 8)
                          | (((unsigned int)scales_ptr[2]) << 16) | (((unsigned int)scales_ptr[3]) << 24);
        unsigned int aux1 = ((unsigned int)scales_ptr[4]) | (((unsigned int)scales_ptr[5]) << 8)
                          | (((unsigned int)scales_ptr[6]) << 16) | (((unsigned int)scales_ptr[7]) << 24);
        unsigned int tmp  = ((unsigned int)scales_ptr[8]) | (((unsigned int)scales_ptr[9]) << 8)
                          | (((unsigned int)scales_ptr[10]) << 16) | (((unsigned int)scales_ptr[11]) << 24);
        // In ggml the final two bytes of the 16-byte aux[3] are set to zero because memcpy only copied 12 bytes and aux is a 4×uint32 array.
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

    /// M7.2f: the GROUP HEADER of a Q3_K 16-weight group, decoded once.
    ///
    /// The per-element decoder re-does the 12-byte ggml aux shuffle for EVERY
    /// weight, which is ~40 ops and 12 byte loads each. The IQ recipe
    /// (the IQ1_S D2-2 change) is to hoist the part of a group that is
    /// CONSTANT.
    ///
    /// It returns the SELECTED scale, not the 16-byte array, and that is not
    /// a detail: a per-thread array indexed by a runtime index lands in
    /// SCRATCH memory, and every weight then pays a scratch load that the 32
    /// lanes of a wave cannot coalesce (each lane has its own scratch
    /// address). Measured: the array form ran the tile at 15.3 ms against
    /// the WMMA path's 7.8 ms; the value form is one register.
    ///
    /// `in_sb` is the index of the group's FIRST weight inside the
    /// super-block. The scale index (n, _j, lo_hi) is constant across the
    /// group's 16 weights, so one shuffle serves the whole group.
    __device__ __forceinline__ void dequant_q3k_header(
        const unsigned char* block_ptr,
        int in_sb,
        float* d_out,
        float* dmin_out,
        float* s0_out,
        float* s1_out)
    {
        const unsigned char* scales_ptr = block_ptr + 96;
        *d_out    = fp16_to_float_device(((const unsigned short*)(block_ptr + 108))[0]);
        *dmin_out = 0.0f;   // Q6_K-style: Q3_K has no dmin term
        *s1_out   = 0.0f;

        const unsigned int kmask1 = 0x03030303u;
        const unsigned int kmask2 = 0x0F0F0F0Fu;
        unsigned int aux0 = ((unsigned int)scales_ptr[0]) | (((unsigned int)scales_ptr[1]) << 8)
                          | (((unsigned int)scales_ptr[2]) << 16) | (((unsigned int)scales_ptr[3]) << 24);
        unsigned int aux1 = ((unsigned int)scales_ptr[4]) | (((unsigned int)scales_ptr[5]) << 8)
                          | (((unsigned int)scales_ptr[6]) << 16) | (((unsigned int)scales_ptr[7]) << 24);
        unsigned int tmp  = ((unsigned int)scales_ptr[8]) | (((unsigned int)scales_ptr[9]) << 8)
                          | (((unsigned int)scales_ptr[10]) << 16) | (((unsigned int)scales_ptr[11]) << 24);
        unsigned int qw0 = (aux0 & kmask2) | (((tmp >>  0) & kmask1) << 4);
        unsigned int qw1 = (aux1 & kmask2) | (((tmp >>  2) & kmask1) << 4);
        unsigned int qw2 = ((aux0 >> 4) & kmask2) | (((tmp >>  4) & kmask1) << 4);
        unsigned int qw3 = ((aux1 >> 4) & kmask2) | (((tmp >>  6) & kmask1) << 4);
        unsigned int auxs[4] = { qw0, qw1, qw2, qw3 };

        int n     = in_sb / 128;
        int _j    = (in_sb % 128) / 32;
        int lo_hi = (in_sb % 32) / 16;
        int sc_idx = n * 8 + _j * 2 + lo_hi;
        int t = sc_idx / 4;
        int b2 = sc_idx % 4;
        unsigned int v = (auxs[t] >> (8 * b2)) & 0xFF;
        *s0_out = (float)((int)(v >= 128u ? (int)v - 256 : (int)v) - 32);
    }

    /// M7.2f: FOUR Q3_K weights from one already-decoded group header.
    ///
    /// The four codes are four nibbles of four CONSECUTIVE bytes and the four
    /// sign bits are four bits of four CONSECUTIVE hmask bytes, so each field
    /// is one wide read instead of four 1-byte reads. Q3_K's 110-byte
    /// super-block stride only guarantees 2-byte alignment, so the wide read
    /// is a pair of ushorts rather than a dword.
    __device__ __forceinline__ void dequant_q3k_quad(
        const unsigned char* block_ptr,
        int in_sb,
        float d,
        float dmin,
        float scale_val,
        float unused_s1,
        float* o0,
        float* o1,
        float* o2,
        float* o3)
    {
        const unsigned char* hmask = block_ptr + 0;
        const unsigned char* qs    = block_ptr + 32;

        int n     = in_sb / 128;
        int _j    = (in_sb % 128) / 32;
        int lo_hi = (in_sb % 32) / 16;
        int l     = in_sb % 16;
        float dl = d * scale_val;

        int shift  = _j * 2;
        int q_off  = n * 32 + l + lo_hi * 16;
        int hm_off = l + lo_hi * 16;

        unsigned short q01, q23, h01, h23;
        __builtin_memcpy(&q01, qs + q_off, 2);
        __builtin_memcpy(&q23, qs + q_off + 2, 2);
        __builtin_memcpy(&h01, hmask + hm_off, 2);
        __builtin_memcpy(&h23, hmask + hm_off + 2, 2);
        unsigned qw = (unsigned)q01 | ((unsigned)q23 << 16);
        unsigned hw = (unsigned)h01 | ((unsigned)h23 << 16);

        {
            int q_val  = (int)(((qw >> 0) & 0xFFu) >> shift) & 3;
            int hm_bit = (int)(((hw >> 0) & 0xFFu) >> (_j + n * 4)) & 1;
            *o0 = dl * (float)(q_val - (hm_bit ? 0 : 4));
        }
        {
            int q_val  = (int)(((qw >> 8) & 0xFFu) >> shift) & 3;
            int hm_bit = (int)(((hw >> 8) & 0xFFu) >> (_j + n * 4)) & 1;
            *o1 = dl * (float)(q_val - (hm_bit ? 0 : 4));
        }
        {
            int q_val  = (int)(((qw >> 16) & 0xFFu) >> shift) & 3;
            int hm_bit = (int)(((hw >> 16) & 0xFFu) >> (_j + n * 4)) & 1;
            *o2 = dl * (float)(q_val - (hm_bit ? 0 : 4));
        }
        {
            int q_val  = (int)(((qw >> 24) & 0xFFu) >> shift) & 3;
            int hm_bit = (int)(((hw >> 24) & 0xFFu) >> (_j + n * 4)) & 1;
            *o3 = dl * (float)(q_val - (hm_bit ? 0 : 4));
        }
    }

    // Latent-cache embedding gather for Q3_K tables (xing4.0 token_embd):
    // mirrors grim_embedding_q4k_gather with the 110-byte Q3_K super-block.
    extern "C" __global__ void grim_embedding_q3k_gather(
        const unsigned char* packed, float* out, int* indices, int dim, int total, int rows) {
        const int QK_BLOCK = 256;
        const int QK_BLOCK_BYTES = 110;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) return;
        int i = idx / dim;
        int j = idx % dim;
        int row = indices[i];
        if (row < 0 || row >= rows) {
            out[idx] = 0.0f;
            return;
        }
        long long e = (long long)row * (long long)dim + (long long)j;
        const unsigned char* blk = packed + (e / QK_BLOCK) * QK_BLOCK_BYTES;
        out[idx] = dequant_q3k_element(blk, (int)(e % QK_BLOCK));
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
