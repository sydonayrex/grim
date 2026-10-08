//! Q6_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q6_K`]

/// HIP source for `grim_fused_dequant_gemm_q6k` and `grim_fused_dequant_backward_gemm_q6k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize one Q6_K element from a 210-byte super-block.
    /// `in_sb` is the weight index within the 256-weight super-block (0..255).
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
        // 4 columns per thread: one activation load feeds 4 MACs. Same
        // traffic argument as the Q4_K kernel.
        const unsigned long long ncols4 = ((unsigned long long)N + 3) / 4;
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * ncols4;
        if (idx >= total) return;

        const int row = (int)(idx / ncols4);
        const int col0 = (int)(idx % ncols4) * 4;
        const int active = (col0 + 4 <= N) ? 4 : (N - col0);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 210;
        const unsigned char* bcol[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            bcol[j] = (j < active)
                ? B_q6k + (long long)(col0 + j) * row_bytes
                : (const unsigned char*)0;
        }
        const float* Arow = A + (long long)row * K;

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int sb = 0; sb < blocks_per_row; ++sb) {
            // Hoist d/scales decode out of the per-weight loop. Q6_K's
            // super-block is 2 outer blocks x 4 quarters x 2 sub-blocks of 16,
            // so each scale byte is loaded once per 16 weights instead of once
            // per weight.
            // Quarter pairs (0,2) and (1,3) share one ql byte: the low nibble
            // is quarter qa, the high nibble quarter qb=qa+2. One byte load
            // yields 2 weights; previously each quarter ran its own loop.
            for (int n = 0; n < 2; ++n) {
                for (int p = 0; p < 2; ++p) {
                    const int qa = p, qb = p + 2;
                    for (int is = 0; is < 2; ++is) {
                        float dsc4[4][2];
                        const unsigned char* ql4[4];
                        const unsigned char* qh4[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            if (j < active) {
                                const unsigned char* block_ptr = bcol[j] + sb * 210;
                                const unsigned char* ql = block_ptr;
                                const unsigned char* qh = block_ptr + 128;
                                const signed char* scales = (const signed char*)(block_ptr + 192);
                                const unsigned short* d_ptr = (const unsigned short*)(block_ptr + 208);
                                const float d = fp16_to_float_device(d_ptr[0]);
                                dsc4[j][0] = d * (float)scales[n * 8 + is + 2 * qa];
                                dsc4[j][1] = d * (float)scales[n * 8 + is + 2 * qb];
                                ql4[j] = ql;
                                qh4[j] = qh;
                            }
                        }
                        const int ql_base = n * 64;
                        const int ql_off  = ((qa & 1) ? 32 : 0);
                        const int qh_base = n * 32;
                        const int kbase_a = sb * 256 + n * 128 + qa * 32;
                        const int kbase_b = sb * 256 + n * 128 + qb * 32;
                        for (int l = is * 16; l < is * 16 + 16; l += 4) {
                            float4 a4a, a4b;
                            __builtin_memcpy(&a4a, Arow + kbase_a + l, 16);
                            __builtin_memcpy(&a4b, Arow + kbase_b + l, 16);
                            const float* afa = (const float*)&a4a;
                            const float* afb = (const float*)&a4b;
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const int ll = l + e;
                                #pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    if (j < active) {
                                        const unsigned char ql_byte = ql4[j][ql_base + ll + ql_off];
                                        const unsigned char qh_byte = qh4[j][qh_base + ll];
                                        const int nib_lo = ql_byte & 0x0F;
                                        const int nib_hi = ql_byte >> 4;
                                        const int bh_a = (qh_byte >> (2 * qa)) & 0x03;
                                        const int bh_b = (qh_byte >> (2 * qb)) & 0x03;
                                        acc[j] += afa[e] * (dsc4[j][0] * ((float)(nib_lo | (bh_a << 4)) - 32.0f));
                                        acc[j] += afb[e] * (dsc4[j][1] * ((float)(nib_hi | (bh_b << 4)) - 32.0f));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < active) C[(long long)row * N + col0 + j] = acc[j];
        }
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

        // Hoist the loop-invariant index decomposition out of the N loop.
        // `dequant_q6k_element` recomputed n/pos/quarter/l/is/sc_idx
        // (5 divs/mods) per N element; all depend only on in_sb.
        const int k_n       = in_sb / 128;
        const int k_pos     = in_sb % 128;
        const int k_quarter = k_pos / 32;
        const int k_l       = k_pos % 32;
        const int k_is      = k_l / 16;
        const int k_sc_idx  = k_n * 8 + k_is + 2 * k_quarter;
        const int k_ql_base = k_n * 64 + k_l;
        const int k_ql_off  = ((k_quarter & 1) ? 32 : 0);
        const int k_qh_idx  = k_n * 32 + k_l;
        const int k_qh_shift = 2 * k_quarter;
        const int k_hi_nib  = (k_quarter & 2) ? 1 : 0;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q6k + n * row_bytes + sb_idx * 210;
            const unsigned char* ql = block_ptr;
            const unsigned char* qh = block_ptr + 128;
            const signed char* scales = (const signed char*)(block_ptr + 192);
            const unsigned short* d_ptr = (const unsigned short*)(block_ptr + 208);
            float d = fp16_to_float_device(d_ptr[0]);
            signed char sc = scales[k_sc_idx];
            unsigned char ql_byte = ql[k_ql_base + k_ql_off];
            int nibble = k_hi_nib ? (ql_byte >> 4) : (ql_byte & 0x0F);
            unsigned char qh_byte = qh[k_qh_idx];
            int qh_bits = (qh_byte >> k_qh_shift) & 0x03;
            int q_code = nibble | (qh_bits << 4);
            acc += dy_val * (d * (float)sc * ((float)q_code - 32.0f));
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
