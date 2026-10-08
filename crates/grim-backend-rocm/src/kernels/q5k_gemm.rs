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

        // ggml layout: four 64-weight groups. Within group n, the first 32 weights take the low nibble of qs[n*32 + l] with high bit qh[l] &
        // (1 << 2n) and scale sub-block 2n; the next 32 take the high nibble with bit qh[l] & (1 << (2n+1)) and scale sub-block 2n+1.
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
        // 4 columns per thread: one activation load feeds 4 MACs. See the Q4_K
        // kernel for the traffic argument; same structure here.
        const unsigned long long ncols4 = ((unsigned long long)N + 3) / 4;
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * ncols4;
        if (idx >= total) return;

        const int row = (int)(idx / ncols4);
        const int col0 = (int)(idx % ncols4) * 4;
        const int active = (col0 + 4 <= N) ? 4 : (N - col0);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 176;
        const unsigned char* bcol[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            bcol[j] = (j < active)
                ? B_q5k + (long long)(col0 + j) * row_bytes
                : (const unsigned char*)0;
        }
        const float* Arow = A + (long long)row * K;

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int sb = 0; sb < blocks_per_row; ++sb) {
            // Hoist d/dmin/sc/m access to one read per sub-block.
            // Same scalar MAC structure, but no per-weight metadata decode.
            for (int is = 0; is < 8; ++is) {
                float dsc4[4], dm4[4];
                const unsigned char* qs4[4];
                const unsigned char* qh4[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j < active) {
                        const unsigned char* block_ptr = bcol[j] + sb * 176;
                        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
                        const float d    = fp16_to_float_device(h_ptr[0]);
                        const float dmin = fp16_to_float_device(h_ptr[1]);
                        const unsigned char* scales = block_ptr + 4;
                        unsigned char sc, m;
                        if (is < 4) {
                            sc = scales[is] & 63;
                            m  = scales[is + 4] & 63;
                        } else {
                            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
                        }
                        dsc4[j] = d * (float)sc;
                        dm4[j]  = dmin * (float)m;
                        // is = 2*n + hi, so n = is/2 group, hi = is%2
                        const int n = is >> 1;
                        qs4[j] = block_ptr + 48 + n * 32;
                        qh4[j] = block_ptr + 16;
                    }
                }
                // is = 2*n + hi, so n = is/2 group, hi = is%2 sub-block of 32
                const int n  = is >> 1;
                const int hi = is & 1;
                const int msb_shift = 2 * n + hi;
                for (int l = 0; l < 32; l += 4) {
                    float4 a4;
                    __builtin_memcpy(&a4, Arow + sb * 256 + is * 32 + l, 16);
                    const float* af = (const float*)&a4;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float a_val = af[e];
                        const int ll = l + e;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            if (j < active) {
                                const unsigned char packed = qs4[j][ll];
                                const int q_low = hi ? (packed >> 4) : (packed & 0x0F);
                                const int msb = (qh4[j][ll] >> msb_shift) & 1;
                                acc[j] += a_val * (dsc4[j] * (float)(q_low | (msb << 4)) - dm4[j]);
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

        // Hoist the loop-invariant index decomposition out of the N loop.
        // `dequant_q5k_element` recomputed n/j/l/hi/is (4 divs/mods + branch)
        // per N element; all depend only on in_sb.
        const int k_n  = in_sb / 64;
        const int k_j  = in_sb % 64;
        const int k_l  = k_j & 31;
        const int k_hi = k_j >> 5;
        const int k_is = 2 * k_n + k_hi;
        const int k_qs_off = k_n * 32 + k_l;
        const int k_msb_shift = 2 * k_n + k_hi;
        const int k_is_low = (k_is < 4) ? 1 : 0;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q5k + n * row_bytes + sb_idx * 176;
            const unsigned short* h_ptr = (const unsigned short*)block_ptr;
            float d = fp16_to_float_device(h_ptr[0]);
            float dmin = fp16_to_float_device(h_ptr[1]);
            const unsigned char* scales = block_ptr + 4;
            unsigned char sc, m;
            if (k_is_low) {
                sc = scales[k_is] & 63;
                m  = scales[k_is + 4] & 63;
            } else {
                sc = (scales[k_is + 4] & 0xF) | ((scales[k_is - 4] >> 6) << 4);
                m  = (scales[k_is + 4] >> 4)  | ((scales[k_is] >> 6) << 4);
            }
            const unsigned char* qs = block_ptr + 48;
            const unsigned char* qh = block_ptr + 16;
            unsigned char packed = qs[k_qs_off];
            unsigned char q_low = k_hi ? (packed >> 4) : (packed & 0x0F);
            unsigned char msb = (qh[k_l] >> k_msb_shift) & 1;
            int q_code = (int)q_low | ((int)msb << 4);
            acc += dy_val * (d * (float)sc * (float)q_code - dmin * (float)m);
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
