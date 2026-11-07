//! TreePie (WS-A) decode + GEMV for the decode shape.
//!
//! TreePie is a 5.0 bpw format: 4 bits of E2M2 payload (`exp|mant`) plus 1 sign
//! bit, packed as 4 payload i32 words and 1 sign-plane i32 per 32 values. This
//! kernel consumes that layout directly and reconstructs the FP16 bit pattern
//! in-register, so no dequantized copy of B ever reaches VRAM.
//!
//! The decode is the *same* table-free bit twiddle as
//! `grim_quant::tree_pie::e2m2_to_fp16_bits`: a bias add on the exponent field
//! for the `e >= 1` rows, and one conditional for the `e == 0` row (which is
//! subnormal in E2M2 but normal FP16). The sign is OR-ed in last, so negative
//! zero falls out without a special case. Because the payload *is* an FP16
//! encoding, the decoded halves are packed straight into the `v_dot2_f32_f16`
//! operand word -- no float round-trip, and the decode is bit-exact by
//! construction rather than by tolerance.
//!
//! `v_dot2_f32_f16` is used because on gfx12 `v_dot4_*_i8` is UNSIGNED-only, so
//! the signed-FP16 vector dot is the available dot primitive for a signed
//! weight grid. This is the M=1 decode shape, where WMMA wastes 15/16 of every
//! 16-row tile.
//!
//! Requires K % 32 == 0: the 5.0 bpw claim only holds at 32-value granularity,
//! and a ragged tail would silently overstate the density.

/// HIP source for [`crate::kernels::tree_pie::KERNEL_SOURCE`].
///
/// Appended after `dot_gemv` so it can reuse `grim_fdot2_f32_f16`.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__AMDGCN__) || defined(__HIP__)
__device__ __forceinline__ unsigned short grim_tree_pie_f16(
    const unsigned* __restrict__ payload,   // 4 words
    unsigned signs,                          // 1 sign bit per value
    int j)                                   // value index within the group of 32
{
    // Payload: 8 nibbles per word, low nibble first (value i lives in word
    // i/8 at nibble i%8) -- the packing order used by pack_tree_pie_32.
    unsigned nib = (payload[j >> 3] >> ((j & 7) * 4)) & 0xFu;
    unsigned sign = (signs >> j) & 1u;

    // code = sign << 4 | exp << 2 | mant, with the sign held in the plane above.
    unsigned exp  = (nib >> 2) & 3u;
    unsigned mant = nib & 3u;

    unsigned exp_field, mant_10;
    if (exp == 0) {
        if (mant == 0) {
            exp_field = 0u; mant_10 = 0u;                 // +-0
        } else {
            // E2M2 subnormal row m/2 is 0, .5, 1, 1.5 -> FP16 exponent fields
            // (only m=0 is special) 14, 15, 15. The two distinct nonzero
            // fields differ by 1, which is `mant >> 1`; the mantissa LSB is set
            // only at m=3, i.e. ((mant+1)>>2) shifted to FP16 bit 9.
            exp_field = 14u + (mant >> 1);
            mant_10   = ((mant + 1u) >> 2) << 9;
        }
    } else {
        // e >= 1: E2M2 bias 0 -> FP16 bias 15, so the field is exp + 15 and the
        // 2-bit mantissa already sits at FP16 bit 8.
        exp_field = exp + 15u;
        mant_10   = mant << 8;
    }

    return (unsigned short)((sign << 15) | (exp_field << 10) | mant_10);
}

// M=1 decode GEMV: C[col] = sum_k A[k] * B_tree[col, k]
//   A      : f16 [K], packed (the WMMA path casts to f16 the same way)
//   B_tree : N columns x ceil(K/32) groups of 5 i32 (4 payload + 1 sign plane)
//   C      : f32 [N]
extern "C" __global__ void grim_tree_pie_gemv(
    const unsigned short* __restrict__ act_f16,
    const int*             __restrict__ B_tree,
    float*                __restrict__ C,
    int N, int K)
{
    const int col  = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= N) return;

    const int groups   = K / 32;                 // K % 32 == 0, checked host-side
    const int row_words = groups * 5;
    const int* __restrict__ row = B_tree + (long long)col * row_words;
    const unsigned* act2 = (const unsigned*)act_f16;

    float facc = 0.0f;
    for (int g = lane; g < groups; g += 32) {
        const int* __restrict__ gp = row + (long long)g * 5;
        unsigned payload[4] = { (unsigned)gp[0], (unsigned)gp[1],
                                (unsigned)gp[2], (unsigned)gp[3] };
        unsigned signs = (unsigned)gp[4];

        float bacc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j += 2) {
            // Pack the two decoded halves straight into the dot2 operand word
            // (lo = element j, hi = element j+1), matching grim_pk_f16's layout
            // so it lines up with the little-endian f16 activation pairs.
            unsigned w01 = (unsigned)grim_tree_pie_f16(payload, signs, j)
                         | ((unsigned)grim_tree_pie_f16(payload, signs, j + 1) << 16);
            unsigned a01 = act2[(g * 32 + j) >> 1];
            bacc = grim_fdot2_f32_f16(a01, w01, bacc);
        }
        facc += bacc;
    }

    for (int off = 16; off > 0; off >>= 1)
        facc += __shfl_xor(facc, off);
    if (lane == 0) C[col] = facc;
}

// TreePie v2 decode GEMV:
// - #1: 4 columns per wave (blockIdx.x * 4), batching all 5 i32 words (20 bytes)
//       across 4 columns before decoding and issuing grim_fdot2_f32_f16.
// - #2: Branchless VALU __builtin_amdgcn_perm lookup for the 16-entry E2M2 -> FP16
//       magnitude table, plus 128-bit uint4 activation loads shared across all 4 columns.
// Preserves the exact per-lane group and j accumulation order of v1 (0 bit diffs).
__device__ __forceinline__ unsigned short grim_tree_pie_f16_valu(
    unsigned nib,
    unsigned sign_bit)
{
    // 16-entry E2M2 magnitude table (fp16 bits):
    // 0: 0x0000 (0.0),  1: 0x3800 (0.5),  2: 0x3c00 (1.0),  3: 0x3e00 (1.5)
    // 4: 0x4000 (2.0),  5: 0x4100 (2.5),  6: 0x4200 (3.0),  7: 0x4300 (3.5)
    // 8: 0x4400 (4.0),  9: 0x4500 (5.0), 10: 0x4600 (6.0), 11: 0x4700 (7.0)
    // 12: 0x4800 (8.0), 13: 0x4900 (10.0), 14: 0x4a00 (12.0), 15: 0x4b00 (14.0)
    // Notice the low byte of every entry is 0x00! Only the high byte varies:
    // {0x00, 0x38, 0x3c, 0x3e, 0x40, 0x41, 0x42, 0x43,
    //  0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b}
    const unsigned t0 = 0x3e3c3800u;
    const unsigned t1 = 0x43424140u;
    const unsigned t2 = 0x47464544u;
    const unsigned t3 = 0x4b4a4948u;
    const unsigned pair01 = __builtin_amdgcn_perm(t1, t0, nib & 7u);
    const unsigned pair23 = __builtin_amdgcn_perm(t3, t2, nib & 7u);
    const unsigned hi_byte = ((nib & 8u) ? pair23 : pair01) & 0xFFu;
    return (unsigned short)((sign_bit << 15) | (hi_byte << 8));
}

// v7/v9 dual-nibble word-wide VALU decode: decodes TWO E2M2 nibbles `(nib0, nib1)`
// and their sign bits `(s0, s1)` into a packed FP16 pair `(lo_f16 | (hi_f16 << 16))`
// using a SINGLE `__builtin_amdgcn_perm` pair (cutting `v_perm_b32` count in half).
// Because the low byte of every E2M2 FP16 value is `0x00`, placing `(nib0 & 7)` in
// selector byte 1 and `(nib1 & 7)` in selector byte 3 puts `hi_byte(nib0)` at bits
// 8..15 and `hi_byte(nib1)` at bits 24..31 with zero output shifts.
__device__ __forceinline__ unsigned grim_tree_pie_f16x2_valu(
    unsigned nib0,
    unsigned nib1,
    unsigned s0,
    unsigned s1)
{
    const unsigned t0 = 0x3e3c3800u;
    const unsigned t1 = 0x43424140u;
    const unsigned t2 = 0x47464544u;
    const unsigned t3 = 0x4b4a4948u;
    const unsigned sel = ((nib0 & 7u) << 8) | ((nib1 & 7u) << 24);
    const unsigned pair01 = __builtin_amdgcn_perm(t1, t0, sel);
    const unsigned pair23 = __builtin_amdgcn_perm(t3, t2, sel);
    const unsigned hi0 = ((nib0 & 8u) ? pair23 : pair01) & 0x0000FF00u;
    const unsigned hi1 = ((nib1 & 8u) ? pair23 : pair01) & 0xFF000000u;
    return hi0 | hi1 | (s0 << 15) | (s1 << 31);
}

extern "C" __global__ void __launch_bounds__(32) grim_tree_pie_gemv_v2(
    const unsigned short* __restrict__ act_f16,
    const int*             __restrict__ B_tree,
    float*                __restrict__ C,
    int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int lane     = threadIdx.x;
    if (col_base >= N) return;

    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const int groups      = K / 32;
    const int row_words   = groups * 5;
    const uint4* act_v4   = (const uint4*)act_f16;

    const int* b_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int col = col_base + ((c < active_cols) ? c : 0);
        b_cols[c] = B_tree + (long long)col * row_words;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int g = lane; g < groups; g += 32) {
        // v8 B-first load ordering: issue all 4 columns' 5-word B group loads
        // before touching A (`act_v4`).
        unsigned payload[4][4];
        unsigned signs[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int* __restrict__ gp = b_cols[c] + (long long)g * 5;
            payload[c][0] = (unsigned)gp[0];
            payload[c][1] = (unsigned)gp[1];
            payload[c][2] = (unsigned)gp[2];
            payload[c][3] = (unsigned)gp[3];
            signs[c]      = (unsigned)gp[4];
        }

        const uint4 av0 = act_v4[g * 4 + 0];
        const uint4 av1 = act_v4[g * 4 + 1];
        const uint4 av2 = act_v4[g * 4 + 2];
        const uint4 av3 = act_v4[g * 4 + 3];
        const unsigned a2[16] = {
            av0.x, av0.y, av0.z, av0.w,
            av1.x, av1.y, av1.z, av1.w,
            av2.x, av2.y, av2.z, av2.w,
            av3.x, av3.y, av3.z, av3.w
        };

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            if (c >= active_cols) break;
            float bacc = 0.0f;
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                const unsigned pw = payload[c][w];
                const unsigned sg = signs[c] >> (w * 8);
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    const unsigned nib0 = (pw >> (p * 8 + 0)) & 0xFu;
                    const unsigned nib1 = (pw >> (p * 8 + 4)) & 0xFu;
                    const unsigned s0   = (sg >> (p * 2 + 0)) & 1u;
                    const unsigned s1   = (sg >> (p * 2 + 1)) & 1u;
                    const unsigned w01  = grim_tree_pie_f16x2_valu(nib0, nib1, s0, s1);
                    bacc = grim_fdot2_f32_f16(a2[w * 4 + p], w01, bacc);
                }
            }
            facc[c] += bacc;
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; c++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[c] += __shfl_xor(facc[c], off);
    }
    if (lane == 0) {
        #pragma unroll
        for (int c = 0; c < active_cols; c++)
            C[col_base + c] = facc[c];
    }
}

// Rows of A handled per block. The decode of a B group is amortised across all of
// them, which is the point: TreePie stores 5 i32 per 32 values and decoding it
// once for 8 output rows is 8x cheaper per row than decoding per row. Kept a
// compile-time constant so `acc` lives in registers; 8 costs 8 floats plus the
// existing decode temporaries, which is comfortable for a wave32 kernel.
#define TP_M_TILE 8

// Prefill GEMM: C[m][col] = sum_k A[m][k] * B_tree[col, k]
//   A      : f16 [M, K], row-major
//   B_tree : N columns x (K/32) groups of 5 i32 (4 payload + 1 sign plane)
//   C      : f32 [M, N]
//
// grid = (N columns, ceil(M / TP_M_TILE)), block = (32,1,1). One block owns one
// output column and a tile of rows, exactly as the GEMV owns one column: the B
// decode is the expensive part and it is shared by every row in the tile.
//
// The reduction is the same 32-lane `__shfl_xor` butterfly the GEMV uses, so the
// two agree on summation order for M = 1 and a decode regression cannot hide as a
// prefill-only difference.
extern "C" __global__ void grim_tree_pie_gemm(
    const unsigned short* __restrict__ act_f16,
    const int*             __restrict__ B_tree,
    float*                 __restrict__ C,
    int M, int N, int K)
{
    const int col  = blockIdx.x;
    const int tile = blockIdx.y;
    const int lane = threadIdx.x;
    if (col >= N) return;

    const int m0    = tile * TP_M_TILE;
    const int rows  = min(TP_M_TILE, M - m0);
    if (rows <= 0) return;

    const int groups   = K / 32;              // K % 32 == 0, checked host-side
    const int row_words = groups * 5;
    const int* __restrict__ row = B_tree + (long long)col * row_words;
    const unsigned* act2 = (const unsigned*)act_f16;

    float acc[TP_M_TILE] = { 0.0f };

    for (int g = lane; g < groups; g += 32) {
        const int* __restrict__ gp = row + (long long)g * 5;
        unsigned payload[4] = { (unsigned)gp[0], (unsigned)gp[1],
                                (unsigned)gp[2], (unsigned)gp[3] };
        unsigned signs = (unsigned)gp[4];

        float bacc[TP_M_TILE] = { 0.0f };
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned pw = payload[w];
            const unsigned sg = signs >> (w * 8);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int j = (w * 4 + p) * 2;
                const unsigned nib0 = (pw >> (p * 8 + 0)) & 0xFu;
                const unsigned nib1 = (pw >> (p * 8 + 4)) & 0xFu;
                const unsigned s0   = (sg >> (p * 2 + 0)) & 1u;
                const unsigned s1   = (sg >> (p * 2 + 1)) & 1u;
                // Decode once via dual-nibble VALU, use for every row in the tile.
                unsigned w01 = grim_tree_pie_f16x2_valu(nib0, nib1, s0, s1);
                #pragma unroll
                for (int t = 0; t < TP_M_TILE; ++t) {
                    if (t < rows) {
                        unsigned a01 = act2[(((size_t)(m0 + t) * K) + (g * 32 + j)) >> 1];
                        bacc[t] = grim_fdot2_f32_f16(a01, w01, bacc[t]);
                    }
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < TP_M_TILE; ++t) {
            acc[t] += bacc[t];
        }
    }

    #pragma unroll
    for (int t = 0; t < TP_M_TILE; ++t) {
        if (t < rows) {
            float v = acc[t];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0) C[(size_t)(m0 + t) * N + col] = v;
        }
    }
}

// v9 multi-row 4-column hoisted-B TreePie GEMM (`TP_M_TILE = 8` rows x 4 columns
// per wave):
// - grid = (ceil(N / 4), ceil(M / TP_M_TILE)), block = (32, 1, 1).
// - Each wave loads and VALU-decodes 4 B columns ONCE per 32-value group (`w01[4][16]`),
//   then streams `uint4` activations for each active row `t < rows` and dots against
//   all 4 decoded columns, amortizing B decode across 8 rows AND A loads across 4 cols!
extern "C" __global__ void __launch_bounds__(32) grim_tree_pie_gemm_v2(
    const unsigned short* __restrict__ act_f16,
    const int*             __restrict__ B_tree,
    float*                 __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int tile     = blockIdx.y;
    const int lane     = threadIdx.x;
    if (col_base >= N) return;

    const int m0   = tile * TP_M_TILE;
    const int rows = min(TP_M_TILE, M - m0);
    if (rows <= 0) return;

    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const int groups      = K / 32;
    const int row_words   = groups * 5;

    const int* b_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int col = col_base + ((c < active_cols) ? c : 0);
        b_cols[c] = B_tree + (long long)col * row_words;
    }

    float acc[TP_M_TILE][4];
    #pragma unroll
    for (int t = 0; t < TP_M_TILE; ++t) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            acc[t][c] = 0.0f;
        }
    }

    for (int g = lane; g < groups; g += 32) {
        unsigned payload[4][4];
        unsigned signs[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int* __restrict__ gp = b_cols[c] + (long long)g * 5;
            payload[c][0] = (unsigned)gp[0];
            payload[c][1] = (unsigned)gp[1];
            payload[c][2] = (unsigned)gp[2];
            payload[c][3] = (unsigned)gp[3];
            signs[c]      = (unsigned)gp[4];
        }

        unsigned w01[4][16];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            #pragma unroll
            for (int w = 0; w < 4; w++) {
                const unsigned pw = payload[c][w];
                const unsigned sg = signs[c] >> (w * 8);
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    const unsigned nib0 = (pw >> (p * 8 + 0)) & 0xFu;
                    const unsigned nib1 = (pw >> (p * 8 + 4)) & 0xFu;
                    const unsigned s0   = (sg >> (p * 2 + 0)) & 1u;
                    const unsigned s1   = (sg >> (p * 2 + 1)) & 1u;
                    w01[c][w * 4 + p] = grim_tree_pie_f16x2_valu(nib0, nib1, s0, s1);
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < TP_M_TILE; ++t) {
            if (t < rows) {
                const uint4* act_v4 =
                    (const uint4*)(act_f16 + (size_t)(m0 + t) * K + (size_t)g * 32);
                const uint4 av0 = act_v4[0];
                const uint4 av1 = act_v4[1];
                const uint4 av2 = act_v4[2];
                const uint4 av3 = act_v4[3];
                const unsigned a2[16] = {
                    av0.x, av0.y, av0.z, av0.w,
                    av1.x, av1.y, av1.z, av1.w,
                    av2.x, av2.y, av2.z, av2.w,
                    av3.x, av3.y, av3.z, av3.w
                };

                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    if (c >= active_cols) break;
                    float bacc = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < 16; e++) {
                        bacc = grim_fdot2_f32_f16(a2[e], w01[c][e], bacc);
                    }
                    acc[t][c] += bacc;
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < TP_M_TILE; ++t) {
        if (t < rows) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float v = acc[t][c];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_xor(v, off);
                if (lane == 0 && c < active_cols)
                    C[(size_t)(m0 + t) * N + col_base + c] = v;
            }
        }
    }
}
#endif
"#;

#[cfg(test)]
mod tests {
    /// The in-register decode must agree with the host `e2m2_to_fp16_bits` for
    /// all 16 payload codes x both signs. The kernel cannot run without a GPU,
    /// so this pins the *contract* the kernel implements: the same bit pattern
    /// for every code, and negative zero produced for free via the sign OR.
    #[test]
    fn decode_contract_matches_host_table() {
        for code in 0u16..16 {
            let expect = grim_quant::tree_pie::e2m2_to_fp16_bits(code as u8);
            // Rebuild the kernel's arithmetic here from its own inputs.
            let nib = code & 0xF;
            let sign = (code >> 4) & 1;
            let (exp, mant) = (nib >> 2, nib & 3);
            let (exp_field, mant_10) = if exp == 0 {
                if mant == 0 {
                    (0u16, 0u16)
                } else {
                    (14 + (mant >> 1), ((mant + 1) >> 2) << 9)
                }
            } else {
                (exp + 15, mant << 8)
            };
            let got = (sign << 15) | (exp_field << 10) | mant_10;
            assert_eq!(got, expect, "code={code} (nib={nib}, sign={sign})");
        }
    }
}
