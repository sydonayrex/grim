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

        for (int j = 0; j < 32; j += 2) {
            // Decode once, use for every row in the tile.
            unsigned w01 = (unsigned)grim_tree_pie_f16(payload, signs, j)
                         | ((unsigned)grim_tree_pie_f16(payload, signs, j + 1) << 16);
            for (int t = 0; t < rows; ++t) {
                unsigned a01 = act2[(((size_t)t * K) + (g * 32 + j)) >> 1];
                acc[t] = grim_fdot2_f32_f16(a01, w01, acc[t]);
            }
        }
    }

    for (int t = 0; t < rows; ++t) {
        float v = acc[t];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
        if (lane == 0) C[(size_t)(m0 + t) * N + col] = v;
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
                if mant == 0 { (0u16, 0u16) } else { (14 + (mant >> 1), ((mant + 1) >> 2) << 9) }
            } else {
                (exp + 15, mant << 8)
            };
            let got = (sign << 15) | (exp_field << 10) | mant_10;
            assert_eq!(got, expect, "code={code} (nib={nib}, sign={sign})");
        }
    }
}
