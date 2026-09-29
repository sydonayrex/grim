//! ScrubJay (WS-B) fused dequant + GEMV for the decode shape.
//!
//! ScrubJay is 5.5 bpw: a per-tensor fp32 pre-scale, then per 8-value block a
//! 4-bit codebook selector, 8 four-bit indices, an 8-bit sign plane, and one
//! E4M3 block scale. The decode is
//!
//!     v = sign(j) * CB[selector][index] * e4m3_scale * pre_scale
//!
//! with the frozen 16x16 codebook compiled in as a constant, so no dequantized
//! copy of B is ever uploaded -- the weights stay at 5.5 bpw in VRAM and the
//! reconstruction happens in-register.
//!
//! The sign plane is not optional. An earlier revision of the spec omitted it,
//! reasoning that the pre-scale's sign could flip a whole block; that yields
//! the right magnitude and the wrong sign for any block containing both
//! signs, which real weight blocks always do.
//!
//! Codebook entries are magnitudes, not signed values: the quantizer picks an
//! index by matching `|v| * inv` against the entries, and the sign arrives
//! separately through the sign plane.
//!
//! Requires K % 8 == 0 (the block granularity).

/// HIP source for [`crate::kernels::scrub_jay::KERNEL_SOURCE`].
pub const KERNEL_SOURCE: &str = r#"
#if defined(__AMDGCN__) || defined(__HIP__)
/// Frozen universal codebook: 16 books x 16 entries, generated from
/// `grim_quant::scrub_jay::SCRUB_JAY_CODEBOOK`. Do not hand-edit; a mismatch
/// here is silent, because the decode is table-driven and still returns
/// plausible numbers. `scrub_jay_gpu.rs` pins this table against the host
/// constant so a drift cannot reach hardware.
__constant__ static const signed char GRIM_SCRUB_JAY_CB[256] = {
     1,  8, 11, 14, 16, 18, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31,   // book 0
     0,  1,  9, 12, 14, 16, 18, 20, 21, 23, 24, 26, 27, 28, 30, 31,   // book 1
     0,  1,  8, 10, 12, 14, 16, 18, 20, 22, 23, 25, 27, 28, 30, 31,   // book 2
     0,  1,  6,  9, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 29, 31,   // book 3
     0,  1,  2,  7,  9, 12, 14, 16, 18, 20, 22, 23, 25, 27, 29, 31,   // book 4
     0,  1,  2,  6,  8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 31,   // book 5
     0,  1,  1,  2,  7,  9, 11, 13, 16, 18, 20, 22, 24, 26, 29, 31,   // book 6
     0,  1,  1,  2,  6,  8, 10, 12, 15, 17, 19, 21, 24, 26, 29, 31,   // book 7
     0,  1,  1,  2,  6,  7,  9, 12, 14, 16, 18, 21, 23, 26, 28, 31,   // book 8
     0,  1,  1,  2,  2,  7,  9, 11, 13, 15, 18, 20, 23, 25, 28, 31,   // book 9
     0,  1,  1,  2,  2,  6,  8, 10, 12, 14, 17, 19, 22, 25, 28, 31,   // book 10
     0,  1,  1,  2,  2,  5,  7,  9, 11, 14, 16, 19, 22, 25, 28, 31,   // book 11
     0,  1,  1,  1,  2,  2,  7,  8, 11, 13, 16, 18, 21, 24, 28, 31,   // book 12
     0,  1,  1,  1,  2,  2,  6,  8, 10, 12, 15, 18, 21, 24, 27, 31,   // book 13
     0,  1,  1,  1,  2,  2,  5,  7,  9, 12, 14, 17, 20, 24, 27, 31,   // book 14
     0,  1,  1,  1,  2,  2,  5,  7,  9, 11, 14, 17, 20, 23, 27, 31,   // book 15
};

/// Values per ScrubJay block.
#define GRIM_SJ_BLOCK 8

// M=1 decode GEMV: C[col] = sum_k A[k] * dequant(B[col, k])
//   A       : f32 [K]
//   B_sel   : u8  [N, K/8]   codebook selector per block
//   B_idx   : u8  [N, K]     4-bit index per value, one byte (unpacked)
//   B_sgn   : u8  [N, K/8]   sign bitmask per block, bit j = value j
//   B_scl   : f32 [N, K/8]   E4M3 block scale, decoded to f32 on the host
//   pre     : f32           per-tensor pre-scale
//   C       : f32 [N]
extern "C" __global__ void grim_scrub_jay_gemv(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_sel,
    const unsigned char* __restrict__ B_idx,
    const unsigned char* __restrict__ B_sgn,
    const float* __restrict__ B_scl,
    float pre,
    float* __restrict__ C,
    int N, int K)
{
    const int col  = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= N) return;

    const int n_blocks = K / GRIM_SJ_BLOCK;      // K % 8 == 0, checked host-side
    const long long blk_base = (long long)col * n_blocks;
    const long long idx_base = (long long)col * K;

    float facc = 0.0f;
    for (int blk = lane; blk < n_blocks; blk += 32) {
        const int sel = B_sel[blk_base + blk];
        const unsigned sgn = B_sgn[blk_base + blk];
        const float scl = B_scl[blk_base + blk] * pre;
        const signed char* book = GRIM_SCRUB_JAY_CB + sel * 16;

        float bacc = 0.0f;
        #pragma unroll
        for (int j = 0; j < GRIM_SJ_BLOCK; j++) {
            const int k = blk * GRIM_SJ_BLOCK + j;
            float w = (float)book[B_idx[idx_base + k]];
            if (sgn & (1u << j)) w = -w;
            bacc += A[k] * (w * scl);
        }
        facc += bacc;
    }

    for (int off = 16; off > 0; off >>= 1)
        facc += __shfl_xor(facc, off);
    if (lane == 0) C[col] = facc;
}

// Values per ScrubJay block.
#define GRIM_SJ_BLOCK 8
/// Values per launch group: one wave (32 lanes) x 4 values.
#define GRIM_SJ_GROUP 128

// Activation quantizer for the ScrubJay plan. Mirrors grim_quantize_u4_group128
// in structure -- max-reduce, then argmin over the codebooks, then pack -- with
// two differences forced by ScrubJay's shape:
//
//   * Per block of 8, not per group of 128. A group of 128 is 16 ScrubJay
//     blocks, each with its own selector and its own E4M3 scale, so the
//     max-reduce is over 8 values held by a lane pair, not over 128 held by a
//     wave. Lane `tid` owns values [4t, 4t+4), so block `tid/2` is owned by
//     lanes 2i and 2i+1 together and the 8 values are this lane's four plus
//     its partner's four, obtained with a single __shfl_xor.
//
//   * The codebook choice is an argmin over 16 books x 8 values, not a
//     uniform quantisation. Only the even lane of each pair evaluates it, and
//     broadcasts, so the 16x8x16 search runs once per block rather than twice.
//
// The scale is chosen so the block's peak magnitude lands on the codebook's top
// entry (31), which is the same convention quantize_block uses, so a
// quantize-then-dequantize round trip through this kernel is consistent with the
// weight-side codec.
extern "C" __global__ void grim_scrub_jay_quantize_u8(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst_idx,   // [n_rows, K]      one byte per value, 0..15
    unsigned char* __restrict__ dst_sel,   // [n_rows, K/8]    codebook selector
    float* __restrict__ dst_scl,           // [n_rows, K/8]    E4M3 block scale (f32 here)
    unsigned char* __restrict__ dst_sgn,   // [n_rows, K/8]    sign bitmask
    int K, int n_rows)
{
    const int n_groups = K / GRIM_SJ_GROUP;
    const int group = blockIdx.x;
    const int row  = group / n_groups;
    const int g    = group % n_groups;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;              // 0..31
    const int base = g * GRIM_SJ_GROUP + tid * 4;
    const long long src_row = (long long)row * K;

    const float v0 = src[src_row + base + 0];
    const float v1 = src[src_row + base + 1];
    const float v2 = src[src_row + base + 2];
    const float v3 = src[src_row + base + 3];

    // Partner lane's four values complete this block's 8.
    const float p0 = __shfl_xor(v0, 1);
    const float p1 = __shfl_xor(v1, 1);
    const float p2 = __shfl_xor(v2, 1);
    const float p3 = __shfl_xor(v3, 1);

    float bmax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    bmax = fmaxf(bmax, fmaxf(fmaxf(fabsf(p0), fabsf(p1)), fmaxf(fabsf(p2), fabsf(p3))));

    const float d     = bmax / 31.0f;
    const float inv_d = (bmax > 1e-9f) ? (31.0f / bmax) : 0.0f;

    // Codebook selection: argmin over the 16 books of the squared distance
    // from each value's scaled magnitude to that book's nearest entry. Done
    // once per block on the even lane, then broadcast.
    unsigned int sel = 0u;
    if ((tid & 1) == 0) {
        const float mag[8] = { fabsf(v0)*inv_d, fabsf(v1)*inv_d,
                               fabsf(v2)*inv_d, fabsf(v3)*inv_d,
                               fabsf(p0)*inv_d, fabsf(p1)*inv_d,
                               fabsf(p2)*inv_d, fabsf(p3)*inv_d };
        float best_err = 3.4e38f;
        for (int c = 0; c < 16; c++) {
            const signed char* book = GRIM_SCRUB_JAY_CB + c * 16;
            float err = 0.0f;
            for (int i = 0; i < 8; i++) {
                // nearest_level: min over entries of (entry - mag)^2
                float bd = 3.4e38f;
                for (int j = 0; j < 16; j++) {
                    const float t = (float)book[j] - mag[i];
                    bd = fminf(bd, t * t);
                }
                err += bd;
            }
            if (err < best_err) { best_err = err; sel = (unsigned)c; }
        }
    }
    // Broadcast from the even lane of the pair, NOT __shfl_xor.
    //
    // __shfl_xor(sel, 1) swaps: lane 0 ends up holding lane 1's value and vice
    // versa, and since only the even lane computed the selection, both lanes
    // kept the odd lane's zero. Every block then selected book 0 -- which
    // still dequantizes to finite weights and still produces plausible
    // numbers, so the GEMV parity test passed while the quantizer was silently
    // ignoring 15 of the 16 codebooks.
    sel = __shfl(sel, tid & ~1, 32);
    const signed char* book = GRIM_SCRUB_JAY_CB + (int)sel * 16;

    const int blk = g * (GRIM_SJ_GROUP / GRIM_SJ_BLOCK) + (tid >> 1);
    const long long blk_i = (long long)row * (K / GRIM_SJ_BLOCK) + blk;

    if ((tid & 1) == 0) {
        dst_sel[blk_i] = (unsigned char)sel;
        dst_scl[blk_i] = d;
        // Sign plane covers the whole 8-value block, so both lanes must agree
        // and both must write. Computed from the same eight magnitudes.
        unsigned int sgn = 0u;
        const float raw[8] = { v0, v1, v2, v3, p0, p1, p2, p3 };
        #pragma unroll
        for (int i = 0; i < 8; i++) if (raw[i] < 0.0f) sgn |= (1u << i);
        dst_sgn[blk_i] = (unsigned char)sgn;
    }

    // Map this lane's four values onto the selected book.
    const float mine[4] = { v0, v1, v2, v3 };
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const float scaled = fabsf(mine[i]) * inv_d;
        float bd = 3.4e38f;
        unsigned int best = 0u;
        for (int j = 0; j < 16; j++) {
            const float t = (float)book[j] - scaled;
            const float e = t * t;
            if (e < bd) { bd = e; best = (unsigned)j; }
        }
        dst_idx[src_row + base + i] = (unsigned char)best;
    }
}
#endif
"#;
