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
#endif
"#;
