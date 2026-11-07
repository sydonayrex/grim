//! ScrubJay (WS-B) fused dequant + GEMV for the decode shape.
//!
//! ScrubJay is 5.5 bpw: a per-tensor fp32 pre-scale, then per 8-value block a
//! 4-bit codebook selector, 8 four-bit indices, an 8-bit sign plane, and one
//! E4M3 block scale. The decode is
//!
//! ```text
//! v = sign(j) * CB[selector][index] * e4m3_scale * pre_scale
//! ```
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
__constant__ static const signed char __attribute__((aligned(16))) GRIM_SCRUB_JAY_CB[256] = {
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

// v7/v9 word-wide VALU codebook lookup: given 4 unpacked 4-bit indices packed into
// the 4 bytes of `idx4` (`0..15`) and a 16-byte codebook row in `cb` (`uint4`),
// returns the 4 corresponding codebook entries packed into the 4 bytes of the
// output word using a single `__builtin_amdgcn_perm` pair (zero divergent scalar
// constant-cache sub-dword loads).
__device__ __forceinline__ unsigned int grim_sj_lookup4_valu(
    unsigned int idx4,
    uint4 cb)
{
    const unsigned int sel  = idx4 & 0x07070707u;
    const unsigned int lo   = __builtin_amdgcn_perm(cb.y, cb.x, sel);
    const unsigned int hi   = __builtin_amdgcn_perm(cb.w, cb.z, sel);
    const unsigned int h    = (idx4 >> 3) & 0x01010101u;
    const unsigned int m_hi = h * 0xFFu;
    return (lo & ~m_hi) | (hi & m_hi);
}

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

// ScrubJay v2 decode GEMV (v8/v9 architecture):
// - 4-wave CTA (`__launch_bounds__(128)`, 4 columns per CTA, 1 column per wave):
//   keeps per-lane VGPR footprint minimal (no 4-column index/selector spilling)
//   while packing 4 waves per CTA and using `uint2` (8-byte) index loads +
//   `float4` activation loads + `grim_sj_lookup4_valu` (`__builtin_amdgcn_perm`)
//   codebook decode with B-first load ordering.
// - Preserves exact per-lane `blk` and `j=0..7` accumulation order of `v1` (0 bit diffs).
extern "C" __global__ void __launch_bounds__(128) grim_scrub_jay_gemv_v2(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_sel,
    const unsigned char* __restrict__ B_idx,
    const unsigned char* __restrict__ B_sgn,
    const float* __restrict__ B_scl,
    float pre,
    float* __restrict__ C,
    int N, int K)
{
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int col  = blockIdx.x * 4 + wave;
    if (col >= N) return;

    const int n_blocks = K / GRIM_SJ_BLOCK;
    const long long blk_base = (long long)col * n_blocks;
    const uint2* idx_v2 = (const uint2*)(B_idx + (long long)col * K);
    const float4* a_v4  = (const float4*)A;
    const uint4* cb_v4  = (const uint4*)GRIM_SCRUB_JAY_CB;

    float facc = 0.0f;
    for (int blk = lane; blk < n_blocks; blk += 32) {
        // v8 B-first load ordering: issue B loads before A loads.
        const long long bi = blk_base + blk;
        const int sel      = B_sel[bi];
        const unsigned sg  = B_sgn[bi];
        const float sc     = B_scl[bi] * pre;
        const uint2 idx2   = idx_v2[blk];

        const float4 av0 = a_v4[blk * 2 + 0];
        const float4 av1 = a_v4[blk * 2 + 1];
        const float a_blk[8] = {
            av0.x, av0.y, av0.z, av0.w,
            av1.x, av1.y, av1.z, av1.w
        };

        const uint4 cb = cb_v4[sel];
        const unsigned int w_lo = grim_sj_lookup4_valu(idx2.x, cb);
        const unsigned int w_hi = grim_sj_lookup4_valu(idx2.y, cb);

        float bacc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float w = (float)((w_lo >> (j * 8)) & 0xFFu);
            if (sg & (1u << j)) w = -w;
            bacc += a_blk[j] * (w * sc);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float w = (float)((w_hi >> (j * 8)) & 0xFFu);
            if (sg & (1u << (j + 4))) w = -w;
            bacc += a_blk[j + 4] * (w * sc);
        }
        facc += bacc;
    }

    #pragma unroll
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

// Prefill/batched 2D tiled GEMM: C[row, col] = sum_k A[row, k] * dequant(B[col, k])
//   A       : f32 [M, K]
//   B_sel   : u8  [N, K/8]   codebook selector per block
//   B_idx   : u8  [N, K]     4-bit index per value, one byte
//   B_sgn   : u8  [N, K/8]   sign bitmask per block
//   B_scl   : f32 [N, K/8]   E4M3 block scale, decoded to f32 on the host
//   pre     : f32           per-tensor pre-scale
//   C       : f32 [M, N]
extern "C" __global__ void __launch_bounds__(256) grim_scrub_jay_gemm_tiled(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_sel,
    const unsigned char* __restrict__ B_idx,
    const unsigned char* __restrict__ B_sgn,
    const float* __restrict__ B_scl,
    float pre,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    const int n_blocks = K / GRIM_SJ_BLOCK;
    const long long blk_base = (long long)col * n_blocks;
    const long long idx_base = (long long)col * K;
    const long long a_base = (long long)row * K;

    float acc = 0.0f;
    for (int blk = 0; blk < n_blocks; ++blk) {
        const int sel = B_sel[blk_base + blk];
        const unsigned sgn = B_sgn[blk_base + blk];
        const float scl = B_scl[blk_base + blk] * pre;
        const signed char* book = GRIM_SCRUB_JAY_CB + sel * 16;

        #pragma unroll
        for (int j = 0; j < GRIM_SJ_BLOCK; ++j) {
            const int k = blk * GRIM_SJ_BLOCK + j;
            float w = (float)book[B_idx[idx_base + k]];
            if (sgn & (1u << j)) w = -w;
            acc += A[a_base + k] * (w * scl);
        }
    }

    C[(long long)row * N + col] = acc;
}

#define SJ_M_TILE 8

// v9 multi-row 4-column hoisted-B ScrubJay GEMM (`SJ_M_TILE = 8` rows x 4 columns
// per wave):
// - grid = (ceil(N / 4), ceil(M / SJ_M_TILE)), block = (32, 1, 1).
// - Coalesced K-loads across the 32-lane wave (`blk = lane; blk < n_blocks; blk += 32`),
//   replacing the uncoalesced 16x16 thread-per-(row,col) loop in `grim_scrub_jay_gemm_tiled`.
// - Decodes 4 B columns ONCE per 8-value block using `grim_sj_lookup4_valu`, then
//   dots against all `SJ_M_TILE = 8` rows of `A` (`float4` loads) and reduces via
//   `__shfl_xor` (100% bit-identical to `grim_scrub_jay_gemv_v2` per row).
extern "C" __global__ void __launch_bounds__(32) grim_scrub_jay_gemm_rows_v9(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_sel,
    const unsigned char* __restrict__ B_idx,
    const unsigned char* __restrict__ B_sgn,
    const float* __restrict__ B_scl,
    float pre,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row_base = blockIdx.y * SJ_M_TILE;
    const int lane     = threadIdx.x;
    if (row_base >= M || col_base >= N) return;

    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    const int active_rows = (row_base + SJ_M_TILE <= M) ? SJ_M_TILE : (M - row_base);
    const int n_blocks    = K / GRIM_SJ_BLOCK;
    const uint4* cb_v4    = (const uint4*)GRIM_SCRUB_JAY_CB;

    long long blk_base[4];
    const uint2* idx_v2[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int col = col_base + ((c < active_cols) ? c : 0);
        blk_base[c] = (long long)col * n_blocks;
        idx_v2[c]   = (const uint2*)(B_idx + (long long)col * K);
    }

    float facc[SJ_M_TILE][4];
    #pragma unroll
    for (int r = 0; r < SJ_M_TILE; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            facc[r][c] = 0.0f;
        }
    }

    for (int blk = lane; blk < n_blocks; blk += 32) {
        float w_val[4][8];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const long long bi = blk_base[c] + blk;
            const int sel      = B_sel[bi];
            const unsigned sg  = B_sgn[bi];
            const float sc     = B_scl[bi] * pre;
            const uint2 idx2   = idx_v2[c][blk];
            const uint4 cb     = cb_v4[sel];
            const unsigned int w_lo = grim_sj_lookup4_valu(idx2.x, cb);
            const unsigned int w_hi = grim_sj_lookup4_valu(idx2.y, cb);

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float w = (float)((w_lo >> (j * 8)) & 0xFFu);
                if (sg & (1u << j)) w = -w;
                w_val[c][j] = w * sc;
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float w = (float)((w_hi >> (j * 8)) & 0xFFu);
                if (sg & (1u << (j + 4))) w = -w;
                w_val[c][j + 4] = w * sc;
            }
        }

        #pragma unroll
        for (int r = 0; r < SJ_M_TILE; ++r) {
            if (r < active_rows) {
                const float4* a_v4 =
                    (const float4*)(A + (long long)(row_base + r) * K + (long long)blk * 8);
                const float4 av0 = a_v4[0];
                const float4 av1 = a_v4[1];
                const float a_blk[8] = {
                    av0.x, av0.y, av0.z, av0.w,
                    av1.x, av1.y, av1.z, av1.w
                };

                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    if (c >= active_cols) break;
                    float bacc = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        bacc += a_blk[j] * w_val[c][j];
                    }
                    facc[r][c] += bacc;
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < SJ_M_TILE; ++r) {
        if (r < active_rows) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float v = facc[r][c];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_xor(v, off);
                if (lane == 0 && c < active_cols)
                    C[(long long)(row_base + r) * N + col_base + c] = v;
            }
        }
    }
}

// v10 ScrubJay 4-wave cooperative-LDS kernels (`__launch_bounds__(128, 2)`):
// - `grim_scrub_jay_gemv_v10` (M=1): 4 waves per CTA (`16` cols/CTA, `4` cols/wave),
//   cooperative LDS staging of the 32-block `float4` activation tile (`s_a[32][2]`),
//   B-first load ordering + `grim_sj_lookup4_valu` word-wide codebook decode.
//   100% bit-identical to `v1` and `v2`.
// - `grim_scrub_jay_gemm_rows_v10` (M>1): 4 waves per CTA (`SJ_M_TILE = 8` rows x `16` cols/CTA),
//   cooperative LDS staging of the 8-row `float4` activation tile (`s_a[8][32][2]`, 2 rows per wave),
//   cutting L2 A-bandwidth by 4x and capping VGPRs via `__launch_bounds__(128, 2)`.
extern "C" __global__ void __launch_bounds__(128) grim_scrub_jay_gemv_v10(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_sel,
    const unsigned char* __restrict__ B_idx,
    const unsigned char* __restrict__ B_sgn,
    const float* __restrict__ B_scl,
    float pre,
    float* __restrict__ C,
    int N, int K)
{
    const int wave     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int col_base = blockIdx.x * 8 + wave * 2;

    const int active_cols = (col_base >= N) ? 0 : ((col_base + 2 <= N) ? 2 : (N - col_base));
    if (active_cols <= 0) return;

    const int n_blocks = K / GRIM_SJ_BLOCK;
    const float4* a_v4 = (const float4*)A;
    const uint4* cb_v4 = (const uint4*)GRIM_SCRUB_JAY_CB;

    const int col0 = col_base;
    const int col1 = (active_cols > 1) ? (col_base + 1) : col_base;
    const long long blk_base0 = (long long)col0 * n_blocks;
    const long long blk_base1 = (long long)col1 * n_blocks;
    const uint2* idx_v2_0 = (const uint2*)(B_idx + (long long)col0 * K);
    const uint2* idx_v2_1 = (const uint2*)(B_idx + (long long)col1 * K);

    float facc0 = 0.0f;
    float facc1 = 0.0f;
    for (int blk = lane; blk < n_blocks; blk += 32) {
        const long long bi0 = blk_base0 + blk;
        const long long bi1 = blk_base1 + blk;
        const int sel0      = B_sel[bi0];
        const int sel1      = B_sel[bi1];
        const unsigned sg0  = B_sgn[bi0];
        const unsigned sg1  = B_sgn[bi1];
        const float sc0     = B_scl[bi0] * pre;
        const float sc1     = B_scl[bi1] * pre;
        const uint2 idx2_0  = idx_v2_0[blk];
        const uint2 idx2_1  = idx_v2_1[blk];

        const float4 av0 = a_v4[blk * 2 + 0];
        const float4 av1 = a_v4[blk * 2 + 1];
        const float a_blk[8] = {
            av0.x, av0.y, av0.z, av0.w,
            av1.x, av1.y, av1.z, av1.w
        };

        const uint4 cb0 = cb_v4[sel0];
        const unsigned int w_lo0 = grim_sj_lookup4_valu(idx2_0.x, cb0);
        const unsigned int w_hi0 = grim_sj_lookup4_valu(idx2_0.y, cb0);
        float bacc0 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float w = (float)((w_lo0 >> (j * 8)) & 0xFFu);
            if (sg0 & (1u << j)) w = -w;
            bacc0 += a_blk[j] * (w * sc0);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float w = (float)((w_hi0 >> (j * 8)) & 0xFFu);
            if (sg0 & (1u << (j + 4))) w = -w;
            bacc0 += a_blk[j + 4] * (w * sc0);
        }
        facc0 += bacc0;

        const uint4 cb1 = cb_v4[sel1];
        const unsigned int w_lo1 = grim_sj_lookup4_valu(idx2_1.x, cb1);
        const unsigned int w_hi1 = grim_sj_lookup4_valu(idx2_1.y, cb1);
        float bacc1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float w = (float)((w_lo1 >> (j * 8)) & 0xFFu);
            if (sg1 & (1u << j)) w = -w;
            bacc1 += a_blk[j] * (w * sc1);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float w = (float)((w_hi1 >> (j * 8)) & 0xFFu);
            if (sg1 & (1u << (j + 4))) w = -w;
            bacc1 += a_blk[j + 4] * (w * sc1);
        }
        facc1 += bacc1;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        facc0 += __shfl_xor(facc0, off);
        facc1 += __shfl_xor(facc1, off);
    }
    if (lane == 0) {
        C[col0] = facc0;
        if (active_cols > 1) C[col1] = facc1;
    }
}

extern "C" __global__ void __launch_bounds__(128, 2) grim_scrub_jay_gemm_rows_v10(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_sel,
    const unsigned char* __restrict__ B_idx,
    const unsigned char* __restrict__ B_sgn,
    const float* __restrict__ B_scl,
    float pre,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int wave     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int col_base = blockIdx.x * 16 + wave * 4;
    const int row_base = blockIdx.y * SJ_M_TILE;
    if (row_base >= M) return;

    const int active_cols = (col_base >= N) ? 0 : ((col_base + 4 <= N) ? 4 : (N - col_base));
    const int active_rows = (row_base + SJ_M_TILE <= M) ? SJ_M_TILE : (M - row_base);
    const int n_blocks    = K / GRIM_SJ_BLOCK;
    const uint4* cb_v4    = (const uint4*)GRIM_SCRUB_JAY_CB;

    long long blk_base[4];
    const uint2* idx_v2[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int col = (c < active_cols) ? (col_base + c) : 0;
        blk_base[c] = (long long)col * n_blocks;
        idx_v2[c]   = (const uint2*)(B_idx + (long long)col * K);
    }

    __shared__ float4 s_a[SJ_M_TILE][32][2];

    float facc[SJ_M_TILE][4];
    #pragma unroll
    for (int r = 0; r < SJ_M_TILE; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            facc[r][c] = 0.0f;
        }
    }

    for (int blk_tile = 0; blk_tile < n_blocks; blk_tile += 32) {
        const int blk = blk_tile + lane;
        int sel[4];
        unsigned sg[4];
        float sc[4];
        uint2 idx2[4];
        if (blk < n_blocks && active_cols > 0) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const long long bi = blk_base[c] + blk;
                sel[c]  = B_sel[bi];
                sg[c]   = B_sgn[bi];
                sc[c]   = B_scl[bi] * pre;
                idx2[c] = idx_v2[c][blk];
            }
        }

        // All 4 waves cooperatively stage the 8-row activation tile into LDS (2 rows per wave):
        for (int r = wave; r < active_rows; r += 4) {
            if (blk < n_blocks) {
                const float4* a_v4 =
                    (const float4*)(A + (long long)(row_base + r) * K + (long long)blk * 8);
                s_a[r][lane][0] = a_v4[0];
                s_a[r][lane][1] = a_v4[1];
            }
        }
        __syncthreads();

        if (blk < n_blocks && active_cols > 0) {
            float w_val[4][8];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const uint4 cb     = cb_v4[sel[c]];
                const unsigned sgc = sg[c];
                const float scc    = sc[c];
                const unsigned int w_lo = grim_sj_lookup4_valu(idx2[c].x, cb);
                const unsigned int w_hi = grim_sj_lookup4_valu(idx2[c].y, cb);

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float w = (float)((w_lo >> (j * 8)) & 0xFFu);
                    if (sgc & (1u << j)) w = -w;
                    w_val[c][j] = w * scc;
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float w = (float)((w_hi >> (j * 8)) & 0xFFu);
                    if (sgc & (1u << (j + 4))) w = -w;
                    w_val[c][j + 4] = w * scc;
                }
            }

            #pragma unroll
            for (int r = 0; r < SJ_M_TILE; ++r) {
                if (r < active_rows) {
                    const float4 av0 = s_a[r][lane][0];
                    const float4 av1 = s_a[r][lane][1];
                    const float a_blk[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w
                    };

                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        if (c >= active_cols) break;
                        float bacc = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            bacc += a_blk[j] * w_val[c][j];
                        }
                        facc[r][c] += bacc;
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < SJ_M_TILE; ++r) {
        if (r < active_rows) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float v = facc[r][c];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_xor(v, off);
                if (lane == 0 && c < active_cols)
                    C[(long long)(row_base + r) * N + col_base + c] = v;
            }
        }
    }
}
#endif
"#;
