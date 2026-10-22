//! SPEED-ROC-8: LDS-tiled fused dequant GEMM kernels for the Q8_0, Q5_K, Q6_K
//! and IQ quant families, stamped out from one HIP macro.
//!
//! Identical tiling discipline to `grim_fused_dequant_gemm_q4k_tiled`
//! (kernels::q4k_gemm): 4x64 output tile per block, 16 KB weight LDS, each
//! weight element dequantized once per M-tile instead of once per output row.
//!
//! This source MUST ride the aggregate JIT unit AFTER iq_gemm / q5k_gemm /
//! q6k_gemm (it calls their `dequant_*` device helpers). See
//! `tiled_quant_kernel_source` + the push site in kernels::source_asm.

/// Always-included family: Q8_0 + all IQ formats (their scalar kernels and
/// `dequant_*` helpers in kernels::iq_gemm are not feature-gated).
pub const IQ_Q80_TILED_KERNEL_SOURCE: &str = r#"
#define GRIM_TILED_QUANT_FWD_BWD(FWD, BWD, DEQ, BLK, BYTES)                            \
    __global__ void FWD(const float* __restrict__ A,                                   \
                        const unsigned char* __restrict__ B,                           \
                        float* __restrict__ C,                                         \
                        int M, int N, int K)                                           \
    {                                                                                  \
        __shared__ float sW[64][64];                                                   \
        __shared__ float sA[4][64];                                                    \
        const int col0 = blockIdx.x * 64;                                              \
        const int row0 = blockIdx.y * 4;                                               \
        const int tx = threadIdx.x;                                                    \
        const int ty = threadIdx.y;                                                    \
        const int row = row0 + ty;                                                     \
        const int col = col0 + tx;                                                     \
        const bool tile_ok = (row < M) && (col < N);                                   \
        const int row_bytes = (K / (BLK)) * (BYTES);                                   \
        float acc = 0.0f;                                                              \
        for (int k0 = 0; k0 < K; k0 += 64) {                                           \
            for (int t = ty * 64 + tx; t < 64 * 64; t += 256) {                        \
                int kk = t >> 6;                                                       \
                int cc = t & 63;                                                       \
                int gk = k0 + kk;                                                      \
                int gcc = col0 + cc;                                                   \
                float w = 0.0f;                                                        \
                if (gcc < N && gk < K) {                                               \
                    w = DEQ(B + (long long)gcc * row_bytes                             \
                              + (long long)(gk / (BLK)) * (BYTES),                     \
                            gk % (BLK));                                               \
                }                                                                      \
                sW[kk][cc] = w;                                                        \
            }                                                                          \
            {                                                                          \
                int gk = k0 + tx;                                                      \
                int rr = row0 + ty;                                                    \
                sA[ty][tx] = (rr < M && gk < K) ? A[(long long)rr * K + gk] : 0.0f;    \
            }                                                                          \
            __syncthreads();                                                           \
            if (tile_ok) {                                                             \
                for (int kk = 0; kk < 64; ++kk) acc += sA[ty][kk] * sW[kk][tx];        \
            }                                                                          \
            __syncthreads();                                                           \
        }                                                                              \
        if (tile_ok) {                                                                 \
            C[(long long)row * N + col] = acc;                                         \
        }                                                                              \
    }                                                                                  \
    __global__ void BWD(const float* __restrict__ dY,                                  \
                        const unsigned char* __restrict__ B,                           \
                        float* __restrict__ dX,                                        \
                        int M, int N, int K)                                           \
    {                                                                                  \
        __shared__ float sW[64][64];                                                   \
        __shared__ float sY[4][64];                                                    \
        const int k0 = blockIdx.x * 64;                                                \
        const int row0 = blockIdx.y * 4;                                               \
        const int tx = threadIdx.x;                                                    \
        const int ty = threadIdx.y;                                                    \
        const int row = row0 + ty;                                                     \
        const int kcol = k0 + tx;                                                      \
        const bool tile_ok = (row < M) && (kcol < K);                                  \
        const int row_bytes = (K / (BLK)) * (BYTES);                                   \
        float acc = 0.0f;                                                              \
        for (int n0 = 0; n0 < N; n0 += 64) {                                           \
            for (int t = ty * 64 + tx; t < 64 * 64; t += 256) {                        \
                int kk = t >> 6;                                                       \
                int cc = t & 63;                                                       \
                int gk = k0 + kk;                                                      \
                int gn = n0 + cc;                                                      \
                float w = 0.0f;                                                        \
                if (gn < N && gk < K) {                                                \
                    w = DEQ(B + (long long)gn * row_bytes                              \
                              + (long long)(gk / (BLK)) * (BYTES),                     \
                            gk % (BLK));                                               \
                }                                                                      \
                sW[cc][kk] = w;                                                        \
            }                                                                          \
            {                                                                          \
                int gn = n0 + tx;                                                      \
                int rr = row0 + ty;                                                    \
                sY[ty][tx] = (rr < M && gn < N) ? dY[(long long)rr * N + gn] : 0.0f;   \
            }                                                                          \
            __syncthreads();                                                           \
            if (tile_ok) {                                                             \
                for (int cc = 0; cc < 64; ++cc) acc += sY[ty][cc] * sW[cc][tx];        \
            }                                                                          \
            __syncthreads();                                                           \
        }                                                                              \
        if (tile_ok) {                                                                 \
            dX[(long long)row * K + kcol] = acc;                                       \
        }                                                                              \
    }

// IQ family (256-element super-blocks). Byte geometry mirrors the scalar
// kernels in kernels::iq_gemm. extern "C" is required: hipModuleGetFunction
// resolves by unmangled name (the launch path looks these up by entry).
extern "C" {
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq2xxs_tiled,
                         grim_fused_dequant_gemm_iq2xxs_backward_tiled,
                         dequant_iq2xxs, 256, 66)
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq2xs_tiled,
                         grim_fused_dequant_gemm_iq2xs_backward_tiled,
                         dequant_iq2xs, 256, 74)
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq2s_tiled,
                         grim_fused_dequant_gemm_iq2s_backward_tiled,
                         dequant_iq2s, 256, 82)
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq3xxs_tiled,
                         grim_fused_dequant_gemm_iq3xxs_backward_tiled,
                         dequant_iq3xxs, 256, 98)
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq3s_tiled,
                         grim_fused_dequant_gemm_iq3s_backward_tiled,
                         dequant_iq3s, 256, 110)
// IQ4_NL has NO tiled entry. Its block is QK4_NL = 32 weights in 18 bytes,
// which cannot fill a 256-element WMMA tile; listing it here also imposed
// `k % tiled_k_multiple(256) == 0` on a 32-element-block format. The
// untiled `grim_fused_dequant_gemm_iq4nl` is the correct path, and
// `tiled_quant_block_sizes_match_llama_cpp` fails the build if iq4nl ever
// reappears in the dispatch table.
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq4xs_tiled,
                         grim_fused_dequant_gemm_iq4xs_backward_tiled,
                         dequant_iq4xs, 256, 136)
// Q8_0: 32-element blocks, 34 bytes each (fp16 delta + 32 int8 codes).
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_q8_0_tiled,
                         grim_fused_dequant_gemm_q8_0_backward_tiled,
                         dequant_q80_standalone, 32, 34)
}

// NOTE: the macro is deliberately NOT #undef'd here — the feature-gated
// Q5_K / Q6_K fragments (Q5K_TILED_KERNEL_SOURCE / Q6K_TILED_KERNEL_SOURCE)
// are appended after this source and reuse it.
"#;

/// Q5_K tiled kernels — only compiled when the `q5k` feature gates the scalar
/// kernel (and its `dequant_q5k_element` helper) into the aggregate.
#[cfg(feature = "q5k")]
pub const Q5K_TILED_KERNEL_SOURCE: &str = r#"
extern "C" {
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_q5k_tiled,
                         grim_fused_dequant_gemm_q5k_backward_tiled,
                         dequant_q5k_element, 256, 176)
}
"#;

/// Q6_K tiled kernels — see `Q5K_TILED_KERNEL_SOURCE` for the gating rationale.
#[cfg(feature = "q6k")]
pub const Q6K_TILED_KERNEL_SOURCE: &str = r#"
extern "C" {
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_q6k_tiled,
                         grim_fused_dequant_gemm_q6k_backward_tiled,
                         dequant_q6k_element, 256, 210)
}
"#;

/// Q3_K / Q2_K tiled kernels — M7.2a.
///
/// The IQ recipe (the LDS-tiled prompt GEMM) was landed for Q8_0, Q5_K, Q6_K
/// and the IQ families, and NOT for these two. The macro is format-agnostic
/// — it takes the `dequant_*` device helper as a parameter, and both helpers
/// exist — so the omission was a gap in the stamps, not a limitation of the
/// recipe.
///
/// It cost real time, and it is measurable at the 27B's dominant projection
/// shape (n=17408, k=5120): at m=64 the scalar fallback re-reads every
/// weight block M times, and Q3_K measured 7.81 ms (4.9 GB/s) and Q2_K 6.90
/// ms (4.2 GB/s) where Q5_K/Q6_K/Q8_0 measure ~1.25 ms. The IQ formats
/// reach ~0.5 ms on the same shape.
///
/// Q4_K is NOT here: it already has a hand-written tiled kernel in
/// `kernels::q4k_gemm` (`grim_fused_dequant_gemm_q4k_tiled`, SPEED-ROC-6),
/// and a second definition of the same symbol is a redefinition error.
///
/// The byte geometry mirrors the scalar kernels: Q3_K 110 B / 256 elements,
/// Q2_K 84 B / 256.
#[cfg(feature = "q3k")]
pub const Q3K_TILED_KERNEL_SOURCE: &str = r#"
extern "C" {
GRIM_TILED_QUANT_FWD_BWD_GROUPED(grim_fused_dequant_gemm_q3k_tiled,
                                 grim_fused_dequant_gemm_q3k_backward_tiled,
                                 dequant_q3k_header, dequant_q3k_quad, 256, 110)
}
"#;

/// Q2_K tiled kernels — see `Q3K_TILED_KERNEL_SOURCE` for the rationale.
#[cfg(feature = "q2k")]
pub const Q2K_TILED_KERNEL_SOURCE: &str = r#"
extern "C" {
GRIM_TILED_QUANT_FWD_BWD_GROUPED(grim_fused_dequant_gemm_q2k_tiled,
                                 grim_fused_dequant_gemm_q2k_backward_tiled,
                                 dequant_q2k_header, dequant_q2k_quad, 256, 84)
}
"#;


/// M7.2f: the GROUPED tiling macro — the LDS-tiled recipe WITH the group
/// hoist, which the plain macro cannot express.
///
/// The plain `GRIM_TILED_QUANT_FWD_BWD` calls `DEQ(block, in_sb)` once per
/// staged weight, so a format whose per-element decoder re-derives the
/// super-block header pays for it 4096 times per tile. For the K-quants that
/// is the whole cost: measured at the 27B's shape, the untiled-group Q3_K
/// tile ran 12.7x SLOWER than the scalar path (99.4 ms against 7.8 ms),
/// because Q3_K's header is a 12-byte ggml aux shuffle into 16 scale bytes
/// and the tile's 4096 weights come from 64 DIFFERENT weight rows, so
/// there is no header to amortise inside the weight loop.
///
/// This macro gives the tile the IQ recipe's shape: a 64-wide k chunk lies
/// inside ONE super-block (BLK=256 divides it), so the header is decoded
/// ONCE per (column, chunk) and the weights of that column are then read
/// four at a time with wide loads.
///
/// `HDR(blk, &d, &dmin, sc16)` decodes the group header; `W4(blk, in_sb, d,
/// dmin, sc16, out4)` decodes FOUR consecutive weights of that group into
/// `out4`. Both are format-specific; the tiling is not.
pub const GROUPED_TILED_MACRO_SOURCE: &str = r#"
#define GRIM_TILED_QUANT_FWD_BWD_GROUPED(FWD, BWD, HDR, W4, BLK, BYTES)            \
    __global__ void FWD(const float* __restrict__ A,                                 \
                        const unsigned char* __restrict__ B,                         \
                        float* __restrict__ C,                                       \
                        int M, int N, int K)                                         \
    {                                                                                \
        __shared__ float sW[64][64];                                                 \
        __shared__ float sA[4][64];                                                  \
        const int col0 = blockIdx.x * 64;                                            \
        const int row0 = blockIdx.y * 4;                                             \
        const int tx = threadIdx.x;                                                  \
        const int ty = threadIdx.y;                                                  \
        const int row = row0 + ty;                                                   \
        const int col = col0 + tx;                                                   \
        const bool tile_ok = (row < M) && (col < N);                                 \
        const int row_bytes = (K / (BLK)) * (BYTES);                                 \
        float acc = 0.0f;                                                            \
        for (int k0 = 0; k0 < K; k0 += 64) {                                         \
            float hd = 0.0f, hdm = 0.0f, hsc = 0.0f, hsm = 0.0f;                    \
            const unsigned char* blk = (const unsigned char*)0;                      \
            if (col < N && k0 < K) {                                                 \
                blk = B + (long long)col * row_bytes                                 \
                    + (long long)(k0 / (BLK)) * (BYTES);                             \
                HDR(blk, (k0 % (BLK)) + ty * 16, &hd, &hdm, &hsc, &hsm);            \
            }                                                                        \
            for (int kg = 0; kg < 4; ++kg) {                                         \
                /* FOUR SEPARATE SCALARS, not an array: a local array      \
                   indexed at runtime lands in SCRATCH memory, and the     \
                   scratch traffic is per-thread, so the 32 lanes of a wave \
                   cannot coalesce it. */                                      \
                float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;                  \
                if (blk != (const unsigned char*)0) {                                \
                    W4(blk, (k0 % (BLK)) + ty * 16 + kg * 4,                        \
                       hd, hdm, hsc, hsm, &w0, &w1, &w2, &w3);                     \
                }                                                                    \
                const int kk0 = ty * 16 + kg * 4;                                    \
                sW[kk0 + 0][tx] = (kk0 + 0 < 64 && k0 + kk0 + 0 < K) ? w0 : 0.0f;   \
                sW[kk0 + 1][tx] = (kk0 + 1 < 64 && k0 + kk0 + 1 < K) ? w1 : 0.0f;   \
                sW[kk0 + 2][tx] = (kk0 + 2 < 64 && k0 + kk0 + 2 < K) ? w2 : 0.0f;   \
                sW[kk0 + 3][tx] = (kk0 + 3 < 64 && k0 + kk0 + 3 < K) ? w3 : 0.0f;   \
            }                                                                        \
            {                                                                        \
                const int gk = k0 + tx;                                              \
                const int rr = row0 + ty;                                            \
                sA[ty][tx] = (rr < M && gk < K) ? A[(long long)rr * K + gk] : 0.0f;  \
            }                                                                        \
            __syncthreads();                                                         \
            if (tile_ok) {                                                           \
                for (int kk = 0; kk < 64; ++kk) acc += sA[ty][kk] * sW[kk][tx];      \
            }                                                                        \
            __syncthreads();                                                         \
        }                                                                            \
        if (tile_ok) C[(long long)row * N + col] = acc;                              \
    }                                                                                \
    __global__ void BWD(const float* __restrict__ dY,                                \
                        const unsigned char* __restrict__ B,                         \
                        float* __restrict__ dX,                                      \
                        int M, int N, int K)                                         \
    {                                                                                \
        __shared__ float sW[64][64];                                                 \
        __shared__ float sY[4][64];                                                  \
        const int k0 = blockIdx.x * 64;                                              \
        const int row0 = blockIdx.y * 4;                                             \
        const int tx = threadIdx.x;                                                  \
        const int ty = threadIdx.y;                                                  \
        const int row = row0 + ty;                                                   \
        const int kcol = k0 + tx;                                                    \
        const bool tile_ok = (row < M) && (kcol < K);                                \
        const int row_bytes = (K / (BLK)) * (BYTES);                                 \
        float acc = 0.0f;                                                            \
        for (int n0 = 0; n0 < N; n0 += 64) {                                         \
            float hd = 0.0f, hdm = 0.0f, hsc = 0.0f, hsm = 0.0f;                    \
            const unsigned char* blk = (const unsigned char*)0;                      \
            const int gn = n0 + tx;                                                  \
            if (gn < N && k0 < K) {                                                  \
                blk = B + (long long)gn * row_bytes                                  \
                    + (long long)(k0 / (BLK)) * (BYTES);                             \
                HDR(blk, (k0 % (BLK)) + ty * 16, &hd, &hdm, &hsc, &hsm);            \
            }                                                                        \
            for (int kg = 0; kg < 4; ++kg) {                                         \
                float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;                  \
                if (blk != (const unsigned char*)0) {                                \
                    W4(blk, (k0 % (BLK)) + ty * 16 + kg * 4,                        \
                       hd, hdm, hsc, hsm, &w0, &w1, &w2, &w3);                     \
                }                                                                    \
                const int kk0 = ty * 16 + kg * 4;                                    \
                sW[tx][kk0 + 0] = (kk0 + 0 < 64 && k0 + kk0 + 0 < K) ? w0 : 0.0f;   \
                sW[tx][kk0 + 1] = (kk0 + 1 < 64 && k0 + kk0 + 1 < K) ? w1 : 0.0f;   \
                sW[tx][kk0 + 2] = (kk0 + 2 < 64 && k0 + kk0 + 2 < K) ? w2 : 0.0f;   \
                sW[tx][kk0 + 3] = (kk0 + 3 < 64 && k0 + kk0 + 3 < K) ? w3 : 0.0f;   \
            }                                                                        \
            {                                                                        \
                const int gn2 = n0 + tx;                                             \
                const int rr = row0 + ty;                                            \
                sY[ty][tx] = (rr < M && gn2 < N) ? dY[(long long)rr * N + gn2] : 0.0f; \
            }                                                                        \
            __syncthreads();                                                         \
            if (tile_ok) {                                                           \
                for (int cc = 0; cc < 64; ++cc) acc += sY[ty][cc] * sW[cc][tx];      \
            }                                                                        \
            __syncthreads();                                                         \
        }                                                                            \
        if (tile_ok) dX[(long long)row * K + kcol] = acc;                            \
    }
"#;

/// Assemble the tiled-source fragment for the enabled feature set. The
/// `#define GRIM_TILED_QUANT_FWD_BWD` lives in `IQ_Q80_TILED_KERNEL_SOURCE`,
/// so that fragment must be pushed first whenever any part of this module is.
pub fn tiled_quant_kernel_source() -> String {
    let mut s = String::new();
    // The grouped macro must be defined before any stamp that uses it.
    s.push_str(GROUPED_TILED_MACRO_SOURCE);
    s.push_str(IQ_Q80_TILED_KERNEL_SOURCE);
    #[cfg(feature = "q5k")]
    s.push_str(Q5K_TILED_KERNEL_SOURCE);
    #[cfg(feature = "q6k")]
    s.push_str(Q6K_TILED_KERNEL_SOURCE);
    // M7.2a: the two formats the IQ recipe never reached. (Q4_K already has
    // a hand-written tiled kernel in kernels::q4k_gemm.)
    #[cfg(feature = "q3k")]
    s.push_str(Q3K_TILED_KERNEL_SOURCE);
    #[cfg(feature = "q2k")]
    s.push_str(Q2K_TILED_KERNEL_SOURCE);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_source_contains_all_included_formats() {
        let src = tiled_quant_kernel_source();
        // IQ4_NL is intentionally absent: its QK4_NL = 32-element blocks (18 bytes)
        // cannot fill a 256-element WMMA tile, and listing it here imposes
        // `k % tiled_k_multiple(256) == 0` on a 32-element-block format.
        // The untiled `grim_fused_dequant_gemm_iq4nl` is the correct path.
        // See the comment above `GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq4xs_tiled`
        // (commit 431e95f5).
        for fmt in ["iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", "iq4xs", "q8_0"] {
            assert!(
                src.contains(&format!("grim_fused_dequant_gemm_{fmt}_tiled")),
                "missing tiled forward kernel for {fmt}"
            );
            assert!(
                src.contains(&format!("grim_fused_dequant_gemm_{fmt}_backward_tiled")),
                "missing tiled backward kernel for {fmt}"
            );
        }
        // Pin the explicit exclusion: iq4nl must NOT appear as a tiled kernel.
        assert!(
            !src.contains("grim_fused_dequant_gemm_iq4nl_tiled"),
            "iq4nl must not have a tiled kernel (32-elem blocks cannot fill a 256-elem tile)"
        );
    }

    #[cfg(feature = "q5k")]
    #[test]
    fn tiled_source_contains_q5k() {
        assert!(tiled_quant_kernel_source().contains("grim_fused_dequant_gemm_q5k_tiled"));
    }

    #[cfg(feature = "q6k")]
    #[test]
    fn tiled_source_contains_q6k() {
        assert!(tiled_quant_kernel_source().contains("grim_fused_dequant_gemm_q6k_tiled"));
    }
}
