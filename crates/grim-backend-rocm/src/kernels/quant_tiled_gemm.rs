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
                         dequant_iq3xxs, 256, 96)
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq3s_tiled,
                         grim_fused_dequant_gemm_iq3s_backward_tiled,
                         dequant_iq3s, 256, 110)
GRIM_TILED_QUANT_FWD_BWD(grim_fused_dequant_gemm_iq4nl_tiled,
                         grim_fused_dequant_gemm_iq4nl_backward_tiled,
                         dequant_iq4nl, 256, 170)
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

/// Assemble the tiled-source fragment for the enabled feature set. The
/// `#define GRIM_TILED_QUANT_FWD_BWD` lives in `IQ_Q80_TILED_KERNEL_SOURCE`,
/// so that fragment must be pushed first whenever any part of this module is.
pub fn tiled_quant_kernel_source() -> String {
    let mut s = String::new();
    s.push_str(IQ_Q80_TILED_KERNEL_SOURCE);
    #[cfg(feature = "q5k")]
    s.push_str(Q5K_TILED_KERNEL_SOURCE);
    #[cfg(feature = "q6k")]
    s.push_str(Q6K_TILED_KERNEL_SOURCE);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_source_contains_all_included_formats() {
        let src = tiled_quant_kernel_source();
        for fmt in [
            "iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", "iq4nl", "iq4xs", "q8_0",
        ] {
            assert!(
                src.contains(&format!("grim_fused_dequant_gemm_{fmt}_tiled")),
                "missing tiled forward kernel for {fmt}"
            );
            assert!(
                src.contains(&format!("grim_fused_dequant_gemm_{fmt}_backward_tiled")),
                "missing tiled backward kernel for {fmt}"
            );
        }
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
