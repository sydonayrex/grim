//! **WhiteRaven** — RDNA4 FP8 E4M3 WMMA GEMM (`V_WMMA_F32_16X16X16_FP8_FP8`).
//!
//! Computes `C[M,N] = A[M,K] @ B^T` with A and B both FP8 E4M3, accumulating in
//! FP32. One wavefront owns a 16x32 output tile (two 16x16 tiles) and stages A
//! once, reusing it across both `mma_sync` calls from registers.
//!
//! # This kernel is the WhiteRaven format and nothing else
//!
//! Per §0 of `plans/PLAN-corvid-precision.md`, a kernel carries a codename only
//! if it contains that format and no other. An earlier revision of this file
//! declared `_Float16` fragments on `_Float16*` inputs while its doc header
//! claimed the FP8 instruction — an FP16 WMMA wearing the FP8 name. It was dead
//! code (its only launcher is `#[allow(dead_code)]` with no callers), so nothing
//! caught it. There is deliberately **no FP16 fallback branch here**: RDNA3 has
//! no FP8 WMMA, so dispatch selects the FP16 WMMA kernels instead, and mixing the
//! two into one translation unit would make this a composite that the §6 A/B
//! could not interpret.
//!
//! # Instruction
//!
//! ROCm 7.2.4's rocWMMA carries FP8 as `rocwmma::float8_t` (= `hip_fp8_e4m3`)
//! and lowers `fragment<..., float8_t, ...>` to
//! `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`, verified by
//! disassembly on gfx1201. The `_w32` and `_gfx12` suffixes are both required —
//! the unsuffixed spelling is not a builtin.
//!
//! The "2x the throughput of FP16" figure is a vendor claim, not a grim
//! measurement. It is the subject of step D5, not an assumption here.

/// HIP source for the WhiteRaven FP8 E4M3 WMMA GEMM kernel.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// WhiteRaven: FP8 E4M3 x FP8 E4M3 -> FP32, 16x32 output tile per wavefront.
extern "C" __global__ void grim_wmma_gemm_fp8_e4m3(
    const float8_t* __restrict__ A,  // [M, K] FP8 E4M3, row-major
    const float8_t* __restrict__ B,  // [N, K] FP8 E4M3 (B^T is [K, N])
    float* __restrict__ C,           // [M, N] FP32, tightly packed
    int M, int N, int K) {

    const int tile_row = blockIdx.y;
    const int tile_col_base = blockIdx.x * 2;  // two 16-wide N tiles per block

    if (tile_row * 16 >= M || tile_col_base * 16 >= N) return;

    const int row_base = tile_row * 16;
    const int col_base = tile_col_base * 16;
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);

    fragment<matrix_a, 16, 16, 16, float8_t, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, float8_t, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, float8_t, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, float> frag_c0;
    fragment<accumulator, 16, 16, 16, float> frag_c1;
    fill_fragment(frag_c0, 0.0f);
    fill_fragment(frag_c1, 0.0f);

    __shared__ float c_out[16 * 16];

    for (int k0 = 0; k0 < K; k0 += 16) {
        // A is reused by both N tiles: one load serves both mma_sync calls.
        load_matrix_sync(frag_a, A + (long long)row_base * K + k0, K);
        load_matrix_sync(frag_b0, B + (long long)col_base * K + k0, K);
        mma_sync(frag_c0, frag_a, frag_b0, frag_c0);
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
        if (valid_col1) {
            load_matrix_sync(frag_b1, B + (long long)(col_base + 16) * K + k0, K);
            mma_sync(frag_c1, frag_a, frag_b1, frag_c1);
        }
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
    }

    float* c0_ptr = C + (long long)row_base * N + col_base;
    store_matrix_sync(c_out, frag_c0, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (row_base + i < M && col_base + j < N) c0_ptr[(long long)i * N + j] = c_out[idx];
    }

    if (valid_col1) {
        float* c1_ptr = C + (long long)row_base * N + (col_base + 16);
        store_matrix_sync(c_out, frag_c1, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (row_base + i < M && col_base + 16 + j < N)
                c1_ptr[(long long)i * N + j] = c_out[idx];
        }
    }
}
#endif // gfx1200/gfx1201 — deliberately no FP16 fallback; see module docs
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn source_contains_fp8_kernel_entry() {
        assert!(
            KERNEL_SOURCE.contains("grim_wmma_gemm_fp8_e4m3"),
            "WhiteRaven kernel entry must be JIT-discoverable"
        );
    }

    #[test]
    fn source_is_gfx12_guarded() {
        // FP8 WMMA is only available on gfx1200/gfx1201 (RDNA4).
        assert!(
            KERNEL_SOURCE.contains("defined(__gfx1200__)"),
            "WhiteRaven must be guarded for RDNA4 targets"
        );
        // Should NOT include gfx11xx (RDNA3) — FP8 WMMA is RDNA4-only, and an
        // FP16 fallback branch here would make this a mixed kernel.
        assert!(
            !KERNEL_SOURCE.contains("defined(__gfx1100__)"),
            "FP8 WMMA is RDNA4-only, should not include RDNA3 guard"
        );
    }

    #[test]
    fn source_uses_dual_issue() {
        assert!(
            KERNEL_SOURCE.matches("mma_sync").count() >= 2,
            "WhiteRaven should dual-issue: frag_a reused across two N tiles"
        );
    }

    #[test]
    fn source_uses_fp8_element_type() {
        assert!(
            KERNEL_SOURCE.contains("float8_t"),
            "fragments must be FP8 (rocwmma::float8_t), not _Float16"
        );
        assert!(
            !KERNEL_SOURCE.contains("_Float16"),
            "no _Float16 may remain in the WhiteRaven kernel"
        );
    }

    #[test]
    fn a_fragment_is_loaded_once_for_both_n_tiles() {
        // The win over the old kernel is that A is staged once and reused; two
        // mma_sync calls must share one A load.
        assert_eq!(
            KERNEL_SOURCE.matches("load_matrix_sync(frag_a").count(),
            1,
            "frag_a must be loaded once per k-step and reused across both N tiles"
        );
        assert_eq!(KERNEL_SOURCE.matches("mma_sync(frag_c").count(), 2);
    }
}
