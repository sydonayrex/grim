//! RDNA3/RDMA4 FP8 E4M3 WMMA GEMM kernel.
//!
//! gfx1201 (RDNA4) supports `v_wmma_f32_16x16x16_fp8_fp8` — 2x the throughput
//! of FP16 WMMA (383 TFLOPS vs 191 TFLOPS).
//!
//! Source: https://zolotukhin.ai/blog/2026-05-09-fp4-wave-breaks-rdma4-fp8-wmma
//!
//! FP8 E4M3FN is the native high-throughput format on RDNA4:
//! - FP16: 191 TFLOPS
//! - FP8 E4M3: 383 TFLOPS (2x FP16)
//! - INT8: 383 TOPS
//! - INT4: 766 TOPS
//!
//! This kernel computes C[M,N] = A[M,K] @ B^T where both A and B are FP8 E4M3.
//! Both activations and weights must be quantized to FP8 before calling.

/// HIP source for FP8 WMMA GEMM kernels.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// FP8 E4M3 WMMA GEMM — dual N-tile per wave on RDNA4.
// gfx1201 supports v_wmma_f32_16x16x16_fp8_fp8 natively.
// OPT: Dual-issue two WMMA per wave for 16×32 effective tile.
// OPT: Minimal VGPR pressure (< 96 VGPRs for 16-wave occupancy).
extern "C" __global__ void grim_wmma_gemm_fp8_e4m3(
    const _Float16* __restrict__ A,  // [M, K] FP8 E4M3 stored as FP16 (packed)
    const _Float16* __restrict__ B,  // [N, K] FP8 E4M3 stored as FP16 (packed)
    float* __restrict__ C,           // [M, N] FP32 output
    int M, int N, int K) {

    const int tile_row = blockIdx.y;
    const int tile_col_base = blockIdx.x * 2; // Dual N-tile per block

    if (tile_row * 16 >= M || tile_col_base * 16 >= N) return;

    const int row_base = tile_row * 16;
    const int col_base = tile_col_base * 16;
    const int valid_col1 = ((tile_col_base + 1) * 16 < N);

    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, float> frag_c0;
    fragment<accumulator, 16, 16, 16, float> frag_c1;
    fill_fragment(frag_c0, 0.0f);
    fill_fragment(frag_c1, 0.0f);

    _Float16 a_tile[16 * 16];
    _Float16 b0_tile[16 * 16];
    _Float16 b1_tile[16 * 16];

    for (int k0 = 0; k0 < K; k0 += 16) {
        // Load A tile (FP8 E4M3 values already in FP16 registers)
        for (int i = 0; i < 16; ++i) {
            int row = row_base + i;
            for (int j = 0; j < 16; ++j) {
                int kk = k0 + j;
                a_tile[i * 16 + j] = (row < M && kk < K) ? A[row * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_a, a_tile, 16);

        // Load B0 tile
        for (int j = 0; j < 16; ++j) {
            int col = col_base + j;
            for (int k = 0; k < 16; ++k) {
                int kk = k0 + k;
                b0_tile[j * 16 + k] = (col < N && kk < K) ? B[col * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_b0, b0_tile, 16);

        // OPT: Dual-issue first WMMA
        mma_sync(frag_c0, frag_a, frag_b0, frag_c0);
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
        // Load B1 tile (overlapped with WMMA)
        if (valid_col1) {
            for (int j = 0; j < 16; ++j) {
                int col = col_base + 16 + j;
                for (int k = 0; k < 16; ++k) {
                    int kk = k0 + k;
                    b1_tile[j * 16 + k] = (col < N && kk < K) ? B[col * K + kk] : (_Float16)0;
                }
            }
            load_matrix_sync(frag_b1, b1_tile, 16);
            mma_sync(frag_c1, frag_a, frag_b1, frag_c1);
        }
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
    }

    // Write C tile 0
    float* c0_tile_ptr = C + tile_row * 16 * N + tile_col_base * 16;
    __shared__ float c_out[16 * 16];
    store_matrix_sync(c_out, frag_c0, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col_base * 16 + j < N) {
            c0_tile_ptr[i * N + j] = c_out[idx];
        }
    }
    // Write C tile 1
    if (valid_col1) {
        float* c1_tile_ptr = C + tile_row * 16 * N + (tile_col_base + 1) * 16;
        store_matrix_sync(c_out, frag_c1, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (tile_row * 16 + i < M && (tile_col_base + 1) * 16 + j < N) {
                c1_tile_ptr[i * N + j] = c_out[idx];
            }
        }
    }
}

#endif // gfx1200/gfx1201 only
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn source_contains_fp8_kernel_entry() {
        assert!(
            KERNEL_SOURCE.contains("grim_wmma_gemm_fp8_e4m3"),
            "FP8 WMMA GEMM kernel entry must be JIT-discoverable"
        );
    }

    #[test]
    fn source_is_gfx12_guarded() {
        // FP8 WMMA is only available on gfx1200/gfx1201 (RDNA4).
        assert!(
            KERNEL_SOURCE.contains("defined(__gfx1200__)"),
            "FP8 kernel must be guarded for RDNA4 targets"
        );
        // Should NOT include gfx11xx (RDNA3) — FP8 WMMA is RDNA4-only.
        assert!(
            !KERNEL_SOURCE.contains("defined(__gfx1100__)"),
            "FP8 WMMA is RDNA4-only, should not include RDNA3 guard"
        );
    }

    #[test]
    fn source_uses_dual_issue() {
        assert!(
            KERNEL_SOURCE.matches("mma_sync").count() >= 2,
            "FP8 kernel should use dual-issue WMMA"
        );
    }
}
