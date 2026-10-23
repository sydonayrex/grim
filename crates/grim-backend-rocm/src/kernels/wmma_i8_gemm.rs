//! res0-T1-3: INT8 WMMA probe (RDNA4, rocwmma int8 fragments).
//!
//! Why: grim's prefill dequantizes K-weights to f16 and runs f16 WMMA; RDNA4's
//! I8 matrix-core rate is 2048 FLOP/clk/CU = 2x the FP16 rate (GPUOpen matrix
//! cores table), and INTEGER accumulation is exactly associative, so any
//! K-split order gives the same i32 result — the bit-identity gate is free.
//!
//! Skeleton copied from `wmma_fp8_gemm.rs` (the proven rocwmma-on-gfx12 path:
//! `fragment`/`load_matrix_sync`/`mma_sync`, two 16-wide N tiles per block).
//! rocwmma on gfx12 rejects the signed x unsigned mix (the SOA_INT register
//! layouts of int8_t and uint8_t differ — measured: static-assert
//! "Input fragment register layouts do not match"), so BOTH operands are
//! int8_t. Grim's unsigned weight codes map to signed at dequant-once time
//! via (code - 8) with the -8*sum(a) correction folded into the per-256-block
//! scale epilogue — the same two-dot decomposition the decode dot4 kernels
//! already use for Q2_K's dmin term.
//!
//! Gate: D (i32) must be BIT-IDENTICAL to the host reference — enforced by
//! tests/wmma_i8_probe.rs. NOT wired into the dispatch; probe launcher only.

pub const WMMA_I8_KERNEL_SOURCE: &str = r#"
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// INT8 x UINT8 -> INT32, 16x32 output tile per wavefront (two 16-wide N
// tiles), same structure as grim_wmma_gemm_fp8_e4m3.
//   A [M, K] int8, row-major (Q8_1-style activation codes)
//   B [N, K] uint8, row-major (unsigned weight codes; B^T is the GEMM's B)
//   D [M, N] int32, tightly packed
extern "C" __global__ __launch_bounds__(32) void grim_wmma_gemm_iu8(
    const int8_t* __restrict__ A,   // [M, K] row-major
    const int8_t* __restrict__ B,   // [N, K] row-major (both operands signed:
    int* __restrict__ D,            //  rocwmma gfx12 rejects int8 x uint8)
    int M, int N, int K) {

    const int tile_row = blockIdx.y;
    const int tile_col_base = blockIdx.x * 2;

    if (tile_row * 16 >= M || tile_col_base * 16 >= N) return;

    const int row_base = tile_row * 16;
    const int col_base = tile_col_base * 16;
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, int32_t> frag_c0;
    fragment<accumulator, 16, 16, 16, int32_t> frag_c1;
    fill_fragment(frag_c0, 0);
    fill_fragment(frag_c1, 0);

    __shared__ int d_out[16 * 16];

    for (int k0 = 0; k0 < K; k0 += 16) {
        load_matrix_sync(frag_a, A + (long long)row_base * K + k0, K);
        load_matrix_sync(frag_b0, B + (long long)col_base * K + k0, K);
        mma_sync(frag_c0, frag_a, frag_b0, frag_c0);
        if (valid_col1) {
            load_matrix_sync(frag_b1, B + (long long)(col_base + 16) * K + k0, K);
            mma_sync(frag_c1, frag_a, frag_b1, frag_c1);
        }
    }

    int* d0_ptr = D + (long long)row_base * N + col_base;
    store_matrix_sync(d_out, frag_c0, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (row_base + i < M && col_base + j < N) d0_ptr[(long long)i * N + j] = d_out[idx];
    }

    if (valid_col1) {
        int* d1_ptr = D + (long long)row_base * N + (col_base + 16);
        store_matrix_sync(d_out, frag_c1, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (row_base + i < M && col_base + 16 + j < N)
                d1_ptr[(long long)i * N + j] = d_out[idx];
        }
    }
}

// MTILE variant: one block computes M_TILES(4) x 16 = 64 rows x 32 cols,
// loading each B fragment ONCE per k-step and reusing it across all 4 A
// fragments — the transpose of the fp8 kernel's A-reuse. The naive tile's
// measured failure was B re-read traffic ((M/16) x N x K bytes); this
// divides it by M_TILES.
extern "C" __global__ __launch_bounds__(32) void grim_wmma_gemm_iu8_mtile(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int* __restrict__ D,
    int M, int N, int K) {

    const int MT = 4;
    const int tile_m_base = blockIdx.y * (MT * 16);
    const int tile_col_base = blockIdx.x * 2;

    if (tile_m_base >= M || tile_col_base * 16 >= N) return;

    const int col_base = tile_col_base * 16;
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> frag_a[MT];
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, int32_t> frag_c0[MT];
    fragment<accumulator, 16, 16, 16, int32_t> frag_c1[MT];
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        fill_fragment(frag_c0[t], 0);
        fill_fragment(frag_c1[t], 0);
    }

    __shared__ int d_out[16 * 16];

    for (int k0 = 0; k0 < K; k0 += 16) {
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int row_base = tile_m_base + t * 16;
            if (row_base < M) load_matrix_sync(frag_a[t], A + (long long)row_base * K + k0, K);
        }
        load_matrix_sync(frag_b0, B + (long long)col_base * K + k0, K);
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int row_base = tile_m_base + t * 16;
            if (row_base < M) mma_sync(frag_c0[t], frag_a[t], frag_b0, frag_c0[t]);
        }
        if (valid_col1) {
            load_matrix_sync(frag_b1, B + (long long)(col_base + 16) * K + k0, K);
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                const int row_base = tile_m_base + t * 16;
                if (row_base < M) mma_sync(frag_c1[t], frag_a[t], frag_b1, frag_c1[t]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int row_base = tile_m_base + t * 16;
        if (row_base >= M) continue;
        int* d0_ptr = D + (long long)row_base * N + col_base;
        store_matrix_sync(d_out, frag_c0[t], 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (col_base + j < N) d0_ptr[(long long)i * N + j] = d_out[idx];
        }
        if (valid_col1) {
            int* d1_ptr = D + (long long)row_base * N + (col_base + 16);
            store_matrix_sync(d_out, frag_c1[t], 16, layout_t::mem_row_major);
            __builtin_amdgcn_wave_barrier();
            for (int idx = threadIdx.x; idx < 256; idx += 32) {
                int i = idx / 16;
                int j = idx % 16;
                if (col_base + 16 + j < N) d1_ptr[(long long)i * N + j] = d_out[idx];
            }
        }
    }
}
#endif
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn wmma_i8_source_contains_entries() {
        assert!(super::WMMA_I8_KERNEL_SOURCE.contains("grim_wmma_gemm_iu8"));
        assert!(super::WMMA_I8_KERNEL_SOURCE.contains("rocwmma"));
    }
}
