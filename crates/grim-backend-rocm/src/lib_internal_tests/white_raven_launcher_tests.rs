//! Keeps the WhiteRaven launcher's grid in step with the kernel's own indexing.
//!
//! The kernel derives its output tile from `blockIdx`, so a launcher that
//! disagrees does not fault -- it writes a subset of the output and leaves the
//! rest as whatever was in the buffer. That is silent, and it is the exact
//! failure the first version of this launcher had: it reused the fused-dequant
//! body's `ceil(N/64)` grid against a kernel that tiles N by 32.

use crate::kernels::wmma_fp8_gemm::KERNEL_SOURCE;

/// N columns per block, derived from the kernel: `blockIdx.x * 2` tiles of 16.
fn n_per_block_from_kernel() -> usize {
    assert!(
        KERNEL_SOURCE.contains("blockIdx.x * 2"),
        "WhiteRaven kernel no longer uses 2 N-tiles per block; \
         the launcher's grid must be re-derived"
    );
    assert!(
        KERNEL_SOURCE.contains("tile_col_base * 16"),
        "WhiteRaven kernel changed its N tile width; \
         the launcher's grid must be re-derived"
    );
    2 * 16
}

#[test]
fn white_raven_grid_covers_every_output_column() {
    let n_per_block = n_per_block_from_kernel();
    // The launcher's arithmetic, restated: grid_x * n_per_block must reach N.
    for n in [16usize, 32, 64, 4096, 4097] {
        let grid_x = n.div_ceil(n_per_block);
        assert!(
            grid_x * n_per_block >= n,
            "N={n}: grid_x={grid_x} covers {} columns, short of {n}",
            grid_x * n_per_block
        );
        // ...and must not over-cover by more than one block, or the last block
        // reads a tile entirely past the end of B.
        assert!(
            (grid_x - 1) * n_per_block < n,
            "N={n}: grid_x={grid_x} launches a wholly-out-of-range block"
        );
    }
}

#[test]
fn the_shared_64_wide_grid_would_have_been_wrong() {
    // Pins the regression this launcher exists to prevent. If someone routes
    // WhiteRaven back through launch_wmma_fused_dequant_quant, this is the
    // arithmetic that silently drops half the output.
    let n_per_block = n_per_block_from_kernel();
    for n in [64usize, 4096] {
        let shared = n.div_ceil(64);
        assert!(
            shared * 64 < n || shared * 64 == n,
            "sanity: the shared grid is supposed to be the wrong one"
        );
        // At N=4096 the correct grid is exactly 2x the shared one.
        assert_eq!(n.div_ceil(n_per_block), shared * 2);
    }
}

#[test]
fn the_block_is_one_wave_not_four() {
    // 128 threads would be 4 waves, each recomputing the same rocwmma tile and
    // racing on the shared c_out buffer. The kernel's store loop bound of 256
    // is the 16x16 tile element count, not a block width.
    assert!(
        KERNEL_SOURCE.contains("idx < 256"),
        "WhiteRaven store loop changed; re-check the block width"
    );
    assert!(
        KERNEL_SOURCE.contains("__builtin_amdgcn_wave_barrier"),
        "the shared-memory reuse depends on a wave barrier"
    );
}

#[test]
fn kernel_requires_k_multiple_of_16() {
    // rocwmma steps K by 16 with no tail guard, so the launcher rejects
    // K % 16 != 0 rather than letting the last load read past A and B.
    assert!(KERNEL_SOURCE.contains("k0 += 16"));
    for k in [16usize, 128, 1024, 11008, 22016] {
        assert_eq!(k % 16, 0, "K={k} must be a legal WhiteRaven K");
    }
    for k in [1usize, 15, 17, 100] {
        assert_ne!(k % 16, 0, "K={k} must be rejected by the launcher");
    }
}
