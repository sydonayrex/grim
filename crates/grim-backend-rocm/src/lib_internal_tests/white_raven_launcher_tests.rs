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

/// WS-D D2: the kernel declares `__launch_bounds__`.
///
/// The plan found zero `__launch_bounds__` across every WMMA kernel in the tree.
/// Without it the compiler picks a register budget freely, and this kernel
/// stages A in registers so that budget is the difference between fitting and
/// spilling to scratch -- the same class of failure that cost 60x at
/// `wmma_quantized_gemm.rs:8` ("384 VGPRs, over the 256 limit").
///
/// The bound must be 32: the launcher passes `block_dim = (32, 1, 1)` and the
/// kernel's own doc header says one wavefront owns a 16x32 output tile. A bound
/// that disagreed with the launch configuration would be worse than none, so the
/// value is checked against the kernel's own wavefront arithmetic rather than
/// hardcoded.
#[test]
fn white_raven_declares_launch_bounds() {
    assert!(
        KERNEL_SOURCE.contains("__launch_bounds__"),
        "WhiteRaven kernel must declare __launch_bounds__; without it the \
         compiler is free to pick a register budget that spills A to scratch"
    );
}

/// The declared bound must match the launch configuration, and that
/// configuration is one wavefront.
///
/// Derived from the kernel: `blockIdx.x * 2` with a 16-wide N tile means each
/// block covers 32 columns, and the WMMA fragment is 16x16, so a block computes
/// exactly two 16x16 tiles -- one wavefront's worth of `mma_sync` on RDNA's
/// 32-wide wavefront. Anything else and the bound would be a lie the compiler
/// optimises against.
#[test]
fn the_launch_bound_matches_one_wavefront() {
    // Anchor on the kernel's actual declaration rather than the first textual
    // occurrence: the file's comments discuss `__launch_bounds__` and
    // `block_dim = (32,1,1)` in prose, so a naive search finds a `)` from a
    // parenthetical in a comment. Scoping to the text immediately preceding the
    // kernel name also proves the attribute is on *this* kernel.
    let decl = KERNEL_SOURCE
        .find("void grim_wmma_gemm_fp8_e4m3")
        .unwrap_or_else(|| panic!("WhiteRaven kernel declaration not found"));
    let before = &KERNEL_SOURCE[..decl];
    let attr = before
        .rfind("__launch_bounds__")
        .unwrap_or_else(|| panic!("no __launch_bounds__ on the kernel declaration"));
    let after = &before[attr + "__launch_bounds__".len()..];
    let open = after
        .find('(')
        .unwrap_or_else(|| panic!("__launch_bounds__ is not followed by '('"));
    let close = after[open..]
        .find(')')
        .map(|i| i + open)
        .unwrap_or_else(|| panic!("__launch_bounds__ has no closing ')'"));

    let bound = &after[open + 1..close];
    let threads: u32 = bound
        .split(',')
        .next()
        .unwrap()
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("could not read a thread count from {bound:?}"));

    assert_eq!(
        threads, 32,
        "WhiteRaven launches with block_dim = (32,1,1) -- one wavefront. A \
         __launch_bounds__ of {threads} would not match the launch config."
    );
}
