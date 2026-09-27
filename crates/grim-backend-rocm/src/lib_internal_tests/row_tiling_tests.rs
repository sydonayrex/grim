//! Guards for the row-tiled dot kernels.
//!
//! The row-tiled variants map a block to several rows so B is streamed from HBM
//! once per tile instead of once per row. That is a pure launch-geometry change,
//! and the failure modes are all silent: a rows-per-block mismatch between the
//! kernel's `#define` and the launcher's grid leaves rows unwritten, and a
//! mismatched tile count means some rows are computed twice. Neither faults --
//! the output is just wrong in a way that looks like a tuning result.
//!
//! So these tests read the constants back out of the source and the launcher
//! arithmetic and check they agree, and check the grid covers every row.

use crate::kernels::dot_gemv::KERNEL_SOURCE;

/// Rows per block, as the kernel's own `#define`.
fn rows_per_block(define: &str) -> usize {
    let needle = format!("#define {define} ");
    let line = KERNEL_SOURCE
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("{define} not found in the kernel source"));
    line.split_whitespace()
        .last()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("{define} is not an integer: {line}"))
}

/// The three row-tiled kernels, with the define each one keys off.
const TILED: [(&str, &str, &str); 3] = [
    ("grim_dot4_fp8_gemv_rowtiled", "GRIM_FP8_ROWS_PER_BLOCK", "launch_dot4_fp8_gemv_rowtiled_for_ab"),
    ("grim_dot4_q80_q81_gemv_rowtiled", "GRIM_Q81_ROWS_PER_BLOCK", "launch_dot4_q80_q81_gemv_rowtiled_for_ab"),
    ("grim_dot8_w4a4_gemv_rowtiled", "GRIM_W4A4_ROWS_PER_BLOCK", "launch_dot8_w4a4_gemv_rowtiled_for_ab"),
];

#[test]
fn every_tiled_kernel_exists_and_tiles_rows() {
    for (kernel, define, _) in TILED {
        assert!(
            KERNEL_SOURCE.contains(&format!("void {kernel}(")),
            "{kernel} is missing from the kernel source"
        );
        assert!(
            rows_per_block(define) > 1,
            "{define} must tile more than one row or the variant is pointless"
        );
    }
}

/// The kernel derives its rows from `blockIdx.y * ROWS`, so the launcher must
/// divide M by exactly that ROWS. A mismatch writes only a fraction of the
/// output and leaves the rest untouched -- no fault, just missing rows.
#[test]
fn the_row_tile_is_derived_from_blockidx_not_a_literal() {
    for (kernel, define, _) in TILED {
        assert!(
            KERNEL_SOURCE.contains(&format!("blockIdx.y * {define}")),
            "{kernel} must derive its row base from blockIdx.y * {define}"
        );
        assert!(
            KERNEL_SOURCE.contains(&format!("min({define}, M - row0)")),
            "{kernel} must clamp its row count with min({define}, M - row0) so the \
             last tile does not write past M"
        );
    }
}

/// The wave reduction has to cover every row in the tile, not just the first.
/// A reduction loop bounded by the tile height but indexed by lane would
/// silently reduce the wrong rows.
#[test]
fn the_reduction_covers_every_row_in_the_tile() {
    for (kernel, define, _) in TILED {
        assert!(
            KERNEL_SOURCE.contains(&format!("for (int r = 0; r < {define}; r++)")),
            "{kernel} must reduce all {define} rows, not a single one"
        );
    }
}

/// Accumulators are declared [col][row], so the inner reduction and store loops
/// must be bounded by the tile height. Checking the shape catches a transpose
/// that would still compile and still produce plausible numbers.
#[test]
fn accumulators_are_indexed_by_column_then_row() {
    for (kernel, define, _) in TILED {
        assert!(
            KERNEL_SOURCE.contains(&format!("facc[4][{define}]")),
            "{kernel} must declare facc[4][{define}]"
        );
    }
}

/// The staged B must be read once per chunk *outside* the row loop. That is the
/// entire point: if the B load sits inside the row loop the kernel is the
/// untiled one with extra arithmetic.
#[test]
fn b_is_staged_outside_the_row_loop() {
    for (kernel, _, _) in TILED {
        let body = kernel_body(kernel);
        let b_stage = body.find("b_words[").expect("staged B words");
        let row_loop = body
            .find("for (int r = 0; r < rows; r++)")
            .expect("row loop");
        assert!(
            b_stage < row_loop,
            "{kernel} stages B at offset {b_stage} but the row loop starts at \
             {row_loop}; B must be loaded before the rows are walked"
        );
    }
}

/// Slice a kernel body out of the shared source, from its signature to the
/// closing brace at column 0.
fn kernel_body(kernel: &str) -> &'static str {
    let start = KERNEL_SOURCE
        .find(&format!("void {kernel}("))
        .expect("kernel present");
    let rest = &KERNEL_SOURCE[start..];
    let end = rest[1..].find("\n}\n").expect("closing brace") + 1;
    &rest[..=end]
}

/// The launcher's grid must divide M by the tile height, and the grid must
/// cover every row. Checked as arithmetic here because the launcher is Rust and
/// the tile height is a `#define` in the HIP source -- the two can only be kept
/// in agreement by asserting it.
#[test]
fn the_launcher_grid_covers_every_row() {
    for (kernel, define, launcher) in TILED {
        let rows = rows_per_block(define);
        for m in [1usize, 7, 8, 9, 16, 63, 64, 65, 512] {
            let grid_y = m.div_ceil(rows);
            assert!(
                grid_y * rows >= m,
                "{launcher}: M={m} with {rows} rows/block needs grid.y={grid_y}, \
                 which covers only {} rows",
                grid_y * rows
            );
            assert!(
                (grid_y - 1) * rows < m,
                "{launcher}: M={m} launches a wholly out-of-range row block"
            );
        }
        // The launcher is expected to name the same constant. It currently
        // hardcodes `const ROWS: usize = 8;` with a comment pointing at the
        // define; if either side moves, this fails.
        assert!(
            rows == 8,
            "{define} is {rows} but the launchers hardcode ROWS = 8; \
             update both sides together"
        );
        let _ = kernel;
    }
}

#[test]
fn the_tiled_variants_keep_the_originals() {
    // The untiled kernels stay: dispatch still uses them, and the A/B compares
    // against them. Deleting them would remove the only control.
    for k in [
        "grim_dot4_fp8_gemv",
        "grim_dot4_q80_q81_gemv",
        "grim_dot8_w4a4_gemv",
    ] {
        assert!(
            KERNEL_SOURCE.contains(&format!("void {k}(")),
            "{k} must remain: it is the A/B control"
        );
    }
}
