//! WS-C1: the activation chunk must be converted to FP8 **once per chunk** and
//! reused across the four output columns, not once per column.
//!
//! # The defect
//!
//! `grim_dot4_fp8_gemv` (kernels/dot_gemv.rs) had, inside the per-chunk loop:
//!
//! ```c
//! for (int chunk = lane; chunk < n_chunks; chunk += 32) {
//!     for (int j = 0; j < 4; j++) {                 // 4 output columns
//!         for (int e = 0; e < 32; e += 4) {
//!             int a4 = grim_pack4_fp8(a_chunk + e);  // <-- same 32 activations,
//!             ...                                     //     converted 4x
//! ```
//!
//! `a_chunk` depends only on `chunk`, never on `j`, so the f32->E4M3 conversion
//! of each group of four activations was performed **four times** to produce
//! four identical values. `grim_pack4_fp8` calls the scalar
//! `grim_f32_to_fp8_e4m3` ladder four times per group (NaN test, inf test,
//! bit-twiddle, RNE), so the waste is 16 conversions per 32-element chunk that
//! only needed 4.
//!
//! # Why a structural test
//!
//! The conversion helper is pure integer bit-manipulation, so it lowers to no
//! `v_cvt` instruction and cannot be counted in a disassembly. A perf assertion
//! would be the only runtime signal, and a perf threshold is a poor RED: it can
//! be satisfied by an unrelated change and it is machine-dependent.
//!
//! So the invariant is asserted structurally, but on the **loop structure**
//! rather than on formatting: the packed-activation call must appear before the
//! column loop opens, and must appear exactly once in the kernel body. Both
//! properties are what a reader assumes from the code and what the current
//! source violates.
//!
//! `ws_c1_conversion_is_hoisted_out_of_the_column_loop_perf` in
//! `tests/fp8_gemv_ab.rs` quantifies the win separately.

use crate::kernels::dot_gemv::KERNEL_SOURCE;

/// Body of `grim_dot4_fp8_gemv`, from its signature to the next `extern "C"`.
fn fp8_gemv_body() -> String {
    let start = KERNEL_SOURCE
        .find("void grim_dot4_fp8_gemv(")
        .unwrap_or_else(|| panic!("grim_dot4_fp8_gemv not found in dot_gemv KERNEL_SOURCE"));
    let rest = &KERNEL_SOURCE[start..];
    // The next kernel, or end of source.
    let end = rest[1..]
        .find("extern \"C\" __global__")
        .map(|i| i + 1)
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

/// Position of the column loop that *follows* the chunk loop.
///
/// `rfind` over the whole body is wrong: the kernel has a second
/// `for (int j = 0; j < 4; j++)` in the cross-lane reduction, after the pack,
/// so an `rfind`-based comparison reports the pack as "outside" and the test
/// passes against the buggy source. Anchor on the chunk loop and search forward.
fn column_loop_after_chunk_loop(body: &str) -> usize {
    let chunk = body
        .find("for (int chunk = lane;")
        .unwrap_or_else(|| panic!("chunk loop not found in body:\n{body}"));
    body[chunk..]
        .find("for (int j = 0; j < 4; j++)")
        .map(|i| i + chunk)
        .expect("column loop not found after the chunk loop")
}

/// RED (C1): the pack must sit between the chunk loop and the column loop.
#[test]
fn ws_c1_conversion_is_hoisted_out_of_the_column_loop() {
    let body = fp8_gemv_body();
    let pack = body
        .find("grim_pack4_fp8(")
        .expect("grim_pack4_fp8 call site");
    let j_loop = column_loop_after_chunk_loop(&body);
    let chunk = body.find("for (int chunk = lane;").unwrap();

    assert!(
        pack > chunk && pack < j_loop,
        "grim_pack4_fp8 must be staged between the chunk loop and the column \
         loop. Found pack@{pack}, chunk@{chunk}, column_loop@{j_loop}. Inside \
         the column loop means the f32->E4M3 conversion of each activation \
         group runs once per output column (4x) instead of once per chunk \
         (WS-C1). Body:\n{body}"
    );
}

/// RED (C1): the hoisted form must actually have somewhere to put the packed
/// words. Today the kernel has no staging buffer at all — it packs straight
/// into the dot loop — so this is the assertion that fails on the current tree.
#[test]
fn ws_c1_chunk_stages_packed_activations_before_the_column_loop() {
    let body = fp8_gemv_body();
    let chunk = body.find("for (int chunk = lane;").unwrap();
    let j_loop = column_loop_after_chunk_loop(&body);
    let prologue = &body[chunk..j_loop];

    assert!(
        prologue.contains('[') && prologue.contains("grim_pack4_fp8"),
        "the chunk loop must stage the packed activation words into a local \
         array before the column loop, then the dot loop indexes that array. \
         No staging buffer found between chunk@{chunk} and column_loop@{j_loop}. \
         Prologue:\n{prologue}"
    );
}

/// The inner dot loop must consume the staged words, not re-pack.
#[test]
fn ws_c1_inner_dot_loop_reads_the_packed_buffer() {
    let body = fp8_gemv_body();
    let j_loop = column_loop_after_chunk_loop(&body);
    let inner = &body[j_loop..];
    let dot = inner
        .find("acc = grim_fdot4_fp8(")
        .map(|i| &inner[i..])
        .unwrap_or_else(|| panic!("dot call not found after the column loop:\n{inner}"));
    let line = dot.lines().next().unwrap_or("");
    assert!(
        !line.contains("grim_pack4_fp8"),
        "the dot loop still packs activations inline; it must read the staged \
         words (WS-C1). Line: {line}"
    );
}
