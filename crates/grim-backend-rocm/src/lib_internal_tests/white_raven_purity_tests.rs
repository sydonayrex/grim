//! WS-D: `wmma_fp8_gemm.rs` must contain the **WhiteRaven** format and nothing
//! else — FP8 E4M3 operands through `V_WMMA_F32_16X16X16_FP8_FP8`, with no FP16
//! path sharing the translation unit.
//!
//! # The defect
//!
//! The file's own doc header claimed *"gfx1201 supports
//! `v_wmma_f32_16x16x16_fp8_fp8` — 2x the throughput"*, but the kernel declared:
//!
//! ```cpp
//! fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
//! ```
//! on `_Float16*` inputs. It was an **FP16 WMMA with FP8 in the filename** —
//! the exact shape §0 of the plan forbids, and the reason the plan's step D1
//! exists. It was also dead code (its only launcher is `#[allow(dead_code)]`
//! with no callers) and its grid math over-launched 4x against the kernel's
//! `blockIdx.x * 2`.
//!
//! # Purity, not preference
//!
//! Merging an FP16 fallback into this file would make it a mixed kernel, which
//! is neither WhiteRaven nor anything else, and would make the §6 A/B measure a
//! composite instead of an instruction. The guard below enforces the split.
//!
//! ROCm 7.2.4's rocWMMA carries FP8 as `rocwmma::float8_t` (= `hip_fp8_e4m3`)
//! and lowers `fragment<..., float8_t, ...>` to
//! `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.

fn src() -> &'static str {
    crate::kernels::wmma_fp8_gemm::KERNEL_SOURCE
}

/// RED (D1): the kernel must use an FP8 element type, not `_Float16`.
#[test]
fn ws_d_white_raven_kernel_uses_fp8_fragments_not_fp16() {
    let s = src();
    assert!(
        s.contains("float8_t") || s.contains("hip_fp8_e4m3"),
        "the WhiteRaven kernel must declare FP8 fragments (rocwmma::float8_t \
         or hip_fp8_e4m3); it currently has no FP8 element type at all"
    );
    assert!(
        !s.contains("fragment<matrix_a, 16, 16, 16, _Float16"),
        "the WhiteRaven kernel still declares _Float16 A fragments — that is an \
         FP16 WMMA wearing the FP8 name, which is the defect WS-D exists to fix"
    );
    assert!(
        !s.contains("fragment<matrix_b, 16, 16, 16, _Float16"),
        "the WhiteRaven kernel still declares _Float16 B fragments"
    );
    // NB: must not test for a bare `float` prefix — `float8_t` starts with
    // `float`, so `fragment<matrix_a, 16, 16, 16, float8_t, ...>` matches a naive
    // substring search and the assertion rejects the very thing it wants.
    assert!(
        !s.contains("fragment<matrix_a, 16, 16, 16, float,")
            && !s.contains("fragment<matrix_a, 16, 16, 16, float>"),
        "FP32 fragments are not a WMMA A-operand type; expected an 8-bit type"
    );
}

/// The kernel's own doc header must not keep claiming things the body does not
/// do. The old header is why the defect survived review: it read correctly.
#[test]
fn ws_d_kernel_header_does_not_claim_unused_fp8_types() {
    let s = src();
    // FP32 accumulate is correct and expected; FP16 *fragments* are not.
    assert!(
        s.contains("accumulator, 16, 16, 16, float"),
        "FP8 WMMA still accumulates in F32 via an accumulator fragment — keep it"
    );
}

/// The existing gfx12-only guard is correct and must survive: there is no
/// pre-gfx12 FP8 WMMA, and adding an FP16 `#elif` here would break purity.
#[test]
fn ws_d_guard_stays_gfx12_only_with_no_fp16_fallback() {
    let s = src();
    assert!(
        s.contains("defined(__gfx1200__)") && s.contains("defined(__gfx1201__)"),
        "WhiteRaven must stay gated on gfx1200/gfx1201"
    );
    assert!(
        !s.contains("defined(__gfx1100__)"),
        "no RDNA3 FP8 WMMA exists; an FP16 fallback branch here would make \
         this a mixed kernel, not the WhiteRaven kernel"
    );
}
