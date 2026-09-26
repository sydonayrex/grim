//! WS-C2: `W4A4OstQuant` (WhiteCrow, `V_DOT8_I32_IU4`) must be gated on the
//! architecture that actually implements it.
//!
//! # The defect
//!
//! `device/quant/mod.rs` validated only `k % 128` before dispatching to
//! `launch_w4a4_ostquant_gemv_blob` -> `grim_dot8_w4a4_gemv`. That kernel lives
//! inside `#if defined(__gfx1200__) || defined(__gfx1201__)`
//! (kernels/dot_gemv.rs), so on a gfx1100 card the preprocessor strips the
//! symbol and the launch fails at `hipModuleGetFunction` with status 500 —
//! exactly the `SPEED-ROC` cold-start failure documented in Cargo.toml. The
//! user-visible symptom is an opaque module-load error instead of "this GPU
//! cannot run W4A4".
//!
//! Sibling arms in the same `match` do gate (`self.gpu_target.starts_with("gfx12")`
//! on the FP8 arm), which is why this one reads as an oversight rather than a
//! policy.
//!
//! # Why a unit test
//!
//! The failure needs a non-gfx12 device to reproduce, and grim's GPU tests are
//! `#[ignore]`d. But the *decision* is pure: given an arch string, does this arm
//! admit or reject? Extracting it as a predicate makes that testable on CPU and
//! makes the gate readable at the call site.

use crate::quantization::{w4a4_ostquant_supported, GcnArch};

/// Whether the current device implements `V_DOT8_I32_IU4` in a form grim ships.
///
/// NOTE on scope: `V_DOT8_I32_IU4` is documented in the RDNA3 manual as well as
/// RDNA4 (rdna3.txt carries `V_DOT8_I32_IU4`; RDNA2 has the pre-rename
/// `V_DOT8_I32_I4`). So this is *grim's* narrower gate, not an ISA limit: the
/// W4A4 blob layout assumes dot8's 4x-per-instruction rate and the gfx12
/// throughput. Do not "fix" this by widening to gfx11 without measuring —
/// the gate is a deliberate choice and the comment says so.
/// RED (C2): every non-gfx12 arch is rejected with a message, not admitted.
#[test]
fn ws_c2_w4a4_is_rejected_off_gfx12() {
    for arch in [
        GcnArch::RDNA1,
        GcnArch::RDNA2,
        GcnArch::RDNA3,
        GcnArch::CDNA1,
        GcnArch::CDNA2,
        GcnArch::CDNA3,
        GcnArch::CDNA4,
        GcnArch::Other,
    ] {
        let err = w4a4_ostquant_supported(arch, 256)
            .expect_err("non-gfx12 arch must be rejected, not silently admitted");
        assert!(
            err.contains("gfx12"),
            "error for {arch:?} must name the required arch so the failure is \
             diagnosable: {err}"
        );
    }
}

/// The arch gate must be checked before the K gate: reporting "K must be
/// divisible by 128" to a gfx1100 user whose real problem is the GPU is a
/// misleading diagnostic.
#[test]
fn ws_c2_arch_gate_precedes_the_k_gate() {
    let err = w4a4_ostquant_supported(GcnArch::RDNA3, 100)
        .expect_err("must reject");
    assert!(
        err.contains("gfx12"),
        "on a non-gfx12 arch the arch error must win over the K error: {err}"
    );
}

/// gfx12 and gfx13 are admitted; the K constraint still applies.
#[test]
fn ws_c2_gfx12_admits_and_still_checks_k() {
    assert!(w4a4_ostquant_supported(GcnArch::RDNA4, 256).is_ok());
    assert!(w4a4_ostquant_supported(GcnArch::UDNA, 256).is_ok());
    let err = w4a4_ostquant_supported(GcnArch::RDNA4, 100).expect_err("K must be checked");
    assert!(err.contains("divisible by 128"), "{err}");
}

/// The dispatch arm must actually call the gate. Without this the predicate
/// above is decoration — the exact failure mode C2 is about: a correct helper
/// that nothing consults.
#[test]
fn ws_c2_dispatch_arm_consults_the_gate() {
    let src = include_str!("../device/quant/mod.rs");
    let arm = src
        .split("DTypeStorage::W4A4OstQuant(cfg) => {")
        .nth(1)
        .and_then(|rest| rest.split("DTypeStorage::").next())
        .expect("W4A4OstQuant arm not found in device/quant/mod.rs");
    assert!(
        arm.contains("w4a4_ostquant_supported"),
        "the W4A4OstQuant dispatch arm must consult w4a4_ostquant_supported \
         before launching; it currently only checks k % 128, so a gfx11 card \
         reaches a kernel the preprocessor stripped (WS-C2)."
    );
    assert!(
        !arm.contains("if k % 128 != 0"),
        "the bare k % 128 check should be replaced by the gate, not duplicated"
    );
}
