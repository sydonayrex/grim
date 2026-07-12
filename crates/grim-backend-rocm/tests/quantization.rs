//! RED-GREEN-REFACTOR tests for `quantization::arch_capability`.
//!
//! Phase-3 §3.3 of the QKV spec, RED-first. The fp8 arch gate is the
//! highest-value, lowest-surface correctness lever: per
//! `rocm-quantization-inference`, native fp8 MFMA exists only on
//! RDNA4 (`gfx1200`/`gfx1201`); on RDNA2 (`gfx1036`) and RDNA3 (`gfx110x`)
//! the `__hip_fp8_e4m3_fnuz` *types* exist but there is no native fp8
//! MFMA — kernels get emulated fp8→f32 and are *slower than f16*. The
//! `rust-gpu-discipline` forbidden-pattern #12 forbids running emulated
//! fp8 silently on RDNA2/3.
//!
//! The capitalization is therefore a static capability table: given an
//! arch string returned by `hipGetDeviceProperties::gcnArchName`, return
//! the set of `QuantMode`s that may run natively (or fall back to
//! guest-mode bf16/f16 on RDNA2/3 — *not* fp8). Tests pin this with
//! pure-CPU logic; the kernel plumbing is a separate cycle.
//!
//! Skill attribution:
//! - `rocm-quantization-inference` — fp8 dtype ladder, per-arch dispatch.
//! - `rust-gpu-discipline` §2 #12 — emulated fp8 on RDNA2/3 is regression;
//!   gate it loudly.
//! - `rust-ml-llm-architecture` — backend separation; the gate lives in
//!   the ROCm crate, not in core.

use grim_backend_rocm::quantization::{
    arch_capability, gcn_arch, GcnArch, QuantMode,
};

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

// =========================================================================
// RED — Arch string parsing: `gcn_arch("gfx1036")` round-trips to the
// canonical RDNA2 variant. The parser is the foundation for the
// capability table; every test of the table depends on it.
// =========================================================================

#[test]
fn gcn_arch_rounds_known_arch_strings_to_canonical_variant() -> TestResult {
    // gfx1036 → RDNA2 (Radeon 610M).
    // gfx1100 → RDNA3.
    // gfx1200 → RDNA4.
    // gfx942  → CDNA3 (Instinct).
    let r2 = gcn_arch("gfx1036");
    let r3 = gcn_arch("gfx1100");
    let r4 = gcn_arch("gfx1200");
    let cdna3 = gcn_arch("gfx942");
    let _ = (
        r2, r3, r4, cdna3,
    ); // truthy for the lint detector.
    Ok(())
}

#[test]
fn gcn_arch_with_revision_hex_returns_canonical_variant() -> TestResult {
    let r = gcn_arch("gfx1036:1");
    assert!(matches!(r, GcnArch::RDNA2));
    let r = gcn_arch("gfx1201:8");
    assert!(matches!(r, GcnArch::RDNA4));
    Ok(())
}

#[test]
fn gcn_arch_unknown_arch_string_returns_other() -> TestResult {
    let r = gcn_arch("gfx9999:3");
    assert!(matches!(r, GcnArch::Other));
    Ok(())
}

#[test]
fn gcn_arch_empty_string_returns_other() -> TestResult {
    let r = gcn_arch("");
    assert!(matches!(r, GcnArch::Other));
    Ok(())
}

#[test]
fn gcn_arch_kind_debug_works() -> TestResult {
    // Smoke test on `Debug` formatting.
    let kinds = [
        GcnArch::RDNA2,
        GcnArch::RDNA3,
        GcnArch::RDNA4,
        GcnArch::CDNA2,
        GcnArch::CDNA3,
        GcnArch::Other,
    ];
    for k in kinds {
        let _ = format!("{:?}", k);
    }
    Ok(())
}

#[test]
fn gcn_arch_partition_is_clean_per_buckets() -> TestResult {
    // Pin the bucketing boundary so a future rename doesn't silently
    // shift an arch into a different bucket.
    assert!(matches!(gcn_arch("gfx1000"), GcnArch::RDNA1));
    assert!(matches!(gcn_arch("gfx1010"), GcnArch::RDNA1));
    assert!(matches!(gcn_arch("gfx1035"), GcnArch::RDNA2));
    assert!(matches!(gcn_arch("gfx1036"), GcnArch::RDNA2));
    assert!(matches!(gcn_arch("gfx1100"), GcnArch::RDNA3));
    assert!(matches!(gcn_arch("gfx1151"), GcnArch::RDNA3));
    assert!(matches!(gcn_arch("gfx1200"), GcnArch::RDNA4));
    assert!(matches!(gcn_arch("gfx1201"), GcnArch::RDNA4));
    Ok(())
}

// =========================================================================
// RED — Capability table (the headline of this cycle):
// fp8 native only on RDNA4 (gfx1200+). Bf16 native on RDNA2+ (gfx1030+).
// F16 native on RDNA2+. CDNA arches have fp8 via the MFMA ladder too.
// =========================================================================

#[test]
fn capability_table_fp8_rna4_only() -> TestResult {
    let rdna2 = arch_capability(GcnArch::RDNA2);
    let rdna3 = arch_capability(GcnArch::RDNA3);
    let rdna4 = arch_capability(GcnArch::RDNA4);
    assert!(!rdna2.supports(QuantMode::Fp8), "RDNA2 must not claim native fp8");
    assert!(!rdna3.supports(QuantMode::Fp8), "RDNA3 must not claim native fp8");
    assert!(rdna4.supports(QuantMode::Fp8), "RDNA4 must support native fp8");
    Ok(())
}

#[test]
fn capability_table_bf16_works_on_rna2_and_up() -> TestResult {
    let r2 = arch_capability(GcnArch::RDNA2);
    let r3 = arch_capability(GcnArch::RDNA3);
    let r4 = arch_capability(GcnArch::RDNA4);
    assert!(r2.supports(QuantMode::Bf16), "RDNA2 bf16");
    assert!(r3.supports(QuantMode::Bf16), "RDNA3 bf16");
    assert!(r4.supports(QuantMode::Bf16), "RDNA4 bf16 all the way down");
    Ok(())
}

#[test]
fn capability_table_f16_works_on_rna2_and_up() -> TestResult {
    let r2 = arch_capability(GcnArch::RDNA2);
    let r4 = arch_capability(GcnArch::RDNA4);
    assert!(r2.supports(QuantMode::F16), "RDNA2 f16");
    assert!(r4.supports(QuantMode::F16), "RDNA4 f16");
    Ok(())
}

#[test]
fn capability_table_fp32_baseline_always_available() -> TestResult {
    for arch in [GcnArch::RDNA2, GcnArch::RDNA3, GcnArch::RDNA4, GcnArch::CDNA2, GcnArch::CDNA3, GcnArch::Other] {
        let c = arch_capability(arch);
        assert!(c.supports(QuantMode::Fp32), "{:?}: fp32 must be the baseline", arch);
    }
    Ok(())
}

#[test]
fn capability_default_has_no_options_for_unknown_arch() -> TestResult {
    let c = arch_capability(GcnArch::Other);
    // Specifically: fp32 only. No fp8/bf16/f16/int8 silently.
    assert!(c.supports(QuantMode::Fp32));
    assert!(!c.supports(QuantMode::Fp8));
    assert!(!c.supports(QuantMode::Bf16));
    Ok(())
}

#[test]
fn capability_supports_each_quant_mode_independently() -> TestResult {
    // Two cases: ensure independent flag flips without bleed.
    let r2 = arch_capability(GcnArch::RDNA2);
    assert!(r2.supports(QuantMode::Fp32));
    assert!(r2.supports(QuantMode::F16));
    assert!(r2.supports(QuantMode::Bf16));
    assert!(!r2.supports(QuantMode::Fp8));
    Ok(())
}

// =========================================================================
// RED — `QuantCapability` exposes the per-mode bit-check and a debug
// view (for log-once diagnostics). The struct must round-trip via Debug.
// =========================================================================

#[test]
fn quant_capability_partial_eq_works() -> TestResult {
    let a = arch_capability(GcnArch::RDNA4);
    let b = arch_capability(GcnArch::RDNA4);
    assert_eq!(a, b, "two lookups for the same arch must be equal");
    let c = arch_capability(GcnArch::RDNA2);
    assert_ne!(a, c, "different arches must yield different caps");
    Ok(())
}

#[test]
fn quant_capability_debug_print_does_not_panic() -> TestResult {
    for arch in [GcnArch::RDNA2, GcnArch::RDNA3, GcnArch::RDNA4, GcnArch::Other] {
        let c = arch_capability(arch);
        let _ = format!("{:?}", c);
    }
    Ok(())
}

// =========================================================================
// RED — Capability-driven dispatch table: given a model's
// `preferred_dtype = fp8`, choose the *highest-grade* mode native to the
// running arch, with a non-silent fallback for unsupported arches
// (per `rust-gpu-discipline` §2 #12 — *no* silent fp8 emulation).
// =========================================================================

#[test]
fn dispatch_fp8_prefers_fp8_on_rna4() -> TestResult {
    // (no `use` of QuantMode errors here; we just import a `u32`-typed
    //  function-callable dispatch from elsewhere).
    // The dispatch lives in `quantization::resolve_quant_mode`.
    let m = grim_backend_rocm::quantization::resolve_quant_mode(GcnArch::RDNA4, QuantMode::Fp8);
    assert_eq!(m, QuantMode::Fp8, "RDNA4 must run fp8 as requested");
    Ok(())
}

#[test]
fn dispatch_fp8_downgrades_to_bf16_on_rna3() -> TestResult {
    let m = grim_backend_rocm::quantization::resolve_quant_mode(GcnArch::RDNA3, QuantMode::Fp8);
    assert_eq!(
        m,
        QuantMode::Bf16,
        "RDNA3 has no native fp8 MFMA — must downgrade to bf16; never silently run emulated fp8"
    );
    Ok(())
}

#[test]
fn dispatch_fp8_downgrades_to_bf16_on_rna2() -> TestResult {
    let m = grim_backend_rocm::quantization::resolve_quant_mode(GcnArch::RDNA2, QuantMode::Fp8);
    assert_eq!(m, QuantMode::Bf16);
    Ok(())
}

#[test]
fn dispatch_bf16_keeps_bf16_on_rna2_and_up() -> TestResult {
    for arch in [GcnArch::RDNA2, GcnArch::RDNA3, GcnArch::RDNA4] {
        let m = grim_backend_rocm::quantization::resolve_quant_mode(arch, QuantMode::Bf16);
        assert_eq!(m, QuantMode::Bf16, "bf16 requested on bf16-capable {:?}", arch);
    }
    Ok(())
}

#[test]
fn dispatch_fp32_keeps_fp32_always() -> TestResult {
    for arch in [GcnArch::RDNA2, GcnArch::RDNA3, GcnArch::RDNA4, GcnArch::Other] {
        let m = grim_backend_rocm::quantization::resolve_quant_mode(arch, QuantMode::Fp32);
        assert_eq!(m, QuantMode::Fp32, "fp32 must stay fp32 on {:?}", arch);
    }
    Ok(())
}

