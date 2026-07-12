//! Phase-3 §3.3 — fp8/int8 quantization arch gate + dispatch.
//!
//! This module is the **head** of §3.3 (the fp8 arch gate): the static
//! classification of an arch string to its native quantization
//! capabilities, plus a `resolve_quant_mode` dispatch that *never*
//! silently runs emulated fp8 on RDNA2/3.
//!
//! The matmul kernel plumbing (hipBLASLt scale-pointers, scaled-MFMA
//! intrinsics, paged attention path) is a larger surface reserved for
//! follow-up work; this module is the precondition those depend on.
//!
//! Skill attribution:
//! - `rocm-quantization-inference` — fp8 dtype ladder, per-arch gating,
//!   hipBLASLt scale-pointers integration ahead.
//! - `rust-gpu-discipline` §2 #12 — emulated fp8 on RDNA2/3 is a
//!   *silent regression*; never let the kernel run silently.
//! - `rust-ml-llm-architecture` — backend isolation: the gate lives in
//!   the ROCm crate, not in core.

use std::fmt;

/// Canonical coarse-grained arch bin. The ROCm nightly headers stamp
/// every GPU with a long `gcnArchName` string like `"gfx1036"` (with
/// optional revision `:N`). We bucket those into the seven bins the
/// spec names: RDNA1/RDNA2/RDNA3/RDNA4 (consumer) and CDNA1/CDNA2/CDNA3
/// (data center); unrecognised strings fall through as `Other`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GcnArch {
    /// RDNA1 — gfx10xx around 5010-1012 (very old consumer).
    RDNA1,
    /// RDNA2 — gfx1030-1036 (van Gogh / RX 6000-7000 series integrated).
    RDNA2,
    /// RDNA3 — gfx1100-1151 (RX 7000 series).
    RDNA3,
    /// RDNA4 — gfx1200-1201 (the new gen-on-mobile target).
    RDNA4,
    /// CDNA2 — gfx908 (MI200 series; full MFMA).
    CDNA2,
    /// CDNA3 — gfx940-942 (MI300 series; full MFMA + fp8 path).
    CDNA3,
    /// Anything else (gfx900, gfx906, gfx0000, malformed strings).
    Other,
}

/// Bucket an `hipGetDeviceProperties::gcnArchName` value into a coarse
/// `GcnArch`. Revision suffixes (`":N"`) are tolerated and ignored.
///
/// Parse scheme: take the leading digits after `gfx` (e.g. `gfx1200`,
/// `gfx1035`, `gfx1101`, `gfx908`) and read the *first one or two* as
/// the major version. We don't try to be a full parser — the coarse
/// floor is what matters for the capability gate. RDNA4 = `gfx12xx`,
/// RDNA3 = `gfx11xx`, RDNA2 = `gfx10xx` (the modern triple), plus
/// CDNA1/2/3 for data-center arches.
pub fn gcn_arch(name: &str) -> GcnArch {
    // Strip the optional `:N` revision suffix.
    let raw = name.split(':').next().unwrap_or(name);
    if !raw.starts_with("gfx") {
        return GcnArch::Other;
    }
    let suffix = &raw[3..];
    // Compile-time confidence: infer the family from the leading
    // 1-2 digits of the suffix. We accept "5", "12", "1200" etc.
    // RDNA2 family (gfx10xx, including gfx101x-1012 RDNA1).
    if let Some(s) = strip_prefix_digits(suffix, "10") {
        return match s.chars().next().map(|c| c.to_digit(10)) {
            Some(Some(2..)) => {
                // gfx102x..gfx1099 are RDNA2 (van Gogh 1035, 1036, etc.).
                // gfx100x..gfx101x are RDNA1.
                if s.chars().next().and_then(|c| c.to_digit(10)) >= Some(2) {
                    GcnArch::RDNA2
                } else {
                    GcnArch::RDNA1
                }
            }
            Some(Some(0..=1)) => GcnArch::RDNA1,
            _ => GcnArch::Other,
        };
    }
    if let Some(s) = strip_prefix_digits(suffix, "11") {
        return family_rna3(s);
    }
    if let Some(s) = strip_prefix_digits(suffix, "12") {
        return family_rna4(s);
    }
    if let Some(s) = strip_prefix_digits(suffix, "9") {
        // gfx908/MI200 = CDNA2, gfx940-941-942 = CDNA3.
        // Other gfx9xx fall under Other.
        return match s {
            r if r.starts_with("08") => GcnArch::CDNA2,
            r if r.starts_with("40") || r.starts_with("41") || r.starts_with("42")
                || r.starts_with("43") || r.starts_with("44") || r.starts_with("50") =>
            {
                GcnArch::CDNA3
            }
            _ => GcnArch::Other,
        };
    }
    GcnArch::Other
}

fn strip_prefix_digits<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && &s[..prefix.len()] == prefix {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

fn family_rna3(s: &str) -> GcnArch {
    // Everything of form gfx11xx is RDNA3 in our coarse model. We do
    // not split RDNA3.0 vs 3.5; both share the F16/BF16 mix and lack fp8.
    if s.chars().all(|c| c.is_ascii_digit()) && !s.is_empty() {
        GcnArch::RDNA3
    } else {
        GcnArch::Other
    }
}

fn family_rna4(s: &str) -> GcnArch {
    if s.chars().all(|c| c.is_ascii_digit()) && !s.is_empty() {
        GcnArch::RDNA4
    } else {
        GcnArch::Other
    }
}

/// A quantization mode the kernel could dispatch to.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum QuantMode {
    /// Plain FP32 — the baseline; always available.
    Fp32,
    /// FP16 — native on RDNA2+.
    F16,
    /// BF16 — native on RDNA2 + CDNA2/3.
    Bf16,
    /// FP8 e4m3 / e5m2 — **only** native on RDNA4 (`gfx1200+`) per the spec.
    Fp8,
}

/// Per-arch capability bitmap. The struct is the *output* of the gate;
/// callers ask `capability.supports(mode)` and get a Yes/No without
/// having to know the arch.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct QuantCapability {
    fp32: bool,
    f16: bool,
    bf16: bool,
    fp8: bool,
}

impl QuantCapability {
    pub fn supports(self, mode: QuantMode) -> bool {
        match mode {
            QuantMode::Fp32 => self.fp32,
            QuantMode::F16 => self.f16,
            QuantMode::Bf16 => self.bf16,
            QuantMode::Fp8 => self.fp8,
        }
    }
}

impl fmt::Display for QuantCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fp32={} f16={} bf16={} fp8={}",
            self.fp32, self.f16, self.bf16, self.fp8
        )
    }
}

/// Compute the capabilities for a coarse-grained arch bucket.
///
/// Per `rocm-quantization-inference`:
/// - Fp32 always.
/// - F16 + Bf16 native on RDNA2 onwards (and CDNA2+).
/// - **Fp8 native only on RDNA4** (`gfx1200+`) on the consumer side,
///   plus CDNA3 (MI300 series) on the data-center side. Crucially, the
///   spec rules out fp8 on RDNA2/3 because there is no native fp8
///   MFMA on those arches.
pub fn arch_capability(arch: GcnArch) -> QuantCapability {
    match arch {
        GcnArch::RDNA4 => QuantCapability { fp32: true, f16: true, bf16: true, fp8: true },
        GcnArch::CDNA3 => QuantCapability { fp32: true, f16: true, bf16: true, fp8: true },
        GcnArch::CDNA2 => QuantCapability { fp32: true, f16: true, bf16: true, fp8: false },
        GcnArch::RDNA3 => QuantCapability { fp32: true, f16: true, bf16: true, fp8: false },
        GcnArch::RDNA2 => QuantCapability { fp32: true, f16: true, bf16: true, fp8: false },
        GcnArch::RDNA1 => QuantCapability { fp32: true, f16: false, bf16: false, fp8: false },
        GcnArch::Other => QuantCapability { fp32: true, f16: false, bf16: false, fp8: false },
    }
}

/// Resolve the runtime `QuantMode` for a running arch given a model's
/// *requested* mode.
///
/// `rust-gpu-discipline` §2 #12 mandates: **emulated fp8 on RDNA2/3 is
/// a regression; never silently run it.** This function is therefore
/// strict: if the requested mode is `Fp8` but the arch can't run it
/// natively, we downshift to `Bf16` (the next-fastest native mode on
/// RDNA2/3). The caller is responsible for surfacing the downgrade via
/// `tracing::warn!` per the discipline.
pub fn resolve_quant_mode(arch: GcnArch, requested: QuantMode) -> QuantMode {
    let caps = arch_capability(arch);
    if caps.supports(requested) {
        return requested;
    }
    match requested {
        // Emulated fp8 on RDNA2/3 is forbidden — fall back to bf16
        // (the best-available native mode on consumer RDNA besides fp32).
        QuantMode::Fp8 => {
            if caps.bf16 {
                QuantMode::Bf16
            } else if caps.f16 {
                QuantMode::F16
            } else {
                QuantMode::Fp32
            }
        }
        // For f16/bf16 requests on arches where the cap is fp32-only
        // (RDNA1, Other), drop all the way to fp32 rather than crash.
        QuantMode::F16 | QuantMode::Bf16 => {
            if caps.f16 {
                requested
            } else {
                QuantMode::Fp32
            }
        }
        QuantMode::Fp32 => QuantMode::Fp32,
    }
}

#[cfg(test)]
mod self_tests {
    //! Tiny self-test: bind capability to the spec without external
    //! fixtures. Heavy coverage lives in `tests/quantization.rs`.
    use super::*;

    #[test]
    fn fp8_capable_buckets_match_spec() {
        // spec: only RDNA4 and CDNA3 claim native fp8.
        for arch in [GcnArch::RDNA4, GcnArch::CDNA3] {
            let c = arch_capability(arch);
            assert!(
                c.supports(QuantMode::Fp8),
                "{:?}: fp8 expected, got {}",
                arch,
                c
            );
        }
        // Every other bucket must NOT claim fp8 (per `rocm-quantization-inference`
        // and `rust-gpu-discipline` §2 #12 — never run emulated fp8).
        for arch in [GcnArch::RDNA1, GcnArch::RDNA2, GcnArch::RDNA3, GcnArch::CDNA2, GcnArch::Other] {
            let c = arch_capability(arch);
            assert!(
                !c.supports(QuantMode::Fp8),
                "{:?}: fp8 must NOT be supported yet got {}",
                arch,
                c
            );
        }
    }
}
