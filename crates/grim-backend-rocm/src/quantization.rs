//! Phase-3 §3.3 — fp8/int8 quantization arch gate + dispatch. [see: `resolve_quant_mode`, `rocm-quantization-inference`]

use std::fmt;

/// Canonical coarse-grained arch bin. The ROCm nightly headers stamp [see: `gcnArchName`, `"gfx1036"`, `:N`, `Other`]
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

/// Bucket an `hipGetDeviceProperties::gcnArchName` value into a coarse [see: `GcnArch`, `":N"`, `gfx`, `gfx1200`]
pub fn gcn_arch(name: &str) -> GcnArch {
    // Strip the optional `:N` revision suffix.
    let raw = name.split(':').next().unwrap_or(name);
    if !raw.starts_with("gfx") {
        return GcnArch::Other;
    }
    let suffix = &raw[3..];
    // Compile-time confidence: infer the family from the leading
    if let Some(s) = strip_prefix_digits(suffix, "10") {
        return match s.chars().next().map(|c| c.to_digit(10)) {
            Some(Some(2..)) => {
                // gfx102x..gfx1099 are RDNA2 (van Gogh 1035, 1036, etc.).
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
        return match s {
            r if r.starts_with("08") => GcnArch::CDNA2,
            r if r.starts_with("40")
                || r.starts_with("41")
                || r.starts_with("42")
                || r.starts_with("43")
                || r.starts_with("44")
                || r.starts_with("50") =>
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
    /// FP8 e4m3 / e5m2 native MFMA — **only** native on RDNA4 (`gfx1200+`) and CDNA3 per the spec.
    Fp8Native,
    /// Rook: MXFP4 E2M1 emulated (dequant in LDS to BF16, WMMA BF16 GEMM). Safe RDNA2+.
    MxFp4Emulated,
    /// Jackdaw: MXFP8 E4M3 emulated (dequant in LDS to BF16, WMMA BF16 GEMM). Safe RDNA2+.
    MxFp8Emulated,
}

/// Per-arch capability bitmap. The struct is the *output* of the gate; [see: `capability.supports(mode)`]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct QuantCapability {
    fp32: bool,
    f16: bool,
    bf16: bool,
    fp8_native: bool,
    mxfp4_emulated: bool,
    mxfp8_emulated: bool,
}

impl QuantCapability {
    pub fn supports(self, mode: QuantMode) -> bool {
        match mode {
            QuantMode::Fp32 => self.fp32,
            QuantMode::F16 => self.f16,
            QuantMode::Bf16 => self.bf16,
            QuantMode::Fp8Native => self.fp8_native,
            QuantMode::MxFp4Emulated => self.mxfp4_emulated,
            QuantMode::MxFp8Emulated => self.mxfp8_emulated,
        }
    }
}

impl fmt::Display for QuantCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fp32={} f16={} bf16={} fp8_native={} mxfp4_emulated={} mxfp8_emulated={}",
            self.fp32,
            self.f16,
            self.bf16,
            self.fp8_native,
            self.mxfp4_emulated,
            self.mxfp8_emulated
        )
    }
}

/// Compute the capabilities for a coarse-grained arch bucket.
pub fn arch_capability(arch: GcnArch) -> QuantCapability {
    match arch {
        GcnArch::RDNA4 | GcnArch::CDNA3 => QuantCapability {
            fp32: true,
            f16: true,
            bf16: true,
            fp8_native: true,
            mxfp4_emulated: true,
            mxfp8_emulated: true,
        },
        GcnArch::RDNA2 | GcnArch::RDNA3 | GcnArch::CDNA2 => QuantCapability {
            fp32: true,
            f16: true,
            bf16: true,
            fp8_native: false,
            mxfp4_emulated: true,
            mxfp8_emulated: true,
        },
        GcnArch::RDNA1 | GcnArch::Other => QuantCapability {
            fp32: true,
            f16: false,
            bf16: false,
            fp8_native: false,
            mxfp4_emulated: false,
            mxfp8_emulated: false,
        },
    }
}

/// Resolve the runtime `QuantMode` for a running arch given a model's
pub fn resolve_quant_mode(arch: GcnArch, requested: QuantMode) -> QuantMode {
    let caps = arch_capability(arch);
    if caps.supports(requested) {
        return requested;
    }
    match requested {
        // Native FP8 on RDNA2/3 is forbidden — fall back to BF16
        QuantMode::Fp8Native => {
            if caps.bf16 {
                QuantMode::Bf16
            } else if caps.f16 {
                QuantMode::F16
            } else {
                QuantMode::Fp32
            }
        }
        QuantMode::MxFp4Emulated | QuantMode::MxFp8Emulated => {
            if caps.bf16 {
                requested
            } else {
                QuantMode::Fp32
            }
        }
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
    use super::*;

    #[test]
    fn fp8_capable_buckets_match_spec() {
        for arch in [GcnArch::RDNA4, GcnArch::CDNA3] {
            let c = arch_capability(arch);
            assert!(
                c.supports(QuantMode::Fp8Native),
                "{:?}: fp8_native expected",
                arch
            );
        }
        for arch in [
            GcnArch::RDNA1,
            GcnArch::RDNA2,
            GcnArch::RDNA3,
            GcnArch::CDNA2,
            GcnArch::Other,
        ] {
            let c = arch_capability(arch);
            assert!(
                !c.supports(QuantMode::Fp8Native),
                "{:?}: fp8_native must NOT be supported",
                arch
            );
        }
    }

    #[test]
    fn rook_and_jackdaw_allowed_on_rdna2_and_rdna3() {
        for arch in [
            GcnArch::RDNA2,
            GcnArch::RDNA3,
            GcnArch::RDNA4,
            GcnArch::CDNA3,
        ] {
            assert_eq!(
                resolve_quant_mode(arch, QuantMode::MxFp4Emulated),
                QuantMode::MxFp4Emulated
            );
            assert_eq!(
                resolve_quant_mode(arch, QuantMode::MxFp8Emulated),
                QuantMode::MxFp8Emulated
            );
        }
    }
}
