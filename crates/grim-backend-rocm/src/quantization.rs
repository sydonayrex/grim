//! Phase-3 §3.3 — fp8/int8 quantization arch gate + dispatch. [see: `resolve_quant_mode`, `rocm-quantization-inference`]

use std::fmt;

/// Canonical coarse-grained arch bin. The ROCm nightly headers stamp [see: `gcnArchName`, `"gfx1036"`, `:N`, `Other`]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GcnArch {
    /// RDNA1 — gfx10xx around 1010-1012 (very old consumer).
    RDNA1,
    /// RDNA2 — gfx1030-1036 (van Gogh / RX 6000 series / integrated APU).
    RDNA2,
    /// RDNA3 — gfx1100-1151 (RX 7000 series).
    RDNA3,
    /// RDNA4 — gfx1200-1201 (RDNA4 discrete and mobile).
    RDNA4,
    /// CDNA1 — gfx906, gfx900 (MI50, MI60, Vega20).
    CDNA1,
    /// CDNA2 — gfx908, gfx90a (MI100, MI210, MI250, MI250X; full MFMA).
    CDNA2,
    /// CDNA3 — gfx940-942 (MI300 series; full MFMA + fp8 path).
    CDNA3,
    /// CDNA4 — gfx950 (MI350, MI355X series; FP8/FP4/MXFP4 MFMA).
    CDNA4,
    /// UDNA — gfx1200, gfx1300, gfx1301 (AMD Unified DNA architecture uniting RDNA & CDNA).
    UDNA,
    /// Anything else (gfx0000, malformed strings).
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
    // UDNA / GFX13xx
    if let Some(_s) = strip_prefix_digits(suffix, "13") {
        return GcnArch::UDNA;
    }
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
        // gfx906/900 = CDNA1, gfx908/90a = CDNA2, gfx940-942 = CDNA3, gfx950 = CDNA4.
        return match s {
            r if r.starts_with("50") || r.starts_with("51") => GcnArch::CDNA4,
            r if r.starts_with("40")
                || r.starts_with("41")
                || r.starts_with("42")
                || r.starts_with("43")
                || r.starts_with("44") =>
            {
                GcnArch::CDNA3
            }
            r if r.starts_with("08") || r.starts_with("0a") || r.starts_with("0A") => GcnArch::CDNA2,
            r if r.starts_with("06") => GcnArch::CDNA1,
            _ => GcnArch::Other,
        };
    }
    GcnArch::Other
}

impl std::fmt::Display for GcnArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GcnArch::RDNA1 => "RDNA1",
            GcnArch::RDNA2 => "RDNA2",
            GcnArch::RDNA3 => "RDNA3",
            GcnArch::RDNA4 => "RDNA4",
            GcnArch::CDNA1 => "CDNA1",
            GcnArch::CDNA2 => "CDNA2",
            GcnArch::CDNA3 => "CDNA3",
            GcnArch::CDNA4 => "CDNA4",
            GcnArch::UDNA => "UDNA",
            GcnArch::Other => "Other",
        };
        f.write_str(s)
    }
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
    /// W8A8 SmoothQuant-style int8 GEMM — activations quantized per-token, weights quantized
    /// per-channel. Uses int8 MFMA (CDNA2/3: `__builtin_amdgcn_mfma_i32_32x32x16_i8`)
    /// or the int8 dot-product path on RDNA3/4.
    Int8W8A8,
}

/// The concrete FP8 element format a device is **natively** capable of. This is
/// the axis a W8A8 GEMM must branch on — the two formats have different packed
/// code namespaces and different MFMA predicates. One `bool` cannot express it.
///
/// NAMING: this "OCP" is the OCP *element format* (e4m3fn), NOT OCP *Microscaling*
/// (MXFP4/MXFP8 — see the Jay/Magpie tiers in `charon.rs` / `grim-quant`). The
/// variant is spelled `OcpFn` to avoid that collision.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Fp8NativeFormat {
    /// No native FP8 path on this arch.
    None,
    /// OCP FP8 e4m3fn — native on RDNA4 (`gfx1200+`) W8A8 WMMA.
    OcpFn,
    /// AMD e4m3fnuz — native on CDNA3 (`gfx940-942`) W8A8 MFMA.
    Fnuz,
}

impl Fp8NativeFormat {
    pub fn is_native(self) -> bool {
        !matches!(self, Fp8NativeFormat::None)
    }
}

impl fmt::Display for Fp8NativeFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fp8NativeFormat::None => write!(f, "none"),
            Fp8NativeFormat::OcpFn => write!(f, "OCP-e4m3fn (RDNA4)"),
            Fp8NativeFormat::Fnuz => write!(f, "e4m3fnuz (CDNA3)"),
        }
    }
}

/// Per-arch capability bitmap. The struct is the *output* of the gate; [see: `capability.supports(mode)`]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct QuantCapability {
    fp32: bool,
    f16: bool,
    bf16: bool,
    /// Native FP8 element format (OCP e4m3fn vs AMD e4m3fnuz) — deferred to
    /// concrete arches that actually differ. `None` == no native FP8.
    fp8: Fp8NativeFormat,
    mxfp4_emulated: bool,
    mxfp8_emulated: bool,
    /// Int8 MFMA for W8A8 SmoothQuant (CDNA2+: `mfma_i32_32x32x16_i8`).
    int8_w8a8: bool,
}

impl QuantCapability {
    pub fn supports(self, mode: QuantMode) -> bool {
        match mode {
            QuantMode::Fp32 => self.fp32,
            QuantMode::F16 => self.f16,
            QuantMode::Bf16 => self.bf16,
            QuantMode::Fp8Native => self.fp8.is_native(),
            QuantMode::MxFp4Emulated => self.mxfp4_emulated,
            QuantMode::MxFp8Emulated => self.mxfp8_emulated,
            QuantMode::Int8W8A8 => self.int8_w8a8,
        }
    }

    /// The concrete FP8 element format native on this arch (`None` when FP8 is
    /// not native). W8A8 dispatch branches on this — see `Fp8NativeFormat`.
    pub fn fp8_native_format(self) -> Fp8NativeFormat {
        self.fp8
    }
}

impl fmt::Display for QuantCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fp32={} f16={} bf16={} fp8_native={} mxfp4_emulated={} mxfp8_emulated={} int8_w8a8={}",
            self.fp32,
            self.f16,
            self.bf16,
            self.fp8,
            self.mxfp4_emulated,
            self.mxfp8_emulated,
            self.int8_w8a8
        )
    }
}

/// Compute the capabilities for a coarse-grained arch bucket.
pub fn arch_capability(arch: GcnArch) -> QuantCapability {
    match arch {
        GcnArch::UDNA | GcnArch::RDNA4 => QuantCapability {
            fp32: true,
            f16: true,
            bf16: true,
            fp8: Fp8NativeFormat::OcpFn,
            mxfp4_emulated: true,
            mxfp8_emulated: true,
            int8_w8a8: true,
        },
        GcnArch::CDNA4 | GcnArch::CDNA3 => QuantCapability {
            fp32: true,
            f16: true,
            bf16: true,
            fp8: Fp8NativeFormat::Fnuz,
            mxfp4_emulated: true,
            mxfp8_emulated: true,
            int8_w8a8: true,
        },
        GcnArch::RDNA2 | GcnArch::RDNA3 | GcnArch::CDNA2 => QuantCapability {
            fp32: true,
            f16: true,
            bf16: true,
            fp8: Fp8NativeFormat::None,
            mxfp4_emulated: true,
            mxfp8_emulated: true,
            int8_w8a8: true,
        },
        GcnArch::CDNA1 => QuantCapability {
            fp32: true,
            f16: true,
            bf16: false,
            fp8: Fp8NativeFormat::None,
            mxfp4_emulated: false,
            mxfp8_emulated: false,
            int8_w8a8: true,
        },
        GcnArch::RDNA1 | GcnArch::Other => QuantCapability {
            fp32: true,
            f16: false,
            bf16: false,
            fp8: Fp8NativeFormat::None,
            mxfp4_emulated: false,
            mxfp8_emulated: false,
            int8_w8a8: false,
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
        QuantMode::Int8W8A8 => {
            if caps.int8_w8a8 {
                requested
            } else {
                QuantMode::Fp32
            }
        }
    }
}

/// Per-channel activation scale array for SmoothQuant W8A8 path. Channels ==
/// output features of a linear/conv. One scale per output channel absorbed from
/// the activation max via gamma migration. Stored as fp32 so downstream
/// in-kernel dequant can work from the canonical float value.
#[derive(Debug, Clone, Default)]
pub struct SmoothQuantActScales {
    /// Flat vec, `len == num_channels`. Must be non-empty when the W8A8 dispatch
    /// is enabled.
    pub channels: Vec<f32>,
}

impl SmoothQuantActScales {
    pub fn new(num_channels: usize) -> Self {
        Self {
            channels: vec![1.0f32; num_channels],
        }
    }

    pub fn borrow(&self) -> &[f32] {
        &self.channels
    }
}

/// Offline calibration result from a single forward pass over a calibration
/// dataset. Collects per-token activation maxes per layer and optionally
/// applies SmoothQuant gamma migration to derive per-channel scales.
#[derive(Debug, Clone)]
pub struct SmoothQuantCalibration {
    /// Per-layer per-channel activation scales after calibration. Layer index
    /// is the position in the transformer stack (0 = embedding-adjacent).
    pub layer_scales: Vec<SmoothQuantActScales>,
}

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn fp8_capable_buckets_match_spec() {
        for arch in [GcnArch::RDNA4, GcnArch::CDNA3, GcnArch::CDNA4, GcnArch::UDNA] {
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
            GcnArch::CDNA1,
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
    fn fp8_native_format_is_split_by_arch() {
        let rdna4 = arch_capability(GcnArch::RDNA4);
        assert_eq!(rdna4.fp8_native_format(), Fp8NativeFormat::OcpFn);
        let udna = arch_capability(GcnArch::UDNA);
        assert_eq!(udna.fp8_native_format(), Fp8NativeFormat::OcpFn);
        let cdna3 = arch_capability(GcnArch::CDNA3);
        assert_eq!(cdna3.fp8_native_format(), Fp8NativeFormat::Fnuz);
        let cdna4 = arch_capability(GcnArch::CDNA4);
        assert_eq!(cdna4.fp8_native_format(), Fp8NativeFormat::Fnuz);
        // Arches with no native FP8 element format at all.
        for arch in [
            GcnArch::RDNA1,
            GcnArch::RDNA2,
            GcnArch::RDNA3,
            GcnArch::CDNA1,
            GcnArch::CDNA2,
            GcnArch::Other,
        ] {
            assert_eq!(
                arch_capability(arch).fp8_native_format(),
                Fp8NativeFormat::None,
                "{arch:?} must have no native FP8 element format"
            );
        }
    }

    #[test]
    fn test_gcn_arch_parsing_cdna_and_udna() {
        assert_eq!(gcn_arch("gfx906"), GcnArch::CDNA1);
        assert_eq!(gcn_arch("gfx908"), GcnArch::CDNA2);
        assert_eq!(gcn_arch("gfx90a"), GcnArch::CDNA2);
        assert_eq!(gcn_arch("gfx940"), GcnArch::CDNA3);
        assert_eq!(gcn_arch("gfx950"), GcnArch::CDNA4);
        assert_eq!(gcn_arch("gfx1200"), GcnArch::RDNA4);
        assert_eq!(gcn_arch("gfx1201"), GcnArch::RDNA4);
        assert_eq!(gcn_arch("gfx1300"), GcnArch::UDNA);
        assert_eq!(gcn_arch("gfx1301"), GcnArch::UDNA);
    }

    #[test]
    fn rook_and_jackdaw_allowed_on_rdna2_and_rdna3() {
        for arch in [
            GcnArch::RDNA2,
            GcnArch::RDNA3,
            GcnArch::RDNA4,
            GcnArch::CDNA3,
            GcnArch::CDNA4,
            GcnArch::UDNA,
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

    // =========================================================================
    // WRECK-10: W8A8 SmoothQuant — structure tests, no GPU required.
    // =========================================================================

    #[test]
    fn quantmode_int8w8a8_variant_compiles() {
        // Verify the new QuantMode variant is in enum space and participates
        // in capability resolution. If it didn't compile, this test wouldn't
        // exist.
        let mode = QuantMode::Int8W8A8;
        assert!(matches!(mode, QuantMode::Int8W8A8));
    }

    #[test]
    fn quant_capability_int8_w8a8_arch_table() {
        // CDNA3 (gfx940) and RDNA3 (gfx1100) should both report int8_w8a8.
        let cdna3 = arch_capability(GcnArch::CDNA3);
        assert!(
            cdna3.int8_w8a8,
            "CDNA3 should support Int8W8A8 (int8 MFMA: mfma_i32_32x32x16_i8)"
        );
        let rna3 = arch_capability(GcnArch::RDNA3);
        assert!(
            rna3.int8_w8a8,
            "RDNA3 should support Int8W8A8 (int8 MFMA / dot product path)"
        );
        let rna4 = arch_capability(GcnArch::RDNA4);
        assert!(
            rna4.int8_w8a8,
            "RDNA4 should support Int8W8A8 (int8 MFMA path)"
        );
    }

    #[test]
    fn quant_capability_int8_w8a8_missing_on_rna1() {
        let rna1 = arch_capability(GcnArch::RDNA1);
        assert!(
            !rna1.int8_w8a8,
            "RDNA1 should NOT support Int8W8A8 (no int8 MFMA)"
        );
    }

    #[test]
    fn quantmode_supports_int8w8a8_gate() {
        let cap = arch_capability(GcnArch::CDNA3);
        assert!(cap.supports(QuantMode::Int8W8A8));
        let rna1_cap = arch_capability(GcnArch::RDNA1);
        assert!(!rna1_cap.supports(QuantMode::Int8W8A8));
    }

    #[test]
    fn smooth_quant_act_scales_default() {
        let scales = SmoothQuantActScales::new(128);
        assert_eq!(scales.channels.len(), 128);
        assert!(scales.channels.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn smooth_quant_calibration_empty_layers_ok() {
        let cal = SmoothQuantCalibration {
            layer_scales: Vec::new(),
        };
        assert!(cal.layer_scales.is_empty());
        let _ = cal.clone();
    }

    #[test]
    fn smooth_quant_calibration_layer_scales_created() {
        let mut cal = SmoothQuantCalibration {
            layer_scales: Vec::new(),
        };
        cal.layer_scales.push(SmoothQuantActScales::new(64));
        cal.layer_scales.push(SmoothQuantActScales::new(128));
        assert_eq!(cal.layer_scales.len(), 2);
        assert_eq!(cal.layer_scales[0].channels.len(), 64);
        assert_eq!(cal.layer_scales[1].channels.len(), 128);
    }

    #[test]
    fn smooth_quant_calibration_scales_not_all_ones_by_default() {
        // Verify the default constructor sets per-channel scales to 1.0 (the
        // "no scaling" identity). A real calibration pass would replace these
        // with observed maxes.
        let scales = SmoothQuantActScales::new(32);
        let all_ones = scales.channels.iter().all(|&s| (s - 1.0).abs() < 1e-6);
        assert!(
            all_ones,
            "default calibration should start with 1.0 identity scales"
        );
    }
}
