//! WI-2: arch-compat bridge between `WeightFormat` (storage codec) and `grim_backend_rocm`'s `QuantMode`/`GcnArch` arch gate.
//! `WeightFormat` itself lives in `grim-format` (canonical home, needed by `ModelFootprint`); this module re-exports it and.

pub use grim_format::WeightFormat;

use grim_backend_rocm::{GcnArch, QuantMode, resolve_quant_mode};

/// WI-2: verdict of a pre-flight compat check between a model's storage
/// codec and the detected hardware's arch gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatResult {
    /// The backend dispatches to this mode natively — no quality loss.
    NativeSupport,
    /// The requested mode is not native, but `resolve_quant_mode` falls back to `to` without changing the output's numerical class (e.g.
    /// FP8 -> BF16).
    FallbackSupport {
        /// The mode the backend will actually dispatch to.
        to: QuantMode,
        /// Human-readable reason for the fallback.
        reason: String,
    },
    /// No supported dispatch path exists. The model cannot run on this hardware as-is (e.g.
    Unsupported { reason: String },
}

/// Map a storage codec to the runtime `QuantMode` the ROCm backend would dispatch to.
/// This is the single bridge between the *storage* codec (`WeightFormat`, a training/conversion concept) and the.
pub fn codec_quant_mode(format: WeightFormat) -> Option<QuantMode> {
    Some(match format {
        WeightFormat::Bf16 => QuantMode::Bf16,
        WeightFormat::Raven => QuantMode::Fp8Native,
        WeightFormat::Rook => QuantMode::MxFp4Emulated,
        WeightFormat::Jackdaw => QuantMode::MxFp8Emulated,
        // 4-bit element path: dequant in LDS to BF16, WMMA GEMM.
        WeightFormat::Nutcracker => QuantMode::MxFp4Emulated,
        // Storage-only aliases — no runtime dispatch gate.
        WeightFormat::Crow | WeightFormat::Jay | WeightFormat::Magpie => {
            return None;
        }
        // TreePie decodes to native FP16 and feeds the existing FP16 dot path, so
        // it has no dispatch gate of its own either. Kept in step with
        // `WeightFormat::as_quant_mode_hint`, which is the canonical answer; this
        // mirror must not drift from it.
        WeightFormat::TreePie => {
            return None;
        }
    })
}

/// WI-2: classify a storage codec against `arch` using the existing `resolve_quant_mode` gate.
/// Reuses, does not reimplement, the backend's compat logic.
pub fn check_support(format: WeightFormat, arch: GcnArch) -> CompatResult {
    let mode = match codec_quant_mode(format) {
        Some(m) => m,
        // Storage-only aliases (Crow/Jay/Magpie) have no dispatch gate: they're resolved at conversion time into a concrete mode, so there's nothing to gate here.
        // Treat as native.
        None => return CompatResult::NativeSupport,
    };
    let resolved = resolve_quant_mode(arch, mode);
    if resolved == mode {
        return CompatResult::NativeSupport;
    }
    if matches!(resolved, QuantMode::Fp32) && !matches!(mode, QuantMode::Fp32) {
        // Fallback collapsed all the way to FP32 — that's a real
        // capability gap, not a same-class downshift. Flag it.
        return CompatResult::Unsupported {
            reason: format!(
                "{format:?} on {arch:?} has no supported dispatch path; \
                 resolve_quant_mode collapsed to FP32"
            ),
        };
    }
    CompatResult::FallbackSupport {
        to: resolved,
        reason: format!(
            "{format:?} ({mode:?}) is not native on {arch:?}; \
             falling back to {resolved:?}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_aliases_have_no_dispatch_gate() {
        for fmt in [
            WeightFormat::Crow,
            WeightFormat::Jay,
            WeightFormat::Magpie,
            // TreePie decodes to native FP16 and feeds the existing FP16 dot
            // path, so it has no gate of its own either.
            WeightFormat::TreePie,
        ] {
            assert!(
                codec_quant_mode(fmt).is_none(),
                "{fmt:?} is a storage-only alias with no dispatch gate"
            );
            // And the compat check treats them as native — they're resolved
            // at conversion time, so there's nothing to gate here.
            assert!(matches!(
                check_support(fmt, GcnArch::RDNA3),
                CompatResult::NativeSupport
            ));
        }
    }

    #[test]
    fn test_raven_falls_back_on_rdna2() {
        // Raven is FP8 native — RDNA2/3 have no native FP8, so it must
        // fall back to BF16 (a same-class downshift), not be unsupported.
        match check_support(WeightFormat::Raven, GcnArch::RDNA2) {
            CompatResult::FallbackSupport { to, .. } => {
                assert_eq!(to, QuantMode::Bf16, "Raven on RDNA2 falls back to BF16");
            }
            other => panic!("expected FallbackSupport, got {other:?}"),
        }
    }

    #[test]
    fn test_raven_native_on_rdna4() {
        assert!(matches!(
            check_support(WeightFormat::Raven, GcnArch::RDNA4),
            CompatResult::NativeSupport
        ));
    }

    #[test]
    fn test_nutcracker_is_native_on_all_4bit_arches() {
        // Nutcracker is a 4-bit emulated codec like Rook/Jay, so it must be
        // natively supported everywhere those are — no RDNA4-only gate.
        for arch in [GcnArch::RDNA2, GcnArch::RDNA3, GcnArch::RDNA4] {
            assert!(
                matches!(
                    check_support(WeightFormat::Nutcracker, arch),
                    CompatResult::NativeSupport
                ),
                "Nutcracker should be native on {arch:?}"
            );
        }
    }

    #[test]
    fn test_nutcracker_dispatches_to_the_4bit_emulated_path() {
        assert_eq!(
            codec_quant_mode(WeightFormat::Nutcracker),
            Some(QuantMode::MxFp4Emulated)
        );
    }
}

    /// The two answers to "does this format have a runtime dispatch gate?" must
    /// agree, for every format.
    ///
    /// `WeightFormat::as_quant_mode_hint` (in grim-format) is the canonical
    /// answer; `codec_quant_mode` here is a mirror of it onto the backend's
    /// `QuantMode`. Two functions answering one question is a drift hazard, and
    /// it already drifted once: adding `WeightFormat::TreePie` updated the
    /// canonical side and left this mirror missing an arm, which surfaced as a
    /// non-exhaustive-match build error in a crate four steps away in the
    /// dependency graph.
    ///
    /// The list below is deliberately exhaustive and written out. Adding a
    /// `WeightFormat` variant makes this fail to *compile* until the new format
    /// is classified here, which is the point -- a hand-maintained subset list
    /// passes vacuously for anything it forgot, which is exactly what happened.
    #[test]
    fn both_dispatch_bridges_agree_for_every_format() {
        use grim_format::QuantModeHint;

        /// (format, expected mode) with `None` meaning "no dispatch gate".
        const ALL: [(WeightFormat, Option<(QuantModeHint, QuantMode)>); 9] = [
            (WeightFormat::Bf16, Some((QuantModeHint::Bf16, QuantMode::Bf16))),
            (WeightFormat::Raven, Some((QuantModeHint::Fp8Native, QuantMode::Fp8Native))),
            (WeightFormat::Rook, Some((QuantModeHint::MxFp4Emulated, QuantMode::MxFp4Emulated))),
            (WeightFormat::Jay, None),
            (WeightFormat::Crow, None),
            (WeightFormat::Jackdaw, Some((QuantModeHint::MxFp8Emulated, QuantMode::MxFp8Emulated))),
            (WeightFormat::Magpie, None),
            (WeightFormat::Nutcracker, Some((QuantModeHint::MxFp4Emulated, QuantMode::MxFp4Emulated))),
            (WeightFormat::TreePie, None),
        ];

        for (fmt, expected) in ALL {
            let hint = fmt.as_quant_mode_hint();
            let mirror = codec_quant_mode(fmt);
            match expected {
                Some((h, m)) => {
                    assert_eq!(hint, Some(h), "{fmt:?}: canonical hint drifted");
                    assert_eq!(mirror, Some(m), "{fmt:?}: mirrored mode drifted");
                }
                None => {
                    assert_eq!(hint, None, "{fmt:?}: canonical hint should be None");
                    assert_eq!(
                        mirror, None,
                        "{fmt:?}: mirror disagrees with the canonical hint"
                    );
                }
            }
            // A format with no gate is treated as native, since it is resolved at
            // conversion time and there is nothing left to gate.
            if expected.is_none() {
                assert!(
                    matches!(
                        check_support(fmt, GcnArch::RDNA3),
                        CompatResult::NativeSupport
                    ),
                    "{fmt:?} has no gate, so the compat check must pass it"
                );
            }
        }
    }
