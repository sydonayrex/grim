//! WI-2: arch-compat bridge between `WeightFormat` (storage codec) and
//! `grim_backend_rocm`'s `QuantMode`/`GcnArch` arch gate.
//!
//! `WeightFormat` itself lives in `grim-format` (canonical home, needed by
//! `ModelFootprint`); this module re-exports it and adds the WI-2
//! `CompatResult` type + `check_support` helper, which live here because
//! they depend on `grim-backend-rocm`, which `grim-format` must not.

pub use grim_format::WeightFormat;

use grim_backend_rocm::{GcnArch, QuantMode, resolve_quant_mode};

/// WI-2: verdict of a pre-flight compat check between a model's storage
/// codec and the detected hardware's arch gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatResult {
    /// The backend dispatches to this mode natively — no quality loss.
    NativeSupport,
    /// The requested mode is not native, but `resolve_quant_mode` falls
    /// back to `to` without changing the output's numerical class (e.g.
    /// FP8 -> BF16). The model still runs correctly, just denser.
    FallbackSupport {
        /// The mode the backend will actually dispatch to.
        to: QuantMode,
        /// Human-readable reason for the fallback.
        reason: String,
    },
    /// No supported dispatch path exists. The model cannot run on this
    /// hardware as-is (e.g. an int8-only kernel path on an arch with no
    /// int8 MFMA). This is a hard stop, not a soft warning.
    Unsupported { reason: String },
}

/// Map a storage codec to the runtime `QuantMode` the ROCm backend would
/// dispatch to. This is the single bridge between the *storage* codec
/// (`WeightFormat`, a training/conversion concept) and the *dispatch*
/// mode (`QuantMode`, a kernel-selection concept). WI-2 pre-flight
/// uses it to classify native vs. fallback support.
///
/// `Crow`/`Jay`/`Magpie` have no `QuantMode` equivalent — they are
/// storage-only aliases resolved at conversion time. `None` here means
/// "no runtime dispatch gate applies", not "unsupported".
pub fn codec_quant_mode(format: WeightFormat) -> Option<QuantMode> {
    Some(match format {
        WeightFormat::Bf16 => QuantMode::Bf16,
        WeightFormat::Raven => QuantMode::Fp8Native,
        WeightFormat::Rook => QuantMode::MxFp4Emulated,
        WeightFormat::Jackdaw => QuantMode::MxFp8Emulated,
        // Storage-only aliases — no runtime dispatch gate.
        WeightFormat::Crow | WeightFormat::Jay | WeightFormat::Magpie => {
            return None;
        }
    })
}

/// WI-2: classify a storage codec against `arch` using the existing
/// `resolve_quant_mode` gate. Reuses, does not reimplement, the backend's
/// compat logic.
pub fn check_support(format: WeightFormat, arch: GcnArch) -> CompatResult {
    let mode = match codec_quant_mode(format) {
        Some(m) => m,
        // Storage-only aliases (Crow/Jay/Magpie) have no dispatch gate:
        // they're resolved at conversion time into a concrete mode, so
        // there's nothing to gate here. Treat as native.
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
        for fmt in [WeightFormat::Crow, WeightFormat::Jay, WeightFormat::Magpie] {
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
}