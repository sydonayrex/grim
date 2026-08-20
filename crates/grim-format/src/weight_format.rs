//! Storage codec enum for model weights.
//!
//! `WeightFormat` names the codec used to store base model weights during
//! training. The names are grim's internal bird-themed aliases. It lives in
//! `grim-format` (not `grim-garage`) because `ModelFootprint` — a
//! header-only model descriptor — needs it, and `grim-format` must not
//! depend on `grim-garage`. `grim-garage` re-exports it.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Codec used to store base model weights during training.
///
/// On arches that don't support a format natively, grim falls back via
/// `resolve_quant_mode`: Raven -> Bf16 on RDNA2/3; all others pass through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WeightFormat {
    /// BF16 full precision. No quantization. Default for all arches.
    #[default]
    Bf16,
    /// Crow: Q4_K GGML super-block 4-bit. 4.5 bpw. RDNA2+.
    Crow,
    /// Raven: FP8 E4M3 native GEMM. 8 bpw. RDNA4 and CDNA3 only.
    /// Downshifts to Bf16 on RDNA2/3 via resolve_quant_mode.
    Raven,
    /// Rook: MXFP4 E2M1 emulated. Dequant in LDS to BF16, WMMA GEMM. ~4.1 bpw. RDNA2+.
    Rook,
    /// Jay: MXFP4 block-16. Alias for Fp4Block16. ~4.1 bpw. RDNA2+.
    Jay,
    /// Jackdaw: MXFP8 E4M3 emulated. Dequant in LDS to BF16, WMMA GEMM. ~8 bpw. RDNA2+.
    /// Better than Raven at same bpw: shared E8M0 exponent captures outlier blocks.
    Jackdaw,
    /// Magpie: MXFP8 block-16. Alias for Fp8Block16. ~8 bpw. RDNA2+.
    Magpie,
}

impl WeightFormat {
    /// Bits-per-weight for this codec. Used by the VRAM estimate in
    /// `ModelFootprint::estimate_vram_bytes`. Conservative upper bounds —
    /// the on-disk size may be smaller, but predicting *under* rather than
    /// *over* is the failure mode that causes OOM.
    pub fn bpw(self) -> f32 {
        match self {
            WeightFormat::Bf16 => 16.0,
            WeightFormat::Crow => 4.5,
            WeightFormat::Raven => 8.0,
            WeightFormat::Rook => 4.1,
            WeightFormat::Jay => 4.1,
            WeightFormat::Jackdaw => 8.0,
            WeightFormat::Magpie => 8.0,
        }
    }

    /// Map this codec to a backend-agnostic `QuantModeHint`.
    ///
    /// This is the single bridge between the *storage* codec
    /// (`WeightFormat`, a training/conversion concept) and the *dispatch*
    /// mode a backend selects. It returns a `QuantModeHint` (defined here,
    /// in `grim-format`) rather than a concrete `QuantMode` (which lives in
    /// each backend crate) so `grim-format` stays backend-free. WI-2
    /// pre-flight uses the hint to classify native vs. fallback support.
    ///
    /// `Crow`/`Jay`/`Magpie` have no runtime dispatch equivalent — they are
    /// storage-only aliases resolved at conversion time. `None` here means
    /// "no runtime dispatch gate applies", not "unsupported".
    pub fn as_quant_mode_hint(self) -> Option<QuantModeHint> {
        Some(match self {
            WeightFormat::Bf16 => QuantModeHint::Bf16,
            WeightFormat::Raven => QuantModeHint::Fp8Native,
            WeightFormat::Rook => QuantModeHint::MxFp4Emulated,
            WeightFormat::Jackdaw => QuantModeHint::MxFp8Emulated,
            // Storage-only aliases — no runtime dispatch gate.
            WeightFormat::Crow | WeightFormat::Jay | WeightFormat::Magpie => {
                return None;
            }
        })
    }
}

/// Backend-agnostic quantization dispatch hint. Mirrors
/// `grim_backend_rocm::QuantMode`'s variants without depending on any
/// backend crate. `grim-garage` maps this to a concrete `QuantMode` before
/// running the arch gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantModeHint {
    Fp32,
    F16,
    Bf16,
    Fp8Native,
    MxFp4Emulated,
    MxFp8Emulated,
    Int8W8A8,
}

impl std::fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_string(self).unwrap_or_else(|_| "?".to_string());
        f.write_str(&s)
    }
}

/// Parse a codec name from a `.grim` header's `target_weight_format`
/// string. Accepts both the serde `snake_case` spelling ("bf16", "crow")
/// and the canonical variant names ("Bf16", "Crow"). Unknown strings
/// return `Err` so the caller falls back to the raw byte sum rather than
/// guessing a smaller bpw.
impl FromStr for WeightFormat {
    type Err = ParseWeightFormatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bf16" | "Bf16" | "BF16" => Ok(WeightFormat::Bf16),
            "crow" | "Crow" => Ok(WeightFormat::Crow),
            "raven" | "Raven" => Ok(WeightFormat::Raven),
            "rook" | "Rook" => Ok(WeightFormat::Rook),
            "jay" | "Jay" => Ok(WeightFormat::Jay),
            "jackdaw" | "Jackdaw" => Ok(WeightFormat::Jackdaw),
            "magpie" | "Magpie" => Ok(WeightFormat::Magpie),
            _ => Err(ParseWeightFormatError { raw: s.to_string() }),
        }
    }
}

/// Error returned when a codec name cannot be parsed into a `WeightFormat`.
#[derive(Debug, Clone)]
pub struct ParseWeightFormatError {
    pub raw: String,
}

impl std::fmt::Display for ParseWeightFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown WeightFormat name '{}'", self.raw)
    }
}

impl std::error::Error for ParseWeightFormatError {}