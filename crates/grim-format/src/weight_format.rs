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

/// Summary descriptor of a model's memory footprint and architecture.
#[derive(Debug, Clone)]
pub struct ModelFootprint {
    pub architecture: String,
    pub param_count: Option<u64>,
    pub quant_format: Option<WeightFormat>,
    pub estimated_weight_bytes: u64,
    pub context_length_default: Option<u32>,
    pub is_moe: bool,
}

impl ModelFootprint {
    pub fn from_gguf_file(gguf: &crate::gguf::GgufFile) -> Self {
        let arch = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let context_length_default = gguf
            .metadata
            .get(&format!("{arch}.context_length"))
            .or_else(|| gguf.metadata.get("context_length"))
            .and_then(|v| v.as_u32());

        let mut param_count: u64 = 0;
        let mut estimated_weight_bytes: u64 = 0;
        let mut is_moe = arch.to_lowercase().contains("moe");

        for t in &gguf.tensors {
            param_count += t.elem_count() as u64;
            estimated_weight_bytes += t.size_bytes;
            if t.name.contains("expert") || t.name.contains("block_sparse") {
                is_moe = true;
            }
        }

        let quant_format = gguf
            .metadata
            .get("general.target_weight_format")
            .or_else(|| gguf.metadata.get("target_weight_format"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<WeightFormat>().ok())
            .or_else(|| {
                gguf.tensors.first().and_then(|t| match t.dtype {
                    crate::gguf::GgufDType::BF16 => Some(WeightFormat::Bf16),
                    crate::gguf::GgufDType::Q4K | crate::gguf::GgufDType::Q4_0 => {
                        Some(WeightFormat::Crow)
                    }
                    crate::gguf::GgufDType::MXFP4 => Some(WeightFormat::Rook),
                    _ => None,
                })
            });

        Self {
            architecture: arch,
            param_count: if param_count > 0 {
                Some(param_count)
            } else {
                None
            },
            quant_format,
            estimated_weight_bytes,
            context_length_default,
            is_moe,
        }
    }

    pub fn from_grim_file(grim: &crate::format::GrimFile) -> Self {
        let mut arch = grim
            .metadata
            .target_gcn
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let mut ctx = None;

        if let Some(ref gguf_meta) = grim.metadata.gguf_metadata {
            if let Some(a) = gguf_meta.get("general.architecture").and_then(|v| v.as_str()) {
                arch = a.to_string();
            }
            ctx = gguf_meta
                .get(&format!("{arch}.context_length"))
                .or_else(|| gguf_meta.get("context_length"))
                .and_then(|v| v.as_u32());
        }

        let mut param_count: u64 = 0;
        let mut estimated_weight_bytes: u64 = 0;
        let mut is_moe = arch.to_lowercase().contains("moe");

        for t in &grim.tensors {
            let elems: u64 = t.shape.iter().map(|&d| d as u64).product();
            param_count += elems;
            estimated_weight_bytes +=
                t.payload_size + (t.outlier_count as u64 * 6) + t.kv_compressed_size;
            if t.name.contains("expert") || t.name.contains("block_sparse") {
                is_moe = true;
            }
        }

        let quant_format = grim
            .metadata
            .target_weight_format
            .as_deref()
            .and_then(|s| s.parse::<WeightFormat>().ok());

        Self {
            architecture: arch,
            param_count: if param_count > 0 {
                Some(param_count)
            } else {
                None
            },
            quant_format,
            estimated_weight_bytes,
            context_length_default: ctx,
            is_moe,
        }
    }
}

/// Conservative estimation of VRAM bytes needed for loading and running a model.
///
/// Formula: weight bytes + KV cache estimate + 10% overhead heuristic.
pub fn estimate_vram_bytes(
    footprint: &ModelFootprint,
    context_length: u32,
    batch_size: u32,
    kv_layers: u32,
    kv_heads: u32,
    head_dim: u32,
) -> u64 {
    // 2 bytes per element (BF16/FP16) * 2 (key + value)
    let kv_bytes = 2u64
        * 2
        * (kv_layers as u64)
        * (kv_heads as u64)
        * (head_dim as u64)
        * (context_length as u64)
        * (batch_size as u64);
    let base = footprint.estimated_weight_bytes.saturating_add(kv_bytes);
    let overhead = (base as f64 * 0.10) as u64;
    base.saturating_add(overhead)
}