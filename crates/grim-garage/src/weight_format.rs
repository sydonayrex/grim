use serde::{Deserialize, Serialize};

/// Names the codec used to store base model weights during training.
/// The names are grim's internal bird-themed aliases.
///
/// On arches that don't support a format natively, grim falls back
/// via resolve_quant_mode: Raven -> Bf16 on RDNA2/3; all others pass through.
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
