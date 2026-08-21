//! ROCm kernel fusion configurations for Unsloth-inspired performance optimizations. [see: `.grim`, `grim-backend-rocm`]

pub use crate::HipDim3 as hipDim3;

const RMSNORM_LDS_MAX_BYTES: u32 = 65536;
const ATTENTION_SHARED_MAX_BYTES: usize = 32768;

/// HIP kernel launch geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipKernelLaunch {
    pub grid_dim: hipDim3,
    pub block_dim: hipDim3,
    pub shared_mem_bytes: usize,
}

/// Fusion configuration for RMSNorm + MatMul operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormMatMulFusionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub wavefront_size: u32,
    pub lds_size: u32,
}

use crate::quantization::QuantMode;

/// Fusion configuration for QKV Projection + Attention operation. [see: `enabled`, `RocmDevice::qkv_attention`, `true`, `false`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvAttentionFusionConfig {
    pub enabled: bool,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub wavefront_size: u32,
    pub quant_mode: QuantMode,
}

impl Default for QkvAttentionFusionConfig {
    fn default() -> Self {
        // Default = true: the backend runs the QKV fused kernel inline
        Self {
            enabled: true,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            wavefront_size: 64,
            quant_mode: QuantMode::Fp32,
        }
    }
}

impl RmsNormMatMulFusionConfig {
    pub fn hip_launch_params(&self) -> HipKernelLaunch {
        let block_dim_x = if self.wavefront_size == 32 { 128 } else { 256 };
        let grid_x = (self.intermediate_size + block_dim_x - 1) / block_dim_x;
        HipKernelLaunch {
            grid_dim: hipDim3::new(grid_x as u32, 1, 1),
            block_dim: hipDim3::new(block_dim_x as u32, 1, 1),
            shared_mem_bytes: self.lds_size.min(RMSNORM_LDS_MAX_BYTES) as usize,
        }
    }
}

impl QkvAttentionFusionConfig {
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Launch geometry for Phase-1 QKV attention. Parallel implementation:
    /// block = (wavefront_size, 1, 1) with 256 threads covering head_dim up to
    /// 256 in parallel. Wavefront-level online-softmax reduction produces
    /// numerically stable results; small rounding differences vs the CPU
    /// sequential reference are expected and covered by test tolerances.
    pub fn hip_launch_params(&self) -> HipKernelLaunch {
        let block_dim_x = if self.wavefront_size == 32 { 128 } else { 256 };
        let grid_x = self.max_seq_len as u32;
        let grid_y = self.num_heads as u32;
        let shared_mem_bytes = (self.head_dim * 4).min(ATTENTION_SHARED_MAX_BYTES);
        HipKernelLaunch {
            grid_dim: hipDim3::new(grid_x, grid_y, 1),
            block_dim: hipDim3::new(block_dim_x, 1, 1),
            shared_mem_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// KI — WRECK-5: KV-cache quantization format enum (replaces legacy quant_bits integer).
// -------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeGemmConfig {
    /// Runtime gate: `false` = always use rocBLAS, `true` = dispatch to the [see: `grim_decode_gemm_f16`, `RocmDevice::matmul`]
    pub enabled: bool,
    /// Wavefront size of the active arch. Tile geometry is the same for [see: `warpSize`]
    pub wavefront_size: u32,
}

impl Default for DecodeGemmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            wavefront_size: 64,
        }
    }
}

/// Configuration for fused dequantization matmul kernels (WI-C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedDequantGemmConfig {
    /// Runtime gate: `false` = always use standard paths, `true` = dispatch to the [see: `grim_fused_dequant_gemm_f16`]
    pub enabled: bool,
    /// Wavefront size of the active arch.
    pub wavefront_size: u32,
}

impl Default for FusedDequantGemmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            wavefront_size: 64,
        }
    }
}

/// Configuration for SplitK matmul reduction (WI-D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitKGemmConfig {
    /// Runtime gate: `false` = always clamp split_k to 1, `true` = allow split_k > 1 with reduction.
    pub enabled: bool,
}

impl Default for SplitKGemmConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// KV-cache quantization format enum (WRECK-5). Replaces the legacy
/// `quant_bits: u8` integer with explicit format descriptors that map to the
/// block/super-block dequant formulas in `kernels::q8_0_dequant` (Q8_0) and
/// `kernels::q4k_dequant` (Q4K).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvQuantFormat {
    /// Dense FP16 K/V (no dequant needed in-kernel; kernel reads fp16 directly).
    Fp16,
    /// Q8_0 block-quantized KV: each 32-element block is 34 bytes (2-byte fp16
    /// delta + 32× int8 codes). Per-block scale is the fp16 delta; k_scales[] unused.
    Q8_0,
    /// Q4K super-block-quantized KV: each 256-element super-block is 144 bytes
    /// (2-byte fp16 d + 2-byte fp16 min + 12-byte packed scales + 128 bytes nibbles).
    /// Per-super-block scale embedded in block; k_scales[] unused.
    Q4K,
    /// Legacy nibble dequant path (quant_bits == 4): 4-bit per nibble, 2 nibbles per byte,
    /// external scale from k_scales[]. Backward compat; use Q4K for the new super-block path.
    LegacyNibble,
    /// Legacy int8 dequant path (quant_bits == 8): ((int8 - 128) / 127) * external scale,
    /// external scale from k_scales[]. Backward compat; use Q8_0 for the new block path.
    LegacyInt8,
}

impl KvQuantFormat {
    /// Convert a legacy `quant_bits` integer to a KvQuantFormat for backward
    /// compat. `quant_bits == 4` with `use_legacy_nibble_path == true` maps to
    /// Q4K so the existing nibble-dequant behavior is preserved until call sites
    /// migrate to explicit KvQuantFormat::Q4K. `quant_bits == 8` maps to Q8_0.
    pub fn from_legacy_quant_bits(quant_bits: u8, use_legacy_path: bool) -> Self {
        match quant_bits {
            4 => {
                if use_legacy_path {
                    Self::LegacyNibble
                } else {
                    Self::Q4K
                }
            }
            8 => {
                if use_legacy_path {
                    Self::LegacyInt8
                } else {
                    Self::Q8_0
                }
            }
            _ => Self::Fp16,
        }
    }

    /// Bits per weight element (for logging / config serialization).
    pub fn bits(&self) -> u8 {
        match self {
            Self::Fp16 => 16,
            Self::Q8_0 => 8,
            Self::Q4K => 4,
            Self::LegacyNibble => 4,
            Self::LegacyInt8 => 8,
        }
    }

    /// In-kernel dequant kind selector, passed as the `quant_format` arg to the
    /// kernel so it can select the right dequant formula. 0 = Fp16, 1 = Q8_0,
    /// 2 = Q4K, -1 = legacy nibble (quant_bits == 4), -2 = legacy int8 (quant_bits == 8).
    pub fn kernel_arg(&self) -> i32 {
        match self {
            Self::Fp16 => 0,
            Self::Q8_0 => 1,
            Self::Q4K => 2,
            Self::LegacyNibble => -1,
            Self::LegacyInt8 => -2,
        }
    }
}

/// Configuration for the fused KV-dequant-attention kernel (WI-R5). [see: `CompressedKvBlock`, `off`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvDequantAttentionConfig {
    /// Runtime gate. `false` = fall back to the dense attention path.
    pub enabled: bool,
    /// Number of query heads (GQA: `num_heads >= num_kv_heads`).
    pub num_heads: usize,
    /// Number of KV heads in the compressed block.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Quantization format of the cached K/V (Fp16, Q8_0 block-quantized, or Q4K super-block-quantized).
    pub quant_format: KvQuantFormat,
    /// Wavefront size of the active arch.
    pub wavefront_size: u32,
}

impl Default for KvDequantAttentionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            quant_format: KvQuantFormat::Fp16,
            wavefront_size: 64,
        }
    }
}

impl KvDequantAttentionConfig {
    /// Build a config from a legacy `quant_bits` integer, preserving backward
    /// compat for existing call sites that haven't migrated to KvQuantFormat yet.
    /// When `quant_bits == 4`, maps to Q4K (the existing nibble-dequant behavior
    /// is preserved via the kernel's quant_format==2 path until the kernel source
    /// is updated). When `quant_bits == 8`, maps to Q8_0. Otherwise Fp16.
    pub fn from_legacy_quant_bits(quant_bits: u8) -> Self {
        Self {
            enabled: true,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            quant_format: KvQuantFormat::from_legacy_quant_bits(quant_bits, true),
            wavefront_size: 64,
        }
    }
}

/// Configuration for the WMMA (Wave Matrix Multiply-Accumulate) GEMM kernel (WI-G). [see: `enabled`, `RocmDevice::matmul`, `true`, `false`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WmmaGemmConfig {
    /// Runtime gate: `false` = always use standard paths, `true` = dispatch to the [see: `grim_wmma_gemm`]
    pub enabled: bool,
    /// Wavefront size of the active arch.
    pub wavefront_size: u32,
}

impl Default for WmmaGemmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            wavefront_size: 64,
        }
    }
}

/// WI-F1 — Load-time concatenation of the per-layer Q/K/V projection weights
/// into a single row-major `[hidden, q_dim + k_dim + v_dim]` matrix, so all
/// three projections run as one GEMM launch (`RocmDevice::fused_qkv_proj`).
/// One-time host cost at model load; must never run per forward pass.
pub fn concat_qkv_weights(
    q_w: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    hidden: usize,
) -> crate::Result<Vec<f32>> {
    if hidden == 0 || q_w.len() % hidden != 0 || k_w.len() % hidden != 0 || v_w.len() % hidden != 0
    {
        return Err(crate::Error::Shape(format!(
            "concat_qkv_weights: weights must be row-major [hidden, _] with hidden={hidden} (got lens {}/{}/{})",
            q_w.len(),
            k_w.len(),
            v_w.len()
        )));
    }
    let q_dim = q_w.len() / hidden;
    let k_dim = k_w.len() / hidden;
    let v_dim = v_w.len() / hidden;
    let qkv_dim = q_dim + k_dim + v_dim;
    let mut out = Vec::with_capacity(hidden * qkv_dim);
    for r in 0..hidden {
        out.extend_from_slice(&q_w[r * q_dim..(r + 1) * q_dim]);
        out.extend_from_slice(&k_w[r * k_dim..(r + 1) * k_dim]);
        out.extend_from_slice(&v_w[r * v_dim..(r + 1) * v_dim]);
    }
    Ok(out)
}
