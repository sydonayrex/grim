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

    /// Launch geometry for Phase-1 QKV attention. Sequential implementation:
    /// 1 thread per (query position, head) pair, block = (1,1,1). The
    /// sequential KV walk and sequential online softmax produce the exact
    /// same floating-point results as the CPU fallback (same reduction order).
    pub fn hip_launch_params(&self) -> HipKernelLaunch {
        let grid_x = self.max_seq_len as u32;
        let grid_y = self.num_heads as u32;
        HipKernelLaunch {
            grid_dim: hipDim3::new(grid_x, grid_y, 1),
            block_dim: hipDim3::new(1, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// WI 2.4.4-2 — decode GEMM config (Rust-centric, replaces vendored CK wrapper). [see: `ck_gemm.cpp`, `ck`]
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
    /// Quantization bits of the cached K/V (4 or 8).
    pub quant_bits: u8,
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
            quant_bits: 4,
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
