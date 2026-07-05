//! ROCm kernel fusion configurations for Unsloth-inspired performance optimizations.
//!
//! These configs encode launch-time parameters for the fused HIP kernels that
//! the Oxidizer CLI can reference when baking `.grim` artifacts. They are pure
//! CPU-side data structures; runtime device execution lives in the parent
//! `grim-backend-rocm` crate.

/// HIP kernel launch geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipKernelLaunch {
    pub grid_dim: hipDim3,
    pub block_dim: hipDim3,
    pub shared_mem_bytes: usize,
}

/// HIP grid/block dimension triple — matches `hipDim3` semantics but is plain Rust
/// so callers don't need real HIP FFI at config-construction time.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct hipDim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl hipDim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

/// Fusion configuration for RMSNorm + MatMul operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormMatMulFusionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub wavefront_size: u32,
    pub lds_size: u32,
}

/// Fusion configuration for QKV Projection + Attention operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvAttentionFusionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub wavefront_size: u32,
}

impl RmsNormMatMulFusionConfig {
    /// Generate HIP kernel launch parameters for fused RMSNorm+MatMul.
    ///
    /// Pick a block size that fills one wavefront × 4 (wavefront-aware LDS reuse)
    /// and as many blocks as needed to cover the intermediate projection.
    pub fn hip_launch_params(&self) -> HipKernelLaunch {
        let block_dim_x = if self.wavefront_size == 32 { 128 } else { 256 };
        let grid_x = (self.intermediate_size + block_dim_x - 1) / block_dim_x;
        HipKernelLaunch {
            grid_dim: hipDim3::new(grid_x as u32, 1, 1),
            block_dim: hipDim3::new(block_dim_x as u32, 1, 1),
            shared_mem_bytes: self.lds_size.min(65536) as usize,
        }
    }
}

impl QkvAttentionFusionConfig {
    /// Generate HIP kernel launch parameters for fused QKV+Attention.
    ///
    /// One block per attention head; block size depends on wavefront size.
    pub fn hip_launch_params(&self) -> HipKernelLaunch {
        let block_dim_x = if self.wavefront_size == 32 { 128 } else { 256 };
        let grid_x = (self.num_heads + block_dim_x - 1) / block_dim_x;
        HipKernelLaunch {
            grid_dim: hipDim3::new(grid_x as u32, 1, 1),
            block_dim: hipDim3::new(block_dim_x as u32, 1, 1),
            shared_mem_bytes: (self.head_dim * 4).min(32768) as usize,
        }
    }
}
