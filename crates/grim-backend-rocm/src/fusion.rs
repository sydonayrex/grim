//! ROCm kernel fusion configurations for Unsloth-inspired performance optimizations.
//!
//! These configs encode launch-time parameters for the fused HIP kernels that
//! the Oxidizer CLI can reference when baking `.grim` artifacts. They are pure
//! CPU-side data structures; runtime device execution lives in the parent
//! `grim-backend-rocm` crate.

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

/// Fusion configuration for QKV Projection + Attention operation.
///
/// `enabled` is the runtime gate for the fused QKV-attention kernel:
/// `RocmDevice::qkv_attention` only launches the kernel when this is `true`.
/// Default = `false`; flip to `true` after Step 4 tests pass. The field is
/// kept long-term so a regression can be gated off without an emergency patch.
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
        // Spec: `enabled` defaults to `false`. Detailed numeric defaults for
        // the GQA / launch geometry match the Phase-1 contract (a typical
        // 4:1 GQA Llama-style head layout); callers should always set
        // `enabled` explicitly anyway.
        Self {
            enabled: false,
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
    /// Launch geometry for Phase-1 QKV attention.
    ///
    /// Phase 1 contract (see `grim_qkv_attention_kernel_spec.md`):
    /// - One block per `(seq_position, head)` pair — flattened to a 2-D grid
    ///   where `grid.x = max_seq_len` (= `seq_len` for this call) and
    ///   `grid.y = num_heads`.
    /// - Block size picks a multiple of 64 on RDNA (Wave64 mandate for
    ///   gfx10xx / gfx11xx / gfx12xx); BLOCK_64 is the minimum 1-wave case,
    ///   BLOCK_256 covers 4 waves and keeps the head_dim dot product busy.
    /// - LDS budget stays under `ATTENTION_SHARED_MAX_BYTES` (32768). With
    ///   online softmax (running max + running weighted sum in registers),
    ///   we never materialize a kv-sized score buffer — shared memory only
    ///   needs to hold partial dot products for cross-thread combination.
    pub fn hip_launch_params(&self) -> HipKernelLaunch {
        // Per-head dimension of work, sized to keep one block per
        // `(seq_position, head)` pair. Smaller heads use 64, larger use 256.
        let block_dim_x = if self.wavefront_size == 32 { 128 } else { 256 };
        let grid_x = self.max_seq_len as u32;
        let grid_y = self.num_heads as u32;
        // 4 KB scratch for partial reductions is plenty for f32 head_dim <= 256.
        let shared_mem_bytes = (self.head_dim * 4).min(ATTENTION_SHARED_MAX_BYTES);
        HipKernelLaunch {
            grid_dim: hipDim3::new(grid_x, grid_y, 1),
            block_dim: hipDim3::new(block_dim_x, 1, 1),
            shared_mem_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// WI 2.4.4-2 — decode GEMM config (Rust-centric, replaces vendored CK wrapper).
//
// grim is Rust-centric: there is no `ck_gemm.cpp` and no `ck` cargo feature.
// The decode-shaped F16 GEMM lives in `kernels::decode_gemm::KERNEL_SOURCE`
// and is JIT-compiled at runtime through the same `hipModuleLoad` path
// every grim compute kernel uses. Dispatch from `RocmDevice::matmul` is
// gated by this config's `enabled` flag (default off), matching the
// `QkvAttentionFusionConfig::enabled` pattern from this same file.
//
// Per-plan gating rules (`grim_rocm_consumer_perf_plan.md` WI 2.4.4-2c):
//   - `enabled` must be `true` for dispatch to skip rocBLAS.
//   - dtype must be FP16 (CK-style kernel is F16-only; BF16/F32 are out of
//     scope per plan limits).
//   - `m <= 8` (the only decode M-slot the 8×64×64 tile is shaped for).
//   - Otherwise the kernel is skipped and rocBLAS handles the GEMM as today.
//
// perf gate (WI 2.6.4): the flag should NOT be flipped to `true` until a
// positive benchmark vs. rocBLAS is in hand. Plan §2.4.4-4 (SMALL-BATCH-MC)
// warns that double-buffered LDS can *reduce* decode throughput vs. plain
// rocBLAS when m is already tiny.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeGemmConfig {
    /// Runtime gate: `false` = always use rocBLAS, `true` = dispatch to the
    /// JIT'd `grim_decode_gemm_f16` kernel subject to the dtype/M filter
    /// in `RocmDevice::matmul`.
    pub enabled: bool,
    /// Wavefront size of the active arch. Tile geometry is the same for
    /// wave32 and wave64 (the kernel sizes the block to 256 and divides
    /// by `warpSize` at runtime), but this is recorded so a future
    /// architecture-specific tile resize hook (e.g. tile=128 on Wave32
    /// to keep occupancy) has the data it needs without an env lookup.
    pub wavefront_size: u32,
}

impl Default for DecodeGemmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            wavefront_size: 64,
        }
    }
}
