//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! Module root: fused-weight structs and submodule re-exports.

use crate::memory::storage::RocmStorage;

/// Fused QKV weight blob for the ROCm decode path (Item 1): the concatenated
/// Q8_0 weights `[n_q + 2·n_kv, hidden]` plus the per-head row counts needed to
/// slice the single GEMV output back into q/k/v.
pub struct FusedQkvWeights {
    pub storage: RocmStorage,
    pub n_q: usize,
    pub n_k: usize,
    pub n_v: usize,
    pub hidden: usize,
}

impl FusedQkvWeights {
    /// Total number of output columns = n_q + n_k + n_v (= n_q + 2·n_kv for GQA).
    pub fn n_total(&self) -> usize {
        self.n_q + self.n_k + self.n_v
    }
}

/// Fused Gate+Up weight blob for dense SwiGLU FFN (Phase 4c).
/// Concatenated Q8_0 weights `[n_gate + n_up, hidden]` for single-launch GEMV.
pub struct FusedGateUpWeights {
    pub storage: RocmStorage,
    pub n_gate: usize,
    pub n_up: usize,
    pub hidden: usize,
}

impl FusedGateUpWeights {
    pub fn n_total(&self) -> usize {
        self.n_gate + self.n_up
    }
}

pub mod autograd_ops;
pub mod core_tensor_ops;
pub mod dot_gemv;
pub mod elementwise_ops;
pub mod fused_ops;
pub mod fusion_ops;
pub mod gemm_launchers;
pub mod kernel_infra;
pub mod layer_elementwise;
pub mod optimizer_ops;
pub mod sampling_ops;

#[allow(unused_imports)] // flat public API: `device::compute::<method>`
pub use autograd_ops::*;
#[allow(unused_imports)]
pub use core_tensor_ops::*;
#[allow(unused_imports)]
pub use dot_gemv::*;
#[allow(unused_imports)]
pub use elementwise_ops::*;
#[allow(unused_imports)]
pub use fused_ops::*;
#[allow(unused_imports)]
pub use fusion_ops::*;
#[allow(unused_imports)]
pub use gemm_launchers::*;
#[allow(unused_imports)]
pub use kernel_infra::*;
#[allow(unused_imports)]
pub use layer_elementwise::*;
#[allow(unused_imports)]
pub use optimizer_ops::*;
#[allow(unused_imports)]
pub use sampling_ops::*;
