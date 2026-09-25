//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! Module root: fused-weight structs and submodule re-exports.

use std::sync::Arc;

use grim_tensor::backend::BackendStorage;

use crate::memory::storage::RocmStorage;
use crate::memory::view::RocmStorageView;

/// Fused QKV weight blob for the ROCm decode path (Item 1): the concatenated
/// Q8_0 weights `[n_q + n_k + n_v + n_gb + n_gw + n_gf, hidden]` plus the row
/// counts needed to slice the single GEMV output back into q/k/v/gates.
pub struct FusedQkvWeights {
    pub storage: RocmStorage,
    pub n_q: usize,
    pub n_k: usize,
    pub n_v: usize,
    pub n_gb: usize,
    pub n_gw: usize,
    pub n_gf: usize,
    pub hidden: usize,
}

impl FusedQkvWeights {
    /// Total number of Q/K/V output columns (= n_q + 2·n_kv for GQA).
    pub fn n_total(&self) -> usize {
        self.n_q + self.n_k + self.n_v
    }

    /// Total number of output columns including the three gate projections.
    pub fn n_total_with_gates(&self) -> usize {
        self.n_total() + self.n_gb + self.n_gw + self.n_gf
    }

    /// Row offset of the erase-gate projection in the fused blob/output.
    pub fn gb_offset(&self) -> usize {
        self.n_total()
    }

    /// Row offset of the write-gate projection in the fused blob/output.
    pub fn gw_offset(&self) -> usize {
        self.gb_offset() + self.n_gb
    }

    /// Row offset of the decay-gate projection in the fused blob/output.
    pub fn gf_offset(&self) -> usize {
        self.gw_offset() + self.n_gw
    }

    /// Row offsets for `(gb, gw, gf)` in the fused blob/output.
    pub fn gate_offsets(&self) -> (usize, usize, usize) {
        (self.gb_offset(), self.gw_offset(), self.gf_offset())
    }
}

/// Gate-logit views produced by one fused QKV+gate GEMV.
///
/// All views borrow the same output allocation; dropping a view never frees the
/// parent buffer.
pub struct FusedQkvGateLogits {
    pub output: Arc<dyn BackendStorage>,
    pub gb: RocmStorageView,
    pub gw: RocmStorageView,
    pub gf: RocmStorageView,
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

/// Fused Q4_K Gate+Up weight blob for a local tensor-parallel shard.
///
/// Q4_K dot4 kernels are not valid on RDNA3/RDNA4 because their scale
/// shuffle is architecture-specific. This blob therefore feeds the existing
/// view-safe Q4_K fused-dequant/WMMA path rather than enabling the unsafe
/// dot4 route.
pub struct FusedGateUpQ4KWeights {
    pub storage: RocmStorage,
    pub n_gate: usize,
    pub n_up: usize,
    pub hidden: usize,
}

impl FusedGateUpQ4KWeights {
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
pub mod gla_launchers;
pub mod gla_mega_launchers;
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
pub use gla_launchers::*;
#[allow(unused_imports)]
pub use kernel_infra::*;
#[allow(unused_imports)]
pub use layer_elementwise::*;
#[allow(unused_imports)]
pub use optimizer_ops::*;
#[allow(unused_imports)]
pub use sampling_ops::*;
