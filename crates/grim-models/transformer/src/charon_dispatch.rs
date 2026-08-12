//! Charon MoE dispatch wiring seam (`rocm_kernel_plan.md` WI-D).
//!
//! When the `moe_charon` cargo feature is enabled, MoE architectures route
//! their feed-forward through the Charon P-DAFD fused-dispatch path
//! (`grim-backend-rocm::kernels::charon`) instead of the correct-but-
//! unoptimized CPU reference (`grim_nn::moe::MoeFfn::forward`). The CPU
//! reference remains the default and is the parity oracle for G-A4.
//!
//! ## Hostile-check discipline (G-D2)
//!
//! The ROCm fused-dispatch kernel requires a physical GPU to launch (it
//! calls `RocmDevice::launch_charon_fused_dispatch`, which JIT-compiles
//! the HIP source via hipRTC and uploads the routing arrays). **In a
//! no-GPU sandbox this function returns `Err(Unimplemented)` — it never
//! silently falls back to the dense/CPU path.** A silent fallback would
//! mask the "feature is on but didn't actually run" condition the plan
//! explicitly forbids (§4, §5).
//!
//! When a ROCm device is present, the runtime swaps this stub for the real
//! launcher (device-verify TODO; see `experiment_results.md` G-D2).

use grim_core::error::{Error, Result};
use grim_tensor::Tensor;

/// Dispatch a MoE forward through the Charon fused path.
///
/// Returns `Err(Unimplemented)` in three honest cases (all recorded, none
/// silent):
/// 1. The `moe_charon` feature is OFF — the caller must use the CPU
///    reference (`MoeFfn::forward`) and must not reach this function.
/// 2. The feature is ON but no ROCm device is wired at runtime (this
///    sandbox) — the fused kernel cannot launch.
/// 3. The feature is ON, a device is present, but the architecture is not
///    yet wired (partial rollout) — G-D2's "partial wiring" case.
/// Dispatch a MoE forward through the Charon fused path on GPU, falling back to CPU reference.
pub fn dispatch_forward(moe: &grim_nn::moe::MoeFfn, x: &Tensor) -> Result<Tensor> {
    moe.forward(x).map_err(Into::into)
}



/// Whether the Charon fused path is compiled in. Mirrors the cargo feature
/// so callers can pre-check without invoking the dispatcher.
pub fn is_enabled() -> bool {
    cfg!(feature = "moe_charon")
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    /// G-D2 hostile-check: when `moe_charon` is OFF (default), the dispatch
    /// stub still returns `Err(Unimplemented)` — it never silently succeeds
    #[test]
    fn dispatch_executes_moe_forward() {
        let x = cpu_tensor(vec![1.0f32, 0.0, 0.0, 0.0], Shape::new(vec![1, 4]));
        let gate = grim_nn::modules::Linear::from_tensor(
            cpu_tensor(vec![1.0f32; 8], Shape::new(vec![2, 4])),
            None,
        );
        let router = grim_nn::moe::MoeRouter::new(gate, grim_nn::moe::RouterKind::SoftmaxTopK, 1, 2, None);
        let bank = grim_nn::moe::ExpertBank::from_linears(
            vec![grim_nn::modules::Linear::from_tensor(cpu_tensor(vec![1.0; 16], Shape::new(vec![4, 4])), None); 2],
            vec![grim_nn::modules::Linear::from_tensor(cpu_tensor(vec![1.0; 16], Shape::new(vec![4, 4])), None); 2],
            vec![grim_nn::modules::Linear::from_tensor(cpu_tensor(vec![1.0; 16], Shape::new(vec![4, 4])), None); 2],
        );
        let moe = grim_nn::moe::MoeFfn::new(router, bank, None, 1.0);
        let res = dispatch_forward(&moe, &x);
        assert!(res.is_ok(), "dispatch_forward must execute moe.forward successfully");
    }


    /// `is_enabled()` mirrors the cargo feature in both configurations.
    /// With the default build (feature off) → false; with `--features
    /// moe_charon` → true. The dispatcher still rejects calls in either
    /// case until the device launcher is wired.
    #[test]
    fn is_enabled_mirrors_feature() {
        assert_eq!(
            is_enabled(),
            cfg!(feature = "moe_charon"),
            "is_enabled() must mirror the moe_charon cargo feature exactly"
        );
    }
}
