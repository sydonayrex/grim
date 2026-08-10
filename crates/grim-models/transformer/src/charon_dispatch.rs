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
pub fn dispatch_forward(_x: &Tensor) -> Result<Tensor> {
    // The feature flag itself routes the caller here; this stub always
    // errors because the device launcher is not wired in this build. The
    // real launcher (grim-backend-rocm) replaces this body when wired.
    Err(Error::Unimplemented(
        "charon_dispatch::dispatch_forward: moe_charon feature is enabled but the \
         ROCm fused-dispatch launcher is not wired in this build (no-GPU sandbox). \
         Use the CPU reference (grim_nn::moe::MoeFfn::forward), or run on a \
         gfx1036/gfx1200 box with grim-backend-rocm wired. See \
         experiment_results.md gate G-D2."
            .into(),
    ))
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
    /// or fabricates output. A caller reaching this path without the feature
    /// has a wiring bug, and the error surfaces it.
    #[test]
    fn dispatch_returns_unimplemented_when_not_wired() {
        let x = cpu_tensor(vec![1.0f32, 0.0, 0.0, 0.0], Shape::new(vec![1, 4]));
        let res = dispatch_forward(&x);
        assert!(
            matches!(res, Err(Error::Unimplemented(_))),
            "dispatch must return Err(Unimplemented), never a silent fallback"
        );
    }

    /// G-D2: the error message must name the feature and the device-verify
    /// TODO so a no-GPU session cannot mistake it for a generic backend
    /// error.
    #[test]
    fn dispatch_error_names_feature_and_device_todo() {
        let x = cpu_tensor(vec![1.0f32; 4], Shape::new(vec![1, 4]));
        let err = dispatch_forward(&x).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("moe_charon"), "error must name the feature flag");
        assert!(
            msg.contains("G-D2") || msg.contains("experiment_results"),
            "error must reference the device-verify TODO / evidence doc"
        );
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
