//! `grim-autograd` — Scoped autograd for adapter-only backward pass.
//!
//! WI-T1 of the grim training plan (`grim_party_plan.md`). This crate
//! provides a minimal reverse-mode autodiff engine specifically designed
//! for LoRA/QLoRA training where the base model weights are frozen. Only
//! the LoRA adapter parameters (A/B matrices) require gradients.
//!
//! # Architectural thesis
//!
//! Unsloth's core trick: never materialize the full unquantized model in
//! VRAM. Frozen base weights stay quantized; only LoRA adapters + optimizer
//! state are kept in full precision; dequantization happens fused,
//! just-in-time, per-op, and is thrown away immediately after use.
//!
//! This crate implements the *bookkeeping* half of that story: a small,
//! purpose-built reverse-mode tape over just the trainable path. It is much
//! easier to make correct and fast on ROCm than a general-purpose autodiff
//! engine (à la PyTorch), and is the only thing QLoRA needs.
//!
//! # Scope limits (from §WI-T1)
//!
//! - No autodiff for the frozen base weights — that is WI-T8's problem.
//! - No reimplementing `grim-tensor-graph`'s fusion IR; that's a different shape.
//! - No reaching into `grim-backend-rocm` kernel internals — goes through
//!   `BackendDevice` like existing forward code.
//!
//! # Op set
//!
//! Only the ops touching adapter parameters during forward are recorded:
//! - `MatMul` (the linear layer, the LoRA A, the LoRA B),
//! - `Add` (LoRA delta added into the frozen base output, trivially routes gradient),
//! - `Scale` (the α/r factor).
//!
//! Backward for this exact op set is implemented; nothing more. Cross-entropy
//! loss backward arrives with WI-T5 (it slots in as one more op).

pub mod injection;
pub mod ops;
pub mod param;
pub mod tape;
pub mod backward;
pub mod registry;
pub mod adamw;
pub mod loss;
pub mod preference_loss;

pub use injection::{LoRAInjectionPoint, LoRAInjectionConfig, InjectionConfig, LoRAInjectionRegistry};
pub use ops::{MatMulArgs, AddArgs, ScaleArgs, matmul_backward, add_backward, scale_backward, lora_backward};
pub use param::{ParamId, TrainableParam, TrainableParams};
pub use tape::{Tape, TapeEntry, TensorId, TapeKind};
pub use backward::{backward, BackwardContext};
pub use registry::AutogradRegistry;
pub use adamw::{AdamW, AdamWConfig};
pub use loss::cross_entropy_loss;
pub use preference_loss::{dpo_loss, orpo_odds_ratio_loss, grpo_normalize_rewards};

use grim_tensor::{BackendDevice, Device, Tensor};

/// Pick the `BackendDevice` that matches the storage location of `x` so
/// arithmetic ops dispatch to GPU kernels when the tensor lives on a GPU.
/// Falls back to CPU if the requested backend is unavailable in this build.
/// Mirrors `grim_nn::modules::pick_device_for_tensor`.
pub fn pick_device_for_tensor(x: &Tensor) -> Box<dyn BackendDevice> {
    match x.device() {
        Device::Cpu => Box::new(grim_backend_cpu::CpuDevice::new()),
        #[cfg(feature = "cuda-mem")]
        Device::Cuda(ordinal) => Box::new(grim_backend_cuda::CudaDevice::new(*ordinal)),
        #[cfg(feature = "rocm-mem")]
        Device::Rocm(ordinal) => Box::new(grim_backend_rocm::RocmDevice::new(*ordinal)),
        _ => Box::new(grim_backend_cpu::CpuDevice::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    #[test]
    fn param_id_distinguishes_a_and_b() {
        let a = ParamId::a(0, 1);
        let b = ParamId::b(0, 1);
        assert!(a.is_a);
        assert!(!b.is_a);
        assert_ne!(a, b);
    }

    #[test]
    fn trainable_param_initializes_zero_grad() {
        let id = ParamId::a(0, 1);
        let data = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let param = TrainableParam::new(id, data).unwrap();
        let g = param.grad().to_vec_f32().unwrap();
        assert!(g.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn standard_qlora_has_seven_injection_points() {
        assert_eq!(LoRAInjectionPoint::all_standard_qlora().len(), 7);
    }

    #[test]
    fn injection_point_attention_vs_mlp_classification() {
        assert!(LoRAInjectionPoint::QProj.is_attention());
        assert!(!LoRAInjectionPoint::QProj.is_mlp());
        assert!(LoRAInjectionPoint::GateProj.is_mlp());
        assert!(!LoRAInjectionPoint::GateProj.is_attention());
    }

    #[test]
    fn op_set_only_records_adapter_touching_ops() {
        // Tape only records MatMul / Add / Scale (the plan's "exactly this op set").
        let mut tape = Tape::new();
        let t = cpu_tensor(vec![1.0], Shape::new(vec![1]));
        let id = tape.register(t);
        tape.record_scale(id, cpu_tensor(vec![2.0], Shape::new(vec![1])), 2.0, None);
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.entries()[0].kind, TapeKind::Scale);
    }
}
