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

/// Controls which parameters are recorded on the backward tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutogradScope {
    /// Only LoRA adapter + base-weight deltas at injection points (current QLoRA behavior).
    LoRAOnly,
    /// All trainable parameters including frozen base weights (full fine-tuning WI-T8).
    /// Requires recording MatMul, Add, Scale ops for every weight matrix in the model.
    FullParameter,
}

impl Default for AutogradScope {
    fn default() -> Self {
        AutogradScope::LoRAOnly
    }
}

pub mod adamw;
pub mod backward;
pub mod collate;
pub mod injection;
pub mod loss;
pub mod lr_schedule;
pub mod ops;
pub mod param;
pub mod preference_loss;
pub mod registry;
pub mod soul_eater;
pub mod tape;

pub use soul_eater::{SoulEaterAdapter, SoulEaterOptimizer};

pub use adamw::{
    Adafactor, AdafactorConfig, AdamW, AdamWConfig, LRScheduler, Lion8Bit, Lion8BitConfig,
    Optimizer, OptimizerKind, PagedAdamW, PagedAdamWConfig,
};
pub use backward::{BackwardContext, backward};
pub use collate::{PackedBatch, TokenSequence, VarLenCollator};
pub use injection::{
    InjectionConfig, LoRAInjectionConfig, LoRAInjectionPoint, LoRAInjectionRegistry,
    loftq_initialize, pissa_initialize,
};
pub use loss::cross_entropy_loss;
pub use lr_schedule::CosineWarmupSchedule;
pub use ops::{
    AddArgs, FakeQuantInt4Args, MatMulArgs, ScaleArgs, add_backward, apply_and_record_lora,
    fake_quant_int4_backward, fake_quant_int4_forward, lora_backward, matmul_backward,
    scale_backward, vera_backward, vera_forward,
};
pub use param::{ParamId, TrainableParam, TrainableParams};
pub use preference_loss::{
    dpo_loss, dpo_loss_autograd, grpo_loss, grpo_loss_autograd, grpo_normalize_rewards, kto_loss,
    olora_orthogonality_penalty, orpo_odds_ratio_loss, orpo_odds_ratio_loss_autograd, simpo_loss,
};
pub use registry::AutogradRegistry;
pub use tape::{Tape, TapeEntry, TapeKind, TensorId};

use grim_tensor::{BackendDevice, Device, Tensor};

/// Pick the `BackendDevice` that matches the storage location of `x` so
/// arithmetic ops dispatch to GPU kernels when the tensor lives on a GPU.
/// Falls back to CPU if the requested backend is unavailable in this build.
/// Mirrors `grim_nn::modules::pick_device_for_tensor`.
pub fn pick_device_for_tensor(x: &Tensor) -> Box<dyn BackendDevice> {
    match x.device() {
        Device::Cpu => Box::new(grim_backend_cpu::CpuDevice::new()),
        #[cfg(feature = "cuda-mem")]
        Device::Cuda(ordinal) => {
            if let Ok(dev) = grim_backend_cuda::CudaDevice::new(*ordinal) {
                Box::new(dev)
            } else {
                Box::new(grim_backend_cpu::CpuDevice::new())
            }
        }
        #[cfg(feature = "rocm-mem")]
        Device::Rocm(ordinal) => {
            if let Ok(dev) = grim_backend_rocm::RocmDevice::try_new(*ordinal) {
                Box::new(dev)
            } else {
                Box::new(grim_backend_cpu::CpuDevice::new())
            }
        }
        #[cfg(feature = "vulkan-mem")]
        Device::Vulkan => Box::new(grim_backend_vulkan::VulkanDevice::new()),
        #[cfg(feature = "metal-mem")]
        Device::Metal(ordinal) => {
            if let Ok(dev) = grim_backend_metal::MetalDevice::new(*ordinal) {
                Box::new(dev)
            } else {
                Box::new(grim_backend_cpu::CpuDevice::new())
            }
        }
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
        let a = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let b = ParamId::b(0, 1, LoRAInjectionPoint::QProj);
        assert!(a.is_a);
        assert!(!b.is_a);
        assert_ne!(a, b);
    }

    #[test]
    fn trainable_param_initializes_zero_grad() {
        let id = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
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
        // Tape only records MatMul / Add / Scale / LoRAApply.
        let mut tape = Tape::new();
        let t = cpu_tensor(vec![1.0], Shape::new(vec![1]));
        let id = tape.register(t);
        tape.record_scale(id, cpu_tensor(vec![2.0], Shape::new(vec![1])), 2.0, None);
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.entries()[0].kind, TapeKind::Scale);
    }
}
