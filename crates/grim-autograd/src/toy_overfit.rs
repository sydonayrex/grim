//! CPU training-correctness smoke test (salamander.md P0.1 / WI-F4 gate).
//!
//! A tiny two-stage LoRA network must overfit a fixed `(input, target)`
//! through the *real* autograd tape, LoRA backward, cross-entropy, and the
//! AdamW optimizer. Proves the training stack learns end-to-end.
//!
//! The tape is bookkeeping-only: `record_lora_apply` takes the *precomputed*
//! LoRA output, so this test recomputes the forward each step via
//! `BackendDevice::lora_accumulate` (the same device kernel `train.rs` uses).

#![cfg(test)]

use crate::adamw::OptimizerKind;
use crate::injection::LoRAInjectionPoint;
use crate::param::{ParamId, TrainableParam, TrainableParams};
use crate::tape::Tape;
use crate::{Optimizer, backward, cross_entropy_loss};
use grim_backend_cpu::CpuDevice;
use grim_tensor::{AutogradOps, DType, Shape, Tensor};
use std::sync::Arc;

fn vec2(rows: usize, cols: usize, vals: Vec<f32>) -> Tensor {
    assert_eq!(vals.len(), rows * cols);
    grim_backend_cpu::cpu_tensor(vals, Shape::new(vec![rows, cols]))
}

fn init2(rows: usize, cols: usize, seed: i32) -> Tensor {
    let vals: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let h = (i as i32).wrapping_mul(374761393).wrapping_add(seed) as u32;
            0.1 * ((h % 1000) as f32 / 1000.0 - 0.5)
        })
        .collect();
    vec2(rows, cols, vals)
}

/// Run one LoRA stage forward `y = base + scale * (x @ A^T) @ B^T` on CPU,
/// recording it on `tape` exactly as `train.rs` would.
#[allow(clippy::too_many_arguments)]
fn lora_stage(
    tape: &mut Tape,
    dev: &CpuDevice,
    base_id: crate::tape::TensorId,
    x_id: crate::tape::TensorId,
    base: &Tensor,
    x: &Tensor,
    a_id: crate::tape::TensorId,
    b_id: crate::tape::TensorId,
    pa: ParamId,
    pb: ParamId,
) -> crate::tape::TensorId {
    let shape_out = base.shape().dims().to_vec();
    let (out_storage, _handle) = dev
        .lora_accumulate(
            base.storage().as_ref(),
            x.storage().as_ref(),
            tape.get(a_id).unwrap().storage().as_ref(),
            tape.get(b_id).unwrap().storage().as_ref(),
            1.0,
            &Shape::new(shape_out),
        )
        .unwrap();
    let out = Tensor::new(
        Arc::from(out_storage),
        base.shape().clone(),
        DType::F32,
        base.provenance().clone(),
        base.device().clone(),
    );
    tape.record_lora_apply(base_id, x_id, a_id, b_id, out, 1.0, 1, pa, pb)
}

#[test]
fn toy_overfit_loss_decreases() {
    let input = vec2(1, 4, vec![1.0, 0.5, -0.3, 0.8]);
    let target = vec![2usize];
    let lr = 0.1f32;
    let steps = 400usize;
    let dev = CpuDevice::new();

    let (pa1, pb1) = (
        ParamId::a(0, 1, LoRAInjectionPoint::QProj),
        ParamId::b(0, 1, LoRAInjectionPoint::QProj),
    );
    let (pa2, pb2) = (
        ParamId::a(1, 1, LoRAInjectionPoint::QProj),
        ParamId::b(1, 1, LoRAInjectionPoint::QProj),
    );

    let mut params = TrainableParams::new();
    // Stage 1: A1 [rank=2, in=4], B1 [out=4, rank=2].
    params.insert(TrainableParam::new(pa1, init2(2, 4, 1)).unwrap());
    params.insert(TrainableParam::new(pb1, init2(4, 2, 2)).unwrap());
    // Stage 2: A2 [rank=2, in=4], B2 [out=5, rank=2].
    params.insert(TrainableParam::new(pa2, init2(2, 4, 3)).unwrap());
    params.insert(TrainableParam::new(pb2, init2(5, 2, 4)).unwrap());

    let mut optimizer = Optimizer::new(OptimizerKind::AdamW, lr).unwrap();
    let mut first_loss = 0.0f32;

    for step in 0..steps {
        let mut tape = Tape::new();

        let base1 = vec2(1, 4, vec![0.0; 4]);
        let base2 = vec2(1, 5, vec![0.0; 5]);
        let base1_id = tape.register(base1.clone());
        let base2_id = tape.register(base2.clone());
        let x_id = tape.register(input.clone());

        let a1_id = tape.register_param(pa1, params.get(pa1).unwrap().data.clone());
        let b1_id = tape.register_param(pb1, params.get(pb1).unwrap().data.clone());
        let a2_id = tape.register_param(pa2, params.get(pa2).unwrap().data.clone());
        let b2_id = tape.register_param(pb2, params.get(pb2).unwrap().data.clone());

        // True LoRA forward on CPU (lora_accumulate), recorded on the tape.
        let h_id = lora_stage(
            &mut tape, &dev, base1_id, x_id, &base1, &input, a1_id, b1_id, pa1, pb1,
        );
        let h = tape.get(h_id).unwrap().clone();
        let logits_id = lora_stage(
            &mut tape, &dev, base2_id, h_id, &base2, &h, a2_id, b2_id, pa2, pb2,
        );
        let logits = tape.get(logits_id).unwrap().clone();

        let (loss, loss_grad) = cross_entropy_loss(&logits, &target).unwrap();
        if step == 0 {
            first_loss = loss;
        }

        backward(&tape, loss_grad, logits_id, &mut params).unwrap();
        optimizer.step(&mut params).unwrap();
        params.zero_all_grads().unwrap();

        if step == steps - 1 {
            assert!(
                loss < first_loss * 0.5,
                "overfit failed: loss {first_loss:.4} -> {loss:.4} (expected < {:.4})",
                first_loss * 0.5,
            );
        }
    }
}
