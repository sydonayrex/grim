//! Reverse-mode tape autograd traversal (WI-T1 item 3).
//!
//! Iterates over the tape entries in reverse order, executing backward functions
//! for each recorded operation and accumulating gradients into `TrainableParams`.
//!
//! WI-X13: when gradient checkpointing dropped a segment's intermediates
//! (`Tape::free_intermediate_activations`), the affected segment is replayed
//! on demand into a per-pass overlay (see [`crate::replay`]). Every tensor
//! lookup consults the overlay before the tape, so checkpointed and
//! non-checkpointed runs take identical code paths.

use crate::ops::{
    AddArgs, MatMulArgs, ScaleArgs, add_backward, lora_backward, matmul_backward, scale_backward,
    silu_mul_backward,
};
use crate::param::TrainableParams;
use crate::replay::replay_segment;
use crate::tape::{Tape, TapeEntry, TapeMetadata, TensorId};
use grim_tensor::{
    Tensor,
    error::{Error, Result},
};
use std::collections::HashMap;

/// Context state during backward pass traversal.
#[derive(Debug)]
pub struct BackwardContext<'a> {
    pub tape: &'a Tape,
    pub grads: HashMap<TensorId, Tensor>,
}

impl<'a> BackwardContext<'a> {
    pub fn new(tape: &'a Tape, loss_grad: Tensor, loss_tensor_id: TensorId) -> Self {
        let mut grads = HashMap::new();
        grads.insert(loss_tensor_id, loss_grad);
        Self { tape, grads }
    }

    /// Retrieve gradient for a tensor ID, or return error if not present.
    pub fn get_grad(&self, id: TensorId) -> Result<&Tensor> {
        self.grads
            .get(&id)
            .ok_or_else(|| Error::Backend(format!("missing gradient for tensor {:?}", id)))
    }
}

/// Look a tensor up in the checkpoint-replay overlay first, then in the tape.
fn get_any<'a>(
    tape: &'a Tape,
    overlay: &'a HashMap<TensorId, Tensor>,
    id: TensorId,
) -> Option<&'a Tensor> {
    overlay.get(&id).or_else(|| tape.get(id))
}

/// WI-X13: if any tensor this entry needs was freed by checkpointing, replay
/// its whole segment once into `overlay`. Cheap no-op while every activation
/// is still resident (`checkpoint_segs <= 1`, or already replayed).
fn ensure_entry_resolved(
    tape: &Tape,
    entry: &TapeEntry,
    overlay: &mut HashMap<TensorId, Tensor>,
) -> Result<()> {
    let missing = entry
        .inputs
        .iter()
        .chain(std::iter::once(&entry.output))
        .any(|id| !overlay.contains_key(id) && tape.get(*id).is_none());
    if missing {
        replay_segment(tape, entry.segment_idx, overlay)?;
    }
    Ok(())
}

/// Drop a finished segment's replayed activations from the overlay to bound
/// peak memory. Live (retained) tensors stay untouched in the tape.
fn evict_segment_overlay(tape: &Tape, seg: usize, overlay: &mut HashMap<TensorId, Tensor>) {
    for e in tape.entries().iter().filter(|e| e.segment_idx == seg) {
        overlay.remove(&e.output);
    }
}

/// Execute reverse-mode autograd pass over `tape`, starting from `loss_grad` at `loss_tensor_id`.
///
/// Accumulates parameter gradients into `trainable_params`. Returns the complete map of intermediate tensor gradients.
pub fn backward(
    tape: &Tape,
    loss_grad: Tensor,
    loss_tensor_id: TensorId,
    trainable_params: &mut TrainableParams,
) -> Result<HashMap<TensorId, Tensor>> {
    let mut ctx = BackwardContext::new(tape, loss_grad, loss_tensor_id);
    // Checkpoint-replay overlay (WI-X13): recomputed activations live here.
    let mut overlay: HashMap<TensorId, Tensor> = HashMap::new();
    let mut active_segment: Option<usize> = None;

    for entry in tape.iter_rev() {
        if !ctx.grads.contains_key(&entry.output) {
            continue;
        }

        // Leaving segment S downward: release its replayed outputs.
        if let Some(prev_seg) = active_segment {
            if prev_seg != entry.segment_idx {
                evict_segment_overlay(tape, prev_seg, &mut overlay);
            }
        }
        ensure_entry_resolved(tape, entry, &mut overlay)?;
        active_segment = Some(entry.segment_idx);

        let out_grad = ctx.get_grad(entry.output)?.clone();

        match &entry.metadata {
            TapeMetadata::LoRAApply { alpha, rank, a, b } => {
                let _base = get_any(tape, &overlay, entry.inputs[0])
                    .ok_or_else(|| Error::Backend("missing base tensor".into()))?;
                let x = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing x tensor".into()))?;
                let a_t = get_any(tape, &overlay, entry.inputs[2])
                    .ok_or_else(|| Error::Backend("missing a tensor".into()))?;
                let b_t = get_any(tape, &overlay, entry.inputs[3])
                    .ok_or_else(|| Error::Backend("missing b tensor".into()))?;

                let scale = alpha / (*rank as f32);
                let (g_base, g_x, g_a, g_b) = lora_backward(&out_grad, x, a_t, b_t, scale)?;

                // Accumulate gradients into trainable params
                if let Some(param_a) = trainable_params.get_mut(*a) {
                    param_a.accumulate_grad(&g_a)?;
                }
                if let Some(param_b) = trainable_params.get_mut(*b) {
                    param_b.accumulate_grad(&g_b)?;
                }

                // Propagate gradients to inputs
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], g_base)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], g_x)?;
            }
            TapeMetadata::MatMul {
                transpose_a,
                transpose_b,
                ..
            } => {
                let a = get_any(tape, &overlay, entry.inputs[0])
                    .ok_or_else(|| Error::Backend("missing matmul input a".into()))?;
                let b = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing matmul input b".into()))?;

                let args = MatMulArgs {
                    a: a.clone(),
                    b: b.clone(),
                    out_grad: out_grad.clone(),
                    transpose_a: *transpose_a,
                    transpose_b: *transpose_b,
                };
                let (g_a, g_b) = matmul_backward(&args)?;

                if let Some(pid) = entry.param_id {
                    if let Some(param) = trainable_params.get_mut(pid) {
                        param.accumulate_grad(&g_a)?;
                    }
                }

                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], g_a)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], g_b)?;
            }
            TapeMetadata::Add => {
                let args = AddArgs {
                    out_grad: out_grad.clone(),
                };
                let (gl, gr) = add_backward(&args)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], gl)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], gr)?;
            }
            TapeMetadata::Scale { factor } => {
                let args = ScaleArgs {
                    input_grad: out_grad.clone(),
                    factor: *factor,
                };
                let g = scale_backward(&args)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], g)?;
            }
            TapeMetadata::SiluMul => {
                let gate = get_any(tape, &overlay, entry.inputs[0])
                    .ok_or_else(|| Error::Backend("missing silu_mul gate".into()))?;
                let up = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing silu_mul up".into()))?;
                let (d_gate, d_up) = silu_mul_backward(gate, up, &out_grad)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], d_gate)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], d_up)?;
            }
            TapeMetadata::RmsNorm { eps, weight_param } => {
                let x = get_any(tape, &overlay, entry.inputs[0])
                    .ok_or_else(|| Error::Backend("missing rmsnorm input x".into()))?;
                let weight = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing rmsnorm input weight".into()))?;

                let (dx, dw) = crate::ops::rmsnorm_backward(x, weight, &out_grad, *eps)?;

                if let Some(pid) = weight_param {
                    if let Some(param) = trainable_params.get_mut(*pid) {
                        param.accumulate_grad(&dw)?;
                    }
                }

                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], dx)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], dw)?;
            }
            TapeMetadata::Rope => {
                let cos = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing rope cos".into()))?;
                let sin = get_any(tape, &overlay, entry.inputs[2])
                    .ok_or_else(|| Error::Backend("missing rope sin".into()))?;

                let dx = crate::ops::rope_backward(&out_grad, cos, sin)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], dx)?;
            }
            TapeMetadata::Softmax => {
                let s = get_any(tape, &overlay, entry.output)
                    .ok_or_else(|| Error::Backend("missing softmax output".into()))?;
                let dx = crate::ops::softmax_backward(&out_grad, s)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], dx)?;
            }
            TapeMetadata::Embedding {
                token_ids,
                vocab_size,
                hidden_dim,
                weight_param,
            } => {
                let dw =
                    crate::ops::embedding_backward(&out_grad, token_ids, *vocab_size, *hidden_dim)?;

                if let Some(pid) = weight_param {
                    if let Some(param) = trainable_params.get_mut(*pid) {
                        param.accumulate_grad(&dw)?;
                    }
                }

                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], dw)?;
            }
        }
    }

    Ok(ctx.grads)
}

/// Execute reverse-mode autograd pass over `tape`, directly fusing backward gradient calculation
/// with optimizer parameter update stepping (LOMO style).
///
/// This eliminates the need to retain parameter gradient tensors in memory across the entire backward pass.
pub fn backward_step(
    tape: &Tape,
    loss_grad: Tensor,
    loss_tensor_id: TensorId,
    trainable_params: &mut TrainableParams,
    optimizer: &mut crate::adamw::Optimizer,
) -> Result<HashMap<TensorId, Tensor>> {
    let mut ctx = BackwardContext::new(tape, loss_grad, loss_tensor_id);
    // Checkpoint-replay overlay (WI-X13): recomputed activations live here.
    let mut overlay: HashMap<TensorId, Tensor> = HashMap::new();
    let mut active_segment: Option<usize> = None;
    // Audit fix (grim-models-adjacent pass): a parameter contributing through
    // MULTIPLE tape entries gets an optimizer step PER ENTRY here — partial
    // gradient stepped, zeroed, next partial stepped again. That silently
    // mis-trains tied/shared params (Adam moments update once per fragment,
    // not once per summed gradient). Fail loudly instead.
    let mut stepped_params: std::collections::HashSet<crate::param::ParamId> =
        std::collections::HashSet::new();

    for entry in tape.iter_rev() {
        if !ctx.grads.contains_key(&entry.output) {
            continue;
        }

        // Leaving segment S downward: release its replayed outputs.
        if active_segment.is_some_and(|s| s != entry.segment_idx) {
            evict_segment_overlay(tape, active_segment.unwrap(), &mut overlay);
        }
        ensure_entry_resolved(tape, entry, &mut overlay)?;
        active_segment = Some(entry.segment_idx);

        let out_grad = ctx.get_grad(entry.output)?.clone();

        match &entry.metadata {
            TapeMetadata::LoRAApply { alpha, rank, a, b } => {
                let _base = get_any(tape, &overlay, entry.inputs[0])
                    .ok_or_else(|| Error::Backend("missing base tensor".into()))?;
                let x = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing x tensor".into()))?;
                let a_t = get_any(tape, &overlay, entry.inputs[2])
                    .ok_or_else(|| Error::Backend("missing a tensor".into()))?;
                let b_t = get_any(tape, &overlay, entry.inputs[3])
                    .ok_or_else(|| Error::Backend("missing b tensor".into()))?;

                let scale = alpha / (*rank as f32);
                let (g_base, g_x, g_a, g_b) = lora_backward(&out_grad, x, a_t, b_t, scale)?;

                // Fused LOMO step: immediately step parameter and release gradient buffer
                if let Some(param_a) = trainable_params.get_mut(*a) {
                    if !stepped_params.insert(*a) {
                        return Err(Error::Backend(format!(
                            "backward_step: parameter {a:?} contributes through multiple tape \
                             entries; the fused path would step it once per fragment. Sum the \
                             gradients on the plain `backward` path instead."
                        )));
                    }
                    param_a.accumulate_grad(&g_a)?;
                    optimizer.step_param(*a, param_a)?;
                    param_a.zero_grad()?;
                }
                if let Some(param_b) = trainable_params.get_mut(*b) {
                    if !stepped_params.insert(*b) {
                        return Err(Error::Backend(format!(
                            "backward_step: parameter {b:?} contributes through multiple tape \
                             entries; the fused path would step it once per fragment. Sum the \
                             gradients on the plain `backward` path instead."
                        )));
                    }
                    param_b.accumulate_grad(&g_b)?;
                    optimizer.step_param(*b, param_b)?;
                    param_b.zero_grad()?;
                }

                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], g_base)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], g_x)?;
            }
            TapeMetadata::MatMul {
                transpose_a,
                transpose_b,
                ..
            } => {
                let a = get_any(tape, &overlay, entry.inputs[0])
                    .ok_or_else(|| Error::Backend("missing matmul input a".into()))?;
                let b = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing matmul input b".into()))?;

                let args = MatMulArgs {
                    a: a.clone(),
                    b: b.clone(),
                    out_grad: out_grad.clone(),
                    transpose_a: *transpose_a,
                    transpose_b: *transpose_b,
                };
                let (g_a, g_b) = matmul_backward(&args)?;

                if let Some(pid) = entry.param_id {
                    if let Some(param) = trainable_params.get_mut(pid) {
                        if !stepped_params.insert(pid) {
                            return Err(Error::Backend(format!(
                                "backward_step: parameter {pid:?} contributes through multiple tape \
                                 entries; the fused path would step it once per fragment. Sum the \
                                 gradients on the plain `backward` path instead."
                            )));
                        }
                        param.accumulate_grad(&g_a)?;
                        optimizer.step_param(pid, param)?;
                        param.zero_grad()?;
                    }
                }

                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], g_a)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], g_b)?;
            }
            TapeMetadata::Add => {
                let args = AddArgs {
                    out_grad: out_grad.clone(),
                };
                let (gl, gr) = add_backward(&args)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], gl)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], gr)?;
            }
            TapeMetadata::Scale { factor } => {
                let args = ScaleArgs {
                    input_grad: out_grad.clone(),
                    factor: *factor,
                };
                let g = scale_backward(&args)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], g)?;
            }
            TapeMetadata::SiluMul => {
                let gate = get_any(tape, &overlay, entry.inputs[0])
                    .ok_or_else(|| Error::Backend("missing silu_mul gate".into()))?;
                let up = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing silu_mul up".into()))?;
                let (d_gate, d_up) = silu_mul_backward(gate, up, &out_grad)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], d_gate)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], d_up)?;
            }
            TapeMetadata::RmsNorm { eps, weight_param } => {
                let x = get_any(tape, &overlay, entry.inputs[0])
                    .ok_or_else(|| Error::Backend("missing rmsnorm input x".into()))?;
                let weight = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing rmsnorm input weight".into()))?;

                let (dx, dw) = crate::ops::rmsnorm_backward(x, weight, &out_grad, *eps)?;

                if let Some(pid) = weight_param {
                    if let Some(param) = trainable_params.get_mut(*pid) {
                        if !stepped_params.insert(*pid) {
                            return Err(Error::Backend(format!(
                                "backward_step: parameter {pid:?} contributes through multiple tape \
                                 entries; the fused path would step it once per fragment. Sum the \
                                 gradients on the plain `backward` path instead."
                            )));
                        }
                        param.accumulate_grad(&dw)?;
                        optimizer.step_param(*pid, param)?;
                        param.zero_grad()?;
                    }
                }

                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], dx)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[1], dw)?;
            }
            TapeMetadata::Rope => {
                let cos = get_any(tape, &overlay, entry.inputs[1])
                    .ok_or_else(|| Error::Backend("missing rope cos".into()))?;
                let sin = get_any(tape, &overlay, entry.inputs[2])
                    .ok_or_else(|| Error::Backend("missing rope sin".into()))?;

                let dx = crate::ops::rope_backward(&out_grad, cos, sin)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], dx)?;
            }
            TapeMetadata::Softmax => {
                let s = get_any(tape, &overlay, entry.output)
                    .ok_or_else(|| Error::Backend("missing softmax output".into()))?;
                let dx = crate::ops::softmax_backward(&out_grad, s)?;
                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], dx)?;
            }
            TapeMetadata::Embedding {
                token_ids,
                vocab_size,
                hidden_dim,
                weight_param,
            } => {
                let dw =
                    crate::ops::embedding_backward(&out_grad, token_ids, *vocab_size, *hidden_dim)?;

                if let Some(pid) = weight_param {
                    if let Some(param) = trainable_params.get_mut(*pid) {
                        if !stepped_params.insert(*pid) {
                            return Err(Error::Backend(format!(
                                "backward_step: parameter {pid:?} contributes through multiple tape \
                                 entries; the fused path would step it once per fragment. Sum the \
                                 gradients on the plain `backward` path instead."
                            )));
                        }
                        param.accumulate_grad(&dw)?;
                        optimizer.step_param(*pid, param)?;
                        param.zero_grad()?;
                    }
                }

                accumulate_tensor_grad(&mut ctx.grads, entry.inputs[0], dw)?;
            }
        }
    }

    Ok(ctx.grads)
}

fn accumulate_tensor_grad(
    grads: &mut HashMap<TensorId, Tensor>,
    id: TensorId,
    g: Tensor,
) -> Result<()> {
    if let Some(existing) = grads.get_mut(&id) {
        let dev = crate::pick_device_for_tensor(existing);
        let g_storage = if g.device() == existing.device() {
            g.storage().clone()
        } else {
            std::sync::Arc::from(dev.from_cpu(
                &g.to_vec_f32()?,
                existing.shape(),
                grim_tensor::DType::F32,
            )?)
        };
        let (sum_storage, handle) = grim_tensor::CoreTensorOps::add(
            &*dev,
            existing.storage().as_ref(),
            g_storage.as_ref(),
            existing.shape(),
        )?;
        handle.synchronize()?;
        *existing = Tensor::new(
            std::sync::Arc::from(sum_storage),
            existing.shape().clone(),
            grim_tensor::DType::F32,
            existing.provenance().clone(),
            existing.device().clone(),
        );
    } else {
        grads.insert(id, g);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::injection::LoRAInjectionPoint;
    use crate::param::ParamId;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    #[test]
    fn backward_accumulates_lora_gradients() {
        let mut tape = Tape::new();
        let mut params = TrainableParams::new();

        let base = tape.register(cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2])));
        let x = tape.register(cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2])));

        let pid_a = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let pid_b = ParamId::b(0, 1, LoRAInjectionPoint::QProj);

        let a_data = cpu_tensor(vec![0.5, 0.5], Shape::new(vec![1, 2]));
        let b_data = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![2, 1]));

        let a_id = tape.register_param(pid_a, a_data.clone());
        let b_id = tape.register_param(pid_b, b_data.clone());

        params.insert(crate::param::TrainableParam::new(pid_a, a_data).unwrap());
        params.insert(crate::param::TrainableParam::new(pid_b, b_data).unwrap());

        let out = tape.record_lora_apply(
            base,
            x,
            a_id,
            b_id,
            cpu_tensor(vec![2.0, 3.0], Shape::new(vec![1, 2])),
            1.0,
            1,
            pid_a,
            pid_b,
        );

        let loss_grad = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2]));
        let grads = backward(&tape, loss_grad, out, &mut params).unwrap();

        assert!(grads.contains_key(&base));
        assert!(grads.contains_key(&x));
        assert!(
            !params
                .get(pid_a)
                .unwrap()
                .grad()
                .to_vec_f32()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn backward_step_accumulates_and_steps_optimizer() {
        let mut tape = Tape::new();
        let mut params = TrainableParams::new();

        let base = tape.register(cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2])));
        let x = tape.register(cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2])));

        let pid_a = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let pid_b = ParamId::b(0, 1, LoRAInjectionPoint::QProj);

        let a_data = cpu_tensor(vec![0.5, 0.5], Shape::new(vec![1, 2]));
        let b_data = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![2, 1]));

        let a_id = tape.register_param(pid_a, a_data.clone());
        let b_id = tape.register_param(pid_b, b_data.clone());

        params.insert(crate::param::TrainableParam::new(pid_a, a_data).unwrap());
        params.insert(crate::param::TrainableParam::new(pid_b, b_data).unwrap());

        let out = tape.record_lora_apply(
            base,
            x,
            a_id,
            b_id,
            cpu_tensor(vec![2.0, 3.0], Shape::new(vec![1, 2])),
            1.0,
            1,
            pid_a,
            pid_b,
        );

        let loss_grad = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2]));
        let mut optimizer =
            crate::adamw::Optimizer::new(crate::adamw::OptimizerKind::AdamW, 1e-3).unwrap();
        let initial_a = params.get(pid_a).unwrap().data.to_vec_f32().unwrap();

        let grads = backward_step(&tape, loss_grad, out, &mut params, &mut optimizer).unwrap();

        assert!(grads.contains_key(&base));
        let stepped_a = params.get(pid_a).unwrap().data.to_vec_f32().unwrap();
        assert_ne!(initial_a, stepped_a);
    }

    /// Audit gate: a parameter contributing through MULTIPLE tape entries
    /// must make `backward_step` fail loudly — stepping each fragment
    /// separately mis-trains tied/shared params.
    #[test]
    fn backward_step_refuses_multi_entry_param() {
        let mut tape = Tape::new();
        let mut params = TrainableParams::new();

        let x = tape.register(cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2])));
        let pid_a = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let pid_b = ParamId::b(0, 1, LoRAInjectionPoint::QProj);
        // SAME A matrix used by two CHAINED LoRA applications (weight tying):
        // out1 = f(x; A, B1), out2 = f(out1; A, B2), loss = out2 — so A
        // contributes through two entries that BOTH lie on the loss path.
        let a_data = cpu_tensor(vec![0.5, 0.5], Shape::new(vec![1, 2]));
        let b1_data = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![2, 1]));
        let b2_data = cpu_tensor(vec![2.0, 2.0], Shape::new(vec![2, 1]));

        let a_id = tape.register_param(pid_a, a_data.clone());
        let b1_id = tape.register_param(pid_b, b1_data.clone());
        let base1 = tape.register(cpu_tensor(vec![0.0, 0.0], Shape::new(vec![1, 2])));
        let out1 = tape.record_lora_apply(
            base1,
            x,
            a_id,
            b1_id,
            cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2])),
            1.0,
            1,
            pid_a,
            pid_b,
        );

        let b2_pid = ParamId::b(0, 2, LoRAInjectionPoint::VProj);
        let b2_id = tape.register_param(b2_pid, b2_data.clone());
        let base2 = tape.register(cpu_tensor(vec![0.0, 0.0], Shape::new(vec![1, 2])));
        let out2 = tape.record_lora_apply(
            base2,
            out1,
            a_id,
            b2_id,
            cpu_tensor(vec![2.0, 2.0], Shape::new(vec![1, 2])),
            1.0,
            1,
            pid_a,
            b2_pid,
        );

        params.insert(crate::param::TrainableParam::new(pid_a, a_data.clone()).unwrap());
        params.insert(crate::param::TrainableParam::new(pid_b, b1_data).unwrap());
        params.insert(crate::param::TrainableParam::new(b2_pid, b2_data).unwrap());

        let loss_grad = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2]));
        let mut optimizer =
            crate::adamw::Optimizer::new(crate::adamw::OptimizerKind::AdamW, 1e-3).unwrap();
        let res =
            crate::backward::backward_step(&tape, loss_grad, out2, &mut params, &mut optimizer);
        let err = match res {
            Err(e) => e,
            Ok(_) => panic!("backward_step must refuse a multi-entry parameter"),
        };
        assert!(
            err.to_string().contains("multiple tape entries"),
            "error should explain the refusal: {err}"
        );
    }
}
