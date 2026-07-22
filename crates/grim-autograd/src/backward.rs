//! Reverse-mode tape autograd traversal (WI-T1 item 3).
//!
//! Iterates over the tape entries in reverse order, executing backward functions
//! for each recorded operation and accumulating gradients into `TrainableParams`.

use crate::ops::{add_backward, lora_backward, matmul_backward, scale_backward, AddArgs, MatMulArgs, ScaleArgs};
use crate::param::TrainableParams;
use crate::tape::{Tape, TapeMetadata, TensorId};
use grim_tensor::{error::{Error, Result}, Tensor};
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

    for entry in tape.iter_rev() {
        if !ctx.grads.contains_key(&entry.output) {
            continue;
        }

        let out_grad = ctx.get_grad(entry.output)?.clone();

        match &entry.metadata {
            TapeMetadata::LoRAApply { alpha, rank, a, b } => {
                let _base = tape.get(entry.inputs[0]).ok_or_else(|| Error::Backend("missing base tensor".into()))?;
                let x = tape.get(entry.inputs[1]).ok_or_else(|| Error::Backend("missing x tensor".into()))?;
                let a_t = tape.get(entry.inputs[2]).ok_or_else(|| Error::Backend("missing a tensor".into()))?;
                let b_t = tape.get(entry.inputs[3]).ok_or_else(|| Error::Backend("missing b tensor".into()))?;

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
            TapeMetadata::MatMul { transpose_a, transpose_b, .. } => {
                let a = tape.get(entry.inputs[0]).ok_or_else(|| Error::Backend("missing matmul input a".into()))?;
                let b = tape.get(entry.inputs[1]).ok_or_else(|| Error::Backend("missing matmul input b".into()))?;

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
                let args = AddArgs { out_grad: out_grad.clone() };
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
        }
    }

    Ok(ctx.grads)
}

fn accumulate_tensor_grad(grads: &mut HashMap<TensorId, Tensor>, id: TensorId, g: Tensor) -> Result<()> {
    if let Some(existing) = grads.get_mut(&id) {
        let dev = crate::pick_device_for_tensor(existing);
        let (sum_storage, handle) = grim_tensor::BackendDevice::add(
            &*dev,
            existing.storage().as_ref(),
            g.storage().as_ref(),
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
    use crate::param::ParamId;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    #[test]
    fn backward_accumulates_lora_gradients() {
        let mut tape = Tape::new();
        let mut params = TrainableParams::new();

        let base = tape.register(cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2])));
        let x = tape.register(cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2])));

        let pid_a = ParamId::a(0, 1);
        let pid_b = ParamId::b(0, 1);

        let a_data = cpu_tensor(vec![0.5, 0.5], Shape::new(vec![1, 2]));
        let b_data = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![2, 1]));

        let a_id = tape.register_param(pid_a, a_data.clone());
        let b_id = tape.register_param(pid_b, b_data.clone());

        params.insert(crate::param::TrainableParam::new(pid_a, a_data).unwrap());
        params.insert(crate::param::TrainableParam::new(pid_b, b_data).unwrap());

        let out = tape.record_lora_apply(
            base, x, a_id, b_id,
            cpu_tensor(vec![2.0, 3.0], Shape::new(vec![1, 2])),
            1.0, 1, pid_a, pid_b,
        );

        let loss_grad = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![1, 2]));
        let grads = backward(&tape, loss_grad, out, &mut params).unwrap();

        assert!(grads.contains_key(&base));
        assert!(grads.contains_key(&x));
        assert!(!params.get(pid_a).unwrap().grad().to_vec_f32().unwrap().is_empty());
    }
}
