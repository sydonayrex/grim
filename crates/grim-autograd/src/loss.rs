//! Cross-entropy loss and backward gradient computation (WI-T5 item 2).
//!
//! Provides `cross_entropy_loss` returning `(loss_val, loss_grad_tensor)`.

use grim_tensor::{DType, Tensor, error::{Error, Result}};
use std::sync::Arc;

/// Compute cross-entropy loss and its backward gradient w.r.t logits.
///
/// `logits` has shape `[batch_size, vocab_size]`; `targets` has shape `[batch_size]`.
/// Returns `(loss_float, loss_grad_tensor)`. CONTRACT: target token IDs must be `< vocab_size`.
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Result<(f32, Tensor)> {
    let dims = logits.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Backend("logits tensor must be 2D [batch_size, vocab_size]".into()));
    }

    let batch_size = dims[0];
    let vocab_size = dims[1];

    if targets.len() != batch_size {
        return Err(Error::Backend(format!(
            "targets count ({}) must match batch_size ({})",
            targets.len(),
            batch_size
        )));
    }

    let logits_vec = logits.to_vec_f32()?;
    let mut grad_vec = vec![0.0f32; batch_size * vocab_size];
    let mut total_loss = 0.0f32;

    for b in 0..batch_size {
        let target_token = targets[b];
        if target_token >= vocab_size {
            return Err(Error::Backend(format!(
                "target token {} out of bounds for vocab_size {}",
                target_token, vocab_size
            )));
        }

        let row_start = b * vocab_size;
        let row_logits = &logits_vec[row_start..row_start + vocab_size];

        // Max trick for numerical stability
        let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut exp_logits = vec![0.0f32; vocab_size];

        for v in 0..vocab_size {
            let exp_val = (row_logits[v] - max_logit).exp();
            exp_logits[v] = exp_val;
            sum_exp += exp_val;
        }

        let log_sum_exp = max_logit + sum_exp.ln();
        let sample_loss = log_sum_exp - row_logits[target_token];
        total_loss += sample_loss;

        // Gradient dL/dLogits = (softmax - one_hot) / batch_size
        let inv_batch = 1.0 / (batch_size as f32);
        for v in 0..vocab_size {
            let prob = exp_logits[v] / sum_exp;
            let target_indicator = if v == target_token { 1.0f32 } else { 0.0f32 };
            grad_vec[row_start + v] = (prob - target_indicator) * inv_batch;
        }
    }

    let avg_loss = total_loss / (batch_size as f32);
    let dev = crate::pick_device_for_tensor(logits);
    let storage = dev.from_cpu(&grad_vec, logits.shape(), DType::F32)?;
    let grad_tensor = Tensor::new(
        Arc::from(storage),
        logits.shape().clone(),
        DType::F32,
        logits.provenance().clone(),
        logits.device().clone(),
    );

    Ok((avg_loss, grad_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    #[test]
    fn cross_entropy_loss_zero_when_confident_correct() {
        // Logits heavily favor index 0 for sample 0, and index 1 for sample 1
        let logits = cpu_tensor(
            vec![10.0, -10.0, -10.0, 10.0],
            Shape::new(vec![2, 2]),
        );
        let targets = vec![0, 1];
        let (loss, grad) = cross_entropy_loss(&logits, &targets).unwrap();
        assert!(loss < 1e-4);
        assert_eq!(grad.shape().dims(), &[2, 2]);
    }
}
