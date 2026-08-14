//! Cross-entropy loss and backward gradient computation (WI-T5 item 2).
//!
//! Provides `cross_entropy_loss` returning `(loss_val, loss_grad_tensor)`.

use grim_tensor::{
    BackendDevice, DType, Shape, Storage, Tensor,
    error::{Error, Result},
};
use std::sync::Arc;

/// Compute cross-entropy loss and its backward gradient w.r.t logits.
///
/// `logits` has shape `[batch_size, vocab_size]`; `targets` has shape `[batch_size]`.
/// Returns `(loss_float, loss_grad_tensor)`. CONTRACT: target token IDs must be `< vocab_size`.

/// Compute Fused Linear Cross Entropy without allocating full [B, V] logits tensor.
///
/// Multiplies hidden states `hidden` [batch_size, hidden_dim] by LM head `lm_head` [vocab_size, hidden_dim]
/// in chunks of `chunk_size` tokens, computing online cross entropy loss and gradient w.r.t `hidden`.
pub fn fused_linear_cross_entropy_loss(
    hidden: &Tensor,
    lm_head: &Tensor,
    targets: &[usize],
    chunk_size: usize,
) -> Result<(f32, Tensor)> {
    let h_dims = hidden.shape().dims();
    let w_dims = lm_head.shape().dims();
    if h_dims.len() != 2 || w_dims.len() != 2 {
        return Err(Error::Backend("hidden and lm_head must be 2D".into()));
    }

    let batch_size = h_dims[0];
    let hidden_dim = h_dims[1];
    let vocab_size = w_dims[0];
    if w_dims[1] != hidden_dim {
        return Err(Error::Backend("lm_head hidden_dim mismatch".into()));
    }
    if batch_size == 0 {
        return Err(Error::Backend("batch_size must be > 0".into()));
    }
    if targets.len() != batch_size {
        return Err(Error::Backend("targets length mismatch".into()));
    }

    if let grim_tensor::Device::Rocm(ordinal) = hidden.device() {
        for &target in targets {
            if target >= vocab_size {
                return Err(Error::Backend(format!("target_token {} >= vocab_size {}", target, vocab_size)));
            }
        }
        let dev = grim_backend_rocm::RocmDevice::try_new(*ordinal)?;
        let target_bytes: Vec<u8> = targets.iter().flat_map(|&v| (v as u32).to_ne_bytes()).collect();
        let target_dtype = DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native };
        let target_storage = dev.from_cpu_bytes(&target_bytes, &Shape::new(vec![batch_size]), target_dtype)?;
        let (loss_out, lse_out, _forward_handle) = dev.fused_linear_cross_entropy_forward(
            &**hidden.storage(), &**lm_head.storage(), &*target_storage, 4096,
        )?;
        let loss_sum: f32 = loss_out.to_cpu_vec_f32()?.iter().sum();
        let (grad_storage, _backward_handle) = dev.fused_linear_cross_entropy_backward(
            &**hidden.storage(), &**lm_head.storage(), &*target_storage, &*lse_out,
            4096, 1.0 / batch_size as f32,
        )?;
        let grad_tensor = Tensor::new(
            Arc::from(grad_storage), hidden.shape().clone(), DType::F32,
            hidden.provenance().clone(), hidden.device().clone(),
        );
        return Ok((loss_sum / batch_size as f32, grad_tensor));
    }

    let h_vec = hidden.to_vec_f32()?;
    let w_vec = lm_head.to_vec_f32()?;

    let mut grad_h = vec![0.0f32; batch_size * hidden_dim];
    let mut total_loss = 0.0f32;

    let chunk = chunk_size.max(1);
    let inv_b = 1.0 / (batch_size as f32);

    for chunk_start in (0..batch_size).step_by(chunk) {
        let chunk_end = (chunk_start + chunk).min(batch_size);

        for b in chunk_start..chunk_end {
            let target_token = targets[b];
            if target_token >= vocab_size {
                return Err(Error::Backend(format!(
                    "target_token {} >= vocab_size {}",
                    target_token, vocab_size
                )));
            }

            let h_row = &h_vec[b * hidden_dim..(b + 1) * hidden_dim];

            // Pass 1: Online LogSumExp & target logit extraction over vocabulary tiles (4096 tokens)
            let v_chunk_size = 4096.min(vocab_size);
            let mut max_logit = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            let mut target_logit = 0.0f32;

            for v_start in (0..vocab_size).step_by(v_chunk_size) {
                let v_end = (v_start + v_chunk_size).min(vocab_size);
                for v in v_start..v_end {
                    let w_row = &w_vec[v * hidden_dim..(v + 1) * hidden_dim];
                    let mut logit = 0.0f32;
                    for d in 0..hidden_dim {
                        logit += h_row[d] * w_row[d];
                    }
                    if v == target_token {
                        target_logit = logit;
                    }
                    if logit > max_logit {
                        let scale = (max_logit - logit).exp();
                        sum_exp = sum_exp * scale + 1.0f32;
                        max_logit = logit;
                    } else {
                        sum_exp += (logit - max_logit).exp();
                    }
                }
            }

            let log_sum_exp = max_logit + sum_exp.ln();
            let sample_loss = log_sum_exp - target_logit;
            total_loss += sample_loss;

            // Pass 2: Online gradient accumulation over vocabulary tiles
            let grad_h_row = &mut grad_h[b * hidden_dim..(b + 1) * hidden_dim];
            for v_start in (0..vocab_size).step_by(v_chunk_size) {
                let v_end = (v_start + v_chunk_size).min(vocab_size);
                for v in v_start..v_end {
                    let w_row = &w_vec[v * hidden_dim..(v + 1) * hidden_dim];
                    let mut logit = 0.0f32;
                    for d in 0..hidden_dim {
                        logit += h_row[d] * w_row[d];
                    }
                    let p = (logit - max_logit).exp() / sum_exp;
                    let target_ind = if v == target_token { 1.0f32 } else { 0.0f32 };
                    let d_logits = (p - target_ind) * inv_b;

                    for d in 0..hidden_dim {
                        grad_h_row[d] += d_logits * w_row[d];
                    }
                }
            }
        }
    }

    let avg_loss = total_loss / (batch_size as f32);
    let dev = crate::pick_device_for_tensor(hidden);
    let storage = dev.from_cpu(&grad_h, hidden.shape(), DType::F32)?;
    let grad_tensor = Tensor::new(
        Arc::from(storage),
        hidden.shape().clone(),
        DType::F32,
        hidden.provenance().clone(),
        hidden.device().clone(),
    );

    Ok((avg_loss, grad_tensor))
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Result<(f32, Tensor)> {
    let dims = logits.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Backend(
            "logits tensor must be 2D [batch_size, vocab_size]".into(),
        ));
    }

    let batch_size = dims[0];
    let vocab_size = dims[1];

    if batch_size == 0 {
        return Err(Error::Backend("batch_size must be > 0".into()));
    }
    if targets.len() != batch_size {
        return Err(Error::Backend(format!(
            "targets count ({}) must match batch_size ({})",
            targets.len(),
            batch_size
        )));
    }

    if let grim_tensor::Device::Rocm(ordinal) = logits.device() {
        match grim_backend_rocm::RocmDevice::try_new(*ordinal) {
            Ok(dev) => {
                match dev.cross_entropy_gpu(&**logits.storage(), targets, None) {
                    Ok((avg_loss, grad_storage)) => {
                        let grad_tensor = Tensor::new(
                            Arc::from(grad_storage),
                            logits.shape().clone(),
                            logits.dtype(),
                            logits.provenance().clone(),
                            logits.device().clone(),
                        );
                        return Ok((avg_loss, grad_tensor));
                    }
                    Err(_e) => {
                        // GPU cross-entropy failed; fall through to CPU reference path.
                    }
                }
            }
            Err(_e) => {
                // No ROCm device available at this ordinal; fall through to CPU.
            }
        }
    }

    let logits_vec = logits.to_vec_f32()?;
    if logits_vec.len() < batch_size * vocab_size {
        return Err(Error::Backend(format!(
            "logits tensor length ({}) is less than required batch_size * vocab_size ({})",
            logits_vec.len(),
            batch_size * vocab_size
        )));
    }
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

    #[test]
    fn test_fused_linear_cross_entropy_loss_matches_unfused() {
        let hidden = cpu_tensor(vec![0.5f32, 0.2, -0.1, 0.8], Shape::new(vec![2, 2]));
        let lm_head = cpu_tensor(vec![1.0f32, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let targets = vec![0, 1];
        let (fused_loss, fused_grad) =
            fused_linear_cross_entropy_loss(&hidden, &lm_head, &targets, 1).unwrap();
        assert!(fused_loss > 0.0);
        assert_eq!(fused_grad.shape().dims(), &[2, 2]);
    }
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    #[test]
    fn cross_entropy_loss_zero_when_confident_correct() {
        // Logits heavily favor index 0 for sample 0, and index 1 for sample 1
        let logits = cpu_tensor(vec![10.0, -10.0, -10.0, 10.0], Shape::new(vec![2, 2]));
        let targets = vec![0, 1];
        let (loss, grad) = cross_entropy_loss(&logits, &targets).unwrap();
        assert!(loss < 1e-4);
        assert_eq!(grad.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_cross_entropy_loss_hand_calculated() {
        let logits = cpu_tensor(vec![1.0f32, 2.0], Shape::new(vec![1, 2]));
        let targets = vec![1];
        let (loss, grad) = cross_entropy_loss(&logits, &targets).expect("cross entropy");

        // Softmax(1.0, 2.0) = [0.2689414, 0.7310586]
        // Loss = -ln(0.7310586) = 0.3132617
        assert!(
            (loss - 0.3132617).abs() < 1e-5,
            "loss = {}, want 0.3132617",
            loss
        );

        // Grad: p_0 / 1 = 0.2689414, (p_1 - 1) / 1 = -0.2689414
        let g = grad.to_vec_f32().expect("to vec");
        assert!(
            (g[0] - 0.2689414).abs() < 1e-5,
            "g[0] = {}, want 0.2689414",
            g[0]
        );
        assert!(
            (g[1] - (-0.2689414)).abs() < 1e-5,
            "g[1] = {}, want -0.2689414",
            g[1]
        );
    }
}
