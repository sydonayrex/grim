//! DPO, ORPO, and GRPO preference optimization loss functions (WI-T7).
//!
//! Provides loss routines for alignment fine-tuning on top of scoped autograd:
//! - `dpo_loss`: Direct Preference Optimization loss.
//! - `orpo_loss`: Odds Ratio Preference Optimization loss.
//! - `grpo_normalize_rewards`: Group-relative reward normalization for GRPO.
//! - `olora_orthogonality_penalty`: OLoRA regularization loss.

use grim_tensor::error::{Error, Result};

/// Compute Direct Preference Optimization (DPO) loss.
///
/// Inputs:
/// - `policy_chosen_logps`: `log π_θ(y_w | x)`
/// - `policy_rejected_logps`: `log π_θ(y_l | x)`
/// - `ref_chosen_logps`: `log π_ref(y_w | x)`
/// - `ref_rejected_logps`: `log π_ref(y_l | x)`
/// - `beta`: scaling parameter (e.g. `0.1`)
///
/// Returns `(loss_float, chosen_rewards, rejected_rewards)`.
pub fn dpo_loss(
    policy_chosen_logps: &[f32],
    policy_rejected_logps: &[f32],
    ref_chosen_logps: &[f32],
    ref_rejected_logps: &[f32],
    beta: f32,
) -> Result<(f32, Vec<f32>, Vec<f32>)> {
    let n = policy_chosen_logps.len();
    if policy_rejected_logps.len() != n
        || ref_chosen_logps.len() != n
        || ref_rejected_logps.len() != n
    {
        return Err(Error::Backend("DPO logps slice length mismatch".into()));
    }

    let mut total_loss = 0.0f32;
    let mut chosen_rewards = Vec::with_capacity(n);
    let mut rejected_rewards = Vec::with_capacity(n);

    for i in 0..n {
        let chosen_logr = policy_chosen_logps[i] - ref_chosen_logps[i];
        let rejected_logr = policy_rejected_logps[i] - ref_rejected_logps[i];

        let chosen_r = beta * chosen_logr;
        let rejected_r = beta * rejected_logr;

        chosen_rewards.push(chosen_r);
        rejected_rewards.push(rejected_r);

        let logits = chosen_r - rejected_r;
        // `-sigmoid(logits).ln()` == `ln(1 + exp(-logits))` == softplus(-logits),
        // but the direct form underflows to ±inf for |logits| > ~88 (sigmoid
        // saturates to 0/1, then ln(0) = -inf). Use the numerically-stable
        // softplus with the max trick: `max(-x,0) + ln(1+exp(-|x|))`, which is
        // exact for all finite inputs and never produces inf/NaN here.
        let loss = softplus(-logits);
        total_loss += loss;
    }

    let avg_loss = total_loss / (n as f32);
    Ok((avg_loss, chosen_rewards, rejected_rewards))
}

/// Compute Odds Ratio Preference Optimization (ORPO) odds ratio loss.
///
/// `policy_chosen_logps` and `policy_rejected_logps` are averaged log probabilities of chosen and rejected tokens.
/// Returns `loss_float`.
pub fn orpo_odds_ratio_loss(
    policy_chosen_logps: &[f32],
    policy_rejected_logps: &[f32],
    lambda: f32,
) -> Result<f32> {
    let n = policy_chosen_logps.len();
    if policy_rejected_logps.len() != n {
        return Err(Error::Backend("ORPO logps length mismatch".into()));
    }

    let mut total_loss = 0.0f32;
    for i in 0..n {
        let p_chosen = policy_chosen_logps[i].exp().clamp(1e-7, 1.0 - 1e-7);
        let p_rejected = policy_rejected_logps[i].exp().clamp(1e-7, 1.0 - 1e-7);

        let odds_chosen = p_chosen / (1.0 - p_chosen);
        let odds_rejected = p_rejected / (1.0 - p_rejected);

        let log_odds_ratio = (odds_chosen / odds_rejected).ln();
        // See dpo_loss: `-sigmoid(x).ln()` == softplus(-x), numerically stable.
        let loss = softplus(-log_odds_ratio);
        total_loss += loss;
    }

    let avg_loss = lambda * (total_loss / (n as f32));
    Ok(avg_loss)
}

/// Normalize rollout rewards for Group Relative Policy Optimization (GRPO).
///
/// Computes `r_norm_i = (r_i - mean(r)) / (std(r) + eps)` across candidate outputs for a prompt.

/// Compute Kahneman-Tversky Optimization (KTO) loss.
pub fn kto_loss(
    policy_chosen_logps: &[f32],
    policy_rejected_logps: &[f32],
    ref_chosen_logps: &[f32],
    ref_rejected_logps: &[f32],
    beta: f32,
    desirable_weight: f32,
    undesirable_weight: f32,
) -> Result<(f32, Vec<f32>, Vec<f32>)> {
    let n_w = policy_chosen_logps.len();
    let n_l = policy_rejected_logps.len();
    if ref_chosen_logps.len() != n_w || ref_rejected_logps.len() != n_l {
        return Err(Error::Backend("KTO logps length mismatch".into()));
    }

    let mut chosen_logr_sum = 0.0f32;
    for i in 0..n_w {
        chosen_logr_sum += policy_chosen_logps[i] - ref_chosen_logps[i];
    }
    let kl_est = if n_w > 0 {
        chosen_logr_sum / n_w as f32
    } else {
        0.0
    };

    let mut total_loss = 0.0f32;
    let mut chosen_losses = Vec::with_capacity(n_w);
    let mut rejected_losses = Vec::with_capacity(n_l);

    for i in 0..n_w {
        let v_w = policy_chosen_logps[i] - ref_chosen_logps[i];
        let loss_i = desirable_weight * softplus(-beta * (v_w - kl_est));
        chosen_losses.push(loss_i);
        total_loss += loss_i;
    }

    for j in 0..n_l {
        let v_l = policy_rejected_logps[j] - ref_rejected_logps[j];
        let loss_j = undesirable_weight * softplus(-beta * (kl_est - v_l));
        rejected_losses.push(loss_j);
        total_loss += loss_j;
    }

    let count = (n_w + n_l).max(1) as f32;
    Ok((total_loss / count, chosen_losses, rejected_losses))
}

/// Compute Simple Preference Optimization (SimPO) loss.
pub fn simpo_loss(
    policy_chosen_logps: &[f32],
    policy_rejected_logps: &[f32],
    chosen_lens: &[usize],
    rejected_lens: &[usize],
    beta: f32,
    gamma: f32,
) -> Result<f32> {
    let n = policy_chosen_logps.len();
    if policy_rejected_logps.len() != n || chosen_lens.len() != n || rejected_lens.len() != n {
        return Err(Error::Backend("SimPO length mismatch".into()));
    }

    let mut total_loss = 0.0f32;
    for i in 0..n {
        let len_w = chosen_lens[i].max(1) as f32;
        let len_l = rejected_lens[i].max(1) as f32;
        let p_w = policy_chosen_logps[i] / len_w;
        let p_l = policy_rejected_logps[i] / len_l;

        let margin = beta * (p_w - p_l) - gamma;
        let loss = softplus(-margin);
        total_loss += loss;
    }

    Ok(total_loss / (n as f32))
}

/// Compute Group Relative Policy Optimization (GRPO) clipped surrogate loss.
pub fn grpo_loss(
    policy_logps: &[f32],
    old_policy_logps: &[f32],
    ref_logps: &[f32],
    rewards: &[f32],
    beta: f32,
    epsilon: f32,
) -> Result<(f32, Vec<f32>)> {
    let n = policy_logps.len();
    if old_policy_logps.len() != n || ref_logps.len() != n || rewards.len() != n {
        return Err(Error::Backend("GRPO inputs length mismatch".into()));
    }

    let advantages = grpo_normalize_rewards(rewards, 1e-8);
    let mut total_loss = 0.0f32;
    let mut per_sample_losses = Vec::with_capacity(n);

    for i in 0..n {
        let log_ratio = policy_logps[i] - old_policy_logps[i];
        let ratio = log_ratio.exp();
        let adv = advantages[i];

        let surr1 = ratio * adv;
        let surr2 = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * adv;
        let obj = surr1.min(surr2);

        let log_kl = ref_logps[i] - policy_logps[i];
        let kl_div = log_kl.exp() - log_kl - 1.0;

        let sample_loss = -obj + beta * kl_div;
        per_sample_losses.push(sample_loss);
        total_loss += sample_loss;
    }

    Ok((total_loss / (n as f32), per_sample_losses))
}

pub fn grpo_normalize_rewards(rewards: &[f32], eps: f32) -> Vec<f32> {
    if rewards.is_empty() {
        return Vec::new();
    }

    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let var = rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();

    rewards.iter().map(|&r| (r - mean) / std).collect()
}

/// Compute DPO loss and gradient tensors for autograd backward traversal.
///
/// Returns `(avg_loss_val, chosen_grad_tensor, rejected_grad_tensor)`.
pub fn dpo_loss_autograd(
    policy_chosen_logps: &grim_tensor::Tensor,
    policy_rejected_logps: &grim_tensor::Tensor,
    ref_chosen_logps: &[f32],
    ref_rejected_logps: &[f32],
    beta: f32,
) -> Result<(f32, grim_tensor::Tensor, grim_tensor::Tensor)> {
    let chosen_vec = policy_chosen_logps.to_vec_f32()?;
    let rejected_vec = policy_rejected_logps.to_vec_f32()?;
    let (loss_val, _, _) = dpo_loss(
        &chosen_vec,
        &rejected_vec,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
    )?;

    let n = chosen_vec.len();
    let mut g_chosen = vec![0.0f32; n];
    let mut g_rejected = vec![0.0f32; n];

    let inv_n = 1.0 / (n as f32);
    for i in 0..n {
        let chosen_logr = chosen_vec[i] - ref_chosen_logps[i];
        let rejected_logr = rejected_vec[i] - ref_rejected_logps[i];
        let logits = beta * (chosen_logr - rejected_logr);

        // sigmoid(-logits) = 1 / (1 + exp(logits))
        let sig_neg = 1.0 / (1.0 + logits.exp().min(1e10));
        g_chosen[i] = -beta * sig_neg * inv_n;
        g_rejected[i] = beta * sig_neg * inv_n;
    }

    let dev = crate::pick_device_for_tensor(policy_chosen_logps);
    let grad_c = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(
            &g_chosen,
            policy_chosen_logps.shape(),
            grim_tensor::dtype::DType::F32,
        )?),
        policy_chosen_logps.shape().clone(),
        grim_tensor::dtype::DType::F32,
        policy_chosen_logps.provenance().clone(),
        policy_chosen_logps.device().clone(),
    );
    let grad_r = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(
            &g_rejected,
            policy_rejected_logps.shape(),
            grim_tensor::dtype::DType::F32,
        )?),
        policy_rejected_logps.shape().clone(),
        grim_tensor::dtype::DType::F32,
        policy_rejected_logps.provenance().clone(),
        policy_rejected_logps.device().clone(),
    );

    Ok((loss_val, grad_c, grad_r))
}

/// Compute ORPO odds ratio loss and gradient tensors for autograd backward traversal.
///
/// Returns `(loss_val, chosen_grad_tensor, rejected_grad_tensor)`.
pub fn orpo_odds_ratio_loss_autograd(
    policy_chosen_logps: &grim_tensor::Tensor,
    policy_rejected_logps: &grim_tensor::Tensor,
    lambda: f32,
) -> Result<(f32, grim_tensor::Tensor, grim_tensor::Tensor)> {
    let chosen_vec = policy_chosen_logps.to_vec_f32()?;
    let rejected_vec = policy_rejected_logps.to_vec_f32()?;
    let loss_val = orpo_odds_ratio_loss(&chosen_vec, &rejected_vec, lambda)?;

    let n = chosen_vec.len();
    let mut g_chosen = vec![0.0f32; n];
    let mut g_rejected = vec![0.0f32; n];
    let inv_n = lambda / (n as f32);

    for i in 0..n {
        let p_chosen = chosen_vec[i].exp().clamp(1e-7, 1.0 - 1e-7);
        let p_rejected = rejected_vec[i].exp().clamp(1e-7, 1.0 - 1e-7);

        let odds_chosen = p_chosen / (1.0 - p_chosen);
        let odds_rejected = p_rejected / (1.0 - p_rejected);
        let log_odds_ratio = (odds_chosen / odds_rejected).ln();

        let sig_neg = 1.0 / (1.0 + log_odds_ratio.exp().min(1e10));
        g_chosen[i] = -inv_n * sig_neg / (1.0 - p_chosen).max(1e-7);
        g_rejected[i] = inv_n * sig_neg / (1.0 - p_rejected).max(1e-7);
    }

    let dev = crate::pick_device_for_tensor(policy_chosen_logps);
    let grad_c = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(
            &g_chosen,
            policy_chosen_logps.shape(),
            grim_tensor::dtype::DType::F32,
        )?),
        policy_chosen_logps.shape().clone(),
        grim_tensor::dtype::DType::F32,
        policy_chosen_logps.provenance().clone(),
        policy_chosen_logps.device().clone(),
    );
    let grad_r = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(
            &g_rejected,
            policy_rejected_logps.shape(),
            grim_tensor::dtype::DType::F32,
        )?),
        policy_rejected_logps.shape().clone(),
        grim_tensor::dtype::DType::F32,
        policy_rejected_logps.provenance().clone(),
        policy_rejected_logps.device().clone(),
    );

    Ok((loss_val, grad_c, grad_r))
}

/// Compute GRPO policy loss and gradient tensor for autograd backward traversal.
///
/// Returns `(mean_loss_val, policy_grad_tensor)`.
pub fn grpo_loss_autograd(
    policy_logps: &grim_tensor::Tensor,
    rewards: &[f32],
    eps: f32,
) -> Result<(f32, grim_tensor::Tensor)> {
    let logps_vec = policy_logps.to_vec_f32()?;
    let norm_advantages = grpo_normalize_rewards(rewards, eps);

    let n = logps_vec.len();
    if norm_advantages.len() != n {
        return Err(Error::Backend("GRPO rewards length mismatch".into()));
    }

    let mut total_loss = 0.0f32;
    let mut grads = vec![0.0f32; n];
    let inv_n = 1.0 / (n as f32);

    for i in 0..n {
        let adv = norm_advantages[i];
        let loss_i = -adv * logps_vec[i];
        total_loss += loss_i;
        grads[i] = -adv * inv_n;
    }

    let avg_loss = total_loss * inv_n;
    let dev = crate::pick_device_for_tensor(policy_logps);
    let grad_tensor = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(
            &grads,
            policy_logps.shape(),
            grim_tensor::dtype::DType::F32,
        )?),
        policy_logps.shape().clone(),
        grim_tensor::dtype::DType::F32,
        policy_logps.provenance().clone(),
        policy_logps.device().clone(),
    );

    Ok((avg_loss, grad_tensor))
}

/// OLoRA orthogonality penalty: `||AᵀA − I||_F² + ||BBᵀ − I||_F²`.
///
/// `a` has shape `[out, r]` (the LoRA down-projection) and `b` has shape
/// `[r, in]` (the LoRA up-projection). The penalty encourages the columns of
/// `A` and the rows of `B` to be orthonormal, which keeps the low-rank
/// subspace of the adapter well-conditioned during training.
///
/// Computed on host floats (via `to_vec_f32`) so it can be added to the scalar
/// CE/DPO/GRPO loss before `backward()` without extending the autograd tape.
pub fn olora_orthogonality_penalty(
    a: &grim_tensor::Tensor,
    b: &grim_tensor::Tensor,
) -> Result<f32> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(Error::Shape(format!(
            "OLoRA expects 2D A and B tensors, got shapes {:?} and {:?}",
            a_dims, b_dims
        )));
    }

    let a_rows = a_dims[0];
    let a_cols = a_dims[1];
    let b_rows = b_dims[0];
    let b_cols = b_dims[1];

    let a_vec = a.to_vec_f32()?;
    let b_vec = b.to_vec_f32()?;
    if a_vec.len() != a_rows * a_cols || b_vec.len() != b_rows * b_cols {
        return Err(Error::Backend("OLoRA tensor length mismatch".into()));
    }

    // ||AᵀA − I||_F² over the r×r Gram matrix.
    let a_gram = gram_penalty(&a_vec, a_rows, a_cols, a_cols, true);
    // ||BBᵀ − I||_F² over the r×r Gram matrix (rows of B become orthonormal).
    let b_gram = gram_penalty(&b_vec, b_rows, b_cols, b_rows, false);

    Ok(a_gram + b_gram)
}

/// Compute `||AᵀA − I||_F²` (when `transpose_a = true`) or `||BBᵀ − I||_F²`
/// (when `transpose_a = false`). `rows`/`cols` describe the raw matrix layout;
/// `rank` is the LoRA rank `r` (the size of the Gram matrix).
fn gram_penalty(m: &[f32], rows: usize, cols: usize, rank: usize, transpose_a: bool) -> f32 {
    let mut total = 0.0f32;
    for i in 0..rank {
        for j in 0..rank {
            let mut dot = 0.0f32;
            if transpose_a {
                // AᵀA[i,j] = Σ_k A[k,i] * A[k,j], k over rows (out dim).
                for k in 0..rows {
                    dot += m[k * cols + i] * m[k * cols + j];
                }
            } else {
                // BBᵀ[i,j] = Σ_k B[i,k] * B[j,k], k over cols (in dim).
                for k in 0..cols {
                    dot += m[i * cols + k] * m[j * cols + k];
                }
            }
            let target = if i == j { 1.0 } else { 0.0 };
            let diff = dot - target;
            total += diff * diff;
        }
    }
    total
}

/// Numerically-stable softplus: `ln(1 + exp(x))` = `max(x, 0) + ln(1 + exp(-|x|))`.
///
/// Equivalent to `-sigmoid(-x).ln()` (and to `-sigmoid(x).ln()` when called with
/// `-x`), but never overflows to ±inf for large |x| because the `max`/abs trick
/// bounds the argument to `exp` to `[0, ∞)` — for very negative inputs the
/// two-term form collapses toward 0 (the true softplus value), and for very
/// positive inputs it collapses toward `x`.
fn softplus(x: f32) -> f32 {
    let max_term = x.max(0.0);
    max_term + (1.0 + (-x.abs()).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kto_simpo_grpo_losses_compute_cleanly() {
        let chosen = vec![-1.2, -0.8];
        let rejected = vec![-2.5, -2.1];
        let ref_c = vec![-1.5, -1.0];
        let ref_r = vec![-2.0, -1.8];

        let (kto_val, _, _) = kto_loss(&chosen, &rejected, &ref_c, &ref_r, 0.1, 1.0, 1.0).unwrap();
        assert!(kto_val > 0.0);

        let lens_c = vec![10, 12];
        let lens_r = vec![10, 12];
        let simpo_val = simpo_loss(&chosen, &rejected, &lens_c, &lens_r, 2.0, 0.5).unwrap();
        assert!(simpo_val > 0.0);

        let old = vec![-1.3, -0.9];
        let rewards = vec![1.0, 0.0];
        let (grpo_val, _) = grpo_loss(&chosen, &old, &ref_c, &rewards, 0.04, 0.2).unwrap();
        assert!(grpo_val.is_finite());
    }

    #[test]
    fn dpo_loss_decreases_when_policy_improves_chosen() {
        let pol_c = vec![-1.0];
        let pol_r = vec![-3.0];
        let ref_c = vec![-2.0];
        let ref_r = vec![-2.0];

        let (loss, c_r, r_r) = dpo_loss(&pol_c, &pol_r, &ref_c, &ref_r, 0.1).unwrap();
        // Exact rewards: chosen = 0.1 * (-1.0 - (-2.0)) = 0.1, rejected = 0.1 * (-3.0 - (-2.0)) = -0.1
        assert!((c_r[0] - 0.1).abs() < 1e-6);
        assert!((r_r[0] - (-0.1)).abs() < 1e-6);
        // Exact loss: softplus(-0.2) = ln(1 + exp(-0.2)) = 0.5981424
        assert!(
            (loss - 0.5981424).abs() < 1e-5,
            "loss = {}, want 0.5981424",
            loss
        );
    }

    #[test]
    fn grpo_reward_normalization_has_zero_mean() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let norm = grpo_normalize_rewards(&rewards, 1e-8);
        let mean = norm.iter().sum::<f32>() / (norm.len() as f32);
        assert!(mean.abs() < 1e-6);

        // Assert unit variance: std_dev of [1, 2, 3, 4, 5] = sqrt((4+1+0+1+4)/5) = sqrt(2)
        // Normalized values must have sample variance ~1.0
        let var = norm.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (norm.len() as f32);
        assert!(
            (var - 1.0).abs() < 1e-4,
            "Normalized variance = {}, want 1.0",
            var
        );
    }

    #[test]
    fn olora_penalty_is_zero_for_orthonormal_adapters() {
        use grim_backend_cpu::cpu_tensor;
        use grim_tensor::Shape;

        // A = [[1, 0], [0, 1], [0, 0]] → AᵀA = I.
        let a = cpu_tensor(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], Shape::new(vec![3, 2]));
        // B = [[1, 0, 0], [0, 1, 0]] → BBᵀ = I.
        let b = cpu_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], Shape::new(vec![2, 3]));

        let penalty = olora_orthogonality_penalty(&a, &b).unwrap();
        assert!(
            penalty.abs() < 1e-5,
            "orthonormal adapters should have ~0 penalty, got {penalty}"
        );
    }

    #[test]
    fn olora_penalty_is_positive_for_non_orthonormal_adapters() {
        use grim_backend_cpu::cpu_tensor;
        use grim_tensor::Shape;

        // Non-orthonormal A: [[1, 1], [1, 1], [0, 0]] → AᵀA = [[2,2],[2,2]].
        let a = cpu_tensor(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0], Shape::new(vec![3, 2]));
        // Non-orthonormal B: [[1, 1, 1], [1, 1, 1]] → BBᵀ = [[3,3],[3,3]].
        let b = cpu_tensor(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], Shape::new(vec![2, 3]));

        let penalty = olora_orthogonality_penalty(&a, &b).unwrap();
        assert!(
            penalty > 1.0,
            "degenerate adapters should produce a large penalty, got {penalty}"
        );
    }

    #[test]
    fn olora_penalty_rejects_non_2d_tensors() {
        use grim_backend_cpu::cpu_tensor;
        use grim_tensor::Shape;

        let a = cpu_tensor(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let b = cpu_tensor(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            Shape::new(vec![2, 3, 1]),
        );
        assert!(olora_orthogonality_penalty(&a, &b).is_err());
    }
}
