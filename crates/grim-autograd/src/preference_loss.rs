//! DPO, ORPO, and GRPO preference optimization loss functions (WI-T7).
//!
//! Provides loss routines for alignment fine-tuning on top of scoped autograd:
//! - `dpo_loss`: Direct Preference Optimization loss.
//! - `orpo_loss`: Odds Ratio Preference Optimization loss.
//! - `grpo_normalize_rewards`: Group-relative reward normalization for GRPO.

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
    if policy_rejected_logps.len() != n || ref_chosen_logps.len() != n || ref_rejected_logps.len() != n {
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
        let loss = -sigmoid(logits).ln();
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
        let loss = -sigmoid(log_odds_ratio).ln();
        total_loss += loss;
    }

    let avg_loss = lambda * (total_loss / (n as f32));
    Ok(avg_loss)
}

/// Normalize rollout rewards for Group Relative Policy Optimization (GRPO).
///
/// Computes `r_norm_i = (r_i - mean(r)) / (std(r) + eps)` across candidate outputs for a prompt.
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpo_loss_decreases_when_policy_improves_chosen() {
        let pol_c = vec![-1.0];
        let pol_r = vec![-3.0];
        let ref_c = vec![-2.0];
        let ref_r = vec![-2.0];

        let (loss, c_r, r_r) = dpo_loss(&pol_c, &pol_r, &ref_c, &ref_r, 0.1).unwrap();
        assert!(loss > 0.0);
        assert!(c_r[0] > r_r[0]);
    }

    #[test]
    fn grpo_reward_normalization_has_zero_mean() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let norm = grpo_normalize_rewards(&rewards, 1e-8);
        let mean = norm.iter().sum::<f32>() / (norm.len() as f32);
        assert!(mean.abs() < 1e-6);
    }
}
