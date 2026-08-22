//! Shared preference trainer for DPO, KTO, SimPO, ORPO, and GRPO (WI-T7 / F7).
//!
//! Provides unified sequence log-probability reduction, preference loss evaluation,
//! and exact log-softmax vector-Jacobian product (VJP) gradient computation for
//! alignment fine-tuning across CLI training and Garage distributed workers.

use crate::preference_loss::{
    dpo_loss, grpo_loss, kto_loss, orpo_odds_ratio_loss, simpo_loss,
};
use grim_tensor::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Supported preference optimization algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreferenceKind {
    /// Direct Preference Optimization.
    Dpo,
    /// Odds Ratio Preference Optimization.
    Orpo,
    /// Simple Preference Optimization (reference-free, length-normalized).
    Simpo,
    /// Kahneman-Tversky Optimization (prospect theory / unpaired preferences).
    Kto,
    /// Group Relative Policy Optimization.
    Grpo,
}

impl std::str::FromStr for PreferenceKind {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "dpo" => Ok(Self::Dpo),
            "orpo" => Ok(Self::Orpo),
            "simpo" => Ok(Self::Simpo),
            "kto" => Ok(Self::Kto),
            "grpo" => Ok(Self::Grpo),
            other => Err(Error::Backend(format!(
                "Unknown preference kind '{other}'. Expected dpo, orpo, simpo, kto, or grpo."
            ))),
        }
    }
}

/// Hyperparameters for preference training steps.
#[derive(Debug, Clone, PartialEq)]
pub struct PreferenceStepConfig {
    /// KL divergence regularization strength $\beta$ (default 0.1).
    pub beta: f32,
    /// ORPO odds ratio loss multiplier $\lambda$ (default 0.1).
    pub orpo_lambda: f32,
    /// SimPO target reward margin $\gamma$ (default 0.5).
    pub simpo_gamma: f32,
    /// KTO weight for desirable examples (default 1.0).
    pub kto_desirable_weight: f32,
    /// KTO weight for undesirable examples (default 1.0).
    pub kto_undesirable_weight: f32,
    /// GRPO clipping parameter $\epsilon$ (default 0.2).
    pub grpo_epsilon: f32,
}

impl Default for PreferenceStepConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            orpo_lambda: 0.1,
            simpo_gamma: 0.5,
            kto_desirable_weight: 1.0,
            kto_undesirable_weight: 1.0,
            grpo_epsilon: 0.2,
        }
    }
}

/// Shared preference trainer executing loss evaluation and VJP gradient propagation.
pub struct PreferenceTrainer {
    pub config: PreferenceStepConfig,
}

impl PreferenceTrainer {
    /// Create a new preference trainer with configuration.
    pub fn new(config: PreferenceStepConfig) -> Self {
        Self { config }
    }

    /// Default preference trainer.
    pub fn with_default_config() -> Self {
        Self::new(PreferenceStepConfig::default())
    }

    /// Compute cumulative sequence log-probability $\sum_{t} \log P(y_t | x, y_{<t})$
    /// over non-ignored target tokens using numerically stable log-softmax.
    ///
    /// # Contract
    /// `logits.len() == targets.len() * vocab_size`.
    /// Tokens with `target == ignore_index` are excluded from the sum and count.
    /// Returns `(total_logp, valid_token_count)`.
    pub fn compute_sequence_logps(
        logits: &[f32],
        targets: &[u32],
        vocab_size: usize,
        ignore_index: u32,
    ) -> (f32, usize) {
        let seq_len = targets.len();
        if logits.len() != seq_len * vocab_size || vocab_size == 0 {
            return (0.0, 0);
        }

        let mut total_logp = 0.0f32;
        let mut valid_count = 0;

        for (t, &target) in targets.iter().enumerate() {
            if target == ignore_index {
                continue;
            }
            let target_tok = target as usize;
            if target_tok >= vocab_size {
                continue;
            }

            let row_start = t * vocab_size;
            let row = &logits[row_start..row_start + vocab_size];

            // Numerically stable log-softmax: log(exp(z_k) / sum(exp(z_j))) = z_k - (max + ln(sum(exp(z_j - max))))
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp = row
                .iter()
                .map(|&z| (z - max_val).exp())
                .sum::<f32>();
            let log_denom = max_val + sum_exp.ln();

            let logp = row[target_tok] - log_denom;
            total_logp += logp;
            valid_count += 1;
        }

        (total_logp, valid_count)
    }

    /// Compute preference loss and per-sample logp scalar gradients $\frac{\partial \mathcal{L}}{\partial \log \pi_\theta(y_w)}$
    /// and $\frac{\partial \mathcal{L}}{\partial \log \pi_\theta(y_l)}$.
    ///
    /// # Contract
    /// Evaluates the configured preference loss (`Dpo`, `Kto`, `Simpo`, `Orpo`, `Grpo`) and
    /// returns `(loss_val, chosen_logp_grad, rejected_logp_grad)`.
    pub fn compute_preference_step(
        &self,
        kind: PreferenceKind,
        policy_chosen_logps: &[f32],
        policy_rejected_logps: &[f32],
        ref_chosen_logps: &[f32],
        ref_rejected_logps: &[f32],
        chosen_lens: &[usize],
        rejected_lens: &[usize],
        rewards: Option<&[f32]>,
    ) -> Result<(f32, Vec<f32>, Vec<f32>)> {
        match kind {
            PreferenceKind::Dpo => {
                let (loss, chosen_r, rejected_r) = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    self.config.beta,
                )?;
                // Analytical gradients: dL/d(logp_w) = -beta * sigmoid(r_l - r_w)
                let mut d_chosen = Vec::with_capacity(chosen_r.len());
                let mut d_rejected = Vec::with_capacity(rejected_r.len());
                let n = policy_chosen_logps.len().max(1) as f32;

                for i in 0..chosen_r.len() {
                    let logit_diff = chosen_r[i] - rejected_r[i];
                    let sig_neg = 1.0 / (1.0 + logit_diff.exp());
                    d_chosen.push((-self.config.beta * sig_neg) / n);
                    d_rejected.push((self.config.beta * sig_neg) / n);
                }
                Ok((loss, d_chosen, d_rejected))
            }
            PreferenceKind::Orpo => {
                let loss = orpo_odds_ratio_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    self.config.orpo_lambda,
                )?;
                let n = policy_chosen_logps.len().max(1) as f32;
                let scale = (-self.config.orpo_lambda * 0.5) / n;
                let d_chosen = vec![scale; policy_chosen_logps.len()];
                let d_rejected = vec![-scale; policy_rejected_logps.len()];
                Ok((loss, d_chosen, d_rejected))
            }
            PreferenceKind::Simpo => {
                let loss = simpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    chosen_lens,
                    rejected_lens,
                    self.config.beta,
                    self.config.simpo_gamma,
                )?;
                let n = policy_chosen_logps.len().max(1) as f32;
                let mut d_chosen = Vec::with_capacity(policy_chosen_logps.len());
                let mut d_rejected = Vec::with_capacity(policy_rejected_logps.len());

                for i in 0..policy_chosen_logps.len() {
                    let len_w = chosen_lens[i].max(1) as f32;
                    let len_l = rejected_lens[i].max(1) as f32;
                    let margin = self.config.beta * (policy_chosen_logps[i] / len_w - policy_rejected_logps[i] / len_l) - self.config.simpo_gamma;
                    let sig_neg = 1.0 / (1.0 + margin.exp());

                    d_chosen.push((-self.config.beta / len_w * sig_neg) / n);
                    d_rejected.push((self.config.beta / len_l * sig_neg) / n);
                }
                Ok((loss, d_chosen, d_rejected))
            }
            PreferenceKind::Kto => {
                let (loss, chosen_losses, rejected_losses) = kto_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    self.config.beta,
                    self.config.kto_desirable_weight,
                    self.config.kto_undesirable_weight,
                )?;
                let n = (policy_chosen_logps.len() + policy_rejected_logps.len()).max(1) as f32;
                let d_chosen = chosen_losses.iter().map(|&l| -self.config.beta * l / n).collect();
                let d_rejected = rejected_losses.iter().map(|&l| self.config.beta * l / n).collect();
                Ok((loss, d_chosen, d_rejected))
            }
            PreferenceKind::Grpo => {
                let default_rewards = vec![1.0f32; policy_chosen_logps.len()];
                let rew = rewards.unwrap_or(&default_rewards);
                let (loss, per_sample_losses) = grpo_loss(
                    policy_chosen_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    rew,
                    self.config.beta,
                    self.config.grpo_epsilon,
                )?;
                let n = policy_chosen_logps.len().max(1) as f32;
                let d_chosen = per_sample_losses.iter().map(|&l| -l / n).collect();
                let d_rejected = vec![0.0f32; policy_rejected_logps.len()];
                Ok((loss, d_chosen, d_rejected))
            }
        }
    }

    /// Compute exact vector-Jacobian product (VJP) gradient for cross-entropy / log-softmax:
    /// \[
    /// \frac{\partial \mathcal{L}}{\partial z_{t, v}} = \frac{\partial \mathcal{L}}{\partial \log \pi} \cdot (\mathbb{I}(v = y_t) - P(v))
    /// \]
    ///
    /// # Contract
    /// `logits.len() == targets.len() * vocab_size`.
    /// Returns the full gradient tensor $\nabla_{\text{logits}} \mathcal{L}$ of size `[seq_len * vocab_size]`.
    pub fn compute_log_softmax_vjp(
        logits: &[f32],
        targets: &[u32],
        vocab_size: usize,
        logp_grad: f32,
        ignore_index: u32,
    ) -> Vec<f32> {
        let seq_len = targets.len();
        let mut grad = vec![0.0f32; seq_len * vocab_size];
        if logp_grad == 0.0 || vocab_size == 0 {
            return grad;
        }

        for (t, &target) in targets.iter().enumerate() {
            if target == ignore_index {
                continue;
            }
            let target_tok = target as usize;
            if target_tok >= vocab_size {
                continue;
            }

            let row_start = t * vocab_size;
            let row = &logits[row_start..row_start + vocab_size];

            // Softmax probabilities: P(v) = exp(z_v - max) / sum_exp
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for &z in row {
                sum_exp += (z - max_val).exp();
            }

            let grad_row = &mut grad[row_start..row_start + vocab_size];
            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                for v in 0..vocab_size {
                    let p_v = (row[v] - max_val).exp() * inv_sum;
                    let delta = if v == target_tok { 1.0 } else { 0.0 };
                    // dL/dz = dL/dlogp * (delta - P(v))
                    grad_row[v] = logp_grad * (delta - p_v);
                }
            }
        }

        grad
    }

    /// Compute exact vector-Jacobian product (VJP) for a tensor on its native backend or CPU fallback.
    pub fn compute_log_softmax_vjp_tensor(
        logits: &grim_tensor::Tensor,
        targets: &[u32],
        vocab_size: usize,
        logp_grad: f32,
        ignore_index: u32,
    ) -> Result<grim_tensor::Tensor> {
        let logits_vec = logits.to_vec_f32().map_err(|e| Error::Backend(format!("Tensor to_vec_f32 failed: {e}")))?;
        let grad_vec = Self::compute_log_softmax_vjp(&logits_vec, targets, vocab_size, logp_grad, ignore_index);
        let grad_tensor = grim_backend_cpu::cpu_tensor(
            grad_vec,
            grim_tensor::Shape::new(vec![targets.len(), vocab_size]),
        );
        Ok(grad_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preference_trainer_logps_and_dpo_step() {
        let trainer = PreferenceTrainer::with_default_config();

        let vocab_size = 4;
        let targets = vec![1, 2, 255]; // 2 valid tokens, 1 ignored (255)
        let logits = vec![
            1.0, 3.0, 0.5, 0.2, // t=0, target=1 (high logp)
            0.1, 0.2, 4.0, 0.3, // t=1, target=2 (high logp)
            0.0, 0.0, 0.0, 0.0, // t=2, ignored
        ];

        let (logp, count) = PreferenceTrainer::compute_sequence_logps(&logits, &targets, vocab_size, 255);
        assert_eq!(count, 2);
        assert!(logp < 0.0 && logp > -1.0, "Logp should be high for top predictions: {logp}");

        // DPO step
        let chosen_logps = vec![logp];
        let rejected_logps = vec![logp - 2.0];
        let ref_chosen = vec![logp - 0.5];
        let ref_rejected = vec![logp - 1.5];

        let (loss, d_chosen, d_rejected) = trainer
            .compute_preference_step(
                PreferenceKind::Dpo,
                &chosen_logps,
                &rejected_logps,
                &ref_chosen,
                &ref_rejected,
                &[2],
                &[2],
                None,
            )
            .unwrap();

        assert!(loss > 0.0);
        assert!(d_chosen[0] < 0.0, "Chosen gradient should push logp higher");
        assert!(d_rejected[0] > 0.0, "Rejected gradient should push logp lower");

        // VJP gradient computation
        let vjp_grad = PreferenceTrainer::compute_log_softmax_vjp(&logits, &targets, vocab_size, d_chosen[0], 255);
        assert_eq!(vjp_grad.len(), logits.len());
        // Row 0, target 1 should receive non-zero gradient
        assert!(vjp_grad[1] != 0.0);
        // Ignored row 2 should remain zero
        assert_eq!(vjp_grad[8], 0.0);
    }
}
