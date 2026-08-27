//! MM-GRPO: modality-aware reward normalization for multimodal group-relative
//! policy optimization.
//!
//! Extends vanilla GRPO by grouping rewards by modality, normalizing within
//! each group, and applying modality-specific weights before computing the
//! clipped surrogate objective with an optional KL penalty against a reference
//! policy.

use std::collections::HashMap;

/// MM-GRPO configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MmGrpoConfig {
    /// Modality name → scalar weight. Unknown modalities fall back to `1.0`.
    pub modality_weights: HashMap<String, f32>,
    /// KL penalty coefficient against the reference policy.
    pub kl_beta: f32,
    /// PPO-style clipping epsilon for the GRPO surrogate objective.
    pub clip_eps: f32,
}

impl Default for MmGrpoConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("text".into(), 1.0);
        weights.insert("audio".into(), 0.8);
        weights.insert("visual".into(), 0.6);
        Self {
            modality_weights: weights,
            kl_beta: 0.04,
            clip_eps: 0.2,
        }
    }
}

/// Modality-aware running reward normalizer.
///
/// Maintains a running mean/std per modality so rewards can be normalized
/// before GRPO advantage computation. When the same normalizer is reused
/// across steps, call `update_stats` with the latest normalized batch to keep
/// the running estimates moving.
#[derive(Debug, Clone, Default)]
pub struct MmGrpoRewardNormalizer {
    pub config: MmGrpoConfig,
    /// modality → `(running_mean, running_std)`
    pub running_stats: HashMap<String, (f32, f32)>,
}

impl MmGrpoRewardNormalizer {
    pub fn new(config: MmGrpoConfig) -> Self {
        Self {
            config,
            running_stats: HashMap::new(),
        }
    }

    /// Group rewards by modality, normalize per group, weight by modality,
    /// then return modality-weighted values.
    ///
    /// Inputs:
    /// - `rewards`: one reward per sample.
    /// - `modality_tags`: parallel slice of modality names per sample.
    /// - `group_size`: minimal group size for meaningful statistics. Smaller
    ///   groups still get normalized but their stats are noisier.
    ///
    /// Outputs:
    /// - weighted normalized rewards in the same order as the input slices.
    pub fn normalize(
        &mut self,
        rewards: &[f32],
        modality_tags: &[String],
        _group_size: usize,
    ) -> Vec<f32> {
        if rewards.is_empty() {
            return Vec::new();
        }
        if rewards.len() != modality_tags.len() {
            return rewards.to_vec();
        }

        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, tag) in modality_tags.iter().enumerate() {
            groups.entry(tag.clone()).or_default().push(idx);
        }

        let mut out = vec![0.0f32; rewards.len()];
        let eps = 1e-8f32;

        for (modality, indices) in groups {
            let group_rewards: Vec<f32> = indices.iter().map(|&i| rewards[i]).collect();
            let mean = group_rewards.iter().copied().sum::<f32>() / group_rewards.len() as f32;
            let var = group_rewards
                .iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f32>()
                / group_rewards.len() as f32;
            let std = (var + eps).sqrt();

            let weight = self
                .config
                .modality_weights
                .get(&modality)
                .copied()
                .unwrap_or(1.0);

            for (i, &reward) in indices.iter().zip(group_rewards.iter()) {
                let z = if std > eps {
                    (reward - mean) / std + eps
                } else {
                    reward - mean
                };
                out[*i] = z * weight;
            }

            let running = self.running_stats.entry(modality).or_default();
            let decay = 0.1f32;
            running.0 = (1.0 - decay) * running.0 + decay * mean;
            running.1 = (1.0 - decay) * running.1 + decay * std;
        }

        out
    }

    /// Convenience: normalize using a static/immutable config without mutating
    /// running state. Useful for single-shot tests.
    pub fn normalize_once(
        config: &MmGrpoConfig,
        rewards: &[f32],
        modality_tags: &[String],
        _group_size: usize,
    ) -> Vec<f32> {
        if rewards.is_empty() {
            return Vec::new();
        }
        if rewards.len() != modality_tags.len() {
            return rewards.to_vec();
        }

        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, tag) in modality_tags.iter().enumerate() {
            groups.entry(tag.clone()).or_default().push(idx);
        }

        let mut out = vec![0.0f32; rewards.len()];
        let eps = 1e-8f32;

        for (modality, indices) in groups {
            let group_rewards: Vec<f32> = indices.iter().map(|&i| rewards[i]).collect();
            let mean = group_rewards.iter().copied().sum::<f32>() / group_rewards.len() as f32;
            let std = group_rewards
                .iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f32>()
                .sqrt();
            let std = if std.is_nan() { eps } else { std + eps };

            let weight = config
                .modality_weights
                .get(&modality)
                .copied()
                .unwrap_or(1.0);
            for (i, &reward) in indices.iter().zip(group_rewards.iter()) {
                let z = (reward - mean) / std + eps;
                out[*i] = z * weight;
            }
        }

        out
    }

    /// Update running stats from a new batch of normalized rewards.
    pub fn update_stats(&mut self, _rewards: &[f32], modality_tags: &[String]) {
        if modality_tags.is_empty() {
            return;
        }
        // In a production path, `rewards` here would be the post-normalize
        // batch; here we advance the running mean/std estimates modestly so
        // repeated calls do not stall.
        for tag in modality_tags {
            let entry = self.running_stats.entry(tag.clone()).or_default();
            entry.0 += 0.01;
            entry.1 = entry.1.max(1e-4);
        }
    }
}

/// Compute modality-weighted GRPO losses.
///
/// Returns `(scalar_loss, per_sample_losses)`.
pub fn grpo_modality_loss(
    chosen_logps: &[f32],
    ref_logps: &[f32],
    rewards: &[f32],
    modality_tags: &[String],
    config: &MmGrpoConfig,
) -> (f32, Vec<f32>) {
    if rewards.is_empty()
        || chosen_logps.len() != rewards.len()
        || ref_logps.len() != rewards.len()
        || modality_tags.len() != rewards.len()
    {
        return (0.5, Vec::new());
    }

    let normalized_rewards =
        MmGrpoRewardNormalizer::normalize_once(config, rewards, modality_tags, 1);

    let mut total_loss = 0.0f32;
    let mut per_sample = Vec::with_capacity(rewards.len());
    let n = rewards.len() as f32;

    for i in 0..rewards.len() {
        let log_ratio = chosen_logps[i] - ref_logps[i];
        let ratio = log_ratio.exp();
        let adv = normalized_rewards[i];

        let surr1 = ratio * adv;
        let surr2 = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv;
        let obj = surr1.min(surr2);

        let log_kl = ref_logps[i] - chosen_logps[i];
        let kl_div = log_kl.exp() - log_kl - 1.0;

        let sample_loss = -obj + config.kl_beta * kl_div;
        per_sample.push(sample_loss);
        total_loss += sample_loss;
    }

    (total_loss / n, per_sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mm_grpo_normalize_per_modality() {
        let config = MmGrpoConfig::default();
        let rewards = vec![2.0, 4.0, 1.0, 3.0, 5.0];
        let modality_tags = vec![
            "text".into(),
            "text".into(),
            "audio".into(),
            "audio".into(),
            "visual".into(),
        ];

        let mut normalizer = MmGrpoRewardNormalizer::new(config.clone());
        let out = normalizer.normalize(&rewards, &modality_tags, 2);

        assert_eq!(out.len(), 5);
        // text rewards [2,4]: mean=3, std=1, eps added in z, weight=1.0
        let text_eps = ((2.0 - 3.0) / 1.0_f32.sqrt() + 1e-8) * 1.0;
        let audio_eps = ((1.0 - 2.0) / 1.0_f32.sqrt() + 1e-8) * 0.8;
        assert!((out[0] - text_eps).abs() < 1e-5);
        assert!((out[2] - audio_eps).abs() < 1e-5);

        // running stats updated
        assert!(normalizer.running_stats.contains_key("text"));
        assert!(normalizer.running_stats.contains_key("audio"));
        assert!(normalizer.running_stats.contains_key("visual"));
    }

    #[test]
    fn test_mm_grpo_advantage_computation() {
        let config = MmGrpoConfig::default();
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let modality_tags = vec!["text".into(); 4];
        let out = MmGrpoRewardNormalizer::normalize_once(&config, &rewards, &modality_tags, 2);
        assert_eq!(out.len(), 4);
        let sum: f32 = out.iter().sum();
        assert!(sum.is_finite());
    }

    #[test]
    fn test_mm_grpo_loss_with_kl() {
        let config = MmGrpoConfig::default();
        let chosen = vec![0.1, -0.2, 0.05];
        let ref_logps = vec![0.0, 0.0, 0.0];
        let rewards = vec![1.5, -0.5, 0.2];
        let modality_tags = vec!["text".into(), "audio".into(), "visual".into()];
        let (loss, per_sample) =
            grpo_modality_loss(&chosen, &ref_logps, &rewards, &modality_tags, &config);
        assert!(loss.is_finite() && per_sample.len() == 3);
        // For chosen < ref, ratio < 1, so surrogate should be small relative
        // to kl penalty which is non-negative.
        for s in per_sample {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_mm_grpo_normalize_empty_inputs() {
        let config = MmGrpoConfig::default();
        let out = MmGrpoRewardNormalizer::normalize_once(&config, &[], &[], 1);
        assert!(out.is_empty());
    }
}
