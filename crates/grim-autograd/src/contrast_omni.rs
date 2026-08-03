//! CONTRAST-OMNI: Fréchet regularizer for hierarchical cross-modal contrastive learning.
//!
//! This module provides `ContrastOmniLoss`, which combines:
//! - Cross-modal Fréchet distance between Gaussian embeddings of each modality.
//! - Within-modality InfoNCE-style contrastive loss.
//! - Utility-weighted adjustment over the contrastive term.
//!
//! All math is expressed over flat `f32` slices to keep the crate backend-agnostic.

use std::collections::HashMap;

/// Configuration for the CONTRAST-OMNI loss.
#[derive(Debug, Clone)]
pub struct ContrastOmniConfig {
    /// Softmax temperature for the within-modality contrastive loss.
    pub temperature: f32,
    /// Per-modality weight for the cross-modal Fréchet penalty.
    pub modality_weights: HashMap<String, f32>,
    /// Number of hierarchy levels used to weight cross-modal vs within-modal terms.
    pub hierarchy_levels: usize,
}

impl Default for ContrastOmniConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            modality_weights: HashMap::new(),
            hierarchy_levels: 3,
        }
    }
}

/// CONTRAST-OMNI loss container.
#[derive(Debug, Clone)]
pub struct ContrastOmniLoss {
    pub config: ContrastOmniConfig,
}

impl ContrastOmniLoss {
    /// Construct a new loss helper from `config`.
    pub fn new(config: ContrastOmniConfig) -> Self {
        Self { config }
    }

    /// Squared Fréchet distance between two diagonal Gaussians.
    ///
    /// Uses the closed-form diagonal approximation:
    /// `sum((mu1 - mu2)^2) + sum((sqrt(cov1) - sqrt(cov2))^2)`.
    pub fn compute_frechet_distance(
        mean1: &[f32],
        mean2: &[f32],
        cov1: &[f32],
        cov2: &[f32],
        d: usize,
    ) -> f32 {
        if d == 0 {
            return 0.0;
        }
        let mut mean_term = 0.0f32;
        let mut cov_term = 0.0f32;
        for i in 0..d {
            let delta = mean1[i] - mean2[i];
            mean_term += delta * delta;
            let s1 = cov1[i].max(0.0).sqrt();
            let s2 = cov2[i].max(0.0).sqrt();
            let diff = s1 - s2;
            cov_term += diff * diff;
        }
        mean_term + cov_term
    }

    /// Hierarchical cross-modal contrastive loss.
    ///
    /// * `features` is a flat buffer of `[sample, dim]` row-major data.
    /// * `modality_ids` assigns each sample to a modality bucket.
    /// * `modality_names` maps modality id to a human-readable tag used for weighting.
    /// * `labels` class ids for within-modality positives.
    ///
    /// Returns the scalar total loss as `f32`.
    pub fn hierarchical_contrastive(
        &self,
        features: &[f32],
        modality_ids: &[usize],
        modality_names: &[String],
        labels: &[usize],
    ) -> f32 {
        let num_samples = features.len() / modality_ids.len().max(1);
        let dim = if num_samples == 0 {
            self.config.hierarchy_levels.max(1)
        } else {
            num_samples
        };
        if num_samples == 0 {
            return 0.0;
        }

        // Group indices by modality id.
        let mut modality_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &mid) in modality_ids.iter().enumerate() {
            modality_groups.entry(mid).or_default().push(idx);
        }

        // Compute per-modality mean and diagonal covariance (variance) over features.
        let mut modality_stats: HashMap<usize, (Vec<f32>, Vec<f32>)> = HashMap::new();
        for (mid, indices) in &modality_groups {
            let mut mean = vec![0.0f32; dim];
            let mut var = vec![0.0f32; dim];
            for &idx in indices {
                let start = idx * dim;
                for d in 0..dim {
                    mean[d] += features[start + d];
                }
            }
            let count = indices.len() as f32;
            for d in 0..dim {
                mean[d] /= count;
            }
            for &idx in indices {
                let start = idx * dim;
                for d in 0..dim {
                    let delta = features[start + d] - mean[d];
                    var[d] += delta * delta;
                }
            }
            for d in 0..dim {
                var[d] /= count.max(1.0);
            }
            modality_stats.insert(*mid, (mean, var));
        }

        // Cross-modal Fréchet penalty between each pair of modalities.
        let mut cross_modal_penalty = 0.0f32;
        let modality_ids_list: Vec<usize> = modality_stats.keys().copied().collect();
        for i in 0..modality_ids_list.len() {
            for j in (i + 1)..modality_ids_list.len() {
                let m1 = modality_ids_list[i];
                let m2 = modality_ids_list[j];
                let (mean1, var1) = &modality_stats[&m1];
                let (mean2, var2) = &modality_stats[&m2];
                let w1 = modality_names
                    .get(m1)
                    .and_then(|name| self.config.modality_weights.get(name))
                    .copied()
                    .unwrap_or(1.0);
                let w2 = modality_names
                    .get(m2)
                    .and_then(|name| self.config.modality_weights.get(name))
                    .copied()
                    .unwrap_or(1.0);
                let dist = Self::compute_frechet_distance(mean1, mean2, var1, var2, dim);
                cross_modal_penalty += (w1 + w2) * dist;
            }
        }

        // Within-modality InfoNCE contrastive loss.
        let temp = self.config.temperature.max(1e-6);
        let mut within_modality_loss = 0.0f32;
        let mut valid_pairs = 0usize;

        for (mid, indices) in &modality_groups {
            if indices.len() < 2 {
                continue;
            }
            let weight = modality_names
                .get(*mid)
                .and_then(|n| self.config.modality_weights.get(n))
                .copied()
                .unwrap_or(1.0);

            for &anchor_idx in indices {
                let anchor_start = anchor_idx * dim;
                let anchor_label = labels[anchor_idx];
                let mut pos_sim = None;
                let mut all_sims: Vec<f32> = Vec::with_capacity(indices.len());

                for &idx in indices {
                    if idx == anchor_idx {
                        continue;
                    }
                    let start = idx * dim;
                    let mut sim = 0.0f32;
                    for d in 0..dim {
                        sim += features[anchor_start + d] * features[start + d];
                    }
                    all_sims.push(sim);
                    if labels[idx] == anchor_label {
                        pos_sim = Some(sim);
                    }
                }

                let pos = if let Some(s) = pos_sim {
                    s
                } else {
                    // No positive in this modality view: skip sample.
                    continue;
                };

                // Numerically stable softmax denominator.
                let max_sim = all_sims.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                for s in &all_sims {
                    denom += (s - max_sim).exp();
                }
                let loss = -((pos - max_sim).exp() / denom).ln();
                within_modality_loss += weight * loss;
                valid_pairs += 1;
            }
        }

        if valid_pairs > 0 {
            within_modality_loss /= valid_pairs as f32;
        }

        // Hierarchy weighting: higher hierarchy_levels increases cross-modal penalty weight.
        let hierarchy_weight = (self.config.hierarchy_levels as f32).ln_1p();
        let total = hierarchy_weight * cross_modal_penalty + within_modality_loss;
        if total.is_nan() || total.is_infinite() {
            0.0
        } else {
            total
        }
    }

    /// Weight an existing contrastive loss by modality-tagged utility.
    ///
    /// * `scores` is a flat list of per-sample contrastive scores.
    /// * `modality_tags` are the modality names (one per sample).
    /// * `utility` global utility scalar applied to every score.
    pub fn utility_weighted_contrastive(
        &self,
        scores: &[f32],
        modality_tags: &[String],
        utility: f32,
    ) -> f32 {
        if scores.len() != modality_tags.len() || scores.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f32;
        for (score, tag) in scores.iter().zip(modality_tags.iter()) {
            let weight = self
                .config
                .modality_weights
                .get(tag)
                .copied()
                .unwrap_or(1.0);
            total += weight * score * utility;
        }
        let avg = total / scores.len() as f32;
        if avg.is_nan() || avg.is_infinite() {
            0.0
        } else {
            avg
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contrast_omni_frechet_distance() {
        // Identical diagonal Gaussians -> distance 0.
        let mean1 = [0.0f32, 1.0];
        let mean2 = [0.0f32, 1.0];
        let cov1 = [1.0, 4.0];
        let cov2 = [1.0, 4.0];
        let dist = ContrastOmniLoss::compute_frechet_distance(&mean1, &mean2, &cov1, &cov2, 2);
        assert!(
            (dist - 0.0).abs() < 1e-5,
            "expected 0.0 for identical distributions, got {dist}"
        );

        // Pure mean shift of 3 in dim0: 9.0; cov shift sqrt(1)-sqrt(4) = -1 -> squared = 1; sqrt(4)-sqrt(1)=1 -> squared=1.
        let mean1 = [0.0f32, 1.0];
        let mean2 = [3.0, 1.0];
        let cov1 = [1.0, 4.0];
        let cov2 = [4.0, 1.0];
        let dist = ContrastOmniLoss::compute_frechet_distance(&mean1, &mean2, &cov1, &cov2, 2);
        let expected = 9.0
            + ((1.0f32.sqrt() - 4.0f32.sqrt()).powi(2))
            + ((4.0f32.sqrt() - 1.0f32.sqrt()).powi(2));
        assert!(
            (dist - expected).abs() < 1e-5,
            "expected {expected}, got {dist}"
        );
    }

    #[test]
    fn test_contrast_omni_within_modality_loss() {
        let cfg = ContrastOmniConfig::default();
        let loss = ContrastOmniLoss::new(cfg);
        // Two modalities, 4 samples, dim=3.
        let features = vec![
            1.0, 0.0, 0.0, 0.9, 0.1, 0.0, // modality 0
            0.0, 1.0, 0.0, 0.0, 0.95, 0.05, // modality 1
        ];
        let modality_ids = vec![0, 0, 1, 1];
        let modality_names = vec![
            String::from("mod0"),
            String::from("mod0"),
            String::from("mod1"),
            String::from("mod1"),
        ];
        let labels = vec![0, 0, 1, 1];
        let total =
            loss.hierarchical_contrastive(&features, &modality_ids, &modality_names, &labels);
        // Within-modality loss should be near 0 because positives are the nearest neighbor.
        assert!(
            total >= 0.0,
            "within-modality loss must be non-negative, got {total}"
        );
        assert!(total.is_finite(), "loss must be finite, got {total}");
    }

    #[test]
    fn test_contrast_omni_cross_modal_penalty() {
        let mut weights = HashMap::new();
        weights.insert(String::from("mod0"), 1.0);
        weights.insert(String::from("mod1"), 2.0);
        let cfg = ContrastOmniConfig {
            temperature: 0.07,
            modality_weights: weights,
            hierarchy_levels: 3,
        };
        let loss = ContrastOmniLoss::new(cfg);
        // Modality 0 cluster around [10,0,0], modality 1 around [0,10,0].
        let features = vec![
            10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 11.0, 0.0,
        ];
        let modality_ids = vec![0, 0, 1, 1];
        let modality_names = vec![
            String::from("mod0"),
            String::from("mod0"),
            String::from("mod1"),
            String::from("mod1"),
        ];
        let labels = vec![0, 1, 2, 3];
        let total =
            loss.hierarchical_contrastive(&features, &modality_ids, &modality_names, &labels);
        // Cross-modal Fréchet term should be positive because the clusters are far apart.
        assert!(
            total > 0.0,
            "cross-modal penalty should be positive, got {total}"
        );
    }
}
