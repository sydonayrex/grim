//! OMNILO-PRUNE: joint rank allocation across modalities in LoRA adapter training.
//!
//! Distributes a global LoRA rank budget across transformer layers based on
//! modality weights (e.g. text vs vision tokens) and layer salience. Supports
//! iterative reallocation from gradient-norm feedback.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for joint rank allocation across modalities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniloConfig {
    /// Total rank budget to distribute across all layers.
    #[serde(default = "default_total_rank_budget")]
    pub total_rank_budget: usize,
    /// Per-modality weight multipliers. Layers whose modality is absent from
    /// this map receive a weight of `1.0`.
    #[serde(default)]
    pub modality_rank_weights: HashMap<String, f32>,
    /// Minimum rank assigned to any single layer.
    #[serde(default = "default_min_rank_per_layer")]
    pub min_rank_per_layer: usize,
    /// Maximum rank assigned to any single layer.
    #[serde(default = "default_max_rank_per_layer")]
    pub max_rank_per_layer: usize,
}

fn default_total_rank_budget() -> usize {
    128
}

fn default_min_rank_per_layer() -> usize {
    4
}

fn default_max_rank_per_layer() -> usize {
    32
}

impl Default for OmniloConfig {
    fn default() -> Self {
        Self {
            total_rank_budget: 128,
            modality_rank_weights: HashMap::new(),
            min_rank_per_layer: 4,
            max_rank_per_layer: 32,
        }
    }
}

/// Joint rank allocator across modalities for LoRA adapter training.
#[derive(Debug, Clone)]
pub struct OmniloRankAllocator {
    pub config: OmniloConfig,
    pub current_rank_per_layer: Vec<usize>,
    pub layer_modality: Vec<String>,
}

impl OmniloRankAllocator {
    /// Create a new allocator with an empty current-rank vector sized to
    /// `num_layers`.
    pub fn new(config: OmniloConfig, num_layers: usize) -> Self {
        Self {
            config,
            current_rank_per_layer: vec![0; num_layers],
            layer_modality: vec![String::new(); num_layers],
        }
    }

    /// Distribute `total_budget` across `num_layers` layers according to
    /// modality weights and optional per-layer salience values.
    ///
    /// Importance score per layer = `modality_weight[modality] * salience`.
    /// Raw allocation = importance / sum(importance) * total_budget.
    /// After clamping to [min_rank, max_rank], an iterative correction loop
    /// restores the sum to exactly `total_budget`.
    pub fn allocate(
        &mut self,
        total_budget: usize,
        num_layers: usize,
        layer_modalities: &[String],
        salience: Option<&[f32]>,
    ) -> Vec<usize> {
        let config = &self.config;

        if num_layers == 0 {
            self.current_rank_per_layer.clear();
            self.layer_modality.clear();
            return Vec::new();
        }

        let default_salience = vec![1.0f32; num_layers];
        let salience = salience.unwrap_or(&default_salience);
        let mut weights: Vec<f32> = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let modality = if i < layer_modalities.len() {
                layer_modalities[i].clone()
            } else {
                String::new()
            };
            let modality_weight = config
                .modality_rank_weights
                .get(&modality)
                .copied()
                .unwrap_or(1.0);
            let s = salience.get(i).copied().unwrap_or(1.0);
            weights.push(modality_weight * s.max(0.0));
        }

        let total_weight: f32 = weights.iter().sum();
        let mut ranks: Vec<usize> = if total_weight > 0.0 {
            weights
                .iter()
                .map(|w| ((*w / total_weight) * total_budget as f32).round() as usize)
                .collect()
        } else {
            vec![total_budget / num_layers; num_layers]
        };

        // Clamp to [min_rank, max_rank]
        for rank in &mut ranks {
            *rank = (*rank).clamp(config.min_rank_per_layer, config.max_rank_per_layer);
        }

        // Iteratively adjust to honor total_budget exactly.
        let mut diff = total_budget as isize - ranks.iter().map(|&r| r as isize).sum::<isize>();
        while diff != 0 {
            let mut moved = false;
            for slot in ranks.iter_mut() {
                if diff == 0 {
                    break;
                }
                let r = *slot as isize;
                if diff > 0 && r < config.max_rank_per_layer as isize {
                    *slot += 1;
                    diff -= 1;
                    moved = true;
                } else if diff < 0 && r > config.min_rank_per_layer as isize {
                    *slot -= 1;
                    diff += 1;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }

        self.current_rank_per_layer = ranks.clone();
        self.layer_modality = layer_modalities.to_vec();
        ranks
    }

    /// Reallocate ranks based on observed gradient norms.
    ///
    /// Layers with higher gradient norms receive more rank. Uses the heuristic:
    /// rank_i = clamp(min_rank + (grad_norm_i / max_grad_norm) * avg_budget,
    ///                min_rank, max_rank), then renormalized to total_budget.
    pub fn update_ranks_from_grad_norms(
        &mut self,
        grad_norms: &[f32],
        current_ranks: &[usize],
    ) -> Vec<usize> {
        let config = &self.config;
        let num_layers = current_ranks.len();
        if num_layers == 0 {
            return Vec::new();
        }

        let max_grad = grad_norms.iter().cloned().fold(0.0f32, f32::max);
        let avg_budget = config.total_rank_budget as f32 / num_layers as f32;
        let mut raw: Vec<usize> = if max_grad > 0.0 {
            grad_norms
                .iter()
                .take(num_layers)
                .map(|g| {
                    let candidate = config.min_rank_per_layer as f32 + (g / max_grad) * avg_budget;
                    candidate.round().clamp(
                        config.min_rank_per_layer as f32,
                        config.max_rank_per_layer as f32,
                    ) as usize
                })
                .collect()
        } else {
            current_ranks.to_vec()
        };

        // Renormalize to total_budget.
        let current_sum: usize = raw.iter().sum();
        if current_sum != config.total_rank_budget && current_sum > 0 {
            let mut diff = config.total_rank_budget as isize - current_sum as isize;
            while diff != 0 {
                let mut moved = false;
                for slot in raw.iter_mut() {
                    if diff == 0 {
                        break;
                    }
                    let r = *slot as isize;
                    if diff > 0 && r < config.max_rank_per_layer as isize {
                        *slot += 1;
                        diff -= 1;
                        moved = true;
                    } else if diff < 0 && r > config.min_rank_per_layer as isize {
                        *slot -= 1;
                        diff += 1;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }

        self.current_rank_per_layer = raw.clone();
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omnilo_allocate_respects_budget() {
        let cfg = OmniloConfig {
            total_rank_budget: 128,
            ..Default::default()
        };
        let mut alloc = OmniloRankAllocator::new(cfg, 4);
        let modalities = vec![
            "text".into(),
            "vision".into(),
            "text".into(),
            "vision".into(),
        ];
        let ranks = alloc.allocate(128, 4, &modalities, None);
        assert_eq!(ranks.iter().sum::<usize>(), 128);
    }

    #[test]
    fn test_omnilo_allocate_min_max() {
        let cfg = OmniloConfig {
            total_rank_budget: 128,
            min_rank_per_layer: 4,
            max_rank_per_layer: 32,
            ..Default::default()
        };
        let mut alloc = OmniloRankAllocator::new(cfg, 8);
        let modalities = vec![
            "text".into(),
            "vision".into(),
            "text".into(),
            "vision".into(),
            "text".into(),
            "vision".into(),
            "text".into(),
            "vision".into(),
        ];
        let ranks = alloc.allocate(128, 8, &modalities, None);
        for r in &ranks {
            assert!(*r >= 4, "rank {r} below min");
            assert!(*r <= 32, "rank {r} above max");
        }
    }

    #[test]
    fn test_omnilo_update_from_grads() {
        let cfg = OmniloConfig {
            total_rank_budget: 128,
            max_rank_per_layer: 128,
            ..OmniloConfig::default()
        };
        let mut alloc = OmniloRankAllocator::new(cfg.clone(), 3);
        let current = vec![4, 4, 4];
        let grad_norms = vec![1.0, 10.0, 0.1];
        let updated = alloc.update_ranks_from_grad_norms(&grad_norms, &current);
        assert_eq!(updated.iter().sum::<usize>(), cfg.total_rank_budget);
        assert!(updated[1] >= updated[0]);
        assert!(updated[1] >= updated[2]);
    }
}
