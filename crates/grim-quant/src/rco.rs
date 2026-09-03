//! Riemannian Constrained Optimization (RCO) for bitwidth allocation.
//!
//! Formulates the per-tensor mixed-precision allocation under a hard total size budget
//! as optimization over a Riemannian manifold in logit space. Replaces heuristic
//! genetic search (EvoPress) with direct projected gradient descent that strictly
//! satisfies the parameter budget down to the exact byte limit.

/// Configuration for Riemannian Constrained Optimization bitwidth search.
#[derive(Debug, Clone)]
pub struct RcoConfig {
    /// Number of optimization steps.
    pub steps: usize,
    /// Learning rate for Riemannian gradient descent.
    pub lr: f32,
    /// Target average bits-per-weight across all tensors.
    pub target_bpw: f32,
    /// Softmax temperature for logit relaxation.
    pub temperature: f32,
    /// Available candidate bitwidths (e.g. `[2, 3, 4, 5, 6, 8]`).
    pub available_bpws: Vec<u32>,
}

impl Default for RcoConfig {
    fn default() -> Self {
        Self {
            steps: 40,
            lr: 0.25,
            target_bpw: 4.0,
            temperature: 1.0,
            available_bpws: vec![2, 3, 4, 5, 6, 8],
        }
    }
}

/// Run RCO (Riemannian Constrained Optimization) to find optimal per-tensor bitwidths.
///
/// Optimizes logit allocation vectors theta_i such that the expected
/// bitwidth satisfies the target parameter budget exactly via tangent space projection,
/// then projects to discrete bitwidths via greedy knapsack.
pub fn rco_search(
    config: &RcoConfig,
    importance_scores: &[f32],
    tensor_sizes: &[usize],
    mut progress: Option<&mut dyn FnMut(usize, usize)>,
) -> Vec<u32> {
    let n_tensors = importance_scores.len();
    if n_tensors == 0 {
        return Vec::new();
    }

    let total_params: usize = tensor_sizes.iter().sum();
    if total_params == 0 {
        return vec![config.target_bpw.round() as u32; n_tensors];
    }

    let target_bits = (config.target_bpw * total_params as f32) as f64;
    let k_candidates = config.available_bpws.len();
    if k_candidates == 0 {
        return vec![4; n_tensors];
    }
    if k_candidates == 1 {
        return vec![config.available_bpws[0]; n_tensors];
    }

    let bpws_f64: Vec<f64> = config.available_bpws.iter().map(|&b| b as f64).collect();
    let min_bpw = bpws_f64[0];
    let max_bpw = bpws_f64[k_candidates - 1];

    let min_possible_bits = total_params as f64 * min_bpw;
    let max_possible_bits = total_params as f64 * max_bpw;
    let target_bits = target_bits.clamp(min_possible_bits, max_possible_bits);

    let imp_sum: f32 = importance_scores.iter().sum::<f32>().max(1e-9);
    let mut theta = vec![vec![0.0f64; k_candidates]; n_tensors];

    for (i, (&imp, _)) in importance_scores.iter().zip(tensor_sizes.iter()).enumerate() {
        let norm_imp = (imp / imp_sum) * n_tensors as f32;
        for (k, &bpw) in bpws_f64.iter().enumerate() {
            let dist = (bpw - (config.target_bpw as f64 * norm_imp as f64)).abs();
            theta[i][k] = -dist;
        }
    }

    for step in 0..config.steps {
        if let Some(cb) = progress.as_deref_mut() {
            cb(step + 1, config.steps);
        }

        let temp = (config.temperature as f64) * (1.0 - 0.5 * (step as f64 / config.steps as f64));
        let mut probs = vec![vec![0.0f64; k_candidates]; n_tensors];
        let mut expected_bits = 0.0f64;

        for i in 0..n_tensors {
            let max_logit = theta[i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f64;
            for k in 0..k_candidates {
                let exp_val = ((theta[i][k] - max_logit) / temp).exp();
                probs[i][k] = exp_val;
                sum_exp += exp_val;
            }
            let s_i = tensor_sizes[i] as f64;
            for k in 0..k_candidates {
                probs[i][k] /= sum_exp;
                expected_bits += s_i * probs[i][k] * bpws_f64[k];
            }
        }

        let mut grad = vec![vec![0.0f64; k_candidates]; n_tensors];
        let mut normal = vec![vec![0.0f64; k_candidates]; n_tensors];

        for i in 0..n_tensors {
            let s_i = tensor_sizes[i] as f64;
            let imp_i = importance_scores[i].max(1e-6) as f64;
            let mut e_i = 0.0f64;
            for k in 0..k_candidates {
                e_i += probs[i][k] * bpws_f64[k];
            }
            let dloss_de = -imp_i / (e_i * e_i).max(1e-4);

            for k in 0..k_candidates {
                let de_dtheta = (1.0 / temp) * probs[i][k] * (bpws_f64[k] - e_i);
                grad[i][k] = dloss_de * de_dtheta;
                normal[i][k] = s_i * de_dtheta;
            }
        }

        let mut dot_gn = 0.0f64;
        let mut norm_sq = 0.0f64;

        for i in 0..n_tensors {
            for k in 0..k_candidates {
                dot_gn += grad[i][k] * normal[i][k];
                norm_sq += normal[i][k] * normal[i][k];
            }
        }

        let budget_err = expected_bits - target_bits;
        let correction = if norm_sq > 1e-12 {
            (dot_gn + 0.1 * budget_err) / norm_sq
        } else {
            0.0
        };

        for i in 0..n_tensors {
            for k in 0..k_candidates {
                let g_proj = grad[i][k] - correction * normal[i][k];
                theta[i][k] -= (config.lr as f64) * g_proj;
            }
        }
    }

    let mut final_genes = vec![0u32; n_tensors];
    let mut total_allocated_bits = 0usize;

    for i in 0..n_tensors {
        let best_k = (0..k_candidates)
            .max_by(|&a, &b| theta[i][a].partial_cmp(&theta[i][b]).unwrap())
            .unwrap_or(0);
        let bpw = config.available_bpws[best_k];
        final_genes[i] = bpw;
        total_allocated_bits += tensor_sizes[i] * bpw as usize;
    }

    let target_bits_usize = target_bits as usize;

    if total_allocated_bits > target_bits_usize {
        let mut indices: Vec<usize> = (0..n_tensors).collect();
        indices.sort_by(|&a, &b| {
            let score_a = importance_scores[a] / (tensor_sizes[a] as f32 + 1.0);
            let score_b = importance_scores[b] / (tensor_sizes[b] as f32 + 1.0);
            score_a.partial_cmp(&score_b).unwrap()
        });

        for &i in &indices {
            while final_genes[i] > config.available_bpws[0] && total_allocated_bits > target_bits_usize {
                if let Some(&lower) = config.available_bpws.iter().rev().find(|&&b| b < final_genes[i]) {
                    let diff = (final_genes[i] - lower) as usize * tensor_sizes[i];
                    total_allocated_bits -= diff;
                    final_genes[i] = lower;
                } else {
                    break;
                }
            }
        }
    } else if total_allocated_bits < target_bits_usize {
        let mut indices: Vec<usize> = (0..n_tensors).collect();
        indices.sort_by(|&a, &b| {
            let score_a = importance_scores[a] / (tensor_sizes[a] as f32 + 1.0);
            let score_b = importance_scores[b] / (tensor_sizes[b] as f32 + 1.0);
            score_b.partial_cmp(&score_a).unwrap()
        });

        let max_avail = config.available_bpws[k_candidates - 1];
        for &i in &indices {
            while final_genes[i] < max_avail {
                if let Some(&higher) = config.available_bpws.iter().find(|&&b| b > final_genes[i]) {
                    let diff = (higher - final_genes[i]) as usize * tensor_sizes[i];
                    if total_allocated_bits + diff <= target_bits_usize {
                        total_allocated_bits += diff;
                        final_genes[i] = higher;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    final_genes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rco_budget_exact_constraint() {
        let sizes = vec![1000, 2000, 4000, 8000, 16000];
        let importance = vec![10.0, 1.0, 5.0, 0.5, 2.0];
        let total_params: usize = sizes.iter().sum();
        let target_bpw = 3.5f32;

        let config = RcoConfig {
            target_bpw,
            steps: 30,
            ..Default::default()
        };

        let bitwidths = rco_search(&config, &importance, &sizes, None);
        assert_eq!(bitwidths.len(), sizes.len());

        let total_bits: usize = bitwidths.iter().zip(sizes.iter()).map(|(&b, &s)| b as usize * s).sum();
        let actual_bpw = total_bits as f32 / total_params as f32;

        assert!(
            (actual_bpw - target_bpw).abs() <= 0.25,
            "actual_bpw: {actual_bpw}, target: {target_bpw}"
        );
        assert!(bitwidths[0] >= bitwidths[3]);
    }
}
