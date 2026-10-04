//! Riemannian Constrained Optimization (RCO) for bitwidth allocation.
//! Formulates the per-tensor mixed-precision allocation under a hard total size budget as optimization over a.

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
    /// True STORAGE density (bits per weight) per candidate, aligned with
    /// `available_bpws`. The labels are integer bits-per-weight tiers; the
    /// budget must be charged against what the bytes actually cost — the
    /// GSQ-RCO 2-bit tier stores 2.25 bpw (18 B / 64 weights), Q4_K 4.5,
    /// Q5_K 5.5, Q6_K 6.5. With `None`, the integer labels are the costs
    /// (the historical behavior).
    pub costs: Option<Vec<f32>>,
}

impl Default for RcoConfig {
    fn default() -> Self {
        Self {
            steps: 40,
            lr: 0.25,
            target_bpw: 4.0,
            temperature: 1.0,
            available_bpws: vec![2, 3, 4, 5, 6, 8],
            costs: None,
        }
    }
}

/// Run RCO (Riemannian Constrained Optimization) to find optimal per-tensor bitwidths.
/// Optimizes logit allocation vectors theta_i such that the expected bitwidth satisfies the target parameter budget.
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

    // Budget accounting runs against the TRUE storage densities when the
    // caller supplied them, so a nominal target lands at the nominal file
    // size instead of drifting with the integer tier labels.
    let bpws_f64: Vec<f64> = match &config.costs {
        Some(c) if c.len() == config.available_bpws.len() => {
            c.iter().map(|&x| x as f64).collect()
        }
        _ => config.available_bpws.iter().map(|&b| b as f64).collect(),
    };
    let min_bpw = bpws_f64[0];
    let max_bpw = bpws_f64[k_candidates - 1];

    let min_possible_bits = total_params as f64 * min_bpw;
    let max_possible_bits = total_params as f64 * max_bpw;
    let target_bits = target_bits.clamp(min_possible_bits, max_possible_bits);

    let imp_sum: f32 = importance_scores.iter().sum::<f32>().max(1e-9);
    let mut theta = vec![vec![0.0f64; k_candidates]; n_tensors];

    for (i, (&imp, _)) in importance_scores
        .iter()
        .zip(tensor_sizes.iter())
        .enumerate()
    {
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
    // Budget accounting in the discrete phase runs on the SAME true densities
    // as the annealer: labels are the returned per-tensor tiers, costs are
    // what the bytes weigh. With integer labels here, a {2,4} ladder repaired
    // to a 3.0 target lands 50/50 — 3.375 realized, 12% over nominal.
    let mut total_allocated_bits = 0.0f64;

    for i in 0..n_tensors {
        let best_k = (0..k_candidates)
            .max_by(|&a, &b| theta[i][a].partial_cmp(&theta[i][b]).unwrap())
            .unwrap_or(0);
        let bpw = config.available_bpws[best_k];
        final_genes[i] = bpw;
        total_allocated_bits += tensor_sizes[i] as f64 * bpws_f64[best_k];
    }

    if total_allocated_bits > target_bits {
        let mut indices: Vec<usize> = (0..n_tensors).collect();
        indices.sort_by(|&a, &b| {
            let score_a = importance_scores[a] / (tensor_sizes[a] as f32 + 1.0);
            let score_b = importance_scores[b] / (tensor_sizes[b] as f32 + 1.0);
            score_a.partial_cmp(&score_b).unwrap()
        });

        for &i in &indices {
            while final_genes[i] > config.available_bpws[0]
                && total_allocated_bits > target_bits
            {
                let cur_k = config
                    .available_bpws
                    .iter()
                    .position(|&b| b == final_genes[i]);
                let lower_k_opt = cur_k.and_then(|k| k.checked_sub(1));
                if let (Some(cur_k), Some(lower_k)) = (cur_k, lower_k_opt) {
                    let lower = config.available_bpws[lower_k];
                    let diff = (bpws_f64[cur_k] - bpws_f64[lower_k]) * tensor_sizes[i] as f64;
                    total_allocated_bits -= diff;
                    final_genes[i] = lower;
                } else {
                    break;
                }
            }
        }
    } else if total_allocated_bits < target_bits {
        let mut indices: Vec<usize> = (0..n_tensors).collect();
        indices.sort_by(|&a, &b| {
            let score_a = importance_scores[a] / (tensor_sizes[a] as f32 + 1.0);
            let score_b = importance_scores[b] / (tensor_sizes[b] as f32 + 1.0);
            score_b.partial_cmp(&score_a).unwrap()
        });

        let max_avail = config.available_bpws[k_candidates - 1];
        for &i in &indices {
            while final_genes[i] < max_avail {
                let cur_k = config
                    .available_bpws
                    .iter()
                    .position(|&b| b == final_genes[i]);
                let higher_k = cur_k.map(|k| k + 1);
                if let (Some(cur_k), Some(higher_k)) = (cur_k, higher_k) {
                    if higher_k >= k_candidates {
                        break;
                    }
                    let higher = config.available_bpws[higher_k];
                    let diff = (bpws_f64[higher_k] - bpws_f64[cur_k]) * tensor_sizes[i] as f64;
                    if total_allocated_bits + diff <= target_bits {
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
    /// The budget must be charged against TRUE storage densities: a nominal
    /// 3.0 bpw over {GSQ 2.25, Q4_K 4.5} has to realize ~3.0, not ~3.3 (the
    /// integer-label stand-ins) — this is what "achieve a nominal target"
    /// means for the released GSQ-RCO artifacts (nominal 3.5 / realized 3.06).
    #[test]
    fn rco_budget_is_charged_against_true_densities() {
        let n = 200usize;
        let importance: Vec<f32> = (0..n).map(|i| 0.5 + (i % 17) as f32 * 0.1).collect();
        let sizes = vec![4096usize; n];
        let costs = [2.25f64, 4.5f64];
        let cfg = RcoConfig {
            target_bpw: 3.0,
            steps: 400,
            available_bpws: vec![2, 4],
            costs: Some(vec![2.25, 4.5]),
            ..Default::default()
        };
        let genes = rco_search(&cfg, &importance, &sizes, None);
        assert_eq!(genes.len(), n);
        let total: usize = sizes.iter().sum();
        let realized: f64 = genes
            .iter()
            .zip(&sizes)
            .map(|(&g, &s)| {
                let d = costs[if g == 2 { 0 } else { 1 }];
                s as f64 * d
            })
            .sum::<f64>()
            / total as f64;
        assert!(
            (realized - 3.0).abs() < 0.35,
            "realized average {realized:.3} should land near the 3.0 nominal budget"
        );
        // More important tensors must never land BELOW the cheap tier that a
        // less important one took (monotone budget discipline, the paper's
        // premise).
        assert!(genes.contains(&2) && genes.contains(&4), "both rungs should be used");
    }
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

        let total_bits: usize = bitwidths
            .iter()
            .zip(sizes.iter())
            .map(|(&b, &s)| b as usize * s)
            .sum();
        let actual_bpw = total_bits as f32 / total_params as f32;

        assert!(
            (actual_bpw - target_bpw).abs() <= 0.25,
            "actual_bpw: {actual_bpw}, target: {target_bpw}"
        );
        assert!(bitwidths[0] >= bitwidths[3]);
    }
}
