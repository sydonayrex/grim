//! Fused Mixture-of-Experts grouped dispatch primitive for CPU backend.

use grim_tensor::error::{Error, Result};

use crate::device::gemm_dispatch;


/// Fused CPU MoE grouped dispatch: Top-K routing, expert token gather, grouped GEMM,
/// SwiGLU gating, and weighted scatter reduction.
pub fn moe_fused_dispatch(
    tokens: &[f32],
    gate_logits: &[f32],
    w_gate: &[Vec<f32>],
    w_up: &[Vec<f32>],
    w_down: &[Vec<f32>],
    num_tokens: usize,
    hidden_dim: usize,
    inter_dim: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<Vec<f32>> {
    if tokens.len() != num_tokens * hidden_dim {
        return Err(Error::ShapeMismatch {
            expected: vec![num_tokens * hidden_dim],
            got: vec![tokens.len()],
        });
    }

    let mut out = vec![0.0f32; num_tokens * hidden_dim];

    for t in 0..num_tokens {
        let tok_src = &tokens[t * hidden_dim..(t + 1) * hidden_dim];
        let logits = &gate_logits[t * num_experts..(t + 1) * num_experts];

        // Softmax over gate_logits
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

        // Top-K selection
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);

        // Normalize top-k weights
        let top_sum: f32 = indexed.iter().map(|(_, w)| *w).sum();
        let norm_weights: Vec<(usize, f32)> = if top_sum > 0.0 {
            indexed.into_iter().map(|(e, w)| (e, w / top_sum)).collect()
        } else {
            indexed
        };

        for &(exp_idx, weight) in &norm_weights {
            if exp_idx >= num_experts {
                continue;
            }

            // Gate GEMM: [1, hidden_dim] @ [hidden_dim, inter_dim] -> [1, inter_dim]
            let mut gate_out = vec![0.0f32; inter_dim];
            if exp_idx < w_gate.len() && w_gate[exp_idx].len() == hidden_dim * inter_dim {
                gemm_dispatch(tok_src, &w_gate[exp_idx], &mut gate_out, 1, inter_dim, hidden_dim);
            }

            // Up GEMM: [1, hidden_dim] @ [hidden_dim, inter_dim] -> [1, inter_dim]
            let mut up_out = vec![0.0f32; inter_dim];
            if exp_idx < w_up.len() && w_up[exp_idx].len() == hidden_dim * inter_dim {
                gemm_dispatch(tok_src, &w_up[exp_idx], &mut up_out, 1, inter_dim, hidden_dim);
            }

            // SwiGLU: silu(gate_out) * up_out
            let mut activated = vec![0.0f32; inter_dim];
            for i in 0..inter_dim {
                let g = gate_out[i];
                let silu = g / (1.0 + (-g).exp());
                activated[i] = silu * up_out[i];
            }

            // Down GEMM: [1, inter_dim] @ [inter_dim, hidden_dim] -> [1, hidden_dim]
            let mut down_out = vec![0.0f32; hidden_dim];
            if exp_idx < w_down.len() && w_down[exp_idx].len() == inter_dim * hidden_dim {
                gemm_dispatch(&activated, &w_down[exp_idx], &mut down_out, 1, hidden_dim, inter_dim);
            }

            // Weighted scatter accumulate
            let tok_dst = &mut out[t * hidden_dim..(t + 1) * hidden_dim];
            for i in 0..hidden_dim {
                tok_dst[i] += weight * down_out[i];
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_dispatch_evaluates_tokens() {
        let (num_tokens, hidden_dim, inter_dim, num_experts, top_k) = (1, 4, 8, 2, 1);
        let tokens = vec![1.0, 1.0, 1.0, 1.0];
        let logits = vec![10.0, 0.0]; // Expert 0 selected
        let w_gate = vec![vec![0.1; 32], vec![0.0; 32]];
        let w_up = vec![vec![0.1; 32], vec![0.0; 32]];
        let w_down = vec![vec![0.1; 32], vec![0.0; 32]];

        let res = moe_fused_dispatch(
            &tokens,
            &logits,
            &w_gate,
            &w_up,
            &w_down,
            num_tokens,
            hidden_dim,
            inter_dim,
            num_experts,
            top_k,
        )
        .expect("moe_fused_dispatch");

        assert_eq!(res.len(), 4);
        assert!(res[0] > 0.0);
    }
}
