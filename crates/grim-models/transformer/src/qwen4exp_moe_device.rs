//! Device-oriented and expert-major MoE helpers for Qwen3.8-Flash-Next (Q1b, Q2).

use std::collections::BTreeMap;

use crate::qwen4exp_baseline_probe::{self as probe, ProbeStage};
use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_tensor::{Shape, Tensor};

/// Returns `true` when `GRIM_MOE_EXPERT_MAJOR` is truthy (`1` or `true`).
#[inline]
pub fn moe_expert_major_enabled() -> bool {
    std::env::var("GRIM_MOE_EXPERT_MAJOR")
        .ok()
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

/// Host routing oracle matching `Qwen38MoeBlock::forward`:
/// global max-subtracted softmax over all experts, stable descending sort
/// (ties -> lower expert index), no top-k renormalization, scaled by
/// `routed_scaling_factor`.
pub fn route_row_host(
    row: &[f32],
    top_k: usize,
    routed_scaling_factor: f32,
) -> (Vec<usize>, Vec<f32>) {
    let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.min(row.len());
    let topk = &indexed[..k];

    let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = row.iter().map(|l| (l - max_l).exp()).sum::<f32>() + 1e-12;
    let weights: Vec<f32> = topk
        .iter()
        .map(|(_, l)| ((l - max_l).exp() / denom) * routed_scaling_factor)
        .collect();
    let ids: Vec<usize> = topk.iter().map(|(id, _)| *id).collect();
    (ids, weights)
}

/// Expert-major MoE prefill grouping:
/// - Groups `(token_idx, slot_idx)` assignments by `expert_id` so each distinct
///   active expert is staged and executed once per chunk.
/// - Stores per-`(token, slot)` outputs and reduces each token in its ORIGINAL
///   `0..k` top-k slot order so CPU output is bitwise identical to the
///   token-major loop.
/// - Batches the shared expert once over the full `[seq_len, hidden_dim]` input.
#[allow(clippy::too_many_arguments)]
pub fn expert_major_prefill<F, G>(
    x: &Tensor,
    x_vec: &[f32],
    logits_vec: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    num_exp: usize,
    top_k: usize,
    routed_scaling_factor: f32,
    has_shared: bool,
    mut run_expert: F,
    mut run_shared: G,
) -> Result<Vec<f32>>
where
    F: FnMut(usize, &Tensor) -> Result<Tensor>,
    G: FnMut(&Tensor) -> Result<Tensor>,
{
    let k = top_k.min(num_exp);
    let mut token_weights: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();

    for s in 0..seq_len {
        let (ids, weights) = probe::span(ProbeStage::Router, || {
            route_row_host(
                &logits_vec[s * num_exp..(s + 1) * num_exp],
                k,
                routed_scaling_factor,
            )
        });
        for (slot, &exp_id) in ids.iter().enumerate() {
            groups.entry(exp_id).or_default().push((s, slot));
        }
        token_weights.push(weights);
    }

    let mut slot_outputs: Vec<Vec<f32>> = vec![Vec::new(); seq_len * k];

    probe::span(ProbeStage::ExpertStage, || -> Result<()> {
        for (exp_id, assignments) in &groups {
            let n_tok = assignments.len();
            let mut batch_host_vec = Vec::with_capacity(n_tok * hidden_dim);
            for &(s, _) in assignments {
                batch_host_vec.extend_from_slice(&x_vec[s * hidden_dim..(s + 1) * hidden_dim]);
            }
            let batch_host = cpu_tensor(batch_host_vec, Shape::new(vec![n_tok, hidden_dim]));
            let batch_x = if x.device().is_cpu() {
                batch_host
            } else {
                probe::record_h2d(
                    ProbeStage::ExpertStage,
                    n_tok * hidden_dim * std::mem::size_of::<f32>(),
                );
                grim_nn::modules::move_to_device(&batch_host, x.device())?
            };
            let exp_out = run_expert(*exp_id, &batch_x)?.to_vec_f32()?;
            if !x.device().is_cpu() {
                probe::record_d2h(
                    ProbeStage::ExpertStage,
                    exp_out.len() * std::mem::size_of::<f32>(),
                );
            }
            for (r, &(s, slot)) in assignments.iter().enumerate() {
                slot_outputs[s * k + slot] =
                    exp_out[r * hidden_dim..(r + 1) * hidden_dim].to_vec();
            }
        }
        Ok(())
    })?;

    let mut out_vec = vec![0.0f32; seq_len * hidden_dim];
    for s in 0..seq_len {
        let weights = &token_weights[s];
        for (slot, &w) in weights.iter().enumerate() {
            let eo = &slot_outputs[s * k + slot];
            for d in 0..hidden_dim {
                out_vec[s * hidden_dim + d] += w * eo[d];
            }
        }
    }

    if has_shared {
        let shared_out = probe::span(ProbeStage::ExpertStage, || -> Result<Vec<f32>> {
            let v = run_shared(x)?.to_vec_f32()?;
            if !x.device().is_cpu() {
                probe::record_d2h(ProbeStage::ExpertStage, v.len() * std::mem::size_of::<f32>());
            }
            Ok(v)
        })?;
        for s in 0..seq_len {
            for d in 0..hidden_dim {
                out_vec[s * hidden_dim + d] += shared_out[s * hidden_dim + d];
            }
        }
    }

    Ok(out_vec)
}
