//! Shared Mixture-of-Experts dispatch (Phase 3a).
//!
//! Consolidates the per-expert loop pattern used by deepseek2/32/4, kimi_k3,
//! mellum into a single `fused_moe_dispatch` entry point. On ROCm the dispatch
//! can route through Charon's grouped fused-kernel (`grim_moe_fused_grouped`),
//! which runs all selected experts for a batch of tokens in far fewer launches
//! than the per-token/per-expert loop. Falls back to the stock per-expert loop
//! on CPU, non-ROCm, or when the backend lacks the needed primitives.
//!
//! ADOPTION STATUS (2026-09-12): the module and its per-expert fallback path
//! are wired and compile. Full numerical parity of a Charon-backed path vs the
//! per-expert loop requires checkpoint-level testing on deepseek2/32/4 and
//! kimi_k3 (not verifiable in this environment). The fallback path is
//! semantically identical to the pre-existing per-expert loops.

use std::sync::Arc;

use grim_core::error::Result;
use grim_nn::modules::silu_mul_on_device;
use grim_nn::Linear;
use grim_tensor::{Shape, Tensor};

/// One expert's SwiGLU FFN as three Linear layers (gate/up projection + down
/// projection), matching the `w1`/`w3`/`w2` naming used across the MoE models.
pub struct MoeExpert {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// Routing result for a single token: (expert_index, combine_weight) pairs.
pub type TokenRouting = Vec<(usize, f32)>;

/// Fused MoE dispatch: routes `x` through the selected experts and returns the
/// weighted sum (plus optional shared-expert contribution).
///
/// * `experts` - per-expert FFN weights.
/// * `shared_expert` - optional always-on shared expert applied to all tokens.
/// * `routings` - per-token selected-expert indices + combine weights (top-k).
/// * `routed_scaling_factor` - architecture-specific scale (DeepSeek dedup gating).
///
/// On ROCm with a capable backend this could dispatch through Charon's grouped
/// fused kernel; currently it runs the per-expert loop on-device, which is the
/// reference semantics a Charon path must match.
pub fn fused_moe_dispatch(
    dev: &dyn grim_tensor::BackendDevice,
    x: &Tensor,
    experts: &[MoeExpert],
    shared_expert: Option<&MoeExpert>,
    routings: &[TokenRouting],
    routed_scaling_factor: f32,
) -> Result<Tensor> {
    per_expert_loop(dev, x, experts, shared_expert, routings, routed_scaling_factor)
}

fn per_expert_loop(
    dev: &dyn grim_tensor::BackendDevice,
    x: &Tensor,
    experts: &[MoeExpert],
    shared_expert: Option<&MoeExpert>,
    routings: &[TokenRouting],
    routed_scaling_factor: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    let seq_len = dims[0];
    let hidden_dim = dims[1];

    let out_st = dev.zeros(x.shape(), grim_tensor::DType::F32)?;

    for s in 0..seq_len {
        let routing = &routings[s];
        if routing.is_empty() {
            continue;
        }
        let tok_shape = Shape::new(vec![1, hidden_dim]);
        let token_st = dev.alloc_storage(&tok_shape, grim_tensor::DType::F32)?;
        dev.copy_slice_range(
            token_st.as_ref(),
            0,
            x.storage().as_ref(),
            s * hidden_dim,
            hidden_dim,
        )?;
        let token_x = Tensor::new(
            Arc::from(token_st),
            tok_shape,
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::default(),
            x.device().clone(),
        );

        let mut acc: Option<Tensor> = None;
        for (exp_idx, weight) in routing {
            let expert = &experts[*exp_idx];
            let exp_out = expert_forward(expert, &token_x)?;
            let w = *weight * routed_scaling_factor;
            let (scaled_st, _handle) = dev.mul_scalar(exp_out.storage().as_ref(), w, exp_out.shape())?;
            let scaled = Tensor::new(
                Arc::from(scaled_st),
                exp_out.shape().clone(),
                grim_tensor::DType::F32,
                grim_tensor::QuantProvenance::default(),
                exp_out.device().clone(),
            );
            acc = Some(match acc {
                Some(a) => grim_nn::modules::add_on_device(&a, &scaled)?,
                None => scaled,
            });
        }
        if let Some(acc) = acc {
            dev.copy_slice_into(out_st.as_ref(), acc.storage().as_ref(), s * hidden_dim, hidden_dim)?;
        }
    }

    let mut out_t = Tensor::new(
        Arc::from(out_st),
        x.shape().clone(),
        grim_tensor::DType::F32,
        grim_tensor::QuantProvenance::default(),
        x.device().clone(),
    );

    if let Some(shared) = shared_expert {
        let sh_out = expert_forward(shared, x)?;
        out_t = grim_nn::modules::add_on_device(&out_t, &sh_out)?;
    }

    Ok(out_t)
}

/// Compute per-token top-k routing from flat gate logits. `logits_v` has layout
/// `[seq_len, num_experts]`. Returns one `TokenRouting` per token (sorted by raw
/// logit, descending) with softmax-normalized combine weights (no architecture
/// scaling applied — multiply by `routed_scaling_factor` at dispatch time).
pub fn route_topk(
    logits_v: &[f32],
    num_experts: usize,
    top_k: usize,
) -> Result<Vec<TokenRouting>> {
    let seq_len = if num_experts == 0 { 0 } else { logits_v.len() / num_experts };
    let mut out = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let row = &logits_v[s * num_experts..(s + 1) * num_experts];
        let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(num_experts);
        let topk = &indexed[..k];
        let mut entry: Vec<(usize, f32)> = topk.iter().map(|(idx, _)| (*idx, 0.0)).collect();
        let weights = normalize_weights(topk);
        for (j, (idx, _)) in topk.iter().enumerate() {
            entry[j] = (*idx, weights[j]);
        }
        out.push(entry);
    }
    Ok(out)
}

/// Softmax-normalize the top-k combine weights.
pub fn normalize_weights(topk: &[(usize, f32)]) -> Vec<f32> {
    if topk.is_empty() {
        return Vec::new();
    }
    let max_l = topk.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
    let sum_e: f32 = exps.iter().sum();
    exps.iter().map(|e| e / (sum_e + 1e-12)).collect()
}

fn expert_forward(expert: &MoeExpert, x: &Tensor) -> Result<Tensor> {
    let lift = |r: std::result::Result<Tensor, grim_tensor::Error>| r.map_err(grim_core::error::Error::from);
    let gate = lift(expert.gate.forward(x))?;
    let up = lift(expert.up.forward(x))?;
    let swiglu = silu_mul_on_device(&gate, &up)?;
    lift(expert.down.forward(&swiglu))
}
