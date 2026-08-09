//! Shared Mixture-of-Experts block (WI-M2/M3).
//!
//! A `MoeBlock` replaces the dense SwiGLU FFN in a transformer layer: the
//! attention output is RMS-normalized, then routed through a `grim_nn::moe`
//! router + expert bank (+ optional shared expert). Every MoE architecture
//! (Qwen2/3-MoE, Laguna, GLM4, Granite-MoE, Phi, DBRX, OLMoE, BailingMoE,
//! Nemotron-hMoE, ...) funnels through this single implementation and
//! differs only by its `MoESpec` (expert counts, router kind, shared expert).

use grim_core::error::Result;
use grim_nn::moe::{ExpertBank, ExpertTriple, MoeFfn, MoeRouter, RouterKind};
use grim_nn::{
    Linear, RmsNorm, TensorParallelConfig, WeightSource,
};
use grim_tensor::{Shape, Tensor};

use crate::model::LlamaConfig;

/// Per-architecture routing configuration that distinguishes MoE families.
#[derive(Debug, Clone)]
pub struct MoESpec {
    /// Total number of experts (`expert_count`).
    pub num_experts: usize,
    /// Experts activated per token (`expert_used_count` / top-k).
    pub top_k: usize,
    /// Router scoring convention: softmax (Qwen/GLM/Granite/Phi/...) or
    /// sigmoid+bias (Laguna, DeepSeek-V2/V3 dedup gating).
    pub router_kind: RouterKind,
    /// Scaling applied to the (routed) expert output before adding the
    /// shared-expert contribution (`routed_scaling_factor`).
    pub routed_scaling_factor: f32,
    /// Whether this architecture carries an always-on shared expert
    /// (`ffn_gate_she` / `ffn_up_she` / `ffn_down_she`).
    pub has_shared_expert: bool,
}

/// A single MoE transformer layer's feed-forward (routing) block.
pub struct MoeBlock {
    pub ffn_norm: RmsNorm,
    pub moe: MoeFfn,
    pub tp_config: TensorParallelConfig,
}

impl MoeBlock {
    /// Load a MoE block from a `WeightSource` positioned at the layer root
    /// (i.e. the caller has already done `ws.pp("layers").pp(&i.to_string())`).
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &LlamaConfig,
        spec: &MoESpec,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        // Router gate. llama.cpp stores the expert router as
        // `ffn_gate_inp.weight` = [hidden, num_experts].
        // `Linear::load` is TP-aware via `ws`'s tensor-parallel config.
        let gate = Linear::load(
            &ws.pp("ffn").pp("gate_inp"),
            cfg.hidden_size,
            spec.num_experts,
            /*has_bias=*/ false,
        )?;

        // Optional dedup/noisy-router correction bias (Laguna / DeepSeek).
        let correction_bias = match &spec.router_kind {
            RouterKind::SigmoidTopKWithBias { .. } => {
                let b = ws.get(Shape::new(vec![spec.num_experts]), "exp_probs_b")?;
                Some(b)
            }
            RouterKind::SoftmaxTopK => None,
        };

        let router = MoeRouter::new(
            gate,
            spec.router_kind.clone(),
            spec.top_k,
            spec.num_experts,
            correction_bias,
        );

        // Per-expert SwiGLU triples from the 3D GGUF layout
        // (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`).
        let experts = ExpertBank::load(
            ws,
            spec.num_experts,
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/ false,
        )?;

        // Optional always-on shared expert.
        let shared_expert: Option<ExpertTriple> = if spec.has_shared_expert {
            Some(ExpertTriple::load(
                ws,
                cfg.hidden_size,
                cfg.intermediate_size,
                /*has_bias=*/ false,
            )?)
        } else {
            None
        };

        let moe = MoeFfn::new(router, experts, shared_expert, spec.routed_scaling_factor);

        Ok(Self {
            ffn_norm,
            moe,
            tp_config: tp,
        })
    }

    /// Forward: RMS-norm the attention output, then route through the MoE.
    /// `x` is `[batch, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.ffn_norm.forward(x)?;
        let routed = self.moe.forward(&normed)?;
        Ok(routed)
    }
}
