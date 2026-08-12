//! Laguna 2 MoE transformer — routes every layer through a shared `MoeBlock`.
//!
//! Laguna uses a sigmoid router with a noisy-router / dedup correction bias
//! (`exp_probs_b`) and an always-on shared expert. Attention towers are plain
//! Llama-style full-context causal GQA; the MoE routing replaces the dense FFN
//! per layer.
//!
//! Note: sliding-window / interleaved local-global attention is *not* supported
//! here. The attention kernels in `block.rs` have no windowing path, and
//! Laguna's design uses plain causal attention. (Hybrid windowed attention
//! exists elsewhere — see `muse_glimmer` — but is unrelated to this model.)

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, Model, ModelConfig, ModalityHint};
use grim_core::session::SessionT;
use grim_nn::moe::RouterKind;
use grim_nn::TensorParallelConfig;
use grim_tensor::{ArithType, Device, Tensor};

use crate::model::{Llama, LlamaConfig};
use crate::moe_block::MoESpec;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LagunaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Scaling applied to the routed expert output before adding the shared
    /// expert contribution (`routed_scaling_factor`).
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for LagunaConfig {
    fn name(&self) -> &str {
        "laguna"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Laguna {
    pub cfg: LagunaConfig,
    pub device: Device,
    pub inner: Llama,
}

impl Laguna {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: LagunaConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: LagunaConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let llama_cfg = LlamaConfig {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            num_layers: cfg.num_layers,
            intermediate_size: cfg.intermediate_size,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            max_seq_len: cfg.max_seq_len,
        };

        // Laguna routes every layer through the MoE block: sigmoid router with
        // correction bias + always-on shared expert. `MoeBlock::load` reads the
        // real `exp_probs_b` correction bias from the checkpoint when the router
        // kind is `SigmoidTopKWithBias`, so no bias tensor is constructed here.
        let spec = MoESpec {
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            router_kind: RouterKind::SigmoidTopKWithBias,
            routed_scaling_factor: cfg.routed_scaling_factor,
            has_shared_expert: true,
        };
        let moe_spec: Vec<Option<MoESpec>> = vec![Some(spec); cfg.num_layers];

        let inner = Llama::load_tp_moe(device.clone(), ws, llama_cfg, &moe_spec, tp)?;
        Ok(Self {
            cfg,
            device: inner.device.clone(),
            inner,
        })
    }
}

impl Model for Laguna {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for Laguna {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.inner.new_session()
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        self.inner.forward(session, input_ids, positions, adapters)
    }
}
