//! Qwen3-MoE transformer — routes every layer through a shared `MoeBlock`.
//!
//! Qwen3-MoE uses a softmax router (no correction bias) and no shared expert.
//! Attention towers are plain Llama-style; the MoE routing replaces the dense
//! FFN per layer.

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
pub struct Qwen3MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Scaling applied to the routed expert output (`routed_scaling_factor`).
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for Qwen3MoeConfig {
    fn name(&self) -> &str {
        "qwen3moe"
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

pub struct Qwen3Moe {
    pub cfg: Qwen3MoeConfig,
    pub device: Device,
    pub inner: Llama,
}

impl Qwen3Moe {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: Qwen3MoeConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen3MoeConfig,
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

        // Qwen3-MoE routes every layer through the MoE block: softmax router,
        // no shared expert.
        let spec = MoESpec {
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            router_kind: RouterKind::SoftmaxTopK,
            routed_scaling_factor: cfg.routed_scaling_factor,
            has_shared_expert: false,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
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

impl Model for Qwen3Moe {
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

impl CausalLm for Qwen3Moe {
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
