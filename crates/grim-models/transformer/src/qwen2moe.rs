//! Thin wrapper around `Llama` for qwen2moe uses a Llama-style transformer.
// ponytail: dense Llama wrapper alias (no expert stack). Qwen MoE architectures with routed experts are `Qwen3Moe` (`qwen3moe.rs`) and `Qwen35Moe` (`qwen35moe.rs`).

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::TensorParallelConfig;
use grim_tensor::{ArithType, Device, Tensor};

use crate::model::{Llama, LlamaConfig};

// Config

#[derive(Debug, Clone)]
pub struct Qwen2MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for Qwen2MoeConfig {
    fn name(&self) -> &str {
        "qwen2moe"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Model - Qwen2-MoE sparse mixture of experts / Llama wrapper

pub struct Qwen2Moe {
    pub cfg: Qwen2MoeConfig,
    pub device: Device,
    pub inner: Llama,
}

impl Qwen2Moe {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen2MoeConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen2MoeConfig,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        };

        // If num_experts > 0, wire through MoE blocks (Qwen2-MoE: 60 routed experts, 4 active, optional shared expert)
        let inner = if cfg.num_experts > 0 {
            use crate::moe_block::MoESpec;
            use grim_nn::moe::RouterKind;
            let spec = MoESpec {
                num_experts: cfg.num_experts,
                top_k: cfg.num_experts_per_tok,
                router_kind: RouterKind::SoftmaxTopK,
                routed_scaling_factor: if cfg.routed_scaling_factor == 0.0 {
                    1.0
                } else {
                    cfg.routed_scaling_factor
                },
                has_shared_expert: cfg.shared_expert_intermediate_size.is_some(),
                moe_intermediate_size: cfg.moe_intermediate_size,
                shared_expert_intermediate_size: cfg.shared_expert_intermediate_size,
                transposed_expert_layout: false,
            };
            let moe_spec: Vec<Option<MoESpec>> = vec![Some(spec); cfg.num_layers];
            Llama::load_tp_moe(device.clone(), ws, llama_cfg, &moe_spec, tp)?
        } else {
            Llama::load_tp(device.clone(), ws, llama_cfg, tp)?
        };

        Ok(Self {
            cfg,
            device: inner.device.clone(),
            inner,
        })
    }
}

impl Model for Qwen2Moe {
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

impl CausalLm for Qwen2Moe {
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
