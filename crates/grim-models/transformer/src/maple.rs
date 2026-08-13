//! Maple 20B-A1B MoE transformer — 256 experts, 8 active per token, 3:1 SWA-512 hybrid attention.

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
pub struct MapleConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub mlp_only_layers: Vec<usize>,
    pub layer_types: Vec<String>,
    pub sliding_window: usize,
    pub num_attention_heads_per_layer: Vec<usize>,
    pub full_rope_theta: f32,
    pub sliding_rope_theta: f32,
    pub full_partial_rotary_factor: f32,
    pub sliding_partial_rotary_factor: f32,
    pub gating: String,
    pub full_yarn: Option<grim_tensor::YaRNParams>,
}

impl Default for MapleConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128000,
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            num_layers: 24,
            intermediate_size: 8192,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            num_experts: 256,
            num_experts_per_tok: 8,
            routed_scaling_factor: 1.0,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 131072,
            mlp_only_layers: vec![],
            layer_types: (0..24)
                .map(|i| {
                    if i % 4 == 3 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect(),
            sliding_window: 512,
            num_attention_heads_per_layer: (0..24).map(|_| 32).collect(),
            full_rope_theta: 500000.0,
            sliding_rope_theta: 10000.0,
            full_partial_rotary_factor: 1.0,
            sliding_partial_rotary_factor: 1.0,
            gating: "per-head".to_string(),
            full_yarn: None,
        }
    }
}

impl ModelConfig for MapleConfig {
    fn name(&self) -> &str {
        "maple"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Model — thin wrapper around Llama with MoE & Hybrid Attention specs
// ---------------------------------------------------------------------------

pub struct Maple {
    pub cfg: MapleConfig,
    pub device: Device,
    pub inner: Llama,
}

impl Maple {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: MapleConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: MapleConfig,
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

        let router_kind = RouterKind::SigmoidTopKWithBias;

        let spec = MoESpec {
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            router_kind,
            routed_scaling_factor: cfg.routed_scaling_factor,
            has_shared_expert: true,
            moe_intermediate_size: Some(cfg.moe_intermediate_size),
            shared_expert_intermediate_size: Some(cfg.shared_expert_intermediate_size),
        };

        let moe_spec: Vec<Option<MoESpec>> = (0..cfg.num_layers)
            .map(|i| {
                if cfg.mlp_only_layers.contains(&i) {
                    None
                } else {
                    Some(spec.clone())
                }
            })
            .collect();

        let attn_specs: Vec<crate::block::LayerAttentionSpec> = (0..cfg.num_layers)
            .map(|i| {
                let layer_type = cfg.layer_types.get(i).map(|s| s.as_str()).unwrap_or("full_attention");
                let is_sliding = layer_type == "sliding_attention";
                let attn_type = if is_sliding {
                    crate::block::AttentionType::Sliding
                } else {
                    crate::block::AttentionType::Full
                };

                let num_heads = cfg
                    .num_attention_heads_per_layer
                    .get(i)
                    .copied()
                    .unwrap_or(cfg.num_heads);

                let theta = if is_sliding {
                    cfg.sliding_rope_theta
                } else {
                    cfg.full_rope_theta
                };

                let partial_factor = if is_sliding {
                    cfg.sliding_partial_rotary_factor
                } else {
                    cfg.full_partial_rotary_factor
                };

                let rotary_dim = (cfg.head_dim as f32 * partial_factor).round() as usize;
                let yarn = if !is_sliding { cfg.full_yarn } else { None };

                let rope = grim_tensor::RopeConfig {
                    dim: cfg.head_dim,
                    base: theta,
                    rotary_dim,
                    yarn,
                };

                let sliding_window = if is_sliding {
                    Some(cfg.sliding_window)
                } else {
                    None
                };

                let has_attn_gate = cfg.gating == "per-head";

                crate::block::LayerAttentionSpec {
                    attn_type,
                    num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    rope,
                    sliding_window,
                    has_attn_gate,
                }
            })
            .collect();

        let inner = Llama::load_tp_moe_specs(device.clone(), ws, llama_cfg, &moe_spec, &attn_specs, tp)?;

        Ok(Self {
            cfg,
            device: inner.device.clone(),
            inner,
        })
    }
}

impl Model for Maple {
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

impl CausalLm for Maple {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maple_config_default_structure() {
        let cfg = MapleConfig::default();
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.sliding_window, 512);

        // Verify 3:1 SWA ratio (layers 0..2 sliding, layer 3 full)
        assert_eq!(cfg.layer_types[0], "sliding_attention");
        assert_eq!(cfg.layer_types[1], "sliding_attention");
        assert_eq!(cfg.layer_types[2], "sliding_attention");
        assert_eq!(cfg.layer_types[3], "full_attention");
    }
}
