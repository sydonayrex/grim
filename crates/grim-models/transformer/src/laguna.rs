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
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Scaling applied to the routed expert output before adding the shared
    /// expert contribution (`routed_scaling_factor`).
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
}

impl Default for LagunaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 3072,
            num_heads: 48,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 48,
            intermediate_size: 12288,
            moe_intermediate_size: 1024,
            shared_expert_intermediate_size: 1024,
            num_experts: 256,
            num_experts_per_tok: 8,
            routed_scaling_factor: 2.5,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 4096,
            mlp_only_layers: vec![0],
            layer_types: (0..48)
                .map(|i| {
                    if i % 4 == 0 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect(),
            sliding_window: 512,
            num_attention_heads_per_layer: (0..48)
                .map(|i| if i % 4 == 0 { 48 } else { 72 })
                .collect(),
            full_rope_theta: 500000.0,
            sliding_rope_theta: 10000.0,
            full_partial_rotary_factor: 0.5,
            sliding_partial_rotary_factor: 1.0,
            gating: "per-head".to_string(),
        }
    }
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

        let router_kind = if cfg.gating == "per-head" {
            RouterKind::SigmoidTopKPerHead
        } else {
            RouterKind::SigmoidTopKWithBias
        };

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

                let yarn = if !is_sliding && theta > 100000.0 {
                    Some(grim_tensor::YaRNParams {
                        factor: 1.0,
                        original_max_pos: cfg.max_seq_len,
                        beta_fast: 32.0,
                        beta_slow: 1.0,
                        attention_factor: 1.0,
                    })
                } else {
                    None
                };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laguna_s_2_1_layer_attention_specs() {
        let cfg = LagunaConfig::default();
        let num_layers = cfg.num_layers;
        assert_eq!(num_layers, 48);

        let specs: Vec<crate::block::LayerAttentionSpec> = (0..num_layers)
            .map(|i| {
                let layer_type = cfg.layer_types.get(i).map(|s| s.as_str()).unwrap_or(if i % 4 == 0 { "full_attention" } else { "sliding_attention" });
                let is_sliding = layer_type == "sliding_attention";

                let num_heads = cfg.num_attention_heads_per_layer.get(i).copied().unwrap_or(if is_sliding { 72 } else { 48 });

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

                let yarn = if !is_sliding && theta > 100000.0 {
                    Some(grim_tensor::YaRNParams {
                        factor: 1.0,
                        original_max_pos: cfg.max_seq_len,
                        beta_fast: 32.0,
                        beta_slow: 1.0,
                        attention_factor: 1.0,
                    })
                } else {
                    None
                };

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
                    attn_type: if is_sliding { crate::block::AttentionType::Sliding } else { crate::block::AttentionType::Full },

                    num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    rope,
                    sliding_window,
                    has_attn_gate,
                }
            })
            .collect();

        // Layer 0: Full attention
        assert_eq!(specs[0].num_heads, 48);
        assert_eq!(specs[0].rope.base, 500000.0);
        assert_eq!(specs[0].rope.rotary_dim, 64); // 0.5 * 128
        assert!(specs[0].rope.yarn.is_some());
        assert_eq!(specs[0].sliding_window, None);
        assert!(specs[0].has_attn_gate);

        // Layer 1: Sliding attention
        assert_eq!(specs[1].num_heads, 72);
        assert_eq!(specs[1].rope.base, 10000.0);
        assert_eq!(specs[1].rope.rotary_dim, 128); // 1.0 * 128
        assert!(specs[1].rope.yarn.is_none());
        assert_eq!(specs[1].sliding_window, Some(512));
        assert!(specs[1].has_attn_gate);
    }
}

