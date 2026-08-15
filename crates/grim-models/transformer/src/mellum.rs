//! Compatibility loader for `JetBrains/Mellum2-12B-A2.5B-Thinking`
//! (HuggingFace `model_type = "mellum"`).
//!
//! ## What this model is
//!
//! Mellum2 is a **sparse-MoE** CausalLM with Llama-style attention:
//!
//! * 28 layers, each with sliding-window or full attention and a sparse MoE FFN.
//! * MoE: 64 routed experts, `num_experts_per_tok = 8`, softmax router,
//!   no shared expert, `routed_scaling_factor = 1.0`.
//! * Attention: 32 query heads, 4 KV heads, `head_dim = 128`,
//!   sliding window = 1024, `use_sliding_window = true`.
//! * Yarn RoPE: `rope_theta = 500000.0`, `factor = 16.0`,
//!   `original_max_position_embeddings = 8192`, `beta_fast = 32.0`,
//!   `beta_slow = 1.0`, `attention_factor = 1.2772588722239782`.
//! * `max_position_embeddings = 131072`, `rms_norm_eps = 1e-06`.
//! * Vocabulary: 98304 tokens (SentencePiece).
//!
//! ## Compatibility status
//!
//! Mellum2 shares the Llama-style transformer backbone with Qwen3-MoE and
//! other sparse-MoE models in grim. The `Mellum` struct wraps `Llama` with
//! a per-layer MoE spec (softmax router, 64 experts, 8 per token, no shared
//! expert).
//!
//! Remaining gaps (tracked, not silently fallen back):
//! * Yarn RoPE parameters (`factor`, `beta_fast`, `beta_slow`,
//!   `attention_factor`) are not yet differentiated from default yarn — the
//!   loader passes `rope_theta = 500000.0` and the standard yarn config.
//!   Full Yarn fidelity (scaling factors) is a follow-up once the Yarn
//!   config struct is extended.
//! * Sliding-window attention is noted in config but the Llama loader
//!   currently treats all layers uniformly; windowed layers run the same
//!   attention path.

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::TensorParallelConfig;
use grim_nn::moe::RouterKind;
use grim_tensor::{ArithType, Device, Tensor};

use crate::model::{Llama, LlamaConfig};
use crate::moe_block::MoESpec;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for JetBrains/Mellum2-12B-A2.5B-Thinking.
///
/// Matches the HuggingFace `config.json` fields:
/// `vocab_size=98304, hidden_size=2304, num_attention_heads=32,
/// num_key_value_heads=4, head_dim=128, num_hidden_layers=28,
/// intermediate_size=7168, moe_intermediate_size=896, num_experts=64,
/// num_experts_per_tok=8, rms_norm_eps=1e-06, rope_theta=500000.0,
/// max_position_embeddings=131072, sliding_window=1024`.
#[derive(Debug, Clone)]
pub struct MellumConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub sliding_window: usize,
}

impl ModelConfig for MellumConfig {
    fn name(&self) -> &str {
        "mellum"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MellumConfig {
    /// Build from the raw HuggingFace `config.json` `serde_json::Value`.
    ///
    /// Panics are avoided: every field is read with a `get` + `as_*` + fallback
    /// so a slightly different Mellum variant still parses.
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        // MoE intermediate size may be under "moe_intermediate_size" or
        // derived from "intermediate_size" for non-MoE variants.
        let moe_intermediate_size = value
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| value.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(0) as u64)
            as usize;

        // Sliding window: may be "sliding_window" or "max_window_layers" with
        // a separate window size. Mellum2 uses "sliding_window": 1024.
        let sliding_window = value
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        MellumConfig {
            vocab_size: u("vocab_size"),
            hidden_size: u("hidden_size"),
            num_heads: u("num_attention_heads"),
            num_kv_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            num_layers: u("num_hidden_layers"),
            intermediate_size: u("intermediate_size"),
            moe_intermediate_size,
            num_experts: u("num_experts"),
            num_experts_per_tok: u("num_experts_per_tok"),
            rms_norm_eps: f("rms_norm_eps"),
            rope_theta: f("rope_theta"),
            max_seq_len: u("max_position_embeddings"),
            sliding_window,
        }
    }
}

// ---------------------------------------------------------------------------
// Model — Llama backbone with per-layer MoE spec
// ---------------------------------------------------------------------------

pub struct Mellum {
    pub cfg: MellumConfig,
    pub device: Device,
    pub inner: Llama,
}

impl Mellum {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: MellumConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: MellumConfig,
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

        // Mellum2 uses a softmax router (no correction bias), no shared expert.
        // Every layer routes through the MoE FFN with 64 experts, 8 per token.
        let spec = MoESpec {
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            router_kind: RouterKind::SoftmaxTopK,
            routed_scaling_factor: 1.0,
            has_shared_expert: false,
            moe_intermediate_size: Some(cfg.moe_intermediate_size),
            shared_expert_intermediate_size: None,
            transposed_expert_layout: false,
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

impl Model for Mellum {
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

impl CausalLm for Mellum {
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

// ---------------------------------------------------------------------------
// Architecture dispatch
// ---------------------------------------------------------------------------

impl MellumConfig {
    /// Whether this config represents a MoE model.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Architecture string for registry matching.
    pub fn architecture_name(&self) -> &str {
        "mellum"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const MELLUM2_CONFIG: &str = r#"{
        "architectures": ["MellumForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dtype": "bfloat16",
        "eos_token_id": 28,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2304,
        "initializer_range": 0.02,
        "intermediate_size": 7168,
        "max_position_embeddings": 131072,
        "max_window_layers": 0,
        "model_type": "mellum",
        "moe_intermediate_size": 896,
        "norm_topk_prob": true,
        "num_attention_heads": 32,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "output_router_logits": false,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.2772588722239782
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 500000.0
            }
        },
        "router_aux_loss_coef": 0.001,
        "sliding_window": 1024,
        "tie_word_embeddings": false,
        "use_cache": true,
        "vocab_size": 98304,
        "use_sliding_window": true
    }"#;

    #[test]
    fn parses_mellum2_config() {
        let v: serde_json::Value = serde_json::from_str(MELLUM2_CONFIG).unwrap();
        let cfg = MellumConfig::from_hf(&v);
        assert_eq!(cfg.vocab_size, 98304);
        assert_eq!(cfg.hidden_size, 2304);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.intermediate_size, 7168);
        assert_eq!(cfg.moe_intermediate_size, 896);
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9);
        assert!((cfg.rope_theta - 500000.0).abs() < 1e-6);
        assert_eq!(cfg.max_seq_len, 131072);
        assert_eq!(cfg.sliding_window, 1024);
    }

    #[test]
    fn mellum2_config_is_moe() {
        let v: serde_json::Value = serde_json::from_str(MELLUM2_CONFIG).unwrap();
        let cfg = MellumConfig::from_hf(&v);
        assert!(cfg.is_moe());
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
    }

    #[test]
    fn hf_model_type_dispatches_to_mellum() {
        // The HF `model_type` ("mellum") must resolve to the registered
        // architecture.
        assert_eq!(
            ModelArchitecture::from_str("mellum"),
            ModelArchitecture::Mellum
        );
        assert!(ModelArchitecture::Mellum.is_moe());
    }

    #[test]
    fn mellum_config_name_matches_registry() {
        let v: serde_json::Value = serde_json::from_str(MELLUM2_CONFIG).unwrap();
        let cfg = MellumConfig::from_hf(&v);
        assert_eq!(cfg.name(), "mellum");
        assert_eq!(cfg.architecture_name(), "mellum");
    }
}
