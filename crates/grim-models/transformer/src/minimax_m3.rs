//! Compatibility loader for `MiniMaxAI/MiniMax-M3` (HuggingFace `model_type = "minimax_m3"`).

use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_tensor::{ArithType, Device, Tensor};

/// Native mirror of `MiniMaxM3Config` (HuggingFace `minimax_m3`).
#[derive(Debug, Clone)]
pub struct MiniMaxM3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl ModelConfig for MiniMaxM3Config {
    fn name(&self) -> &str {
        "minimax_m3"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MiniMaxM3Config {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        MiniMaxM3Config {
            vocab_size: u("vocab_size"),
            hidden_size: u("hidden_size"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            num_hidden_layers: u("num_hidden_layers"),
            intermediate_size: u("intermediate_size"),
            num_experts: u("num_experts"),
            num_experts_per_tok: u("num_experts_per_tok"),
            rms_norm_eps: f("rms_norm_eps"),
            rope_theta: f("rope_theta"),
            max_position_embeddings: u("max_position_embeddings"),
        }
    }
}

#[allow(dead_code)]
pub const MINIMAX_M3_TENSOR_KEYS: &[&str] = &[
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
    "model.layers.{i}.input_layernorm.weight",
    "model.layers.{i}.post_attention_layernorm.weight",
    "model.layers.{i}.self_attn.q_proj.weight",
    "model.layers.{i}.self_attn.k_proj.weight",
    "model.layers.{i}.self_attn.v_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight",
    "model.layers.{i}.block_sparse_moe.gate.weight",
    "model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight",
    "model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight",
    "model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight",
];

pub struct MiniMaxM3 {
    pub cfg: MiniMaxM3Config,
    pub device: Device,
}

impl MiniMaxM3 {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: MiniMaxM3Config,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        _device: Device,
        _ws: &grim_nn::WeightSource<'_>,
        _cfg: MiniMaxM3Config,
    ) -> Result<Self> {
        Err(Error::Unimplemented("MiniMaxM3 load_tp is not yet implemented".into()))
    }
}

impl Model for MiniMaxM3 {
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

impl CausalLm for MiniMaxM3 {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }
    fn forward(
        &self,
        _session: &mut dyn SessionT,
        _input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        Err(Error::Unimplemented("MiniMaxM3 forward pass".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const MINIMAX_M3_CONFIG: &str = r#"{
        "architectures": ["MiniMaxM3ForCausalLM"],
        "hidden_size": 3072,
        "num_hidden_layers": 36,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "rms_norm_eps": 1e-05,
        "rope_theta": 100000.0,
        "vocab_size": 128000
    }"#;

    #[test]
    fn parses_minimax_m3_config() {
        let v: serde_json::Value = serde_json::from_str(MINIMAX_M3_CONFIG).unwrap();
        let cfg = MiniMaxM3Config::from_hf(&v);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_experts, 32);
        assert_eq!(cfg.name(), "minimax_m3");
    }

    #[test]
    fn dispatches_minimax_m3_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("minimax_m3"),
            ModelArchitecture::MiniMaxM3
        );
    }
}
