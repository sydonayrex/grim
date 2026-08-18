//! Compatibility loader for `zai-org/GLM-5.2` (HuggingFace `model_type = "glm5_2"`).

use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_tensor::{ArithType, Device, Tensor};

/// Native mirror of `Glm52Config` (HuggingFace `glm5_2`).
#[derive(Debug, Clone)]
pub struct Glm52Config {
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

impl ModelConfig for Glm52Config {
    fn name(&self) -> &str {
        "glm5_2"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Glm52Config {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        Glm52Config {
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
pub const GLM5_2_TENSOR_KEYS: &[&str] = &[
    "transformer.embedding.word_embeddings.weight",
    "transformer.output_layer.weight",
    "transformer.encoder.final_layernorm.weight",
    "transformer.encoder.layers.{i}.input_layernorm.weight",
    "transformer.encoder.layers.{i}.post_attention_layernorm.weight",
    "transformer.encoder.layers.{i}.self_attention.query_key_value.weight",
    "transformer.encoder.layers.{i}.self_attention.dense.weight",
    "transformer.encoder.layers.{i}.mlp.gate.weight",
    "transformer.encoder.layers.{i}.mlp.experts.{e}.dense_h_to_4h.weight",
    "transformer.encoder.layers.{i}.mlp.experts.{e}.dense_4h_to_h.weight",
];

pub struct Glm52 {
    pub cfg: Glm52Config,
    pub device: Device,
}

impl Glm52 {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: Glm52Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        _device: Device,
        _ws: &grim_nn::WeightSource<'_>,
        _cfg: Glm52Config,
    ) -> Result<Self> {
        Err(Error::Unimplemented("Glm52 load_tp is not yet implemented".into()))
    }
}

impl Model for Glm52 {
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

impl CausalLm for Glm52 {
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
        Err(Error::Unimplemented("Glm52 forward pass".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const GLM5_2_CONFIG: &str = r#"{
        "architectures": ["Glm52ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 13824,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "vocab_size": 151552
    }"#;

    #[test]
    fn parses_glm5_2_config() {
        let v: serde_json::Value = serde_json::from_str(GLM5_2_CONFIG).unwrap();
        let cfg = Glm52Config::from_hf(&v);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.name(), "glm5_2");
    }

    #[test]
    fn dispatches_glm5_2_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("glm5_2"),
            ModelArchitecture::Glm52
        );
    }
}
