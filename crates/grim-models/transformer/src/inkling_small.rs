//! Compatibility loader for `thinkingmachines/Inkling-Small` (HuggingFace `model_type = "inkling_small"`).

use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_tensor::{ArithType, Device, Tensor};

/// Native mirror of `InklingSmallConfig` (HuggingFace `inkling_small`).
#[derive(Debug, Clone)]
pub struct InklingSmallConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl ModelConfig for InklingSmallConfig {
    fn name(&self) -> &str {
        "inkling_small"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl InklingSmallConfig {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        InklingSmallConfig {
            vocab_size: u("vocab_size"),
            hidden_size: u("hidden_size"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            num_hidden_layers: u("num_hidden_layers"),
            intermediate_size: u("intermediate_size"),
            rms_norm_eps: f("rms_norm_eps"),
            rope_theta: f("rope_theta"),
            max_position_embeddings: u("max_position_embeddings"),
        }
    }
}

#[allow(dead_code)]
pub const INKLING_SMALL_TENSOR_KEYS: &[&str] = &[
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
    "model.layers.{i}.input_layernorm.weight",
    "model.layers.{i}.post_attention_layernorm.weight",
    "model.layers.{i}.self_attn.q_proj.weight",
    "model.layers.{i}.self_attn.k_proj.weight",
    "model.layers.{i}.self_attn.v_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight",
    "model.layers.{i}.mlp.gate_proj.weight",
    "model.layers.{i}.mlp.up_proj.weight",
    "model.layers.{i}.mlp.down_proj.weight",
];

pub struct InklingSmall {
    pub cfg: InklingSmallConfig,
    pub device: Device,
}

impl InklingSmall {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: InklingSmallConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        device: Device,
        _ws: &grim_nn::WeightSource<'_>,
        cfg: InklingSmallConfig,
    ) -> Result<Self> {
        Ok(InklingSmall { cfg, device })
    }
}

impl Model for InklingSmall {
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

impl CausalLm for InklingSmall {
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
        Err(Error::Unimplemented("InklingSmall forward pass".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const INKLING_SMALL_CONFIG: &str = r#"{
        "architectures": ["InklingSmallForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "intermediate_size": 3072,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "vocab_size": 32000
    }"#;

    #[test]
    fn parses_inkling_small_config() {
        let v: serde_json::Value = serde_json::from_str(INKLING_SMALL_CONFIG).unwrap();
        let cfg = InklingSmallConfig::from_hf(&v);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 16);
        assert_eq!(cfg.name(), "inkling_small");
    }

    #[test]
    fn dispatches_inkling_small_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("inkling_small"),
            ModelArchitecture::InklingSmall
        );
    }
}
