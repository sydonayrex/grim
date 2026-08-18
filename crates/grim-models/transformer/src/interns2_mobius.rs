//! Compatibility loader for `internlm/Intern-S2-Mobius` (HuggingFace `model_type = "interns2_mobius"`).

use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_tensor::{ArithType, Device, Tensor};

/// Native mirror of `InternS2MobiusConfig` (HuggingFace `interns2_mobius`).
#[derive(Debug, Clone)]
pub struct InternS2MobiusConfig {
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

impl ModelConfig for InternS2MobiusConfig {
    fn name(&self) -> &str {
        "interns2_mobius"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl InternS2MobiusConfig {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        InternS2MobiusConfig {
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
pub const INTERN_S2_MOBIUS_TENSOR_KEYS: &[&str] = &[
    "model.tok_embeddings.weight",
    "model.norm.weight",
    "output.weight",
    "model.layers.{i}.attention_norm.weight",
    "model.layers.{i}.ffn_norm.weight",
    "model.layers.{i}.attention.wqkv.weight",
    "model.layers.{i}.attention.wo.weight",
    "model.layers.{i}.feed_forward.w1.weight",
    "model.layers.{i}.feed_forward.w2.weight",
    "model.layers.{i}.feed_forward.w3.weight",
];

pub struct InternS2Mobius {
    pub cfg: InternS2MobiusConfig,
    pub device: Device,
}

impl InternS2Mobius {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: InternS2MobiusConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        _device: Device,
        _ws: &grim_nn::WeightSource<'_>,
        _cfg: InternS2MobiusConfig,
    ) -> Result<Self> {
        Err(Error::Unimplemented("InternS2Mobius load_tp is not yet implemented".into()))
    }
}

impl Model for InternS2Mobius {
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

impl CausalLm for InternS2Mobius {
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
        Err(Error::Unimplemented("InternS2Mobius forward pass".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const INTERN_S2_CONFIG: &str = r#"{
        "architectures": ["InternS2MobiusForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "vocab_size": 92544
    }"#;

    #[test]
    fn parses_intern_s2_mobius_config() {
        let v: serde_json::Value = serde_json::from_str(INTERN_S2_CONFIG).unwrap();
        let cfg = InternS2MobiusConfig::from_hf(&v);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.name(), "interns2_mobius");
    }

    #[test]
    fn dispatches_intern_s2_mobius_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("interns2_mobius"),
            ModelArchitecture::InternS2Mobius
        );
    }
}
