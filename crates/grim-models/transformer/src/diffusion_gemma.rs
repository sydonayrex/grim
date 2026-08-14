//! Compatibility loader for `google/diffusiongemma-26B-A4B-it` (HuggingFace `model_type = "diffusion_gemma"`).

use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_tensor::{ArithType, Device, Tensor};

/// Native mirror of `DiffusionGemmaConfig` (HuggingFace `diffusion_gemma`).
#[derive(Debug, Clone)]
pub struct DiffusionGemmaConfig {
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

impl ModelConfig for DiffusionGemmaConfig {
    fn name(&self) -> &str {
        "diffusion_gemma"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl DiffusionGemmaConfig {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        DiffusionGemmaConfig {
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
pub const DIFFUSION_GEMMA_TENSOR_KEYS: &[&str] = &[
    "model.embed_tokens.weight",
    "model.norm.weight",
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

pub struct DiffusionGemma {
    pub cfg: DiffusionGemmaConfig,
    pub device: Device,
}

impl DiffusionGemma {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DiffusionGemmaConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        device: Device,
        _ws: &grim_nn::WeightSource<'_>,
        cfg: DiffusionGemmaConfig,
    ) -> Result<Self> {
        Ok(DiffusionGemma { cfg, device })
    }
}

impl Model for DiffusionGemma {
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

impl CausalLm for DiffusionGemma {
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
        Err(Error::Unimplemented("DiffusionGemma forward pass".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const DIFFUSION_GEMMA_CONFIG: &str = r#"{
        "architectures": ["DiffusionGemmaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 46,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "intermediate_size": 16384,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "vocab_size": 256000
    }"#;

    #[test]
    fn parses_diffusion_gemma_config() {
        let v: serde_json::Value = serde_json::from_str(DIFFUSION_GEMMA_CONFIG).unwrap();
        let cfg = DiffusionGemmaConfig::from_hf(&v);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 46);
        assert_eq!(cfg.name(), "diffusion_gemma");
    }

    #[test]
    fn dispatches_diffusion_gemma_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("diffusion_gemma"),
            ModelArchitecture::DiffusionGemma
        );
    }
}
