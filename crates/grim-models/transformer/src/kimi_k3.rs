//! Compatibility loader for `moonshotai/Kimi-K3` (HuggingFace `model_type = "kimi_k3"`).

use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, Model, ModelConfig, ModalityHint};
use grim_core::session::SessionT;
use grim_tensor::{ArithType, Device, Tensor};

/// Native mirror of `KimiK3Config` (HuggingFace `kimi_k3`).
#[derive(Debug, Clone)]
pub struct KimiK3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
}

impl ModelConfig for KimiK3Config {
    fn name(&self) -> &str {
        "kimi_k3"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl KimiK3Config {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        KimiK3Config {
            vocab_size: u("vocab_size"),
            hidden_size: u("hidden_size"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            num_hidden_layers: u("num_hidden_layers"),
            q_lora_rank: u("q_lora_rank"),
            kv_lora_rank: u("kv_lora_rank"),
            qk_nope_head_dim: u("qk_nope_head_dim"),
            qk_rope_head_dim: u("qk_rope_head_dim"),
            v_head_dim: u("v_head_dim"),
            num_experts: u("num_experts"),
            num_experts_per_tok: u("num_experts_per_tok"),
            routed_scaling_factor: f("routed_scaling_factor"),
            rms_norm_eps: f("rms_norm_eps"),
        }
    }
}

#[allow(dead_code)]
pub const KIMI_K3_TENSOR_KEYS: &[&str] = &[
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
    "model.layers.{i}.input_layernorm.weight",
    "model.layers.{i}.post_attention_layernorm.weight",
    "model.layers.{i}.self_attn.q_a_proj.weight",
    "model.layers.{i}.self_attn.q_b_proj.weight",
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.{i}.self_attn.kv_b_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight",
    "model.layers.{i}.moe.gate.weight",
    "model.layers.{i}.moe.experts.{e}.w1.weight",
    "model.layers.{i}.moe.experts.{e}.w2.weight",
    "model.layers.{i}.moe.experts.{e}.w3.weight",
];

pub struct KimiK3 {
    pub cfg: KimiK3Config,
    pub device: Device,
}

impl KimiK3 {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: KimiK3Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        device: Device,
        _ws: &grim_nn::WeightSource<'_>,
        cfg: KimiK3Config,
    ) -> Result<Self> {
        Ok(KimiK3 { cfg, device })
    }
}

impl Model for KimiK3 {
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

impl CausalLm for KimiK3 {
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
        Err(Error::Unimplemented("KimiK3 forward pass".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const KIMI_K3_CONFIG: &str = r#"{
        "architectures": ["KimiK3ForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "q_lora_rank": 256,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 6,
        "routed_scaling_factor": 2.0,
        "rms_norm_eps": 1e-06,
        "vocab_size": 163840
    }"#;

    #[test]
    fn parses_kimi_k3_config() {
        let v: serde_json::Value = serde_json::from_str(KIMI_K3_CONFIG).unwrap();
        let cfg = KimiK3Config::from_hf(&v);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.name(), "kimi_k3");
    }

    #[test]
    fn dispatches_kimi_k3_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("kimi_k3"),
            ModelArchitecture::KimiK3
        );
    }
}
