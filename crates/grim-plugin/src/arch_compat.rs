//! Architecture compatibility generator and spec parser for HuggingFace `config.json`.
//!
//! §6 of Grim architecture. Ingests raw HuggingFace model `config.json` files (e.g., Ling-2.6-flash,
//! Qwen, LFM2, custom models) and generates a structured `ArchCompatSpec` containing parameter mappings,
//! tensor remapping rules, and architecture capability declarations for dynamic plugin registration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use grim_core::error::{Error, Result};
use grim_core::architecture::{ModelArchitecture, TensorNamingRegistry};

/// Architecture compatibility specification generated from HuggingFace `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchCompatSpec {
    pub name: String,
    pub model_type: String,
    pub base_architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub is_moe: bool,
    pub is_ssm: bool,
    pub expert_count: Option<usize>,
    pub expert_used_count: Option<usize>,
    pub tensor_name_mapping: HashMap<String, String>,
}

/// Raw HuggingFace `config.json` layout for dynamic parsing.
#[derive(Debug, Deserialize)]
struct RawHfConfig {
    #[serde(rename = "architectures")]
    architectures: Option<Vec<String>>,
    #[serde(rename = "model_type")]
    model_type: Option<String>,
    #[serde(rename = "hidden_size")]
    hidden_size: Option<usize>,
    #[serde(rename = "num_hidden_layers")]
    num_hidden_layers: Option<usize>,
    #[serde(rename = "vocab_size")]
    vocab_size: Option<usize>,
    #[serde(rename = "rms_norm_eps")]
    rms_norm_eps: Option<f32>,
    #[serde(rename = "layer_norm_eps")]
    layer_norm_eps: Option<f32>,
    #[serde(rename = "rope_theta")]
    rope_theta: Option<f32>,
    #[serde(rename = "num_attention_heads")]
    num_attention_heads: Option<usize>,
    #[serde(rename = "num_key_value_heads")]
    num_key_value_heads: Option<usize>,
    #[serde(rename = "head_dim")]
    head_dim: Option<usize>,
    #[serde(rename = "intermediate_size")]
    intermediate_size: Option<usize>,
    #[serde(rename = "max_position_embeddings")]
    max_position_embeddings: Option<usize>,
    #[serde(rename = "num_local_experts")]
    num_local_experts: Option<usize>,
    #[serde(rename = "num_experts_per_tok")]
    num_experts_per_tok: Option<usize>,
}

impl ArchCompatSpec {
    /// Parse a HuggingFace `config.json` string and construct an architecture compatibility spec.
    pub fn from_hf_config_json(json_str: &str) -> Result<Self> {
        let raw: RawHfConfig = serde_json::from_str(json_str)
            .map_err(|e| Error::Config(format!("Failed to parse HF config.json: {e}")))?;

        let model_type = raw.model_type
            .or_else(|| raw.architectures.and_then(|a| a.first().cloned()))
            .unwrap_or_else(|| "custom".to_string());

        let model_arch = ModelArchitecture::from_str(&model_type);
        let hidden_size = raw.hidden_size.unwrap_or(4096);
        let num_layers = raw.num_hidden_layers.unwrap_or(32);
        let vocab_size = raw.vocab_size.unwrap_or(32000);
        let num_heads = raw.num_attention_heads.unwrap_or(32);
        let num_kv_heads = raw.num_key_value_heads.unwrap_or(num_heads);
        let head_dim = raw.head_dim.unwrap_or_else(|| if num_heads > 0 { hidden_size / num_heads } else { 128 });
        let intermediate_size = raw.intermediate_size.unwrap_or(hidden_size * 4);
        let rms_norm_eps = raw.rms_norm_eps.or(raw.layer_norm_eps).unwrap_or(1e-5);
        let rope_theta = raw.rope_theta.unwrap_or(10000.0);
        let max_seq_len = raw.max_position_embeddings.unwrap_or(2048);

        let is_moe = model_arch.is_moe() || raw.num_local_experts.is_some();
        let is_ssm = model_arch.is_ssm();

        let tensor_name_mapping = TensorNamingRegistry::remap_hf_to_gguf(model_arch, num_layers);

        Ok(Self {
            name: format!("{}-compat", model_type),
            model_type: model_type.clone(),
            base_architecture: model_arch.as_str().to_string(),
            hidden_size,
            num_layers,
            vocab_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
            max_seq_len,
            is_moe,
            is_ssm,
            expert_count: raw.num_local_experts,
            expert_used_count: raw.num_experts_per_tok,
            tensor_name_mapping,
        })
    }

    /// Serialize the architecture compatibility spec to a formatted JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize ArchCompatSpec to JSON: {e}")))
    }

    /// Serialize the architecture compatibility spec to a formatted TOML string.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize ArchCompatSpec to TOML: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arch_compat_spec_from_hf_json() {
        let sample_json = r#"{
            "architectures": ["LingForCausalLM"],
            "model_type": "ling",
            "hidden_size": 4096,
            "num_hidden_layers": 28,
            "vocab_size": 128000,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "rms_norm_eps": 1e-6
        }"#;

        let spec = ArchCompatSpec::from_hf_config_json(sample_json).unwrap();
        assert_eq!(spec.model_type, "ling");
        assert_eq!(spec.num_layers, 28);
        assert_eq!(spec.hidden_size, 4096);
        assert_eq!(spec.vocab_size, 128000);
        assert_eq!(spec.num_kv_heads, 8);

        let json = spec.to_json().unwrap();
        assert!(json.contains("ling"));
    }
}
