//! Architecture compatibility generator and spec parser for HuggingFace `config.json`.
//!
//! §6 of Grim architecture. Ingests raw HuggingFace model `config.json` files (e.g., Ling-2.6-flash,
//! Qwen, LFM2, custom models) and generates a structured `ArchCompatSpec` containing parameter mappings,
//! tensor remapping rules, and architecture capability declarations for dynamic plugin registration.

use grim_core::architecture::{ModelArchitecture, TensorNamingRegistry};
use grim_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vision encoder sub-specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisionEncoderSpec {
    pub vision_encoder_type: String,
    pub decoder_dmodel: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub n_channels: usize,
    pub n_layers: usize,
    pub use_vision_norm: bool,
}

/// Audio encoder sub-specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioEncoderSpec {
    pub decoder_dmodel: usize,
    pub n_mel_bins: usize,
    pub mel_vocab_size: usize,
    pub bias: bool,
    pub use_audio_norm: bool,
    pub audio_mode: String,
}

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
    pub is_multimodal: bool,
    pub vision_spec: Option<VisionEncoderSpec>,
    pub audio_spec: Option<AudioEncoderSpec>,
    pub expert_count: Option<usize>,
    pub expert_used_count: Option<usize>,
    /// Scaling applied to routed-expert output before adding the shared expert.
    /// `None` means "not specified by this source"; callers fall back to 1.0.
    pub routed_scaling_factor: Option<f32>,
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
    #[serde(rename = "model_max_length")]
    model_max_length: Option<usize>,
    #[serde(rename = "num_local_experts")]
    num_local_experts: Option<usize>,
    #[serde(rename = "n_routed_experts")]
    n_routed_experts: Option<usize>,
    #[serde(rename = "num_experts_per_tok")]
    num_experts_per_tok: Option<usize>,
    // MoE routed-expert output scaling. HF configs use either
    // `routed_scaling_factor` (Qwen3-MoE / DeepSeek-V2) or, in some
    // SmolLM2-derived checkpoints, `expert_gating_func`.
    #[serde(rename = "routed_scaling_factor")]
    routed_scaling_factor: Option<f32>,
    #[serde(rename = "expert_gating_func")]
    expert_gating_func: Option<f32>,
    #[serde(rename = "text_config")]
    text_config: Option<Box<RawHfConfig>>,
    #[serde(rename = "vision_config")]
    vision_config: Option<VisionEncoderSpec>,
    #[serde(rename = "audio_config")]
    audio_config: Option<AudioEncoderSpec>,
}

impl ArchCompatSpec {
    /// Parse a HuggingFace `config.json` string and construct an architecture compatibility spec.
    pub fn from_hf_config_json(json_str: &str) -> Result<Self> {
        let raw: RawHfConfig = serde_json::from_str(json_str)
            .map_err(|e| Error::Config(format!("Failed to parse HF config.json: {e}")))?;

        let text = raw.text_config.as_deref();

        let model_type = raw
            .model_type
            .or_else(|| raw.architectures.as_ref().and_then(|a| a.first().cloned()))
            .or_else(|| text.and_then(|t| t.model_type.clone()))
            .unwrap_or_else(|| "custom".to_string());

        let model_arch = ModelArchitecture::from_str(&model_type);
        let hidden_size = raw
            .hidden_size
            .or_else(|| text.and_then(|t| t.hidden_size))
            .unwrap_or(4096);
        let num_layers = raw
            .num_hidden_layers
            .or_else(|| text.and_then(|t| t.num_hidden_layers))
            .unwrap_or(32);
        let vocab_size = raw
            .vocab_size
            .or_else(|| text.and_then(|t| t.vocab_size))
            .unwrap_or(32000);
        let num_heads = raw
            .num_attention_heads
            .or_else(|| text.and_then(|t| t.num_attention_heads))
            .unwrap_or(32);
        let num_kv_heads = raw
            .num_key_value_heads
            .or_else(|| text.and_then(|t| t.num_key_value_heads))
            .unwrap_or(num_heads);
        let head_dim = raw
            .head_dim
            .or_else(|| text.and_then(|t| t.head_dim))
            .unwrap_or_else(|| {
                if num_heads > 0 {
                    hidden_size / num_heads
                } else {
                    128
                }
            });
        let intermediate_size = raw
            .intermediate_size
            .or_else(|| text.and_then(|t| t.intermediate_size))
            .unwrap_or(hidden_size * 4);
        let rms_norm_eps = raw
            .rms_norm_eps
            .or(raw.layer_norm_eps)
            .or_else(|| text.and_then(|t| t.rms_norm_eps.or(t.layer_norm_eps)))
            .unwrap_or(1e-5);
        let rope_theta = raw
            .rope_theta
            .or_else(|| text.and_then(|t| t.rope_theta))
            .unwrap_or(10000.0);
        let max_seq_len = raw
            .max_position_embeddings
            .or(raw.model_max_length)
            .or_else(|| text.and_then(|t| t.max_position_embeddings.or(t.model_max_length)))
            .unwrap_or(2048);

        let expert_count = raw
            .num_local_experts
            .or(raw.n_routed_experts)
            .or_else(|| text.and_then(|t| t.num_local_experts.or(t.n_routed_experts)));
        let expert_used_count = raw
            .num_experts_per_tok
            .or_else(|| text.and_then(|t| t.num_experts_per_tok));
        let routed_scaling_factor = raw
            .routed_scaling_factor
            .or(raw.expert_gating_func)
            .or_else(|| {
                text.and_then(|t| t.routed_scaling_factor.or(t.expert_gating_func))
            });

        let is_moe = model_arch.is_moe() || expert_count.is_some();
        let is_ssm = model_arch.is_ssm();
        let vision_spec = raw.vision_config;
        let audio_spec = raw.audio_config;
        let is_multimodal = vision_spec.is_some() || audio_spec.is_some();

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
            is_multimodal,
            vision_spec,
            audio_spec,
            expert_count,
            expert_used_count,
            routed_scaling_factor,
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

    /// Translate a tensor name using the bidirectional mapping table generated for this spec.
    ///
    /// Checks forward mapping (`hf -> gguf`) and reverse mapping (`gguf -> hf`). If a match
    /// is found, returns the translated tensor name; otherwise returns the input name unchanged.
    ///
    /// The reverse lookup is deterministic: when multiple HF names map to the same GGUF name,
    /// the HF standard naming (with `model.` prefix) is preferred over internal loader
    /// canonical names.
    pub fn remap_tensor_name(&self, name: &str) -> String {
        if let Some(mapped) = self.tensor_name_mapping.get(name) {
            return mapped.clone();
        }
        // Reverse lookup: prefer HF standard naming (with `model.` prefix)
        // over internal loader canonical names for deterministic results.
        if let Some(hf_name) = self
            .tensor_name_mapping
            .iter()
            .find(|(hf, gguf)| gguf == &name && hf.starts_with("model."))
            .map(|(hf, _)| hf)
        {
            return hf_name.clone();
        }
        self.tensor_name_mapping
            .iter()
            .find(|(_, gguf)| gguf == &name)
            .map(|(hf, _)| hf.clone())
            .unwrap_or_else(|| name.to_string())
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

    #[test]
    fn test_inkling_config_json_ingestion() {
        let inkling_json =
            std::fs::read_to_string("/D/rex/projects/grim/models/inkling/config.json")
                .or_else(|_| std::fs::read_to_string("models/inkling/config.json"))
                .expect("read inkling config.json");

        let spec = ArchCompatSpec::from_hf_config_json(&inkling_json).unwrap();
        assert_eq!(spec.model_type, "inkling_mm_model");
        assert_eq!(spec.num_layers, 42);
        assert_eq!(spec.hidden_size, 4096);
        assert_eq!(spec.vocab_size, 201024);
        assert_eq!(spec.num_heads, 32);
        assert_eq!(spec.num_kv_heads, 8);
        assert_eq!(spec.head_dim, 128);
        assert_eq!(spec.max_seq_len, 1048576);
        assert!(spec.is_moe);
        assert!(spec.is_multimodal);
        assert!(spec.vision_spec.is_some());
        assert_eq!(spec.vision_spec.as_ref().unwrap().patch_size, 40);
        assert!(spec.audio_spec.is_some());
        assert_eq!(spec.audio_spec.as_ref().unwrap().n_mel_bins, 80);
        assert_eq!(spec.expert_count, Some(256));
        assert_eq!(spec.expert_used_count, Some(6));

        let spec_json = spec.to_json().unwrap();
        assert!(spec_json.contains("inkling_mm_model"));

        let spec_toml = spec.to_toml().unwrap();
        assert!(spec_toml.contains("inkling_mm_model"));
    }
}
