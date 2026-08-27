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
///
/// Lenient deserialization: all fields have defaults so that any HF vision_config
/// shape (e.g. Qwen3.8-27B's `qwen3_5`-type vision_config, older
/// `vision_encoder`-type configs, or partial configs) parses without error.
/// Callers should check `vision_spec.is_some()` to detect the presence of a
/// vision config, then read only the fields they need — missing fields will have
/// their default values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisionEncoderSpec {
    #[serde(default)]
    pub vision_encoder_type: String,
    #[serde(default = "VisionEncoderSpec::default_decoder_dmodel")]
    pub decoder_dmodel: usize,
    #[serde(default = "VisionEncoderSpec::default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "VisionEncoderSpec::default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "VisionEncoderSpec::default_n_channels")]
    pub n_channels: usize,
    #[serde(default = "VisionEncoderSpec::default_n_layers")]
    pub n_layers: usize,
    #[serde(default)]
    pub use_vision_norm: bool,
}

impl VisionEncoderSpec {
    fn default_decoder_dmodel() -> usize {
        2048
    }
    fn default_patch_size() -> usize {
        16
    }
    fn default_temporal_patch_size() -> usize {
        16
    }
    fn default_n_channels() -> usize {
        3
    }
    fn default_n_layers() -> usize {
        12
    }
}

/// Audio encoder sub-specification.
///
/// Lenient deserialization: all fields have defaults so that any HF audio_config
/// shape parses without error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioEncoderSpec {
    #[serde(default = "AudioEncoderSpec::default_decoder_dmodel")]
    pub decoder_dmodel: usize,
    #[serde(default = "AudioEncoderSpec::default_n_mel_bins")]
    pub n_mel_bins: usize,
    #[serde(default = "AudioEncoderSpec::default_mel_vocab_size")]
    pub mel_vocab_size: usize,
    #[serde(default)]
    pub bias: bool,
    #[serde(default)]
    pub use_audio_norm: bool,
    #[serde(default)]
    pub audio_mode: String,
}

impl AudioEncoderSpec {
    fn default_decoder_dmodel() -> usize {
        2048
    }
    fn default_n_mel_bins() -> usize {
        80
    }
    fn default_mel_vocab_size() -> usize {
        3000
    }
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
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct RawHfConfig {
    #[serde(rename = "architectures")]
    architectures: Option<Vec<String>>,
    #[serde(rename = "model_type")]
    model_type: Option<String>,
    #[serde(rename = "hidden_size", alias = "d_model", alias = "dim")]
    hidden_size: Option<usize>,
    #[serde(
        rename = "num_hidden_layers",
        alias = "num_layers",
        alias = "n_layer",
        alias = "n_layers",
        alias = "num_blocks"
    )]
    num_hidden_layers: Option<usize>,
    #[serde(rename = "vocab_size")]
    vocab_size: Option<usize>,
    #[serde(rename = "rms_norm_eps")]
    rms_norm_eps: Option<f32>,
    #[serde(rename = "layer_norm_eps")]
    layer_norm_eps: Option<f32>,
    #[serde(rename = "rope_theta")]
    rope_theta: Option<f32>,
    #[serde(
        rename = "num_attention_heads",
        alias = "num_heads",
        alias = "n_head",
        alias = "n_heads"
    )]
    num_attention_heads: Option<usize>,
    #[serde(
        rename = "num_key_value_heads",
        alias = "num_kv_heads",
        alias = "n_kv_head"
    )]
    num_key_value_heads: Option<usize>,
    #[serde(rename = "head_dim")]
    head_dim: Option<usize>,
    #[serde(rename = "intermediate_size", alias = "ffn_dim")]
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
    // SmolLM2-derived checkpoints, `expert_gating_func` (a string like "softmax"
    // or "silu", NOT a float).
    #[serde(rename = "routed_scaling_factor")]
    routed_scaling_factor: Option<f32>,
    #[serde(rename = "expert_gating_func")]
    expert_gating_func: Option<String>,
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
                    hidden_size.checked_div(num_heads).unwrap_or(hidden_size)
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
            .or_else(|| text.and_then(|t| t.routed_scaling_factor));

        let is_moe = model_arch.is_moe() || expert_count.is_some();
        let is_ssm = model_arch.is_ssm();
        let vision_spec = raw.vision_config;
        let audio_spec = raw.audio_config;
        let is_multimodal = vision_spec.is_some() || audio_spec.is_some();

        let tensor_name_mapping = TensorNamingRegistry::remap_hf_to_gguf(model_arch, num_layers);

        // When the model_type doesn't correspond to a native ModelArchitecture
        // variant, report the base_architecture as "dynamic:{model_type}" rather
        // than "unknown" — the diagnostic in model_loader.rs's fallback arm will
        // print this, giving operators a readable hint about what was resolved
        // rather than a generic "unknown".
        let base_architecture = if model_arch == ModelArchitecture::Unknown {
            format!("dynamic:{model_type}")
        } else {
            model_arch.as_str().to_string()
        };

        Ok(Self {
            name: format!("{}-compat", model_type),
            model_type: model_type.clone(),
            base_architecture,
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

    /// Fetch a HuggingFace repo's `config.json` via the Hub `resolve/main/` endpoint
    /// and build the spec.
    ///
    /// Hits `GET https://huggingface.co/{org}/{repo}/resolve/main/config.json` with a
    /// `User-Agent: hf-cli/0.1` header, which is required by the HF CDN to return the
    /// full config.json (without it, the response may be a truncated/LFS-pointer subset).
    /// Delegates to `from_hf_config_json` for parsing.
    ///
    /// # Failure modes
    ///
    /// - Network / HTTP errors → `Error::Config`.
    /// - Non-200 response (404, etc.) → `Error::Config` with the status code.
    /// - Response that doesn't parse as JSON → `Error::Config`.
    /// - Parsed config missing required fields (model_type, num_hidden_layers, hidden_size)
    ///   → `Error::Config` via `validate_required_fields`. Unlike `from_hf_config_json`
    ///   alone (which silently defaults), the network-fetch path rejects incomplete configs
    ///   because a real HF repo's config.json should always have these fields.
    ///
    /// # Why `resolve/main/` and not `/api/models/`?
    ///
    /// The `/api/models/{org}/{repo}` endpoint embeds a *deserialised* `config` object,
    /// but for Qwen3.8-27B that embedded object is missing key fields (`num_hidden_layers`,
    /// `hidden_size`, `vocab_size`, etc. — only `model_type` and `chat_template_jinja`
    /// are present). The raw `resolve/main/config.json` file has the complete nested
    /// structure (including `text_config` sub-object with all the real parameters).
    /// The raw file is the authoritative source and is what `from_hf_config_json` was
    /// designed to parse.
    pub async fn from_hf_model_id(org_repo: &str) -> Result<Self> {
        let config_url = format!("https://huggingface.co/{org_repo}/resolve/main/config.json");
        let client = reqwest::Client::builder()
            .user_agent("hf-cli/0.1")
            .build()
            .map_err(|e| Error::Config(format!("build HTTP client: {e}")))?;

        let resp = client.get(&config_url).send().await.map_err(|e| {
            Error::Config(format!(
                "Failed to fetch config.json from {config_url}: {e}"
            ))
        })?;

        if !resp.status().is_success() {
            return Err(Error::Config(format!(
                "HF returned {} when fetching config.json from {config_url}",
                resp.status()
            )));
        }

        let config_json = resp.text().await.map_err(|e| {
            Error::Config(format!(
                "Failed to read config.json body from {config_url}: {e}"
            ))
        })?;

        let spec = Self::from_hf_config_json(&config_json)?;
        validate_required_fields(&spec)?;
        Ok(spec)
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

/// Validate that a spec parsed from a *real* HF repo's config.json has the
/// required fields. `from_hf_config_json` silently defaults missing fields
/// (e.g. `model_type` → `"custom"`, `hidden_size` → 4096), so a network
/// fetch that returns a genuinely incomplete config must be rejected here
/// rather than silently installed with wrong defaults.
///
/// Rejects:
/// - `model_type` that is empty OR the `"custom"` sentinel (meaning
///   `from_hf_config_json` found no `model_type` in the config).
/// - `num_layers == 0` (a config that specified 0 layers, or a malformed
///   config where the field parsed as 0).
/// - `hidden_size == 0` (same logic).
fn validate_required_fields(spec: &ArchCompatSpec) -> Result<()> {
    if spec.model_type.is_empty() || spec.model_type == "custom" {
        return Err(Error::Config(
            "model_type is required but was empty or missing from config.json".into(),
        ));
    }
    if spec.num_layers == 0 {
        return Err(Error::Config("num_hidden_layers must be > 0".into()));
    }
    if spec.hidden_size == 0 {
        return Err(Error::Config("hidden_size must be > 0".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Red target: from_hf_model_id fetches Qwen3.8-27B's config and returns
    /// a spec with model_type == "qwen3_5" and the correct num_layers.
    #[test]
    fn from_hf_config_json_parses_qwen38_nested_config() {
        let config_json = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/qwen38_config.json"),
        )
        .expect("qwen38_config.json fixture must exist");

        let spec = ArchCompatSpec::from_hf_config_json(&config_json)
            .expect("qwen38_config.json must parse");

        validate_required_fields(&spec).expect("qwen38 config must have required fields");

        assert_eq!(spec.model_type, "qwen3_5");
        assert_eq!(spec.base_architecture, "qwen35");
        assert!(!spec.is_moe, "Qwen3.8-27B is dense, not MoE");
        assert!(!spec.is_ssm);
        // Qwen3.8-27B text_config.num_hidden_layers == 64
        assert_eq!(
            spec.num_layers, 64,
            "num_hidden_layers from text_config must be 64, got {}",
            spec.num_layers
        );
        assert_eq!(
            spec.hidden_size, 5120,
            "hidden_size from text_config must be 5120"
        );
        assert_eq!(
            spec.vocab_size, 248320,
            "vocab_size from text_config must be 248320"
        );
        assert_eq!(spec.num_heads, 24);
        assert_eq!(spec.num_kv_heads, 4);
        assert_eq!(spec.head_dim, 256);
        assert_eq!(spec.intermediate_size, 17408);
        assert_eq!(spec.rms_norm_eps, 1e-6);
        assert_eq!(spec.max_seq_len, 262144);
        // NOTE: ArchCompatSpec has no partial_rotary_factor field today.
        // Qwen3.8-27B's config.json specifies partial_rotary_factor: 0.25 in
        // text_config, but from_hf_config_json does NOT read it (it's not in
        // RawHfConfig). This is a known gap — see the design question doc and
        // the model_loader.rs GGUF-load path which reads
        // config.partial_rotary_factor separately. For the plugin-generation path
        // this means a Qwen3.8-27B .grimplugin would ship with the default
        // partial_rotary_factor (1.0, set in Qwen35/Qwen3 wrappers) unless the
        // loader path is also updated. Flagged for follow-up; not blocking the
        // plugin-generation work item.
        assert!(
            spec.rope_theta > 0.0,
            "rope_theta must be set (default 10000.0 if absent)"
        );
    }

    /// Green target: a non-existent org/repo returns `Err`, not a panic or a
    /// defaulted spec. The error should surface the failed fetch, not silently
    /// produce garbage.
    ///
    /// NOTE: requires network access to huggingface.co. Ignored by default —
    /// run with `--ignored` to exercise.
    #[tokio::test]
    #[ignore]
    async fn from_hf_model_id_rejects_nonexistent_repo() {
        let result =
            ArchCompatSpec::from_hf_model_id("nonexistent-org-12345/nonexistent-repo-67890").await;

        assert!(result.is_err(), "non-existent repo must return Err");
        // The error should not be a successfully-parsed spec with defaulted fields.
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("nonexistent") || err_msg.contains("404") || err_msg.contains("API"),
            "error should mention the failed fetch, got: {err_msg}"
        );
    }

    /// Green target: a config missing `model_type` is rejected by
    /// `validate_required_fields`, not silently defaulted.
    #[test]
    fn validate_required_fields_rejects_empty_model_type() {
        let spec = ArchCompatSpec::from_hf_config_json(
            r#"{"hidden_size": 4096, "num_hidden_layers": 32}"#,
        )
        .unwrap();

        assert!(
            validate_required_fields(&spec).is_err(),
            "spec with empty model_type must be rejected"
        );
    }

    /// Green target: a config missing `num_hidden_layers` (num_layers == 0
    /// after defaulting? No — from_hf_config_json defaults num_layers to 32 when
    /// missing, so this test must construct a spec where num_layers is
    /// *actually* 0, which requires a config with `"num_hidden_layers": 0`).
    ///
    /// Note: `from_hf_config_json` will parse `"num_hidden_layers": 0` as 0
    /// (it's `Option<usize>` deserialized from JSON, and 0 is a valid usize).
    /// The `.unwrap_or(32)` only fires when the field is *absent*, not when it's
    /// present and zero. So this test is valid: a config with
    /// `"num_hidden_layers": 0` produces num_layers == 0, and validation must
    /// reject it.
    #[test]
    fn validate_required_fields_rejects_zero_num_layers() {
        let spec = ArchCompatSpec::from_hf_config_json(
            r#"{"model_type": "test", "hidden_size": 4096, "num_hidden_layers": 0}"#,
        )
        .unwrap();

        assert_eq!(spec.num_layers, 0);
        assert!(
            validate_required_fields(&spec).is_err(),
            "spec with num_layers == 0 must be rejected"
        );
    }

    /// Green target: a config missing `hidden_size` (with hidden_size == 0 after
    /// the `.unwrap_or(4096)` default — wait, hidden_size defaults to 4096 when
    /// absent, so this test needs `"hidden_size": 0` to get hidden_size == 0).
    ///
    /// Same logic as num_layers: `.unwrap_or(4096)` fires only when the field is
    /// absent, not when it's present and zero.
    #[test]
    fn validate_required_fields_rejects_zero_hidden_size() {
        let spec = ArchCompatSpec::from_hf_config_json(
            r#"{"model_type": "test", "num_hidden_layers": 32, "hidden_size": 0}"#,
        )
        .unwrap();

        assert_eq!(spec.hidden_size, 0);
        assert!(
            validate_required_fields(&spec).is_err(),
            "spec with hidden_size == 0 must be rejected"
        );
    }

    /// Existing test — must keep passing through the refactor.
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

    /// Integration test: pull Inkling-Small's config.json from HF via
    /// `from_hf_model_id` and verify the spec matches the known values.
    ///
    /// NOTE: requires network access to huggingface.co. Ignored by default —
    /// run with `--ignored` to exercise.
    ///
    /// This replaces the local-file `test_inkling_config_json_ingestion` test
    /// with a real HF API pull, exercising the full `resolve/main/config.json`
    /// → `from_hf_config_json` → spec pipeline end-to-end.
    ///
    /// Known-good values from Inkling-Small's config.json (verified against the
    /// repo's raw config.json):
    ///   - text_config.num_hidden_layers: 42
    ///   - text_config.hidden_size: 4096
    ///   - text_config.vocab_size: 201024
    ///   - text_config.num_attention_heads: 32
    ///   - text_config.num_key_value_heads: 8
    ///   - text_config.head_dim: 128
    ///   - model_max_length: 1048576
    ///   - text_config.n_routed_experts: 256 (expert_count)
    ///   - text_config.num_experts_per_tok: 6 (expert_used_count)
    ///   - vision_config.patch_size: 40
    ///   - audio_config.n_mel_bins: 80
    #[tokio::test]
    #[ignore]
    async fn from_hf_model_id_pulls_inkling_small_from_hf() {
        let spec = ArchCompatSpec::from_hf_model_id("thinkingmachines/Inkling-Small")
            .await
            .expect("Inkling-Small should be fetchable from HF");

        // Validate required fields (from_hf_model_id does this internally, but
        // re-check as a defense-in-depth gate).
        validate_required_fields(&spec).expect("Inkling-Small config must have required fields");

        assert_eq!(spec.model_type, "inkling_mm_model");
        // base_architecture for an unrecognized model_type is now "dynamic:{model_type}"
        // (set in from_hf_config_json when model_arch == Unknown), giving operators
        // a readable diagnostic instead of a generic "unknown".
        assert_eq!(
            spec.base_architecture, "dynamic:inkling_mm_model",
            "unrecognized model_type should get dynamic:inkling_mm_model base_architecture"
        );
        assert_eq!(
            spec.num_layers, 42,
            "text_config.num_hidden_layers must be 42"
        );
        assert_eq!(
            spec.hidden_size, 4096,
            "text_config.hidden_size must be 4096"
        );
        assert_eq!(
            spec.vocab_size, 201024,
            "text_config.vocab_size must be 201024"
        );
        assert_eq!(spec.num_heads, 32);
        assert_eq!(spec.num_kv_heads, 8);
        assert_eq!(spec.head_dim, 128);
        assert_eq!(
            spec.max_seq_len, 1048576,
            "model_max_length must be 1048576"
        );
        assert!(spec.is_moe, "Inkling-Small is MoE (256 experts)");
        assert!(
            spec.is_multimodal,
            "Inkling-Small has vision + audio encoders"
        );
        assert!(spec.vision_spec.is_some(), "vision_config must be present");
        assert_eq!(
            spec.vision_spec.as_ref().unwrap().patch_size,
            40,
            "vision_config.patch_size must be 40"
        );
        assert!(spec.audio_spec.is_some(), "audio_config must be present");
        assert_eq!(
            spec.audio_spec.as_ref().unwrap().n_mel_bins,
            80,
            "audio_config.n_mel_bins must be 80"
        );
        assert_eq!(
            spec.expert_count,
            Some(256),
            "text_config.n_routed_experts must be 256"
        );
        assert_eq!(
            spec.expert_used_count,
            Some(6),
            "text_config.num_experts_per_tok must be 6"
        );

        let spec_json = spec.to_json().unwrap();
        assert!(spec_json.contains("inkling_mm_model"));

        let spec_toml = spec.to_toml().unwrap();
        assert!(spec_toml.contains("inkling_mm_model"));
    }
}
