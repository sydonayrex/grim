//! Model loading utilities for GGUF and safetensors files.

use grim_core::architecture::{ModelArchitecture, TensorNamingRegistry};
use grim_core::error::{Error, Result};
use grim_core::grim_plugins_dir;
use grim_core::hyperparams::{HyperparameterExtractor, MetadataLookup};
use grim_core::model::CausalLm;
use grim_format::{
    gguf::GgufValue,
    tprov::{RemappingTensorProvider, SafetensorsProvider},
    GgufProvider,
};
use grim_models_mamba::{
    GraniteHybridConfig, JambaConfig, Mamba, Mamba2Config, MambaConfig, NemotronHConfig, Rwkv,
    Rwkv6Config, Rwkv7Config, RwkvConfig,
};
use grim_models_transformer::{
    BloomConfig, DeepSeek, DeepSeekConfig, FalconConfig, Gemma, GemmaConfig, Gpt2, Gpt2Config,
    Lfm2, Lfm2Config, Llama, LlamaConfig, MoeConfig, PhiConfig, QwenConfig, T5, T5Config,
};
use grim_models_vision::{Bert, BertConfig, ModernBertConfig, NomicBertConfig, T5EncoderConfig};
use grim_nn::WeightSource;
use grim_plugin::ArchCompatSpec;
use grim_tensor::{Device, TensorProvider};
use serde::Deserialize;
use std::path::Path;

/// Attempt to resolve an `ArchCompatSpec` for an unknown architecture string,
/// first from an inline HF `config.json` string, and second by searching installed
/// `.grimplugin` manifests in `grim_plugins_dir()`.
fn resolve_arch_compat_spec(arch_str: &str, config_raw: Option<&str>) -> Option<ArchCompatSpec> {
    if let Some(json_str) = config_raw {
        if let Ok(spec) = ArchCompatSpec::from_hf_config_json(json_str) {
            if !spec.model_type.is_empty() {
                return Some(spec);
            }
        }
    }

    let plugins_dir = grim_plugins_dir();
    if plugins_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(plugins_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if ext == "grimplugin" || ext == "json" {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            if let Ok(spec) = ArchCompatSpec::from_hf_config_json(&content) {
                                if spec.model_type.eq_ignore_ascii_case(arch_str)
                                    || spec.name.eq_ignore_ascii_case(arch_str)
                                {
                                    return Some(spec);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Helper function to get metadata as string from GGUF provider
fn get_meta_str(provider: &GgufProvider, key: &str) -> Option<String> {
    let v: Option<&GgufValue> = provider.metadata(key);
    let v: &GgufValue = match v {
        Some(val) => val,
        None => return None,
    };
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(u) = v.as_u32() {
        return Some(u.to_string());
    }
    if let Some(f) = v.as_f32() {
        return Some(f.to_string());
    }
    None
}

/// Helper function to get metadata as array from GGUF provider
fn get_meta_array<'a>(provider: &'a GgufProvider, key: &str) -> Option<&'a [GgufValue]> {
    let v: Option<&GgufValue> = provider.metadata(key);
    if let Some(v) = v {
        v.as_array()
    } else {
        None
    }
}

/// Metadata accessor implementation wrapping `GgufProvider`.
struct GgufMetadataLookup<'a>(&'a GgufProvider);

impl<'a> MetadataLookup for GgufMetadataLookup<'a> {
    fn get_str(&self, key: &str) -> Option<String> {
        get_meta_str(self.0, key)
    }
    fn get_u32(&self, key: &str) -> Option<u32> {
        let v = self.0.metadata(key)?;
        if let Some(u) = v.as_u32() {
            return Some(u);
        }
        if let Some(s) = v.as_str() {
            if let Ok(u) = s.parse::<u32>() {
                return Some(u);
            }
        }
        None
    }
    fn get_f32(&self, key: &str) -> Option<f32> {
        let v = self.0.metadata(key)?;
        if let Some(f) = v.as_f32() {
            return Some(f);
        }
        if let Some(s) = v.as_str() {
            if let Ok(f) = s.parse::<f32>() {
                return Some(f);
            }
        }
        None
    }
}

/// Load a model from a GGUF file.
pub fn load_model_from_gguf(path: &str, device: Device) -> Result<Box<dyn CausalLm>> {
    let provider = GgufProvider::open(path)?;
    load_model_with_providers(&provider, &provider, device, path)
}

/// Load a model from a native `.grim` file with a sibling `.gguf` file containing metadata.
pub fn load_model_from_grim(path: &str, device: Device) -> Result<Box<dyn CausalLm>> {
    let gguf_path = std::path::Path::new(path).with_extension("gguf");
    let gguf_path_str = gguf_path.to_str().ok_or_else(|| {
        Error::Config(format!(
            "Invalid path for sibling GGUF file: {:?}",
            gguf_path
        ))
    })?;
    let gguf_provider = GgufProvider::open(gguf_path_str)?;
    let grim_provider = grim_format::tprov::GrimProvider::open(path)?;
    load_model_with_providers(&gguf_provider, &grim_provider, device, path)
}

/// Load a model from a safetensors file with a sibling config.json.
pub fn load_model_from_safetensors(path: &str, device: Device) -> Result<Box<dyn CausalLm>> {
    let path_obj = Path::new(path);
    // config.json is in the same directory as the model file
    let config_path = path_obj.parent().unwrap_or(path_obj).join("config.json");
    if config_path.to_str().is_none() {
        return Err(Error::Config(format!(
            "Invalid path for sibling config.json: {:?}",
            config_path
        )));
    }

    // Read and parse config.json
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::Config(format!("Failed to read config.json: {e}")))?;
    let config: SafetensorsConfig = serde_json::from_str(&config_str)
        .map_err(|e| Error::Config(format!("Failed to parse config.json: {e}")))?;

    // Open safetensors provider
    let provider = SafetensorsProvider::open(path)?;

    // Delegate to the config-based loader with raw config_str for ArchCompatSpec fallback
    load_model_from_config(config, &provider, device, path, Some(&config_str))
}

/// Minimal config extracted from HF config.json for model loading.
#[derive(Debug, Deserialize)]
struct SafetensorsConfig {
    #[serde(rename = "architectures")]
    architectures: Option<Vec<String>>,
    #[serde(rename = "model_type")]
    model_type: Option<String>,
    #[serde(rename = "hidden_size")]
    hidden_size: usize,
    #[serde(rename = "num_hidden_layers")]
    num_hidden_layers: usize,
    #[serde(rename = "vocab_size")]
    vocab_size: usize,
    #[serde(rename = "rms_norm_eps")]
    rms_norm_eps: Option<f32>,
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
    // LFM2-specific
    #[serde(rename = "shortconv_l_cache")]
    shortconv_l_cache: Option<usize>,
    #[serde(rename = "conv_l_cache")]
    conv_l_cache: Option<usize>,
    #[serde(rename = "attention_head_count_kv")]
    attention_head_count_kv: Option<Vec<u32>>,
    /// LFM2 layer types: "conv" (recurrent) or "full_attention".
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    // MoE specific
    #[serde(rename = "num_local_experts")]
    num_local_experts: Option<usize>,
    #[serde(rename = "num_experts_per_tok")]
    num_experts_per_tok: Option<usize>,
}

fn load_model_from_config(
    config: SafetensorsConfig,
    provider: &SafetensorsProvider,
    device: Device,
    _path: &str,
    raw_config_str: Option<&str>,
) -> Result<Box<dyn CausalLm>> {
    // Determine architecture
    let arch_str = config
        .model_type
        .or_else(|| config.architectures.and_then(|a| a.first().cloned()))
        .ok_or_else(|| {
            Error::Config("config.json missing model_type or architectures".into())
        })?;
    let model_arch = ModelArchitecture::from_str(&arch_str);

    let vocab_size = config.vocab_size;
    let hidden_size = config.hidden_size;
    let num_layers = config.num_hidden_layers;
    let rms_norm_eps = config.rms_norm_eps.unwrap_or(1e-5);
    let num_heads = config.num_attention_heads.unwrap_or(32);
    let num_kv_heads = config.num_key_value_heads.unwrap_or(num_heads);
    let head_dim = config.head_dim.unwrap_or_else(|| if num_heads > 0 { hidden_size / num_heads } else { 128 });
    let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
    let max_seq_len = config.max_position_embeddings.unwrap_or(2048);
    let rope_theta = config.rope_theta.unwrap_or(10000.0);

    eprintln!(
        "[grim] Loading config from safetensors: architecture={:?}, layers={}, hidden={}, vocab={}",
        model_arch, num_layers, hidden_size, vocab_size
    );

    let device_clone = device.clone();
    let ws = WeightSource::root(provider, device);

    match model_arch {
        ModelArchitecture::Falcon => {
            let falcon_cfg = FalconConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                max_seq_len,
            };
            eprintln!("[grim] Loading Falcon model with config: {:?}", falcon_cfg);
            let llama_cfg = LlamaConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            let m = Llama::load(&ws, llama_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Bloom => {
            let bloom_cfg = BloomConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                layer_norm_epsilon: rms_norm_eps,
                max_seq_len,
            };
            eprintln!("[grim] Loading BLOOM model with config: {:?}", bloom_cfg);
            let gpt2_cfg = Gpt2Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                layer_norm_epsilon: rms_norm_eps,
                max_seq_len,
            };
            let m = Gpt2::load(&ws, gpt2_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Phi2 | ModelArchitecture::Phi3 | ModelArchitecture::PhiMoe => {
            let phi_cfg = PhiConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            eprintln!("[grim] Loading Phi model with config: {:?}", phi_cfg);
            let llama_cfg = LlamaConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            let m = Llama::load(&ws, llama_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen | ModelArchitecture::Qwen2 | ModelArchitecture::Qwen3 | ModelArchitecture::Qwen35 => {
            let qwen_cfg = QwenConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            eprintln!("[grim] Loading Qwen model with config: {:?}", qwen_cfg);
            let llama_cfg = LlamaConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            let m = Llama::load(&ws, llama_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen2Moe | ModelArchitecture::Qwen3Moe | ModelArchitecture::Qwen35Moe | ModelArchitecture::Qwen3VlMoe => {
            let qwen_moe_cfg = MoeConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                expert_count: config.num_local_experts.unwrap_or(8),
                expert_used_count: config.num_experts_per_tok.unwrap_or(2),
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            eprintln!("[grim] Loading Qwen-MoE model with config: {:?}", qwen_moe_cfg);
            let deepseek_cfg = DeepSeekConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                q_lora_rank: num_heads,
                kv_lora_rank: num_kv_heads * 4,
            };
            let m = DeepSeek::load(&ws, deepseek_cfg)?;
            Ok(Box::new(m))
        }
        arch if arch.is_moe() => {
            let moe_cfg = MoeConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                expert_count: config.num_local_experts.unwrap_or(8),
                expert_used_count: config.num_experts_per_tok.unwrap_or(2),
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            eprintln!("[grim] Loading MoE model with config: {:?}", moe_cfg);
            let deepseek_cfg = DeepSeekConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                q_lora_rank: num_heads,
                kv_lora_rank: num_kv_heads * 4,
            };
            let m = DeepSeek::load(&ws, deepseek_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Mamba2 => {
            let mamba2_cfg = Mamba2Config {
                vocab_size,
                hidden_size,
                d_state: 16,
                d_inner: intermediate_size,
                d_conv: 4,
                num_heads,
                num_layers,
                rms_norm_eps,
            };
            eprintln!("[grim] Loading Mamba2 model with config: {:?}", mamba2_cfg);
            let mamba_cfg = MambaConfig {
                vocab_size,
                hidden_size,
                d_state: 16,
                d_inner: intermediate_size,
                d_conv: 4,
                num_layers,
                conv_kernel: 4,
                rms_norm_eps,
            };
            let m = Mamba::load(&ws, mamba_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Jamba => {
            let jamba_cfg = JambaConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                num_layers,
                intermediate_size,
                expert_count: config.num_local_experts.unwrap_or(8),
                expert_used_count: config.num_experts_per_tok.unwrap_or(2),
                ssm_d_state: 16,
                rms_norm_eps,
                max_seq_len,
            };
            eprintln!("[grim] Loading Jamba model with config: {:?}", jamba_cfg);
            let mamba_cfg = MambaConfig {
                vocab_size,
                hidden_size,
                d_state: 16,
                d_inner: intermediate_size,
                d_conv: 4,
                num_layers,
                conv_kernel: 4,
                rms_norm_eps,
            };
            let m = Mamba::load(&ws, mamba_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::NemotronH => {
            let nemotron_cfg = NemotronHConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                ssm_d_state: 16,
                rms_norm_eps,
                max_seq_len,
            };
            eprintln!("[grim] Loading NemotronH model with config: {:?}", nemotron_cfg);
            let mamba_cfg = MambaConfig {
                vocab_size,
                hidden_size,
                d_state: 16,
                d_inner: intermediate_size,
                d_conv: 4,
                num_layers,
                conv_kernel: 4,
                rms_norm_eps,
            };
            let m = Mamba::load(&ws, mamba_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::GraniteHybrid => {
            let granite_cfg = GraniteHybridConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                ssm_d_state: 16,
                rms_norm_eps,
                max_seq_len,
            };
            eprintln!("[grim] Loading GraniteHybrid model with config: {:?}", granite_cfg);
            let mamba_cfg = MambaConfig {
                vocab_size,
                hidden_size,
                d_state: 16,
                d_inner: intermediate_size,
                d_conv: 4,
                num_layers,
                conv_kernel: 4,
                rms_norm_eps,
            };
            let m = Mamba::load(&ws, mamba_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::ModernBert => {
            let modern_bert_cfg = ModernBertConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                layer_norm_eps: rms_norm_eps,
                max_seq_len,
            };
            eprintln!("[grim] Loading ModernBERT model with config: {:?}", modern_bert_cfg);
            let bert_cfg = BertConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                max_seq_len,
            };
            let m = Bert::load(&ws, bert_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::NomicBert | ModelArchitecture::NomicBertMoe | ModelArchitecture::NeoBert | ModelArchitecture::JinaBertV2 | ModelArchitecture::JinaBertV3 => {
            let nomic_bert_cfg = NomicBertConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                layer_norm_eps: rms_norm_eps,
                max_seq_len,
            };
            eprintln!("[grim] Loading NomicBERT model with config: {:?}", nomic_bert_cfg);
            let bert_cfg = BertConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                max_seq_len,
            };
            let m = Bert::load(&ws, bert_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::T5Encoder => {
            let t5_enc_cfg = T5EncoderConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                max_seq_len,
            };
            eprintln!("[grim] Loading T5Encoder model with config: {:?}", t5_enc_cfg);
            let t5_cfg = T5Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
            };
            let m = T5::load(&ws, t5_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Rwkv6 | ModelArchitecture::Rwkv6Qwen2 => {
            let rwkv6_cfg = Rwkv6Config {
                vocab_size,
                hidden_size,
                num_layers,
                head_dim,
                max_seq_len,
            };
            eprintln!("[grim] Loading RWKV6 model with config: {:?}", rwkv6_cfg);
            let rwkv_cfg = RwkvConfig {
                vocab_size,
                hidden_size,
                num_layers,
            };
            let m = Rwkv::load(&ws, rwkv_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Rwkv7 | ModelArchitecture::ARwkv7 => {
            let rwkv7_cfg = Rwkv7Config {
                vocab_size,
                hidden_size,
                num_layers,
                head_dim,
                max_seq_len,
            };
            eprintln!("[grim] Loading RWKV7 model with config: {:?}", rwkv7_cfg);
            let rwkv_cfg = RwkvConfig {
                vocab_size,
                hidden_size,
                num_layers,
            };
            let m = Rwkv::load(&ws, rwkv_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Lfm2 | ModelArchitecture::Lfm2Moe => {
            let mapping = TensorNamingRegistry::remap_hf_to_gguf(model_arch, num_layers);
            let remap_fn = move |name: &str| -> String {
                if let Some(mapped) = mapping.get(name) {
                    return mapped.clone();
                }
                name.to_string()
            };
            let remapped_provider = RemappingTensorProvider::new(provider, remap_fn);
            let ws = WeightSource::root(&remapped_provider, device_clone);

            let intermediate_size = remapped_provider
                .meta("blk.0.ffn_gate.weight")
                .ok()
                .and_then(|m| m.shape.first().copied())
                .unwrap_or_else(|| config.intermediate_size.unwrap_or(4608));
            let n_shortconv_l_cache = config
                .shortconv_l_cache
                .or(config.conv_l_cache)
                .unwrap_or(3);

            let mut is_recr: Vec<bool> = Vec::with_capacity(num_layers);
            if let Some(layer_types) = &config.layer_types {
                for lt in layer_types.iter().take(num_layers) {
                    is_recr.push(lt == "conv");
                }
            } else if let Some(kv_array) = &config.attention_head_count_kv {
                for &n in kv_array.iter().take(num_layers) {
                    is_recr.push(n == 0);
                }
            }
            is_recr.resize(num_layers, false);

            eprintln!("[grim] LFM2 layer-type map (T=shortconv): {:?}", is_recr);

            let cfg = Lfm2Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                n_shortconv_l_cache,
                is_recr,
            };

            let m = Lfm2::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Mamba => {
            let d_state = 16;
            let d_inner = config.intermediate_size.unwrap_or(hidden_size * 2);
            let d_conv = 4;
            let cfg = MambaConfig {
                vocab_size,
                hidden_size,
                d_state,
                d_inner,
                d_conv,
                num_layers,
                conv_kernel: d_conv,
                rms_norm_eps,
            };
            let m = Mamba::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Gpt2 => {
            let cfg = Gpt2Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                layer_norm_epsilon: rms_norm_eps,
                max_seq_len,
            };
            let m = Gpt2::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Gemma | ModelArchitecture::Gemma2 | ModelArchitecture::Gemma3 | ModelArchitecture::Gemma4 => {
            let cfg = GemmaConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim: config.head_dim.unwrap_or(256),
                num_layers,
                intermediate_size: config.intermediate_size.unwrap_or(16384),
                rms_norm_eps,
            };
            let m = Gemma::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek | ModelArchitecture::DeepSeek2 | ModelArchitecture::DeepSeek32 | ModelArchitecture::DeepSeek4 => {
            let cfg = DeepSeekConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                q_lora_rank: num_heads,
                kv_lora_rank: num_kv_heads * 4,
            };
            let m = DeepSeek::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        arch if arch.is_encoder() => {
            let cfg = BertConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                max_seq_len,
            };
            let m = Bert::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::T5 => {
            let cfg = T5Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
            };
            let m = T5::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        _ => {
            if let Some(spec) = resolve_arch_compat_spec(&arch_str, raw_config_str) {
                eprintln!(
                    "[grim] Resolved unknown architecture '{}' via ArchCompatSpec plugin (base='{}', is_moe={})",
                    arch_str, spec.base_architecture, spec.is_moe
                );
                if spec.is_moe {
                    let deepseek_cfg = DeepSeekConfig {
                        vocab_size: spec.vocab_size,
                        hidden_size: spec.hidden_size,
                        num_heads: spec.num_heads,
                        num_layers: spec.num_layers,
                        intermediate_size: spec.intermediate_size,
                        rms_norm_eps: spec.rms_norm_eps,
                        q_lora_rank: spec.num_heads,
                        kv_lora_rank: spec.num_kv_heads * 4,
                    };
                    let m = DeepSeek::load(&ws, deepseek_cfg)?;
                    return Ok(Box::new(m));
                } else if spec.is_ssm {
                    let mamba_cfg = MambaConfig {
                        vocab_size: spec.vocab_size,
                        hidden_size: spec.hidden_size,
                        d_state: 16,
                        d_inner: spec.intermediate_size,
                        d_conv: 4,
                        num_layers: spec.num_layers,
                        conv_kernel: 4,
                        rms_norm_eps: spec.rms_norm_eps,
                    };
                    let m = Mamba::load(&ws, mamba_cfg)?;
                    return Ok(Box::new(m));
                } else {
                    let llama_cfg = LlamaConfig {
                        vocab_size: spec.vocab_size,
                        hidden_size: spec.hidden_size,
                        num_heads: spec.num_heads,
                        num_kv_heads: spec.num_kv_heads,
                        head_dim: spec.head_dim,
                        num_layers: spec.num_layers,
                        intermediate_size: spec.intermediate_size,
                        rms_norm_eps: spec.rms_norm_eps,
                        rope_theta: spec.rope_theta,
                        max_seq_len: spec.max_seq_len,
                    };
                    let m = Llama::load(&ws, llama_cfg)?;
                    return Ok(Box::new(m));
                }
            }

            eprintln!("[grim] Unknown architecture '{}' with no plugin compat spec found; using default Llama loader", arch_str);
            let cfg = LlamaConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            let m = Llama::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
    }
}

fn load_model_with_providers(
    provider: &GgufProvider,
    weight_provider: &dyn grim_tensor::TensorProvider,
    device: Device,
    _path: &str,
) -> Result<Box<dyn CausalLm>> {
    // Extract architecture from GGUF metadata
    let arch_str = provider.architecture().ok_or_else(|| {
        Error::Config(format!(
            "GGUF file has no 'general.architecture' metadata; cannot determine model family"
        ))
    })?;

    let model_arch = ModelArchitecture::from_str(arch_str);
    let lookup = GgufMetadataLookup(provider);
    let hparams = HyperparameterExtractor::extract(model_arch, &lookup);

    eprintln!(
        "[grim] Loading config: architecture={:?}, layers={}, hidden={}, vocab={}",
        model_arch, hparams.num_layers, hparams.hidden_size, hparams.vocab_size
    );

    let ws = WeightSource::root(weight_provider, device);

    match model_arch {
        ModelArchitecture::Falcon => {
            let falcon_cfg = FalconConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading Falcon model with config: {:?}", falcon_cfg);
            let llama_cfg = LlamaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Llama::load(&ws, llama_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Bloom => {
            let bloom_cfg = BloomConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                layer_norm_epsilon: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading BLOOM model with config: {:?}", bloom_cfg);
            let gpt2_cfg = Gpt2Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                layer_norm_epsilon: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Gpt2::load(&ws, gpt2_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Phi2 | ModelArchitecture::Phi3 | ModelArchitecture::PhiMoe => {
            let phi_cfg = PhiConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading Phi model with config: {:?}", phi_cfg);
            let llama_cfg = LlamaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Llama::load(&ws, llama_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen | ModelArchitecture::Qwen2 | ModelArchitecture::Qwen3 | ModelArchitecture::Qwen35 => {
            let qwen_cfg = QwenConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading Qwen model with config: {:?}", qwen_cfg);
            let llama_cfg = LlamaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Llama::load(&ws, llama_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen2Moe | ModelArchitecture::Qwen3Moe | ModelArchitecture::Qwen35Moe | ModelArchitecture::Qwen3VlMoe => {
            let qwen_moe_cfg = MoeConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                expert_count: hparams.expert_count.unwrap_or(8),
                expert_used_count: hparams.expert_used_count.unwrap_or(2),
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading Qwen-MoE model with config: {:?}", qwen_moe_cfg);
            let deepseek_cfg = DeepSeekConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                q_lora_rank: hparams.num_heads,
                kv_lora_rank: hparams.num_kv_heads * 4,
            };
            let m = DeepSeek::load(&ws, deepseek_cfg)?;
            Ok(Box::new(m))
        }
        arch if arch.is_moe() => {
            let moe_cfg = MoeConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                expert_count: hparams.expert_count.unwrap_or(8),
                expert_used_count: hparams.expert_used_count.unwrap_or(2),
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading MoE model with config: {:?}", moe_cfg);
            let deepseek_cfg = DeepSeekConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                q_lora_rank: hparams.num_heads,
                kv_lora_rank: hparams.num_kv_heads * 4,
            };
            let m = DeepSeek::load(&ws, deepseek_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Mamba2 => {
            let mamba2_cfg = Mamba2Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                d_state: hparams.ssm_d_state.unwrap_or(16),
                d_inner: hparams.ssm_d_inner.unwrap_or(hparams.hidden_size * 2),
                d_conv: hparams.ssm_d_conv.unwrap_or(4),
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                rms_norm_eps: hparams.rms_norm_eps,
            };
            eprintln!("[grim] Loading Mamba2 model with config: {:?}", mamba2_cfg);
            let mamba_cfg = MambaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                d_state: hparams.ssm_d_state.unwrap_or(16),
                d_inner: hparams.ssm_d_inner.unwrap_or(hparams.hidden_size * 2),
                d_conv: hparams.ssm_d_conv.unwrap_or(4),
                num_layers: hparams.num_layers,
                conv_kernel: hparams.ssm_d_conv.unwrap_or(4),
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = Mamba::load(&ws, mamba_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Jamba => {
            let jamba_cfg = JambaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                expert_count: hparams.expert_count.unwrap_or(8),
                expert_used_count: hparams.expert_used_count.unwrap_or(2),
                ssm_d_state: hparams.ssm_d_state.unwrap_or(16),
                rms_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading Jamba model with config: {:?}", jamba_cfg);
            let mamba_cfg = MambaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                d_state: hparams.ssm_d_state.unwrap_or(16),
                d_inner: hparams.ssm_d_inner.unwrap_or(hparams.hidden_size * 2),
                d_conv: hparams.ssm_d_conv.unwrap_or(4),
                num_layers: hparams.num_layers,
                conv_kernel: hparams.ssm_d_conv.unwrap_or(4),
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = Mamba::load(&ws, mamba_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::NemotronH => {
            let nemotron_cfg = NemotronHConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                ssm_d_state: hparams.ssm_d_state.unwrap_or(16),
                rms_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading NemotronH model with config: {:?}", nemotron_cfg);
            let mamba_cfg = MambaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                d_state: hparams.ssm_d_state.unwrap_or(16),
                d_inner: hparams.ssm_d_inner.unwrap_or(hparams.hidden_size * 2),
                d_conv: hparams.ssm_d_conv.unwrap_or(4),
                num_layers: hparams.num_layers,
                conv_kernel: hparams.ssm_d_conv.unwrap_or(4),
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = Mamba::load(&ws, mamba_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::GraniteHybrid => {
            let granite_cfg = GraniteHybridConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                ssm_d_state: hparams.ssm_d_state.unwrap_or(16),
                rms_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading GraniteHybrid model with config: {:?}", granite_cfg);
            let mamba_cfg = MambaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                d_state: hparams.ssm_d_state.unwrap_or(16),
                d_inner: hparams.ssm_d_inner.unwrap_or(hparams.hidden_size * 2),
                d_conv: hparams.ssm_d_conv.unwrap_or(4),
                num_layers: hparams.num_layers,
                conv_kernel: hparams.ssm_d_conv.unwrap_or(4),
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = Mamba::load(&ws, mamba_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::ModernBert => {
            let modern_bert_cfg = ModernBertConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                layer_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading ModernBERT model with config: {:?}", modern_bert_cfg);
            let bert_cfg = BertConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Bert::load(&ws, bert_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::NomicBert | ModelArchitecture::NomicBertMoe | ModelArchitecture::NeoBert | ModelArchitecture::JinaBertV2 | ModelArchitecture::JinaBertV3 => {
            let nomic_bert_cfg = NomicBertConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                layer_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading NomicBERT model with config: {:?}", nomic_bert_cfg);
            let bert_cfg = BertConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Bert::load(&ws, bert_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::T5Encoder => {
            let t5_enc_cfg = T5EncoderConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading T5Encoder model with config: {:?}", t5_enc_cfg);
            let t5_cfg = T5Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = T5::load(&ws, t5_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Rwkv6 | ModelArchitecture::Rwkv6Qwen2 => {
            let rwkv6_cfg = Rwkv6Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_layers: hparams.num_layers,
                head_dim: hparams.head_dim,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading RWKV6 model with config: {:?}", rwkv6_cfg);
            let rwkv_cfg = RwkvConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_layers: hparams.num_layers,
            };
            let m = Rwkv::load(&ws, rwkv_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Rwkv7 | ModelArchitecture::ARwkv7 => {
            let rwkv7_cfg = Rwkv7Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_layers: hparams.num_layers,
                head_dim: hparams.head_dim,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading RWKV7 model with config: {:?}", rwkv7_cfg);
            let rwkv_cfg = RwkvConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_layers: hparams.num_layers,
            };
            let m = Rwkv::load(&ws, rwkv_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Lfm2 | ModelArchitecture::Lfm2Moe => {
            let mut head_count_kv_vec: Vec<u32> = Vec::with_capacity(hparams.num_layers);
            if let Some(arr_val) = get_meta_array(provider, "lfm2.attention.head_count_kv") {
                for v in arr_val.iter().take(hparams.num_layers) {
                    let v: &grim_format::gguf::GgufValue = v;
                    let n: u32 = v.as_u32().unwrap_or_else(|| {
                        if let Some(s) = v.as_str() {
                            s.parse::<u32>().unwrap_or(0u32)
                        } else {
                            0u32
                        }
                    });
                    head_count_kv_vec.push(n);
                }
            }
            for i in 0..hparams.num_layers {
                if i < head_count_kv_vec.len() {
                    continue;
                }
                let key = format!("lfm2.attention.head_count_kv.{i}");
                let n: u32 = if let Some(val) = provider.metadata(&key) {
                    let val: &grim_format::gguf::GgufValue = val;
                    val.as_u32().unwrap_or(0u32)
                } else {
                    0u32
                };
                if (i + 1) > head_count_kv_vec.len() {
                    head_count_kv_vec.resize(i + 1, 0);
                }
                head_count_kv_vec[i] = n;
            }
            head_count_kv_vec.resize(hparams.num_layers, 0);
            let is_recr: Vec<bool> = head_count_kv_vec.iter().map(|&n| n == 0).collect();
            eprintln!("[grim] LFM2 layer-type map (T=shortconv): {:?}", is_recr);
            let n_shortconv_l_cache = 3usize;
            let num_kv_heads = head_count_kv_vec
                .iter()
                .find(|&&n| n > 0)
                .copied()
                .map(|n| n as usize)
                .unwrap_or(hparams.num_kv_heads);
            let cfg = Lfm2Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                n_shortconv_l_cache,
                is_recr: is_recr.clone(),
            };
            let m = Lfm2::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Mamba => {
            let d_state = hparams.ssm_d_state.unwrap_or(16);
            let d_inner = hparams.ssm_d_inner.unwrap_or(hparams.hidden_size * 2);
            let d_conv = hparams.ssm_d_conv.unwrap_or(4);
            let cfg = MambaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                d_state,
                d_inner,
                d_conv,
                num_layers: hparams.num_layers,
                conv_kernel: d_conv,
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = Mamba::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Gpt2 => {
            let cfg = Gpt2Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                layer_norm_epsilon: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Gpt2::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Gemma | ModelArchitecture::Gemma2 | ModelArchitecture::Gemma3 | ModelArchitecture::Gemma4 => {
            let cfg = GemmaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = Gemma::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek | ModelArchitecture::DeepSeek2 | ModelArchitecture::DeepSeek32 | ModelArchitecture::DeepSeek4 => {
            let cfg = DeepSeekConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                q_lora_rank: hparams.num_heads,
                kv_lora_rank: hparams.num_kv_heads * 4,
            };
            let m = DeepSeek::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        arch if arch.is_encoder() => {
            let cfg = BertConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Bert::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::T5 => {
            let cfg = T5Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = T5::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
        _ => {
            if let Some(spec) = resolve_arch_compat_spec(arch_str, None) {
                eprintln!(
                    "[grim] Resolved unknown GGUF architecture '{}' via ArchCompatSpec plugin (base='{}', is_moe={})",
                    arch_str, spec.base_architecture, spec.is_moe
                );
                if spec.is_moe {
                    let deepseek_cfg = DeepSeekConfig {
                        vocab_size: hparams.vocab_size,
                        hidden_size: hparams.hidden_size,
                        num_heads: hparams.num_heads,
                        num_layers: hparams.num_layers,
                        intermediate_size: hparams.intermediate_size,
                        rms_norm_eps: hparams.rms_norm_eps,
                        q_lora_rank: hparams.num_heads,
                        kv_lora_rank: hparams.num_kv_heads * 4,
                    };
                    let m = DeepSeek::load(&ws, deepseek_cfg)?;
                    return Ok(Box::new(m));
                } else if spec.is_ssm {
                    let mamba_cfg = MambaConfig {
                        vocab_size: hparams.vocab_size,
                        hidden_size: hparams.hidden_size,
                        d_state: hparams.ssm_d_state.unwrap_or(16),
                        d_inner: hparams.ssm_d_inner.unwrap_or(hparams.hidden_size * 2),
                        d_conv: hparams.ssm_d_conv.unwrap_or(4),
                        num_layers: hparams.num_layers,
                        conv_kernel: hparams.ssm_d_conv.unwrap_or(4),
                        rms_norm_eps: hparams.rms_norm_eps,
                    };
                    let m = Mamba::load(&ws, mamba_cfg)?;
                    return Ok(Box::new(m));
                } else {
                    let llama_cfg = LlamaConfig {
                        vocab_size: hparams.vocab_size,
                        hidden_size: hparams.hidden_size,
                        num_heads: hparams.num_heads,
                        num_kv_heads: hparams.num_kv_heads,
                        head_dim: hparams.head_dim,
                        num_layers: hparams.num_layers,
                        intermediate_size: hparams.intermediate_size,
                        rms_norm_eps: hparams.rms_norm_eps,
                        rope_theta: hparams.rope_theta,
                        max_seq_len: hparams.max_seq_len,
                    };
                    let m = Llama::load(&ws, llama_cfg)?;
                    return Ok(Box::new(m));
                }
            }

            eprintln!("[grim] Unknown GGUF architecture '{}' with no plugin compat spec found; using default Llama loader", arch_str);
            let cfg = LlamaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Llama::load(&ws, cfg)?;
            Ok(Box::new(m))
        }
    }
}

/// Convenience wrapper: detect the best available device and load a GGUF or GRIM model.
///
/// Device priority: ROCm → CUDA → Metal → CPU.  This is the entry point called by
/// `grim-server`'s on-demand model loader so callers don't need to manage
/// device selection themselves.
pub fn load_from_path(path: &str) -> Result<Box<dyn CausalLm>> {
    let is_grim = path.ends_with(".grim");
    let is_safetensors = path.ends_with(".safetensors");

    // Check for forced device first
    if let Ok(s) = std::env::var("GRIM_FORCE_DEVICE") {
        match s.as_str() {
            "cuda" => {
                if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
                    if let Some(first) = cuda_devices.first() {
                        eprintln!("[model_loader] Using CUDA device {} (forced)", first.ordinal());
                        let dev = Device::Cuda(first.ordinal());
                        return if is_grim {
                            load_model_from_grim(path, dev)
                        } else if is_safetensors {
                            load_model_from_safetensors(path, dev)
                        } else {
                            load_model_from_gguf(path, dev)
                        };
                    }
                }
            }
            "rocm" => {
                if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
                    if let Some(first) = rocm_devices.first() {
                        eprintln!("[model_loader] Using ROCm device {} (forced)", first.ordinal());
                        let dev = Device::Rocm(first.ordinal());
                        return if is_grim {
                            load_model_from_grim(path, dev)
                        } else if is_safetensors {
                            load_model_from_safetensors(path, dev)
                        } else {
                            load_model_from_gguf(path, dev)
                        };
                    }
                }
            }
            "cpu" => {
                eprintln!("[model_loader] Using CPU (forced)");
                let dev = Device::Cpu;
                return if is_grim {
                    load_model_from_grim(path, dev)
                } else if is_safetensors {
                    load_model_from_safetensors(path, dev)
                } else {
                    load_model_from_gguf(path, dev)
                };
            }
            _ => {}
        }
    }

    // Attempt ROCm first (AMD GPU — primary grim target).
    if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
        if let Some(first) = rocm_devices.first() {
            eprintln!("[model_loader] Using ROCm device {}", first.ordinal());
            let dev = Device::Rocm(first.ordinal());
            return if is_grim {
                load_model_from_grim(path, dev)
            } else if is_safetensors {
                load_model_from_safetensors(path, dev)
            } else {
                load_model_from_gguf(path, dev)
            };
        }
    }
    // Fall back to CUDA.
    if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
        if let Some(first) = cuda_devices.first() {
            eprintln!("[model_loader] Using CUDA device {}", first.ordinal());
            let dev = Device::Cuda(first.ordinal());
            return if is_grim {
                load_model_from_grim(path, dev)
            } else if is_safetensors {
                load_model_from_safetensors(path, dev)
            } else {
                load_model_from_gguf(path, dev)
            };
        }
    }
    // Fall back to Metal.
    if let Ok(metal_devices) = grim_backend_metal::MetalDevice::probe() {
        if let Some(first) = metal_devices.first() {
            eprintln!("[model_loader] Using Metal device {}", first.ordinal());
            let dev = Device::Metal(first.ordinal());
            return if is_grim {
                load_model_from_grim(path, dev)
            } else if is_safetensors {
                load_model_from_safetensors(path, dev)
            } else {
                load_model_from_gguf(path, dev)
            };
        }
    }
    // CPU fallback.
    eprintln!("[model_loader] No GPU detected; using CPU.");
    let dev = Device::Cpu;
    if is_grim {
        load_model_from_grim(path, dev)
    } else if is_safetensors {
        load_model_from_safetensors(path, dev)
    } else {
        load_model_from_gguf(path, dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_from_path_picks_grim_extension() {
        let is_grim_dispatch = |p: &str| p.ends_with(".grim");
        assert!(is_grim_dispatch("/models/llama3.grim"));
        assert!(!is_grim_dispatch("/models/llama3.gguf"));
        assert!(!is_grim_dispatch("/nonexistent"));
    }

    #[test]
    fn load_from_path_dispatches_to_grim_loader() {
        let r = load_from_path("/tmp/__grim_does_not_exist__.grim");
        match r {
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    !msg.is_empty(),
                    "expected error message from grim dispatch, got empty"
                );
            }
            Ok(_) => panic!("non-existent .grim must not load successfully"),
        }
    }
}
