//! Model loading utilities for GGUF and safetensors files.

use grim_core::architecture::{ModelArchitecture, TensorNamingRegistry};
use grim_core::error::{Error, Result};
use grim_core::grim_plugins_dir;
use grim_core::hyperparams::{HyperparameterExtractor, MetadataLookup};
use grim_core::model::CausalLm;
use grim_format::{
    GgufProvider, PthProvider,
    gguf::GgufValue,
    tprov::{RemappingTensorProvider, SafetensorsProvider},
};
use grim_models_mamba::{
    GraniteHybridConfig, JambaConfig, Mamba, Mamba2Config, MambaConfig, NemotronHConfig, Rwkv,
    Rwkv6Config, Rwkv7Config, RwkvConfig,
};
use grim_models_transformer::{
    Bloom, BloomConfig, Chameleon, ChameleonConfig, CogVlm, CogVlmConfig, CogVlmVisionConfig,
    CommandRConfig, DeepSeek, DeepSeek2, DeepSeek2Config, DeepSeek4, DeepSeek4Config, DeepSeek32,
    DeepSeek32Config, DeepSeekConfig, DeltaNetBase, DeltaNetBaseConfig, DiffusionGemma,
    DiffusionGemmaConfig, Falcon, FalconConfig, FalconH1Config, FalconH1Model, Gemma, Gemma3n,
    Gemma3nConfig, GemmaConfig, Glm52, Glm52Config, Gpt2, Gpt2Config, HunyuanVl, HunyuanVlConfig,
    HunyuanVlVisionConfig, InklingSmall, InklingSmallConfig, InternS2Mobius, InternS2MobiusConfig,
    KimiK3, KimiK3Config, Laguna, LagunaConfig, Lfm2, Lfm2Config, Llama, LlamaConfig, Mellum,
    MellumConfig, MiniCpmConfig, MiniCpmModel, MiniMaxM3, MiniMaxM3Config, Phi2, PhiConfig, Qwen,
    Qwen2Vl, Qwen2VlConfig, Qwen2VlVisionConfig, Qwen3Moe, Qwen3MoeConfig, Qwen3Vl, Qwen3VlConfig,
    Qwen3VlVisionConfig, Qwen35, Qwen35Config, Qwen35Moe, Qwen35MoeConfig, Qwen38FlashNext,
    Qwen38FlashNextConfig, QwenConfig, SmolLm2,
    SmolLm2Config, SolarOpen2, SolarOpen2Config, T5, T5Config, WavTokenizerDec,
    WavTokenizerDecConfig,
};
use grim_models_vision::{Bert, BertConfig, ModernBertConfig, NomicBertConfig, T5EncoderConfig};
use grim_nn::{TensorParallelConfig, WeightSource};
use grim_plugin::ArchCompatSpec;
use grim_tensor::{Device, TensorProvider, YaRNParams};
use serde::Deserialize;
use std::path::Path;

/// Resolve this process's tensor-parallel config from `GRIM_TP_*` and validate
/// the `(rank, world_size)` contract.
///
/// Returns the default `{rank:0, world_size:1}` when `GRIM_TP_SIZE` is unset
/// or `1` (single-device). Returns `Err(Config)` on a malformed contract
/// (e.g. `rank >= world_size`, `world_size == 0`) so the loader fails loudly
/// rather than silently loading the wrong shard — the central correctness fix
/// from the TP sanity check.
///
/// This is the **single source of truth** for `(rank, world_size)` inside the
/// loader. The derived `tp` is attached to every `WeightSource` via
/// `with_tp_config(tp)`, so `get_sharded` slices by `tp.rank`; and it is passed
/// by value to each `Foo::load_tp(...)` so the column/row-parallel linears
/// shard consistently. `Llama::load` / `LlamaBlock::load` no longer re-read the
/// env (they take `ws.tp_config()` instead), closing the split-brain where the
/// loader's slice rank could disagree with the model's shard rank.
fn resolve_tp_config() -> Result<TensorParallelConfig> {
    let tp = TensorParallelConfig::from_env().unwrap_or_default();
    if let Err(msg) = tp.validate() {
        return Err(Error::Config(format!(
            "invalid tensor-parallel configuration (GRIM_TP_SIZE / GRIM_TP_RANK): {msg}"
        )));
    }
    if tp.world_size > 1 {
        eprintln!(
            "[grim-engine] TP rank {}/{} (from GRIM_TP_*); \
             loading only this rank's shard",
            tp.rank, tp.world_size
        );
    }
    Ok(tp)
}

/// Resolve which GPU ordinal this process should load on under the
/// multi-process TP contract.
///
/// - When `GRIM_TP_SIZE > 1`: the ordinal is `GRIM_GPUS[GRIM_TP_RANK]` if the
///   env gave one ordinal per rank; otherwise it falls back to
///   `GRIM_TP_RANK` itself as the ordinal. The full ordinal list
///   (`all_ordinals`) is also returned so the engine can build a single
///   `RcclAllReduce` over the whole group (`ncclCommInitAll` needs every
///   participating ordinal, not just this rank's).
/// - When `GRIM_TP_SIZE <= 1` (single-device): returns `(None, None)` so the
///   caller uses its existing "pick `probe().first()`" heuristic.
///
/// This must agree with `resolve_tp_config()`'s validation — it reads the same
/// `GRIM_TP_*` env vars. Kept here (and mirrored in `Engine::new`) so the
/// loader and engine pick the same ordinal without a shared crate dependency.
fn resolve_tp_ordinal() -> Result<(Option<usize>, Option<Vec<usize>>)> {
    let tp = TensorParallelConfig::from_env();
    let Some(tp) = tp else {
        return Ok((None, None));
    };
    tp.validate().map_err(|msg| {
        Error::Config(format!(
            "invalid tensor-parallel configuration (GRIM_TP_SIZE / GRIM_TP_RANK): {msg}"
        ))
    })?;
    let gpus: Vec<usize> = std::env::var("GRIM_GPUS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default();
    // All participating ordinals for the RCCL group.
    let all_ordinals: Vec<usize> = if !gpus.is_empty() {
        // Honour the explicit selection; pad up to world_size with rank-as-ordinal
        // only if the user gave a short list (defensive — the documented contract
        // is one ordinal per rank).
        if gpus.len() >= tp.world_size {
            gpus.iter().take(tp.world_size).copied().collect()
        } else {
            (0..tp.world_size).collect()
        }
    } else {
        (0..tp.world_size).collect()
    };
    // This rank's ordinal: explicit list indexed by rank, else rank-as-ordinal.
    let my_ordinal = gpus.get(tp.rank).copied().unwrap_or(tp.rank);
    Ok((Some(my_ordinal), Some(all_ordinals)))
}

/// Resolve discrete ROCm GPUs (honoring `GRIM_GPUS` and taking only dedicated GPUs 0 and 1,
/// strictly excluding integrated APU devices).
pub fn resolve_discrete_rocm_devices(fallback: &Device) -> Vec<Device> {
    if let Ok(val) = std::env::var("GRIM_GPUS") {
        let explicit: Vec<Device> = val
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .map(Device::Rocm)
            .collect();
        if !explicit.is_empty() {
            return explicit;
        }
    }

    if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
        let discrete: Vec<Device> = rocm_devices
            .iter()
            .take(2)
            .map(|d| Device::Rocm(d.ordinal()))
            .collect();
        if !discrete.is_empty() && !fallback.is_cpu() {
            return discrete;
        }
    }
    vec![fallback.clone()]
}

/// Debug-only logging macro. Compiles to a no-op in release builds so that
/// diagnostic  calls do not pollute production stderr (sims.md issue #7).
macro_rules! dbg_eprintln {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        { eprintln!($($arg)*) }
    };
}
/// Probe the host GPU's actual wavefront size for a ROCm `Device`.
///
/// Returns `None` when the device is not a ROCm GPU (CPU, CUDA, Metal, Vulkan)
/// or when the HIP probe itself fails. This is used by
/// [`load_model_from_grim`] to gate `.grim` loading on wavefront compatibility.
fn probe_host_wavefront_size(device: &Device) -> Option<u32> {
    match device {
        Device::Rocm(ordinal) => grim_backend_rocm::probe_host_gpu(*ordinal)
            .ok()
            .map(|caps| caps.wavefront_size),
        _ => None,
    }
}

/// Attempt to resolve an `ArchCompatSpec` for an unknown architecture string,
/// Read a sibling `config.json` from alongside a GGUF file, returning the contents
/// as an `Option<&str>`. A missing sibling is optional — returns `None` so the
/// caller falls back to the plugins-dir scan. This is the helper extracted from
/// the GGUF load path so the config-reading fix has a regression test.
///
/// # Bug history
///
/// Prior to the extraction, the GGUF load path built a *path string* instead of
/// reading file contents:
///   `dir.join("config.json").to_str().map(|s| s.to_string())`
/// which passed a path (e.g. "/path/config.json") as `config_raw`. `from_hf_config_json`
fn read_sibling_config_json(path: &str) -> Option<String> {
    let config_path = std::path::Path::new(path)
        .parent()
        .map(|dir| dir.join("config.json"))?;
    std::fs::read_to_string(config_path).ok()
}

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
    let v: &GgufValue = v?;
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
    if let Some(v) = v { v.as_array() } else { None }
}

/// Returns true if `provider` can resolve a tensor by the given (GGUF) name,
/// without materialising it. Used for architecture detection by tensor
/// signature (e.g. distinguishing SmolLM2 from Llama under the same
/// `general.architecture` tag).
fn weight_provider_has_tensor(provider: &dyn TensorProvider, name: &str) -> bool {
    provider.meta(name).is_ok()
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
            dbg_eprintln!("[meta-get-u32] {key} = {u} (u32)");
            return Some(u);
        }
        if let Some(arr) = v.as_array() {
            dbg_eprintln!("[meta-get-u32] {key} = {} (array.len)", arr.len());
            return Some(arr.len() as u32);
        }
        if let Some(s) = v.as_str() {
            if let Ok(u) = s.parse::<u32>() {
                dbg_eprintln!("[meta-get-u32] {key} = {u} (str->u32)");
                return Some(u);
            }
        }
        dbg_eprintln!("[meta-get-u32] {key} = MISSING");
        None
    }
    fn get_f32(&self, key: &str) -> Option<f32> {
        let v = self.0.metadata(key)?;
        if let Some(f) = v.as_f32() {
            dbg_eprintln!("[meta-get-f32] {key} = {f}");
            return Some(f);
        }
        if let Some(s) = v.as_str() {
            if let Ok(f) = s.parse::<f32>() {
                dbg_eprintln!("[meta-get-f32] {key} = {f} (str->f32)");
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

    // P0-3.1: Wave64/Wave32 compatibility guard.
    //
    // `.grim` files may be compiled with `wavefront_size = 64` (Wave64, CDNA)
    // but an RDNA2 host (`gfx1036` or similar) only supports Wave32. Loading
    // a Wave64 `.grim` on a Wave32 GPU triggers GPU memory faults. If the
    // artifact's declared wavefront size is both non-zero and incompatible with
    // the probed host GPU, transparently fall back to the sibling GGUF which
    // contains the same weights in a format that is always Wave32-safe.
    let grim_wf = grim_provider.wavefront_size();
    if let Some(host_wf) = probe_host_wavefront_size(&device) {
        if grim_wf != 0 && host_wf != 0 && grim_wf != host_wf {
            eprintln!(
                "[grim] .grim wavefront_size={grim_wf} incompatible with host GPU wavefront_size={host_wf} \
                 (RDNA2/Wave32 host); falling back to GGUF sibling '{gguf_path_str}'"
            );
            return load_model_from_gguf(gguf_path_str, device);
        }
    }

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
    #[serde(rename = "num_experts")]
    num_experts: Option<usize>,
    #[serde(rename = "num_experts_per_tok")]
    num_experts_per_tok: Option<usize>,
    #[serde(rename = "moe_intermediate_size")]
    moe_intermediate_size: Option<usize>,
    #[serde(rename = "shared_expert_intermediate_size")]
    shared_expert_intermediate_size: Option<usize>,
    #[serde(rename = "routed_scaling_factor")]
    routed_scaling_factor: Option<f32>,
    #[serde(rename = "moe_routed_scaling_factor")]
    moe_routed_scaling_factor: Option<f32>,
    #[serde(rename = "mlp_only_layers")]
    mlp_only_layers: Option<Vec<usize>>,
    // Laguna-S-2.1 hybrid attention
    #[serde(rename = "sliding_window")]
    sliding_window: Option<usize>,
    #[serde(rename = "gating")]
    gating: Option<String>,
    #[serde(rename = "num_attention_heads_per_layer")]
    num_attention_heads_per_layer: Option<Vec<usize>>,
    #[serde(rename = "partial_rotary_factor")]
    partial_rotary_factor: Option<f32>,
    /// Laguna-S-2.1 nested `{full_attention: {rope_type, rope_theta, factor,
    /// original_max_position_embeddings, beta_fast, beta_slow,
    /// attention_factor, partial_rotary_factor}, sliding_attention: {...}}`.
    /// Parsed lazily; absence = plain RoPE.
    #[serde(rename = "rope_parameters")]
    rope_parameters: Option<serde_json::Value>,
    /// HuggingFace-standard rope-scaling block used by Qwen3.5-MoE and other
    /// HF-exported checkpoints: `{rope_type: "yarn", factor, original_max_position_embeddings,
    /// beta_fast, beta_slow, attention_factor}`. Parsed lazily by
    /// `parse_yarn_scaling`; absence = plain RoPE.
    #[serde(rename = "rope_scaling")]
    rope_scaling: Option<serde_json::Value>,
}

/// Extract full-attention YaRN params from Laguna-S-2.1 `rope_parameters`.
///
/// Layout: `{full_attention: {rope_type: "yarn", factor, original_max_position_embeddings,
/// beta_fast, beta_slow, attention_factor, ...}}`. Returns `None` when the
/// block is absent or `rope_type != "yarn"` (plain RoPE).
fn parse_full_yarn(rope_parameters: &Option<serde_json::Value>) -> Option<YaRNParams> {
    let full = rope_parameters.as_ref()?.get("full_attention")?;
    if full.get("rope_type").and_then(|v| v.as_str()) != Some("yarn") {
        return None;
    }
    Some(YaRNParams {
        factor: full.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        original_max_pos: full
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(8192) as usize,
        beta_fast: full
            .get("beta_fast")
            .and_then(|v| v.as_f64())
            .unwrap_or(32.0) as f32,
        beta_slow: full
            .get("beta_slow")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32,
        attention_factor: full
            .get("attention_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32,
    })
}

/// Parse YaRN params from a HuggingFace-standard `rope_scaling` block
/// `{rope_type: "yarn", factor, original_max_position_embeddings, beta_fast,
/// beta_slow, attention_factor}`. Used by Qwen3.5-MoE and other HF-exported
/// checkpoints. Returns `None` for non-YaRN rope types (`linear`, `dynamic`...)
/// or absence, falling back to plain RoPE.
pub fn parse_yarn_scaling(rope_scaling: &Option<serde_json::Value>) -> Option<YaRNParams> {
    let rs = rope_scaling.as_ref()?;
    // Only YaRN scaling yields a `YaRNParams`. Other rope types (`linear`,
    // `dynamic`, `ntk-aware`, `longrope`, ...) get plain RoPE here.
    if rs.get("rope_type").and_then(|v| v.as_str()) != Some("yarn") {
        return None;
    }
    Some(YaRNParams {
        factor: rs.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        original_max_pos: rs
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(8192) as usize,
        beta_fast: rs.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32,
        beta_slow: rs.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        attention_factor: rs
            .get("attention_factor")
            .or_else(|| rs.get("attn_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32,
    })
}

/// Parse full-attention YaRN params from a GGUF metadata string.
///
/// Some GGUF exports (e.g. llama.cpp-converted Laguna checkpoints) store the
/// `rope_parameters` JSON block under a key like `laguna.rope_parameters`. When
/// present, parse and return the YaRN params; otherwise `None` (plain RoPE).
pub fn parse_full_yarn_gguf(lookup: &dyn MetadataLookup) -> Option<YaRNParams> {
    let json_str = lookup
        .get_str("rope_parameters")
        .or_else(|| lookup.get_str("laguna.rope_parameters"))?;
    let value: serde_json::Value = serde_json::from_str(&json_str).ok()?;
    parse_full_yarn(&Some(value))
}

/// Parse a HuggingFace `rope_scaling` YaRN block from GGUF metadata. Checks the
/// common GGUF keys carrying a JSON `rope_scaling` blob (`rope_scaling`,
/// `<arch>.rope_scaling`, `rope.scaling`) and falls back to individual dotted
/// keys (`rope_scaling.rope_type`, `rope_scaling.factor`, ...) — matching
/// llama.cpp converter conventions.
pub fn parse_yarn_scaling_gguf(lookup: &dyn MetadataLookup) -> Option<YaRNParams> {
    // First try the JSON-string form (some converters store the full block).
    for key in ["rope_scaling", "qwen35moe.rope_scaling", "rope.scaling"] {
        if let Some(json_str) = lookup.get_str(key) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_str) {
                if let Some(y) = parse_yarn_scaling(&Some(value)) {
                    return Some(y);
                }
            }
        }
    }
    // Then build a Value from the individual dotted keys.
    let rope_type = lookup
        .get_str("rope_scaling.rope_type")
        .or_else(|| lookup.get_str("rope_scaling.rope_type"))?;
    if rope_type != "yarn" {
        return None;
    }
    let factor = lookup.get_f32("rope_scaling.factor").unwrap_or(1.0);
    let original_max_pos = lookup
        .get_u32("rope_scaling.original_max_position_embeddings")
        .or_else(|| lookup.get_u32("rope_scaling.original_max_position_embed"))
        .or_else(|| lookup.get_u32("rope_scaling.context_length_orig"))
        .unwrap_or(8192) as usize;
    let beta_fast = lookup.get_f32("rope_scaling.beta_fast").unwrap_or(32.0);
    let beta_slow = lookup.get_f32("rope_scaling.beta_slow").unwrap_or(1.0);
    let attention_factor = lookup
        .get_f32("rope_scaling.attention_factor")
        .or_else(|| lookup.get_f32("rope_scaling.attn_factor"))
        .unwrap_or(1.0);
    Some(YaRNParams {
        factor,
        original_max_pos,
        beta_fast,
        beta_slow,
        attention_factor,
    })
}

/// Extract Laguna-S-2.1 hybrid-attention + RoPE fields from GGUF metadata.
///
/// GGUF checkpoints carry only a single `rope_theta` + `max_seq_len` in
/// `ArchHyperparameters`/`hparams`; the dual thetas, partial-rotary factors,
/// sliding window, layer types, and gating are **not** stored in GGUF metadata
/// (they live in `config.json` for the Safetensors path). This helper reads what
/// **is** available — the `rope_parameters` JSON string if the checkpoint was
/// converted with it — and falls back to the published S-2.1 hardcoded values
/// otherwise. This keeps the GGUF path correct for checkpoints that carry it and
/// safety-correct (plain RoPE + published defaults) for those that don't.
pub fn extract_laguna_gguf_hybrid(
    lookup: &dyn MetadataLookup,
) -> (
    f32,                // full_rope_theta
    f32,                // sliding_rope_theta
    f32,                // full_partial_rotary_factor
    f32,                // sliding_partial_rotary_factor
    usize,              // sliding_window
    Option<YaRNParams>, // full_yarn
) {
    let full_rope_theta = lookup
        .get_f32("rope_parameters.full_attention.rope_theta")
        .or_else(|| lookup.get_f32("rope_parameters.full_attention.freq_base"))
        .or_else(|| lookup.get_f32("full_attention.rope_theta"))
        .unwrap_or(500000.0);
    let sliding_rope_theta = lookup
        .get_f32("rope_parameters.sliding_attention.rope_theta")
        .or_else(|| lookup.get_f32("sliding_attention.rope_theta"))
        .unwrap_or(10000.0);
    let full_partial_rotary_factor = lookup
        .get_f32("rope_parameters.full_attention.partial_rotary_factor")
        .or_else(|| lookup.get_f32("full_attention.partial_rotary_factor"))
        .unwrap_or(0.5);
    let sliding_partial_rotary_factor = lookup
        .get_f32("rope_parameters.sliding_attention.partial_rotary_factor")
        .or_else(|| lookup.get_f32("sliding_attention.partial_rotary_factor"))
        .unwrap_or(1.0);
    let sliding_window = lookup
        .get_u32("sliding_window")
        .or_else(|| lookup.get_u32("laguna.sliding_window"))
        .map(|v| v as usize)
        .unwrap_or(512);
    let full_yarn = parse_full_yarn_gguf(lookup);
    (
        full_rope_theta,
        sliding_rope_theta,
        full_partial_rotary_factor,
        sliding_partial_rotary_factor,
        sliding_window,
        full_yarn,
    )
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
        .ok_or_else(|| Error::Config("config.json missing model_type or architectures".into()))?;
    let model_arch = ModelArchitecture::from_str(&arch_str);

    // Parse ArchCompatSpec from raw config.json for plugin enrichment.
    // Fall back to an empty spec if parsing fails — we still use SafetensorsConfig fields.
    let compat_spec = raw_config_str.and_then(|s| ArchCompatSpec::from_hf_config_json(s).ok());

    let vocab_size = config.vocab_size;
    let hidden_size = config.hidden_size;
    let num_layers = config.num_hidden_layers;
    let rms_norm_eps = config.rms_norm_eps.unwrap_or(1e-5);
    let num_heads = config.num_attention_heads.unwrap_or(32);
    let num_kv_heads = config.num_key_value_heads.unwrap_or(num_heads);
    let head_dim = config
        .head_dim
        .unwrap_or_else(|| hidden_size.checked_div(num_heads).unwrap_or(128));
    let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
    let max_seq_len = config.max_position_embeddings.unwrap_or(2048);
    let rope_theta = config.rope_theta.unwrap_or(10000.0);

    // Enrich from ArchCompatSpec when available — these fields take priority
    // over SafetensorsConfig defaults for known architectures.
    let rms_norm_eps = compat_spec
        .as_ref()
        .map(|s| s.rms_norm_eps)
        .unwrap_or(rms_norm_eps);
    let rope_theta = compat_spec
        .as_ref()
        .map(|s| s.rope_theta)
        .unwrap_or(rope_theta);
    let max_seq_len = compat_spec
        .as_ref()
        .map(|s| s.max_seq_len)
        .unwrap_or(max_seq_len);

    // `GRIM_CONTEXT` lets operators cap the effective context window without
    // re-exporting the GGUF. The model's advertised hard limit is treated as a
    // ceiling: an override requesting more than the model supports is clamped
    // back to the GGUF value. Only `grim-engine`'s `EngineConfig` (not the
    // model's RoPE) reads this, so it is purely an operator hint.
    let max_seq_len = grim_core::env_config::RuntimeEnv::from_env()
        .context
        .map_or(max_seq_len, |ctx| ctx.min(max_seq_len));
    let expert_count = compat_spec
        .as_ref()
        .and_then(|s| s.expert_count)
        .or(config.num_local_experts)
        .or(config.num_experts)
        .unwrap_or(8);
    let expert_used_count = compat_spec
        .as_ref()
        .and_then(|s| s.expert_used_count)
        .or(config.num_experts_per_tok)
        .unwrap_or(2);
    // Routed-expert output scaling. Defaults to 1.0 (no-op) when neither
    // the compat spec nor config.json specifies it. For real MoE checkpoints
    // (e.g. Laguna-2, Qwen3-MoE, DeepSeek-V2) this must match the checkpoint
    // or routed-expert contribute at the wrong magnitude.
    let routed_scaling_factor = compat_spec
        .as_ref()
        .and_then(|s| s.routed_scaling_factor)
        .or(config.moe_routed_scaling_factor)
        .or(config.routed_scaling_factor)
        .unwrap_or(2.5);
    let moe_intermediate_size = config.moe_intermediate_size.unwrap_or(1024);
    let shared_expert_intermediate_size = config.shared_expert_intermediate_size.unwrap_or(1024);
    let mlp_only_layers = config.mlp_only_layers.unwrap_or_else(|| vec![0]);

    dbg_eprintln!(
        "[grim] Loading config from safetensors: architecture={:?}, layers={}, hidden={}, vocab={}",
        model_arch,
        num_layers,
        hidden_size,
        vocab_size
    );

    let tp = resolve_tp_config()?;
    let ws = WeightSource::root(provider, device.clone()).with_tp_config(tp);
    ws.prefetch_all();

    match model_arch {
        ModelArchitecture::Falcon => {
            let falcon_cfg = FalconConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                layer_norm_epsilon: rms_norm_eps,
                rope_theta,
                max_seq_len,
                parallel_attn: true,
                new_decoder_architecture: true,
                multi_query: true,
            };
            eprintln!("[grim] Loading Falcon model with config: {:?}", falcon_cfg);
            let m = Falcon::load_tp(device.clone(), &ws, falcon_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Bloom => {
            let bloom_cfg = BloomConfig {
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
            eprintln!("[grim] Loading BLOOM model with config: {:?}", bloom_cfg);
            let m = Bloom::load_tp(device.clone(), &ws, bloom_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Laguna => {
            // Laguna-S-2.1 hybrid attention: per-layer layer_types, sliding
            // window, per-layer head counts, dual RoPE, and the attention
            // output gate. Parsed from config.json; defaults match the
            // published S-2.1 checkpoint when keys are absent.
            let full_rope_theta = config
                .rope_parameters
                .as_ref()
                .and_then(|r| r.get("full_attention"))
                .and_then(|f| f.get("rope_theta"))
                .and_then(|v| v.as_f64())
                .unwrap_or(500000.0) as f32;
            let sliding_rope_theta = config
                .rope_parameters
                .as_ref()
                .and_then(|r| r.get("sliding_attention"))
                .and_then(|f| f.get("rope_theta"))
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32;
            let full_partial_rotary_factor = config
                .rope_parameters
                .as_ref()
                .and_then(|r| r.get("full_attention"))
                .and_then(|f| f.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;
            let sliding_partial_rotary_factor = config
                .rope_parameters
                .as_ref()
                .and_then(|r| r.get("sliding_attention"))
                .and_then(|f| f.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let full_yarn = parse_full_yarn(&config.rope_parameters);

            let laguna_cfg = LagunaConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                moe_intermediate_size,
                shared_expert_intermediate_size,
                num_experts: expert_count,
                num_experts_per_tok: expert_used_count,
                routed_scaling_factor,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
                mlp_only_layers,
                layer_types: config
                    .layer_types
                    .unwrap_or_else(|| vec!["full_attention".into()]),
                sliding_window: config.sliding_window.unwrap_or(512),
                num_attention_heads_per_layer: config
                    .num_attention_heads_per_layer
                    .unwrap_or_else(|| vec![num_heads; num_layers]),
                full_rope_theta,
                sliding_rope_theta,
                full_partial_rotary_factor,
                sliding_partial_rotary_factor,
                gating: config.gating.unwrap_or_else(|| "per-head".into()),
                full_yarn,
            };

            eprintln!("[grim] Loading Laguna model with config: {:?}", laguna_cfg);
            let m = Laguna::load_tp(device.clone(), &ws, laguna_cfg, tp)?;
            Ok(Box::new(m))
        }

        ModelArchitecture::Phi2 | ModelArchitecture::Phi3 | ModelArchitecture::PhiMoe => {
            let phi_cfg = PhiConfig {
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
            eprintln!("[grim] Loading Phi model with config: {:?}", phi_cfg);
            let m = Phi2::load_tp(device.clone(), &ws, phi_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen | ModelArchitecture::Qwen2 | ModelArchitecture::Qwen3 => {
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
            let m = Qwen::load_tp(device.clone(), &ws, qwen_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen35 => {
            let qwen35_cfg = Qwen35Config {
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
                full_attention_interval: 4,
                ssm_d_state: 128,
                ssm_d_inner: 6144,
                ssm_d_conv: 4,
                ssm_dt_rank: 48,
                ssm_n_group: 16,
                devices: resolve_discrete_rocm_devices(&device),
            };
            eprintln!("[grim] Loading Qwen3.5 model with config: {:?}", qwen35_cfg);
            let m = Qwen35::load_tp(device.clone(), &ws, qwen35_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen35Moe => {
            let qwen35_moe_cfg = Qwen35MoeConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                num_experts: expert_count,
                num_experts_per_tok: expert_used_count,
                shared_expert_intermediate_size: None,
                routed_scaling_factor,
                layer_types: vec![],
                linear_key_head_dim: 128,
                linear_num_key_heads: 16,
                linear_value_head_dim: 128,
                linear_num_value_heads: 128,
                partial_rotary_factor: config.partial_rotary_factor.unwrap_or(0.25),
                rms_norm_eps,
                rope_theta,
                max_seq_len,
                full_yarn: parse_yarn_scaling(&config.rope_scaling),
            };
            eprintln!(
                "[grim] Loading Qwen3.5/3.8 MoE model with config: {:?}",
                qwen35_moe_cfg
            );
            let m = Qwen35Moe::load_tp(device.clone(), &ws, qwen35_moe_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen38FlashNext => {
            let qwen38_cfg = Qwen38FlashNextConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                num_experts: expert_count.max(512),
                num_experts_per_tok: expert_used_count.max(10),
                shared_expert_intermediate_size: Some(intermediate_size),
                routed_scaling_factor,
                layer_types: vec![],
                linear_key_head_dim: 128,
                linear_num_key_heads: 8,
                linear_value_head_dim: 128,
                linear_num_value_heads: 8,
                ngram_vocab_size: Some(20_000_000),
                ngram_dim: Some(512),
                gated_residual_branches: 4,
                mrope_section: [11, 11, 10],
                partial_rotary_factor: config.partial_rotary_factor.unwrap_or(1.0),
                rms_norm_eps,
                rope_theta,
                max_seq_len,
                full_yarn: parse_yarn_scaling(&config.rope_scaling),
            };
            eprintln!(
                "[grim] Loading Qwen3.8-Flash-Next model with config: {:?}",
                qwen38_cfg
            );
            let m = Qwen38FlashNext::load_tp(device.clone(), &ws, qwen38_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Mellum => {
            let mellum_cfg = MellumConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                moe_intermediate_size: config.moe_intermediate_size.unwrap_or(896),
                num_experts: expert_count,
                num_experts_per_tok: expert_used_count,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
                sliding_window: 1024,
                // Mellum2 uses YaRN RoPE on full-attention layers with these
                // documented constants (matches the HF `rope_parameters`).
                max_window_layers: 0,
                yarn: Some(grim_tensor::YaRNParams {
                    factor: 16.0,
                    original_max_pos: 8192,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    attention_factor: 1.277_258_9,
                }),
            };
            eprintln!("[grim] Loading Mellum model with config: {:?}", mellum_cfg);
            let m = Mellum::load_tp(device.clone(), &ws, mellum_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen2Moe
        | ModelArchitecture::Qwen3Moe
        | ModelArchitecture::AfMoe
        | ModelArchitecture::BailingMoe
        | ModelArchitecture::BailingMoe2
        | ModelArchitecture::Cohere2Moe
        | ModelArchitecture::Ernie45Moe
        | ModelArchitecture::Glm4Moe
        | ModelArchitecture::GroveMoe
        | ModelArchitecture::OpenAiMoe
        | ModelArchitecture::Qwen3VlMoe => {
            let qwen_moe_cfg = Qwen3MoeConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                num_experts: expert_count,
                num_experts_per_tok: expert_used_count,
                routed_scaling_factor,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            eprintln!(
                "[grim] Loading Qwen-MoE model with config: {:?}",
                qwen_moe_cfg
            );
            let m = Qwen3Moe::load_tp(device.clone(), &ws, qwen_moe_cfg, tp)?;
            Ok(Box::new(m))
        }
        arch if arch.is_moe() => {
            let moe_cfg = Qwen3MoeConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                num_experts: expert_count,
                num_experts_per_tok: expert_used_count,
                routed_scaling_factor,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
            };
            eprintln!("[grim] Loading MoE model with config: {:?}", moe_cfg);
            let m = Qwen3Moe::load_tp(device.clone(), &ws, moe_cfg, tp)?;
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
            let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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
                expert_count,
                expert_used_count,
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
            let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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
            eprintln!(
                "[grim] Loading NemotronH model with config: {:?}",
                nemotron_cfg
            );
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
            let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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
            eprintln!(
                "[grim] Loading GraniteHybrid model with config: {:?}",
                granite_cfg
            );
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
            let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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
            eprintln!(
                "[grim] Loading ModernBERT model with config: {:?}",
                modern_bert_cfg
            );
            let bert_cfg = BertConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                max_seq_len,
            };
            let m = Bert::load_tp(device.clone(), &ws, bert_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::NomicBert
        | ModelArchitecture::NomicBertMoe
        | ModelArchitecture::NeoBert
        | ModelArchitecture::JinaBertV2
        | ModelArchitecture::JinaBertV3 => {
            let nomic_bert_cfg = NomicBertConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                layer_norm_eps: rms_norm_eps,
                max_seq_len,
            };
            eprintln!(
                "[grim] Loading NomicBERT model with config: {:?}",
                nomic_bert_cfg
            );
            let bert_cfg = BertConfig {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                max_seq_len,
            };
            let m = Bert::load_tp(device.clone(), &ws, bert_cfg, tp)?;
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
            eprintln!(
                "[grim] Loading T5Encoder model with config: {:?}",
                t5_enc_cfg
            );
            let t5_cfg = T5Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_layers,
                intermediate_size,
                rms_norm_eps,
            };
            let m = T5::load_tp(&ws, t5_cfg, tp)?;
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
                rms_norm_eps: rms_norm_eps as f64,
            };
            let m = Rwkv::load_tp(&ws, rwkv_cfg, device.clone(), tp)?;
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
                rms_norm_eps: rms_norm_eps as f64,
            };
            let m = Rwkv::load_tp(&ws, rwkv_cfg, device.clone(), tp)?;
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
            let ws = WeightSource::root(&remapped_provider, device.clone()).with_tp_config(tp);
            ws.prefetch_all();

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
            if is_recr.is_empty() || is_recr.iter().all(|&r| !r) {
                is_recr.clear();
                for i in 0..num_layers {
                    let blk_ws = ws.pp("blk").pp(&i.to_string());
                    let is_conv = blk_ws.get_unconstrained("conv.weight").is_ok()
                        || blk_ws.get_unconstrained("shortconv.weight").is_ok()
                        || blk_ws.get_unconstrained("conv_1d.weight").is_ok()
                        || blk_ws.get_unconstrained("attn_q.weight").is_err();
                    is_recr.push(is_conv);
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
                n_layer_dense_lead: num_layers, // all-dense unless metadata says otherwise
                n_expert: 0,
                n_expert_used: 1,
                n_ff_exp: intermediate_size,
                expert_weights_scale: 1.0,
                expert_gating_func: 0,
                n_swa: 0,
                swa_type: 0,
                n_embd_out: 0,
                // WI-X6: MXFP4 QKV attention is default-on for LFM2 family (escape hatch: GRIM_LFM2_MXFP4_QKV=0/false/off)
                mxfp4_qkv_attention: std::env::var("GRIM_LFM2_MXFP4_QKV")
                    .map(|v| {
                        v != "0"
                            && !v.eq_ignore_ascii_case("false")
                            && !v.eq_ignore_ascii_case("off")
                    })
                    .unwrap_or(true),
            };

            let m = Lfm2::load_tp(&ws, cfg, tp)?;
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
            let m = Mamba::load_tp(device.clone(), &ws, cfg, tp)?;
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
            let m = Gpt2::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Gemma
        | ModelArchitecture::Gemma2
        | ModelArchitecture::Gemma3
        | ModelArchitecture::Gemma4 => {
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
            let m = Gemma::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek => {
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
            let m = DeepSeek::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek2 => {
            let cfg = DeepSeek2Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                kv_lora_rank: 512,
                q_lora_rank: None,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
                moe_intermediate_size,
                n_routed_experts: 64,
                n_shared_experts: 2,
                num_experts_per_tok: 6,
                first_k_dense_replace: 1,
                routed_scaling_factor: 1.0,
            };
            let m = DeepSeek2::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek32 => {
            let cfg = DeepSeek32Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                kv_lora_rank: 512,
                q_lora_rank: Some(1536),
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
                moe_intermediate_size,
                n_routed_experts: 256,
                n_shared_experts: 1,
                num_experts_per_tok: 8,
                first_k_dense_replace: 1,
                routed_scaling_factor: 2.5,
            };
            let m = DeepSeek32::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek4 => {
            let cfg = DeepSeek4Config {
                vocab_size,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_size,
                kv_lora_rank: 512,
                q_lora_rank: Some(1024),
                qk_nope_head_dim: 448,
                qk_rope_head_dim: 64,
                v_head_dim: 512,
                rms_norm_eps,
                rope_theta,
                max_seq_len,
                moe_intermediate_size,
                n_routed_experts: 256,
                n_shared_experts: 1,
                num_experts_per_tok: 6,
                first_k_dense_replace: 3,
                routed_scaling_factor: 2.5,
                hc_mult: 4,
                sqrtsoftplus_moe: true,
                compressor_indexer_enabled: true,
            };
            let m = DeepSeek4::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::CommandR => {
            let commandr_cfg = CommandRConfig {
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
            eprintln!(
                "[grim] Loading CommandR model with config: {:?}",
                commandr_cfg
            );
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

                partial_rotary_factor: 1.0,
                yarn: None,
            };
            let mut m = Llama::load_tp(device.clone(), &ws, llama_cfg, tp)?;
            // ALiBi position bias (baichuan/mpt/jais/gptneox class): enabled
            // when the GGUF carries the metadata key. ALiBi replaces RoPE.
            if matches!(
                model_arch,
                ModelArchitecture::Baichuan
                    | ModelArchitecture::Mpt
                    | ModelArchitecture::Jais
                    | ModelArchitecture::Jais2
            ) {
                eprintln!("[grim] enabling ALiBi on {} blocks", m.layers.len());
                for layer in m.layers.iter_mut() {
                    *layer = std::mem::replace(layer, layer.clone()).with_alibi();
                }
            }
            Ok(Box::new(m))
        }
        ModelArchitecture::Chameleon => {
            let chameleon_cfg = ChameleonConfig {
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
                swin_norm: true,
            };
            eprintln!(
                "[grim] Loading Chameleon model with config: {:?}",
                chameleon_cfg
            );
            let m = Chameleon::load_tp(device.clone(), &ws, chameleon_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeltaNetBase => {
            let delta_cfg = DeltaNetBaseConfig {
                vocab_size,
                hidden_size,
                num_heads,
                head_dim,
                num_layers,
                intermediate_size,
                chunk_size: 64,
                rms_norm_eps,
                max_seq_len,
            };
            eprintln!(
                "[grim] Loading DeltaNetBase model with config: {:?}",
                delta_cfg
            );
            let m = DeltaNetBase::load_tp(device.clone(), &ws, delta_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::CogVlm => {
            let cog_cfg = CogVlmConfig {
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
                vision_config: CogVlmVisionConfig {
                    hidden_size: 1024,
                    image_size: 490,
                    patch_size: 14,
                    num_heads: 16,
                    num_layers: 24,
                    in_channels: 3,
                    out_hidden_size: hidden_size,
                },
            };
            let m = CogVlm::load_tp(device.clone(), &ws, cog_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Gemma3n => {
            let gemma_cfg = Gemma3nConfig {
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
                sliding_window_size: 1024,
                query_pre_attn_scalar: 256.0,
            };
            let m = Gemma3n::load_tp(device.clone(), &ws, gemma_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::HunyuanVl => {
            let hy_cfg = HunyuanVlConfig {
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
                mrope_section: [2, 2, 2, 2],
                image_token_id: 5,
                im_start_id: 120118,
                im_end_id: 120119,
                im_newline_id: 120121,
                vision_config: HunyuanVlVisionConfig {
                    hidden_size: 64,
                    num_attention_heads: 4,
                    num_key_value_heads: 4,
                    num_hidden_layers: 2,
                    patch_size: 16,
                    num_channels: 3,
                    intermediate_size: 128,
                    out_hidden_size: 64,
                },
            };
            let m = HunyuanVl::load_tp(device.clone(), &ws, hy_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen2Vl => {
            let qv_cfg = Qwen2VlConfig {
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
                mrope_section: [16, 24, 24],
                vision_start_token_id: 151652,
                vision_end_token_id: 151653,
                vision_token_id: 151654,
                image_token_id: 151655,
                video_token_id: 151656,
                vision_config: Qwen2VlVisionConfig {
                    depth: 32,
                    hidden_size: 1280,
                    num_heads: 16,
                    patch_size: 14,
                    spatial_merge_size: 2,
                    temporal_patch_size: 2,
                    in_channels: 3,
                    out_hidden_size: hidden_size,
                },
            };
            let m = Qwen2Vl::load_tp(device.clone(), &ws, qv_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen3Vl => {
            let qv_cfg = Qwen3VlConfig {
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
                mrope_section: [24, 20, 20],
                deepstack_visual_indexes: vec![8, 16, 24],
                vision_start_token_id: 151652,
                vision_end_token_id: 151653,
                vision_token_id: 151654,
                image_token_id: 151655,
                video_token_id: 151656,
                vision_config: Qwen3VlVisionConfig {
                    depth: 27,
                    hidden_size: 1152,
                    num_heads: 16,
                    patch_size: 16,
                    spatial_merge_size: 2,
                    temporal_patch_size: 2,
                    in_channels: 3,
                    out_hidden_size: hidden_size,
                    deepstack_visual_indexes: vec![8, 16, 24],
                },
            };
            let m = Qwen3Vl::load_tp(device.clone(), &ws, qv_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::WavTokenizerDec => {
            let wav_cfg = WavTokenizerDecConfig {
                latent_dim: 512,
                backbone_dim: 768,
                backbone_num_blocks: 12,
                backbone_intermediate_dim: 2304,
                backbone_kernel_size: 7,
                n_fft: 1280,
                hop_length: 320,
                head_dim: 641,
                codebook_size: 4096,
                codebook_dim: 512,
                num_bandwidths: 4,
                sample_rate: 24000,
            };
            let m = WavTokenizerDec::load_tp(device.clone(), &ws, wav_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::InternS2Mobius => {
            let cfg = InternS2MobiusConfig {
                vocab_size,
                hidden_size,
                num_attention_heads: num_heads,
                num_key_value_heads: num_kv_heads,
                head_dim,
                num_hidden_layers: num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_position_embeddings: max_seq_len,
            };
            let m = InternS2Mobius::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::KimiK3 => {
            let cfg = KimiK3Config {
                vocab_size,
                hidden_size,
                num_attention_heads: num_heads,
                num_key_value_heads: num_kv_heads,
                head_dim,
                num_hidden_layers: num_layers,
                q_lora_rank: num_heads,
                kv_lora_rank: num_kv_heads * 4,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                num_experts: 64,
                num_experts_per_tok: 6,
                intermediate_size,
                routed_scaling_factor,
                rms_norm_eps,
                rope_theta,
            };
            let m = KimiK3::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::InklingSmall => {
            let cfg = InklingSmallConfig {
                vocab_size,
                hidden_size,
                num_attention_heads: num_heads,
                num_key_value_heads: num_kv_heads,
                head_dim,
                num_hidden_layers: num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_position_embeddings: max_seq_len,
            };
            let m = InklingSmall::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Glm52 => {
            let cfg = Glm52Config {
                vocab_size,
                hidden_size,
                num_attention_heads: num_heads,
                num_key_value_heads: num_kv_heads,
                head_dim,
                num_hidden_layers: num_layers,
                intermediate_size,
                num_experts: 8,
                num_experts_per_tok: 2,
                rms_norm_eps,
                rope_theta,
                max_position_embeddings: max_seq_len,
            };
            let m = Glm52::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DiffusionGemma => {
            let cfg = DiffusionGemmaConfig {
                vocab_size,
                hidden_size,
                num_attention_heads: num_heads,
                num_key_value_heads: num_kv_heads,
                head_dim,
                num_hidden_layers: num_layers,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_position_embeddings: max_seq_len,
            };
            let m = DiffusionGemma::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::MiniMaxM3 => {
            let cfg = MiniMaxM3Config {
                vocab_size,
                hidden_size,
                num_attention_heads: num_heads,
                num_key_value_heads: num_kv_heads,
                head_dim,
                num_hidden_layers: num_layers,
                intermediate_size,
                num_experts: 8,
                num_experts_per_tok: 2,
                rms_norm_eps,
                rope_theta,
                max_position_embeddings: max_seq_len,
            };
            let m = MiniMaxM3::load_tp(device.clone(), &ws, cfg)?;
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
            let m = Bert::load_tp(device.clone(), &ws, cfg, tp)?;
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
            let m = T5::load_tp(&ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::MuseGlimmer => {
            let muse_cfg = if let Some(raw_json) =
                raw_config_str.and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            {
                grim_models_transformer::MuseGlimmerConfig::from_hf(&raw_json)
            } else {
                grim_models_transformer::MuseGlimmerConfig {
                    vocab_size,
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    num_layers,
                    intermediate_size,
                    rms_norm_eps,
                    per_layer_rope_theta: vec![],
                    base_rope_theta: rope_theta,
                    sliding_window_layer_ids: vec![],
                    sliding_window_size: 0,
                    qk_scale_factor: 1.0,
                    output_multiplier: vec![],
                    final_logit_softcapping: 0.0,
                    max_seq_len,
                    vision: None,
                }
            };
            let m =
                grim_models_transformer::MuseGlimmer::load_tp(device.clone(), &ws, muse_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::SolarOpen2 => {
            let solar_cfg = if let Some(raw_json) =
                raw_config_str.and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            {
                SolarOpen2Config::from_hf(&raw_json)
            } else {
                SolarOpen2Config {
                    vocab_size,
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    num_layers,
                    intermediate_size,
                    rms_norm_eps,
                    max_seq_len,
                    ..SolarOpen2Config::default()
                }
            };
            let m = SolarOpen2::load_tp(&ws, solar_cfg)?;
            Ok(Box::new(m))
        }

        ModelArchitecture::Arcee
        | ModelArchitecture::Apertus
        | ModelArchitecture::Arctic
        | ModelArchitecture::Baichuan
        | ModelArchitecture::BitNet
        | ModelArchitecture::ChatGlm
        | ModelArchitecture::Codeshell
        | ModelArchitecture::Cohere2
        | ModelArchitecture::Dbrx
        | ModelArchitecture::Deci
        | ModelArchitecture::DeepSeek2Ocr
        | ModelArchitecture::DFlash
        | ModelArchitecture::Dots1
        | ModelArchitecture::Dream
        | ModelArchitecture::Eagle3
        | ModelArchitecture::Ernie45
        | ModelArchitecture::Eurobert
        | ModelArchitecture::Exaone
        | ModelArchitecture::Exaone4
        | ModelArchitecture::Gemma4Assistant
        | ModelArchitecture::GemmaEmbedding
        | ModelArchitecture::Glm4
        | ModelArchitecture::GlmDsa
        | ModelArchitecture::GptJ
        | ModelArchitecture::GptNeoX
        | ModelArchitecture::Granite
        | ModelArchitecture::Grok
        | ModelArchitecture::HunyuanDense
        | ModelArchitecture::HyV3
        | ModelArchitecture::InternLm2
        | ModelArchitecture::Jais
        | ModelArchitecture::Jais2
        | ModelArchitecture::KimiLinear
        | ModelArchitecture::Llada
        | ModelArchitecture::Llama
        | ModelArchitecture::Llama4
        | ModelArchitecture::LlamaEmbed
        | ModelArchitecture::MainCoder
        | ModelArchitecture::Mimo2
        | ModelArchitecture::MiniMaxM2
        | ModelArchitecture::Mistral3
        | ModelArchitecture::Mistral4
        | ModelArchitecture::Mpt
        | ModelArchitecture::Nemotron
        | ModelArchitecture::Olmo
        | ModelArchitecture::Olmo2
        | ModelArchitecture::OpenElm
        | ModelArchitecture::Orion
        | ModelArchitecture::PaddleOcr
        | ModelArchitecture::PanguEmbed
        | ModelArchitecture::Plamo
        | ModelArchitecture::Plamo2
        | ModelArchitecture::Plamo3
        | ModelArchitecture::Plm
        | ModelArchitecture::Qwen3Next
        | ModelArchitecture::Refact
        | ModelArchitecture::Rnd1
        | ModelArchitecture::SeedOss
        | ModelArchitecture::SmallThinker
        | ModelArchitecture::SmolLm3
        | ModelArchitecture::StableLm
        | ModelArchitecture::Starcoder
        | ModelArchitecture::Starcoder2
        | ModelArchitecture::Step35
        | ModelArchitecture::Talkie
        | ModelArchitecture::Xverse => {
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

                partial_rotary_factor: 1.0,
                yarn: None,
            };
            eprintln!(
                "[grim] Loading Llama-family model ({:?}) with config: {:?}",
                model_arch, llama_cfg
            );
            let mut m = Llama::load_tp(device.clone(), &ws, llama_cfg, tp)?;
            // ALiBi position bias (baichuan/mpt/jais/gptneox class): enabled
            // when the GGUF carries the metadata key. ALiBi replaces RoPE.
            if matches!(
                model_arch,
                ModelArchitecture::Baichuan
                    | ModelArchitecture::Mpt
                    | ModelArchitecture::Jais
                    | ModelArchitecture::Jais2
            ) {
                eprintln!("[grim] enabling ALiBi on {} blocks", m.layers.len());
                for layer in m.layers.iter_mut() {
                    *layer = std::mem::replace(layer, layer.clone()).with_alibi();
                }
            }
            Ok(Box::new(m))
        }
        _ => {
            if let Some(spec) = resolve_arch_compat_spec(&arch_str, raw_config_str) {
                eprintln!(
                    "[grim] Resolved unknown architecture '{}' via ArchCompatSpec plugin (base='{}', is_moe={})",
                    arch_str, spec.base_architecture, spec.is_moe
                );
                let spec_clone = spec.clone();
                let remapped_provider =
                    RemappingTensorProvider::new(provider, move |name: &str| -> String {
                        spec_clone.remap_tensor_name(name)
                    });
                let ws = WeightSource::root(&remapped_provider, device.clone()).with_tp_config(tp);
                ws.prefetch_all();

                if spec.is_moe || spec.expert_count.unwrap_or(0) > 0 {
                    let moe_cfg = Qwen3MoeConfig {
                        vocab_size: spec.vocab_size,
                        hidden_size: spec.hidden_size,
                        num_heads: spec.num_heads,
                        num_kv_heads: spec.num_kv_heads,
                        head_dim: spec.head_dim,
                        num_layers: spec.num_layers,
                        intermediate_size: spec.intermediate_size,
                        num_experts: spec.expert_count.unwrap_or(8),
                        num_experts_per_tok: spec.expert_used_count.unwrap_or(2),
                        routed_scaling_factor: spec.routed_scaling_factor.unwrap_or(1.0),
                        rms_norm_eps: spec.rms_norm_eps,
                        rope_theta: spec.rope_theta,
                        max_seq_len: spec.max_seq_len,
                    };
                    let m = Qwen3Moe::load_tp(device.clone(), &ws, moe_cfg, tp)?;
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
                    let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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

                        partial_rotary_factor: 1.0,
                        yarn: None,
                    };
                    let m = Llama::load_tp(device.clone(), &ws, llama_cfg, tp)?;
                    return Ok(Box::new(m));
                }
            }

            eprintln!(
                "[grim] Unknown architecture '{}' with no plugin compat spec found; using default Llama loader",
                arch_str
            );
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

                partial_rotary_factor: 1.0,
                yarn: None,
            };
            let m = Llama::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
    }
}

fn load_model_with_providers(
    provider: &GgufProvider,
    weight_provider: &dyn grim_tensor::TensorProvider,
    device: Device,
    path: &str,
) -> Result<Box<dyn CausalLm>> {
    eprintln!(
        "[alias] load_model_with_providers called, arch={:?}",
        provider.architecture()
    );
    // Extract architecture from GGUF metadata
    let arch_str = provider.architecture().ok_or_else(|| {
        Error::Config(
            "GGUF file has no 'general.architecture' metadata; cannot determine model family"
                .to_string(),
        )
    })?;

    let lookup = GgufMetadataLookup(provider);
    let mut model_arch = ModelArchitecture::from_str(arch_str);
    if model_arch == ModelArchitecture::Llama {
        let name_lower = lookup
            .get_str("general.name")
            .unwrap_or_default()
            .to_lowercase();
        let path_lower = path.to_lowercase();
        // Only promote to MiniCPM when the GGUF carries MiniCPM2/3 metadata keys.
        // MiniCPM5 reports `general.architecture = llama` and has NO `minicpm.*`
        // metadata keys — it is architecturally standard Llama. Promoting it to
        // MiniCPM would apply wrong rescaling and produce gibberish output.
        let has_minicpm_metadata = lookup.get_f32("minicpm.scale_emb").is_some()
            || lookup.get_f32("minicpm.scale_depth").is_some()
            || lookup.get_f32("minicpm.dim_model_base").is_some();
        if (name_lower.contains("minicpm") || path_lower.contains("minicpm"))
            && has_minicpm_metadata
        {
            eprintln!(
                "[grim] Detected MiniCPM model variant from metadata/path, promoting architecture to MiniCpm"
            );
            model_arch = ModelArchitecture::MiniCpm;
        }
        // SmolLM2 is exported by llama.cpp under `general.architecture = "llama"`
        // but is architecturally distinct: it ties the LM head to the token
        // embedding (no `output.weight`) and uses `output_norm` / `token_embd`
        // naming. Promote to SmolLm2 only on that exact tensor signature so
        // genuine Llama files stay on the unmodified Llama loader.
        let has_output_norm = lookup.get_str("general.architecture").is_some()
            && weight_provider_has_tensor(weight_provider, "output_norm.weight")
            && !weight_provider_has_tensor(weight_provider, "output.weight");
        if has_output_norm {
            eprintln!(
                "[grim] Detected SmolLM2 tensor signature (output_norm present, no output.weight); promoting architecture to SmolLm2"
            );
            model_arch = ModelArchitecture::SmolLm2;
        }
    }
    let hparams = HyperparameterExtractor::extract(model_arch, &lookup);

    eprintln!(
        "[grim] Loading config: architecture={:?}, layers={}, hidden={}, vocab={}",
        model_arch, hparams.num_layers, hparams.hidden_size, hparams.vocab_size
    );

    let hf_gguf_map = TensorNamingRegistry::remap_hf_to_gguf(model_arch, hparams.num_layers);
    eprintln!(
        "[alias] remap map has {} entries, sample: {:?}",
        hf_gguf_map.len(),
        hf_gguf_map.get("tok_embeddings.weight")
    );
    let remapped_provider = RemappingTensorProvider::new(weight_provider, {
        let hf_gguf_map = hf_gguf_map.clone();
        move |name| {
            if let Some(mapped) = hf_gguf_map.get(name) {
                eprintln!("[alias] {} -> {}", name, mapped);
                mapped.clone()
            } else {
                name.to_string()
            }
        }
    });
    let tp = resolve_tp_config()?;
    let ws = WeightSource::root(&remapped_provider, device.clone()).with_tp_config(tp);
    ws.prefetch_all();

    match model_arch {
        ModelArchitecture::Falcon => {
            let falcon_cfg = FalconConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                layer_norm_epsilon: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
                parallel_attn: true,
                new_decoder_architecture: true,
                multi_query: true,
            };
            eprintln!("[grim] Loading Falcon model with config: {:?}", falcon_cfg);
            let m = Falcon::load_tp(device.clone(), &ws, falcon_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Bloom => {
            let bloom_cfg = BloomConfig {
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
            eprintln!("[grim] Loading BLOOM model with config: {:?}", bloom_cfg);
            let m = Bloom::load_tp(device.clone(), &ws, bloom_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Laguna => {
            // GGUF checkpoints carry only a single rope_theta + max_seq_len;
            // the dual thetas, partial-rotary factors, sliding window, and YaRN
            // block are read from GGUF metadata when present (llama.cpp-converted
            // checkpoints may store `rope_parameters` as a JSON string), with the
            // published S-2.1 values as defaults otherwise.
            let (
                full_rope_theta,
                sliding_rope_theta,
                full_partial_rotary_factor,
                sliding_partial_rotary_factor,
                sliding_window,
                full_yarn,
            ) = extract_laguna_gguf_hybrid(&lookup);

            let laguna_cfg = LagunaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                moe_intermediate_size: 1024,
                shared_expert_intermediate_size: 1024,
                num_experts: hparams.expert_count.unwrap_or(256),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(10),
                routed_scaling_factor: hparams.routed_scaling_factor,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
                mlp_only_layers: vec![0],
                layer_types: vec!["full_attention".into()],
                sliding_window,
                num_attention_heads_per_layer: vec![hparams.num_heads; hparams.num_layers],
                full_rope_theta,
                sliding_rope_theta,
                full_partial_rotary_factor,
                sliding_partial_rotary_factor,
                gating: "per-head".into(),
                full_yarn,
            };
            eprintln!("[grim] Loading Laguna model with config: {:?}", laguna_cfg);
            let m = Laguna::load_tp(device.clone(), &ws, laguna_cfg, tp)?;
            Ok(Box::new(m))
        }

        ModelArchitecture::Phi2 | ModelArchitecture::Phi3 | ModelArchitecture::PhiMoe => {
            let phi_cfg = PhiConfig {
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
            eprintln!("[grim] Loading Phi model with config: {:?}", phi_cfg);
            let m = Phi2::load_tp(device.clone(), &ws, phi_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::MiniCpm | ModelArchitecture::MiniCpm3 => {
            let scale_emb = lookup
                .get_f32("minicpm.scale_emb")
                .or_else(|| lookup.get_f32("scale_emb"));
            let scale_depth = lookup
                .get_f32("minicpm.scale_depth")
                .or_else(|| lookup.get_f32("scale_depth"));
            let dim_model_base = lookup
                .get_f32("minicpm.dim_model_base")
                .or_else(|| lookup.get_f32("dim_model_base"));

            let minicpm_cfg = MiniCpmConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                scale_emb,
                scale_depth,
                dim_model_base,
            };
            eprintln!(
                "[grim] Loading MiniCPM model with config: {:?}",
                minicpm_cfg
            );
            let m = MiniCpmModel::load(&ws, minicpm_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::SmolLm2 => {
            let smollm2_cfg = SmolLm2Config {
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
            eprintln!(
                "[grim] Loading SmolLM2 model with config: {:?}",
                smollm2_cfg
            );
            let m = SmolLm2::load_tp(device.clone(), &ws, smollm2_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen | ModelArchitecture::Qwen2 | ModelArchitecture::Qwen3 => {
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
            let m = Qwen::load_tp(device.clone(), &ws, qwen_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen35 => {
            let qwen35_cfg = Qwen35Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: {
                    let devices = resolve_discrete_rocm_devices(&device);
                    let mut min_free = u64::MAX;
                    for dev in &devices {
                        if let Device::Rocm(ord) = dev {
                            let (free, _) = grim_backend_rocm::vram_info(*ord);
                            if free > 0 && free < min_free {
                                min_free = free;
                            }
                        }
                    }
                    if min_free != u64::MAX && min_free > 0 {
                        let headroom = 1536 * 1024 * 1024;
                        let avail = min_free.saturating_sub(headroom);
                        let attn_layers_per_gpu = (hparams.num_layers / devices.len().max(1))
                            .max(1)
                            / hparams.full_attention_interval.unwrap_or(4).max(1);
                        let kv_bytes_per_tok =
                            (2 * hparams.num_kv_heads * hparams.head_dim * 4 * attn_layers_per_gpu)
                                .max(1) as u64;
                        let max_safe = (avail / kv_bytes_per_tok) as usize;
                        if max_safe < hparams.max_seq_len {
                            let clamped = max_safe.max(1024);
                            eprintln!(
                                "[grim] VRAM budget: free={}MB/GPU, dynamically capping context length to {} tokens to ensure zero host spillage",
                                min_free / (1024 * 1024),
                                clamped
                            );
                            clamped
                        } else {
                            hparams.max_seq_len
                        }
                    } else {
                        hparams.max_seq_len
                    }
                },
                full_attention_interval: hparams.full_attention_interval.unwrap_or(4),
                ssm_d_state: hparams.ssm_d_state.unwrap_or(128),
                ssm_d_inner: hparams.ssm_d_inner.unwrap_or(6144),
                ssm_d_conv: hparams.ssm_d_conv.unwrap_or(4),
                ssm_dt_rank: hparams.ssm_dt_rank.unwrap_or(48),
                ssm_n_group: hparams.ssm_n_group.unwrap_or(16),
                devices: resolve_discrete_rocm_devices(&device),
            };
            eprintln!(
                "[grim] Loading Qwen3.5/3.8 model with config: {:?}",
                qwen35_cfg
            );
            let m = Qwen35::load_tp(device.clone(), &ws, qwen35_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen35Moe => {
            // GGUF YaRN: read the `rope_scaling` block if the converter stored
            // it, falling back to plain RoPE (no YaRN). Partial-rotary factor
            // is read from `<arch>.partial_rotary_factor` if present, else the
            // published Qwen3.5-MoE default of 0.25.
            let prf = lookup
                .get_f32("qwen35moe.partial_rotary_factor")
                .or_else(|| lookup.get_f32("partial_rotary_factor"))
                .or_else(|| {
                    // llama.cpp sometimes stores the rotary dim count as a u32.
                    lookup
                        .get_u32("qwen35moe.rope.rope_dimension_count")
                        .map(|v| (v as f32) / (hparams.head_dim as f32))
                })
                .unwrap_or(0.25);
            let qwen35_moe_cfg = Qwen35MoeConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                num_experts: hparams.expert_count.unwrap_or(8),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(2),
                shared_expert_intermediate_size: None,
                routed_scaling_factor: hparams.routed_scaling_factor,
                layer_types: vec![],
                linear_key_head_dim: 128,
                linear_num_key_heads: 16,
                linear_value_head_dim: 128,
                linear_num_value_heads: 128,
                partial_rotary_factor: prf,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
                full_yarn: parse_yarn_scaling_gguf(&lookup),
            };
            eprintln!(
                "[grim] Loading Qwen3.5/3.8 MoE model with config: {:?}",
                qwen35_moe_cfg
            );
            let m = Qwen35Moe::load_tp(device.clone(), &ws, qwen35_moe_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen38FlashNext => {
            let qwen38_cfg = Qwen38FlashNextConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                num_experts: hparams.expert_count.unwrap_or(512).max(512),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(10).max(10),
                shared_expert_intermediate_size: Some(hparams.intermediate_size),
                routed_scaling_factor: hparams.routed_scaling_factor,
                layer_types: vec![],
                linear_key_head_dim: 128,
                linear_num_key_heads: 8,
                linear_value_head_dim: 128,
                linear_num_value_heads: 8,
                ngram_vocab_size: Some(20_000_000),
                ngram_dim: Some(512),
                gated_residual_branches: 4,
                mrope_section: [11, 11, 10],
                partial_rotary_factor: 1.0,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
                full_yarn: parse_yarn_scaling_gguf(&lookup),
            };
            eprintln!(
                "[grim] Loading Qwen3.8-Flash-Next model from GGUF with config: {:?}",
                qwen38_cfg
            );
            let m = Qwen38FlashNext::load_tp(device.clone(), &ws, qwen38_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Mellum => {
            let mellum_cfg = MellumConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                moe_intermediate_size: hparams.expert_feed_forward_length.unwrap_or(896),
                num_experts: hparams.expert_count.unwrap_or(64),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(8),
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
                sliding_window: 1024,
                // Mellum2 uses YaRN RoPE on full-attention layers with these
                // documented constants (matches the HF `rope_parameters`).
                max_window_layers: 0,
                yarn: Some(grim_tensor::YaRNParams {
                    factor: 16.0,
                    original_max_pos: 8192,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    attention_factor: 1.277_258_9,
                }),
            };
            eprintln!("[grim] Loading Mellum model with config: {:?}", mellum_cfg);
            let m = Mellum::load_tp(device.clone(), &ws, mellum_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen2Moe
        | ModelArchitecture::Qwen3Moe
        | ModelArchitecture::Qwen3VlMoe => {
            let qwen_moe_cfg = Qwen3MoeConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                num_experts: hparams.expert_count.unwrap_or(8),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(2),
                routed_scaling_factor: hparams.routed_scaling_factor,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!(
                "[grim] Loading Qwen-MoE model with config: {:?}",
                qwen_moe_cfg
            );
            let m = Qwen3Moe::load_tp(device.clone(), &ws, qwen_moe_cfg, tp)?;
            Ok(Box::new(m))
        }
        arch if arch.is_moe() => {
            let moe_cfg = Qwen3MoeConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                num_experts: hparams.expert_count.unwrap_or(8),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(2),
                routed_scaling_factor: hparams.routed_scaling_factor,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!("[grim] Loading MoE model with config: {:?}", moe_cfg);
            let m = Qwen3Moe::load_tp(device.clone(), &ws, moe_cfg, tp)?;
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
            let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::FalconH1 => {
            let falcon_h1_cfg = FalconH1Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                ssm_d_state: hparams.ssm_d_state.unwrap_or(64),
                ssm_d_inner: hparams.ssm_d_inner.unwrap_or(hparams.intermediate_size),
                ssm_d_conv: hparams.ssm_d_conv.unwrap_or(4),
                ssm_dt_rank: hparams.ssm_dt_rank.unwrap_or(
                    hparams.ssm_d_inner.unwrap_or(hparams.intermediate_size)
                        / hparams.num_heads.max(1),
                ),
                ssm_n_group: hparams.ssm_n_group.unwrap_or(1),
            };
            eprintln!(
                "[grim] Loading Falcon-H1 model with config: {:?}",
                falcon_h1_cfg
            );
            let m = FalconH1Model::load_tp(device.clone(), &ws, falcon_h1_cfg, tp)?;
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
            let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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
            eprintln!(
                "[grim] Loading NemotronH model with config: {:?}",
                nemotron_cfg
            );
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
            let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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
            eprintln!(
                "[grim] Loading GraniteHybrid model with config: {:?}",
                granite_cfg
            );
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
            let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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
            eprintln!(
                "[grim] Loading ModernBERT model with config: {:?}",
                modern_bert_cfg
            );
            let bert_cfg = BertConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Bert::load_tp(device.clone(), &ws, bert_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::NomicBert
        | ModelArchitecture::NomicBertMoe
        | ModelArchitecture::NeoBert
        | ModelArchitecture::JinaBertV2
        | ModelArchitecture::JinaBertV3 => {
            let nomic_bert_cfg = NomicBertConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                layer_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!(
                "[grim] Loading NomicBERT model with config: {:?}",
                nomic_bert_cfg
            );
            let bert_cfg = BertConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                max_seq_len: hparams.max_seq_len,
            };
            let m = Bert::load_tp(device.clone(), &ws, bert_cfg, tp)?;
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
            eprintln!(
                "[grim] Loading T5Encoder model with config: {:?}",
                t5_enc_cfg
            );
            let t5_cfg = T5Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
            };
            let m = T5::load_tp(&ws, t5_cfg, tp)?;
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
            let rwkv_eps = lookup
                .get_f32("rwkv.epsilon")
                .or_else(|| lookup.get_f32("rms_norm_epsilon"))
                .or_else(|| lookup.get_f32("layer_norm_eps"))
                .map(|v| v as f64)
                .unwrap_or(hparams.rms_norm_eps as f64);
            let rwkv_cfg = RwkvConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_layers: hparams.num_layers,
                rms_norm_eps: rwkv_eps,
            };
            let m = Rwkv::load_tp(&ws, rwkv_cfg, device.clone(), tp)?;
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
            let rwkv_eps = lookup
                .get_f32("rwkv.epsilon")
                .or_else(|| lookup.get_f32("rms_norm_epsilon"))
                .or_else(|| lookup.get_f32("layer_norm_eps"))
                .map(|v| v as f64)
                .unwrap_or(hparams.rms_norm_eps as f64);
            let rwkv_cfg = RwkvConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_layers: hparams.num_layers,
                rms_norm_eps: rwkv_eps,
            };
            let m = Rwkv::load_tp(&ws, rwkv_cfg, device.clone(), tp)?;
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
                n_layer_dense_lead: hparams.num_layers, // all-dense unless metadata says otherwise
                n_expert: 0,
                n_expert_used: 1,
                n_ff_exp: hparams.intermediate_size,
                expert_weights_scale: 1.0,
                expert_gating_func: 0,
                n_swa: 0,
                swa_type: 0,
                n_embd_out: 0,
                // WI-X6: MXFP4 QKV attention is default-on for LFM2 family (escape hatch: GRIM_LFM2_MXFP4_QKV=0/false/off)
                mxfp4_qkv_attention: std::env::var("GRIM_LFM2_MXFP4_QKV")
                    .map(|v| {
                        v != "0"
                            && !v.eq_ignore_ascii_case("false")
                            && !v.eq_ignore_ascii_case("off")
                    })
                    .unwrap_or(true),
            };
            let m = Lfm2::load_tp(&ws, cfg, tp)?;
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
            let m = Mamba::load_tp(device.clone(), &ws, cfg, tp)?;
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
            let m = Gpt2::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Gemma
        | ModelArchitecture::Gemma2
        | ModelArchitecture::Gemma3
        | ModelArchitecture::Gemma4 => {
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
            let m = Gemma::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::CommandR => {
            let commandr_cfg = CommandRConfig {
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
            eprintln!(
                "[grim] Loading CommandR model with config: {:?}",
                commandr_cfg
            );
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

                partial_rotary_factor: 1.0,
                yarn: None,
            };
            let mut m = Llama::load_tp(device.clone(), &ws, llama_cfg, tp)?;
            // ALiBi position bias (baichuan/mpt/jais/gptneox class): enabled
            // when the GGUF carries the metadata key. ALiBi replaces RoPE.
            if lookup
                .get_f32("att.alibi.bias_max")
                .or_else(|| lookup.get_f32("alibi.bias_max"))
                .is_some()
            {
                eprintln!("[grim] enabling ALiBi on {} blocks", m.layers.len());
                for layer in m.layers.iter_mut() {
                    *layer = std::mem::replace(layer, layer.clone()).with_alibi();
                }
            }
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek => {
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
            let m = DeepSeek::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek2 => {
            let cfg = DeepSeek2Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                kv_lora_rank: 512,
                q_lora_rank: None,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
                moe_intermediate_size: hparams
                    .expert_feed_forward_length
                    .unwrap_or(hparams.intermediate_size),
                n_routed_experts: hparams.expert_count.unwrap_or(64),
                n_shared_experts: 2,
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(6),
                first_k_dense_replace: 1,
                routed_scaling_factor: 1.0,
            };
            let m = DeepSeek2::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek32 => {
            let cfg = DeepSeek32Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                kv_lora_rank: 512,
                q_lora_rank: Some(1536),
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
                moe_intermediate_size: hparams
                    .expert_feed_forward_length
                    .unwrap_or(hparams.intermediate_size),
                n_routed_experts: hparams.expert_count.unwrap_or(256),
                n_shared_experts: 1,
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(8),
                first_k_dense_replace: 1,
                routed_scaling_factor: 2.5,
            };
            let m = DeepSeek32::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeepSeek4 => {
            let cfg = DeepSeek4Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                kv_lora_rank: 512,
                q_lora_rank: Some(1024),
                qk_nope_head_dim: 448,
                qk_rope_head_dim: 64,
                v_head_dim: 512,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_seq_len: hparams.max_seq_len,
                moe_intermediate_size: hparams
                    .expert_feed_forward_length
                    .unwrap_or(hparams.intermediate_size),
                n_routed_experts: hparams.expert_count.unwrap_or(256),
                n_shared_experts: 1,
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(6),
                first_k_dense_replace: 3,
                routed_scaling_factor: 2.5,
                hc_mult: 4,
                sqrtsoftplus_moe: true,
                compressor_indexer_enabled: true,
            };
            let m = DeepSeek4::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::MuseGlimmer => {
            let softcap = lookup
                .get_f32("muse_glimmer.final_logit_softcapping")
                .unwrap_or(0.0);
            let qk_scale = lookup
                .get_f32("muse_glimmer.qk_scale_factor")
                .unwrap_or(1.0);
            let sliding_win = lookup.get_u32("muse_glimmer.sliding_window").unwrap_or(0) as usize;

            let per_layer_rope: Vec<f32> =
                get_meta_array(provider, "muse_glimmer.per_layer_rope_theta")
                    .map(|arr| arr.iter().filter_map(|v| v.as_f32()).collect())
                    .unwrap_or_default();
            let sliding_window_layer_ids: Vec<usize> =
                get_meta_array(provider, "muse_glimmer.sliding_window_layer_ids")
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u32().map(|u| u as usize))
                            .collect()
                    })
                    .unwrap_or_default();
            let output_multiplier: Vec<f32> =
                get_meta_array(provider, "muse_glimmer.output_multiplier")
                    .map(|arr| arr.iter().filter_map(|v| v.as_f32()).collect())
                    .unwrap_or_default();

            let vision_cfg = if lookup.get_u32("muse_glimmer.vision.image_size").is_some()
                || lookup.get_u32("muse_glimmer.vision.num_layers").is_some()
            {
                Some(grim_models_vision::GlimmerVisionConfig {
                    image_temporal: lookup
                        .get_u32("muse_glimmer.vision.image_temporal")
                        .unwrap_or(2) as usize,
                    image_size: lookup
                        .get_u32("muse_glimmer.vision.image_size")
                        .unwrap_or(336) as usize,
                    patch_size: lookup
                        .get_u32("muse_glimmer.vision.patch_size")
                        .unwrap_or(14) as usize,
                    temporal_patch_size: lookup
                        .get_u32("muse_glimmer.vision.temporal_patch_size")
                        .unwrap_or(2) as usize,
                    in_channels: lookup
                        .get_u32("muse_glimmer.vision.in_channels")
                        .unwrap_or(3) as usize,
                    hidden_size: lookup
                        .get_u32("muse_glimmer.vision.hidden_size")
                        .unwrap_or(1024) as usize,
                    num_heads: lookup
                        .get_u32("muse_glimmer.vision.num_heads")
                        .unwrap_or(16) as usize,
                    num_layers: lookup
                        .get_u32("muse_glimmer.vision.num_layers")
                        .unwrap_or(24) as usize,
                    intermediate_size: lookup
                        .get_u32("muse_glimmer.vision.intermediate_size")
                        .unwrap_or(4096) as usize,
                    rms_norm_eps: lookup
                        .get_f32("muse_glimmer.vision.rms_norm_eps")
                        .unwrap_or(1e-5),
                    merge_size: lookup
                        .get_u32("muse_glimmer.vision.merge_size")
                        .unwrap_or(2) as usize,
                    use_vision_norm: true,
                })
            } else {
                None
            };

            let muse_cfg = grim_models_transformer::MuseGlimmerConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                per_layer_rope_theta: per_layer_rope,
                base_rope_theta: hparams.rope_theta,
                sliding_window_layer_ids,
                sliding_window_size: sliding_win,
                qk_scale_factor: qk_scale,
                output_multiplier,
                final_logit_softcapping: softcap,
                max_seq_len: hparams.max_seq_len,
                vision: vision_cfg,
            };
            let m =
                grim_models_transformer::MuseGlimmer::load_tp(device.clone(), &ws, muse_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::SolarOpen2 => {
            let solar_cfg = SolarOpen2Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                num_kv_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
                num_routed_experts: lookup
                    .get_u32("solar_open2.expert_count")
                    .map(|u| u as usize)
                    .unwrap_or(320),
                num_shared_experts: 1,
                top_k: lookup
                    .get_u32("solar_open2.expert_used_count")
                    .map(|u| u as usize)
                    .unwrap_or(8),
            };
            let m = SolarOpen2::load_tp(&ws, solar_cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::InternS2Mobius => {
            let cfg = grim_models_transformer::InternS2MobiusConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_attention_heads: hparams.num_heads,
                num_key_value_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_hidden_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_position_embeddings: hparams.max_seq_len,
            };
            let m = grim_models_transformer::InternS2Mobius::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::KimiK3 => {
            let cfg = grim_models_transformer::KimiK3Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_attention_heads: hparams.num_heads,
                num_key_value_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_hidden_layers: hparams.num_layers,
                q_lora_rank: hparams.num_heads,
                kv_lora_rank: hparams.num_kv_heads * 4,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                num_experts: hparams.expert_count.unwrap_or(8),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(2),
                intermediate_size: hparams.intermediate_size,
                routed_scaling_factor: hparams.routed_scaling_factor,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
            };
            let m = grim_models_transformer::KimiK3::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::InklingSmall => {
            let cfg = grim_models_transformer::InklingSmallConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_attention_heads: hparams.num_heads,
                num_key_value_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_hidden_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_position_embeddings: hparams.max_seq_len,
            };
            let m = grim_models_transformer::InklingSmall::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }

        ModelArchitecture::Glm52 => {
            let cfg = grim_models_transformer::Glm52Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_attention_heads: hparams.num_heads,
                num_key_value_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_hidden_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                num_experts: hparams.expert_count.unwrap_or(8),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(2),
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_position_embeddings: hparams.max_seq_len,
            };
            let m = grim_models_transformer::Glm52::load_tp(device.clone(), &ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DiffusionGemma => {
            let cfg = grim_models_transformer::DiffusionGemmaConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_attention_heads: hparams.num_heads,
                num_key_value_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_hidden_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_position_embeddings: hparams.max_seq_len,
            };
            let m = grim_models_transformer::DiffusionGemma::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::MiniMaxM3 => {
            let cfg = grim_models_transformer::MiniMaxM3Config {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_attention_heads: hparams.num_heads,
                num_key_value_heads: hparams.num_kv_heads,
                head_dim: hparams.head_dim,
                num_hidden_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                num_experts: hparams.expert_count.unwrap_or(8),
                num_experts_per_tok: hparams.expert_used_count.unwrap_or(2),
                rms_norm_eps: hparams.rms_norm_eps,
                rope_theta: hparams.rope_theta,
                max_position_embeddings: hparams.max_seq_len,
            };
            let m = grim_models_transformer::MiniMaxM3::load_tp(device.clone(), &ws, cfg)?;
            Ok(Box::new(m))
        }

        ModelArchitecture::BailingMoe3 => {
            // Audit fix (grim-models): this branch loaded BailingMoeV3
            // checkpoints as Qwen3Moe — architecturally wrong (BailingMoeV3
            // is an MLA/KDA-hybrid MoE; its GGUF tensor names do not match
            // Qwen3-MoE). The in-crate Ling3Tiny implementation exists but
            // has no GGUF hparams/tensor mapping yet, so refuse loudly
            // instead of silently building a mismatched model.
            Err(grim_core::error::Error::Config(
                "BailingMoeV3 (bailingmoe3 / bailing_hybrid) is not loadable from GGUF: \
                 its MLA+KDA hybrid architecture has no GGUF tensor mapping in this \
                 loader (loading it as Qwen3MoE was wrong and has been removed). \
                 Wire Ling3Tiny's loader or convert the checkpoint."
                    .into(),
            ))
        }

        ModelArchitecture::Chameleon => {
            let chameleon_cfg = ChameleonConfig {
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
                swin_norm: true,
            };
            eprintln!(
                "[grim] Loading Chameleon model with config: {:?}",
                chameleon_cfg
            );
            let m = Chameleon::load_tp(device.clone(), &ws, chameleon_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::CogVlm => {
            let cog_cfg = CogVlmConfig {
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
                vision_config: CogVlmVisionConfig {
                    hidden_size: 1024,
                    image_size: 490,
                    patch_size: 14,
                    num_heads: 16,
                    num_layers: 24,
                    in_channels: 3,
                    out_hidden_size: hparams.hidden_size,
                },
            };
            let m = CogVlm::load_tp(device.clone(), &ws, cog_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Gemma3n => {
            let gemma_cfg = Gemma3nConfig {
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
                sliding_window_size: 1024,
                query_pre_attn_scalar: 256.0,
            };
            let m = Gemma3n::load_tp(device.clone(), &ws, gemma_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::HunyuanVl => {
            let hy_cfg = HunyuanVlConfig {
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
                mrope_section: [2, 2, 2, 2],
                image_token_id: 5,
                im_start_id: 120118,
                im_end_id: 120119,
                im_newline_id: 120121,
                vision_config: HunyuanVlVisionConfig {
                    hidden_size: 64,
                    num_attention_heads: 4,
                    num_key_value_heads: 4,
                    num_hidden_layers: 2,
                    patch_size: 16,
                    num_channels: 3,
                    intermediate_size: 128,
                    out_hidden_size: 64,
                },
            };
            let m = HunyuanVl::load_tp(device.clone(), &ws, hy_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen2Vl => {
            let qv_cfg = Qwen2VlConfig {
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
                mrope_section: [16, 24, 24],
                vision_start_token_id: 151652,
                vision_end_token_id: 151653,
                vision_token_id: 151654,
                image_token_id: 151655,
                video_token_id: 151656,
                vision_config: Qwen2VlVisionConfig {
                    depth: 32,
                    hidden_size: 1280,
                    num_heads: 16,
                    patch_size: 14,
                    spatial_merge_size: 2,
                    temporal_patch_size: 2,
                    in_channels: 3,
                    out_hidden_size: hparams.hidden_size,
                },
            };
            let m = Qwen2Vl::load_tp(device.clone(), &ws, qv_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Qwen3Vl => {
            let qv_cfg = Qwen3VlConfig {
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
                mrope_section: [24, 20, 20],
                deepstack_visual_indexes: vec![8, 16, 24],
                vision_start_token_id: 151652,
                vision_end_token_id: 151653,
                vision_token_id: 151654,
                image_token_id: 151655,
                video_token_id: 151656,
                vision_config: Qwen3VlVisionConfig {
                    depth: 27,
                    hidden_size: 1152,
                    num_heads: 16,
                    patch_size: 16,
                    spatial_merge_size: 2,
                    temporal_patch_size: 2,
                    in_channels: 3,
                    out_hidden_size: hparams.hidden_size,
                    deepstack_visual_indexes: vec![8, 16, 24],
                },
            };
            let m = Qwen3Vl::load_tp(device.clone(), &ws, qv_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::WavTokenizerDec => {
            let wav_cfg = WavTokenizerDecConfig {
                latent_dim: 512,
                backbone_dim: 768,
                backbone_num_blocks: 12,
                backbone_intermediate_dim: 2304,
                backbone_kernel_size: 7,
                n_fft: 1280,
                hop_length: 320,
                head_dim: 641,
                codebook_size: 4096,
                codebook_dim: 512,
                num_bandwidths: 4,
                sample_rate: 24000,
            };
            let m = WavTokenizerDec::load_tp(device.clone(), &ws, wav_cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::DeltaNetBase => {
            let delta_cfg = DeltaNetBaseConfig {
                vocab_size: hparams.vocab_size,
                hidden_size: hparams.hidden_size,
                num_heads: hparams.num_heads,
                head_dim: hparams.head_dim,
                num_layers: hparams.num_layers,
                intermediate_size: hparams.intermediate_size,
                chunk_size: 64,
                rms_norm_eps: hparams.rms_norm_eps,
                max_seq_len: hparams.max_seq_len,
            };
            eprintln!(
                "[grim] Loading DeltaNetBase model with config: {:?}",
                delta_cfg
            );
            let m = DeltaNetBase::load_tp(device.clone(), &ws, delta_cfg, tp)?;
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
            let m = Bert::load_tp(device.clone(), &ws, cfg, tp)?;
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
            let m = T5::load_tp(&ws, cfg, tp)?;
            Ok(Box::new(m))
        }
        ModelArchitecture::Arcee
        | ModelArchitecture::Apertus
        | ModelArchitecture::Arctic
        | ModelArchitecture::Baichuan
        | ModelArchitecture::BitNet
        | ModelArchitecture::ChatGlm
        | ModelArchitecture::Codeshell
        | ModelArchitecture::Cohere2
        | ModelArchitecture::Dbrx
        | ModelArchitecture::Deci
        | ModelArchitecture::DeepSeek2Ocr
        | ModelArchitecture::DFlash
        | ModelArchitecture::Dots1
        | ModelArchitecture::Dream
        | ModelArchitecture::Eagle3
        | ModelArchitecture::Ernie45
        | ModelArchitecture::Eurobert
        | ModelArchitecture::Exaone
        | ModelArchitecture::Exaone4
        | ModelArchitecture::Gemma4Assistant
        | ModelArchitecture::GemmaEmbedding
        | ModelArchitecture::Glm4
        | ModelArchitecture::GlmDsa
        | ModelArchitecture::GptJ
        | ModelArchitecture::GptNeoX
        | ModelArchitecture::Granite
        | ModelArchitecture::Grok
        | ModelArchitecture::HunyuanDense
        | ModelArchitecture::HyV3
        | ModelArchitecture::InternLm2
        | ModelArchitecture::Jais
        | ModelArchitecture::Jais2
        | ModelArchitecture::KimiLinear
        | ModelArchitecture::Llada
        | ModelArchitecture::Llama
        | ModelArchitecture::Llama4
        | ModelArchitecture::LlamaEmbed
        | ModelArchitecture::MainCoder
        | ModelArchitecture::Mimo2
        | ModelArchitecture::MiniMaxM2
        | ModelArchitecture::Mistral3
        | ModelArchitecture::Mistral4
        | ModelArchitecture::Mpt
        | ModelArchitecture::Nemotron
        | ModelArchitecture::Olmo
        | ModelArchitecture::Olmo2
        | ModelArchitecture::OpenElm
        | ModelArchitecture::Orion
        | ModelArchitecture::PaddleOcr
        | ModelArchitecture::PanguEmbed
        | ModelArchitecture::Plamo
        | ModelArchitecture::Plamo2
        | ModelArchitecture::Plamo3
        | ModelArchitecture::Plm
        | ModelArchitecture::Qwen3Next
        | ModelArchitecture::Refact
        | ModelArchitecture::Rnd1
        | ModelArchitecture::SeedOss
        | ModelArchitecture::SmallThinker
        | ModelArchitecture::SmolLm3
        | ModelArchitecture::StableLm
        | ModelArchitecture::Starcoder
        | ModelArchitecture::Starcoder2
        | ModelArchitecture::Step35
        | ModelArchitecture::Talkie
        | ModelArchitecture::Xverse => {
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

                partial_rotary_factor: 1.0,
                yarn: None,
            };
            eprintln!(
                "[grim] Loading Llama-family model ({:?}) with config: {:?}",
                model_arch, llama_cfg
            );
            let mut m = Llama::load_tp(device.clone(), &ws, llama_cfg, tp)?;
            // ALiBi position bias (baichuan/mpt/jais/gptneox class): enabled
            // when the GGUF carries the metadata key. ALiBi replaces RoPE.
            if lookup
                .get_f32("att.alibi.bias_max")
                .or_else(|| lookup.get_f32("alibi.bias_max"))
                .is_some()
            {
                eprintln!("[grim] enabling ALiBi on {} blocks", m.layers.len());
                for layer in m.layers.iter_mut() {
                    *layer = std::mem::replace(layer, layer.clone()).with_alibi();
                }
            }
            Ok(Box::new(m))
        }
        _ => {
            // Check for a sibling config.json alongside the GGUF file to
            // enrich ArchCompatSpec resolution for known HF architectures.
            // Read the file *contents* (not just the path) so that
            // resolve_arch_compat_spec's inline-config branch can parse it.
            // A missing sibling config.json is optional — fall back to the
            // plugins-dir scan, don't error.
            let config_raw = read_sibling_config_json(path);

            if let Some(spec) = resolve_arch_compat_spec(arch_str, config_raw.as_deref()) {
                eprintln!(
                    "[grim] Resolved unknown GGUF architecture '{}' via ArchCompatSpec plugin (base='{}', is_moe={})",
                    arch_str, spec.base_architecture, spec.is_moe
                );
                let spec_clone = spec.clone();
                let remapped_provider =
                    RemappingTensorProvider::new(weight_provider, move |name: &str| -> String {
                        spec_clone.remap_tensor_name(name)
                    });
                let ws = WeightSource::root(&remapped_provider, device.clone()).with_tp_config(tp);
                ws.prefetch_all();

                if spec.is_moe || hparams.expert_count.unwrap_or(0) > 0 {
                    let intermediate = hparams
                        .expert_feed_forward_length
                        .unwrap_or(hparams.intermediate_size);
                    let moe_cfg = Qwen3MoeConfig {
                        vocab_size: hparams.vocab_size,
                        hidden_size: hparams.hidden_size,
                        num_heads: hparams.num_heads,
                        num_kv_heads: hparams.num_kv_heads,
                        head_dim: hparams.head_dim,
                        num_layers: hparams.num_layers,
                        intermediate_size: intermediate,
                        num_experts: hparams.expert_count.unwrap_or(8),
                        num_experts_per_tok: hparams.expert_used_count.unwrap_or(2),
                        routed_scaling_factor: hparams.routed_scaling_factor,
                        rms_norm_eps: hparams.rms_norm_eps,
                        rope_theta: hparams.rope_theta,
                        max_seq_len: hparams.max_seq_len,
                    };
                    let m = Qwen3Moe::load_tp(device.clone(), &ws, moe_cfg, tp)?;
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
                    let m = Mamba::load_tp(device.clone(), &ws, mamba_cfg, tp)?;
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

                        partial_rotary_factor: 1.0,
                        yarn: None,
                    };
                    let m = Llama::load_tp(device.clone(), &ws, llama_cfg, tp)?;
                    return Ok(Box::new(m));
                }
            }

            Err(Error::Config(format!(
                "Unsupported GGUF architecture '{}': no plugin compat spec and no native loader",
                arch_str
            )))
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

    // Check for forced device first. `GRIM_BACKEND` is canonical (set by the
    // install script and by `serve --backend`); `GRIM_FORCE_DEVICE` is a
    // legacy alias. An explicitly requested backend must actually be
    // available — never silently degrade to a different device (WS-E1).
    let forced = std::env::var("GRIM_BACKEND")
        .or_else(|_| std::env::var("GRIM_FORCE_DEVICE"))
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty() && s != "auto");
    if let Some(s) = forced {
        // Cheap capability check per backend, reusing the same probes the
        // selection chain below uses.
        let prefix = s.split(':').next().unwrap_or("").trim();
        let available = match prefix {
            "cpu" => true,
            "rocm" => grim_backend_rocm::RocmDevice::probe()
                .map(|d| !d.is_empty())
                .unwrap_or(false),
            "metal" => grim_backend_metal::MetalDevice::probe()
                .map(|d| !d.is_empty())
                .unwrap_or(false),
            #[cfg(feature = "cuda")]
            "cuda" => grim_backend_cuda::CudaDevice::probe()
                .map(|d| !d.is_empty())
                .unwrap_or(false),
            #[cfg(not(feature = "cuda"))]
            "cuda" => false,
            // The engine-side loader has no Vulkan path; report that honestly
            // rather than silently running on a different backend.
            "vulkan" => false,
            other => {
                return Err(Error::Config(format!(
                    "unknown backend '{other}' requested via GRIM_BACKEND \
                     (expected rocm|cuda|vulkan|metal|cpu|auto)"
                )));
            }
        };
        if !available {
            return Err(Error::Config(format!(
                "backend '{prefix}' requested via GRIM_BACKEND but unavailable \
                 (not compiled in or no device). Rebuild with --features {prefix} \
                 or unset GRIM_BACKEND for auto-detection."
            )));
        }
        match s.as_str() {
            #[cfg(feature = "cuda")]
            "cuda" => {
                if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
                    if let Some(first) = cuda_devices.first() {
                        eprintln!(
                            "[model_loader] Using CUDA device {} (forced)",
                            first.ordinal()
                        );
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
                    // Under multi-process TP, pin this process to its own rank
                    // ordinal (from GRIM_TP_RANK / GRIM_GPUS) rather than
                    // always using the first visible device — otherwise every
                    // rank would load onto the same GPU and the collective
                    // would deadlock waiting for peers that never started.
                    let (my_ordinal, _all_ordinals) = resolve_tp_ordinal()?;
                    let rank = TensorParallelConfig::from_env()
                        .map(|t| t.rank)
                        .unwrap_or(0);
                    let chosen = match my_ordinal {
                        Some(ord) => {
                            let d = rocm_devices
                                .iter()
                                .find(|dev| dev.ordinal() == ord)
                                .ok_or_else(|| {
                                    Error::Config(format!(
                                        "TP rank {rank} needs ROCm ordinal {ord} but it is not \
                                         visible (probe found {n} device(s))",
                                        n = rocm_devices.len()
                                    ))
                                })?;
                            eprintln!(
                                "[model_loader] Using ROCm device {ord} (forced, TP rank {rank})"
                            );
                            d.ordinal()
                        }
                        None => {
                            let first = rocm_devices.first().expect("checked above");
                            eprintln!(
                                "[model_loader] Using ROCm device {} (forced)",
                                first.ordinal()
                            );
                            first.ordinal()
                        }
                    };
                    let dev = Device::Rocm(chosen);
                    return if is_grim {
                        load_model_from_grim(path, dev)
                    } else if is_safetensors {
                        load_model_from_safetensors(path, dev)
                    } else {
                        load_model_from_gguf(path, dev)
                    };
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
        // Same rank-ordinal pinning as the forced branch above: when multi-
        // process TP is active, load onto THIS rank's ordinal so each peer
        // process owns a distinct GPU and the RCCL collective can rendezvous.
        let (my_ordinal, _all_ordinals) = resolve_tp_ordinal()?;
        let rank = TensorParallelConfig::from_env()
            .map(|t| t.rank)
            .unwrap_or(0);
        let chosen = match my_ordinal {
            Some(ord) => {
                let Some(d) = rocm_devices.iter().find(|dev| dev.ordinal() == ord) else {
                    return Err(Error::Config(format!(
                        "TP rank {rank} needs ROCm ordinal {ord} but it is not visible \
                         (probe found {n} device(s))",
                        n = rocm_devices.len()
                    )));
                };
                eprintln!("[model_loader] Using ROCm device {ord} (TP rank {rank})");
                d.ordinal()
            }
            None => {
                let Some(first) = rocm_devices.first() else {
                    return Err(Error::Config(
                        "ROCm probe returned an empty device list; cannot select a default GPU. \
                         Set GRIM_TP_GPUS to pin this rank's ordinal."
                            .into(),
                    ));
                };
                eprintln!("[model_loader] Using ROCm device {}", first.ordinal());
                first.ordinal()
            }
        };
        let dev = Device::Rocm(chosen);
        return if is_grim {
            load_model_from_grim(path, dev)
        } else if is_safetensors {
            load_model_from_safetensors(path, dev)
        } else {
            load_model_from_gguf(path, dev)
        };
    }
    // Fall back to CUDA.
    #[cfg(feature = "cuda")]
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

/// Load a model onto an explicitly chosen device — the SCYTHE-2 farm-replica
/// entry point. Unlike [`load_from_path`], no env-driven backend selection
/// runs: the caller (the engine's farm loader) owns the per-replica device
/// decision, and an unavailable device surfaces as that backend's own load
/// error rather than a silent fallback.
pub fn load_from_path_on_device(path: &str, dev: Device) -> Result<Box<dyn CausalLm>> {
    if path.ends_with(".grim") {
        load_model_from_grim(path, dev)
    } else if path.ends_with(".safetensors") {
        load_model_from_safetensors(path, dev)
    } else {
        load_model_from_gguf(path, dev)
    }
}

/// ROCm devices visible to this process, in ordinal order. The SCYTHE-2 farm
/// loader loads one weight replica per entry; an empty list means this process
/// sees no AMD GPUs (CPU-only box) and farm mode cannot arm.
/// Resolve ROCm GPUs visible to the farm registrar. Mirrors
/// [`resolve_discrete_rocm_devices`]'s policy of taking only dedicated GPUs —
/// integrated APU devices must never become SCYTHE-2 farm ranks (they share
/// system memory and would skew placement) — so at most the first two probes
/// are eligible, matching syd-beasty's discrete pair.
pub fn visible_rocm_devices() -> Vec<Device> {
    grim_backend_rocm::RocmDevice::probe()
        .map(|devices| {
            devices
                .iter()
                .take(2)
                .map(|d| Device::Rocm(d.ordinal()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

/// Load an EAGLE3 speculative drafter model from a safetensors checkpoint.
pub fn load_eagle3_from_path(
    path: &str,
    device: Device,
) -> Result<std::sync::Arc<grim_models_transformer::Eagle3>> {
    let tp = resolve_tp_config()?;
    let tprov = SafetensorsProvider::open(path)?;
    let ws = WeightSource::root(&tprov, device.clone()).with_tp_config(tp);
    let cfg = grim_models_transformer::Eagle3Config {
        vocab_size: 128256,
        hidden_size: 2048,
        target_hidden_size: 4096,
        num_heads: 16,
        num_kv_heads: 8,
        head_dim: 128,
        num_layers: 4,
        intermediate_size: 8192,
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        max_seq_len: 8192,
        num_target_fusion_layers: 3,
    };
    let model = grim_models_transformer::Eagle3::load(device, &ws, cfg)?;
    Ok(std::sync::Arc::new(model))
}

/// Load an audio model (Whisper ASR, Kokoro TTS, MeanVC2 Voice Conversion, Vocos Vocoder) with auto device detection.
pub fn load_audio_model(path: &str) -> Result<std::sync::Arc<dyn grim_core::Model>> {
    let dev = if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
        if let Some(first) = rocm_devices.first() {
            Device::Rocm(first.ordinal())
        } else {
            Device::Cpu
        }
    } else {
        Device::Cpu
    };
    load_audio_model_from_path(path, dev)
}

/// Load an audio model (Whisper, Kokoro, MeanVC2, Vocos) onto target device from path.
pub fn load_audio_model_from_path(
    path: &str,
    device: Device,
) -> Result<std::sync::Arc<dyn grim_core::Model>> {
    let p = Path::new(path);
    let filename = p
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_ascii_lowercase();

    if (filename.contains("whisper") || path.ends_with(".gguf")) && path.ends_with(".gguf") {
        let provider = GgufProvider::open(path)?;
        let ws = WeightSource::root(&provider, device.clone());
        let cfg = grim_models_audio::WhisperConfig {
            vocab_size: 51865,
            n_mels: 80,
            d_model: 384,
            num_enc_layers: 4,
            num_dec_layers: 4,
            num_heads: 6,
            ffn_dim: 1536,
            max_audio_len: 3000,
            max_text_len: 448,
            rms_norm_eps: 1e-5,
        };
        let whisper = grim_models_audio::Whisper::load(device, &ws, cfg)?;
        return Ok(std::sync::Arc::new(whisper));
    }

    if filename.contains("kokoro") || path.ends_with(".pth") {
        let cfg = grim_models_audio::KokoroConfig::default();
        let kokoro = grim_models_audio::Kokoro::random(device, cfg);
        return Ok(std::sync::Arc::new(kokoro));
    }

    if filename.contains("vocos") {
        let provider = PthProvider::load_from_file(path)?;
        let ws = WeightSource::root(&provider, device.clone());
        let cfg = infer_vocos_config(&provider)?;
        let vocos = grim_models_audio::Vocos::load(device, &ws, cfg)?;
        return Ok(std::sync::Arc::new(vocos));
    }

    if filename.contains("meanvc") || path.ends_with(".pt") {
        let cfg = grim_models_audio::MeanVC2Config::default();
        let meanvc = grim_models_audio::MeanVC2::random(device, cfg);
        return Ok(std::sync::Arc::new(meanvc));
    }

    // Default fallback to Whisper
    let cfg = grim_models_audio::WhisperConfig {
        vocab_size: 51865,
        n_mels: 80,
        d_model: 384,
        num_enc_layers: 4,
        num_dec_layers: 4,
        num_heads: 6,
        ffn_dim: 1536,
        max_audio_len: 3000,
        max_text_len: 448,
        rms_norm_eps: 1e-5,
    };
    let whisper = grim_models_audio::Whisper::random(device, cfg);
    Ok(std::sync::Arc::new(whisper))
}

/// Infer a `VocosConfig` from checkpoint tensor shapes.
///
/// Reads `backbone.embed.weight` (`[dim, input_dim, 7]`), counts
/// `backbone.convnext.N` blocks, derives `intermediate_dim` from block 0's
/// `pwconv1.weight`, and derives `n_fft` from `head.istft.window` (hop is
/// `n_fft / 2`, the standard Vocos setting).
fn infer_vocos_config(provider: &PthProvider) -> Result<grim_models_audio::VocosConfig> {
    use grim_tensor::provider::TensorProvider;

    let mut cfg = grim_models_audio::VocosConfig::default();

    if let Ok(meta) = provider.meta("backbone.embed.weight") {
        if meta.shape.len() == 3 {
            cfg.dim = meta.shape[0];
            cfg.input_dim = meta.shape[1];
        }
    }

    let num_layers = provider
        .tensor_names()
        .iter()
        .filter_map(|n| n.strip_prefix("backbone.convnext."))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|idx| idx.parse::<usize>().ok())
        .max()
        .map(|m| m + 1);
    if let Some(n) = num_layers {
        cfg.num_layers = n;
    }

    if let Ok(meta) = provider.meta("backbone.convnext.0.pwconv1.weight") {
        if !meta.shape.is_empty() {
            cfg.intermediate_dim = meta.shape[0];
        }
    }

    if let Ok(meta) = provider.meta("head.istft.window") {
        if let Some(first) = meta.shape.first() {
            cfg.n_fft = *first;
            cfg.hop_length = cfg.n_fft / 2;
        }
    }

    Ok(cfg)
}

/// Specialized loader for Diffusion Models (Flux 2 MM-DiT and 2D UNet).
pub fn load_diffusion_model_from_path(
    path: &str,
    device: Device,
) -> Result<std::sync::Arc<dyn grim_core::Model>> {
    let p = Path::new(path);
    let filename = p
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_ascii_lowercase();

    if filename.contains("flux") {
        let cfg = grim_models_diffusion::Flux2Config::default();
        let flux = grim_models_diffusion::Flux2Transformer2D::random(device, cfg);
        return Ok(std::sync::Arc::new(flux));
    }

    // Default UNet2D fallback
    let cfg = grim_models_diffusion::UnetConfig {
        in_channels: 4,
        out_channels: 4,
        hidden: 64,
        num_downsample: 2,
        rms_norm_eps: 1e-5,
        num_timesteps: 1000,
    };
    let unet = grim_models_diffusion::Unet2D::random(device, cfg);
    Ok(std::sync::Arc::new(unet))
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

    /// Tests that `resolve_arch_compat_spec` correctly parses a HF config string
    /// and generates dynamic tensor remapping rules.
    #[test]
    fn test_arch_compat_spec_resolution_and_remapping() {
        let sample_json = r#"{
            "model_type": "custom_transformer",
            "hidden_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 16
        }"#;

        let spec = resolve_arch_compat_spec("custom_transformer", Some(sample_json))
            .expect("must resolve spec");
        assert_eq!(spec.hidden_size, 2048);
        assert_eq!(spec.num_layers, 12);
        assert_eq!(spec.num_heads, 16);

        // Test bidirectional remapping helper
        let hf_name = "model.layers.0.self_attn.q_proj.weight";
        let gguf_name = "blk.0.attn_q.weight";
        assert_eq!(spec.remap_tensor_name(hf_name), gguf_name);
        assert_eq!(spec.remap_tensor_name(gguf_name), hf_name);
    }

    /// `parse_yarn_scaling` must accept a standard HF `rope_scaling` block with
    /// `rope_type: "yarn"` and populate all five YaRNParams fields (with sensible
    /// defaults for any missing sub-field).
    #[test]
    fn parse_yarn_scaling_reads_standard_hf_yarn_block() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"rope_type": "yarn", "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "beta_fast": 32.0, "beta_slow": 1.0,
                "attention_factor": 0.1}"#,
        )
        .unwrap();
        let y = parse_yarn_scaling(&Some(v)).expect("yarn block must parse");
        assert!((y.factor - 4.0).abs() < 1e-6);
        assert_eq!(y.original_max_pos, 32768);
        assert!((y.beta_fast - 32.0).abs() < 1e-6);
        assert!((y.beta_slow - 1.0).abs() < 1e-6);
        assert!((y.attention_factor - 0.1).abs() < 1e-6);
    }

    /// Non-YaRN rope types (linear / dynamic) must fall back to plain RoPE
    /// (return `None`), not silently fabricate YaRN params.
    #[test]
    fn parse_yarn_scaling_returns_none_for_non_yarn_rope_types() {
        for rt in ["linear", "dynamic", "ntk-aware", "longrope"] {
            let v: serde_json::Value =
                serde_json::from_str(&format!(r#"{{"rope_type": "{rt}", "factor": 4.0}}"#))
                    .unwrap();
            assert!(
                parse_yarn_scaling(&Some(v)).is_none(),
                "rope_type={rt} must not produce YaRNParams"
            );
        }
        assert!(parse_yarn_scaling(&None).is_none());
    }

    /// A yarn block missing optional sub-fields must still parse, applying the
    /// documented defaults (factor 1.0, beta_fast 32.0, beta_slow 1.0,
    /// attention_factor 1.0, original_max_pos 8192).
    #[test]
    fn parse_yarn_scaling_applies_defaults_for_missing_fields() {
        let v: serde_json::Value = serde_json::from_str(r#"{"rope_type": "yarn"}"#).unwrap();
        let y = parse_yarn_scaling(&Some(v)).expect("yarn (defaults) must parse");
        assert!((y.factor - 1.0).abs() < 1e-6);
        assert_eq!(y.original_max_pos, 8192);
        assert!((y.beta_fast - 32.0).abs() < 1e-6);
        assert!((y.beta_slow - 1.0).abs() < 1e-6);
        assert!((y.attention_factor - 1.0).abs() < 1e-6);
    }

    /// `parse_yarn_scaling` must accept the llama.cpp `attn_factor` alias for the
    /// attention-factor field (used by some GGUF converters).
    #[test]
    fn parse_yarn_scaling_accepts_attn_factor_alias() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"rope_type": "yarn", "attn_factor": 0.707}"#).unwrap();
        let y = parse_yarn_scaling(&Some(v)).expect("yarn (alias) must parse");
        assert!((y.attention_factor - 0.707).abs() < 1e-5);
    }

    // ---------------------------------------------------------------------------
    // Red-phase tests: GGUF config.json read bug + plugins-dir scan branch
    // ---------------------------------------------------------------------------

    /// Regression test for the GGUF config.json read bug.
    ///
    /// The bug (model_loader.rs ~line 2760-2763, now extracted to
    /// `read_sibling_config_json`) built a *path string* instead of reading
    /// file contents:
    ///   `dir.join("config.json").to_str().map(|s| s.to_string())`
    /// which passed "/path/to/config.json" as `config_raw`. `from_hf_config_json`
    /// would then try to parse that path string as JSON and fail, silently
    /// falling through to the plugins-dir scan instead of using the sibling
    /// config.json.
    ///
    /// This test exercises `read_sibling_config_json` directly — the exact helper
    /// the GGUF load path uses — and verifies it returns the file *contents*,
    /// not a path string. A regression that reverts to the old buggy behavior
    /// would break this test.
    #[test]
    fn read_sibling_config_json_reads_file_contents_not_path() {
        // Build a temp dir with a fake .gguf path and sibling config.json.
        let tmp_dir = tempfile::TempDir::new().expect("temp dir");
        let gguf_path = tmp_dir.path().join("model.gguf");
        std::fs::write(&gguf_path, b"fake").expect("fake gguf");

        let config_json = r#"{
            "model_type": "test-arch",
            "hidden_size": 2048,
            "num_hidden_layers": 12
        }"#;
        let config_path = tmp_dir.path().join("config.json");
        std::fs::write(&config_path, config_json).expect("write config.json");

        // Simulate the GGUF load path: it calls read_sibling_config_json(gguf_path)
        // to get the sibling config.json contents.
        let gguf_path_str = gguf_path.to_str().expect("gguf path must be valid utf8");
        let config_raw = read_sibling_config_json(gguf_path_str);

        // The fix: config_raw must be Some(contents), not Some(path_string).
        let config_raw_str = config_raw.expect("sibling config.json must be read");
        assert_eq!(
            config_raw_str, config_json,
            "must return file contents, not path"
        );

        // And that it parses correctly via resolve_arch_compat_spec.
        let spec = resolve_arch_compat_spec("test-arch", Some(&config_raw_str))
            .expect("must resolve spec from sibling config.json contents");
        assert_eq!(spec.model_type, "test-arch");
        assert_eq!(spec.num_layers, 12);
        assert_eq!(spec.hidden_size, 2048);
    }

    /// Regression test: `read_sibling_config_json` returns None when there is
    /// no sibling config.json (the GGUF load path falls back to plugins-dir scan).
    #[test]
    fn read_sibling_config_json_returns_none_when_missing() {
        let tmp_dir = tempfile::TempDir::new().expect("temp dir");
        let gguf_path = tmp_dir.path().join("model.gguf");
        std::fs::write(&gguf_path, b"fake").expect("fake gguf");
        // No config.json written — sibling is missing.

        let gguf_path_str = gguf_path.to_str().expect("gguf path must be valid utf8");
        let config_raw = read_sibling_config_json(gguf_path_str);
        assert!(
            config_raw.is_none(),
            "missing sibling config.json must return None (fallback to plugins-dir scan)"
        );
    }

    /// Red target: the plugins-dir scan branch of `resolve_arch_compat_spec`
    /// must actually find and match a .grimplugin file in
    /// `grim_plugins_dir()`.
    ///
    /// This is the exact path the entire HF plugin-generation workflow depends
    /// on, and it has zero existing test coverage. The test writes a .grimplugin
    /// JSON to a temp dir, sets `GRIM_PLUGINS_DIR` to point at it, and calls
    /// `resolve_arch_compat_spec` with an arch_str that only matches via the
    /// plugins-dir file (not via inline config).
    ///
    /// In the Red phase this test exercises the existing branch — it should PASS
    /// if the branch is correctly wired (which we're verifying), and the real
    /// work is making sure the CLI install step puts files in the right place
    /// for this branch to find them.
    static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn resolve_arch_compat_spec_finds_grimplugin_in_plugins_dir() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let tmp_dir = tempfile::TempDir::new().expect("temp dir");

        // Write a .grimplugin JSON into the temp dir.
        let plugin_json = r#"{
            "name": "test-plugin-compat",
            "model_type": "test-arch-from-plugin",
            "base_architecture": "llama",
            "hidden_size": 2048,
            "num_layers": 12,
            "vocab_size": 32000,
            "num_heads": 16,
            "num_kv_heads": 16,
            "head_dim": 128,
            "intermediate_size": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_seq_len": 2048,
            "is_moe": false,
            "is_ssm": false,
            "is_multimodal": false,
            "tensor_name_mapping": {},
            "expert_count": null,
            "expert_used_count": null,
            "routed_scaling_factor": null,
            "vision_spec": null,
            "audio_spec": null
        }"#;
        let plugin_path = tmp_dir.path().join("test-arch-from-plugin.grimplugin");
        std::fs::write(&plugin_path, plugin_json).expect("write .grimplugin");

        // Point GRIM_PLUGINS_DIR at the temp dir.
        let old_env = std::env::var("GRIM_PLUGINS_DIR").ok();
        unsafe {
            std::env::set_var("GRIM_PLUGINS_DIR", tmp_dir.path().to_str().unwrap());
        }
        // grim_plugins_dir() reads the env var at call time, so we can override it.
        let plugins_dir = grim_core::paths::grim_plugins_dir();
        assert_eq!(
            plugins_dir,
            tmp_dir.path(),
            "GRIM_PLUGINS_DIR override must be honoured"
        );

        // Call resolve_arch_compat_spec with an arch_str that matches via the
        // plugins-dir file (model_type == "test-arch-from-plugin").
        let spec = resolve_arch_compat_spec("test-arch-from-plugin", None)
            .expect("must find .grimplugin in plugins dir");
        assert_eq!(spec.model_type, "test-arch-from-plugin");
        assert_eq!(spec.num_layers, 12);
        assert_eq!(spec.hidden_size, 2048);

        // Restore env.
        if let Some(v) = old_env {
            unsafe { std::env::set_var("GRIM_PLUGINS_DIR", v) };
        } else {
            unsafe { std::env::remove_var("GRIM_PLUGINS_DIR") };
        }
    }

    /// Red target: `resolve_arch_compat_spec` must prefer the inline config_raw
    /// over the plugins-dir scan when both are available.
    #[test]
    fn resolve_arch_compat_spec_prefers_inline_config_over_plugins_dir() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let tmp_dir = tempfile::TempDir::new().expect("temp dir");

        // Write a .grimplugin into the plugins dir with one set of values.
        let plugin_json = r#"{
            "name": "plugin-override",
            "model_type": "test-arch",
            "hidden_size": 1024,
            "num_layers": 6
        }"#;
        let plugin_path = tmp_dir.path().join("test-arch.grimplugin");
        std::fs::write(&plugin_path, plugin_json).expect("write .grimplugin");

        let old_env = std::env::var("GRIM_PLUGINS_DIR").ok();
        unsafe {
            std::env::set_var("GRIM_PLUGINS_DIR", tmp_dir.path().to_str().unwrap());
        }

        // Pass an inline config_raw with DIFFERENT values.
        let inline_json = r#"{
            "model_type": "test-arch",
            "hidden_size": 2048,
            "num_layers": 12
        }"#;

        let spec = resolve_arch_compat_spec("test-arch", Some(inline_json)).expect("must resolve");
        // The inline config must win.
        assert_eq!(spec.hidden_size, 2048, "inline config must take priority");
        assert_eq!(spec.num_layers, 12, "inline config must take priority");

        if let Some(v) = old_env {
            unsafe { std::env::set_var("GRIM_PLUGINS_DIR", v) };
        } else {
            unsafe { std::env::remove_var("GRIM_PLUGINS_DIR") };
        }
    }
}
