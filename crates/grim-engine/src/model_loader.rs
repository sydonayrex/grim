//! Model loading utilities for GGUF files.

use grim_core::error::{Error, Result};
use grim_core::model::CausalLm;
use grim_format::{GgufProvider, gguf::GgufValue};
use grim_models_mamba::{Mamba, MambaConfig};
use grim_models_transformer::{
    Llama, LlamaConfig, Lfm2, Lfm2Config, Gpt2, Gpt2Config, Gemma, GemmaConfig, DeepSeek, DeepSeekConfig, T5, T5Config
};
use grim_models_vision::{Bert, BertConfig};
use grim_models_mamba::RwkvConfig;
use grim_nn::WeightSource;
use grim_tensor::{Device, Shape};

/// Helper function to get metadata as string from GGUF provider
fn get_meta_str(provider: &GgufProvider, key: &str) -> Option<String> {
    let v: Option<&GgufValue> = provider.metadata(key);
    let v: &GgufValue = match v {
        Some(val) => val,
        None => return None,
    };
    if let Some(s) = v.as_str() { return Some(s.to_string()); }
    if let Some(u) = v.as_u32() { return Some(u.to_string()); }
    if let Some(f) = v.as_f32() { return Some(f.to_string()); }
    None
}

/// Helper function to get metadata as u32 from GGUF provider
fn get_meta_u32(provider: &GgufProvider, key: &str, default: u32) -> u32 {
    let v: Option<&GgufValue> = provider.metadata(key);
    if let Some(v) = v {
        if let Some(u) = v.as_u32() { return u; }
        if let Some(s) = v.as_str() { if let Ok(u) = s.parse::<u32>() { return u; } }
    }
    get_meta_str(provider, key).and_then(|s| s.parse::<u32>().ok()).unwrap_or(default)
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

/// Load a model from a GGUF file.
pub fn load_model_from_gguf(path: &str, device: Device) -> Result<Box<dyn CausalLm>> {
    let provider = GgufProvider::open(path)?;

    // Extract architecture from GGUF metadata
    let arch = provider
        .architecture()
        .ok_or_else(|| Error::Config(
            format!("GGUF file '{}' has no 'general.architecture' metadata; cannot determine model family", path)
        ))?;

    // Extract common metadata
    let vocab_size = get_meta_str(&provider, "tokenizer.ggml.vocab_size")
        .or_else(|| get_meta_str(&provider, &format!("{}.vocab_size", arch)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32000);

    let hidden_size = get_meta_u32(&provider, &format!("{}.embedding_length", arch), 4096) as usize;
    let num_layers = get_meta_u32(&provider, &format!("{}.block_count", arch), 32) as usize;
    let rms_norm_eps = get_meta_str(&provider, &format!("{}.attention.layer_norm_eps", arch))
        .or_else(|| get_meta_str(&provider, &format!("{}.attention.layernorm_rms_eps", arch)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-5_f32);

    eprintln!(
        "[grim] GGUF config: architecture={}, layers={}, hidden={}, vocab={}",
        arch, num_layers, hidden_size, vocab_size
    );

    let ws = WeightSource::root(&provider, device);

    if arch.contains("mamba") {
        let d_state = get_meta_u32(&provider, &format!("{}.ssm.state_size", arch), 16) as usize;
        let d_inner = get_meta_u32(&provider, &format!("{}.ssm.inner_size", arch), (hidden_size * 2) as u32) as usize;
        let d_conv = get_meta_u32(&provider, &format!("{}.ssm.conv_kernel", arch), 4) as usize;
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
    } else if arch.contains("lfm2") {
        // Determine layer types from per-layer head_count_kv (canonical LFM2 schema).
        // A layer with head_count_kv == 0 is a recurrent shortconv layer; otherwise it is
        // a full attention layer. llama.cpp reads this from `<arch>.attention.head_count_kv`
        // which is stored (in GGUF v3) as an ARRAY of length = block_count.
        let mut head_count_kv_vec: Vec<u32> = Vec::with_capacity(num_layers);
        if let Some(arr_val) = get_meta_array(&provider, "lfm2.attention.head_count_kv") {
            for v in arr_val.iter().take(num_layers) {
                let v: &grim_format::gguf::GgufValue = v;
                let n: u32 = v.as_u32().unwrap_or_else(|| {
                    if let Some(s) = v.as_str() { s.parse::<u32>().unwrap_or(0u32) } else { 0u32 }
                });
                head_count_kv_vec.push(n);
            }
        }
        // Fallback per-index keys (`block_count_kv.0`, etc.) — fill gaps with 0 (recurrent).
        for i in 0..num_layers {
            if i < head_count_kv_vec.len() { continue; }
            let key = format!("lfm2.attention.head_count_kv.{i}");
            let n: u32 = if let Some(val) = provider.metadata(&key) {
                let val: &grim_format::gguf::GgufValue = val;
                val.as_u32().unwrap_or(0u32)
            } else { 0u32 };
            if (i + 1) > head_count_kv_vec.len() {
                head_count_kv_vec.resize(i + 1, 0);
            }
            head_count_kv_vec[i] = n;
        }
        // If array shorter than num_layers, extend with 0 (recurrent default).
        head_count_kv_vec.resize(num_layers, 0);
        // is_recr[k] == true means shortconv (recurrent) layer.
        let is_recr: Vec<bool> = head_count_kv_vec.iter().map(|&n| n == 0).collect();
        eprintln!("[grim] LFM2 layer-type map (T=shortconv): {:?}", is_recr);
        // shortconv kernel size is fixed at 3 in canonical LFM2 (conv.weight shape = [3, n_embd]).
        let n_shortconv_l_cache = 3usize;
        let num_heads = get_meta_u32(&provider, "lfm2.attention.head_count", 16) as usize;
        let num_kv_heads = head_count_kv_vec.iter().find(|&&n| n > 0).copied().unwrap_or(8) as usize;
        let head_dim = get_meta_u32(&provider, "lfm2.attention.key_length", 64) as usize;
        let intermediate_size = get_meta_u32(&provider, "lfm2.feed_forward_length", 4608) as usize;
        let cfg = Lfm2Config {
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            intermediate_size,
            rms_norm_eps,
            n_shortconv_l_cache,
            is_recr: is_recr.clone(),
        };
        let m = Lfm2::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("gpt2") {
        let cfg = Gpt2Config {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&provider, &format!("{}.attention.head_count", arch), 12) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&provider, &format!("{}.intermediate_size", arch), (hidden_size * 4) as u32) as usize,
            layer_norm_epsilon: get_meta_str(&provider, &format!("{}.attention.layer_norm_epsilon", arch))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-5_f32),
            max_seq_len: get_meta_u32(&provider, &format!("{}.context_length", arch), 1024) as usize,
        };
        let m = Gpt2::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("gemma") {
        let cfg = GemmaConfig {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&provider, &format!("{}.attention.head_count", arch), 8) as usize,
            num_kv_heads: get_meta_u32(&provider, &format!("{}.attention.head_count_kv", arch), 8) as usize,
            head_dim: get_meta_u32(&provider, &format!("{}.attention.key_length", arch), 256) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&provider, &format!("{}.intermediate_size", arch), 16384) as usize,
            rms_norm_eps,
        };
        let m = Gemma::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("deepseek") {
        let cfg = DeepSeekConfig {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&provider, &format!("{}.attention.head_count", arch), 128) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&provider, &format!("{}.intermediate_size", arch), 7168) as usize,
            rms_norm_eps,
            q_lora_rank: get_meta_u32(&provider, &format!("{}.attention.q_lora_rank", arch), 128) as usize,
            kv_lora_rank: get_meta_u32(&provider, &format!("{}.attention.kv_lora_rank", arch), 512) as usize,
        };
        let m = DeepSeek::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("bert") {
        let cfg = BertConfig {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&provider, &format!("{}.attention.head_count", arch), 12) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&provider, &format!("{}.intermediate_size", arch), (hidden_size * 4) as u32) as usize,
            max_seq_len: get_meta_u32(&provider, &format!("{}.context_length", arch), 512) as usize,
        };
        let m = Bert::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("t5") {
        let cfg = T5Config {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&provider, &format!("{}.attention.head_count", arch), 8) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&provider, &format!("{}.intermediate_size", arch), (hidden_size * 4) as u32) as usize,
            rms_norm_eps,
        };
        let m = T5::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("rwkv") {
        let cfg = RwkvConfig {
            vocab_size,
            hidden_size,
            num_layers,
        };
        let m = grim_models_mamba::Rwkv::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else {
        let intermediate_size = get_meta_u32(&provider, &format!("{}.intermediate_size", arch), 11008) as usize;
        let num_heads = get_meta_u32(&provider, &format!("{}.attention.head_count", arch), 32) as usize;
        let num_kv_heads = get_meta_u32(&provider, &format!("{}.attention.head_count_kv", arch), num_heads as u32) as usize;
        let head_dim = get_meta_u32(&provider, &format!("{}.attention.key_length", arch), 128) as usize;
        let rope_theta = get_meta_str(&provider, &format!("{}.rope.freq_base", arch))
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000.0_f32);
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
            max_seq_len: 2048,
        };
        let m = Llama::load(&ws, cfg)?;
        Ok(Box::new(m))
    }
}

/// Convenience wrapper: detect the best available device and load a GGUF model.
///
/// Device priority: ROCm → CUDA → CPU.  This is the entry point called by
/// `grim-server`'s on-demand model loader so callers don't need to manage
/// device selection themselves.
pub fn load_from_path(path: &str) -> Result<Box<dyn CausalLm>> {
    // Attempt ROCm first (AMD GPU — primary grim target).
    if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
        if let Some(first) = rocm_devices.first() {
            eprintln!("[model_loader] Using ROCm device {}", first.ordinal());
            return load_model_from_gguf(path, Device::Rocm(first.ordinal()));
        }
    }
    // Fall back to CUDA.
    if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
        if let Some(first) = cuda_devices.first() {
            eprintln!("[model_loader] Using CUDA device {}", first.ordinal());
            return load_model_from_gguf(path, Device::Cuda(first.ordinal()));
        }
    }
    // CPU fallback.
    eprintln!("[model_loader] No GPU detected; using CPU.");
    load_model_from_gguf(path, Device::Cpu)
}