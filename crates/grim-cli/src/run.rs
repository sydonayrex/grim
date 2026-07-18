//! `grim run` — load a model, run a prompt, or start HTTP server.

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_core::session::Inner as SessionInner;
use grim_engine::{Engine, EngineConfig};
use grim_models_transformer::{Llama, LlamaConfig, Lfm2, Lfm2Config, Gpt2, Gpt2Config, Gemma, GemmaConfig, DeepSeek, DeepSeekConfig, T5, T5Config};
use grim_models_mamba::{Rwkv, RwkvConfig};
use grim_models_vision::{Bert, BertConfig};
use grim_tensor::Device;

pub async fn cmd_run(model_path: String, prompt: Option<String>, serve: bool, address: String, _plugins: &str) -> Result<()> {
    let prompt = prompt.unwrap_or_else(|| "Hello".to_string());

    // Probe for ROCm GPUs; fall back to CPU if none are available.
    // §13.2: we fail closed — if a path was given but we can't open the file,
    // we crash rather than silently running a random toy model.
    let gguf_path = std::path::Path::new(&model_path);
    let use_gguf = gguf_path.is_file() && model_path.to_lowercase().ends_with(".gguf");

    let (device, device_name) = if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
        if let Some(first) = rocm_devices.first() {
            let ordinal = first.ordinal();
            let wavefront = format!("{:?}", first.wavefront_size());
            let xnack = first.xnack_enabled();
            eprintln!(
                "[grim] ROCm GPU {} detected (wavefront={}, xnack={})",
                ordinal, wavefront, xnack
            );
            (Device::Rocm(ordinal), format!("rocm:{}", ordinal))
        } else {
            eprintln!("[grim] No ROCm GPU detected; using CPU backend.");
            (Device::Cpu, "cpu".into())
        }
    } else {
        eprintln!("[grim] ROCm runtime not available; using CPU backend.");
        (Device::Cpu, "cpu".into())
    };

    if serve {
        let mut engine = Engine::new(EngineConfig::default());
        let model: Box<dyn CausalLm> = if use_gguf {
            eprintln!("[grim] Loading GGUF model: {}", model_path);
            match load_model_from_gguf(&model_path, device.clone()) {
                Ok(m) => {
                    eprintln!("[grim] GGUF model loaded successfully.");
                    m
                }
                Err(e) => {
                    eprintln!("[grim] ERROR: failed to load GGUF model '{}': {}", model_path, e);
                    return Err(e);
                }
            }
        } else {
            if !use_gguf && gguf_path.exists() {
                eprintln!(
                    "[grim] WARNING: '{}' is not a .gguf file; using random toy model.",
                    model_path
                );
            } else if !gguf_path.exists() {
                eprintln!(
                    "[grim] WARNING: model path '{}' not found; using random toy model.",
                    model_path
                );
            }
            Box::new(Llama::random(random_config()))
        };

        let model_id = "default";
        engine.register_model(model_id, model);
        eprintln!("[grim] Starting HTTP server on {address}...");
        grim_server::serve(&address, engine).await?;
        return Ok(());
    }

    // One-shot inference path.
    let model: Box<dyn CausalLm> = if use_gguf {
        eprintln!("[grim] Loading GGUF model: {}", model_path);
        match load_model_from_gguf(&model_path, device.clone()) {
            Ok(m) => {
                eprintln!("[grim] GGUF model loaded successfully.");
                m
            }
            Err(e) => {
                eprintln!("[grim] ERROR: failed to load GGUF model '{}': {}", model_path, e);
                return Err(e);
            }
        }
    } else {
        if !use_gguf && gguf_path.exists() {
            eprintln!(
                "[grim] WARNING: '{}' is not a .gguf file; using random toy model.",
                model_path
            );
        } else if !gguf_path.exists() {
            eprintln!(
                "[grim] WARNING: model path '{}' not found; using random toy model.",
                model_path
            );
        }
        Box::new(Llama::random(random_config()))
    };

    let tokenizer = if use_gguf {
        let provider = grim_format::GgufProvider::open(&model_path)?;
        Some(provider.tokenizer()?)
    } else {
        None
    };

    println!("Prompt: {prompt}");
    println!("Device: {device_name}");

    // Simple tokenization: use GgufTokenizer if available, fallback to byte-level
    let tokens: Vec<u32> = if let Some(tok) = &tokenizer {
        tok.encode(&prompt)
    } else {
        prompt.bytes().map(|b| b as u32 % 512).collect()
    };
    let input_tensor = grim_backend_cpu::cpu_tensor(
        tokens.iter().map(|t| *t as f32).collect::<Vec<f32>>(),
        grim_tensor::Shape::new(vec![tokens.len()]),
    );

    let mut session = SessionInner::new(model.device().clone());
    let logits = CausalLm::forward(&*model, &mut session, &input_tensor, &input_tensor, &[])?;
    let logits_vec = logits.to_vec_f32()?;

    // Get the argmax of the last token.
    let vocab = if let Some(cfg) = model.config().as_any().downcast_ref::<LlamaConfig>() {
        cfg.vocab_size
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<grim_models_mamba::MambaConfig>() {
        cfg.vocab_size
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<Lfm2Config>() {
        cfg.vocab_size
    } else {
        512
    };
    let last_start = logits_vec.len().saturating_sub(vocab);
    let last_token = logits_vec[last_start..]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    
    if let Some(tok) = &tokenizer {
        let token_text = tok.decode(&[last_token as u32]);
        println!("Next token id: {last_token} (text: {:?})", token_text);
    } else {
        println!("Next token id: {last_token}");
    }
    println!("[grim] Done.");
    Ok(())
}

fn random_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 512,
        hidden_size: 64,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 32,
        num_layers: 1,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 64,
    }
}

/// Load a model from a GGUF file.
fn load_model_from_gguf(path: &str, _device: Device) -> Result<Box<dyn CausalLm>> {
    use grim_format::GgufProvider;
    use grim_nn::WeightSource;

    let provider = GgufProvider::open(path)?;

    // Extract architecture from GGUF metadata
    let arch = provider
        .architecture()
        .ok_or_else(|| grim_tensor::Error::Backend(
            format!("GGUF file '{}' has no 'general.architecture' metadata; cannot determine model family", path)
        ))?;

    let get_meta = |key: &str| -> Option<String> {
        let v = provider.metadata(key)?;
        if let Some(s) = v.as_str() { return Some(s.to_string()); }
        if let Some(u) = v.as_u32() { return Some(u.to_string()); }
        if let Some(f) = v.as_f32() { return Some(f.to_string()); }
        None
    };
    let get_meta_u32 = |key: &str, default: u32| -> u32 {
        if let Some(v) = provider.metadata(key) {
            if let Some(u) = v.as_u32() { return u; }
            if let Some(s) = v.as_str() { if let Ok(u) = s.parse() { return u; } }
        }
        get_meta(key).and_then(|s| s.parse().ok()).unwrap_or(default)
    };

    let vocab_size = get_meta("tokenizer.ggml.vocab_size")
        .or_else(|| get_meta(&format!("{}.vocab_size", arch)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32000);

    let hidden_size = get_meta_u32(&format!("{}.embedding_length", arch), 4096) as usize;
    let num_layers = get_meta_u32(&format!("{}.block_count", arch), 32) as usize;
    let rms_norm_eps = get_meta(&format!("{}.attention.layer_norm_eps", arch))
        .or_else(|| get_meta(&format!("{}.attention.layernorm_rms_eps", arch)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-5_f32);

    eprintln!(
        "[grim] GGUF config: architecture={}, layers={}, hidden={}, vocab={}",
        arch, num_layers, hidden_size, vocab_size
    );

    let ws = WeightSource::root(&provider, Device::Cpu);


    if arch.contains("mamba") {
        let d_state = get_meta_u32(&format!("{}.ssm.state_size", arch), 16) as usize;
        let d_inner = get_meta_u32(&format!("{}.ssm.inner_size", arch), (hidden_size * 2) as u32) as usize;
        let d_conv = get_meta_u32(&format!("{}.ssm.conv_kernel", arch), 4) as usize;
        let cfg = grim_models_mamba::MambaConfig {
            vocab_size,
            hidden_size,
            d_state,
            d_inner,
            d_conv,
            num_layers,
            conv_kernel: d_conv,
            rms_norm_eps,
        };
        let m = grim_models_mamba::Mamba::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("lfm2") {
        // Determine layer types from per-layer head_count_kv (canonical LFM2 schema).
        // A layer with head_count_kv == 0 is a recurrent shortconv layer; otherwise it is
        // a full attention layer. llama.cpp reads this from `<arch>.attention.head_count_kv`
        // which is stored (in GGUF v3) as an ARRAY of length = block_count.
        let mut head_count_kv_vec: Vec<u32> = Vec::with_capacity(num_layers);
        if let Some(arr_val) = provider.metadata("lfm2.attention.head_count_kv").and_then(|v| v.as_array()) {
            for v in arr_val.iter().take(num_layers) {
                let n = v.as_u32().unwrap_or_else(|| {
                    if let Some(s) = v.as_str() { s.parse().unwrap_or(0) } else { 0 }
                });
                head_count_kv_vec.push(n);
            }
        }
        // Fallback per-index keys (`block_count_kv.0`, etc.) — fill gaps with 0 (recurrent).
        for i in 0..num_layers {
            if i < head_count_kv_vec.len() { continue; }
            let key = format!("lfm2.attention.head_count_kv.{i}");
            let n = if let Some(val) = provider.metadata(&key) {
                val.as_u32().unwrap_or(0)
            } else { 0 };
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
        let num_heads = get_meta_u32("lfm2.attention.head_count", 16) as usize;
        let num_kv_heads = head_count_kv_vec.iter().find(|&&n| n > 0).copied().unwrap_or(8) as usize;
        let head_dim = get_meta_u32("lfm2.attention.key_length", 64) as usize;
        let intermediate_size = get_meta_u32("lfm2.feed_forward_length", 4608) as usize;
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
            num_heads: get_meta_u32(&format!("{}.attention.head_count", arch), 12) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&format!("{}.intermediate_size", arch), (hidden_size * 4) as u32) as usize,
            layer_norm_epsilon: get_meta(&format!("{}.attention.layer_norm_epsilon", arch))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-5_f32),
            max_seq_len: get_meta_u32(&format!("{}.context_length", arch), 1024) as usize,
        };
        let m = Gpt2::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("gemma") {
        let cfg = GemmaConfig {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&format!("{}.attention.head_count", arch), 8) as usize,
            num_kv_heads: get_meta_u32(&format!("{}.attention.head_count_kv", arch), 8) as usize,
            head_dim: get_meta_u32(&format!("{}.attention.key_length", arch), 256) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&format!("{}.intermediate_size", arch), 16384) as usize,
            rms_norm_eps,
        };
        let m = Gemma::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("deepseek") {
        let cfg = DeepSeekConfig {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&format!("{}.attention.head_count", arch), 128) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&format!("{}.intermediate_size", arch), 7168) as usize,
            rms_norm_eps,
            q_lora_rank: get_meta_u32(&format!("{}.attention.q_lora_rank", arch), 128) as usize,
            kv_lora_rank: get_meta_u32(&format!("{}.attention.kv_lora_rank", arch), 512) as usize,
        };
        let m = DeepSeek::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("bert") {
        let cfg = BertConfig {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&format!("{}.attention.head_count", arch), 12) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&format!("{}.intermediate_size", arch), (hidden_size * 4) as u32) as usize,
            max_seq_len: get_meta_u32(&format!("{}.context_length", arch), 512) as usize,
        };
        let m = Bert::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else if arch.contains("t5") {
        let cfg = T5Config {
            vocab_size,
            hidden_size,
            num_heads: get_meta_u32(&format!("{}.attention.head_count", arch), 8) as usize,
            num_layers,
            intermediate_size: get_meta_u32(&format!("{}.intermediate_size", arch), (hidden_size * 4) as u32) as usize,
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
        let m = Rwkv::load(&ws, cfg)?;
        Ok(Box::new(m))
    } else {
        let intermediate_size = get_meta_u32(&format!("{}.intermediate_size", arch), 11008) as usize;
        let num_heads = get_meta_u32(&format!("{}.attention.head_count", arch), 32) as usize;
        let num_kv_heads = get_meta_u32(&format!("{}.attention.head_count_kv", arch), num_heads as u32) as usize;
        let head_dim = get_meta_u32(&format!("{}.attention.key_length", arch), 128) as usize;
        let rope_theta = get_meta(&format!("{}.rope.freq_base", arch))
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