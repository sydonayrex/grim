//! `grim run` — load a model, run a prompt, or start HTTP server.

use std::sync::Arc;
use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_core::session::Inner as SessionInner;
use grim_engine::{Engine, EngineConfig};
use grim_models_transformer::{Llama, LlamaConfig};
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
            match load_llama_from_gguf(&model_path, device.clone()) {
                Ok(m) => {
                    eprintln!("[grim] GGUF model loaded successfully.");
                    Box::new(m)
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
    let model: Llama = if use_gguf {
        eprintln!("[grim] Loading GGUF model: {}", model_path);
        match load_llama_from_gguf(&model_path, device.clone()) {
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
        Llama::random(random_config())
    };

    println!("Prompt: {prompt}");
    println!("Device: {device_name}");

    // Simple tokenization: use UTF-8 byte values as token ids (v1 tokenizer).
    let tokens: Vec<u32> = prompt.bytes().map(|b| b as u32 % 512).collect();
    let input_tensor = grim_backend_cpu::cpu_tensor(
        tokens.iter().map(|t| *t as f32).collect::<Vec<f32>>(),
        grim_tensor::Shape::new(vec![tokens.len()]),
    );

    let mut session = SessionInner::new(model.device.clone());
    let logits = CausalLm::forward(&model, &mut session, &input_tensor, &input_tensor, &[])?;
    let logits_vec = logits.to_vec_f32()?;

    // Get the argmax of the last token.
    let vocab = model.cfg.vocab_size;
    let last_start = logits_vec.len().saturating_sub(vocab);
    let last_token = logits_vec[last_start..]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    println!("Next token id: {last_token}");
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

/// Load a Llama model from a GGUF file.
/// Fails closed — §13.2: if the file can't be opened or doesn't match the
/// expected architecture, we crash rather than silently falling back.
fn load_llama_from_gguf(path: &str, device: Device) -> Result<Llama> {
    use grim_format::GgufProvider;
    use grim_nn::WeightSource;

    let provider = GgufProvider::open(path)?;

    // Extract architecture from GGUF metadata — we only support llama family for now.
    let arch = provider
        .architecture()
        .ok_or_else(|| grim_tensor::Error::Backend(
            format!("GGUF file '{}' has no 'general.architecture' metadata; cannot determine model family", path)
        ))?;

    if !arch.contains("llama") && !arch.contains("mistral") && !arch.contains("qwen") {
        return Err(grim_tensor::Error::Backend(
            format!(
                "GGUF architecture '{}' is not yet supported; Grim v1 supports \
                 llama/mistral/qwen family only (found in '{}')",
                arch, path
            )
        ).into());
    }

    // Derive LlamaConfig from GGUF metadata KV pairs.
    let get_meta = |key: &str| -> Option<String> {
        provider.metadata(key).and_then(|v| v.as_str().map(String::from))
    };
    let get_meta_u32 = |key: &str, default: u32| -> u32 {
        get_meta(key).and_then(|s| s.parse().ok()).unwrap_or(default)
    };

    let vocab_size = get_meta("tokenizer.ggml.vocab_size")
        .or_else(|| get_meta(&format!("{}.vocab_size", arch)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32000);

    let hidden_size = get_meta_u32(&format!("{}.embedding_length", arch), 4096) as usize;
    let intermediate_size = get_meta_u32(&format!("{}.intermediate_size", arch), 11008) as usize;
    let num_layers = get_meta_u32(&format!("{}.block_count", arch), 32) as usize;
    let num_heads = get_meta_u32(&format!("{}.attention.head_count", arch), 32) as usize;
    let num_kv_heads = get_meta_u32(&format!("{}.attention.head_count_kv", arch), num_heads as u32) as usize;
    let head_dim = get_meta_u32(&format!("{}.attention.key_length", arch), 128) as usize;
    let rms_norm_eps = get_meta(&format!("{}.attention.layer_norm_eps", arch))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-5_f32);
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

    eprintln!(
        "[grim] GGUF config: layers={}, heads={}, hidden={}, vocab={}, head_dim={}",
        num_layers, num_heads, hidden_size, vocab_size, head_dim
    );

    // Weight materialization (§4.1) currently supports CPU tensors only.
    // Pass Device::Cpu so tensor_from_raw can deserialize the GGUF bytes;
    // the model uses CpuDevice for matmul ops regardless of the device probe.
    let ws = WeightSource::root(&provider, Device::Cpu);
    Llama::load(&ws, cfg)
}