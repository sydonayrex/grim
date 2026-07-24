//! `grim run` — load a model, run a prompt, or start HTTP server.

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_core::session::Inner as SessionInner;
use grim_core::sampler::{SamplingParams, Sampler};
use grim_engine::{Engine, EngineConfig, model_loader::{load_model_from_grim, load_model_from_safetensors}};
use grim_models_transformer::{Llama, LlamaConfig, Lfm2, Lfm2Config, Gpt2, Gpt2Config, Gemma, GemmaConfig, DeepSeek, DeepSeekConfig, T5, T5Config};
use grim_models_mamba::{Rwkv, RwkvConfig};
use grim_models_vision::{Bert, BertConfig};
use std::sync::Arc;
use grim_tensor::BackendDevice;
use grim_tensor::Device;
use grim_backend_cpu;
#[cfg(feature = "rocm")]
use grim_backend_rocm;
use crate::catalog::resolve_model_path;

pub async fn cmd_run(
    model_path: String,
    prompt: Option<String>,
    serve: bool,
    address: String,
    _plugins: &str,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    max_tokens: usize,
    seed: u64,
    repeat_penalty: f32,
) -> Result<()> {
    let prompt = prompt.unwrap_or_else(|| "Hello".to_string());

    // Resolve model name to actual file path
    let resolved_path = resolve_model_path(&model_path)
        .or_else(|| {
            // Accept a direct file path if it exists on disk.
            let p = std::path::Path::new(&model_path);
            if p.exists() { Some(p.to_path_buf()) } else { None }
        })
        .ok_or_else(|| grim_core::error::Error::Config(
            format!("Model '{}' not found. Run 'grim pull {}' to download it.",
                model_path, model_path)
        ))?;
    let model_path_str = resolved_path.to_string_lossy().to_string();
    eprintln!("[grim] Resolved model path: {}", model_path_str);

    // Probe for ROCm GPUs; fall back to CPU if none are available.
    // §13.2: we fail closed — if a path was given but we can't open the file,
    // we crash rather than silently running a random toy model.
    let path_obj = std::path::Path::new(&model_path_str);
    let use_gguf = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".gguf");
    let use_grim = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".grim");
    let use_safetensors = path_obj.is_file() && (model_path_str.to_lowercase().ends_with(".safetensors") || model_path_str.to_lowercase().ends_with(".bin"));

    let (device, device_name) = if let Ok(s) = std::env::var("GRIM_FORCE_DEVICE") {
        match s.as_str() {
            "cuda" => {
                if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
                    if let Some(first) = cuda_devices.first() {
                        (Device::Cuda(first.ordinal()), format!("cuda:{}", first.ordinal()))
                    } else {
                        (Device::Cpu, "cpu".into())
                    }
                } else {
                    (Device::Cpu, "cpu".into())
                }
            }
            "rocm" => {
                if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
                    if let Some(first) = rocm_devices.first() {
                        (Device::Rocm(first.ordinal()), format!("rocm:{}", first.ordinal()))
                    } else {
                        (Device::Cpu, "cpu".into())
                    }
                } else {
                    (Device::Cpu, "cpu".into())
                }
            }
            "cpu" => (Device::Cpu, "cpu".into()),
            _ => (Device::Cpu, "cpu".into()),
        }
    } else if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
        if let Some(first) = rocm_devices.first() {
            let ordinal = first.ordinal();
            let wavefront = format!("{:?}", first.wavefront_size());
            let xnack = first.xnack_enabled();
            eprintln!(
                "[grim] ROCm GPU {} detected (wavefront={}, xnack={})",
                ordinal, wavefront, xnack
            );
            (Device::Rocm(ordinal), format!("rocm:{}", ordinal))
        } else if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
            if let Some(first) = cuda_devices.first() {
                let ordinal = first.ordinal();
                eprintln!("[grim] CUDA GPU {} detected", ordinal);
                (Device::Cuda(ordinal), format!("cuda:{}", ordinal))
            } else {
                eprintln!("[grim] No GPU detected; using CPU backend.");
                (Device::Cpu, "cpu".into())
            }
        } else {
            eprintln!("[grim] No ROCm GPU detected; using CPU backend.");
            (Device::Cpu, "cpu".into())
        }
    } else if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
        if let Some(first) = cuda_devices.first() {
            let ordinal = first.ordinal();
            eprintln!("[grim] CUDA GPU {} detected", ordinal);
            (Device::Cuda(ordinal), format!("cuda:{}", ordinal))
        } else {
            eprintln!("[grim] No GPU detected; using CPU backend.");
            (Device::Cpu, "cpu".into())
        }
    } else {
        eprintln!("[grim] GPU runtime not available; using CPU backend.");
        (Device::Cpu, "cpu".into())
    };

    if serve {
        let mut engine = Engine::new(EngineConfig::default());
        let model: Box<dyn CausalLm> = if use_gguf {
            eprintln!("[grim] Loading GGUF model: {}", model_path_str);
            match load_model_from_gguf(&model_path_str, device.clone()) {
                Ok(m) => {
                    eprintln!("[grim] GGUF model loaded successfully.");
                    m
                }
                Err(e) => {
                    eprintln!("[grim] ERROR: failed to load GGUF model '{}': {}", model_path_str, e);
                    return Err(e);
                }
            }
        } else if use_grim {
            eprintln!("[grim] Loading GRIM model: {}", model_path_str);
            match load_model_from_grim(&model_path_str, device.clone()) {
                Ok(m) => {
                    eprintln!("[grim] GRIM model loaded successfully.");
                    m
                }
                Err(e) => {
                    eprintln!("[grim] ERROR: failed to load GRIM model '{}': {}", model_path_str, e);
                    return Err(e);
                }
            }
        } else if use_safetensors {
            eprintln!("[grim] Loading safetensors model: {}", model_path_str);
            match load_model_from_safetensors(&model_path_str, device.clone()) {
                Ok(m) => {
                    eprintln!("[grim] safetensors model loaded successfully.");
                    m
                }
                Err(e) => {
                    eprintln!("[grim] ERROR: failed to load safetensors model '{}': {}", model_path_str, e);
                    return Err(e);
                }
            }
        } else {
            // Never silently run a toy model — error loudly so the user
            // knows they need to pull a real model first.
            return Err(grim_core::error::Error::Config(format!(
                "Model '{}' is not a valid .gguf, .grim, or .safetensors file or does not exist. \
                 Run 'grim pull <name>' to download a model first.",
                model_path_str
            )));
        };

        let model_id = "default";
        engine.register_model(model_id, model);
        eprintln!("[grim] Starting HTTP server on {address}...");
        let serve_model_path = Some(std::path::PathBuf::from(&model_path_str));
        grim_server::serve(&address, engine, serve_model_path).await?;
        return Ok(());
    }

    // One-shot inference path with generation loop.
    let model: Box<dyn CausalLm> = if use_gguf {
        eprintln!("[grim] Loading GGUF model: {}", model_path_str);
        match load_model_from_gguf(&model_path_str, device.clone()) {
            Ok(m) => {
                eprintln!("[grim] GGUF model loaded successfully.");
                m
            }
            Err(e) => {
                eprintln!("[grim] ERROR: failed to load GGUF model '{}': {}", model_path_str, e);
                return Err(e);
            }
        }
    } else if use_grim {
        eprintln!("[grim] Loading GRIM model: {}", model_path_str);
        match load_model_from_grim(&model_path_str, device.clone()) {
            Ok(m) => {
                eprintln!("[grim] GRIM model loaded successfully.");
                m
            }
            Err(e) => {
                eprintln!("[grim] ERROR: failed to load GRIM model '{}': {}", model_path_str, e);
                return Err(e);
            }
        }
    } else if use_safetensors {
        eprintln!("[grim] Loading safetensors model: {}", model_path_str);
        match load_model_from_safetensors(&model_path_str, device.clone()) {
            Ok(m) => {
                eprintln!("[grim] safetensors model loaded successfully.");
                m
            }
            Err(e) => {
                eprintln!("[grim] ERROR: failed to load safetensors model '{}': {}", model_path_str, e);
                return Err(e);
            }
        }
    } else {
        // Fail loudly — never generate from a toy model.
        return Err(grim_core::error::Error::Config(format!(
            "Model '{}' is not a valid .gguf, .grim, or .safetensors file or could not be found.\n\
             Run 'grim pull <name>' to download a model, or provide an\n\
             explicit path to a .gguf, .grim, or .safetensors file.",
            model_path_str
        )));
    };

    let tokenizer = if use_gguf {
        let provider = grim_format::GgufProvider::open(&model_path_str)?;
        Some(provider.tokenizer()?)
    } else if use_grim {
        // For .grim files, get tokenizer from sibling .gguf file
        let gguf_path = path_obj.with_extension("gguf");
        if gguf_path.exists() {
            let provider = grim_format::GgufProvider::open(gguf_path.to_str().unwrap())?;
            Some(provider.tokenizer()?)
        } else {
            None
        }
    } else if use_safetensors {
        // For safetensors, load tokenizer from the sibling tokenizer.json
        // (HuggingFace format) in the same directory.
        let dir = path_obj.parent().unwrap_or(std::path::Path::new("."));
        let tokenizer_json = dir.join("tokenizer.json");
        if tokenizer_json.exists() {
            grim_format::GgufTokenizer::from_hf_json(tokenizer_json.to_str().unwrap()).ok()
        } else {
            None
        }
    } else {
        None
    };

    // Create sampler based on parameters
    let sampling_params = SamplingParams {
        temperature,
        top_p,
        top_k,
        repeat_penalty,
    };
    let sampler: Box<dyn Sampler> = sampling_params.into_sampler(seed);

    // Tokenize prompt
    let mut tokens: Vec<u32> = if let Some(tok) = &tokenizer {
        let mut ids = Vec::new();

        // Prepend BOS token for models that expect it (e.g. <|startoftext|> for LFM2).
        let bos_candidates = ["<|startoftext|>", "<s>", "<|im_start|>"];
        for bos in &bos_candidates {
            if let Some(&id) = tok.token_to_id.get(*bos) {
                ids.push(id);
                break;
            }
        }

        ids.extend(tok.encode(&prompt));
        eprintln!("[grim] Encoded prompt: {} tokens: {:?}", ids.len(), ids);
        let decoded: Vec<&str> = ids.iter()
            .filter_map(|&id| tok.tokens.get(id as usize).map(|s| s.as_str()))
            .collect();
        eprintln!("[grim] Decoded tokens: {:?}", decoded);
        ids
    } else {
        prompt.bytes().map(|b| b as u32 % 512).collect()
    };

    // Determine vocab size
    let vocab = if let Some(cfg) = model.config().as_any().downcast_ref::<LlamaConfig>() {
        cfg.vocab_size
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<grim_models_mamba::MambaConfig>() {
        cfg.vocab_size
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<Lfm2Config>() {
        cfg.vocab_size
    } else {
        512
    };

    println!("Prompt: {prompt}");
    println!("Device: {device_name}");
    println!("Sampling: temp={}, top_p={}, top_k={}, max_tokens={}, seed={}", 
             temperature, top_p, top_k, max_tokens, seed);
    print!("\nResponse: ");
    use std::io::Write;
    std::io::stdout().flush().unwrap();

    let mut session = SessionInner::new(model.device().clone());
    let mut generated = 0;
    let mut history: Vec<u32> = Vec::new();
    let mut first_pass = true;

    // Generation loop
    while generated < max_tokens {
        // First pass: prefill with all prompt tokens to populate KV/conv caches.
        // Subsequent passes: incremental decode — only pass the latest token
        // so the caches (KV for attention, state for ShortConv) accumulate
        // correctly instead of seeing the same tokens repeated.
        let input_ids: Vec<f32> = if first_pass {
            first_pass = false;
            tokens.iter().map(|t| *t as f32).collect()
        } else {
            vec![*tokens.last().unwrap() as f32]
        };

        // Build tensor from the selected token(s)
        let shape = grim_tensor::Shape::new(vec![input_ids.len()]);
        let float_tokens = input_ids;
        let dtype = grim_tensor::dtype::DType::F32;
        let storage: Arc<dyn grim_tensor::BackendStorage> = match device {
            grim_tensor::Device::Cpu => {
                let dev = grim_backend_cpu::CpuDevice::new();
                Arc::from(dev.from_cpu(&float_tokens, &shape, dtype.clone())?)
            }
            grim_tensor::Device::Cuda(ordinal) => {
                let dev = grim_backend_cuda::CudaDevice::new(ordinal);
                Arc::from(dev.from_cpu(&float_tokens, &shape, dtype.clone())?)
            }
            grim_tensor::Device::Rocm(ordinal) => {
                let dev = grim_backend_rocm::RocmDevice::new(ordinal);
                Arc::from(dev.from_cpu(&float_tokens, &shape, dtype.clone())?)
            }
            grim_tensor::Device::Vulkan => {
                let dev = grim_backend_vulkan::VulkanDevice::new();
                Arc::from(dev.from_cpu(&float_tokens, &shape, dtype.clone())?)
            }
            grim_tensor::Device::Metal(ordinal) => {
                let dev = grim_backend_metal::MetalDevice::try_new(ordinal)?;
                Arc::from(dev.from_cpu(&float_tokens, &shape, dtype.clone())?)
            }
        };
        let input_tensor = grim_tensor::Tensor::new(
            storage,
            shape,
            dtype,
            grim_tensor::dtype::QuantProvenance::default(),
            device.clone(),
        );

        // Forward pass
        let logits = CausalLm::forward(&*model, &mut session, &input_tensor, &input_tensor, &[])?;
        
        // Get logits for the last token position only
        let logits_vec = logits.to_vec_f32()?;
        let last_start = logits_vec.len().saturating_sub(vocab);
        let last_logits = &logits_vec[last_start..];

        // Build a single-position logits tensor containing only the last-token
        // logits, so the sampler sees exactly the distribution for the next
        // token (not every position in the sequence). This fixes the bug where
        // `sampler.sample(&logits, &history)` sees logits for the wrong slot
        // and returns a non-final-position argmax.
        let last_shape = grim_tensor::Shape::new(vec![vocab]);
        let last_storage: Arc<dyn grim_tensor::BackendStorage> = match device {
            grim_tensor::Device::Cpu => {
                let dev = grim_backend_cpu::CpuDevice::new();
                Arc::from(dev.from_cpu(last_logits, &last_shape, grim_tensor::dtype::DType::F32)?)
            }
            grim_tensor::Device::Cuda(ordinal) => {
                let dev = grim_backend_cuda::CudaDevice::new(ordinal);
                Arc::from(dev.from_cpu(last_logits, &last_shape, grim_tensor::dtype::DType::F32)?)
            }
            grim_tensor::Device::Rocm(ordinal) => {
                let dev = grim_backend_rocm::RocmDevice::new(ordinal);
                Arc::from(dev.from_cpu(last_logits, &last_shape, grim_tensor::dtype::DType::F32)?)
            }
            grim_tensor::Device::Vulkan => {
                let dev = grim_backend_vulkan::VulkanDevice::new();
                Arc::from(dev.from_cpu(last_logits, &last_shape, grim_tensor::dtype::DType::F32)?)
            }
            grim_tensor::Device::Metal(ordinal) => {
                let dev = grim_backend_metal::MetalDevice::try_new(ordinal)?;
                Arc::from(dev.from_cpu(last_logits, &last_shape, grim_tensor::dtype::DType::F32)?)
            }
        };
        let last_logits_tensor = grim_tensor::Tensor::new(
            last_storage,
            last_shape,
            grim_tensor::dtype::DType::F32,
            grim_tensor::dtype::QuantProvenance::default(),
            device.clone(),
        );

        // Sample next token from the *last-position* logits, not the full tensor.
        let next_token = sampler.sample(&last_logits_tensor, &history)?;
        
        // Decode and print token
        if let Some(tok) = &tokenizer {
            let token_text = tok.decode(&[next_token]);
            print!("{}", token_text);
            std::io::stdout().flush().unwrap();
        } else {
            // Fallback: print as raw token
            print!("{} ", next_token);
            std::io::stdout().flush().unwrap();
        }

        // Update state
        tokens.push(next_token);
        history.push(next_token);
        generated += 1;

        // Stop on EOS token: in GGUF the canonical EOS id is `vocab_size - 1`.
        // We deliberately do NOT kill on token ids 0/1/2 — those are BOS/pad/unk
        // for some tokenizers and legitimate content tokens for others; killing
        // on them prematurely truncates output for LFM2-style models.
        let vocab_u32 = vocab as u32;
        if next_token >= vocab_u32.saturating_sub(1) {
            break;
        }
    }

    println!("\n[grim] Done. Generated {} tokens.", generated);
    Ok(())
}

/// Load a model from a GGUF file.
pub fn load_model_from_gguf(path: &str, device: Device) -> Result<Box<dyn CausalLm>> {
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

    let ws = WeightSource::root(&provider, device);


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
        let rope_theta = provider.metadata("lfm2.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(1000000.0f32);
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