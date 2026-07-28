//! `grim run` — load a model, run a prompt, or start HTTP server.

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_core::session::Inner as SessionInner;
use grim_core::sampler::{SamplingParams, Sampler};
use grim_engine::{Engine, EngineConfig, model_loader::{load_model_from_gguf, load_model_from_grim, load_model_from_safetensors}};
use grim_models_transformer::{Lfm2Config, LlamaConfig};
use grim_tensor::Device;
use std::sync::Arc;
use grim_tensor::BackendDevice;
use grim_backend_cpu;
#[cfg(feature = "rocm")]
use grim_backend_rocm;
use grim_format::GgufTokenizer;
use crate::catalog::resolve_model_path;

/// Auto-detect the best available device.  Extracted from `cmd_run` so
/// the interactive REPL (B.4) can probe once and reuse the result.
fn probe_device() -> (Device, String) {
    if let Ok(s) = std::env::var("GRIM_FORCE_DEVICE") {
        match s.as_str() {
            "cuda" => {
                if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
                    if let Some(first) = cuda_devices.first() {
                        return (Device::Cuda(first.ordinal()), format!("cuda:{}", first.ordinal()));
                    }
                }
                (Device::Cpu, "cpu".into())
            }
            "rocm" => {
                if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
                    if let Some(first) = rocm_devices.first() {
                        return (Device::Rocm(first.ordinal()), format!("rocm:{}", first.ordinal()));
                    }
                }
                (Device::Cpu, "cpu".into())
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
    }
}

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

	    let (device, device_name) = probe_device();

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

        // If the tokenizer carries a Jinja chat template, render the
        // single-turn prompt through it for instruction-tuned models.
        // Otherwise fall back to raw prompt + best-effort BOS.
        let prompt_text = if tok.chat_template.is_some() {
            let messages = vec![grim_format::ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            }];
            grim_format::render_messages_or_last(tok, &messages)
        } else {
            // Prepend BOS token for models that expect it (e.g. <|startoftext|> for LFM2).
            let bos_candidates = ["<|startoftext|>", "<s>", "<|im_start|>"];
            for bos in &bos_candidates {
                if let Some(&id) = tok.token_to_id.get(*bos) {
                    ids.push(id);
                    break;
                }
            }
            prompt.clone()
        };

        ids.extend(tok.encode(&prompt_text));
        eprintln!("[grim] Encoded prompt: {} tokens: {:?}", ids.len(), ids);
        let decoded: Vec<&str> = ids.iter()
            .filter_map(|&id| tok.tokens.get(id as usize).map(|s| s.as_str()))
            .collect();
        eprintln!("[grim] Decoded tokens: {:?}", decoded);
        ids
    } else {
        prompt.bytes().map(|b| b as u32 % 512).collect()
    };

    // Determine vocab size — fall back to tokenizer vocab length if model
    // config type is unknown (GPT2, Gemma, DeepSeek, etc.)
    let vocab: usize = if let Some(cfg) = model.config().as_any().downcast_ref::<LlamaConfig>() {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<grim_models_mamba::MambaConfig>() {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<Lfm2Config>() {
        cfg.vocab_size as usize
    } else if let Some(tok) = &tokenizer {
        tok.tokens.len()
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
    let mut generated_tokens: Vec<u32> = Vec::new();

    // Generation loop
    while generated < max_tokens {
        // First pass: prefill with all prompt tokens to populate KV/conv caches.
        // Subsequent passes: incremental decode — only pass the latest token
        // so the caches (KV for attention, state for ShortConv) accumulate
        // correctly instead of seeing the same tokens repeated.
        // HIGH-3: save first_pass BEFORE the input_ids block mutates it,
        // otherwise the positions check below always reads `false` and the
        // first forward pass uses wrong positions (decode-style single value
        // instead of sequential 0..n).
        let is_prefill = first_pass;
        let input_ids: Vec<f32> = if first_pass {
            first_pass = false;
            tokens.iter().map(|t| *t as f32).collect()
        } else {
            vec![*tokens.last().unwrap() as f32]
        };

        // Build tensor from the selected token(s)
        let n_tokens = input_ids.len();
        let shape = grim_tensor::Shape::new(vec![n_tokens]);
        let float_tokens = input_ids;
        let dtype = grim_tensor::dtype::DType::F32;
        let input_tensor = build_tensor(&float_tokens, &shape, &device)?;

        // Forward pass
        // CRIT-1: Need to pass proper positions tensor, not the same as input_ids
        let positions: Vec<f32> = if is_prefill {
            (0..n_tokens).map(|i| i as f32).collect()
        } else {
            vec![n_tokens as f32 - 1.0]
        };
        let pos_shape = grim_tensor::Shape::new(vec![positions.len()]);
        let positions_tensor = build_tensor(&positions, &pos_shape, &device)?;
        
        let logits = CausalLm::forward(&*model, &mut session, &input_tensor, &positions_tensor, &[])?;
        
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
        let last_logits_tensor = build_tensor(last_logits, &last_shape, &device)?;

        // Sample next token from the *last-position* logits, not the full tensor.
        let next_token = sampler.sample(&last_logits_tensor, &history)?;
        
        // Accumulate generated tokens; decode full sequence at end for
        // correct BPE/SentencePiece boundary handling.
        generated_tokens.push(next_token);

        // Update state
        tokens.push(next_token);
        history.push(next_token);
        generated += 1;

        // Check for EOS token to stop generation
        if let Some(tok) = &tokenizer {
            if let Some(eos_id) = tok.eos_token_id {
                if next_token == eos_id {
                    eprintln!("[grim] EOS token {} reached, stopping generation.", eos_id);
                    break;
                }
            }
        }
    }

    // Decode all generated tokens together for correct BPE/SentencePiece
    // boundary handling (single-token decode can produce incomplete output).
    if let Some(tok) = &tokenizer {
        let text = tok.decode(&generated_tokens);
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    } else {
        for t in &generated_tokens {
            print!("{} ", t);
        }
        std::io::stdout().flush().unwrap();
    }

    println!("\n[grim] Done. Generated {} tokens.", generated);
    Ok(())
}

/// Holds everything needed for one or more generation runs against the same
/// loaded model. Used by both `cmd_run` (one-shot) and the interactive REPL
/// (B.4: avoid reloading the model from disk every turn).
pub struct GenerationContext {
    pub model: Box<dyn CausalLm>,
    pub session: SessionInner,
    pub tokenizer: Option<GgufTokenizer>,
    pub sampler: Box<dyn Sampler>,
    pub device: Device,
    pub vocab: usize,
    pub max_tokens: usize,
}

/// Load a model and prepare a fresh generation context.  The caller may
/// call `run_generation_turn` repeatedly on the same context; each turn
/// creates a *new* `SessionInner` (the model may need to re-prefill),
/// but the model itself is loaded only once and the tokenizer/sampler
/// are kept live between turns.
pub fn init_generation(
    model_path: String,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: u64,
    repeat_penalty: f32,
    max_tokens: usize,
) -> Result<GenerationContext> {
    let prompt = String::new(); // placeholder, not used for init

    // Resolve model name to actual file path
    let resolved_path = resolve_model_path(&model_path)
        .or_else(|| {
            let p = std::path::Path::new(&model_path);
            if p.exists() { Some(p.to_path_buf()) } else { None }
        })
        .ok_or_else(|| grim_core::error::Error::Config(
            format!("Model '{}' not found. Run 'grim pull {}' to download it.",
                model_path, model_path)
        ))?;
    let model_path_str = resolved_path.to_string_lossy().to_string();

    let path_obj = std::path::Path::new(&model_path_str);
    let use_gguf = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".gguf");
    let use_grim = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".grim");
    let use_safetensors = path_obj.is_file()
        && (model_path_str.to_lowercase().ends_with(".safetensors")
            || model_path_str.to_lowercase().ends_with(".bin"));

    let (device, _device_name) = probe_device();

    let model: Box<dyn CausalLm> = if use_gguf {
        eprintln!("[grim] Loading GGUF model: {}", model_path_str);
        load_model_from_gguf(&model_path_str, device.clone())?
    } else if use_grim {
        eprintln!("[grim] Loading GRIM model: {}", model_path_str);
        load_model_from_grim(&model_path_str, device.clone())?
    } else if use_safetensors {
        eprintln!("[grim] Loading safetensors model: {}", model_path_str);
        load_model_from_safetensors(&model_path_str, device.clone())?
    } else {
        return Err(grim_core::error::Error::Config(format!(
            "Model '{}' is not a valid .gguf, .grim, or .safetensors file or does not exist.",
            model_path_str
        )));
    };

    let tokenizer = if use_gguf {
        let provider = grim_format::GgufProvider::open(&model_path_str)?;
        Some(provider.tokenizer()?)
    } else if use_grim {
        let gguf_path = path_obj.with_extension("gguf");
        if gguf_path.exists() {
            let provider = grim_format::GgufProvider::open(gguf_path.to_str().unwrap())?;
            Some(provider.tokenizer()?)
        } else {
            None
        }
    } else if use_safetensors {
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

    let sampling_params = SamplingParams { temperature, top_p, top_k, repeat_penalty };
    let sampler: Box<dyn Sampler> = sampling_params.into_sampler(seed);

    let vocab: usize = if let Some(cfg) = model.config().as_any().downcast_ref::<LlamaConfig>() {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<grim_models_mamba::MambaConfig>() {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<Lfm2Config>() {
        cfg.vocab_size as usize
    } else if let Some(tok) = &tokenizer {
        tok.tokens.len()
    } else {
        512
    };

    let session = SessionInner::new(model.device().clone());

    Ok(GenerationContext {
        model,
        session,
        tokenizer,
        sampler,
        device,
        vocab,
        max_tokens,
    })
}

/// Build an F32 tensor from host data on the given device.
/// Eliminates the 5-way device match duplication that was repeated
/// for every tensor construction in the generation loop.
fn build_tensor(
    data: &[f32],
    shape: &grim_tensor::Shape,
    device: &grim_tensor::Device,
) -> Result<grim_tensor::Tensor> {
    let dtype = grim_tensor::dtype::DType::F32;
    let storage: Arc<dyn grim_tensor::BackendStorage> = match device {
        grim_tensor::Device::Cpu => {
            let dev = grim_backend_cpu::CpuDevice::new();
            Arc::from(dev.from_cpu(data, shape, dtype.clone())?)
        }
        grim_tensor::Device::Cuda(ordinal) => {
            let dev = grim_backend_cuda::CudaDevice::new(*ordinal);
            Arc::from(dev.from_cpu(data, shape, dtype.clone())?)
        }
        grim_tensor::Device::Rocm(ordinal) => {
            let dev = grim_backend_rocm::RocmDevice::new(*ordinal);
            Arc::from(dev.from_cpu(data, shape, dtype.clone())?)
        }
        grim_tensor::Device::Vulkan => {
            let dev = grim_backend_vulkan::VulkanDevice::new();
            Arc::from(dev.from_cpu(data, shape, dtype.clone())?)
        }
        grim_tensor::Device::Metal(ordinal) => {
            let dev = grim_backend_metal::MetalDevice::try_new(*ordinal)?;
            Arc::from(dev.from_cpu(data, shape, dtype.clone())?)
        }
    };
    Ok(grim_tensor::Tensor::new(
        storage,
        shape.clone(),
        dtype,
        grim_tensor::dtype::QuantProvenance::default(),
        device.clone(),
    ))
}

/// Interactive REPL: loads the model ONCE, then loops reading prompts
/// and generating responses without reloading or discarding the session.
/// Fixes B.4: the previous code called `cmd_run` per turn, reloading the
/// model and rebuilding the session (and its KV cache) from scratch.
pub async fn cmd_run_interactive(
    model_path: String,
    address: String,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    max_tokens: usize,
    seed: u64,
    repeat_penalty: f32,
) -> Result<()> {
    // ---- resolve path ----
    let resolved_path = resolve_model_path(&model_path)
        .or_else(|| {
            let p = std::path::Path::new(&model_path);
            if p.exists() { Some(p.to_path_buf()) } else { None }
        })
        .ok_or_else(|| grim_core::error::Error::Config(
            format!("Model '{}' not found. Run 'grim pull {}' to download it.",
                model_path, model_path)
        ))?;
    let model_path_str = resolved_path.to_string_lossy().to_string();
    eprintln!("[grim] Resolved model path: {}", model_path_str);

    let path_obj = std::path::Path::new(&model_path_str);
    let use_gguf = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".gguf");
    let use_grim = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".grim");
    let use_safetensors = path_obj.is_file()
        && (model_path_str.to_lowercase().ends_with(".safetensors")
            || model_path_str.to_lowercase().ends_with(".bin"));

    let (device, device_name) = probe_device();

    // ---- model (loaded once) ----
    let model: Box<dyn CausalLm> = if use_gguf {
        eprintln!("[grim] Loading GGUF model: {}", model_path_str);
        load_model_from_gguf(&model_path_str, device.clone())?
    } else if use_grim {
        eprintln!("[grim] Loading GRIM model: {}", model_path_str);
        load_model_from_grim(&model_path_str, device.clone())?
    } else if use_safetensors {
        eprintln!("[grim] Loading safetensors model: {}", model_path_str);
        load_model_from_safetensors(&model_path_str, device.clone())?
    } else {
        return Err(grim_core::error::Error::Config(format!(
            "Model '{}' is not a valid .gguf, .grim, or .safetensors file or does not exist.",
            model_path_str
        )));
    };

    // ---- tokenizer (loaded once) ----
    let tokenizer = if use_gguf {
        let provider = grim_format::GgufProvider::open(&model_path_str)?;
        Some(provider.tokenizer()?)
    } else if use_grim {
        let gguf_path = path_obj.with_extension("gguf");
        if gguf_path.exists() {
            let provider = grim_format::GgufProvider::open(gguf_path.to_str().unwrap())?;
            Some(provider.tokenizer()?)
        } else {
            None
        }
    } else if use_safetensors {
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

    // ---- sampler (created once) ----
    let sampling_params = SamplingParams { temperature, top_p, top_k, repeat_penalty };
    let sampler: Box<dyn Sampler> = sampling_params.into_sampler(seed);

    // ---- vocab size (computed once) ----
    let vocab: usize = if let Some(cfg) = model.config().as_any().downcast_ref::<LlamaConfig>() {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<grim_models_mamba::MambaConfig>() {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<Lfm2Config>() {
        cfg.vocab_size as usize
    } else if let Some(tok) = &tokenizer {
        tok.tokens.len()
    } else {
        512
    };

    eprintln!("[grim] Device: {device_name}");
    eprintln!("[grim] Sampling: temp={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}, seed={seed}");
    eprintln!("[grim] Type your prompt below (Ctrl+C to exit):");

    // Session persists across turns so the KV cache carries forward context.
    let mut session = SessionInner::new(model.device().clone());
    // Accumulated conversation history for multi-turn chat templates.
    let mut messages: Vec<grim_format::ChatMessage> = Vec::new();
    // Repeat-penalty history persists across turns.
    let mut history: Vec<u32> = Vec::new();
    // Running token count for position offset across turns (KV cache persists).
    let mut total_tokens: usize = 0;

    use std::io::Write;
    loop {
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        // Append the user message to the conversation history.
        messages.push(grim_format::ChatMessage {
            role: "user".to_string(),
            content: trimmed.to_string(),
        });

        let mut tokens: Vec<u32> = if let Some(tok) = &tokenizer {
            let mut ids = Vec::new();
            let prompt_text = if tok.chat_template.is_some() {
                grim_format::render_messages_or_last(tok, &messages)
            } else {
                let bos_candidates = ["<|startoftext|>", "<s>", "<|im_start|>"];
                for bos in &bos_candidates {
                    if let Some(&id) = tok.token_to_id.get(*bos) {
                        ids.push(id);
                        break;
                    }
                }
                trimmed.to_string()
            };
            ids.extend(tok.encode(&prompt_text));
            ids
        } else {
            trimmed.bytes().map(|b| b as u32 % 512).collect()
        };

        let mut generated = 0;
        let mut first_pass = true;
        let mut generated_tokens: Vec<u32> = Vec::new();

        while generated < max_tokens {
            let is_prefill = first_pass;
            let input_ids: Vec<f32> = if first_pass {
                first_pass = false;
                tokens.iter().map(|t| *t as f32).collect()
            } else {
                vec![*tokens.last().unwrap() as f32]
            };

            let n_tokens = input_ids.len();
            let shape = grim_tensor::Shape::new(vec![n_tokens]);
            let input_tensor = build_tensor(&input_ids, &shape, &device)?;

            let positions: Vec<f32> = if is_prefill {
                (0..n_tokens).map(|i| (total_tokens + i) as f32).collect()
            } else {
                vec![(total_tokens + n_tokens - 1) as f32]
            };
            let pos_shape = grim_tensor::Shape::new(vec![positions.len()]);
            let positions_tensor = build_tensor(&positions, &pos_shape, &device)?;

            let logits = CausalLm::forward(&*model, &mut session, &input_tensor, &positions_tensor, &[])?;

            let logits_vec = logits.to_vec_f32()?;
            let last_start = logits_vec.len().saturating_sub(vocab);
            let last_logits = &logits_vec[last_start..];

            let last_shape = grim_tensor::Shape::new(vec![vocab]);
            let last_logits_tensor = build_tensor(last_logits, &last_shape, &device)?;

            let next_token = sampler.sample(&last_logits_tensor, &history)?;

            generated_tokens.push(next_token);
            tokens.push(next_token);
            history.push(next_token);
            total_tokens += n_tokens;
            generated += 1;

            if let Some(tok) = &tokenizer {
                if let Some(eos_id) = tok.eos_token_id {
                    if next_token == eos_id { break; }
                }
            }
        }

        if let Some(tok) = &tokenizer {
            let text = tok.decode(&generated_tokens);
            print!("{}", text);
            // Record the assistant response so the next turn's chat template
            // has full conversation history.
            messages.push(grim_format::ChatMessage {
                role: "assistant".to_string(),
                content: text,
            });
        } else {
            for t in &generated_tokens {
                print!("{} ", t);
            }
        }
        std::io::stdout().flush().unwrap();
        println!();
    }
}