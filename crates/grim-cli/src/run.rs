//! `grim run` — load a model, run a prompt, or start HTTP server.

use crate::catalog::resolve_model_path;
use grim_backend_cpu;
#[cfg(feature = "cuda")]
use grim_backend_cuda;
use grim_backend_metal;
#[cfg(feature = "rocm")]
use grim_backend_rocm;
use grim_backend_vulkan;
use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_core::sampler::{Sampler, SamplingParams};
use grim_core::session::Inner as SessionInner;
use grim_engine::{
    Engine, EngineConfig,
    model_loader::{load_model_from_gguf, load_model_from_grim, load_model_from_safetensors},
};
use grim_format::GgufTokenizer;
use grim_models_transformer::{Lfm2Config, LlamaConfig};
use grim_tensor::BackendDevice;
use grim_tensor::Device;
use std::sync::Arc;

/// Resolve the GPU ordinal for this TP rank's process. Only returns `Some`
/// when multi-process TP is active (`GRIM_TP_SIZE > 1`). Mirrors the ordinal
/// resolution in `model_loader::resolve_tp_ordinal` and
/// `RocmDevice::auto_init_rccl` — they all read the same env vars to stay in
/// sync. Returns `None` for single-device or when the configured ordinal isn't
/// among the probed devices.
fn tp_ordinal(devices: &[grim_backend_rocm::RocmDevice]) -> Option<usize> {
    let world_size = std::env::var("GRIM_TP_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&w| w > 1)?;
    let rank = std::env::var("GRIM_TP_RANK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    if rank >= world_size {
        return None;
    }
    // GRIM_GPUS may specify ordinals per rank; fall back to rank-as-ordinal.
    let gpus: Vec<usize> = std::env::var("GRIM_GPUS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default();
    let my_ordinal = gpus.get(rank).copied().unwrap_or(rank);
    // Verify the ordinal is visible among the probed devices.
    devices
        .iter()
        .any(|d| d.ordinal() == my_ordinal)
        .then_some(my_ordinal)
}

/// Auto-detect best available device. Probed once, reused by interactive REPL.
/// An explicitly requested backend must be available — never silently
/// degrade to CPU (WS-E1).
fn probe_device() -> Result<(Device, String)> {
    // `GRIM_BACKEND` is canonical (set by the install script); `GRIM_FORCE_DEVICE`
    // is accepted as a legacy alias for backward compatibility.
    let requested = std::env::var("GRIM_BACKEND").or_else(|_| std::env::var("GRIM_FORCE_DEVICE"));
    probe_device_with(requested.ok().as_deref())
}

/// Hard error for an explicitly requested backend that is unavailable in
/// this build or on this host (WS-E1: no silent CPU fallback).
fn backend_unavailable(name: &str, why: &str) -> grim_core::error::Error {
    grim_core::error::Error::Config(format!(
        "backend '{name}' requested via GRIM_BACKEND but unavailable ({why}). \
         Rebuild with --features {name} or unset GRIM_BACKEND for auto-detection."
    ))
}

/// Resolve the device for an explicit selection string (`"rocm"`, `"cuda:1"`,
/// `"auto"`, ...); `None` means auto-detect. Split from [`probe_device`] so
/// the unavailable-backend error path is unit-testable without mutating the
/// process environment. `auto`/unset keep the probe-chain-with-fallback
/// behavior; any other explicitly named backend that is not compiled in or
/// has no device is a hard error.
fn probe_device_with(requested: Option<&str>) -> Result<(Device, String)> {
    if let Some(s) = requested
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty() && s != "auto")
    {
        let prefix = s.split(':').next().unwrap_or("").trim();
        return match prefix {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    let ord_req = s
                        .split(':')
                        .nth(1)
                        .and_then(|x| x.parse::<u32>().ok())
                        .unwrap_or(0);
                    if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
                        if let Some(dev) = cuda_devices
                            .iter()
                            .find(|d| d.ordinal() == ord_req)
                            .or_else(|| cuda_devices.first())
                        {
                            return Ok((
                                Device::Cuda(dev.ordinal()),
                                format!("cuda:{}", dev.ordinal()),
                            ));
                        }
                    }
                    Err(backend_unavailable("cuda", "no CUDA device found"))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(backend_unavailable("cuda", "not compiled in"))
                }
            }
            "rocm" => {
                let ord_req = s.split(':').nth(1).and_then(|x| x.parse::<usize>().ok());
                if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
                    if let Some(req) = ord_req {
                        if let Some(dev) = rocm_devices.iter().find(|d| d.ordinal() == req) {
                            return Ok((
                                Device::Rocm(dev.ordinal()),
                                format!("rocm:{}", dev.ordinal()),
                            ));
                        }
                    }
                    if let Some(ord) = tp_ordinal(&rocm_devices) {
                        return Ok((Device::Rocm(ord), format!("rocm:{}", ord)));
                    }
                    if let Some(first) = rocm_devices.first() {
                        return Ok((
                            Device::Rocm(first.ordinal()),
                            format!("rocm:{}", first.ordinal()),
                        ));
                    }
                }
                Err(backend_unavailable("rocm", "no ROCm devices probed"))
            }
            "metal" => {
                let ord_req = s
                    .split(':')
                    .nth(1)
                    .and_then(|x| x.parse::<usize>().ok())
                    .unwrap_or(0);
                match grim_backend_metal::vram_info(ord_req) {
                    Some((_free, total)) if total > 0 => {
                        Ok((Device::Metal(ord_req), format!("metal:{ord_req}")))
                    }
                    _ => Err(backend_unavailable(
                        "metal",
                        "host unsupported or no Metal device found",
                    )),
                }
            }
            "vulkan" => {
                if let Ok(vulkan_devices) = grim_backend_vulkan::VulkanDevice::probe() {
                    if !vulkan_devices.is_empty() {
                        return Ok((Device::Vulkan, "vulkan".into()));
                    }
                }
                Err(backend_unavailable("vulkan", "no Vulkan devices found"))
            }
            "cpu" => Ok((Device::Cpu, "cpu".into())),
            other => Err(grim_core::error::Error::Config(format!(
                "unknown backend '{other}' requested via GRIM_BACKEND \
                 (expected rocm|cuda|vulkan|metal|cpu|auto)"
            ))),
        };
    }
    if let Ok(rocm_devices) = grim_backend_rocm::RocmDevice::probe() {
        if let Some(first) = rocm_devices.first() {
            // Under multi-process TP, pin this rank process to its own GPU.
            let ordinal = tp_ordinal(&rocm_devices).unwrap_or_else(|| first.ordinal());
            let wavefront = format!("{:?}", first.wavefront_size());
            let xnack = first.xnack_enabled();
            eprintln!(
                "[grim] ROCm GPU {} detected (wavefront={}, xnack={})",
                ordinal, wavefront, xnack
            );
            Ok((Device::Rocm(ordinal), format!("rocm:{}", ordinal)))
        } else {
            // ROCm available but no devices; check Metal → CUDA → Vulkan fallback.
            #[cfg(target_vendor = "apple")]
            {
                let Some((free, total)) = grim_backend_metal::vram_info(0) else {
                    // vram_info failed; fall through to CUDA/Vulkan
                };
                if total > 0 {
                    eprintln!("[grim] Metal GPU detected");
                    return Ok((Device::Metal(0), "metal:0".into()));
                }
            }
            #[cfg(feature = "cuda")]
            if let Ok(cuda_devices) = grim_backend_cuda::CudaDevice::probe() {
                if let Some(first) = cuda_devices.first() {
                    let ordinal = first.ordinal();
                    eprintln!("[grim] CUDA GPU {} detected", ordinal);
                    return Ok((Device::Cuda(ordinal), format!("cuda:{}", ordinal)));
                }
            }
            if let Ok(vulkan_devices) = grim_backend_vulkan::VulkanDevice::probe() {
                if !vulkan_devices.is_empty() {
                    eprintln!("[grim] Vulkan GPU detected");
                    Ok((Device::Vulkan, "vulkan".into()))
                } else {
                    eprintln!("[grim] No GPU detected; using CPU backend.");
                    Ok((Device::Cpu, "cpu".into()))
                }
            } else {
                eprintln!("[grim] No GPU detected; using CPU backend.");
                Ok((Device::Cpu, "cpu".into()))
            }
        }
    } else {
        // Check Metal on Apple platforms first, then Vulkan as fallback
        #[cfg(target_vendor = "apple")]
        {
            let Some((_free, total)) = grim_backend_metal::vram_info(0) else {
                return Ok((Device::Cpu, "cpu".into()));
            };
            if total > 0 {
                eprintln!("[grim] Metal GPU detected");
                return Ok((Device::Metal(0), "metal:0".into()));
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            if let Ok(vulkan_devices) = grim_backend_vulkan::VulkanDevice::probe() {
                if !vulkan_devices.is_empty() {
                    eprintln!("[grim] Vulkan GPU detected");
                    return Ok((Device::Vulkan, "vulkan".into()));
                }
            }
        }
        eprintln!("[grim] GPU runtime not available; using CPU backend.");
        Ok((Device::Cpu, "cpu".into()))
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

    // Resolve model name to file path
    let resolved_path = resolve_model_path(&model_path)
        .or_else(|| {
            // Accept a direct file path if it exists on disk.
            let p = std::path::Path::new(&model_path);
            if p.exists() {
                Some(p.to_path_buf())
            } else {
                None
            }
        })
        .ok_or_else(|| {
            grim_core::error::Error::Config(format!(
                "Model '{}' not found. Run 'grim pull {}' to download it.",
                model_path, model_path
            ))
        })?;
    let model_path_str = resolved_path.to_string_lossy().to_string();
    eprintln!("[grim] Resolved model path: {}", model_path_str);

    // Probe for ROCm GPUs; fail closed if path can't be opened (§13.2).
    let path_obj = std::path::Path::new(&model_path_str);
    let use_gguf = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".gguf");
    let use_grim = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".grim");
    let use_safetensors = path_obj.is_file()
        && (model_path_str.to_lowercase().ends_with(".safetensors")
            || model_path_str.to_lowercase().ends_with(".bin"));

    let (device, device_name) = probe_device()?;

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
                    eprintln!(
                        "[grim] ERROR: failed to load GGUF model '{}': {}",
                        model_path_str, e
                    );
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
                    eprintln!(
                        "[grim] ERROR: failed to load GRIM model '{}': {}",
                        model_path_str, e
                    );
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
                    eprintln!(
                        "[grim] ERROR: failed to load safetensors model '{}': {}",
                        model_path_str, e
                    );
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
        grim_server::serve(&address, engine, serve_model_path, None).await?;
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
                eprintln!(
                    "[grim] ERROR: failed to load GGUF model '{}': {}",
                    model_path_str, e
                );
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
                eprintln!(
                    "[grim] ERROR: failed to load GRIM model '{}': {}",
                    model_path_str, e
                );
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
                eprintln!(
                    "[grim] ERROR: failed to load safetensors model '{}': {}",
                    model_path_str, e
                );
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
        thinking_level: grim_core::sampler::ThinkingLevel::Default,
    };
    let sampler: Box<dyn Sampler> = sampling_params.into_sampler(seed);

    // Tokenize prompt
    let mut tokens: Vec<u32> = if let Some(tok) = &tokenizer {
        let mut ids = Vec::new();

        // If the tokenizer carries a Jinja chat template, render the
        // single-turn prompt through it for instruction-tuned models.
        // Otherwise fall back to raw prompt + best-effort BOS.
        let prompt_text = if tok.chat_template.is_some() {
            // The chat template itself is responsible for inserting BOS via
            // `{{ bos_token }}` (grim-format resolves it to the tokenizer's
            // `<s>` string). We must NOT prepend BOS here — that would
            // double-inject it for models like MiniCPM5 whose template opens
            // with `{{- bos_token }}`.
            let messages = vec![grim_format::ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
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
        let decoded: Vec<&str> = ids
            .iter()
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
    } else if let Some(cfg) = model
        .config()
        .as_any()
        .downcast_ref::<grim_models_mamba::MambaConfig>()
    {
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
    println!(
        "Sampling: temp={}, top_p={}, top_k={}, max_tokens={}, seed={}",
        temperature, top_p, top_k, max_tokens, seed
    );
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
        // Prefill on first pass to populate KV caches; decode one token at a time after.
        // HIGH-3: save first_pass before input_ids block mutates it to avoid wrong positions.
        let is_prefill = first_pass;
        let input_ids: Vec<f32> = if first_pass {
            first_pass = false;
            tokens.iter().map(|t| *t as f32).collect()
        } else {
            vec![*tokens.last().unwrap() as f32]
        };

        // Build tensor from selected token(s)
        let n_tokens = input_ids.len();
        let shape = grim_tensor::Shape::new(vec![n_tokens]);
        let float_tokens = input_ids;
        let _dtype = grim_tensor::dtype::DType::F32;
        let input_tensor = build_tensor(&float_tokens, &shape, &device)?;

        // Forward pass with proper positions tensor (CRIT-1).
        let positions: Vec<f32> = if is_prefill {
            (0..n_tokens).map(|i| i as f32).collect()
        } else {
            vec![n_tokens as f32 - 1.0]
        };
        let pos_shape = grim_tensor::Shape::new(vec![positions.len()]);
        let positions_tensor = build_tensor(&positions, &pos_shape, &device)?;

        let logits =
            CausalLm::forward(&*model, &mut session, &input_tensor, &positions_tensor, &[])?;

        // Get logits for last token position only
        let logits_vec = logits.to_vec_f32()?;
        let last_start = logits_vec.len().saturating_sub(vocab);
        let last_logits = &logits_vec[last_start..];

        // Single-position logits tensor so sampler sees next-token distribution only, not full sequence.
        let last_shape = grim_tensor::Shape::new(vec![vocab]);
        let last_logits_tensor = build_tensor(last_logits, &last_shape, &device)?;

        // Sample from last-position logits only.
        let next_token = sampler.sample(&last_logits_tensor, &history)?;

        // Accumulate tokens; decode full sequence at end for correct BPE boundary handling.
        generated_tokens.push(next_token);

        // Update state
        tokens.push(next_token);
        history.push(next_token);
        generated += 1;

        // Check for EOS or ChatML stop tokens
        if let Some(tok) = &tokenizer {
            let is_eos = tok.eos_token_id.map_or(false, |id| next_token == id)
                || tok.token_to_id.get("<|im_end|>").copied() == Some(next_token)
                || tok.token_to_id.get("<|endoftext|>").copied() == Some(next_token)
                || tok.token_to_id.get("</s>").copied() == Some(next_token);
            if is_eos {
                eprintln!(
                    "[grim] EOS token {} reached, stopping generation.",
                    next_token
                );
                break;
            }
        }
    }

    // Decode all tokens together for correct BPE boundary handling.
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

/// Holds state for one or more generation runs against the same model. Avoids reloading per turn.
pub struct GenerationContext {
    pub model: Box<dyn CausalLm>,
    pub session: SessionInner,
    pub tokenizer: Option<GgufTokenizer>,
    pub sampler: Box<dyn Sampler>,
    pub device: Device,
    pub vocab: usize,
    pub max_tokens: usize,
}

/// Load model and prepare generation context. Model loaded once; tokenizer/sampler persist between turns.
pub fn init_generation(
    model_path: String,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: u64,
    repeat_penalty: f32,
    max_tokens: usize,
) -> Result<GenerationContext> {
    // Resolve model name to file path
    let resolved_path = resolve_model_path(&model_path)
        .or_else(|| {
            let p = std::path::Path::new(&model_path);
            if p.exists() {
                Some(p.to_path_buf())
            } else {
                None
            }
        })
        .ok_or_else(|| {
            grim_core::error::Error::Config(format!(
                "Model '{}' not found. Run 'grim pull {}' to download it.",
                model_path, model_path
            ))
        })?;
    let model_path_str = resolved_path.to_string_lossy().to_string();

    let path_obj = std::path::Path::new(&model_path_str);
    let use_gguf = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".gguf");
    let use_grim = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".grim");
    let use_safetensors = path_obj.is_file()
        && (model_path_str.to_lowercase().ends_with(".safetensors")
            || model_path_str.to_lowercase().ends_with(".bin"));

    let (device, _device_name) = probe_device()?;

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

    let sampling_params = SamplingParams {
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        thinking_level: grim_core::sampler::ThinkingLevel::Default,
    };
    let sampler: Box<dyn Sampler> = sampling_params.into_sampler(seed);

    let vocab: usize = if let Some(cfg) = model.config().as_any().downcast_ref::<LlamaConfig>() {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model
        .config()
        .as_any()
        .downcast_ref::<grim_models_mamba::MambaConfig>()
    {
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

/// Build an F32 tensor from host data. Eliminates 5-way device match duplication.
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
        #[cfg(feature = "cuda")]
        grim_tensor::Device::Cuda(ordinal) => {
            let dev = grim_backend_cuda::CudaDevice::new(*ordinal)?;
            Arc::from(dev.from_cpu(data, shape, dtype.clone())?)
        }
        #[cfg(not(feature = "cuda"))]
        grim_tensor::Device::Cuda(_) => {
            return Err(grim_core::error::Error::Unimplemented(
                "CUDA backend is not enabled in this build".into(),
            ));
        }
        grim_tensor::Device::Rocm(ordinal) => {
            // Shared singleton: per-weight `new()` + drop would flush the
            // allocator cache and unload HIP modules on every upload.
            let dev = grim_backend_rocm::RocmDevice::shared(*ordinal);
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

/// Interactive REPL: loads model once, loops reading prompts without reloading. Fixes B.4.
pub async fn cmd_run_interactive(
    model_path: String,
    _address: String,
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
            if p.exists() {
                Some(p.to_path_buf())
            } else {
                None
            }
        })
        .ok_or_else(|| {
            grim_core::error::Error::Config(format!(
                "Model '{}' not found. Run 'grim pull {}' to download it.",
                model_path, model_path
            ))
        })?;
    let model_path_str = resolved_path.to_string_lossy().to_string();
    eprintln!("[grim] Resolved model path: {}", model_path_str);

    let path_obj = std::path::Path::new(&model_path_str);
    let use_gguf = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".gguf");
    let use_grim = path_obj.is_file() && model_path_str.to_lowercase().ends_with(".grim");
    let use_safetensors = path_obj.is_file()
        && (model_path_str.to_lowercase().ends_with(".safetensors")
            || model_path_str.to_lowercase().ends_with(".bin"));

    let (device, device_name) = probe_device()?;

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
    let sampling_params = SamplingParams {
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        thinking_level: grim_core::sampler::ThinkingLevel::Default,
    };
    let sampler: Box<dyn Sampler> = sampling_params.into_sampler(seed);

    // ---- vocab size (computed once) ----
    let vocab: usize = if let Some(cfg) = model.config().as_any().downcast_ref::<LlamaConfig>() {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model
        .config()
        .as_any()
        .downcast_ref::<grim_models_mamba::MambaConfig>()
    {
        cfg.vocab_size as usize
    } else if let Some(cfg) = model.config().as_any().downcast_ref::<Lfm2Config>() {
        cfg.vocab_size as usize
    } else if let Some(tok) = &tokenizer {
        tok.tokens.len()
    } else {
        512
    };

    eprintln!("[grim] Device: {device_name}");
    eprintln!(
        "[grim] Sampling: temp={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}, seed={seed}"
    );
    eprintln!("[grim] Type your prompt below (Ctrl+C to exit):");

    // Session and KV cache persist across turns.
    let mut session = SessionInner::new(model.device().clone());
    // Multi-turn chat template history.
    let mut messages: Vec<grim_format::ChatMessage> = Vec::new();
    // Repeat-penalty history persists across turns.
    let mut history: Vec<u32> = Vec::new();
    // Running token count for position offset across turns.
    let mut total_tokens: usize = 0;

    use std::io::Write;
    loop {
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Append the user message to the conversation history.
        messages.push(grim_format::ChatMessage {
            role: "user".to_string(),
            content: trimmed.to_string(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });

        let mut tokens: Vec<u32> = if let Some(tok) = &tokenizer {
            let mut ids = Vec::new();
            let prompt_text = if tok.chat_template.is_some() {
                if tok.add_bos_token {
                    if let Some(bos_id) = tok.bos_token_id {
                        ids.push(bos_id);
                    }
                }
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

            let logits =
                CausalLm::forward(&*model, &mut session, &input_tensor, &positions_tensor, &[])?;

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
                let is_eos = tok.eos_token_id.map_or(false, |id| next_token == id)
                    || tok.token_to_id.get("<|im_end|>").copied() == Some(next_token)
                    || tok.token_to_id.get("<|endoftext|>").copied() == Some(next_token)
                    || tok.token_to_id.get("</s>").copied() == Some(next_token);
                if is_eos {
                    break;
                }
            }
        }

        if let Some(tok) = &tokenizer {
            let text = tok.decode(&generated_tokens);
            print!("{}", text);
            // Record assistant response for next turn's full conversation history.
            messages.push(grim_format::ChatMessage {
                role: "assistant".to_string(),
                content: text,
                tool_calls: None,
                tool_call_id: None,
                name: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_unavailable_backend_errors_loudly() {
        // On the default (no-cuda) build this exercises the "not compiled in"
        // path; on a cuda build without a GPU it exercises "no device". In
        // both cases it must hard-error naming the backend and the env var —
        // never silently fall back to CPU (WS-E1).
        match probe_device_with(Some("cuda")) {
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("cuda"), "error must name the backend: {msg}");
                assert!(
                    msg.contains("GRIM_BACKEND"),
                    "error must name the env var: {msg}"
                );
            }
            // A cuda build with a live device legitimately resolves cuda.
            Ok((Device::Cuda(_), name)) => assert!(name.starts_with("cuda")),
            Ok((_dev, name)) => panic!("GRIM_BACKEND=cuda silently resolved to {name}"),
        }
    }

    #[test]
    fn requested_cpu_always_works() {
        let (dev, name) = probe_device_with(Some("cpu")).expect("cpu is always available");
        assert!(matches!(dev, Device::Cpu));
        assert_eq!(name, "cpu");
    }

    #[test]
    fn unknown_backend_is_rejected() {
        let msg = probe_device_with(Some("quantum")).unwrap_err().to_string();
        assert!(
            msg.contains("quantum"),
            "error must name the backend: {msg}"
        );
        assert!(
            msg.contains("GRIM_BACKEND"),
            "error must name the env var: {msg}"
        );
    }

    #[test]
    fn auto_and_unset_keep_the_fallback_chain() {
        // `auto` and unset must never hard-error: the probe chain falls back
        // through GPU backends down to CPU.
        assert!(probe_device_with(Some("auto")).is_ok());
        assert!(probe_device_with(None).is_ok());
        assert!(probe_device_with(Some("")).is_ok());
    }
}
