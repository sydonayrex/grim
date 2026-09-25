//! `grim run` — load a model, run a prompt, or start HTTP server.

use crate::catalog::resolve_model_path;
use grim_backend_cpu;
#[cfg(feature = "cuda")]
use grim_backend_cuda;
#[cfg(feature = "metal")]
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
use grim_models_transformer::{
    Chameleon, DecodeGraphModel, DeepSeek2, DeepSeek4, DeepSeek32, Gemma2, Lfm2, Lfm2Config, Llama,
    LlamaConfig, Mistral3, Mistral4, Qwen35,
};
use grim_tensor::{CoreTensorOps, Device};
use std::sync::Arc;

enum GraphDecodeResult {
    Sampled(u32),
    Logits(Box<grim_tensor::Tensor>),
}

/// Shared tokenizer loader for the `grim run` entry points (M12/M15).
/// Sibling `.gguf` provider or sibling `tokenizer.json`; `None` when neither
/// exists. Non-UTF8 paths are a typed Config error (never `to_str().unwrap()`);
/// a present-but-unreadable tokenizer warns loudly instead of vanishing into
/// `.ok()` — sampling defaults then stay CLI-provided, visibly.
fn load_run_tokenizer(
    path_obj: &std::path::Path,
    model_path_str: &str,
    use_gguf: bool,
    use_grim: bool,
    use_safetensors: bool,
) -> Result<Option<GgufTokenizer>> {
    if use_gguf {
        let provider = grim_format::GgufProvider::open(model_path_str)?;
        return Ok(Some(provider.tokenizer()?));
    }
    if use_grim {
        let gguf_path = path_obj.with_extension("gguf");
        if !gguf_path.exists() {
            return Ok(None);
        }
        let gguf_str = gguf_path.to_str().ok_or_else(|| {
            grim_core::error::Error::Config(format!(
                "sibling GGUF path is not valid UTF-8: {}",
                gguf_path.display()
            ))
        })?;
        let provider = grim_format::GgufProvider::open(gguf_str)?;
        return Ok(Some(provider.tokenizer()?));
    }
    if use_safetensors {
        let dir = path_obj.parent().unwrap_or(std::path::Path::new("."));
        let tokenizer_json = dir.join("tokenizer.json");
        if !tokenizer_json.exists() {
            return Ok(None);
        }
        let json_str = tokenizer_json.to_str().ok_or_else(|| {
            grim_core::error::Error::Config(format!(
                "tokenizer.json path is not valid UTF-8: {}",
                tokenizer_json.display()
            ))
        })?;
        match grim_format::GgufTokenizer::from_hf_json(json_str) {
            Ok(t) => return Ok(Some(t)),
            Err(e) => {
                eprintln!(
                    "[grim run] WARNING: present tokenizer.json unreadable ({e}); continuing without tokenizer defaults"
                );
                return Ok(None);
            }
        }
    }
    Ok(None)
}

/// Resolve a `&dyn DecodeGraphModel` from a model reference, unwrapping `SpeculativeCausalLm`
/// when operating under single-token `Strategy::Plain`. Fails closed (`None`) if wrapped under
/// multi-token strategies (e.g. DSpark) to prevent KV state / token offset corruption.
pub(crate) fn resolve_graph_model<'a>(model: &'a dyn CausalLm) -> Option<&'a dyn DecodeGraphModel> {
    let target: &'a dyn CausalLm = if let Some(spec) =
        model.as_any().downcast_ref::<grim_speculative::SpeculativeCausalLm>()
    {
        if spec.strategy() != grim_speculative::Strategy::Plain {
            return None;
        }
        spec.inner_target()
    } else {
        model
    };

    if let Some(m) = target.as_any().downcast_ref::<Lfm2>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<Llama>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<Mistral3>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<Mistral4>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<Qwen35>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<Gemma2>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<DeepSeek2>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<DeepSeek32>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<DeepSeek4>() {
        Some(m)
    } else if let Some(m) = target.as_any().downcast_ref::<Chameleon>() {
        Some(m)
    } else if let Some(m) = target
        .as_any()
        .downcast_ref::<grim_models_transformer::MiniMaxM3>()
    {
        Some(m)
    } else if let Some(m) = target
        .as_any()
        .downcast_ref::<grim_models_transformer::Glm4MoeLite>()
    {
        Some(m)
    } else if let Some(m) = target
        .as_any()
        .downcast_ref::<grim_models_transformer::GraniteMoeHybrid>()
    {
        Some(m)
    } else if let Some(m) = target
        .as_any()
        .downcast_ref::<grim_models_transformer::HyV3>()
    {
        Some(m)
    } else {
        grim_models_transformer::llama_wrapper_graph_model(target.as_any())
    }
}

/// i-was-dumb-graph.md Phase 5 + twinkie-zombieland P0: decode-loop graph trigger (LFM2 + ROCm only).
/// If GPU sampler can be used, samples directly from device logits with zero D2H!
/// Falls back to reading logits to host only when repeat penalty or CPU sampler is forced.
fn try_graph_decode_step(
    model: &dyn CausalLm,
    device: &Device,
    token_id: u32,
    vocab: usize,
    graph: &mut Option<grim_backend_rocm::FullDecodeGraph>,
    graph_retry: &mut grim_backend_rocm::graph_capture::GraphRetryPolicy,
    graph_fallback_step: &mut Option<usize>,
    sampling_params: &SamplingParams,
    seed: u64,
    step: usize,
    allow_gpu_sample: bool,
    history: &[u32],
    // Session reference for generic KV arena seeding (`None` → fail closed to eager).
    session: Option<&dyn grim_core::session::SessionT>,
    // Rows valid in every dense arena (prompt_len + decode steps so far).
    valid_rows: u32,
    // GRAVE task 0.3 follow-up: set when decode has reached the graph's hard
    // capacity. Capacity exhaustion is a CLEAN STOP (like any engine's context
    // window) — falling back to eager here would run against a stale session
    // (the graph owns all rows appended since capture) and emit garbage.
    ctx_exhausted: &mut bool,
    // Set when the previous call captured a fresh graph: the capture bracket
    // only RECORDS (it does not execute), so the capture-step token never
    // entered graph state. The eager forward for that step ran in the caller
    // afterwards — re-seed from the session once, at the first replay.
    just_captured: &mut bool,
) -> Option<GraphDecodeResult> {
    // who-dat.md P3-10: a transient capture/replay failure no longer kills
    // graph decode for the rest of the run — the policy re-arms after
    // `GRIM_GRAPH_RETRY_INTERVAL` eager steps (default 32), capped by
    // `GRIM_GRAPH_MAX_RETRIES` (default 3; 0 = old one-shot behaviour).
    if !graph_retry.should_attempt(step) || !grim_backend_rocm::decode_graph_enabled() {
        return None;
    }
    if !matches!(device, Device::Rocm(_)) {
        return None;
    }

    let graph_model: &dyn DecodeGraphModel = resolve_graph_model(model)?;
    // Lazily allocate once; stable addresses across steps.
    if graph.is_none() {
        let graph_ctx = std::env::var("GRIM_KV_CACHE_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096);
        // GRAVE task 0.3 (revised after research review): capacity exhaustion
        // is a clean stop, NOT an eager fallback — the eager session is stale
        // by every row the graph appended since capture, so switching modes
        // mid-run emits garbage. Raise GRIM_KV_CACHE_LEN for longer runs.
        if (valid_rows as usize) + 1 >= graph_ctx {
            *ctx_exhausted = true;
            eprintln!(
                "[grim] decode-graph: context capacity {} rows reached at step {}; \
                 stopping. Raise GRIM_KV_CACHE_LEN for longer contexts.",
                graph_ctx, step
            );
            return None;
        }
        match graph_model.get_or_create_decode_graph(graph_ctx, 1) {
            Ok(mut g) => {
                // Seed = copy prompt state (KV arenas + conv rings) from the
                // eager session into the graph pool. Runs OUTSIDE any capture
                // bracket (D2D + H2D + sync are capture-poison) and AFTER the
                // warmups (their state writes are discarded by the seed).
                macro_rules! seed_from_session {
                    ($g:expr) => {
                        (|| -> std::result::Result<(), String> {
                            let sess = session.ok_or_else(|| "no session".to_string())?;
                            let srcs = graph_model
                                .eager_kv_seed_sources(sess, valid_rows)
                                .map_err(|e| format!("export: {e}"))?;
                            let Device::Rocm(ordinal) = *device else {
                                return Err("non-ROCm device".to_string());
                            };
                            let dev = grim_backend_rocm::RocmDevice::shared(ordinal);
                            $g.buffers
                                .seed_kv_arena_from_eager(&dev, &srcs)
                                .map_err(|e| format!("seed: {e}"))?;
                            // GDL layers: D2D copy eager recurrent state into the
                            // graph's GDL state buffer so replay doesn't start
                            // from zero after prefill.
                            $g.buffers
                                .seed_gdl_state_from_eager(&dev, &srcs)
                                .map_err(|e| format!("gdl seed: {e}"))?;
                            // Recurrent (ShortConv) layers: upload the host conv rings
                            // so replay doesn't run them against a zeroed ring after
                            // prefill. Fail-closed like the KV seed on any mismatch.
                            let conv_seeds = graph_model
                                .eager_conv_seed_rings(sess)
                                .map_err(|e| format!("conv export: {e}"))?;
                            if !conv_seeds.is_empty() {
                                $g.buffers
                                    .seed_conv_rings(&conv_seeds)
                                    .map_err(|e| format!("conv seed: {e}"))?;
                            }
                            Ok(())
                        })()
                    };
                }
                // who-dat.md P3-11: Warmup before capture to prime JIT compiler,
                // Scythe WI-SB0 calibration, and caching allocator (hipModuleLaunchKernel 901 prevention).
                // Warmups run on UNSEEDED buffers and their state writes are
                // discarded — the seed below re-establishes prompt state
                // afterwards, so warmup side effects (KV appends, conv-ring
                // advances) can't desync graph state from the eager session.
                let _ = graph_model.forward_capture(&mut g, token_id);
                let _ = graph_model.forward_capture(&mut g, token_id);

                if let Err(e) = seed_from_session!(g) {
                    graph_retry.note_fail(step);
                    *graph_fallback_step = Some(step);
                    eprintln!(
                        "[grim] decode-graph: KV seed failed at step {step} ({e}); eager fallback (retry {})",
                        graph_retry.failures()
                    );
                    return None;
                }

                // First step: capture. Any failure -> abort the open capture
                // (else the stream stays capturing and later copies fail
                // with hipMemcpyDtoH 906), then eager fallback.
                // B3: log on the fallback side only (never inside the capture
                // bracket) so logging itself can't abort a capture.
                if let Err(e) = g.begin_capture() {
                    graph_retry.note_fail(step);
                    *graph_fallback_step = Some(step);
                    eprintln!(
                        "[grim] decode-graph: begin_capture failed at step {step} ({e}); eager fallback (retry {})",
                        graph_retry.failures()
                    );
                    return None;
                }
                let cap = graph_model.forward_capture(&mut g, token_id);
                let end = g.end_capture();
                if cap.is_err() || end.is_err() {
                    let _ = g.abort_capture();
                    graph_retry.note_fail(step);
                    *graph_fallback_step = Some(step);
                    eprintln!(
                        "[grim] decode-graph: capture failed at step {step} (forward: {}, end: {}); eager fallback (retry {})",
                        cap.as_ref()
                            .err()
                            .map(|e| e.to_string())
                            .unwrap_or_else(|| "ok".into()),
                        end.as_ref()
                            .err()
                            .map(|e| e.to_string())
                            .unwrap_or_else(|| "ok".into()),
                        graph_retry.failures()
                    );
                    return None;
                }
                // The capture bracket only RECORDS — it does not execute — so
                // the capture-step token never entered graph state. The eager
                // forward for this step runs in the caller after we return;
                // `just_captured` makes the FIRST replay re-seed from the
                // session (which then includes the capture-step token).
                *just_captured = true;
                *graph = Some(g);
                graph_retry.note_success();
                return None; // capture step ran eagerly path this token; replay from next.
            }
            Err(e) => {
                graph_retry.note_fail(step);
                *graph_fallback_step = Some(step);
                eprintln!(
                    "[grim] decode-graph: graph allocation failed at step {step} ({e}); eager fallback (retry {})",
                    graph_retry.failures()
                );
                return None;
            }
        }
    }
    let g = graph.as_mut()?;
    // First replay after a fresh capture: the capture step's token went
    // through the eager session (the caller ran it after we returned), but
    // the recorded graph never applied it. Re-seed so graph state matches
    // the session exactly before the first replay appends to it.
    if *just_captured {
        macro_rules! seed_from_session {
            ($g:expr) => {
                (|| -> std::result::Result<(), String> {
                    let sess = session.ok_or_else(|| "no session".to_string())?;
                    let srcs = graph_model
                        .eager_kv_seed_sources(sess, valid_rows)
                        .map_err(|e| format!("export: {e}"))?;
                    let Device::Rocm(ordinal) = *device else {
                        return Err("non-ROCm device".to_string());
                    };
                    let dev = grim_backend_rocm::RocmDevice::shared(ordinal);
                    $g.buffers
                        .seed_kv_arena_from_eager(&dev, &srcs)
                        .map_err(|e| format!("seed: {e}"))?;
                    $g.buffers
                        .seed_gdl_state_from_eager(&dev, &srcs)
                        .map_err(|e| format!("gdl seed: {e}"))?;
                    let conv_seeds = graph_model
                        .eager_conv_seed_rings(sess)
                        .map_err(|e| format!("conv export: {e}"))?;
                    if !conv_seeds.is_empty() {
                        $g.buffers
                            .seed_conv_rings(&conv_seeds)
                            .map_err(|e| format!("conv seed: {e}"))?;
                    }
                    Ok(())
                })()
            };
        }
        if let Err(e) = seed_from_session!(g) {
            graph_retry.note_fail(step);
            *graph_fallback_step = Some(step);
            *graph = None;
            *just_captured = false;
            eprintln!(
                "[grim] decode-graph: post-capture re-seed failed at step {step} ({e}); eager fallback (retry {})",
                graph_retry.failures()
            );
            return None;
        }
        *just_captured = false;
    }
    if let Err(e) = graph_model.forward_replay(g, token_id) {
        // Replay failure drops the stale graph so the next attempt (after
        // the retry interval) re-seeds + re-captures from the eager caches.
        graph_retry.note_fail(step);
        *graph_fallback_step = Some(step);
        *graph = None;
        eprintln!(
            "[grim] decode-graph: replay failed at step {step} ({e}); eager fallback (retry {})",
            graph_retry.failures()
        );
        return None;
    }
    g.buffers.current_pos = g.buffers.current_pos.wrapping_add(1);

    if allow_gpu_sample {
        // G1: launch the sampler on `g.stream` (the replay stream) so it is
        // stream-ordered AFTER `hipGraphLaunch`. Ambient `active_stream()`
        // would order only by today's implicit pool-0 == default_stream
        // coincidence — a future split of graph/sampler streams would
        // silently race. On miss (kernel not yet pre-resolved), sync the
        // graph stream once and take the ambient path (one-time cost).
        if let Device::Rocm(ordinal) = device {
            let dev = grim_backend_rocm::RocmDevice::shared(*ordinal);
            let stream = g.stream;
            let gate = sampling_params;
            match grim_backend_rocm::sample_logits_on_device_with_penalty_at_stream(
                &dev,
                g.logits_device_storage(),
                vocab,
                gate.temperature,
                gate.top_k as i32,
                gate.top_p,
                (seed & 0xffff_ffff) | ((step as u64) << 32),
                step as u32,
                gate.repeat_penalty,
                history,
                stream,
            ) {
                Ok(Some(tok)) => return Some(GraphDecodeResult::Sampled(tok)),
                // Miss or error: anything launched after this point (ambient
                // sampler, fallback D2H) must be ordered after the replayed
                // graph — one blocking sync, then fall through.
                Ok(None) | Err(_) => {
                    let _ = unsafe { grim_backend_rocm::hipStreamSynchronize(stream) };
                }
            }
        }
        if let Ok(tok) = sample_storage_on_rocm(
            device,
            g.logits_device_storage(),
            sampling_params,
            seed,
            step,
            history,
        ) {
            return Some(GraphDecodeResult::Sampled(tok));
        }
    }

    // who-dat P1-4: this full-vocab D2H is reached ONLY when the user forces
    // the CPU sampler (GRIM_CPU_SAMPLER) — the GPU sampler above already
    // applies repeat penalty on-device, so penalty-active steps never D2H.
    // Ordering: `to_cpu_vec_f32` copies via blocking `hipMemcpy`, which is
    // ordered after the replayed graph on every stream.
    let flat = g.read_logits_f32().ok()?;
    if flat.len() != vocab {
        return None;
    }
    let shape = grim_tensor::Shape::new(vec![1, vocab]);
    let tensor = build_tensor(&flat, &shape, device).ok()?;
    Some(GraphDecodeResult::Logits(Box::new(tensor)))
}

/// Resolve the GPU ordinal for this TP rank's process.
/// Only returns `Some` when multi-process TP is active (`GRIM_TP_SIZE > 1`).
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

/// Resolve the device for an explicit selection string (`"rocm"`, `"cuda:1"`, `"auto"`, ...); `None` means auto-detect.
/// Split from [`probe_device`] so the unavailable-backend error path is unit-testable without mutating the process environment.
pub(crate) fn probe_device_with(requested: Option<&str>) -> Result<(Device, String)> {
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
                        .and_then(|x| x.parse::<usize>().ok())
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
                #[cfg(feature = "metal")]
                {
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
                #[cfg(not(feature = "metal"))]
                Err(backend_unavailable("metal", "not compiled in"))
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
            #[cfg(all(target_vendor = "apple", feature = "metal"))]
            {
                if let Some((_free, total)) = grim_backend_metal::vram_info(0) {
                    if total > 0 {
                        eprintln!("[grim] Metal GPU detected");
                        return Ok((Device::Metal(0), "metal:0".into()));
                    }
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
        #[cfg(all(target_vendor = "apple", feature = "metal"))]
        {
            let Some((_free, total)) = grim_backend_metal::vram_info(0) else {
                return Ok((Device::Cpu, "cpu".into()));
            };
            if total > 0 {
                eprintln!("[grim] Metal GPU detected");
                return Ok((Device::Metal(0), "metal:0".into()));
            }
        }
        #[cfg(not(all(target_vendor = "apple", feature = "metal")))]
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
    min_tokens: u32,
    draft_model: Option<String>,
    _lookahead: bool,
    raw: bool,
    system: Option<String>,
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

    // Speculative decoding: if --draft-model was provided, wrap the base model.
    // Full DSpark (markov+confidence scheduling) is wired in the HTTP engine path;
    // CLI one-shot uses plain autoregressive + pre-loaded draft for future extension.
    // ponytail: plain wrapper — add DSpark when CLI speculation throughput is measured.
    let model: Box<dyn CausalLm> = if let Some(ref d_path) = draft_model {
        let dev = model.device().clone();
        match grim_engine::model_loader::load_eagle3_from_path(d_path, dev) {
            Ok(eagle3) => {
                eprintln!("[grim] Speculative decoding: Eagle3 draft loaded from {d_path}");
                let _drafter = Arc::new(grim_speculative::Eagle3Drafter::new(eagle3));
                // ponytail: plain for now; swap to with_dspark when markov/confidence CLI path added
                Box::new(grim_speculative::SpeculativeCausalLm::plain(model)) as Box<dyn CausalLm>
            }
            Err(_) => match grim_engine::model_loader::load_from_path(d_path) {
                Ok(_draft_raw) => {
                    eprintln!(
                        "[grim] Draft loaded from {d_path} (plain autoregressive; full DSpark via HTTP engine path)"
                    );
                    Box::new(grim_speculative::SpeculativeCausalLm::plain(model))
                        as Box<dyn CausalLm>
                }
                Err(e) => {
                    eprintln!(
                        "[grim] WARNING: draft model '{d_path}' load failed: {e}; using base model"
                    );
                    model
                }
            },
        }
    } else {
        model
    };

    let tokenizer = load_run_tokenizer(
        path_obj,
        &model_path_str,
        use_gguf,
        use_grim,
        use_safetensors,
    )?;

    // Apply model-recommended sampling defaults if CLI didn't override and not in raw mode.
    let (temperature, repeat_penalty, top_k) = if !raw {
        if let Some((rec_temp, rec_rep, rec_k)) =
            tokenizer.as_ref().and_then(|t| t.default_sampling_params())
        {
            // Use recommended default if temperature was default 0.7
            let t = if (temperature - 0.7).abs() < 1e-4 {
                rec_temp
            } else {
                temperature
            };
            let r = if (repeat_penalty - 1.1).abs() < 1e-4 {
                rec_rep
            } else {
                repeat_penalty
            };
            let k = if top_k == 40 { rec_k } else { top_k };
            (t, r, k)
        } else {
            (temperature, repeat_penalty, top_k)
        }
    } else {
        (temperature, repeat_penalty, top_k)
    };

    // Create sampler based on parameters
    let sampling_params = SamplingParams {
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        thinking_level: grim_core::sampler::ThinkingLevel::Default,
        min_tokens: 0,
    };
    let sampler: Box<dyn Sampler> = sampling_params.into_sampler(seed);

    // Tokenize prompt
    let mut tokens: Vec<u32> = if let Some(tok) = &tokenizer {
        let mut ids = Vec::new();

        // If raw mode is requested, pass prompt unformatted.
        // Otherwise, render through chat template and inject default/explicit system prompt.
        let prompt_text = if !raw && tok.chat_template.is_some() {
            let mut messages = Vec::new();
            if let Some(ref sys) = system {
                messages.push(grim_format::ChatMessage {
                    role: "system".to_string(),
                    content: sys.clone(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            } else if let Some(default_sys) = tok.default_system_prompt() {
                messages.push(grim_format::ChatMessage {
                    role: "system".to_string(),
                    content: default_sys.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            messages.push(grim_format::ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
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
        // GRAVE Phase 0.1: long-context baseline harness — `GRIM_CTX_TOKENS=N`
        // pads the prompt with neutral filler up to N total prompt tokens so
        // decode throughput can be measured as a function of context length.
        if let Ok(ctx) = std::env::var("GRIM_CTX_TOKENS").map(|v| v.parse::<usize>()) {
            if let Ok(ctx) = ctx {
                // Leave headroom for the chat-template suffix tokens AND at
                // least one generated position, or the KV arena overflows
                // (seed_kv_arena > arena) and the eager fallback faults.
                let kv_cap = std::env::var("GRIM_KV_CACHE_LEN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4096);
                let ctx = ctx.min(kv_cap.saturating_sub(16));
                let filler_text = "The history of grain production in the northern valley is documented in great detail across many seasons. ";
                let filler = tok.encode(filler_text);
                let target = ctx.saturating_sub(ids.len());
                let mut pad = Vec::with_capacity(target);
                let mut i = 0;
                while pad.len() < target {
                    pad.push(filler[i % filler.len()]);
                    i += 1;
                }
                pad.truncate(target);
                // Insert filler AFTER the BOS token so the template stays intact.
                let bos_count = ids.iter().take(2).count().min(1);
                let split = bos_count;
                let tail = ids.split_off(split);
                let pad_len = pad.len();
                ids.extend(pad);
                ids.extend(tail);
                eprintln!(
                    "[grim] ctx pad: {} filler tokens, total prompt {}",
                    pad_len,
                    ids.len()
                );
            }
        }
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
    std::io::stdout()
        .flush()
        .map_err(|e| grim_core::error::Error::Config(format!("stdout flush failed: {e}")))?;

    let mut session = SessionInner::new(model.device().clone());

    // GRAVE Phase 0.2: teacher-forced perplexity scorer (`GRIM_SCORE_FILE`).
    // Scores a text file chunk-by-chunk with plain forward passes (no chat
    // template, no sampling) and prints the aggregate PPL. The SAME harness
    // later scores distilled GRAVE students, so quality gates compare like
    // with like. Cross-checked against llama.cpp logprobs at gate G0b.
    if let Ok(score_path) = std::env::var("GRIM_SCORE_FILE") {
        let Some(tok) = tokenizer.as_ref() else {
            return Err(grim_core::error::Error::Config(
                "GRIM_SCORE_FILE requires a tokenizer (GGUF path)".into(),
            ));
        };
        let text = std::fs::read_to_string(&score_path)
            .map_err(|e| grim_core::error::Error::Backend(format!("score file read: {e}")))?;
        let full = tok.encode(&text);
        // G3c: context-length sweep via GRIM_SCORE_CHUNK_LEN (default 1024,
        // the historical value). Measures PPL flatness vs context — the
        // GRAVE quality gate for the GDL swap's context-invariance claim.
        let chunk_len: usize = std::env::var("GRIM_SCORE_CHUNK_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v >= 64 && v <= 32768)
            .unwrap_or(1024);
        let mut nll = 0.0f64;
        let mut n_target = 0usize;
        let mut chunks = 0usize;
        let mut start = 0usize;
        let score_start = std::time::Instant::now();
        while start + chunk_len + 1 <= full.len() {
            let ids_c: Vec<u32> = full[start..start + chunk_len].to_vec();
            let targets: Vec<u32> = full[start + 1..start + chunk_len + 1].to_vec();
            start += chunk_len;
            let mut sess = SessionInner::new(model.device().clone());
            let seq = ids_c.len();
            let in_t = grim_tensor::Tensor::new(
                std::sync::Arc::from(grim_backend_cpu::CpuDevice::new().from_cpu(
                    &ids_c.iter().map(|&i| i as f32).collect::<Vec<f32>>(),
                    &grim_tensor::Shape::new(vec![1, seq]),
                    grim_tensor::dtype::DType::F32,
                )?),
                grim_tensor::Shape::new(vec![1, seq]),
                grim_tensor::dtype::DType::F32,
                grim_tensor::dtype::QuantProvenance::default(),
                model.device().clone(),
            );
            let pos_t = grim_tensor::Tensor::new(
                std::sync::Arc::from(grim_backend_cpu::CpuDevice::new().from_cpu(
                    &(0..seq as u32).map(|i| i as f32).collect::<Vec<f32>>(),
                    &grim_tensor::Shape::new(vec![1, seq]),
                    grim_tensor::dtype::DType::F32,
                )?),
                grim_tensor::Shape::new(vec![1, seq]),
                grim_tensor::dtype::DType::F32,
                grim_tensor::dtype::QuantProvenance::default(),
                model.device().clone(),
            );
            let logits = CausalLm::forward(&*model, &mut sess, &in_t, &pos_t, &[])?;
            let lv = logits.to_vec_f32()?;
            for t in 0..seq - 1 {
                let row = &lv[t * vocab..(t + 1) * vocab];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let logsumexp: f32 = row
                    .iter()
                    .map(|&x| ((x - max) as f64).exp() as f32)
                    .sum::<f32>()
                    .ln()
                    + max;
                let tgt = targets[t] as usize;
                nll += -((row[tgt] - logsumexp) as f64);
                n_target += 1;
            }
            chunks += 1;
        }
        let elapsed_s = score_start.elapsed().as_secs_f64();
        let ppl = (nll / n_target as f64).exp();
        let tps = if elapsed_s > 0.0 {
            n_target as f64 / elapsed_s
        } else {
            0.0
        };
        let model_size_bytes = std::fs::metadata(&model_path_str)
            .map(|m| m.len())
            .unwrap_or(0);
        let bw_str = if model_size_bytes > 0 && tps > 0.0 {
            format!(
                ", {:.1} GB/s bandwidth",
                (model_size_bytes as f64 * tps) / 1e9
            )
        } else {
            String::new()
        };
        println!(
            "[ppl] chunks={} scored_tokens={} ppl={:.4} ({:.1} tokens/sec{}, elapsed={:.2}s)",
            chunks, n_target, ppl, tps, bw_str, elapsed_s
        );
        return Ok(());
    }

    let mut generated = 0;
    let mut history: Vec<u32> = Vec::new();
    let mut first_pass = true;
    let mut generated_tokens: Vec<u32> = Vec::new();

    // SPEED-ROC: decode-step tensors preallocated on first decode pass and
    // rewritten in place afterwards (write_f32_into) — no per-token device
    // allocs on the Rocm hot path.
    let mut decode_input: Option<grim_tensor::Tensor> = None;
    let mut decode_pos: Option<grim_tensor::Tensor> = None;

    // i-was-dumb-graph.md Phase 5: full-model graph state. Allocated once,
    // reused for stable addresses. `None` + `graph_failed` = eager fallback.
    let mut ctx_exhausted = false;
    let mut decode_graph: Option<grim_backend_rocm::FullDecodeGraph> = None;
    let mut graph_just_captured = false;
    let mut graph_retry = grim_backend_rocm::graph_capture::GraphRetryPolicy::default_policy();
    let mut graph_fallback_step: Option<usize> = None;
    // Owns the one prefill tensor pair so the borrow lives across the step.
    let mut decode_prefill_cache: Vec<(grim_tensor::Tensor, grim_tensor::Tensor)> = Vec::new();

    // Generation loop
    let mut decode_total_us: u64 = 0;
    let mut decode_count: u64 = 0;
    while generated < max_tokens {
        // Prefill on first pass to populate KV caches; decode one token at a time after.
        // HIGH-3: save first_pass before input_ids block mutates it to avoid wrong positions.
        let is_prefill = first_pass;
        let input_ids: Vec<f32> = if first_pass {
            first_pass = false;
            tokens.iter().map(|t| *t as f32).collect()
        } else if let Some(&last) = tokens.last() {
            vec![last as f32]
        } else {
            break;
        };

        // Host vecs: no transfer yet. Positions value matches old logic
        // (decode step pos = n_tokens-1 == 0 for [1] inputs).
        let n_hint = input_ids.len();
        let positions: Vec<f32> = if is_prefill {
            (0..n_hint).map(|i| i as f32).collect()
        } else {
            vec![n_hint as f32 - 1.0]
        };
        // SPEED-ROC: ROCm decode steps reuse preallocated [1]-shape tensors —
        // async in-place update, zero H2D allocs on the reuse path (old code
        // built + discarded two tensors per token). Non-Rocm: per-step build.
        let cached_tensors = if !is_prefill && matches!(device, Device::Rocm(_)) {
            if let (Some(din), Some(dpos)) = (&decode_input, &decode_pos) {
                // Update in place async on the active stream, ordered vs forward.
                if let Device::Rocm(ordinal) = &device {
                    let dev = grim_backend_rocm::RocmDevice::shared(*ordinal);
                    dev.write_f32_into_async(din.storage().as_ref(), &input_ids)?;
                    dev.write_f32_into_async(dpos.storage().as_ref(), &positions)?;
                }
                Some((din, dpos))
            } else {
                None
            }
        } else {
            None
        };

        let (input_tensor, positions_tensor) = if let Some((din, dpos)) = cached_tensors {
            (din, dpos)
        } else {
            // Build tensor from selected token(s). Prefill + first decode
            // step allocate; later ROCm steps never reach here.
            let n_tokens = input_ids.len();
            let shape = grim_tensor::Shape::new(vec![n_tokens]);
            let float_tokens = input_ids;
            let input_tensor = build_tensor(&float_tokens, &shape, &device)?;

            // Forward pass with proper positions tensor (CRIT-1).
            let pos_shape = grim_tensor::Shape::new(vec![positions.len()]);
            let positions_tensor = build_tensor(&positions, &pos_shape, &device)?;
            if !is_prefill {
                decode_input = Some(input_tensor);
                decode_pos = Some(positions_tensor);
                // Proof: assigned `Some` two lines above in this same block;
                // no await/branch between assignment and unwrap. `expect`
                // stays (not `?`): `None` here is a logic bug, not bad input.
                let din = decode_input.as_ref().expect("just set");
                let dpos = decode_pos.as_ref().expect("just set");
                (din, dpos)
            } else {
                // Prefill temps: leak via small Vec to keep borrow simple.
                // One alloc on prefill only, never on decode.
                decode_prefill_cache.push((input_tensor, positions_tensor));
                // Proof: pushed one element on the line above, so `last`
                // cannot be `None`. Same logic-bug rationale as above.
                let (i, p) = decode_prefill_cache.last().unwrap();
                (i, p)
            }
        };

        let step_start = std::time::Instant::now();
        // Phase 5 trigger: LFM2 decode steps try single-launch replay first.
        // Capture step + any miss fall through to eager (spec §Fallback).
        // B1: repeat penalty now applies on-device (pre-pass kernel), so the
        // GPU sampler stays eligible on penalty-active steps. `Err` inside
        // still falls back to the CPU sampler per call site below.
        let allow_gpu_sample =
            std::env::var("GRIM_CPU_SAMPLER").is_err() && matches!(device, Device::Rocm(_));

        if ctx_exhausted {
            eprintln!("[grim] context window exhausted; stopping generation.");
            break;
        }

        let graph_hit = if !is_prefill {
            let tid = tokens.last().copied().unwrap_or(0);
            try_graph_decode_step(
                &*model,
                &device,
                tid,
                vocab,
                &mut decode_graph,
                &mut graph_retry,
                &mut graph_fallback_step,
                &sampling_params,
                seed,
                generated,
                allow_gpu_sample,
                &history,
                Some(&session as &dyn grim_core::session::SessionT),
                tokens.len() as u32,
                &mut ctx_exhausted,
                &mut graph_just_captured,
            )
        } else {
            None
        };

        let next_token = match graph_hit {
            Some(GraphDecodeResult::Sampled(tok)) => {
                let step_us = step_start.elapsed().as_micros() as u64;
                decode_total_us += step_us;
                decode_count += 1;
                tok
            }
            Some(GraphDecodeResult::Logits(logits)) => {
                let step_us = step_start.elapsed().as_micros() as u64;
                decode_total_us += step_us;
                decode_count += 1;
                let logits_vec = logits.to_vec_f32()?;
                let last_start = logits_vec.len().saturating_sub(vocab);
                let last_logits = &logits_vec[last_start..];
                let last_shape = grim_tensor::Shape::new(vec![vocab]);
                let last_logits_tensor = grim_tensor::Tensor::new(
                    std::sync::Arc::from(grim_backend_cpu::CpuDevice::new().from_cpu(
                        last_logits,
                        &last_shape,
                        grim_tensor::dtype::DType::F32,
                    )?),
                    last_shape,
                    grim_tensor::dtype::DType::F32,
                    grim_tensor::dtype::QuantProvenance::default(),
                    grim_tensor::Device::Cpu,
                );
                sampler.sample(&last_logits_tensor, &history)?
            }
            None => {
                let logits =
                    CausalLm::forward(&*model, &mut session, input_tensor, positions_tensor, &[])?;
                let step_us = step_start.elapsed().as_micros() as u64;
                if !is_prefill {
                    decode_total_us += step_us;
                    decode_count += 1;
                }

                // SPEED-ROC: on decode steps (single-row logits) sample straight from the
                // device tensor via the WI-X3 GPU sampler — skips the full-vocab D2H +
                // CPU sampling that dominated per-token overhead. Prefill (multi-row)
                // steps fall back to the CPU sampler.
                // B1: repeat penalty applies on-device now; a device-kernel miss
                // degrades to CPU sampling (never errors the run).
                let gpu_sample_ok = logits.shape().elem_count() == vocab
                    && std::env::var("GRIM_CPU_SAMPLER").is_err()
                    && matches!(device, Device::Rocm(_));

                let device_token: Option<u32> = if gpu_sample_ok {
                    sample_on_rocm(
                        &device,
                        &logits,
                        vocab,
                        &sampling_params,
                        seed,
                        generated,
                        &history,
                    )
                    .ok()
                } else {
                    None
                };
                // CPU fallback closure (device-kernel miss or prefill shape).
                let cpu_sample = || -> Result<u32> {
                    let logits_vec = logits.to_vec_f32()?;
                    let last_start = logits_vec.len().saturating_sub(vocab);
                    let last_logits = &logits_vec[last_start..];
                    if std::env::var_os("GRIM_DEBUG_TOPLOGITS").is_some() {
                        if let Ok(path) = std::env::var("GRIM_DEBUG_LOGITS_PATH") {
                            use std::io::Write as _;
                            let mut f = std::fs::File::create(&path).ok();
                            if let Some(f) = f.as_mut() {
                                for v in last_logits {
                                    let b = v.to_le_bytes();
                                    let _ = f.write_all(&b);
                                }
                            }
                            eprintln!(
                                "[toplogits] dumped {} logits to {}",
                                last_logits.len(),
                                path
                            );
                        }
                        let mut idx: Vec<u32> = (0..vocab as u32).collect();
                        idx.sort_by(|&a, &b| {
                            last_logits[b as usize].total_cmp(&last_logits[a as usize])
                        });
                        eprintln!(
                            "[toplogits] step={} top5: {:?}",
                            generated,
                            idx[..5]
                                .iter()
                                .map(|&i| (
                                    i,
                                    (last_logits[i as usize] * 10_000.0).round() / 10_000.0
                                ))
                                .collect::<Vec<_>>()
                        );
                    }
                    let last_shape = grim_tensor::Shape::new(vec![vocab]);
                    let last_logits_tensor = grim_tensor::Tensor::new(
                        std::sync::Arc::from(grim_backend_cpu::CpuDevice::new().from_cpu(
                            last_logits,
                            &last_shape,
                            grim_tensor::dtype::DType::F32,
                        )?),
                        last_shape,
                        grim_tensor::dtype::DType::F32,
                        grim_tensor::dtype::QuantProvenance::default(),
                        grim_tensor::Device::Cpu,
                    );
                    Ok(sampler.sample(&last_logits_tensor, &history)?)
                };

                match device_token {
                    Some(tok) => tok,
                    None => cpu_sample()?,
                }
            }
        };

        // Accumulate tokens; decode full sequence at end for correct BPE boundary handling.
        generated_tokens.push(next_token);

        // Update state
        tokens.push(next_token);
        history.push(next_token);
        generated += 1;

        // Check for EOS or ChatML stop tokens, respecting min_tokens budget.
        if let Some(tok) = &tokenizer {
            let stop_tokens: Vec<u32> = vec![
                tok.token_to_id.get("<|im_end|>").copied(),
                tok.token_to_id.get("<|endoftext|>").copied(),
                tok.token_to_id.get("</s>").copied(),
            ]
            .into_iter()
            .flatten()
            .collect();

            if grim_core::sampler::should_stop(
                next_token,
                generated as u32,
                min_tokens,
                tok.eos_token_id,
                &stop_tokens,
            ) {
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
        std::io::stdout()
            .flush()
            .map_err(|e| grim_core::error::Error::Config(format!("stdout flush failed: {e}")))?;
    } else {
        for t in &generated_tokens {
            print!("{} ", t);
        }
        std::io::stdout()
            .flush()
            .map_err(|e| grim_core::error::Error::Config(format!("stdout flush failed: {e}")))?;
    }

    println!("\n[grim] Done. Generated {} tokens.", generated);
    if decode_count > 0 {
        let avg_us = decode_total_us / decode_count;
        let tps = 1_000_000.0 / avg_us as f64;
        let model_size_bytes = std::fs::metadata(&model_path_str)
            .map(|m| m.len())
            .unwrap_or(0);
        let bw_str = if model_size_bytes > 0 {
            let gb_s = (model_size_bytes as f64 * tps) / 1e9;
            format!(", {:.1} GB/s achieved GEMV bandwidth", gb_s)
        } else {
            String::new()
        };
        eprintln!(
            "[grim] Decode: {:.2} ms/token avg ({} tokens, {:.0} tokens/sec{})",
            avg_us as f64 / 1000.0,
            decode_count,
            tps,
            bw_str
        );
    }
    // B3: operator-visible graph status (PLAN-reduce-d2h-h2d).
    if matches!(device, Device::Rocm(_)) {
        if decode_graph.is_some() {
            eprintln!("[grim] decode-graph: active (replay path served decode steps)");
        } else if let Some(s) = graph_fallback_step {
            eprintln!(
                "[grim] decode-graph: eager at end-of-run ({} failure(s), last at step {s})",
                graph_retry.failures()
            );
        } else {
            eprintln!("[grim] decode-graph: inactive (eager; capture never succeeded)");
        }
    }
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

    let tokenizer = load_run_tokenizer(
        path_obj,
        &model_path_str,
        use_gguf,
        use_grim,
        use_safetensors,
    )?;

    let sampling_params = SamplingParams {
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        thinking_level: grim_core::sampler::ThinkingLevel::Default,
        min_tokens: 0,
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
/// SPEED-ROC: sample one token entirely on the GPU via the WI-X3 device sampler.
/// Reads the caller's device-resident logits tensor — no D2H, no CPU sort.
/// RNG stream advances per step: `SamplingOps::sample_on_device` packs the
/// position into the high 32 bits of `seed` (see `device_sampler::sample_impl`).
/// B1: repeat penalty applies on-device via `sample_on_device_with_penalty`
/// (pre-pass kernel over host-deduped history ids); `Err` → caller CPU-fallback.
fn sample_storage_on_rocm(
    device: &Device,
    storage: &dyn grim_tensor::BackendStorage,
    params: &SamplingParams,
    seed: u64,
    step: usize,
    history: &[u32],
) -> Result<u32> {
    use grim_tensor::SamplingOps;
    let Device::Rocm(ordinal) = device else {
        return Err(grim_core::error::Error::Unimplemented(
            "sample_on_rocm requires a ROCm device".into(),
        ));
    };
    let dev = grim_backend_rocm::RocmDevice::shared(*ordinal);
    let step_seed = (seed & 0xffff_ffff) | ((step as u64) << 32);
    Ok(dev.sample_on_device_with_penalty(
        storage,
        params.temperature,
        params.top_p,
        params.top_k,
        step_seed,
        params.repeat_penalty,
        history,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn sample_on_rocm(
    device: &Device,
    logits: &grim_tensor::Tensor,
    _vocab: usize,
    params: &SamplingParams,
    seed: u64,
    step: usize,
    history: &[u32],
) -> Result<u32> {
    sample_storage_on_rocm(
        device,
        logits.storage().as_ref(),
        params,
        seed,
        step,
        history,
    )
}

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
        #[cfg(feature = "metal")]
        grim_tensor::Device::Metal(ordinal) => {
            let dev = grim_backend_metal::MetalDevice::try_new(*ordinal)?;
            Arc::from(dev.from_cpu(data, shape, dtype.clone())?)
        }
        #[cfg(not(feature = "metal"))]
        grim_tensor::Device::Metal(_) => {
            return Err(grim_core::error::Error::Unimplemented(
                "Metal backend is not enabled in this build".into(),
            ));
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
    draft_model: Option<String>,
    _lookahead: bool,
    raw: bool,
    system: Option<String>,
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

    // Speculative decoding: wrap base model when --draft-model was provided.
    // ponytail: plain wrapper — full DSpark via HTTP engine path.
    // DSpark speculative wrapping (see cmd_run): draft backbone + markov +
    // entropy confidence + PID depth tuner; the old plain() ignored the draft.
    let model: Box<dyn CausalLm> = if let Some(ref d_path) = draft_model {
        let dev = model.device().clone();
        match grim_engine::model_loader::load_eagle3_from_path(d_path, dev) {
            Ok(eagle3) => {
                eprintln!("[grim] Speculative decoding: Eagle3 draft loaded from {d_path}");
                let drafter = Arc::new(grim_speculative::Eagle3Drafter::new(eagle3));
                grim_engine::Engine::build_dspark_model(model, drafter) as Box<dyn CausalLm>
            }
            Err(_) => match grim_engine::model_loader::load_from_path(d_path) {
                Ok(_draft_raw) => {
                    eprintln!(
                        "[grim] Draft loaded from {d_path} (DSpark: tiny backbone + markov + confidence)"
                    );
                    let dims = model
                        .arch_hyperparams()
                        .map(|h| (h.vocab_size, h.hidden_size))
                        .unwrap_or((128256, 2048));
                    let drafter = Arc::new(grim_speculative::TinyDraftBackbone::new(
                        dims.0, dims.1, 4, 42,
                    ));
                    grim_engine::Engine::build_dspark_model(model, drafter) as Box<dyn CausalLm>
                }
                Err(e) => {
                    eprintln!(
                        "[grim] WARNING: draft model '{d_path}' load failed: {e}; using base model"
                    );
                    model
                }
            },
        }
    } else {
        model
    };

    // ---- tokenizer (loaded once) ----
    let tokenizer = load_run_tokenizer(
        path_obj,
        &model_path_str,
        use_gguf,
        use_grim,
        use_safetensors,
    )?;

    // Apply model-recommended sampling defaults if CLI didn't override and not in raw mode.
    let (temperature, repeat_penalty, top_k) = if !raw {
        if let Some((rec_temp, rec_rep, rec_k)) =
            tokenizer.as_ref().and_then(|t| t.default_sampling_params())
        {
            let t = if (temperature - 0.7).abs() < 1e-4 {
                rec_temp
            } else {
                temperature
            };
            let r = if (repeat_penalty - 1.1).abs() < 1e-4 {
                rec_rep
            } else {
                repeat_penalty
            };
            let k = if top_k == 40 { rec_k } else { top_k };
            (t, r, k)
        } else {
            (temperature, repeat_penalty, top_k)
        }
    } else {
        (temperature, repeat_penalty, top_k)
    };

    // ---- sampler (created once) ----
    let sampling_params = SamplingParams {
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        thinking_level: grim_core::sampler::ThinkingLevel::Default,
        min_tokens: 0,
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
    // Repeat-penalty history persists across turns.
    let mut history: Vec<u32> = Vec::new();
    // Multi-turn chat template history.
    let mut messages: Vec<grim_format::ChatMessage> = Vec::new();
    let mut raw_mode = raw;

    // Initialize system prompt if configured or model provides a default
    let active_sys_prompt = system
        .as_deref()
        .or_else(|| tokenizer.as_ref().and_then(|t| t.default_system_prompt()));
    if let Some(sys) = active_sys_prompt {
        messages.push(grim_format::ChatMessage {
            role: "system".to_string(),
            content: sys.to_string(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    // Running token count for position offset across turns.
    let mut total_tokens: usize = 0;

    use std::io::Write;
    loop {
        print!("> ");
        std::io::stdout()
            .flush()
            .map_err(|e| grim_core::error::Error::Config(format!("stdout flush failed: {e}")))?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            break Ok(());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "/reset" {
            session = SessionInner::new(model.device().clone());
            messages.clear();
            if let Some(sys) = active_sys_prompt {
                messages.push(grim_format::ChatMessage {
                    role: "system".to_string(),
                    content: sys.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            history.clear();
            total_tokens = 0;
            println!("(conversation reset)");
            continue;
        }
        if trimmed == "/raw" {
            raw_mode = !raw_mode;
            println!("(raw mode: {})", if raw_mode { "ON" } else { "OFF" });
            continue;
        }
        if trimmed == "/exit" || trimmed == "/quit" {
            break Ok(());
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
            let prompt_text = if !raw_mode && tok.chat_template.is_some() {
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

        // SPEED-ROC: per-turn decode tensor reuse (reset each turn —
        // positions depend on total_tokens which shifts between turns).
        let mut decode_input: Option<grim_tensor::Tensor> = None;
        let mut decode_pos: Option<grim_tensor::Tensor> = None;

        while generated < max_tokens {
            let is_prefill = first_pass;
            let input_ids: Vec<f32> = if first_pass {
                first_pass = false;
                tokens.iter().map(|t| *t as f32).collect()
            } else if let Some(&last) = tokens.last() {
                vec![last as f32]
            } else {
                break;
            };

            let n_tokens = input_ids.len();
            let shape = grim_tensor::Shape::new(vec![n_tokens]);

            let positions: Vec<f32> = if is_prefill {
                (0..n_tokens).map(|i| (total_tokens + i) as f32).collect()
            } else {
                vec![(total_tokens + n_tokens - 1) as f32]
            };
            let pos_shape = grim_tensor::Shape::new(vec![positions.len()]);

            // SPEED-ROC: decode steps reuse preallocated [1]-shape tensors
            // (write_f32_into, no per-token alloc). Prefill + first decode
            // step allocate and seed the reuse buffers.
            let input_tensor;
            let positions_tensor;
            let it;
            let pt;
            if let (false, Some(din), Some(dpos)) = (is_prefill, &decode_input, &decode_pos) {
                if let Device::Rocm(ordinal) = &device {
                    let dev = grim_backend_rocm::RocmDevice::shared(*ordinal);
                    dev.write_f32_into(din.storage().as_ref(), &input_ids)?;
                    dev.write_f32_into(dpos.storage().as_ref(), &positions)?;
                }
                input_tensor = din;
                positions_tensor = dpos;
            } else {
                it = build_tensor(&input_ids, &shape, &device)?;
                pt = build_tensor(&positions, &pos_shape, &device)?;
                if !is_prefill {
                    decode_input = Some(it.clone());
                    decode_pos = Some(pt.clone());
                    // Proof: assigned `Some` two lines above; no await/branch
                    // between. `None` is a logic bug, not bad input — `expect`
                    // stays per the unwrap-sweep rule.
                    input_tensor = decode_input.as_ref().expect("just assigned");
                    positions_tensor = decode_pos.as_ref().expect("just assigned");
                } else {
                    input_tensor = &it;
                    positions_tensor = &pt;
                }
            };

            let logits =
                CausalLm::forward(&*model, &mut session, input_tensor, positions_tensor, &[])?;

            // SPEED-ROC: GPU-direct sampling on decode steps — see one-shot loop.
            // B1: penalty on-device; kernel miss degrades to CPU (never errors).
            let gpu_sample_ok = logits.shape().elem_count() == vocab
                && std::env::var("GRIM_CPU_SAMPLER").is_err()
                && matches!(device, Device::Rocm(_));

            let device_token = if gpu_sample_ok {
                sample_on_rocm(
                    &device,
                    &logits,
                    vocab,
                    &sampling_params,
                    seed,
                    generated,
                    &history,
                )
                .ok()
            } else {
                None
            };
            let next_token = match device_token {
                Some(tok) => tok,
                None => {
                    let logits_vec = logits.to_vec_f32()?;
                    let last_start = logits_vec.len().saturating_sub(vocab);
                    let last_logits = &logits_vec[last_start..];

                    let last_shape = grim_tensor::Shape::new(vec![vocab]);
                    let last_logits_tensor = build_tensor(last_logits, &last_shape, &device)?;

                    sampler.sample(&last_logits_tensor, &history)?
                }
            };

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
        std::io::stdout()
            .flush()
            .map_err(|e| grim_core::error::Error::Config(format!("stdout flush failed: {e}")))?;
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_unavailable_backend_errors_loudly() {
        // On the default (no-cuda) build this exercises the "not compiled in" path; on a cuda build without a GPU it exercises "no device".
        // In both cases it must hard-error naming the backend and the env var - never.
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

    /// M12: missing tokenizer sources degrade to `None` (CLI defaults apply),
    /// never panic. Present-but-unreadable warns loudly inside the loader.
    #[test]
    fn missing_tokenizer_sources_yield_none() {
        let dir = tempfile::tempdir().unwrap();
        let fake_model = dir.path().join("model.safetensors");
        std::fs::write(&fake_model, b"stub").unwrap();
        let out = load_run_tokenizer(&fake_model, "model.safetensors", false, false, true)
            .expect("missing tokenizer.json must not error");
        assert!(out.is_none());
        // Garbage tokenizer.json: warns + None, never Err, never panic.
        std::fs::write(dir.path().join("tokenizer.json"), b"not json{{{").unwrap();
        let out = load_run_tokenizer(&fake_model, "model.safetensors", false, false, true)
            .expect("unreadable tokenizer.json must not error");
        assert!(out.is_none());
    }

    /// M15: non-UTF8 sibling paths are a typed Config error, not a
    /// `to_str().unwrap()` panic. Unix-only: only Unix allows such names.
    /// `m\xffdel.grim`'s `.gguf` sibling exists but is not valid UTF-8.
    #[cfg(unix)]
    #[test]
    fn non_utf8_sibling_path_errors_loudly() {
        use std::ffi::OsString;
        use std::os::unix::ffi::{OsStrExt, OsStringExt};
        let dir = tempfile::tempdir().unwrap();
        let mk = |name: &[u8]| {
            let mut v = dir.path().as_os_str().as_bytes().to_vec();
            v.extend_from_slice(name);
            let p = std::path::PathBuf::from(OsString::from_vec(v));
            std::fs::write(&p, b"stub").unwrap();
            p
        };
        mk(b"/m\xffdel.gguf");
        let bad_grim = mk(b"/m\xffdel.grim");
        let err = match load_run_tokenizer(&bad_grim, "x", false, true, false) {
            Ok(_) => panic!("non-UTF8 existing sibling must Err"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("UTF-8"),
            "error must name the cause: {err}"
        );
    }
}
