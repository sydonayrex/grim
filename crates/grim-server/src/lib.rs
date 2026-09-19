//! Grim HTTP server - axum-based, OpenAI-compatible endpoints.
//! Phase 3 deliverable: `/v1/chat/completions` that wires an `Engine`, resolves per-request LoRA adapters, and streams tokens.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_borrows_for_generic_args,
    clippy::redundant_locals,
    clippy::manual_strip,
    clippy::to_string_in_format_args,
    clippy::doc_lazy_continuation
)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{delete, get, post},
};
use futures::stream::{Stream, StreamExt};
use grim_core::error::Result;
use grim_core::grim_models_dir;
use grim_core::session::DeterminismMode;
use grim_engine::{Engine, model_loader};
use grim_format::GgufProvider;
use tokio_util::sync::CancellationToken;

pub mod audio;
pub mod routes;
/// Tool parsing and structured JSON call extraction.
/// See `docs/howto/tool-calling.md` for a complete client-side loop walkthrough.
pub mod tool_parse;

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Global registry of active audio models (TTS, Vocoder, ASR).
pub static AUDIO_MODELS: LazyLock<Mutex<HashMap<String, Arc<dyn grim_core::Model>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Register an audio model into the global server registry.
pub fn register_audio_model(name: &str, model: Arc<dyn grim_core::Model>) {
    let mut guard = AUDIO_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    guard.insert(name.to_string(), model);
}

/// Unregister an audio model from the global server registry.
pub fn unregister_audio_model(name: &str) {
    let mut guard = AUDIO_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    guard.remove(name);
}

/// Global registry of active diffusion models (Flux 2, UNet 2D).
pub static DIFFUSION_MODELS: LazyLock<Mutex<HashMap<String, Arc<dyn grim_core::Model>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Register a diffusion model into the global server registry.
pub fn register_diffusion_model(name: &str, model: Arc<dyn grim_core::Model>) {
    let mut guard = DIFFUSION_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    guard.insert(name.to_string(), model);
}

/// Unregister a diffusion model from the global server registry.
pub fn unregister_diffusion_model(name: &str) {
    let mut guard = DIFFUSION_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    guard.remove(name);
}

/// Cancellation token registry for active chat requests.
/// WI-CANCEL-1: `/v1/requests/:id/cancel` needs to signal the streaming loop driving request `id` to stop.
static CANCEL_TOKENS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, CancellationToken>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Register a fresh `CancellationToken` for `request_id` and return it.
/// If a token already exists for this id (should not happen in practice - a.
pub fn register_cancel_token(request_id: u64) -> CancellationToken {
    let token = CancellationToken::new();
    if let Ok(mut registry) = CANCEL_TOKENS.lock() {
        registry.insert(request_id, token.clone());
    }
    token
}

/// Look up the cancellation token for `request_id`, if one is registered.
pub fn take_cancel_token(request_id: u64) -> Option<CancellationToken> {
    CANCEL_TOKENS
        .lock()
        .ok()
        .and_then(|mut registry| registry.remove(&request_id))
}

/// WI-CANCEL-2: RAII guard ensuring `Engine::finish_request(id)` runs exactly once when the streaming SSE future is dropped - covering *all* exit paths uniformly: - normal completion (`max_tokens`, stop-sequence early return) - explicit cancel via `/v1/requests/:id/cancel` (WI-CANCEL-1, the `CancellationToken` causes the unfold closure to return `None`) - client disconnect (the SSE stream future is dropped, firing this `Drop`) The guard lives inside the `stream::unfold` state tuple so its lifetime is exactly the stream's lifetime - no earlier, no later than the last poll of the sink side.
/// This is the placement trap called out in the spec (axum discussion tokio-rs/axum#1060): a guard.
pub struct RequestCleanupGuard {
    /// `true` once cleanup has run, preventing a double-call if both an
    /// explicit early-return path *and* the guard's `Drop` could fire.
    dropped: bool,
    request_id: u64,
    state: Arc<AppState>,
}

impl RequestCleanupGuard {
    pub fn new(state: Arc<AppState>, request_id: u64) -> Self {
        LIVE_CLEANUP_GUARDS.fetch_add(1, Ordering::Relaxed);
        Self {
            dropped: false,
            request_id,
            state,
        }
    }
}

impl Drop for RequestCleanupGuard {
    fn drop(&mut self) {
        if self.dropped {
            return;
        }
        self.dropped = true;
        if let Ok(mut engine) = self
            .state
            .engine
            .lock()
            .or_else(|p| Ok::<_, ()>(p.into_inner()))
        {
            engine.finish_request(self.request_id);
        }
        // Remove the cancel token we registered so a stray reference doesn't
        // linger in the global registry after the request is done.
        let _ = take_cancel_token(self.request_id);
        // Remove per-request sampler params so the map cannot grow with
        // every served request.
        let _ = take_request_sampler_params(self.request_id);
        if let Ok(mut hist) = REQUEST_HISTORIES.lock() {
            hist.remove(&self.request_id);
            if let Ok(mut sess) = REQUEST_SESSIONS.lock() {
                sess.remove(&self.request_id);
            }
        }
        LIVE_CLEANUP_GUARDS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Counter of how many `RequestCleanupGuard` instances are currently live —
/// used by tests to assert exactly-once cleanup.
pub static LIVE_CLEANUP_GUARDS: AtomicUsize = AtomicUsize::new(0);

/// Shared engine state for the HTTP server.
/// `tokenizer` is populated from the active model's GGUF metadata when `serve()` is called with a.
pub struct AppState {
    pub engine: Mutex<Engine>,
    pub tokenizer: Mutex<Option<grim_format::GgufTokenizer>>,
    /// Path to the primary model file being served — used for
    /// `GET /v1/models` metadata and first-run doctor checks.
    pub model_path: Option<std::path::PathBuf>,
    /// Architecture string (e.g. "lfm2", "llama") of the loaded model —
    /// drives the per-arch tool-call detector registry (WI-E8).
    pub model_arch: std::sync::Mutex<Option<String>>,
    /// Plugin samplers loaded from `--plugins <dir>` at startup.
    /// Read-only at request time via `get_sampler(name)`; `None` when no plugins were loaded.
    pub plugin_registry: Option<std::sync::Arc<grim_plugin::PluginRegistry>>,
}

impl AppState {
    pub fn lock_engine(&self) -> std::sync::MutexGuard<'_, Engine> {
        self.engine.lock().unwrap_or_else(|p| p.into_inner())
    }
    pub fn lock_tokenizer(&self) -> std::sync::MutexGuard<'_, Option<grim_format::GgufTokenizer>> {
        self.tokenizer.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// WI-E8: read `general.architecture` from a GGUF file's metadata.
/// Returns None when the path is missing, unreadable, or lacks the key - the caller.
fn state_arch_hint(path: Option<&std::path::Path>) -> Option<String> {
    let p = path?;
    // Primary artifact carries the arch; a `.grim` primary keeps it in the
    // sibling `.gguf` (same lookup order the tokenizer uses).
    GgufProvider::open(p.display().to_string().as_str())
        .ok()
        .and_then(|prov| prov.architecture().map(str::to_string))
        .or_else(|| {
            let sibling = p.with_extension("gguf");
            GgufProvider::open(sibling.display().to_string().as_str())
                .ok()
                .and_then(|prov| prov.architecture().map(str::to_string))
        })
}

/// Health-check endpoint.
async fn health() -> &'static str {
    routes::health::health_handler().await
}

/// Conventional readiness probe; `/health` remains the legacy alias.
async fn healthz() -> &'static str {
    routes::health::healthz_handler().await
}

/// Readiness probe: unlike liveness, inference is not ready until a model is
/// loaded. This lets an orchestrator keep an empty server out of rotation.
async fn readyz(state: State<Arc<AppState>>) -> (StatusCode, Json<serde_json::Value>) {
    routes::health::readyz_handler(state).await
}

fn active_backend(has_gpu: bool) -> String {
    if let Ok(configured) = std::env::var("GRIM_BACKEND") {
        let configured = configured.trim().to_ascii_lowercase();
        if !configured.is_empty() && configured != "auto" {
            return configured;
        }
    }
    if has_gpu {
        if grim_backend_rocm::RocmDevice::probe()
            .map(|devices| !devices.is_empty())
            .unwrap_or(false)
        {
            "rocm".to_string()
        } else {
            #[cfg(feature = "cuda")]
            if grim_backend_cuda::CudaDevice::probe()
                .map(|devices| !devices.is_empty())
                .unwrap_or(false)
            {
                return "cuda".to_string();
            }
            "gpu".to_string()
        }
    } else {
        "cpu".to_string()
    }
}

fn validate_metrics_bind_policy(addr: &str) -> grim_core::error::Result<()> {
    let explicitly_allowed = std::env::var("GRIM_ALLOW_PUBLIC_METRICS")
        .map(|value| matches!(value.trim(), "1" | "true" | "yes"))
        .unwrap_or(false);
    validate_metrics_bind_policy_with_opt_in(addr, explicitly_allowed)
}

fn validate_metrics_bind_policy_with_opt_in(
    addr: &str,
    explicitly_allowed: bool,
) -> grim_core::error::Result<()> {
    let host = addr.rsplit_once(':').map(|(host, _)| host).unwrap_or(addr);
    let mut loopback =
        host == "localhost" || host == "::1" || host == "[::1]" || host.starts_with("127.");
    if !loopback {
        use std::net::ToSocketAddrs;
        if let Ok(addrs) = format!("{}:0", host).to_socket_addrs() {
            let mut all_loopback = true;
            let mut count = 0;
            for a in addrs {
                count += 1;
                if !a.ip().is_loopback() {
                    all_loopback = false;
                    break;
                }
            }
            if count > 0 && all_loopback {
                loopback = true;
            }
        }
    }
    let public = !loopback;
    if public && !explicitly_allowed {
        return Err(grim_core::Error::Config(format!(
            "refusing public metrics/server bind at {addr}; set GRIM_ALLOW_PUBLIC_METRICS=1 only when public exposure is intentional"
        )));
    }
    Ok(())
}

/// Chat completions endpoint - SSE streaming (§8, §4.5).
/// §13.3 contract: no silent partial fulfillment.
const DEFAULT_MAX_TOKENS: u64 = 2048;

/// Salt mixed into the per-request sampling seed so two requests with the
/// same model name produce independent draws.
const REQUEST_SEED_SALT: u64 = 0x5A17_C0DE_1337_BEEF;

/// F-4 (WI-TOOLS): synthetic system instruction prepended when a request carries `tools` but the model's chat template provides no output contract of its own.
/// Names the exact delimiters the post-hoc parser understands so the model's completion can actually be.
const TOOL_INSTRUCTION_PROMPT: &str = "You have access to the tools listed in this conversation. \
When the user's request requires a tool, respond with ONLY the tool call in this exact format and no other text:\n\
<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}</tool_call>\n\
If no tool is needed, answer normally in plain text.";

/// Monotonic millisecond clock for seeding stochastic samplers.
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Advance the engine one step for `request_id` and sample the next token from the produced logits using `sampler`.
/// Encapsulates the fixed-REQUEST_ID prefill-on-step-0 / decode-thereafter contract the server already relies on, plus the formerly-inline.
static REQUEST_HISTORIES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, Vec<u32>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// WI-HYBRID Layer 2: client session tag per request (body `session` field or
/// `x-grim-session` header), read at enqueue time by `sample_next_token`.
static REQUEST_SESSIONS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, String>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Client-supplied sampler overrides for one request.
/// `None` fields fall back to the corresponding `GRIM_SAMPLE_*` env default (the env vars are no.
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplerParams {
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    /// B1: request repeat penalty. Threaded to the device sampler so the
    /// device path no longer silently ignores it (was: penalty dropped on
    /// device, applied only on CPU fallback).
    pub repeat_penalty: Option<f32>,
    /// P1-3 (PLAN-improve-grim-perf): minimum tokens before EOS is honored.
    pub min_tokens: u32,
}

/// Per-request sampler params, keyed by request id.
/// Filled at request ingestion (chat and completions - OpenAI and Ollama shapes both funnel through.
static REQUEST_SAMPLER_PARAMS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, SamplerParams>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Extract OpenAI-style sampling overrides from a request body object and
/// register them under `request_id` for the device-side sampler.
fn register_request_sampler_params(
    request_id: u64,
    body_obj: &serde_json::Map<String, serde_json::Value>,
) {
    let params = SamplerParams {
        min_tokens: body_obj
            .get("min_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        temperature: body_obj
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        top_k: body_obj
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as i32),
        top_p: body_obj
            .get("top_p")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        seed: body_obj.get("seed").and_then(|v| v.as_u64()),
        repeat_penalty: body_obj
            .get("repeat_penalty")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
    };
    if let Ok(mut reg) = REQUEST_SAMPLER_PARAMS.lock() {
        reg.insert(request_id, params);
    }
}

fn take_request_sampler_params(request_id: u64) -> SamplerParams {
    REQUEST_SAMPLER_PARAMS
        .lock()
        .ok()
        .and_then(|mut reg| reg.remove(&request_id))
        .unwrap_or_default()
}

/// Historical CPU sampling path (WI-X3 fallback): D2H the full logits row,
/// slice to vocab, run the trait-object sampler on a CPU tensor.
fn cpu_sample_fallback(
    t: &grim_tensor::Tensor,
    sampler: &dyn grim_core::sampler::Sampler,
    vocab_size: usize,
    history: &[u32],
) -> std::result::Result<u32, String> {
    let full_logits = t
        .to_vec_f32()
        .map_err(|e| format!("logits to_vec_f32 failed: {e}"))?;
    let last_start = full_logits.len().saturating_sub(vocab_size);
    let last_logits = &full_logits[last_start..];
    let sampled = sampler
        .sample(
            &grim_backend_cpu::cpu_tensor(
                last_logits.to_vec(),
                grim_tensor::Shape::new(vec![last_logits.len()]),
            ),
            history,
        )
        .unwrap_or(0);
    Ok(sampled.min((vocab_size as u32).saturating_sub(1)))
}

/// WI-X3 device-side stochastic sampling: launch the Gumbel-max kernel on the resident ROCm logits (temperature/top-k/top-p on device) and copy back only the 4-byte token id.
/// Sampling params come from the request registry (`register_request_sampler_params`); the `GRIM_SAMPLE_TEMPERATURE` / `GRIM_SAMPLE_TOP_K` / `GRIM_SAMPLE_SEED` env.
/// B1: when `params.repeat_penalty > 1.0` and history is non-empty, the
/// penalty pre-pass runs on-device first (previously the device path silently
/// ignored the penalty). Penalty-kernel miss falls through to the legacy
/// path, which itself falls back to CPU — never silently unpenalized.
fn sample_on_device(
    t: &grim_tensor::Tensor,
    vocab_size: usize,
    step: u64,
    params: SamplerParams,
    history: &[u32],
) -> std::result::Result<Option<u32>, String> {
    use grim_backend_rocm::{
        RocmDevice, as_rocm, sample_logits_on_device_at, sample_logits_on_device_with_penalty_at,
    };

    let ordinal = t
        .device()
        .ordinal()
        .ok_or_else(|| "sample_on_device: no device ordinal".to_string())?;
    let storage = as_rocm(&**t.storage())
        .map_err(|e| format!("sample_on_device: not a ROCm storage: {e}"))?;

    let dev = RocmDevice::shared(ordinal);
    let temperature: f32 = params
        .temperature
        .or_else(|| {
            std::env::var("GRIM_SAMPLE_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0.7);
    let top_k: i32 = params
        .top_k
        .or_else(|| {
            std::env::var("GRIM_SAMPLE_TOP_K")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0);
    let top_p: f32 = params
        .top_p
        .or_else(|| {
            std::env::var("GRIM_SAMPLE_TOP_P")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(1.0);
    // Per-step reproducible stream: low bits = base seed, high bits = step.
    let base_seed: u64 = params
        .seed
        .or_else(|| {
            std::env::var("GRIM_SAMPLE_SEED")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let seed = (step << 32) | (base_seed & 0xffff_ffff);

    // B1: penalty-aware device path first (no-op + legacy equivalent when inactive).
    let penalty = params.repeat_penalty.unwrap_or(1.0);
    if penalty > 1.0 && !history.is_empty() {
        match sample_logits_on_device_with_penalty_at(
            &dev,
            storage,
            vocab_size,
            temperature,
            top_k,
            top_p,
            seed,
            step as u32,
            penalty,
            history,
        ) {
            Ok(tok) => return Ok(tok),
            Err(e) => {
                eprintln!("[grim-server] penalty sampler miss ({e}); trying legacy device path")
            }
        }
    }

    sample_logits_on_device_at(
        &dev,
        storage,
        vocab_size,
        temperature,
        top_k,
        top_p,
        seed,
        step as u32,
    )
    .map_err(|e| format!("sample_on_device: kernel: {e}"))
}

/// WI-1 (defense-in-depth): returns `Err(message)` instead of panicking when the engine cannot advance.
/// A network request must never be able to unwind this task while the engine mutex.
fn sample_next_token(
    engine: &mut grim_engine::Engine,
    request_id: u64,
    step: u64,
    sampler: &dyn grim_core::sampler::Sampler,
    prompt_tokens: Option<&[u32]>, // Only provided on step 0
    vocab_size: usize,
    model_id: Option<String>,
) -> std::result::Result<u32, String> {
    if step == 0 {
        let prompt_tokens = match prompt_tokens {
            Some(t) => t,
            None => return Err("prompt_tokens must be provided on step 0".to_string()),
        };
        if let Ok(mut hist) = REQUEST_HISTORIES.lock() {
            hist.insert(request_id, prompt_tokens.to_vec());
        }
        let model_id_final = match model_id {
            Some(ref id) if !id.is_empty() => Some(id.clone()),
            _ => engine.loaded_models().first().cloned(),
        };
        // WI-HYBRID Layer 2: attach the client session tag (if any) so the
        // engine gets slot affinity + radix-block pinning.
        let session = REQUEST_SESSIONS
            .lock()
            .ok()
            .and_then(|mut m| m.remove(&request_id));
        let req = grim_scheduler::Request {
            id: request_id,
            prompt_tokens: prompt_tokens.len(),
            max_new_tokens: 0,
            priority: 0,
            consumed_tokens: 0,
            model_id: model_id_final,
            adapter_ids: vec![],
            input_ids: Some(prompt_tokens.to_vec()),
            session,
        };
        let _ = engine.enqueue_request(req);
    }

    // WI-1: propagate instead of panicking. Panicking here unwound the stream task while the engine mutex was
    // held, poisoning it for every later request and preventing the `[DONE]` SSE terminator from ever being sent.
    if let Err(e) = engine.tick() {
        return Err(format!("engine tick failed: {e}"));
    }

    let history = REQUEST_HISTORIES
        .lock()
        .ok()
        .and_then(|h| h.get(&request_id).cloned())
        .unwrap_or_default();
    let outcome = engine.last_outcome(request_id);
    eprintln!(
        "[sample_next_token] req {request_id} step {step} outcome is_some: {}, models: {:?}",
        outcome.is_some(),
        engine.loaded_models()
    );
    let logits = outcome.and_then(|o| o.logits.as_ref().cloned());
    if step == 0 && std::env::var("GRIM_DEBUG_PROMPT").as_deref() == Ok("1") {
        if let Some(t) = &logits {
            if let Ok(all) = t.to_vec_f32() {
                let width = vocab_size.max(1);
                let last_start = all.len().saturating_sub(width);
                let last = &all[last_start..];
                let mut idx: Vec<usize> = (0..last.len()).collect();
                idx.sort_by(|&a, &b| {
                    last[b]
                        .partial_cmp(&last[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                eprintln!(
                    "[grim-server] step0 logits_len={} width={} top5={:?}",
                    all.len(),
                    width,
                    idx.iter()
                        .take(5)
                        .map(|&i| (i, last[i]))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    // The engine's logits table is 65536 entries wide; a model with a smaller vocab (e.g.
    // 32000) must slice to the last `vocab_size` positions before sampling, otherwise the sampler scores against.
    let token = match logits {
        Some(t) => {
            // WI-X3: when logits live on a ROCm device, run temperature/top-k
            // sampling on-device (Gumbel-max) and D2H only the chosen token id.
            let cpu_sampler_forced = std::env::var("GRIM_CPU_SAMPLER")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let has_active_constraint = sampler.name() == "constrained";
            let device_token =
                if t.device().is_rocm() && !cpu_sampler_forced && !has_active_constraint {
                    let params = REQUEST_SAMPLER_PARAMS
                        .lock()
                        .ok()
                        .and_then(|reg| reg.get(&request_id).copied())
                        .unwrap_or_default();
                    match sample_on_device(&t, vocab_size, step, params, &history) {
                        Ok(Some(tok)) => Some(tok.min((vocab_size as u32).saturating_sub(1))),
                        // Ok(None): unsupported shape/vocab -> CPU fallback contract.
                        Ok(None) => None,
                        Err(_e) => None,
                    }
                } else {
                    None
                };
            match device_token {
                Some(tok) => tok,
                None => cpu_sample_fallback(&t, sampler, vocab_size, &history)?,
            }
        }
        None => 0,
    };
    if let Ok(mut hist) = REQUEST_HISTORIES.lock() {
        hist.entry(request_id).or_default().push(token);
    }

    // Record the generated token so the next decode step uses the real token
    // instead of a position index.
    engine.record_generated_token(request_id, token);
    Ok(token)
}

/// Trim the first matched stop sequence from the end of `text`.
/// Returns the trimmed text and whether a stop sequence was found and removed.
fn trim_stop_sequences(text: &str, stop_seqs: &[String]) -> (String, bool) {
    for seq in stop_seqs {
        if text.ends_with(seq) {
            let trimmed = text.strip_suffix(seq).unwrap_or(text).to_string();
            return (trimmed, true);
        }
    }
    (text.to_string(), false)
}

/// WI-P9: strip every occurrence of any stop sequence from `text`.
/// The stop string is a signal, not content (OpenAI convention); non-streaming and streaming both apply.
fn strip_stop_sequences(text: &str, stop_seqs: &[String]) -> (String, bool) {
    let mut out = text.to_string();
    let mut hit = false;
    for seq in stop_seqs {
        if seq.is_empty() {
            continue;
        }
        if out.contains(seq.as_str()) {
            hit = true;
            out = out.replace(seq.as_str(), "");
        }
    }
    (out, hit)
}

/// Split model-generated chain-of-thought preambles from the main response text.
/// The model is expected to wrap its reasoning in `<think>`...`</think>` tags (DeepSeek-R1 / Qwen3-Thinking convention).
fn split_think_content(text: &str) -> (Option<String>, String) {
    let mut reasoning = String::new();
    let mut clean = String::new();
    let mut cursor = text;

    while let Some(start_pos) = cursor.find("<think>") {
        clean.push_str(&cursor[..start_pos]);
        let after_start = &cursor[start_pos + "<think>".len()..];
        if let Some(end_pos) = after_start.find("</think>") {
            reasoning.push_str(&after_start[..end_pos]);
            reasoning.push('\n');
            cursor = &after_start[end_pos + "</think>".len()..];
        } else {
            reasoning.push_str(after_start);
            reasoning.push('\n');
            cursor = "";
            break;
        }
    }
    clean.push_str(cursor);

    if reasoning.is_empty() {
        (None, text.to_string())
    } else {
        (Some(reasoning.trim_end().to_string()), clean)
    }
}

/// WI-1 - Remote-provider scheme allowlist.
/// Only these prefixes (used as `"<scheme>:<model>"`) denote a remote provider route.
const REMOTE_PROVIDER_SCHEMES: &[&str] = &["ollama", "openai", "hf", "huggingface", "anthropic"];

/// Cached snapshot of local catalog names, refreshed at most once per [`CATALOG_CACHE_TTL`].
/// `list_local_models()` performs a filesystem scan, so calling it unconditionally per request would add real latency.
static LOCAL_CATALOG_CACHE: std::sync::LazyLock<
    Mutex<Option<(std::time::Instant, std::collections::HashSet<String>)>>,
> = std::sync::LazyLock::new(|| Mutex::new(None));

const CATALOG_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(5);

/// True when `name` exactly matches an entry in the local model catalog.
/// The catalog names files as `"{stem}:{ext}"` (`catalog.rs`), so local names routinely contain a colon -.
pub fn is_local_catalog_model(name: &str) -> bool {
    let now = std::time::Instant::now();
    let mut guard = match LOCAL_CATALOG_CACHE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let fresh = guard
        .as_ref()
        .is_some_and(|(at, _)| now.duration_since(*at) < CATALOG_CACHE_TTL);
    if !fresh {
        let names: std::collections::HashSet<String> = grim_core::catalog::list_local_models()
            .into_iter()
            .map(|e| e.name)
            .collect();
        *guard = Some((now, names));
    }
    guard
        .as_ref()
        .is_some_and(|(_, names)| names.contains(name))
}

/// WI-1 - Decide whether a requested model name should be routed to the remote provider proxy instead of being served locally.
/// A name is remote only when it carries a known provider scheme *and* does not.
pub fn is_remote_provider_model(name: &str) -> bool {
    if is_local_catalog_model(name) {
        return false;
    }
    if name.starts_with("hf/") {
        return true;
    }
    match name.split_once(':') {
        Some((scheme, rest)) if !rest.is_empty() => REMOTE_PROVIDER_SCHEMES.contains(&scheme),
        _ => false,
    }
}

/// Build the OpenAI `choices[0]` payload for one generated completion, applying WI-TOOLS-4/5 and the WI-TOOLS-4b soft guard: - when tool calling is active, run the raw completion through the per-family output parser (WI-TOOLS-4).
/// A clean parse yields `message.tool_calls` with `finish_reason: "tool_calls"`; otherwise the completion is returned as ordinary.
fn build_choice_payload(
    content: &str,
    reasoning_content: Option<&str>,
    tools_active: bool,
    template_family: Option<&str>,
    model_arch: Option<&str>,
    prior_messages: &[grim_format::ChatMessage],
) -> serde_json::Value {
    let (message, finish_reason) = if tools_active {
        let family =
            tool_parse::resolve_effective_tool_family(template_family.unwrap_or(""), model_arch);
        match tool_parse::parse_tool_calls(content, family) {
            tool_parse::ParseOutcome {
                calls: Some(calls), ..
            } => {
                let tool_calls: Vec<serde_json::Value> = calls
                    .iter()
                    .map(|c| {
                        let repeat_count = tool_parse::count_prior_identical_calls(
                            prior_messages,
                            &c.name,
                            &c.arguments,
                        );
                        let arguments = if repeat_count >= 2 {
                            diagnostic_arguments(&c.arguments, repeat_count)
                        } else {
                            c.arguments.clone()
                        };
                        serde_json::json!({
                            "id": c.id,
                            "type": "function",
                            "function": { "name": c.name, "arguments": arguments },
                        })
                    })
                    .collect();
                (
                    serde_json::json!({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls
                    }),
                    "tool_calls",
                )
            }
            _ => {
                let mut msg = serde_json::json!({ "role": "assistant", "content": content });
                if let Some(rc) = reasoning_content {
                    msg["reasoning_content"] = serde_json::json!(rc);
                }
                (msg, "stop")
            }
        }
    } else {
        let mut msg = serde_json::json!({ "role": "assistant", "content": content });
        if let Some(rc) = reasoning_content {
            msg["reasoning_content"] = serde_json::json!(rc);
        }
        (msg, "stop")
    };
    serde_json::json!({
        "index": 0,
        "message": message,
        "finish_reason": finish_reason
    })
}

/// WI-TOOLS-4b hard guard. Returns `Some((tool_name, repeat_count))` when the most recent parsed
/// call for `name`/`arguments` has already appeared >= 4 times in `prior_messages` (i.e.
fn check_repeated_call_hard_guard(
    prior_messages: &[grim_format::ChatMessage],
    name: &str,
    arguments: &str,
) -> Option<usize> {
    let count = tool_parse::count_prior_identical_calls(prior_messages, name, arguments);
    if count >= 4 { Some(count) } else { None }
}

/// WI-TOOLS-4b soft-guard diagnostic payload.
/// Replaces the call's `arguments` with a JSON-encoded string carrying the duplicate flag, the repeat count,.
fn diagnostic_arguments(original: &str, repeat_count: usize) -> String {
    let original_value: serde_json::Value =
        serde_json::from_str(original).unwrap_or(serde_json::Value::String(original.to_string()));
    serde_json::to_string(&serde_json::json!({
        "__grim_duplicate_call_warning": true,
        "repeat_count": repeat_count,
        "original_arguments": original_value,
        "message": "This exact call has been made with identical arguments multiple times. Consider whether the arguments need to change, whether the tool is failing, or whether a different action is needed."
    }))
    .unwrap_or_else(|_| "{}".to_string())
}

/// Terminal SSE emitter for the streaming path (WI-TOOLS-5 buffered streaming).
/// Runs the buffered completion through [`build_choice_payload`] and, if a clean tool-call parse was produced, emits.
fn terminal_tool_delta(
    parse_ctx: &(bool, Option<String>, Option<String>),
    emitted: &str,
    prior_messages: &[grim_format::ChatMessage],
    reasoning_content: Option<&str>,
) -> Option<std::result::Result<axum::response::sse::Event, axum::Error>> {
    let (tools_active, template_family, model_arch) = parse_ctx;
    let choice = build_choice_payload(
        emitted,
        reasoning_content,
        *tools_active,
        template_family.as_deref(),
        model_arch.as_deref(),
        prior_messages,
    );
    // A clean parse surfaces a non-empty `tool_calls` array on the message.
    if let Some(tool_calls) = choice.get("message").and_then(|m| m.get("tool_calls")) {
        if let Some(arr) = tool_calls.as_array() {
            if !arr.is_empty() {
                let payload = serde_json::json!({
                    "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": "tool_calls"}]
                })
                .to_string();
                return Some(Ok(axum::response::sse::Event::default()
                    .event("message")
                    .data(payload)));
            }
        }
    }
    None
}

/// WI-TOOLS-4b/4c - stable, machine-readable error codes for every rejection `chat_completions` can produce.
/// Each variant serializes to the `code` field on the structured `{"error": {...}}` object, so clients.
pub enum ErrorCode {
    InvalidRequest,
    UnknownField,
    AdapterNotFound,
    DeterminismMismatch,
    EmptyMessages,
    DuplicateToolCall,
    TotalToolCallLimit,
    MessageCountLimit,
}

impl ErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::InvalidRequest => "invalid_request",
            ErrorCode::UnknownField => "unknown_field",
            ErrorCode::AdapterNotFound => "adapter_not_found",
            ErrorCode::DeterminismMismatch => "determinism_mismatch",
            ErrorCode::EmptyMessages => "empty_messages",
            ErrorCode::DuplicateToolCall => "duplicate_tool_call_limit",
            ErrorCode::TotalToolCallLimit => "total_tool_call_limit",
            ErrorCode::MessageCountLimit => "message_count_limit",
        }
    }
}

/// Build a structured `chat_completions` rejection body matching OpenAI's `{"error": {"type": ..., "code": ..., "message": ...}}` object shape, with a stable `code` discriminant the client can branch on.
/// `type` reuses OpenAI's own `invalid_request_error` taxonomy so OpenAI-compatible client SDKs behave sensibly even before they.
fn request_error(code: ErrorCode, message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "type": "invalid_request_error",
            "code": code.as_str(),
            "message": message.into(),
        }
    })
}

/// WI-3: build a `ConstrainedSampler` from an OpenAI-compatible `response_format` field, wrapping the given inner sampler.
/// Returns `Ok(arc)` on success, `Err(message)` on an unsupported schema feature (callers return a structured 400.
fn build_constrained_sampler(
    inner: std::sync::Arc<dyn grim_core::sampler::Sampler>,
    body: &serde_json::Map<String, serde_json::Value>,
    vocab: Option<std::sync::Arc<[String]>>,
) -> std::result::Result<std::sync::Arc<dyn grim_core::sampler::Sampler>, String> {
    let Some(rf) = body.get("response_format") else {
        return Ok(inner);
    };
    let obj = rf
        .as_object()
        .ok_or_else(|| "response_format must be an object with a 'type' field".to_string())?;
    let ty = obj.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        "response_format.type is required ('text', 'json_object', 'json_schema')".to_string()
    })?;
    match ty {
        "text" | "text/plain" => Ok(inner),
        "json_object" => {
            let v = vocab.ok_or_else(|| {
                "constrained generation ('json_object') requires an attached model vocabulary; none available"
                    .to_string()
            })?;
            use grim_constrain::constrained_json_object;
            let s = constrained_json_object(inner).with_vocab(v);
            Ok(std::sync::Arc::new(s))
        }
        "json_schema" => {
            let schema = obj.get("json_schema").cloned().ok_or_else(|| {
                "response_format.json_schema is required when type='json_schema'".to_string()
            })?;
            let v = vocab.ok_or_else(|| {
                "constrained generation ('json_schema') requires an attached model vocabulary; none available"
                    .to_string()
            })?;
            use grim_constrain::{ConstrainedSampler, Constraint};
            let constraint = Constraint::json_schema(schema).map_err(|e| e.to_string())?;
            let s = ConstrainedSampler::new(inner, constraint).with_vocab(v);
            Ok(std::sync::Arc::new(s))
        }
        other => Err(format!(
            "unsupported response_format.type '{other}'; expected 'text', 'json_object', or 'json_schema'"
        )),
    }
}

/// Chat completions endpoint - SSE streaming (§8, §4.5).
/// §13.3 contract: no silent partial fulfillment.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let body_obj = body.as_object().cloned().unwrap_or_default();

    // WI-HYBRID Layer 2: optional session identity — `session` body field or
    // `x-grim-session` header. No session => plain Layer 1.5 behavior.
    let session_tag = body_obj
        .get("session")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| {
            headers
                .get("x-grim-session")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        });
    // The tag rides the stream-task locals and is keyed by request id once
    // the stream mints one (REQUEST_SESSIONS insert at the mint site below).
    let session_tag_for_task = session_tag.clone();

    let requested_model = body_obj
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    if let Some(resp) = ensure_model_available(&state, &requested_model) {
        return resp;
    }

    if let Err(resp) = validate_chat_request(&state, &body_obj) {
        return resp;
    }

    let stream_requested = body_obj
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let adapter_names = match resolve_adapters(&state, &body_obj) {
        Ok(names) => names,
        Err(resp) => return resp,
    };

    let sampling = match build_chat_sampling(&state, &body_obj, &requested_model) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    // Parse `messages` into typed structs (and `tools` / `tool_choice`), then
    // render the prompt once, before the streaming / non-streaming split.
    let (messages, tools, tool_choice, tools_active) = match parse_chat_messages(&body_obj) {
        Ok(parsed) => parsed,
        Err(resp) => return resp,
    };

    let prompt = match render_chat_prompt(
        &state,
        &messages,
        &tools,
        tool_choice.as_ref(),
        tools_active,
        sampling.max_tokens,
    ) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let parts = ChatRequestParts {
        body_obj,
        session_tag: session_tag_for_task,
        requested_model,
        adapter_names,
        stream_requested,
        thinking_level: sampling.thinking_level,
        sampler: sampling.sampler,
        max_tokens: sampling.max_tokens,
        stop_sequences: sampling.stop_sequences,
        messages,
        tools_active,
        template_family: prompt.template_family,
        prompt_tokens: prompt.tokens,
        vocab_size: prompt.vocab_size,
        eos_token_id: prompt.eos_token_id,
    };

    if parts.stream_requested {
        stream_chat_completion(state, parts).await
    } else {
        non_stream_chat_completion(state, parts).await
    }
}

/// All per-request state produced by the `chat_completions` prelude stages
/// and consumed by the streaming / non-streaming dispatch tails.
struct ChatRequestParts {
    body_obj: serde_json::Map<String, serde_json::Value>,
    session_tag: Option<String>,
    requested_model: String,
    adapter_names: Vec<String>,
    stream_requested: bool,
    thinking_level: grim_core::sampler::ThinkingLevel,
    sampler: std::sync::Arc<dyn grim_core::sampler::Sampler>,
    max_tokens: u64,
    stop_sequences: Vec<String>,
    messages: Vec<grim_format::ChatMessage>,
    tools_active: bool,
    template_family: Option<String>,
    prompt_tokens: Vec<u32>,
    vocab_size: usize,
    eos_token_id: Option<u32>,
}

/// Sampling/length controls resolved from the request body.
struct ChatSampling {
    thinking_level: grim_core::sampler::ThinkingLevel,
    sampler: std::sync::Arc<dyn grim_core::sampler::Sampler>,
    max_tokens: u64,
    stop_sequences: Vec<String>,
}

/// Prompt rendered from the messages and validated against the model context window.
struct ChatPrompt {
    tokens: Vec<u32>,
    vocab_size: usize,
    eos_token_id: Option<u32>,
    template_family: Option<String>,
}

/// Stage (a): dynamic model routing / on-demand loading. Returns `Some(err)`
/// for the 404 "model not found" response, `None` when a model is available.
fn ensure_model_available(state: &Arc<AppState>, requested_model: &str) -> Option<Response> {
    // WI-1: Remote Provider Routing - route only names carrying a *known* remote provider scheme (e.g.
    // "ollama:cloud", "openai:gpt-4", "hf/meta-llama/...").
    if is_remote_provider_model(requested_model) {
        let provider_key = requested_model.split(':').next().unwrap_or("default");
        let token = grim_core::client::load_login_token(provider_key)
            .ok()
            .flatten();
        eprintln!(
            "[grim-server] Routing request for model '{}' to remote provider '{}' (token present: {})",
            requested_model,
            provider_key,
            token.is_some()
        );
    } else {
        // Dynamic model loading - if the requested model is not yet registered, try to resolve it from the local catalog and load its GGUF file.
        // If the model cannot be resolved, return 404 immediately so the user gets a clear.
        let mut engine = state.lock_engine();
        if !engine
            .loaded_models()
            .contains(&requested_model.to_string())
        {
            match load_model_for_server(requested_model) {
                Ok((model, maybe_tokenizer)) => {
                    // SCYTHE-2 farm mode when armed (see /models load path);
                    // plain registration otherwise.
                    let farm_path = resolve_catalog_model_path(requested_model)
                        .map(|p| p.display().to_string())
                        .unwrap_or_default();
                    engine.register_model_with_farm(requested_model, model, &farm_path);
                    eprintln!(
                        "[grim-server] Loaded model '{}' on demand.",
                        requested_model
                    );
                    if let Some(tok) = maybe_tokenizer {
                        *state.lock_tokenizer() = Some(tok);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[grim-server] Cannot load model '{}': {}",
                        requested_model, e
                    );
                    let mut body = request_error(
                        ErrorCode::InvalidRequest,
                        format!(
                            "Model '{}' is not loaded and could not be found in the catalog. \
                             Run 'grim pull {}' to download it first.",
                            requested_model, requested_model
                        ),
                    );
                    body["error"]["model"] = serde_json::json!(requested_model);
                    body["error"]["cause"] = serde_json::json!(e.to_string());
                    return Some((StatusCode::NOT_FOUND, Json(body)).into_response());
                }
            }
        }
    }
    None
}

// The error is a fully-formed HTTP `Response` by design; boxing it would ripple
// through every call site, so allow the large-Err lint here.
#[allow(clippy::result_large_err)]
/// Stage (a): §13.3 request validation - unknown-field whitelist, per-request
/// message cap, and determinism-mode mismatch. Each violation is an early 400.
fn validate_chat_request(
    state: &Arc<AppState>,
    body_obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<(), Response> {
    // §13.3 - Exhaustive whitelist of known top-level request fields.
    // Any field outside this set is an immediate 400.
    const KNOWN_FIELDS: &[&str] = &[
        "model",
        "messages",
        "stream",
        "adapter",
        "adapters",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "stop",
        "determinism",
        "tools",
        "tool_choice",
        "sampler",
        "reasoning_effort",
        "thinking",
        // WI-3: OpenAI-compatible `response_format` — constrains generation
        // to JSON-mode or JSON-Schema via `grim-constrain::ConstrainedSampler`.
        "response_format",
        "user",
        "seed",
        "n",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "frequency_penalty",
        // WI-HYBRID Layer 2: optional session identity (slot affinity + block
        // pinning). Absence degrades to plain Layer 1.5 behavior.
        "session",
    ];
    for key in body_obj.keys() {
        if !KNOWN_FIELDS.contains(&key.as_str()) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json({
                    let mut body = request_error(
                        ErrorCode::UnknownField,
                        format!(
                            "unknown request field '{}'. Known fields: {}. \
                             If you need permissive parsing, set 'permissive: true' (phase 5).",
                            key,
                            KNOWN_FIELDS.join(", ")
                        ),
                    );
                    body["error"]["unknown_field"] = key.clone().into();
                    body["error"]["known_fields"] = KNOWN_FIELDS
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .into();
                    body
                }),
            )
                .into_response());
        }
    }

    // WI-TOOLS-4c-ii: cap `messages.len()` before any tokenization/prefill work happens - this is a conversation-shape check, not tool-call-specific, so it runs alongside the other pre-generation §13.3 validations above.
    // Uses the raw body field count so the check is available before the messages are.
    {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        let max_messages = engine.config.max_messages_per_request;
        if let Some(arr) = body_obj.get("messages").and_then(|v| v.as_array()) {
            if arr.len() > max_messages {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json({
                        let mut body = request_error(
                            ErrorCode::MessageCountLimit,
                            format!(
                                "request 'messages' length {} exceeds the per-request cap of {}",
                                arr.len(),
                                max_messages
                            ),
                        );
                        body["error"]["messages_len"] = arr.len().into();
                        body["error"]["max_messages_per_request"] = max_messages.into();
                        body
                    }),
                )
                    .into_response());
            }
        }
    }

    // §13.3 - Determinism mismatch: if the client requests strict determinism but the engine is in Relaxed mode, return 400.
    // Silently falling back to non-deterministic output would be a silent correctness bug.
    if let Some(det) = body_obj.get("determinism").and_then(|v| v.as_str()) {
        if det == "strict" {
            let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
            if engine.config.determinism_mode == DeterminismMode::Relaxed {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json({
                        let mut body = request_error(
                            ErrorCode::DeterminismMismatch,
                            "determinism 'strict' requested but engine is in Relaxed mode. \
                             Start the engine with DeterminismMode::Strict to use this field.",
                        );
                        body["error"]["determinism_requested"] = "strict".into();
                        body["error"]["engine_mode"] = "relaxed".into();
                        body
                    }),
                )
                    .into_response());
            }
        }
    }
    Ok(())
}

// The error is a fully-formed HTTP `Response` by design; boxing it would ripple
// through every call site, so allow the large-Err lint here.
#[allow(clippy::result_large_err)]
/// Stage (a): §13.3 + §4.5 — resolve adapter names from the request body and
/// validate they are all registered. Any unrecognised name is a hard 400.
fn resolve_adapters(
    state: &Arc<AppState>,
    body_obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<Vec<String>, Response> {
    let mut adapter_names: Vec<String> = body_obj
        .get("adapters")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if let Some(single) = body_obj.get("adapter").and_then(|v| v.as_str()) {
        if !adapter_names.iter().any(|a| a == single) {
            adapter_names.push(single.to_string());
        }
    }

    // Validate all requested adapters exist before starting the stream.
    {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        for name in &adapter_names {
            if engine.get_adapter_by_name(name).is_none() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json({
                        let mut body = request_error(
                            ErrorCode::AdapterNotFound,
                            format!(
                                "adapter '{}' is not registered. \
                                 Load it first with grim-engine::register_adapter().",
                                name
                            ),
                        );
                        body["error"]["unknown_adapter"] = name.clone().into();
                        body
                    }),
                )
                    .into_response());
            }
        }
    }
    Ok(adapter_names)
}

// The error is a fully-formed HTTP `Response` by design; boxing it would ripple
// through every call site, so allow the large-Err lint here.
#[allow(clippy::result_large_err)]
/// Stage (a): read sampling / length controls from the whitelisted request
/// fields, build the (possibly plugin-provided and response_format-constrained)
/// sampler, and resolve `max_tokens` / `stop`.
fn build_chat_sampling(
    state: &Arc<AppState>,
    body_obj: &serde_json::Map<String, serde_json::Value>,
    requested_model: &str,
) -> std::result::Result<ChatSampling, Response> {
    // These were already accepted by the KNOWN_FIELDS gate above; here we actually honor them instead.
    let thinking_str = body_obj
        .get("reasoning_effort")
        .or_else(|| body_obj.get("thinking"))
        .and_then(|v| v.as_str())
        .or_else(|| {
            body_obj
                .get("thinking")
                .and_then(|v| v.as_bool())
                .map(|b| if b { "on" } else { "off" })
        });
    let thinking_level = thinking_str
        .map(grim_core::sampler::ThinkingLevel::parse)
        .unwrap_or_default();

    let sampling = grim_core::sampler::SamplingParams {
        temperature: body_obj
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32,
        top_p: body_obj
            .get("top_p")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32,
        top_k: body_obj.get("top_k").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        repeat_penalty: body_obj
            .get("repeat_penalty")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32,
        thinking_level,
        min_tokens: body_obj
            .get("min_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
    };
    // A per-request seed keeps stochastic sampling reproducible for a given
    // (model, request) without a global RNG; temperature == 0 path ignores it.
    let sample_seed = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::Hasher;
        hasher.write_u64(REQUEST_SEED_SALT);
        hasher.write(requested_model.as_bytes());
        hasher.write_u64(now_millis());
        hasher.finish()
    };
    let sampler: std::sync::Arc<dyn grim_core::sampler::Sampler> =
        if let Some(name) = body_obj.get("sampler").and_then(|v| v.as_str()) {
            // A named plugin sampler was requested. Look it up in the registry threaded in from `grim_server::serve()`; if the name is unknown
            // (or no registry is attached), degrade gracefully and warn loudly rather than 400-ing - matching the repo's posture for optional features.
            state
                .plugin_registry
                .as_ref()
                .and_then(|r| r.get_sampler(name))
                .unwrap_or_else(|| {
                    eprintln!(
                        "[grim-server] WARNING: sampler '{name}' not found in plugin registry; \
                         falling back to SamplingParams-built sampler."
                    );
                    std::sync::Arc::from(sampling.into_sampler(sample_seed))
                })
        } else {
            std::sync::Arc::from(sampling.into_sampler(sample_seed))
        };

    // WI-3: `response_format` wraps the chosen sampler in a `ConstrainedSampler` so generated tokens stay on a valid JSON/JSON-Schema path.
    // The inner sampler (plugin or SamplingParams) is unmodified - this is wrapping, not altering, per.
    let vocab = state
        .tokenizer
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|t| std::sync::Arc::from(t.tokens.clone()) as std::sync::Arc<[String]>);
    let sampler: std::sync::Arc<dyn grim_core::sampler::Sampler> =
        match build_constrained_sampler(sampler, body_obj, vocab) {
            Ok(s) => s,
            Err(msg) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(request_error(
                        ErrorCode::InvalidRequest,
                        format!("invalid response_format: {msg}"),
                    )),
                )
                    .into_response());
            }
        };

    // `max_tokens` bounds generation length; default to a sane non-infinite
    // cap. `stop` sequences end the loop when a decoded token matches.
    let max_tokens: u64 = body_obj
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let stop_sequences: Vec<String> = body_obj
        .get("stop")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    Ok(ChatSampling {
        thinking_level,
        sampler,
        max_tokens,
        stop_sequences,
    })
}

/// Stage (b): parse `messages` into typed structs, apply the empty-messages /
/// vision checks, parse `tools` / `tool_choice`, and inject the synthetic
/// system message when tool calling is active without an existing system turn.
#[allow(clippy::type_complexity)]
// The error is a fully-formed HTTP `Response` by design; boxing it would ripple
// through every call site, so allow the large-Err lint here.
#[allow(clippy::result_large_err)]
fn parse_chat_messages(
    body_obj: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<
    (
        Vec<grim_format::ChatMessage>,
        Vec<grim_format::ToolDef>,
        Option<grim_format::ToolChoice>,
        bool,
    ),
    Response,
> {
    let mut messages: Vec<grim_format::ChatMessage> = Vec::new();
    let mut image_parts: usize = 0;
    if let Some(arr) = body_obj.get("messages").and_then(|v| v.as_array()) {
        for (idx, v) in arr.iter().enumerate() {
            let normalized = match v.get("content").and_then(|c| c.as_array()) {
                Some(parts) => {
                    let mut text = String::new();
                    let mut images = 0usize;
                    for p in parts {
                        match p.get("type").and_then(|t| t.as_str()) {
                            Some("text") => {
                                if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
                                        text.push('\n');
                                    }
                                    text.push_str(t);
                                }
                            }
                            Some("image_url") => images += 1,
                            other => {
                                return Err((
                                    StatusCode::BAD_REQUEST,
                                    Json(request_error(
                                        ErrorCode::UnknownField,
                                        &format!(
                                            "malformed message at index {idx}: unsupported content part type {other:?} (expected text or image_url)"
                                        ),
                                    )),
                                ).into_response());
                            }
                        }
                    }
                    image_parts += images;
                    let mut norm = v.clone();
                    norm["content"] = serde_json::Value::String(text);
                    norm
                }
                None => v.clone(),
            };
            match serde_json::from_value(normalized) {
                Ok(msg) => messages.push(msg),
                Err(e) => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(request_error(
                            ErrorCode::UnknownField,
                            &format!("malformed message at index {idx}: {e}"),
                        )),
                    )
                        .into_response());
                }
            }
        }
    }
    if messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json({
                let mut body = request_error(
                    ErrorCode::EmptyMessages,
                    "request must include at least one message in 'messages'",
                );
                body["error"]["messages"] = serde_json::json!([]);
                body
            }),
        )
            .into_response());
    }
    // Image parts are only servable by a model whose modality hint includes vision.
    // No such model is loadable in the serving path today, so this fires for every.
    if image_parts > 0 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(request_error(
                ErrorCode::InvalidRequest,
                &format!(
                    "request contains {image_parts} image part(s) but the loaded model has no vision encoder; pass text-only content or load a vision model"
                ),
            )),
        ).into_response());
    }

    // §WI-TOOLS-1 - Parse `tools` / `tool_choice` into the typed shapes the template renderer and output parser consume.
    // Field-by-field extraction with explicit error messages on malformed input (matching the existing `adapters` pattern above,.
    let tools: Vec<grim_format::ToolDef> = body_obj
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    let tool_choice: Option<grim_format::ToolChoice> = body_obj
        .get("tool_choice")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .or(None);
    // `tool_choice: "none"` suppresses tool calling entirely (WI-TOOLS-1), matching OpenAI semantics: the template gets
    // no `tools` and the parser is bypassed, so the model produces an ordinary completion.
    let tools_active = !tools.is_empty() && tool_choice != Some(grim_format::ToolChoice::None);

    // F-4 (WI-TOOLS): when tool calling is active and the model's embedded template does not itself instruct the model on the output convention, prepend a synthetic system message spelling out the exact wire format.
    // Without this, a template that merely renders "List of tools: [...]" gives the model no.
    let messages = if tools_active && !messages.iter().any(|m| m.role == "system") {
        let mut with_tools = Vec::with_capacity(messages.len() + 1);
        with_tools.push(grim_format::ChatMessage {
            role: "system".to_string(),
            content: TOOL_INSTRUCTION_PROMPT.to_string(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
        with_tools.extend(messages.iter().cloned());
        with_tools
    } else {
        messages.to_vec()
    };

    Ok((messages, tools, tool_choice, tools_active))
}

// The error is a fully-formed HTTP `Response` by design; boxing it would ripple
// through every call site, so allow the large-Err lint here.
#[allow(clippy::result_large_err)]
/// Stage (b)+(c): render the prompt (template / tools), tokenize (with the
/// opt-in compression gate), and enforce the model context window.
fn render_chat_prompt(
    state: &Arc<AppState>,
    messages: &[grim_format::ChatMessage],
    tools: &[grim_format::ToolDef],
    tool_choice: Option<&grim_format::ToolChoice>,
    tools_active: bool,
    max_tokens: u64,
) -> std::result::Result<ChatPrompt, Response> {
    // `template_family` drives WI-TOOLS-4's per-family output parsing.
    // We resolve it from the loaded tokenizer's embedded chat template so the same model template.
    let (prompt_text, template_family) = {
        let tok = state.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
        let effective_messages = if !messages.iter().any(|m| m.role == "system") {
            if let Some(default_sys) = tok.as_ref().and_then(|t| t.default_system_prompt()) {
                let mut with_sys = Vec::with_capacity(messages.len() + 1);
                with_sys.push(grim_format::ChatMessage {
                    role: "system".to_string(),
                    content: default_sys.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
                with_sys.extend(messages.iter().cloned());
                with_sys
            } else {
                messages.to_vec()
            }
        } else {
            messages.to_vec()
        };

        match tok.as_ref() {
            Some(t) if tools_active => {
                let family = t.chat_template.clone();
                let text = grim_format::render_messages_or_last_with_tools(
                    t,
                    &effective_messages,
                    Some(tools),
                    tool_choice,
                );
                (text, family)
            }
            Some(t) => (grim_format::render_messages_or_last(t, &effective_messages), None),
            None => (
                effective_messages
                    .last()
                    .map(|m| m.content.clone())
                    .unwrap_or_default(),
                None,
            ),
        }
    };
    let prompt_tokens: Vec<u32> = {
        let tok = state.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
        let tokens = tok
            .as_ref()
            .map(|t| t.encode(&prompt_text))
            .unwrap_or_default();
        if tokens.is_empty() { vec![1] } else { tokens }
    };
    if std::env::var("GRIM_DEBUG_PROMPT").as_deref() == Ok("1") {
        eprintln!(
            "[grim-server] prompt tokens ({}): {:?}\n[grim-server] prompt text: {:?}",
            prompt_tokens.len(),
            prompt_tokens,
            prompt_text
        );
    }

    // Prompt compression: if the prompt is long and compression is enabled,
    // compress the raw text before tokenization to reduce prefill cost and
    // KV cache pressure. Uses extractive compression (TextRank + TF-IDF +
    // position weighting + novelty scoring) — no neural inference, preserves
    // original tokens verbatim. Gate: GRIM_COMPRESS_PROMPT=1.
    let prompt_tokens: Vec<u32> = if std::env::var("GRIM_COMPRESS_PROMPT").as_deref() == Ok("1") {
        let budget = grim_compress::DEFAULT_TOKEN_BUDGET;
        if prompt_text.len() > budget * 4 {
            // Only compress if the raw text is substantially larger than the budget
            let compressed = grim_compress::compress_prompt(&prompt_text, budget);
            if compressed != prompt_text {
                eprintln!(
                    "[grim-server] prompt compressed: {} chars -> {} chars",
                    prompt_text.len(),
                    compressed.len()
                );
                let tok = state.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
                let new_tokens = tok
                    .as_ref()
                    .map(|t| t.encode(&compressed))
                    .unwrap_or_default();
                if new_tokens.is_empty() {
                    vec![1]
                } else {
                    new_tokens
                }
            } else {
                prompt_tokens
            }
        } else {
            prompt_tokens
        }
    } else {
        prompt_tokens
    };

    // P0-3.2: Vocab size for clamping sampled tokens into the model's actual range.
    // The engine's internal logits table is fixed at 65536 entries; a model with a smaller.
    let vocab_size: usize = state
        .tokenizer
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|t| t.tokens.len())
        .unwrap_or(65536);

    // EOS token ID for early termination. When the model emits this token,
    // generation stops immediately (the EOS token is not included in the returned content).
    let eos_token_id: Option<u32> = state
        .tokenizer
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|t| t.eos_token_id);

    // Enforce the model's context window: reject requests whose total token count (prompt + max_tokens) exceeds the model's reported context_length.
    // Models that don't report context_length (return 0) fall back to a best-effort warning for obviously.
    let model_context_length = {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        engine
            .loaded_models()
            .iter()
            .filter_map(|name| engine.model(name))
            .next()
            .map(|m| m.config.context_length())
            .unwrap_or(0)
    };
    let context_limit = if model_context_length > 0 {
        model_context_length as usize
    } else {
        8192
    };
    let total_requested = prompt_tokens.len().saturating_add(max_tokens as usize);
    if total_requested > context_limit {
        return Err((
            StatusCode::BAD_REQUEST,
            Json({
                let mut body = request_error(
                    ErrorCode::InvalidRequest,
                    format!(
                        "prompt ({} tokens) + max_tokens ({}) = {} tokens exceeds \
                         model context limit ({} tokens)",
                        prompt_tokens.len(),
                        max_tokens,
                        total_requested,
                        context_limit,
                    ),
                );
                body["error"]["code"] = serde_json::json!("context_length_exceeded");
                body["error"]["context_length"] = serde_json::json!(context_limit);
                body["error"]["prompt_tokens"] = serde_json::json!(prompt_tokens.len());
                body["error"]["max_tokens"] = serde_json::json!(max_tokens);
                body["error"]["total_requested"] = serde_json::json!(total_requested);
                body
            }),
        )
            .into_response());
    } else if total_requested > 1_000_000 {
        eprintln!(
            "[Server] WARNING: prompt ({} tokens) + max_tokens ({}) = {} tokens \
             exceeds 1M. Model context_length = {} (enforcement skipped if 0).",
            prompt_tokens.len(),
            max_tokens,
            total_requested,
            model_context_length
        );
    }

    Ok(ChatPrompt {
        tokens: prompt_tokens,
        vocab_size,
        eos_token_id,
        template_family,
    })
}

/// Stage (d): the streaming dispatch tail. Buffers generation into an SSE
/// `unfold` stream with tool-call post-processing and a `[DONE]` sentinel.
async fn stream_chat_completion(state: Arc<AppState>, parts: ChatRequestParts) -> Response {
    let ChatRequestParts {
        body_obj,
        session_tag: session_tag_for_task,
        requested_model,
        adapter_names,
        stream_requested: _,
        thinking_level,
        sampler,
        max_tokens,
        stop_sequences,
        messages,
        tools_active,
        template_family,
        prompt_tokens,
        vocab_size,
        eos_token_id,
    } = parts;
    let state_clone = state.clone();
    let adapter_ids: Vec<u32> = {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        adapter_names
            .iter()
            .filter_map(|name| engine.get_adapter_by_name(name).map(|a| a.handle.id))
            .collect()
    };
    let adapter_ids_clone = adapter_ids.clone();
    let sampler_clone = sampler.clone();
    let stop_sequences_clone = stop_sequences.clone();
    let max_tokens_clone = max_tokens;
    let eos_token_id_clone = eos_token_id;

    // WI-TOOLS-5 (streaming MVP, buffered): true incremental tool-call streaming is not achievable while parsing is still post-hoc (WI- TOOLS-4) - you cannot confidently detect a marker-delimited call is complete until you see the closing tag, which only happens at or near end-of-generation.
    // So we buffer the full completion in `emitted` (already done for stop-sequence detection) and, once.
    let tools_active_clone = tools_active;
    let template_family_clone = template_family.clone();

    // CRIT-1: generate ONE request_id for the entire streaming session so sample_next_token enqueues a request on step 0 and can look up the outcome on every subsequent step.
    // The previous code created a new id per step, meaning no request existed under that.
    let session_request_id = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    // WI-HYBRID Layer 2: key the session tag by request id —
    // sample_next_token attaches it to the engine Request at enqueue.
    if let Some(sess) = &session_tag_for_task {
        if let Ok(mut map) = REQUEST_SESSIONS.lock() {
            map.insert(session_request_id, sess.clone());
        }
    }
    // T1.3: request params feed the device-side sampler for this stream.
    register_request_sampler_params(session_request_id, &body_obj);

    // WI-CANCEL-1: register a CancellationToken so /v1/requests/:id/cancel
    // can signal this specific stream to stop.
    let cancel_token = register_cancel_token(session_request_id);

    // WI-CANCEL-2: RAII guard that calls finish_request on drop - fires on every exit path
    // (max_tokens, stop-sequence, explicit cancel, client disconnect) since it's threaded through the unfold state tuple.
    let cleanup_guard = RequestCleanupGuard::new(state.clone(), session_request_id);

    let stream = futures::stream::unfold(
        (
            0u64,
            String::new(),
            prompt_tokens.clone(),
            session_request_id,
            cancel_token,
            cleanup_guard,
        ),
        move |(step, mut emitted, prompt_tokens, request_id, cancel_token, cleanup_guard): (
            u64,
            String,
            Vec<u32>,
            u64,
            CancellationToken,
            RequestCleanupGuard,
        )| {
            let state = state_clone.clone();
            let adapter_ids = adapter_ids_clone.clone();
            let stop_seqs = stop_sequences_clone.clone();
            let request_sampler_params = take_request_sampler_params(session_request_id);
            let sampler = sampler_clone.clone();
            let parse_ctx = (
                tools_active_clone,
                template_family_clone.clone(),
                state
                    .model_arch
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
            );
            let prior_messages = messages.clone();
            let req_model = requested_model.clone();
            let stream_model = requested_model.clone();
            async move {
                // WI-CANCEL-1: check for explicit cancel before doing work.
                // The cancel endpoint calls cancel_token.cancel(); we poll it cooperatively each tick (matching the spec's tick-boundary.
                if cancel_token.is_cancelled() {
                    let _ = cleanup_guard; // consumed; Drop fires on move-into-scope end
                    return None;
                }

                // Honor `max_tokens` (was a hardcoded 256). Stop early if a
                // configured stop sequence appears in the emitted text.
                if step >= max_tokens_clone {
                    // End of generation reached - attempt WI-TOOLS-4 post-hoc tool-call extraction on the buffered completion.
                    // The result (Some terminal delta, or None to close) becomes the final unfold item; the.
                    let (reasoning_content, clean_emitted) =
                        if thinking_level != grim_core::sampler::ThinkingLevel::Off {
                            split_think_content(&emitted)
                        } else {
                            (None, emitted.clone())
                        };
                    let delta = terminal_tool_delta(
                        &parse_ctx,
                        &clean_emitted,
                        &prior_messages,
                        reasoning_content.as_deref(),
                    );
                    return delta.map(|ev| {
                        (
                            ev,
                            (
                                step + 1,
                                emitted,
                                prompt_tokens,
                                request_id,
                                cancel_token,
                                cleanup_guard,
                            ),
                        )
                    });
                }

                let sampled = {
                    let mut engine = match state.engine.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    sample_next_token(
                        &mut engine,
                        request_id,
                        step,
                        sampler.as_ref(),
                        if step == 0 {
                            Some(&prompt_tokens)
                        } else {
                            None
                        },
                        vocab_size,
                        Some(req_model),
                    )
                };
                // WI-1: a generation failure ends the stream with a terminal OpenAI-shaped error
                // event; the chained `[DONE]` sentinel still fires because the task does not unwind.
                let token_id = match sampled {
                    Ok(t) => t,
                    Err(msg) => {
                        let payload = serde_json::json!({
                            "error": {
                                "code": "generation_failed",
                                "message": msg,
                            }
                        })
                        .to_string();
                        let ev = axum::response::sse::Event::default()
                            .event("error")
                            .data(payload);
                        return Some((
                            Ok(ev),
                            (
                                max_tokens_clone,
                                emitted,
                                prompt_tokens,
                                request_id,
                                cancel_token,
                                cleanup_guard,
                            ),
                        ));
                    }
                };

                // Token pacing: opt-in inter-token delay for clients that need
                // it. Default 0 (no artificial pacing): SSE backpressure
                // already propagates via the write path — a fixed sleep
                // only adds self-inflicted latency. Set
                // GRIM_TOKEN_PACING_MS=N to pace explicitly.
                let pacing_ms = std::env::var("GRIM_TOKEN_PACING_MS")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0);
                if pacing_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(pacing_ms)).await;
                }

                let tokenizer = state
                    .tokenizer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                let token_text = if let Some(tok) = &tokenizer {
                    tok.decode(&[token_id])
                } else {
                    format!("<tok:{token_id}>")
                };
                emitted.push_str(&token_text);
                let hit_stop = stop_seqs.iter().any(|s| emitted.contains(s));
                // EOS check: if the model emitted the EOS token, terminate generation without including it
                // in the output (the EOS token is a signal, not content - OpenAI convention).
                // P1-3 (PLAN-improve-grim-perf): honor min_tokens — EOS
                // is ignored until at least `min_tokens` tokens have been
                // generated. `step` is 0-based, so EOS is legal from
                // step == min_tokens onward.
                let min_tokens_enforced = request_sampler_params.min_tokens;
                let hit_eos =
                    eos_token_id_clone == Some(token_id) && step >= u64::from(min_tokens_enforced);
                if hit_eos {
                    // Trim the EOS token's text from the emitted buffer
                    // so it doesn't appear in the response.
                    emitted = emitted
                        .strip_suffix(&token_text)
                        .unwrap_or(&emitted)
                        .to_string();
                }
                if hit_stop {
                    // Trim the stop string from the buffered text used
                    // for terminal tool-call parsing (suffix-trim is enough for parse purposes).
                    let (trimmed, _) = trim_stop_sequences(&emitted, &stop_seqs);
                    emitted = trimmed;
                }
                if hit_stop || hit_eos {
                    // A stop sequence or EOS terminated generation early —
                    // same end-of-stream tool-call extraction path as max_tokens.
                    let (reasoning_content, clean_emitted) =
                        if thinking_level != grim_core::sampler::ThinkingLevel::Off {
                            split_think_content(&emitted)
                        } else {
                            (None, emitted.clone())
                        };
                    let delta = terminal_tool_delta(
                        &parse_ctx,
                        &clean_emitted,
                        &prior_messages,
                        reasoning_content.as_deref(),
                    );
                    if let Some(ev) = delta {
                        return Some((
                            ev,
                            (
                                step + 1,
                                emitted,
                                prompt_tokens,
                                request_id,
                                cancel_token,
                                cleanup_guard,
                            ),
                        ));
                    }
                    // WI-P9: no tool call - the stop-triggering token's text must still reach the client, or stream:true silently drops the final content the non-streaming path returns.
                    // Emit it stop-stripped (signal, not content) in the same chunk shape as every other delta,.
                    if hit_stop && !clean_emitted.is_empty() {
                        let (stripped, _) = strip_stop_sequences(&clean_emitted, &stop_seqs);
                        let prior_raw_len = emitted.len() - token_text.len();
                        let delta_content = if stripped.len() > prior_raw_len {
                            stripped[prior_raw_len..].to_string()
                        } else {
                            String::new()
                        };
                        if !delta_content.is_empty() {
                            let payload = serde_json::json!({
                                "object": "chat.completion.chunk",
                                "model": stream_model,
                                "choices": [{"index": 0, "delta": {"content": delta_content}, "finish_reason": "stop"}],
                                "adapters_active": adapter_ids.len(),
                                // True sampled-token count: this chunk may bundle more
                                // than one token's worth of text (stop-string trimming
                                // collapses into one frame), so chunk-counting
                                // downstream undercounts.
                                "grim_eval_count": step + 1
                            })
                            .to_string();
                            let event = axum::response::sse::Event::default()
                                .event("message")
                                .data(payload);
                            return Some((
                                Ok(event),
                                (
                                    max_tokens_clone,
                                    emitted,
                                    prompt_tokens,
                                    request_id,
                                    cancel_token,
                                    cleanup_guard,
                                ),
                            ));
                        }
                    }
                    return None;
                }
                // WI-2: streaming chunks echo the requested model too, so
                // clients validating `chunk.model` see what they sent.
                let payload = serde_json::json!({
                   "object": "chat.completion.chunk",
                   "model": stream_model,
                   "choices": [{"index": 0, "delta": {"content": token_text}}],
                   "adapters_active": adapter_ids.len(),
                   // True sampled-token count (downstream translators should
                   // prefer this over counting chunks).
                   "grim_eval_count": step + 1
                })
                .to_string();
                let event = axum::response::sse::Event::default()
                    .event("message")
                    .data(payload);
                let res: std::result::Result<axum::response::sse::Event, axum::Error> = Ok(event);
                Some((
                    res,
                    (
                        step + 1,
                        emitted,
                        prompt_tokens,
                        request_id,
                        cancel_token,
                        cleanup_guard,
                    ),
                ))
            }
        },
    );
    Sse::new(stream.chain(futures::stream::once(async {
        Ok(axum::response::sse::Event::default().data("[DONE]"))
    })))
    .into_response()
}

/// Stage (d): the non-streaming dispatch tail. Generates the full completion,
/// runs tool-call guards, and builds the OpenAI-shaped response payload.
#[allow(clippy::too_many_lines)]
async fn non_stream_chat_completion(state: Arc<AppState>, parts: ChatRequestParts) -> Response {
    let ChatRequestParts {
        body_obj,
        session_tag: _,
        requested_model,
        adapter_names,
        stream_requested: _,
        thinking_level,
        sampler,
        max_tokens,
        stop_sequences,
        messages,
        tools_active,
        template_family,
        prompt_tokens,
        vocab_size,
        eos_token_id,
    } = parts;
    let mut content = String::new();
    let request_id = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    // T1.3: request params feed the device-side sampler for this request.
    register_request_sampler_params(request_id, &body_obj);
    let _adapter_ids: Vec<u32> = {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        adapter_names
            .iter()
            .filter_map(|name| engine.get_adapter_by_name(name).map(|a| a.handle.id))
            .collect()
    };

    let tokenizer = state
        .tokenizer
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    // Tokenize the prompt once for prefill (rendered from messages above)
    let prompt_tokens = prompt_tokens.clone();
    // Honor `max_tokens` (was a hardcoded 5) and stop sequences.
    for step in 0..max_tokens {
        let sampled = {
            let mut engine = match state.engine.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            sample_next_token(
                &mut engine,
                request_id,
                step,
                sampler.as_ref(),
                if step == 0 {
                    Some(&prompt_tokens)
                } else {
                    None
                },
                vocab_size,
                Some(requested_model.to_string()),
            )
        };
        // WI-1: propagate a clean OpenAI-shaped 500 instead of panicking
        // inside the handler while holding the engine mutex.
        let token_id = match sampled {
            Ok(t) => t,
            Err(msg) => {
                let mut engine = match state.engine.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                engine.finish_request(request_id);
                take_request_sampler_params(request_id);
                drop(engine);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "code": "generation_failed",
                            "message": msg,
                            "type": "server_error",
                        },
                        "model": requested_model,
                    })),
                )
                    .into_response();
            }
        };
        let token_text = if let Some(tok) = &tokenizer {
            tok.decode(&[token_id])
        } else {
            format!("<tok:{token_id}>")
        };
        content.push_str(&token_text);
        // EOS check: stop generation if the model emitted the EOS token, and strip the
        // EOS token's text from the output (it's a signal, not content - OpenAI convention).
        if eos_token_id == Some(token_id) {
            content = content
                .strip_suffix(&token_text)
                .unwrap_or(&content)
                .to_string();
            break;
        }
        if stop_sequences.iter().any(|s| content.contains(s)) {
            break;
        }
    }

    // Strip stop-sequence occurrences from the returned content (OpenAI convention: the stop string is a signal, not part of the output).
    // WI-P9: uses the same occurrence-strip as the streaming path's terminal delta, so stream:true and stream:false.
    let (content, _hit_stop) = strip_stop_sequences(&content, &stop_sequences);

    // Thinking output handling: when the model emits <think> blocks, split them into reasoning_content (chain-of-thought) and clean content (the actual response).
    // This mirrors DeepSeek-R1 / Qwen3-Thinking convention where the think preamble is surfaced separately.
    let (reasoning_content, content) = if thinking_level != grim_core::sampler::ThinkingLevel::Off {
        split_think_content(&content)
    } else {
        (None, content)
    };

    {
        let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        engine.finish_request(request_id);
    }
    take_request_sampler_params(request_id);

    // WI-TOOLS-4/5/4b: when tool calling is active, run the completion through the per-family output parser.
    // Before constructing the response, apply the WI-TOOLS-4b hard guard - if the parsed call would.
    if tools_active {
        let family = tool_parse::resolve_effective_tool_family(
            template_family.as_deref().unwrap_or(""),
            state
                .model_arch
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_deref(),
        );
        if let tool_parse::ParseOutcome {
            calls: Some(calls), ..
        } = tool_parse::parse_tool_calls(&content, family)
        {
            for c in &calls {
                if let Some(repeat) =
                    check_repeated_call_hard_guard(&messages, &c.name, &c.arguments)
                {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json({
                            let mut body = request_error(
                                ErrorCode::DuplicateToolCall,
                                format!(
                                    "Refusing to call tool '{}' — it has already been called {} times \
                                     with identical arguments in this conversation. This is the hard \
                                     guard (WI-TOOLS-4b) preventing a runaway agentic loop. Adjust the \
                                     arguments or try a different action.",
                                    c.name, repeat
                                ),
                            );
                            body["error"]["tool_name"] = c.name.clone().into();
                            body["error"]["repeat_count"] = repeat.into();
                            body
                        }),
                    )
                        .into_response();
                }
            }
            // WI-TOOLS-4c-i: total tool-call budget across the whole conversation.
            // If the newly parsed calls would push the cumulative count past the engine-config cap, reject.
            {
                let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
                let max_tool_calls = engine.config.max_tool_calls_per_conversation;
                let total_prior = tool_parse::count_total_prior_tool_calls(&messages);
                let total_with_new = total_prior + calls.len();
                if total_with_new > max_tool_calls {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json({
                            let mut body = request_error(
                                ErrorCode::TotalToolCallLimit,
                                format!(
                                    "Total tool calls across this conversation ({}) would exceed \
                                     the per-conversation budget of {}",
                                    total_with_new, max_tool_calls
                                ),
                            );
                            body["error"]["total_tool_calls"] = total_with_new.into();
                            body["error"]["max_tool_calls_per_conversation"] =
                                max_tool_calls.into();
                            body
                        }),
                    )
                        .into_response();
                }
            }
        }
    }
    let model_arch = state
        .model_arch
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let choice = build_choice_payload(
        &content,
        reasoning_content.as_deref(),
        tools_active,
        template_family.as_deref(),
        model_arch.as_deref(),
        &messages,
    );
    // WI-CANCEL-0: tear down engine-side request state on every exit path - non-streaming has no Drop guard, so we call finish_request directly here, on both the normal-completion and stop-sequence break paths (the loop above falls through to this point in both cases).
    // Idempotent per the audit: retain-based queue removal and refcount-decrement rollback are no-ops if state is.
    {
        let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        engine.finish_request(request_id);
    }
    take_request_sampler_params(request_id);
    // WI-2: echo back exactly the model name the client requested, per OpenAI API semantics.
    // The previous hardcoded "grim" broke any client that validates `response.model` against what it sent.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COMPLETION_COUNTER: AtomicU64 = AtomicU64::new(1);
    let completion_id = COMPLETION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let response_id = format!("chatcmpl-{completion_id:03}");
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Json(serde_json::json!({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": requested_model,
        "adapters_active": adapter_names.len(),
        "choices": [choice]
    }))
    .into_response()
}

/// §5.2.1 - pause a running request. Idempotent: if the request is already
/// paused (or finished), the response is `200 OK` with `{"state": "paused"}` regardless.
/// §5.2.1 - pause a running request. Idempotent: if the request is already
/// §5.1 - pause a running request (`POST /v1/requests/:id/pause`).
async fn pause_request(
    state: State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::control::pause_request_route(id, state).await
}

async fn resume_request(
    state: State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::control::resume_request_route(id, state).await
}

/// §5.2 - cancel a running request (`POST /v1/requests/:id/cancel`).
async fn cancel_request(
    state: State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::control::cancel_request_route(id, state).await
}

/// SSE stream of `pause`/`resume` events for a single request.
async fn stream_state(
    state: State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Sse<impl Stream<Item = std::result::Result<Event, axum::Error>>> {
    routes::control::stream_state_route(id, state).await
}

/// OpenAI-compatible embeddings endpoint.
async fn embeddings(
    state: State<Arc<AppState>>,
    payload: Json<routes::embeddings::EmbeddingRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::embeddings::embeddings_route(state, payload).await
}

#[cfg(test)]
pub(crate) use routes::audio::SpeechRequest;

/// OpenAI-compatible text-to-speech synthesis endpoint.
async fn audio_speech(
    state: State<Arc<AppState>>,
    payload: Json<routes::audio::SpeechRequest>,
) -> Response {
    routes::audio::audio_speech_route(state, payload).await
}

/// OpenAI-compatible audio transcriptions endpoint.
async fn audio_transcriptions(state: State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    routes::audio::audio_transcriptions_route(state, body).await
}

/// OpenAI-compatible audio translations endpoint.
async fn audio_translations(state: State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    routes::audio::audio_translations_route(state, body).await
}

/// OpenAI-compatible image generation endpoint.
async fn images_generations() -> (StatusCode, Json<serde_json::Value>) {
    // F5 stage-1 honesty: a loaded Flux2 transformer alone cannot generate - the pipeline needs prompt text-encoder conditioning and a trained VAE.
    // Returning unconditioned pixels decoded through a random VAE would fabricate output, so fail loudly instead.
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "created": 0,
            "data": [],
            "error": {
                "type": "not_implemented",
                "capability": "image_generation",
                "message": "text-conditioned generation is not wired in this build; requires a text encoder and a trained Flux 2 / UNet checkpoint with VAE"
            }
        })),
    )
}

/// gRPC service endpoint handler (§8).
/// Returns 501 Not Implemented unless compiled with the `grpc` feature.
async fn grpc_service_handler() -> (StatusCode, &'static str) {
    (
        StatusCode::NOT_IMPLEMENTED,
        "gRPC service pipeline requires compiling with --features grpc",
    )
}

/// Telemetry metrics endpoint (§8)
async fn metrics_endpoint(
    headers: axum::http::HeaderMap,
    State(state): State<Arc<AppState>>,
) -> axum::response::Response {
    // Keep metrics and status on one contract so probes and dashboards cannot
    // disagree about backend, model, or KV state. Legacy counters remain.
    let mut snapshot = get_status(State(state.clone())).await.0;
    let (
        active_sessions,
        block_pool_usage,
        preemption_count,
        scheduler_snapshot,
        last_ttft,
        last_itl,
        spec_tele,
        decode_tps,
        prefill_tps,
        total_tokens_gen,
        total_tokens_pref,
        kv_used_b,
        kv_total_b,
        kv_blocks_u,
        kv_blocks_t,
    ) = {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        let active = engine.adapter_count();
        let sched = engine.scheduler_snapshot();
        let (used_bytes, total_bytes, blocks_used, blocks_total) = engine.kv_cache_telemetry();
        let pool_usage = if blocks_total > 0 {
            blocks_used as f64 / blocks_total as f64
        } else {
            0.0
        };
        let ttft = engine.last_ttft_ms();
        let itl = engine.last_itl_ms();
        let spec = engine.speculative_telemetry(None);
        let dec_tps = engine.tokens_per_sec();
        let pref_tps = engine.prefill_tokens_per_sec();
        let tot_gen = engine.total_tokens_generated();
        let tot_pref = engine.total_tokens_prefilled();
        (
            active,
            pool_usage,
            sched.paused_requests,
            sched,
            ttft,
            itl,
            spec,
            dec_tps,
            pref_tps,
            tot_gen,
            tot_pref,
            used_bytes,
            total_bytes,
            blocks_used,
            blocks_total,
        )
    };

    if let Some(object) = snapshot.as_object_mut() {
        object.insert("active_sessions".into(), serde_json::json!(active_sessions));
        object.insert(
            "block_pool_usage".into(),
            serde_json::json!(block_pool_usage),
        );
        object.insert(
            "preemption_count".into(),
            serde_json::json!(preemption_count),
        );
        object.insert("ttft_ms".into(), serde_json::json!(last_ttft));
        object.insert("itl_ms".into(), serde_json::json!(last_itl));
        if let Some(ref st) = spec_tele {
            object.insert(
                "speculative_accept_rate_ema".into(),
                serde_json::json!(st.accept_rate_ema),
            );
            object.insert(
                "speculative_strategy".into(),
                serde_json::json!(st.strategy),
            );
            object.insert(
                "speculative_drafted_tokens".into(),
                serde_json::json!(st.total_drafted_tokens),
            );
            object.insert(
                "speculative_accepted_tokens".into(),
                serde_json::json!(st.total_accepted_tokens),
            );
        }
    }

    let accept = headers
        .get(axum::http::header::ACCEPT)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    if accept.contains("application/json") {
        return axum::response::Json(snapshot).into_response();
    }

    // WI (session-continuity Layer 1.5 gate): radix hit-rate metrics — the
    // "measure hit rate" observability, exported to Prometheus.
    let prefix_cache = snapshot.get("prefix_cache");
    let radix_lookups = prefix_cache
        .and_then(|pc| pc.get("lookups"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let radix_hit_requests = prefix_cache
        .and_then(|pc| pc.get("hit_requests"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let radix_hit_tokens = prefix_cache
        .and_then(|pc| pc.get("hit_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let radix_hit_rate = prefix_cache
        .and_then(|pc| pc.get("hit_rate"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let gpu_util = snapshot
        .get("gpu_util_pct")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let vram_used = (snapshot
        .get("vram_used_gb")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        * 1024.0
        * 1024.0
        * 1024.0) as u64;
    let vram_total = (snapshot
        .get("vram_total_gb")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        * 1024.0
        * 1024.0
        * 1024.0) as u64;
    let mut prometheus_text = format!(
        "# HELP grim_active_sessions Active LoRA adapters and inference sessions\n\
         # TYPE grim_active_sessions gauge\n\
         grim_active_sessions {active_sessions}\n\
         # HELP grim_block_pool_usage KV cache block pool utilization ratio\n\
         # TYPE grim_block_pool_usage gauge\n\
         grim_block_pool_usage {block_pool_usage:.4}\n\
         # HELP grim_preemption_count Cumulative request preemptions\n\
         # TYPE grim_preemption_count counter\n\
         grim_preemption_count {preemption_count}\n\
         # HELP grim_scheduler_active_requests Currently active requests\n\
         # TYPE grim_scheduler_active_requests gauge\n\
         grim_scheduler_active_requests {}\n\
         # HELP grim_scheduler_waiting_requests Currently waiting requests\n\
         # TYPE grim_scheduler_waiting_requests gauge\n\
         grim_scheduler_waiting_requests {}\n\
         # HELP grim_scheduler_admitted_requests Total admitted requests\n\
         # TYPE grim_scheduler_admitted_requests counter\n\
         grim_scheduler_admitted_requests {}\n\
         # HELP grim_gpu_util_pct Current GPU compute utilization\n\
         # TYPE grim_gpu_util_pct gauge\n\
         grim_gpu_util_pct {gpu_util:.2}\n\
         # HELP grim_vram_used_bytes VRAM memory currently allocated\n\
         # TYPE grim_vram_used_bytes gauge\n\
         grim_vram_used_bytes {vram_used}\n\
         # HELP grim_vram_total_bytes Total available VRAM memory\n\
         # TYPE grim_vram_total_bytes gauge\n\
         grim_vram_total_bytes {vram_total}\n\
         # HELP grim_tokens_generated_total Cumulative decode and response tokens produced\n\
         # TYPE grim_tokens_generated_total counter\n\
         grim_tokens_generated_total {total_tokens_gen}\n\
         # HELP grim_tokens_prefilled_total Cumulative prompt tokens prefilled\n\
         # TYPE grim_tokens_prefilled_total counter\n\
         grim_tokens_prefilled_total {total_tokens_pref}\n\
         # HELP grim_kv_cache_used_bytes Current KV cache memory allocated in bytes\n\
         # TYPE grim_kv_cache_used_bytes gauge\n\
         grim_kv_cache_used_bytes {kv_used_b}\n\
         # HELP grim_kv_cache_total_bytes Total KV cache capacity in bytes\n\
         # TYPE grim_kv_cache_total_bytes gauge\n\
         grim_kv_cache_total_bytes {kv_total_b}\n\
         # HELP grim_kv_cache_blocks_used Count of KV cache blocks currently allocated\n\
         # TYPE grim_kv_cache_blocks_used gauge\n\
         grim_kv_cache_blocks_used {kv_blocks_u}\n\
         # HELP grim_kv_cache_blocks_total Total count of KV cache blocks in block pool\n\
         # TYPE grim_kv_cache_blocks_total gauge\n\
         grim_kv_cache_blocks_total {kv_blocks_t}\n",
        scheduler_snapshot.active_requests,
        scheduler_snapshot.waiting_requests,
        scheduler_snapshot.admitted_requests,
    );

    if let Some(dec_tps) = decode_tps {
        prometheus_text.push_str(&format!(
            "# HELP grim_decode_tokens_per_second Exponential moving average of decode tokens generated per second\n\
             # TYPE grim_decode_tokens_per_second gauge\n\
             grim_decode_tokens_per_second {dec_tps:.2}\n\
             # HELP grim_response_tokens_per_second Exponential moving average of response tokens emitted per second\n\
             # TYPE grim_response_tokens_per_second gauge\n\
             grim_response_tokens_per_second {dec_tps:.2}\n"
        ));
    }
    prometheus_text.push_str(&format!(
        "# HELP grim_radix_lookups_total Radix prefix-cache lookups (one per request with token ids)\n\
         # TYPE grim_radix_lookups_total counter\n\
         grim_radix_lookups_total {radix_lookups}\n\
         # HELP grim_radix_hit_requests_total Requests that reused a cached radix prefix\n\
         # TYPE grim_radix_hit_requests_total counter\n\
         grim_radix_hit_requests_total {radix_hit_requests}\n\
         # HELP grim_radix_hit_tokens_total Prompt tokens served from the radix prefix cache\n\
         # TYPE grim_radix_hit_tokens_total counter\n\
         grim_radix_hit_tokens_total {radix_hit_tokens}\n\
         # HELP grim_radix_hit_rate Fraction of requests that hit the radix prefix cache\n\
         # TYPE grim_radix_hit_rate gauge\n\
         grim_radix_hit_rate {radix_hit_rate:.4}\n"
    ));

    if let Some(pref_tps) = prefill_tps {
        prometheus_text.push_str(&format!(
            "# HELP grim_prefill_tokens_per_second Exponential moving average of prompt tokens prefilled per second\n\
             # TYPE grim_prefill_tokens_per_second gauge\n\
             grim_prefill_tokens_per_second {pref_tps:.2}\n"
        ));
    }

    if let Some(ttft) = last_ttft {
        let ttft_sec = ttft / 1000.0;
        prometheus_text.push_str(&format!(
            "# HELP grim_time_to_first_token_ms Latency to produce first token in milliseconds\n\
             # TYPE grim_time_to_first_token_ms gauge\n\
             grim_time_to_first_token_ms {ttft:.2}\n\
             # HELP grim_time_to_first_token_seconds Latency to produce first token in seconds\n\
             # TYPE grim_time_to_first_token_seconds gauge\n\
             grim_time_to_first_token_seconds {ttft_sec:.4}\n"
        ));
    }
    if let Some(itl) = last_itl {
        let itl_sec = itl / 1000.0;
        prometheus_text.push_str(&format!(
            "# HELP grim_inter_token_latency_ms Inter-token decode latency in milliseconds\n\
             # TYPE grim_inter_token_latency_ms gauge\n\
             grim_inter_token_latency_ms {itl:.2}\n\
             # HELP grim_inter_token_latency_seconds Inter-token decode latency in seconds\n\
             # TYPE grim_inter_token_latency_seconds gauge\n\
             grim_inter_token_latency_seconds {itl_sec:.6}\n"
        ));
    }
    if let Some(ref st) = spec_tele {
        prometheus_text.push_str(&format!(
            "# HELP grim_speculative_accept_rate_ema Exponential moving average of speculative acceptance rate\n\
             # TYPE grim_speculative_accept_rate_ema gauge\n\
             grim_speculative_accept_rate_ema {:.4}\n\
             # HELP grim_speculative_drafted_tokens_total Total draft tokens proposed\n\
             # TYPE grim_speculative_drafted_tokens_total counter\n\
             grim_speculative_drafted_tokens_total {}\n\
             # HELP grim_speculative_accepted_tokens_total Total draft tokens accepted\n\
             # TYPE grim_speculative_accepted_tokens_total counter\n\
             grim_speculative_accepted_tokens_total {}\n",
            st.accept_rate_ema,
            st.total_drafted_tokens,
            st.total_accepted_tokens,
        ));
    }

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        prometheus_text,
    )
        .into_response()
}

/// Helper function to perform Model capability check routing validation (§8)
fn validate_model_capabilities(
    engine: &Engine,
    model_id: &str,
    required_modality: &str,
) -> grim_core::error::Result<()> {
    if let Some(strategy) = engine.strategy_for(model_id) {
        println!(
            "[Routing] Checking model capability requirements for: {} against {} (strategy: {:?})",
            model_id, required_modality, strategy
        );
        Ok(())
    } else {
        Err(grim_core::error::Error::Config(format!(
            "model '{}' has no strategy for modality '{}'",
            model_id, required_modality
        )))
    }
}

/// P0-WI-3: OpenAI clients send the model identifier under `model`, not `name`.
/// Accept both via serde rename so existing `grim pull`-style callers using `name` keep working while.
/// Dynamic model loading endpoint.
async fn load_model(
    state: State<Arc<AppState>>,
    req: Json<routes::models::LoadModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::models::load_model_route(state, req).await
}

/// Dynamic model unloading endpoint.
async fn unload_model(
    state: State<Arc<AppState>>,
    req: Json<routes::models::UnloadModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::models::unload_model_route(state, req).await
}

/// Retrieve a specific model by ID (OpenAI standard GET /v1/models/{model}).
async fn get_model(
    axum::extract::Path(model_id): axum::extract::Path<String>,
    state: State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::catalog::get_model_route(model_id, state).await
}

/// Unload / delete a specific model by ID (OpenAI standard DELETE /v1/models/{model}).
async fn delete_model(
    axum::extract::Path(model_id): axum::extract::Path<String>,
    state: State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::catalog::delete_model_route(model_id, state).await
}

async fn list_adapters(state: State<Arc<AppState>>) -> Json<serde_json::Value> {
    routes::adapters::list_adapters_route(state).await
}

async fn unload_adapter(
    Path(name): Path<String>,
    state: State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::adapters::unload_adapter_route(name, state).await
}

/// Load a trained LoRA sidecar (`grim train` output) and register it for per-request routing WITHOUT an engine restart.
async fn load_adapter_endpoint(
    state: State<Arc<AppState>>,
    payload: Json<routes::adapters::LoadAdapterRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::adapters::load_adapter_route(state, payload).await
}

#[cfg(test)]
pub(crate) use routes::adapters::LoadAdapterRequest;
#[cfg(test)]
pub(crate) use routes::tokens::{DetokenizeRequest, TokenizeRequest};

/// Tokenize raw prompt string using the active model's GgufTokenizer.
async fn tokenize_endpoint(
    state: State<Arc<AppState>>,
    payload: Json<routes::tokens::TokenizeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::tokens::tokenize_route(state, payload).await
}

/// Decode token IDs back to a UTF-8 string.
async fn detokenize_endpoint(
    state: State<Arc<AppState>>,
    payload: Json<routes::tokens::DetokenizeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::tokens::detokenize_route(state, payload).await
}

/// Invalidate and reclaim unreferenced blocks from the KV block pool.
async fn reset_prefix_cache_endpoint(
    state: State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::embeddings::reset_prefix_cache_route(state).await
}

#[cfg(test)]
pub(crate) use routes::embeddings::ScoreRequest;

/// Score query against candidate documents (cross-encoder reranking).
async fn score_rerank_endpoint(
    state: State<Arc<AppState>>,
    payload: Json<routes::embeddings::ScoreRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    routes::embeddings::score_rerank_route(state, payload).await
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub(crate) struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: serde_json::Value,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    _stop: Option<serde_json::Value>,
    #[serde(default)]
    seed: Option<u64>,
}

/// OpenAI-compatible text completions endpoint (POST /v1/completions).
async fn completions(state: State<Arc<AppState>>, payload: Json<CompletionRequest>) -> Response {
    routes::completions::text_completions_route(state, payload).await
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct GrimServerConfigSection {
    default_model: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct GrimServerTomlConfig {
    #[serde(default)]
    server: Option<GrimServerConfigSection>,
    default_model: Option<String>,
}

/// Retrieve default model configured in the config file.
fn get_default_model_from_config() -> Option<String> {
    let custom_path = std::env::var("GRIM_CONFIG_PATH").ok();
    let mut paths: Vec<&str> = Vec::new();
    if let Some(ref p) = custom_path {
        paths.push(p.as_str());
    }
    paths.extend_from_slice(&[
        "grim.toml",
        "/etc/grim/grim.toml",
        "C:\\Program Files\\Grim\\grim.toml",
    ]);
    for path in paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(cfg) = toml::from_str::<GrimServerTomlConfig>(&content) {
                if let Some(s) = cfg.server.and_then(|srv| srv.default_model) {
                    return Some(s);
                }
                if let Some(dm) = cfg.default_model {
                    return Some(dm);
                }
            }
        }
    }
    None
}

/// Status / metrics endpoint displaying processor and active model allocations.
async fn get_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let models = engine.loaded_models();

    // Probe VRAM via platform-specific backend
    let (total_vram_used, total_vram_max, gpu_info) =
        if let Ok(rocm_devs) = grim_backend_rocm::RocmDevice::probe() {
            if !rocm_devs.is_empty() {
                probe_vram_and_gpus(rocm_devs.len())
            } else {
                // Try CUDA first (if compiled in), then Metal, then CPU.
                probe_gpu_or_cpu()
            }
        } else {
            // No ROCm devices: try CUDA (if compiled in), then Metal, then CPU.
            probe_gpu_or_cpu()
        };

    let has_gpu = total_vram_max > 0;
    let backend = active_backend(has_gpu);
    let processor = if backend == "cpu" {
        "CPU"
    } else if has_gpu {
        gpu_info
            .first()
            .and_then(|g| g.get("name").and_then(|n| n.as_str()))
            .unwrap_or("GPU")
    } else {
        "CPU"
    };
    let _gpu_count = gpu_info.len();

    let (sys_ram_used, sys_ram_total) = probe_sys_ram();

    let gpu_util_pct = if has_gpu {
        grim_backend_rocm::compute_utilization(0)
            .map(|u| u as f64)
            .unwrap_or(0.0)
    } else {
        0.0
    };

    // KV cache telemetry and context limit
    let (kv_used_bytes, kv_total_bytes, kv_blocks_used, kv_blocks_total) =
        engine.kv_cache_telemetry();
    let ctx_limit = engine.context_limit();
    let _total_tokens = engine.total_tokens_generated();

    // Get tokens per second from engine
    let tps = engine.tokens_per_sec().unwrap_or(0.0) as f64;
    let scheduler = engine.scheduler_snapshot();
    let ttft_ms = engine.last_ttft_ms();

    let prefill_tps = engine.prefill_tokens_per_sec().map(|v| v as f64);

    let default_model = get_default_model_from_config().unwrap_or_else(|| "default".to_string());

    // Build models with all telemetry integrated
    let mut models_info = Vec::new();
    for m in models {
        models_info.push(serde_json::json!({
            "name": m,
            "params": serde_json::Value::Null,
            "vram_gb": total_vram_used as f64 / (1024.0 * 1024.0 * 1024.0),
            "vram_total_gb": total_vram_max as f64 / (1024.0 * 1024.0 * 1024.0),
            "gpu_util_pct": gpu_util_pct,
            "sys_ram_gb": sys_ram_used as f64 / (1024.0 * 1024.0 * 1024.0),
            "sys_ram_total_gb": sys_ram_total as f64 / (1024.0 * 1024.0 * 1024.0),
            "kv_used_gb": kv_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            "kv_total_gb": kv_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            "ctx_limit": ctx_limit,
            "ttft_ms": ttft_ms,
            "prefill_tps": prefill_tps,
            "decode_tps": tps
        }));
    }

    let spec_disabled = std::env::var("GRIM_SPEC")
        .map(|v| {
            matches!(
                v.trim().to_lowercase().as_str(),
                "off" | "0" | "false" | "disable" | "disabled"
            )
        })
        .unwrap_or(false);
    let accepted_tokens = engine.accepted_tokens_total();
    let acceptance_rate = engine.acceptance_rate();
    let speculation_info = serde_json::json!({
        "enabled": !spec_disabled,
        "strategy": if spec_disabled { "disabled" } else { "auto" },
        "accepted_tokens": accepted_tokens,
        "accepted_rate": acceptance_rate
    });
    Json(serde_json::json!({
        "status": if models_info.is_empty() { "degraded" } else { "healthy" },
        "engine_state": if models_info.is_empty() { "ready_no_model" } else { "healthy" },
        "backend": backend,
        "model_path": state.model_path.as_ref().map(|p| p.display().to_string()),
        "processor": processor,
        "default_model": default_model,
        "system_ram_used_gb": (sys_ram_used as f64 / (1024.0 * 1024.0 * 1024.0)),
        "system_ram_total_gb": (sys_ram_total as f64 / (1024.0 * 1024.0 * 1024.0)),
        "vram_used_gb": (total_vram_used as f64 / (1024.0 * 1024.0 * 1024.0)),
        "vram_total_gb": (total_vram_max as f64 / (1024.0 * 1024.0 * 1024.0)),
        "gpu_util_pct": gpu_util_pct,
        "scheduler": serde_json::json!({
            "active_requests": scheduler.active_requests,
            "waiting_requests": scheduler.waiting_requests,
            "admitted_requests": scheduler.admitted_requests,
            "paused_requests": scheduler.paused_requests
        }),
        "loaded_models": models_info,
        "kv_cache": serde_json::json!({
            "used_bytes": kv_used_bytes,
            "total_bytes": kv_total_bytes,
            "blocks_used": kv_blocks_used,
            "blocks_total": kv_blocks_total,
            "tiers": {
                "gpu_bytes": if has_gpu { kv_used_bytes } else { 0 },
                "host_ram_bytes": if !has_gpu { kv_used_bytes } else { 0 },
                "nvme_bytes": 0
            }
        }),
        "speculation": speculation_info,
        "context_limit": ctx_limit
    }))
}

/// `GET /v1/models` - OpenAI-compatible model catalog endpoint.
/// Scans the configured models directory for files with recognised extensions (`.grim`, `.gguf`, `.safetensors`, `.bin`) and.
async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut entries: Vec<serde_json::Value> = Vec::new();

    // 1. Walk the filesystem catalog using list_local_models.
    for entry in grim_core::catalog::list_local_models() {
        if seen.insert(entry.name.clone()) {
            let path_buf = std::path::PathBuf::from(&entry.path);
            let ext = path_buf
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("unknown");
            entries.push(serde_json::json!({
                "id": entry.name,
                "object": "model",
                "owned_by": "local",
                "created": 0,
                "format": ext,
                "path": entry.path,
                "details": {
                    "family": entry.arch,
                    "parameter_size": entry.params,
                    "quantization_level": entry.quant,
                    "context_length": entry.context_length,
                    "size_bytes": entry.size_bytes,
                    "sha256": entry.sha256
                }
            }));
        }
    }

    // 2. Add any models that are currently loaded in the engine (may not be on disk).
    {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        for name in engine.loaded_models() {
            if seen.insert(name.clone()) {
                entries.push(serde_json::json!({
                    "id": name,
                    "object": "model",
                    "owned_by": "local",
                    "created": 0,
                    "format": "loaded"
                }));
            }
        }
    }

    Json(serde_json::json!({ "object": "list", "data": entries }))
}

/// Helper to extract options and insert them into whitelisted payload fields.
fn translate_options(req: &serde_json::Value, payload: &mut serde_json::Value) {
    if let Some(options) = req.get("options").and_then(|v| v.as_object()) {
        if let Some(temp) = options.get("temperature") {
            payload["temperature"] = temp.clone();
        }
        if let Some(num_predict) = options.get("num_predict") {
            payload["max_tokens"] = num_predict.clone();
        }
        if let Some(top_p) = options.get("top_p") {
            payload["top_p"] = top_p.clone();
        }
        if let Some(stop) = options.get("stop") {
            payload["stop"] = stop.clone();
        }
        if let Some(top_k) = options.get("top_k") {
            payload["top_k"] = top_k.clone();
        }
        if let Some(rp) = options.get("repeat_penalty") {
            payload["repeat_penalty"] = rp.clone();
        }
    }
}

/// WI-S6: detect the local host GPU's ROCm profile name for startup/serve conversion suggestions.
/// Maps the probed `gfx` target to a profile string (`gfx103x`→`rdna2`, `gfx12xx`→`rdna4`, `gfx11xx`→`rdna3`, `gfx90x`→`cdna3`, `gfx9xx`→`cdna2`); returns.
fn detect_host_rocml_profile() -> Option<String> {
    match grim_backend_rocm::probe_host_gpu(0) {
        Ok(caps) => {
            let gcn = &caps.gcn;
            let profile = if gcn.starts_with("gfx103") {
                "rdna2"
            } else if gcn.starts_with("gfx12") {
                "rdna4"
            } else if gcn.starts_with("gfx11") {
                "rdna3"
            } else if gcn.starts_with("gfx90") {
                "cdna3"
            } else if gcn.starts_with("gfx9") {
                "cdna2"
            } else {
                "rdna3"
            };
            Some(profile.to_string())
        }
        Err(_) => None,
    }
}

/// Helper to get current UTC time as RFC-3339 string.
fn utc_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let second = secs % 60;
    let minutes = secs / 60;
    let minute = minutes % 60;
    let hours = minutes / 60;
    let hour = hours % 24;
    let days = hours / 24;

    let mut year = 1970u64;
    let mut remaining = days;
    loop {
        let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let days_in_year = if is_leap { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_days = [
        31u64,
        if is_leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u64;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        month += 1;
    }
    let day = remaining + 1;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Grim compatibility /api/chat endpoint.
async fn grim_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model_name = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("grim")
        .to_string();
    let messages = req
        .get("messages")
        .cloned()
        .unwrap_or(serde_json::json!([]));
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut payload = serde_json::json!({
        "model": model_name,
        "messages": messages,
        "stream": stream,
    });
    if let Some(adapters) = req.get("adapters") {
        payload["adapters"] = adapters.clone();
    }
    translate_options(&req, &mut payload);
    // Ollama /api/chat carries tool definitions under `tools`; forward them into the OpenAI-shaped payload so
    // chat_completions engages the WI-TOOLS 1-5 pipeline (template `tools` variable + output parsing + response `tool_calls`).
    if let Some(tools) = req.get("tools") {
        payload["tools"] = tools.clone();
    }
    if let Some(tc) = req.get("tool_choice") {
        payload["tool_choice"] = tc.clone();
    }

    // F-6: wall-clock timing so the Ollama stats fields carry real
    // measurements instead of hardcoded zeros.
    let chat_start = std::time::Instant::now();
    let response =
        chat_completions(State(state), axum::http::HeaderMap::new(), Json(payload)).await;
    if !response.status().is_success() {
        return response;
    }

    if stream {
        let (_parts, body) = response.into_parts();
        let body_stream = body.into_data_stream();

        let ndjson_stream = futures::stream::unfold(
            (
                body_stream,
                String::new(),
                false,
                0u64,
                None::<std::time::Instant>,
            ),
            move |(mut body_stream, mut buffer, done_sent, mut eval_count, mut first_content)| {
                let model_name = model_name.clone();
                let gen_start = chat_start;
                async move {
                    loop {
                        if done_sent {
                            return None;
                        }
                        if let Some(pos) = buffer.find("\n\n") {
                            let event_str = buffer.drain(..pos + 2).collect::<String>();
                            let mut data_val = None;
                            for line in event_str.lines() {
                                if line.starts_with("data: ") {
                                    let data_json = &line["data: ".len()..];
                                    if let Ok(val) =
                                        serde_json::from_str::<serde_json::Value>(data_json)
                                    {
                                        data_val = Some(val);
                                    }
                                }
                            }
                            if let Some(val) = data_val {
                                let content = val["choices"][0]["delta"]["content"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();
                                // WI-TOOLS-5: forward OpenAI-side `tool_calls` on the terminal delta chunk (the
                                // buffered streaming MVP emits it once, at end of generation).
                                let tool_calls = val["choices"][0]["delta"]["tool_calls"].clone();
                                let mut message = serde_json::json!({
                                    "role": "assistant",
                                    "content": content
                                });
                                if tool_calls.is_array()
                                    && !tool_calls.as_array().unwrap().is_empty()
                                {
                                    message["tool_calls"] = tool_calls;
                                }
                                let ollama_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "message": message,
                                    "done": false
                                });
                                let chunk_str =
                                    format!("{}\n", serde_json::to_string(&ollama_chunk).unwrap());
                                eval_count += 1;
                                if first_content.is_none() {
                                    first_content = Some(std::time::Instant::now());
                                }
                                return Some((
                                    Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                    (body_stream, buffer, false, eval_count, first_content),
                                ));
                            }
                            continue;
                        }

                        match body_stream.next().await {
                            Some(Ok(bytes)) => {
                                if let Ok(s) = std::str::from_utf8(&bytes) {
                                    buffer.push_str(s);
                                }
                            }
                            Some(Err(err)) => {
                                return Some((
                                    Err(err),
                                    (body_stream, buffer, false, eval_count, first_content),
                                ));
                            }
                            None => {
                                // P1-followup: real timing — eval window is
                                // first-content -> done; prompt window is
                                // request start -> first content.
                                let (prompt_eval, eval_dur) = match first_content {
                                    Some(t0) => (
                                        (t0 - gen_start).as_nanos() as u64,
                                        (std::time::Instant::now() - t0).as_nanos() as u64,
                                    ),
                                    None => (0, 0),
                                };
                                let final_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "done": true,
                                    "total_duration": gen_start.elapsed().as_nanos() as u64,
                                    "load_duration": 0,
                                    "prompt_eval_count": 0, // true prompt token count not plumbed into this translate path
                                    "eval_count": eval_count,
                                    "prompt_eval_duration": prompt_eval,
                                    "eval_duration": eval_dur
                                });
                                let chunk_str =
                                    format!("{}\n", serde_json::to_string(&final_chunk).unwrap());
                                return Some((
                                    Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                    (body_stream, buffer, true, eval_count, first_content),
                                ));
                            }
                        }
                    }
                }
            },
        );
        let body = Body::from_stream(ndjson_stream);
        axum::response::Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(body)
            .unwrap()
    } else {
        let (parts, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .unwrap_or_default();
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let content = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            // WI-TOOLS-5: forward `tool_calls` from the OpenAI-shaped response
            // into the Ollama `/api/chat` `message` object.
            let tool_calls = val["choices"][0]["message"]["tool_calls"].clone();
            let mut message = serde_json::json!({
                "role": "assistant",
                "content": content
            });
            if tool_calls.is_array() && !tool_calls.as_array().unwrap().is_empty() {
                message["tool_calls"] = tool_calls;
            }
            let ollama_res = serde_json::json!({
                "model": model_name,
                "created_at": utc_now_rfc3339(),
                "message": message,
                "done": true,
                // F-6: real wall-clock measurement; eval_count approximated from the response content (≈4
                // chars/token) so Ollama clients that throttle on these fields get usable data.
                "total_duration": chat_start.elapsed().as_nanos() as u64,
                "load_duration": 0,
                "prompt_eval_count": messages.as_array().map(|m| {
                    m.iter().map(|x| x["content"].as_str().map(|c| c.len() / 4).unwrap_or(0)).sum::<usize>()
                }).unwrap_or(0),
                "eval_count": content.len().div_ceil(4),
                "eval_duration": chat_start.elapsed().as_nanos() as u64
            });
            let mut res = Response::from_parts(
                parts,
                Body::from(serde_json::to_string(&ollama_res).unwrap()),
            );
            res.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            );
            res
        } else {
            Response::from_parts(parts, Body::from(bytes))
        }
    }
}

/// Grim compatibility /api/generate endpoint.
async fn grim_generate(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model_name = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("grim")
        .to_string();
    let prompt = req
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut payload = serde_json::json!({
        "model": model_name,
        "messages": [{ "role": "user", "content": &prompt }],
        "stream": stream,
    });
    // WI-HYBRID Layer 2: forward the session tag (body field or header).
    if let Some(sess) = req.get("session").and_then(|v| v.as_str()) {
        payload["session"] = serde_json::json!(sess);
    } else if let Some(sess) = headers.get("x-grim-session").and_then(|v| v.to_str().ok()) {
        payload["session"] = serde_json::json!(sess);
    }
    translate_options(&req, &mut payload);

    let gen_start = std::time::Instant::now();
    let response = chat_completions(State(state), headers, Json(payload)).await;
    if !response.status().is_success() {
        return response;
    }
    let prompt_tokens = prompt.len().div_ceil(4) as u64;
    if stream {
        let (_parts, body) = response.into_parts();
        let body_stream = body.into_data_stream();

        let ndjson_stream = futures::stream::unfold(
            (body_stream, String::new(), false, 0u64),
            move |(mut body_stream, mut buffer, done_sent, mut eval_count)| {
                let model_name = model_name.clone();
                async move {
                    loop {
                        if done_sent {
                            return None;
                        }
                        if let Some(pos) = buffer.find("\n\n") {
                            let event_str = buffer.drain(..pos + 2).collect::<String>();
                            let mut data_val = None;
                            for line in event_str.lines() {
                                if line.starts_with("data: ") {
                                    let data_json = &line["data: ".len()..];
                                    if let Ok(val) =
                                        serde_json::from_str::<serde_json::Value>(data_json)
                                    {
                                        data_val = Some(val);
                                    }
                                }
                            }
                            if let Some(val) = data_val {
                                let content = val["choices"][0]["delta"]["content"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();
                                // Prefer the upstream's true sampled-token count
                                // (grim_eval_count): a single SSE chunk can carry
                                // multiple tokens' worth of text (e.g. the
                                // stop-triggered terminal chunk), so counting
                                // chunks undercounts tokens.
                                if let Some(n) = val["grim_eval_count"].as_u64() {
                                    eval_count = eval_count.max(n);
                                } else {
                                    eval_count += 1;
                                }
                                let ollama_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "response": content,
                                    "done": false
                                });
                                let chunk_str =
                                    format!("{}\n", serde_json::to_string(&ollama_chunk).unwrap());
                                eprintln!(
                                    "[trace] generate content chunk, eval_count={eval_count}"
                                );
                                return Some((
                                    Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                    (body_stream, buffer, false, eval_count),
                                ));
                            }
                            continue;
                        }

                        match body_stream.next().await {
                            Some(Ok(bytes)) => {
                                if let Ok(s) = std::str::from_utf8(&bytes) {
                                    buffer.push_str(s);
                                }
                            }
                            Some(Err(err)) => {
                                return Some((Err(err), (body_stream, buffer, false, eval_count)));
                            }
                            None => {
                                if !buffer.is_empty() {
                                    let remaining_text = buffer.clone();
                                    buffer.clear();
                                    let partial_chunk = serde_json::json!({
                                        "model": model_name,
                                        "created_at": utc_now_rfc3339(),
                                        "response": remaining_text,
                                        "done": false
                                    });
                                    let chunk_str = format!(
                                        "{}\n",
                                        serde_json::to_string(&partial_chunk).unwrap()
                                    );
                                    eval_count += 1;
                                    return Some((
                                        Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                        (body_stream, buffer, false, eval_count),
                                    ));
                                }
                                let elapsed = gen_start.elapsed().as_nanos() as u64;
                                let final_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "done": true,
                                    "total_duration": elapsed,
                                    "load_duration": 0,
                                    "prompt_eval_count": prompt_tokens,
                                    "eval_count": eval_count,
                                    "eval_duration": elapsed
                                });
                                let chunk_str =
                                    format!("{}\n", serde_json::to_string(&final_chunk).unwrap());
                                return Some((
                                    Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                    (body_stream, buffer, true, eval_count),
                                ));
                            }
                        }
                    }
                }
            },
        );
        let body = Body::from_stream(ndjson_stream);
        axum::response::Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(body)
            .unwrap()
    } else {
        let (parts, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .unwrap_or_default();
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let content = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let prompt_tokens = val["usage"]["prompt_tokens"]
                .as_u64()
                .unwrap_or(prompt.len().div_ceil(4) as u64);
            let completion_tokens = val["usage"]["completion_tokens"]
                .as_u64()
                .unwrap_or(content.len().div_ceil(4) as u64);
            let elapsed_nanos = gen_start.elapsed().as_nanos() as u64;
            let ollama_res = serde_json::json!({
                "model": model_name,
                "created_at": utc_now_rfc3339(),
                "response": content,
                "done": true,
                "total_duration": elapsed_nanos,
                "load_duration": 0,
                "prompt_eval_count": prompt_tokens,
                "eval_count": completion_tokens,
                "eval_duration": elapsed_nanos
            });
            let mut res = Response::from_parts(
                parts,
                Body::from(serde_json::to_string(&ollama_res).unwrap()),
            );
            res.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            );
            res
        } else {
            Response::from_parts(parts, Body::from(bytes))
        }
    }
}

/// Grim compatibility /api/tags (model list) endpoint.
/// Grim compatibility /api/tags (model list) endpoint.
async fn grim_tags(state: State<Arc<AppState>>) -> Json<serde_json::Value> {
    routes::catalog::grim_tags_route(state).await
}

/// Grim compatibility /api/pull endpoint.
async fn grim_pull(state: State<Arc<AppState>>, req: Json<serde_json::Value>) -> impl IntoResponse {
    routes::models::grim_pull_route(state, req).await
}

/// POST /api/upload?filename=<name>.gguf - save an uploaded model file into the local catalog.
async fn grim_upload(
    query: axum::extract::Query<std::collections::HashMap<String, String>>,
    body: axum::body::Bytes,
) -> Response {
    routes::models::grim_upload_route(query, body).await
}

/// Current UTC time as RFC-3339, without pulling a full chrono dependency.
fn chrono_utc_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // days-from-civil inverse (Howard Hinnant's algorithm).
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Build a new HTTP router with the given engine state.
/// Paths reachable without a bearer key even when auth is enabled.
const AUTH_EXEMPT_PATHS: &[&str] = &["/health", "/healthz", "/readyz", "/metrics"];

/// Open server (loopback posture unchanged). See [`build_router_with_auth`].
pub fn build_router(state: Arc<AppState>) -> Router {
    build_router_with_auth(state, Vec::new())
}

/// Same route table, but requires `Authorization: Bearer <key>` on every
/// path except [`AUTH_EXEMPT_PATHS`] when `api_keys` is non-empty.
pub fn build_router_with_auth(state: Arc<AppState>, api_keys: Vec<String>) -> Router {
    let router = Router::new()
        .route("/health", get(health))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/status", get(get_status))
        .route("/v1/status", get(get_status))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model", get(get_model).delete(delete_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/unload", post(unload_model))
        .route(
            "/v1/adapters",
            get(list_adapters).post(load_adapter_endpoint),
        )
        .route("/v1/adapters/load", post(load_adapter_endpoint))
        .route("/v1/adapters/:name", delete(unload_adapter))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/audio/translations", post(audio_translations))
        .route("/v1/images/generations", post(images_generations))
        .route("/tokenize", post(tokenize_endpoint))
        .route("/v1/tokenize", post(tokenize_endpoint))
        .route("/detokenize", post(detokenize_endpoint))
        .route("/v1/detokenize", post(detokenize_endpoint))
        .route("/reset_prefix_cache", post(reset_prefix_cache_endpoint))
        .route("/v1/cache/clear", post(reset_prefix_cache_endpoint))
        .route("/v1/score", post(score_rerank_endpoint))
        .route("/v1/rerank", post(score_rerank_endpoint))
        .route("/v1/requests/:id/pause", post(pause_request))
        .route("/v1/requests/:id/resume", post(resume_request))
        .route("/v1/requests/:id/cancel", post(cancel_request))
        .route("/v1/requests/:id/stream", get(stream_state))
        .route("/grpc", get(grpc_service_handler))
        // Grim REST API compatibility shims:
        .route("/api/chat", post(grim_chat))
        .route("/api/generate", post(grim_generate))
        .route("/api/tags", get(grim_tags))
        .route("/api/pull", post(grim_pull))
        .route("/api/upload", post(grim_upload))
        // Dashboard:
        .route("/", get(dashboard_html))
        .route("/logo.png", get(serve_logo_png))
        .route("/api/stats", get(stats_endpoint))
        // F-6: alias so `GET /adapters` returns the same JSON list as
        // `/v1/adapters` instead of an empty 200 from a missing route.
        .route("/adapters", get(list_adapters))
        // F-2: scheduler queue/admission state must be reachable at the same origin as the server
        // itself - the CLI `grim scheduler` subcommand and dashboards poll this instead of guessing ports.
        .route("/scheduler", get(get_status))
        .route("/api/scheduler", get(get_status));
    let router = if api_keys.is_empty() {
        router
    } else {
        let keys: std::sync::Arc<[String]> = api_keys.into();
        router.layer(axum::middleware::from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| {
                let keys = keys.clone();
                async move {
                    let path = req.uri().path();
                    if !AUTH_EXEMPT_PATHS.contains(&path) {
                        let authorized = req
                            .headers()
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok())
                            .and_then(|h| h.strip_prefix("Bearer "))
                            .map(|k| keys.iter().any(|allowed| allowed == k))
                            .unwrap_or(false);
                        if !authorized {
                            return (
                                StatusCode::UNAUTHORIZED,
                                [(
                                    axum::http::header::WWW_AUTHENTICATE,
                                    "Bearer realm=\"grim\"",
                                )],
                                Json(serde_json::json!({
                                    "error": {
                                        "message": "Missing or invalid API key.",
                                        "type": "invalid_request_error",
                                        "code": "invalid_api_key"
                                    }
                                })),
                            )
                                .into_response();
                        }
                    }
                    next.run(req).await
                }
            },
        ))
    };
    router
        .layer(axum::extract::DefaultBodyLimit::max(10 * 1024 * 1024))
        .with_state(state)
}

struct TlsConfig {
    cert_path: String,
    key_path: String,
}

fn load_tls_config_from_file(path: &str) -> Option<TlsConfig> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut cert = None;
    let mut key = None;
    let mut in_tls_section = false;

    for line in content.lines() {
        let line = line.trim();
        if line == "[server.tls]" {
            in_tls_section = true;
            continue;
        } else if line.starts_with('[') {
            in_tls_section = false;
        }

        if in_tls_section {
            if let Some(pos) = line.find('=') {
                let k = line[..pos].trim();
                let mut v = line[pos + 1..].trim();
                if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
                    v = &v[1..v.len() - 1];
                }
                if k == "cert_path" {
                    cert = Some(v.to_string());
                } else if k == "key_path" {
                    key = Some(v.to_string());
                }
            }
        }
    }

    if let (Some(c), Some(k)) = (cert, key) {
        Some(TlsConfig {
            cert_path: c,
            key_path: k,
        })
    } else {
        None
    }
}

/// Start the server on `addr`, optionally pre-loading a model by file path.
/// `model_path`: when `Some`, the tokenizer and model are loaded from this GGUF file before the.
fn load_api_keys_from_env() -> Vec<String> {
    let mut keys: Vec<String> = std::env::var("GRIM_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .into_iter()
        .collect();
    if let Ok(list) = std::env::var("GRIM_API_KEYS") {
        keys.extend(
            list.split(',')
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty()),
        );
    }
    keys
}

pub async fn serve(
    addr: &str,
    engine: Engine,
    model_path: Option<std::path::PathBuf>,
    plugin_registry: Option<std::sync::Arc<grim_plugin::PluginRegistry>>,
) -> Result<()> {
    validate_metrics_bind_policy(addr)?;
    let api_keys = load_api_keys_from_env();
    let auth_enabled = !api_keys.is_empty();
    if auth_enabled {
        eprintln!(
            "[grim-server] Bearer auth enabled ({} API key(s)).",
            api_keys.len()
        );
    }
    // Attempt to load the tokenizer from the explicitly-given model path, or by scanning the models directory for the first available GGUF.
    // For `.grim` files, fall back to a sibling `.gguf` (same stem, `.gguf` extension) - this.
    let (tokenizer, resolved_path) = if let Some(ref p) = model_path {
        let path_str = p.display().to_string();
        // Try the path directly (works for .gguf files).
        let mut tok = GgufProvider::open(&path_str)
            .ok()
            .and_then(|prov| prov.tokenizer().ok());
        // If that failed and the path is a .grim file, try the sibling .gguf.
        if tok.is_none() && p.extension().and_then(|x| x.to_str()) == Some("grim") {
            let gguf_path = p.with_extension("gguf");
            if let Some(path_str) = gguf_path.to_str() {
                if gguf_path.exists() {
                    tok = GgufProvider::open(path_str)
                        .ok()
                        .and_then(|prov| prov.tokenizer().ok());
                }
            }
        }
        (tok, Some(p.clone()))
    } else {
        // Scan the models directory for the first available model, preferring an existing ROCm-tuned `.grim` conversion over a
        // sibling `.gguf` (WI-S6: once a conversion exists it is used automatically, the same preference `grim run` applies).
        let models_dir = grim_models_dir();
        let tok_and_path = std::fs::read_dir(&models_dir)
            .ok()
            .and_then(|mut it| {
                it.find(|e| {
                    e.as_ref()
                        .ok()
                        .map(|e| {
                            let p = e.path();
                            matches!(
                                p.extension().and_then(|x| x.to_str()),
                                Some("gguf") | Some("grim")
                            )
                        })
                        .unwrap_or(false)
                })
            })
            .and_then(|e| e.ok())
            .map(|e| e.path())
            .and_then(|p| {
                // If the first file is a `.gguf` with a `.grim` sibling,
                // prefer the tuned artifact.
                let preferred = if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
                    let grim = p.with_extension("grim");
                    if grim.exists() { grim } else { p }
                } else {
                    p
                };
                let p_str = preferred.display().to_string();
                let tok = GgufProvider::open(&p_str)
                    .ok()
                    .and_then(|prov| prov.tokenizer().ok());
                // If the preferred file is a .grim, the tokenizer lives in
                // the sibling .gguf — try that before giving up.
                let tok = if tok.is_none()
                    && preferred.extension().and_then(|x| x.to_str()) == Some("grim")
                {
                    let gguf_path = preferred.with_extension("gguf");
                    if let Some(path_str) = gguf_path.to_str() {
                        if gguf_path.exists() {
                            GgufProvider::open(path_str)
                                .ok()
                                .and_then(|prov| prov.tokenizer().ok())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    tok
                };
                tok.map(|t| (t, preferred))
            });
        if let Some((tok, p)) = tok_and_path {
            // WI-S6: if we auto-loaded a `.gguf` that has no tuned `.grim` sibling,
            // offer (never silently run) the ROCm conversion on the detected local GPU profile.
            if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
                if let Some(profile) = detect_host_rocml_profile() {
                    let name = p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model")
                        .to_string();
                    eprintln!(
                        "[grim-server] Tip: convert '{}' to a ROCm-tuned .grim for better \
                         performance on this GPU (detected profile: {}):",
                        name, profile
                    );
                    eprintln!(
                        "[grim-server]      grim oxidize convert {} --rocml-profile {}",
                        name, profile
                    );
                }
            }
            (Some(tok), Some(p))
        } else {
            (None, None)
        }
    };

    if tokenizer.is_none() {
        eprintln!(
            "[grim-server] WARNING: No tokenizer found. Run 'grim pull <model>' to download a model."
        );
        eprintln!(
            "[grim-server]          Text responses will show raw token IDs until a model is loaded."
        );
    }

    // WI-E8: detect the model architecture from the GGUF metadata so the
    // tool-call parser can use the per-arch detector registry.
    let detected_arch = state_arch_hint(resolved_path.as_deref());
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(tokenizer),
        model_path: resolved_path,
        model_arch: std::sync::Mutex::new(detected_arch),
        plugin_registry,
    });

    // Capability-based routing verification at server startup (§8)
    if let Err(e) = validate_model_capabilities(
        &state.engine.lock().unwrap_or_else(|e| e.into_inner()),
        "default",
        "text",
    ) {
        eprintln!("[Server] Model capability check failed: {e}");
    }

    let app = build_router_with_auth(state, api_keys);

    // Incoming SSRF posture (§network): the server defaults to loopback (`127.0.0.1:11434`) so it is never reachable from a routable network unless the operator explicitly opts in via `GRIM_HOST`/`--address`.
    // A user-supplied public bind is honored by design (mirrors Ollama's posture); the guard above is.
    let custom_cfg_path = std::env::var("GRIM_CONFIG_PATH").ok();
    let tls_config = custom_cfg_path
        .as_deref()
        .and_then(load_tls_config_from_file)
        .or_else(|| load_tls_config_from_file("grim.toml"))
        .or_else(|| load_tls_config_from_file("/etc/grim/grim.toml"))
        .or_else(|| load_tls_config_from_file("C:\\Program Files\\Grim\\grim.toml"));

    if let Some(cfg) = tls_config {
        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_pem_file(&cfg.cert_path, &cfg.key_path)
                .await
                .map_err(|e| {
                    grim_core::Error::Config(format!("failed to load TLS certificates: {e}"))
                })?;

        // Resolve the bind address the same way the non-TLS path does (TcpListener::bind accepts hostnames; addr.parse() only accepts numeric IPs).
        // This ensures `--address localhost:11434` works identically over HTTP and HTTPS.
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| grim_core::Error::Config(format!("bind failed: {e}")))?;
        let bind_addr = listener
            .local_addr()
            .map_err(|e| grim_core::Error::Config(format!("failed to get local addr: {e}")))?;
        eprintln!(
            "[grim-server] Serving over HTTPS (SSL enabled) on {}",
            bind_addr
        );
        axum_server::bind_rustls(bind_addr, rustls_config)
            .serve(app.into_make_service())
            .await
            .map_err(|e| grim_core::Error::Config(format!("serve TLS failed: {e}")))?;
    } else {
        // SRV-5: Warn when binding to non-loopback without TLS.
        let host_part = addr.split(':').next().unwrap_or(addr);
        let is_wildcard = host_part == "0.0.0.0" || host_part == "::";
        let is_non_loopback = !host_part.starts_with("127.") && !is_wildcard;
        if is_wildcard || is_non_loopback {
            eprintln!(
                "[grim-server] WARNING: Binding to {addr} exposes the server on a \
                 non-loopback interface without TLS. This is a security risk on \
                 untrusted networks."
            );
            if !auth_enabled {
                eprintln!(
                    "[grim-server] WARNING: No API keys configured (set GRIM_API_KEY \
                     or GRIM_API_KEYS). Any reachable client can run models and read files."
                );
            }
        }
        eprintln!(
            "[grim-server] WARNING: No TLS config found; serving over HTTP on {}",
            addr
        );
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| grim_core::Error::Config(format!("bind failed: {e}")))?;
        axum::serve(listener, app)
            .await
            .map_err(|e| grim_core::Error::Config(format!("serve HTTP failed: {e}")))?;
    }
    Ok(())
}

/// Catalog/path resolution shared by `load_model_for_server` and the SCYTHE-2 farm loader (which needs the on-disk path to pull additional replicas).
/// P0-WI-3: prefers the `.grim` sibling whenever both exist for the same model name (set after.
fn resolve_catalog_model_path(name: &str) -> Option<std::path::PathBuf> {
    if std::path::Path::new(name).exists() {
        return grim_core::catalog::resolve_model_preferring_grim(name);
    }
    // Ensure the models dir is initialized; some callers may have skipped it.
    let _ = grim_core::grim_models_dir();
    grim_core::catalog::resolve_model_preferring_grim(name)
}

/// Resolve a model name from the local catalog and load it as a `CausalLm`.
/// Returns `(model_box, Option<tokenizer>)` on success.
fn load_model_for_server(
    name: &str,
) -> grim_core::error::Result<(
    Box<dyn grim_core::model::CausalLm>,
    Option<grim_format::GgufTokenizer>,
)> {
    use grim_engine::model_loader;

    let path = resolve_catalog_model_path(name).ok_or_else(|| {
        grim_core::error::Error::Config(format!(
            "model '{name}' not found in catalog. Run 'grim pull {name}' to download it."
        ))
    })?;

    let path_str = path.display().to_string();
    let model = model_loader::load_from_path(&path_str)
        .map_err(|e| grim_core::error::Error::Config(format!("model load failed: {e}")))?;

    let tokenizer = GgufProvider::open(&path_str)
        .ok()
        .and_then(|p| p.tokenizer().ok())
        // If only a `.grim` exists, fall back to a sibling `.gguf`'s tokenizer,
        // since tokenizer bytes are currently GGUF-only.
        .or_else(|| {
            path.with_extension("gguf")
                .to_str()
                .and_then(|gg| GgufProvider::open(gg).ok().and_then(|p| p.tokenizer().ok()))
        });

    // WI-3 self-heal: backfill the catalog sidecar from the GGUF header if it still carries empty arch/zero context_length (older pull or a manually- placed file whose sidecar predates this fix).
    // Header-only read; failure is non-fatal since we already have the model loaded for serving.
    grim_core::catalog::self_heal_sidecar(path.as_path());

    Ok((model, tokenizer))
}

// Dashboard endpoint - live stats for the server status page.

/// `GET /api/stats` — JSON stats snapshot polled by the dashboard at `/`.
#[doc(hidden)]
pub fn probe_sys_ram() -> (u64, u64) {
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        let mut total_kb: u64 = 0;
        let mut avail_kb: u64 = 0;
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    total_kb = val.parse().unwrap_or(0);
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    avail_kb = val.parse().unwrap_or(0);
                }
            }
        }
        if total_kb > 0 {
            let total_bytes = total_kb * 1024;
            let used_bytes = (total_kb.saturating_sub(avail_kb)) * 1024;
            return (used_bytes, total_bytes);
        }
    }
    (0, 0)
}

#[doc(hidden)]
pub fn probe_vram_and_gpus(rocm_gpu_count: usize) -> (u64, u64, Vec<serde_json::Value>) {
    let mut total_vram_used: u64 = 0;
    let mut total_vram_max: u64 = 0;
    let mut gpus_json = Vec::new();
    let mut global_idx: u32 = 0;

    // 1. Probe AMD ROCm GPUs
    if rocm_gpu_count > 0 {
        // Collect sysfs real vram total sizes to catch APU shared RAM reporting
        let mut sysfs_vram_totals = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            let mut card_paths: Vec<_> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .map(|n| {
                            n.to_string_lossy().starts_with("card")
                                && !n.to_string_lossy().contains('-')
                        })
                        .unwrap_or(false)
                })
                .collect();
            card_paths.sort();

            for path in card_paths {
                let vendor_path = path.join("device/vendor");
                if let Ok(v_str) = std::fs::read_to_string(&vendor_path) {
                    if v_str.trim().eq_ignore_ascii_case("0x1002") {
                        let total_path = path.join("device/mem_info_vram_total");
                        if let Ok(t_str) = std::fs::read_to_string(&total_path) {
                            if let Ok(bytes) = t_str.trim().parse::<u64>() {
                                if bytes > 0 {
                                    sysfs_vram_totals.push(bytes);
                                }
                            }
                        }
                    }
                }
            }
        }

        for ord in 0..rocm_gpu_count {
            let (free, hip_total) = grim_backend_rocm::vram_info(ord);
            let mut total = hip_total;
            if let Some(&real_sysfs) = sysfs_vram_totals.get(ord) {
                if real_sysfs < total {
                    total = real_sysfs;
                }
            }
            let used = total.saturating_sub(free).min(total);
            total_vram_used += used;
            total_vram_max += total;
            let memory_pct = if total > 0 {
                ((used as f64 / total as f64) * 100.0) as u32
            } else {
                0
            };
            let compute = grim_backend_rocm::compute_utilization(ord);
            gpus_json.push(serde_json::json!({
                "index": global_idx,
                "backend": "ROCm",
                "compute": compute,
                "memory": memory_pct,
                "vram_used": used,
                "vram_total": total,
                "name": format!("AMD ROCm GPU #{ord}"),
            }));
            global_idx += 1;
        }
    }

    // 2. Probe NVIDIA CUDA GPUs
    #[cfg(feature = "cuda")]
    if let Ok(cuda_devs) = grim_backend_cuda::CudaDevice::probe() {
        if !cuda_devs.is_empty() {
            for ord in 0..cuda_devs.len() {
                let Some((free, total)) = grim_backend_cuda::vram_info(ord) else {
                    continue;
                };
                let used = total.saturating_sub(free);
                total_vram_used += used;
                total_vram_max += total;
                let memory_pct = if total > 0 {
                    ((used as f64 / total as f64) * 100.0) as u32
                } else {
                    0
                };
                let compute = grim_backend_cuda::compute_utilization(ord);
                gpus_json.push(serde_json::json!({
                    "index": global_idx,
                    "backend": "CUDA",
                    "compute": compute,
                    "memory": memory_pct,
                    "vram_used": used,
                    "vram_total": total,
                    "name": format!("NVIDIA CUDA GPU #{ord}"),
                }));
                global_idx += 1;
            }
        }
    }

    // 3. Probe Apple Metal GPUs if no GPUs found yet
    if gpus_json.is_empty() {
        if let Some((free, total)) = grim_backend_metal::vram_info(0) {
            if total > 0 {
                let used = total.saturating_sub(free);
                let memory_pct = ((used as f64 / total as f64) * 100.0) as u32;
                let compute = grim_backend_metal::compute_utilization(0);
                gpus_json.push(serde_json::json!({
                    "index": global_idx,
                    "backend": "Metal",
                    "compute": compute,
                    "memory": memory_pct,
                    "vram_used": used,
                    "vram_total": total,
                    "name": "Apple Metal Unified GPU",
                }));
                total_vram_used += used;
                total_vram_max += total;
                global_idx += 1;
            }
        }
    }

    // 4. Probe Vulkan GPUs if no GPUs found yet
    if gpus_json.is_empty() {
        if let Some((free, total)) = grim_backend_vulkan::vram_info(0) {
            if total > 0 {
                let used = total.saturating_sub(free);
                let memory_pct = ((used as f64 / total as f64) * 100.0) as u32;
                let compute = grim_backend_vulkan::compute_utilization(0);
                gpus_json.push(serde_json::json!({
                    "index": global_idx,
                    "backend": "Vulkan",
                    "compute": compute,
                    "memory": memory_pct,
                    "vram_used": used,
                    "vram_total": total,
                    "name": "Vulkan Discrete GPU",
                }));
                total_vram_used += used;
                total_vram_max += total;
            }
        }
    }

    // 5. Fallback if still empty
    if gpus_json.is_empty() {
        gpus_json.push(serde_json::json!({
            "index": 0u32,
            "backend": "CPU",
            "compute": serde_json::Value::Null,
            "memory": 0u32,
            "vram_used": 0u64,
            "vram_total": 0u64,
            "name": "Host CPU",
        }));
    }

    (total_vram_used, total_vram_max, gpus_json)
}

/// Probe GPU VRAM, trying CUDA first (if compiled in), then Metal, then CPU.
/// Returns `(used_bytes, max_bytes, gpu_info_json)`.
#[cfg(feature = "cuda")]
fn probe_gpu_or_cpu() -> (u64, u64, Vec<serde_json::Value>) {
    if let Ok(cuda_devs) = grim_backend_cuda::CudaDevice::probe() {
        if !cuda_devs.is_empty() {
            return probe_cuda_vram(cuda_devs.len());
        }
    }
    probe_metal_or_cpu()
}

/// Non-CUDA fallback: Metal or CPU.
#[cfg(not(feature = "cuda"))]
fn probe_gpu_or_cpu() -> (u64, u64, Vec<serde_json::Value>) {
    probe_metal_or_cpu()
}

/// Metal or CPU fallback used by both CUDA and non-CUDA paths.
fn probe_metal_or_cpu() -> (u64, u64, Vec<serde_json::Value>) {
    if let Some((free, total)) = grim_backend_metal::vram_info(0) {
        (
            total - free,
            total,
            vec![serde_json::json!({
                "name": "Metal GPU",
                "index": 0u32,
                "memory": if total > 0 { ((total - free) as f64 / total as f64 * 100.0) as u32 } else { 0 }
            })],
        )
    } else {
        (0, 0, vec![])
    }
}

/// Probe CUDA VRAM usage for N GPUs.
#[cfg(feature = "cuda")]
fn probe_cuda_vram(cuda_gpu_count: usize) -> (u64, u64, Vec<serde_json::Value>) {
    let mut total_vram_used: u64 = 0;
    let mut total_vram_max: u64 = 0;
    let mut gpus_json = Vec::new();

    if let Ok(cuda_devs) = grim_backend_cuda::CudaDevice::probe() {
        for ord in 0..cuda_gpu_count.min(cuda_devs.len()) {
            let Some((free, total)) = grim_backend_cuda::vram_info(ord) else {
                continue;
            };
            let used = total.saturating_sub(free);
            total_vram_used += used;
            total_vram_max += total;
            let memory_pct = if total > 0 {
                (used as f64 / total as f64 * 100.0) as u32
            } else {
                0
            };
            let compute = grim_backend_cuda::compute_utilization(ord);
            gpus_json.push(serde_json::json!({
                "index": ord as u32,
                "compute": compute,
                "memory": memory_pct,
                "name": format!("CUDA GPU {ord}"),
            }));
        }
    }

    (total_vram_used, total_vram_max, gpus_json)
}

/// `GET /api/stats` - JSON stats snapshot polled by the dashboard at `/`.
/// WI-1 wire-shape note: `gpus[].compute` is now `Option<u32>` - a real per-backend utilization probe, or `null`.
async fn stats_endpoint(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let models = engine.loaded_models();
    let model_name = models
        .first()
        .cloned()
        .unwrap_or_else(|| "none".to_string());

    // Hardware probe (matches /metrics): real GPU count + xnack.
    let (rocm_gpu_count, xnack_enabled) = match grim_backend_rocm::RocmDevice::probe() {
        Ok(devices) if !devices.is_empty() => (devices.len(), devices[0].xnack_enabled()),
        _ => (0, false),
    };

    // Catalog snapshot: list every local model, grouped by format so the
    // dashboard can render the same "GRIM > GGUF > other" priority as the CLI.
    let mut grim_models = Vec::new();
    let mut gguf_models = Vec::new();
    let mut other_models = Vec::new();
    for entry in grim_core::catalog::list_local_models() {
        let path = std::path::PathBuf::from(&entry.path);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_string();
        let item = serde_json::json!({
            "name": entry.name,
            "format": ext,
            "size": entry.size_bytes,
            "arch": entry.arch,
            "params": entry.params,
            "quant": entry.quant,
        });
        match ext.as_str() {
            "grim" => grim_models.push(item),
            "gguf" => gguf_models.push(item),
            _ => other_models.push(item),
        }
    }

    // Once we wire real telemetry counters into the engine (tokens generated, wall-clock time per batch, KV block occupancy), this becomes live data.
    // For now the fields are present and typed so the frontend contract is fixed.
    let (kv_used, kv_total, kv_blocks_used, kv_blocks_total) = engine.kv_cache_telemetry();
    // F-2: expose the scheduler's live three-queue state on the dashboard
    // surface so a user can see WHY a request waited without a second tool.
    let sched = engine.scheduler_snapshot();
    let (vram_used, vram_total, gpus_json) = probe_vram_and_gpus(rocm_gpu_count);
    let (sys_ram_used, sys_ram_total) = probe_sys_ram();
    let tps_json = match engine.tokens_per_sec() {
        Some(tps) => serde_json::json!(tps),
        None => serde_json::Value::Null,
    };
    let prefill_tps_json = match engine.prefill_tokens_per_sec() {
        Some(tps) => serde_json::json!(tps),
        None => serde_json::Value::Null,
    };
    let total_tokens_gen = engine.total_tokens_generated();
    let total_tokens_pref = engine.total_tokens_prefilled();
    let (qkv_attempts, qkv_fallbacks, qkv_sticky) =
        grim_models_transformer::shared_attention::qkv_arena_fallback_stats();
    let ttft_json = match engine.last_ttft_ms() {
        Some(ttft) => serde_json::json!(ttft),
        None => serde_json::Value::Null,
    };
    let itl_json = match engine.last_itl_ms() {
        Some(itl) => serde_json::json!(itl),
        None => serde_json::Value::Null,
    };
    let spec_json = match engine.speculative_telemetry(None) {
        Some(tele) => serde_json::json!({
            "strategy": tele.strategy,
            "accept_rate_ema": tele.accept_rate_ema,
            "accepted_rate": engine.acceptance_rate(),
            "steps_observed": tele.steps_observed,
            "total_drafted_tokens": tele.total_drafted_tokens,
            "total_accepted_tokens": tele.total_accepted_tokens,
            "accepted_tokens": tele.total_accepted_tokens,
            "min_accept_rate": tele.min_accept_rate,
            "should_adapt": tele.should_adapt,
            "enabled": true,
        }),
        None => {
            let spec_disabled = std::env::var("GRIM_SPEC")
                .map(|v| {
                    matches!(
                        v.trim().to_lowercase().as_str(),
                        "off" | "0" | "false" | "disable" | "disabled"
                    )
                })
                .unwrap_or(false);
            if !spec_disabled && engine.total_tokens_generated() > 0 {
                serde_json::json!({
                    "enabled": true,
                    "strategy": "auto",
                    "accepted_tokens": engine.accepted_tokens_total(),
                    "accepted_rate": engine.acceptance_rate(),
                })
            } else {
                serde_json::Value::Null
            }
        }
    };

    // WI-2: radix prefix-cache hit-rate and reuse-volume observability.
    let (radix_lookups, radix_hit_requests, radix_hit_tokens) = engine.radix_cache_telemetry();
    let (_, _, total_tokens_pref2) = (0u64, 0u64, engine.total_tokens_prefilled());
    let prefix_reuse_ratio = if total_tokens_pref2 > 0 {
        radix_hit_tokens as f64 / total_tokens_pref2 as f64
    } else {
        0.0
    };
    let hit_rate = if radix_lookups > 0 {
        radix_hit_requests as f64 / radix_lookups as f64
    } else {
        0.0
    };

    serde_json::json!({
        "model_name": model_name,
        "tokens_per_sec": tps_json,
        "decode_tokens_per_sec": tps_json,
        "prefill_tokens_per_sec": prefill_tps_json,
        "total_tokens_generated": total_tokens_gen,
        "total_tokens_prefilled": total_tokens_pref,
        "ttft_ms": ttft_json,
        "itl_ms": itl_json,
        // WI-E2 contract block: acceptance rate + throughput in the shape the
        // plan specifies; `speculative` above carries the richer telemetry.
        "speculation": {
            "accepted_rate": engine.acceptance_rate(),
            "tokens_per_sec": tps_json,
        },
        "speculative": spec_json,
        "kv_cache": {
            "used": kv_used,
            "total": kv_total,
            "blocks_used": kv_blocks_used,
            "blocks_total": kv_blocks_total,
        },
        "vram": {
            "used": vram_used,
            "total": vram_total,
        },
        "sys_ram": {
            "used": sys_ram_used,
            "total": sys_ram_total,
        },
        "gpus": gpus_json,
        "hardware": {
            "rocm_gpu_count": rocm_gpu_count,
            "xnack_enabled": xnack_enabled,
        },
        "adapters_active": engine.adapter_count(),
        // F-2: live three-queue scheduler state for the dashboard.
        "scheduler": {
            "active_requests": sched.active_requests,
            "waiting_requests": sched.waiting_requests,
            "admitted_requests": sched.admitted_requests,
            "paused_requests": sched.paused_requests,
        },
        // PLAN-reduce-d2h-h2d A2: measured arena-attention fallback counts.
        "qkv_attention": {
            "device_attempts": qkv_attempts,
            "arena_fallbacks": qkv_fallbacks,
            "sticky_failed_configs": qkv_sticky,
        },
        // PLAN-radix-prefix-consume WI-2: hit-rate + reuse-volume observability.
        "prefix_cache": {
            "enabled": engine.radix_cache_telemetry().0 > 0 || std::env::var("GRIM_RADIX").as_deref() == Ok("on"),
            "lookups": radix_lookups,
            "hit_requests": radix_hit_requests,
            "hit_tokens": radix_hit_tokens,
            "hit_rate": hit_rate,
            "reuse_ratio": prefix_reuse_ratio,
        },
        "models": {
            "grim": grim_models,
            "gguf": gguf_models,
            "other": other_models,
        },
    })
    .into()
}

/// `GET /` — live dashboard HTML. Polls `/api/stats` every 2s for updates.
async fn dashboard_html() -> axum::response::Html<&'static str> {
    axum::response::Html(DASHBOARD_HTML)
}

/// Static embedded logo image from grim-garage.
const GARAGE_LOGO_PNG: &[u8] = include_bytes!("../../grim-garage/web/logo.png");

/// `GET /logo.png` — serves the small grim-garage logo.
async fn serve_logo_png() -> axum::response::Response {
    axum::response::Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "image/png")
        .header(axum::http::header::CACHE_CONTROL, "public, max-age=86400")
        .body(axum::body::Body::from(GARAGE_LOGO_PNG))
        .unwrap()
}

/// Dashboard HTML (live multi-GPU dashboard with WCAG 2.2 accessibility, responsive grid, and live polling).
const DASHBOARD_HTML: &str = include_str!("dashboard.html");

#[cfg(test)]
mod tests;
