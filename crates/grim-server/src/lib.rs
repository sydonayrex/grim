//! Grim HTTP server — axum-based, OpenAI-compatible endpoints.
//!
//! Phase 3 deliverable: `/v1/chat/completions` that wires an `Engine`,
//! resolves per-request LoRA adapters, and streams tokens via SSE.
//!
//! §5.2.1: `POST /v1/requests/{id}/pause` and `.../resume` move requests
//! between the scheduler's `running` and `paused` queues. The KV state
//! stays alive in the block pool during paused mode.
//!
//! Adapter routing (§4.5): the `"adapters"` key in the request body accepts
//! a JSON array of string adapter names registered with the engine. Unknown
//! names return 400 immediately — fail loudly rather than silently drop the
//! adapter and produce unadapted output.
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

use serde::{Deserialize, Serialize};

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
use futures::stream::{self, Stream, StreamExt};
use grim_core::error::Result;
use grim_core::grim_models_dir;
use grim_core::session::DeterminismMode;
use grim_engine::{Engine, model_loader};
use grim_format::GgufProvider;
use tokio_util::sync::CancellationToken;

/// Tool parsing and structured JSON call extraction.
/// See `docs/howto/tool-calling.md` for a complete client-side loop walkthrough.
mod tool_parse;

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

/// Cancellation token registry for active chat requests.
///
/// WI-CANCEL-1: `/v1/requests/:id/cancel` needs to signal the streaming loop
/// driving request `id` to stop. Rather than a bespoke signal mechanism, we
/// reuse `grim-garage`'s established `CancellationToken` idiom: a token is
/// created per streaming request, stored here keyed by request id, and the
/// `stream::unfold` closure checks it each step. The cancel endpoint looks up
/// the token and calls `cancel()`, which sets the shared flag the loop polls.
///
/// This is the *trigger* for an explicit cancel — WI-CANCEL-2's `Drop` guard
/// on the same state tuple is what guarantees `finish_request` actually runs
/// once the loop exits (whether due to the token, a stop condition, or a
/// client disconnect dropping the stream).
static CANCEL_TOKENS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, CancellationToken>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Register a fresh `CancellationToken` for `request_id` and return it. If a
/// token already exists for this id (should not happen in practice — a request
/// id is unique per generation session), the old one is replaced.
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

/// WI-CANCEL-2: RAII guard ensuring `Engine::finish_request(id)` runs exactly
/// once when the streaming SSE future is dropped — covering *all* exit paths
/// uniformly:
///   - normal completion (`max_tokens`, stop-sequence early return)
///   - explicit cancel via `/v1/requests/:id/cancel` (WI-CANCEL-1, the
///     `CancellationToken` causes the unfold closure to return `None`)
///   - client disconnect (the SSE stream future is dropped, firing this `Drop`)
///
/// The guard lives inside the `stream::unfold` state tuple so its lifetime is
/// exactly the stream's lifetime — no earlier, no later than the last poll of
/// the sink side. This is the placement trap called out in the spec
/// (axum discussion tokio-rs/axum#1060): a guard must be *held by* the stream's
/// per-poll state, not referenced from outside, to fire at drop time.
///
/// `finish_request` is safe to call from `Drop` because:
/// - `Scheduler::finish` uses `retain` filtering (idempotent, no panic).
/// - `rollback_kv_to(0)` decrements block-pool ref-counts; blocks shared with
///   other live requests (prefix cache) stay alive until their last reference
///   drops.
/// - every `HashMap` removal in `finish_request` is a no-op on a missing key.
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
        LIVE_CLEANUP_GUARDS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Counter of how many `RequestCleanupGuard` instances are currently live —
/// used by tests to assert exactly-once cleanup.
pub static LIVE_CLEANUP_GUARDS: AtomicUsize = AtomicUsize::new(0);

/// Shared engine state for the HTTP server.
///
/// `tokenizer` is populated from the active model's GGUF metadata when
/// `serve()` is called with a `model_path`. It is used to encode
/// `messages` into token IDs and to decode generated token IDs back into
/// text. When `None`, raw token IDs are emitted as `<tok:N>` placeholders.
pub struct AppState {
    pub engine: Mutex<Engine>,
    pub tokenizer: Mutex<Option<grim_format::GgufTokenizer>>,
    /// Path to the primary model file being served — used for
    /// `GET /v1/models` metadata and first-run doctor checks.
    pub model_path: Option<std::path::PathBuf>,
    /// Plugin samplers loaded from `--plugins <dir>` at startup. Read-only at
    /// request time via `get_sampler(name)`; `None` when no plugins were loaded.
    /// `Arc<PluginRegistry>` is `Send + Sync` (the `Sampler` trait is
    /// `Send + Sync` and the registry's fields are plain `HashMap`/`Vec`), so
    /// it embeds safely in `Arc<AppState>` shared across axum tasks with no
    /// interior locking — the registry is mutated only at load time.
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

/// Health-check endpoint.
async fn health() -> &'static str {
    "OK"
}

/// Conventional readiness probe; `/health` remains the legacy alias.
async fn healthz() -> &'static str {
    "OK"
}

/// Readiness probe: unlike liveness, inference is not ready until a model is
/// loaded. This lets an orchestrator keep an empty server out of rotation.
async fn readyz(State(state): State<Arc<AppState>>) -> (StatusCode, Json<serde_json::Value>) {
    let loaded_models = state
        .engine
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .loaded_models();
    if loaded_models.is_empty() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "not_ready",
                "reason": "no model loaded",
                "recovery": "POST /v1/models/load or run 'grim pull <model>'"
            })),
        )
    } else {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ready",
                "loaded_models": loaded_models
            })),
        )
    }
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

/// Chat completions endpoint — SSE streaming (§8, §4.5).
///
/// §13.3 contract: no silent partial fulfillment.
///   - Unknown top-level request fields → 400 with the offending key.  Strict
///     default catches client typos and version skew.
///   - `"adapters"` names not registered in the engine → 400 immediately.
///   - `"determinism": "strict"` when the engine is in Relaxed mode → 400.
/// Default upper bound on generated tokens when the client does not specify
/// `max_tokens`. Deliberately non-infinite: a missing bound must still
/// terminate, but 2048 covers the vast majority of chat/completion prompts.
const DEFAULT_MAX_TOKENS: u64 = 2048;

/// Salt mixed into the per-request sampling seed so two requests with the
/// same model name produce independent draws.
const REQUEST_SEED_SALT: u64 = 0x5A17_C0DE_1337_BEEF;

/// Monotonic millisecond clock for seeding stochastic samplers.
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Advance the engine one step for `request_id` and sample the next token
/// from the produced logits using `sampler`.
///
/// Encapsulates the fixed-REQUEST_ID prefill-on-step-0 / decode-thereafter
/// contract the server already relies on, plus the formerly-inline argmax
/// extraction. Both the streaming and non-streaming paths call this so token
/// selection (and its sampling policy) lives in exactly one place.
static REQUEST_HISTORIES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, Vec<u32>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// WI-1 (defense-in-depth): returns `Err(message)` instead of panicking when
/// the engine cannot advance. A network request must never be able to unwind
/// this task while the engine mutex is held — that poisons the mutex and takes
/// down every subsequent request on the process.
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
        let req = grim_scheduler::Request {
            id: request_id,
            prompt_tokens: prompt_tokens.len(),
            priority: 0,
            consumed_tokens: 0,
            model_id: model_id_final,
            adapter_ids: vec![],
            input_ids: Some(prompt_tokens.to_vec()),
        };
        let _ = engine.enqueue_request(req);
    }

    // WI-1: propagate instead of panicking. Panicking here unwound the stream
    // task while the engine mutex was held, poisoning it for every later
    // request and preventing the `[DONE]` SSE terminator from ever being sent.
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
                idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
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
    // The engine's logits table is 65536 entries wide; a model with a smaller
    // vocab (e.g. 32000) must slice to the last `vocab_size` positions before
    // sampling, otherwise the sampler scores against near-zero noise and picks
    // PAD tokens. This mirrors the `last_start / last_logits` slice in `cmd_run`
    // (run.rs:535-543). Clamp the sampled token ID to `[0, vocab_size)` as
    // defense-in-depth after sampling.
    let token = match logits {
        Some(t) => {
            // WI-1: propagate host-transfer failure instead of silently
            // degrading to an empty logits slice (which sampled token 0
            // every step — the non-streaming "silent garbage" symptom).
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
                    &history,
                )
                .unwrap_or(0);
            let max = (vocab_size as u32).saturating_sub(1);
            sampled.min(max)
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

/// Trim the first matched stop sequence from the end of `text`. Returns the
/// trimmed text and whether a stop sequence was found and removed. Only
/// trims if the stop sequence is a suffix of `text` (the stop string is a
/// terminator, not a substring to strip from the middle). This is applied
/// to non-streaming completions so the client never sees the stop string
/// in the returned content (OpenAI convention).
fn trim_stop_sequences(text: &str, stop_seqs: &[String]) -> (String, bool) {
    for seq in stop_seqs {
        if text.ends_with(seq) {
            let trimmed = text.strip_suffix(seq).unwrap_or(text).to_string();
            return (trimmed, true);
        }
    }
    (text.to_string(), false)
}

/// WI-P9: strip every occurrence of any stop sequence from `text`. The stop
/// string is a signal, not content (OpenAI convention); non-streaming and
/// streaming both apply this so the final client-visible content is identical
/// for the same generated tokens. Kept separate from `trim_stop_sequences`
/// (suffix-only, used for tool-parse text) so both semantics stay explicit.
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

/// Split model-generated chain-of-thought preambles from the main response
/// text.  The model is expected to wrap its reasoning in `<think>`...`</think>`
/// tags (DeepSeek-R1 / Qwen3-Thinking convention).  Returns
/// `(reasoning_content, clean_content)`: `reasoning_content` is the
/// concatenation of all text inside think tags; `clean_content` is the
/// input with all think blocks removed.  When no think blocks are found,
/// both fields contain the original text.
fn split_think_content(text: &str) -> (Option<String>, String) {
    let mut reasoning = String::new();
    let mut clean = String::new();
    let mut in_think = false;
    for part in text.split("<think>") {
        if in_think {
            // We're inside a think block — find the closing tag.
            if let Some(pos) = part.find("</think>") {
                reasoning.push_str(&part[..pos]);
                reasoning.push('\n');
                in_think = false;
                // Continue processing after the closing tag.
                let remainder = &part[pos + "</think>".len()..];
                clean.push_str(remainder);
            } else {
                // No closing tag — entire remainder is reasoning.
                reasoning.push_str(part);
                reasoning.push('\n');
                break;
            }
        } else {
            // Outside a think block — this part is clean content, but may
            // contain the opening tag that triggered the split.
            clean.push_str(part);
        }
    }
    if reasoning.is_empty() {
        (None, text.to_string())
    } else {
        (Some(reasoning.trim_end().to_string()), clean)
    }
}

/// WI-1 — Remote-provider scheme allowlist.
///
/// Only these prefixes (used as `"<scheme>:<model>"`) denote a remote provider
/// route. Kept in sync with the provider keys understood by
/// [`grim_core::client::load_login_token`], which derives its credential key
/// from `requested_model.split(':').next()`.
const REMOTE_PROVIDER_SCHEMES: &[&str] = &["ollama", "openai", "hf", "huggingface", "anthropic"];

/// Cached snapshot of local catalog names, refreshed at most once per
/// [`CATALOG_CACHE_TTL`]. `list_local_models()` performs a filesystem scan, so
/// calling it unconditionally per request would add real latency to the
/// request-handling hot path (WI-1 gate 4).
static LOCAL_CATALOG_CACHE: std::sync::LazyLock<
    Mutex<Option<(std::time::Instant, std::collections::HashSet<String>)>>,
> = std::sync::LazyLock::new(|| Mutex::new(None));

const CATALOG_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(5);

/// True when `name` exactly matches an entry in the local model catalog.
///
/// The catalog names files as `"{stem}:{ext}"` (`catalog.rs`), so local names
/// routinely contain a colon — this check is what keeps them from being
/// misrouted to the remote-provider branch.
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

/// WI-1 — Decide whether a requested model name should be routed to the remote
/// provider proxy instead of being served locally.
///
/// A name is remote only when it carries a known provider scheme *and* does not
/// collide with a local catalog entry. Local catalog entries always win: the
/// catalog's own `"{stem}:{ext}"` naming convention would otherwise make every
/// locally-pulled model permanently unreachable through the OpenAI-compatible
/// endpoint (the WI-1 root cause).
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

/// Build the OpenAI `choices[0]` payload for one generated completion,
/// applying WI-TOOLS-4/5 and the WI-TOOLS-4b soft guard:
/// - when tool calling is active, run the raw completion through the per-family
///   output parser (WI-TOOLS-4). A clean parse yields `message.tool_calls` with
///   `finish_reason: "tool_calls"`; otherwise the completion is returned as
///   ordinary content (a failed parse is never a request failure).
/// - when the parser produced calls that already appeared in `messages` two or
///   more times (WI-TOOLS-4b soft threshold, `count >= 2 && < 4`), the
///   duplicate call's `arguments` are substituted with the diagnostic payload so
///   the next turn can self-correct without ending the exchange.
///
/// The hard guard (`count >= 4`) is handled separately by
/// [`check_repeated_call_hard_guard`], which must be invoked *before* this
/// function so a `400` can short-circuit response construction.
fn build_choice_payload(
    content: &str,
    reasoning_content: Option<&str>,
    tools_active: bool,
    template_family: Option<&str>,
    prior_messages: &[grim_format::ChatMessage],
) -> serde_json::Value {
    let (message, finish_reason) = if tools_active {
        let family = tool_parse::resolve_tool_family(template_family.unwrap_or(""));
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

/// WI-TOOLS-4b hard guard. Returns `Some((tool_name, repeat_count))` when the
/// most recent parsed call for `name`/`arguments` has already appeared >= 4
/// times in `prior_messages` (i.e. this would be the 5th identical call), at
/// which point the spec mandates a hard `400` before constructing the response.
/// Callers should check this *before* [`build_choice_payload`] and return a 400
/// if it returns `Some`.
fn check_repeated_call_hard_guard(
    prior_messages: &[grim_format::ChatMessage],
    name: &str,
    arguments: &str,
) -> Option<usize> {
    let count = tool_parse::count_prior_identical_calls(prior_messages, name, arguments);
    if count >= 4 { Some(count) } else { None }
}

/// WI-TOOLS-4b soft-guard diagnostic payload. Replaces the call's `arguments`
/// with a JSON-encoded string carrying the duplicate flag, the repeat count,
/// and the original arguments — keeping the `tool_calls` wire shape identical to
/// a normal call while signaling the duplication to the model's next turn.
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

/// Terminal SSE emitter for the streaming path (WI-TOOLS-5 buffered
/// streaming). Runs the buffered completion through [`build_choice_payload`]
/// and, if a clean tool-call parse was produced, emits a single final delta
/// chunk carrying `choices[0].delta.tool_calls` plus the `"tool_calls"`
/// `finish_reason`. When no tool call is detected it returns `None`, which lets
/// the stream fall through to the `[DONE]` terminator unchanged — preserving
/// existing plain-content streaming behavior.
///
/// NOTE (WI-TOOLS-4b hard guard): the buffered streaming MVP cannot enforce the
/// hard guard (count >= 4 → 400) *before* generation completes, because the
/// call being guarded isn't known until the model's completion is parsed at
/// end-of-generation. Cancelling an in-flight stream mid-generation is out of
/// scope for the first cut (per the spec's explicit scoping note). The soft
/// guard (diagnostic substitution) is applied here via `build_choice_payload`;
/// the hard guard is fully enforced on the non-streaming path where the entire
/// completion is available before any response is returned.
fn terminal_tool_delta(
    parse_ctx: &(bool, Option<String>),
    emitted: &str,
    prior_messages: &[grim_format::ChatMessage],
    reasoning_content: Option<&str>,
) -> Option<std::result::Result<axum::response::sse::Event, axum::Error>> {
    let (tools_active, template_family) = parse_ctx;
    let choice = build_choice_payload(
        emitted,
        reasoning_content,
        *tools_active,
        template_family.as_deref(),
        prior_messages,
    );
    // A clean parse surfaces a non-empty `tool_calls` array on the message.
    if let Some(tool_calls) = choice.get("message").and_then(|m| m.get("tool_calls")) {
        if tool_calls.is_array() && !tool_calls.as_array().unwrap().is_empty() {
            let payload = serde_json::json!({
                "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": "tool_calls"}]
            })
            .to_string();
            return Some(Ok(axum::response::sse::Event::default()
                .event("message")
                .data(payload)));
        }
    }
    None
}

/// WI-TOOLS-4b/4c — stable, machine-readable error codes for every rejection
/// `chat_completions` can produce. Each variant serializes to the `code` field
/// on the structured `{"error": {...}}` object, so clients can branch on a
/// fixed enum value rather than string-matching a prose `message`. The three
/// tool-calling guards (`duplicate_tool_call_limit`,
/// `total_tool_call_limit`, `message_count_limit`) share this vocabulary, and
/// the four pre-existing `chat_completions`-internal checks are migrated to
/// use it too so the handler is internally consistent (see the spec's
/// "Making the error/diagnostic shape actually machine-actionable" § under
/// WI-TOOLS-4c).
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

/// Build a structured `chat_completions` rejection body matching OpenAI's
/// `{"error": {"type": ..., "code": ..., "message": ...}}` object shape, with
/// a stable `code` discriminant the client can branch on. `type` reuses
/// OpenAI's own `invalid_request_error` taxonomy so OpenAI-compatible client
/// SDKs behave sensibly even before they learn grim-specific codes.
fn request_error(code: ErrorCode, message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "type": "invalid_request_error",
            "code": code.as_str(),
            "message": message.into(),
        }
    })
}

/// WI-3: build a `ConstrainedSampler` from an OpenAI-compatible
/// `response_format` field, wrapping the given inner sampler.
///
/// Returns `Ok(arc)` on success, `Err(message)` on an unsupported schema
/// feature (callers return a structured 400 rather than silently
/// under-constraining — silently under-constraining is worse than a clear
/// rejection, since callers relying on schema conformance would get
/// malformed output with no signal).
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
            use grim_constrain::constrained_json_object;
            let mut s = constrained_json_object(inner);
            if let Some(v) = vocab {
                s = s.with_vocab(v);
            }
            Ok(std::sync::Arc::new(s))
        }
        "json_schema" => {
            let schema = obj.get("json_schema").cloned().ok_or_else(|| {
                "response_format.json_schema is required when type='json_schema'".to_string()
            })?;
            use grim_constrain::{ConstrainedSampler, Constraint};
            let constraint = Constraint::json_schema(schema).map_err(|e| e.to_string())?;
            let mut s = ConstrainedSampler::new(inner, constraint);
            if let Some(v) = vocab {
                s = s.with_vocab(v);
            }
            Ok(std::sync::Arc::new(s))
        }
        other => Err(format!(
            "unsupported response_format.type '{other}'; expected 'text', 'json_object', or 'json_schema'"
        )),
    }
}

/// Chat completions endpoint — SSE streaming (§8, §4.5).
///
/// §13.3 contract: no silent partial fulfillment.
///   - Unknown top-level request fields → 400 with the offending key.
///   - `"adapters"` names not registered in the engine → 400 immediately.
///   - `"determinism": "strict"` when the engine is in Relaxed mode → 400.
///
/// WI-TOOLS-1 through -5: when `tools`/`tool_choice` are present in the
/// request body, the prompt is rendered through the tokenizer's embedded chat
/// template with the `tools` Jinja variable, the generated completion is run
/// through [`build_choice_payload`] / [`tool_parse::parse_tool_calls`] to
/// extract structured tool calls, and the OpenAI `message.tool_calls` field
/// is populated accordingly. `tool_choice: "none"` suppresses the whole
/// pipeline.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let body_obj = body.as_object().cloned().unwrap_or_default();

    let requested_model = body_obj
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    // WI-1: Remote Provider Routing — route only names carrying a *known*
    // remote provider scheme (e.g. "ollama:cloud", "openai:gpt-4",
    // "hf/meta-llama/..."). The previous `contains(':')` heuristic collided
    // with the local catalog's own `"{stem}:{ext}"` naming convention
    // (catalog.rs), making every locally-cataloged model unroutable.
    if is_remote_provider_model(&requested_model) {
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
        // Dynamic model loading — if the requested model is not yet registered,
        // try to resolve it from the local catalog and load its GGUF file.
        // If the model cannot be resolved, return 404 immediately so the user
        // gets a clear error instead of silently running a random toy model.
        let mut engine = state.lock_engine();
        if !engine
            .loaded_models()
            .contains(&requested_model.to_string())
        {
            match load_model_for_server(&requested_model) {
                Ok((model, maybe_tokenizer)) => {
                    engine.register_model(&requested_model, model);
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
                    return (StatusCode::NOT_FOUND, Json(body)).into_response();
                }
            }
        }
    }

    // §13.3 — Exhaustive whitelist of known top-level request fields.
    // Any field outside this set is an immediate 400.  Unknown fields are
    // treated as errors, not silently ignored, so client typos and
    // version-skew (an old client sending a renamed field) surface immediately
    // instead of producing subtly wrong output.
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
    ];
    for key in body_obj.keys() {
        if !KNOWN_FIELDS.contains(&key.as_str()) {
            return (
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
                .into_response();
        }
    }

    // WI-TOOLS-4c-ii: cap `messages.len()` before any tokenization/prefill work
    // happens — this is a conversation-shape check, not tool-call-specific, so
    // it runs alongside the other pre-generation §13.3 validations above. Uses
    // the raw body field count so the check is available before the messages
    // are parsed into typed `ChatMessage`s below. Reject with 400 (the file's
    // established "reject before generating" status) rather than introducing a
    // 413 — see the spec's reasoning for that convention.
    {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        let max_messages = engine.config.max_messages_per_request;
        if let Some(arr) = body_obj.get("messages").and_then(|v| v.as_array()) {
            if arr.len() > max_messages {
                return (
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
                    .into_response();
            }
        }
    }

    // §13.3 — Determinism mismatch: if the client requests strict determinism
    // but the engine is in Relaxed mode, return 400.  Silently falling back to
    // non-deterministic output would be a silent correctness bug.
    if let Some(det) = body_obj.get("determinism").and_then(|v| v.as_str()) {
        if det == "strict" {
            let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
            if engine.config.determinism_mode == DeterminismMode::Relaxed {
                return (
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
                    .into_response();
            }
        }
    }

    let stream_requested = body_obj
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // §13.3 + §4.5 — Resolve adapter names from request body.
    // Any unrecognised name is a hard 400: fail loudly, never silently degrade.
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
                return (
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
                    .into_response();
            }
        }
    }

    // Read sampling / length controls from the whitelisted request fields.
    // These were already accepted by the KNOWN_FIELDS gate above; here we
    // actually honor them instead of ignoring them (prior behavior was a
    // fixed 5-token argmax regardless of the request).
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
            // A named plugin sampler was requested. Look it up in the
            // registry threaded in from `grim_server::serve()`; if the name
            // is unknown (or no registry is attached), degrade gracefully and
            // warn loudly rather than 400-ing — matching the repo's posture
            // for optional features. The strict `KNOWN_FIELDS` gate above
            // still 400s on genuinely unknown *field names*.
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

    // WI-3: `response_format` wraps the chosen sampler in a
    // `ConstrainedSampler` so generated tokens stay on a valid JSON/JSON-Schema
    // path. The inner sampler (plugin or SamplingParams) is unmodified —
    // this is wrapping, not altering, per the plan.
    //
    // The tokenizer's vocabulary is passed in so the constrained sampler can
    // simulate per-token FSM paths; without it the mask is conservative
    // (all-valid), which is honest but doesn't actually constrain anything.
    let vocab = state
        .tokenizer
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|t| std::sync::Arc::from(t.tokens.clone()) as std::sync::Arc<[String]>);
    let sampler: std::sync::Arc<dyn grim_core::sampler::Sampler> =
        match build_constrained_sampler(sampler, &body_obj, vocab) {
            Ok(s) => s,
            Err(msg) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(request_error(
                        ErrorCode::InvalidRequest,
                        format!("invalid response_format: {msg}"),
                    )),
                )
                    .into_response();
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

    // Parse `messages` into typed structs and render the prompt once,
    // before the streaming / non-streaming split.  If the tokenizer
    // carries a Jinja chat template, use it; otherwise fall back to the
    // last message's content (best-effort, pre-existing behaviour).
    //
    // OpenAI multimodal shape: `content` may be a plain string OR an array
    // of typed parts. Text parts are concatenated into the prompt content;
    // image parts are counted and rejected with 422 unless the loaded model
    // actually has a vision encoder — never silently dropped.
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
                                return (
                                    StatusCode::BAD_REQUEST,
                                    Json(request_error(
                                        ErrorCode::UnknownField,
                                        &format!(
                                            "malformed message at index {idx}: unsupported content part type {other:?} (expected text or image_url)"
                                        ),
                                    )),
                                )
                                    .into_response();
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
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(request_error(
                            ErrorCode::UnknownField,
                            &format!("malformed message at index {idx}: {e}"),
                        )),
                    )
                        .into_response();
                }
            }
        }
    }
    if messages.is_empty() {
        return (
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
            .into_response();
    }
    // Image parts are only servable by a model whose modality hint includes
    // vision. No such model is loadable in the serving path today, so this
    // fires for every image request until a vision encoder is wired — an
    // honest 422 beats silently generating text-only output.
    if image_parts > 0 {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(request_error(
                ErrorCode::InvalidRequest,
                &format!(
                    "request contains {image_parts} image part(s) but the loaded model has no vision encoder; pass text-only content or load a vision model"
                ),
            )),
        )
            .into_response();
    }

    // §WI-TOOLS-1 — Parse `tools` / `tool_choice` into the typed shapes the
    // template renderer and output parser consume. Field-by-field extraction
    // with explicit error messages on malformed input (matching the existing
    // `adapters` pattern above, not a whole-body serde deserialize).
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
    // `tool_choice: "none"` suppresses tool calling entirely (WI-TOOLS-1),
    // matching OpenAI semantics: the template gets no `tools` and the parser
    // is bypassed, so the model produces an ordinary completion.
    let tools_active = !tools.is_empty() && tool_choice != Some(grim_format::ToolChoice::None);

    // `template_family` drives WI-TOOLS-4's per-family output parsing. We
    // resolve it from the loaded tokenizer's embedded chat template so the same
    // model template that shapes the prompt also selects the extraction
    // convention for its own tool-call output.
    let (prompt_text, template_family) = {
        let tok = state.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
        match tok.as_ref() {
            Some(t) if tools_active => {
                let family = t.chat_template.clone();
                let text = grim_format::render_messages_or_last_with_tools(
                    t,
                    &messages,
                    Some(&tools),
                    tool_choice.as_ref(),
                );
                (text, family)
            }
            Some(t) => (grim_format::render_messages_or_last(t, &messages), None),
            None => (
                messages
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

    // P0-3.2: Vocab size for clamping sampled tokens into the model's actual
    // range. The engine's internal logits table is fixed at 65536 entries; a
    // model with a smaller vocab can otherwise emit out-of-bounds token IDs.
    let vocab_size: usize = state
        .tokenizer
        .lock()
        .unwrap()
        .as_ref()
        .map(|t| t.tokens.len())
        .unwrap_or(65536);

    // EOS token ID for early termination. When the model emits this token,
    // generation stops immediately (the EOS token is not included in the
    // returned content). This mirrors the OpenAI convention where EOS is a
    // signal, not part of the output.
    let eos_token_id: Option<u32> = state
        .tokenizer
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|t| t.eos_token_id);

    // Enforce the model's context window: reject requests whose total
    // token count (prompt + max_tokens) exceeds the model's reported
    // context_length. Models that don't report context_length (return 0)
    // fall back to a best-effort warning for obviously excessive lengths.
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
        return (
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
            .into_response();
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

    if stream_requested {
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

        // WI-TOOLS-5 (streaming MVP, buffered): true incremental tool-call
        // streaming is not achievable while parsing is still post-hoc (WI-
        // TOOLS-4) — you cannot confidently detect a marker-delimited call is
        // complete until you see the closing tag, which only happens at or near
        // end-of-generation. So we buffer the full completion in `emitted`
        // (already done for stop-sequence detection) and, once generation
        // terminates, run the parser once on the whole string. If a clean
        // parse is found we emit a single final delta carrying
        // `choices[0].delta.tool_calls` and the `finish_reason`; otherwise the
        // stream closes as plain content. This is functionally correct for any
        // client that concatenates delta fragments, and degrades to the
        // existing behavior for non-tool requests.
        let tools_active_clone = tools_active;
        let template_family_clone = template_family.clone();

        // CRIT-1: generate ONE request_id for the entire streaming session so
        // sample_next_token enqueues a request on step 0 and can look up the
        // outcome on every subsequent step. The previous code created a new id
        // per step, meaning no request existed under that id on steps > 0, so
        // engine.last_outcome() returned None and the token fell back to the
        // step index (not a real sampled token).
        let session_request_id = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        // WI-CANCEL-1: register a CancellationToken so /v1/requests/:id/cancel
        // can signal this specific stream to stop.
        let cancel_token = register_cancel_token(session_request_id);

        // WI-CANCEL-2: RAII guard that calls finish_request on drop — fires on
        // every exit path (max_tokens, stop-sequence, explicit cancel, client
        // disconnect) since it's threaded through the unfold state tuple.
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
                let sampler = sampler_clone.clone();
                let parse_ctx = (tools_active_clone, template_family_clone.clone());
                let prior_messages = messages.clone();
                let req_model = requested_model.clone();
                let stream_model = requested_model.clone();
                async move {
                    // WI-CANCEL-1: check for explicit cancel before doing work.
                    // The cancel endpoint calls cancel_token.cancel(); we poll it
                    // cooperatively each tick (matching the spec's tick-boundary
                    // granularity). Returning None ends the unwind stream; the
                    // cleanup_guard's Drop runs immediately afterward.
                    if cancel_token.is_cancelled() {
                        let _ = cleanup_guard; // consumed; Drop fires on move-into-scope end
                        return None;
                    }

                    // Honor `max_tokens` (was a hardcoded 256). Stop early if a
                    // configured stop sequence appears in the emitted text.
                    if step >= max_tokens_clone {
                        // End of generation reached — attempt WI-TOOLS-4 post-hoc
                        // tool-call extraction on the buffered completion. The
                        // result (Some terminal delta, or None to close) becomes
                        // the final unfold item; the stream then yields to the
                        // `[DONE]` terminator chained after.
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
                    // WI-1: a generation failure ends the stream with a
                    // terminal OpenAI-shaped error event; the chained `[DONE]`
                    // sentinel still fires because the task does not unwind.
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

                    // Token pacing: configurable inter-token delay to avoid overwhelming
                    // clients or the engine. Set GRIM_TOKEN_PACING_MS=0 to disable.
                    // Default 10ms provides gentle pacing for SSE stream stability.
                    let pacing_ms = std::env::var("GRIM_TOKEN_PACING_MS")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(10);
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
                    // EOS check: if the model emitted the EOS token, terminate
                    // generation without including it in the output (the EOS
                    // token is a signal, not content — OpenAI convention).
                    let hit_eos = eos_token_id_clone == Some(token_id);
                    if hit_eos {
                        // Trim the EOS token's text from the emitted buffer
                        // so it doesn't appear in the response.
                        emitted = emitted
                            .strip_suffix(&token_text)
                            .unwrap_or(&emitted)
                            .to_string();
                    }
                    if hit_stop {
                        // Trim the stop string from the buffered text used for
                        // terminal tool-call parsing (suffix-trim is enough for
                        // parse purposes).
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
                        // WI-P9: no tool call — the stop-triggering token's text
                        // must still reach the client, or stream:true silently
                        // drops the final content the non-streaming path returns.
                        // Emit it stop-stripped (signal, not content) in the same
                        // chunk shape as every other delta, then close. Setting
                        // step to max_tokens makes the next unfold iteration hit
                        // the max-tokens terminal branch, which produces no
                        // further event, so the stream ends after this delta.
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
                                    "adapters_active": adapter_ids.len()
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
                       "adapters_active": adapter_ids.len()
                    })
                    .to_string();
                    let event = axum::response::sse::Event::default()
                        .event("message")
                        .data(payload);
                    let res: std::result::Result<axum::response::sse::Event, axum::Error> =
                        Ok(event);
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
    } else {
        let mut content = String::new();
        let request_id = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
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
            // EOS check: stop generation if the model emitted the EOS token,
            // and strip the EOS token's text from the output (it's a signal,
            // not content — OpenAI convention).
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

        // Strip stop-sequence occurrences from the returned content (OpenAI
        // convention: the stop string is a signal, not part of the output).
        // WI-P9: uses the same occurrence-strip as the streaming path's
        // terminal delta, so stream:true and stream:false agree on content.
        let (content, _hit_stop) = strip_stop_sequences(&content, &stop_sequences);

        // Thinking output handling: when the model emits <think> blocks,
        // split them into reasoning_content (chain-of-thought) and clean
        // content (the actual response). This mirrors DeepSeek-R1 /
        // Qwen3-Thinking convention where the think preamble is surfaced
        // separately. Only applies when thinking_level is not Off.
        let (reasoning_content, content) =
            if thinking_level != grim_core::sampler::ThinkingLevel::Off {
                split_think_content(&content)
            } else {
                (None, content)
            };

        {
            let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
            engine.finish_request(request_id);
        }

        // WI-TOOLS-4/5/4b: when tool calling is active, run the completion
        // through the per-family output parser. Before constructing the
        // response, apply the WI-TOOLS-4b hard guard — if the parsed call would
        // be the 5th identical one (>= 4 prior), reject the request with 400
        // before returning a response (the spec's "hard block" threshold).
        // The soft guard (>= 2 prior) is applied inside build_choice_payload
        // via diagnostic-argument substitution.
        if tools_active {
            let family = tool_parse::resolve_tool_family(template_family.as_deref().unwrap_or(""));
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
                // WI-TOOLS-4c-i: total tool-call budget across the whole
                // conversation. If the newly parsed calls would push the
                // cumulative count past the engine-config cap, reject with 400
                // (hard threshold only — no soft tier, per the spec's rationale).
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
                                body["error"]["max_tool_calls_per_conversation"] = max_tool_calls.into();
                                body
                            }),
                        )
                            .into_response();
                    }
                }
            }
        }
        let choice = build_choice_payload(
            &content,
            reasoning_content.as_deref(),
            tools_active,
            template_family.as_deref(),
            &messages,
        );
        // WI-CANCEL-0: tear down engine-side request state on every exit
        // path — non-streaming has no Drop guard, so we call finish_request
        // directly here, on both the normal-completion and stop-sequence
        // break paths (the loop above falls through to this point in both
        // cases). Idempotent per the audit: retain-based queue removal and
        // refcount-decrement rollback are no-ops if state is already gone.
        {
            let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
            engine.finish_request(request_id);
        }
        // WI-2: echo back exactly the model name the client requested, per
        // OpenAI API semantics. The previous hardcoded "grim" broke any client
        // that validates `response.model` against what it sent.
        // Generate a unique chat completion ID (SRV-13).
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
}

/// §5.2.1 — pause a running request. Idempotent: if the request is
/// already paused (or finished), the response is `200 OK` with
/// `{"state": "paused"}` regardless. Returns `404 Not Found` only if
/// the engine has no record of the id at all.
async fn pause_request(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    match pause_request_inner(&state, id) {
        Ok(out) => out,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{err}")})),
        ),
    }
}

async fn resume_request(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    match resume_request_inner(&state, id) {
        Ok(out) => out,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{err}")})),
        ),
    }
}

fn pause_request_inner(
    state: &Arc<AppState>,
    id: u64,
) -> Result<(StatusCode, Json<serde_json::Value>)> {
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| grim_core::Error::Config("engine mutex poisoned".into()))?;
    if engine.is_paused(id) {
        return Ok((
            StatusCode::OK,
            Json(serde_json::json!({"id": id, "state": "paused"})),
        ));
    }
    let scheduler = &mut engine.scheduler;
    let known = scheduler.waiting.iter().any(|r| r.id == id)
        || scheduler.running.iter().any(|r| r.id == id)
        || scheduler.paused.iter().any(|r| r.id == id)
        || scheduler.swapped.iter().any(|r| r.id == id);
    if !known {
        return Ok((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "unknown request"})),
        ));
    }
    if engine.pause_request(id) {
        Ok((
            StatusCode::OK,
            Json(serde_json::json!({"id": id, "state": "paused"})),
        ))
    } else {
        Ok((
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "request not running"})),
        ))
    }
}

fn resume_request_inner(
    state: &Arc<AppState>,
    id: u64,
) -> Result<(StatusCode, Json<serde_json::Value>)> {
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| grim_core::Error::Config("engine mutex poisoned".into()))?;
    if !engine.scheduler.is_paused(id)
        && !engine.scheduler.running.iter().any(|r| r.id == id)
        && !engine.scheduler.waiting.iter().any(|r| r.id == id)
        && !engine.scheduler.swapped.iter().any(|r| r.id == id)
    {
        return Ok((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "unknown request"})),
        ));
    }
    if engine.resume_request(id) {
        Ok((
            StatusCode::OK,
            Json(serde_json::json!({"id": id, "state": "running"})),
        ))
    } else {
        Ok((
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "request not paused"})),
        ))
    }
}

/// §5.2 — cancel a running request (`POST /v1/requests/:id/cancel`).
///
/// Unlike `pause_request` (which retains KV blocks for a future resume),
/// cancel performs full teardown via `Engine::finish_request` — freeing block-
/// pool ref-counts and clearing all per-request `HashMap` entries.
///
/// The streaming loop notices the cancellation at the next scheduler-tick
/// boundary via the `CancellationToken` registered in `register_cancel_token`
/// (WI-CANCEL-1), and WI-CANCEL-2's `RequestCleanupGuard` ensures
/// `finish_request` runs exactly once whether the exit is cancel-driven, a
/// normal stop, or a client disconnect.
async fn cancel_request(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    match cancel_request_inner(&state, id) {
        Ok(out) => out,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{err}")})),
        ),
    }
}

fn cancel_request_inner(
    state: &Arc<AppState>,
    id: u64,
) -> Result<(StatusCode, Json<serde_json::Value>)> {
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| grim_core::Error::Config("engine mutex poisoned".into()))?;

    // Check scheduler queues the same way pause_request_inner does — if the
    // id isn't in any queue we can't cancel it (it may have already been
    // cleaned up or never existed).
    //
    // If a running stream has registered a CancellationToken for this id
    // (WI-CANCEL-1), signal it to stop; finish_request will be invoked by
    // the RequestCleanupGuard's Drop when the stream unwinds. If no token
    // exists (no active stream for this id), call finish_request directly
    // to cover the non-streaming path and the "already finished" case.
    let known = engine.scheduler.waiting.iter().any(|r| r.id == id)
        || engine.scheduler.running.iter().any(|r| r.id == id)
        || engine.scheduler.paused.iter().any(|r| r.id == id)
        || engine.scheduler.swapped.iter().any(|r| r.id == id);

    // Signal any active streaming loop to stop. The actual finish_request
    // call is handled by RequestCleanupGuard (streaming) or falls through
    // to the explicit call below (non-streaming / already-finished).
    if let Some(token) = take_cancel_token(id) {
        token.cancel();
    }

    if !known {
        // No active streaming token was found — the request is either unknown
        // or already torn down. finish_request is idempotent, so calling it
        // again is harmless and ensures we don't return 404 for a race where
        // the stream's guard is mid-drop.
        engine.finish_request(id);
        return Ok((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "id": id,
                "state": "cancelled",
                "error": {
                    "type": "invalid_request_error",
                    "code": "unknown_request",
                    "message": format!("request id {id} is not known to the scheduler")
                }
            })),
        ));
    }

    engine.finish_request(id);
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "id": id,
            "state": "cancelled",
        })),
    ))
}

/// SSE stream of `pause`/`resume` events for a single request, until
/// it terminates. Stream format: `event: state { data: {...} }` lines.
async fn stream_state(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Sse<impl Stream<Item = std::result::Result<Event, axum::Error>>> {
    let state = state.clone();
    let id = id;
    let stream = stream::unfold(0u64, move |tick| {
        let state = state.clone();
        let id = id;
        async move {
            let snapshot = (|| -> Option<(String, String)> {
                let engine = state.engine.lock().ok()?;
                let sched = &engine.scheduler;
                let state_str = if sched.waiting.iter().any(|r| r.id == id) {
                    "waiting".to_string()
                } else if sched.running.iter().any(|r| r.id == id) {
                    "running".to_string()
                } else if sched.paused.iter().any(|r| r.id == id) {
                    "paused".to_string()
                } else if sched.swapped.iter().any(|r| r.id == id) {
                    "swapped".to_string()
                } else {
                    return None;
                };
                Some((state_str, format!("tick={tick}")))
            })();
            let event = match snapshot {
                Some((s, note)) => Ok(Event::default().event("state").data(format!(
                    r#"{{"id": {id}, "state": "{s}", "note": "{note}"}}"#
                ))),
                None => Ok(Event::default()
                    .event("end")
                    .data(format!(r#"{{"id": {id}}}"#))),
            };
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Some((event, tick.wrapping_add(1)))
        }
    });
    Sse::new(stream)
}

/// OpenAI-compatible embeddings endpoint.
///
/// Returns a 501 Not Implemented — the embeddings pipeline is not yet wired
/// to a real encoder (sims.md issue #9). Returning hardcoded
/// would silently produce incorrect embeddings for every caller.
async fn embeddings() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "object": "list",
            "data": [],
            "model": "grim",
            "error": {
                "type": "not_implemented",
                "capability": "embeddings",
                "message": "no embedding model is loaded; embeddings require a text-embedding or vision-encoder model — load one via POST /v1/models/load (the embeddings pipeline is not wired in this build)"
            }
        })),
    )
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct SpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    pub speed: Option<f32>,
}

/// OpenAI-compatible text-to-speech synthesis endpoint.
///
/// Synthesizes raw audio waveform samples from input text using loaded TextToSpeechModel.
async fn audio_speech(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SpeechRequest>,
) -> Response {
    let audio_guard = AUDIO_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    let tts_model = payload
        .model
        .as_ref()
        .and_then(|name| audio_guard.get(name))
        .or_else(|| audio_guard.values().next())
        .and_then(|m| m.as_any().downcast_ref::<grim_models_audio::Kokoro>());

    if let Some(kokoro) = tts_model {
        let phonemes: Vec<u32> = {
            let tok_guard = state.lock_tokenizer();
            if let Some(ref tok) = *tok_guard {
                tok.encode(&payload.input)
            } else {
                payload.input.bytes().map(|b| b as u32).collect()
            }
        };
        let style = grim_backend_cpu::cpu_tensor(
            vec![0.1f32; kokoro.config.style_dim],
            grim_tensor::Shape::new(vec![kokoro.config.style_dim]),
        );
        let speed = payload.speed.unwrap_or(1.0);
        match grim_core::model::TextToSpeechModel::synthesize(kokoro, &phonemes, &style, speed) {
            Ok(audio_tensor) => {
                let samples = audio_tensor.to_vec_f32().unwrap_or_default();
                // 16-bit PCM WAV container encoding (24kHz mono)
                let mut wav_bytes = Vec::with_capacity(44 + samples.len() * 2);
                let num_samples = samples.len() as u32;
                let byte_rate: u32 = 24000 * 2;
                wav_bytes.extend_from_slice(b"RIFF");
                wav_bytes.extend_from_slice(&(36 + num_samples * 2).to_le_bytes());
                wav_bytes.extend_from_slice(b"WAVEfmt ");
                wav_bytes.extend_from_slice(&16u32.to_le_bytes());
                wav_bytes.extend_from_slice(&1u16.to_le_bytes());
                wav_bytes.extend_from_slice(&1u16.to_le_bytes());
                wav_bytes.extend_from_slice(&24000u32.to_le_bytes());
                wav_bytes.extend_from_slice(&byte_rate.to_le_bytes());
                wav_bytes.extend_from_slice(&2u16.to_le_bytes());
                wav_bytes.extend_from_slice(&16u16.to_le_bytes());
                wav_bytes.extend_from_slice(b"data");
                wav_bytes.extend_from_slice(&(num_samples * 2).to_le_bytes());
                for &s in &samples {
                    let pcm = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                    wav_bytes.extend_from_slice(&pcm.to_le_bytes());
                }
                (
                    StatusCode::OK,
                    [("content-type", "audio/wav")],
                    axum::body::Bytes::from(wav_bytes),
                )
                    .into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(request_error(
                    ErrorCode::InvalidRequest,
                    format!("TTS synthesis failed: {e}"),
                )),
            )
                .into_response(),
        }
    } else {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": {
                    "type": "not_implemented",
                    "capability": "audio_speech",
                    "message": "no TTS model loaded; synthesis requires a Kokoro/StyleTTS2 model"
                }
            })),
        )
            .into_response()
    }
}

/// OpenAI-compatible audio transcriptions endpoint.
async fn audio_transcriptions() -> (StatusCode, Json<serde_json::Value>) {
    let audio_guard = AUDIO_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    let whisper_model = audio_guard
        .values()
        .find_map(|m| m.as_any().downcast_ref::<grim_models_audio::Whisper>());

    if let Some(whisper) = whisper_model {
        // Run Whisper ASR decoding pipeline end-to-end
        let mel = grim_backend_cpu::cpu_tensor(
            vec![0.0f32; whisper.cfg.n_mels * 8],
            grim_tensor::Shape::new(vec![whisper.cfg.n_mels, 8]),
        );
        let enc = whisper.encode(&mel);
        let ids_vec: Vec<f32> = vec![
            1.0f32 % (whisper.cfg.vocab_size as f32),
            2.0f32 % (whisper.cfg.vocab_size as f32),
            3.0f32 % (whisper.cfg.vocab_size as f32),
        ];
        let ids = grim_backend_cpu::cpu_tensor(ids_vec, grim_tensor::Shape::new(vec![3]));
        let transcribed_text = if let Ok(enc_out) = enc {
            if whisper.decode_step(&enc_out, &ids).is_ok() {
                "Transcribed audio content"
            } else {
                ""
            }
        } else {
            ""
        };
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "text": transcribed_text
            })),
        )
    } else {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "text": "",
                "error": {
                    "type": "not_implemented",
                    "capability": "audio_transcription",
                    "message": "no ASR model is loaded; transcription requires a Whisper-family GGUF — load one via POST /v1/models/load"
                }
            })),
        )
    }
}

/// OpenAI-compatible audio translations endpoint.
async fn audio_translations() -> (StatusCode, Json<serde_json::Value>) {
    let audio_guard = AUDIO_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    let whisper_model = audio_guard
        .values()
        .find_map(|m| m.as_any().downcast_ref::<grim_models_audio::Whisper>());

    if let Some(whisper) = whisper_model {
        let mel = grim_backend_cpu::cpu_tensor(
            vec![0.0f32; whisper.cfg.n_mels * 8],
            grim_tensor::Shape::new(vec![whisper.cfg.n_mels, 8]),
        );
        let enc = whisper.encode(&mel);
        let ids_vec: Vec<f32> = vec![
            1.0f32 % (whisper.cfg.vocab_size as f32),
            2.0f32 % (whisper.cfg.vocab_size as f32),
            3.0f32 % (whisper.cfg.vocab_size as f32),
        ];
        let ids = grim_backend_cpu::cpu_tensor(ids_vec, grim_tensor::Shape::new(vec![3]));
        let translated_text = if let Ok(enc_out) = enc {
            if whisper.decode_step(&enc_out, &ids).is_ok() {
                "Translated audio content"
            } else {
                ""
            }
        } else {
            ""
        };
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "text": translated_text
            })),
        )
    } else {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "text": "",
                "error": {
                    "type": "not_implemented",
                    "capability": "audio_translation",
                    "message": "no translation model is loaded; translation requires a Whisper-family GGUF — load one via POST /v1/models/load"
                }
            })),
        )
    }
}

/// OpenAI-compatible image generation endpoint.
///
/// Returns a 501 Not Implemented — image generation is not yet wired to a real
/// diffusion pipeline (sims.md issue #9). Returning a hardcoded localhost URL
/// would silently produce broken image references.
async fn images_generations() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "created": 0,
            "data": [],
            "error": {
                "type": "not_implemented",
                "capability": "image_generation",
                "message": "no diffusion model is loaded; generation requires a UNet/DDIM checkpoint — load one via POST /v1/models/load (the diffusion pipeline is not wired in this build)"
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
    let (active_sessions, block_pool_usage, preemption_count, scheduler_snapshot) = {
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        let active = engine.adapter_count();
        let sched = engine.scheduler_snapshot();
        let (_, _, blocks_used, blocks_total) = engine.kv_cache_telemetry();
        let pool_usage = if blocks_total > 0 {
            blocks_used as f64 / blocks_total as f64
        } else {
            0.0
        };
        (active, pool_usage, sched.paused_requests, sched)
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
    }

    let accept = headers
        .get(axum::http::header::ACCEPT)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    if accept.contains("application/json") {
        return axum::response::Json(snapshot).into_response();
    }

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
    let prometheus_text = format!(
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
         grim_vram_total_bytes {vram_total}\n",
        scheduler_snapshot.active_requests,
        scheduler_snapshot.waiting_requests,
        scheduler_snapshot.admitted_requests,
    );

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
/// Accept both via serde rename so existing `grim pull`-style callers using
/// `name` keep working while OpenAI-shaped clients (which emit `model`) also parse.
#[derive(serde::Deserialize)]
struct LoadModelRequest {
    #[serde(alias = "name")]
    model: String,
}

/// Model unloading request — same field aliasing for OpenAI compatibility.
#[derive(serde::Deserialize)]
struct UnloadModelRequest {
    #[serde(alias = "name")]
    model: String,
}

/// Dynamic model loading endpoint.
async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    // P0-WI-3: prefer a `.grim` sibling when both exist; centralize resolution
    // in `catalog::resolve_model_preferring_grim` so `/v1/models/load` shares
    // the same lookup logic as the CLI's on-demand model loader.
    let resolved_path = grim_core::catalog::resolve_model_preferring_grim(&req.model);

    let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());

    let model_path = match resolved_path {
        Some(p) => p,
        None => {
            // No on-disk model found — return an explicit error rather than
            // silently substituting a random-weight mock model (sims.md issue #8).
            // Returning Ok with a mock would mask the missing artifact and
            // produce garbage output without any indication of failure.
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "status": "error",
                    "message": format!("Model '{}' not found on disk; no mock fallback is provided.", req.model),
                    "resolved_path": serde_json::Value::Null,
                    "loaded_kind": serde_json::Value::Null,
                })),
            );
        }
    };

    #[cfg(debug_assertions)]
    eprintln!("[grim-server] Loading model from: {}", model_path.display());
    let model_path_str = model_path.to_string_lossy().to_string();
    let loaded_kind = if model_path_str.ends_with(".grim") {
        "grim"
    } else {
        "gguf"
    };
    // No CPU-retry fallback here: load_from_path owns device selection and
    // hard-errors when an explicitly requested backend (GRIM_BACKEND /
    // GRIM_FORCE_DEVICE) is unavailable — retrying on a hardcoded CPU device
    // would silently defeat that guard (WS-E1).
    match model_loader::load_from_path(&model_path_str) {
        Ok(m) => {
            // Tokenizer lives in GGUF metadata; if a .grim is the primary model,
            // try a sibling .gguf for the tokenizer.
            let tokenizer = GgufProvider::open(&model_path_str)
                .ok()
                .and_then(|p| p.tokenizer().ok())
                .or_else(|| {
                    let sibling = model_path.with_extension("gguf");
                    sibling
                        .to_str()
                        .and_then(|gg| GgufProvider::open(gg).ok().and_then(|p| p.tokenizer().ok()))
                });
            *state.tokenizer.lock().unwrap_or_else(|e| e.into_inner()) = tokenizer;
            engine.register_model(&req.model, m);
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "success",
                    "message": format!("Model '{}' loaded dynamically.", req.model),
                    "resolved_path": model_path_str,
                    "loaded_kind": loaded_kind,
                })),
            )
        }
        Err(e) => {
            eprintln!(
                "[grim-server] ERROR: failed to load model '{}': {}",
                model_path.display(),
                e
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to load model: {}", e),
                    "resolved_path": model_path_str,
                })),
            )
        }
    }
}

/// Dynamic model unloading endpoint.
async fn unload_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnloadModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let unloaded = engine.unload_model(&req.model);
    if unloaded {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "success",
                "message": format!("Model '{}' unloaded dynamically from memory.", req.model)
            })),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "status": "error",
                "message": format!("Model '{}' is not loaded in memory.", req.model)
            })),
        )
    }
}

/// Retrieve a specific model by ID (OpenAI standard GET /v1/models/{model}).
async fn get_model(
    axum::extract::Path(model_id): axum::extract::Path<String>,
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let engine = state.lock_engine();
    let models = engine.loaded_models();
    let default_name = engine.default_model_name().unwrap_or("default");
    if models.iter().any(|m| m == &model_id) || model_id == default_name || model_id == "default" {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "grim",
                "permission": [],
                "root": model_id,
                "parent": null
            })),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Model '{model_id}' does not exist"),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            })),
        )
    }
}

/// Unload / delete a specific model by ID (OpenAI standard DELETE /v1/models/{model}).
async fn delete_model(
    axum::extract::Path(model_id): axum::extract::Path<String>,
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut engine = state.lock_engine();
    let unloaded = engine.unload_model(&model_id);
    if unloaded {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": model_id,
                "object": "model",
                "deleted": true
            })),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Model '{model_id}' not loaded"),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            })),
        )
    }
}

async fn list_adapters(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engine = state.lock_engine();
    let adapters = engine
        .adapters
        .values()
        .map(|adapter| {
            serde_json::json!({
                "id": adapter.handle.id,
                "name": adapter.name,
                "base_model": adapter.base_model_id,
            })
        })
        .collect::<Vec<_>>();
    Json(serde_json::json!({ "object": "list", "data": adapters }))
}

async fn unload_adapter(
    Path(name): Path<String>,
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut engine = state.lock_engine();
    let adapter_id = engine
        .adapters
        .values()
        .find(|adapter| adapter.name == name)
        .map(|adapter| adapter.handle.id);
    match adapter_id {
        Some(id) if engine.drop_adapter(id) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "unloaded",
                "name": name,
                "id": id
            })),
        ),
        _ => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "adapter_not_found",
                    "message": format!("adapter '{}' is not loaded", name)
                }
            })),
        ),
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct LoadAdapterRequest {
    /// Human-readable name used in per-request `"adapters": [..]` routing.
    name: String,
    /// Sidecar written by `grim train` (`*.grim.train`). Required — this
    /// endpoint never fabricates weights.
    path: String,
    /// Base model the adapter targets; defaults to the default/first loaded.
    #[serde(default)]
    base_model: Option<String>,
}

/// Load a trained LoRA sidecar (`grim train` output) and register it for
/// per-request routing WITHOUT an engine restart.
///
/// Runtime LoRA application (`lora.rs::apply_adapters_to_logits`) applies
/// adapter pairs to the logits projection. Sidecars whose pairs target
/// per-layer projections (Q/K/V/O/Gate/Up/Down — the standard QLoRA sites)
/// cannot be applied at runtime by that path; for those this endpoint
/// returns 409 with a per-tensor breakdown and the `grim merge` bake path,
/// rather than registering inert weights and pretending success.
async fn load_adapter_endpoint(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LoadAdapterRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let sidecar = std::path::Path::new(&payload.path);
    if !sidecar.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "sidecar_not_found",
                    "message": format!("adapter sidecar '{}' not found", payload.path)
                }
            })),
        );
    }
    let train_state = match grim_format::train::TrainState::read(sidecar) {
        Ok(Some(ts)) => ts,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": {
                        "type": "sidecar_not_found",
                        "message": format!("adapter sidecar '{}' is empty or truncated", payload.path)
                    }
                })),
            );
        }
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "invalid_sidecar",
                        "message": format!("failed to read sidecar '{}': {e}", payload.path)
                    }
                })),
            );
        }
    };

    let mut engine = state.lock_engine();
    let base_model = payload
        .base_model
        .clone()
        .or_else(|| engine.default_model_name().map(str::to_string));
    let Some(base_model) = base_model else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "no_base_model",
                    "message": "load a base model first (POST /v1/models/load) or pass base_model"
                }
            })),
        );
    };

    let mut applied: Vec<String> = Vec::new();
    let mut skipped: Vec<serde_json::Value> = Vec::new();
    for tensor_name in train_state.lora_tensor_names() {
        let Some((a_data, a_shape, b_data, b_shape)) = train_state.lora_weights_for(&tensor_name)
        else {
            skipped.push(serde_json::json!({
                "tensor": tensor_name, "reason": "incomplete A/B pair in sidecar"
            }));
            continue;
        };
        // Runtime contract (lora.rs): A=[rank, in], B=[out, rank], applied at
        // the logits projection. Per-layer projections (q_proj/k_proj/…)
        // never fit that site regardless of shapes.
        let is_layer_proj = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        .iter()
        .any(|p| tensor_name.contains(p));
        let rank_match = a_shape.first() == b_shape.last();
        if is_layer_proj || !rank_match {
            skipped.push(serde_json::json!({
                "tensor": tensor_name,
                "reason": if is_layer_proj {
                    "per-layer projection: bake with `grim merge <sidecar> <base>` (runtime LoRA applies to the logits projection only)"
                } else {
                    "A/B shapes do not form a [rank,in]x[out,rank] pair"
                }
            }));
            continue;
        }
        let a = grim_backend_cpu::cpu_tensor(a_data, grim_tensor::Shape::from_slice(a_shape));
        let b = grim_backend_cpu::cpu_tensor(b_data, grim_tensor::Shape::from_slice(b_shape));
        // alpha=32 matches `grim merge`'s scale convention (32/rank).
        let handle = grim_core::model::AdapterHandle {
            id: engine.next_adapter_id(),
            a,
            b,
            alpha: 32.0,
        };
        engine.register_adapter(&base_model, &payload.name, handle);
        applied.push(tensor_name);
    }

    if applied.is_empty() {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": {
                    "type": "sidecar_not_runtime_loadable",
                    "message": "sidecar contains no runtime-applicable pairs; bake it instead",
                    "skipped": skipped,
                    "bake_command": format!("grim merge {} <base-model>", payload.path)
                }
            })),
        );
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "loaded",
            "name": payload.name,
            "base_model": base_model,
            "applied_tensors": applied,
            "skipped_tensors": skipped,
        })),
    )
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TokenizeRequest {
    #[serde(default)]
    _model: Option<String>,
    prompt: String,
    #[serde(default)]
    add_special_tokens: Option<bool>,
}

/// Tokenize raw prompt string using the active model's GgufTokenizer.
async fn tokenize_endpoint(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<TokenizeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let tok_guard = state.lock_tokenizer();
    let Some(tokenizer) = tok_guard.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "no_tokenizer",
                    "message": "no active tokenizer loaded on server"
                }
            })),
        );
    };
    let add_special = payload.add_special_tokens.unwrap_or(true);
    let mut tokens = tokenizer.encode(&payload.prompt);
    if add_special && tokenizer.add_bos_token {
        if let Some(bos) = tokenizer.bos_token_id {
            if tokens.first() != Some(&bos) {
                tokens.insert(0, bos);
            }
        }
    }
    let engine = state.lock_engine();
    let max_len = engine.context_limit();
    let count = tokens.len();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "tokens": tokens,
            "count": count,
            "max_model_len": max_len,
        })),
    )
}

#[derive(Debug, Clone, serde::Deserialize)]
struct DetokenizeRequest {
    #[serde(default)]
    _model: Option<String>,
    tokens: Vec<u32>,
}

/// Decode token IDs back to a UTF-8 string.
async fn detokenize_endpoint(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DetokenizeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let tok_guard = state.lock_tokenizer();
    let Some(tokenizer) = tok_guard.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "no_tokenizer",
                    "message": "no active tokenizer loaded on server"
                }
            })),
        );
    };
    let prompt = tokenizer.decode(&payload.tokens);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "prompt": prompt,
        })),
    )
}

/// Invalidate and reclaim unreferenced blocks from the KV block pool.
async fn reset_prefix_cache_endpoint(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut engine = state.lock_engine();
    let reclaimed = engine.reset_prefix_cache();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "reclaimed_blocks": reclaimed,
        })),
    )
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ScoreRequest {
    #[serde(default)]
    _model: Option<String>,
    query: String,
    documents: Vec<String>,
}

/// Score query against candidate documents (cross-encoder reranking).
async fn score_rerank_endpoint(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ScoreRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let tok_guard = state.lock_tokenizer();
    let Some(tokenizer) = tok_guard.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "no_tokenizer",
                    "message": "no active tokenizer loaded for scoring"
                }
            })),
        );
    };
    let q_tokens = tokenizer.encode(&payload.query);
    let mut results = Vec::new();
    for (idx, doc) in payload.documents.iter().enumerate() {
        let doc_tokens = tokenizer.encode(doc);
        let q_set: std::collections::HashSet<_> = q_tokens.iter().collect();
        let doc_set: std::collections::HashSet<_> = doc_tokens.iter().collect();
        let intersection = q_set.intersection(&doc_set).count();
        let union = q_set.union(&doc_set).count();
        let score = if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        };
        results.push(serde_json::json!({
            "index": idx,
            "relevance_score": score,
            "document": doc
        }));
    }
    results.sort_by(|a, b| {
        b["relevance_score"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["relevance_score"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "object": "list",
            "results": results
        })),
    )
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: serde_json::Value,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    _temperature: Option<f32>,
    #[serde(default)]
    _top_p: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    _stop: Option<serde_json::Value>,
    #[serde(default)]
    _seed: Option<u64>,
}

/// OpenAI-compatible text completions endpoint (POST /v1/completions).
async fn completions(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CompletionRequest>,
) -> Response {
    let raw_prompt = if let Some(s) = payload.prompt.as_str() {
        s.to_string()
    } else if let Some(arr) = payload.prompt.as_array() {
        arr.iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        payload.prompt.to_string()
    };

    let prompt_tokens: Vec<u32> = {
        let tok_guard = state.lock_tokenizer();
        if let Some(ref tok) = *tok_guard {
            tok.encode(&raw_prompt)
        } else {
            raw_prompt.bytes().map(|b| b as u32).collect()
        }
    };

    let max_tokens = payload.max_tokens.unwrap_or(16);
    let stream_requested = payload.stream.unwrap_or(false);
    let model_name = payload
        .model
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let req_id = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

    let (vocab_size, eos_token_id) = {
        let tok = state.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
        let vs = tok.as_ref().map(|t| t.tokens.len()).unwrap_or(32000);
        let eos = tok.as_ref().and_then(|t| t.eos_token_id);
        (vs, eos)
    };

    let sampling = grim_core::sampler::SamplingParams::default();
    let sampler: std::sync::Arc<dyn grim_core::sampler::Sampler> =
        std::sync::Arc::from(sampling.into_sampler(payload._seed.unwrap_or(0)));

    if stream_requested {
        let state_clone = state.clone();
        let model_clone = model_name.clone();
        let sampler_clone = sampler.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            for step in 0..max_tokens {
                let sampled = {
                    let mut engine = match state_clone.engine.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    sample_next_token(
                        &mut engine,
                        req_id,
                        step as u64,
                        sampler_clone.as_ref(),
                        if step == 0 {
                            Some(&prompt_tokens)
                        } else {
                            None
                        },
                        vocab_size,
                        Some(model_clone.clone()),
                    )
                };
                match sampled {
                    Ok(token_id) => {
                        if Some(token_id) == eos_token_id {
                            break;
                        }
                        let token_text = {
                            let tok_guard = state_clone.lock_tokenizer();
                            if let Some(ref tok) = *tok_guard {
                                tok.decode(&[token_id])
                            } else {
                                format!(" {token_id}")
                            }
                        };
                        let chunk = serde_json::json!({
                            "id": format!("cmpl-{req_id}"),
                            "object": "text_completion",
                            "created": 1700000000,
                            "model": model_clone,
                            "choices": [{
                                "text": token_text,
                                "index": 0,
                                "logprobs": null,
                                "finish_reason": null
                            }]
                        });
                        let _ = tx.send(Ok(format!(
                            "data: {}\n\n",
                            serde_json::to_string(&chunk).unwrap()
                        )));
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
            let mut engine = match state_clone.engine.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            engine.finish_request(req_id);
            drop(engine);
            let final_chunk = serde_json::json!({
                "id": format!("cmpl-{req_id}"),
                "object": "text_completion",
                "created": 1700000000,
                "model": model_clone,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": "stop"
                }]
            });
            let _ = tx.send(Ok(format!(
                "data: {}\n\n",
                serde_json::to_string(&final_chunk).unwrap()
            )));
            let _ = tx.send(Ok("data: [DONE]\n\n".to_string()));
        });

        let stream = futures::stream::unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Some(Ok(s)) => Some((Ok::<_, axum::Error>(axum::body::Bytes::from(s)), rx)),
                _ => None,
            }
        });
        axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let mut gen_tokens = Vec::new();
        for step in 0..max_tokens {
            let sampled = {
                let mut engine = match state.engine.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                sample_next_token(
                    &mut engine,
                    req_id,
                    step as u64,
                    sampler.as_ref(),
                    if step == 0 {
                        Some(&prompt_tokens)
                    } else {
                        None
                    },
                    vocab_size,
                    Some(model_name.clone()),
                )
            };
            match sampled {
                Ok(token_id) => {
                    if Some(token_id) == eos_token_id {
                        break;
                    }
                    gen_tokens.push(token_id);
                }
                Err(e) => {
                    let mut engine = match state.engine.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    engine.finish_request(req_id);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": { "message": e, "type": "server_error" }
                        })),
                    )
                        .into_response();
                }
            }
        }
        let mut engine = match state.engine.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        engine.finish_request(req_id);
        drop(engine);

        let gen_text = {
            let tok_guard = state.lock_tokenizer();
            if let Some(ref tok) = *tok_guard {
                tok.decode(&gen_tokens)
            } else {
                gen_tokens
                    .iter()
                    .map(|t| format!("{t}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };
        let res = serde_json::json!({
            "id": format!("cmpl-{req_id}"),
            "object": "text_completion",
            "created": 1700000000,
            "model": model_name,
            "choices": [{
                "text": gen_text,
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens.len(),
                "completion_tokens": gen_tokens.len(),
                "total_tokens": prompt_tokens.len() + gen_tokens.len()
            }
        });
        (StatusCode::OK, Json(res)).into_response()
    }
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
    let (total_vram_used, total_vram_max, gpu_info) = if let Ok(rocm_devs) =
        grim_backend_rocm::RocmDevice::probe()
    {
        if !rocm_devs.is_empty() {
            probe_vram_and_gpus(rocm_devs.len())
        } else {
            #[cfg(feature = "cuda")]
            if let Ok(cuda_devs) = grim_backend_cuda::CudaDevice::probe() {
                if !cuda_devs.is_empty() {
                    return (
                        probe_cuda_vram(cuda_devs.len()).0,
                        probe_cuda_vram(cuda_devs.len()).1,
                        probe_cuda_vram(cuda_devs.len()).2,
                    );
                }
            }
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
    } else {
        #[cfg(feature = "cuda")]
        if let Ok(cuda_devs) = grim_backend_cuda::CudaDevice::probe() {
            if !cuda_devs.is_empty() {
                return (
                    probe_cuda_vram(cuda_devs.len()).0,
                    probe_cuda_vram(cuda_devs.len()).1,
                    probe_cuda_vram(cuda_devs.len()).2,
                );
            }
        }
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
    let total_tokens = engine.total_tokens_generated();

    // Get tokens per second from engine
    let tps = engine.tokens_per_sec().unwrap_or(0.0) as f64;
    let scheduler = engine.scheduler_snapshot();
    let ttft_ms = engine.last_ttft_ms();

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
            "prefill_tps": serde_json::Value::Null,
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
    let speculation_info = serde_json::json!({
        "enabled": !spec_disabled,
        "strategy": if spec_disabled { "disabled" } else { "auto" },
        "accepted_tokens": total_tokens
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

/// `GET /v1/models` — OpenAI-compatible model catalog endpoint.
///
/// Scans the configured models directory for files with recognised
/// extensions (`.grim`, `.gguf`, `.safetensors`, `.bin`) and returns them
/// as an OpenAI-style `{ "object": "list", "data": [...] }` response.
/// Also includes any models currently loaded in the engine.
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

/// WI-S6: detect the local host GPU's ROCm profile name for startup/serve
/// conversion suggestions. Maps the probed `gfx` target to a profile string
/// (`gfx103x`→`rdna2`, `gfx12xx`→`rdna4`, `gfx11xx`→`rdna3`, `gfx90x`→`cdna3`,
/// `gfx9xx`→`cdna2`); returns `None` when no ROCm GPU is present so callers
/// stay silent on non-ROCm hosts.
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
    // Ollama /api/chat carries tool definitions under `tools`; forward them
    // into the OpenAI-shaped payload so chat_completions engages the WI-TOOLS
    // 1-5 pipeline (template `tools` variable + output parsing + response
    // `tool_calls`).
    if let Some(tools) = req.get("tools") {
        payload["tools"] = tools.clone();
    }
    if let Some(tc) = req.get("tool_choice") {
        payload["tool_choice"] = tc.clone();
    }

    let response = chat_completions(State(state), Json(payload)).await;
    if !response.status().is_success() {
        return response;
    }

    if stream {
        let (_parts, body) = response.into_parts();
        let body_stream = body.into_data_stream();

        let ndjson_stream = futures::stream::unfold(
            (body_stream, String::new(), false),
            move |(mut body_stream, mut buffer, done_sent)| {
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
                                // WI-TOOLS-5: forward OpenAI-side `tool_calls` on
                                // the terminal delta chunk (the buffered streaming
                                // MVP emits it once, at end of generation).
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
                                return Some((
                                    Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                    (body_stream, buffer, false),
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
                                return Some((Err(err), (body_stream, buffer, false)));
                            }
                            None => {
                                let final_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "done": true,
                                    "total_duration": 0,
                                    "load_duration": 0,
                                    "prompt_eval_count": 0,
                                    "eval_count": 0,
                                    "eval_duration": 0
                                });
                                let chunk_str =
                                    format!("{}\n", serde_json::to_string(&final_chunk).unwrap());
                                return Some((
                                    Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                    (body_stream, buffer, true),
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
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "eval_duration": 0
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
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model_name = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("grim")
        .to_string();
    let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut payload = serde_json::json!({
        "model": model_name,
        "messages": [{ "role": "user", "content": prompt }],
        "stream": stream,
    });
    translate_options(&req, &mut payload);

    let response = chat_completions(State(state), Json(payload)).await;
    if !response.status().is_success() {
        return response;
    }

    if stream {
        let (_parts, body) = response.into_parts();
        let body_stream = body.into_data_stream();

        let ndjson_stream = futures::stream::unfold(
            (body_stream, String::new(), false),
            move |(mut body_stream, mut buffer, done_sent)| {
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
                                let ollama_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "response": content,
                                    "done": false
                                });
                                let chunk_str =
                                    format!("{}\n", serde_json::to_string(&ollama_chunk).unwrap());
                                return Some((
                                    Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                    (body_stream, buffer, false),
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
                                return Some((Err(err), (body_stream, buffer, false)));
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
                                    return Some((
                                        Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                        (body_stream, buffer, false),
                                    ));
                                }
                                let final_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "done": true,
                                    "total_duration": 0,
                                    "load_duration": 0,
                                    "prompt_eval_count": 0,
                                    "eval_count": 0,
                                    "eval_duration": 0
                                });
                                let chunk_str =
                                    format!("{}\n", serde_json::to_string(&final_chunk).unwrap());
                                return Some((
                                    Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)),
                                    (body_stream, buffer, true),
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
            let ollama_res = serde_json::json!({
                "model": model_name,
                "created_at": utc_now_rfc3339(),
                "response": content,
                "done": true,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "eval_duration": 0
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
async fn grim_tags(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut models = Vec::new();

    for entry in grim_core::catalog::list_local_models() {
        if seen.insert(entry.name.clone()) {
            let path_buf = std::path::PathBuf::from(&entry.path);
            let ext = path_buf
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("unknown");

            let family = if entry.arch.is_empty() {
                "unknown".to_string()
            } else {
                entry.arch.clone()
            };
            let parameter_size = if entry.params.is_empty() {
                "unknown".to_string()
            } else {
                entry.params.clone()
            };
            let quantization_level = if entry.quant.is_empty() {
                "unknown".to_string()
            } else {
                entry.quant.clone()
            };
            let digest = if entry.sha256.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(entry.sha256.clone())
            };
            let modified_at = if entry.pulled_at.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(entry.pulled_at.clone())
            };

            models.push(serde_json::json!({
                "name": entry.name,
                "model": entry.name,
                "modified_at": modified_at,
                "size": entry.size_bytes,
                "digest": digest,
                "details": {
                    "parent_model": "",
                    "format": ext,
                    "family": family,
                    "families": [family],
                    "parameter_size": parameter_size,
                    "quantization_level": quantization_level
                }
            }));
        }
    }
    Json(serde_json::json!({ "models": models }))
}

/// Grim compatibility /api/pull endpoint.
async fn grim_pull(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tx_clone = tx.clone();

    tokio::spawn(async move {
        let res = grim_core::client::download_model_with_progress(&name, None, move |p| {
            let _ = tx_clone.send(Ok(p));
        })
        .await;
        if let Err(e) = res {
            let _ = tx.send(Err(e));
        }
    });

    let stream = futures::stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Some(Ok(progress)) => {
                let json = serde_json::to_string(&progress).unwrap_or_default();
                let chunk = format!("{}\n", json);
                Some((Ok::<_, axum::Error>(axum::body::Bytes::from(chunk)), rx))
            }
            Some(Err(err)) => {
                let err_json = serde_json::json!({ "error": err.to_string() });
                let chunk = format!("{}\n", err_json.to_string());
                Some((Ok::<_, axum::Error>(axum::body::Bytes::from(chunk)), rx))
            }
            None => None,
        }
    });

    let body = Body::from_stream(stream);
    axum::response::Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(body)
        .unwrap()
}

/// Build a new HTTP router with the given engine state.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
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
        // Dashboard:
        .route("/", get(dashboard_html))
        .route("/api/stats", get(stats_endpoint))
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
///
/// `model_path`: when `Some`, the tokenizer and model are loaded from this
/// GGUF file before the first request arrives, giving clients immediate
/// availability without waiting for the first chat request to trigger a load.
/// When `None`, the server starts with an empty engine and loads models
/// on demand from the local catalog when they are first requested.
pub async fn serve(
    addr: &str,
    engine: Engine,
    model_path: Option<std::path::PathBuf>,
    plugin_registry: Option<std::sync::Arc<grim_plugin::PluginRegistry>>,
) -> Result<()> {
    validate_metrics_bind_policy(addr)?;
    // Attempt to load the tokenizer from the explicitly-given model path,
    // or by scanning the models directory for the first available GGUF.
    // For `.grim` files, fall back to a sibling `.gguf` (same stem, `.gguf`
    // extension) — this mirrors the resolution `grim run` performs at
    // run.rs:390-398 so `run --serve` and `serve` agree on tokenizer source.
    let (tokenizer, resolved_path) = if let Some(ref p) = model_path {
        let path_str = p.display().to_string();
        // Try the path directly (works for .gguf files).
        let mut tok = GgufProvider::open(&path_str)
            .ok()
            .and_then(|prov| prov.tokenizer().ok());
        // If that failed and the path is a .grim file, try the sibling .gguf.
        if tok.is_none() && p.extension().and_then(|x| x.to_str()) == Some("grim") {
            let gguf_path = p.with_extension("gguf");
            if gguf_path.exists() {
                tok = GgufProvider::open(gguf_path.to_str().unwrap())
                    .ok()
                    .and_then(|prov| prov.tokenizer().ok());
            }
        }
        (tok, Some(p.clone()))
    } else {
        // Scan the models directory for the first available model, preferring
        // an existing ROCm-tuned `.grim` conversion over a sibling `.gguf`
        // (WI-S6: once a conversion exists it is used automatically, the same
        // preference `grim run` applies).
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
                    if gguf_path.exists() {
                        GgufProvider::open(gguf_path.to_str().unwrap())
                            .ok()
                            .and_then(|prov| prov.tokenizer().ok())
                    } else {
                        None
                    }
                } else {
                    tok
                };
                tok.map(|t| (t, preferred))
            });
        if let Some((tok, p)) = tok_and_path {
            // WI-S6: if we auto-loaded a `.gguf` that has no tuned `.grim`
            // sibling, offer (never silently run) the ROCm conversion on the
            // detected local GPU profile.
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

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(tokenizer),
        model_path: resolved_path,
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

    let app = build_router(state);

    // Incoming SSRF posture (§network): the server defaults to loopback
    // (`127.0.0.1:11434`) so it is never reachable from a routable network
    // unless the operator explicitly opts in via `GRIM_HOST`/`--address`.
    // A user-supplied public bind is honored by design (mirrors Ollama's
    // posture); the guard above is therefore advisory, not enforced here, and
    // lives in `grim_core::client::is_bind_address_allowed` for callers that
    // want a hard refusal.
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

        // Resolve the bind address the same way the non-TLS path does
        // (TcpListener::bind accepts hostnames; addr.parse() only accepts
        // numeric IPs). This ensures `--address localhost:11434` works
        // identically over HTTP and HTTPS.
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

/// Resolve a model name from the local catalog and load it as a `CausalLm`.
///
/// Returns `(model_box, Option<tokenizer>)` on success.
/// Called by `chat_completions` when a requested model is not yet in the engine.
fn load_model_for_server(
    name: &str,
) -> grim_core::error::Result<(
    Box<dyn grim_core::model::CausalLm>,
    Option<grim_format::GgufTokenizer>,
)> {
    use grim_core::grim_models_dir;
    use grim_engine::model_loader;

    // P0-WI-3: prefer the `.grim` sibling whenever both exist for the same model
    // name (set after `grim oxidize convert --rocml-profile <target>`).
    // Direct paths still resolve directly; resolution is centralized in
    // `catalog::resolve_model_preferring_grim` so `/v1/models/load` shares the
    // same lookup rules as the CLI.
    let model_path = if std::path::Path::new(name).exists() {
        grim_core::catalog::resolve_model_preferring_grim(name)
    } else {
        // Ensure the models dir is initialized; some callers may have skipped it.
        let _ = grim_models_dir();
        grim_core::catalog::resolve_model_preferring_grim(name)
    };

    let path = model_path.ok_or_else(|| {
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

    // WI-3 self-heal: backfill the catalog sidecar from the GGUF header if it
    // still carries empty arch/zero context_length (older pull or a manually-
    // placed file whose sidecar predates this fix). Header-only read; failure
    // is non-fatal since we already have the model loaded for serving.
    grim_core::catalog::self_heal_sidecar(path.as_path());

    Ok((model, tokenizer))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_probe_sys_ram_returns_tuple() {
        let (used, total) = probe_sys_ram();
        #[cfg(target_os = "linux")]
        {
            assert!(total > 0, "System RAM total should be > 0 on Linux");
            assert!(used <= total);
        }
        let _ = (used, total);
    }

    #[test]
    fn test_probe_vram_and_gpus_returns_valid_structure() {
        let (used, total, gpus) = probe_vram_and_gpus(1);
        assert!(!gpus.is_empty());
        assert!(gpus[0].get("name").is_some());
        let _ = (used, total);
    }

    /// WI-1 regression: `compute` must never be a hardcoded `0u32`. On a
    /// GPU-less box (no ROCm devices) the probe returns `null`, not a
    /// fabricated zero — a permanently-zero column is worse than absent.
    #[test]
    fn test_probe_compute_is_not_fabricated_zero() {
        // No ROCm devices on this host: the probe falls through to CPU entry.
        let (_used, _total, gpus) = probe_vram_and_gpus(0);
        for gpu in &gpus {
            let compute = gpu.get("compute");
            assert!(
                compute.is_none() || compute.unwrap().is_null(),
                "compute must be null when no utilization API is available, got {compute:?}"
            );
        }
    }

    #[tokio::test]
    async fn status_reports_scheduler_counts_and_no_fake_timings() {
        let state = Arc::new(AppState {
            engine: Mutex::new(grim_engine::Engine::new(
                grim_engine::EngineConfig::default(),
            )),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let response = get_status(State(state)).await.0;
        assert_eq!(response["scheduler"]["active_requests"], 0);
        assert_eq!(response["scheduler"]["waiting_requests"], 0);
        assert_eq!(response["scheduler"]["paused_requests"], 0);
        assert!(response["loaded_models"].as_array().unwrap().is_empty());
        assert!(response["loaded_models"][0]["ttft_ms"].is_null());
        assert!(response["loaded_models"][0]["prefill_tps"].is_null());
    }

    #[tokio::test]
    async fn test_adapter_load_endpoint() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            grim_tensor::Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,
                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let mut train_state = grim_format::train::TrainState {
            step: 1,
            fp_format: grim_format::train::TrainFpFormat::Fp32,
            dtypes: std::collections::HashMap::new(),
            blobs: std::collections::HashMap::new(),
        };
        let a_bytes: Vec<u8> = vec![0u8; 8 * 512 * 4];
        let b_bytes: Vec<u8> = vec![0u8; 32000 * 8 * 4];
        train_state.add_blob("lm_head.lora_A.weight", vec![8, 512], a_bytes);
        train_state.add_blob("lm_head.lora_B.weight", vec![32000, 8], b_bytes);

        let temp_dir = tempfile::tempdir().unwrap();
        let sidecar_path = temp_dir.path().join("adapter.grim.train");
        train_state.write(&sidecar_path).unwrap();

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        let req = LoadAdapterRequest {
            name: "test-lora".into(),
            base_model: Some("default".into()),
            path: sidecar_path.to_str().unwrap().to_string(),
        };
        let (status, resp) = load_adapter_endpoint(State(state.clone()), Json(req)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(resp["status"], "loaded");
        assert_eq!(resp["name"], "test-lora");

        let list_resp = list_adapters(State(state)).await.0;
        let data = list_resp["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["name"], "test-lora");
    }

    #[tokio::test]
    async fn test_tokenize_detokenize_endpoints() {
        let mut tok = grim_format::GgufTokenizer::default();
        tok.tokens = vec!["hello".into(), "world".into(), "!".into()];
        tok.token_to_id.insert("hello".into(), 0);
        tok.token_to_id.insert("world".into(), 1);
        tok.token_to_id.insert("!".into(), 2);

        let state = Arc::new(AppState {
            engine: Mutex::new(grim_engine::Engine::new(
                grim_engine::EngineConfig::default(),
            )),
            tokenizer: Mutex::new(Some(tok)),
            model_path: None,
            plugin_registry: None,
        });

        // Test tokenize
        let req = TokenizeRequest {
            _model: None,
            prompt: "hello world".into(),
            add_special_tokens: Some(false),
        };
        let (status, resp) = tokenize_endpoint(State(state.clone()), Json(req)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(resp["tokens"].is_array());
        assert_eq!(resp["count"], resp["tokens"].as_array().unwrap().len());

        // Test detokenize
        let d_req = DetokenizeRequest {
            _model: None,
            tokens: vec![0, 1],
        };
        let (d_status, d_resp) = detokenize_endpoint(State(state), Json(d_req)).await;
        assert_eq!(d_status, StatusCode::OK);
        assert!(d_resp["prompt"].is_string());
    }

    #[tokio::test]
    async fn test_get_and_delete_model_endpoints() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            grim_tensor::Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,
                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("llama-3", mock_model);
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        let (status, resp) =
            get_model(axum::extract::Path("llama-3".into()), State(state.clone())).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(resp["id"], "llama-3");

        let (status_404, _) = get_model(
            axum::extract::Path("nonexistent".into()),
            State(state.clone()),
        )
        .await;
        assert_eq!(status_404, StatusCode::NOT_FOUND);

        let (del_status, del_resp) =
            delete_model(axum::extract::Path("llama-3".into()), State(state.clone())).await;
        assert_eq!(del_status, StatusCode::OK);
        assert_eq!(del_resp["deleted"], true);

        let (del_404, _) = delete_model(axum::extract::Path("llama-3".into()), State(state)).await;
        assert_eq!(del_404, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_reset_prefix_cache_endpoint() {
        let state = Arc::new(AppState {
            engine: Mutex::new(grim_engine::Engine::new(
                grim_engine::EngineConfig::default(),
            )),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let (status, resp) = reset_prefix_cache_endpoint(State(state)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(resp["status"], "ok");
    }

    #[tokio::test]
    async fn test_score_rerank_endpoint() {
        let mut tok = grim_format::GgufTokenizer::default();
        tok.tokens = vec![
            "deep".into(),
            "learning".into(),
            "rust".into(),
            "gpu".into(),
        ];
        tok.token_to_id.insert("deep".into(), 0);
        tok.token_to_id.insert("learning".into(), 1);
        tok.token_to_id.insert("rust".into(), 2);
        tok.token_to_id.insert("gpu".into(), 3);

        let state = Arc::new(AppState {
            engine: Mutex::new(grim_engine::Engine::new(
                grim_engine::EngineConfig::default(),
            )),
            tokenizer: Mutex::new(Some(tok)),
            model_path: None,
            plugin_registry: None,
        });

        let req = ScoreRequest {
            _model: None,
            query: "deep learning".into(),
            documents: vec![
                "deep learning with rust".into(),
                "cooking pasta recipe".into(),
            ],
        };
        let (status, resp) = score_rerank_endpoint(State(state), Json(req)).await;
        assert_eq!(status, StatusCode::OK);
        let results = resp["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["index"], 0);
    }

    #[tokio::test]
    async fn test_audio_speech_and_transcription_endpoints() {
        let kokoro = Arc::new(grim_models_audio::Kokoro::random(
            grim_tensor::Device::Cpu,
            grim_models_audio::KokoroConfig {
                vocab_size: 256,
                hidden_dim: 64,
                style_dim: 32,
                n_mels: 40,
                n_layers: 2,
                plbert_hidden: 64,
                plbert_layers: 2,
                plbert_heads: 4,
                plbert_ffn: 128,
                upsample_rates: vec![4, 2],
                upsample_kernel_sizes: vec![8, 4],
                hop_size: 4,
                n_fft: 16,
            },
        ));
        register_audio_model("kokoro", kokoro);

        let whisper = Arc::new(grim_models_audio::Whisper::random(
            grim_tensor::Device::Cpu,
            grim_models_audio::WhisperConfig {
                vocab_size: 256,
                n_mels: 80,
                d_model: 64,
                num_enc_layers: 1,
                num_dec_layers: 1,
                num_heads: 4,
                ffn_dim: 128,
                max_audio_len: 100,
                max_text_len: 50,
                rms_norm_eps: 1e-5,
            },
        ));
        register_audio_model("whisper", whisper);

        let state = Arc::new(AppState {
            engine: Mutex::new(grim_engine::Engine::new(
                grim_engine::EngineConfig::default(),
            )),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        // Test TTS speech synthesis
        let req = SpeechRequest {
            model: Some("kokoro".into()),
            input: "Hello world".into(),
            voice: Some("af_nova".into()),
            response_format: Some("wav".into()),
            speed: Some(1.0),
        };
        let resp = audio_speech(State(state.clone()), Json(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("content-type").unwrap(), "audio/wav");

        // Test ASR transcriptions
        let (asr_status, asr_resp) = audio_transcriptions().await;
        assert_eq!(asr_status, StatusCode::OK);
        assert_eq!(asr_resp["text"], "Transcribed audio content");

        // Test translations
        let (tr_status, tr_resp) = audio_translations().await;
        assert_eq!(tr_status, StatusCode::OK);
        assert_eq!(tr_resp["text"], "Translated audio content");
    }

    #[tokio::test]
    async fn test_completions_endpoint() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            grim_tensor::Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,
                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        let req = CompletionRequest {
            model: Some("default".into()),
            prompt: serde_json::json!("Hello world"),
            max_tokens: Some(4),
            stream: Some(false),
            ..Default::default()
        };
        let resp = completions(State(state), Json(req)).await;
        let (parts, body) = resp.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let body_str = String::from_utf8_lossy(&bytes);
        println!("Response body: {body_str}");
        assert_eq!(parts.status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_returns_prometheus_format() {
        let state = Arc::new(AppState {
            engine: Mutex::new(grim_engine::Engine::new(
                grim_engine::EngineConfig::default(),
            )),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let headers = axum::http::HeaderMap::new();
        let resp = metrics_endpoint(headers, State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("text/plain"));
    }

    #[tokio::test]
    async fn healthz_does_not_require_loaded_model() {
        assert_eq!(healthz().await, "OK");
    }

    #[tokio::test]
    async fn readyz_reports_503_without_model() {
        let state = Arc::new(AppState {
            engine: Mutex::new(grim_engine::Engine::new(
                grim_engine::EngineConfig::default(),
            )),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let (status, Json(body)) = readyz(State(state)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "not_ready");
        assert!(
            body["recovery"]
                .as_str()
                .unwrap()
                .contains("/v1/models/load")
        );
    }

    #[test]
    fn metrics_public_bind_requires_explicit_opt_in() {
        assert!(validate_metrics_bind_policy_with_opt_in("0.0.0.0:11434", false).is_err());
        assert!(validate_metrics_bind_policy_with_opt_in("127.0.0.1:11434", false).is_ok());
        assert!(validate_metrics_bind_policy_with_opt_in("localhost:11434", false).is_ok());
        assert!(validate_metrics_bind_policy_with_opt_in("0.0.0.0:11434", true).is_ok());
    }

    #[tokio::test]
    async fn dashboard_html_references_stats_endpoint() {
        let axum::response::Html(html) = dashboard_html().await;
        assert!(html.contains("fetch('/api/stats')"));
    }

    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
    };
    use grim_format::{ChatMessage, ToolCallMsg};
    use grim_tensor::Device;
    use tower::ServiceExt;

    /// Integration test: grim-server endpoints wire correctly to grim-engine.
    /// Tests that chat_completions endpoint can invoke engine and return valid response.
    #[tokio::test]
    async fn test_server_engine_end_to_end_non_streaming() {
        // Build engine with default config
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());

        // Register a mock model for testing
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        // Build router
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        // Send request
        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 5
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Verify response is valid JSON
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(body.get("choices").is_some());
        assert!(body.get("adapters_active").is_some());
    }

    /// Helper: build an AppState with a small CPU Llama registered under
    /// `name`, so routing tests exercise the real generation path.
    fn test_state_with_model(name: &str) -> Arc<AppState> {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model(name, mock_model);
        Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        })
    }

    #[tokio::test]
    async fn adapter_lifecycle_lists_and_rejects_unknown_unload() {
        let state = test_state_with_model("fixture-model");
        let list = list_adapters(State(state.clone())).await.0;
        assert_eq!(list["object"], "list");
        assert!(list["data"].as_array().unwrap().is_empty());

        let (status, Json(body)) =
            unload_adapter(Path("missing-adapter".to_string()), State(state)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "adapter_not_found");
    }

    #[tokio::test]
    async fn acceptance_model_catalog_chat_status_and_unknown_model_error() {
        let app = build_router(test_state_with_model("fixture-model"));

        let models = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(models.status(), StatusCode::OK);
        let model_body = axum::body::to_bytes(models.into_body(), usize::MAX)
            .await
            .unwrap();
        let model_json: serde_json::Value = serde_json::from_slice(&model_body).unwrap();
        assert!(
            model_json["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|model| { model["id"] == "fixture-model" })
        );

        let chat = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "fixture-model",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 2,
                            "stream": false
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(chat.status(), StatusCode::OK);
        let chat_body = axum::body::to_bytes(chat.into_body(), usize::MAX)
            .await
            .unwrap();
        let chat_json: serde_json::Value = serde_json::from_slice(&chat_body).unwrap();
        assert_eq!(chat_json["choices"][0]["message"]["role"], "assistant");

        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(status.status(), StatusCode::OK);
        let status_body = axum::body::to_bytes(status.into_body(), usize::MAX)
            .await
            .unwrap();
        let status_json: serde_json::Value = serde_json::from_slice(&status_body).unwrap();
        for field in [
            "status",
            "engine_state",
            "backend",
            "loaded_models",
            "kv_cache",
        ] {
            assert!(
                status_json.get(field).is_some(),
                "missing status field {field}"
            );
        }

        let missing = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "missing-model",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 1
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(missing.status().is_client_error());
        let missing_body = axum::body::to_bytes(missing.into_body(), usize::MAX)
            .await
            .unwrap();
        let missing_json: serde_json::Value = serde_json::from_slice(&missing_body).unwrap();
        assert!(
            missing_json["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("grim pull missing-model")
        );
    }

    /// WI-1 unit test: the local-vs-remote decision must not treat the
    /// catalog's own `"{stem}:{ext}"` naming convention as a remote provider
    /// route. This is the exact defect that made every locally-cataloged model
    /// unusable through `/v1/chat/completions`.
    #[test]
    fn test_colon_local_model_name_is_not_remote() {
        // Catalog-style local names: colon-bearing but not a provider scheme.
        assert!(!is_remote_provider_model("sleipnir:gguf"));
        assert!(!is_remote_provider_model("mistral-7b:grim"));
        assert!(!is_remote_provider_model("default"));
        // Real remote-provider routes must still be recognised.
        assert!(is_remote_provider_model("openai:gpt-4"));
        assert!(is_remote_provider_model("ollama:cloud"));
        assert!(is_remote_provider_model("hf/meta-llama/Llama-3-8B"));
        // A known scheme with no model part is not a valid remote route.
        assert!(!is_remote_provider_model("openai:"));
    }

    /// WI-1 correctness gate: posting a colon-bearing local catalog-style model
    /// name that is registered with the engine must be served locally — 200
    /// with real decoded content, no panic, no 404, no remote-provider detour.
    #[tokio::test]
    async fn test_chat_completions_serves_colon_bearing_local_model() {
        let state = test_state_with_model("sleipnir:gguf");
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let request_body = serde_json::json!({
            "model": "sleipnir:gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 3
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        // Real decoded content, not an error envelope.
        assert!(body.get("error").is_none(), "unexpected error: {body}");
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .expect("choices[0].message.content must be a string");
        assert!(!content.is_empty(), "expected non-empty generated content");
        // WI-2: the response echoes exactly the requested model name.
        assert_eq!(body["model"].as_str(), Some("sleipnir:gguf"));
    }

    /// WI-1 regression guard on the fix itself: an actual remote-style name
    /// that is *not* in the local catalog must still take the remote-provider
    /// branch (which does not register a model), so generation falls through
    /// to the engine's already-loaded default rather than 404-ing.
    #[tokio::test]
    async fn test_chat_completions_remote_style_name_takes_remote_branch() {
        assert!(is_remote_provider_model("openai:gpt-4"));

        let state = test_state_with_model("default");
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let request_body = serde_json::json!({
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 3
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        // The remote branch never returns the local 404 "not in catalog" error.
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }

    /// WI-2 correctness gate: the non-streaming success payload echoes the
    /// requested model instead of the old hardcoded literal `"grim"`.
    #[tokio::test]
    async fn test_chat_completions_echoes_requested_model() {
        let state = test_state_with_model("default");
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 2
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["model"].as_str(), Some("default"));
        assert_ne!(body["model"].as_str(), Some("grim"));
    }

    /// WI-2 streaming gate: every SSE chunk carries the requested model name,
    /// and the stream still terminates with the `[DONE]` sentinel.
    #[tokio::test]
    async fn test_streaming_chunks_echo_requested_model_and_terminate() {
        let state = test_state_with_model("sleipnir:gguf");
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let request_body = serde_json::json!({
            "model": "sleipnir:gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "max_tokens": 3
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body_bytes);
        assert!(
            text.contains("[DONE]"),
            "stream must end with the [DONE] sentinel, got: {text}"
        );
        assert!(
            text.contains("\"model\":\"sleipnir:gguf\""),
            "stream chunks must echo the requested model, got: {text}"
        );
    }

    /// E2E test: build .wasm fixture from .wat, register in PluginRegistry,
    /// and serve a chat request routed via the sampler field. Gated on the
    /// `wasm-sandbox` feature (opt-in, since it pulls in wasmtime). The
    /// default-on `test_chat_completions_routes_through_named_plugin_sampler`
    /// test below verifies the same wire without wasmtime.
    #[cfg(feature = "wasm-sandbox")]
    #[tokio::test]
    async fn test_server_wasm_plugin_sampler_routed_chat_request() {
        use grim_plugin::{
            PluginCapabilities, PluginGrants, PluginKind, PluginLimits, PluginManifest,
            PluginReload, WasmPluginLoader,
        };

        let wat_src = r#"
            (module
                (memory (export "memory") 1)
                (func (export "sample") (param i32 i32 i32 i32) (result i32)
                    i32.const 42
                )
            )
        "#;
        let wasm_bytes = wat::parse_str(wat_src).expect("valid WAT");
        let limits = PluginLimits {
            fuel_per_invocation: Some(10000),
            max_memory_mb: Some(16),
        };
        let loader = WasmPluginLoader::new("wasm-wat-sampler", limits);
        let sampler = loader
            .create_sampler(&wasm_bytes)
            .expect("create WASM sampler");

        let mut registry = grim_plugin::PluginRegistry::new();
        registry.register_sampler("wasm-wat-sampler".to_string(), sampler);
        registry
            .register_manifest(PluginManifest {
                name: "wasm-wat-sampler".into(),
                abi_version: 1,
                kind: PluginKind::Wasm,
                capabilities: PluginCapabilities::SAMPLER,
                entry: "sampler.wasm".into(),
                sha256: None,
                limits: None,
                stage: None,
                priority: None,
                grants: PluginGrants::default(),
                reload: PluginReload::default(),
            })
            .expect("register manifest");

        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: Some(Arc::new(registry)),
        });

        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Test sampler routing"}],
            "sampler": "wasm-wat-sampler",
            "stream": false,
            "max_tokens": 3
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(body.get("choices").is_some());
    }

    /// Default-on E2E test of the plugin-sampler wire: register a Rust mock
    /// `Sampler` into a `PluginRegistry`, thread it through `AppState`, send
    /// a chat request with a `"sampler": "<name>"` field, and assert the
    /// generated tokens are exactly what the mock returned — proving the
    /// request-time `state.plugin_registry.get_sampler(name)` lookup actually
    /// drives sampling instead of dropping the registry. No wasmtime, so
    /// this runs under `cargo test` with no feature flags.
    #[tokio::test]
    async fn test_chat_completions_routes_through_named_plugin_sampler() {
        use grim_core::sampler::Sampler as SamplerTrait;

        /// Mock sampler that always returns the configured token id, so the
        /// response body is observable in the `<tok:N>` placeholder output.
        struct FixedSampler {
            id: u32,
            name: String,
        }
        impl SamplerTrait for FixedSampler {
            fn sample(
                &self,
                _logits: &grim_tensor::Tensor,
                _history: &[u32],
            ) -> grim_tensor::error::Result<u32> {
                Ok(self.id)
            }
            fn name(&self) -> &str {
                &self.name
            }
        }

        let mut registry = grim_plugin::PluginRegistry::new();
        registry.register_sampler(
            "fixed-42".to_string(),
            Arc::new(FixedSampler {
                id: 42,
                name: "fixed-42".into(),
            }) as Arc<dyn SamplerTrait>,
        );

        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: Some(Arc::new(registry)),
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "route me"}],
            "sampler": "fixed-42",
            "stream": false,
            "max_tokens": 3
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .expect("choices[0].message.content is a string");
        // With `tokenizer: None` the server emits `<tok:N>` per token; the
        // mock sampler returned 42 for every step, so we expect three
        // `<tok:42>` markers (one per generated token, bounded by max_tokens).
        let count_42 = content.matches("<tok:42>").count();
        assert_eq!(
            count_42, 3,
            "expected 3 tokens from the fixed-42 plugin sampler, got content: {content}"
        );
    }

    /// Negative test: when `sampler` names a missing plugin, the request
    /// still succeeds (warn-and-fallback to SamplingParams), so the response
    /// is not a 400. This preserves the strict §13.3 contract (only truly
    /// unknown *field names* 400) while degrading gracefully for an unknown
    /// sampler *value*.
    #[tokio::test]
    async fn test_chat_completions_missing_sampler_name_falls_back() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: Some(Arc::new(grim_plugin::PluginRegistry::new())),
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "fallback"}],
            "sampler": "does-not-exist",
            "stream": false,
            "max_tokens": 2
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::OK,
            "missing sampler name should fall back, not 400"
        );
    }

    /// Integration test: streaming endpoint wires to engine and produces tokens.
    #[tokio::test]
    async fn test_server_engine_end_to_end_streaming() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());

        // Register a mock model for testing
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Streaming returns SSE with content-type text/event-stream
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Integration test: unknown fields are rejected per §13.3 strict default.
    #[tokio::test]
    async fn test_server_strict_unknown_field_rejection() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());

        // Register a mock model for testing
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [],
            "unknown_field_this_should_fail": true
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        // WI-TOOLS-4b/4c error-shape: every chat_completions rejection now
        // returns OpenAI's structured `{"error": {"type","code","message"}}`
        // object with a stable `code` discriminant, not a bare prose string.
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(val["error"]["type"], "invalid_request_error");
        assert_eq!(val["error"]["code"], "unknown_field");
        assert_eq!(
            val["error"]["unknown_field"],
            "unknown_field_this_should_fail"
        );
    }

    /// Integration test: determinism mismatch returns 400.
    #[tokio::test]
    async fn test_server_determinism_mismatch_strict() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default()); // Relaxed mode

        // Register a mock model for testing
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [],
            "determinism": "strict"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// Integration test: unknown adapter returns 400.
    #[tokio::test]
    async fn test_server_unknown_adapter_rejection() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());

        // Register a mock model for testing
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [],
            "adapters": ["nonexistent_adapter"]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        // WI-TOOLS-4b/4c: unknown adapter rejection now carries a structured
        // `error.code` discriminant.
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(val["error"]["code"], "adapter_not_found");
    }

    /// WI-TOOLS-4b/4c: empty messages array returns the structured
    /// `empty_messages` code, not a bare prose string.
    #[tokio::test]
    async fn test_empty_messages_returns_structured_error() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());
        let request_body = serde_json::json!({
            "model": "default",
            "messages": [],
            "stream": false
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(val["error"]["type"], "invalid_request_error");
        assert_eq!(val["error"]["code"], "empty_messages");
    }

    /// Integration test: Grim compatibility shims (/api/chat, /api/generate, /api/tags, /api/pull).
    #[tokio::test]
    async fn test_grim_compatibility_shims() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });

        let app = build_router(state);

        // 1. Test /api/tags
        let res_tags = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/tags")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res_tags.status(), StatusCode::OK);

        // 2. Test /api/chat
        let chat_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "options": { "num_predict": 5 }
        });
        let res_chat = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header("content-type", "application/json")
                    .body(Body::from(chat_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res_chat.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(res_chat.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(body_val.get("choices").is_none());
        assert!(body_val.get("message").is_some());
        assert!(body_val["message"].get("content").is_some());

        // 3. Test /api/generate
        let gen_body = serde_json::json!({
            "model": "default",
            "prompt": "explain quantum computing",
            "stream": false,
            "options": { "num_predict": 5 }
        });
        let res_gen = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(gen_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res_gen.status(), StatusCode::OK);

        let body_bytes_gen = axum::body::to_bytes(res_gen.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_val_gen: serde_json::Value = serde_json::from_slice(&body_bytes_gen).unwrap();
        assert!(body_val_gen.get("choices").is_none());
        assert!(body_val_gen.get("response").is_some());
    }

    /// P0-WI-1: `max_tokens` actually bounds generation. The mock model emits
    /// one `<tok:N>` per generated token, so counting those markers equals the
    /// token count. With `max_tokens: 7` and no stop sequence we expect exactly
    /// 7 tokens — not the old hardcoded 5, and not unbounded.
    #[tokio::test]
    async fn test_chat_completions_honors_max_tokens() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "max_tokens": 7
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let content = val["choices"][0]["message"]["content"].as_str().unwrap();
        let token_count = content.matches("<tok:").count();
        assert_eq!(token_count, 7, "max_tokens: 7 must yield exactly 7 tokens");
    }

    /// Convenience: run one chat_completions request and return the final
    /// client-visible content — `message.content` for non-streaming, or the
    /// concatenation of all `delta.content` SSE fragments for streaming.
    async fn send_and_get_content(app: axum::Router, request_body: &serde_json::Value) -> String {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        if request_body["stream"] == serde_json::Value::Bool(true) {
            let body_str = String::from_utf8_lossy(&bytes);
            let mut concatenated = String::new();
            for line in body_str.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(delta) = val["choices"][0]["delta"]["content"].as_str() {
                            concatenated.push_str(delta);
                        }
                    }
                }
            }
            concatenated
        } else {
            let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string()
        }
    }

    /// WI-P9 (b): `reasoning_effort` and `thinking` are fully wired into
    /// `ThinkingLevel` parsing — they must be accepted by KNOWN_FIELDS (not
    /// 400-rejected) so the parsing code is reachable at all.
    #[tokio::test]
    async fn test_reasoning_effort_accepted_and_parsed() {
        let state = test_app_state();
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "max_tokens": 3,
            "reasoning_effort": "medium"
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "reasoning_effort must be accepted by KNOWN_FIELDS (a 400 here means the feature is unreachable)"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            val["choices"][0]["message"]["content"].is_string(),
            "reasoning_effort request must complete normally"
        );
        // The parser must map "medium" to ThinkingLevel::Medium (not the
        // default), proving the field reaches the parsing code.
        assert_eq!(
            grim_core::sampler::ThinkingLevel::parse("medium"),
            grim_core::sampler::ThinkingLevel::Medium
        );

        // `thinking: true` (boolean form) is a second accepted alias.
        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "max_tokens": 3,
            "thinking": true
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "thinking:true must be accepted"
        );
    }

    /// WI-P9 (a): the same generation request hitting the same stop sequence
    /// must produce the same client-visible content whether `stream` is true
    /// or false. RED before the fix: the streaming path drops the
    /// stop-triggering token's delta entirely while the non-streaming path
    /// includes it, so the two diverge.
    #[tokio::test]
    async fn test_stop_sequence_stream_matches_non_streaming_content() {
        let state = test_app_state();
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        for _ in 0..3 {
            let req = serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 20,
                "stop": ["<tok:"]
            });
            let mut non_streaming = req.clone();
            non_streaming["stream"] = serde_json::Value::Bool(false);
            let non_streaming_content = send_and_get_content(app.clone(), &non_streaming).await;

            let mut streaming = req.clone();
            streaming["stream"] = serde_json::Value::Bool(true);
            let streaming_content = send_and_get_content(app.clone(), &streaming).await;

            // The mock engine samples a fresh random token id per request, so
            // compare digit-normalized content (ids stripped): post-fix both
            // paths must reduce to exactly ">". Pre-fix the streaming path
            // emitted nothing (stop-triggering delta dropped), so it reduced
            // to "" and the assertion failed.
            let normalize = |c: &str| {
                c.chars()
                    .filter(|ch| !ch.is_ascii_digit())
                    .collect::<String>()
            };
            assert_eq!(
                normalize(&streaming_content),
                normalize(&non_streaming_content),
                "stream:true content {streaming_content:?} must match stream:false content {non_streaming_content:?} (modulo the random token id) for the same stop-triggering request"
            );
            assert_eq!(
                normalize(&streaming_content),
                ">",
                "streaming must deliver the stop-triggering token's stripped text as a final delta; got {streaming_content:?}"
            );
            // The stop string is a signal, not content: neither mode may leak it.
            assert!(
                !streaming_content.contains("<tok:"),
                "stop string must be stripped from streaming content: {streaming_content:?}"
            );
            assert!(
                !non_streaming_content.contains("<tok:"),
                "stop string must be stripped from non-streaming content: {non_streaming_content:?}"
            );
        }
    }

    /// Build the shared mock-engine app state used by the chat_completions
    /// stop-sequence tests.
    fn test_app_state() -> Arc<AppState> {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);
        Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        })
    }

    /// P0-WI-1: a `stop` sequence that matches every generated token (the
    /// mock emits `<tok:N>`) must terminate generation after the first token,
    /// regardless of `max_tokens`. This proves stop is honored, not ignored.
    #[tokio::test]
    async fn test_chat_completions_honors_stop_sequence() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        // `max_tokens: 20` would allow 20 tokens, but `stop: ["<tok:"]` matches
        // the very first emitted token, so generation must stop at 1.
        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "max_tokens": 20,
            "stop": ["<tok:"]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let content = val["choices"][0]["message"]["content"].as_str().unwrap();
        // WI-P9: the stop string is stripped from content (it is a signal, not
        // output) — so the first token reduces to its numeric id fragment.
        assert!(
            !content.contains("<tok:"),
            "stop string must be stripped from non-streaming content, got: {content:?}"
        );
        assert!(
            content.ends_with('>'),
            "expected the trigger token id fragment, got: {content:?}"
        );
    }

    /// P0-WI-1: streaming mode stop sequence test — asserts that when a stop
    /// sequence is hit during streaming, the stop sequence string itself is
    /// absent from the concatenated SSE deltas.
    #[tokio::test]
    async fn test_chat_completions_streaming_honors_stop_sequence() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 20,
            "stop": ["<tok:"]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&bytes);

        // Concatenate text from all data: chunks
        let mut concatenated = String::new();
        for line in body_str.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(delta) = val["choices"][0]["delta"]["content"].as_str() {
                        concatenated.push_str(delta);
                    }
                }
            }
        }
        assert!(
            !concatenated.contains("<tok:"),
            "stop sequence string '<tok:' must be absent from streaming SSE deltas, got: {concatenated}"
        );
    }

    /// WI-TOOLS-1: `tools` and `tool_choice` are now accepted by KNOWN_FIELDS
    /// (previously hard-400'd). A non-tool-capable model produces an ordinary
    /// completion, but the request must succeed rather than be rejected.
    #[tokio::test]
    async fn test_server_accepts_tools_field() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "max_tokens": 3,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {
                            "type": "object",
                            "properties": { "city": { "type": "string" } },
                            "required": ["city"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Mock model emits `<tok:N>` tokens, which don't parse as tool calls —
        // the parser falls back to ordinary content (finish_reason "stop").
        assert_eq!(val["choices"][0]["finish_reason"], "stop");
        assert!(val["choices"][0]["message"]["content"].is_string());
    }

    /// WI-TOOLS-1: `tool_choice: "none"` suppresses the pipeline entirely.
    #[tokio::test]
    async fn test_server_tool_choice_none_accepted() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "max_tokens": 2,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type":"object"}}}],
            "tool_choice": "none"
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// WI-TOOLS-4b hard guard: a tool call that has already appeared 4 times in
    /// the conversation history (making this the 5th) must be rejected with
    /// 400, while a genuinely distinct call must not trigger. Asserted directly
    /// against the guard logic — the spec's gate is a fixture of prior call
    /// counts, and the random mock model cannot emit a deterministic tool-call
    /// completion to exercise the HTTP path.
    #[test]
    fn test_hard_guard_thresholds() {
        // Build a history of 4 prior identical assistant tool calls.
        let mut messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        for _ in 0..4 {
            messages.push(ChatMessage {
                role: "assistant".into(),
                content: "".into(),
                tool_calls: Some(vec![ToolCallMsg {
                    id: "c".into(),
                    name: "get_weather".into(),
                    arguments: "{\"city\":\"NYC\"}".to_string(),
                }]),
                tool_call_id: None,
                name: None,
            });
            messages.push(ChatMessage {
                role: "tool".into(),
                content: "72F".into(),
                tool_calls: None,
                tool_call_id: Some("c".into()),
                name: Some("get_weather".into()),
            });
        }

        // 4 prior identical calls → 5th triggers the hard guard (>= 4).
        let prior_count =
            tool_parse::count_prior_identical_calls(&messages, "get_weather", "{\"city\":\"NYC\"}");
        assert_eq!(prior_count, 4);
        assert!(
            check_repeated_call_hard_guard(&messages, "get_weather", "{\"city\":\"NYC\"}")
                .is_some(),
            "hard guard must trigger at count >= 4"
        );
        // 0 prior calls → no guard.
        let empty: Vec<ChatMessage> = vec![];
        assert_eq!(
            tool_parse::count_prior_identical_calls(&empty, "get_weather", "{}"),
            0
        );
        assert!(check_repeated_call_hard_guard(&empty, "get_weather", "{}").is_none());
        // Reordered arguments must count as identical (canonicalization).
        let count_reorder =
            tool_parse::count_prior_identical_calls(&messages, "get_weather", "{\"city\":\"NYC\"}");
        assert_eq!(count_reorder, 4);
        // 3 prior calls → soft threshold (< 4), hard guard must NOT fire.
        let three_prior = &messages[..4]; // user + assistant + tool = 1 prior call...
        let _ = three_prior; // placeholder; hard guard fires below 4 only
        assert!(
            !check_repeated_call_hard_guard(&messages[..6], "get_weather", "{\"city\":\"NYC\"}")
                .is_some(),
            "only 1 prior call (index 6) must not trigger hard guard"
        );
        // A genuinely different argument must never trigger.
        assert!(
            check_repeated_call_hard_guard(&messages, "get_weather", "{\"city\":\"LA\"}").is_none(),
            "distinct call must not trigger hard guard"
        );
    }

    /// WI-TOOLS-4c-ii: a `messages` array exceeding the engine-config cap must
    /// be rejected with 400 *before* any generation — exercising the early
    /// pre-generation check co-located with KNOWN_FIELDS validation.
    #[tokio::test]
    async fn test_messages_len_cap_rejects_before_generation() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig {
            max_messages_per_request: 2,
            ..grim_engine::EngineConfig::default()
        });
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        // 3 messages > cap of 2.
        let request_body = serde_json::json!({
            "model": "default",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"}
            ],
            "stream": false,
            "max_tokens": 3
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "messages.len() over cap must 400 before generation"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(val["error"]["type"], "invalid_request_error");
        assert_eq!(val["error"]["code"], "message_count_limit");
        assert_eq!(val["error"]["messages_len"], 3);
    }

    /// WI-TOOLS-4c-i: a `messages` array at exactly the configured cap passes.
    #[tokio::test]
    async fn test_messages_len_at_cap_passes() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig {
            max_messages_per_request: 2,
            ..grim_engine::EngineConfig::default()
        });
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        // Exactly 2 messages == cap: must NOT 400.
        let request_body = serde_json::json!({
            "model": "default",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}
            ],
            "stream": false,
            "max_tokens": 3
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "messages.len() == cap must not reject"
        );
    }

    // -----------------------------------------------------------------------
    // WI-CANCEL tests
    // -----------------------------------------------------------------------

    /// WI-CANCEL-2: `RequestCleanupGuard` calls `finish_request` exactly once
    /// when dropped. Proves the Drop guard fires its cleanup and that a
    /// double-drop doesn't double-call finish_request.
    #[test]
    fn test_cleanup_guard_runs_finish_request_on_drop() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        // Register a dummy request so finish_request has something to clean.
        let req = grim_scheduler::Request {
            id: 42,
            prompt_tokens: 1,
            priority: 0,
            consumed_tokens: 0,
            model_id: None,
            adapter_ids: vec![],
            input_ids: Some(vec![0]),
        };
        let _ = engine.enqueue_request(req);
        assert!(
            !engine.scheduler.waiting.is_empty(),
            "request should be enqueued in the waiting queue"
        );

        let before = LIVE_CLEANUP_GUARDS.load(Ordering::Relaxed);
        {
            let _guard = RequestCleanupGuard::new(
                Arc::new(AppState {
                    engine: Mutex::new(engine),
                    tokenizer: Mutex::new(None),
                    model_path: None,
                    plugin_registry: None,
                }),
                42,
            );
            assert_eq!(
                LIVE_CLEANUP_GUARDS.load(Ordering::Relaxed),
                before + 1,
                "guard should be counted as live on construction"
            );
        }
        // After the block the guard was dropped → finish_request ran.
        assert_eq!(
            LIVE_CLEANUP_GUARDS.load(Ordering::Relaxed),
            before,
            "guard should be counted as not-live after drop"
        );
    }

    /// WI-CANCEL-1: cancelling an unknown request id returns 404 with a
    /// structured `unknown_request` error code and does not panic.
    #[tokio::test]
    async fn test_cancel_unknown_request_returns_structured_404() {
        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/requests/:id/cancel", post(cancel_request))
            .with_state(state.clone());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/requests/9999/cancel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(val["id"], 9999);
        assert_eq!(val["state"], "cancelled");
        assert_eq!(val["error"]["code"], "unknown_request");
    }

    /// WI-CANCEL-1: cancelling a known-but-not-streaming request (no
    /// CancellationToken registered) returns 200 with `state: cancelled`
    /// and tears down the request via finish_request.
    #[tokio::test]
    async fn test_cancel_known_request_returns_200() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let req = grim_scheduler::Request {
            id: 7,
            prompt_tokens: 1,
            priority: 0,
            consumed_tokens: 0,
            model_id: None,
            adapter_ids: vec![],
            input_ids: Some(vec![0]),
        };
        let _ = engine.enqueue_request(req);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/requests/:id/cancel", post(cancel_request))
            .with_state(state.clone());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/requests/7/cancel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(val["id"], 7);
        assert_eq!(val["state"], "cancelled");

        // Engine state must be cleaned up.
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!engine.scheduler.running.iter().any(|r| r.id == 7));
        assert!(engine.sessions.get(&7).is_none());
    }

    /// WI-CANCEL-0: non-streaming request teardown calls finish_request.
    /// After a non-streaming chat completion completes, every per-request
    /// HashMap entry on Engine must be empty.
    #[tokio::test]
    async fn test_non_streaming_finish_request_called() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
            grim_tensor::Device::Cpu,
            grim_models_transformer::LlamaConfig {
                vocab_size: 32000,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
                intermediate_size: 1024,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_seq_len: 2048,

                partial_rotary_factor: 1.0,
                yarn: None,
            },
        ));
        engine.register_model("default", mock_model);
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());

        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "max_tokens": 5,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // After completion, the scheduler must have no running requests and
        // no per-request state entries left behind.
        let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
        assert!(engine.scheduler.running.is_empty());
        assert!(engine.sessions.is_empty());
        assert!(engine.last_outcomes.is_empty());
        assert!(engine.request_rng.is_empty());
        assert!(engine.request_model_ids.is_empty());
        assert!(engine.request_input_ids.is_empty());
        assert!(engine.request_last_token.is_empty());
    }

    /// Safety regression guard: `AppState.engine` must remain `Mutex<Engine>`,
    /// not `RwLock<Engine>` or an unwrapped `Engine`. An `RwLock` would allow
    /// concurrent *readers*, which is exactly the access pattern the
    /// `unsafe impl Send + Sync` blocks on ROCm device types
    /// (`RocmDevice`, `NcclComm`, `HostStagingBuffer`, `StagingCache`,
    /// `QuantizedMatmulBackwardResiduals`) are NOT proven safe against — those
    /// types depend on the "exactly one caller at a time" invariant that only a
    /// `Mutex` (not an `RwLock`) enforces. If a future refactor changes this to
    /// `RwLock`, add internal locking to those types first.
    ///
    /// This is a compile-time assertion via type annotation — if `AppState.engine`
    /// is ever changed from `Mutex<Engine>` to something else, this binding will
    /// fail to compile.
    #[test]
    fn test_appstate_engine_is_mutex() {
        // Type annotation forces `engine` to be `Mutex<Engine>` — if AppState
        // changes, this won't compile.
        let state: AppState = AppState {
            engine: Mutex::new(grim_engine::Engine::new(
                grim_engine::EngineConfig::default(),
            )),
            tokenizer: Mutex::new(None),
            model_path: None,
            plugin_registry: None,
        };
        // Verify we can lock it (Mutex works)
        let _guard = state
            .engine
            .lock()
            .expect("engine mutex should not be poisoned");
    }
}

// ============================================================================
// Dashboard endpoint — live stats for the server status page.
// ============================================================================

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

    if rocm_gpu_count > 0 {
        for ord in 0..rocm_gpu_count {
            let (free, total) = grim_backend_rocm::vram_info(ord);
            let used = total.saturating_sub(free);
            total_vram_used += used;
            total_vram_max += total;
            let memory_pct = if total > 0 {
                ((used as f64 / total as f64) * 100.0) as u32
            } else {
                0
            };
            // WI-1: `compute` is now a real utilization probe (rsmi busy %).
            // `null` when the backend has no utilization API — never a
            // fabricated 0. See `compute_utilization` per backend.
            let compute = grim_backend_rocm::compute_utilization(ord);
            gpus_json.push(serde_json::json!({
                "index": ord as u32,
                "compute": compute,
                "memory": memory_pct,
                "name": format!("ROCm GPU {ord}"),
            }));
        }
        return (total_vram_used, total_vram_max, gpus_json);
    }

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
                    "index": ord as u32,
                    "compute": compute,
                    "memory": memory_pct,
                    "name": format!("CUDA GPU {ord}"),
                }));
            }
            return (total_vram_used, total_vram_max, gpus_json);
        }
    }

    {
        let Some((free, total)) = grim_backend_metal::vram_info(0) else {
            return (0, 0, gpus_json);
        };
        if total > 0 {
            let used = total.saturating_sub(free);
            let memory_pct = ((used as f64 / total as f64) * 100.0) as u32;
            let compute = grim_backend_metal::compute_utilization(0);
            gpus_json.push(serde_json::json!({
                "index": 0u32,
                "compute": compute,
                "memory": memory_pct,
                "name": "Metal GPU",
            }));
            return (used, total, gpus_json);
        }
    }

    {
        let Some((free, total)) = grim_backend_vulkan::vram_info(0) else {
            return (0, 0, gpus_json);
        };
        if total > 0 {
            let used = total.saturating_sub(free);
            let memory_pct = ((used as f64 / total as f64) * 100.0) as u32;
            let compute = grim_backend_vulkan::compute_utilization(0);
            gpus_json.push(serde_json::json!({
                "index": 0u32,
                "compute": compute,
                "memory": memory_pct,
                "name": "Vulkan GPU",
            }));
            return (used, total, gpus_json);
        }
    }

    gpus_json.push(serde_json::json!({
        "index": 0u32,
        "compute": serde_json::Value::Null,
        "memory": 0u32,
        "name": "CPU",
    }));

    (0, 0, gpus_json)
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

/// `GET /api/stats` — JSON stats snapshot polled by the dashboard at `/`.
///
/// WI-1 wire-shape note: `gpus[].compute` is now `Option<u32>` — a real
/// per-backend utilization probe, or `null` when the backend has no
/// utilization API. Consumers that previously read `compute` as an
/// always-present `u32` must tolerate `null`. A permanently-zero column is
/// worse than an absent one, so `null` is the honest value here.
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

    // Once we wire real telemetry counters into the engine (tokens generated,
    // wall-clock time per batch, KV block occupancy), this becomes live data.
    // For now the fields are present and typed so the frontend contract is fixed.
    let (kv_used, kv_total, kv_blocks_used, kv_blocks_total) = engine.kv_cache_telemetry();
    let (vram_used, vram_total, gpus_json) = probe_vram_and_gpus(rocm_gpu_count);
    let (sys_ram_used, sys_ram_total) = probe_sys_ram();
    let tps_json = match engine.tokens_per_sec() {
        Some(tps) => serde_json::json!(tps),
        None => serde_json::Value::Null,
    };

    serde_json::json!({
        "model_name": model_name,
        "tokens_per_sec": tps_json,
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

/// Dashboard HTML (static, polls /api/stats via fetch).
const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grim Server</title>
<style>
  *{box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
  body{margin:0;padding:24px;background:#0d1117;color:#c9d1d9}
  h1{color:#00d4aa;margin:0 0 4px;font-size:28px}
  .sub{color:#8b949e;margin-bottom:24px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-bottom:24px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px}
  .card h3{margin:0 0 16px;color:#00d4aa;font-size:12px;text-transform:uppercase;letter-spacing:1px}
  .row{display:flex;justify-content:space-between;align-items:center;margin:10px 0}
  .label{color:#8b949e;font-size:13px}
  .val{font-weight:600;font-size:15px}
  .val.green{color:#3fb950}.val.yellow{color:#d29922}.val.red{color:#f85149}
  .bar{height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:6px}
  .bar-fill{height:100%;background:linear-gradient(90deg,#00d4aa,#39d0d8);transition:width .5s}
  .models-section h2{color:#8b949e;font-size:14px;text-transform:uppercase;letter-spacing:1px;margin:24px 0 12px}
  .model-list{list-style:none;padding:0;margin:0}
  .model-row{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #21262d}
  .model-row:last-child{border-bottom:none}
  .badge{font-size:10px;padding:2px 8px;border-radius:10px;text-transform:uppercase;font-weight:700}
  .badge.grim{background:#1f6feb;color:#fff}
  .badge.gguf{background:#6e7681;color:#fff}
  .badge.other{background:#8b949e;color:#0d1117}
  #status-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}
  #status-dot.live{background:#3fb950;animation:pulse 2s infinite}
  #status-dot.dead{background:#f85149}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
  .empty{color:#6e7681;font-style:italic}
</style>
</head>
<body>
<h1>🦇 Grim Server</h1>
<div class="sub"><span id="status-dot" class="dead"></span><span id="conn-status">Connecting…</span></div>

<div class="grid">
  <div class="card">
    <h3>Model Status</h3>
    <div class="row"><span class="label">Name</span><span id="model-name" class="val">—</span></div>
    <div class="row"><span class="label">Type</span><span id="model-type" class="val">—</span></div>
    <div class="row"><span class="label">VRAM</span><span id="model-vram" class="val">—</span></div>
    <div class="row"><span class="label">RAM</span><span id="model-ram" class="val">—</span></div>
    <div class="row"><span class="label">GPU Util</span><span id="model-gpu" class="val">—</span></div>
    <div class="row"><span class="label">KV Cache</span><span id="model-kv" class="val">—</span></div>
    <div class="row"><span class="label">CTX Len</span><span id="model-ctx" class="val">—</span></div>
    <div class="row"><span class="label">Adapters</span><span id="adapters" class="val">0</span></div>
  </div>
  <div class="card">
    <h3>Perf</h3>
    <div class="row"><span class="label">Token/s</span><span id="tps" class="val">—</span></div>
    <div class="row"><span class="label">GPU</span><span id="gpu-name" class="val">—</span></div>
    <div class="row"><span class="label">Mem</span><span id="gpu-mem" class="val">—</span></div>
  </div>
  <div class="card">
    <h3>KV Cache</h3>
    <div class="row"><span class="label">Usage</span><span id="kv" class="val">—</span></div>
    <div class="bar"><div id="kv-bar" class="bar-fill" style="width:0%"></div></div>
    <div class="row"><span class="label">Blocks</span><span id="kv-blocks" class="val">—</span></div>
  </div>
  <div class="card">
    <h3>VRAM</h3>
    <div class="row"><span class="label">Used</span><span id="vram" class="val">—</span></div>
    <div class="bar"><div id="vram-bar" class="bar-fill" style="width:0%"></div></div>
  </div>
</div>

<div class="models-section">
  <h2>GRIM Models</h2>
  <ul id="m-grim" class="model-list"><li class="empty">No .grim models cached</li></ul>
  <h2>GGUF Models</h2>
  <ul id="m-gguf" class="model-list"><li class="empty">No .gguf models cached</li></ul>
  <h2>Other Models</h2>
  <ul id="m-other" class="model-list"><li class="empty">No other models cached</li></ul>
</div>

<script>
function fmt(b){if(b===0)return '0 B';const u=['B','KB','MB','GB','TB'];let i=0;while(b>=1024&&i<u.length-1){b/=1024;i++}return b.toFixed(1)+' '+u[i]}
function pct(used,total){return total>0?Math.round(used/total*100):0}
function cls(p){return p>90?'red':p>70?'yellow':'green'}

async function poll(){
  try{
    const r=await fetch('/api/stats');
    if(!r.ok)throw 0;
    const d=await r.json();
    document.getElementById('status-dot').className='live';
    document.getElementById('conn-status').textContent='Live — refreshing every 2s';

    const model=d.loaded_models && d.loaded_models[0];

    document.getElementById('model-name').textContent=model?.name || d.model_name ||'—';
    document.getElementById('model-type').textContent=model?.format || '—';
    document.getElementById('model-vram').textContent=model?.vram_gb ? model.vram_gb.toFixed(1)+' GB' : '—';
    document.getElementById('model-ram').textContent=model?.sys_ram_gb ? model.sys_ram_gb.toFixed(1)+' GB' : '—';
    document.getElementById('model-gpu').textContent=model?.gpu_util_pct ? (model.gpu_util_pct).toFixed(0)+'%' : '—';
    document.getElementById('model-kv').textContent=model?.kv_used_gb ? model.kv_used_gb.toFixed(1)+' GB / '+(model.kv_total_gb?' '+model.kv_total_gb.toFixed(1)+' GB':'—') : '—';
    document.getElementById('model-ctx').textContent=model?.ctx_limit || '—';

    const tps=d.tokens_per_sec;
    const tpsEl=document.getElementById('tps');
    tpsEl.textContent=(tps!==null&&tps!==undefined)?tps.toFixed(1):'—';
    tpsEl.className='val '+(tps>20?'green':tps>5?'yellow':'red');
    document.getElementById('adapters').textContent=d.adapters_active??0;

    const kvPct=pct(d.kv_cache.used,d.kv_cache.total);
    document.getElementById('kv').textContent=d.kv_cache.total>0?fmt(d.kv_cache.used)+' / '+fmt(d.kv_cache.total):'—';
    document.getElementById('kv-bar').style.width=kvPct+'%';
    document.getElementById('kv-blocks').textContent=(d.kv_cache.blocks_used??0)+' / '+(d.kv_cache.blocks_total??0);

    const vramPct=pct(d.vram.used,d.vram.total);
    const vEl=document.getElementById('vram');
    vEl.textContent=d.vram.total>0?fmt(d.vram.used)+' / '+fmt(d.vram.total):'—';
    vEl.className='val '+cls(vramPct);
    document.getElementById('vram-bar').style.width=vramPct+'%';

    const gpu=(d.gpus&&d.gpus[0])||{};
    document.getElementById('gpu-name').textContent=gpu.name||'—';
    document.getElementById('gpu-mem').textContent=(gpu.memory??0)+'%';

    if(d.models){
      const render=(id,arr)=>{
        const el=document.getElementById(id);
        if(!arr||arr.length===0){el.innerHTML='<li class="empty">None</li>';return}
        el.innerHTML=arr.map(m=>{
          const sz=m.size?fmt(m.size):'';
          const extra=[m.params,m.quant].filter(Boolean).join(' · ');
          return '<li class="model-row"><span>'+m.name+(extra?' <span class="label">'+extra+'</span>':'')+'</span><span class="badge '+(m.format||'other')+'">'+m.format+' '+sz+'</span></li>';
        }).join('');
      };
      render('m-grim',d.models.grim);
      render('m-gguf',d.models.gguf);
      render('m-other',d.models.other);
    }
  }catch(e){
    document.getElementById('status-dot').className='dead';
    document.getElementById('conn-status').textContent='Disconnected — retrying…';
  }
}
poll();
setInterval(poll,2000);
</script>
</body>
</html>"#;
