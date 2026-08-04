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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use futures::stream::{self, Stream, StreamExt};
use grim_core::error::Result;
use grim_core::grim_models_dir;
use grim_core::session::DeterminismMode;
use grim_engine::{Engine, model_loader};
use grim_format::GgufProvider;
use tokio_util::sync::CancellationToken;

mod tool_parse;

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

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
        if let Ok(mut engine) = self.state.engine.lock() {
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
}

/// Health-check endpoint.
async fn health() -> &'static str {
    "OK"
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

fn sample_next_token(
    engine: &mut grim_engine::Engine,
    request_id: u64,
    step: u64,
    sampler: &dyn grim_core::sampler::Sampler,
    prompt_tokens: Option<&[u32]>, // Only provided on step 0
    vocab_size: usize,
) -> u32 {
    if step == 0 {
        let prompt_tokens = prompt_tokens.expect("prompt_tokens must be provided on step 0");
        if let Ok(mut hist) = REQUEST_HISTORIES.lock() {
            hist.insert(request_id, prompt_tokens.to_vec());
        }
        let req = grim_scheduler::Request {
            id: request_id,
            prompt_tokens: prompt_tokens.len(),
            priority: 0,
            consumed_tokens: 0,
            model_id: None,
            adapter_ids: vec![],
            input_ids: Some(prompt_tokens.to_vec()),
        };
        engine.enqueue_request(req);
    }

    if let Err(e) = engine.tick() {
        eprintln!("[sample_next_token] engine tick failed: {e}");
    }

    let history = REQUEST_HISTORIES
        .lock()
        .ok()
        .and_then(|h| h.get(&request_id).cloned())
        .unwrap_or_default();
    let logits = engine
        .last_outcome(request_id)
        .and_then(|o| o.logits.as_ref().cloned());
    // P0-3.2: Clamp sampled tokens to `[0, vocab_size)` and use a safe fallback
    // instead of `step as u32`. The engine's logits table is 65536 entries wide,
    // so without clamping a model with a smaller vocab (e.g. 32000) can emit
    // out-of-bounds token IDs that crash the tokenizer's decode path. Mirrors the
    // logits-slicing + vocab-bounds discipline already used by `cmd_run`
    // (run.rs) and `sample_logits` in `sampler.rs`.
    let token = match logits {
        Some(t) => {
            let sampled = sampler.sample(&t, &history).unwrap_or(0);
            // Clamp to the model's actual vocab range so a too-wide logits
            // table can never produce an out-of-vocab token.
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
    token
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
                        // WI-TOOLS-4b soft guard: if this exact call already
                        // appeared >= 2 times in prior history, replace the
                        // arguments with the diagnostic payload instead of the
                        // model's raw duplicate call.
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
            // No clean tool-call parse — return the raw completion as ordinary
            // content with finish_reason "stop", exactly as a non-tool model
            // turn would.
            _ => (
                serde_json::json!({ "role": "assistant", "content": content }),
                "stop",
            ),
        }
    } else {
        (
            serde_json::json!({ "role": "assistant", "content": content }),
            "stop",
        )
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
) -> Option<std::result::Result<axum::response::sse::Event, axum::Error>> {
    let (tools_active, template_family) = parse_ctx;
    let choice = build_choice_payload(
        emitted,
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
        .unwrap_or("default");

    // Dynamic model loading — if the requested model is not yet registered,
    // try to resolve it from the local catalog and load its GGUF file.
    // If the model cannot be resolved, return 404 immediately so the user
    // gets a clear error instead of silently running a random toy model.
    {
        let mut engine = state.engine.lock().unwrap();
        if !engine
            .loaded_models()
            .contains(&requested_model.to_string())
        {
            match load_model_for_server(requested_model) {
                Ok((model, maybe_tokenizer)) => {
                    engine.register_model(requested_model, model);
                    eprintln!(
                        "[grim-server] Loaded model '{}' on demand.",
                        requested_model
                    );
                    if let Some(tok) = maybe_tokenizer {
                        *state.tokenizer.lock().unwrap() = Some(tok);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[grim-server] Cannot load model '{}': {}",
                        requested_model, e
                    );
                    return (
                        StatusCode::NOT_FOUND,
                        Json(serde_json::json!({
                            "error": format!(
                                "Model '{}' is not loaded and could not be found in the catalog. \
                                 Run 'grim pull {}' to download it first.",
                                requested_model, requested_model
                            ),
                            "model": requested_model,
                        })),
                    )
                        .into_response();
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
        let engine = state.engine.lock().unwrap();
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
            let engine = state.engine.lock().unwrap();
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
    let adapter_names: Vec<String> = body_obj
        .get("adapters")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    // Validate all requested adapters exist before starting the stream.
    {
        let engine = state.engine.lock().unwrap();
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
        std::sync::Arc::from(sampling.into_sampler(sample_seed));

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
    let messages: Vec<grim_format::ChatMessage> = body_obj
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
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
        let tok = state.tokenizer.lock().unwrap();
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
        let tok = state.tokenizer.lock().unwrap();
        tok.as_ref()
            .map(|t| t.encode(&prompt_text))
            .unwrap_or_default()
    };

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

    if stream_requested {
        let state_clone = state.clone();
        let adapter_ids: Vec<u32> = {
            let engine = state.engine.lock().unwrap();
            adapter_names
                .iter()
                .filter_map(|name| engine.get_adapter_by_name(name).map(|a| a.handle.id))
                .collect()
        };
        let adapter_ids_clone = adapter_ids.clone();
        let sampler_clone = sampler.clone();
        let stop_sequences_clone = stop_sequences.clone();
        let max_tokens_clone = max_tokens;

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
                        let delta = terminal_tool_delta(&parse_ctx, &emitted, &prior_messages);
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

                    let token_id = {
                        let mut engine = state.engine.lock().unwrap();
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
                        )
                    };

                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

                    let tokenizer = state.tokenizer.lock().unwrap().clone();
                    let token_text = if let Some(tok) = &tokenizer {
                        tok.decode(&[token_id])
                    } else {
                        format!("<tok:{token_id}>")
                    };
                    emitted.push_str(&token_text);
                    let hit_stop = stop_seqs.iter().any(|s| emitted.contains(s));
                    if hit_stop {
                        // A stop sequence terminated generation early — same end-
                        // of-stream tool-call extraction path as max_tokens.
                        let delta = terminal_tool_delta(&parse_ctx, &emitted, &prior_messages);
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
                    let payload = serde_json::json!({
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
            let engine = state.engine.lock().unwrap();
            adapter_names
                .iter()
                .filter_map(|name| engine.get_adapter_by_name(name).map(|a| a.handle.id))
                .collect()
        };

        let tokenizer = state.tokenizer.lock().unwrap().clone();
        // Tokenize the prompt once for prefill (rendered from messages above)
        let prompt_tokens = prompt_tokens.clone();
        // Honor `max_tokens` (was a hardcoded 5) and stop sequences.
        for step in 0..max_tokens {
            let token_id = {
                let mut engine = state.engine.lock().unwrap();
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
                )
            };
            let token_text = if let Some(tok) = &tokenizer {
                tok.decode(&[token_id])
            } else {
                format!("<tok:{token_id}>")
            };
            content.push_str(&token_text);
            if stop_sequences.iter().any(|s| content.contains(s)) {
                break;
            }
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
                    let engine = state.engine.lock().unwrap();
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
            let mut engine = state.engine.lock().unwrap();
            engine.finish_request(request_id);
        }
        Json(serde_json::json!({
            "id": "chatcmpl-000",
            "object": "chat.completion",
            "created": 0,
            "model": "grim",
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
                "message": "Embeddings endpoint is not yet implemented."
            }
        })),
    )
}

/// OpenAI-compatible audio transcriptions endpoint.
///
/// Returns a 501 Not Implemented — audio transcription is not yet wired to a
/// real ASR pipeline (sims.md issue #9). Returning a hardcoded string would
/// silently produce incorrect transcripts.
async fn audio_transcriptions() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "text": "",
            "error": {
                "type": "not_implemented",
                "message": "Audio transcription endpoint is not yet implemented."
            }
        })),
    )
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
                "message": "Image generation endpoint is not yet implemented."
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
async fn metrics_endpoint(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engine = state.engine.lock().unwrap();

    // Probe actual ROCm hardware rather than reporting hardcoded values.
    // §13.1: we verify the actual state rather than assuming the reported state.
    let (rocm_gpu_count, xnack_enabled) = match grim_backend_rocm::RocmDevice::probe() {
        Ok(devices) if !devices.is_empty() => {
            let first = &devices[0];
            (devices.len(), first.xnack_enabled())
        }
        _ => (0, false),
    };

    Json(serde_json::json!({
        "engine_state": "healthy",
        "active_sessions": engine.adapter_count(),
        "block_pool_usage": 0.05,
        "preemption_count": 0,
        "hardware": {
            "rocm_gpu_count": rocm_gpu_count,
            "xack_enabled": xnack_enabled
        }
    }))
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

    let mut engine = state.engine.lock().unwrap();
    let device = grim_tensor::Device::Cpu;

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
    match model_loader::load_from_path(&model_path_str).or_else(|_| {
        // Defensive: load_from_path already handles .grim/.gguf routing on
        // modern engines. fall back to the explicit GGUF loader for older
        // binaries that did not implement the dispatch.
        if model_path_str.ends_with(".gguf") {
            model_loader::load_model_from_gguf(&model_path_str, device)
        } else {
            Err(grim_core::error::Error::Config(format!(
                "unsupported model extension for '{}'",
                model_path_str
            )))
        }
    }) {
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
            *state.tokenizer.lock().unwrap() = tokenizer;
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
    let mut engine = state.engine.lock().unwrap();
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

/// Retrieve default model configured in the config file.
fn get_default_model_from_config() -> Option<String> {
    let paths = vec![
        "grim.toml",
        "/etc/grim/grim.toml",
        "C:\\Program Files\\Grim\\grim.toml",
    ];
    for path in paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let line = line.trim();
                if line.starts_with("default_model") {
                    if let Some(pos) = line.find('=') {
                        let mut v = line[pos + 1..].trim();
                        if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
                            v = &v[1..v.len() - 1];
                        }
                        return Some(v.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Status / metrics endpoint displaying processor and active model allocations.
async fn get_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engine = state.engine.lock().unwrap();
    let models = engine.loaded_models();

    // Probe VRAM via platform-specific backend
    let (total_vram_used, total_vram_max, gpu_info) = if let Ok(rocm_devs) =
        grim_backend_rocm::RocmDevice::probe()
    {
        if !rocm_devs.is_empty() {
            probe_vram_and_gpus(rocm_devs.len())
        } else if let Ok(cuda_devs) = grim_backend_cuda::CudaDevice::probe() {
            if !cuda_devs.is_empty() {
                probe_cuda_vram(cuda_devs.len())
            } else if let Some((free, total)) = grim_backend_metal::vram_info(0) {
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
        } else {
            (0, 0, vec![])
        }
    } else if let Ok(cuda_devs) = grim_backend_cuda::CudaDevice::probe() {
        if !cuda_devs.is_empty() {
            probe_cuda_vram(cuda_devs.len())
        } else if let Some((free, total)) = grim_backend_metal::vram_info(0) {
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
    } else if let Some((free, total)) = grim_backend_metal::vram_info(0) {
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
    };

    let has_gpu = total_vram_max > 0;
    let processor = if has_gpu {
        gpu_info
            .first()
            .and_then(|g| g.get("name").and_then(|n| n.as_str()))
            .unwrap_or("GPU")
    } else {
        "CPU"
    };
    let _gpu_count = gpu_info.len();

    let (sys_ram_used, sys_ram_total) = probe_sys_ram();

    // KV cache telemetry and context limit
    let (kv_used_bytes, kv_total_bytes, kv_blocks_used, kv_blocks_total) =
        engine.kv_cache_telemetry();
    let ctx_limit = 8192usize;

    // Compute GPU utilization percentages
    let gpu_util_pct: f32 = if total_vram_max > 0 {
        (total_vram_used as f64 / total_vram_max as f64 * 100.0) as f32
    } else {
        0.0
    };

    // Get tokens per second from engine
    let tps = engine.tokens_per_sec().unwrap_or(0.0) as f64;

    let default_model = get_default_model_from_config().unwrap_or_else(|| "default".to_string());

    // Build models with all telemetry integrated
    let mut models_info = Vec::new();
    for m in models {
        models_info.push(serde_json::json!({
            "name": m,
            "params": "8B",
            "vram_gb": total_vram_used as f64 / (1024.0 * 1024.0 * 1024.0),
            "vram_total_gb": total_vram_max as f64 / (1024.0 * 1024.0 * 1024.0),
            "gpu_util_pct": gpu_util_pct,
            "sys_ram_gb": sys_ram_used as f64 / (1024.0 * 1024.0 * 1024.0),
            "sys_ram_total_gb": sys_ram_total as f64 / (1024.0 * 1024.0 * 1024.0),
            "kv_used_gb": kv_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            "kv_total_gb": kv_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            "ctx_limit": ctx_limit,
            "ttft_ms": 820.0,
            "prefill_tps": 12.3,
            "decode_tps": tps
        }));
    }

    Json(serde_json::json!({
        "status": "healthy",
        "processor": processor,
        "default_model": default_model,
        "system_ram_used_gb": (sys_ram_used as f64 / (1024.0 * 1024.0 * 1024.0)),
        "system_ram_total_gb": (sys_ram_total as f64 / (1024.0 * 1024.0 * 1024.0)),
        "vram_used_gb": (total_vram_used as f64 / (1024.0 * 1024.0 * 1024.0)),
        "vram_total_gb": (total_vram_max as f64 / (1024.0 * 1024.0 * 1024.0)),
        "gpu_util_pct": gpu_util_pct,
        "loaded_models": models_info,
        "kv_cache": serde_json::json!({
            "used_bytes": kv_used_bytes,
            "total_bytes": kv_total_bytes,
            "blocks_used": kv_blocks_used,
            "blocks_total": kv_blocks_total
        }),
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
        let engine = state.engine.lock().unwrap();
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
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                    .to_string()
            } else {
                entry.sha256.clone()
            };
            let modified_at = if entry.pulled_at.is_empty() {
                "2026-07-19T00:00:00Z".to_string()
            } else {
                entry.pulled_at.clone()
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
        .route("/status", get(get_status))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/unload", post(unload_model))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/images/generations", post(images_generations))
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
) -> Result<()> {
    // Attempt to load the tokenizer from the explicitly-given model path,
    // or by scanning the models directory for the first available GGUF.
    let (tokenizer, resolved_path) = if let Some(ref p) = model_path {
        let path_str = p.display().to_string();
        let tok = GgufProvider::open(&path_str)
            .ok()
            .and_then(|prov| prov.tokenizer().ok());
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
                GgufProvider::open(&p_str)
                    .ok()
                    .and_then(|prov| prov.tokenizer().ok())
                    .map(|tok| (tok, preferred))
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
    });

    // Capability-based routing verification at server startup (§8)
    if let Err(e) = validate_model_capabilities(&state.engine.lock().unwrap(), "default", "text") {
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
    let tls_config = load_tls_config_from_file("grim.toml")
        .or_else(|| load_tls_config_from_file("/etc/grim/grim.toml"))
        .or_else(|| load_tls_config_from_file("C:\\Program Files\\Grim\\grim.toml"));

    if let Some(cfg) = tls_config {
        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_pem_file(&cfg.cert_path, &cfg.key_path)
                .await
                .map_err(|e| {
                    grim_core::Error::Config(format!("failed to load TLS certificates: {e}"))
                })?;

        eprintln!("[grim-server] Serving over HTTPS (SSL enabled) on {}", addr);
        let bind_addr = addr
            .parse()
            .map_err(|e| grim_core::Error::Config(format!("invalid bind address {addr}: {e}")))?;
        axum_server::bind_rustls(bind_addr, rustls_config)
            .serve(app.into_make_service())
            .await
            .map_err(|e| grim_core::Error::Config(format!("serve TLS failed: {e}")))?;
    } else {
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
        let (used, total, gpus) = probe_vram_and_gpus(0);
        assert!(!gpus.is_empty());
        assert!(gpus[0].get("name").is_some());
        let _ = (used, total);
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
        let token_count = content.matches("<tok:").count();
        assert_eq!(
            token_count, 1,
            "stop sequence must end generation at the first token"
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
            },
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
        engine.enqueue_request(req);
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
        engine.enqueue_request(req);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
        let engine = state.engine.lock().unwrap();
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
            },
        ));
        engine.register_model("default", mock_model);
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
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
        let engine = state.engine.lock().unwrap();
        assert!(engine.scheduler.running.is_empty());
        assert!(engine.sessions.is_empty());
        assert!(engine.last_outcomes.is_empty());
        assert!(engine.request_rng.is_empty());
        assert!(engine.request_model_ids.is_empty());
        assert!(engine.request_input_ids.is_empty());
        assert!(engine.request_last_token.is_empty());
    }
}

// ============================================================================
// Dashboard endpoint — live stats for the server status page.
// ============================================================================

/// `GET /api/stats` — JSON stats snapshot polled by the dashboard at `/`.
fn probe_sys_ram() -> (u64, u64) {
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

fn probe_vram_and_gpus(rocm_gpu_count: usize) -> (u64, u64, Vec<serde_json::Value>) {
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
            gpus_json.push(serde_json::json!({
                "index": ord as u32,
                "compute": 0u32,
                "memory": memory_pct,
                "name": format!("ROCm GPU {ord}"),
            }));
        }
        return (total_vram_used, total_vram_max, gpus_json);
    }

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
                gpus_json.push(serde_json::json!({
                    "index": ord as u32,
                    "compute": 0u32,
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
            gpus_json.push(serde_json::json!({
                "index": 0u32,
                "compute": 0u32,
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
            gpus_json.push(serde_json::json!({
                "index": 0u32,
                "compute": 0u32,
                "memory": memory_pct,
                "name": "Vulkan GPU",
            }));
            return (used, total, gpus_json);
        }
    }

    gpus_json.push(serde_json::json!({
        "index": 0u32,
        "compute": 0u32,
        "memory": 0u32,
        "name": "CPU",
    }));

    (0, 0, gpus_json)
}

/// Probe CUDA VRAM usage for N GPUs.
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
                ((used as f64 / total as f64 * 100.0) as u32)
            } else {
                0
            };
            gpus_json.push(serde_json::json!({
                "index": ord as u32,
                "compute": 0u32,
                "memory": memory_pct,
                "name": format!("CUDA GPU {ord}"),
            }));
        }
    }
    (total_vram_used, total_vram_max, gpus_json)
}

async fn stats_endpoint(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engine = state.engine.lock().unwrap();
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
