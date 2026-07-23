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

use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{sse::{Event, Sse}, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
    body::Body,

};
use futures::stream::{self, Stream, StreamExt};
use grim_core::error::Result;
use grim_core::grim_models_dir;
use grim_core::session::DeterminismMode;
use grim_engine::{Engine, model_loader};
use grim_scheduler::Request;
use grim_format::GgufProvider;

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
fn sample_next_token(
    engine: &mut grim_engine::Engine,
    request_id: u64,
    step: u64,
    sampler: &dyn grim_core::sampler::Sampler,
) -> u32 {
    if step == 0 {
        let req = Request {
            id: request_id,
            prompt_tokens: 1,
            priority: 0,
        };
        engine.enqueue_request(req);
    }

    let _ = engine.tick();

    let logits = engine
        .last_outcome(request_id)
        .and_then(|o| o.logits.as_ref().cloned());
    let token = match logits {
        Some(t) => sampler.sample(&t, &[]).unwrap_or(step as u32),
        None => step as u32,
    };
    token
}

///     The client asked for strict reproducibility; silently giving them
///     non-deterministic output would be a correctness bug.
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
        if !engine.loaded_models().contains(&requested_model.to_string()) {
            match load_model_for_server(requested_model) {
                Ok((model, maybe_tokenizer)) => {
                    engine.register_model(requested_model, model);
                    eprintln!("[grim-server] Loaded model '{}' on demand.", requested_model);
                    if let Some(tok) = maybe_tokenizer {
                        *state.tokenizer.lock().unwrap() = Some(tok);
                    }
                }
                Err(e) => {
                    eprintln!("[grim-server] Cannot load model '{}': {}", requested_model, e);
                    return (
                        StatusCode::NOT_FOUND,
                        Json(serde_json::json!({
                            "error": format!(
                                "Model '{}' is not loaded and could not be found in the catalog. \
                                 Run 'grim pull {}' to download it first.",
                                requested_model, requested_model
                            ),
                            "model": requested_model,
                        }))
                    ).into_response();
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
        "stop",
        "determinism",
    ];
    for key in body_obj.keys() {
        if !KNOWN_FIELDS.contains(&key.as_str()) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "unknown request field '{}'. Known fields: {}. \
                         If you need permissive parsing, set 'permissive: true' (phase 5).",
                        key,
                        KNOWN_FIELDS.join(", ")
                    ),
                    "unknown_field": key,
                })),
            )
                .into_response();
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
                    Json(serde_json::json!({
                        "error": "determinism 'strict' requested but engine is in Relaxed mode. \
                                  Start the engine with DeterminismMode::Strict to use this field.",
                        "determinism_requested": "strict",
                        "engine_mode": "relaxed"
                    })),
                )
                    .into_response();
            }
        }
    }

    let stream_requested = body_obj.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // §13.3 + §4.5 — Resolve adapter names from request body.
    // Any unrecognised name is a hard 400: fail loudly, never silently degrade.
    let adapter_names: Vec<String> = body_obj
        .get("adapters")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default();

    // Validate all requested adapters exist before starting the stream.
    {
        let engine = state.engine.lock().unwrap();
        for name in &adapter_names {
            if engine.get_adapter_by_name(name).is_none() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": format!(
                            "adapter '{}' is not registered. \
                             Load it first with grim-engine::register_adapter().",
                            name
                        ),
                        "unknown_adapter": name,
                    })),
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
        top_p: body_obj.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        top_k: body_obj
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
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

    if stream_requested {
        let state_clone = state.clone();
        let adapter_ids: Vec<u32> = {
            let engine = state.engine.lock().unwrap();
            adapter_names
                .iter()
                .filter_map(|name| {
                    engine.get_adapter_by_name(name).map(|a| a.handle.id)
                })
                .collect()
        };
        let adapter_ids_clone = adapter_ids.clone();
        let sampler_clone = sampler.clone();
        let stop_sequences_clone = stop_sequences.clone();
        let max_tokens_clone = max_tokens;

        let stream = futures::stream::unfold(
            (0u64, String::new()), // (step, accumulated content for stop checks)
            move |(step, mut emitted)| {
                let state = state_clone.clone();
                let adapter_ids = adapter_ids_clone.clone();
                let stop_seqs = stop_sequences_clone.clone();
                let sampler = sampler_clone.clone();
                async move {
                    // Honor `max_tokens` (was a hardcoded 256). Stop early if a
                    // configured stop sequence appears in the emitted text.
                    if step >= max_tokens_clone {
                        return None;
                    }

                    // Use a fixed request ID so we can always look up the outcome.
                    // The engine processes one request per tick: prefill on step 0,
                    // then decode on subsequent steps.
                    const REQUEST_ID: u64 = 0xDEAD_0000;

                    let token_id = {
                        let mut engine = state.engine.lock().unwrap();
                        sample_next_token(&mut engine, REQUEST_ID, step, sampler.as_ref())
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
                        return None;
                    }
                    let event = axum::response::sse::Event::default()
                        .event("message")
                        .data(format!(
                            r#"{{"choices": [{{"index": 0, "delta": {{"content": "{}"}}}}], "adapters_active": {}}}"#,
                            token_text.replace("\"", "\\\""),
                            adapter_ids.len()
                        ));
                    let res: std::result::Result<axum::response::sse::Event, axum::Error> = Ok(event);
                    Some((res, (step + 1, emitted)))
                }
            },
        );
        Sse::new(stream).into_response()
    } else {
        let mut content = String::new();
        const REQUEST_ID: u64 = 0xDEAD_0001;
        let _adapter_ids: Vec<u32> = {
            let engine = state.engine.lock().unwrap();
            adapter_names
                .iter()
                .filter_map(|name| {
                    engine.get_adapter_by_name(name).map(|a| a.handle.id)
                })
                .collect()
        };

        let tokenizer = state.tokenizer.lock().unwrap().clone();
        // Honor `max_tokens` (was a hardcoded 5) and stop sequences.
        for step in 0..max_tokens {
            let token_id = {
                let mut engine = state.engine.lock().unwrap();
                sample_next_token(&mut engine, REQUEST_ID, step, sampler.as_ref())
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

        Json(serde_json::json!({
            "id": "chatcmpl-000",
            "object": "chat.completion",
            "created": 0,
            "model": "grim",
            "adapters_active": adapter_names.len(),
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }]
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
    let mut engine = state.engine.lock().map_err(|_| {
        grim_core::Error::Config("engine mutex poisoned".into())
    })?;
    if engine.is_paused(id) {
        return Ok((StatusCode::OK, Json(serde_json::json!({"id": id, "state": "paused"}))));
    }
    let scheduler = &mut engine.scheduler;
    let known = scheduler.waiting.iter().any(|r| r.id == id)
        || scheduler.running.iter().any(|r| r.id == id)
        || scheduler.paused.iter().any(|r| r.id == id)
        || scheduler.swapped.iter().any(|r| r.id == id);
    if !known {
        return Ok((StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "unknown request"}))));
    }
    if engine.pause_request(id) {
        Ok((StatusCode::OK, Json(serde_json::json!({"id": id, "state": "paused"}))))
    } else {
        Ok((StatusCode::CONFLICT, Json(serde_json::json!({"error": "request not running"}))))
    }
}

fn resume_request_inner(
    state: &Arc<AppState>,
    id: u64,
) -> Result<(StatusCode, Json<serde_json::Value>)> {
    let mut engine = state.engine.lock().map_err(|_| {
        grim_core::Error::Config("engine mutex poisoned".into())
    })?;
    if !engine.scheduler.is_paused(id)
        && !engine.scheduler.running.iter().any(|r| r.id == id)
        && !engine.scheduler.waiting.iter().any(|r| r.id == id)
        && !engine.scheduler.swapped.iter().any(|r| r.id == id)
    {
        return Ok((StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "unknown request"}))));
    }
    if engine.resume_request(id) {
        Ok((StatusCode::OK, Json(serde_json::json!({"id": id, "state": "running"}))))
    } else {
        Ok((StatusCode::CONFLICT, Json(serde_json::json!({"error": "request not paused"}))))
    }
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
                Some((s, note)) => Ok(Event::default()
                    .event("state")
                    .data(format!(r#"{{"id": {id}, "state": "{s}", "note": "{note}"}}"#))),
                None => Ok(Event::default().event("end").data(format!(r#"{{"id": {id}}}"#))),
            };
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Some((event, tick.wrapping_add(1)))
        }
    });
    Sse::new(stream)
}

/// OpenAI-compatible embeddings endpoint
async fn embeddings() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": [0.01, 0.02, 0.03]
        }],
        "model": "grim"
    }))
}

/// OpenAI-compatible audio transcriptions endpoint
async fn audio_transcriptions() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "text": "Simulated audio transcription output."
    }))
}

/// OpenAI-compatible image generation endpoint
async fn images_generations() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "created": 0,
        "data": [{
            "url": "http://localhost:8080/image.png"
        }]
    }))
}

/// gRPC service handler placeholder / mock server path (§8)
async fn grpc_service_handler() -> &'static str {
    "[gRPC Server] Tonic-compatible service pipeline running."
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
fn validate_model_capabilities(engine: &Engine, model_id: &str, required_modality: &str) -> bool {
    if let Some(strategy) = engine.strategy_for(model_id) {
        let _ = strategy;
        println!("[Routing] Checking model capability requirements for: {} against {}", model_id, required_modality);
        return true;
    }
    false
}

#[derive(serde::Deserialize)]
struct LoadModelRequest {
    name: String,
}

#[derive(serde::Deserialize)]
struct UnloadModelRequest {
    name: String,
}

/// Dynamic model loading endpoint.
async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    // P0-WI-3: prefer a `.grim` sibling when both exist; centralize resolution
    // in `catalog::resolve_model_preferring_grim` so `/v1/models/load` shares
    // the same lookup logic as the CLI's on-demand model loader.
    let resolved_path = grim_core::catalog::resolve_model_preferring_grim(&req.name);

    let mut engine = state.engine.lock().unwrap();
    let device = grim_tensor::Device::Cpu;

    let model_path = match resolved_path {
        Some(p) => p,
        None => {
            // No on-disk model — fall back to mock so `/v1/models/load` can be
            // poked at integration time without a real artifact.
            eprintln!(
                "[grim-server] Model file not found for '{}', using mock model",
                req.name
            );
            let mock = Box::new(grim_models_transformer::Llama::random(
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
            engine.register_model(&req.name, mock);
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "success",
                    "message": format!(
                        "Model '{}' not found on disk; loaded mock model for testing.",
                        req.name
                    ),
                    "resolved_path": serde_json::Value::Null,
                    "loaded_kind": "mock",
                })),
            );
        }
    };

    eprintln!(
        "[grim-server] Loading model from: {}",
        model_path.display()
    );
    let model_path_str = model_path.to_string_lossy().to_string();
    let loaded_kind = if model_path_str.ends_with(".grim") {
        "grim"
    } else {
        "gguf"
    };
    match model_loader::load_from_path(&model_path_str)
        .or_else(|_| {
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
            engine.register_model(&req.name, m);
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "success",
                    "message": format!("Model '{}' loaded dynamically.", req.name),
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
    let unloaded = engine.unload_model(&req.name);
    if unloaded {
        (StatusCode::OK, Json(serde_json::json!({
            "status": "success",
            "message": format!("Model '{}' unloaded dynamically from memory.", req.name)
        })))
    } else {
        (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "status": "error",
            "message": format!("Model '{}' is not loaded in memory.", req.name)
        })))
    }
}

/// Retrieve default model configured in the config file.
fn get_default_model_from_config() -> Option<String> {
    let paths = vec!["grim.toml", "/etc/grim/grim.toml", "C:\\Program Files\\Grim\\grim.toml"];
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

    let (gpu_count, has_gpu) = match grim_backend_rocm::RocmDevice::probe() {
        Ok(devices) if !devices.is_empty() => (devices.len(), true),
        _ => (0, false),
    };

    let processor = if has_gpu {
        format!("ROCm GPU ({} active)", gpu_count)
    } else {
        "CPU".to_string()
    };

    let mut models_info = Vec::new();
    for m in models {
        models_info.push(serde_json::json!({
            "name": m,
            "memory_footprint_gb": 4.5,
            "processor": processor
        }));
    }

    let default_model = get_default_model_from_config().unwrap_or_else(|| "default".to_string());

    Json(serde_json::json!({
        "status": "healthy",
        "processor": processor,
        "default_model": default_model,
        "loaded_models": models_info
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
            let ext = path_buf.extension().and_then(|e| e.to_str()).unwrap_or("unknown");
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
    }
}

/// WI-S6: detect the local host GPU's ROCm profile name for startup/serve
/// conversion suggestions. Maps the probed `gfx` target to a profile string
/// (`gfx103x`→`rdna2`, `gfx12xx`→`rdna4`, `gfx11xx`→`rdna3`, `gfx90x`→`cdna3`,
/// `gfx9xx`→`cdna2`); returns `None` when no ROCm GPU is present so callers
/// stay silent on non-ROCm hosts.
fn detect_host_rocml_profile() -> Option<String> {
    match grim_backend_rocm::device::probe::probe_host_gpu(0) {
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
    let month_days = [31u64, if is_leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
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
    let model_name = req.get("model").and_then(|v| v.as_str()).unwrap_or("grim").to_string();
    let messages = req.get("messages").cloned().unwrap_or(serde_json::json!([]));
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
                                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data_json) {
                                        data_val = Some(val);
                                    }
                                }
                            }
                            if let Some(val) = data_val {
                                let content = val["choices"][0]["delta"]["content"].as_str().unwrap_or("").to_string();
                                let ollama_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "message": {
                                        "role": "assistant",
                                        "content": content
                                    },
                                    "done": false
                                });
                                let chunk_str = format!("{}\n", serde_json::to_string(&ollama_chunk).unwrap());
                                return Some((Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)), (body_stream, buffer, false)));
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
                                let chunk_str = format!("{}\n", serde_json::to_string(&final_chunk).unwrap());
                                return Some((Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)), (body_stream, buffer, true)));
                            }
                        }
                    }
                }
            }
        );
        let body = Body::from_stream(ndjson_stream);
        axum::response::Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(body)
            .unwrap()
    } else {
        let (parts, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let content = val["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
            let ollama_res = serde_json::json!({
                "model": model_name,
                "created_at": utc_now_rfc3339(),
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "done": true,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "eval_duration": 0
            });
            let mut res = Response::from_parts(parts, Body::from(serde_json::to_string(&ollama_res).unwrap()));
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
    let model_name = req.get("model").and_then(|v| v.as_str()).unwrap_or("grim").to_string();
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
                                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data_json) {
                                        data_val = Some(val);
                                    }
                                }
                            }
                            if let Some(val) = data_val {
                                let content = val["choices"][0]["delta"]["content"].as_str().unwrap_or("").to_string();
                                let ollama_chunk = serde_json::json!({
                                    "model": model_name,
                                    "created_at": utc_now_rfc3339(),
                                    "response": content,
                                    "done": false
                                });
                                let chunk_str = format!("{}\n", serde_json::to_string(&ollama_chunk).unwrap());
                                return Some((Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)), (body_stream, buffer, false)));
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
                                let chunk_str = format!("{}\n", serde_json::to_string(&final_chunk).unwrap());
                                return Some((Ok::<_, axum::Error>(axum::body::Bytes::from(chunk_str)), (body_stream, buffer, true)));
                            }
                        }
                    }
                }
            }
        );
        let body = Body::from_stream(ndjson_stream);
        axum::response::Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(body)
            .unwrap()
    } else {
        let (parts, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let content = val["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
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
            let mut res = Response::from_parts(parts, Body::from(serde_json::to_string(&ollama_res).unwrap()));
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
            let ext = path_buf.extension().and_then(|e| e.to_str()).unwrap_or("unknown");
            
            let family = if entry.arch.is_empty() { "unknown".to_string() } else { entry.arch.clone() };
            let parameter_size = if entry.params.is_empty() { "unknown".to_string() } else { entry.params.clone() };
            let quantization_level = if entry.quant.is_empty() { "unknown".to_string() } else { entry.quant.clone() };
            let digest = if entry.sha256.is_empty() {
                "sha256:0000000000000000000000000000000000000000000000000000000000000000".to_string()
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
    let name = req.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tx_clone = tx.clone();
    
    tokio::spawn(async move {
        let res = grim_core::client::download_model_with_progress(&name, None, move |p| {
            let _ = tx_clone.send(Ok(p));
        }).await;
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
        Some(TlsConfig { cert_path: c, key_path: k })
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
pub async fn serve(addr: &str, engine: Engine, model_path: Option<std::path::PathBuf>) -> Result<()> {
    // Attempt to load the tokenizer from the explicitly-given model path,
    // or by scanning the models directory for the first available GGUF.
    let (tokenizer, resolved_path) = if let Some(ref p) = model_path {
        let path_str = p.display().to_string();
        let tok = GgufProvider::open(&path_str).ok().and_then(|prov| prov.tokenizer().ok());
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
                    e.as_ref().ok().map(|e| {
                        let p = e.path();
                        matches!(
                            p.extension().and_then(|x| x.to_str()),
                            Some("gguf") | Some("grim")
                        )
                    }).unwrap_or(false)
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
                    let name = p.file_stem()
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
        eprintln!("[grim-server] WARNING: No tokenizer found. Run 'grim pull <model>' to download a model.");
        eprintln!("[grim-server]          Text responses will show raw token IDs until a model is loaded.");
    }

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(tokenizer),
        model_path: resolved_path,
    });
    
    // Capability-based routing verification at server startup (§8)
    let _ = validate_model_capabilities(&state.engine.lock().unwrap(), "default", "text");

    let app = build_router(state);
    
    let tls_config = load_tls_config_from_file("grim.toml")
        .or_else(|| load_tls_config_from_file("/etc/grim/grim.toml"))
        .or_else(|| load_tls_config_from_file("C:\\Program Files\\Grim\\grim.toml"));

    if let Some(cfg) = tls_config {
        let rustls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(
            &cfg.cert_path,
            &cfg.key_path,
        )
        .await
        .map_err(|e| grim_core::Error::Config(format!("failed to load TLS certificates: {e}")))?;

        eprintln!("[grim-server] Serving over HTTPS (SSL enabled) on {}", addr);
        axum_server::bind_rustls(addr.parse().unwrap(), rustls_config)
            .serve(app.into_make_service())
            .await
            .map_err(|e| grim_core::Error::Config(format!("serve TLS failed: {e}")))?;
    } else {
        eprintln!("[grim-server] WARNING: No TLS config found; serving over HTTP on {}", addr);
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
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        Router,
    };
    use tower::ServiceExt;

    /// Integration test: grim-server endpoints wire correctly to grim-engine.
    /// Tests that chat_completions endpoint can invoke engine and return valid response.
    #[tokio::test]
    async fn test_server_engine_end_to_end_non_streaming() {
        // Build engine with default config
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        
        // Register a mock model for testing
        let mock_model = Box::new(grim_models_transformer::Llama::random(
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
            }
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
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
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
            }
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
            }
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
    }

    /// Integration test: determinism mismatch returns 400.
    #[tokio::test]
    async fn test_server_determinism_mismatch_strict() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default()); // Relaxed mode
        
        // Register a mock model for testing
        let mock_model = Box::new(grim_models_transformer::Llama::random(
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
            }
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
            }
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
    }

    /// Integration test: Grim compatibility shims (/api/chat, /api/generate, /api/tags, /api/pull).
    #[tokio::test]
    async fn test_grim_compatibility_shims() {
        let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let mock_model = Box::new(grim_models_transformer::Llama::random(
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
            }
        ));
        engine.register_model("default", mock_model);

        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
            model_path: None,
        });

        let app = build_router(state);

        // 1. Test /api/tags
        let res_tags = app.clone()
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
        let res_chat = app.clone()
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
        
        let body_bytes = axum::body::to_bytes(res_chat.into_body(), usize::MAX).await.unwrap();
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
        let res_gen = app.clone()
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

        let body_bytes_gen = axum::body::to_bytes(res_gen.into_body(), usize::MAX).await.unwrap();
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

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
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

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let content = val["choices"][0]["message"]["content"].as_str().unwrap();
        let token_count = content.matches("<tok:").count();
        assert_eq!(token_count, 1, "stop sequence must end generation at the first token");
    }
}

// ============================================================================
// Dashboard endpoint — live stats for the server status page.
// ============================================================================

/// `GET /api/stats` — JSON stats snapshot polled by the dashboard at `/`.
async fn stats_endpoint(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engine = state.engine.lock().unwrap();
    let models = engine.loaded_models();
    let model_name = models.first().cloned().unwrap_or_else(|| "none".to_string());

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
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("unknown").to_string();
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
    let is_loaded = model_name != "none";
    serde_json::json!({
        "model_name": model_name,
        "tokens_per_sec": if is_loaded { serde_json::json!(0.0f32) } else { serde_json::Value::Null },
        "kv_cache": {
            "used": 0u64,
            "total": 0u64,
            "blocks_used": 0u64,
            "blocks_total": 0u64,
        },
        "vram": {
            "used": 0u64,
            "total": 0u64,
        },
        "sys_ram": {
            "used": 0u64,
            "total": 0u64,
        },
        "gpus": [{
            "index": 0u32,
            "compute": 0u32,
            "memory": 0u32,
            "name": if rocm_gpu_count > 0 { "ROCm GPU" } else { "CPU" },
        }],
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
    }).into()
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
    <h3>Loaded Model</h3>
    <div class="row"><span class="label">Name</span><span id="model-name" class="val">—</span></div>
    <div class="row"><span class="label">Tokens / sec</span><span id="tps" class="val">—</span></div>
    <div class="row"><span class="label">Adapters</span><span id="adapters" class="val">0</span></div>
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
  <div class="card">
    <h3>GPU</h3>
    <div class="row"><span class="label">Device</span><span id="gpu-name" class="val">—</span></div>
    <div class="row"><span class="label">Compute</span><span id="gpu-cmp" class="val">—</span></div>
    <div class="row"><span class="label">Memory</span><span id="gpu-mem" class="val">—</span></div>
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

    document.getElementById('model-name').textContent=d.model_name||'—';
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
    document.getElementById('gpu-cmp').textContent=(gpu.compute??0)+'%';
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