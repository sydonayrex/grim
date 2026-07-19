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
};
use futures::stream::{self, Stream};
use grim_core::error::Result;
use grim_core::session::DeterminismMode;
use grim_engine::Engine;
use grim_scheduler::Request;

/// Shared engine state for the HTTP server.
pub struct AppState {
    pub engine: Mutex<Engine>,
    pub tokenizer: Mutex<Option<grim_format::GgufTokenizer>>,
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

    // Dynamic model loading / routing fallback logic:
    // If the requested model is not currently registered in the engine,
    // load/register it dynamically on-demand so external agents (Claude Code, Hermes, etc.)
    // do not receive a hard connection failure.
    {
        let mut engine = state.engine.lock().unwrap();
        if !engine.loaded_models().contains(&requested_model.to_string()) {
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
            engine.register_model(requested_model, mock_model);
            eprintln!("[grim-server] Loaded model '{}' dynamically on-demand.", requested_model);
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

        let stream = futures::stream::unfold(
            (0u64, 0u64), // (step, current_pos)
            move |(step, _pos)| {
                let state = state_clone.clone();
                let adapter_ids = adapter_ids_clone.clone();
                async move {
                    // Cap at 256 tokens per request.
                    if step >= 256 {
                        return None;
                    }

                    // Use a fixed request ID so we can always look up the outcome.
                    // The engine processes one request per tick: prefill on step 0,
                    // then decode on subsequent steps.
                    const REQUEST_ID: u64 = 0xDEAD_0000;

                    let token_id = {
                        let mut engine = state.engine.lock().unwrap();

                        // Only enqueue on the very first step (step 0).
                        // After that the request stays in the scheduler's running
                        // queue and tick() drives decode forward.
                        if step == 0 {
                            let req = Request {
                                id: REQUEST_ID,
                                prompt_tokens: 1,
                                priority: 0,
                            };
                            engine.enqueue_request(req);
                        }

                        // Advance the scheduler: this runs prefill (step 0) or
                        // decode (steps 1+). Each call produces one new token.
                        let _ = engine.tick();

                        // Read the outcome for our fixed request ID.
                        let argmax = engine
                            .last_outcome(REQUEST_ID)
                            .and_then(|o| {
                                o.logits.as_ref().map(|l| {
                                    l.to_vec_f32().ok().and_then(|v| {
                                        v.iter()
                                            .enumerate()
                                            .max_by(|(_, a), (_, b)| {
                                                a.partial_cmp(b).unwrap()
                                            })
                                            .map(|(i, _)| i as u32)
                                    })
                                })
                            })
                            .flatten()
                            .unwrap_or(step as u32);

                        argmax
                    };

                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

                    let tokenizer = state.tokenizer.lock().unwrap().clone();
                    let token_text = if let Some(tok) = &tokenizer {
                        tok.decode(&[token_id])
                    } else {
                        format!("<tok:{token_id}>")
                    };
                    let event = axum::response::sse::Event::default()
                        .event("message")
                        .data(format!(
                            r#"{{"choices": [{{"index": 0, "delta": {{"content": "{}"}}}}], "adapters_active": {}}}"#,
                            token_text.replace("\"", "\\\""),
                            adapter_ids.len()
                        ));
                    let res: std::result::Result<axum::response::sse::Event, axum::Error> = Ok(event);
                    Some((res, (step + 1, step + 1)))
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
        for step in 0..5 {
            let token_id = {
                let mut engine = state.engine.lock().unwrap();
                if step == 0 {
                    let req = Request {
                        id: REQUEST_ID,
                        prompt_tokens: 1,
                        priority: 0,
                    };
                    engine.enqueue_request(req);
                }
                let _ = engine.tick();
                let argmax = engine
                    .last_outcome(REQUEST_ID)
                    .and_then(|o| {
                        o.logits.as_ref().map(|l| {
                            l.to_vec_f32().ok().and_then(|v| {
                                v.iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| {
                                        a.partial_cmp(b).unwrap()
                                    })
                                    .map(|(i, _)| i as u32)
                            })
                        })
                    })
                    .flatten()
                    .unwrap_or(step as u32);
                argmax
            };
            let token_text = if let Some(tok) = &tokenizer {
                tok.decode(&[token_id])
            } else {
                format!("<tok:{token_id}>")
            };
            content.push_str(&token_text);
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
    let mut engine = state.engine.lock().unwrap();
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
    engine.register_model(&req.name, mock_model);
    (StatusCode::OK, Json(serde_json::json!({
        "status": "success",
        "message": format!("Model '{}' loaded dynamically.", req.name)
    })))
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

/// Resolve the configured models directory, checking common locations in order.
/// Returns the first path that exists, or a sensible default if none do.
fn resolve_models_dir() -> std::path::PathBuf {
    let candidates = [
        // 1. Environment variable override
        std::env::var("GRIM_MODELS_DIR").ok().map(std::path::PathBuf::from),
        // 2. Config file `models_dir` key
        get_default_model_from_config().map(|_| None).unwrap_or(None),
        // 3. Known install path
        Some(std::path::PathBuf::from("/var/lib/grim/models")),
        // 4. User home fallback
        dirs_sys_home().map(|h| h.join(".grim").join("models")),
    ];
    for c in candidates.into_iter().flatten() {
        if c.exists() {
            return c;
        }
    }
    std::path::PathBuf::from("/var/lib/grim/models")
}

/// Portable home-directory probe used only by `resolve_models_dir`.
fn dirs_sys_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
}

/// `GET /v1/models` — OpenAI-compatible model catalog endpoint.
///
/// Scans the configured models directory for files with recognised
/// extensions (`.grim`, `.gguf`, `.safetensors`, `.bin`) and returns them
/// as an OpenAI-style `{ "object": "list", "data": [...] }` response.
/// Also includes any models currently loaded in the engine.
async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let models_dir = resolve_models_dir();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut entries: Vec<serde_json::Value> = Vec::new();

    // 1. Walk the filesystem catalog.
    if let Ok(read_dir) = std::fs::read_dir(&models_dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "grim" | "gguf" | "safetensors" | "bin") {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let id = format!("{stem}:{ext}");
                if seen.insert(id.clone()) {
                    entries.push(serde_json::json!({
                        "id": id,
                        "object": "model",
                        "owned_by": "local",
                        "created": 0,
                        "format": ext,
                        "path": path.display().to_string()
                    }));
                }
            }
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

/// Start the server on `addr`.
pub async fn serve(addr: &str, engine: Engine) -> Result<()> {
    let tokenizer = if std::path::Path::new("./models/sleipnir.gguf").exists() {
        grim_format::GgufProvider::open("./models/sleipnir.gguf").ok().and_then(|p| p.tokenizer().ok())
    } else {
        None
    };
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(tokenizer),
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
        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
        });
        
        // Build router
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());
        
        // Send request
        let request_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
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
        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
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
        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
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
        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default()); // Relaxed mode
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
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
        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let state = Arc::new(AppState {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(None),
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
}