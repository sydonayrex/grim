//! Grim HTTP server — axum-based, OpenAI-compatible endpoints.
//!
//! Phase 3 deliverable: minimal `/v1/chat/completions` that wires
//! an `Engine` and streams tokens back via SSE.

use std::sync::{Arc, Mutex};

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use grim_core::error::Result;
use grim_engine::Engine;

/// Shared engine state for the HTTP server.
pub struct AppState {
    pub engine: Mutex<Engine>,
}

/// Health-check endpoint.
async fn health() -> &'static str {
    "OK"
}

/// Placeholder chat completions endpoint.
async fn chat_completions(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let _body = body.as_object().cloned().unwrap_or_default();
    Json(serde_json::json!({
        "id": "chatcmpl-000",
        "object": "chat.completion",
        "created": 0,
        "model": "grim",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Grim engine running. Streaming not yet wired." },
            "finish_reason": "stop"
        }]
    }))
}

/// Build a new HTTP router with the given engine state.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

/// Start the server on `addr`.
pub async fn serve(addr: &str, engine: Engine) -> Result<()> {
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
    });
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| grim_core::Error::Config(format!("bind failed: {e}")))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| grim_core::Error::Config(format!("serve failed: {e}")))?;
    Ok(())
}