use crate::AppState;
use axum::{Json, extract::State, http::StatusCode};
use std::sync::Arc;

pub async fn health_handler() -> &'static str {
    "OK"
}

pub async fn healthz_handler() -> &'static str {
    "OK"
}

pub async fn readyz_handler(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
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
