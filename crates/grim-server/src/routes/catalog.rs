use crate::AppState;
use axum::{Json, extract::State};
use std::sync::Arc;

pub async fn grim_tags_route(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value> {
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

pub async fn get_model_route(
    model_id: String,
    State(state): State<Arc<AppState>>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    let engine = state.lock_engine();
    let models = engine.loaded_models();
    let default_name = engine.default_model_name().unwrap_or("default");
    if models.iter().any(|m| m == &model_id) || model_id == default_name || model_id == "default" {
        (
            axum::http::StatusCode::OK,
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
            axum::http::StatusCode::NOT_FOUND,
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

pub async fn delete_model_route(
    model_id: String,
    State(state): State<Arc<AppState>>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    let mut engine = state.lock_engine();
    let unloaded = engine.unload_model(&model_id);
    if unloaded {
        (
            axum::http::StatusCode::OK,
            Json(serde_json::json!({
                "id": model_id,
                "object": "model",
                "deleted": true
            })),
        )
    } else {
        (
            axum::http::StatusCode::NOT_FOUND,
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
