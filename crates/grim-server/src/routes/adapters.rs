use crate::AppState;
use axum::{Json, extract::State, http::StatusCode};
use std::sync::Arc;

pub async fn list_adapters_route(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
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

pub async fn unload_adapter_route(
    name: String,
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
pub struct LoadAdapterRequest {
    /// Human-readable name used in per-request `"adapters": [..]` routing.
    pub name: String,
    /// Sidecar written by `grim train` (`*.grim.train`). Required — this
    /// endpoint never fabricates weights.
    pub path: String,
    /// Base model the adapter targets; defaults to the default/first loaded.
    #[serde(default)]
    pub base_model: Option<String>,
}

/// Load a trained LoRA sidecar (`grim train` output) and register it for per-request routing WITHOUT an engine restart.
/// Runtime LoRA application (`lora.rs::apply_adapters_to_logits`) applies adapter pairs to the logits projection.
pub async fn load_adapter_route(
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
        // Runtime contract (lora.rs): A=[rank, in], B=[out, rank], applied at the logits projection.
        // Per-layer projections (q_proj/k_proj/…) never fit that site regardless of shapes.
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
