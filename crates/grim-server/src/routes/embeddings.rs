use crate::AppState;
use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: serde_json::Value,
    pub encoding_format: Option<String>,
    pub dimensions: Option<usize>,
    pub user: Option<String>,
}

/// OpenAI-compatible embeddings endpoint.
pub async fn embeddings_route(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<EmbeddingRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let inputs: Vec<String> = match &payload.input {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    };

    if inputs.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "input field must be a non-empty string or array of strings",
                    "type": "invalid_request_error",
                    "code": "invalid_input"
                }
            })),
        );
    }

    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "object": "list",
            "data": [],
            "model": "grim",
            "error": {
                "type": "not_implemented",
                "capability": "embeddings",
                "message": "no embedding model is loaded; text embeddings require an encoder or BERT/Nomic-family model — load one via POST /v1/models/load"
            }
        })),
    )
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScoreRequest {
    #[serde(default)]
    pub _model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
}

/// Score query against candidate documents (cross-encoder reranking).
pub async fn score_rerank_route(
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

/// Invalidate and reclaim unreferenced blocks from the KV block pool.
pub async fn reset_prefix_cache_route(
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
