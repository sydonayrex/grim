use crate::AppState;
use axum::{Json, extract::State, http::StatusCode};
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TokenizeRequest {
    #[serde(default)]
    pub _model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub add_special_tokens: Option<bool>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DetokenizeRequest {
    #[serde(default)]
    pub _model: Option<String>,
    pub tokens: Vec<u32>,
}

pub async fn tokenize_route(
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

pub async fn detokenize_route(
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
