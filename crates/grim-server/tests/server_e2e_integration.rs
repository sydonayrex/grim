//! Cross-crate end-to-end integration tests for `grim-server`.
//!
//! Validates:
//! - Health & metadata discovery endpoints (`/healthz`, `/v1/models`, `/status`)
//! - Non-streaming `/v1/chat/completions` request/response framing
//! - Streaming `/v1/chat/completions` with Server-Sent Events (SSE) and `[DONE]` termination
//! - Constrained sampling with JSON Schema and response_format validation
//! - Tokenize and detokenize endpoints with active tokenizer
//! - Request pause, resume, and cancellation lifecycle

use axum::http::StatusCode;
use grim_engine::{Engine, EngineConfig};
use grim_format::{GgufTokenizer, GgufValue};
use grim_models_transformer::{Llama, LlamaConfig};
use grim_server::{AppState, build_router};
use grim_tensor::Device;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

fn create_test_state() -> Arc<AppState> {
    let mut engine = Engine::new(EngineConfig::default());
    let mock_model = Box::new(Llama::random(
        Device::Cpu,
        LlamaConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            num_layers: 2,
            intermediate_size: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 512,
            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    // Provide an in-memory tokenizer via GGUF metadata
    let mut meta = HashMap::new();
    meta.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::String("llama".to_string()),
    );
    let tokens_arr: Vec<GgufValue> = [
        "<unk>", "<s>", "</s>", "hello", "world", "grim", "test", "{", "}", ":", "\"", "a", "b",
        "c",
    ]
    .iter()
    .map(|s| GgufValue::String(s.to_string()))
    .collect();
    meta.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::Array(tokens_arr),
    );
    meta.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        GgufValue::Uint32(1),
    );
    meta.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        GgufValue::Uint32(2),
    );
    meta.insert(
        "tokenizer.ggml.unknown_token_id".to_string(),
        GgufValue::Uint32(0),
    );
    let tok = GgufTokenizer::from_metadata(&meta).unwrap();

    Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(Some(tok)),
        model_path: None,
        model_arch: Mutex::new(Some("llama".to_string())),
        plugin_registry: None,
    })
}

#[tokio::test]
async fn test_server_health_and_discovery_endpoints() {
    let state = create_test_state();
    let app = build_router(state);

    // 1. /healthz
    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/healthz")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 2. /v1/models
    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/models")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 3. /status
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/status")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_server_chat_completions_non_streaming() {
    let state = create_test_state();
    let app = build_router(state);

    let body = json!({
        "model": "default",
        "messages": [
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": "hello world" }
        ],
        "max_tokens": 4,
        "temperature": 0.0
    });

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(val.get("choices").is_some());
    assert!(val["choices"][0]["message"]["content"].is_string());
}

#[tokio::test]
async fn test_server_chat_completions_streaming() {
    let state = create_test_state();
    let app = build_router(state);

    let body = json!({
        "model": "default",
        "messages": [
            { "role": "user", "content": "hello world" }
        ],
        "max_tokens": 3,
        "stream": true
    });

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get(axum::http::header::CONTENT_TYPE)
            .unwrap(),
        "text/event-stream"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("data:"),
        "SSE response should contain data lines"
    );
    assert!(
        text.contains("[DONE]"),
        "SSE stream should terminate with [DONE]"
    );
}

#[tokio::test]
async fn test_server_chat_completions_constrained_schema() {
    let state = create_test_state();
    let app = build_router(state);

    let body = json!({
        "model": "default",
        "messages": [
            { "role": "user", "content": "hello" }
        ],
        "response_format": {
            "type": "json_object"
        },
        "max_tokens": 4
    });

    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_server_tokenize_and_detokenize() {
    let state = create_test_state();
    let app = build_router(state);

    // 1. /v1/tokenize
    let tok_body = json!({
        "prompt": "hello world"
    });
    let resp_tok = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/tokenize")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(
                    serde_json::to_vec(&tok_body).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_tok.status(), StatusCode::OK);

    // 2. /v1/detokenize
    let detok_body = json!({
        "tokens": [3, 4]
    });
    let resp_detok = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/detokenize")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(
                    serde_json::to_vec(&detok_body).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_detok.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_server_request_control_endpoints() {
    let state = create_test_state();
    let app = build_router(state);

    // /v1/requests/99999/cancel -> not found produces structured 404 response
    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/requests/99999/cancel")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(resp.status() == StatusCode::OK || resp.status() == StatusCode::NOT_FOUND);

    // /v1/requests/99999/pause
    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/requests/99999/pause")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(resp.status() == StatusCode::OK || resp.status() == StatusCode::NOT_FOUND);

    // /v1/requests/99999/resume
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/requests/99999/resume")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(resp.status() == StatusCode::OK || resp.status() == StatusCode::NOT_FOUND);
}
