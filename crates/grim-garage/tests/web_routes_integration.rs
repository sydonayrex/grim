//! Integration tests for grim-garage web app static asset serving and API endpoints.

use axum::http::{Request, StatusCode};
use grim_garage::{jobs::JobRegistry, routes::{build_router, AppState}};
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn test_serves_embedded_index_html() {
    let state = AppState {
        registry: Arc::new(JobRegistry::new()),
    };
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.contains("text/html"));
}

#[tokio::test]
async fn test_serves_embedded_app_js() {
    let state = AppState {
        registry: Arc::new(JobRegistry::new()),
    };
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/app.js")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.contains("javascript"));
}

#[tokio::test]
async fn test_chat_endpoint_returns_ok() {
    let state = AppState {
        registry: Arc::new(JobRegistry::new()),
    };
    let app = build_router(state);

    let payload = serde_json::json!({
        "model_id": "/path/to/test-model.grim",
        "prompt": "Hello, model!",
        "temperature": 0.7
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
