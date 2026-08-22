//! Integration tests for grim-garage web app static asset serving and API endpoints.

use axum::http::{Request, StatusCode};
use grim_garage::routes::{build_router, new_app_state};
use tower::ServiceExt;

#[tokio::test]
async fn test_serves_embedded_index_html() {
    let state = new_app_state();
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
    let state = new_app_state();
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

// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[tokio::test]
async fn test_chat_endpoint_fails_when_model_path_does_not_exist() {
    let state = new_app_state();
    let app = build_router(state);

    let payload = serde_json::json!({
        "model_id": "test-model.grim",
        "prompt": "Hello, model!",
        "temperature": 0.7
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::to_vec(&payload).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // The handler attempts on-demand model loading which fails (file not found)
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn test_start_training_accepts_weight_format() {
    let state = new_app_state();
    let app = build_router(state);

    let payload = serde_json::json!({
        "model_path": "models/test_model.grim",
        "dataset_path": "datasets/test_data.jsonl",
        "training_mode": "Lora",
        "weight_format": "crow",
        "lora_rank": 16,
        "learning_rate": 0.0001
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/train/start")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::to_vec(&payload).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn cannot_exceed_max_concurrent_jobs() {
    use grim_garage::jobs::JobRegistry;
    let registry = std::sync::Arc::new(JobRegistry::with_max_concurrent(1));
    let engine = std::sync::Arc::new(std::sync::Mutex::new(grim_engine::Engine::new(
        grim_engine::EngineConfig::default(),
    )));
    let state = grim_garage::routes::AppState {
        registry,
        engine,
        tokenizer: std::sync::Arc::new(std::sync::Mutex::new(None)),
        model_path: None,
    };
    let app = build_router(state);

    let payload = serde_json::json!({
        "model_path": "models/model1.grim",
        "dataset_path": "datasets/data1.jsonl",
        "training_mode": "Lora"
    });

    let resp1 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/train/start")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::to_vec(&payload).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp1.status(), StatusCode::OK);

    let payload2 = serde_json::json!({
        "model_path": "models/model2.grim",
        "dataset_path": "datasets/data2.jsonl",
        "training_mode": "Lora"
    });

    let resp2 = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/train/start")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::to_vec(&payload2).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn test_start_training_accepts_advanced_optimizers() {
    let optimizers = ["LOMO", "Adalomo", "CAME", "Sophia", "GaloreAdamW"];
    for opt in optimizers {
        let state = new_app_state();
        let app = build_router(state);

        let payload = serde_json::json!({
            "model_path": "models/model.grim",
            "dataset_path": "datasets/data.jsonl",
            "training_mode": "Lora",
            "optimizer": opt
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/train/start")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&payload).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "failed starting job with optimizer {opt}"
        );
    }
}
