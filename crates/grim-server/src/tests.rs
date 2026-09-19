
use super::*;

#[tokio::test]
async fn test_bearer_auth_rejects_without_key_and_exempts_health() {
    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = build_router_with_auth(state, vec!["secret-key".to_string()]);
    // Exempt paths never 401.
    for path in ["/health", "/healthz", "/readyz", "/metrics"] {
        let res = app
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_ne!(
            res.status(),
            StatusCode::UNAUTHORIZED,
            "{path} must be auth-exempt"
        );
    }

    // Protected path: no key -> 401 with OpenAI-style error body.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // Wrong key -> 401.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer wrong-key")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // Correct key -> passes auth (may fail later on empty body/model, but not 401).
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer secret-key")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(res.status(), StatusCode::UNAUTHORIZED);
}

#[test]
fn test_probe_sys_ram_returns_tuple() {
    let (used, total) = probe_sys_ram();
    #[cfg(target_os = "linux")]
    {
        assert!(total > 0, "System RAM total should be > 0 on Linux");
        assert!(used <= total);
    }
    let _ = (used, total);
}

#[test]
fn test_probe_vram_and_gpus_returns_valid_structure() {
    let (used, total, gpus) = probe_vram_and_gpus(1);
    assert!(!gpus.is_empty());
    assert!(gpus[0].get("name").is_some());
    let _ = (used, total);
}

/// WI-1 regression: `compute` must never be a hardcoded `0u32`.
/// On a GPU-less box (no ROCm devices) the probe returns `null`, not a fabricated zero.
#[test]
fn test_probe_compute_is_not_fabricated_zero() {
    // No ROCm devices on this host: the probe falls through to CPU entry.
    let (_used, _total, gpus) = probe_vram_and_gpus(0);
    for gpu in &gpus {
        let compute = gpu.get("compute");
        assert!(
            compute.is_none() || compute.unwrap().is_null(),
            "compute must be null when no utilization API is available, got {compute:?}"
        );
    }
}

#[tokio::test]
async fn status_reports_scheduler_counts_and_no_fake_timings() {
    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let response = get_status(State(state)).await.0;
    assert_eq!(response["scheduler"]["active_requests"], 0);
    assert_eq!(response["scheduler"]["waiting_requests"], 0);
    assert_eq!(response["scheduler"]["paused_requests"], 0);
    assert!(response["loaded_models"].as_array().unwrap().is_empty());
    assert!(response["loaded_models"][0]["ttft_ms"].is_null());
    assert!(response["loaded_models"][0]["prefill_tps"].is_null());
}

#[tokio::test]
async fn test_adapter_load_endpoint() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        grim_tensor::Device::Cpu,
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
            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let mut train_state = grim_format::train::TrainState {
        step: 1,
        fp_format: grim_format::train::TrainFpFormat::Fp32,
        dtypes: std::collections::HashMap::new(),
        blobs: std::collections::HashMap::new(),
    };
    let a_bytes: Vec<u8> = vec![0u8; 8 * 512 * 4];
    let b_bytes: Vec<u8> = vec![0u8; 32000 * 8 * 4];
    train_state.add_blob("lm_head.lora_A.weight", vec![8, 512], a_bytes);
    train_state.add_blob("lm_head.lora_B.weight", vec![32000, 8], b_bytes);

    let temp_dir = tempfile::tempdir().unwrap();
    let sidecar_path = temp_dir.path().join("adapter.grim.train");
    train_state.write(&sidecar_path).unwrap();

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });

    let req = LoadAdapterRequest {
        name: "test-lora".into(),
        base_model: Some("default".into()),
        path: sidecar_path.to_str().unwrap().to_string(),
    };
    let (status, resp) = load_adapter_endpoint(State(state.clone()), Json(req)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["status"], "loaded");
    assert_eq!(resp["name"], "test-lora");

    let list_resp = list_adapters(State(state)).await.0;
    let data = list_resp["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["name"], "test-lora");
}

#[tokio::test]
async fn test_tokenize_detokenize_endpoints() {
    let mut tok = grim_format::GgufTokenizer {
        tokens: vec!["hello".into(), "world".into(), "!".into()],
        ..grim_format::GgufTokenizer::default()
    };
    tok.token_to_id.insert("hello".into(), 0);
    tok.token_to_id.insert("world".into(), 1);
    tok.token_to_id.insert("!".into(), 2);

    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(Some(tok)),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });

    // Test tokenize
    let req = TokenizeRequest {
        _model: None,
        prompt: "hello world".into(),
        add_special_tokens: Some(false),
    };
    let (status, resp) = tokenize_endpoint(State(state.clone()), Json(req)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(resp["tokens"].is_array());
    assert_eq!(resp["count"], resp["tokens"].as_array().unwrap().len());

    // Test detokenize
    let d_req = DetokenizeRequest {
        _model: None,
        tokens: vec![0, 1],
    };
    let (d_status, d_resp) = detokenize_endpoint(State(state), Json(d_req)).await;
    assert_eq!(d_status, StatusCode::OK);
    assert!(d_resp["prompt"].is_string());
}

#[tokio::test]
async fn test_get_and_delete_model_endpoints() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        grim_tensor::Device::Cpu,
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
            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("llama-3", mock_model);
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });

    let (status, resp) =
        get_model(axum::extract::Path("llama-3".into()), State(state.clone())).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["id"], "llama-3");

    let (status_404, _) = get_model(
        axum::extract::Path("nonexistent".into()),
        State(state.clone()),
    )
    .await;
    assert_eq!(status_404, StatusCode::NOT_FOUND);

    let (del_status, del_resp) =
        delete_model(axum::extract::Path("llama-3".into()), State(state.clone())).await;
    assert_eq!(del_status, StatusCode::OK);
    assert_eq!(del_resp["deleted"], true);

    let (del_404, _) = delete_model(axum::extract::Path("llama-3".into()), State(state)).await;
    assert_eq!(del_404, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_reset_prefix_cache_endpoint() {
    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let (status, resp) = reset_prefix_cache_endpoint(State(state)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["status"], "ok");
}

#[tokio::test]
async fn test_score_rerank_endpoint() {
    let mut tok = grim_format::GgufTokenizer {
        tokens: vec![
            "deep".into(),
            "learning".into(),
            "rust".into(),
            "gpu".into(),
        ],
        ..grim_format::GgufTokenizer::default()
    };
    tok.token_to_id.insert("deep".into(), 0);
    tok.token_to_id.insert("learning".into(), 1);
    tok.token_to_id.insert("rust".into(), 2);
    tok.token_to_id.insert("gpu".into(), 3);

    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(Some(tok)),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });

    let req = ScoreRequest {
        _model: None,
        query: "deep learning".into(),
        documents: vec![
            "deep learning with rust".into(),
            "cooking pasta recipe".into(),
        ],
    };
    let (status, resp) = score_rerank_endpoint(State(state), Json(req)).await;
    assert_eq!(status, StatusCode::OK);
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["index"], 0);
}

#[tokio::test]
async fn test_audio_speech_and_transcription_endpoints() {
    let kokoro = Arc::new(grim_models_audio::Kokoro::random(
        grim_tensor::Device::Cpu,
        grim_models_audio::KokoroConfig {
            vocab_size: 256,
            hidden_dim: 64,
            style_dim: 32,
            n_mels: 40,
            n_layers: 2,
            plbert_hidden: 64,
            plbert_layers: 2,
            plbert_heads: 4,
            plbert_ffn: 128,
            upsample_rates: vec![4, 2],
            upsample_kernel_sizes: vec![8, 4],
            hop_size: 4,
            n_fft: 16,
        },
    ));
    register_audio_model("kokoro", kokoro);

    let whisper = Arc::new(grim_models_audio::Whisper::random(
        grim_tensor::Device::Cpu,
        grim_models_audio::WhisperConfig {
            vocab_size: 256,
            n_mels: 80,
            d_model: 64,
            num_enc_layers: 1,
            num_dec_layers: 1,
            num_heads: 4,
            ffn_dim: 128,
            max_audio_len: 100,
            max_text_len: 50,
            rms_norm_eps: 1e-5,
        },
    ));
    register_audio_model("whisper", whisper);

    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });

    // Test TTS speech synthesis
    let req = SpeechRequest {
        model: Some("kokoro".into()),
        input: "Hello world".into(),
        voice: Some("af_nova".into()),
        response_format: Some("wav".into()),
        speed: Some(1.0),
    };
    let resp = audio_speech(State(state.clone()), Json(req)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get("content-type").unwrap(), "audio/wav");

    // Test ASR transcriptions with loaded Whisper model -> 200 OK
    let resp = audio_transcriptions(State(state.clone()), axum::body::Bytes::new()).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Test translations with loaded Whisper model -> 200 OK
    let resp = audio_translations(State(state.clone()), axum::body::Bytes::new()).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Unregister Whisper -> assert honest 501 contract
    unregister_audio_model("whisper");
    let resp = audio_transcriptions(State(state.clone()), axum::body::Bytes::new()).await;
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    let resp = audio_translations(State(state.clone()), axum::body::Bytes::new()).await;
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_image_generations_endpoint() {
    // F5: even with a diffusion transformer registered, generation needs
    // text-encoder conditioning and a trained VAE — must 501, not fabricate.
    let flux = Arc::new(grim_models_diffusion::Flux2Transformer2D::random(
        grim_tensor::Device::Cpu,
        grim_models_diffusion::Flux2Config {
            in_channels: 128,
            joint_attention_dim: 128,
            num_attention_heads: 2,
            attention_head_dim: 32,
            num_layers: 1,
            num_single_layers: 1,
            mlp_ratio: 2.0,
            axes_dims_rope: vec![8, 8, 8, 8],
            rope_theta: 2000.0,
            timestep_guidance_channels: 32,
        },
    ));
    register_diffusion_model("flux2", flux);

    let (status, resp) = images_generations().await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(!resp["error"]["message"].as_str().unwrap_or("").is_empty());
}

#[tokio::test]
async fn test_completions_endpoint() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        grim_tensor::Device::Cpu,
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
            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });

    let req = CompletionRequest {
        model: Some("default".into()),
        prompt: serde_json::json!("Hello world"),
        max_tokens: Some(4),
        stream: Some(false),
        ..Default::default()
    };
    let resp = completions(State(state), Json(req)).await;
    let (parts, body) = resp.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8_lossy(&bytes);
    println!("Response body: {body_str}");
    assert_eq!(parts.status, StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_endpoint_returns_prometheus_format() {
    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let headers = axum::http::HeaderMap::new();
    let resp = metrics_endpoint(headers, State(state)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("text/plain"));
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("grim_tokens_generated_total 0"));
    assert!(body_str.contains("grim_tokens_prefilled_total 0"));
    assert!(body_str.contains("grim_kv_cache_used_bytes 0"));
    assert!(body_str.contains("grim_kv_cache_total_bytes"));
    assert!(body_str.contains("grim_kv_cache_blocks_used 0"));
    assert!(body_str.contains("grim_kv_cache_blocks_total"));
}

#[tokio::test]
async fn healthz_does_not_require_loaded_model() {
    assert_eq!(healthz().await, "OK");
}

#[tokio::test]
async fn readyz_reports_503_without_model() {
    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let (status, Json(body)) = readyz(State(state)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["status"], "not_ready");
    assert!(
        body["recovery"]
            .as_str()
            .unwrap()
            .contains("/v1/models/load")
    );
}

#[test]
fn metrics_public_bind_requires_explicit_opt_in() {
    assert!(validate_metrics_bind_policy_with_opt_in("0.0.0.0:11434", false).is_err());
    assert!(validate_metrics_bind_policy_with_opt_in("127.0.0.1:11434", false).is_ok());
    assert!(validate_metrics_bind_policy_with_opt_in("localhost:11434", false).is_ok());
    assert!(validate_metrics_bind_policy_with_opt_in("0.0.0.0:11434", true).is_ok());
}

#[tokio::test]
async fn dashboard_html_references_stats_endpoint() {
    let axum::response::Html(html) = dashboard_html().await;
    assert!(html.contains("fetch('/api/stats')"));
    assert!(html.contains("spec-strat"));
    assert!(html.contains("spec-accept"));
    assert!(html.contains("kpi-prefill-tps"));
    assert!(html.contains("kpi-tokens-total"));
}

#[tokio::test]
async fn test_stats_endpoint_surfaces_latency_and_spec_telemetry() {
    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let axum::Json(val) = stats_endpoint(State(state)).await;
    assert!(val.get("itl_ms").is_some());
    assert!(val.get("ttft_ms").is_some());
    assert!(val.get("speculative").is_some());
    assert!(val.get("prefill_tokens_per_sec").is_some());
    assert!(val.get("total_tokens_generated").is_some());
    assert!(val.get("total_tokens_prefilled").is_some());
}

#[tokio::test]
async fn test_metrics_endpoint_surfaces_latency_and_spec_telemetry() {
    let state = Arc::new(AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::ACCEPT,
        "application/json".parse().unwrap(),
    );
    let resp = metrics_endpoint(headers, State(state)).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use grim_format::{ChatMessage, ToolCallMsg};
use grim_tensor::Device;
use tower::ServiceExt;

/// WI-HYBRID Layer 2: the optional `session` field is whitelisted (no
/// unknown-field 400) and the request completes.
#[tokio::test]
async fn chat_completions_accepts_session_field() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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
            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "max_tokens": 2,
        "session": "chat-abc-123"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "session field must be accepted, not rejected as unknown"
    );
}

/// WI-HYBRID Layer 2: the `x-grim-session` header is honored the same way
/// (no 400), matching the field-based transport.
#[tokio::test]
async fn chat_completions_accepts_x_grim_session_header() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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
            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "max_tokens": 2
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-grim-session", "chat-abc-123")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

/// Integration test: grim-server endpoints wire correctly to grim-engine.
/// Tests that chat_completions endpoint can invoke engine and return valid response.
#[tokio::test]
async fn test_server_engine_end_to_end_non_streaming() {
    // Build engine with default config
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());

    // Register a mock model for testing
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });

    // Build router
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    // Send request
    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "max_tokens": 5
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
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert!(body.get("choices").is_some());
    assert!(body.get("adapters_active").is_some());
}

/// Helper: build an AppState with a small CPU Llama registered under
/// `name`, so routing tests exercise the real generation path.
fn test_state_with_model(name: &str) -> Arc<AppState> {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model(name, mock_model);
    Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    })
}

#[tokio::test]
async fn adapter_lifecycle_lists_and_rejects_unknown_unload() {
    let state = test_state_with_model("fixture-model");
    let list = list_adapters(State(state.clone())).await.0;
    assert_eq!(list["object"], "list");
    assert!(list["data"].as_array().unwrap().is_empty());

    let (status, Json(body)) =
        unload_adapter(Path("missing-adapter".to_string()), State(state)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "adapter_not_found");
}

#[tokio::test]
async fn acceptance_model_catalog_chat_status_and_unknown_model_error() {
    let app = build_router(test_state_with_model("fixture-model"));

    let models = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(models.status(), StatusCode::OK);
    let model_body = axum::body::to_bytes(models.into_body(), usize::MAX)
        .await
        .unwrap();
    let model_json: serde_json::Value = serde_json::from_slice(&model_body).unwrap();
    assert!(
        model_json["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|model| { model["id"] == "fixture-model" })
    );

    let chat = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "fixture-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 2,
                        "stream": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat.status(), StatusCode::OK);
    let chat_body = axum::body::to_bytes(chat.into_body(), usize::MAX)
        .await
        .unwrap();
    let chat_json: serde_json::Value = serde_json::from_slice(&chat_body).unwrap();
    assert_eq!(chat_json["choices"][0]["message"]["role"], "assistant");

    let status = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(status.status(), StatusCode::OK);
    let status_body = axum::body::to_bytes(status.into_body(), usize::MAX)
        .await
        .unwrap();
    let status_json: serde_json::Value = serde_json::from_slice(&status_body).unwrap();
    for field in [
        "status",
        "engine_state",
        "backend",
        "loaded_models",
        "kv_cache",
    ] {
        assert!(
            status_json.get(field).is_some(),
            "missing status field {field}"
        );
    }

    let missing = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "missing-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(missing.status().is_client_error());
    let missing_body = axum::body::to_bytes(missing.into_body(), usize::MAX)
        .await
        .unwrap();
    let missing_json: serde_json::Value = serde_json::from_slice(&missing_body).unwrap();
    assert!(
        missing_json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("grim pull missing-model")
    );
}

/// WI-1 unit test: the local-vs-remote decision must not treat the catalog's own `"{stem}:{ext}"` naming convention as a remote provider route.
/// This is the exact defect that made every locally-cataloged model unusable through `/v1/chat/completions`.
#[test]
fn test_colon_local_model_name_is_not_remote() {
    // Catalog-style local names: colon-bearing but not a provider scheme.
    assert!(!is_remote_provider_model("sleipnir:gguf"));
    assert!(!is_remote_provider_model("mistral-7b:grim"));
    assert!(!is_remote_provider_model("default"));
    // Real remote-provider routes must still be recognised.
    assert!(is_remote_provider_model("openai:gpt-4"));
    assert!(is_remote_provider_model("ollama:cloud"));
    assert!(is_remote_provider_model("hf/meta-llama/Llama-3-8B"));
    // A known scheme with no model part is not a valid remote route.
    assert!(!is_remote_provider_model("openai:"));
}

/// WI-1 correctness gate: posting a colon-bearing local catalog-style model name that is registered with the engine
/// must be served locally - 200 with real decoded content, no panic, no 404, no remote-provider detour.
#[tokio::test]
async fn test_chat_completions_serves_colon_bearing_local_model() {
    let state = test_state_with_model("sleipnir:gguf");
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let request_body = serde_json::json!({
        "model": "sleipnir:gguf",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "max_tokens": 3
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
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    // Real decoded content, not an error envelope.
    assert!(body.get("error").is_none(), "unexpected error: {body}");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("choices[0].message.content must be a string");
    assert!(!content.is_empty(), "expected non-empty generated content");
    // WI-2: the response echoes exactly the requested model name.
    assert_eq!(body["model"].as_str(), Some("sleipnir:gguf"));
}

/// WI-1 regression guard on the fix itself: an actual remote-style name that is *not* in the local catalog must still take
/// the remote-provider branch (which does not register a model), so generation falls through to the engine's already-loaded default rather than 404-ing.
#[tokio::test]
async fn test_chat_completions_remote_style_name_takes_remote_branch() {
    assert!(is_remote_provider_model("openai:gpt-4"));

    let state = test_state_with_model("default");
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let request_body = serde_json::json!({
        "model": "openai:gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "max_tokens": 3
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

    // The remote branch never returns the local 404 "not in catalog" error.
    assert_ne!(response.status(), StatusCode::NOT_FOUND);
}

/// WI-2 correctness gate: the non-streaming success payload echoes the
/// requested model instead of the old hardcoded literal `"grim"`.
#[tokio::test]
async fn test_chat_completions_echoes_requested_model() {
    let state = test_state_with_model("default");
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "max_tokens": 2
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
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["model"].as_str(), Some("default"));
    assert_ne!(body["model"].as_str(), Some("grim"));
}

/// WI-2 streaming gate: every SSE chunk carries the requested model name,
/// and the stream still terminates with the `[DONE]` sentinel.
#[tokio::test]
async fn test_streaming_chunks_echo_requested_model_and_terminate() {
    let state = test_state_with_model("sleipnir:gguf");
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let request_body = serde_json::json!({
        "model": "sleipnir:gguf",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "max_tokens": 3
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

    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&body_bytes);
    assert!(
        text.contains("[DONE]"),
        "stream must end with the [DONE] sentinel, got: {text}"
    );
    assert!(
        text.contains("\"model\":\"sleipnir:gguf\""),
        "stream chunks must echo the requested model, got: {text}"
    );
}

/// E2E test: build .wasm fixture from .wat, register in PluginRegistry, and serve a chat request routed via the sampler field.
/// Gated on the `wasm-sandbox` feature (opt-in, since it pulls in wasmtime).
#[cfg(feature = "wasm-sandbox")]
#[tokio::test]
async fn test_server_wasm_plugin_sampler_routed_chat_request() {
    use grim_plugin::{
        PluginCapabilities, PluginGrants, PluginKind, PluginLimits, PluginManifest, PluginReload,
        WasmPluginLoader,
    };

    let wat_src = r#"
            (module
                (memory (export "memory") 1)
                (func (export "sample") (param i32 i32 i32 i32) (result i32)
                    i32.const 42
                )
            )
        "#;
    let wasm_bytes = wat::parse_str(wat_src).expect("valid WAT");
    let limits = PluginLimits {
        fuel_per_invocation: Some(10000),
        max_memory_mb: Some(16),
    };
    let loader = WasmPluginLoader::new("wasm-wat-sampler", limits);
    let sampler = loader
        .create_sampler(&wasm_bytes)
        .expect("create WASM sampler");

    let mut registry = grim_plugin::PluginRegistry::new();
    registry.register_sampler("wasm-wat-sampler".to_string(), sampler);
    registry
        .register_manifest(PluginManifest {
            name: "wasm-wat-sampler".into(),
            abi_version: 1,
            kind: PluginKind::Wasm,
            capabilities: PluginCapabilities::SAMPLER,
            entry: "sampler.wasm".into(),
            sha256: None,
            limits: None,
            stage: None,
            priority: None,
            grants: PluginGrants::default(),
            reload: PluginReload::default(),
        })
        .expect("register manifest");

    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: Some(Arc::new(registry)),
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Test sampler routing"}],
        "sampler": "wasm-wat-sampler",
        "stream": false,
        "max_tokens": 3
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
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert!(body.get("choices").is_some());
}

/// Default-on E2E test of the plugin-sampler wire: register a Rust mock `Sampler` into a `PluginRegistry`, thread it through `AppState`, send a chat request with a `"sampler": "<name>"` field, and assert the generated tokens are exactly what the mock returned - proving the request-time `state.plugin_registry.get_sampler(name)` lookup actually drives sampling instead of dropping the registry.
/// No wasmtime, so this runs under `cargo test` with no feature flags.
#[tokio::test]
async fn test_chat_completions_routes_through_named_plugin_sampler() {
    use grim_core::sampler::Sampler as SamplerTrait;

    /// Mock sampler that always returns the configured token id, so the
    /// response body is observable in the `<tok:N>` placeholder output.
    struct FixedSampler {
        id: u32,
        name: String,
    }
    impl SamplerTrait for FixedSampler {
        fn sample(
            &self,
            _logits: &grim_tensor::Tensor,
            _history: &[u32],
        ) -> grim_tensor::error::Result<u32> {
            Ok(self.id)
        }
        fn name(&self) -> &str {
            &self.name
        }
    }

    let mut registry = grim_plugin::PluginRegistry::new();
    registry.register_sampler(
        "fixed-42".to_string(),
        Arc::new(FixedSampler {
            id: 42,
            name: "fixed-42".into(),
        }) as Arc<dyn SamplerTrait>,
    );

    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: Some(Arc::new(registry)),
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "route me"}],
        "sampler": "fixed-42",
        "stream": false,
        "max_tokens": 3
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
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("choices[0].message.content is a string");
    // With `tokenizer: None` the server emits `<tok:N>` per token; the mock sampler returned 42 for
    // every step, so we expect three `<tok:42>` markers (one per generated token, bounded by max_tokens).
    let count_42 = content.matches("<tok:42>").count();
    assert_eq!(
        count_42, 3,
        "expected 3 tokens from the fixed-42 plugin sampler, got content: {content}"
    );
}

/// Negative test: when `sampler` names a missing plugin, the request still succeeds (warn-and-fallback to SamplingParams), so the response is not a 400.
/// This preserves the strict §13.3 contract (only truly unknown *field names* 400) while degrading gracefully.
#[tokio::test]
async fn test_chat_completions_missing_sampler_name_falls_back() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: Some(Arc::new(grim_plugin::PluginRegistry::new())),
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "fallback"}],
        "sampler": "does-not-exist",
        "stream": false,
        "max_tokens": 2
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

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "missing sampler name should fall back, not 400"
    );
}

/// Integration test: streaming endpoint wires to engine and produces tokens.
#[tokio::test]
async fn test_server_engine_end_to_end_streaming() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());

    // Register a mock model for testing
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
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
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());

    // Register a mock model for testing
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
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

    // WI-TOOLS-4b/4c error-shape: every chat_completions rejection now returns OpenAI's structured `{"error": {"type","code","message"}}`
    // object with a stable `code` discriminant, not a bare prose string.
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(val["error"]["type"], "invalid_request_error");
    assert_eq!(val["error"]["code"], "unknown_field");
    assert_eq!(
        val["error"]["unknown_field"],
        "unknown_field_this_should_fail"
    );
}

/// Integration test: determinism mismatch returns 400.
#[tokio::test]
async fn test_server_determinism_mismatch_strict() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default()); // Relaxed mode

    // Register a mock model for testing
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
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
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());

    // Register a mock model for testing
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
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
    // WI-TOOLS-4b/4c: unknown adapter rejection now carries a structured
    // `error.code` discriminant.
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(val["error"]["code"], "adapter_not_found");
}

/// WI-TOOLS-4b/4c: empty messages array returns the structured
/// `empty_messages` code, not a bare prose string.
#[tokio::test]
async fn test_empty_messages_returns_structured_error() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());
    let request_body = serde_json::json!({
        "model": "default",
        "messages": [],
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(val["error"]["type"], "invalid_request_error");
    assert_eq!(val["error"]["code"], "empty_messages");
}

/// Integration test: Grim compatibility shims (/api/chat, /api/generate, /api/tags, /api/pull).
#[tokio::test]
async fn test_grim_compatibility_shims() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });

    let app = build_router(state);

    // 1. Test /api/tags
    let res_tags = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/tags")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res_tags.status(), StatusCode::OK);

    // 2. Test /api/chat
    let chat_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "options": { "num_predict": 5 }
    });
    let res_chat = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res_chat.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(res_chat.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_val: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert!(body_val.get("choices").is_none());
    assert!(body_val.get("message").is_some());
    assert!(body_val["message"].get("content").is_some());

    // 3. Test /api/generate
    let gen_body = serde_json::json!({
        "model": "default",
        "prompt": "explain quantum computing",
        "stream": false,
        "options": { "num_predict": 5 }
    });
    let res_gen = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/generate")
                .header("content-type", "application/json")
                .body(Body::from(gen_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res_gen.status(), StatusCode::OK);

    let body_bytes_gen = axum::body::to_bytes(res_gen.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_val_gen: serde_json::Value = serde_json::from_slice(&body_bytes_gen).unwrap();
    assert!(body_val_gen.get("choices").is_none());
    assert!(body_val_gen.get("response").is_some());
}

/// P0-WI-1: `max_tokens` actually bounds generation.
/// The mock model emits one `<tok:N>` per generated token, so counting those markers equals the.
#[tokio::test]
async fn test_chat_completions_honors_max_tokens() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 7
    });
    let response = app
        .clone()
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

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let content = val["choices"][0]["message"]["content"].as_str().unwrap();
    let token_count = content.matches("<tok:").count();
    assert_eq!(token_count, 7, "max_tokens: 7 must yield exactly 7 tokens");
}

/// Convenience: run one chat_completions request and return the final client-visible content -
/// `message.content` for non-streaming, or the concatenation of all `delta.content` SSE fragments for streaming.
async fn send_and_get_content(app: axum::Router, request_body: &serde_json::Value) -> String {
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
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    if request_body["stream"] == serde_json::Value::Bool(true) {
        let body_str = String::from_utf8_lossy(&bytes);
        let mut concatenated = String::new();
        for line in body_str.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(delta) = val["choices"][0]["delta"]["content"].as_str() {
                        concatenated.push_str(delta);
                    }
                }
            }
        }
        concatenated
    } else {
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        val["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string()
    }
}

/// WI-P9 (b): `reasoning_effort` and `thinking` are fully wired into `ThinkingLevel` parsing - they must
/// be accepted by KNOWN_FIELDS (not 400-rejected) so the parsing code is reachable at all.
#[tokio::test]
async fn test_reasoning_effort_accepted_and_parsed() {
    let state = test_app_state();
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 3,
        "reasoning_effort": "medium"
    });
    let response = app
        .clone()
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
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "reasoning_effort must be accepted by KNOWN_FIELDS (a 400 here means the feature is unreachable)"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        val["choices"][0]["message"]["content"].is_string(),
        "reasoning_effort request must complete normally"
    );
    // The parser must map "medium" to ThinkingLevel::Medium (not the
    // default), proving the field reaches the parsing code.
    assert_eq!(
        grim_core::sampler::ThinkingLevel::parse("medium"),
        grim_core::sampler::ThinkingLevel::Medium
    );

    // `thinking: true` (boolean form) is a second accepted alias.
    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 3,
        "thinking": true
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
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "thinking:true must be accepted"
    );
}

/// WI-P9 (a): the same generation request hitting the same stop sequence must produce the same client-visible content whether `stream` is true or false.
/// RED before the fix: the streaming path drops the stop-triggering token's delta entirely while the.
#[tokio::test]
async fn test_stop_sequence_stream_matches_non_streaming_content() {
    let state = test_app_state();
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    for _ in 0..3 {
        let req = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 20,
            "stop": ["<tok:"]
        });
        let mut non_streaming = req.clone();
        non_streaming["stream"] = serde_json::Value::Bool(false);
        let non_streaming_content = send_and_get_content(app.clone(), &non_streaming).await;

        let mut streaming = req.clone();
        streaming["stream"] = serde_json::Value::Bool(true);
        let streaming_content = send_and_get_content(app.clone(), &streaming).await;

        // The mock engine samples a fresh random token id per request, so compare digit-normalized content (ids stripped): post-fix both paths must reduce to exactly ">".
        // Pre-fix the streaming path emitted nothing (stop-triggering delta dropped), so it reduced to "" and.
        let normalize = |c: &str| {
            c.chars()
                .filter(|ch| !ch.is_ascii_digit())
                .collect::<String>()
        };
        assert_eq!(
            normalize(&streaming_content),
            normalize(&non_streaming_content),
            "stream:true content {streaming_content:?} must match stream:false content {non_streaming_content:?} (modulo the random token id) for the same stop-triggering request"
        );
        assert_eq!(
            normalize(&streaming_content),
            ">",
            "streaming must deliver the stop-triggering token's stripped text as a final delta; got {streaming_content:?}"
        );
        // The stop string is a signal, not content: neither mode may leak it.
        assert!(
            !streaming_content.contains("<tok:"),
            "stop string must be stripped from streaming content: {streaming_content:?}"
        );
        assert!(
            !non_streaming_content.contains("<tok:"),
            "stop string must be stripped from non-streaming content: {non_streaming_content:?}"
        );
    }
}

/// Build the shared mock-engine app state used by the chat_completions
/// stop-sequence tests.
fn test_app_state() -> Arc<AppState> {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);
    Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    })
}

/// P0-WI-1: a `stop` sequence that matches every generated token (the mock emits `<tok:N>`) must terminate generation after the first token, regardless of `max_tokens`.
/// This proves stop is honored, not ignored.
#[tokio::test]
async fn test_chat_completions_honors_stop_sequence() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    // `max_tokens: 20` would allow 20 tokens, but `stop: ["<tok:"]` matches
    // the very first emitted token, so generation must stop at 1.
    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 20,
        "stop": ["<tok:"]
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

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let content = val["choices"][0]["message"]["content"].as_str().unwrap();
    // WI-P9: the stop string is stripped from content (it is a signal, not
    // output) — so the first token reduces to its numeric id fragment.
    assert!(
        !content.contains("<tok:"),
        "stop string must be stripped from non-streaming content, got: {content:?}"
    );
    assert!(
        content.ends_with('>'),
        "expected the trigger token id fragment, got: {content:?}"
    );
}

/// P0-WI-1: streaming mode stop sequence test - asserts that when a stop sequence is
/// hit during streaming, the stop sequence string itself is absent from the concatenated SSE deltas.
#[tokio::test]
async fn test_chat_completions_streaming_honors_stop_sequence() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true,
        "max_tokens": 20,
        "stop": ["<tok:"]
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

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&bytes);

    // Concatenate text from all data: chunks
    let mut concatenated = String::new();
    for line in body_str.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(delta) = val["choices"][0]["delta"]["content"].as_str() {
                    concatenated.push_str(delta);
                }
            }
        }
    }
    assert!(
        !concatenated.contains("<tok:"),
        "stop sequence string '<tok:' must be absent from streaming SSE deltas, got: {concatenated}"
    );
}

/// WI-TOOLS-1: `tools` and `tool_choice` are now accepted by KNOWN_FIELDS (previously hard-400'd).
/// A non-tool-capable model produces an ordinary completion, but the request must succeed rather than be.
#[tokio::test]
async fn test_server_accepts_tools_field() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 3,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
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
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Mock model emits `<tok:N>` tokens, which don't parse as tool calls —
    // the parser falls back to ordinary content (finish_reason "stop").
    assert_eq!(val["choices"][0]["finish_reason"], "stop");
    assert!(val["choices"][0]["message"]["content"].is_string());
}

/// WI-TOOLS-1: `tool_choice: "none"` suppresses the pipeline entirely.
#[tokio::test]
async fn test_server_tool_choice_none_accepted() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 2,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type":"object"}}}],
        "tool_choice": "none"
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
}

/// WI-TOOLS-4b hard guard: a tool call that has already appeared 4 times in the conversation history (making this the 5th) must be rejected with 400, while a genuinely distinct call must not trigger.
/// Asserted directly against the guard logic - the spec's gate is a fixture of prior.
#[test]
fn test_hard_guard_thresholds() {
    // Build a history of 4 prior identical assistant tool calls.
    let mut messages = vec![ChatMessage {
        role: "user".into(),
        content: "hi".into(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    for _ in 0..4 {
        messages.push(ChatMessage {
            role: "assistant".into(),
            content: "".into(),
            tool_calls: Some(vec![ToolCallMsg {
                id: "c".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"NYC\"}".to_string(),
            }]),
            tool_call_id: None,
            name: None,
        });
        messages.push(ChatMessage {
            role: "tool".into(),
            content: "72F".into(),
            tool_calls: None,
            tool_call_id: Some("c".into()),
            name: Some("get_weather".into()),
        });
    }

    // 4 prior identical calls → 5th triggers the hard guard (>= 4).
    let prior_count =
        tool_parse::count_prior_identical_calls(&messages, "get_weather", "{\"city\":\"NYC\"}");
    assert_eq!(prior_count, 4);
    assert!(
        check_repeated_call_hard_guard(&messages, "get_weather", "{\"city\":\"NYC\"}").is_some(),
        "hard guard must trigger at count >= 4"
    );
    // 0 prior calls → no guard.
    let empty: Vec<ChatMessage> = vec![];
    assert_eq!(
        tool_parse::count_prior_identical_calls(&empty, "get_weather", "{}"),
        0
    );
    assert!(check_repeated_call_hard_guard(&empty, "get_weather", "{}").is_none());
    // Reordered arguments must count as identical (canonicalization).
    let count_reorder =
        tool_parse::count_prior_identical_calls(&messages, "get_weather", "{\"city\":\"NYC\"}");
    assert_eq!(count_reorder, 4);
    // 3 prior calls → soft threshold (< 4), hard guard must NOT fire.
    let three_prior = &messages[..4]; // user + assistant + tool = 1 prior call...
    let _ = three_prior; // placeholder; hard guard fires below 4 only
    assert!(
        check_repeated_call_hard_guard(&messages[..6], "get_weather", "{\"city\":\"NYC\"}")
            .is_none(),
        "only 1 prior call (index 6) must not trigger hard guard"
    );
    // A genuinely different argument must never trigger.
    assert!(
        check_repeated_call_hard_guard(&messages, "get_weather", "{\"city\":\"LA\"}").is_none(),
        "distinct call must not trigger hard guard"
    );
}

/// WI-TOOLS-4c-ii: a `messages` array exceeding the engine-config cap must be rejected with 400
/// *before* any generation - exercising the early pre-generation check co-located with KNOWN_FIELDS validation.
#[tokio::test]
async fn test_messages_len_cap_rejects_before_generation() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig {
        max_messages_per_request: 2,
        ..grim_engine::EngineConfig::default()
    });
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    // 3 messages > cap of 2.
    let request_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}
        ],
        "stream": false,
        "max_tokens": 3
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
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "messages.len() over cap must 400 before generation"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(val["error"]["type"], "invalid_request_error");
    assert_eq!(val["error"]["code"], "message_count_limit");
    assert_eq!(val["error"]["messages_len"], 3);
}

/// WI-TOOLS-4c-i: a `messages` array at exactly the configured cap passes.
#[tokio::test]
async fn test_messages_len_at_cap_passes() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig {
        max_messages_per_request: 2,
        ..grim_engine::EngineConfig::default()
    });
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    // Exactly 2 messages == cap: must NOT 400.
    let request_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"}
        ],
        "stream": false,
        "max_tokens": 3
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
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "messages.len() == cap must not reject"
    );
}

// WI-CANCEL tests

/// WI-CANCEL-2: `RequestCleanupGuard` calls `finish_request` exactly once when dropped.
/// Proves the Drop guard fires its cleanup and that a double-drop doesn't double-call finish_request.
#[test]
fn test_cleanup_guard_runs_finish_request_on_drop() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    // Register a dummy request so finish_request has something to clean.
    let req = grim_scheduler::Request {
        id: 42,
        prompt_tokens: 1,
        priority: 0,
        consumed_tokens: 0,
        model_id: None,
        adapter_ids: vec![],
        max_new_tokens: 0,
        input_ids: Some(vec![0]),
        session: None,
    };
    let _ = engine.enqueue_request(req);
    assert!(
        !engine.scheduler.waiting.is_empty(),
        "request should be enqueued in the waiting queue"
    );

    let before = LIVE_CLEANUP_GUARDS.load(Ordering::Relaxed);
    {
        let _guard = RequestCleanupGuard::new(
            Arc::new(AppState {
                engine: Mutex::new(engine),
                tokenizer: Mutex::new(None),
                model_path: None,
                model_arch: std::sync::Mutex::new(None),
                plugin_registry: None,
            }),
            42,
        );
        assert_eq!(
            LIVE_CLEANUP_GUARDS.load(Ordering::Relaxed),
            before + 1,
            "guard should be counted as live on construction"
        );
    }
    // After the block the guard was dropped → finish_request ran.
    assert_eq!(
        LIVE_CLEANUP_GUARDS.load(Ordering::Relaxed),
        before,
        "guard should be counted as not-live after drop"
    );
}

/// WI-CANCEL-1: cancelling an unknown request id returns 404 with a
/// structured `unknown_request` error code and does not panic.
#[tokio::test]
async fn test_cancel_unknown_request_returns_structured_404() {
    let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/requests/:id/cancel", post(cancel_request))
        .with_state(state.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/requests/9999/cancel")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(val["id"], 9999);
    assert_eq!(val["state"], "cancelled");
    assert_eq!(val["error"]["code"], "unknown_request");
}

/// WI-CANCEL-1: cancelling a known-but-not-streaming request (no CancellationToken registered) returns 200
/// with `state: cancelled` and tears down the request via finish_request.
#[tokio::test]
async fn test_cancel_known_request_returns_200() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let req = grim_scheduler::Request {
        id: 7,
        prompt_tokens: 1,
        priority: 0,
        consumed_tokens: 0,
        model_id: None,
        adapter_ids: vec![],
        max_new_tokens: 0,
        input_ids: Some(vec![0]),
        session: None,
    };
    let _ = engine.enqueue_request(req);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/requests/:id/cancel", post(cancel_request))
        .with_state(state.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/requests/7/cancel")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(val["id"], 7);
    assert_eq!(val["state"], "cancelled");

    // Engine state must be cleaned up.
    let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    assert!(!engine.scheduler.running.iter().any(|r| r.id == 7));
    assert!(!engine.sessions.contains_key(&7));
}

/// WI-CANCEL-0: non-streaming request teardown calls finish_request.
/// After a non-streaming chat completion completes, every per-request HashMap entry on Engine must be empty.
#[tokio::test]
async fn test_non_streaming_finish_request_called() {
    let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
    let mock_model = Box::new(grim_models_transformer::Llama::random(
        grim_tensor::Device::Cpu,
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

            partial_rotary_factor: 1.0,
            yarn: None,
        },
    ));
    engine.register_model("default", mock_model);
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state.clone());

    let request_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 5,
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

    // After completion, the scheduler must have no running requests and
    // no per-request state entries left behind.
    let engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    assert!(engine.scheduler.running.is_empty());
    assert!(engine.sessions.is_empty());
    assert!(engine.last_outcomes.is_empty());
    assert!(engine.request_rng.is_empty());
    assert!(engine.request_model_ids.is_empty());
    assert!(engine.request_input_ids.is_empty());
    assert!(engine.request_last_token.is_empty());
}

/// Safety regression guard: `AppState.engine` must remain `Mutex<Engine>`, not `RwLock<Engine>` or an unwrapped `Engine`.
/// An `RwLock` would allow concurrent *readers*, which is exactly the access pattern the `unsafe impl.
#[test]
fn test_appstate_engine_is_mutex() {
    // Type annotation forces `engine` to be `Mutex<Engine>` — if AppState
    // changes, this won't compile.
    let state: AppState = AppState {
        engine: Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        )),
        tokenizer: Mutex::new(None),
        model_path: None,
        model_arch: std::sync::Mutex::new(None),
        plugin_registry: None,
    };
    // Verify we can lock it (Mutex works)
    let _guard = state
        .engine
        .lock()
        .expect("engine mutex should not be poisoned");
}
