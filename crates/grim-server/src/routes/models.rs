use crate::{AppState, chrono_utc_now_rfc3339, grim_models_dir, model_loader};
use axum::{
    Json,
    body::Body,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use grim_format::GgufProvider;
use std::sync::Arc;

#[derive(serde::Deserialize)]
pub struct LoadModelRequest {
    #[serde(alias = "name")]
    pub model: String,
}

/// Model unloading request — same field aliasing for OpenAI compatibility.
#[derive(serde::Deserialize)]
pub struct UnloadModelRequest {
    #[serde(alias = "name")]
    pub model: String,
}

/// Dynamic model loading endpoint.
pub async fn load_model_route(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Validate model identifier: disallow directory traversal sequences ('..', absolute paths, backslashes).
    if req.model.contains("..") || req.model.starts_with('/') || req.model.contains('\\') {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "status": "error",
                "message": format!("Invalid model identifier '{}': path traversal sequences are not permitted.", req.model),
                "resolved_path": serde_json::Value::Null,
                "loaded_kind": serde_json::Value::Null,
            })),
        );
    }

    // P0-WI-3: prefer a `.grim` sibling when both exist; centralize resolution in `catalog::resolve_model_preferring_grim`
    // so `/v1/models/load` shares the same lookup logic as the CLI's on-demand model loader.
    let resolved_path = grim_core::catalog::resolve_model_preferring_grim(&req.model);

    let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());

    let model_path = match resolved_path {
        Some(p) => p,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "status": "error",
                    "message": format!("Model '{}' not found on disk; no mock fallback is provided.", req.model),
                    "resolved_path": serde_json::Value::Null,
                    "loaded_kind": serde_json::Value::Null,
                })),
            );
        }
    };

    #[cfg(debug_assertions)]
    eprintln!("[grim-server] Loading model from: {}", model_path.display());
    let model_path_str = model_path.to_string_lossy().to_string();
    let loaded_kind = if model_path_str.ends_with(".grim") {
        "grim"
    } else {
        "gguf"
    };

    match model_loader::load_from_path(&model_path_str) {
        Ok(m) => {
            let tokenizer = GgufProvider::open(&model_path_str)
                .ok()
                .and_then(|p| p.tokenizer().ok())
                .or_else(|| {
                    let sibling = model_path.with_extension("gguf");
                    sibling
                        .to_str()
                        .and_then(|gg| GgufProvider::open(gg).ok().and_then(|p| p.tokenizer().ok()))
                });
            *state.tokenizer.lock().unwrap_or_else(|e| e.into_inner()) = tokenizer;
            let arch = GgufProvider::open(&model_path_str)
                .ok()
                .and_then(|p| p.architecture().map(str::to_string))
                .or_else(|| {
                    let sibling = model_path.with_extension("gguf");
                    sibling.to_str().and_then(|gg| {
                        GgufProvider::open(gg)
                            .ok()
                            .and_then(|p| p.architecture().map(str::to_string))
                    })
                });
            *state.model_arch.lock().unwrap_or_else(|e| e.into_inner()) = arch;
            engine.register_model_with_farm(&req.model, m, &model_path_str);

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "success",
                    "message": format!("Model '{}' loaded dynamically.", req.model),
                    "resolved_path": model_path_str,
                    "loaded_kind": loaded_kind,
                })),
            )
        }
        Err(e) => {
            eprintln!(
                "[grim-server] ERROR: failed to load model '{}': {}",
                model_path.display(),
                e
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to load model: {}", e),
                    "resolved_path": model_path_str,
                })),
            )
        }
    }
}

/// Dynamic model unloading endpoint.
pub async fn unload_model_route(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnloadModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let unloaded = engine.unload_model(&req.model);
    if unloaded {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "success",
                "message": format!("Model '{}' unloaded dynamically from memory.", req.model)
            })),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "status": "error",
                "message": format!("Model '{}' is not loaded in memory.", req.model)
            })),
        )
    }
}

/// Grim compatibility /api/pull endpoint.
pub async fn grim_pull_route(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tx_clone = tx.clone();

    tokio::spawn(async move {
        let res = grim_core::client::download_model_with_progress(&name, None, move |p| {
            let _ = tx_clone.send(Ok(p));
        })
        .await;
        if let Err(e) = res {
            let _ = tx.send(Err(e));
        }
    });

    let stream = futures::stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Some(Ok(progress)) => {
                let json = serde_json::to_string(&progress).unwrap_or_default();
                let chunk = format!("{}\n", json);
                Some((Ok::<_, axum::Error>(axum::body::Bytes::from(chunk)), rx))
            }
            Some(Err(err)) => {
                let err_json = serde_json::json!({ "error": err.to_string() });
                let chunk = format!("{}\n", err_json.to_string());
                Some((Ok::<_, axum::Error>(axum::body::Bytes::from(chunk)), rx))
            }
            None => None,
        }
    });

    let body = Body::from_stream(stream);
    axum::response::Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(body)
        .unwrap()
}

/// POST /api/upload?filename=<name>.gguf - save an uploaded model file into the local catalog.
pub async fn grim_upload_route(
    Query(query): Query<std::collections::HashMap<String, String>>,
    body: axum::body::Bytes,
) -> Response {
    use grim_core::catalog::{ModelEntry, apply_gguf_enrichment};

    let Some(filename) = query.get("filename") else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "missing 'filename' query parameter" })),
        )
            .into_response();
    };

    let safe_name = std::path::Path::new(filename)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");
    if safe_name.is_empty()
        || !(safe_name.to_ascii_lowercase().ends_with(".gguf")
            || safe_name.to_ascii_lowercase().ends_with(".grim")
            || safe_name.to_ascii_lowercase().ends_with(".safetensors"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!(
                "invalid filename '{filename}': expected a bare .gguf, .grim, or .safetensors filename"
            )})),
        )
            .into_response();
    }

    if body.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "empty upload body" })),
        )
            .into_response();
    }

    let models_dir = grim_models_dir();
    if let Err(e) = tokio::fs::create_dir_all(&models_dir).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("cannot create models dir: {e}") })),
        )
            .into_response();
    }
    let dest_path = models_dir.join(safe_name);
    if let Err(e) = tokio::fs::write(&dest_path, &body).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("failed to write model file: {e}") })),
        )
            .into_response();
    }

    use sha2::Digest as _;
    let mut sha = sha2::Sha256::new();
    sha.update(&body);
    let sha256_hex = format!("{:x}", sha.finalize());

    let is_gguf = safe_name.to_ascii_lowercase().ends_with(".gguf");
    let display_name = dest_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(safe_name)
        .to_string();

    let mut entry = ModelEntry {
        name: display_name,
        path: dest_path.display().to_string(),
        arch: String::new(),
        params: String::new(),
        quant: String::new(),
        context_length: 0,
        size_bytes: body.len() as u64,
        sha256: sha256_hex,
        pulled_at: chrono_utc_now_rfc3339(),
        source: "upload".to_string(),
        preferred_dtype: String::new(),
    };
    if is_gguf {
        apply_gguf_enrichment(&mut entry, &dest_path);
    }
    if let Err(e) = entry.save(&dest_path) {
        eprintln!(
            "[grim-server] WARNING: uploaded '{}' saved but sidecar write failed: {e}",
            safe_name
        );
    }

    eprintln!(
        "[grim-server] Uploaded model '{}' ({:.2} MB) -> {}",
        safe_name,
        body.len() as f64 / 1_048_576.0,
        dest_path.display()
    );

    Json(serde_json::json!({
        "status": "success",
        "name": entry.name,
        "path": entry.path,
        "size_bytes": entry.size_bytes,
    }))
    .into_response()
}
