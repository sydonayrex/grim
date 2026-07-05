//! HTTP routes for Grim's Garage.
//!
//! These are mounted under `/api/...` and `/sse/...` and (eventually)
//! served by `cvkg-webkit-server`'s axum instance via `Router::nest`.
//!
//! v1 endpoints:
//! - `GET  /api/models`                       — list local models
//! - `GET  /api/datasets`                     — list local datasets
//! - `GET  /api/rocm/devices`                 — GPU probe
//! - `POST /api/train/start`                  — create + start a job
//! - `GET  /api/train/jobs`                   — list jobs + statuses
//! - `GET  /api/train/status/{id}`           — single-job snapshot
//! - `POST /api/train/cancel/{id}`           — request cancellation
//! - `SSE  /sse/metrics/{id}`                — live loss/vram events

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use cvkg_webkit_server::router as cvkg_router;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::discovery::{default_datasets_dir, default_models_dir, DatasetEntry, ModelEntry};
use crate::jobs::{JobId, JobRegistry, TrainingJob, TrainingMode};
use crate::rocm::{probe_rocm_devices, RocmDeviceInfo};

/// Shared state passed to every handler.
#[derive(Debug, Clone)]
pub struct AppState {
    pub registry: Arc<JobRegistry>,
}

#[derive(Debug, Deserialize)]
pub struct StartTrainingRequest {
    pub model_path: String,
    pub dataset_path: String,
    pub training_mode: TrainingMode,
    #[serde(default = "default_rank")]
    pub lora_rank: u32,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default)]
    pub rocm_fusion_rmsnorm_matmul: bool,
    #[serde(default)]
    pub rocm_fusion_qkv_attention: bool,
}

fn default_rank() -> u32 { 16 }
fn default_lr() -> f64 { 2e-5 }
fn default_epochs() -> u32 { 1 }

#[derive(Debug, Serialize)]
pub struct StartTrainingResponse {
    pub job_id: String,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct DatasetsResponse {
    pub datasets: Vec<DatasetEntry>,
}

#[derive(Debug, Serialize)]
pub struct RocmDevicesResponse {
    pub devices: Vec<RocmDeviceInfo>,
}

#[derive(Debug, Serialize)]
pub struct JobsListResponse {
    pub jobs: Vec<JobSummary>,
}

#[derive(Debug, Serialize)]
pub struct JobSummary {
    pub job_id: String,
    pub status: String,
    pub model_path: String,
    pub dataset_path: String,
    pub training_mode: TrainingMode,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/api/models", get(get_models))
        .route("/api/datasets", get(get_datasets))
        .route("/api/rocm/devices", get(get_rocm_devices))
        .route("/api/train/jobs", get(list_jobs))
        .route("/api/train/start", post(start_training))
        .route("/api/train/status/{id}", get(get_job_status))
        .route("/api/train/cancel/{id}", post(cancel_job))
        .route("/sse/metrics/{id}", get(sse_metrics))
        .with_state(state)
}

/// Merge Grim's Garage API routes with the CVKG dev-server router
/// from `cvkg-webkit-server` 0.3.3 (`cvkg_webkit_server::router`).
///
/// CVKG contributed routes:
///   - `GET  /`                              — loading screen / last VDOM snapshot
///   - `POST /snapshot`                       — capture VDOM snapshot
///   - `POST /build`                          — trigger a build
///   - `GET  /health/liveness`                — always 200 (no auth)
///   - `GET  /health/readiness`               — always 200 (no auth)
///   - `GET  /metrics`                        — Prometheus handle
///   - `GET  /api/system/time`                — SystemTime JSON
///   - `WS   /cvkg-ws`                        — runtime WebSocket
///   - `WS   /hmr`                            — HMR WebSocket
///   - `GET  /cvkg-webkit-server/{pkg,assets,static}/*` — static dirs
///
/// All endpoints share one axum `Router` driven from `axum::serve`.
pub fn build_combined_router(state: AppState) -> Router {
    let grim = build_router(state);

    // Build CVKG's dev-server router with a minimal in-memory AppState.
    // Tunables come from env via CVKG's defaults (`CVKG_BIND_ADDR=0.0.0.0:3000`,
    // `CVKG_PKG_DIR=...`, etc.) — grim-garage does not need to control them
    // because grim's own bind address is bound by `main.rs`.
    //
    // We construct Config via the clap Parser path so all CVKG_BIND_ADDR
    // / CVKG_PKG_DIR / CVKG_STATIC_DIR / CVKG_RATE_LIMIT_RPS / etc. env
    // paths work the same way they do in the standalone binary.
    let cfg = <cvkg_router::Config as clap::Parser>::parse_from(std::iter::empty::<String>());
    let (hmr_tx, _rx) = tokio::sync::broadcast::channel::<String>(16);
    let cvkg_state = std::sync::Arc::new(cvkg_router::AppState::new(cfg, hmr_tx));
    let cvkg = cvkg_router::create_router(cvkg_state, None);

    grim.merge(cvkg)
}

async fn get_models() -> Json<ModelsResponse> {
    let dir = default_models_dir();
    match crate::discovery::discover_models(&dir) {
        Ok(models) => Json(ModelsResponse { models }),
        Err(_) => Json(ModelsResponse { models: Vec::new() }),
    }
}

async fn get_datasets() -> Json<DatasetsResponse> {
    let dir = default_datasets_dir();
    match crate::discovery::discover_datasets(&dir) {
        Ok(datasets) => Json(DatasetsResponse { datasets }),
        Err(_) => Json(DatasetsResponse { datasets: Vec::new() }),
    }
}

async fn get_rocm_devices() -> Json<RocmDevicesResponse> {
    Json(RocmDevicesResponse { devices: probe_rocm_devices() })
}

async fn list_jobs(State(state): State<AppState>) -> Json<JobsListResponse> {
    let jobs = state.registry.list().await;
    let summaries: Vec<JobSummary> = futures::future::join_all(jobs.into_iter().map(|(id, status)| {
        let st = state.clone();
        async move {
            if let Some(job) = st.registry.get(&id).await {
                JobSummary {
                    job_id: id.0,
                    status: status_label(status).to_string(),
                    model_path: job.model_path,
                    dataset_path: job.dataset_path,
                    training_mode: job.training_mode,
                }
            } else {
                JobSummary {
                    job_id: id.0,
                    status: status_label(status).to_string(),
                    model_path: String::new(),
                    dataset_path: String::new(),
                    training_mode: TrainingMode::Lora,
                }
            }
        }
    }))
    .await;
    Json(JobsListResponse { jobs: summaries })
}

async fn start_training(
    State(state): State<AppState>,
    Json(req): Json<StartTrainingRequest>,
) -> Result<Json<StartTrainingResponse>, (StatusCode, Json<serde_json::Value>)> {
    let job = TrainingJob {
        model_path: req.model_path,
        dataset_path: req.dataset_path,
        training_mode: req.training_mode,
        lora_rank: req.lora_rank,
        learning_rate: req.learning_rate,
        epochs: req.epochs,
        rocm_fusion_rmsnorm_matmul: req.rocm_fusion_rmsnorm_matmul,
        rocm_fusion_qkv_attention: req.rocm_fusion_qkv_attention,
        status: crate::jobs::JobStatus::Pending,
        metrics: Vec::new(),
    };

    match state.registry.create(job).await {
        Ok(id) => {
            Ok(Json(StartTrainingResponse {
                job_id: id.0,
                status: "pending".into(),
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

async fn get_job_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let jid = JobId(id);
    match state.registry.get(&jid).await {
        Some(job) => Ok(Json(json!({
            "job_id": jid.0,
            "status": status_label(job.status),
            "model_path": job.model_path,
            "dataset_path": job.dataset_path,
            "training_mode": job.training_mode,
            "lora_rank": job.lora_rank,
            "learning_rate": job.learning_rate,
            "epochs": job.epochs,
            "metric_count": job.metrics.len(),
        }))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("job not found: {}", jid.0) })),
        )),
    }
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let jid = JobId(id);
    // v1: mark as Failed — actual cancellation ack would target the running worker.
    match state.registry.update_status(&jid, crate::jobs::JobStatus::Failed).await {
        Ok(()) => Ok(Json(json!({ "job_id": jid.0, "status": "failed" }))),
        Err(_) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("job not found: {}", jid.0) })),
        )),
    }
}

async fn sse_metrics(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<
    Sse<impl Stream<Item = std::result::Result<Event, axum::Error>>>,
    (StatusCode, Json<serde_json::Value>),
> {
    let jid = JobId(id);
    if state.registry.get(&jid).await.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("job not found: {}", jid.0) })),
        ));
    }
    let mut rx = state.registry.subscribe_metrics();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) if event.job_id == jid.0 => {
                    let payload = serde_json::to_string(&event).unwrap_or_default();
                    yield std::result::Result::<Event, axum::Error>::Ok(
                        Event::default().event("metric").data(payload)
                    );
                }
                Ok(_) => continue,
                Err(_) => {
                    yield std::result::Result::<Event, axum::Error>::Ok(
                        Event::default().event("end").data("done")
                    );
                    break;
                }
            }
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::new()))
}

fn status_label(status: crate::jobs::JobStatus) -> &'static str {
    use crate::jobs::JobStatus;
    match status {
        JobStatus::Pending => "pending",
        JobStatus::Running => "running",
        JobStatus::Completed => "completed",
        JobStatus::Failed => "failed",
    }
}

/// Health endpoint for kube-style probes.
async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "ok"})))
}

pub fn health_router() -> Router {
    Router::new().route("/healthz", get(health))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_router_returns_ok() {
        let r = health_router();
        let resp = r
            .oneshot(
                axum::http::Request::builder()
                    .uri("/healthz")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn start_training_request_applies_defaults() {
        let json = r#"{"model_path":"/m","dataset_path":"/d","training_mode":"Lora"}"#;
        let parsed: StartTrainingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.lora_rank, 16);
        assert!((parsed.learning_rate - 2e-5).abs() < 1e-9);
        assert_eq!(parsed.epochs, 1);
        assert!(!parsed.rocm_fusion_rmsnorm_matmul);
        assert!(!parsed.rocm_fusion_qkv_attention);
    }

    #[tokio::test]
    async fn combined_router_serves_grim_models_endpoint() {
        let state = AppState { registry: std::sync::Arc::new(crate::jobs::JobRegistry::new()) };
        let r = build_combined_router(state);
        let resp = r
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // 200 with empty ModelEntry vec because no models dir exists in test environment.
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn combined_router_serves_cvkg_liveness_probes() {
        // cvkg-webkit-server 0.3.3 router contributes `/health/liveness`
        // and `/health/readiness` (mercilessly returning 200 even when no
        // real subscribers are wired). Verify they coexist with grim routes.
        let state = AppState { registry: std::sync::Arc::new(crate::jobs::JobRegistry::new()) };
        let r = build_combined_router(state);

        for path in ["/health/liveness", "/health/readiness"] {
            let resp = r
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(path)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "CVKG probe {path} should return 200"
            );
        }
    }

    #[tokio::test]
    async fn combined_router_serves_cvkg_system_time_endpoint() {
        let state = AppState { registry: std::sync::Arc::new(crate::jobs::JobRegistry::new()) };
        let r = build_combined_router(state);
        let resp = r
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/system/time")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
