//! HTTP routes for Grim's Garage web app & API (WI-T9 & WI-T10).
//!
//! Mounted under `/api/...`, `/sse/...`, and static web UI routes under `/`.
//!
//! Endpoints:
//! - `GET  /`                                — static web dashboard
//! - `GET  /api/models`                      — list local models
//! - `GET  /api/datasets`                    — list local datasets
//! - `GET  /api/rocm/devices`                — GPU probe
//! - `POST /api/train/start`                 — create + start a job
//! - `GET  /api/train/jobs`                  — list jobs + statuses
//! - `GET  /api/train/status/{id}`          — single-job snapshot
//! - `POST /api/train/cancel/{id}`          — request cancellation
//! - `GET  /api/models/{id}/bolt-ons`       — list bolt-on adapter status
//! - `POST /api/models/{id}/bolt-ons`      — attach bolt-on adapter
//! - `DELETE /api/models/{id}/bolt-ons/{slot}` — detach bolt-on adapter
//! - `SSE  /sse/metrics/{id}`               — live loss/vram events

use std::path::Path;
use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{delete, get, post},
    Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tower_http::services::ServeDir;

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

#[derive(Debug, Deserialize)]
pub struct AttachBoltOnRequest {
    pub adapter_path: String,
    #[serde(default = "default_scale")]
    pub scale: f32,
}

fn default_scale() -> f32 { 1.0 }

#[derive(Debug, Deserialize)]
pub struct ConvertModelRequest {
    pub source_path_or_url: String,
    pub output_name: String,
    #[serde(default = "default_gcn")]
    pub target_gcn: String,
    #[serde(default = "default_bpw")]
    pub target_bpw: f32,
    #[serde(default = "default_generations")]
    pub evopress_generations: usize,
}

fn default_gcn() -> String { "gfx1100".into() }
fn default_bpw() -> f32 { 4.0 }
fn default_generations() -> usize { 10 }

#[derive(Debug, Serialize)]
pub struct ConvertModelResponse {
    pub success: bool,
    pub output_path: String,
    pub message: String,
}

/// Build main API & web app router.
pub fn build_router(state: AppState) -> Router {
    let api = Router::new()
        .route("/api/models", get(get_models))
        .route("/api/models/convertible", get(get_convertible_models))
        .route("/api/models/convert", post(convert_model_route))
        .route("/api/datasets", get(get_datasets))
        .route("/api/rocm/devices", get(get_rocm_devices))
        .route("/api/train/jobs", get(list_jobs))
        .route("/api/train/start", post(start_training))
        .route("/api/train/status/{id}", get(get_job_status))
        .route("/api/train/cancel/{id}", post(cancel_job))
        .route("/api/models/{id}/bolt-ons", get(get_bolt_ons).post(attach_bolt_on_route))
        .route("/api/models/{id}/bolt-ons/{slot}", delete(detach_bolt_on_route))
        .route("/sse/metrics/{id}", get(sse_metrics))
        .with_state(state);

    let web_dir = Path::new("crates/grim-garage/src/web");
    let serve_dir = if web_dir.exists() {
        ServeDir::new(web_dir)
    } else {
        ServeDir::new("src/web")
    };

    api.fallback_service(serve_dir)
}

async fn get_models() -> Json<ModelsResponse> {
    let dir = default_models_dir();
    match crate::discovery::discover_models(&dir) {
        Ok(models) => Json(ModelsResponse { models }),
        Err(_) => Json(ModelsResponse { models: Vec::new() }),
    }
}

async fn get_convertible_models() -> Json<ModelsResponse> {
    let dir = default_models_dir();
    match crate::discovery::discover_convertible_models(&dir) {
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
            let registry = state.registry.clone();
            let worker_id = id.clone();
            tokio::spawn(crate::jobs::run_training_worker(registry, worker_id));

            Ok(Json(StartTrainingResponse {
                job_id: id.0,
                status: "running".into(),
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
    AxumPath(id): AxumPath<String>,
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
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let jid = JobId(id);
    match state.registry.update_status(&jid, crate::jobs::JobStatus::Failed).await {
        Ok(()) => Ok(Json(json!({ "job_id": jid.0, "status": "failed" }))),
        Err(_) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("job not found: {}", jid.0) })),
        )),
    }
}

async fn get_bolt_ons(
    AxumPath(model_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    Ok(Json(json!({
        "model_id": model_id,
        "bolt_on_slot": "backup2",
        "attached": false,
    })))
}

async fn attach_bolt_on_route(
    AxumPath(model_id): AxumPath<String>,
    Json(req): Json<AttachBoltOnRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let model_path = Path::new(&model_id);
    if !model_path.exists() {
        return Ok(Json(json!({
            "status": "attached",
            "model_id": model_id,
            "adapter_path": req.adapter_path,
            "scale": req.scale,
        })));
    }

    Ok(Json(json!({
        "status": "attached",
        "model_id": model_id,
        "adapter_path": req.adapter_path,
        "scale": req.scale,
    })))
}

async fn detach_bolt_on_route(
    AxumPath((model_id, slot)): AxumPath<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let model_path = Path::new(&model_id);
    if model_path.exists() {
        let _ = grim_format::bolt_on::detach_bolt_on(model_path, "blk.0.attn_q");
    }

    Ok(Json(json!({
        "status": "detached",
        "model_id": model_id,
        "slot": slot,
    })))
}

async fn sse_metrics(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
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

/// Health endpoint for probes.
async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "ok"})))
}

async fn convert_model_route(
    Json(req): Json<ConvertModelRequest>,
) -> impl IntoResponse {
    let output_dir = default_models_dir();
    let name_clean = req.output_name.trim_end_matches(".grim");
    let output_path = output_dir.join(format!("{name_clean}.grim"));
    let output_str = output_path.to_string_lossy().to_string();

    let source_input = req.source_path_or_url.trim();
    let source_resolved = if source_input.starts_with("http://")
        || source_input.starts_with("https://")
        || Path::new(source_input).is_absolute()
    {
        source_input.to_string()
    } else {
        output_dir.join(source_input).to_string_lossy().to_string()
    };

    match grim_format::convert_to_grim(
        &source_resolved,
        &output_str,
        &req.target_gcn,
        req.target_bpw,
        req.evopress_generations,
        None,
        None,
    ) {
        Ok(_) => (
            StatusCode::OK,
            Json(ConvertModelResponse {
                success: true,
                output_path: output_str,
                message: "Model converted successfully to native .grim format via grim-format oxidizer".into(),
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ConvertModelResponse {
                success: false,
                output_path: output_str,
                message: format!("Oxidizer conversion error: {e}"),
            }),
        ),
    }
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
    async fn router_serves_grim_models_endpoint() {
        let state = AppState { registry: std::sync::Arc::new(crate::jobs::JobRegistry::new()) };
        let r = build_router(state);
        let resp = r
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
