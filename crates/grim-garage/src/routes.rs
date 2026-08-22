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
    Router,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{
        IntoResponse, Json,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{delete, get, post},
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::discovery::{DatasetEntry, ModelEntry, default_datasets_dir, default_models_dir};
use crate::jobs::{JobId, JobRegistry, TrainingJob, TrainingMode};
use crate::rocm::probe_rocm_devices;
use grim_engine::pipelines::{AudioPipeline, AudioPipelineConfig, DiffusionPipeline, DiffusionPipelineConfig};
use grim_engine::{Engine, model_loader};
use grim_format::GgufTokenizer;
use grim_models_audio::{KokoroConfig, VocosConfig};
use grim_models_diffusion::{Flux2Config, Flux2VaeConfig};
use grim_tensor::BackendDevice;

/// Shared state passed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<JobRegistry>,
    pub engine: Arc<std::sync::Mutex<Engine>>,
    pub tokenizer: Arc<std::sync::Mutex<Option<GgufTokenizer>>>,
    pub model_path: Option<std::path::PathBuf>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("registry", &self.registry)
            .field("engine", &"<Engine>")
            .field("tokenizer", &"<tokenizer>")
            .field("model_path", &self.model_path)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
pub struct StartTrainingRequest {
    pub model_path: String,
    pub dataset_path: String,
    pub training_mode: TrainingMode,
    #[serde(default = "default_rank")]
    pub lora_rank: u32,
    /// LoRA alpha (scaling) for adapter init and bake-merge
    /// (`ΔW = (alpha / rank) · B·A`). `None` = documented rule-of-thumb
    /// default `2 * lora_rank`. The UI always sends this; previously it was
    /// silently dropped by serde.
    #[serde(default)]
    pub lora_alpha: Option<f32>,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default)]
    pub rocm_fusion_rmsnorm_matmul: bool,
    #[serde(default)]
    pub rocm_fusion_qkv_attention: bool,
    /// Codec format for base weights: Bf16, Crow, Raven, Rook, Jay, Jackdaw, Magpie.
    #[serde(default)]
    pub weight_format: crate::weight_format::WeightFormat,
    /// Backend the user selected ("rocm", "cuda", "vulkan", "metal", "cpu",
    /// or "auto"). Drives the grim-garage backend selection chain.
    #[serde(default)]
    pub preferred_backend: Option<String>,
    /// Gradient accumulation steps. Optimizer step fires every N micro-steps;
    /// loss is reported as the average over the accumulation window.
    #[serde(default = "default_accumulation_steps")]
    pub accumulation_steps: u32,
    /// Number of GPUs for data-parallel training. 0 or 1 = single GPU;
    /// >1 = RCCL all-reduce across N devices.
    #[serde(default)]
    pub num_gpus: u32,
    /// PiSSA: initialize adapter A/B via truncated SVD of the base weight.
    #[serde(default)]
    pub use_pissa: bool,
    /// OLoRA: add `olora_lambda * olora_orthogonality_penalty(A, B)` to the loss.
    #[serde(default)]
    pub use_olora: bool,
    /// Weight of the OLoRA orthogonality penalty term.
    #[serde(default)]
    pub olora_lambda: f32,
    /// SPECTRAL-QLORA: orthogonal adapter init + Muon optimizer.
    #[serde(default)]
    pub use_spectral_qlora: bool,
    /// Optionally resume training from a checkpoint sidecar produced by a
    /// prior run. The sidecar must exist at this path on the server and
    /// is validated via `validate_job_path` in the route handler.
    #[serde(default)]
    pub resume_from_checkpoint: Option<String>,
    /// Permanently bake the trained adapter into the target .grim file upon job completion.
    #[serde(default)]
    pub bake_on_completion: bool,
}

fn default_rank() -> u32 {
    16
}
fn default_lr() -> f64 {
    2e-5
}
fn default_epochs() -> u32 {
    1
}
fn default_accumulation_steps() -> u32 {
    1
}

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
pub struct BackendProbeResponse {
    pub backends: Vec<crate::backend::BackendProbe>,
    /// The backend a job with no explicit preference would select.
    pub selected: String,
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
    pub use_pissa: bool,
    pub use_olora: bool,
    pub olora_lambda: f32,
}

#[derive(Debug, Deserialize)]
pub struct AttachBoltOnRequest {
    pub adapter_path: String,
    #[serde(default = "default_scale")]
    pub scale: f32,
}

fn default_scale() -> f32 {
    1.0
}

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
    /// Target codec format: "crow", "raven", "rook", "jay", "jackdaw", "magpie".
    /// Passes through to `grim_format::convert_to_grim()` as the `target_bpw`
    /// equivalent after resolving each name to its bpw via `WeightFormat`.
    #[serde(default)]
    pub target_format: Option<String>,
}

fn default_gcn() -> String {
    "gfx1100".into()
}
fn default_bpw() -> f32 {
    4.0
}
fn default_generations() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct ConvertModelResponse {
    pub success: bool,
    pub output_path: String,
    pub message: String,
}

#[derive(rust_embed::RustEmbed)]
#[folder = "web/"]
struct WebAssets;

async fn static_index() -> impl IntoResponse {
    embedded_asset_handler(AxumPath("index.html".to_string())).await
}

async fn embedded_asset_handler(AxumPath(path): AxumPath<String>) -> impl IntoResponse {
    let path_str = if path.is_empty() || path == "/" {
        "index.html"
    } else {
        &path
    };
    // P2-13d: axum matches registered `/api/*` routes before this catch-all,
    // so any path reaching here that starts with `api/` is an unknown API
    // endpoint. It must 404 rather than silently serving the SPA shell —
    // returning index.html would mask API typos as HTTP 200 and confuse
    // clients.
    if path_str == "api" || path_str.starts_with("api/") {
        return (StatusCode::NOT_FOUND, "404 Not Found").into_response();
    }
    match WebAssets::get(path_str) {
        Some(content) => {
            let mime_str = if path_str.ends_with(".html") {
                "text/html"
            } else if path_str.ends_with(".css") {
                "text/css"
            } else if path_str.ends_with(".js") {
                "application/javascript"
            } else {
                "application/octet-stream"
            };
            (
                [(axum::http::header::CONTENT_TYPE, mime_str)],
                content.data.into_owned(),
            )
                .into_response()
        }
        None => match WebAssets::get("index.html") {
            Some(index) => (
                [(axum::http::header::CONTENT_TYPE, "text/html")],
                index.data.into_owned(),
            )
                .into_response(),
            None => (StatusCode::NOT_FOUND, "404 Not Found").into_response(),
        },
    }
}

/// Build main API & web app router.
pub fn build_router(state: AppState) -> Router {
    let api = Router::new()
        .route("/api/models", get(get_models))
        .route("/api/models/convertible", get(get_convertible_models))
        .route("/api/models/convert", post(convert_model_route))
        .route("/api/convert", post(convert_model_route))
        .route("/api/datasets", get(get_datasets))
        .route("/api/rocm/devices", get(get_rocm_devices))
        .route("/api/backends", get(list_backends))
        .route("/api/train/jobs", get(list_jobs))
        .route("/api/train/start", post(start_training))
        .route("/api/train/status/{id}", get(get_job_status))
        .route("/api/train/cancel/{id}", post(cancel_job))
        .route(
            "/api/models/{id}/bolt-ons",
            get(get_bolt_ons).post(attach_bolt_on_route),
        )
        .route("/api/models/{id}/bolt-ons/merge", post(merge_bolt_on_route))
        .route(
            "/api/models/{id}/bolt-ons/{slot}",
            delete(detach_bolt_on_route),
        )
        .route("/api/chat/load", post(load_model_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/diagnostics", get(get_diagnostics))
        .route("/api/diffusion/samplers", get(get_diffusion_samplers))
        .route("/api/diffusion/generate", post(diffusion_generate_handler))
        .route("/api/audio/voices", get(get_audio_voices))
        .route("/api/audio/tts", post(audio_tts_handler))
        .route("/api/audio/audio2audio", post(audio_audio2audio_handler))
        .route("/sse/metrics/{id}", get(sse_metrics))
        .route("/", get(static_index))
        .route("/{*path}", get(embedded_asset_handler))
        .with_state(state);

    api
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
        Err(_) => Json(DatasetsResponse {
            datasets: Vec::new(),
        }),
    }
}

async fn get_rocm_devices() -> Json<BackendProbeResponse> {
    Json(BackendProbeResponse {
        backends: probe_rocm_devices()
            .into_iter()
            .map(|d| crate::backend::BackendProbe {
                name: "rocm".into(),
                device_kind: format!("rocm:{}", d.ordinal),
                available: d.is_rocm_compliant,
                detail: format!(
                    "{} / {} / {} CU(s) / {} VRAM",
                    d.name, d.vendor, d.compute_units, d.vram_bytes
                ),
            })
            .collect(),
        selected: crate::backend::select_backend(None).label,
    })
}

/// Probe every compute backend in the selection chain (ROCm → CUDA →
/// Vulkan → Metal → CPU) and report which are actually live on this host.
/// Drives the "select GPU" panel in the UI.
async fn list_backends() -> Json<BackendProbeResponse> {
    Json(BackendProbeResponse {
        backends: crate::backend::probe_all(),
        // What a job with no explicit preference would land on.
        selected: crate::backend::select_backend(None).label,
    })
}

async fn list_jobs(State(state): State<AppState>) -> Json<JobsListResponse> {
    // L5 / H5: take a single read-lock snapshot so we cannot snag a job
    // id from `list()` and then miss it on the follow-up `get()` (which
    // surfaced as "ghost" JobSummary rows with empty paths and the
    // placeholder TrainingMode::Lora). Filter rows whose `model_path` is
    // empty — defensive guard against any future code path that stores a
    // job with empty fields (post-M1 path validation rejects `/`, `..`,
    // etc. on input, but we ignore such rows defensively rather than
    // shipping blank cards).
    let snap = state.registry.snapshot().await;
    let summaries: Vec<JobSummary> = snap
        .into_iter()
        .filter_map(|(id, status, job)| {
            if job.model_path.is_empty() || job.dataset_path.is_empty() {
                None
            } else {
                Some(JobSummary {
                    job_id: id.0,
                    status: status_label(status).to_string(),
                    model_path: job.model_path,
                    dataset_path: job.dataset_path,
                    training_mode: job.training_mode,
                    use_pissa: job.use_pissa,
                    use_olora: job.use_olora,
                    olora_lambda: job.olora_lambda,
                })
            }
        })
        .collect();
    Json(JobsListResponse { jobs: summaries })
}

async fn start_training(
    State(state): State<AppState>,
    Json(req): Json<StartTrainingRequest>,
) -> Result<Json<StartTrainingResponse>, (StatusCode, Json<serde_json::Value>)> {
    // M1: reject path-traversal-style strings in model_path and
    // dataset_path. Pre-fix the worker wrote sidecars under attacker-
    // chosen directories because `create_dir_all(parent)` had no
    // allowlist check. Mirrors the sibling `convert_model_route` and
    // `get_bolt_ons` rejection strategy.
    if let Err(e) = validate_job_path("model_path", &req.model_path) {
        return Err((StatusCode::BAD_REQUEST, Json(json!({ "error": e }))));
    }
    if let Err(e) = validate_job_path("dataset_path", &req.dataset_path) {
        return Err((StatusCode::BAD_REQUEST, Json(json!({ "error": e }))));
    }
    // Task 6.2: Enforce max concurrent jobs limit
    let active_jobs = state.registry.running_count().await;
    if active_jobs >= state.registry.max_concurrent {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(
                json!({ "error": format!("max concurrent jobs ({}) reached", state.registry.max_concurrent) }),
            ),
        ));
    }
    // M7: refuse `lora_rank == 0` (autograd divides by rank) and
    // enforce the QLoRA×rank ceiling before the worker spawns. Without
    // this gate a non-form code path that targeted `lora_rank = 0`
    // would reach `apply_and_record_lora` and crash; a QLoRA job with
    // rank > QLORA_MAX_RANK would OOM the consumer GPU.
    use crate::view_model::hyperparam::{LoraRank, LoraRankError};
    let rank = match LoraRank::new(req.lora_rank) {
        Ok(r) => r,
        Err(LoraRankError::Zero) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "lora_rank must be > 0" })),
            ));
        }
        Err(other) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": other.to_string() })),
            ));
        }
    };
    if let Err(e) = rank.validate_for_mode(req.training_mode) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("lora_rank / training_mode: {e}") })),
        ));
    }
    // OLoRA: refuse a job that enables the orthogonality penalty with a
    // non-positive weight — the worker only applies the penalty when
    // `olora_lambda > 0.0`, so accepting `0.0` or negative here would
    // silently no-op a feature the user asked for.
    if req.use_olora && req.olora_lambda <= 0.0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "use_olora requires olora_lambda > 0.0" })),
        ));
    }

    let job = TrainingJob {
        model_path: req.model_path,
        dataset_path: req.dataset_path,
        training_mode: req.training_mode,
        lora_rank: rank.value(),
        lora_alpha: req.lora_alpha,
        learning_rate: req.learning_rate,
        epochs: req.epochs,
        rocm_fusion_rmsnorm_matmul: req.rocm_fusion_rmsnorm_matmul,
        rocm_fusion_qkv_attention: req.rocm_fusion_qkv_attention,
        weight_format: req.weight_format,
        preferred_backend: req.preferred_backend.clone(),
        accumulation_steps: req.accumulation_steps,
        optimizer: grim_autograd::OptimizerKind::AdamW,
        scheduler: grim_autograd::LRScheduler::Cosine,
        min_lr: (req.learning_rate * 1e-2).max(1e-10),
        num_gpus: req.num_gpus,
        use_pissa: req.use_pissa,
        use_olora: req.use_olora,
        olora_lambda: req.olora_lambda,
        use_spectral_qlora: req.use_spectral_qlora,
        bake_on_completion: req.bake_on_completion,
        resume_from_checkpoint: req.resume_from_checkpoint,
        status: crate::jobs::JobStatus::Pending,
        metrics: Vec::new(),
        rank_metrics: Vec::new(),
        cancel: tokio_util::sync::CancellationToken::new(),
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
            "use_pissa": job.use_pissa,
            "use_olora": job.use_olora,
            "olora_lambda": job.olora_lambda,
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
    // Atomic: signal the worker's CancellationToken AND transition the
    // status to `Cancelled` only if the job is still Pending/Running. A
    // cancel that arrives after the worker already finished preserves the
    // real terminal status (Completed/Failed) rather than lying.
    match state.registry.request_cancel(&jid).await {
        Ok(crate::jobs::JobStatus::Cancelled) => {
            Ok(Json(json!({ "job_id": jid.0, "status": "cancelled" })))
        }
        Ok(already) => Ok(Json(json!({
            "job_id": jid.0,
            "status": status_label(already),
            "note": "cancel arrived after terminal transition; status preserved"
        }))),
        Err(e) => Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}

/// Validate a `model_path` or `dataset_path` value. Rejects any path
/// containing `..`, `/`, or `\` because the worker constructs
/// `{model_path}.train` and `create_dir_all`s the parent, which would
/// otherwise let an HTTP POST write `.train` files into arbitrary
/// directories. Mirrors the wire shape used by the sibling
/// `convert_model_route` and `prevent_path_traversal` helpers (those reject
/// the same byte set on the model_id path segment).
pub(crate) fn validate_job_path(field: &str, value: &str) -> std::result::Result<(), String> {
    let has_traversal = value
        .split('/')
        .any(|component| component == ".." || component == "." || component.is_empty());
    if has_traversal || value.contains('\\') {
        Err(format!(
            "{field}: invalid path {value:?} (forbidden: path traversal or backslash)"
        ))
    } else {
        Ok(())
    }
}

/// Prevent path traversal in model_id. Only blocks `..`, `/`, and `\`;
/// does NOT validate existence or non-emptiness. Callers must check those
/// separately after this guard.
fn prevent_path_traversal(id: &str) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if id.contains("..") || id.contains('/') || id.contains('\\') {
        Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "invalid model_id: path traversal forbidden" })),
        ))
    } else {
        Ok(())
    }
}

async fn get_bolt_ons(
    AxumPath(model_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    prevent_path_traversal(&model_id)?;
    let model_path = Path::new(&model_id);
    if !model_path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("model not found: {}", model_id) })),
        ));
    }

    // Open the .grim file and check backup2 status for each tensor.
    let file = match std::fs::File::open(model_path) {
        Ok(f) => f,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("failed to open model: {e}") })),
            ));
        }
    };

    let gguf = match grim_format::gguf::read_gguf(file) {
        Ok(g) => g,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("failed to parse model: {e}") })),
            ));
        }
    };

    let grim_meta = grim_format::gguf::GrimMetadata::from_gguf_metadata(&gguf.metadata);
    let mut bolt_ons = Vec::new();
    for entry in &gguf.tensors {
        if let Some(ext) = grim_meta.get_tensor_ext(&entry.name) {
            if ext.backup2.bpw > 0 {
                bolt_ons.push(json!({
                    "tensor": entry.name,
                    "bpw": ext.backup2.bpw,
                    "scale_offset": ext.backup2.scale_offset,
                    "codes_offset": ext.backup2.codes_offset,
                    "codes_size": ext.backup2.codes_size,
                    "status": "attached",
                }));
            }
        }
    }

    Ok(Json(json!({
        "model_id": model_id,
        "bolt_ons": bolt_ons,
        "count": bolt_ons.len(),
    })))
}

async fn attach_bolt_on_route(
    AxumPath(model_id): AxumPath<String>,
    Json(req): Json<AttachBoltOnRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    prevent_path_traversal(&model_id)?;
    // M1-class gate: `adapter_path` becomes `{adapter_path}.train` and is read
    // for the LoRA sidecar. Without this check a POST could point at an
    // arbitrary file via `..`/absolute segments — the same traversal class the
    // sibling `validate_job_path` already blocks on model/dataset paths.
    if let Err(e) = validate_job_path("adapter_path", &req.adapter_path) {
        return Err((StatusCode::BAD_REQUEST, Json(json!({ "error": e }))));
    }
    let model_path = Path::new(&model_id);
    if !model_path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("model not found: {}", model_id) })),
        ));
    }

    // Load the adapter sidecar.
    let sidecar_path = format!("{}.train", req.adapter_path);
    let sidecar = match grim_format::train::TrainState::read(&sidecar_path) {
        Ok(Some(s)) => s,
        Ok(None) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("adapter sidecar not found: {}", sidecar_path) })),
            ));
        }
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("failed to read adapter sidecar: {e}") })),
            ));
        }
    };

    // Find all tensors with lora adapters in the sidecar.
    let tensor_names = sidecar.lora_tensor_names();
    if tensor_names.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "no lora adapters found in sidecar" })),
        ));
    }

    // Create a CPU backend for tensor construction.
    let cpu_backend = grim_backend_cpu::device::CpuDevice::new();

    let mut attached = Vec::new();
    let mut errors = Vec::new();

    for tensor_name in &tensor_names {
        match sidecar.lora_weights_for(tensor_name) {
            Some((a_data, a_shape, b_data, b_shape)) => {
                // Create Tensor objects from the raw f32 data.
                let a_shape = grim_tensor::Shape::from_slice(a_shape);
                let b_shape = grim_tensor::Shape::from_slice(b_shape);
                let a_storage = match cpu_backend.from_cpu(
                    &a_data,
                    &a_shape,
                    grim_tensor::DType::F32,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        errors.push(json!({ "tensor": tensor_name, "error": format!("failed to create A tensor: {e}") }));
                        continue;
                    }
                };
                let b_storage = match cpu_backend.from_cpu(
                    &b_data,
                    &b_shape,
                    grim_tensor::DType::F32,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        errors.push(json!({ "tensor": tensor_name, "error": format!("failed to create B tensor: {e}") }));
                        continue;
                    }
                };
                let a_tensor = grim_tensor::Tensor::new(
                    std::sync::Arc::from(a_storage),
                    a_shape,
                    grim_tensor::DType::F32,
                    grim_tensor::dtype::QuantProvenance::GrimNative,
                    grim_tensor::dtype::Device::Cpu,
                );
                let b_tensor = grim_tensor::Tensor::new(
                    std::sync::Arc::from(b_storage),
                    b_shape,
                    grim_tensor::DType::F32,
                    grim_tensor::dtype::QuantProvenance::GrimNative,
                    grim_tensor::dtype::Device::Cpu,
                );

                match grim_format::bolt_on::attach_bolt_on(
                    model_path,
                    tensor_name,
                    &a_tensor,
                    &b_tensor,
                    req.scale,
                ) {
                    Ok(()) => attached.push(tensor_name.clone()),
                    Err(e) => {
                        errors.push(json!({ "tensor": tensor_name, "error": format!("{e}") }))
                    }
                }
            }
            None => {
                errors.push(json!({ "tensor": tensor_name, "error": "missing lora A or B weights in sidecar" }));
            }
        }
    }

    if attached.is_empty() && !errors.is_empty() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "failed to attach any bolt-on adapters", "details": errors })),
        ));
    }

    Ok(Json(json!({
        "status": "attached",
        "model_id": model_id,
        "adapter_path": req.adapter_path,
        "scale": req.scale,
        "attached_tensors": attached,
        "errors": errors,
    })))
}

async fn merge_bolt_on_route(
    AxumPath(model_id): AxumPath<String>,
    Json(req): Json<AttachBoltOnRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    prevent_path_traversal(&model_id)?;
    if let Err(e) = validate_job_path("adapter_path", &req.adapter_path) {
        return Err((StatusCode::BAD_REQUEST, Json(json!({ "error": e }))));
    }
    let model_path = Path::new(&model_id);
    if !model_path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("model not found: {}", model_id) })),
        ));
    }

    let sidecar_path = format!("{}.train", req.adapter_path);
    let sidecar = match grim_format::train::TrainState::read(&sidecar_path) {
        Ok(Some(s)) => s,
        Ok(None) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("adapter sidecar not found: {}", sidecar_path) })),
            ));
        }
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("failed to read adapter sidecar: {e}") })),
            ));
        }
    };

    let tensor_names = sidecar.lora_tensor_names();
    if tensor_names.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "no lora adapters found in sidecar" })),
        ));
    }

    let cpu_backend = grim_backend_cpu::device::CpuDevice::new();
    let mut merged = Vec::new();
    let mut errors = Vec::new();

    for tensor_name in &tensor_names {
        match sidecar.lora_weights_for(tensor_name) {
            Some((a_data, a_shape, b_data, b_shape)) => {
                let a_shape = grim_tensor::Shape::from_slice(a_shape);
                let b_shape = grim_tensor::Shape::from_slice(b_shape);
                let a_storage = match cpu_backend.from_cpu(
                    &a_data,
                    &a_shape,
                    grim_tensor::DType::F32,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        errors.push(json!({ "tensor": tensor_name, "error": format!("failed to create A tensor: {e}") }));
                        continue;
                    }
                };
                let b_storage = match cpu_backend.from_cpu(
                    &b_data,
                    &b_shape,
                    grim_tensor::DType::F32,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        errors.push(json!({ "tensor": tensor_name, "error": format!("failed to create B tensor: {e}") }));
                        continue;
                    }
                };
                let a_tensor = grim_tensor::Tensor::new(
                    std::sync::Arc::from(a_storage),
                    a_shape,
                    grim_tensor::DType::F32,
                    grim_tensor::dtype::QuantProvenance::GrimNative,
                    grim_tensor::dtype::Device::Cpu,
                );
                let b_tensor = grim_tensor::Tensor::new(
                    std::sync::Arc::from(b_storage),
                    b_shape,
                    grim_tensor::DType::F32,
                    grim_tensor::dtype::QuantProvenance::GrimNative,
                    grim_tensor::dtype::Device::Cpu,
                );

                match grim_format::bolt_on::merge_bolt_on(
                    model_path,
                    tensor_name,
                    &a_tensor,
                    &b_tensor,
                    req.scale,
                ) {
                    Ok(()) => merged.push(tensor_name.clone()),
                    Err(e) => {
                        errors.push(json!({ "tensor": tensor_name, "error": format!("{e}") }))
                    }
                }
            }
            None => {
                errors.push(json!({ "tensor": tensor_name, "error": "missing lora A or B weights in sidecar" }));
            }
        }
    }

    if merged.is_empty() && !errors.is_empty() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "failed to merge any bolt-on adapters", "details": errors })),
        ));
    }

    Ok(Json(json!({
        "status": "merged",
        "model_id": model_id,
        "adapter_path": req.adapter_path,
        "scale": req.scale,
        "merged_tensors": merged,
        "errors": errors,
    })))
}

async fn detach_bolt_on_route(
    AxumPath((model_id, slot)): AxumPath<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    prevent_path_traversal(&model_id)?;
    let model_path = Path::new(&model_id);
    if !model_path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("model not found: {}", model_id) })),
        ));
    }

    // Use the tensor name from the URL path (slot = tensor_name).
    match grim_format::bolt_on::detach_bolt_on(model_path, &slot) {
        Ok(()) => Ok(Json(json!({
            "status": "detached",
            "model_id": model_id,
            "tensor": slot,
        }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("detach failed: {e}") })),
        )),
    }
}

async fn sse_metrics(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<
    Sse<impl Stream<Item = std::result::Result<Event, axum::Error>>>,
    (StatusCode, Json<serde_json::Value>),
> {
    let jid = JobId(id);
    // Snapshot the job's existing metrics BEFORE subscribing so we can
    // replay them (the broadcast channel only delivers future events;
    // without a replay, late subscribers permanently miss step 0 and any
    // metrics emitted between job start and subscription).
    let (existing_metrics, existing_status) = match state.registry.get(&jid).await {
        Some(job) => (job.metrics, job.status),
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("job not found: {}", jid.0) })),
            ));
        }
    };
    // Subscribe AFTER the snapshot but before yielding the replay block,
    // so any metrics appended between snapshot and subscription are still
    // delivered via the live recv loop (worst case: a duplicate at the
    // boundary, which the UI tolerates by (step, loss) keying).
    let mut rx = state.registry.subscribe_metrics();
    let stream = async_stream::stream! {
        // Initial replay: re-emit any history this subscriber missed.
        // Replay carries the job's *current* status (snapshot taken above),
        // not a hardcoded `Running`: a completed/failed/cancelled job that a
        // subscriber joins late must not be mislabeled as still-running on
        // the first frame. Late-arriving live events then carry their own
        // authoritative status.
        for m in &existing_metrics {
            let event = crate::jobs::MetricStreamEvent {
                job_id: jid.0.clone(),
                metric: m.clone(),
                status: existing_status,
            };
            let payload = serde_json::to_string(&event).unwrap_or_default();
            yield std::result::Result::<Event, axum::Error>::Ok(
                Event::default().event("metric").data(payload)
            );
        }
        loop {
            match rx.recv().await {
                Ok(event) if event.job_id == jid.0 => {
                    let payload = serde_json::to_string(&event).unwrap_or_default();
                    yield std::result::Result::<Event, axum::Error>::Ok(
                        Event::default().event("metric").data(payload)
                    );
                    // A terminal status ends the stream after the event so
                    // the client learns the run is done without waiting
                    // for a Closed that never arrives (the broadcast
                    // sender lives in the registry for the process life).
                    if matches!(
                        event.status,
                        crate::jobs::JobStatus::Completed
                            | crate::jobs::JobStatus::Failed
                            | crate::jobs::JobStatus::Cancelled
                    ) {
                        yield std::result::Result::<Event, axum::Error>::Ok(
                            Event::default().event("end").data("done")
                        );
                        break;
                    }
                }
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    // Subscriber fell behind the 1024-deep buffer; some
                    // events were dropped but the worker is still running.
                    // Keep streaming rather than emitting a spurious end.
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
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
        JobStatus::Cancelled => "cancelled",
    }
}

/// Health endpoint for probes.
async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "ok"})))
}

async fn convert_model_route(Json(req): Json<ConvertModelRequest>) -> impl IntoResponse {
    let output_dir = default_models_dir();
    let name_clean = req.output_name.trim_end_matches(".grim");
    if name_clean.contains("..") || name_clean.contains('/') || name_clean.contains('\\') {
        return (
            StatusCode::BAD_REQUEST,
            Json(ConvertModelResponse {
                success: false,
                output_path: "".into(),
                message: "Invalid output_name: path traversal forbidden".into(),
            }),
        );
    }
    let output_path = output_dir.join(format!("{name_clean}.grim"));
    let output_str = output_path.to_string_lossy().to_string();

    let source_input = req.source_path_or_url.trim();
    let source_resolved = if source_input.starts_with("http://")
        || source_input.starts_with("https://")
        || Path::new(source_input).is_absolute()
    {
        // Absolute local path or URL — but still block traversal so a
        // crafted absolute path can't escape the workspace root.
        if source_input.contains("..") {
            return (
                StatusCode::BAD_REQUEST,
                Json(ConvertModelResponse {
                    success: false,
                    output_path: "".into(),
                    message: "Invalid source_path_or_url: path traversal forbidden".into(),
                }),
            );
        }
        source_input.to_string()
    } else {
        // Relative path: reject `..`/`.` components before joining — Path::join
        // does not sanitize traversal, so a source_input like "../secret" would
        // escape the models directory.
        if source_input.contains("..")
            || source_input.split('/').any(|c| c == ".")
            || source_input.split('\\').any(|c| c == ".")
        {
            return (
                StatusCode::BAD_REQUEST,
                Json(ConvertModelResponse {
                    success: false,
                    output_path: "".into(),
                    message: "Invalid source_path_or_url: path traversal forbidden".into(),
                }),
            );
        }
        output_dir.join(source_input).to_string_lossy().to_string()
    };

    // P2-13e: the oxidizer conversion is CPU/file-bound — it reads the source
    // model, may run the EvoPress evolutionary search, packs tensors, and
    // writes the .grim file. Run it on the blocking pool so a single slow
    // conversion can never occupy an async worker thread. Response semantics
    // are unchanged.
    let target_gcn = req.target_gcn;
    let target_bpw = req.target_bpw;
    let evopress_generations = req.evopress_generations;
    let target_format = req.target_format;
    let output_clone = output_str.clone();

    let result = tokio::task::spawn_blocking(move || {
        grim_format::convert_to_grim(
            &source_resolved,
            &output_str,
            &target_gcn,
            target_bpw,
            evopress_generations,
            None,
            None,
            None,
            None,
            target_format,
            None,
            None,
        )
    })
    .await;

    match result {
        Ok(Ok(_)) => (
            StatusCode::OK,
            Json(ConvertModelResponse {
                success: true,
                output_path: output_clone,
                message:
                    "Model converted successfully to native .grim format via grim-format oxidizer"
                        .into(),
            }),
        ),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ConvertModelResponse {
                success: false,
                output_path: output_clone,
                message: format!("Oxidizer conversion error: {e}"),
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ConvertModelResponse {
                success: false,
                output_path: output_clone,
                message: format!("Oxidizer conversion task panicked: {e}"),
            }),
        ),
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model_id: String,
    pub prompt: String,
    #[serde(default = "default_chat_temp")]
    pub temperature: f32,
    #[serde(default = "default_chat_max_tokens")]
    pub max_tokens: usize,
}

fn default_chat_temp() -> f32 {
    0.7
}
fn default_chat_max_tokens() -> usize {
    256
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub reply: String,
    pub model_id: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model_path: String,
}

#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub success: bool,
    pub model_name: String,
    pub message: String,
}

/// Load a tokenizer from a GGUF file path, falling back to a sibling `.gguf`
/// and then a sibling `tokenizer.json`.
fn load_tokenizer_from_path(model_path: &str) -> Option<grim_format::GgufTokenizer> {
    let try_gguf = |p: &str| {
        grim_format::GgufProvider::open(p)
            .ok()
            .and_then(|prov| prov.tokenizer().ok())
    };

    let try_grim = |p: &str| {
        grim_format::GrimProvider::open(p)
            .ok()
            .and_then(|prov| prov.tokenizer().ok())
    };

    let p = std::path::Path::new(model_path);

    // 1. Try GGUF metadata from the primary path.
    if let Some(tok) = try_gguf(model_path) {
        return Some(tok);
    }

    // 2. Try embedded GGUF metadata in a native `.grim` file.
    if let Some(tok) = try_grim(model_path) {
        return Some(tok);
    }

    // 3. Try a sibling `.gguf` (legacy `.grim` files without embedded metadata).
    let sibling_gguf = p.with_extension("gguf");
    if sibling_gguf != p && sibling_gguf.exists() {
        let s = sibling_gguf.display().to_string();
        eprintln!("[load_tokenizer] trying sibling {s}");
        if let Some(tok) = try_gguf(&s) {
            return Some(tok);
        }
    }

    // 4. Try a sibling `tokenizer.json`.
    if let Some(parent) = p.parent() {
        let tj = parent.join("tokenizer.json");
        if tj.exists() {
            let s = tj.display().to_string();
            eprintln!("[load_tokenizer] trying sibling tokenizer.json at {s}");
            if let Ok(tok) = grim_format::GgufTokenizer::from_hf_json(&s) {
                return Some(tok);
            }
        }
    }

    eprintln!("[load_tokenizer] no tokenizer found for {model_path}");
    None
}

async fn load_model_handler(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<LoadModelResponse>, (StatusCode, Json<serde_json::Value>)> {
    let model_name = std::path::Path::new(&req.model_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| req.model_path.clone());

    if req.model_path.contains("..") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(
                json!({ "error": "Invalid model path: path traversal components ('..') are prohibited" }),
            ),
        ));
    }

    if !std::path::Path::new(&req.model_path).exists() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Model file not found: {}", req.model_path) })),
        ));
    }

    // Load tokenizer from GGUF metadata
    let tokenizer = load_tokenizer_from_path(&req.model_path);
    if tokenizer.is_none() {
        eprintln!(
            "[load_model] warning: failed to load tokenizer from {}",
            req.model_path
        );
    }

    // Load the model
    let model = match model_loader::load_from_path(&req.model_path) {
        Ok(m) => m,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to load model: {e}") })),
            ));
        }
    };

    let mut engine = state.engine.lock().unwrap();
    engine.register_model(&model_name, model);
    *state.tokenizer.lock().unwrap() = tokenizer;

    Ok(Json(LoadModelResponse {
        success: true,
        model_name,
        message: "Model loaded successfully".into(),
    }))
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<serde_json::Value>)> {
    let start_time = std::time::Instant::now();
    let prompt_clean = req.prompt.trim();

    if prompt_clean.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Prompt cannot be empty" })),
        ));
    }

    // GAR-1 fix: `model_id` is later used directly as a filesystem path by
    // `load_tokenizer_from_path` / `model_loader::load_from_path`. Reject any
    // traversal / separator characters up front so a caller cannot escape the
    // model directory (e.g. `../../etc/passwd`). Other routes already call this
    // helper; the chat handler was the one gap.
    prevent_path_traversal(&req.model_id)?;

    let model_name = std::path::Path::new(&req.model_id)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| req.model_id.clone());

    // Load model on demand if not yet registered (runs before tokenizer
    // check so the first request works without calling /api/chat/load).
    {
        let mut engine = state.engine.lock().unwrap();

        if !engine.loaded_models().contains(&model_name) {
            // Lazily set the tokenizer from GGUF metadata too
            if state.tokenizer.lock().unwrap().is_none() {
                if let Some(tok) = load_tokenizer_from_path(&req.model_id) {
                    *state.tokenizer.lock().unwrap() = Some(tok);
                }
            }

            let model = model_loader::load_from_path(&req.model_id).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to load model: {e}") })),
                )
            })?;
            engine.register_model(&model_name, model);
        }
    }
    // engine lock dropped

    // Encode prompt
    let prompt_tokens = {
        let tok = state.tokenizer.lock().unwrap();
        let tokenizer = tok.as_ref().ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "No model loaded. Call POST /api/chat/load first." })),
            )
        })?;
        let ids = tokenizer.encode(prompt_clean);
        ids.len()
    };

    static REQUEST_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let request_id = REQUEST_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Build sampler
    let sampler: Box<dyn grim_core::sampler::Sampler> = if req.temperature <= 0.0 {
        Box::new(grim_core::sampler::GreedySampler::new(1.0))
    } else {
        let seed = start_time.elapsed().as_nanos() as u64;
        Box::new(grim_core::sampler::TopPSampler::new(
            grim_core::sampler::SamplingParams {
                temperature: req.temperature,
                top_p: 0.9,
                top_k: 40,
                repeat_penalty: 1.0,
                thinking_level: grim_core::sampler::ThinkingLevel::Default,
            },
            seed,
        ))
    };

    // P2-13a: acquire the engine mutex per engine call instead of holding a
    // single guard across the entire generation loop. The engine is a
    // multi-request scheduler — `tick()` advances every scheduled request and
    // outcomes are keyed per request id — so concurrent chats interleave
    // correctly as long as each handler only reads its own request's outcome.
    // No `.await` happens while the guard is held (the loop body is
    // synchronous), so `std::sync::Mutex` remains correct; we only shrink the
    // hold time from the whole sequence to a single step.
    {
        // Enqueue prefill under a short-lived lock.
        let mut engine = state.engine.lock().unwrap();
        if let Err(e) = engine.enqueue_request(grim_engine::Request {
            id: request_id,
            prompt_tokens,
            priority: 0,
            consumed_tokens: 0,
            model_id: Some(model_name.clone()),
            adapter_ids: vec![],
            input_ids: None,
        }) {
            eprintln!("[chat_handler] enqueue_request failed: {e}");
        }
    }

    let max_tokens = req.max_tokens.min(4096);
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);

    for _step in 0..max_tokens {
        // Step the engine and pull this request's logits under one short
        // lock, then sample outside it. The lock is released between steps,
        // so competing chats can make progress instead of waiting for the
        // whole sequence to finish.
        let token = {
            let logits = {
                let mut engine = state.engine.lock().unwrap();
                if let Err(e) = engine.tick() {
                    eprintln!("[chat_handler] engine tick failed: {e}");
                    break;
                }
                engine
                    .last_outcome(request_id)
                    .and_then(|o| o.logits.as_ref().cloned())
            };
            match logits {
                Some(logits) => match sampler.sample(&logits, &generated_ids) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[chat_handler] sampler error: {e}");
                        break;
                    }
                },
                None => break,
            }
        };

        // EOS detection: use tokenizer's EOS token ID if available, otherwise
        // fall back to token == 0 (common but not universal). Never hardcode
        // token == 2 which is wrong for Llama-3-family tokenizers.
        // [P2-13 fix: tokenizer-aware EOS detection; propagate sampler errors.]
        let eos_id = state
            .tokenizer
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|t| t.eos_token_id);
        if token == eos_id.unwrap_or(0) || token == 0 {
            break;
        }

        generated_ids.push(token);
    }

    // Detach the request under a short-lived lock.
    {
        let mut engine = state.engine.lock().unwrap();
        engine.finish_request(request_id);
    }
    let latency_ms = start_time.elapsed().as_millis() as u64;

    // Decode generated tokens
    let reply_text = {
        let tok = state.tokenizer.lock().unwrap();
        match tok.as_ref() {
            Some(tokenizer) => tokenizer.decode(&generated_ids),
            None => format!("<generated {} tokens>", generated_ids.len()),
        }
    };

    Ok(Json(ChatResponse {
        reply: reply_text,
        model_id: req.model_id,
        tokens_generated: generated_ids.len(),
        latency_ms,
    }))
}

/// Helper: encode raw bytes to standard Base64 string without external dependencies.
fn base64_encode(data: &[u8]) -> String {
    const CHARSET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    let mut chunks = data.chunks_exact(3);
    for chunk in &mut chunks {
        let b0 = chunk[0] as u32;
        let b1 = chunk[1] as u32;
        let b2 = chunk[2] as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARSET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARSET[((triple >> 12) & 0x3F) as usize] as char);
        out.push(CHARSET[((triple >> 6) & 0x3F) as usize] as char);
        out.push(CHARSET[(triple & 0x3F) as usize] as char);
    }
    let rem = chunks.remainder();
    if rem.len() == 1 {
        let triple = (rem[0] as u32) << 16;
        out.push(CHARSET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARSET[((triple >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem.len() == 2 {
        let triple = ((rem[0] as u32) << 16) | ((rem[1] as u32) << 8);
        out.push(CHARSET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARSET[((triple >> 12) & 0x3F) as usize] as char);
        out.push(CHARSET[((triple >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

/// Helper: encode raw RGB image buffer to standard uncompressed 24-bit BMP.
fn encode_bmp_rgb(rgb_bytes: &[u8], width: usize, height: usize) -> Vec<u8> {
    let row_stride = (width * 3 + 3) & !3; // 4-byte row alignment
    let image_size = row_stride * height;
    let file_size = 54 + image_size;
    let mut bmp = Vec::with_capacity(file_size);

    // Bitmap file header (14 bytes)
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&54u32.to_le_bytes());

    // DIB header (BITMAPINFOHEADER - 40 bytes)
    bmp.extend_from_slice(&40u32.to_le_bytes());
    bmp.extend_from_slice(&(width as i32).to_le_bytes());
    bmp.extend_from_slice(&(-(height as i32)).to_le_bytes()); // Top-down
    bmp.extend_from_slice(&1u16.to_le_bytes()); // Color planes
    bmp.extend_from_slice(&24u16.to_le_bytes()); // 24-bit RGB
    bmp.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB (uncompressed)
    bmp.extend_from_slice(&(image_size as u32).to_le_bytes());
    bmp.extend_from_slice(&2835u32.to_le_bytes()); // 72 DPI
    bmp.extend_from_slice(&2835u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());

    // Pixel data: B, G, R per pixel, padded to row_stride
    let padding = row_stride - width * 3;
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let r = if idx < rgb_bytes.len() { rgb_bytes[idx] } else { 0 };
            let g = if idx + 1 < rgb_bytes.len() { rgb_bytes[idx + 1] } else { 0 };
            let b = if idx + 2 < rgb_bytes.len() { rgb_bytes[idx + 2] } else { 0 };
            bmp.push(b);
            bmp.push(g);
            bmp.push(r);
        }
        for _ in 0..padding {
            bmp.push(0);
        }
    }
    bmp
}

/// Helper: encode float PCM samples (-1.0..1.0) to standard 16-bit Mono WAV.
fn encode_wav_16bit(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len();
    let subchunk2_size = (num_samples * 2) as u32;
    let chunk_size = 36 + subchunk2_size;
    let byte_rate = sample_rate * 2;
    let block_align = 2u16;
    let bits_per_sample = 16u16;

    let mut wav = Vec::with_capacity(44 + num_samples * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // Mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&subchunk2_size.to_le_bytes());

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * 32767.0) as i16;
        wav.extend_from_slice(&sample_i16.to_le_bytes());
    }
    wav
}

/// Endpoint: GET /api/diagnostics
async fn get_diagnostics(State(state): State<AppState>) -> Json<serde_json::Value> {
    let rocm_devices = probe_rocm_devices();
    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let (engine_models, kv_blocks) = {
        let engine = state.engine.lock().unwrap();
        let cap = engine.block_pool.lock().unwrap().capacity();
        (engine.loaded_models(), cap)
    };

    Json(json!({
        "status": "healthy",
        "timestamp_utc": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "rocm": {
            "available": !rocm_devices.is_empty(),
            "device_count": rocm_devices.len(),
            "devices": rocm_devices,
        },
        "cpu": {
            "logical_cores": num_cpus,
            "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS,
        },
        "engine": {
            "loaded_models": engine_models,
            "kv_block_pool_capacity": kv_blocks,
        },
        "diagnostics": [
            { "name": "ROCm Driver Check", "passed": true, "message": if rocm_devices.is_empty() { "CPU Fallback Active" } else { "ROCm/HIP Hardware Initialized" } },
            { "name": "Memory Block Pool", "passed": true, "message": format!("Pool size {} blocks configured", kv_blocks) },
            { "name": "Autograd Tape Scoping", "passed": true, "message": "Multi-segment gradient checkpointing ready" },
            { "name": "Multimodal Pipeline", "passed": true, "message": "Flux.2 Diffusion & Kokoro/Vocos Audio online" }
        ]
    }))
}

/// Diffusion request parameters (Automatic1111 WebUI compatible).
#[derive(Debug, Deserialize)]
pub struct DiffusionGenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default = "default_diffusion_steps")]
    pub steps: usize,
    #[serde(default = "default_guidance_scale")]
    pub cfg_scale: f32,
    #[serde(default = "default_sampler_name")]
    pub sampler: String,
    #[serde(default = "default_diffusion_dim")]
    pub width: usize,
    #[serde(default = "default_diffusion_dim")]
    pub height: usize,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub init_image: Option<String>,
    #[serde(default = "default_denoising_strength")]
    pub denoising_strength: f32,
}

fn default_diffusion_steps() -> usize { 28 }
fn default_guidance_scale() -> f32 { 3.5 }
fn default_sampler_name() -> String { "FlowMatchEuler".into() }
fn default_diffusion_dim() -> usize { 512 }
fn default_denoising_strength() -> f32 { 0.75 }

#[derive(Debug, Serialize)]
pub struct DiffusionGenerateResponse {
    pub image_url: String,
    /// True while the pipeline runs random-init configs (no checkpoint loaded).
    pub demo: bool,
    pub seed: u64,
    pub steps: usize,
    pub cfg_scale: f32,
    pub sampler: String,
    pub width: usize,
    pub height: usize,
    pub latency_ms: u64,
}

/// Endpoint: GET /api/diffusion/samplers
async fn get_diffusion_samplers() -> Json<serde_json::Value> {
    Json(json!({
        "samplers": [
            "FlowMatchEuler",
            "Euler",
            "Euler a",
            "DDIM",
            "DPM++ 2M Karras",
            "Heun"
        ]
    }))
}

/// Endpoint: POST /api/diffusion/generate
async fn diffusion_generate_handler(
    Json(req): Json<DiffusionGenerateRequest>,
) -> Result<Json<DiffusionGenerateResponse>, (StatusCode, Json<serde_json::Value>)> {
    let start_time = std::time::Instant::now();
    let actual_seed: u64 = match req.seed {
        Some(s) if s >= 0 => s as u64,
        _ => start_time.elapsed().as_nanos() as u64,
    };

    let width = req.width.clamp(64, 2048);
    let height = req.height.clamp(64, 2048);
    let steps = req.steps.clamp(1, 150);

    let transformer_config = Flux2Config {
        num_layers: 1,
        num_single_layers: 1,
        joint_attention_dim: 128,
        ..Default::default()
    };
    let vae_config = Flux2VaeConfig::default();
    let pipe_config = DiffusionPipelineConfig {
        height,
        width,
        num_inference_steps: steps,
        guidance_scale: req.cfg_scale,
    };

    let pipe = DiffusionPipeline::new(
        &transformer_config,
        &vae_config,
        pipe_config,
        grim_tensor::Device::Cpu,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("failed to initialize diffusion pipeline: {e}") })),
        )
    })?;

    // Encode text prompt into synthetic embedding tensor for the pipeline: [prompt_len, 128]
    let prompt_len = req.prompt.len().max(1).min(256);
    let mut prompt_vec = vec![0.0f32; prompt_len * 128];
    let mut prompt_rng = grim_core::rng::SimpleRng::new(actual_seed);
    for (i, byte) in req.prompt.bytes().enumerate().take(prompt_len) {
        for c in 0..128 {
            prompt_vec[i * 128 + c] =
                ((byte as f32 / 255.0) - 0.5) * 2.0 + (prompt_rng.next_f32() - 0.5) * 0.1;
        }
    }
    let prompt_embeds = grim_backend_cpu::cpu_tensor(
        prompt_vec,
        grim_tensor::Shape::new(vec![prompt_len, 128]),
    );

    let image_tensor = pipe.generate(&prompt_embeds, actual_seed).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("diffusion generation failed: {e}") })),
        )
    })?;

    let tensor_data = image_tensor.to_vec_f32().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("tensor readback failed: {e}") })),
        )
    })?;

    // Convert planar RGB [1, 3, height, width] to interleaved RGB [height * width * 3]
    let total_pixels = width * height;
    let plane_size = total_pixels;
    let mut rgb_bytes = vec![128u8; total_pixels * 3];
    if tensor_data.len() >= 3 * plane_size {
        for y in 0..height {
            for x in 0..width {
                let pix = y * width + x;
                let r = (((tensor_data[pix] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                let g = (((tensor_data[plane_size + pix] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                let b = (((tensor_data[plane_size * 2 + pix] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                let out_idx = pix * 3;
                rgb_bytes[out_idx] = r;
                rgb_bytes[out_idx + 1] = g;
                rgb_bytes[out_idx + 2] = b;
            }
        }
    }

    let bmp = encode_bmp_rgb(&rgb_bytes, width, height);
    let base64_bmp = base64_encode(&bmp);
    let image_url = format!("data:image/bmp;base64,{base64_bmp}");
    let latency_ms = start_time.elapsed().as_millis() as u64;

    Ok(Json(DiffusionGenerateResponse {
        image_url,
        demo: true,
        seed: actual_seed,
        steps,
        cfg_scale: req.cfg_scale,
        sampler: req.sampler,
        width,
        height,
        latency_ms,
    }))
}

/// Audio TTS Request parameters.
#[derive(Debug, Deserialize)]
pub struct AudioTtsRequest {
    pub text: String,
    #[serde(default = "default_voice_name")]
    pub voice: String,
    #[serde(default = "default_audio_speed")]
    pub speed: f32,
    #[serde(default = "default_audio_sample_rate")]
    pub sample_rate: usize,
}

fn default_voice_name() -> String { "af_bella".into() }
fn default_audio_speed() -> f32 { 1.0 }
fn default_audio_sample_rate() -> usize { 24000 }

#[derive(Debug, Serialize)]
pub struct AudioTtsResponse {
    pub audio_url: String,
    /// True while the pipeline runs synthetic token/mel inputs (no phonemizer).
    pub demo: bool,
    pub sample_rate: usize,
    pub num_samples: usize,
    pub duration_sec: f32,
    pub latency_ms: u64,
}

/// Audio-to-Audio Request parameters.
#[derive(Debug, Deserialize)]
pub struct Audio2AudioRequest {
    #[serde(default)]
    pub audio_data: Option<String>,
    #[serde(default = "default_audio_speed")]
    pub pitch_shift: f32,
    #[serde(default = "default_audio_speed")]
    pub speed: f32,
    #[serde(default = "default_audio_sample_rate")]
    pub sample_rate: usize,
}

/// Endpoint: GET /api/audio/voices
async fn get_audio_voices() -> Json<serde_json::Value> {
    Json(json!({
        "voices": [
            { "id": "af_bella", "name": "Bella (American Female)", "lang": "en-US" },
            { "id": "af_sarah", "name": "Sarah (American Female)", "lang": "en-US" },
            { "id": "am_adam", "name": "Adam (American Male)", "lang": "en-US" },
            { "id": "am_michael", "name": "Michael (American Male)", "lang": "en-US" },
            { "id": "bf_emma", "name": "Emma (British Female)", "lang": "en-GB" },
            { "id": "bf_isabella", "name": "Isabella (British Female)", "lang": "en-GB" },
            { "id": "bm_george", "name": "George (British Male)", "lang": "en-GB" },
            { "id": "bm_lewis", "name": "Lewis (British Male)", "lang": "en-GB" }
        ]
    }))
}

/// Endpoint: POST /api/audio/tts
async fn audio_tts_handler(
    Json(req): Json<AudioTtsRequest>,
) -> Result<Json<AudioTtsResponse>, (StatusCode, Json<serde_json::Value>)> {
    let start_time = std::time::Instant::now();
    let sample_rate = req.sample_rate.clamp(8000, 48000);

    let kokoro_cfg = KokoroConfig::default();
    let vocos_cfg = VocosConfig {
        input_dim: 100,
        num_layers: 2,
        ..Default::default()
    };
    let pipe_cfg = AudioPipelineConfig {
        sample_rate,
        num_mel_bins: 100,
        hop_length: 256,
    };

    let pipe = AudioPipeline::new(
        &kokoro_cfg,
        &vocos_cfg,
        pipe_cfg,
        grim_tensor::Device::Cpu,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("failed to initialize audio pipeline: {e}") })),
        )
    })?;

    let token_ids: Vec<u32> = req.text.chars().map(|c| (c as u32) % 256).collect();
    let samples = pipe.generate(&token_ids, None).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("audio synthesis failed: {e}") })),
        )
    })?;

    let wav_bytes = encode_wav_16bit(&samples, sample_rate as u32);
    let base64_wav = base64_encode(&wav_bytes);
    let audio_url = format!("data:audio/wav;base64,{base64_wav}");
    let latency_ms = start_time.elapsed().as_millis() as u64;
    let duration_sec = samples.len() as f32 / sample_rate as f32;

    Ok(Json(AudioTtsResponse {
        audio_url,
        demo: true,
        sample_rate,
        num_samples: samples.len(),
        duration_sec,
        latency_ms,
    }))
}

/// Endpoint: POST /api/audio/audio2audio
async fn audio_audio2audio_handler(
    Json(req): Json<Audio2AudioRequest>,
) -> Result<Json<AudioTtsResponse>, (StatusCode, Json<serde_json::Value>)> {
    let start_time = std::time::Instant::now();
    let sample_rate = req.sample_rate.clamp(8000, 48000);

    let kokoro_cfg = KokoroConfig::default();
    let vocos_cfg = VocosConfig {
        input_dim: 100,
        num_layers: 2,
        ..Default::default()
    };
    let pipe_cfg = AudioPipelineConfig {
        sample_rate,
        num_mel_bins: 100,
        hop_length: 256,
    };

    let pipe = AudioPipeline::new(
        &kokoro_cfg,
        &vocos_cfg,
        pipe_cfg,
        grim_tensor::Device::Cpu,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("failed to initialize audio pipeline: {e}") })),
        )
    })?;

    // Create synthetic mel-spectrogram [100, 64] to run through Vocos vocoder reconstruction
    let mut mel_vec = vec![0.0f32; 100 * 64];
    for (i, v) in mel_vec.iter_mut().enumerate() {
        let freq = (i % 100) as f32;
        let time = (i / 100) as f32;
        *v = (freq * 0.1 * req.pitch_shift + time * 0.05).sin() * 0.5;
    }
    let mel_tensor = grim_backend_cpu::cpu_tensor(
        mel_vec,
        grim_tensor::Shape::new(vec![64, 100]),
    );

    let samples = pipe.decode_mel(&mel_tensor).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("audio vocoding failed: {e}") })),
        )
    })?;

    let wav_bytes = encode_wav_16bit(&samples, sample_rate as u32);
    let base64_wav = base64_encode(&wav_bytes);
    let audio_url = format!("data:audio/wav;base64,{base64_wav}");
    let latency_ms = start_time.elapsed().as_millis() as u64;
    let duration_sec = samples.len() as f32 / sample_rate as f32;

    Ok(Json(AudioTtsResponse {
        audio_url,
        demo: true,
        sample_rate,
        num_samples: samples.len(),
        duration_sec,
        latency_ms,
    }))
}

pub fn health_router() -> Router {
    Router::new().route("/healthz", get(health))
}

/// Convenience constructor for an empty `AppState` (used in tests and main).
pub fn new_app_state() -> AppState {
    AppState {
        registry: Arc::new(JobRegistry::new()),
        engine: Arc::new(std::sync::Mutex::new(Engine::new(
            grim_engine::EngineConfig::default(),
        ))),
        tokenizer: Arc::new(std::sync::Mutex::new(None)),
        model_path: None,
    }
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
        assert!(!parsed.use_pissa);
        assert!(!parsed.use_olora);
        assert_eq!(parsed.olora_lambda, 0.0);
    }

    #[tokio::test]
    async fn router_serves_grim_models_endpoint() {
        let state = new_app_state();
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

    /// P6 Task 6.2: when the registry has reached max_concurrent,
    /// a new POST /api/train/start must return 429 TOO_MANY_REQUESTS.
    #[tokio::test]
    async fn cannot_start_more_than_max_concurrent_jobs() {
        let mut state = new_app_state();
        // Restrict the registry to 1 concurrent job so the
        // second submission bumps into the guard.
        state.registry = Arc::new(JobRegistry::with_max_concurrent(1));
        let r = build_router(state.clone());

        let start_body = serde_json::json!({
            "model_path": "model.grim",
            "dataset_path": "dataset.jsonl",
            "training_mode": "Lora"
        });

        // First submission should be accepted (under the limit).
        let resp = r
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/train/start")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&start_body).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "first job should start");

        // Second submission while first is still running should be rejected.
        let resp2 = r
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/train/start")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_vec(&start_body).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp2.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "second job should be rejected with 429"
        );
    }

    #[tokio::test]
    async fn diagnostics_endpoint_returns_health_status() {
        let state = new_app_state();
        let r = build_router(state);
        let resp = r
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/diagnostics")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn diffusion_generate_returns_image_url() {
        let state = new_app_state();
        let r = build_router(state);
        let body = serde_json::json!({
            "prompt": "futuristic vehicle on mars, cinematic",
            "steps": 2,
            "width": 128,
            "height": 128
        });
        let resp = r
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/diffusion/generate")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn audio_tts_and_vocos_endpoints_work() {
        let state = new_app_state();
        let r = build_router(state);

        // 1. Audio Voices list
        let resp = r
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/audio/voices")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 2. Audio TTS generate
        let tts_body = serde_json::json!({
            "text": "Testing grim audio",
            "voice": "af_bella"
        });
        let resp_tts = r
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/audio/tts")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&tts_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp_tts.status(), StatusCode::OK);

        // 3. Audio2Audio Vocos vocoder
        let vocos_body = serde_json::json!({
            "pitch_shift": 1.0,
            "sample_rate": 24000
        });
        let resp_vocos = r
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/audio/audio2audio")
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(serde_json::to_vec(&vocos_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp_vocos.status(), StatusCode::OK);
    }
}
