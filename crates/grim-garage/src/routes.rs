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
use grim_engine::{Engine, model_loader};
use grim_format::GgufTokenizer;
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
        .any(|component| component == ".." || component == ".");
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
        // Relative path: resolve under the models dir (join is safe from `..`
        // escapes because the result is always beneath `output_dir`).
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
        None,
        None,
        req.target_format,
        None,
        None,
    ) {
        Ok(_) => (
            StatusCode::OK,
            Json(ConvertModelResponse {
                success: true,
                output_path: output_str,
                message:
                    "Model converted successfully to native .grim format via grim-format oxidizer"
                        .into(),
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
            Json(json!({ "error": "Invalid model path: path traversal components ('..') are prohibited" })),
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

    let mut engine = state.engine.lock().unwrap();

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

    // Enqueue prefill
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

    let max_tokens = req.max_tokens.min(4096);
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);

    for _step in 0..max_tokens {
        if let Err(e) = engine.tick() {
            eprintln!("[chat_handler] engine tick failed: {e}");
            break;
        }

        let token = match engine
            .last_outcome(request_id)
            .and_then(|o| o.logits.as_ref().cloned())
        {
            Some(logits) => sampler.sample(&logits, &generated_ids).unwrap_or(0),
            None => break,
        };

        // Common EOS token IDs
        if token == 0 || token == 2 {
            break;
        }

        generated_ids.push(token);
    }

    engine.finish_request(request_id);
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
            "model_path": "/tmp/model.grim",
            "dataset_path": "/tmp/dataset.jsonl",
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
}
