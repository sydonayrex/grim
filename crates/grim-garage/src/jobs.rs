//! Training jobs: in-memory state machine + tokio task lifecycle.
//!
//! The UI submits a `TrainingJob` via `POST /api/train/start`; the server
//! hands the job id to a worker task and reports status through:
//!   - `GET   /api/train/status/:id`   — single snapshot
//!   - `SSE   /sse/metrics/:id`        — live loss/vram telemetry
//!
//! Workers record per-step metrics into `job.metrics` as they run; the
//! `metrics_watcher` emits each new metric to subscribed SSE clients via
//! a `tokio::sync::broadcast` channel.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum JobError {
    #[error("job not found: {0}")]
    NotFound(String),
    #[error("duplicate job id")]
    Duplicate,
}

/// Coarse job status surface — enough for the UI badge in the history list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl Default for JobStatus {
    fn default() -> Self {
        JobStatus::Pending
    }
}

/// Training mode the UI's "Training Mode" dropdown drives.
///
/// SFT modes: `Lora`, `QLoRA`, `Bf16Full`.
/// Reinforcement-learning modes: `Orpo`, `Dpo`, `Grpo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingMode {
    /// LoRA supervised fine-tuning on compressed weights.
    Lora,
    /// Quantized LoRA — LoRA adapters with block-quantized base weights.
    QLoRA,
    /// Full BF16 supervised fine-tuning (unpacked weights).
    Bf16Full,
    /// Odds-Ratio Preference Optimization (HLRF reinforcement).
    Orpo,
    /// Direct Preference Optimization (HLRF reinforcement).
    Dpo,
    /// Group Relative Policy Optimization (HLRF reinforcement, DeepSeek-R1-style).
    Grpo,
}

/// One per-step metric sample: step id, loss, tokens processed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub step: u64,
    pub loss: f64,
    pub tokens: u64,
}

/// Configuration for a training job — what the React UI submits verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub model_path: String,
    pub dataset_path: String,
    pub training_mode: TrainingMode,
    pub lora_rank: u32,
    pub learning_rate: f64,
    pub epochs: u32,
    pub rocm_fusion_rmsnorm_matmul: bool,
    pub rocm_fusion_qkv_attention: bool,
    /// Mutable state shared with the worker task.
    #[serde(skip)]
    pub status: JobStatus,
    #[serde(skip)]
    pub metrics: Vec<Metric>,
}

impl Default for TrainingJob {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            dataset_path: String::new(),
            training_mode: TrainingMode::Lora,
            lora_rank: 16,
            learning_rate: 2e-5,
            epochs: 1,
            rocm_fusion_rmsnorm_matmul: false,
            rocm_fusion_qkv_attention: false,
            status: JobStatus::Pending,
            metrics: Vec::new(),
        }
    }
}

impl TrainingJob {
    /// Append a metric sample. Used by worker tasks and by tests.
    pub fn push_metric(&mut self, step: u64, loss: f64, tokens: u64) {
        self.metrics.push(Metric { step, loss, tokens });
    }
}

/// Strongly typed UUID wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub String);

impl JobId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Live metric stream sent to SSE subscribers.
#[derive(Debug, Clone, Serialize)]
pub struct MetricStreamEvent {
    pub job_id: String,
    pub metric: Metric,
    pub status: JobStatus,
}

/// In-memory registry of training jobs. Shared via `Arc<RwLock<_>>` between
/// the HTTP server and the worker tasks that update metrics.
#[derive(Debug)]
pub struct JobRegistry {
    inner: Arc<RwLock<HashMap<JobId, TrainingJob>>>,
    metrics_tx: broadcast::Sender<MetricStreamEvent>,
}

impl Default for JobRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl JobRegistry {
    pub fn new() -> Self {
        // Buffer up to 1024 metrics; slow clients drop events rather than block workers.
        let (metrics_tx, _) = broadcast::channel(1024);
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            metrics_tx,
        }
    }

    /// Create a new job with a freshly-generated id. Stored as `Pending`.
    /// Returns the new id so the caller can hand it back to the UI immediately.
    pub async fn create(&self, job: TrainingJob) -> Result<JobId, JobError> {
        let id = JobId::new();
        let mut g = self.inner.write().await;
        g.insert(id.clone(), job);
        Ok(id)
    }

    /// Insert with an explicit id. Used by tests to verify duplicate rejection.
    pub async fn insert_with_id(&self, id: JobId, job: TrainingJob) -> Result<JobId, JobError> {
        let mut g = self.inner.write().await;
        if g.contains_key(&id) {
            return Err(JobError::Duplicate);
        }
        g.insert(id.clone(), job);
        Ok(id)
    }

    pub async fn get(&self, id: &JobId) -> Option<TrainingJob> {
        let g = self.inner.read().await;
        g.get(id).cloned()
    }

    pub async fn list(&self) -> Vec<(JobId, JobStatus)> {
        let g = self.inner.read().await;
        g.iter()
            .map(|(k, v)| (k.clone(), v.status))
            .collect::<Vec<_>>()
    }

    pub async fn update_status(&self, id: &JobId, status: JobStatus) -> Result<(), JobError> {
        let mut g = self.inner.write().await;
        let job = g.get_mut(id).ok_or_else(|| JobError::NotFound(id.0.clone()))?;
        job.status = status;
        Ok(())
    }

    pub async fn append_metric(&self, id: &JobId, metric: Metric) -> Result<(), JobError> {
        let mut g = self.inner.write().await;
        let job = g.get_mut(id).ok_or_else(|| JobError::NotFound(id.0.clone()))?;
        let status = job.status;
        job.push_metric(metric.step, metric.loss, metric.tokens);
        // Best-effort broadcast; if there are no subscribers (SSE clients) this returns Err
        // and we just ignore — the next subscriber would need a snapshot via /api/train/status.
        let _ = self.metrics_tx.send(MetricStreamEvent {
            job_id: id.0.clone(),
            metric,
            status,
        });
        Ok(())
    }

    /// Subscribe to the live metric stream. Each subscriber gets every subsequent event.
    pub fn subscribe_metrics(&self) -> broadcast::Receiver<MetricStreamEvent> {
        self.metrics_tx.subscribe()
    }
}

/// Compute a baseline loss for the given training mode.
///
/// SFT modes start from an empirical cross-entropy target (~2.3);
/// RL modes use an initial reward differential of 0.0 converging upward.
fn initial_loss(mode: TrainingMode) -> f64 {
    match mode {
        TrainingMode::Lora | TrainingMode::QLoRA | TrainingMode::Bf16Full => 2.3,
        TrainingMode::Orpo | TrainingMode::Dpo | TrainingMode::Grpo => 0.0,
    }
}

/// Execute a training job inside a Tokio background task.
///
/// The caller should spawn this with `tokio::spawn`:
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use grim_garage::jobs::{JobId, JobRegistry, run_training_worker};
/// # async fn example(registry: Arc<JobRegistry>, job_id: JobId) {
/// tokio::spawn(run_training_worker(registry.clone(), job_id));
/// # }
/// ```
///
/// Contract:
/// - Transitions `Pending → Running` immediately.
/// - Emits one `Metric` event per simulated step (200 ms sleep).
/// - On completion, transitions to `Completed`.
/// - On any registry error, transitions to `Failed` and logs the error.
pub async fn run_training_worker(registry: Arc<JobRegistry>, id: JobId) {
    // Retrieve the job configuration.
    let job = match registry.get(&id).await {
        Some(j) => j,
        None => {
            eprintln!("[grim-garage] worker: job {} not found — aborting", id);
            return;
        }
    };

    let mode = job.training_mode;
    let epochs = job.epochs.max(1) as u64;
    // Simulated steps per epoch; real implementors would use dataset_len / batch_size.
    let steps_per_epoch: u64 = 10;
    let total_steps = epochs * steps_per_epoch;

    // Transition → Running
    if let Err(e) = registry.update_status(&id, JobStatus::Running).await {
        eprintln!("[grim-garage] worker: failed to mark {} Running: {e}", id);
        return;
    }
    eprintln!("[grim-garage] worker: job {} started (mode={mode:?}, epochs={epochs})", id);

    let mut loss = initial_loss(mode);
    // Exponential decay factor per step for SFT modes.
    let decay: f64 = 0.85;

    for step in 0..total_steps {
        // Simulate one training step.
        // SFT: loss decays exponentially toward zero.
        // RL:  reward differential rises; reported as negative "loss" for chart compat.
        loss = match mode {
            TrainingMode::Lora | TrainingMode::QLoRA | TrainingMode::Bf16Full => {
                loss * decay + rand_noise(0.02)
            }
            TrainingMode::Orpo | TrainingMode::Dpo | TrainingMode::Grpo => {
                let reward = (step as f64 / total_steps as f64) + rand_noise(0.05);
                -reward
            }
        };

        let metric = Metric { step, loss, tokens: (step + 1) * 512 };
        if let Err(e) = registry.append_metric(&id, metric).await {
            eprintln!("[grim-garage] worker: metric append failed for {}: {e}", id);
            let _ = registry.update_status(&id, JobStatus::Failed).await;
            return;
        }

        // Yield so other tasks can run; a real trainer awaits GPU kernel completion here.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    if let Err(e) = registry.update_status(&id, JobStatus::Completed).await {
        eprintln!("[grim-garage] worker: failed to mark {} Completed: {e}", id);
    } else {
        eprintln!("[grim-garage] worker: job {} completed successfully", id);
    }
}

/// Minimal pseudo-random noise for the step simulator.
/// Uses the system-time nanosecond sub-second counter as a lightweight seed.
/// This is acceptable for non-security use (training loss jitter simulation only).
fn rand_noise(amplitude: f64) -> f64 {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as f64;
    // Map nanos in [0, 1e9) → [-amplitude, +amplitude].
    (seed / 1_000_000_000.0 - 0.5) * amplitude * 2.0
}
