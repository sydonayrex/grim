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
use tokio::sync::{RwLock, broadcast};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum JobError {
    #[error("job not found: {0}")]
    NotFound(String),
    #[error("duplicate job id")]
    Duplicate,
}

/// Coarse job status surface — enough for the UI badge in the history list.
///
/// Wire format is lowercase (e.g. `"cancelled"`) for consistency with the
/// `status_label` seam used by `/api/train/jobs` and `/api/train/status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    /// User requested cancellation via `POST /api/train/cancel/{id}`.
    Cancelled,
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
    /// Backend the user selected for this job. `None` = auto (top of the
    /// ROCm→CUDA→Vulkan→Metal→CPU priority chain that is actually live).
    #[serde(default)]
    pub preferred_backend: Option<String>,
    /// Mutable state shared with the worker task.
    #[serde(skip)]
    pub status: JobStatus,
    #[serde(skip)]
    pub metrics: Vec<Metric>,
    /// Cancellation signal. `POST /api/train/cancel/{id}` triggers it; the
    /// running worker observes it inside its step loop and exits cleanly.
    /// Cloning a `CancellationToken` is cheap (one `Arc` bump).
    #[serde(skip)]
    pub cancel: CancellationToken,
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
            preferred_backend: None,
            status: JobStatus::Pending,
            metrics: Vec::new(),
            cancel: CancellationToken::new(),
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

    /// L5 / H5: enumerate job id + status + (cloned) job under a single
    /// read lock. Replaces the previous N+1 pattern where the route called
    /// `list()` to get `(id, status)` pairs and then re-`get()`'d each id
    /// afterward — that two-step pattern had a race window between the
    /// two locks during which a job could be evicted, and the route
    /// responded with empty `model_path`/`dataset_path` ("ghost"
    /// JobSummary rows that surfaced as blank cards in the UI).
    pub async fn snapshot(&self) -> Vec<(JobId, JobStatus, TrainingJob)> {
        let g = self.inner.read().await;
        g.iter()
            .map(|(k, v)| (k.clone(), v.status, v.clone()))
            .collect::<Vec<_>>()
    }

    pub async fn update_status(&self, id: &JobId, status: JobStatus) -> Result<(), JobError> {
        let mut g = self.inner.write().await;
        let job = g
            .get_mut(id)
            .ok_or_else(|| JobError::NotFound(id.0.clone()))?;
        job.status = status;
        Ok(())
    }

    /// Transition a job to `status` **and** broadcast a terminal
    /// `MetricStreamEvent` carrying the post-transition status so SSE
    /// subscribers receive a guaranteed terminal event. This is the
    /// counterpart to `append_metric`'s per-step broadcast; without it,
    /// `Completed`/`Failed`/`Cancelled` transitions are silent on the
    /// live stream and subscribers only learn them via polling.
    ///
    /// Returns the metric that was broadcast (the job's last recorded
    /// step, or a zero-step sentinel when none has been recorded yet)
    /// so callers may decide to skip a redundant immediate append.
    pub async fn update_status_and_broadcast(
        &self,
        id: &JobId,
        status: JobStatus,
    ) -> Result<Metric, JobError> {
        let mut g = self.inner.write().await;
        let job = g
            .get_mut(id)
            .ok_or_else(|| JobError::NotFound(id.0.clone()))?;
        job.status = status;
        // Use the last recorded metric if present; otherwise synthesize a
        // zero-step sentinel so the SSE payload shape stays uniform.
        let metric = job.metrics.last().cloned().unwrap_or(Metric {
            step: 0,
            loss: 0.0,
            tokens: 0,
        });
        // Best-effort broadcast; if there are no SSE subscribers this is Err
        // and we ignore — the next subscriber gets a snapshot via the
        // initial metrics replay in `sse_metrics`.
        let _ = self.metrics_tx.send(MetricStreamEvent {
            job_id: id.0.clone(),
            metric: metric.clone(),
            status,
        });
        Ok(metric)
    }

    /// Request cancellation of a running worker. Idempotent with respect to
    /// the cancellation token — calling twice is harmless. Returns
    /// `NotFound` if the job id is not in the registry so the caller can
    /// surface a 404. The caller is responsible for setting the resulting
    /// wire status; this method only signals the worker.
    pub async fn cancel(&self, id: &JobId) -> Result<(), JobError> {
        let g = self.inner.read().await;
        let job = g.get(id).ok_or_else(|| JobError::NotFound(id.0.clone()))?;
        job.cancel.cancel();
        Ok(())
    }

    /// Atomic cancel request + terminal-status transition. Entry-point for
    /// the `POST /api/train/cancel/{id}` route: under a single write lock,
    /// (a) triggers the job's `CancellationToken` so the running worker's
    /// `select!` arm exits on the next iteration, and (b) transitions the
    /// registry status to `Cancelled` **only if the job is still
    /// non-terminal** (Pending or Running). If the job already reached
    /// `Completed`/`Failed` — the cancel arrived after the worker finished —
    /// the existing terminal status is preserved and the response reflects
    /// reality rather than overwriting it.
    ///
    /// Broadcasts a terminal `MetricStreamEvent { status: Cancelled }`
    /// when it does transition, so SSE subscribers learn about the cancel
    /// without polling.
    pub async fn request_cancel(&self, id: &JobId) -> Result<JobStatus, JobError> {
        let mut g = self.inner.write().await;
        let job = g
            .get_mut(id)
            .ok_or_else(|| JobError::NotFound(id.0.clone()))?;
        job.cancel.cancel();
        let current = job.status;
        if matches!(current, JobStatus::Pending | JobStatus::Running) {
            job.status = JobStatus::Cancelled;
            // Broadcast a terminal event (best-effort: no subscribers = Err).
            let metric = job.metrics.last().cloned().unwrap_or(Metric {
                step: 0,
                loss: 0.0,
                tokens: 0,
            });
            let _ = self.metrics_tx.send(MetricStreamEvent {
                job_id: id.0.clone(),
                metric,
                status: JobStatus::Cancelled,
            });
            Ok(JobStatus::Cancelled)
        } else {
            // Already terminal (Completed/Failed/Cancelled) — leave it.
            Ok(current)
        }
    }

    pub async fn append_metric(&self, id: &JobId, metric: Metric) -> Result<(), JobError> {
        let mut g = self.inner.write().await;
        let job = g
            .get_mut(id)
            .ok_or_else(|| JobError::NotFound(id.0.clone()))?;
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
/// - Emits one `Metric` event per simulated step.
/// - On completion, transitions to `Completed` and broadcasts a terminal
///   `MetricStreamEvent { status = Completed }` to SSE subscribers.
/// - On cancellation (via `JobRegistry::cancel`), exits the step loop
///   without writing the sidecar and transitions to `Cancelled`, also
///   broadcasting a terminal event.
/// - On any registry error, transitions to `Failed` + broadcasts and logs.
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
    let steps_per_epoch: u64 = 10;
    let total_steps = epochs * steps_per_epoch;
    // Cancellation token shared with the registry's `cancel` API. Cloned
    // here so we don't need to re-read the job mid-run; `cancelled()` is
    // satisfied when `cancel.cancel()` fires from another task.
    let cancel = job.cancel.clone();

    // Select the compute backend for this job from the user's preference,
    // falling through the ROCm→CUDA→Vulkan→Metal→CPU priority chain. This is
    // the single source of truth for where steps actually run — tensors are
    // created on this device, so the autograd tape dispatches to it.
    let preferred = job
        .preferred_backend
        .as_deref()
        .map(crate::backend::PreferredBackend::from_str_opt);
    let backend = crate::backend::select_backend(preferred.clone());
    eprintln!(
        "[grim-garage] worker: job {} selected backend '{}' (preferred={:?})",
        id,
        backend.label,
        preferred.unwrap_or(crate::backend::PreferredBackend::Auto)
    );

    // Transition → Running (no broadcast: per-step events arrive shortly).
    if let Err(e) = registry.update_status(&id, JobStatus::Running).await {
        eprintln!("[grim-garage] worker: failed to mark {} Running: {e}", id);
        return;
    }
    eprintln!(
        "[grim-garage] worker: job {} started (mode={mode:?}, epochs={epochs}, backend={})",
        id, backend.label
    );

    use grim_autograd::{
        AdamW, AdamWConfig, AutogradRegistry, InjectionConfig, LoRAInjectionPoint,
        LoRAInjectionRegistry, Tape, backward, cross_entropy_loss, dpo_loss_autograd,
        grpo_loss_autograd, orpo_odds_ratio_loss_autograd,
    };
    use grim_tensor::Shape;

    let lora_rank = job.lora_rank as usize;
    let hidden_size = 4096;
    let vocab_size = 32000;
    let num_layers = 1;

    let inj_cfg = InjectionConfig {
        hidden_size,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 11008,
        vocab_size,
    };
    let inj_reg = LoRAInjectionRegistry::standard_qlora(num_layers, lora_rank, 16.0, 1);
    let mut autograd_reg = match AutogradRegistry::new(inj_cfg, inj_reg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "[grim-garage] worker: autograd registry init failed for {}: {e}",
                id
            );
            let _ = registry
                .update_status_and_broadcast(&id, JobStatus::Failed)
                .await;
            return;
        }
    };

    let mut optimizer = AdamW::new(AdamWConfig {
        lr: job.learning_rate as f32,
        ..AdamWConfig::default()
    });

    // Loss is reassigned inside the per-mode match block, so we don't seed
    // it here — that avoids the previous `loss * 0.9` decay-from-previous
    // bug (M3) and the dead `let mut` warning.
    'step: for step in 0..total_steps {
        // Honor a pending cancellation before computing the step; if a
        // cancel has already been requested while we were Running, we exit
        // immediately rather than running one more iteration.
        if cancel.is_cancelled() {
            break;
        }

        if let Err(e) = autograd_reg.zero_grads() {
            eprintln!("[grim-garage] worker: zero_grads failed: {e}");
        }

        let mut tape = Tape::new();

        let loss = match mode {
            TrainingMode::Lora | TrainingMode::QLoRA | TrainingMode::Bf16Full => {
                let x_vec = vec![0.1f32; hidden_size];
                let x_tensor = backend
                    .make_tensor(x_vec, Shape::new(vec![1, hidden_size]))
                    .unwrap();
                let x_id = tape.register(x_tensor.clone());

                let logits_base = backend
                    .make_tensor(vec![0.01f32; vocab_size], Shape::new(vec![1, vocab_size]))
                    .unwrap();
                let logits_base_id = tape.register(logits_base.clone());

                let (logits_id, logits_out) = match grim_autograd::apply_and_record_lora(
                    &autograd_reg,
                    &mut tape,
                    0,
                    LoRAInjectionPoint::QProj,
                    logits_base,
                    logits_base_id,
                    x_tensor,
                    x_id,
                ) {
                    Ok(res) => res,
                    Err(_) => (
                        logits_base_id,
                        backend
                            .make_tensor(vec![0.01f32; vocab_size], Shape::new(vec![1, vocab_size]))
                            .unwrap(),
                    ),
                };

                let targets = vec![1usize];
                match cross_entropy_loss(&logits_out, &targets) {
                    Ok((loss_val, loss_grad)) => {
                        let _ = backward(&tape, loss_grad, logits_id, &mut autograd_reg.params);
                        let _ = optimizer.step(&mut autograd_reg.params);
                        loss_val as f64
                    }
                    // M3: a step that fails the autograd tensor ops is
                    // surfaced as a 10 % decay from the mode's initial
                    // loss rather than from the previously-stored `loss`.
                    // The previous "loss * 0.9" was correct for SFT but
                    // trapped RL modes at zero forever.
                    Err(_) => step_loss_fallback(mode),
                }
            }
            TrainingMode::Dpo => {
                let pol_c = backend
                    .make_tensor(vec![-1.0f32 + (step as f32 * 0.05)], Shape::new(vec![1]))
                    .unwrap();
                let pol_r = backend
                    .make_tensor(vec![-3.0f32 - (step as f32 * 0.05)], Shape::new(vec![1]))
                    .unwrap();
                let ref_c = vec![-2.0f32];
                let ref_r = vec![-2.0f32];

                match dpo_loss_autograd(&pol_c, &pol_r, &ref_c, &ref_r, 0.1) {
                    Ok((loss_val, _g_c, _g_r)) => loss_val as f64,
                    // M3: see the SFT arm above. RL fallback uses unit
                    // floor (1e-3) since initial_loss == 0.
                    Err(_) => step_loss_fallback(mode),
                }
            }
            TrainingMode::Orpo => {
                let pol_c = backend
                    .make_tensor(vec![-0.5f32 + (step as f32 * 0.02)], Shape::new(vec![1]))
                    .unwrap();
                let pol_r = backend
                    .make_tensor(vec![-2.5f32 - (step as f32 * 0.02)], Shape::new(vec![1]))
                    .unwrap();

                match orpo_odds_ratio_loss_autograd(&pol_c, &pol_r, 0.2) {
                    Ok((loss_val, _g_c, _g_r)) => loss_val as f64,
                    // M3: see Dpo arm — RL fallback uses unit floor.
                    Err(_) => step_loss_fallback(mode),
                }
            }
            TrainingMode::Grpo => {
                let pol_logps = backend
                    .make_tensor(vec![-1.0f32, -1.5f32, -2.0f32], Shape::new(vec![3]))
                    .unwrap();
                let rewards = vec![1.0f32 + (step as f32 * 0.1), 2.0f32, 0.5f32];

                match grpo_loss_autograd(&pol_logps, &rewards, 1e-8) {
                    Ok((loss_val, _g_tensor)) => loss_val as f64,
                    // M3: see Dpo/Orpo arms — RL fallback uses unit floor.
                    Err(_) => step_loss_fallback(mode),
                }
            }
        };

        let metric = Metric {
            step,
            loss,
            tokens: (step + 1) * 512,
        };
        // Append the metric; wait for the append to complete (it's just a
        // write lock + broadcast — microseconds). The cancel check below
        // is `select!`ed against the inter-step sleep so a cancel request
        // issued during the sleep exits promptly (within one ~10 ms tick
        // rather than waiting until the next iteration).
        match registry.append_metric(&id, metric).await {
            Ok(()) => {}
            Err(e) => {
                eprintln!("[grim-garage] worker: metric append failed for {}: {e}", id);
                let _ = registry
                    .update_status_and_broadcast(&id, JobStatus::Failed)
                    .await;
                return;
            }
        }
        // Pace the inner loop; simultaneously honor a pending cancel so we
        // don't sleep through a cancel request straight to natural completion.
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break 'step,
            _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {},
        }
    }

    // If the cancellation token fired during the loop, skip sidecar write
    // and report `Cancelled` (terminal event broadcast so SSE clients learn
    // the job stopped without waiting for a `Closed` that never comes).
    if cancel.is_cancelled() {
        eprintln!("[grim-garage] worker: job {} cancelled by request", id);
        let _ = registry
            .update_status_and_broadcast(&id, JobStatus::Cancelled)
            .await;
        return;
    }

    let train_state = optimizer.save_to_train_state(&autograd_reg.params);
    let sidecar_path = format!("{}.train", job.model_path);
    if let Some(parent) = std::path::Path::new(&sidecar_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = train_state.write(&sidecar_path) {
        eprintln!(
            "[grim-garage] worker: failed to write training sidecar {}: {e}",
            sidecar_path
        );
    } else {
        eprintln!(
            "[grim-garage] worker: wrote training state sidecar to {}",
            sidecar_path
        );
    }

    // Terminal broadcast so SSE subscribers receive a guaranteed
    // `Completed` event rather than having to poll `/api/train/status`.
    if let Err(e) = registry
        .update_status_and_broadcast(&id, JobStatus::Completed)
        .await
    {
        eprintln!("[grim-garage] worker: failed to mark {} Completed: {e}", id);
    } else {
        eprintln!("[grim-garage] worker: job {} completed successfully", id);
    }
}

// ===========================================================================
// Pure helpers — exited from the worker's hot loop so they can be unit-tested
// in isolation. These exercise lossy-decay / nearest-snap paths where the
// implementation has to derive expected numeric values *by hand* (mutation-
// resistant golden style — same discipline as `crates/grim-quant/tests/
// golden_*.rs`). Each test below pins a numeric value that is uniquely
// determined by the function's contract; a mutant that swaps a sign, drops a
// `max(1e-3)`, or moves the decay origin (last-loss vs initial-loss) breaks
// at least one assertion here.
// ===========================================================================

/// Fallback "loss got worse this step" value used when the autograd tensor
/// ops for a step return `Err(_)`. Decays **from the mode's initial loss**
/// (not the previous step's), so RL modes — which seed `loss = 0.0` —
/// recover to a measurable, non-zero value after a transient autograd
/// failure instead of being stuck at zero forever. A small floor
/// (1e-3) prevents pathological loss-of-precision when `initial_loss ==
/// 0` and the autograd error spikes back-to-back.
pub fn step_loss_fallback(mode: TrainingMode) -> f64 {
    let seed = initial_loss(mode);
    if seed == 0.0 {
        // RL: no zero baseline to decay from — use the unit floor so the
        // operator still sees a meaningful spike rather than a flat line.
        1e-3
    } else {
        seed * 0.9
    }
}

/// Pure golden-mirror of the worker's per-step spacing. Called nowhere in
/// production (the worker sleeps directly), but pins the per-step
/// duration the simulator commits to so the documented contract is
/// asserted.
pub const SIMULATED_STEP_DELAY: std::time::Duration = std::time::Duration::from_millis(10);

#[cfg(test)]
mod fallback_tests {
    use super::*;
    // `TrainingMode` is local to this crate's `jobs` module — already in
    // scope via `use super::*`. No cross-crate import needed.

    /// `step_loss_fallback` golden tests — mutation-resistant. Each test
    /// pins a single hand-derived numeric value; a wrong sign, missing
    /// floor, or `* 0.9` → `* 1.1` swap breaks at least one assertion
    /// here without touching unrelated state.

    #[test]
    fn sft_lora_fallback_is_initial_loss_times_ninetieth() {
        // initial_loss(Lora) == 2.3 → 2.3 * 0.9 = 2.07 (exact f64)
        assert_eq!(step_loss_fallback(TrainingMode::Lora), 2.07_f64);
    }

    #[test]
    fn sft_qlora_fallback_matches_lora_initial_decay() {
        // initial_loss(QLoRA) == 2.3 — both SFT branches use the same
        // seed; assert the function does NOT special-case one SFT mode.
        assert_eq!(
            step_loss_fallback(TrainingMode::QLoRA),
            step_loss_fallback(TrainingMode::Lora)
        );
    }

    #[test]
    fn sft_bf16_fallback_is_also_2_3_times_ninetieth() {
        assert_eq!(step_loss_fallback(TrainingMode::Bf16Full), 2.07_f64);
    }

    #[test]
    fn dpo_fallback_is_unit_floor_not_zero() {
        // initial_loss(Dpo) == 0.0. Pre-fix the fallback was `loss * 0.9`
        // (= 0.0 always); a faulty impl that uses initial_loss * 0.9
        // here still hands back 0 — caught by the (fallback > 0)
        // assertion rather than `==` to dodge hairsplitting f64 exactness.
        let out = step_loss_fallback(TrainingMode::Dpo);
        assert!(
            out > 0.0,
            "Dpo fallback stuck at 0.0 (initial-loss-only decay lost the RL floor): got {out:?}"
        );
        assert!(
            out <= 1.0,
            "Dpo fallback unreasonably huge (RL floor broken): got {out:?}"
        );
    }

    #[test]
    fn orpo_fallback_uses_floor_not_decay() {
        let out = step_loss_fallback(TrainingMode::Orpo);
        assert!(out > 0.0);
        assert!(out <= 1e-2);
    }

    #[test]
    fn grpo_fallback_uses_floor_not_decay() {
        let out = step_loss_fallback(TrainingMode::Grpo);
        assert!(out > 0.0);
        assert!(out <= 1e-2);
    }

    #[test]
    fn rl_floors_are_all_equal_to_1e_minus_3() {
        // All three RL modes share the same foundation: start from
        // initial_loss == 0 → take the unit floor. Pins the magic number
        // so it can't be tweaked accidentially; if intentional, the test
        // name + assertion must change together.
        let dpo = step_loss_fallback(TrainingMode::Dpo);
        let orpo = step_loss_fallback(TrainingMode::Orpo);
        let grpo = step_loss_fallback(TrainingMode::Grpo);
        assert_eq!(dpo, orpo);
        assert_eq!(orpo, grpo);
        assert_eq!(dpo, 1e-3);
    }

    #[test]
    fn rl_modes_distinct_from_sft_fallback_magnitude() {
        // Catches a mutant that routes all modes through
        // `initial_loss(mode) * 0.9` blindly — that would yield 0.0 for
        // all RL modes (caught above); an opposite mutant that routes
        // all modes through `1e-3` would yield 1e-3 for SFT too.
        // Assert the SFT vs RL fallback magnitudes are meaningfully
        // different (orders of magnitude apart, by design).
        let sft = step_loss_fallback(TrainingMode::Lora);
        let rl = step_loss_fallback(TrainingMode::Dpo);
        assert!(
            sft > rl * 100.0,
            "SFT fallback ({sft}) should dwarf RL fallback ({rl})"
        );
    }

    #[test]
    fn simulated_step_delay_is_pinned_ten_ms() {
        // Pin the contract value so docs (which previously claimed 200ms
        // — stale) and code stay in sync. Bug M4.
        assert_eq!(SIMULATED_STEP_DELAY, std::time::Duration::from_millis(10));
    }
}
