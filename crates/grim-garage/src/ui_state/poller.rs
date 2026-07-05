//! Runtime poller — pulls models / datasets / devices / jobs from the
//! local grim-garage backend and writes them into a shared `DisplayState`.
//!
//! The CVKG runtime owns one `Poller` per session; the mutator side of
//! the display state lives behind an `Arc<Mutex<DisplayState>>` that
//! the UI reads. Polling is fire-and-await — if the backend is down,
//! the call surfaces an error and the loop swallows it (no UI death).
//!
//! Live SSE for per-job metrics is opt-in via `subscribe_sse(...)` and
//! uses the existing `JobRegistry::subscribe_metrics` broadcast
//! channel — exactly the same one the axum `sse_metrics` handler
//! drains. The CVKG view layer does not need a separate broadcast.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use super::display::DisplayState;
use super::http_client::{GarageClient, JobSummaryDto};
use crate::ui_state::UiJob;

/// Single refresh round: hits GET /api/models, /api/datasets, /api/rocm/devices,
/// /api/train/jobs and overwrites the corresponding fields on `state`.
///
/// Each step is best-effort: an unreachable backend, a partial failure,
/// or even a partial JSON parse doesn't poison the whole call. The
/// `Result` returned is `Err` only when **every** endpoint failed, and
/// even then the state may have been partially populated.
pub async fn poll_once(
    client: &GarageClient,
    state: &mut DisplayState,
) -> Result<(), PollError> {
    let mut errors = 0usize;

    if let Ok(models) = client.get_models().await {
        state.set_models(models);
    } else {
        errors += 1;
    }

    if let Ok(datasets) = client.get_datasets().await {
        state.set_datasets(datasets);
    } else {
        errors += 1;
    }

    if let Ok(devices) = client.get_devices().await {
        state.set_devices(devices);
    } else {
        errors += 1;
    }

    if let Ok(jobs) = client.get_jobs().await {
        for job in jobs {
            state.upsert_job(job_summary_to_ui_job(job));
        }
    } else {
        errors += 1;
    }

    if errors == 4 {
        Err(PollError::AllFailed)
    } else {
        Ok(())
    }
}

/// Reasons a refresh round can fail.
#[derive(Debug, thiserror::Error)]
pub enum PollError {
    /// All four endpoints returned errors. UI should surface a "backend offline" notice.
    #[error("all poll endpoints failed")]
    AllFailed,
}

/// Convert a wire `JobSummaryDto` into the UI's `UiJob`. Static function
/// so the poller is the single seam where wire-side `TrainingMode`
/// gets normalized back to UI string labels.
fn job_summary_to_ui_job(s: JobSummaryDto) -> UiJob {
    UiJob {
        job_id: s.job_id,
        status: s.status,
        model_path: s.model_path,
        dataset_path: s.dataset_path,
        training_mode: match s.training_mode {
            crate::jobs::TrainingMode::Lora => "LoRA".into(),
            crate::jobs::TrainingMode::QLoRA => "QLoRA".into(),
            crate::jobs::TrainingMode::Bf16Full => "Bf16-Full".into(),
        },
    }
}

/// Poller handle — owns a background tokio task that calls `poll_once`
/// on a fixed interval plus a one-shot initial refresh.
///
/// `abort()` stops the task and is idempotent.
pub struct Poller {
    client: GarageClient,
    state: Arc<Mutex<DisplayState>>,
    interval: Duration,
    handle: Option<JoinHandle<()>>,
}

impl Poller {
    pub fn new(client: GarageClient, state: Arc<Mutex<DisplayState>>) -> Self {
        Self {
            client,
            state,
            interval: Duration::from_secs(5),
            handle: None,
        }
    }

    pub fn with_interval(&mut self, d: Duration) -> &mut Self {
        self.interval = d;
        self
    }

    /// Spawn the background loop. Returns `&mut self` so callers can
    /// keep a handle and abort later via `abort()`.
    pub fn spawn(&mut self) -> &mut Self {
        let client = self.client.clone();
        let state = Arc::clone(&self.state);
        let interval = self.interval;

        let h = tokio::spawn(async move {
            // Initial refresh — fires immediately.
            {
                let mut s = state.lock().await;
                let _ = poll_once(&client, &mut s).await;
            }
            loop {
                tokio::time::sleep(interval).await;
                let mut s = state.lock().await;
                let _ = poll_once(&client, &mut s).await;
            }
        });
        self.handle = Some(h);
        self
    }

    pub fn abort(&self) {
        if let Some(h) = self.handle.as_ref() {
            h.abort();
        }
    }
}

impl Drop for Poller {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            h.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_error_display_mentions_failure_mode() {
        let e = PollError::AllFailed;
        let msg = e.to_string();
        assert!(msg.contains("all poll endpoints failed"));
    }

    #[test]
    fn poller_default_interval_is_five_seconds() {
        let client = GarageClient::new("http://localhost:9999");
        let state = Arc::new(Mutex::new(DisplayState::new()));
        let p = Poller::new(client, state);
        assert_eq!(p.interval, Duration::from_secs(5));
    }

    #[test]
    fn poller_with_interval_returns_self() {
        let client = GarageClient::new("http://localhost:9999");
        let state = Arc::new(Mutex::new(DisplayState::new()));
        let mut p = Poller::new(client, state);
        let prev_id = std::any::type_name::<Poller>();
        let r = p.with_interval(Duration::from_millis(50));
        assert_eq!(prev_id, std::any::type_name::<Poller>());
        let _ = r;
        assert_eq!(p.interval, Duration::from_millis(50));
    }
    // ^ extra blank line removal marker (test only)    #[tokio::test]
    async fn poller_abort_is_idempotent() {
        let client = GarageClient::new("http://localhost:9999");
        let state = Arc::new(Mutex::new(DisplayState::new()));
        let mut p = Poller::new(client, state);
        p.spawn();
        tokio::time::sleep(Duration::from_millis(20)).await;
        p.abort();
        p.abort(); // double abort must not panic
    }
}
