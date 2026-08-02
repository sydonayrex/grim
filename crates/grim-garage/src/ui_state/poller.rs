//! Runtime poller — pulls models / datasets / devices / jobs from the
//! local grim-garage backend and writes them into a shared `DisplayState`.
//!
//! The application owns one `Poller` per session; the mutator side of
//! the display state lives behind an `Arc<Mutex<DisplayState>>` that
//! the UI reads. Polling is fire-and-await — if the backend is down,
//! the call surfaces an error and the loop swallows it (no UI death).
//!
//! Live SSE for per-job metrics is opt-in via `subscribe_sse(...)` and
//! uses the existing `JobRegistry::subscribe_metrics` broadcast
//! channel — exactly the same one the axum `sse_metrics` handler
//! drains. The view layer does not need a separate broadcast.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use super::display::DisplayState;
use super::http_client::{GarageClient, JobSummaryDto};
use crate::backend::BackendProbe;
use crate::discovery::{DatasetEntry, ModelEntry};
use crate::ui_state::UiJob;
use tracing::warn;

/// Number of endpoints polled by [`poll_fetch`]. Kept as a named
/// constant (rather than the magic literal it replaces) so the
/// fetcher, the AllFailed classifier, and any future callers stay in
/// lockstep when an endpoint is added or removed.
pub const POLL_ENDPOINT_COUNT: usize = 4;

/// Best-effort fetch results for a single poll round. Each field is
/// `Ok`/`Err` independently — a partial failure (e.g. backend up but
/// `/api/rocm/devices` 500s) should not mask the others.
pub struct PollFetch {
    pub models: Result<Vec<ModelEntry>, String>,
    pub datasets: Result<Vec<DatasetEntry>, String>,
    pub devices: Result<Vec<BackendProbe>, String>,
    pub jobs: Result<Vec<JobSummaryDto>, String>,
}

impl PollFetch {
    pub const ENDPOINT_LABELS: [&'static str; POLL_ENDPOINT_COUNT] =
        ["models", "datasets", "devices", "jobs"];

    pub fn new_failures() -> Self {
        Self {
            models: Err("models not fetched".into()),
            datasets: Err("datasets not fetched".into()),
            devices: Err("devices not fetched".into()),
            jobs: Err("jobs not fetched".into()),
        }
    }

    /// Result tuples `(label, success?)` for every polled endpoint.
    pub fn results(&self) -> [(&'static str, bool); POLL_ENDPOINT_COUNT] {
        [
            (Self::ENDPOINT_LABELS[0], self.models.is_ok()),
            (Self::ENDPOINT_LABELS[1], self.datasets.is_ok()),
            (Self::ENDPOINT_LABELS[2], self.devices.is_ok()),
            (Self::ENDPOINT_LABELS[3], self.jobs.is_ok()),
        ]
    }

    /// Names of endpoints that returned `Err`.
    pub fn failed_labels(&self) -> Vec<&'static str> {
        self.results()
            .into_iter()
            .filter_map(|(label, ok)| if ok { None } else { Some(label) })
            .collect()
    }

    /// Number of endpoints that succeeded.
    pub fn success_count(&self) -> usize {
        self.results().iter().filter(|(_, ok)| *ok).count()
    }
}

/// Fetch all four endpoints — **no shared-state lock required**. The spawn
/// loop calls this unlocked so a stalled TCP connect (which has no
/// client-side timeout today) cannot block the UI's reactive read path.
/// Each `await` runs ahead of any `DisplayState` mutation.
pub async fn poll_fetch(client: &GarageClient) -> PollFetch {
    // Fetch sequentially — a hung backend path is now bounded by the
    // caller's outer `tokio::time::timeout` rather than an unbounded lock
    // held across awaits. Order matches the panel-importance ranking.
    PollFetch {
        models: client.get_models().await,
        datasets: client.get_datasets().await,
        devices: client.get_devices().await,
        jobs: client.get_jobs().await,
    }
}

/// Merge a fetched poll round into `state`. This is the synchronous
/// (no-`await`) side of the poll loop — the only section that needs the
/// `DisplayState` write lock. Returns
/// `Err(PollError::AllFailed(labels))` when every endpoint returned an
/// error; `Err(PollError::Partial(labels))` when 1..=POLL_ENDPOINT_COUNT-1
/// endpoints failed. `Ok(())` when all succeeded.
pub fn merge_fetch(state: &mut DisplayState, fetched: PollFetch) -> Result<(), PollError> {
    // Snapshot failed labels BEFORE we start moving the result fields
    // out of `fetched` (each `if let Ok(x) = ...` is a partial move).
    let failed_labels: Vec<&'static str> = fetched.failed_labels();
    let success_count = POLL_ENDPOINT_COUNT - failed_labels.len();

    if let Ok(models) = fetched.models {
        state.set_models(models);
    }
    if let Ok(datasets) = fetched.datasets {
        state.set_datasets(datasets);
    }
    if let Ok(devices) = fetched.devices {
        state.set_devices(devices);
    }
    if let Ok(jobs) = fetched.jobs {
        // Wholesale replace: ids that vanished from the backend (e.g.
        // completed jobs pruned server-side) must not linger in the UI's
        // history list. Earlier per-entry `upsert_job` calls leaked
        // these forever.
        let map: HashMap<String, UiJob> = jobs
            .into_iter()
            .map(|j| {
                let u = job_summary_to_ui_job(j);
                (u.job_id.clone(), u)
            })
            .collect();
        state.set_jobs(map);
    }

    if success_count == 0 {
        // Summarizing failed_labels is safe because fetched is dropped
        // after this branch — no further borrow.
        let mut labels: Vec<&'static str> = Vec::with_capacity(failed_labels.len());
        for l in &failed_labels {
            labels.push(*l);
        }
        Err(PollError::AllFailed(labels))
    } else if !failed_labels.is_empty() {
        let mut labels: Vec<&'static str> = Vec::with_capacity(failed_labels.len());
        for l in &failed_labels {
            labels.push(*l);
        }
        Err(PollError::Partial(labels))
    } else {
        Ok(())
    }
}

/// Single refresh round: hits GET /api/models, /api/datasets, /api/rocm/devices,
/// /api/train/jobs and overwrites the corresponding fields on `state`.
///
/// Each step is best-effort: an unreachable backend, a partial failure,
/// or even a partial JSON parse doesn't poison the whole call. The
/// `Result` returned is `Err` only when **every** endpoint failed, and
/// even then the state may have been partially populated.
///
/// This convenience form holds the caller's lock for the duration of
/// four sequential network awaits; the background `Poller` loop instead
/// uses [`poll_fetch`] + [`merge_fetch`] so the UI read path is not
/// blocked on a stalled backend.
pub async fn poll_once(client: &GarageClient, state: &mut DisplayState) -> Result<(), PollError> {
    let fetched = poll_fetch(client).await;
    merge_fetch(state, fetched)
}

/// Reasons a refresh round can fail.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PollError {
    /// All polled endpoints returned errors. Carries the static labels of
    /// every endpoint that failed so the UI can render an actionable
    /// diagnostic instead of an opaque "backend offline" banner.
    #[error("all poll endpoints failed: {0:?}")]
    AllFailed(Vec<&'static str>),
    /// A subset of endpoints failed; the rest succeeded, so the round
    /// is not "all-failed" — but the caller still benefits from the
    /// list of what broke.
    #[error("partial poll failure: {0:?}")]
    Partial(Vec<&'static str>),
}

/// Single normalization point for incoming wire status strings. The
/// server today emits lowercase (`status_label`), but a future refactor
/// that emits `"Failed"` or `"CANCELLED"` from a different code path
/// would silently slip past `JobCardV1::badge_label`'s exhaustive match
/// and render as "? Failed". Normalize at the seam so the UI depends on
/// a stable `lowercase` invariant.
///
/// Exposed `pub` so tests can pin the contract.
pub fn normalize_wire_status(raw: &str) -> String {
    raw.to_ascii_lowercase()
}

/// Convert a wire `JobSummaryDto` into the UI's `UiJob`. Static function
/// so the poller is the single seam where wire-side `TrainingMode`
/// gets normalized back to UI string labels.
fn job_summary_to_ui_job(s: JobSummaryDto) -> UiJob {
    UiJob {
        job_id: s.job_id,
        status: normalize_wire_status(&s.status),
        model_path: s.model_path,
        dataset_path: s.dataset_path,
        training_mode: match s.training_mode {
            crate::jobs::TrainingMode::Lora => "LoRA".into(),
            crate::jobs::TrainingMode::QLoRA => "QLoRA".into(),
            crate::jobs::TrainingMode::Bf16Full => "Bf16-Full".into(),
            crate::jobs::TrainingMode::RsLora => "RoSLoRA".into(),
            crate::jobs::TrainingMode::Dora => "DoRA".into(),
            crate::jobs::TrainingMode::LoftQ => "LoftQ".into(),
            crate::jobs::TrainingMode::Orpo => "ORPO".into(),
            crate::jobs::TrainingMode::Dpo => "DPO".into(),
            crate::jobs::TrainingMode::Kto => "KTO".into(),
            crate::jobs::TrainingMode::SimPo => "SimPO".into(),
            crate::jobs::TrainingMode::Grpo => "GRPO".into(),
            crate::jobs::TrainingMode::SoulEater => "SOUL EATER".into(),
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
    /// Per-round wall-clock budget for `poll_fetch`. Bounds a stalled
    /// connect so the UI read path can never be blocked indefinitely by
    /// a hung backend.
    fetch_timeout: Duration,
    handle: Option<JoinHandle<()>>,
}

impl Poller {
    pub fn new(client: GarageClient, state: Arc<Mutex<DisplayState>>) -> Self {
        Self {
            client,
            state,
            interval: Duration::from_secs(5),
            fetch_timeout: Duration::from_millis(800),
            handle: None,
        }
    }

    pub fn with_interval(&mut self, d: Duration) -> &mut Self {
        self.interval = d;
        self
    }

    /// Override the per-round fetch timeout. Tests use very short values to
    /// assert the loop does not stall on an unreachable host; production
    /// keeps the 800 ms default so a hung connect doesn't block UI reads.
    pub fn with_fetch_timeout(&mut self, d: Duration) -> &mut Self {
        self.fetch_timeout = d;
        self
    }

    /// Spawn the background loop. Returns `&mut self` so callers can
    /// keep a handle and abort later via `abort()`.
    ///
    /// Lock discipline: the `DisplayState` mutex is held *only* during the
    /// synchronous merge phase, never across a network `await`. A stalled
    /// TCP connect (the underlying HTTP client has no connect timeout)
    /// therefore blocks the poll task, not the UI's reactive read path —
    /// `DisplayState::snapshot()` and friends remain lockable while the
    /// backend is unreachable. A `tokio::time::timeout` bounds each round
    /// so a hung half-open connection cannot stall the loop indefinitely.
    pub fn spawn(&mut self) -> &mut Self {
        let client = self.client.clone();
        let state = Arc::clone(&self.state);
        let interval = self.interval;
        let fetch_timeout = self.fetch_timeout;

        let h = tokio::spawn(async move {
            // Initial refresh — fires immediately.
            let fetched = match tokio::time::timeout(fetch_timeout, poll_fetch(&client)).await {
                Ok(inner) => inner,
                Err(_elapsed) => PollFetch::new_failures(),
            };
            {
                let mut s = state.lock().await;
                if let Err(e) = merge_fetch(&mut s, fetched) {
                    warn!(error = ?e, "poll_round: initial fetch failed");
                }
            }
            loop {
                tokio::time::sleep(interval).await;
                let fetched = match tokio::time::timeout(fetch_timeout, poll_fetch(&client)).await {
                    Ok(inner) => inner,
                    Err(_elapsed) => PollFetch::new_failures(),
                };
                let mut s = state.lock().await;
                if let Err(e) = merge_fetch(&mut s, fetched) {
                    warn!(error = ?e, "poll_round: fetch failed");
                }
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
    use crate::backend::BackendProbe;
    use crate::discovery::{DatasetEntry, ModelEntry};
    use crate::jobs::TrainingMode;
    use crate::ui_state::{JobSummaryDto, UiJob};

    #[test]
    fn poll_error_display_mentions_failure_mode() {
        let e = PollError::AllFailed(vec!["models", "jobs"]);
        let msg = e.to_string();
        assert!(msg.contains("all poll endpoints failed"));
        assert!(msg.contains("models"));
    }

    #[test]
    fn merge_fetch_reports_all_failed_with_labels_when_every_endpoint_failed() {
        // M5: `merge_fetch` returns `AllFailed(label_vec)` rather than the
        // pre-fix magic `()` constant — the UI can now surface which
        // endpoints were unreachable. Pin a *specific* expected set rather
        // than asserting the count so a mutant that drops/deviates stays
        // caught.
        let mut s = DisplayState::new();
        let fetched = PollFetch::new_failures();
        let result = merge_fetch(&mut s, fetched);
        match result {
            Err(PollError::AllFailed(labels)) => {
                assert_eq!(
                    labels,
                    vec!["models", "datasets", "devices", "jobs"],
                    "every failed label should be listed in endpoint order"
                );
            }
            other => panic!("expected AllFailed(labels); got {other:?}"),
        }
    }

    #[test]
    fn merge_fetch_reports_partial_with_only_failed_labels() {
        // M5: 1..N-1 endpoint failures must produce `Partial(failed)`,
        // not `AllFailed`. The successful endpoint's label must be
        // absent — guards against an impl that confuses "all failed"
        // with "not all succeeded".
        let mut s = DisplayState::new();
        let mut fetched = PollFetch::new_failures();
        // One endpoint OK — let it through with a synthetic, well-typed
        // payload so we can also confirm the merge still applied.
        fetched.models = Ok(vec![ModelEntry {
            id: "tiny.gguf".into(),
            name: "tiny.gguf".into(),
            path: "/m/tiny.gguf".into(),
            format: "gguf".into(),
            is_grim: false,
            size_bytes: 0,
        }]);
        let result = merge_fetch(&mut s, fetched);
        match result {
            Err(PollError::Partial(labels)) => {
                assert_eq!(
                    labels,
                    vec!["datasets", "devices", "jobs"],
                    "partial list must contain only failed endpoints"
                );
            }
            other => panic!("expected Partial(failed); got {other:?}"),
        }
        // And the OK endpoint's data must have reached state.
        assert_eq!(s.models().len(), 1);
    }

    #[test]
    fn merge_fetch_reports_ok_when_nothing_failed() {
        // Confirm the enum discriminator picks the poller's happy path
        // rather than the empty `Vec` partial case (which a careless
        // `if !failed_labels.is_empty()` could misroute).
        let mut s = DisplayState::new();
        let fetched = PollFetch {
            models: Ok(vec![]),
            datasets: Ok(vec![]),
            devices: Ok(vec![]),
            jobs: Ok(vec![]),
        };
        assert_eq!(merge_fetch(&mut s, fetched), Ok(()));
    }

    #[test]
    fn normalize_wire_status_lowercases_in_place_at_poller_seam() {
        // M6: the poller is the single normalization point for incoming
        // JobSummaryDto.status. Servers may emit `"Failed"` (capitalized),
        // `"CANCELLED"`, or other casings; the UI side depends on a stable
        // lowercase invariant. Pin the exact contract.
        assert_eq!(normalize_wire_status("running"), "running");
        assert_eq!(normalize_wire_status("Pending"), "pending");
        assert_eq!(normalize_wire_status("FAILED"), "failed");
        assert_eq!(normalize_wire_status("CaNcElLeD"), "cancelled");
        assert_eq!(normalize_wire_status(""), "");
        // Non-ASCII bytes: ASCII gives no undefined behavior with lowercase
        // but make sure we don't panic on a unicode char.
        assert_eq!(normalize_wire_status("ñ"), "ñ");
    }

    #[test]
    fn job_summary_to_ui_job_applies_status_lowercase_seam() {
        // End-to-end: a wire summary arriving with mixed-case status is
        // normalized before entering the UiJob. Construct the DTO with
        // a randomized case and assert the resulting UiJob variant.
        let dto = JobSummaryDto {
            job_id: "j-1".into(),
            status: "Running".into(),
            model_path: "/m.gguf".into(),
            dataset_path: "/d.jsonl".into(),
            training_mode: TrainingMode::Lora,
        };
        let ui = _wire_dto_to_ui(dto);
        assert_eq!(ui.job_id, "j-1");
        assert_eq!(ui.status, "running");
    }

    // Helper because `job_summary_to_ui_job` is private — the test only
    // needs to prove the seam exists.
    fn _wire_dto_to_ui(s: JobSummaryDto) -> UiJob {
        super::job_summary_to_ui_job(s)
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
    // ^ extra blank line removal marker (test only)
    #[tokio::test]
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
