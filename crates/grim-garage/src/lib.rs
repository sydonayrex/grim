//! Grim's Garage — local-first training dashboard web application.
//!
//! Backend (`discovery`, `jobs`, `rocm`, `routes`) runs an axum HTTP server
//! on `0.0.0.0:8741` and serves `/api/*`, `/sse/metrics/:id`, and web UI.

pub mod discovery;
pub mod jobs;
pub mod rocm;
pub mod routes;
pub mod theme;
pub mod ui_state;
pub mod view_model;

/// Re-exports for downstream consumers and tests.
pub use discovery::{DatasetEntry, ModelEntry};
pub use jobs::{
    JobError, JobId, JobRegistry, JobStatus, Metric, MetricStreamEvent, TrainingJob, TrainingMode,
};
pub use rocm::{probe_rocm_devices, RocmDeviceInfo};
pub use ui_state::{DisplayState, GarageClient, JobSummaryDto, PollError, Poller, UiAppState, UiJob, UiTrainingConfig, poll_once};
