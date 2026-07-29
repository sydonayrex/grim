//! Grim's Garage — local-first training dashboard web application.
//!
//! Backend (`discovery`, `jobs`, `rocm`, `routes`) runs an axum HTTP server
//! on `0.0.0.0:8741` and serves `/api/*`, `/sse/metrics/:id`, and web UI.

pub mod backend;
pub mod discovery;
pub mod jobs;
pub mod rocm;
pub mod routes;
pub mod theme;
pub mod ui_state;
pub mod view_model;
pub mod dataloader;
pub mod weight_format;

/// Re-exports for downstream consumers and tests.
pub use discovery::{DatasetEntry, ModelEntry};
pub use jobs::{
    JobError, JobId, JobRegistry, JobStatus, Metric, MetricStreamEvent, TrainingJob, TrainingMode,
};
pub use weight_format::WeightFormat;
pub use rocm::{RocmDeviceInfo, probe_rocm_devices};
pub use ui_state::{
    DisplayState, GarageClient, JobSummaryDto, PollError, Poller, UiAppState, UiJob,
    UiTrainingConfig, merge_fetch, normalize_wire_status, poll_fetch, poll_once,
};
