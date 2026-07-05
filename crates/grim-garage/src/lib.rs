//! Grim's Garage — local-first training dashboard.
//!
//! Backend (`discovery`, `jobs`, `rocm`, `routes`) runs an axum HTTP server
//! on `0.0.0.0:8741` and serves `/api/*` + `/sse/metrics/:id`.
//!
//! Frontend (`ui_state`, `ui`) is built on CVKG 0.3.1's native Rust UI
//! framework via `cvkg-cli::native_shell::create_window` with headless /
//! Tauri / Wry backends. React/Vite/Tailwind are deliberately NOT used —
//! CVKG provides 180+ components, OKLCH theme tokens, and reactive
//! primitives that cover the dashboard surface directly in Rust.

pub mod discovery;
pub mod jobs;
pub mod renderer_host;
pub mod rocm;
pub mod routes;
pub mod theme;
pub mod ui;
pub mod ui_state;
pub mod view_model;

/// Re-exports for downstream consumers and tests.
pub use discovery::{DatasetEntry, ModelEntry};
pub use jobs::{
    JobError, JobId, JobRegistry, JobStatus, Metric, MetricStreamEvent, TrainingJob, TrainingMode,
};
pub use rocm::{probe_rocm_devices, RocmDeviceInfo};
pub use ui_state::{DisplayState, GarageClient, JobSummaryDto, PollError, Poller, UiAppState, UiJob, UiTrainingConfig, poll_once};
