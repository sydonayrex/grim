//! ViewModel — deterministic projection of `DisplayState` into the
//! strings/structs/IDs that the UI surface consumes.
//!
//! The CVKG widgets (and any future React/Vite/Tauri renderer) render
//! from this layer rather than from the raw `UiAppState`. This keeps
//! every piece of UI text testable and stable across renderer swaps.
//!
//! `ViewModel::from(&state)` is the single entry point. Render-impls
//! consume the resulting struct read-only via the [`panel`] submodules.

pub mod hyperparam;
pub mod job_card;
pub mod layout;
pub mod rocm_panel;
pub mod training_panel;

use crate::discovery::{DatasetEntry, ModelEntry};
use crate::rocm::RocmDeviceInfo;
use crate::ui_state::display::DisplayState;

pub use hyperparam::HyperparamFormV1;
pub use job_card::JobCardV1;
pub use layout::AppShellLayout;
pub use rocm_panel::RocmTogglesV1;
pub use training_panel::TrainingPanelV1;

/// The single ViewModel snapshot the dashboard renders from. Constructed
/// once per refresh round; cheap to clone.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ViewModel {
    pub models: Vec<ModelEntry>,
    pub datasets: Vec<DatasetEntry>,
    pub rocm_devices: Vec<RocmDeviceInfo>,
    pub jobs: Vec<JobCardV1>,
    pub training_config: HyperparamFormV1,
    pub rocm_toggles: RocmTogglesV1,
    pub training_panel: TrainingPanelV1,
    pub layout: AppShellLayout,
}

impl ViewModel {
    /// Build the ViewModel from the current UI state.
    pub fn from(state: &DisplayState) -> Self {
        let snap = state.snapshot();
        let config = HyperparamFormV1::from_training_config(snap.config.clone());
        let rocm_toggles = RocmTogglesV1::default_for_with_devices(
            state.rocm_devices(),
            config.rocm_fusion_rmsnorm_matmul,
            config.rocm_fusion_qkv_attention,
        );
        let training_panel = TrainingPanelV1::from_form(&config);
        let jobs = snap.jobs.values().cloned().map(JobCardV1::from).collect();
        Self {
            models: snap.models,
            datasets: snap.datasets,
            rocm_devices: snap.devices,
            jobs,
            training_config: config,
            rocm_toggles,
            training_panel,
            layout: AppShellLayout::default(),
        }
    }

    /// One-job-per-line history list. Empty when no jobs are known.
    pub fn history_cards(&self) -> &[JobCardV1] {
        &self.jobs
    }

    /// Model selector: zero-or-more entries.
    pub fn model_entries(&self) -> &[ModelEntry] {
        &self.models
    }
}
