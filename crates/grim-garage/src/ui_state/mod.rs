//! UI state — the shared store the CVKG reactive system observes.
//!
//! Display state is split into [`display::DisplayState`] (read-by-view) and
//! service clients (HTTP fetcher). CVKG's reactive primitives watch
//! `display` and re-render when fields change.

pub mod display;
pub mod http_client;
pub mod poller;

pub use display::DisplayState;
pub use http_client::{GarageClient, JobSummaryDto};
pub use poller::{poll_once, PollError, Poller};

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::discovery::{DatasetEntry, ModelEntry};
use crate::rocm::RocmDeviceInfo;

/// Top-level UI state. Owns the data the CVKG views read.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UiAppState {
    pub models: Vec<ModelEntry>,
    pub datasets: Vec<DatasetEntry>,
    pub devices: Vec<RocmDeviceInfo>,
    pub jobs: HashMap<String, UiJob>,
    pub config: UiTrainingConfig,
    pub selected_model: Option<String>,
    pub selected_dataset: Option<String>,
    /// Live metrics per job id: `(step, loss)`.
    pub live_metrics: HashMap<String, Vec<(u64, f64)>>,
}

/// One job as the UI sees it.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UiJob {
    pub job_id: String,
    pub status: String,
    pub model_path: String,
    pub dataset_path: String,
    pub training_mode: String,
}

/// Hyperparameters the UI form binds to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiTrainingConfig {
    pub model: Option<String>,
    pub dataset: Option<String>,
    pub training_mode: String,
    pub quant_format: String,
    pub lora_rank: u32,
    pub learning_rate: f64,
    pub epochs: u32,
    pub rocm_fusion_rmsnorm_matmul: bool,
    pub rocm_fusion_qkv_attention: bool,
    pub auto_wavefront: bool,
    pub xnack_enabled: bool,
}

impl Default for UiTrainingConfig {
    fn default() -> Self {
        Self {
            model: None,
            dataset: None,
            training_mode: "LoRA".into(),
            quant_format: "Q4_K".into(),
            lora_rank: 16,
            learning_rate: 2e-5,
            epochs: 1,
            rocm_fusion_rmsnorm_matmul: true,
            rocm_fusion_qkv_attention: false,
            auto_wavefront: true,
            xnack_enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_training_config_default_round_trips() {
        let cfg = UiTrainingConfig::default();
        assert_eq!(cfg.lora_rank, 16);
        assert!(cfg.rocm_fusion_rmsnorm_matmul);
        assert!(cfg.auto_wavefront);
        assert_eq!(cfg.quant_format, "Q4_K");
    }

    #[test]
    fn ui_job_default_is_empty_string_fields() {
        let job = UiJob::default();
        assert_eq!(job.job_id, "");
        assert_eq!(job.status, "");
    }

    #[test]
    fn ui_app_state_default_has_no_jobs() {
        let app = UiAppState::default();
        assert!(app.models.is_empty());
        assert!(app.datasets.is_empty());
        assert!(app.jobs.is_empty());
    }
}
