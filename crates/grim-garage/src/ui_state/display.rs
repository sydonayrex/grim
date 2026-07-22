//! Reactive display state — CVKG's reactive primitives subscribe here.
//!
//! `DisplayState` is the read-side view-model that the CVKG runtime observes.
//! When something updates one of its fields, the framework re-renders the
//! dependent widgets without manual notification. Tests verify the in-place
//! mutators behave correctly.

use std::collections::HashMap;

use super::{UiAppState, UiJob, UiTrainingConfig};

/// State the CVKG UI reads from. Held behind a `Mutex` in the host runtime;
/// the display methods take `&mut` since reads-then-mutates must be atomic.
#[derive(Debug, Default)]
pub struct DisplayState {
    inner: UiAppState,
}

impl DisplayState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> UiAppState {
        self.inner.clone()
    }

    pub fn set_models(&mut self, models: Vec<crate::discovery::ModelEntry>) {
        self.inner.models = models;
    }

    pub fn set_datasets(&mut self, datasets: Vec<crate::discovery::DatasetEntry>) {
        self.inner.datasets = datasets;
    }

    pub fn set_devices(&mut self, devices: Vec<crate::rocm::RocmDeviceInfo>) {
        self.inner.devices = devices;
    }

    pub fn replace_config(&mut self, config: UiTrainingConfig) {
        self.inner.config = config;
    }

    pub fn upsert_job(&mut self, job: UiJob) {
        self.inner.jobs.insert(job.job_id.clone(), job);
    }

    pub fn select_model(&mut self, id: String) {
        self.inner.selected_model = Some(id);
    }

    pub fn select_dataset(&mut self, id: String) {
        self.inner.selected_dataset = Some(id);
    }

    pub fn push_metric(&mut self, job_id: &str, step: u64, loss: f64) {
        self.inner
            .live_metrics
            .entry(job_id.to_string())
            .or_default()
            .push((step, loss));
    }

    pub fn metric_series(&self, job_id: &str) -> Vec<(u64, f64)> {
        self.inner
            .live_metrics
            .get(job_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Snapshot of all model entries — read by the ModelSelector Picker.
    pub fn models(&self) -> &Vec<crate::discovery::ModelEntry> {
        &self.inner.models
    }

    /// Snapshot of all dataset entries — read by the DatasetPanel.
    pub fn datasets(&self) -> &Vec<crate::discovery::DatasetEntry> {
        &self.inner.datasets
    }

    /// Snapshot of ROCm devices — read by the ROCm toggles panel.
    pub fn rocm_devices(&self) -> &[crate::rocm::RocmDeviceInfo] {
        &self.inner.devices
    }

    /// Active training configuration — read by the hyperparameters form.
    pub fn config(&self) -> &UiTrainingConfig {
        &self.inner.config
    }

    /// All known jobs — read by the job history list.
    pub fn jobs(&self) -> HashMap<String, UiJob> {
        self.inner.jobs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{DatasetEntry, ModelEntry};
    use crate::rocm::RocmDeviceInfo;

    fn sample_state() -> DisplayState {
        let mut s = DisplayState::new();
        s.set_models(vec![ModelEntry {
            id: "tiny.gguf".into(),
            path: "/tmp/tiny.gguf".into(),
            format: "gguf".into(),
            is_grim: false,
        }]);
        s.set_datasets(vec![DatasetEntry {
            id: "train.jsonl".into(),
            path: "/tmp/train.jsonl".into(),
            format: "jsonl".into(),
            size_bytes: 1024,
        }]);
        s.set_devices(vec![RocmDeviceInfo {
            ordinal: 0,
            name: "AMD Radeon RX 7900 XTX".into(),
            gcn_arch: "gfx1100".into(),
            vram_bytes: 16 * 1024 * 1024 * 1024,
            wavefront_size: 32,
            wmma_supported: true,
            mfma_supported: false,
            xnack_enabled: false,
            compute_units: 84,
            max_threads_per_block: 1024,
        }]);
        s.upsert_job(UiJob {
            job_id: "abc".into(),
            status: "running".into(),
            model_path: "/tmp/tiny.gguf".into(),
            dataset_path: "/tmp/train.jsonl".into(),
            training_mode: "LoRA".into(),
        });
        s
    }

    #[test]
    fn display_state_round_trips_snapshot() {
        let s = sample_state();
        let snap = s.snapshot();
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.datasets.len(), 1);
        assert_eq!(snap.devices.len(), 1);
        assert_eq!(snap.jobs.get("abc").map(|j| j.status.clone()), Some("running".into()));
    }

    #[test]
    fn display_state_models_accessor_returns_inserted_entries() {
        let s = sample_state();
        assert_eq!(s.models().len(), 1);
        assert_eq!(s.models()[0].id, "tiny.gguf");
    }

    #[test]
    fn display_state_push_metric_appends_in_order() {
        let mut s = DisplayState::new();
        s.push_metric("j1", 0, 2.5);
        s.push_metric("j1", 1, 2.0);
        s.push_metric("j1", 2, 1.5);
        let series = s.metric_series("j1");
        assert_eq!(series, vec![(0, 2.5), (1, 2.0), (2, 1.5)]);
    }

    #[test]
    fn display_state_select_model_updates_selection() {
        let mut s = DisplayState::new();
        s.select_model("tiny.gguf".into());
        let snap = s.snapshot();
        assert_eq!(snap.selected_model.as_deref(), Some("tiny.gguf"));
    }

    #[test]
    fn display_state_upsert_job_replaces_existing() {
        let mut s = sample_state();
        s.upsert_job(UiJob {
            job_id: "abc".into(),
            status: "completed".into(),
            ..Default::default()
        });
        assert_eq!(s.jobs().get("abc").map(|j| j.status.clone()), Some("completed".into()));
    }

    #[test]
    fn display_state_jobs_returns_independent_clone() {
        let mut s = sample_state();
        let jobs1 = s.jobs();
        s.upsert_job(UiJob { job_id: "xyz".into(), ..Default::default() });
        let jobs2 = s.jobs();
        assert_eq!(jobs1.len(), 1);
        assert_eq!(jobs2.len(), 2);
    }
}
