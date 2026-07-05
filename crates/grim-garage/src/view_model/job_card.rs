//! Job history list — one card per training job.

use crate::ui_state::UiJob;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobCardV1 {
    pub job_id: String,
    pub status: String,
    pub model_path: String,
    pub dataset_path: String,
    pub training_mode: String,
}

impl JobCardV1 {
    /// Map the wire-side `UiJob` into the viewmodel-friendly card.
    pub fn from(job: UiJob) -> Self {
        Self {
            job_id: job.job_id,
            status: job.status,
            model_path: job.model_path,
            dataset_path: job.dataset_path,
            training_mode: job.training_mode,
        }
    }

    /// Short label that pairs the status with a tok-flavoured bullet,
    /// e.g. `"● running"`, `"✓ completed"`, `"✗ failed"`. The CVKG `Badge`
    /// widget uses this verbatim inside `MerkiBadge`.
    pub fn badge_label(&self) -> String {
        let bullet = match self.status.as_str() {
            "running" => "●",
            "completed" => "✓",
            "failed" => "✗",
            "pending" => "○",
            other => return format!("? {other}"),
        };
        format!("{bullet} {}", self.status)
    }

    /// Short model/dataset summary trimmed for the card subtitle.
    pub fn subtitle(&self) -> String {
        let model = self
            .model_path
            .rsplit_once('/')
            .map(|(_, leaf)| leaf)
            .unwrap_or(&self.model_path);
        let dataset = self
            .dataset_path
            .rsplit_once('/')
            .map(|(_, leaf)| leaf)
            .unwrap_or(&self.dataset_path);
        format!("{model} on {dataset}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn badge_uses_running_bullet() {
        let card = JobCardV1 {
            job_id: "abc".into(),
            status: "running".into(),
            model_path: "/m.gguf".into(),
            dataset_path: "/d.jsonl".into(),
            training_mode: "LoRA".into(),
        };
        assert_eq!(card.badge_label(), "● running");
    }

    #[test]
    fn badge_uses_completed_check() {
        let card = JobCardV1 {
            job_id: "x".into(),
            status: "completed".into(),
            model_path: "/m".into(),
            dataset_path: "/d".into(),
            training_mode: "QLoRA".into(),
        };
        assert_eq!(card.badge_label(), "✓ completed");
    }

    #[test]
    fn badge_uses_failed_x() {
        let card = JobCardV1 {
            job_id: "y".into(),
            status: "failed".into(),
            model_path: "/m".into(),
            dataset_path: "/d".into(),
            training_mode: "Bf16-Full".into(),
        };
        assert_eq!(card.badge_label(), "✗ failed");
    }

    #[test]
    fn subtitle_strips_path_components() {
        let card = JobCardV1 {
            job_id: "z".into(),
            status: "running".into(),
            model_path: "/home/user/models/tiny.gguf".into(),
            dataset_path: "/var/data/train.jsonl".into(),
            training_mode: "LoRA".into(),
        };
        assert_eq!(card.subtitle(), "tiny.gguf on train.jsonl");
    }

    #[test]
    fn unknown_status_falls_back_to_question_mark() {
        let card = JobCardV1 {
            job_id: "z".into(),
            status: "frobnicate".into(),
            model_path: "/m".into(),
            dataset_path: "/d".into(),
            training_mode: "LoRA".into(),
        };
        assert_eq!(card.badge_label(), "? frobnicate");
    }
}
