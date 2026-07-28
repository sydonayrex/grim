//! Integration tests for the ViewModel bridge — the deterministic
//! projection of `DisplayState` into strings/structs that the CVKG
//! widgets (and any future React/Vite/Tauri renderer) consume.
//!
//! The goal is to land `GrimViewModel::from(state)` and the
//! supporting fields without depending on CVKG widget constructor
//! specifics. When CVKG 0.3.3 surfaces a stable widget API, we add
//! the View-impl tree in a separate commit that consumes this.

use grim_garage::ui_state::display::DisplayState;
use grim_garage::view_model::ViewModel;
use grim_garage::view_model::hyperparam::HyperparamFormV1;
use grim_garage::view_model::job_card::JobCardV1;
use grim_garage::view_model::layout::AppShellLayout;
use grim_garage::view_model::rocm_panel::RocmTogglesV1;
use grim_garage::view_model::training_panel::TrainingPanelV1;

#[test]
fn viewmodel_default_state_renders_empty_lists_and_zero_devices() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    assert!(vm.models.is_empty());
    assert!(vm.datasets.is_empty());
    assert!(vm.rocm_devices.is_empty());
    assert!(vm.jobs.is_empty());
    assert_eq!(vm.training_config.lora_rank, 16);
    assert_eq!(vm.training_config.training_mode, "LoRA");
    assert!(vm.training_config.rocm_fusion_rmsnorm_matmul);
}

#[test]
fn viewmodel_pulls_through_to_hyperparam_form() {
    let mut s = DisplayState::new();
    s.set_models(vec![model("tiny.gguf"), model("big.grim")]);
    s.set_datasets(vec![dataset("train.jsonl")]);
    let vm = ViewModel::from(&s);
    assert_eq!(vm.models.len(), 2);
    assert_eq!(vm.datasets.len(), 1);
    assert!(vm.models.iter().any(|m| m.is_grim));
}

#[test]
fn hyperparam_form_serializes_to_camel_case_keys() {
    let form = HyperparamFormV1::default();
    let json = serde_json::to_value(&form).expect("serde");
    // UI gets `learningRate` not `learning_rate` — matches CSS/JS conventions.
    assert!(json.get("learningRate").is_some());
    assert!(json.get("loraRank").is_some());
    assert!(json.get("trainingMode").is_some());
}

#[test]
fn hyperparam_form_clamps_lora_rank_to_set() {
    let mut form = HyperparamFormV1::default();
    form.lora_rank = 99; // not in {8,16,32,64}
    let cleaned = form.normalized();
    assert!(
        cleaned.lora_rank == 8
            || cleaned.lora_rank == 16
            || cleaned.lora_rank == 32
            || cleaned.lora_rank == 64
    );
}

#[test]
fn training_panel_routes_attention_to_quantization_floor() {
    let mut form = HyperparamFormV1::default();
    form.training_mode = "QLoRA".into();
    let panel = TrainingPanelV1::from_form(&form);
    assert!(panel.show_quant_format_picker);
    assert_eq!(panel.quant_format_options.len(), 3);
}

#[test]
fn rocm_toggles_panel_lists_all_four_options() {
    let panel = RocmTogglesV1::default_for(&DisplayState::new());
    assert_eq!(panel.toggles.len(), 4);
    assert!(panel.toggles.iter().any(|t| t.id == "rmsnorm_matmul"));
    assert!(panel.toggles.iter().any(|t| t.id == "qkv_attention"));
    assert!(panel.toggles.iter().any(|t| t.id == "auto_wavefront"));
    assert!(panel.toggles.iter().any(|t| t.id == "xnack"));
}

#[test]
fn rocm_toggles_reflect_state_when_devices_present() {
    let mut s = DisplayState::new();
    s.set_devices(vec![grim_garage::RocmDeviceInfo {
        ordinal: 0,
        name: "AMD Instinct MI300X".into(),
        vendor: "AMD".into(),
        backend: "ROCm".into(),
        is_rocm_compliant: true,
        gcn_arch: "gfx942".into(), // MI300X — CDNA3, W64
        vram_bytes: 192 * 1024 * 1024 * 1024,
        vram_used_bytes: 0,
        wavefront_size: 64,
        wmma_supported: true,
        mfma_supported: true,
        xnack_enabled: true,
        compute_units: 304,
        max_threads_per_block: 1024,
    }]);
    let panel = RocmTogglesV1::default_for(&s);
    // MI300X should auto-enable waves-per-eu since RDNA-style is irrelevant.
    assert!(panel.device_summary.contains("CDNA") || panel.device_summary.contains("W64"));

    let _ = grim_garage::ModelEntry::new("a.gguf", "/p", "gguf", false);
}

#[test]
fn job_card_shows_status_emoji_for_running_state() {
    let card = JobCardV1 {
        job_id: "abc".into(),
        status: "running".into(),
        model_path: "/p".into(),
        dataset_path: "/d".into(),
        training_mode: "LoRA".into(),
    };
    let label = card.badge_label();
    assert!(label.contains("running"));
}

#[test]
fn job_card_renders_stop_glyph_for_cancelled_status() {
    // Defensive: the UI side must not silently fall through the wildcard
    // `? cancelled` arm when the wire emits "cancelled". The badge should
    // carry a recognizable stop glyph (U+23F9) instead of a question mark.
    let card = JobCardV1 {
        job_id: "abc".into(),
        status: "cancelled".into(),
        model_path: "/p".into(),
        dataset_path: "/d".into(),
        training_mode: "QLoRA".into(),
    };
    let label = card.badge_label();
    assert!(
        label.contains('\u{23F9}'),
        "cancelled badge should embed U+23F9; got {label:?}"
    );
    assert!(label.contains("cancelled"));
    assert!(
        !label.starts_with('?'),
        "must not hit the unknown-status wildcard; got {label:?}"
    );
}

#[test]
fn app_shell_layout_groups_panels_into_three_columns() {
    let layout = AppShellLayout::default();
    assert!(layout.header_height > 0);
    assert_eq!(layout.columns.len(), 3);
}

#[test]
fn viewmodel_round_trips_through_serde() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let json = serde_json::to_string(&vm).expect("ser");
    let back: ViewModel = serde_json::from_str(&json).expect("de");
    assert_eq!(back.models.len(), vm.models.len());
    assert_eq!(back.training_config.lora_rank, vm.training_config.lora_rank);
}

#[test]
fn display_state_set_jobs_prunes_stale_jobs_and_metrics() {
    // H5: the poller must replace the whole job map rather than merging
    // per-id; otherwise completed/failed jobs the backend pruned (e.g.
    // memory cap) linger in the UI's history forever. `set_jobs` also
    // prunes `live_metrics` for removed ids to keep memory bounded.
    use grim_garage::UiJob;
    use std::collections::HashMap;

    let mut s = DisplayState::new();

    fn job(id: &str, status: &str) -> UiJob {
        UiJob {
            job_id: id.into(),
            status: status.into(),
            model_path: format!("/models/{id}.gguf"),
            dataset_path: "/d.jsonl".into(),
            training_mode: "LoRA".into(),
        }
    }

    // Seed two jobs the long way (mimics old per-id upserts).
    s.upsert_job(job("keep", "running"));
    s.upsert_job(job("gone", "completed"));
    s.push_metric("keep", 0, 2.3);
    s.push_metric("gone", 0, 4.1);
    s.push_metric("gone", 1, 3.9);
    assert_eq!(s.jobs().len(), 2);
    assert_eq!(s.metric_series("gone").len(), 2);

    // Replace the job set with only "keep". "gone" must vanish from jobs
    // AND its metric series must be pruned (else DisplayState leaks).
    let mut next: HashMap<String, UiJob> = HashMap::new();
    next.insert("keep".to_string(), job("keep", "completed"));
    s.set_jobs(next);

    let jobs = s.jobs();
    assert_eq!(jobs.len(), 1, "stale job leaked through set_jobs: {jobs:?}");
    assert!(jobs.contains_key("keep"));
    assert!(
        !jobs.contains_key("gone"),
        "set_jobs did not prune absent 'gone' job"
    );
    assert_eq!(
        s.metric_series("gone").len(),
        0,
        "live_metrics for pruned job was not garbage-collected"
    );
    assert_eq!(
        s.metric_series("keep").len(),
        1,
        "metrics for retained job should survive set_jobs"
    );
}

// ----- helpers -----

fn model(id: &str) -> grim_garage::ModelEntry {
    grim_garage::ModelEntry::new(
        id,
        &format!("/tmp/{id}"),
        if id.ends_with(".grim") {
            "grim"
        } else {
            "gguf"
        },
        id.ends_with(".grim"),
    )
}

fn dataset(id: &str) -> grim_garage::DatasetEntry {
    grim_garage::DatasetEntry {
        id: id.into(),
        path: format!("/tmp/{id}"),
        format: "jsonl".into(),
        size_bytes: 1024,
    }
}
