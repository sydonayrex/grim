//! Integration tests for the CVKG View compositors that assemble
//! the dashboard panels from the ViewModel + DisplayState.
//!
//! These tests prove the consumer-side wiring without invoking a renderer:
//! - The compositor builds a `View` tree (typed VStack/...) from the
//!   ViewModel and DisplayState.
//! - The returned tree is structurally consistent (correct column count,
//!   correct panel presence, no empty panels).
//!
//! That gives us a regression net for the renderer integration once a
//! real wgpu/winit host is wired up — any panics will surface here.

use grim_garage::rocm::RocmDeviceInfo;
use grim_garage::ui_state::display::DisplayState;
use grim_garage::view_model::ViewModel;
use grim_garage::ui::dashboard::build_dashboard;
use grim_garage::ui::panels::{
    build_header, build_job_history_panel, build_rocm_panel, build_training_panel,
};
use grim_garage::ui::view_kind::ViewKind;

#[test]
fn dashboard_renderer_round_trip_returns_vstack() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let view = build_dashboard(&vm);
    assert_eq!(view.kind(), ViewKind::VStack);
    assert_eq!(view.children().len(), 2, "header + main row");
}

#[test]
fn header_emits_title_and_status_subtitle() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let header = build_header(&vm);
    assert_eq!(header.kind(), ViewKind::HStack);
    assert!(!header.children().is_empty());
    // Title text appears somewhere in the header.
    assert!(
        header
            .debug_string()
            .contains(&vm.layout.window_title),
        "header should mention the window title"
    );
}

#[test]
fn training_panel_routes_attention_to_quantization_picker() {
    let mut s = DisplayState::new();
    let mut config = s.config().clone();
    config.training_mode = "QLoRA".into();
    s.replace_config(config);
    let vm = ViewModel::from(&s);
    let panel = build_training_panel(&vm);
    assert_eq!(panel.kind(), ViewKind::Card);
    // Picker only mounts when the panel says so.
    assert!(
        vm.training_panel.show_quant_format_picker,
        "QLoRA must request the quant format picker"
    );
    assert_eq!(
        vm.training_panel.quant_format_options.len(),
        3,
        "QLoRA exposes 3 quant format options"
    );
    assert!(
        panel
            .debug_string()
            .contains(&vm.training_panel.panel_title)
    );
}

#[test]
fn rocm_panel_with_no_devices_disables_toggles() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let panel = build_rocm_panel(&vm);
    assert_eq!(panel.kind(), ViewKind::Card);
    // No devices -> every toggle child is disabled.
    for toggle in &vm.rocm_toggles.toggles {
        assert!(!toggle.enabled);
        assert!(panel.debug_string().contains(&toggle.label));
    }
}

#[test]
fn rocm_panel_with_a_cdna_device_enables_all_toggles() {
    let mut s = DisplayState::new();
    s.set_devices(vec![RocmDeviceInfo {
        ordinal: 0,
        gcn_arch: "gfx942".into(),
        vram_bytes: 192 * 1024 * 1024 * 1024,
        wavefront_size: 64,
        xnack_enabled: true,
    }]);
    let vm = ViewModel::from(&s);
    assert!(vm.rocm_toggles.toggles.iter().all(|t| t.enabled));
    let panel = build_rocm_panel(&vm);
    assert!(panel.debug_string().contains("CDNA"));
}

#[test]
fn job_history_panel_renders_a_card_per_job() {
    let mut s = DisplayState::new();
    // 3 fake-card jobs: pending, running, completed.
    let mut jobs = s.jobs();
    jobs.insert(
        "a".into(),
        grim_garage::UiJob {
            job_id: "a".into(),
            status: "pending".into(),
            ..Default::default()
        },
    );
    jobs.insert(
        "b".into(),
        grim_garage::UiJob {
            job_id: "b".into(),
            status: "running".into(),
            ..Default::default()
        },
    );
    jobs.insert(
        "c".into(),
        grim_garage::UiJob {
            job_id: "c".into(),
            status: "completed".into(),
            ..Default::default()
        },
    );
    // Replace the snapshot's jobs slot.
    s.upsert_job(jobs.remove("a").unwrap());
    s.upsert_job(jobs.remove("b").unwrap());
    s.upsert_job(jobs.remove("c").unwrap());
    let vm = ViewModel::from(&s);
    assert_eq!(vm.jobs.len(), 3);
    let panel = build_job_history_panel(&vm);
    assert_eq!(panel.kind(), ViewKind::Card);
    // Each card produced by JobCardV1 shows its badge label somewhere.
    assert!(panel.debug_string().contains("○ pending"));
    assert!(panel.debug_string().contains("● running"));
    assert!(panel.debug_string().contains("✓ completed"));
}

#[test]
fn dashboard_includes_every_required_panel() {
    let mut s = DisplayState::new();
    s.upsert_job(grim_garage::UiJob {
        job_id: "x".into(),
        status: "running".into(),
        ..Default::default()
    });
    let vm = ViewModel::from(&s);
    let view = build_dashboard(&vm);
    let debug = view.debug_string();
    // Sanity: every major surface is mounted.
    assert!(debug.contains("ROCm"), "ROCm panel present");
    assert!(debug.contains("Training mode"), "Training mode panel present");
    assert!(debug.contains("Job History"), "Job history panel present");
    assert!(debug.contains(&vm.training_panel.panel_title));
    assert!(
        debug.contains(&vm.rocm_toggles.device_summary),
        "device summary mounted"
    );
}
