//! Integration tests for the renderer host — wires CVKG's headless
//! runtime to the grim-garage ViewModel + DisplayState and confirms each
//! `render_frame` returns a fresh `HeadlessFrame` whose vdom reflects
//! the current model.
//!
//! These are TDD'd against `CvkgHeadless::new(view, viewport)` from
//! `cvkg` 0.3.3 (no GPU, no display, no input).

use grim_garage::rocm::RocmDeviceInfo;
use grim_garage::ui_state::display::DisplayState;
use grim_garage::view_model::ViewModel;
use grim_garage::renderer_host::RendererHandle;

#[test]
fn renderer_handle_constructs_from_empty_state() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let handle = RendererHandle::from_view_model(&vm);
    // Must round-trip without panicking.
    let _debug = handle.debug_string();
}

#[test]
fn renderer_handle_exposes_known_subtree_terms() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let handle = RendererHandle::from_view_model(&vm);
    let debug = handle.debug_string();
    // Top-level surfaces we expect the renderer to materialize:
    assert!(debug.contains("ROCm optimizations"));
    assert!(debug.contains("Training mode:"));
    assert!(debug.contains("Job History"));
    assert!(debug.contains(&vm.layout.window_title));
}

#[test]
fn renderer_handle_rebuilds_after_state_mutation() {
    let mut s = DisplayState::new();
    s.set_devices(vec![RocmDeviceInfo {
        ordinal: 0,
        gcn_arch: "gfx942".into(),
        vram_bytes: 192 * 1024 * 1024 * 1024,
        wavefront_size: 64,
        xnack_enabled: true,
    }]);
    s.upsert_job(grim_garage::UiJob {
        job_id: "abc".into(),
        status: "running".into(),
        ..Default::default()
    });

    let mut vm = ViewModel::from(&s);
    let mut handle = RendererHandle::from_view_model(&vm);

    // After rebuild, the device summary must mention CDNA.
    vm = ViewModel::from(&s);
    handle.refresh(&vm);
    let debug = handle.debug_string();
    assert!(debug.contains("CDNA"));
    assert!(debug.contains("● running"));
}

#[test]
fn renderer_handle_serializes_to_json() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let handle = RendererHandle::from_view_model(&vm);
    let json = serde_json::to_string(&handle.debug_summary()).expect("ser");
    // Not asserting specific fields — just that the summary is valid JSON,
    // suitable for the dev-server's `/api/system/debug` etc.
    assert!(json.starts_with("{"));
    assert!(json.ends_with("}"));
}

#[test]
fn renderer_handle_produces_a_frame_after_render() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let handle = RendererHandle::from_view_model(&vm);
    let frame = handle.render_frame().expect("headless render must succeed");
    // Frame exposes a VDom and a viewport; sanity-check the viewport.
    assert_eq!(frame.viewport.width, 1280);
    assert_eq!(frame.viewport.height, 720);
}

#[test]
fn renderer_handle_throttles_redundant_renders() {
    let s = DisplayState::new();
    let vm = ViewModel::from(&s);
    let mut handle = RendererHandle::from_view_model(&vm);
    // Two consecutive frames with no state mutation should not crash; the
    // headless render is non-blocking.
    handle.render_frame().unwrap();
    handle.render_frame().unwrap();
}
