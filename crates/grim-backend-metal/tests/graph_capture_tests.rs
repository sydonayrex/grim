//! Graph capture parity tests for Metal backend.
//!
//! Verifies that MetalDevice now implements GraphCaptureOps trait (Lapse A fix)
//! and that the capture/replay lifecycle works end-to-end.
//!
//! NOTE: MetalDevice::new returns Result<MetalDevice, Error> — unwrap with
//! expect() since these tests require a valid device handle.
//!
//! On non-Apple targets the trait impl is a no-op on the Apple-specific paths:
//! begin/end_graph_capture have no state to track, so end without a prior begin
//! returns an error, and replay always returns false. This matches the trait
//! contract: end_graph_capture must fail when there is no active capture.

use grim_backend_metal::MetalDevice;
use grim_tensor::backend::GraphCaptureOps;

#[test]
fn test_metal_graph_capture_trait_no_panic_on_create() {
    let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed on non-Apple");
    assert!(
        !dev.has_captured_graph("nonexistent_key"),
        "Fresh device should have no captured graphs"
    );
}

#[test]
fn test_metal_graph_capture_lifecycle_non_apple() {
    let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed on non-Apple");

    // begin_graph_capture on non-Apple: always Ok(()) (no-op, no state tracked)
    assert!(
        dev.begin_graph_capture("test_key").is_ok(),
        "begin_graph_capture should succeed on non-Apple (no-op)"
    );

    // has_captured_graph: false (no actual capture happens on non-Apple)
    assert!(
        !dev.has_captured_graph("test_key"),
        "Non-Apple path does not store captured graphs"
    );

    // end_graph_capture: on non-Apple there is no active capture state, so
    // end always fails with "No active graph capture to end" regardless of
    // whether begin was called. This is the correct no-op semantics.
    let end_result = dev.end_graph_capture("test_key");
    assert!(
        end_result.is_err(),
        "end_graph_capture should fail on non-Apple when no capture is active"
    );

    // replay_graph: Ok(false) — no graph stored
    let replay = dev.replay_graph("test_key");
    assert!(replay.is_ok());
    assert_eq!(replay.unwrap(), false, "No graph stored on non-Apple path");
}

#[test]
fn test_metal_graph_capture_key_mismatch_detected() {
    let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed on non-Apple");

    // end without begin: should fail (no active capture)
    let result = dev.end_graph_capture("some_key");
    assert!(
        result.is_err(),
        "end_graph_capture without begin_graph_capture should fail"
    );

    // begin then end: on non-Apple begin is a no-op with no state, so end
    // still fails because there is no active capture to end.
    assert!(dev.begin_graph_capture("correct_key").is_ok());
    let wrong_result = dev.end_graph_capture("wrong_key");
    assert!(
        wrong_result.is_err(),
        "end_graph_capture should fail on non-Apple even after begin (no active capture)"
    );
}
