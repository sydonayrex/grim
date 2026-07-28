//! Integration test for the backend selection chain.
//!
//! Verifies that the full ROCm→CUDA→Vulkan→Metal→CPU priority chain is
//! wired and actually selects a *live* device on this host (the autograd
//! tape then dispatches to it via `pick_device_for_tensor`). This is the
//! "wire the full selection chain" acceptance test — it runs a real
//! `select_backend` and asserts the chosen device tag matches what the
//! hardware probe reported.

use grim_backend_cpu::cpu_tensor;
use grim_garage::backend::{PreferredBackend, probe_all, select_backend};
use grim_tensor::{Device, Tensor};

fn dev_of(t: &Tensor) -> Device {
    t.device().clone()
}

#[test]
fn select_chain_falls_through_to_live_device() {
    let b = select_backend(None);
    let probes = probe_all();
    // The chosen backend must be one the probe reported as available.
    assert!(
        probes.iter().any(|p| p.name == b.label && p.available),
        "selected backend '{}' was not reported live by probe_all",
        b.label
    );
    // At minimum CPU is always live.
    assert!(probes.iter().any(|p| p.name == "cpu" && p.available));
}

#[test]
fn make_tensor_lands_on_selected_device() {
    let b = select_backend(None);
    let t = b
        .make_tensor(
            vec![0.1f32, 0.2f32, 0.3f32, 0.4f32],
            grim_tensor::Shape::new(vec![2, 2]),
        )
        .expect("make_tensor");
    let d = dev_of(&t);
    match b.label.as_str() {
        "rocm" => assert!(
            matches!(d, Device::Rocm(_)),
            "expected Rocm device, got {d:?}"
        ),
        "cuda" => assert!(
            matches!(d, Device::Cuda(_)),
            "expected Cuda device, got {d:?}"
        ),
        "vulkan" => assert!(
            matches!(d, Device::Vulkan),
            "expected Vulkan device, got {d:?}"
        ),
        "metal" => assert!(
            matches!(d, Device::Metal(_)),
            "expected Metal device, got {d:?}"
        ),
        "cpu" => assert!(matches!(d, Device::Cpu), "expected Cpu device, got {d:?}"),
        other => panic!("unexpected backend label {other}"),
    }
}

#[test]
fn preferred_backend_is_respected_when_live() {
    // Force ROCm if it is live; otherwise this asserts the selection honors
    // the explicit preference and only falls back when unavailable.
    let want = PreferredBackend::Rocm;
    let b = select_backend(Some(want));
    if probe_all().iter().any(|p| p.name == "rocm" && p.available) {
        assert_eq!(b.label, "rocm");
    } else {
        // ROCm not live: must still be a live backend, never a dead one.
        assert!(
            probe_all().iter().any(|p| p.name == b.label && p.available),
            "preferred ROCm unavailable but fell to '{}' which is also dead",
            b.label
        );
    }
}

#[test]
fn cpu_always_available() {
    let _ = cpu_tensor(vec![1.0f32], grim_tensor::Shape::new(vec![1]));
    let b = select_backend(Some(PreferredBackend::Cpu));
    assert_eq!(b.label, "cpu");
}
