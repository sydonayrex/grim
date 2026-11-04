//! Audit §4 regression/env — DeviceGuard drift guard.
//!
//! The production launch seams pin `DeviceGuard::set(device_ordinal)` so a
//! thread that has drifted onto a foreign HIP context still runs kernels
//! against the owning device. `context_drift_tests.rs` covers the
//! `last_launch_context()` invariant with a *separate* foreign-context
//! thread; this test covers the numeric twin: the *same* closure executed
//! (a) correctly pinned, (b) explicitly pinned to the WRONG device ordinal,
//! and (c) pinned while a foreign context is parked on the other device,
//! on a tensor pair that indeed lives on one device (cross-residency would
//! make the launch read peer memory, not drift — peer access is tested by
//! `scythe_multi_gpu.rs` / `p2p_route.rs`, not here).
//!
//! Gated: `GRIM_GPU_TEST=1` + two ROCm devices. With <2 devices the test
//! prints `[SKIP] requires 2 devices` and returns (mirroring
//! `multigpu_context_coherence.rs::two_gpu_gate`).

use std::sync::Arc;

use grim_backend_rocm::{DeviceGuard, RocmDevice, as_rocm, enumerate_devices};
use grim_tensor::{CoreTensorOps, DType, Shape};

fn two_gpu_gate() -> bool {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] requires 2 devices (GRIM_GPU_TEST=1 off)");
        return false;
    }
    match enumerate_devices() {
        Ok(n) if n >= 2 => true,
        Ok(n) => {
            eprintln!("[SKIP] requires 2 devices (found {n})");
            false
        }
        Err(e) => {
            eprintln!("[SKIP] requires 2 devices (enumeration failed: {e})");
            false
        }
    }
}

/// One fused add+rms_norm over a [1, 1024] row; returns (sum, norm) host vecs.
/// The weight is unit so any wrong-context execution changes only rounding,
/// not magnitude — bit comparison is the signal.
fn fused_once(dev: &RocmDevice, x: &[f32], r: &[f32], w: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let rows = 1usize;
    let row_len = 1024usize;
    let shape = Shape::new(vec![rows, row_len]);
    let x_s = CoreTensorOps::from_cpu(dev, x, &shape, DType::F32).unwrap();
    let r_s = CoreTensorOps::from_cpu(dev, r, &shape, DType::F32).unwrap();
    let w_s = CoreTensorOps::from_cpu(dev, w, &Shape::new(vec![row_len]), DType::F32).unwrap();
    let sum = CoreTensorOps::from_cpu(dev, &vec![0.0f32; rows * row_len], &shape, DType::F32).unwrap();
    let norm = CoreTensorOps::from_cpu(dev, &vec![0.0f32; rows * row_len], &shape, DType::F32).unwrap();
    {
        let _guard = DeviceGuard::set(dev.ordinal() as i32);
        dev.fused_add_rms_norm_into(
            x_s.as_ref(),
            r_s.as_ref(),
            w_s.as_ref(),
            1e-5,
            as_rocm(sum.as_ref()).unwrap(),
            as_rocm(norm.as_ref()).unwrap(),
            &shape,
        )
        .unwrap();
    }
    dev.synchronize();
    (
        sum.to_cpu_vec_f32().unwrap(),
        norm.to_cpu_vec_f32().unwrap(),
    )
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1 on a 2-GPU box"]
fn rms_norm_bit_equal_under_wrong_device_pin_and_foreign_context() {
    if !two_gpu_gate() {
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // Device acquisition per the audit: require_gpu_device() -> try_new(0),
    // then prove a second ordinal is real before touching any guard.
    let Some(dev0) = grim_backend_rocm::require_gpu_device() else {
        eprintln!("[SKIP] requires 2 devices (no device 0)");
        return;
    };
    let _b = RocmDevice::try_new(1).unwrap_or_else(|e| {
        panic!("try_new(1) failed on a 2-device box: {e}")
    });
    assert_eq!(
        dev0.ordinal(),
        0,
        "require_gpu_device must land on ordinal 0 for this fixture"
    );
    let dev0 = Arc::new(dev0);

    let row_len = 1024usize;
    let n = row_len;
    let x: Vec<f32> = (0..n).map(|i| (0.1 + i as f32 * 0.61).sin() * 0.5).collect();
    let r: Vec<f32> = (0..n).map(|i| (2.7 + i as f32 * 0.31).sin() * 0.5).collect();
    let w: Vec<f32> = vec![1.0; row_len];

    // (a) Clean baseline: tensors uploaded and run pinned to device 0.
    let (s_clean, n_clean) = fused_once(dev0.as_ref(), &x, &r, &w);

    // (b) Wrong pin: the *same* fused launch on device-0 tensors while the
    // thread is pinned to ordinal 1. The launch seam's
    // DeviceGuard::set(0) must win (drift-proof), so the result is
    // bit-identical to the clean baseline.
    let (s_wrong, n_wrong) = {
        let _wrong = DeviceGuard::set(1);
        fused_once(dev0.as_ref(), &x, &r, &w)
    };

    // (c) Foreign context: a second live thread parked on device 1 via a held
    // DeviceGuard while this thread runs the identical launch on device 0.
    // Mirrors context_drift_tests.rs::park_foreign_context inline.
    let (s_foreign, n_foreign) = {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            let _g = DeviceGuard::set(1);
            ready_tx.send(()).expect("park ready signal");
            let _ = release_rx.recv();
        });
        ready_rx
            .recv()
            .expect("worker parked its context on device 1");
        let out = fused_once(dev0.as_ref(), &x, &r, &w);
        let _ = release_tx.send(());
        out
    };

    for i in 0..n {
        assert_eq!(
            s_clean[i].to_bits(),
            s_wrong[i].to_bits(),
            "wrong-device pin drifted the fused sum at {i}: {} vs {}",
            s_clean[i],
            s_wrong[i]
        );
        assert_eq!(
            n_clean[i].to_bits(),
            n_wrong[i].to_bits(),
            "wrong-device pin drifted the fused norm at {i}: {} vs {}",
            n_clean[i],
            n_wrong[i]
        );
        assert_eq!(
            s_clean[i].to_bits(),
            s_foreign[i].to_bits(),
            "foreign-context drifted the fused sum at {i}: {} vs {}",
            s_clean[i],
            s_foreign[i]
        );
        assert_eq!(
            n_clean[i].to_bits(),
            n_foreign[i].to_bits(),
            "foreign-context drifted the fused norm at {i}: {} vs {}",
            n_clean[i],
            n_foreign[i]
        );
    }
}
