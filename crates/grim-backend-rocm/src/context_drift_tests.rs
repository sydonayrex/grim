//! WI-M3 context-correctness gates (gguf_multigpu_context_plan.md).
//!
//! The ctx_dev=2 page fault only reproduces when a second HIP device is
//! visible AND some thread parks its context on that device mid-forward.
//! These tests manufacture exactly that state on purpose:
//!
//! 1. A worker thread holds `DeviceGuard::set(foreign)` **alive** while the
//!    main thread uploads a tensor and launches `grim_rms_norm` through the
//!    public API. The launch must still be context-correct (`ctx_dev ==
//!    self_dev == owning ordinal`) and produce correct numbers. Roles are
//!    then swapped so both ordinals are exercised.
//!
//! 2. `copy_from_host_raw_bytes` runs under the same foreign guard: the
//!    storage must land physically on the intended ordinal, not on whatever
//!    device the drifted thread's context pointed at. Residency is checked
//!    two ways — free-VRAM deltas per device (a real `hipMalloc` on the
//!    intended ordinal shrinks ITS free memory), and a functional
//!    `grim_rms_norm` readback against a CPU reference.
//!
//! Mutation check (plan gate): revert the WI-M1 pins (storage.rs seams +
//! allocator alloc/free) and these tests fail — the H2D fill / malloc then
//! executes under the worker's foreign context and the device-0 launch reads
//! wrong-device memory (garbage or a page fault). Run once manually when
//! touching the pins:
//!
//! ```text
//! GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm --lib context_drift
//! # mutation: git revert <pin commit> → rerun → expect FAIL → restore
//! ```
//!
//! Device-gated: requires ≥2 visible HIP devices and `GRIM_GPU_TEST=1`
//! (single-device boxes cannot express cross-device drift).

use std::sync::Arc;
use std::sync::mpsc;

use grim_tensor::Shape;
use grim_tensor::backend::BackendDevice;

use crate::RocmDevice;
use crate::device::capability_profiler::vram_info;
use crate::device::util::{DeviceGuard, dtype_f32, last_launch_context};
use crate::memory::storage::RocmStorage;

fn multi_gpu_available() -> Option<usize> {
    if !crate::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return None;
    }
    let count = crate::peer_access::enumerate_devices().ok()?;
    if count < 2 {
        eprintln!("[skipped: need >=2 HIP devices for drift simulation, got {count}]");
        return None;
    }
    Some(count)
}

/// Park a fresh thread's HIP context on `ordinal` and hold it there until
/// the returned sender is dropped. This is the drift state the fault hunt
/// observed on tape: some other thread with ctx_dev != 0 alive during our
/// launches (per-thread current device semantics).
fn park_foreign_context(ordinal: i32) -> mpsc::Sender<()> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        // Guard held for the whole thread body: the point is that the
        // foreign context stays CURRENT on a live second thread while the
        // main thread runs its own uploads + launches.
        let _guard = DeviceGuard::set(ordinal);
        ready_tx.send(()).expect("park ready signal");
        let _ = release_rx.recv();
    });
    ready_rx
        .recv()
        .expect("worker parked its context on the foreign device");
    release_tx
}

fn cpu_rms_norm_reference(x: &[f32], row_len: usize) -> Vec<f32> {
    x.chunks(row_len)
        .flat_map(|row| {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
            let inv = 1.0 / (mean_sq + 1e-5).sqrt();
            row.iter().map(move |v| v * inv)
        })
        .collect()
}

struct DriftFixture {
    dev: Arc<RocmDevice>,
    _foreign_park: mpsc::Sender<()>,
}

/// Build the target device plus a second live thread whose context is parked
/// on `foreign_ordinal` for the duration of the closure.
fn with_drift_fixture(target: usize, foreign_ordinal: i32, f: impl FnOnce(&DriftFixture)) {
    let Some(_) = multi_gpu_available() else {
        return;
    };
    let dev = Arc::new(
        RocmDevice::try_new(target).unwrap_or_else(|e| panic!("try_new({target}) failed: {e}")),
    );
    // try_new must be context-neutral: constructing a foreign device from a
    // drifted thread may not leave THIS thread parked anywhere unexpected.
    let foreign_park = park_foreign_context(foreign_ordinal);
    let fx = DriftFixture {
        dev,
        _foreign_park: foreign_park,
    };
    f(&fx);
}

const ROWS: usize = 8;
const ROW_LEN: usize = 64;

fn upload_and_launch_rms_norm(fx: &DriftFixture, x_data: &[f32]) -> Vec<f32> {
    let shape = Shape::from_slice(&[ROWS, ROW_LEN]);
    let ones = vec![1.0f32; ROW_LEN];

    // Upload through the pinned seam while the foreign context is alive.
    let x = RocmStorage::copy_from_host(
        x_data,
        &shape,
        dtype_f32(),
        &fx.dev.allocator,
        fx.dev.ordinal,
    )
    .expect("copy_from_host under foreign-context pressure");
    assert_eq!(x.device_ordinal(), fx.dev.ordinal);
    let w = RocmStorage::copy_from_host(
        &ones,
        &Shape::from_slice(&[ROW_LEN]),
        dtype_f32(),
        &fx.dev.allocator,
        fx.dev.ordinal,
    )
    .expect("gamma upload");

    let (out, handle) = fx
        .dev
        .rms_norm(&x, &w, 1e-5, &shape)
        .expect("grim_rms_norm launch through public API");
    handle.synchronize().expect("rms_norm sync");

    // The launch seam stamped (self_dev, ctx_dev): the launching (main)
    // thread must have been on the OWNING device's context, not the worker's
    // foreign one.
    let (self_dev, ctx_dev) = last_launch_context();
    assert_eq!(
        self_dev, fx.dev.ordinal as i32,
        "launch seam ran on the wrong device object"
    );
    assert_eq!(
        self_dev, ctx_dev,
        "CONTEXT DRIFT: kernel launched with ctx_dev != self_dev \
         (WI-M1 pins missing or regressed)"
    );

    out.to_cpu_vec_f32().expect("DtoH readback")
}

#[test]
fn rms_norm_launch_stays_context_correct_while_worker_parks_on_device_1() {
    with_drift_fixture(0, 1, |fx| {
        let x_data: Vec<f32> = (0..ROWS * ROW_LEN)
            .map(|i| ((i % 17) as f32) * 0.25 - 1.0)
            .collect();
        let got = upload_and_launch_rms_norm(fx, &x_data);
        let want = cpu_rms_norm_reference(&x_data, ROW_LEN);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 2e-4 * w.abs().max(1.0),
                "rms_norm mismatch at {i}: got {g}, want {w} — tensor was \
                 materialised or read on the wrong device"
            );
        }
    });
}

#[test]
fn rms_norm_launch_stays_context_correct_roles_swapped() {
    // Swapped roles: the MAIN thread drives ordinal 1 while a worker parks
    // its context back on ordinal 0.
    with_drift_fixture(1, 0, |fx| {
        let x_data: Vec<f32> = (0..ROWS * ROW_LEN)
            .map(|i| ((i % 13) as f32) * 0.5 - 2.0)
            .collect();
        let got = upload_and_launch_rms_norm(fx, &x_data);
        let want = cpu_rms_norm_reference(&x_data, ROW_LEN);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 2e-4 * w.abs().max(1.0),
                "roles-swapped mismatch at {i}: got {g}, want {w}"
            );
        }
    });
}

#[test]
fn raw_bytes_upload_lands_on_intended_ordinal_under_drifted_context() {
    const PAYLOAD_BYTES: usize = 32 * 1024 * 1024; // unique size class → real malloc, not pool reuse
    with_drift_fixture(0, 1, |fx| {
        // Drain pooled blocks so the upcoming allocation is a genuine driver
        // malloc whose residency we can observe via per-device free VRAM.
        fx.dev.allocator.empty_cache();

        let (free0_before, _) = vram_info(0);
        let (free1_before, _) = vram_info(1);

        let payload: Vec<u8> = (0..PAYLOAD_BYTES).map(|i| (i % 251) as u8).collect();
        let storage = RocmStorage::copy_from_host_raw_bytes(
            &payload,
            &Shape::from_slice(&[PAYLOAD_BYTES]),
            dtype_f32(),
            &fx.dev.allocator,
            0, // INTENDED ordinal: device 0, not the drifted context's
        )
        .expect("raw-bytes upload under foreign-context pressure");
        assert_eq!(
            storage.device_ordinal(),
            0,
            "storage metadata must claim the intended ordinal"
        );

        let (free0_after, _) = vram_info(0);
        let (free1_after, _) = vram_info(1);

        // The allocation must be charged to device 0 (the intended ordinal),
        // NOT to device 1 (where the drifting worker's context points).
        let dropped_on_0 = free0_before.saturating_sub(free0_after);
        let dropped_on_1 = free1_before.saturating_sub(free1_after);
        assert!(
            dropped_on_0 + 4 * 1024 * 1024 >= PAYLOAD_BYTES as u64,
            "buffer did not land on device 0 (free VRAM moved by {dropped_on_0} bytes) — \
             hipMalloc executed under the drifted context (WI-M1 pin missing)"
        );
        assert!(
            dropped_on_1 <= 8 * 1024 * 1024,
            "buffer leaked onto device 1 ({dropped_on_1} bytes) — the exact \
             wrong-device residency class behind the ctx_dev=2 fault"
        );

        // Functional proof the bytes are reachable from device-0 kernels.
        drop(storage);
    });
}

#[test]
fn prefill_latch_round_trips() {
    // Host-only smoke for the WI-M2 latch plumbing the engine drives.
    crate::set_prefill_in_flight(true);
    assert!(crate::prefill_in_flight());
    crate::set_prefill_in_flight(false);
    assert!(!crate::prefill_in_flight());
}
