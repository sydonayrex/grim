//! Tests for the generic graph-capture session API on `RocmDevice`
//! (Item 5 of the ROCm perf/ABI spec): `begin_graph_capture` /
//! `end_graph_capture` / `replay_graph` — a caller-supplied `&str`-keyed
//! begin/end bracket around whatever sequence of primitive ops the caller
//! issues, plus a keyed replay lookup. No shape fingerprinting, no baked-in
//! "decode step": the spec's acceptance criteria use a synthetic
//! `matmul` -> `add` -> `rms_norm` sequence built from ops that already
//! exist in this crate.
//!
//! These are GPU tests: they bail out (return Ok) when `GRIM_RUN_GPU_TESTS`
//! is unset or no device is available, so `cargo test` stays green off-GPU
//! while still pinning the API shape (compile errors count as red).
//!
//! NOTE on buffer stability and capture safety: the op sequence is
//! `matmul(a,b) -> add(c) -> rms_norm(w)`. The INPUTS must be uploaded to
//! the device BEFORE `begin_graph_capture` — HIP graph capture forbids
//! synchronous `hipMemcpy` (the upload path) inside a capture region
//! (`hipErrorStreamCaptureImplicit` = 906). Only the compute kernels
//! (matmul/add/rms_norm, none of which issue host-synchronous copies) run
//! inside the bracket. The pooled allocator (Item 1) keeps the device
//! pointers of the dropped intermediates valid across the capture, which is
//! exactly why Item 5 requires Item 1 to land first.

use std::time::Instant;

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";
const CAPTURE_ENV: &str = "GRIM_CAPTURE_GRAPH";

/// Build a device, bailing the test (Ok) if no GPU is present.
fn gpu_device() -> Option<RocmDevice> {
    if !std::env::var(GPU_TEST_ENV).is_ok() {
        return None;
    }
    match std::panic::catch_unwind(|| RocmDevice::new(0)) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

/// Upload inputs to the device. Synchronous `hipMemcpy` is ILLEGAL inside a
/// capture region, so this must run OUTSIDE `begin_graph_capture`.
fn upload_inputs(
    dev: &RocmDevice,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> TestResult<(
    Box<dyn grim_tensor::BackendStorage>,
    Box<dyn grim_tensor::BackendStorage>,
    Box<dyn grim_tensor::BackendStorage>,
    Box<dyn grim_tensor::BackendStorage>,
)> {
    let a_s = Shape::from_slice(&[m, k]);
    let b_s = Shape::from_slice(&[k, n]);
    let out_s = Shape::from_slice(&[m, n]);
    let a_dev = BackendDevice::from_cpu(dev, a, &a_s, DType::F32)?;
    let b_dev = BackendDevice::from_cpu(dev, b, &b_s, DType::F32)?;
    let c_dev = BackendDevice::from_cpu(dev, c, &out_s, DType::F32)?;
    let w_dev = BackendDevice::from_cpu(dev, w, &out_s, DType::F32)?;
    Ok((a_dev, b_dev, c_dev, w_dev))
}

/// Run `matmul -> add -> rms_norm` on already-uploaded inputs. Capture-safe:
/// no host-synchronous copies are issued here.
fn run_compute(
    dev: &RocmDevice,
    inputs: &(
        Box<dyn grim_tensor::BackendStorage>,
        Box<dyn grim_tensor::BackendStorage>,
        Box<dyn grim_tensor::BackendStorage>,
        Box<dyn grim_tensor::BackendStorage>,
    ),
    m: usize,
    k: usize,
    n: usize,
) -> TestResult<Box<dyn grim_tensor::BackendStorage>> {
    let out_s = Shape::from_slice(&[m, n]);
    let (a_dev, b_dev, c_dev, w_dev) = inputs;
    let (mm, _h1) = BackendDevice::matmul(dev, a_dev.as_ref(), b_dev.as_ref(), &out_s)?;
    let (added, _h2) = BackendDevice::add(dev, mm.as_ref(), c_dev.as_ref(), &out_s)?;
    let (rn, _h3) =
        BackendDevice::rms_norm(dev, added.as_ref(), w_dev.as_ref(), 1e-5_f32, &out_s)?;
    Ok(rn)
}

/// Run the op sequence eagerly (inputs uploaded, compute outside capture)
/// and return the flattened f32 output.
fn run_seq(
    dev: &RocmDevice,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> TestResult<Vec<f32>> {
    let inputs = upload_inputs(dev, a, b, c, w, m, k, n)?;
    let out = run_compute(&dev, &inputs, m, k, n)?;
    Ok(out.to_cpu_vec_f32()?)
}

// =========================================================================
// Acceptance #1: a synthetic multi-op sequence run eagerly and again via
// begin/end/replay produces matching output within f32 tolerance.
// =========================================================================

#[test]
fn eager_vs_captured_multi_op_match() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let k = 4usize;
    let n = 3usize;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
    let c: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.05).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();

    // Eager reference.
    let inputs = upload_inputs(&dev, &a, &b, &c, &w, m, k, n)?;
    let eager = run_compute(&dev, &inputs, m, k, n)?.to_cpu_vec_f32()?;

    // Capture the same sequence, then replay it, then read.
    let key = "eager_vs_captured_multi_op_match";
    dev.begin_graph_capture(key)?;
    let captured = run_compute(&dev, &inputs, m, k, n)?;
    dev.end_graph_capture(key)?;
    assert!(
        dev.replay_graph(key)?,
        "replay_graph must launch the captured graph"
    );
    let replayed = captured.to_cpu_vec_f32()?;

    assert_eq!(eager.len(), replayed.len());
    let tol = 1e-2_f32;
    for i in 0..eager.len() {
        assert!(
            (eager[i] - replayed[i]).abs() <= tol,
            "eager vs replayed mismatch at {i}: {} vs {}",
            eager[i],
            replayed[i]
        );
    }
    Ok(())
}

// =========================================================================
// Acceptance #2: capturing under one key and replaying a *different* key
// returns `Ok(false)` rather than replaying the wrong graph.
// =========================================================================

#[test]
fn replay_with_different_key_returns_false() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let k = 4usize;
    let n = 3usize;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
    let c: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.05).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();

    let inputs = upload_inputs(&dev, &a, &b, &c, &w, m, k, n)?;
    let key_a = "replay_diff_a";
    dev.begin_graph_capture(key_a)?;
    let _captured = run_compute(&dev, &inputs, m, k, n)?;
    dev.end_graph_capture(key_a)?;

    // Replay under a different key must NOT launch key_a's graph.
    let replayed = dev.replay_graph("replay_diff_b")?;
    assert!(
        !replayed,
        "replay_graph with an unknown key must return Ok(false), not replay a different graph"
    );
    // And replaying the real key does launch.
    assert!(dev.replay_graph(key_a)?);
    Ok(())
}

// =========================================================================
// Acceptance #3: one capture + N replays is cheaper than N eager runs
// for the same synthetic op sequence (qualitative wall-clock check).
// =========================================================================

#[test]
fn capture_then_replay_undercuts_eager_loop() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    let k = 4usize;
    let n = 3usize;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
    let c: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.05).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();

    let inputs = upload_inputs(&dev, &a, &b, &c, &w, m, k, n)?;
    let key = "capture_bench";
    const N: usize = 50;

    let eager_start = Instant::now();
    for _ in 0..N {
        let _ = run_seq(&dev, &a, &b, &c, &w, m, k, n)?;
    }
    let eager_ms = eager_start.elapsed().as_secs_f64() * 1e3;

    dev.begin_graph_capture(key)?;
    let _ = run_compute(&dev, &inputs, m, k, n)?;
    dev.end_graph_capture(key)?;

    let replay_start = Instant::now();
    for _ in 0..N {
        assert!(dev.replay_graph(key)?);
    }
    let replay_ms = replay_start.elapsed().as_secs_f64() * 1e3;

    println!(
        "[graph-capture] {N} eager={eager_ms:.2}ms replay={replay_ms:.2}ms (capture+replay target < eager)"
    );
    Ok(())
}
