//! Correctness-parity and infrastructure test for the WMMA GEMM kernel (WI-G).
//!
//! Verifies that when `WmmaGemmConfig` is enabled, the backend correctly compiles
//! the JIT WMMA GEMM kernel, dispatches to it under `RocmDevice::matmul`, and
//! yields outputs matching a CPU host oracle. On the current RDNA2 (gfx1036) system,
//! this executes the scalar fallback path within the JIT kernel, verifying full JIT compile
//! and launch infrastructure safety.

use std::panic;
use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

/// Env var opting into GPU execution. If unset, tests bail Ok to run on CPU-only CI.
const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// Build a RocmDevice if the GPU test environment is enabled.
///
/// Returns `None` if `GRIM_RUN_GPU_TESTS` is not set or device instantiation fails.
fn gpu_device() -> Option<RocmDevice> {
    if std::env::var(GPU_TEST_ENV).is_err() {
        return None;
    }
    match panic::catch_unwind(|| RocmDevice::new(0)) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

/// Computes a standard row-major float GEMM reference on the CPU.
///
/// Matrix dimensions are: A is [m x k], B is [k x n], and out is [m x n].
fn host_gemm_f32(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            out[mi * n + ni] = acc;
        }
    }
}

/// Computes the max absolute difference between two float slices.
fn max_abs_diff(gpu: &[f32], oracle: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..gpu.len().min(oracle.len()) {
        let d = (gpu[i] - oracle[i]).abs();
        if d > max {
            max = d;
        }
    }
    max
}

/// Runs the WMMA GEMM kernel on the device for the given inputs and returns the result.
fn run_wmma_kernel(
    dev: &RocmDevice,
    a_data: &[f32],
    b_data: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> TestResult<Vec<f32>> {
    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[k, n]);
    let out_shape = Shape::from_slice(&[m, n]);

    let a_dev = BackendDevice::from_cpu(dev, a_data, &a_shape, DType::F16)?;
    let b_dev = BackendDevice::from_cpu(dev, b_data, &b_shape, DType::F16)?;

    // Enable the WMMA JIT branch
    dev.set_wmma_gemm_enabled(true);
    let (out, handle) = BackendDevice::matmul(dev, a_dev.as_ref(), b_dev.as_ref(), &out_shape)?;
    handle.synchronize()?;
    dev.set_wmma_gemm_enabled(false);

    Ok(out.as_ref().to_cpu_vec_f32()?)
}

/// Verifies that the JIT compiled WMMA GEMM kernel can be successfully
/// enqueued and its output matches the host reference within a tolerance.
#[test]
fn test_wmma_gemm_infrastructure_and_correctness() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let (m, k, n) = (8usize, 128usize, 128usize);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| (i as f32 * 0.02).sin() * 0.5)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| (i as f32 * 0.04).cos() * 0.5)
        .collect();

    // 1. Run with WMMA JIT kernel enabled
    let gpu_out = run_wmma_kernel(&dev, &a_data, &b_data, m, k, n)?;

    // 2. Compute host reference
    let mut host_out = vec![0.0f32; m * n];
    host_gemm_f32(&a_data, &b_data, &mut host_out, m, k, n);

    // 3. Compare with f16-equivalent tolerance
    let diff = max_abs_diff(&gpu_out, &host_out);
    let tolerance = 1e-2;
    assert!(
        diff <= tolerance,
        "WMMA JIT output mismatch: max abs diff {diff:.e} exceeds tolerance {tolerance:.e}"
    );

    Ok(())
}
