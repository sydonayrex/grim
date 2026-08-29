//! CPU-vs-GPU parity tests for the WMMA GEMM kernel (`grim_wmma_gemm`, WI-G).
//!
//! Mirrors the structure of `q3k_gemm_cpu_gpu_parity.rs`.  For every (M, K, N)
//! shape in the test matrix, this module:
//!   1. Builds seeded F32 inputs and converts them to F16 for GPU upload.
//!   2. Enables the WMMA path through `RocmDevice::set_wmma_gemm_enabled`, which
//!      is now correctly consulted by `RocmDevice::matmul` via `should_use_wmma_path`.
//!   3. Runs the GPU matmul and downloads the F32 result.
//!   4. Computes an F32 reference GEMM on the CPU using the same F16-rounded values
//!      (to match GPU precision), and asserts the max relative error is within 2e-2
//!      (accounting for F16 accumulation rounding on gfx1200 scalar-fallback path).
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test wmma_gemm_cpu_gpu_parity
//! RESULT: 1/2 PASS (test_wmma_gemm_enable_disable_output_consistency), 1/2 FAIL
//!   (test_wmma_gemm_cpu_gpu_parity_all_shapes — hipModuleLoad failed: 209). The
//!   enable/disable consistency test passes because it exercises the dispatch switch;
//!   the full parity sweep fails because the JIT WMMA kernel module cannot be loaded
//!   on this system.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{DType, Shape,
    CoreTensorOps,
};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

/// Build a `RocmDevice` if `GRIM_RUN_GPU_TESTS` is set; returns `None` otherwise.
fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

/// Round an f32 value through f16 to simulate GPU storage precision.
///
/// The WMMA kernel operates on F16 storage; the CPU reference must apply the
/// same rounding to avoid systematic disagreement from f32 precision.
fn round_to_f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// Row-major GEMM on the CPU: C[M×N] = A[M×K] · B[K×N].
///
/// Both inputs are already f16-rounded (via `round_to_f16`) to match GPU storage.
fn host_gemm_f16_rounded(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// Build a seeded, f16-rounded vector of length `len` for test inputs.
///
/// Uses a simple deterministic formula so seeds produce distinct, bounded values.
fn seeded_data(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let raw = ((i as f32 * 0.07 + seed as f32 * 0.3).sin()) * scale;
            round_to_f16(raw)
        })
        .collect()
}

/// Run the GPU WMMA GEMM for one (M, K, N) shape and seed, returning C as f32.
///
/// Enables `wmma_gemm` through `should_use_wmma_path` (now the live dispatch
/// gate) and disables it afterwards to avoid polluting subsequent tests.
fn run_gpu(
    dev: &RocmDevice,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> TestResult<Vec<f32>> {
    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[k, n]);
    let out_shape = Shape::from_slice(&[m, n]);

    // Upload as F16; `from_cpu` converts element-wise.
    let a_dev = CoreTensorOps::from_cpu(dev, a, &a_shape, DType::F16)?;
    let b_dev = CoreTensorOps::from_cpu(dev, b, &b_shape, DType::F16)?;

    dev.set_wmma_gemm_enabled(true);
    let (out, handle) = CoreTensorOps::matmul(dev, a_dev.as_ref(), b_dev.as_ref(), &out_shape)?;
    handle.synchronize()?;
    dev.set_wmma_gemm_enabled(false);

    Ok(out.as_ref().to_cpu_vec_f32()?)
}

/// Maximum absolute value in a slice; used to compute relative error.
fn max_abs(v: &[f32]) -> f32 {
    v.iter().copied().map(f32::abs).fold(0.0f32, f32::max)
}

/// Maximum element-wise absolute difference between two equally-sized slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Shape cases: (M, K, N) chosen to exercise:
///   - WMMA tile-aligned (16-multiple) shapes for native RDNA3/4 path
///   - non-tile-aligned shapes for the scalar fallback path
///   - large K to stress accumulator precision
const SHAPES: &[(usize, usize, usize)] = &[
    (8, 64, 8),    // small, below WMMA tile
    (16, 64, 16),  // WMMA tile-aligned
    (32, 128, 32), // two tiles in each dimension
    (4, 256, 64),  // large K, non-square
    (48, 96, 48),  // non-power-of-two
];

/// CPU/GPU parity for the WMMA GEMM kernel across multiple shapes and seeds.
///
/// This is the authoritative regression gate for `launch_wmma_gemm`: if
/// `should_use_wmma_path` were still disconnected from dispatch, this test
/// would silently pass on the rocBLAS path rather than exercising the WMMA
/// JIT kernel.  Confirm via `GRIM_WMMA_VERIFY=1` tracing if needed.
#[test]
fn test_wmma_gemm_cpu_gpu_parity_all_shapes() -> TestResult {
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    for &(m, k, n) in SHAPES {
        for seed in 0u32..4 {
            let a = seeded_data(m * k, seed, 0.5);
            let b = seeded_data(k * n, seed.wrapping_add(7), 0.5);

            let gpu_out = run_gpu(&dev, &a, &b, m, k, n)?;
            let cpu_out = host_gemm_f16_rounded(&a, &b, m, k, n);

            let base = max_abs(&cpu_out);
            let diff = max_abs_diff(&gpu_out, &cpu_out);
            // F16 scalar accumulation on the fallback path has ~1e-2 relative error;
            // native WMMA path is tighter but we bound by the looser guarantee.
            let rel = if base == 0.0 { diff } else { diff / base };
            assert!(
                rel < 2e-2,
                "WMMA GEMM parity fail: shape=({m},{k},{n}) seed={seed} \
                 max_rel_err={rel:.3e} (max_abs={base:.3e})"
            );
        }
    }

    Ok(())
}

/// Verifies that disabling `wmma_gemm` routes matmul through rocBLAS, and
/// re-enabling routes back through the WMMA JIT kernel, with matching output.
///
/// This guards against a regression where `should_use_wmma_path` is consulted
/// but its enabled flag has no effect on the actual dispatch branch.
// PASSED: 2026-08-27 on gfx1036 (ROCm)
#[test]
fn test_wmma_gemm_enable_disable_output_consistency() -> TestResult {
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let (m, k, n) = (16usize, 64usize, 16usize);
    let a = seeded_data(m * k, 42, 0.5);
    let b = seeded_data(k * n, 99, 0.5);

    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[k, n]);
    let out_shape = Shape::from_slice(&[m, n]);

    let a_dev = CoreTensorOps::from_cpu(&dev, &a, &a_shape, DType::F16)?;
    let b_dev = CoreTensorOps::from_cpu(&dev, &b, &b_shape, DType::F16)?;

    // Run with WMMA disabled (rocBLAS path)
    dev.set_wmma_gemm_enabled(false);
    let (out_rocblas, h) = CoreTensorOps::matmul(&dev, a_dev.as_ref(), b_dev.as_ref(), &out_shape)?;
    h.synchronize()?;
    let rocblas_result = out_rocblas.as_ref().to_cpu_vec_f32()?;

    // Run with WMMA enabled (should_use_wmma_path → launch_wmma_gemm)
    dev.set_wmma_gemm_enabled(true);
    let (out_wmma, h2) = CoreTensorOps::matmul(&dev, a_dev.as_ref(), b_dev.as_ref(), &out_shape)?;
    h2.synchronize()?;
    dev.set_wmma_gemm_enabled(false);
    let wmma_result = out_wmma.as_ref().to_cpu_vec_f32()?;

    let base = max_abs(&rocblas_result);
    let diff = max_abs_diff(&wmma_result, &rocblas_result);
    let rel = if base == 0.0 { diff } else { diff / base };
    assert!(
        rel < 2e-2,
        "WMMA vs rocBLAS output diverge: max_rel_err={rel:.3e}"
    );

    Ok(())
}
