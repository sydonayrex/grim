//! Gate 2.6.4 — decode-GEMM correctness-parity test (`grim_rocm_consumer_perf_planv2.md`).
//!
//! Per v2's "concrete next step for Work Item 2.4.4-2", this is the missing
//! gate that must exist and compile before `DecodeGemmConfig::enabled`'s
//! default may ever flip from `false`. It runs F16, `m <= 8` decode input
//! through the opt-in JIT decode kernel (`grim_decode_gemm_f16`, via
//! `set_decode_gemm_enabled(true)`) and compares the output against an
//! independent CPU host oracle (f32 GEMM with F16 input quantization).
//!
//! The rocBLAS path is used as a *secondary* cross-check when available, but
//! on some consumer hardware (gfx1036 / Radeon 610M) rocBLAS F16 `gemm_ex`
//! returns `rocblas_status_invalid_value` for certain shapes — in that case
//! the test falls back to the CPU oracle alone, which is the plan's actual
//! correctness contract ("compare against a CPU reference").
//!
//! GPU-gated, mirroring the established `graph_capture.rs` pattern: these
//! tests bail `Ok` when `GRIM_RUN_GPU_TESTS` is unset or no device is present.

use std::time::Instant;

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

/// Env var that opts into GPU execution. Unset ⇒ every test below bails Ok.
const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// Build a device, bailing the whole test (Ok) if no GPU is present.
fn gpu_device() -> Option<RocmDevice> {
    if !std::env::var(GPU_TEST_ENV).is_ok() {
        return None;
    }
    match std::panic::catch_unwind(|| RocmDevice::new(0)) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

/// Small host reference GEMM in f32, the independent oracle. `a`/`b`/`out`
/// are row-major f32 views; result is written into `out` (len `m*n`).
fn host_gemm_f32(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            out[mi * n + ni] = acc;
        }
    }
}

/// F16-equivalent tolerance. The decode kernel accumulates in f32 then rounds
/// to F16; the host oracle is pure f32 but the GPU inputs are F16-quantized,
/// introducing ~1e-2 relative error for values near 1.0. We use a slightly
/// looser tolerance for large K (more accumulation steps → more rounding).
fn tolerance_for(k: usize) -> f32 {
    if k >= 4096 { 0.5 } else if k >= 512 { 0.1 } else { 1e-2 }
}

/// Run the decode kernel on the given F16 inputs, return the output as f32.
fn run_decode_kernel(
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
    dev.set_decode_gemm_enabled(true);
    let (out, handle) = BackendDevice::matmul(dev, a_dev.as_ref(), b_dev.as_ref(), &out_shape)?;
    handle.synchronize()?;
    dev.set_decode_gemm_enabled(false);
    Ok(out.as_ref().to_cpu_vec_f32()?)
}

/// Try to run rocBLAS (decode kernel OFF) on the same inputs. Returns None if
/// rocBLAS F16 is unsupported on this hardware (status 11 / invalid_value).
fn try_rocblas(
    dev: &RocmDevice,
    a_data: &[f32],
    b_data: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> TestResult<Option<Vec<f32>>> {
    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[k, n]);
    let out_shape = Shape::from_slice(&[m, n]);
    let a_dev = BackendDevice::from_cpu(dev, a_data, &a_shape, DType::F16)?;
    let b_dev = BackendDevice::from_cpu(dev, b_data, &b_shape, DType::F16)?;
    let result = BackendDevice::matmul(dev, a_dev.as_ref(), b_dev.as_ref(), &out_shape);
    match result {
        Ok((out, handle)) => {
            handle.synchronize()?;
            Ok(Some(out.as_ref().to_cpu_vec_f32()?))
        }
        Err(_) => {
            // rocBLAS F16 may return invalid_value on some consumer hardware.
            // This is a rocBLAS limitation, not a decode-kernel bug.
            eprintln!("[gate 2.6.4] rocBLAS F16 unavailable on this hardware — using CPU oracle only");
            Ok(None)
        }
    }
}

/// Compare GPU output against CPU oracle, return max abs diff.
fn max_abs_diff(gpu: &[f32], oracle: &[f32]) -> f32 {
    let mut max = 0f32;
    for i in 0..gpu.len().min(oracle.len()) {
        let d = (gpu[i] - oracle[i]).abs();
        if d > max {
            max = d;
        }
    }
    max
}

/// Gate 2.6.4 (correctness) — the decode kernel must match the CPU oracle
/// for the canonical Llama decode shape.
#[test]
fn gate_2_6_4_decode_gemm_matches_cpu_oracle() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let (m, k, n) = (1usize, 128, 128);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| (((i as f32 * 0.01).sin() * 0.5) as f32))
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| (((i as f32 * 0.03).cos() * 0.5) as f32))
        .collect();

    let gpu = run_decode_kernel(&dev, &a_data, &b_data, m, k, n)?;
    let mut host = vec![0f32; m * n];
    host_gemm_f32(&a_data, &b_data, &mut host, m, k, n);

    let tol = tolerance_for(k);
    let diff = max_abs_diff(&gpu, &host);
    assert!(
        diff <= tol,
        "decode kernel vs CPU oracle: max abs diff {diff:.e} exceeds tol {tol:.e}"
    );

    // Secondary: cross-check with rocBLAS if available on this hardware.
    if let Some(ref_gpu) = try_rocblas(&dev, &a_data, &b_data, m, k, n)? {
        let rdiff = max_abs_diff(&gpu, &ref_gpu);
        assert!(
            rdiff <= tol,
            "decode kernel vs rocBLAS: max abs diff {rdiff:.e} exceeds tol {tol:.e}"
        );
    }
    Ok(())
}

/// Gate 2.6.4 (m=8 multi-row) — full M-tile decode shape.
#[test]
fn gate_2_6_4_decode_gemm_m8_matches_cpu_oracle() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let (m, k, n) = (8usize, 128, 128);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| (((i as f32 * 0.01).sin() * 0.5) as f32))
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| (((i as f32 * 0.03).cos() * 0.5) as f32))
        .collect();

    let gpu = run_decode_kernel(&dev, &a_data, &b_data, m, k, n)?;
    let mut host = vec![0f32; m * n];
    host_gemm_f32(&a_data, &b_data, &mut host, m, k, n);

    let tol = tolerance_for(k);
    let diff = max_abs_diff(&gpu, &host);
    assert!(
        diff <= tol,
        "m=8 decode kernel vs CPU oracle: max abs diff {diff:.e} exceeds tol {tol:.e}"
    );
    Ok(())
}

/// Gate 2.6.4 (throughput) — the decode kernel must not be catastrophically
/// slow. Rather than asserting it beats rocBLAS (which may not work on this
/// hardware), we assert it completes in a reasonable wall-clock time.
#[test]
fn gate_2_6_4_decode_gemm_completes_in_reasonable_time() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let (m, k, n) = (8usize, 128, 128);
    let a_data = vec![0.5f32; m * k];
    let b_data = vec![0.25f32; k * n];

    let t0 = Instant::now();
    let _gpu = run_decode_kernel(&dev, &a_data, &b_data, m, k, n)?;
    let elapsed = t0.elapsed();

    // The Radeon 610M has 2 CUs; a (8,128,128) F16 GEMM should complete in
    // well under 1 second even with JIT compile overhead on the first call.
    // This is a sanity check, not a performance gate — the real perf gate
    // (Gate 2.6.4 item 4) requires comparing against rocBLAS on hardware
    // that supports F16 gemm_ex.
    assert!(
        elapsed.as_secs() < 10,
        "decode kernel took {elapsed:?} — unreasonably slow for ({m},{k},{n})"
    );
    eprintln!("[gate 2.6.4] decode kernel ({m},{k},{n}) completed in {elapsed:?}");
    Ok(())
}

// ===========================================================================
// Gate 2.6.2b — double-buffer race safety + asymmetric tiling correctness.
// ===========================================================================

/// Gate 2.6.2b (part 1) — decode kernel must match CPU oracle for aligned
/// and irregular N shapes. The irregular shape exercises OOB masking on
/// the last N-block.
#[test]
fn gate_2_6_2b_decode_gemm_aligned_and_irregular_n() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    // (m, k, n) tuples: N_TILE=64, so N=64 is aligned, N=65 exercises OOB.
    let shapes: &[(usize, usize, usize)] = &[
        (1, 64, 64),   // N exactly one tile
        (1, 64, 65),   // N = tile+1 — partial last block
        (8, 64, 64),   // full M_TILE, one N tile
        (8, 64, 128),  // full M_TILE, two N tiles
    ];

    for &(m, k, n) in shapes {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| (((i as f32 * 0.01).sin() * 0.3) as f32))
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| (((i as f32 * 0.03).cos() * 0.3) as f32))
            .collect();

        let gpu = run_decode_kernel(&dev, &a_data, &b_data, m, k, n)?;
        let mut host = vec![0f32; m * n];
        host_gemm_f32(&a_data, &b_data, &mut host, m, k, n);

        let tol = tolerance_for(k);
        let diff = max_abs_diff(&gpu, &host);
        assert!(
            diff <= tol,
            "shape ({m},{k},{n}): decode kernel vs CPU oracle max abs diff {diff:.e} > tol {tol:.e}"
        );

        // Check for NaN/inf — a race or OOB read produces these.
        for (i, &v) in gpu.iter().enumerate() {
            assert!(
                v.is_finite(),
                "shape ({m},{k},{n}) output[{i}] = {v} — non-finite (race or OOB bug)"
            );
        }
    }
    Ok(())
}

/// Gate 2.6.2b (part 2) — double-buffer race safety: a K dimension large
/// enough to require many buffer swaps must still produce correct, finite
/// output. Run twice to catch non-deterministic races.
#[test]
fn gate_2_6_2b_double_buffer_many_k_steps_no_race() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    // k=256 → 256/16 = 16 K-step iterations, each triggering a buffer swap.
    let (m, k, n) = (8usize, 256, 64);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| (((i as f32 * 0.01).sin() * 0.3) as f32))
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| (((i as f32 * 0.02).cos() * 0.3) as f32))
        .collect();

    let mut prev: Option<Vec<f32>> = None;
    let tol = tolerance_for(k);

    for run in 0..2 {
        let gpu = run_decode_kernel(&dev, &a_data, &b_data, m, k, n)?;

        // Must be finite (a race produces inf/nan silently).
        for (i, &v) in gpu.iter().enumerate() {
            assert!(
                v.is_finite(),
                "run {run}: output[{i}] = {v} — non-finite (double-buffer race)"
            );
        }

        // Must match the CPU oracle.
        let mut host = vec![0f32; m * n];
        host_gemm_f32(&a_data, &b_data, &mut host, m, k, n);
        let diff = max_abs_diff(&gpu, &host);
        assert!(
            diff <= tol,
            "run {run}: vs CPU oracle max abs diff {diff:.e} > tol {tol:.e}"
        );

        // Must match previous run (determinism — a non-deterministic race
        // produces different output across runs).
        if let Some(p) = &prev {
            let run_diff = max_abs_diff(&gpu, p);
            assert!(
                run_diff <= tol,
                "run {run}: differs from run 0 by {run_diff:.e} — non-deterministic race"
            );
        }
        prev = Some(gpu);
    }
    Ok(())
}

/// Off-GPU compile guard: the test module must compile even without a GPU.
#[test]
fn gate_2_6_4_harness_compiles_and_bails_without_gpu() -> TestResult {
    let _ = gpu_device();
    Ok(())
}

/// Minimal debug: (1,16,64) — one K-step, one N-tile, smallest possible shape.
#[test]
fn debug_decode_gemm_minimal() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let (m, k, n) = (1usize, 16, 64);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| (((i as f32 * 0.1).sin() * 0.3) as f32))
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| (((i as f32 * 0.1).cos() * 0.3) as f32))
        .collect();

    // First: verify F16 round-trip works (upload + download without compute).
    let a_shape = Shape::from_slice(&[m, k]);
    let a_gpu = BackendDevice::from_cpu(&dev, &a_data, &a_shape, DType::F16)?;
    let a_back = a_gpu.as_ref().to_cpu_vec_f32()?;
    let rt_diff = max_abs_diff(&a_data, &a_back);
    eprintln!("[debug] F16 round-trip max diff = {rt_diff:.e}");
    eprintln!("[debug] a_data[0..4] = {:?}", &a_data[..4]);
    eprintln!("[debug] a_back[0..4]  = {:?}", &a_back[..4]);

    let gpu = run_decode_kernel(&dev, &a_data, &b_data, m, k, n)?;
    let mut host = vec![0f32; m * n];
    host_gemm_f32(&a_data, &b_data, &mut host, m, k, n);

    eprintln!("[debug] gpu[0..4]   = {:?}", &gpu[..4.min(gpu.len())]);
    eprintln!("[debug] host[0..4]  = {:?}", &host[..4.min(host.len())]);
    eprintln!("[debug] max_diff    = {}", max_abs_diff(&gpu, &host));
    Ok(())
}
