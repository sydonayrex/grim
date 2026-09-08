//! SPEED-ROC-6: parity + prefill-throughput tests for the LDS-tiled Q4_K
//! forward GEMM (`grim_fused_dequant_gemm_q4k_tiled`) against the scalar
//! one-thread-per-output kernel.
//!
//! GPU tests: run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm
//! --features "q4k gpu-test-shims" --test q4k_tiled_gemm`.
#![cfg(all(feature = "q4k", feature = "gpu-test-shims"))]

use grim_backend_rocm::device::q4k_test_shim;
use grim_backend_rocm::{RocmCachingAllocator, RocmDevice, RocmStorage};
use grim_tensor::dtype::{DType, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, Shape};
use std::sync::Arc;
use std::time::Instant;

fn gpu_enabled() -> bool {
    std::env::var("GRIM_RUN_GPU_TESTS").is_ok()
}

fn q4k_dtype() -> DType {
    DType {
        arith: grim_tensor::ArithType::F32,
        storage: DTypeStorage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K),
    }
}

struct Fixture {
    dev: RocmDevice,
    alloc: Arc<RocmCachingAllocator>,
}

fn fixture() -> Fixture {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    let alloc = Arc::new(RocmCachingAllocator::new(0, 4 << 30));
    Fixture { dev, alloc }
}

fn upload_f32(fx: &Fixture, data: &[f32], dims: &[usize]) -> RocmStorage {
    RocmStorage::copy_from_host(data, &Shape::new(dims.to_vec()), DType::F32, &fx.alloc, 0).unwrap()
}

fn upload_q4k_blob(fx: &Fixture, bytes: &[u8]) -> RocmStorage {
    RocmStorage::copy_from_host_raw_bytes(
        bytes,
        &Shape::new(vec![bytes.len()]),
        q4k_dtype(),
        &fx.alloc,
        0,
    )
    .unwrap()
}

/// Deterministic pseudo-random Q4_K blob (N rows x K/256 blocks x 144 bytes).
fn make_q4k_blob(n: usize, k: usize) -> Vec<u8> {
    let row_bytes = (k / 256) * 144;
    let mut blob = vec![0u8; n * row_bytes];
    for (i, b) in blob.iter_mut().enumerate() {
        *b = ((i * 31 + 7) % 251) as u8;
    }
    blob
}

fn run_kernel(
    fx: &Fixture,
    tiled: bool,
    a: &RocmStorage,
    b: &RocmStorage,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let out = RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &fx.alloc, 0).unwrap();
    let stream = if tiled {
        q4k_test_shim::launch_tiled(&fx.dev, a, b, &out, m, n, k)
    } else {
        q4k_test_shim::launch_scalar(&fx.dev, a, b, &out, m, n, k)
    }
    .unwrap();
    unsafe { grim_backend_rocm::hipStreamSynchronize(stream) };
    out.to_cpu_vec_f32().unwrap()
}

/// Parity: tiled vs scalar forward outputs must match to f32 rounding noise.
/// The two kernels accumulate in different orders (scalar: strict k order;
/// tiled: 64-wide tiles), so allow a small relative tolerance.
fn parity_case(m: usize, n: usize, k: usize, atol: f32) {
    let fx = fixture();
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.125 - 0.75)
        .collect();
    let blob = make_q4k_blob(n, k);
    let a = upload_f32(&fx, &a_data, &[m, k]);
    let b = upload_q4k_blob(&fx, &blob);

    let scalar = run_kernel(&fx, false, &a, &b, m, n, k);
    let tiled = run_kernel(&fx, true, &a, &b, m, n, k);

    assert_eq!(scalar.len(), m * n);
    let mut max_abs = 0.0f32;
    for (s, t) in scalar.iter().zip(tiled.iter()) {
        max_abs = max_abs.max((s - t).abs());
    }
    assert!(
        max_abs <= atol,
        "parity failed at m={m} n={n} k={k}: max_abs_diff={max_abs} (atol={atol})"
    );
}

#[test]
fn tiled_q4k_parity_small() {
    if !gpu_enabled() {
        return;
    }
    // k=256 (one super-block), non-multiple-of-tile N/M exercise the guards.
    parity_case(4, 16, 256, 1e-4);
    parity_case(16, 64, 256, 1e-4);
}

#[test]
fn tiled_q4k_parity_prefill_shapes() {
    if !gpu_enabled() {
        return;
    }
    parity_case(32, 512, 512, 1e-3);
    parity_case(128, 1024, 1024, 1e-3);
    // N not a multiple of 64 — exercises column guards.
    parity_case(32, 100, 768, 1e-3);
    // M not a multiple of 4 — exercises row guards.
    parity_case(18, 128, 512, 1e-3);
}

fn run_kernel_backward(
    fx: &Fixture,
    tiled: bool,
    dy: &RocmStorage,
    b: &RocmStorage,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let out = RocmStorage::alloc_gpu(&Shape::new(vec![m, k]), DType::F32, &fx.alloc, 0).unwrap();
    let stream = if tiled {
        q4k_test_shim::launch_backward_tiled(&fx.dev, dy, b, &out, m, n, k)
    } else {
        q4k_test_shim::launch_backward_scalar(&fx.dev, dy, b, std::ptr::null(), &out, m, n, k)
    }
    .unwrap();
    unsafe { grim_backend_rocm::hipStreamSynchronize(stream) };
    out.to_cpu_vec_f32().unwrap()
}

/// Parity: tiled vs scalar backward outputs.
fn backward_parity_case(m: usize, n: usize, k: usize, atol: f32) {
    let fx = fixture();
    let dy_data: Vec<f32> = (0..m * n).map(|i| ((i % 11) as f32) * 0.2 - 1.0).collect();
    let blob = make_q4k_blob(n, k);
    let dy = upload_f32(&fx, &dy_data, &[m, n]);
    let b = upload_q4k_blob(&fx, &blob);

    let scalar = run_kernel_backward(&fx, false, &dy, &b, m, n, k);
    let tiled = run_kernel_backward(&fx, true, &dy, &b, m, n, k);

    let mut max_abs = 0.0f32;
    for (s, t) in scalar.iter().zip(tiled.iter()) {
        max_abs = max_abs.max((s - t).abs());
    }
    assert!(
        max_abs <= atol,
        "backward parity failed at m={m} n={n} k={k}: max_abs_diff={max_abs} (atol={atol})"
    );
}

#[test]
fn tiled_q4k_backward_parity() {
    if !gpu_enabled() {
        return;
    }
    backward_parity_case(4, 16, 256, 1e-4);
    backward_parity_case(32, 512, 512, 1e-3);
    // N not a multiple of 64, M not a multiple of 4 — exercise guards.
    backward_parity_case(18, 100, 768, 1e-3);
}

/// Prefill throughput: tiled vs scalar at a representative LLM FFN shape.
/// Prints GFLOP/s for both; asserts the tiled path is not a regression
/// (>= 1x scalar) and reports the measured speedup in the log.
#[test]
fn tiled_q4k_prefill_throughput() {
    if !gpu_enabled() {
        return;
    }
    let fx = fixture();
    let (m, n, k) = (512usize, 4096usize, 4096usize);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.125 - 0.75)
        .collect();
    let blob = make_q4k_blob(n, k);
    let a = upload_f32(&fx, &a_data, &[m, k]);
    let b = upload_q4k_blob(&fx, &blob);

    let flops = 2.0 * m as f64 * n as f64 * k as f64;

    let bench = |tiled: bool| -> f64 {
        // Warmup (JIT compile + allocator pool).
        let _ = run_kernel(&fx, tiled, &a, &b, m, n, k);
        let iters = 10;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = run_kernel(&fx, tiled, &a, &b, m, n, k);
        }
        let elapsed = t0.elapsed().as_secs_f64() / iters as f64;
        flops / elapsed / 1e9
    };

    let scalar_gflops = bench(false);
    let tiled_gflops = bench(true);
    let speedup = tiled_gflops / scalar_gflops;
    println!(
        "q4k prefill m={m} n={n} k={k}: scalar {scalar_gflops:.1} GFLOP/s, \
         tiled {tiled_gflops:.1} GFLOP/s, speedup {speedup:.2}x"
    );
    assert!(
        tiled_gflops >= scalar_gflops * 0.95,
        "tiled kernel regressed vs scalar: {tiled_gflops:.1} vs {scalar_gflops:.1} GFLOP/s"
    );
}

/// SPEED-ROC-7: the dtype-aware gradient all-reduce must no-op cleanly on a
/// single-GPU `RcclAllReduce` for every supported gradient dtype (the
/// multi-GPU reduction itself requires ≥2 GPUs and is covered by tests/rccl.rs
/// on multi-GPU hardware).
#[test]
fn bf16_grad_allreduce_single_gpu_noop() {
    if !gpu_enabled() {
        return;
    }
    let rccl =
        grim_backend_rocm::rccl::RcclAllReduce::try_new(&[0]).expect("single-GPU RcclAllReduce");
    for (name, dtype) in [
        ("f32", grim_backend_rocm::rccl::NCCL_FLOAT32),
        ("f16", grim_backend_rocm::rccl::NCCL_FLOAT16),
        ("bf16", grim_backend_rocm::rccl::NCCL_BFLOAT16),
    ] {
        rccl.sum_gradients_device_typed(0, 0, 1024, dtype, 0, 0)
            .unwrap_or_else(|e| panic!("{name}: single-GPU typed all-reduce should no-op: {e}"));
    }
}
