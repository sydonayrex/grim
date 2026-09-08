//! SPEED-ROC-8: parity tests for the LDS-tiled quant fused dequant GEMMs
//! (Q8_0, Q5_K, Q6_K, IQ2/IQ3/IQ4 families) against their scalar kernels,
//! forward and backward.
//!
//! GPU tests: `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm
//! --features "q5k q6k gpu-test-shims" --test quant_tiled_gemm`.
#![cfg(all(feature = "q5k", feature = "q6k", feature = "gpu-test-shims"))]

use grim_backend_rocm::device::q4k_test_shim;
use grim_backend_rocm::{RocmCachingAllocator, RocmDevice, RocmStorage};
use grim_tensor::dtype::{DType, KQuantScheme, Storage as DTypeStorage};
use grim_tensor::{ArithType, BackendStorage, Shape};
use std::sync::Arc;

fn gpu_enabled() -> bool {
    std::env::var("GRIM_RUN_GPU_TESTS").is_ok()
}

/// (format tag, KQuant scheme for the storage dtype, super-block elements,
/// bytes per block, scalar forward kernel entry)
const FORMATS: &[(&str, KQuantScheme, usize, usize, &str)] = &[
    (
        "q8_0",
        KQuantScheme::Q80,
        32,
        34,
        "grim_fused_dequant_gemm_q8_0",
    ),
    (
        "q5k",
        KQuantScheme::Q5K,
        256,
        176,
        "grim_fused_dequant_gemm_q5k",
    ),
    (
        "q6k",
        KQuantScheme::Q6K,
        256,
        210,
        "grim_fused_dequant_gemm_q6k",
    ),
    (
        "iq2xxs",
        KQuantScheme::IQ2XXS,
        256,
        66,
        "grim_fused_dequant_gemm_iq2xxs",
    ),
    (
        "iq2xs",
        KQuantScheme::IQ2XS,
        256,
        74,
        "grim_fused_dequant_gemm_iq2xs",
    ),
    (
        "iq2s",
        KQuantScheme::IQ2S,
        256,
        82,
        "grim_fused_dequant_gemm_iq2s",
    ),
    (
        "iq3xxs",
        KQuantScheme::IQ3XXS,
        256,
        96,
        "grim_fused_dequant_gemm_iq3xxs",
    ),
    (
        "iq3s",
        KQuantScheme::IQ3S,
        256,
        110,
        "grim_fused_dequant_gemm_iq3s",
    ),
    (
        "iq4nl",
        KQuantScheme::IQ4NL,
        256,
        170,
        "grim_fused_dequant_gemm_iq4nl",
    ),
    (
        "iq4xs",
        KQuantScheme::IQ4XS,
        256,
        136,
        "grim_fused_dequant_gemm_iq4xs",
    ),
];

struct Fixture {
    dev: RocmDevice,
    alloc: Arc<RocmCachingAllocator>,
}

fn fixture() -> Fixture {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    let alloc = Arc::new(RocmCachingAllocator::new(0, 4 << 30));
    Fixture { dev, alloc }
}

fn quant_dtype(scheme: KQuantScheme) -> DType {
    DType {
        arith: ArithType::F32,
        storage: DTypeStorage::KQuant(scheme),
    }
}

fn upload_f32(fx: &Fixture, data: &[f32], dims: &[usize]) -> RocmStorage {
    RocmStorage::copy_from_host(data, &Shape::new(dims.to_vec()), DType::F32, &fx.alloc, 0).unwrap()
}

/// Deterministic pseudo-random quant blob: n rows x (k / blk) blocks x bytes.
fn make_blob(n: usize, k: usize, blk: usize, bytes: usize) -> Vec<u8> {
    let row_bytes = (k / blk) * bytes;
    let mut blob = vec![0u8; n * row_bytes];
    for (i, b) in blob.iter_mut().enumerate() {
        *b = ((i * 31 + 7) % 251) as u8;
    }
    blob
}

fn sync_and_read(_fx: &Fixture, stream: *mut std::ffi::c_void, out: &RocmStorage) -> Vec<f32> {
    unsafe { grim_backend_rocm::hipStreamSynchronize(stream) };
    out.to_cpu_vec_f32().unwrap()
}

fn parity_forward(
    fx: &Fixture,
    tag: &str,
    scheme: KQuantScheme,
    blk: usize,
    bytes: usize,
    scalar_entry: &str,
    m: usize,
    n: usize,
    k: usize,
) -> f32 {
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.125 - 0.75)
        .collect();
    let blob = make_blob(n, k, blk, bytes);
    let a = upload_f32(fx, &a_data, &[m, k]);
    let b_dtype = quant_dtype(scheme);
    let b = RocmStorage::copy_from_host_raw_bytes(
        &blob,
        &Shape::new(vec![blob.len()]),
        b_dtype,
        &fx.alloc,
        0,
    )
    .unwrap();

    let run = |tiled: bool| {
        let out =
            RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &fx.alloc, 0).unwrap();
        let stream = if tiled {
            q4k_test_shim::launch_quant_tiled(
                &fx.dev,
                &format!("grim_fused_dequant_gemm_{tag}_tiled"),
                &a,
                &b,
                &out,
                m,
                n,
                k,
            )
        } else {
            q4k_test_shim::launch_quant_scalar(&fx.dev, scalar_entry, &a, &b, &out, m, n, k)
        }
        .unwrap();
        sync_and_read(fx, stream, &out)
    };

    let scalar = run(false);
    let tiled = run(true);
    let max_abs = scalar
        .iter()
        .zip(tiled.iter())
        .fold(0.0f32, |mx, (s, t)| mx.max((s - t).abs()));
    max_abs
}

fn parity_backward(
    fx: &Fixture,
    tag: &str,
    scheme: KQuantScheme,
    blk: usize,
    bytes: usize,
    scalar_entry: &str,
    m: usize,
    n: usize,
    k: usize,
) -> f32 {
    let dy_data: Vec<f32> = (0..m * n).map(|i| ((i % 11) as f32) * 0.2 - 1.0).collect();
    let blob = make_blob(n, k, blk, bytes);
    let dy = upload_f32(fx, &dy_data, &[m, n]);
    let b = RocmStorage::copy_from_host_raw_bytes(
        &blob,
        &Shape::new(vec![blob.len()]),
        quant_dtype(scheme),
        &fx.alloc,
        0,
    )
    .unwrap();
    let scalar_entry = scalar_entry.replace("dequant_gemm", "dequant_backward_gemm");

    let run = |tiled: bool| {
        let out =
            RocmStorage::alloc_gpu(&Shape::new(vec![m, k]), DType::F32, &fx.alloc, 0).unwrap();
        let stream = if tiled {
            q4k_test_shim::launch_quant_tiled_backward(
                &fx.dev,
                &format!("grim_fused_dequant_gemm_{tag}_backward_tiled"),
                &dy,
                &b,
                &out,
                m,
                n,
                k,
            )
        } else {
            q4k_test_shim::launch_quant_scalar_backward(
                &fx.dev,
                &scalar_entry,
                &dy,
                &b,
                &out,
                m,
                n,
                k,
            )
        }
        .unwrap();
        sync_and_read(fx, stream, &out)
    };

    let scalar = run(false);
    let tiled = run(true);
    let max_abs = scalar
        .iter()
        .zip(tiled.iter())
        .fold(0.0f32, |mx, (s, t)| mx.max((s - t).abs()));
    max_abs
}

#[test]
fn tiled_quant_parity_all_formats_forward() {
    if !gpu_enabled() {
        return;
    }
    let fx = fixture();
    // k = 256 satisfies every format's block geometry; guard shapes included
    // via n=100 (not a multiple of the 64-wide tile) and m=18.
    for (tag, scheme, blk, bytes, entry) in FORMATS {
        let d1 = parity_forward(&fx, tag, *scheme, *blk, *bytes, entry, 32, 128, 256);
        let d2 = parity_forward(&fx, tag, *scheme, *blk, *bytes, entry, 18, 100, 256);
        assert!(
            d1 <= 1e-3 && d2 <= 1e-3,
            "{tag} forward parity: max_abs_diff {d1}/{d2} exceeds 1e-3"
        );
    }
}

#[test]
fn tiled_quant_parity_all_formats_backward() {
    if !gpu_enabled() {
        return;
    }
    let fx = fixture();
    for (tag, scheme, blk, bytes, entry) in FORMATS {
        let d = parity_backward(&fx, tag, *scheme, *blk, *bytes, entry, 18, 100, 256);
        assert!(
            d <= 1e-3,
            "{tag} backward parity: max_abs_diff {d} exceeds 1e-3"
        );
    }
}

/// Q8_0 prefill throughput (the one format with a non-256 block geometry).
#[test]
fn tiled_q8_0_prefill_throughput() {
    if !gpu_enabled() {
        return;
    }
    let fx = fixture();
    let (m, n, k) = (512usize, 4096usize, 4096usize);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.125 - 0.75)
        .collect();
    let blob = make_blob(n, k, 32, 34);
    let a = upload_f32(&fx, &a_data, &[m, k]);
    let b = RocmStorage::copy_from_host_raw_bytes(
        &blob,
        &Shape::new(vec![blob.len()]),
        quant_dtype(KQuantScheme::Q80),
        &fx.alloc,
        0,
    )
    .unwrap();

    let run = |tiled: bool| {
        let out =
            RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &fx.alloc, 0).unwrap();
        let stream = if tiled {
            q4k_test_shim::launch_quant_tiled(
                &fx.dev,
                "grim_fused_dequant_gemm_q8_0_tiled",
                &a,
                &b,
                &out,
                m,
                n,
                k,
            )
        } else {
            q4k_test_shim::launch_quant_scalar(
                &fx.dev,
                "grim_fused_dequant_gemm_q8_0",
                &a,
                &b,
                &out,
                m,
                n,
                k,
            )
        }
        .unwrap();
        sync_and_read(&fx, stream, &out)
    };

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let bench = |tiled: bool| -> f64 {
        let _ = run(tiled); // warmup (JIT + allocator)
        let iters = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = run(tiled);
        }
        flops / (t0.elapsed().as_secs_f64() / iters as f64) / 1e9
    };

    let scalar_gflops = bench(false);
    let tiled_gflops = bench(true);
    println!(
        "q8_0 prefill m={m} n={n} k={k}: scalar {scalar_gflops:.1} GFLOP/s, \
         tiled {tiled_gflops:.1} GFLOP/s ({:.2}x)",
        tiled_gflops / scalar_gflops
    );
    assert!(
        tiled_gflops >= scalar_gflops * 0.95,
        "tiled q8_0 regressed vs scalar: {tiled_gflops:.1} vs {scalar_gflops:.1} GFLOP/s"
    );
}
