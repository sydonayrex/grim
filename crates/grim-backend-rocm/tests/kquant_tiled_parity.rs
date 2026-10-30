//! M7.2a — parity for the Q3_K and Q2_K LDS-tiled GEMMs.
//!
//! The IQ recipe (the LDS-tiled prompt GEMM) was landed for Q8_0, Q5_K, Q6_K
//! and the IQ families. Q3_K and Q2_K never got it, and the omission is
//! measurable: at the 27B's dominant projection shape (n=17408, k=5120),
//! Q3_K at m=64 measured 7.81 ms (4.9 GB/s) and Q2_K 6.90 ms (4.2 GB/s),
//! where the tiled formats reach ~1.25 ms and the IQ formats ~0.5 ms.
//!
//! This gate is the same shape as `tests/quant_tiled_gemm.rs` — the tiled
//! kernel against the SCALAR kernel that ships today, on the same bytes — but
//! it is a separate file for two reasons:
//!
//!  1. The shared gate's Q5_K entry fails on a PRE-EXISTING blob-geometry
//!     bug (its `make_blob` writes 176 B per 256-element block, but the
//!     storage's `expected_bytes` says 15488 for the shape it uploads), so
//!     the shared gate cannot reach the new formats without that fixed
//!     first. That bug is not mine to fix silently inside this change.
//!  2. The new formats' parity deserves its OWN gate, so a failure names
//!     the format that broke.
//!
//! The bar is EQUALITY against the scalar oracle, not a tolerance: a
//! quantized GEMV's rounding is exactly the kind of difference that hides
//! until the logits move, so a "close enough" gate would not catch the
//! failure mode that matters. (The shared gate uses 1e-3; this one demands
//! bit-equality because the tiled kernel and the scalar kernel compute the
//! same values in the same order, so ANY difference is a bug.)
//!
//! GPU tests: `HIP_VISIBLE_DEVICES=1 GRIM_RUN_GPU_TESTS=1 cargo test -p
//! grim-backend-rocm --features "gpu-test-shims" --test kquant_tiled_parity --
//! -- --ignored --test-threads=1`

#![cfg(all(feature = "q3k", feature = "q2k", feature = "gpu-test-shims"))]

use grim_backend_rocm::device::q4k_test_shim;
use grim_backend_rocm::{RocmCachingAllocator, RocmDevice, RocmStorage};
use grim_tensor::dtype::{DType, KQuantScheme, Storage as DTypeStorage};
use grim_tensor::{ArithType, BackendStorage, Shape};
use std::sync::Arc;

/// (format tag, KQuant scheme, super-block elements, bytes per block, the
/// SCALAR kernel that ships today).
///
/// The byte geometry mirrors `grim_quant`'s own layouts and the dispatch
/// table in `device/compute/gemm_launchers.rs`: Q3_K is 110 B per 256
/// elements, Q2_K is 84 B per 256.
const FORMATS: &[(&str, KQuantScheme, usize, usize, &str)] = &[
    (
        "q3k",
        KQuantScheme::Q3K,
        256,
        110,
        "grim_fused_dequant_gemm_q3k",
    ),
    (
        "q2k",
        KQuantScheme::Q2K,
        256,
        84,
        "grim_fused_dequant_gemm_q2k",
    ),
];

fn gpu_enabled() -> bool {
    std::env::var("GRIM_RUN_GPU_TESTS").is_ok()
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
///
/// The bytes are a deterministic NON-UNIFORM pattern, so a kernel that reads
/// the wrong offsets produces a DIFFERENT answer rather than a plausible one.
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

/// Run the scalar and the tiled kernel on the same bytes and return
/// (max_abs_diff, the first index that differs).
fn compare(
    fx: &Fixture,
    tag: &str,
    scheme: KQuantScheme,
    blk: usize,
    bytes: usize,
    scalar_entry: &str,
    m: usize,
    n: usize,
    k: usize,
) -> (f32, usize) {
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.125 - 0.75)
        .collect();
    let blob = make_blob(n, k, blk, bytes);
    let a = upload_f32(fx, &a_data, &[m, k]);
    let b = RocmStorage::copy_from_host_raw_bytes(
        &blob,
        &Shape::new(vec![blob.len()]),
        quant_dtype(scheme),
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
    let mut worst = 0.0f32;
    let mut worst_at = usize::MAX;
    for (i, (s, t)) in scalar.iter().zip(tiled.iter()).enumerate() {
        let d = (s - t).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    (worst, worst_at)
}

/// The forward parity: the tiled kernel must equal the scalar kernel BIT for
/// BIT, on the shipping bytes, at both a skinny shape and a prompt shape.
#[test]
#[ignore]
fn kquant_tiled_matches_the_scalar_kernel_exactly() {
    if !gpu_enabled() {
        return;
    }
    let fx = fixture();
    for (tag, scheme, blk, bytes, entry) in FORMATS {
        // A skinny shape (m=2: the tiled kernel's row-reuse factor is 4, so
        // m=2 exercises the partial-tile path) and a prompt shape (m=16: a
        // full tile).
        for (m, n, k) in [(2usize, 128usize, 256usize), (16, 512, 5120)] {
            let (d, at) = compare(&fx, tag, *scheme, *blk, *bytes, entry, m, n, k);
            assert_eq!(
                d, 0.0,
                "{tag} tiled vs scalar at m={m} n={n} k={k}: max_abs_diff {d} at {at} — the \\
                 tiled kernel must be BIT-IDENTICAL to the scalar kernel that ships today"
            );
        }
    }
}

/// The backward parity, same bar.
///
/// The backward kernel is the one the training path uses, so a difference
/// here is a difference in the GRADIENT, which is harder to notice than a
/// difference in the forward answer.
#[test]
#[ignore]
fn kquant_tiled_backward_matches_the_scalar_kernel_exactly() {
    if !gpu_enabled() {
        return;
    }
    let fx = fixture();
    for (tag, scheme, blk, bytes, _entry) in FORMATS {
        let (d, at) = compare_backward(&fx, tag, *scheme, *blk, *bytes, 2, 128, 256);
        assert_eq!(
            d, 0.0,
            "{tag} tiled backward vs scalar: max_abs_diff {d} at {at} — the tiled \\
             backward must be BIT-IDENTICAL to the scalar backward that ships today"
        );
    }
}

fn compare_backward(
    fx: &Fixture,
    tag: &str,
    scheme: KQuantScheme,
    blk: usize,
    bytes: usize,
    m: usize,
    n: usize,
    k: usize,
) -> (f32, usize) {
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

    let run = |tiled: bool| {
        let out =
            RocmStorage::alloc_gpu(&Shape::new(vec![m, k]), DType::F32, &fx.alloc, 0).unwrap();
        let stream = if tiled {
            q4k_test_shim::launch_quant_tiled_backward(
                &fx.dev,
                // The macro emits the backward as
                // `grim_fused_dequant_gemm_<fmt>_backward_tiled`; the SCALAR
                // backward is `grim_fused_dequant_backward_gemm_<fmt>`. The
                // two families name the segment differently, and the shared
                // gate's `replace("dequant_gemm", "dequant_backward_gemm")`
                // is what bridges them for the scalar side.
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
                // The backward entry is the forward name with `backward`
                // spliced in: `grim_fused_dequant_gemm_q3k` ->
                // `grim_fused_dequant_backward_gemm_q3k`.
                &format!("grim_fused_dequant_backward_gemm_{tag}"),
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
    let mut worst = 0.0f32;
    let mut worst_at = usize::MAX;
    for (i, (s, t)) in scalar.iter().zip(tiled.iter()).enumerate() {
        let d = (s - t).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    (worst, worst_at)
}
