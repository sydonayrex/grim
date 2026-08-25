//! WI-SB6 benchmark: ring-vs-direct decode GEMM latency.
//!
//! Times `RocmDevice::matmul` (rocBLAS direct path) against the
//! `GRIM_SCYTHE_RING=1` production routing (ScytheRing persistent dispatch
//! wave) over representative decode-shape dense-layer GEMMs, validates
//! parity per shape, and emits one JSONL row per shape on stdout.
//!
//! Run: `cargo run --release --example ring_vs_direct_decode [-- ordinal iters]`
//! (release build; the debug build's numbers are not meaningful)

use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::BackendDevice;
use grim_tensor::{DType, Shape};

const DECODE_SHAPES: &[(usize, usize, usize)] = &[
    // (m, n, k) — dense-layer projections at decode batch sizes.
    (1, 576, 576),
    (1, 1536, 576),
    (1, 576, 1536),
    (1, 4096, 4096),
    (1, 12288, 4096),
    (1, 4096, 12288),
    (4, 4096, 4096),
];

fn time_ops<F: FnMut() -> Box<dyn grim_tensor::backend::ComputeHandle>>(
    iters: usize,
    mut op: F,
) -> f64 {
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let handle = op();
        handle.synchronize().expect("op sync");
    }
    t0.elapsed().as_secs_f64() / iters as f64 * 1e6 // µs/op
}

fn main() {
    let ordinal: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    const WARMUP: usize = 5;

    let dev = match RocmDevice::try_new(ordinal) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no ROCm device {ordinal}: {e:?}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "[bench] ordinal {ordinal}, {} iters after {WARMUP} warmup",
        iters
    );

    for &(m, n, k) in DECODE_SHAPES {
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 * 0.11) - 0.9).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 23) as f32 * 0.07) - 0.75).collect();
        let a = dev
            .from_cpu(&a_data, &Shape::from_slice(&[m, k]), DType::F32)
            .expect("a");
        let b = dev
            .from_cpu(&b_data, &Shape::from_slice(&[k, n]), DType::F32)
            .expect("b");
        let out_shape = Shape::from_slice(&[m, n]);

        // Warm both paths.
        unsafe { std::env::remove_var("GRIM_SCYTHE_RING") };
        for _ in 0..WARMUP {
            let (o, h) = dev.matmul(a.as_ref(), b.as_ref(), &out_shape).expect("mm");
            h.synchronize().expect("sync");
            drop(o);
        }
        unsafe { std::env::set_var("GRIM_SCYTHE_RING", "1") };
        for _ in 0..WARMUP {
            let (o, h) = dev.matmul(a.as_ref(), b.as_ref(), &out_shape).expect("mm");
            h.synchronize().expect("sync");
            drop(o);
        }

        // Timed: direct (flag off).
        unsafe { std::env::remove_var("GRIM_SCYTHE_RING") };
        let direct_us = time_ops(iters, || dev.matmul(a.as_ref(), b.as_ref(), &out_shape).expect("mm").1);

        // Timed: ring-routed (flag on).
        unsafe { std::env::set_var("GRIM_SCYTHE_RING", "1") };
        let ring_us = time_ops(iters, || dev.matmul(a.as_ref(), b.as_ref(), &out_shape).expect("mm").1);
        let (routed, h) = dev.matmul(a.as_ref(), b.as_ref(), &out_shape).expect("mm");
        h.synchronize().expect("sync");
        unsafe { std::env::remove_var("GRIM_SCYTHE_RING") };

        // Parity vs the direct result.
        unsafe { std::env::remove_var("GRIM_SCYTHE_RING") };
        let (direct_ref, h) = dev.matmul(a.as_ref(), b.as_ref(), &out_shape).expect("mm");
        h.synchronize().expect("sync");
        let d_ref = direct_ref.to_cpu_vec_f32().expect("direct readback");
        let d_ring = routed.to_cpu_vec_f32().expect("ring readback");
        let max_diff = d_ref
            .iter()
            .zip(d_ring.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);

        let ratio = ring_us / direct_us.max(1e-9);
        eprintln!(
            "[bench] m={m} n={n} k={k}: direct={direct_us:9.2}µs ring={ring_us:9.2}µs ratio={ratio:7.2}x max_abs_diff={max_diff:.3e}"
        );
        println!(
            "{{\"bench\":\"ring_vs_direct_decode\",\"m\":{m},\"n\":{n},\"k\":{k},\
             \"direct_us\":{direct_us:.3},\"ring_us\":{ring_us:.3},\"ratio\":{ratio:.4},\
             \"max_abs_diff\":{max_diff:.3e},\"iters\":{iters},\"ordinal\":{ordinal}}}"
        );
    }
}
