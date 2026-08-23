//! WI-INF4: SCYTHE-2 placement-decision overhead sweep.
//!
//! Measures the host-side cost of `C2plrController::decide` across realistic
//! layer counts and prefill/decode shapes:
//! - cache-miss path (`decide_miss`: WaveTune bilinear eval + MLP + Gumbel),
//!   budgeted at ~10 µs/layer in scythe2.md §3.4 / `scythe2.rs` doc comments;
//! - cache-hit path (decode steady state), budgeted at ~50 ns/layer.
//!
//! These are host-side numbers — pure Rust, no GPU needed — so they bound the
//! CPU overhead the decode/prefill hot path pays for routing. The end-to-end
//! TTFT comparison (route on vs off) still requires the real asymmetric pair:
//!
//! TODO(gpu-verify): run `cargo bench -p grim-engine --bench scythe2_decide_miss`
//! on syd-beasty (RX 9070 XT / RX 9060 XT) AND an end-to-end prefill TTFT
//! A/B there before this ships default-on (WI-INF4 gate). Until then
//! `GRIM_SCYTHE_INFERENCE` stays opt-in.

use std::time::Instant;

use grim_engine::scythe2::{C2plrController, bucketize};
use grim_tensor::backend::{GpuCapability, ScytheLink};

fn caps_asymmetric() -> Vec<GpuCapability> {
    // syd-beasty shape: slow card at rank 0, fast card at rank 1.
    vec![
        GpuCapability {
            tflops_fp16: 8.0,
            tflops_fp8: 0.0,
            hbm_bandwidth_gbps: 51.2,
            vram_free_bytes: 16 << 30,
            throttle_pct: 0.0,
            ordinal: 0,
        },
        GpuCapability {
            tflops_fp16: 80.0,
            tflops_fp8: 160.0,
            hbm_bandwidth_gbps: 960.0,
            vram_free_bytes: 16 << 30,
            throttle_pct: 0.0,
            ordinal: 1,
        },
    ]
}

fn links_full(num_gpus: usize) -> Vec<ScytheLink> {
    let mut v = vec![ScytheLink::Host; num_gpus * num_gpus];
    for i in 0..num_gpus {
        v[i * num_gpus + i] = ScytheLink::PeerDirect;
    }
    v
}

/// Aggregate µs/layer for one full-model decide sweep with a cold cache
/// (every call is a `decide_miss`).
fn measure_miss_sweep(num_layers: usize, shape: &[usize], iters: usize) -> f64 {
    let caps = caps_asymmetric();
    let links = links_full(caps.len());
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let mut ctrl = C2plrController::new(num_layers, caps.len(), 150.0);
        let start = Instant::now();
        for layer_id in 0..num_layers as u32 {
            ctrl.decide(layer_id, shape, &caps, &links, 0);
        }
        let us = start.elapsed().as_secs_f64() * 1e6;
        best = best.min(us);
    }
    best / num_layers as f64
}

/// ns/layer for the warm-cache decode path (same bucket, same epoch).
fn measure_hit_sweep(num_layers: usize, shape: &[usize], reps: usize) -> f64 {
    let caps = caps_asymmetric();
    let links = links_full(caps.len());
    let mut ctrl = C2plrController::new(num_layers, caps.len(), 10.0);
    for layer_id in 0..num_layers as u32 {
        ctrl.decide(layer_id, shape, &caps, &links, 0);
    }
    // Warm up caches/timers.
    for _ in 0..100 {
        for layer_id in 0..num_layers as u32 {
            ctrl.decide(layer_id, shape, &caps, &links, 0);
        }
    }
    let start = Instant::now();
    for _ in 0..reps {
        for layer_id in 0..num_layers as u32 {
            ctrl.decide(layer_id, shape, &caps, &links, 0);
        }
    }
    let ns = start.elapsed().as_nanos() as f64;
    ns / (reps * num_layers) as f64
}

fn main() {
    println!("== WI-INF4: SCYTHE-2 decide() overhead sweep (host-side, best-of-N) ==");
    println!(
        "{:<10} {:<22} {:>14} {:>14}",
        "layers", "shape", "miss µs/layer", "hit ns/layer"
    );

    let decode = [1usize, 1, 4096, 128];
    let prefill_2k = [1usize, 2048, 4096, 128];
    let prefill_8k = [1usize, 8192, 4096, 128];

    let mut worst_miss_us = 0.0f64;
    let mut worst_hit_ns = 0.0f64;
    for &num_layers in &[8usize, 32, 80] {
        for (name, shape) in [
            ("decode [1,1,4096,128]", &decode[..]),
            ("prefill [1,2048,...]", &prefill_2k[..]),
            ("prefill [1,8192,...]", &prefill_8k[..]),
        ] {
            let miss_us = measure_miss_sweep(num_layers, shape, 200);
            let hit_ns = measure_hit_sweep(num_layers, shape, 2_000);
            worst_miss_us = worst_miss_us.max(miss_us);
            worst_hit_ns = worst_hit_ns.max(hit_ns);
            println!(
                "{:<10} {:<22} {:>14.2} {:>14.1}",
                num_layers, name, miss_us, hit_ns
            );
            let _ = bucketize(shape); // keep import honest if shapes change
        }
    }

    // Generous CI gates — the claimed budgets are ~10 µs miss / ~50 ns hit.
    // We fail only on a >5× overrun of the miss claim (the absolute claim is
    // hardware-dependent; the end-to-end gate is WI-INF4's syd-beasty run).
    let mut failed = false;
    if worst_miss_us > 50.0 {
        eprintln!(
            "FAIL: worst decide_miss {worst_miss_us:.1} µs/layer exceeds the 50 µs CI bound \
             (>5× the claimed ~10 µs/layer budget)"
        );
        failed = true;
    }
    if worst_hit_ns > 250.0 {
        eprintln!(
            "FAIL: worst cache-hit {worst_hit_ns:.1} ns/layer exceeds the 250 ns CI bound \
             (>5× the claimed ~50 ns/layer)"
        );
        failed = true;
    }
    if !failed {
        println!(
            "PASS: worst miss {worst_miss_us:.2} µs/layer (<50), worst hit {worst_hit_ns:.1} ns/layer (<250)."
        );
        println!(
            "NOTE: host-side only. End-to-end prefill TTFT on/off A/B still pending on \
             syd-beasty — GRIM_SCYTHE_INFERENCE stays default-off until then (WI-INF4)."
        );
    }
    std::process::exit(if failed { 1 } else { 0 });
}
