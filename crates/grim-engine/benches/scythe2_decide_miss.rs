//! WI-INF4: SCYTHE-2 placement-decision overhead sweep.
//!
//! Measures the host-side cost of `C2plrController::decide` across realistic
//! layer counts and prefill/decode shapes:
//! - cache-miss path (`decide_miss`: WaveTune bilinear eval + MLP + Gumbel),
//!   measured at ~2 µs/layer (SB3 campaign, release host-side; the ~10 µs
//!   figure in scythe2.md §3.4 was the pre-implementation estimate);
//! - cache-hit path (decode steady state), measured at ~50 ns/layer.
//!
//! These are host-side numbers — pure Rust, no GPU needed — so they bound the
//! CPU overhead the decode/prefill hot path pays for routing. The end-to-end
//! A/B has since RUN on syd-beasty (WI-INF4 verdict, 2026-08-23c): mean TTFT
//! overhead −0.09 %/−0.00 % (F/S) and p95 ITL overhead −18.56 %/+2.43 % —
//! the S-first ITL tail exceeds the 2 % budget, so `GRIM_SCYTHE_INFERENCE`
//! stays opt-in and the placement cost model is retuned against that tail.
//! Re-run this bench after placement changes that target the ITL path.

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

    // Retuned CI gates (SB3): the measured budgets are ~2 µs miss / ~50 ns
    // hit in release. We fail only on a >5× overrun of the measured claim
    // (the absolute numbers are hardware-dependent; the end-to-end gate is
    // the WI-INF4 A/B, already run — verdict recorded 2026-08-23c).
    let mut failed = false;
    if worst_miss_us > 10.0 {
        eprintln!(
            "FAIL: worst decide_miss {worst_miss_us:.1} µs/layer exceeds the 10 µs CI bound \
             (>5× the measured ~2 µs/layer budget)"
        );
        failed = true;
    }
    if worst_hit_ns > 250.0 {
        eprintln!(
            "FAIL: worst cache-hit {worst_hit_ns:.1} ns/layer exceeds the 250 ns CI bound \
             (>5× the measured ~50 ns/layer)"
        );
        failed = true;
    }
    if !failed {
        println!(
            "PASS: worst miss {worst_miss_us:.2} µs/layer (<10), worst hit {worst_hit_ns:.1} ns/layer (<250)."
        );
        println!(
            "NOTE: host-side only. WI-INF4 end-to-end verdict (2026-08-23c): STAYS OPT-IN — \
             S-first p95 ITL +2.43% exceeds the 2% budget."
        );
    }
    std::process::exit(if failed { 1 } else { 0 });
}
