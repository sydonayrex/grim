//! SPEED-DOT MoE perf decision bench: dot4/sudot4 grouped kernels vs the
//! scalar grouped kernels at decode shapes, per plan §1.4 — "kernel of
//! choice" is measured, not hardcoded. The winner per (shape, quant) is
//! registered into the `Autotuner` MoE table (`record_moe`) so the selector
//! reads a measured prior instead of a static preference.
//!
//! Wall-clock per roundtrip call includes H2D upload of the packed weights;
//! the upload cost is identical for both variants, so the *ratio* isolates
//! the kernel. RUN: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm \
//!   --test charon_dot4_perf -- --nocapture

use grim_backend_rocm::autotune::{AutotuneConfig, Autotuner, MoeKernelKey};
use grim_backend_rocm::kernels::charon::{
    dot4_supported, grouped_dot4_entry, CharonDot4Quant, RoutingAssignment,
};
use grim_backend_rocm::RocmDevice;
use std::panic;
use std::time::Instant;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

fn quant_q80(w: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; w.len() / 32 * 34];
    for (blk, chunk) in w.chunks(32).enumerate() {
        let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = (amax / 127.0).max(1e-30);
        let bits = half::f16::from_f32(d).to_bits();
        let o = &mut out[blk * 34..blk * 34 + 34];
        o[0] = (bits & 0xFF) as u8;
        o[1] = ((bits >> 8) & 0xFF) as u8;
        for (j, &v) in chunk.iter().enumerate() {
            o[2 + j] = ((v / d).round().clamp(-127.0, 127.0) as i8) as u8;
        }
    }
    out
}

fn bench_case(dev: &RocmDevice, batch: usize, hidden: usize, inter: usize, num_experts: usize) {
    let top_k = 2;
    let mut tokens = Vec::new();
    let mut experts = Vec::new();
    let mut weights = Vec::new();
    for t in 0..batch {
        for k in 0..top_k {
            tokens.push(t as u32);
            experts.push(((t + k) % num_experts) as u32);
            weights.push(0.5);
        }
    }
    let asg = RoutingAssignment { tokens, experts, weights };
    let a_scale = vec![1.0f32; batch];
    let x: Vec<f32> = (0..batch * hidden).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let w: Vec<f32> = (0..num_experts * hidden * inter)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.04)
        .collect();
    let gw_q = quant_q80(&w);

    let scalar_entry = "grim_moe_fused_grouped_q80";
    let dot4_entry = grouped_dot4_entry(CharonDot4Quant::Q8_0);

    let time_q80 = |scalar: bool, iters: usize| -> f32 {
        let mut best = f32::MAX;
        for _ in 0..iters {
            let t0 = Instant::now();
            let out = if scalar {
                dev.charon_grouped_dispatch_roundtrip_q80(
                    &x, &gw_q, &gw_q, &gw_q, &a_scale, &asg, batch, hidden, inter, 1.0,
                )
            } else {
                dev.charon_grouped_dispatch_roundtrip_dot4(
                    dot4_entry,
                    &x, &gw_q, &gw_q, &gw_q, &a_scale, &asg, batch, hidden, inter, num_experts,
                    1.0,
                )
            };
            assert!(out.is_ok(), "roundtrip failed for scalar={scalar}");
            let dt = t0.elapsed().as_secs_f32() * 1e3;
            best = best.min(dt);
        }
        best
    };

    // Correctness of the plan: run dot4 only where supported.
    let run_dot4 = dot4_supported(dev.gcn_arch());
    let scalar_ms = time_q80(true, 10);
    let dot4_ms = if run_dot4 { time_q80(false, 10) } else { f32::MAX };

    let arch = dev.gcn_arch().to_string();
    let winner = if dot4_ms < scalar_ms { dot4_entry } else { scalar_entry };
    println!(
        "dot4-perf batch={batch} hidden={hidden} inter={inter} experts={num_experts} \
         arch={arch}: scalar={scalar_ms:.3}ms dot4={dot4_ms:.3}ms winner={winner}"
    );

    // Register the measured winner as the autotune prior for this shape class
    // (existing Autotuner machinery; cycles are relative, winner-isolated).
    let arch_static: &'static str = Box::leak(arch.clone().into_boxed_str());
    let mut tuner = Autotuner::for_device(0, arch_static);
    let key = MoeKernelKey {
        kernel: winner.to_string(),
        gpu_arch: arch.clone(),
        hidden,
        inter,
        num_experts,
        top_k,
        skew_bucket: grim_backend_rocm::autotune::quantize_routing_skew(0.0),
    };
    let cfg = AutotuneConfig {
        block_dim: 64,
        tile_kv: 64,
        grid_stride: 1,
        cycles_per_invocation: if dot4_ms < scalar_ms {
            (dot4_ms * 1e6) as u64
        } else {
            (scalar_ms * 1e6) as u64
        },
        spec_gamma: 4,
        spec_acceptance_threshold: 0.6,
        spec_alpha: 0.0,
        split_k: 0,
    };
    tuner.record_moe(key, cfg).expect("record_moe dot4 winner");
    assert!(
        !tuner.list_moe_keys().is_empty(),
        "dot4 winner must be registered in the autotune table"
    );
}

#[test]
fn dot4_decode_perf_and_autotune_registration() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    if !dot4_supported(dev.gcn_arch()) {
        eprintln!("[SKIP] {} has no dot4 path; scalar stays kernel of choice", dev.gcn_arch());
        return;
    }
    // Decode-class shapes (small batch, multiples of 32). Kept small: the
    // scalar grouped kernel is one-thread-per-token and fully serial, so
    // large hidden would take minutes per launch — that asymmetry is itself
    // the finding the dot4 kernels exist to fix.
    bench_case(&dev, 1, 256, 256, 8);
    bench_case(&dev, 4, 256, 256, 8);
    bench_case(&dev, 1, 512, 512, 8);
}
