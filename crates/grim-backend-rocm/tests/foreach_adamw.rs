//! SPEED-ROC-10: parity + launch-count benchmark for the multi-tensor
//! (foreach) fused AdamW step against the per-tensor fused_adamw_step.
//!
//! GPU test: `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm
//! --features gpu-test-shims --test foreach_adamw -- --nocapture`.
#![cfg(feature = "gpu-test-shims")]

use grim_backend_rocm::{RocmCachingAllocator, RocmDevice, RocmStorage};
use grim_tensor::dtype::DType;
use grim_tensor::{BackendStorage, OptimizerOps, Shape};
use std::sync::Arc;

fn gpu_enabled() -> bool {
    std::env::var("GRIM_RUN_GPU_TESTS").is_ok()
}

const HYPER: (f32, f32, f32, f32, f32, f32, f32) =
    // lr, beta1, beta2, eps, weight_decay, bc1, bc2
    (1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0 / 0.1, 1.0 / 0.001);

struct Fixture {
    dev: RocmDevice,
    alloc: Arc<RocmCachingAllocator>,
}

fn fixture() -> Fixture {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    let alloc = Arc::new(RocmCachingAllocator::new(0, 2 << 30));
    Fixture { dev, alloc }
}

fn upload(fx: &Fixture, data: &[f32]) -> RocmStorage {
    RocmStorage::copy_from_host(
        data,
        &Shape::new(vec![data.len()]),
        DType::F32,
        &fx.alloc,
        0,
    )
    .unwrap()
}

/// Deliberately ragged sizes: small tensors are where per-tensor launch
/// overhead dominates.
const SIZES: &[usize] = &[7, 100, 333, 1024, 4096, 17];

fn make_state(
    fx: &Fixture,
    seed: f32,
) -> (
    Vec<RocmStorage>,
    Vec<RocmStorage>,
    Vec<RocmStorage>,
    Vec<RocmStorage>,
) {
    let mut ps = Vec::new();
    let mut gs = Vec::new();
    let mut ms = Vec::new();
    let mut vs = Vec::new();
    for (i, &len) in SIZES.iter().enumerate() {
        let p: Vec<f32> = (0..len)
            .map(|j| seed + (i * 31 + j % 13) as f32 * 0.01)
            .collect();
        let g: Vec<f32> = (0..len)
            .map(|j| ((i * 7 + j % 11) as f32) * 0.05 - 0.25)
            .collect();
        let m: Vec<f32> = vec![0.0; len];
        let v: Vec<f32> = vec![0.0; len];
        ps.push(upload(fx, &p));
        gs.push(upload(fx, &g));
        ms.push(upload(fx, &m));
        vs.push(upload(fx, &v));
    }
    (ps, gs, ms, vs)
}

#[test]
fn foreach_adamw_parity() {
    if !gpu_enabled() {
        return;
    }
    let fx = fixture();
    let (lr, b1, b2, eps, wd, bc1, bc2) = HYPER;

    // Arm A: per-tensor steps.
    let (ps_a, gs_a, ms_a, vs_a) = make_state(&fx, 1.0);
    for i in 0..SIZES.len() {
        let handle = fx
            .dev
            .fused_adamw_step(
                &ps_a[i], &gs_a[i], &ms_a[i], &vs_a[i], lr, b1, b2, eps, wd, bc1, bc2, SIZES[i],
            )
            .unwrap();
        handle.synchronize().unwrap();
    }

    // Arm B: one foreach step.
    let (ps_b, gs_b, ms_b, vs_b) = make_state(&fx, 1.0);
    let p_refs: Vec<&dyn BackendStorage> = ps_b.iter().map(|s| s as &dyn BackendStorage).collect();
    let g_refs: Vec<&dyn BackendStorage> = gs_b.iter().map(|s| s as &dyn BackendStorage).collect();
    let m_refs: Vec<&dyn BackendStorage> = ms_b.iter().map(|s| s as &dyn BackendStorage).collect();
    let v_refs: Vec<&dyn BackendStorage> = vs_b.iter().map(|s| s as &dyn BackendStorage).collect();
    let handle = fx
        .dev
        .fused_adamw_step_foreach(
            &p_refs, &g_refs, &m_refs, &v_refs, lr, b1, b2, eps, wd, bc1, bc2,
        )
        .unwrap();
    handle.synchronize().unwrap();

    // Identical fp op sequences per element -> results must match bit-for-bit.
    for i in 0..SIZES.len() {
        let a = ps_a[i].to_cpu_vec_f32().unwrap();
        let b = ps_b[i].to_cpu_vec_f32().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() <= 1e-9,
                "param tensor {i} diverged: {x} vs {y}"
            );
        }
        let ma = ms_a[i].to_cpu_vec_f32().unwrap();
        let mb = ms_b[i].to_cpu_vec_f32().unwrap();
        for (x, y) in ma.iter().zip(mb.iter()) {
            assert!(
                (x - y).abs() <= 1e-9,
                "momentum tensor {i} diverged: {x} vs {y}"
            );
        }
        let va = vs_a[i].to_cpu_vec_f32().unwrap();
        let vb = vs_b[i].to_cpu_vec_f32().unwrap();
        for (x, y) in va.iter().zip(vb.iter()) {
            assert!(
                (x - y).abs() <= 1e-9,
                "variance tensor {i} diverged: {x} vs {y}"
            );
        }
    }
}

#[test]
fn foreach_adamw_throughput() {
    if !gpu_enabled() {
        return;
    }
    let fx = fixture();
    let (lr, b1, b2, eps, wd, bc1, bc2) = HYPER;
    let (ps, gs, ms, vs) = make_state(&fx, 1.0);
    let p_refs: Vec<&dyn BackendStorage> = ps.iter().map(|s| s as &dyn BackendStorage).collect();
    let g_refs: Vec<&dyn BackendStorage> = gs.iter().map(|s| s as &dyn BackendStorage).collect();
    let m_refs: Vec<&dyn BackendStorage> = ms.iter().map(|s| s as &dyn BackendStorage).collect();
    let v_refs: Vec<&dyn BackendStorage> = vs.iter().map(|s| s as &dyn BackendStorage).collect();

    let iters = 200;

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        for i in 0..SIZES.len() {
            let handle = fx
                .dev
                .fused_adamw_step(
                    &ps[i], &gs[i], &ms[i], &vs[i], lr, b1, b2, eps, wd, bc1, bc2, SIZES[i],
                )
                .unwrap();
            let _ = handle;
        }
        fx.dev.synchronize();
    }
    let per_tensor_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let handle = fx
            .dev
            .fused_adamw_step_foreach(
                &p_refs, &g_refs, &m_refs, &v_refs, lr, b1, b2, eps, wd, bc1, bc2,
            )
            .unwrap();
        let _ = handle;
        fx.dev.synchronize();
    }
    let foreach_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    println!(
        "adamw step over {} tensors: per-tensor {per_tensor_us:.1} us, foreach {foreach_us:.1} us ({:.2}x)",
        SIZES.len(),
        per_tensor_us / foreach_us
    );
    assert!(
        foreach_us <= per_tensor_us,
        "foreach slower than per-tensor"
    );
}

fn main() {}
