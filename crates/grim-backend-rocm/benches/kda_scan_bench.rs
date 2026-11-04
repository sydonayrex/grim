//! Kernel-level timing for `kda_gated_delta_rule_scan` at the Qwen3.5-9B
//! recurrent geometry (nv=32, nk=16, head_dim=128), seq 4056 — the shape the
//! 4K-prompt prefill drives 32 times (once per recurrent layer).
//!
//! RUN: GRIM_GPU_TEST=1 HIP_VISIBLE_DEVICES=1 cargo test -p grim-backend-rocm \
//!   --test kda_scan_bench -- --ignored --nocapture

use grim_backend_rocm::{RecurrentOps, RocmDevice};
use grim_tensor::{DType, Shape};

fn linspace(n: usize, lo: f64, hi: f64) -> Vec<f32> {
    (0..n)
        .map(|i| (lo + (hi - lo) * (i as f64) / (n.max(2) - 1) as f64) as f32)
        .collect()
}

#[test]
#[ignore]
fn kda_scan_bench_9b_geometry() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1");
        return;
    }
    let dev = RocmDevice::try_new(0).unwrap();
    let (nv, nk, d, seq_len) = (32usize, 16usize, 128usize, 4056usize);
    let key_dim = nk * d;
    let value_dim = nv * d;
    let conv_dim = 2 * key_dim + value_dim;

    let mut conv = Vec::with_capacity(seq_len * conv_dim);
    let mut alpha = Vec::with_capacity(seq_len * nv);
    let mut beta = Vec::with_capacity(seq_len * nv);
    let mut z = Vec::with_capacity(seq_len * value_dim);
    for t in 0..seq_len {
        let s = t as f64;
        conv.extend(linspace(conv_dim, -1.5 + 0.3 * s, 2.0 - 0.4 * s));
        alpha.extend(linspace(nv, -0.4 + 0.1 * s, 0.6));
        beta.extend(linspace(nv, -1.0, 1.0));
        z.extend(linspace(value_dim, -0.7 + 0.05 * s, 0.7));
    }
    let dt_bias = linspace(nv, 0.1, 0.9);
    let ssm_a = linspace(nv, 0.5, 1.5);
    let norm_w = linspace(d, 0.8, 1.2);
    let seed = linspace(nv * d * d, -0.3, 0.3);

    let up = |data: &[f32], shape: Shape| -> Box<dyn grim_tensor::BackendStorage> {
        grim_tensor::CoreTensorOps::from_cpu(&dev, data, &shape, DType::F32).unwrap()
    };
    let state = up(&seed, Shape::new(vec![nv, d, d]));
    let out_shape = Shape::new(vec![seq_len, value_dim]);

    let run = || {
        let (out, handle) = dev
            .kda_gated_delta_rule_scan(
                up(&conv, Shape::new(vec![seq_len, conv_dim])).as_ref(),
                up(&alpha, Shape::new(vec![seq_len, nv])).as_ref(),
                up(&beta, Shape::new(vec![seq_len, nv])).as_ref(),
                up(&dt_bias, Shape::new(vec![nv])).as_ref(),
                up(&ssm_a, Shape::new(vec![nv])).as_ref(),
                up(&norm_w, Shape::new(vec![d])).as_ref(),
                Some(up(&z, Shape::new(vec![seq_len, value_dim])).as_ref()),
                state.as_ref(),
                seq_len,
                nv,
                nk,
                d,
                1e-6,
                &out_shape,
            )
            .unwrap();
        handle.synchronize().unwrap();
        out
    };

    run(); // warmup + JIT
    let iters = 3;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = run();
    }
    eprintln!(
        "[kda-scan-bench] {:.1} ms/launch at seq={seq_len} nv={nv} nk={nk} d={d}",
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    );
}
