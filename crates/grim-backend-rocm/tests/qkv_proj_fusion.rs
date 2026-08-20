//! WI-F1 — Fused QKV projection GEMM parity tests.
//!
//! RED-first per the fusion-boundary plan: the reference is *computed* by
//! running the existing unfused path (three separate GEMM calls) on device,
//! never a hand-copied constant. `fused_qkv_proj` / `concat_qkv_weights` /
//! the launch-count hooks must fail to resolve until the implementation
//! exists.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)"))
        .ok()
}

/// Deterministic synthetic weight/activation filler.
fn fill(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let s = (seed + i as f32 * 0.37).sin();
            s * 0.25 - 0.05
        })
        .collect()
}

fn host_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f32;
            for d in 0..k {
                acc += a[r * k + d] * b[d * n + c];
            }
            out[r * n + c] = acc;
        }
    }
    out
}

#[test]
fn concat_qkv_weights_layout_matches_slices() {
    let hidden = 8;
    let q_dim = 12;
    let k_dim = 8;
    let v_dim = 4;
    let q_w = fill(hidden, q_dim, 0.1);
    let k_w = fill(hidden, k_dim, 0.2);
    let v_w = fill(hidden, v_dim, 0.3);
    let fused = grim_backend_rocm::concat_qkv_weights(&q_w, &k_w, &v_w, hidden)
        .expect("concat_qkv_weights should succeed for matching row counts");
    assert_eq!(fused.len(), hidden * (q_dim + k_dim + v_dim));
    for r in 0..hidden {
        let row = &fused[r * (q_dim + k_dim + v_dim)..(r + 1) * (q_dim + k_dim + v_dim)];
        assert_eq!(&row[..q_dim], &q_w[r * q_dim..(r + 1) * q_dim], "q slice row {r}");
        assert_eq!(&row[q_dim..q_dim + k_dim], &k_w[r * k_dim..(r + 1) * k_dim], "k slice row {r}");
        assert_eq!(
            &row[q_dim + k_dim..],
            &v_w[r * v_dim..(r + 1) * v_dim],
            "v slice row {r}"
        );
    }
    // Mismatched row counts must be rejected, not silently truncated.
    let bad = grim_backend_rocm::concat_qkv_weights(&q_w, &k_w[..hidden * 4 + 1], &v_w, hidden);
    assert!(bad.is_err(), "mismatched k_w rows must error");
}

#[test]
fn qkv_proj_fusion_matches_unfused() {
    let Some(dev) = gpu_device() else { eprintln!("skipping: GPU test gate off"); return };

    let tokens = 4;
    let hidden = 64;
    let q_dim = 64;
    let k_dim = 64;
    let v_dim = 64;
    let qkv_dim = q_dim + k_dim + v_dim;

    let x_data = fill(tokens, hidden, 0.7);
    let q_w = fill(hidden, q_dim, 0.1);
    let k_w = fill(hidden, k_dim, 0.2);
    let v_w = fill(hidden, v_dim, 0.3);

    let x_shape = Shape::from_slice(&[tokens, hidden]);
    let w_shape = |n| Shape::from_slice(&[hidden, n]);

    let x = dev.from_cpu(&x_data, &x_shape, DType::F32).unwrap();
    let q = dev.from_cpu(&q_w, &w_shape(q_dim), DType::F32).unwrap();
    let k = dev.from_cpu(&k_w, &w_shape(k_dim), DType::F32).unwrap();
    let v = dev.from_cpu(&v_w, &w_shape(v_dim), DType::F32).unwrap();

    // Reference: the existing unfused path — three separate GEMM launches.
    let (q_out, h) = BackendDevice::matmul(&dev, x.as_ref(), q.as_ref(), &Shape::from_slice(&[tokens, q_dim])).unwrap();
    h.synchronize().unwrap();
    let (k_out, h) = BackendDevice::matmul(&dev, x.as_ref(), k.as_ref(), &Shape::from_slice(&[tokens, k_dim])).unwrap();
    h.synchronize().unwrap();
    let (v_out, h) = BackendDevice::matmul(&dev, x.as_ref(), v.as_ref(), &Shape::from_slice(&[tokens, v_dim])).unwrap();
    h.synchronize().unwrap();
    let q_got = q_out.to_cpu_vec_f32().unwrap();
    let k_got = k_out.to_cpu_vec_f32().unwrap();
    let v_got = v_out.to_cpu_vec_f32().unwrap();

    // Fused path: load-time-concatenated weight, one GEMM.
    let fused_w = grim_backend_rocm::concat_qkv_weights(&q_w, &k_w, &v_w, hidden).unwrap();
    let fw = dev.from_cpu(&fused_w, &w_shape(qkv_dim), DType::F32).unwrap();
    let (fused_out, h) = dev
        .fused_qkv_proj(x.as_ref(), fw.as_ref(), &Shape::from_slice(&[tokens, qkv_dim]))
        .expect("fused_qkv_proj should launch");
    h.synchronize().unwrap();
    let fused_got = fused_out.to_cpu_vec_f32().unwrap();

    // Compare per-projection slices by offset — no data movement, just views.
    let tol = 2e-4f32;
    for t in 0..tokens {
        for (name, dim, base, unfused) in [
            ("q", q_dim, 0usize, &q_got),
            ("k", k_dim, q_dim, &k_got),
            ("v", v_dim, q_dim + k_dim, &v_got),
        ] {
            for c in 0..dim {
                let want = unfused[t * dim + c];
                let got = fused_got[t * qkv_dim + base + c];
                let denom = want.abs().max(got.abs()).max(1e-6);
                assert!(
                    ((got - want) / denom).abs() <= tol,
                    "{name} mismatch at token {t} col {c}: fused {got} vs unfused {want}"
                );
            }
        }
    }

    // Guard against a vacuous pass: unfused outputs must be non-trivial and
    // must differ from the raw host reference by only float-accumulation error.
    let host_ref = host_matmul(&x_data, &q_w, tokens, hidden, q_dim);
    for i in 0..host_ref.len() {
        let denom = host_ref[i].abs().max(q_got[i].abs()).max(1e-6);
        assert!(
            ((q_got[i] - host_ref[i]) / denom).abs() < 1e-3,
            "unfused reference disagrees with host math at {i}: {} vs {}",
            q_got[i],
            host_ref[i]
        );
    }
}

#[test]
fn qkv_proj_fused_uses_single_launch() {
    let Some(dev) = gpu_device() else { eprintln!("skipping: GPU test gate off"); return };

    let tokens = 2;
    let hidden = 32;
    let dim = 32;
    let qkv_dim = 3 * dim;

    let x_data = fill(tokens, hidden, 0.7);
    let w = fill(hidden, qkv_dim, 0.4);
    let x_shape = Shape::from_slice(&[tokens, hidden]);
    let x = dev.from_cpu(&x_data, &x_shape, DType::F32).unwrap();
    let fw = dev.from_cpu(&w, &Shape::from_slice(&[hidden, qkv_dim]), DType::F32).unwrap();

    // Unfused: three separate GEMM calls on column slices of the same weight —
    // each slice needs its own upload, so exercise it via three matmuls against
    // three separate weight tensors (the real pre-fusion call shape).
    let wq = dev.from_cpu(&fill(hidden, dim, 0.1), &Shape::from_slice(&[hidden, dim]), DType::F32).unwrap();
    let wk = dev.from_cpu(&fill(hidden, dim, 0.2), &Shape::from_slice(&[hidden, dim]), DType::F32).unwrap();
    let wv = dev.from_cpu(&fill(hidden, dim, 0.3), &Shape::from_slice(&[hidden, dim]), DType::F32).unwrap();
    dev.reset_launch_count();
    for wmat in [&wq, &wk, &wv] {
        let (_, h) = BackendDevice::matmul(&dev, x.as_ref(), wmat.as_ref(), &Shape::from_slice(&[tokens, dim])).unwrap();
        h.synchronize().unwrap();
    }
    let unfused_count = dev.launch_count();
    assert!(
        unfused_count >= 3,
        "launch counter must detect the 3 separate GEMM launches (got {unfused_count}); a counter that cannot fail proves nothing"
    );

    dev.reset_launch_count();
    let (out, h) = dev
        .fused_qkv_proj(x.as_ref(), fw.as_ref(), &Shape::from_slice(&[tokens, qkv_dim]))
        .unwrap();
    h.synchronize().unwrap();
    let fused_count = dev.launch_count();
    assert_eq!(
        fused_count, 1,
        "fused QKV projection must be exactly 1 launch (got {fused_count}; unfused was {unfused_count})"
    );
    assert_eq!(out.shape().elem_count(), tokens * qkv_dim);
}
