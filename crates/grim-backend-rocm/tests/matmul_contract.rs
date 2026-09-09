//! SPEED-ROC-16 contract test: matmul(a, b) MUST compute C = A @ B^T where
//! b is the natural weight (N, K). Verifies the GPU result against a CPU
//! reference for several shapes including non-square and the decode regime (M<=8).

#![cfg(feature = "gpu-test-shims")]

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, Shape};

fn gpu() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    Some(RocmDevice::try_new(0).expect("device"))
}

/// CPU reference: C = A @ B^T, A=[M,K], B=[N,K] -> C=[M,N].
fn cpu_ref(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn check(dev: &RocmDevice, m: usize, n: usize, k: usize) {
    let mut rng = (m * 1000 + n * 100 + k) as u64;
    let mut rnd = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((rng >> 33) as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    };
    let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rnd()).collect();

    let a_s = dev.from_cpu(&a, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let b_s = dev.from_cpu(&b, &Shape::new(vec![n, k]), DType::F32).unwrap();
    let (out, handle) = grim_tensor::CoreTensorOps::matmul(dev, a_s.as_ref(), b_s.as_ref(), &Shape::new(vec![m, n])).unwrap();
    handle.synchronize().unwrap();
    let got: Vec<f32> = out.to_cpu_vec_f32().unwrap();
    let want = cpu_ref(&a, &b, m, n, k);

    let mut max_err = 0.0f32;
    for i in 0..want.len() {
        let denom = want[i].abs().max(1e-6);
        max_err = max_err.max((got[i] - want[i]).abs() / denom);
    }
    assert!(
        max_err <= 1e-4,
        "matmul contract m={m} n={n} k={k}: max rel err {max_err:.3e} (want[:4]={:?} got[:4]={:?})",
        &want[..4.min(want.len())],
        &got[..4.min(got.len())],
    );
    eprintln!("matmul contract m={m} n={n} k={k}: OK (max rel err {max_err:.3e})");
}

#[test]
fn matmul_contract_parity() {
    let Some(dev) = gpu() else { return };
    // Decode regime (M<=8, the R3 target).
    check(&dev, 1, 4096, 4096);
    check(&dev, 8, 14336, 4096);
    // Non-square, prefill regime.
    check(&dev, 512, 4096, 4096);
    check(&dev, 32, 100, 768);
    // Square.
    check(&dev, 64, 64, 64);
}
