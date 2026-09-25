//! PLAN-decode-throughput-restore Fix 3 regression: `fused_add_rms_norm_into`
//! (one `grim_add_rms_norm` launch) must match the split path
//! (`add_into` + `rms_norm_into`) on the same inputs, and must tolerate
//! `norm_out` aliasing `residual` — the decode-graph capture relies on both.

use grim_backend_rocm::{RocmDevice, as_rocm, gpu_test_enabled};
use grim_tensor::{CoreTensorOps, DType, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (seed + i as f32 * 0.61).sin() * 0.5)
        .collect()
}

fn run_case(dev: &RocmDevice, rows: usize, row_len: usize) {
    let n = rows * row_len;
    let shape = Shape::new(vec![rows, row_len]);
    let x = fill(n, 0.1);
    let r = fill(n, 2.7);
    let w: Vec<f32> = (0..row_len).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
    let eps = 1e-5f32;

    // Split reference: add_into then rms_norm_into (fresh output buffers).
    let x_s = dev.from_cpu(&x, &shape, DType::F32).unwrap();
    let r_s = dev.from_cpu(&r, &shape, DType::F32).unwrap();
    let w_s = dev
        .from_cpu(&w, &Shape::new(vec![row_len]), DType::F32)
        .unwrap();
    let sum_ref = dev.from_cpu(&vec![0.0f32; n], &shape, DType::F32).unwrap();
    let norm_ref = dev.from_cpu(&vec![0.0f32; n], &shape, DType::F32).unwrap();
    let sum_ref_r = as_rocm(sum_ref.as_ref()).unwrap();
    let norm_ref_r = as_rocm(norm_ref.as_ref()).unwrap();
    dev.add_into(x_s.as_ref(), r_s.as_ref(), sum_ref_r).unwrap();
    dev.rms_norm_into(sum_ref.as_ref(), w_s.as_ref(), eps, norm_ref_r, &shape)
        .unwrap();

    // Fused, with norm_out ALIASING residual (decode-graph capture pattern):
    // the norm output buffer starts as a copy of the residual input.
    let sum_f = dev.from_cpu(&vec![0.0f32; n], &shape, DType::F32).unwrap();
    let norm_f = dev.from_cpu(&r, &shape, DType::F32).unwrap();
    dev.fused_add_rms_norm_into(
        x_s.as_ref(),
        r_s.as_ref(),
        w_s.as_ref(),
        eps,
        as_rocm(sum_f.as_ref()).unwrap(),
        as_rocm(norm_f.as_ref()).unwrap(),
        &shape,
    )
    .unwrap();

    let got_sum = sum_f.to_cpu_vec_f32().unwrap();
    let want_sum = sum_ref.to_cpu_vec_f32().unwrap();
    let got_norm = norm_f.to_cpu_vec_f32().unwrap();
    let want_norm = norm_ref.to_cpu_vec_f32().unwrap();

    for i in 0..n {
        assert!(
            (got_sum[i] - want_sum[i]).abs() <= 1e-5,
            "rows={rows} sum mismatch at {i}: {} vs {}",
            got_sum[i],
            want_sum[i]
        );
        assert!(
            (got_norm[i] - want_norm[i]).abs() <= 1e-4,
            "rows={rows} norm mismatch at {i}: {} vs {}",
            got_norm[i],
            want_norm[i]
        );
    }
}

#[test]
#[ignore]
fn fused_add_rms_norm_matches_split_path() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    // LFM2 hidden=1024 (decode row), plus an odd row_len to flush lane tails.
    for (rows, row_len) in [(1usize, 1024usize), (4, 1024), (1, 3072), (3, 100)] {
        run_case(&dev, rows, row_len);
    }
}
