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
    for eps in [1e-5f32, 1e-6f32] {
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

        // Fused, with norm_out ALIASING residual (Residual 1 pattern):
        // the norm output buffer is the exact same allocation as the residual input.
        let sum_f = dev.from_cpu(&vec![0.0f32; n], &shape, DType::F32).unwrap();
        let norm_f = dev.from_cpu(&r, &shape, DType::F32).unwrap();
        let norm_f_r = as_rocm(norm_f.as_ref()).unwrap();
        dev.fused_add_rms_norm_into(
            x_s.as_ref(),
            norm_f.as_ref(),
            w_s.as_ref(),
            eps,
            as_rocm(sum_f.as_ref()).unwrap(),
            norm_f_r,
            &shape,
        )
        .unwrap();

        // Fused, with norm_out ALIASING sum_out (Residual 2 + output_norm pattern):
        let sum_norm_f = dev.from_cpu(&vec![0.0f32; n], &shape, DType::F32).unwrap();
        let sum_norm_r = as_rocm(sum_norm_f.as_ref()).unwrap();
        dev.fused_add_rms_norm_into(
            x_s.as_ref(),
            r_s.as_ref(),
            w_s.as_ref(),
            eps,
            sum_norm_r,
            sum_norm_r,
            &shape,
        )
        .unwrap();

        // Fused, with DISTINCT buffers (Residual 2 standard pattern):
        let sum_dist_f = dev.from_cpu(&vec![0.0f32; n], &shape, DType::F32).unwrap();
        let norm_dist_f = dev.from_cpu(&vec![0.0f32; n], &shape, DType::F32).unwrap();
        dev.fused_add_rms_norm_into(
            x_s.as_ref(),
            r_s.as_ref(),
            w_s.as_ref(),
            eps,
            as_rocm(sum_dist_f.as_ref()).unwrap(),
            as_rocm(norm_dist_f.as_ref()).unwrap(),
            &shape,
        )
        .unwrap();

        let got_sum = sum_f.to_cpu_vec_f32().unwrap();
        let want_sum = sum_ref.to_cpu_vec_f32().unwrap();
        let got_norm = norm_f.to_cpu_vec_f32().unwrap();
        let got_sum_norm = sum_norm_f.to_cpu_vec_f32().unwrap();
        let got_dist_norm = norm_dist_f.to_cpu_vec_f32().unwrap();
        let want_norm = norm_ref.to_cpu_vec_f32().unwrap();

        // BIT-exact (review requirement): the fused kernel reads the same fp32
        // sum values the split path stores, so any bit diff means a different
        // reduction order — a divergence the decode graph must not inherit.
        for i in 0..n {
            assert_eq!(
                got_sum[i].to_bits(),
                want_sum[i].to_bits(),
                "rows={rows} sum bit mismatch at {i}: {} vs {}",
                got_sum[i],
                want_sum[i]
            );
            assert_eq!(
                got_norm[i].to_bits(),
                want_norm[i].to_bits(),
                "rows={rows} norm (alias residual) bit mismatch at {i}: {} vs {}",
                got_norm[i],
                want_norm[i]
            );
            assert_eq!(
                got_sum_norm[i].to_bits(),
                want_norm[i].to_bits(),
                "rows={rows} norm (alias sum_out) bit mismatch at {i}: {} vs {}",
                got_sum_norm[i],
                want_norm[i]
            );
            assert_eq!(
                got_dist_norm[i].to_bits(),
                want_norm[i].to_bits(),
                "rows={rows} norm (distinct) bit mismatch at {i}: {} vs {}",
                got_dist_norm[i],
                want_norm[i]
            );
        }
    }
}

#[test]
#[ignore]
fn fused_add_rms_norm_matches_split_path() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    // LFM2 hidden=1024, Qwen3.5 27B hidden=5120, non-multiple-of-1024 decode row (2560),
    // multi-row warp path (4x1024), and odd row_len (3x100) to flush lane tails.
    for (rows, row_len) in [
        (1usize, 1024usize),
        (1, 2560),
        (1, 3072),
        (1, 5120),
        (4, 1024),
        (3, 100),
    ] {
        run_case(&dev, rows, row_len);
    }
}
