//! Split-K matmul dtype-parity regression gate.
//!
//! `grim_split_k_reduction` was hard-typed `_Float16*`: every F32 matmul
//! that took the split-K path (active when `m > 1 || k > 8192`) had its
//! f32 partials read as half-precision pairs and f16 bits written back
//! into the f32 output buffer — silent garbage, caught by the WI-SB6
//! ring-vs-direct benchmark on 2026-08-25. The reduction now dispatches a
//! dtype-matched kernel; this gate pins F32 split-K shapes against a CPU
//! reference so the corruption class cannot return unnoticed.
//!
//! Device-gated: `GRIM_GPU_TEST=1`.

use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::{BackendDevice, BackendStorage};
use grim_tensor::{DType, Shape};

fn gpu_ready() -> bool {
    std::env::var("GRIM_GPU_TEST").as_deref() == Ok("1")
}

fn check_shape(dev: &RocmDevice, m: usize, n: usize, k: usize) -> f32 {
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 17) as f32 * 0.01) - 0.08)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 13) as f32 * 0.01) - 0.06)
        .collect();
    let a = dev
        .from_cpu(&a_data, &Shape::from_slice(&[m, k]), DType::F32)
        .unwrap();
    let b = dev
        .from_cpu(&b_data, &Shape::from_slice(&[k, n]), DType::F32)
        .unwrap();
    let (out, handle) = dev
        .matmul(a.as_ref(), b.as_ref(), &Shape::from_slice(&[m, n]))
        .unwrap();
    handle.synchronize().unwrap();
    let got = out.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0f32;
    for r in 0..m {
        for j in 0..n {
            let want: f32 = (0..k).map(|p| a_data[r * k + p] * b_data[p * n + j]).sum();
            max_diff = max_diff.max((got[r * n + j] - want).abs());
        }
    }
    max_diff
}

#[test]
fn f32_split_k_matmul_matches_cpu_reference() {
    if !gpu_ready() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skipped: no ROCm device: {e:?}]");
            return;
        }
    };

    // Both split-K triggers: m > 1 (prefill-class) and k > 8192
    // (long-reduction decode-class). Thresholds sized so an fp-tolerance
    // failure is unmissable — the pre-fix kernel erred in the ~1e3 range.
    for &(m, n, k) in &[
        (4usize, 64usize, 4096usize),
        (1, 64, 12288),
        (4, 4096, 4096),
    ] {
        let d = check_shape(&dev, m, n, k);
        println!("[split-k] m={m} n={n} k={k} max_abs_diff={d:.3e}");
        assert!(
            d < 1e-3,
            "F32 split-K matmul (m={m}, n={n}, k={k}) diverged from CPU: {d:.3e} \
             — reduction kernel dtype dispatch regressed?"
        );
    }
}
