//! Integration test for GPU-native reductions (reduce_sum, reduce_max, argmax)
//! verifying parity against CPU ground truth across small and large tensors.
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test reduction_parity -- --nocapture

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, ElementwiseOps, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

#[test]
#[ignore]
fn gpu_reduction_parity_vs_cpu() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };

    let test_sizes = [4usize, 32, 64, 256, 1024, 4096, 65536, 262144, 1048576];

    for &n in &test_sizes {
        // Generate pseudo-random f32 data with positive, negative, and extreme values
        let mut seed = 0x1234_5678u64 ^ (n as u64);
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 20.0 - 10.0
        };

        let cpu_data: Vec<f32> = (0..n).map(|_| rand()).collect();
        let shape = Shape::new(vec![n]);
        let dev_storage = dev
            .from_cpu(&cpu_data, &shape, DType::F32)
            .expect("upload tensor to GPU");

        // 1. reduce_sum
        let cpu_sum: f32 = cpu_data.iter().sum();
        let gpu_sum = dev
            .reduce_sum(dev_storage.as_ref())
            .expect("gpu reduce_sum");
        let sum_abs_diff = (cpu_sum - gpu_sum).abs();
        let sum_rel_diff = sum_abs_diff / (cpu_sum.abs().max(gpu_sum.abs()).max(1.0));
        eprintln!(
            "[reduction-sum] N={n} cpu={cpu_sum:.4} gpu={gpu_sum:.4} rel_diff={sum_rel_diff:.6}"
        );
        assert!(
            sum_rel_diff < 1e-4 || sum_abs_diff < 1e-3,
            "reduce_sum mismatch at N={n}: cpu={cpu_sum}, gpu={gpu_sum}, diff={sum_abs_diff}"
        );

        // 2. reduce_max
        let cpu_max = cpu_data
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("cpu max");
        let gpu_max = dev
            .reduce_max(dev_storage.as_ref())
            .expect("gpu reduce_max");
        let max_diff = (cpu_max - gpu_max).abs();
        eprintln!("[reduction-max] N={n} cpu={cpu_max:.4} gpu={gpu_max:.4} diff={max_diff:.6}");
        assert_eq!(
            cpu_max.to_bits(),
            gpu_max.to_bits(),
            "reduce_max exact bit mismatch at N={n}: cpu={cpu_max}, gpu={gpu_max}"
        );

        // 3. argmax
        let cpu_argmax = cpu_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .expect("cpu argmax");
        let gpu_argmax = dev.argmax(dev_storage.as_ref()).expect("gpu argmax");
        eprintln!("[argmax] N={n} cpu={cpu_argmax} gpu={gpu_argmax}");
        assert_eq!(
            cpu_argmax, gpu_argmax,
            "argmax mismatch at N={n}: cpu={cpu_argmax}, gpu={gpu_argmax}"
        );
    }
}

#[test]
#[ignore]
fn gpu_argmax_tie_breaker_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };

    // Test tie breaking: multiple occurrences of the maximum element
    // CPU contract: Iterator::max_by chooses the last index that ties
    let mut data = vec![1.0f32; 1000];
    data[100] = 50.0;
    data[500] = 50.0;
    data[950] = 50.0;

    let shape = Shape::new(vec![data.len()]);
    let dev_storage = dev
        .from_cpu(&data, &shape, DType::F32)
        .expect("upload tensor to GPU");

    let gpu_argmax = dev.argmax(dev_storage.as_ref()).expect("gpu argmax");
    assert_eq!(
        gpu_argmax, 950,
        "argmax must pick the last maximum element on ties"
    );
}

/// Audit §4 regression/env — determinism (task_ring_gemm.rs:219 twice-run
/// byte-equality pattern): `reduce_sum` executed twice with identical inputs
/// must be bit-identical. One canonical sum case only (n=4096), not every size.
#[test]
#[ignore]
fn reduce_sum_deterministic_across_runs() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let n = 4096usize;
    let mut seed = 0x1234_5678u64 ^ (n as u64);
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 20.0 - 10.0
    };
    let cpu_data: Vec<f32> = (0..n).map(|_| rand()).collect();
    let shape = Shape::new(vec![n]);
    let dev_storage = dev
        .from_cpu(&cpu_data, &shape, DType::F32)
        .expect("upload tensor to GPU");
    let r1 = dev.reduce_sum(dev_storage.as_ref()).expect("gpu reduce_sum run 1");
    let r2 = dev.reduce_sum(dev_storage.as_ref()).expect("gpu reduce_sum run 2");
    assert_eq!(
        r1.to_bits(),
        r2.to_bits(),
        "reduce_sum not bit-deterministic across runs: {r1} vs {r2}"
    );
}

/// Audit §2.5/§4 — reduction order (soft `deterministic-repeat` twin coverage is above):
/// Kahan-vs-naive probe on a cancellation-heavy input. The GPU `reduce_sum` must
/// stay within 1e-4 relative of the f64 reference; naive-f32 and Kahan-f32
/// CPU sums are printed for diagnosis (naive may wander far off when a large
/// partial swallows many small ones, Kahan tracks f64 closely).
#[test]
#[ignore]
fn reduce_sum_kahan_vs_naive_on_cancellation_heavy_input() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    // Cancellation-heavy: a huge constant cancelled by its negation, then many
    // small ones. In f32, `1e8 + 1.0` rounds back to `1e8` (ulp at 1e8 is 8),
    // so a naive left-to-right sum swallows every "1.0" before the `-1e8`
    // cancels the big term. f64 keeps them.
    let big: f32 = 1e8;
    let n_small: usize = 2000;
    let mut data: Vec<f32> = Vec::with_capacity(1 + n_small + 1);
    data.push(big);
    for _ in 0..n_small {
        data.push(1.0);
    }
    data.push(-big);
    let n = data.len();

    let dev_storage = dev
        .from_cpu(&data, &Shape::new(vec![n]), DType::F32)
        .expect("upload cancellation-heavy tensor");
    let gpu_sum = dev
        .reduce_sum(dev_storage.as_ref())
        .expect("gpu reduce_sum");

    let f64_ref: f64 = data.iter().map(|v| *v as f64).sum();
    let naive_f32: f32 = data.iter().fold(0.0f32, |a, v| a + v);
    let mut sum_kahan = 0.0f64;
    let mut comp = 0.0f64;
    for &v in &data {
        let y = v as f64 - comp;
        let t = sum_kahan + y;
        comp = (t - sum_kahan) - y;
        sum_kahan = t;
    }
    let kahan_f32 = sum_kahan as f32;

    let rel_gpu = (gpu_sum as f64 - f64_ref).abs() / f64_ref.abs().max(1.0);
    let dist_naive = (naive_f32 as f64 - f64_ref).abs();
    let dist_kahan = (kahan_f32 as f64 - f64_ref).abs();
    let dist_gpu = (gpu_sum as f64 - f64_ref).abs();
    eprintln!(
        "[kahan-probe] n={n} f64={f64_ref:.3} naive_f32={naive_f32} kahan_f32={kahan_f32} gpu={gpu_sum} \
         dist naive={dist_naive:.3} kahan={dist_kahan:.3} gpu={dist_gpu:.3} rel_gpu={rel_gpu:.3e}"
    );
    // Sanity on the diagnosis itself: the fixture really does hurt naive f32.
    assert!(
        dist_naive > dist_kahan + 1.0,
        "fixture must hurt naive more than Kahan (naive {dist_naive}, kahan {dist_kahan})"
    );
    assert!(
        rel_gpu < 1e-4,
        "gpu reduce_sum diverges from f64 reference on cancellation-heavy input: rel={rel_gpu:.3e}"
    );
}
