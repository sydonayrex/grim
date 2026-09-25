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
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
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
