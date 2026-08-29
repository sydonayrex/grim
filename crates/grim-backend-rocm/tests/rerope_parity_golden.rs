//! Golden tests for Fused Re-RoPE (Position Retargeting) on ROCm GPU.
//!
//! Verifies that un-rotating a key tensor from position p_old and re-rotating
//! it to p_new on ROCm GPU yields numerical identity with computing RoPE directly at p_new.

use std::panic;

use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::BackendDevice;
use grim_tensor::dtype::DType;
use grim_tensor::shape::Shape;
use grim_tensor::RopeConfig;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

#[test]
fn test_rerope_rocm_gpu_parity_vs_fresh_rope() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping Re-RoPE GPU test");
        return;
    };

    let b = 2usize;
    let s = 4usize;
    let d = 8usize;
    let shape = Shape::new(vec![b, s, d]);
    let mut data = vec![0.0f32; b * s * d];
    for i in 0..data.len() {
        data[i] = ((i as f32 + 1.0) * 0.1).sin();
    }

    let k_orig_storage = BackendDevice::from_cpu(&dev, &data, &shape, DType::F32).unwrap();
    let old_pos: Vec<u32> = vec![10, 11, 12, 13];
    let new_pos: Vec<u32> = vec![100, 101, 102, 103];

    let cfg = RopeConfig::new(d, 10000.0);

    // 1. Compute direct RoPE at old_pos -> K_old
    let (k_old_storage, _) = dev
        .rope(k_orig_storage.as_ref(), &old_pos, &cfg, &shape)
        .unwrap();

    // 2. Compute direct RoPE at new_pos -> K_expected
    let (k_exp_storage, _) = dev
        .rope(k_orig_storage.as_ref(), &new_pos, &cfg, &shape)
        .unwrap();
    let exp_vec = k_exp_storage.to_cpu_vec_f32().unwrap();

    // 3. Re-RoPE K_old directly from old_pos to new_pos -> K_rerope
    let (k_rerope_storage, _) = dev
        .rerope(k_old_storage.as_ref(), &old_pos, &new_pos, &cfg, &shape)
        .unwrap();
    let rerope_vec = k_rerope_storage.to_cpu_vec_f32().unwrap();

    assert_eq!(exp_vec.len(), rerope_vec.len());
    let mut max_diff = 0.0f32;
    for (e, r) in exp_vec.iter().zip(rerope_vec.iter()) {
        let diff = (e - r).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(
        max_diff <= 1e-5,
        "GPU Re-RoPE max diff {max_diff} vs fresh RoPE exceeds 1e-5 tolerance"
    );
}
