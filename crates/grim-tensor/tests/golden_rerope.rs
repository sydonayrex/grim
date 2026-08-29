//! Golden tests for Fused Re-RoPE (Position Retargeting).
//!
//! Verifies that un-rotating a key tensor from position p_old and re-rotating
//! it to p_new yields numerical identity with computing RoPE directly at p_new.

use grim_tensor::backend::BackendDevice;
use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;
use grim_tensor::RopeConfig;
use std::sync::Arc;

fn make_cpu_tensor(data: Vec<f32>, shape: Shape) -> Tensor {
    let dev = grim_backend_cpu::CpuDevice::new();
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        QuantProvenance::default(),
        Device::Cpu,
    )
}

#[test]
fn test_rerope_cpu_parity_vs_fresh_rope() {
    let b = 2usize;
    let s = 4usize;
    let d = 8usize;
    let mut data = vec![0.0f32; b * s * d];
    for i in 0..data.len() {
        data[i] = ((i as f32 + 1.0) * 0.1).sin();
    }

    let k_orig = make_cpu_tensor(data, Shape::new(vec![b, s, d]));
    let old_pos: Vec<u32> = vec![10, 11, 12, 13];
    let new_pos: Vec<u32> = vec![100, 101, 102, 103];

    let cfg = RopeConfig::new(d, 10000.0);

    let dev = grim_backend_cpu::CpuDevice::new();

    // 1. Compute direct RoPE at old_pos -> K_old
    let (k_old_storage, _) = dev
        .rope(k_orig.storage().as_ref(), &old_pos, &cfg, k_orig.shape())
        .unwrap();

    // 2. Compute direct RoPE at new_pos -> K_expected
    let (k_exp_storage, _) = dev
        .rope(k_orig.storage().as_ref(), &new_pos, &cfg, k_orig.shape())
        .unwrap();
    let exp_vec = k_exp_storage.to_cpu_vec_f32().unwrap();

    // 3. Re-RoPE K_old directly from old_pos to new_pos -> K_rerope
    let (k_rerope_storage, _) = dev
        .rerope(k_old_storage.as_ref(), &old_pos, &new_pos, &cfg, k_orig.shape())
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
        max_diff <= 1e-6,
        "Re-RoPE max diff {max_diff} vs fresh RoPE exceeds 1e-6 tolerance"
    );
}
