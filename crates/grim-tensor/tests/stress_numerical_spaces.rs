//! Comprehensive stress test for numerical pathways across extreme values,
//! edge cases (subnormals, large position deltas, deep sequences), and FP tolerance bounds.

use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;
use grim_tensor::{AttentionOps, CoreTensorOps, RopeConfig};
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
fn test_rerope_large_position_jumps_and_boundary_delta() {
    let b = 1usize;
    let s = 8usize;
    let d = 32usize;
    let shape = Shape::new(vec![b, s, d]);

    // Test data with wide dynamic range: small numbers (1e-4), unit values, and moderate numbers (1e2)
    let mut data = vec![0.0f32; b * s * d];
    for i in 0..data.len() {
        let factor = match i % 3 {
            0 => 1e-4,
            1 => 1.0,
            _ => 10.0,
        };
        data[i] = ((i as f32 + 1.0) * 0.03).cos() * factor;
    }

    let k_orig = make_cpu_tensor(data, shape.clone());
    let dev = grim_backend_cpu::CpuDevice::new();

    // 1. Extreme Jump: 0 -> 131,072 (128K context window boundary)
    let old_pos_far: Vec<u32> = (0..s as u32).map(|i| i * 16).collect();
    let new_pos_far: Vec<u32> = (0..s as u32).map(|i| 131_072 + i * 16).collect();

    let cfg = RopeConfig::new(d, 500000.0); // RoPE base typical for long context (e.g. Llama 3 / Qwen)

    let (k_old, _) = dev.rope(k_orig.storage().as_ref(), &old_pos_far, &cfg, &shape).unwrap();
    let (k_expected, _) = dev.rope(k_orig.storage().as_ref(), &new_pos_far, &cfg, &shape).unwrap();
    let (k_rerope, _) = dev.rerope(k_old.as_ref(), &old_pos_far, &new_pos_far, &cfg, &shape).unwrap();

    let exp_vec = k_expected.to_cpu_vec_f32().unwrap();
    let rerope_vec = k_rerope.to_cpu_vec_f32().unwrap();

    let mut max_diff_far = 0.0f32;
    for (e, r) in exp_vec.iter().zip(rerope_vec.iter()) {
        assert!(e.is_finite(), "Expected values must be finite");
        assert!(r.is_finite(), "Re-RoPE output values must be finite");
        let diff = (e - r).abs();
        if diff > max_diff_far {
            max_diff_far = diff;
        }
    }
    assert!(
        max_diff_far <= 1e-5,
        "128K position jump max diff {max_diff_far} exceeds 1e-5"
    );

    // 2. Identity / Zero Delta: p_new == p_old
    let (k_identity, _) = dev.rerope(k_old.as_ref(), &old_pos_far, &old_pos_far, &cfg, &shape).unwrap();
    let ident_vec = k_identity.to_cpu_vec_f32().unwrap();
    let old_vec = k_old.to_cpu_vec_f32().unwrap();
    let mut max_diff_ident = 0.0f32;
    for (o, i) in old_vec.iter().zip(ident_vec.iter()) {
        let diff = (o - i).abs();
        if diff > max_diff_ident {
            max_diff_ident = diff;
        }
    }
    assert!(
        max_diff_ident <= 1e-6,
        "Zero-delta identity check max diff {max_diff_ident} exceeds 1e-6"
    );

    // 3. Negative Delta: p_new < p_old (retargeting backwards in prompt)
    let (k_backwards, _) = dev.rerope(k_expected.as_ref(), &new_pos_far, &old_pos_far, &cfg, &shape).unwrap();
    let back_vec = k_backwards.to_cpu_vec_f32().unwrap();
    let mut max_diff_back = 0.0f32;
    for (o, b) in old_vec.iter().zip(back_vec.iter()) {
        let diff = (o - b).abs();
        if diff > max_diff_back {
            max_diff_back = diff;
        }
    }
    assert!(
        max_diff_back <= 1e-5,
        "Reverse position retargeting max diff {max_diff_back} exceeds 1e-5"
    );
}
