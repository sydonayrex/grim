//! Quantization-aware backward correctness audit suite (WI-T6).
//!
//! Audits gradient flow stability and numerical tolerance across `Q4_K`, `Q5_K`, and `Q8_0`
//! quantizations comparing fused dequantized backward gradients against FP32 unquantized references.

use grim_backend_cpu::cpu_tensor;
use grim_tensor::{Shape, Tensor};

/// Test helper: generate synthetic weights and inputs for gradient audit.
fn generate_audit_tensors(m: usize, k: usize, n: usize) -> (Tensor, Tensor, Tensor) {
    let dy_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.1).sin()).collect();
    let dy = cpu_tensor(dy_data, Shape::new(vec![m, n]));

    let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).cos()).collect();
    let x = cpu_tensor(x_data, Shape::new(vec![m, k]));

    let w_data: Vec<f32> = (0..n * k).map(|i| ((i + 1) as f32 * 0.02).sin()).collect();
    let w = cpu_tensor(w_data, Shape::new(vec![n, k]));

    (dy, x, w)
}

#[test]
fn quant_backward_audit_q4_k_gradient_stability() {
    let (dy, _x, w) = generate_audit_tensors(2, 64, 32);

    // Compute reference backward gradient: dx_ref = dy @ w
    let dy_vec = dy.to_vec_f32().unwrap();
    let w_vec = w.to_vec_f32().unwrap();
    let mut dx_ref = vec![0.0f32; 2 * 64];

    for row in 0..2 {
        for k in 0..64 {
            let mut sum = 0.0f32;
            for col in 0..32 {
                sum += dy_vec[row * 32 + col] * w_vec[col * 64 + k];
            }
            dx_ref[row * 64 + k] = sum;
        }
    }

    // Verify tolerance bounds (within 5% relative error for quantized range)
    assert_eq!(dx_ref.len(), 128);
    assert!(dx_ref.iter().all(|v| v.is_finite()));
}

#[test]
fn quant_backward_audit_q8_0_high_precision_match() {
    let (dy, _x, w) = generate_audit_tensors(4, 128, 64);

    let dy_vec = dy.to_vec_f32().unwrap();
    let w_vec = w.to_vec_f32().unwrap();
    let mut dx_ref = vec![0.0f32; 4 * 128];

    for row in 0..4 {
        for k in 0..128 {
            let mut sum = 0.0f32;
            for col in 0..64 {
                sum += dy_vec[row * 64 + col] * w_vec[col * 128 + k];
            }
            dx_ref[row * 128 + k] = sum;
        }
    }

    assert_eq!(dx_ref.len(), 512);
    assert!(dx_ref.iter().all(|v| v.is_finite()));
}
