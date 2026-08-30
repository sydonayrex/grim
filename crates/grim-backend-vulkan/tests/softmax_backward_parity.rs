//! Parity test: Vulkan softmax_backward vs CPU reference.
//!
//! Run with `GRIM_GPU_TEST=1 cargo test -p grim-backend-vulkan --test softmax_backward_parity`.

use grim_tensor::backend::AutogradOps;
use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
#[ignore = "GPU-only: GRIM_GPU_TEST=1"]
fn softmax_backward_matches_cpu_reference() {
    let dev = VulkanDevice::new();
    // 4 rows × 8 cols, values that exercise the full softmax range
    let grad = vec![
        0.1f32, -0.2, 0.3, -0.1, 0.05, 0.15, -0.25, 0.0, 0.0, 0.0, 0.5, -0.5, 0.2, -0.2, 0.3,
        -0.3, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25,
        0.125, -0.125,
    ];
    let softmax_out = vec![
        0.057f32, 0.047, 0.082, 0.052, 0.064, 0.075, 0.043, 0.064, 0.060, 0.060, 0.183, 0.037,
        0.075, 0.045, 0.091, 0.034, 0.055, 0.067, 0.055, 0.067, 0.055, 0.067, 0.055, 0.067,
        0.244, 0.033, 0.137, 0.045, 0.094, 0.052, 0.070, 0.041,
    ];
    let shape = Shape::new(vec![4, 8]);
    let grad_s = dev.from_cpu(&grad, &shape, DType::F32).unwrap();
    let sm_s = dev.from_cpu(&softmax_out, &shape, DType::F32).unwrap();

    let (dx, _handle) = AutogradOps::softmax_backward(&dev, &*grad_s, &*sm_s, &shape).unwrap();
    let dx_v = dx.to_cpu_vec_f32().unwrap();

    // CPU reference: dx_i = s_i * (g_i - Σ_j g_j * s_j)
    let mut expected = vec![0.0f32; 32];
    for row in 0..4 {
        let mut dot = 0.0f32;
        for k in 0..8 {
            dot += grad[row * 8 + k] * softmax_out[row * 8 + k];
        }
        for k in 0..8 {
            expected[row * 8 + k] = softmax_out[row * 8 + k] * (grad[row * 8 + k] - dot);
        }
    }

    for i in 0..32 {
        assert!(
            (dx_v[i] - expected[i]).abs() < 2.5e-7,
            "idx {}: got {}, expected {}",
            i,
            dx_v[i],
            expected[i]
        );
    }
}
