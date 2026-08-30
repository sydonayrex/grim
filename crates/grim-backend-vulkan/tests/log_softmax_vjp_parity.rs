//! Parity test: Vulkan log_softmax_vjp vs CPU reference.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test log_softmax_vjp_parity`.

use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
fn log_softmax_vjp_matches_cpu_reference() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    // log_softmax VJP: dx_i = exp(log_p_i) * (g_i - Σ_j g_j)
    let log_probs = vec![
        -2.3f32, -1.6, -0.9, -1.2, -0.5, -1.8, -0.7, -1.0, -1.5, -0.8, -1.1, -0.6, -1.3, -0.9,
        -1.4, -0.4,
    ];
    let grad = vec![
        0.1f32, -0.2, 0.3, -0.1, 0.05, 0.15, -0.25, 0.0, 0.0, 0.0, 0.5, -0.5, 0.2, -0.2, 0.3,
        -0.3,
    ];
    let shape = Shape::new(vec![2, 8]);
    let lp_s = dev.from_cpu(&log_probs, &shape, DType::F32).unwrap();
    let g_s = dev.from_cpu(&grad, &shape, DType::F32).unwrap();

    let (dx, _handle) = dev.log_softmax_vjp(&*g_s, &*lp_s, &shape).unwrap();
    let dx_v = dx.to_cpu_vec_f32().unwrap();

    let mut expected = vec![0.0f32; 16];
    for row in 0..2 {
        let mut g_sum = 0.0f32;
        for k in 0..8 {
            g_sum += grad[row * 8 + k];
        }
        for k in 0..8 {
            let exp_lp = log_probs[row * 8 + k].exp();
            expected[row * 8 + k] = exp_lp * (grad[row * 8 + k] - g_sum);
        }
    }
    for i in 0..16 {
        assert!(
            (dx_v[i] - expected[i]).abs() < 2.5e-7,
            "dx[{}]: {} vs {}",
            i,
            dx_v[i],
            expected[i]
        );
    }
}
