//! Parity test: Vulkan rmsnorm_backward vs CPU reference.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test rmsnorm_backward_parity`.

use grim_tensor::backend::AutogradOps;
use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
fn rmsnorm_backward_matches_cpu_reference() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    let x = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
    ];
    let weight = vec![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let grad = vec![
        0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,
    ];
    let x_shape = Shape::new(vec![2, 8]);
    let w_shape = Shape::new(vec![8]);
    let eps = 1e-6f32;

    let x_s = dev.from_cpu(&x, &x_shape, DType::F32).unwrap();
    let w_s = dev.from_cpu(&weight, &w_shape, DType::F32).unwrap();
    let g_s = dev.from_cpu(&grad, &x_shape, DType::F32).unwrap();

    let (dx, dw, _handle) =
        AutogradOps::rmsnorm_backward(&dev, &*x_s, &*w_s, &*g_s, eps, &x_shape, &w_shape).unwrap();

    let dx_v = dx.to_cpu_vec_f32().unwrap();
    let dw_v = dw.to_cpu_vec_f32().unwrap();

    // CPU reference
    let cols = 8usize;
    let mut dx_exp = vec![0.0f32; 16];
    let mut dw_exp = vec![0.0f32; 8];
    for r in 0..2 {
        let base = r * cols;
        let mean_sq: f32 =
            (0..cols).map(|c| x[base + c] * x[base + c]).sum::<f32>() / cols as f32;
        let rms = (mean_sq + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let sum_xg: f32 = (0..cols).map(|c| x[base + c] * grad[base + c]).sum();
        for c in 0..cols {
            let xn = x[base + c] * inv_rms;
            dx_exp[base + c] =
                weight[c] * (grad[base + c] * inv_rms - xn * sum_xg / cols as f32 * inv_rms);
            dw_exp[c] += grad[base + c] * xn;
        }
    }

    for i in 0..16 {
        assert!(
            (dx_v[i] - dx_exp[i]).abs() < 2.5e-7,
            "dx[{}]: {} vs {}",
            i,
            dx_v[i],
            dx_exp[i]
        );
    }
    for i in 0..8 {
        assert!(
            (dw_v[i] - dw_exp[i]).abs() < 2.5e-7,
            "dw[{}]: {} vs {}",
            i,
            dw_v[i],
            dw_exp[i]
        );
    }
}
