//! Parity test: Vulkan rope_backward vs CPU reference.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test rope_backward_parity`.

use grim_tensor::backend::AutogradOps;
use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
fn rope_backward_matches_cpu_reference() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    // 8 elements = 4 interleaved (cos, sin) pairs
    let grad = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let cos_v = vec![0.96f32, 0.87, 0.77, 0.66];
    let sin_v = vec![0.28f32, 0.49, 0.64, 0.76];
    // Expand cos/sin to interleaved full-length
    let mut cos_full = Vec::with_capacity(8);
    let mut sin_full = Vec::with_capacity(8);
    for i in 0..4 {
        cos_full.push(cos_v[i]);
        cos_full.push(cos_v[i]);
        sin_full.push(sin_v[i]);
        sin_full.push(sin_v[i]);
    }
    let shape = Shape::new(vec![8]);
    let g_s = dev.from_cpu(&grad, &shape, DType::F32).unwrap();
    let c_s = dev.from_cpu(&cos_full, &shape, DType::F32).unwrap();
    let s_s = dev.from_cpu(&sin_full, &shape, DType::F32).unwrap();

    let (dx, _handle) = AutogradOps::rope_backward(&dev, &*g_s, &*c_s, &*s_s, &shape).unwrap();
    let dx_v = dx.to_cpu_vec_f32().unwrap();

    let mut expected = vec![0.0f32; 8];
    for i in (0..8).step_by(2) {
        expected[i] = grad[i] * cos_full[i] + grad[i + 1] * sin_full[i];
        expected[i + 1] = -grad[i] * sin_full[i] + grad[i + 1] * cos_full[i];
    }
    for i in 0..8 {
        assert!(
            (dx_v[i] - expected[i]).abs() < 1e-5,
            "dx[{}]: {} vs {}",
            i,
            dx_v[i],
            expected[i]
        );
    }
}
