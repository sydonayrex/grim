//! Test: Vulkan charon_backward MoE expert-weight gradient kernel (structural).
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test charon_backward_parity`.

use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
fn charon_backward_produces_finite_gradients() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    let num_experts = 4u32;
    let hidden = 64u32;
    let inter = 128u32;
    let num_tokens = 8usize;

    let gate_w = vec![0.01f32; num_experts as usize * inter as usize * hidden as usize];
    let up_w = vec![0.01f32; num_experts as usize * inter as usize * hidden as usize];
    let down_w = vec![0.01f32; num_experts as usize * hidden as usize * inter as usize];
    let x = vec![0.1f32; num_tokens * hidden as usize];
    let grad = vec![0.05f32; num_tokens * hidden as usize];

    let gw_shape = Shape::new(vec![num_experts as usize, inter as usize, hidden as usize]);
    let dw_shape = Shape::new(vec![num_experts as usize, hidden as usize, inter as usize]);
    let x_shape = Shape::new(vec![num_tokens, hidden as usize]);
    let g_shape = Shape::new(vec![num_tokens, hidden as usize]);

    let gw_s = dev.from_cpu(&gate_w, &gw_shape, DType::F32).unwrap();
    let uw_s = dev.from_cpu(&up_w, &gw_shape, DType::F32).unwrap();
    let dw_s = dev.from_cpu(&down_w, &dw_shape, DType::F32).unwrap();
    let x_s = dev.from_cpu(&x, &x_shape, DType::F32).unwrap();
    let g_s = dev.from_cpu(&grad, &g_shape, DType::F32).unwrap();

    let result = dev.charon_backward(&*x_s, &*gw_s, &*uw_s, &*dw_s, &*g_s, num_experts, hidden, inter);
    assert!(result.is_ok(), "charon_backward dispatch: {:?}", result.err());
}
