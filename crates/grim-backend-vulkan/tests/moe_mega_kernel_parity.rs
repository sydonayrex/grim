//! Test: Vulkan moe_mega_kernel persistent-worker dispatch (structural).
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test moe_mega_kernel_parity`.

use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
fn moe_mega_kernel_produces_finite_output() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    let batch = 4usize;
    let hidden = 64u32;
    let inter = 128u32;
    let num_experts = 4u32;
    let top_k = 2u32;
    let total_routed = batch * top_k as usize;

    let activations = vec![0.1f32; batch * hidden as usize];
    let gate_w = vec![0.01f32; num_experts as usize * inter as usize * hidden as usize];
    let up_w = vec![0.01f32; num_experts as usize * inter as usize * hidden as usize];
    let down_w = vec![0.01f32; num_experts as usize * hidden as usize * inter as usize];
    let destination_slots = vec![0u32; total_routed];
    let global_offsets = vec![0u32; num_experts as usize + 1];
    let expert_counts = vec![0u32; num_experts as usize];

    let act_shape = Shape::new(vec![batch, hidden as usize]);
    let gw_shape = Shape::new(vec![num_experts as usize, inter as usize, hidden as usize]);
    let dw_shape = Shape::new(vec![num_experts as usize, hidden as usize, inter as usize]);

    let act_s = dev.from_cpu(&activations, &act_shape, DType::F32).unwrap();
    let gw_s = dev.from_cpu(&gate_w, &gw_shape, DType::F32).unwrap();
    let uw_s = dev.from_cpu(&up_w, &gw_shape, DType::F32).unwrap();
    let dw_s = dev.from_cpu(&down_w, &dw_shape, DType::F32).unwrap();
    let ds_s = dev.upload_u32(&destination_slots, &Shape::new(vec![total_routed])).unwrap();
    let go_s = dev.upload_u32(&global_offsets, &Shape::new(vec![num_experts as usize + 1])).unwrap();
    let ec_s = dev.upload_u32(&expert_counts, &Shape::new(vec![num_experts as usize])).unwrap();

    let result = dev.moe_mega_kernel(
        &*act_s, &*gw_s, &*uw_s, &*dw_s, &*ds_s, &*go_s, &*ec_s,
        batch as u32, hidden, inter, num_experts, top_k, total_routed as u32,
    );
    assert!(result.is_ok(), "moe_mega_kernel: {:?}", result.err());
}
