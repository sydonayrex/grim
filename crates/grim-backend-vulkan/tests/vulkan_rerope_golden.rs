//! Golden numerical parity test for Vulkan Re-RoPE (Position Retargeting).

use grim_backend_cpu::CpuDevice;
use grim_backend_vulkan::VulkanDevice;
use grim_tensor::dtype::DType;
use grim_tensor::shape::Shape;
use grim_tensor::{AttentionOps, CoreTensorOps, RopeConfig};

#[test]
fn test_vulkan_rerope_parity_vs_cpu_oracle() {
    let devices = VulkanDevice::probe().unwrap();
    if devices.is_empty() {
        eprintln!("Vulkan device uninitialized/unavailable; skipping vulkan_rerope test");
        return;
    }
    let cpu_dev = CpuDevice::new();
    let vk_dev = &devices[0];

    let b = 1usize;
    let s = 4usize;
    let d = 16usize;
    let shape = Shape::new(vec![b, s, d]);

    let mut k_init = vec![0.0f32; b * s * d];
    for i in 0..k_init.len() {
        k_init[i] = ((i as f32 + 1.0) * 0.05).sin();
    }

    let old_pos = vec![0u32, 1, 2, 3];
    let new_pos = vec![100u32, 101, 102, 103];
    let cfg = RopeConfig::new(d, 10000.0);

    // 1. Initial RoPE on CPU
    let k_cpu_init = cpu_dev.from_cpu(&k_init, &shape, DType::F32).unwrap();
    let (k_cpu_old, _) = cpu_dev.rope(k_cpu_init.as_ref(), &old_pos, &cfg, &shape).unwrap();

    // 2. CPU Re-RoPE Oracle
    let (k_cpu_retargeted, _) = cpu_dev
        .rerope(k_cpu_old.as_ref(), &old_pos, &new_pos, &cfg, &shape)
        .unwrap();
    let cpu_retargeted_vec = k_cpu_retargeted.to_cpu_vec_f32().unwrap();

    // 3. Vulkan Re-RoPE execution
    let k_vk_old = vk_dev
        .from_cpu(&k_cpu_old.to_cpu_vec_f32().unwrap(), &shape, DType::F32)
        .unwrap();
    let (k_vk_retargeted, _) = vk_dev
        .rerope(k_vk_old.as_ref(), &old_pos, &new_pos, &cfg, &shape)
        .unwrap();
    let vk_retargeted_vec = k_vk_retargeted.to_cpu_vec_f32().unwrap();

    // 4. Parity check
    let mut max_diff = 0.0f32;
    for (c, v) in cpu_retargeted_vec.iter().zip(vk_retargeted_vec.iter()) {
        let diff = (c - v).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(
        max_diff <= 1e-5,
        "Vulkan Re-RoPE max diff {max_diff} vs CPU oracle exceeds 1e-5 tolerance"
    );
}
