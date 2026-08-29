//! Numerical parity test for Vulkan MoE Fused Dispatch on physical GPU.

use grim_backend_vulkan::VulkanDevice;
use grim_tensor::dtype::DType;
use grim_tensor::shape::Shape;
use grim_tensor::CoreTensorOps;

#[test]
fn test_vulkan_moe_fused_dispatch_parity() {
    let devices = VulkanDevice::probe().unwrap();
    if devices.is_empty() {
        eprintln!("Vulkan device uninitialized/unavailable; skipping vulkan_moe test");
        return;
    }
    let vk_dev = &devices[0];

    // Verify FP32 atomic add support for MoE dispatch
    if !vk_dev.caps().supports_fp32_atomic_add {
        eprintln!("Vulkan device does not support FP32 atomic add; skipping GPU MoE dispatch");
        return;
    }

    let batch = 2usize;
    let hidden = 16usize;
    let inter = 32usize;
    let num_experts = 4usize;
    let num_pairs = 4usize; // 2 tokens * 2 top-k experts

    let x_shape = Shape::new(vec![batch, hidden]);
    let gw_shape = Shape::new(vec![num_experts * inter * hidden]);
    let uw_shape = Shape::new(vec![num_experts * inter * hidden]);
    let dw_shape = Shape::new(vec![num_experts * hidden * inter]);

    // Synthetic inputs
    let mut x_data = vec![0.0f32; batch * hidden];
    for i in 0..x_data.len() {
        x_data[i] = ((i as f32 + 1.0) * 0.05).sin();
    }

    let mut gw_data = vec![0.0f32; num_experts * inter * hidden];
    let mut uw_data = vec![0.0f32; num_experts * inter * hidden];
    let mut dw_data = vec![0.0f32; num_experts * hidden * inter];
    for i in 0..gw_data.len() {
        gw_data[i] = ((i as f32 + 1.0) * 0.02).cos() * 0.1;
        uw_data[i] = ((i as f32 + 2.0) * 0.02).sin() * 0.1;
    }
    for i in 0..dw_data.len() {
        dw_data[i] = ((i as f32 + 3.0) * 0.02).cos() * 0.1;
    }

    let rtok = vec![0u32, 0, 1, 1];
    let rexp = vec![0u32, 1, 2, 3];
    let rw = vec![0.6f32, 0.4, 0.7, 0.3];
    let routed_scale = 1.0f32;

    // CPU Reference computation
    let mut cpu_out = vec![0.0f32; batch * hidden];
    for p in 0..num_pairs {
        let tok = rtok[p] as usize;
        let exp = rexp[p] as usize;
        let weight = rw[p] * routed_scale;

        let x_tok = &x_data[tok * hidden..(tok + 1) * hidden];
        let gw_exp = &gw_data[exp * inter * hidden..(exp + 1) * inter * hidden];
        let uw_exp = &uw_data[exp * inter * hidden..(exp + 1) * inter * hidden];
        let dw_exp = &dw_data[exp * hidden * inter..(exp + 1) * hidden * inter];

        for i in 0..inter {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for j in 0..hidden {
                g += gw_exp[i * hidden + j] * x_tok[j];
                u += uw_exp[i * hidden + j] * x_tok[j];
            }
            let silu_g = g / (1.0 + (-g).exp());
            let act = silu_g * u;

            for h in 0..hidden {
                let y = dw_exp[h * inter + i] * act;
                cpu_out[tok * hidden + h] += weight * y;
            }
        }
    }

    // Vulkan GPU computation
    let x_s = vk_dev.from_cpu(&x_data, &x_shape, DType::F32).unwrap();
    let gw_s = vk_dev.upload_f32(&gw_data, &gw_shape).unwrap();
    let uw_s = vk_dev.upload_f32(&uw_data, &uw_shape).unwrap();
    let dw_s = vk_dev.upload_f32(&dw_data, &dw_shape).unwrap();
    let tok_s = vk_dev.upload_u32(&rtok, &Shape::new(vec![num_pairs])).unwrap();
    let exp_s = vk_dev.upload_u32(&rexp, &Shape::new(vec![num_pairs])).unwrap();
    let w_s = vk_dev.upload_f32(&rw, &Shape::new(vec![num_pairs])).unwrap();

    let (out_s, _handle) = vk_dev
        .moe_fused_dispatch(
            x_s.as_ref(),
            gw_s.as_ref(),
            uw_s.as_ref(),
            dw_s.as_ref(),
            tok_s.as_ref(),
            exp_s.as_ref(),
            w_s.as_ref(),
            &x_shape,
            hidden as u32,
            inter as u32,
            num_experts as u32,
            batch as u32,
            routed_scale,
        )
        .unwrap();

    let vk_out = out_s.to_cpu_vec_f32().unwrap();
    assert_eq!(vk_out.len(), cpu_out.len());

    let mut max_diff = 0.0f32;
    for (_i, (&v, &c)) in vk_out.iter().zip(cpu_out.iter()).enumerate() {
        let diff = (v - c).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(
        max_diff <= 1e-4,
        "Vulkan MoE dispatch max diff {max_diff} vs CPU oracle exceeds 1e-4 tolerance"
    );
}
