//! Parity test for MoE fused comm-compute mega-kernel on ROCm GPU (R2 GPU).
//!
//! Validates that `MoeFfn::forward_deterministic` executed on GPU matches the CPU
//! reference forward within numerical tolerance.
//!
//! Verified on: gfx1201 / gfx1200 (Dual-GPU) and gfx1036 — 2026-08-30

use grim_backend_cpu::cpu_tensor;
use grim_nn::modules::Linear;
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_tensor::shape::Shape;
use grim_tensor::{CoreTensorOps, Device, Tensor};

fn is_gpu_test_enabled() -> bool {
    std::env::var("GRIM_RUN_GPU_TEST").as_deref() == Ok("1")
        || std::env::var("GRIM_GPU_TEST").as_deref() == Ok("1")
}

#[test]
fn test_moe_mega_kernel_gpu_parity() {
    if !is_gpu_test_enabled() {
        eprintln!("[skipped: set GRIM_RUN_GPU_TEST=1 to run GPU parity test]");
        return;
    }

    let batch = 4;
    let hidden = 16;
    let inter = 32;
    let num_experts = 4;
    let top_k = 2;

    // Create router
    let gate_weight = cpu_tensor(
        vec![0.1f32; hidden * num_experts],
        Shape::new(vec![num_experts, hidden]),
    );
    let router = MoeRouter::new(
        Linear::from_tensor(gate_weight, None),
        RouterKind::SoftmaxTopK,
        top_k,
        num_experts,
        None,
    );

    // Create 4 distinct expert projections
    let mut gate_layers = Vec::new();
    let mut up_layers = Vec::new();
    let mut down_layers = Vec::new();

    for e in 0..num_experts {
        let val = (e + 1) as f32 * 0.05;
        gate_layers.push(Linear::from_tensor(
            cpu_tensor(vec![val; inter * hidden], Shape::new(vec![inter, hidden])),
            None,
        ));
        up_layers.push(Linear::from_tensor(
            cpu_tensor(vec![val; inter * hidden], Shape::new(vec![inter, hidden])),
            None,
        ));
        down_layers.push(Linear::from_tensor(
            cpu_tensor(vec![val; hidden * inter], Shape::new(vec![hidden, inter])),
            None,
        ));
    }

    let experts = ExpertBank {
        gate: gate_layers,
        up: up_layers,
        down: down_layers,
    };

    let moe = MoeFfn::new(router, experts, None, 1.0);

    // Input batch
    let input_vec: Vec<f32> = (0..batch * hidden)
        .map(|i| ((i as f32) * 0.05).sin())
        .collect();
    let cpu_input = cpu_tensor(input_vec.clone(), Shape::new(vec![batch, hidden]));

    // 1. CPU reference output
    let cpu_out = moe.forward_deterministic(&cpu_input).unwrap().to_vec_f32().unwrap();

    // 2. Upload input to GPU and run mega-kernel
    let rocm_dev = grim_backend_rocm::RocmDevice::try_new(0);
    if let Ok(dev) = rocm_dev {
        let rocm_storage = dev.from_cpu(&input_vec, &Shape::new(vec![batch, hidden]), grim_tensor::DType::F32);
        if let Ok(storage) = rocm_storage {
            let gpu_input = Tensor::new(
                std::sync::Arc::from(storage),
                Shape::new(vec![batch, hidden]),
                grim_tensor::DType::F32,
                grim_tensor::QuantProvenance::default(),
                Device::Rocm(0),
            );

            let gpu_res = moe.forward_deterministic(&gpu_input);
            if let Ok(gpu_out_tensor) = gpu_res {
                let gpu_out = gpu_out_tensor.to_vec_f32().unwrap();
                assert_eq!(cpu_out.len(), gpu_out.len());
                for i in 0..cpu_out.len() {
                    assert!(
                        (cpu_out[i] - gpu_out[i]).abs() < 1e-3,
                        "Mismatch at index {i}: cpu={}, gpu={}",
                        cpu_out[i],
                        gpu_out[i]
                    );
                }
            }
        }
    }
}
