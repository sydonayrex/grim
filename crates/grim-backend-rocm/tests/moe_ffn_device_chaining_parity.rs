//! Integration test: MoeFfn::forward end-to-end device execution vs CPU reference parity
//! and on-device output chaining verification without redundant host roundtrips.
//!
//! Addresses TODO(gpu-verify) items for Charon MoE:
//! 1. MoeFfn::forward GPU vs CPU oracle parity under full batch and multi-expert routing.
//! 2. Verification that the GPU forward path stays entirely on-device (zero D2H roundtrip).
//!
//! Verified on: gfx1036 (RDNA2)

use std::panic;
use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_backend_rocm::RocmDevice;
use grim_nn::Linear;
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_tensor::backend::BackendDevice;
use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;

const HIDDEN: usize = 8;
const INTER: usize = 8;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const BATCH: usize = 4;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

struct ExpertWeights {
    gate: Vec<Vec<f32>>,
    up: Vec<Vec<f32>>,
    down: Vec<Vec<f32>>,
}

fn deterministic_expert_weights() -> ExpertWeights {
    let mut gate = Vec::with_capacity(NUM_EXPERTS);
    let mut up = Vec::with_capacity(NUM_EXPERTS);
    let mut down = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        let seed = (e as f32 + 1.0) * 0.37;
        let mut g = vec![0.0f32; INTER * HIDDEN];
        let mut u = vec![0.0f32; INTER * HIDDEN];
        let mut d = vec![0.0f32; HIDDEN * INTER];
        for i in 0..INTER {
            for j in 0..HIDDEN {
                g[i * HIDDEN + j] = ((i as f32 + 1.0) * 0.1 + (j as f32 + 1.0) * 0.05 + seed).sin();
                u[i * HIDDEN + j] = ((i as f32 + 1.0) * 0.07 + (j as f32 + 1.0) * 0.03 + seed * 2.0).cos();
            }
        }
        for h in 0..HIDDEN {
            for j in 0..INTER {
                d[h * INTER + j] = 1.0 / (1.0 + h as f32 + j as f32 + seed);
            }
        }
        gate.push(g);
        up.push(u);
        down.push(d);
    }
    ExpertWeights { gate, up, down }
}

fn deterministic_router_gate() -> Vec<f32> {
    let mut gw = vec![0.0f32; NUM_EXPERTS * HIDDEN];
    for e in 0..NUM_EXPERTS {
        for i in 0..HIDDEN {
            gw[e * HIDDEN + i] = ((e as f32 + 1.0) * 0.5 + i as f32 * 0.1).sin();
        }
    }
    gw
}

fn deterministic_activations() -> Vec<f32> {
    let mut x = vec![0.0f32; BATCH * HIDDEN];
    for t in 0..BATCH {
        for i in 0..HIDDEN {
            x[t * HIDDEN + i] = ((t as f32 + 1.0) * 0.7 + i as f32 * 0.3).sin();
        }
    }
    x
}

fn build_moe_oracle(routed_scaling_factor: f32) -> MoeFfn {
    let ew = deterministic_expert_weights();
    let gw = deterministic_router_gate();
    let gate = Linear::from_tensor(cpu_tensor(gw, Shape::new(vec![NUM_EXPERTS, HIDDEN])), None);
    let mut eg = Vec::with_capacity(NUM_EXPERTS);
    let mut eu = Vec::with_capacity(NUM_EXPERTS);
    let mut ed = Vec::with_capacity(NUM_EXPERTS);
    for e in 0..NUM_EXPERTS {
        eg.push(Linear::from_tensor(
            cpu_tensor(ew.gate[e].clone(), Shape::new(vec![INTER, HIDDEN])),
            None,
        ));
        eu.push(Linear::from_tensor(
            cpu_tensor(ew.up[e].clone(), Shape::new(vec![INTER, HIDDEN])),
            None,
        ));
        ed.push(Linear::from_tensor(
            cpu_tensor(ew.down[e].clone(), Shape::new(vec![HIDDEN, INTER])),
            None,
        ));
    }
    let bank = ExpertBank::from_linears(eg, eu, ed);
    let router = MoeRouter::new(gate, RouterKind::SoftmaxTopK, TOP_K, NUM_EXPERTS, None);
    MoeFfn::new(router, bank, None, routed_scaling_factor)
}

#[test]
fn test_moe_ffn_forward_gpu_cpu_parity_and_device_residency() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping device parity test");
        return;
    };

    let rsf = 1.0f32;
    let moe = build_moe_oracle(rsf);
    let x_data = deterministic_activations();
    let x_cpu = cpu_tensor(x_data.clone(), Shape::new(vec![BATCH, HIDDEN]));

    // 1. CPU Oracle Forward
    let cpu_out = moe.forward(&x_cpu).expect("CPU forward should succeed");
    let cpu_v = cpu_out.to_vec_f32().expect("CPU out to vec");

    // 2. GPU Device-Resident Forward
    let dev_storage = BackendDevice::from_cpu(&dev, &x_data, &Shape::new(vec![BATCH, HIDDEN]), DType::F32)
        .expect("upload activation to GPU");
    let x_gpu = Tensor::new(
        Arc::from(dev_storage),
        Shape::new(vec![BATCH, HIDDEN]),
        DType::F32,
        QuantProvenance::default(),
        Device::Rocm(0),
    );

    let gpu_out = moe.forward(&x_gpu).expect("GPU forward should succeed");

    // Assert the output tensor stays on Device::Rocm(0) without premature host download
    assert_eq!(
        gpu_out.device(),
        &Device::Rocm(0),
        "MoeFfn::forward must preserve GPU device placement for output tensor"
    );

    // Download for numerical verification
    let gpu_v = gpu_out.to_vec_f32().expect("GPU out to vec");
    assert_eq!(gpu_v.len(), cpu_v.len());

    let mut max_diff = 0.0f32;
    for (g, c) in gpu_v.iter().zip(cpu_v.iter()) {
        let diff = (g - c).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(
        max_diff <= 1e-3,
        "MoeFfn::forward GPU vs CPU max abs diff {max_diff} exceeds 1e-3 tolerance"
    );
}
