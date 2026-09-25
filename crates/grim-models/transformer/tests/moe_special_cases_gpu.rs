//! GPU parity tests for Phase 3 special-case MoE architectures.
//!
//! Covers:
//! 1. GLM-5.2 GELU grouped dispatch (`grim_moe_fused_grouped_gelu`) vs CPU reference.
//! 2. DBRX top-4 real routing (`GRIM_DBRX_REAL_ROUTING=1`) via `fused_moe_dispatch_from_logits`.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_backend_rocm::RocmDevice;
use grim_nn::Linear;
use grim_tensor::shape::Shape;
use grim_tensor::{CoreTensorOps, DType, Device, Tensor};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::device::util::gpu_test_enabled() {
        return None;
    }
    RocmDevice::try_new(0).ok()
}

/// Serializes GPU tests in this binary (one device; concurrent Charon
/// dispatches contend and give false failures under `--test-threads=N`).
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    grim_backend_rocm::device::util::gpu_test_lock()
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
        })
        .collect()
}

fn rocm_tensor(dev: &RocmDevice, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        Device::Rocm(0),
    )
}

fn make_linear_rocm(dev: &RocmDevice, data: Vec<f32>, rows: usize, cols: usize) -> Linear {
    let shape = Shape::new(vec![rows, cols]);
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    let t = Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        Device::Rocm(0),
    );
    Linear::from_tensor(t, None)
}

fn make_linear_cpu(data: Vec<f32>, rows: usize, cols: usize) -> Linear {
    let shape = Shape::new(vec![rows, cols]);
    let t = cpu_tensor(data, shape);
    Linear::from_tensor(t, None)
}

// ── DBRX real routing parity ────────────────────────────────────────────────

/// Validates DBRX real top-k routing dispatch vs CPU SwiGLU per-expert loop.
/// Exercises `fused_moe_dispatch_from_logits` with route_mode=0 (softmax).
#[test]
fn test_dbrx_real_routing_gpu_vs_cpu_parity() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for DBRX routing GPU test]");
        return;
    };

    let seq = 2usize;
    let hidden = 16usize;
    let inter = 32usize;
    let num_experts = 4usize;
    let top_k = 2usize;

    let router_w = rand_vec(hidden * num_experts, 7);
    let w1_ws: Vec<Vec<f32>> = (0..num_experts)
        .map(|i| rand_vec(inter * hidden, 300 + i as u64))
        .collect();
    let v1_ws: Vec<Vec<f32>> = (0..num_experts)
        .map(|i| rand_vec(inter * hidden, 400 + i as u64))
        .collect();
    let w2_ws: Vec<Vec<f32>> = (0..num_experts)
        .map(|i| rand_vec(hidden * inter, 500 + i as u64))
        .collect();
    let x_data = rand_vec(seq * hidden, 99);

    use grim_models_transformer::shared_moe::{
        CharonCache, MoeExpert, fused_moe_dispatch_from_logits,
    };

    let cache = CharonCache::new();
    let experts_gpu: Vec<MoeExpert> = (0..num_experts)
        .map(|i| MoeExpert {
            gate: make_linear_rocm(&dev, w1_ws[i].clone(), inter, hidden),
            up: make_linear_rocm(&dev, v1_ws[i].clone(), inter, hidden),
            down: make_linear_rocm(&dev, w2_ws[i].clone(), hidden, inter),
        })
        .collect();

    let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![seq, hidden]));
    let router_gpu = make_linear_rocm(&dev, router_w.clone(), num_experts, hidden);
    let logits_gpu = router_gpu.forward(&x_gpu).unwrap();

    let out_gpu = fused_moe_dispatch_from_logits(
        &dev,
        &x_gpu,
        &logits_gpu,
        &experts_gpu,
        None,
        top_k,
        1.0,
        0,
        &cache,
    )
    .unwrap()
    .expect("expected GPU dispatch to return Some")
    .to_vec_f32()
    .unwrap();

    // ── CPU reference ──
    let x_cpu = cpu_tensor(x_data.clone(), Shape::new(vec![seq, hidden]));
    let router_cpu = make_linear_cpu(router_w.clone(), num_experts, hidden);
    let logits_cpu = router_cpu.forward(&x_cpu).unwrap().to_vec_f32().unwrap();

    let xv = x_data.clone();
    let mut out_cpu = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let row = &logits_cpu[s * num_experts..(s + 1) * num_experts];
        let mut idx: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let topk = &idx[..top_k];
        let max_l = topk
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let ws: Vec<f32> = exps.iter().map(|e| e / (sum + 1e-12)).collect();

        let token_x: Vec<f32> = xv[s * hidden..(s + 1) * hidden].to_vec();
        for (i, (ei, _)) in topk.iter().enumerate() {
            let w = ws[i];
            let w1 = &w1_ws[*ei];
            let v1 = &v1_ws[*ei];
            let w2 = &w2_ws[*ei];
            let mut g = vec![0.0f32; inter];
            for r in 0..inter {
                for c in 0..hidden {
                    g[r] += w1[r * hidden + c] * token_x[c];
                }
            }
            let mut u = vec![0.0f32; inter];
            for r in 0..inter {
                for c in 0..hidden {
                    u[r] += v1[r * hidden + c] * token_x[c];
                }
            }
            let act: Vec<f32> = g
                .iter()
                .zip(u.iter())
                .map(|(gi, ui)| {
                    let sig = 1.0 / (1.0 + (-gi).exp());
                    gi * sig * ui
                })
                .collect();
            for r in 0..hidden {
                let mut v = 0.0f32;
                for c in 0..inter {
                    v += w2[r * inter + c] * act[c];
                }
                out_cpu[s * hidden + r] += w * v;
            }
        }
    }

    assert_eq!(out_gpu.len(), out_cpu.len());
    let max_diff = out_gpu
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "DBRX real-routing GPU vs CPU max diff {max_diff:.6} exceeds 1e-3"
    );
    eprintln!("DBRX real-routing parity OK  max_diff={max_diff:.2e}");
}

// ── GLM-5.2 GELU parity ────────────────────────────────────────────────────

/// Validates GLM-5.2 GELU dispatch (`grim_moe_fused_grouped_gelu`) vs CPU GELU loop.
/// Uses `gelu_charon_dispatch` directly to avoid private field access on Glm52Moe.
#[test]
fn test_glm52_gelu_gpu_vs_cpu_parity() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for GLM-5.2 GELU GPU test]");
        return;
    };

    let seq = 2usize;
    let hidden = 16usize;
    let inter = 32usize;
    let num_experts = 4usize;
    let top_k = 2usize;

    let gate_w = rand_vec(num_experts * hidden, 1);
    let proj_ws: Vec<Vec<f32>> = (0..num_experts)
        .map(|i| rand_vec(inter * hidden, 100 + i as u64))
        .collect();
    let down_ws: Vec<Vec<f32>> = (0..num_experts)
        .map(|i| rand_vec(hidden * inter, 200 + i as u64))
        .collect();
    let x_data = rand_vec(seq * hidden, 42);

    use grim_models_transformer::shared_moe::{CharonCache, TokenRouting, gelu_charon_dispatch};

    // ── GPU: compute router logits CPU-side then build routings ──
    let gate_cpu = make_linear_cpu(gate_w.clone(), num_experts, hidden);
    let x_cpu_t = cpu_tensor(x_data.clone(), Shape::new(vec![seq, hidden]));
    let logits_v = gate_cpu.forward(&x_cpu_t).unwrap().to_vec_f32().unwrap();

    let mut routings: Vec<TokenRouting> = Vec::with_capacity(seq);
    for s in 0..seq {
        let row = &logits_v[s * num_experts..(s + 1) * num_experts];
        let mut idx: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let topk = &idx[..top_k];
        let max_l = topk
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        routings.push(
            topk.iter()
                .zip(exps.iter())
                .map(|((ei, _), e)| (*ei, e / (sum + 1e-12)))
                .collect(),
        );
    }

    // Build GPU tensors for gate/down weights.
    let gate_tensors: Vec<Tensor> = (0..num_experts)
        .map(|i| rocm_tensor(&dev, proj_ws[i].clone(), Shape::new(vec![inter, hidden])))
        .collect();
    let down_tensors: Vec<Tensor> = (0..num_experts)
        .map(|i| rocm_tensor(&dev, down_ws[i].clone(), Shape::new(vec![hidden, inter])))
        .collect();
    let gate_refs: Vec<&Tensor> = gate_tensors.iter().collect();
    let down_refs: Vec<&Tensor> = down_tensors.iter().collect();

    let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![seq, hidden]));
    let cache = CharonCache::new();

    let out_gpu =
        gelu_charon_dispatch(&dev, &x_gpu, &gate_refs, &down_refs, &routings, 1.0, &cache)
            .unwrap()
            .expect("expected GELU GPU dispatch to return Some")
            .to_vec_f32()
            .unwrap();

    // ── CPU reference: GELU loop ──
    let xv = x_data.clone();
    let mut out_cpu = vec![0.0f32; seq * hidden];
    for (s, routing) in routings.iter().enumerate() {
        let token_x: Vec<f32> = xv[s * hidden..(s + 1) * hidden].to_vec();
        for &(ei, w) in routing {
            let pw = &proj_ws[ei];
            let dw = &down_ws[ei];
            // proj = dense_h_to_4h: [inter, hidden] @ [hidden]
            let mut h = vec![0.0f32; inter];
            for r in 0..inter {
                for c in 0..hidden {
                    h[r] += pw[r * hidden + c] * token_x[c];
                }
            }
            // GELU tanh approx
            let gelu: Vec<f32> = h
                .iter()
                .map(|&v| 0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044715 * v.powi(3))).tanh()))
                .collect();
            // down = dense_4h_to_h: [hidden, inter] @ [inter]
            for r in 0..hidden {
                let mut v = 0.0f32;
                for c in 0..inter {
                    v += dw[r * inter + c] * gelu[c];
                }
                out_cpu[s * hidden + r] += w * v;
            }
        }
    }

    assert_eq!(out_gpu.len(), out_cpu.len());
    let max_diff = out_gpu
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "GLM-5.2 GELU GPU vs CPU max diff {max_diff:.6} exceeds 1e-3"
    );
    eprintln!("GLM-5.2 GELU parity OK  max_diff={max_diff:.2e}");
}
