//! GPU-native MoE multi-architecture dispatch parity test.
//!
//! Validates:
//! 1. `MoeFfn` routing on GPU with `RouterKind::SoftmaxTopK` (`route_mode 0`).
//! 2. `MoeFfn` routing on GPU with `RouterKind::SoftmaxTopKRenorm` (`route_mode 3`).
//!
//! Asserts:
//! - Numerical parity within tolerance (< 2e-3) against CPU reference oracle.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_backend_rocm::RocmDevice;
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
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
/// dispatches contend on scratch and give false failures under default
/// `--test-threads=N`). See `gpu_test_lock` docs.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    grim_backend_rocm::device::util::gpu_test_lock()
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

/// O(1)-magnitude deterministic weights. REQUIRED for meaningful MoE parity:
/// tiny (±0.1) magnitudes make global-softmax vs renorm-over-top-k agree
/// within tolerance and hide normalization bugs (WI-gpu-native-moe 2026-09-18).
fn o1_vec(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 1.0) * 0.37 + seed).sin() + ((i as f32 + 1.0) * 0.11 + seed * 2.0).cos() * 0.5)
        .collect()
}

#[test]
fn test_moe_ffn_gpu_softmax_and_renorm_parity() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 or GRIM_GPU_TEST=1 for GPU MoE test]");
        return;
    };

    let batch = 2usize;
    let hidden = 16usize;
    let inter = 32usize;
    let num_experts = 4usize;
    let top_k = 2usize;

    // Renorm arm: device route_mode 3 and the host reference implement the
    // SAME normalization, so GPU-vs-CPU parity is meaningful (O(1) weights
    // would expose any divergence well above tolerance).
    {
        let kind = RouterKind::SoftmaxTopKRenorm;
        let gate_weight = o1_vec(num_experts * hidden, 1.0);
        let gate_linear_gpu = Linear::from_tensor(
            rocm_tensor(&dev, gate_weight.clone(), Shape::new(vec![num_experts, hidden])),
            None,
        );
        let gate_linear_cpu = Linear::from_tensor(
            cpu_tensor(gate_weight, Shape::new(vec![num_experts, hidden])),
            None,
        );

        let router_gpu = MoeRouter::new(gate_linear_gpu, kind.clone(), top_k, num_experts, None);
        let router_cpu = MoeRouter::new(gate_linear_cpu, kind, top_k, num_experts, None);

        let mut gate_gpu = Vec::new();
        let mut up_gpu = Vec::new();
        let mut down_gpu = Vec::new();

        let mut gate_cpu = Vec::new();
        let mut up_cpu = Vec::new();
        let mut down_cpu = Vec::new();

        for e in 0..num_experts {
            let s = (e as f32 + 1.0) * 3.0;
            let gw = o1_vec(inter * hidden, s + 1.0);
            let uw = o1_vec(inter * hidden, s + 2.0);
            let dw = o1_vec(hidden * inter, s + 3.0);

            gate_gpu.push(Linear::from_tensor(
                rocm_tensor(&dev, gw.clone(), Shape::new(vec![inter, hidden])),
                None,
            ));
            up_gpu.push(Linear::from_tensor(
                rocm_tensor(&dev, uw.clone(), Shape::new(vec![inter, hidden])),
                None,
            ));
            down_gpu.push(Linear::from_tensor(
                rocm_tensor(&dev, dw.clone(), Shape::new(vec![hidden, inter])),
                None,
            ));

            gate_cpu.push(Linear::from_tensor(
                cpu_tensor(gw, Shape::new(vec![inter, hidden])),
                None,
            ));
            up_cpu.push(Linear::from_tensor(
                cpu_tensor(uw, Shape::new(vec![inter, hidden])),
                None,
            ));
            down_cpu.push(Linear::from_tensor(
                cpu_tensor(dw, Shape::new(vec![hidden, inter])),
                None,
            ));
        }

        let moe_gpu = MoeFfn::new(
            router_gpu,
            ExpertBank::from_linears(gate_gpu, up_gpu, down_gpu),
            None,
            1.0,
        );
        let moe_cpu = MoeFfn::new(
            router_cpu,
            ExpertBank::from_linears(gate_cpu, up_cpu, down_cpu),
            None,
            1.0,
        );

        let x_data = o1_vec(batch * hidden, 7.0);
        let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![batch, hidden]));
        let x_cpu = cpu_tensor(x_data, Shape::new(vec![batch, hidden]));

        let out_gpu = moe_gpu.forward(&x_gpu).expect("GPU forward").to_vec_f32().unwrap();
        let out_cpu = moe_cpu.forward(&x_cpu).expect("CPU forward").to_vec_f32().unwrap();

        assert_eq!(out_gpu.len(), out_cpu.len());
        let max_diff = out_gpu
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 2e-3,
            "MoeFfn SoftmaxTopKRenorm GPU vs CPU parity diverged: max_diff={max_diff}"
        );
        eprintln!("MoeFfn SoftmaxTopKRenorm parity OK  max_diff={max_diff:.2e}");
    }

    // SoftmaxTopK arm: device route_mode 0 is GLOBAL softmax (HF Qwen
    // semantics) while the host `MoeRouter::route` renormalizes — comparing
    // GPU MoeFfn against CPU MoeFfn here is a KNOWN semantic mismatch, not
    // a kernel bug. Assert against the global-softmax SwiGLU oracle
    // instead (same oracle as `check_shared_moe_d2d_parity` mode 0).
    {
        let gate_weight = o1_vec(num_experts * hidden, 2.0);
        let gate_linear_gpu = Linear::from_tensor(
            rocm_tensor(&dev, gate_weight.clone(), Shape::new(vec![num_experts, hidden])),
            None,
        );
        let router_gpu = MoeRouter::new(gate_linear_gpu, RouterKind::SoftmaxTopK, top_k, num_experts, None);

        let mut gate_gpu = Vec::new();
        let mut up_gpu = Vec::new();
        let mut down_gpu = Vec::new();
        let mut gw_all = Vec::new();
        let mut uw_all = Vec::new();
        let mut dw_all = Vec::new();

        for e in 0..num_experts {
            let s = (e as f32 + 1.0) * 5.0;
            let gw = o1_vec(inter * hidden, s + 1.0);
            let uw = o1_vec(inter * hidden, s + 2.0);
            let dw = o1_vec(hidden * inter, s + 3.0);

            gate_gpu.push(Linear::from_tensor(
                rocm_tensor(&dev, gw.clone(), Shape::new(vec![inter, hidden])),
                None,
            ));
            up_gpu.push(Linear::from_tensor(
                rocm_tensor(&dev, uw.clone(), Shape::new(vec![inter, hidden])),
                None,
            ));
            down_gpu.push(Linear::from_tensor(
                rocm_tensor(&dev, dw.clone(), Shape::new(vec![hidden, inter])),
                None,
            ));
            gw_all.push(gw);
            uw_all.push(uw);
            dw_all.push(dw);
        }

        let moe_gpu = MoeFfn::new(
            router_gpu,
            ExpertBank::from_linears(gate_gpu, up_gpu, down_gpu),
            None,
            1.0,
        );

        let x_data = o1_vec(batch * hidden, 9.0);
        let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![batch, hidden]));
        let out_gpu = moe_gpu.forward(&x_gpu).expect("GPU forward").to_vec_f32().unwrap();

        // Global-softmax oracle from CPU gate logits.
        let x_cpu = cpu_tensor(x_data.clone(), Shape::new(vec![batch, hidden]));
        let gate_cpu = Linear::from_tensor(
            cpu_tensor(gate_weight, Shape::new(vec![num_experts, hidden])),
            None,
        );
        let logits = gate_cpu.forward(&x_cpu).unwrap().to_vec_f32().unwrap();
        let mut topk_idx = vec![Vec::new(); batch];
        let mut weights = vec![Vec::new(); batch];
        for s in 0..batch {
            let row = &logits[s * num_experts..(s + 1) * num_experts];
            let mut idx: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let max_l = idx.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = idx.iter().map(|(_, l)| (l - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let k = top_k.min(num_experts);
            topk_idx[s] = idx[..k].iter().map(|(i, _)| *i).collect();
            weights[s] = idx[..k]
                .iter()
                .enumerate()
                .map(|(j, _)| exps[idx.iter().position(|(i, _)| *i == topk_idx[s][j]).unwrap()] / (sum + 1e-12))
                .collect();
        }
        let out_cpu = swiglu_oracle(
            &x_data, batch, hidden, inter, &gw_all, &uw_all, &dw_all, &topk_idx, &weights,
        );

        assert_eq!(out_gpu.len(), out_cpu.len());
        let max_diff = out_gpu
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "MoeFfn SoftmaxTopK GPU vs global oracle diverged: max_diff={max_diff}"
        );
        eprintln!("MoeFfn SoftmaxTopK (global) parity OK  max_diff={max_diff:.2e}");
    }
}

#[test]
fn test_granite_moe_block_gpu_dispatch_matches_cpu() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 or GRIM_GPU_TEST=1 for GPU MoE test]");
        return;
    };

    let hidden = 16usize;
    let inter = 32usize;
    let n_exp = 4usize;
    let top_k = 2usize;
    let seq = 2usize;

    let x_data = o1_vec(seq * hidden, 11.0);
    let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![seq, hidden]));
    let x_cpu = cpu_tensor(x_data, Shape::new(vec![seq, hidden]));

    let gate_weight = o1_vec(n_exp * hidden, 12.0);
    let gate_gpu = Linear::from_tensor(
        rocm_tensor(&dev, gate_weight.clone(), Shape::new(vec![n_exp, hidden])),
        None,
    );
    let gate_cpu = Linear::from_tensor(
        cpu_tensor(gate_weight, Shape::new(vec![n_exp, hidden])),
        None,
    );

    let router_gpu = MoeRouter::new(gate_gpu, RouterKind::SoftmaxTopKRenorm, top_k, n_exp, None);
    let router_cpu = MoeRouter::new(gate_cpu, RouterKind::SoftmaxTopKRenorm, top_k, n_exp, None);

    let mut g_gpu = Vec::new();
    let mut u_gpu = Vec::new();
    let mut d_gpu = Vec::new();
    let mut g_cpu = Vec::new();
    let mut u_cpu = Vec::new();
    let mut d_cpu = Vec::new();

    for e in 0..n_exp {
        let s = (e as f32 + 1.0) * 8.0;
        let gw = o1_vec(inter * hidden, s + 1.0);
        let uw = o1_vec(inter * hidden, s + 2.0);
        let dw = o1_vec(hidden * inter, s + 3.0);

        g_gpu.push(Linear::from_tensor(rocm_tensor(&dev, gw.clone(), Shape::new(vec![inter, hidden])), None));
        u_gpu.push(Linear::from_tensor(rocm_tensor(&dev, uw.clone(), Shape::new(vec![inter, hidden])), None));
        d_gpu.push(Linear::from_tensor(rocm_tensor(&dev, dw.clone(), Shape::new(vec![hidden, inter])), None));

        g_cpu.push(Linear::from_tensor(cpu_tensor(gw, Shape::new(vec![inter, hidden])), None));
        u_cpu.push(Linear::from_tensor(cpu_tensor(uw, Shape::new(vec![inter, hidden])), None));
        d_cpu.push(Linear::from_tensor(cpu_tensor(dw, Shape::new(vec![hidden, inter])), None));
    }

    let ffn_gpu = MoeFfn::new(router_gpu, ExpertBank::from_linears(g_gpu, u_gpu, d_gpu), None, 1.0);
    let ffn_cpu = MoeFfn::new(router_cpu, ExpertBank::from_linears(g_cpu, u_cpu, d_cpu), None, 1.0);

    let out_gpu = ffn_gpu.forward(&x_gpu).expect("GPU forward").to_vec_f32().unwrap();
    let out_cpu = ffn_cpu.forward(&x_cpu).expect("CPU forward").to_vec_f32().unwrap();

    let max_diff = out_gpu.iter().zip(out_cpu.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 2e-3, "Granite MoE dispatch parity failed: max_diff={max_diff}");
}

// ── WI-gpu-native-moe Phase 0: shared_moe D2D engagement + parity ──────────
// Regression guard: `fused_moe_dispatch_from_logits` must engage (return
// `Some` + populate `CharonCache` routing) — a silent `Ok(None)` fallback is
// a vacuous pass and must FAIL loudly. Numeric gate: device output must match
// the family reference (global-softmax for mode 0, renorm-over-top-k for
// mode 3) within 1e-3.

fn make_shared_experts_gpu(
    dev: &RocmDevice,
    num_experts: usize,
    hidden: usize,
    inter: usize,
    seed_base: u64,
) -> (
    Vec<grim_models_transformer::shared_moe::MoeExpert>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
) {
    use grim_models_transformer::shared_moe::MoeExpert;
    let mut experts = Vec::with_capacity(num_experts);
    let mut gw_all = Vec::with_capacity(num_experts);
    let mut uw_all = Vec::with_capacity(num_experts);
    let mut dw_all = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let s = seed_base + e as u64 * 1000;
        let gw = rand_vec(inter * hidden, s + 1);
        let uw = rand_vec(inter * hidden, s + 2);
        let dw = rand_vec(hidden * inter, s + 3);
        experts.push(MoeExpert {
            gate: Linear::from_tensor(
                rocm_tensor(dev, gw.clone(), Shape::new(vec![inter, hidden])),
                None,
            ),
            up: Linear::from_tensor(
                rocm_tensor(dev, uw.clone(), Shape::new(vec![inter, hidden])),
                None,
            ),
            down: Linear::from_tensor(
                rocm_tensor(dev, dw.clone(), Shape::new(vec![hidden, inter])),
                None,
            ),
        });
        gw_all.push(gw);
        uw_all.push(uw);
        dw_all.push(dw);
    }
    (experts, gw_all, uw_all, dw_all)
}

/// CPU SwiGLU oracle for one (token, expert, weight) triple set.
/// `weights` must already carry the family's routing normalization.
fn swiglu_oracle(
    x_data: &[f32],
    seq: usize,
    hidden: usize,
    inter: usize,
    gw_all: &[Vec<f32>],
    uw_all: &[Vec<f32>],
    dw_all: &[Vec<f32>],
    topk_idx: &[Vec<usize>],
    weights: &[Vec<f32>],
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let token_x = &x_data[s * hidden..(s + 1) * hidden];
        for (rank, &ei) in topk_idx[s].iter().enumerate() {
            let w = weights[s][rank];
            let gw = &gw_all[ei];
            let uw = &uw_all[ei];
            let dw = &dw_all[ei];
            let mut act = vec![0.0f32; inter];
            for j in 0..inter {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for i in 0..hidden {
                    g += gw[j * hidden + i] * token_x[i];
                    u += uw[j * hidden + i] * token_x[i];
                }
                let silu = g / (1.0 + (-g).exp());
                act[j] = silu * u;
            }
            for h in 0..hidden {
                let mut v = 0.0f32;
                for j in 0..inter {
                    v += dw[h * inter + j] * act[j];
                }
                out[s * hidden + h] += w * v;
            }
        }
    }
    out
}

fn check_shared_moe_d2d_parity(route_mode: i32, label: &str) {
    use grim_models_transformer::shared_moe::{CharonCache, fused_moe_dispatch_from_logits};

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for shared_moe D2D {label} test]");
        return;
    };

    let seq = 2usize;
    let hidden = 16usize;
    let inter = 32usize;
    let num_experts = 4usize;
    let top_k = 2usize;

    let router_w = rand_vec(num_experts * hidden, 7);
    let (experts_gpu, gw_all, uw_all, dw_all) =
        make_shared_experts_gpu(&dev, num_experts, hidden, inter, 300);
    let x_data = rand_vec(seq * hidden, 99);
    let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![seq, hidden]));
    let router_gpu = Linear::from_tensor(
        rocm_tensor(&dev, router_w.clone(), Shape::new(vec![num_experts, hidden])),
        None,
    );
    let logits_gpu = router_gpu.forward(&x_gpu).unwrap();

    let cache = CharonCache::new();
    assert!(
        !cache.is_routing_engaged(),
        "{label}: fresh cache must start disengaged"
    );
    let out_gpu = fused_moe_dispatch_from_logits(
        &dev,
        &x_gpu,
        &logits_gpu,
        &experts_gpu,
        None,
        top_k,
        1.0,
        route_mode,
        &cache,
    )
    .unwrap()
    .unwrap_or_else(|| panic!("{label}: D2D dispatch returned Ok(None) — silent fallback, engagement FAILED"));
    assert!(
        cache.is_routing_engaged(),
        "{label}: CharonCache.routing not populated after forward — engagement FAILED"
    );
    let out_gpu = out_gpu.to_vec_f32().unwrap();

    // CPU reference from the same logits.
    let x_cpu = cpu_tensor(x_data.clone(), Shape::new(vec![seq, hidden]));
    let router_cpu = Linear::from_tensor(
        cpu_tensor(router_w, Shape::new(vec![num_experts, hidden])),
        None,
    );
    let logits_cpu = router_cpu.forward(&x_cpu).unwrap().to_vec_f32().unwrap();

    let mut topk_idx = vec![Vec::new(); seq];
    let mut weights = vec![Vec::new(); seq];
    for s in 0..seq {
        let row = &logits_cpu[s * num_experts..(s + 1) * num_experts];
        let mut idx: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let k = top_k.min(num_experts);
        let topk = &idx[..k];
        topk_idx[s] = topk.iter().map(|(i, _)| *i).collect();
        weights[s] = if route_mode == 3 {
            // Renorm over top-k (shared_moe::normalize_weights).
            let max_l = topk.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / (sum + 1e-12)).collect()
        } else {
            // Global softmax (HF Qwen): denom over ALL experts.
            let max_l = idx.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
            let exps_all: Vec<f32> = idx.iter().map(|(_, l)| (l - max_l).exp()).collect();
            let sum_all: f32 = exps_all.iter().sum();
            let w_of = |ei: usize| {
                let pos = idx.iter().position(|(i, _)| *i == ei).unwrap();
                exps_all[pos] / (sum_all + 1e-12)
            };
            topk.iter().map(|(ei, _)| w_of(*ei)).collect()
        };
    }
    let out_cpu = swiglu_oracle(
        &x_data, seq, hidden, inter, &gw_all, &uw_all, &dw_all, &topk_idx, &weights,
    );

    assert_eq!(out_gpu.len(), out_cpu.len());
    let max_diff = out_gpu
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "{label}: shared_moe D2D vs CPU max diff {max_diff:.6} exceeds 1e-3"
    );
    eprintln!("{label} parity OK  max_diff={max_diff:.2e}");
}

#[test]
fn test_shared_moe_d2d_mode0_engaged_parity() {
    check_shared_moe_d2d_parity(0, "shared_moe-mode0");
}

#[test]
fn test_shared_moe_d2d_mode3_renorm_engaged_parity() {
    check_shared_moe_d2d_parity(3, "shared_moe-mode3-renorm");
}

// ── WI-gpu-native-moe Phase 1: minimax_m3 D2D wiring ───────────────────────
// Production path (`MiniMaxM3BlockSparseMoe::forward`) must engage D2D
// (route_mode 3, renorm) on ROCm and match the host reference within 1e-3.
// Silent `Ok(None)` fallback = vacuous pass → FAIL loudly via the
// `is_routing_engaged` sentinel.

#[test]
fn test_minimax_m3_d2d_matches_host_reference() {
    use grim_models_transformer::minimax_m3::{MiniMaxM3BlockSparseMoe, MiniMaxM3Expert};

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for minimax_m3 D2D test]");
        return;
    };

    let hidden = 16usize;
    let inter = 32usize;
    let num_experts = 4usize;
    let top_k = 2usize;
    let seq = 2usize;

    let gate_w = rand_vec(num_experts * hidden, 11);
    let mut w1_all = Vec::with_capacity(num_experts);
    let mut w3_all = Vec::with_capacity(num_experts);
    let mut w2_all = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        w1_all.push(rand_vec(inter * hidden, 100 + e as u64 * 3));
        w3_all.push(rand_vec(inter * hidden, 200 + e as u64 * 3));
        w2_all.push(rand_vec(hidden * inter, 300 + e as u64 * 3));
    }

    let mk_experts = |rocm: bool| -> Vec<MiniMaxM3Expert> {
        (0..num_experts)
            .map(|e| {
                let (g, u, d) = if rocm {
                    (
                        rocm_tensor(&dev, w1_all[e].clone(), Shape::new(vec![inter, hidden])),
                        rocm_tensor(&dev, w3_all[e].clone(), Shape::new(vec![inter, hidden])),
                        rocm_tensor(&dev, w2_all[e].clone(), Shape::new(vec![hidden, inter])),
                    )
                } else {
                    (
                        cpu_tensor(w1_all[e].clone(), Shape::new(vec![inter, hidden])),
                        cpu_tensor(w3_all[e].clone(), Shape::new(vec![inter, hidden])),
                        cpu_tensor(w2_all[e].clone(), Shape::new(vec![hidden, inter])),
                    )
                };
                MiniMaxM3Expert {
                    w1: Linear::from_tensor(g, None),
                    w3: Linear::from_tensor(u, None),
                    w2: Linear::from_tensor(d, None),
                }
            })
            .collect()
    };

    let moe_gpu = MiniMaxM3BlockSparseMoe {
        gate: Linear::from_tensor(
            rocm_tensor(&dev, gate_w.clone(), Shape::new(vec![num_experts, hidden])),
            None,
        ),
        experts: mk_experts(true),
        num_experts_per_tok: top_k,
        charon_cache: grim_models_transformer::shared_moe::CharonCache::new(),
    };
    let moe_cpu = MiniMaxM3BlockSparseMoe {
        gate: Linear::from_tensor(
            cpu_tensor(gate_w, Shape::new(vec![num_experts, hidden])),
            None,
        ),
        experts: mk_experts(false),
        num_experts_per_tok: top_k,
        charon_cache: grim_models_transformer::shared_moe::CharonCache::new(),
    };

    let x_data = rand_vec(seq * hidden, 77);
    let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![seq, hidden]));
    let x_cpu = cpu_tensor(x_data, Shape::new(vec![seq, hidden]));

    let out_gpu = moe_gpu.forward(&x_gpu).unwrap().to_vec_f32().unwrap();
    assert!(
        moe_gpu.charon_cache.is_routing_engaged(),
        "minimax_m3: D2D dispatch did not engage — silent fallback, FAIL"
    );
    let out_cpu = moe_cpu.forward(&x_cpu).unwrap().to_vec_f32().unwrap();

    assert_eq!(out_gpu.len(), out_cpu.len());
    let max_diff = out_gpu
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "minimax_m3 D2D vs host max diff {max_diff:.6} exceeds 1e-3"
    );
    eprintln!("minimax_m3 D2D parity OK  max_diff={max_diff:.2e}");
}

// ── WI-gpu-native-moe Phase 2: quantized D2D engagement gate ───────────────
// Quantized checkpoints (W4A16 packed experts) must engage the D2D dispatch
// (via the device dequant-to-f32 path) — NOT silently fall back — and match
// the exact-dequant native reference. Native quantized kernels
// (`moe_fused_grouped_dispatch_w8a8_*`, `_awq`, `_fp8`, `_mxfp4`) remain
// perf follow-ups; correctness rides the dequant path.

use std::collections::HashMap;

#[derive(Clone)]
struct MemProvider {
    tensors: Arc<HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)>>,
}

impl grim_tensor::provider::TensorProvider for MemProvider {
    fn get(
        &self,
        name: &str,
    ) -> Result<grim_tensor::provider::RawTensor, grim_tensor::error::Error> {
        let (bytes, shape, dtype, provenance) = self.tensors.get(name).cloned().ok_or_else(|| {
            grim_tensor::error::Error::Backend(format!("missing tensor '{name}'"))
        })?;
        Ok(grim_tensor::provider::RawTensor {
            bytes,
            shape,
            dtype,
            provenance,
        })
    }
    fn meta(
        &self,
        name: &str,
    ) -> Result<grim_tensor::provider::TensorMeta, grim_tensor::error::Error> {
        let (_, shape, dtype, provenance) = self.tensors.get(name).cloned().ok_or_else(|| {
            grim_tensor::error::Error::Backend(format!("missing tensor '{name}'"))
        })?;
        Ok(grim_tensor::provider::TensorMeta {
            dtype,
            provenance,
            shape,
            fusion_mask: 0,
        })
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Per-row symmetric W4A16 quantization with EXACT dequant round-trip:
/// reference weights are rewritten to `(code - 8) * scale` so parity is not
/// limited by quantization error.
fn w4a16_quantize_rowmajor(
    w: &mut [f32],
    rows: usize,
    k: usize,
    group_size: usize,
) -> Vec<u8> {
    let words_per_row = k / 8;
    let groups_per_row = k.div_ceil(group_size);
    let mut codes = vec![0u32; rows * words_per_row];
    let mut scales = vec![0.0f32; rows * groups_per_row];
    for row in 0..rows {
        for g in 0..groups_per_row {
            let lo = g * group_size;
            let hi = (lo + group_size).min(k);
            let max_abs = (lo..hi)
                .map(|c| w[row * k + c].abs())
                .fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1e-12 } else { max_abs / 7.0 };
            scales[row * groups_per_row + g] = scale;
            for c in lo..hi {
                let q = ((w[row * k + c] / scale).round() as i32).clamp(-8, 7);
                codes[row * words_per_row + c / 8] |= ((q + 8) as u32) << ((c % 8) * 4);
                w[row * k + c] = q as f32 * scale;
            }
        }
    }
    let mut blob = Vec::new();
    for x in &codes {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    for x in &scales {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob
}

#[test]
fn test_quantized_w4a16_d2d_engages_and_matches_native() {
    use grim_models_transformer::shared_moe::{CharonCache, MoeExpert, fused_moe_dispatch_from_logits};

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for quantized D2D test]");
        return;
    };

    const E: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    const GROUP: usize = 8;
    const TOP_K: usize = 2; // = E: routing identical regardless of fp noise

    let mut seed = 0xC0FFEEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    // Packed fixture + exact-dequant native twin.
    let mut packed_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    let mut native_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ] {
        let mut w: Vec<f32> = (0..E * out * k).map(|_| rand()).collect();
        let blob = w4a16_quantize_rowmajor(&mut w, E * out, k, GROUP);
        packed_map.insert(
            name.to_string(),
            (
                blob,
                vec![E, out, k],
                DType {
                    arith: grim_tensor::ArithType::F32,
                    storage: grim_tensor::Storage::W4A16(
                        grim_tensor::dtype::W4A16Config { group_size: GROUP },
                    ),
                },
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
        native_map.insert(
            name.to_string(),
            (
                f32_bytes(&w),
                vec![E, out, k],
                DType {
                    arith: grim_tensor::ArithType::F32,
                    storage: grim_tensor::Storage::Native,
                },
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
    }
    let router: Vec<f32> = vec![
        3.0, 0.1, 0.2, -3.0, 2.5, 0.05, -2.5, 0.3, -2.0, 0.4, 2.8, 0.15, -0.5, 3.2, 0.25, 1.0,
    ];
    for map in [&mut packed_map, &mut native_map] {
        map.insert(
            "ffn_gate_inp.weight".to_string(),
            (
                f32_bytes(&router),
                vec![E, HIDDEN],
                DType {
                    arith: grim_tensor::ArithType::F32,
                    storage: grim_tensor::Storage::Native,
                },
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
    }

    let run_d2d = |provider: &MemProvider| -> Vec<f32> {
        let ws = grim_nn::WeightSource::root(provider, Device::Rocm(0));
        let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
        let experts: Vec<MoeExpert> = (0..E)
            .map(|e| MoeExpert {
                gate: bank.gate[e].clone(),
                up: bank.up[e].clone(),
                down: bank.down[e].clone(),
            })
            .collect();
        // Gate logits stay on device: project x through the packed bank's
        // router (router is native f32 in both fixtures).
        let gate_w = ws
            .get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight")
            .unwrap();
        let x_data: Vec<f32> = (0..HIDDEN).map(|i| ((i % 7) as f32 * 0.4) - 1.2).collect();
        let x_gpu = rocm_tensor(&dev, x_data, Shape::new(vec![1, HIDDEN]));
        let router_lin = Linear::from_tensor(gate_w, None);
        let logits_gpu = router_lin.forward(&x_gpu).unwrap();
        let cache = CharonCache::new();
        let out = fused_moe_dispatch_from_logits(
            &dev, &x_gpu, &logits_gpu, &experts, None, TOP_K, 1.0, 0, &cache,
        )
        .unwrap()
        .expect("quantized D2D dispatch returned Ok(None) — silent fallback, FAIL");
        assert!(
            cache.is_routing_engaged(),
            "quantized D2D did not populate routing scratch — engagement FAIL"
        );
        out.to_vec_f32().unwrap()
    };

    let packed_provider = MemProvider {
        tensors: Arc::new(packed_map),
    };
    let native_provider = MemProvider {
        tensors: Arc::new(native_map),
    };
    let got = run_d2d(&packed_provider);
    let want = run_d2d(&native_provider);

    assert_eq!(got.len(), want.len());
    let max_diff = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 5e-3,
        "W4A16 D2D vs exact-dequant native max diff {max_diff:.5} exceeds 5e-3"
    );
    eprintln!("W4A16 D2D engagement parity OK  max_diff={max_diff:.2e}");
}
