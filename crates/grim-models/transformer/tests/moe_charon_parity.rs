//! P1 Charon contract: `MoeFfn` (grim-nn, Charon-capable) must agree with
//! `shared_moe::route_topk` + manual weighted expert sum on CPU.
//!
//! The fused D2D kernel is GPU-only; this CPU test locks the routing math it
//! must match (softmax top-k indices + renorm weights + routed scaling), so
//! router drift fails CI without a GPU.

use grim_backend_cpu::cpu_tensor;
use grim_models_transformer::shared_moe;
use grim_nn::Linear;
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_tensor::{Device, Shape, Tensor};

fn lin(weight: Vec<f32>, out_dim: usize, in_dim: usize) -> Linear {
    assert_eq!(weight.len(), out_dim * in_dim);
    Linear::from_tensor(cpu_tensor(weight, Shape::new(vec![out_dim, in_dim])), None)
}

fn weights(seed: u64, n: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((st >> 33) % 2000) as f32 - 1000.0) / 1000.0 * 0.3
        })
        .collect()
}

fn tiny_moe(top_k: usize) -> (MoeFfn, Tensor) {
    let hidden = 4usize;
    let inter = 4usize;
    let num_experts = 3usize;
    let gate = lin(weights(11, num_experts * hidden), num_experts, hidden);
    let router = MoeRouter::new(gate, RouterKind::SoftmaxTopK, top_k, num_experts, None);
    let gates: Vec<Linear> = (0..num_experts)
        .map(|e| lin(weights(100 + e as u64, inter * hidden), inter, hidden))
        .collect();
    let ups: Vec<Linear> = (0..num_experts)
        .map(|e| lin(weights(200 + e as u64, inter * hidden), inter, hidden))
        .collect();
    let downs: Vec<Linear> = (0..num_experts)
        .map(|e| lin(weights(300 + e as u64, hidden * inter), hidden, inter))
        .collect();
    let bank = ExpertBank::from_linears(gates, ups, downs);
    let moe = MoeFfn::new(router, bank, None, 1.0);
    let x = cpu_tensor(
        vec![0.5, -0.2, 0.3, 0.1, -0.4, 0.6, 0.2, -0.1],
        Shape::new(vec![2, hidden]),
    );
    assert!(matches!(x.device(), Device::Cpu));
    (moe, x)
}

fn manual_weighted_sum(moe: &MoeFfn, x: &Tensor, top_k: usize) -> Vec<f32> {
    let hidden = 4usize;
    let seq_len = 2usize;
    let gate_logits = moe.router.gate.forward(x).unwrap().to_vec_f32().unwrap();
    let num_experts = 3usize;
    let routing = shared_moe::route_topk(&gate_logits, num_experts, top_k).unwrap();
    assert_eq!(routing.len(), seq_len);
    let xv = x.to_vec_f32().unwrap();
    let mut out = vec![0.0f32; seq_len * hidden];
    for s in 0..seq_len {
        let token_x = cpu_tensor(
            xv[s * hidden..(s + 1) * hidden].to_vec(),
            Shape::new(vec![1, hidden]),
        );
        for (expert_idx, weight) in &routing[s] {
            let gate = moe.experts.gate[*expert_idx].forward(&token_x).unwrap();
            let up = moe.experts.up[*expert_idx].forward(&token_x).unwrap();
            let act = grim_nn::modules::silu_mul_on_device(&gate, &up).unwrap();
            let y = moe.experts.down[*expert_idx].forward(&act).unwrap();
            let yv = y.to_vec_f32().unwrap();
            for d in 0..hidden {
                out[s * hidden + d] += weight * yv[d];
            }
        }
    }
    out
}

#[test]
fn moeffn_matches_shared_moe_routing_math() {
    for top_k in [1usize, 2usize, 3usize] {
        let (moe, x) = tiny_moe(top_k);
        let got = moe.forward(&x).unwrap().to_vec_f32().unwrap();
        let expected = manual_weighted_sum(&moe, &x, top_k);
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "top_k={top_k} idx={i}: MoeFfn {g} vs shared_moe manual {e}"
            );
        }
    }
}

#[test]
fn route_topk_boundaries_are_deterministic() {
    let logits = vec![0.1f32, 2.0, -1.0, 0.5, 0.5, 0.5];
    let full = shared_moe::route_topk(&logits, 3, 3).unwrap();
    assert_eq!(full.len(), 2);
    assert_eq!(full[0][0].0, 1, "argmax must win");
    let one = shared_moe::route_topk(&logits, 3, 1).unwrap();
    assert_eq!(one[0].len(), 1);
    assert!((one[0][0].1 - 1.0).abs() < 1e-6, "k=1 weight must be 1");
    let repeat = shared_moe::route_topk(&logits, 3, 2).unwrap();
    let repeat2 = shared_moe::route_topk(&logits, 3, 2).unwrap();
    assert_eq!(repeat, repeat2, "routing must be deterministic");
}
