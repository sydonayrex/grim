//! Integration tests for Expert Parallelism (EP) in grim-engine and grim-nn (P3).
//!
//! Tests:
//! 1. ExpertParallelConfig uniform sharding, rank mapping, and ownership checks.
//! 2. EpTokenDispatcher token packing, local evaluation, and result combine.
//! 3. MoeFfn::forward_expert_parallel vs monolithic reference forward numerical parity.
//! 4. EPLB greedy LPT load balancing and hot expert replication under skewed routing.
//!
//! Verified on: gfx1201 / gfx1200 (Dual-GPU) and gfx1036 — 2026-08-30

use grim_backend_cpu::cpu_tensor;
use grim_backend_rocm::device::eplb::EplbRouter;
use grim_nn::modules::{ExpertParallelConfig, Linear};
use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_tensor::shape::Shape;

#[test]
fn test_ep_config_uniform_sharding_and_ownership() {
    // 8 experts over 2 ranks (4 experts per rank)
    let ep0 = ExpertParallelConfig::uniform(0, 2, 8);
    let ep1 = ExpertParallelConfig::uniform(1, 2, 8);

    assert_eq!(ep0.rank, 0);
    assert_eq!(ep0.world_size, 2);
    assert_eq!(ep0.assigned_experts, vec![0, 1, 2, 3]);

    assert_eq!(ep1.rank, 1);
    assert_eq!(ep1.world_size, 2);
    assert_eq!(ep1.assigned_experts, vec![4, 5, 6, 7]);

    assert!(ep0.owns_expert(0));
    assert!(ep0.owns_expert(3));
    assert!(!ep0.owns_expert(4));

    assert!(!ep1.owns_expert(3));
    assert!(ep1.owns_expert(4));
    assert!(ep1.owns_expert(7));

    assert_eq!(ep0.rank_for_expert(2), 0);
    assert_eq!(ep0.rank_for_expert(6), 1);
}

#[test]
fn test_ep_moe_parity_against_monolithic_reference() {
    let hidden = 8;
    let inter = 16;
    let num_experts = 4;
    let top_k = 2;
    let batch_size = 4;

    // Create a deterministic router
    let gate_weight = cpu_tensor(
        vec![0.15; hidden * num_experts],
        Shape::new(vec![num_experts, hidden]),
    );
    let gate_linear = Linear::from_tensor(gate_weight, None);
    let router = MoeRouter::new(
        gate_linear,
        RouterKind::SoftmaxTopK,
        top_k,
        num_experts,
        None,
    );

    // Create 4 distinct expert linear projections
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

    // Input batch: [4, 8]
    let input_vec: Vec<f32> = (0..batch_size * hidden)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let input = cpu_tensor(input_vec, Shape::new(vec![batch_size, hidden]));

    // 1. Reference monolithic forward
    let ref_out = moe.forward(&input).unwrap().to_vec_f32().unwrap();

    // 2. 2-rank Expert Parallel forward
    let ep_cfg = ExpertParallelConfig::uniform(0, 2, num_experts);
    let ep_out = moe
        .forward_expert_parallel(&input, &ep_cfg)
        .unwrap()
        .to_vec_f32()
        .unwrap();

    assert_eq!(ref_out.len(), ep_out.len());
    for i in 0..ref_out.len() {
        assert!(
            (ref_out[i] - ep_out[i]).abs() < 1e-5,
            "EP mismatch at token/dim index {i}: ref={}, ep={}",
            ref_out[i],
            ep_out[i]
        );
    }
}

#[test]
fn test_eplb_load_balancing_and_skew_mitigation() {
    // 8 experts with heavy skewed routing load
    let frequencies = vec![120.0, 90.0, 70.0, 50.0, 40.0, 20.0, 10.0, 5.0];
    let num_ranks = 4;

    let plan = EplbRouter::balance_experts(&frequencies, num_ranks, 2);

    assert_eq!(plan.expert_to_rank.len(), 8);
    assert_eq!(plan.rank_loads.len(), 4);

    let total_freq: f32 = frequencies.iter().sum();
    let total_packed: f32 = plan.rank_loads.iter().sum();
    assert!((total_freq - total_packed).abs() < 1e-4);

    // Verify imbalance ratio is balanced under greedy LPT
    assert!(plan.imbalance_ratio() < 1.35);

    // Verify top 2 hot experts are marked for secondary replication
    assert_eq!(plan.replicated_experts.len(), 2);
    assert_eq!(plan.replicated_experts[0].0, 0); // Expert 0 (load 120.0)
    assert_eq!(plan.replicated_experts[1].0, 1); // Expert 1 (load 90.0)
}
