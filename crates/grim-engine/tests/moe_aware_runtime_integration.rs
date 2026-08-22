//! End-to-end MoE-aware runtime integration test.
//!
//! Tests the cross-crate coordination of:
//! 1. Semantic-aware prefix state caching (grim-memory + grim-engine).
//! 2. Bandwidth-adaptive q* miss partitioning (grim-scheduler + grim-backend-rocm).
//! 3. Multithreaded persistent CPU worker pool + GPU flag handshake (grim-backend-cpu + grim-backend-rocm).
//! 4. Dynamic VRAM elastic budget reconfiguration at scheduler safe points (grim-memory + grim-engine).
//! 5. Double-buffered full-layer prefill streaming pipeline (grim-engine).
//! 6. FTW flat-bank direct I/O and memory pinning (grim-format).

use grim_backend_cpu::PersistentMoeWorkerPool;
use grim_backend_rocm::{MoeGraphSyncFlag, MoeHybridExecutor};
use grim_format::{FtwDirectLoader, FtwHeader, FtwQuantFormat};
use grim_memory::{
    ElasticMoEAllocation, KvBlockPool, RecurrentLayerState,
    SemanticAnchorRegistry,
};
use grim_scheduler::BandwidthProfile;

#[test]
fn test_moe_aware_semantic_anchor_and_prefix_caching_integration() {
    let mut pool = KvBlockPool::new(32, 4, 64);

    // Register anchor token: 151644 (<think>)
    pool.anchor_registry = SemanticAnchorRegistry::new(vec![151644, 151645]);

    // Construct prompt containing a semantic boundary: [100, 101, ..., 151644, 200, 201]
    let mut prompt = vec![100u32; 16];
    prompt.push(151644); // Semantic anchor at index 16
    prompt.extend(vec![200u32; 15]); // Total 32 tokens (2 blocks of size 16)

    let blocks = vec![1, 2];
    let recurrent_states = vec![
        RecurrentLayerState {
            layer_idx: 0,
            state_data: vec![0.5f32; 128],
            shape: vec![128],
        },
        RecurrentLayerState {
            layer_idx: 1,
            state_data: vec![0.75f32; 128],
            shape: vec![128],
        },
    ];

    // Insert prefix with semantic recurrent state
    pool.insert_prefix_with_recurrent_state(&prompt, &blocks, recurrent_states);

    // Query prefix match: should hit both blocks and return the anchored checkpoint
    let (matched, count, checkpoint) = pool.match_prefix_with_recurrent(&prompt);
    assert_eq!(matched, vec![1, 2]);
    assert_eq!(count, 32);
    assert!(checkpoint.is_some(), "Anchored checkpoint must be retrieved");
    let cp = checkpoint.unwrap();
    assert_eq!(cp.token_offset, 17);
    assert_eq!(cp.layer_states.len(), 2);
    assert_eq!(cp.layer_states[0].state_data[0], 0.5f32);
}

#[test]
fn test_moe_aware_bandwidth_policy_to_hybrid_execution_integration() {
    // 1. Measured PCIe (25 GB/s) and Host DRAM (50 GB/s) -> ratio 0.5
    let profile = BandwidthProfile::new(25_000.0, 50_000.0);
    assert_eq!(profile.compute_q_star(4), 2);
    let (fills, comps) = profile.partition_misses(&[0, 1, 2, 3]);
    assert_eq!(fills, vec![0, 1]);
    assert_eq!(comps, vec![2, 3]);

    // 2. Hybrid executor planning
    let executor = MoeHybridExecutor::new(profile.pcie_bandwidth_mbps, profile.host_bandwidth_mbps);
    let routed = vec![0, 1, 2, 3]; // 4 active experts
    // Simulate: expert 0 is resident on GPU, experts 1, 2, 3 are misses (m = 3)
    let plan = executor.plan_step(0, &routed, |e| e == 0);

    assert_eq!(plan.gpu_resident_experts, vec![0]);
    assert_eq!(plan.gpu_fill_experts.len() + plan.cpu_compute_experts.len(), 3);
    // m = 3 * 0.5 = 1.5 -> round to 2 GPU fills, 1 CPU compute
    assert_eq!(plan.gpu_fill_experts, vec![1, 2]);
    assert_eq!(plan.cpu_compute_experts, vec![3]);

    // 3. Multithreaded CPU pool + GPU concurrent execution with atomic sync flag
    let worker_pool = PersistentMoeWorkerPool::new(Some(2));
    let sync_flag = MoeGraphSyncFlag::new();

    let hidden = 4;
    let inter = 8;
    let num_experts = 4;
    let top_k = 4;

    let tokens = vec![1.0f32, 0.5, -0.5, 2.0];
    let indices = vec![0, 1, 2, 3];
    let weights = vec![0.25f32, 0.25, 0.25, 0.25];

    let mut w_gate = Vec::new();
    let mut w_up = Vec::new();
    let mut w_down = Vec::new();

    for e in 0..num_experts {
        let val = (e + 1) as f32 * 0.1;
        w_gate.push(vec![val; hidden * inter]);
        w_up.push(vec![val; hidden * inter]);
        w_down.push(vec![val; inter * hidden]);
    }

    let out = executor
        .execute_hybrid_step(
            &plan,
            &sync_flag,
            &worker_pool,
            &tokens,
            &indices,
            &weights,
            &w_gate,
            &w_up,
            &w_down,
            1,
            hidden,
            inter,
            num_experts,
            top_k,
            |gpu_active| {
                // Mock GPU GEMM for resident + fills (experts 0, 1, 2)
                let mut mock_gpu = vec![0.0f32; hidden];
                for &e in gpu_active {
                    let w = 0.25;
                    for i in 0..hidden {
                        mock_gpu[i] += w * (e + 1) as f32 * 0.1;
                    }
                }
                Ok(mock_gpu)
            },
        )
        .unwrap();

    assert_eq!(out.len(), hidden);
    assert!(sync_flag.is_cpu_done());
    assert!(sync_flag.is_gpu_ready());
}

#[test]
fn test_moe_aware_elastic_safe_point_rebalance_integration() {
    let mut pool = KvBlockPool::new(100, 4, 64);
    let block_bytes = pool.block_bytes();

    let total_vram = 100 * block_bytes + 10 * 1024 * 1024;
    let slot_bytes = 1024 * 1024;
    let mut elastic = ElasticMoEAllocation::new(
        total_vram,
        100 * block_bytes,
        0,
        slot_bytes,
    )
    .unwrap();
    assert_eq!(elastic.max_expert_slots, 0);

    // Dynamic shift at scheduler safe point: allocate 10 expert slots (10MB) by reducing KV
    let new_kv_bytes = 50 * block_bytes;
    let new_expert_bytes = 10 * slot_bytes;
    let new_slots = elastic.rebalance(new_kv_bytes, new_expert_bytes).unwrap();
    assert_eq!(new_slots, 10);

    // Apply new budget to pool
    pool.resize_capacity(50);
    assert_eq!(pool.capacity(), 50);
}

#[test]
fn test_moe_aware_ftw_direct_io_and_prefill_pipeline_integration() {
    // 1. FTW format direct loading
    let header = FtwHeader::new(4, 8, 128, 512, FtwQuantFormat::Bf16);
    let mut loader = FtwDirectLoader::new(header);

    for (name, bank) in &loader.banks {
        assert_eq!(bank.data.len(), 4 * 8 * loader.header.bank_row_bytes[name]);
    }
    let _pinned = loader.pin_all_banks();

    // 2. Prefill pipelining
    let mut prefill_pipe = grim_engine::pipelines::MoePrefillPipeline::new(4, 8, 1024 * 1024, 4 * 1024 * 1024);
    assert!(prefill_pipe.double_buffering_enabled);

    let mut compute_layers = Vec::new();
    let mut dma_layers = Vec::new();

    prefill_pipe
        .execute_pipelined(
            |layer, _buf| {
                compute_layers.push(layer);
                Ok(())
            },
            |layer, _buf| {
                dma_layers.push(layer);
                Ok(())
            },
        )
        .unwrap();

    assert_eq!(compute_layers, vec![0, 1, 2, 3]);
    assert_eq!(dma_layers, vec![0, 1, 2, 3]);
}
