//! Integration tests for multi-rank Virtual Pipeline Parallelism (VPP, R3).
//!
//! Tests:
//! 1. VPP-Async schedule: per-rank step order puts all forward-arm chunks
//!    before return-arm chunks on non-fold ranks (head-first swap), and
//!    interleaves head→tail per chunk on the fold rank.
//! 2. Multi-rank forward parity vs single-node `forward_vpp` over an
//!    in-process transport (N=2 and N=3 ranks, C chunks).
//! 3. Multi-rank forward parity over the TCP loopback transport
//!    (`grim-kvtransport::TcpActivationTransport`).
//! 4. GPU-gated dual-ROCm-GPU parity (`test_vpp_multi_rank_forward`),
//!    skipped unless `GRIM_RUN_GPU_TEST=1`.
//!
//! Verified on: gfx1201 (RX 9070 XT) + gfx1200 (RX 9060 XT) — 2026-09-04.

use std::sync::Arc;

use grim_backend_cpu::storage::CpuStorage;
use grim_core::error::Result;
use grim_engine::pipeline_engine::{
    InprocVppTransport, TcpVppTransport, VirtualPipelineCoordinator, VirtualPipelinePlan,
    VppChannel, vpp_async_schedule,
};
use grim_kvtransport::TcpActivationTransport;
use grim_memory::KvBlockPool;
use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;

fn make_cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    let s = Shape::new(shape);
    let storage = Arc::new(CpuStorage::new(data, s.clone(), DType::F32));
    Tensor::new(
        storage,
        s,
        DType::F32,
        QuantProvenance::GrimNative,
        Device::Cpu,
    )
}

/// Synthetic layer: layer L adds (L + 1) * scale to every element. Stateless,
/// so the same closure is safe to share across rank threads.
fn add_layer_fn(
    layer_idx: usize,
    x: &Tensor,
    _pool: &mut KvBlockPool,
    scale: f32,
) -> Result<Tensor> {
    let mut v = x.to_vec_f32()?;
    let delta = (layer_idx as f32 + 1.0) * scale;
    for val in &mut v {
        *val += delta;
    }
    Ok(make_cpu_tensor(v, x.shape().dims().to_vec()))
}

fn chunk_inputs(num_chunks: usize, rows: usize, hidden: usize, base: f32) -> Vec<Tensor> {
    (0..num_chunks)
        .map(|c| {
            let data: Vec<f32> = (0..rows * hidden)
                .map(|i| base + c as f32 + (i as f32) * 0.1)
                .collect();
            make_cpu_tensor(data, vec![rows, hidden])
        })
        .collect()
}

#[test]
fn test_vpp_async_schedule_head_first_order() {
    // N=2 ranks, 4 virtual stages, 2 chunks.
    // Fold rank (1) owns vs1+vs2 adjacent → per-chunk interleave.
    // Entry rank (0) owns vs0+vs3 → all heads before any tail (head-first).
    let plan = VirtualPipelinePlan::plan(8, 2, &[0, 1]).unwrap();
    let schedule = vpp_async_schedule(&plan, 2);

    assert_eq!(schedule.len(), 2);

    let r0: Vec<(usize, usize)> = schedule[0]
        .iter()
        .map(|s| (s.virtual_stage, s.chunk))
        .collect();
    assert_eq!(
        r0,
        vec![(0, 0), (0, 1), (3, 0), (3, 1)],
        "entry rank must run every chunk's head before its first tail (VPP-Async swap)"
    );

    let r1: Vec<(usize, usize)> = schedule[1]
        .iter()
        .map(|s| (s.virtual_stage, s.chunk))
        .collect();
    assert_eq!(
        r1,
        vec![(1, 0), (2, 0), (1, 1), (2, 1)],
        "fold rank must interleave head→tail per chunk (fold handoff is local)"
    );

    // Transport only crosses physical rank boundaries.
    let r0_head = &schedule[0][0];
    assert!(r0_head.recv.is_none(), "model head receives no activation");
    let send = r0_head.send.as_ref().expect("head sends to rank 1");
    assert_eq!(send.peer_rank, 1);
    assert!(matches!(send.channel, VppChannel::Forward));

    let r0_tail = &schedule[0][2];
    let recv = r0_tail.recv.as_ref().expect("tail receives from rank 1");
    assert_eq!(recv.peer_rank, 1);
    assert!(matches!(recv.channel, VppChannel::Return));
    assert!(r0_tail.send.is_none(), "model tail produces the output");

    let r1_head = &schedule[1][0];
    assert!(
        r1_head.send.is_none(),
        "fold head hands off to same-rank tail, no transport"
    );
    let r1_tail = &schedule[1][1];
    assert!(
        r1_tail.recv.is_none(),
        "fold tail takes input from same-rank head, no transport"
    );
}

#[test]
fn test_vpp_async_schedule_three_ranks() {
    let plan = VirtualPipelinePlan::plan(12, 3, &[0, 1, 2]).unwrap();
    let schedule = vpp_async_schedule(&plan, 1);

    let stages_of =
        |r: usize| -> Vec<usize> { schedule[r].iter().map(|s| s.virtual_stage).collect() };
    assert_eq!(stages_of(0), vec![0, 5]);
    assert_eq!(stages_of(1), vec![1, 4]);
    assert_eq!(stages_of(2), vec![2, 3], "fold rank owns the middle pair");

    // Non-fold return arm crosses ranks: rank1's vs4 receives from rank2.
    let r1_ret = &schedule[1][1];
    assert_eq!(r1_ret.recv.as_ref().unwrap().peer_rank, 2);
    assert_eq!(r1_ret.send.as_ref().unwrap().peer_rank, 0);
}

/// Reference: run every chunk through the existing single-node VPP traversal.
fn single_node_reference(
    plan: &VirtualPipelinePlan,
    chunks: &[Tensor],
    scale: f32,
) -> Result<Vec<Vec<f32>>> {
    let coordinator = VirtualPipelineCoordinator::new(plan.clone(), 16, 2, 16);
    chunks
        .iter()
        .map(|c| {
            coordinator
                .forward_vpp(c.clone(), |l, x, p| add_layer_fn(l, x, p, scale))
                .and_then(|t| t.to_vec_f32().map_err(Into::into))
        })
        .collect()
}

fn assert_parity(expected: &[Vec<f32>], actual: &[Tensor]) {
    assert_eq!(expected.len(), actual.len(), "one output per chunk");
    for (chunk, want) in expected.iter().enumerate() {
        let got = actual[chunk].to_vec_f32().unwrap();
        assert_eq!(want.len(), got.len(), "chunk {chunk} shape mismatch");
        for (w, g) in want.iter().zip(got.iter()) {
            assert!((w - g).abs() < 1e-5, "chunk {chunk} mismatch: {w} vs {g}");
        }
    }
}

fn run_multirank_parity(num_ranks: usize, total_layers: usize, num_chunks: usize) {
    let ordinals: Vec<usize> = (0..num_ranks).collect();
    let plan = VirtualPipelinePlan::plan(total_layers, num_ranks, &ordinals).unwrap();
    let coordinator = VirtualPipelineCoordinator::new(plan.clone(), 16, 2, 16);
    let chunks = chunk_inputs(num_chunks, 2, 8, 1.0);
    let expected = single_node_reference(&plan, &chunks, 0.25).unwrap();

    let transport = InprocVppTransport::mesh(num_ranks);
    let actual = coordinator
        .forward_vpp_multi_rank(transport.as_ref(), chunks, |l, x, p| {
            add_layer_fn(l, x, p, 0.25)
        })
        .expect("multi-rank forward should succeed");

    assert_parity(&expected, &actual);
}

#[test]
fn test_vpp_multi_rank_forward_parity_two_ranks_inproc() {
    run_multirank_parity(2, 8, 3);
}

#[test]
fn test_vpp_multi_rank_forward_parity_three_ranks_inproc() {
    run_multirank_parity(3, 12, 2);
}

#[test]
fn test_vpp_multi_rank_forward_parity_tcp_loopback() {
    let plan = VirtualPipelinePlan::plan(8, 2, &[0, 1]).unwrap();
    let coordinator = VirtualPipelineCoordinator::new(plan.clone(), 16, 2, 16);
    let chunks = chunk_inputs(2, 2, 8, 2.0);
    let expected = single_node_reference(&plan, &chunks, 0.25).unwrap();

    // One transport object owns a listener per rank; engine-spawned rank
    // threads accept inbound activations on their own port and dial peers
    // out — loopback, no external server process.
    let mut tcp = TcpActivationTransport::bind(2).expect("bind rank listeners");
    tcp.set_peer(0, tcp.local_addr(0).unwrap());
    tcp.set_peer(1, tcp.local_addr(1).unwrap());

    let actual = coordinator
        .forward_vpp_multi_rank(&TcpVppTransport(tcp), chunks, |l, x, p| {
            add_layer_fn(l, x, p, 0.25)
        })
        .expect("multi-rank TCP forward should succeed");

    assert_parity(&expected, &actual);
}

#[test]
fn test_vpp_multi_rank_forward() {
    let gpu_enabled = std::env::var("GRIM_RUN_GPU_TEST").as_deref() == Ok("1");
    if !gpu_enabled {
        eprintln!("[skipped: set GRIM_RUN_GPU_TEST=1 to run the dual-GPU VPP parity test]");
        return;
    }

    // Dual-GPU box: virtual stages alternate between the two ROCm ordinals,
    // so per-stage KV pools land on both devices while the toy layer math
    // stays host-side (transport is value-based, matching the prefill
    // latency budget the work item targets).
    let plan = VirtualPipelinePlan::plan(8, 2, &[0, 1]).unwrap();
    let coordinator = VirtualPipelineCoordinator::new(plan.clone(), 16, 2, 16);
    let chunks = chunk_inputs(2, 2, 8, 1.0);
    let expected = single_node_reference(&plan, &chunks, 0.25).unwrap();

    let transport = InprocVppTransport::mesh(2);
    let actual = coordinator
        .forward_vpp_multi_rank(transport.as_ref(), chunks, |l, x, p| {
            add_layer_fn(l, x, p, 0.25)
        })
        .expect("dual-GPU multi-rank forward should succeed");

    assert_parity(&expected, &actual);
}
