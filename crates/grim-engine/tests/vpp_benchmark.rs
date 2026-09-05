//! VPP bubble-ratio benchmark (R3 multi-node) — hardware-gated.
//!
//! Runs the VPP coordinator across 2 ROCm ranks (per-stage KV pools on both
//! devices) and reports, against the single-rank sequential baseline:
//! - wall-clock total for a multi-chunk prefill,
//! - per-rank bubble ratio = share of wall time each rank spent blocked on
//!   cross-rank recv (the idle_steps / total_steps proxy from the work item).
//!
//! The layer math is synthetic (fixed-cost dense update), so this documents
//! the *mechanism's* overlap on the target hardware, not a model-quality
//! number; the synthesis headline (98% bubble reduction vs DCPP) needs a
//! real 512K-token prefill run wired through `Engine`.
//!
//! Run on the dual-GPU box with:
//!   GRIM_RUN_GPU_TEST=1 cargo test -p grim-engine --test vpp_benchmark -- --nocapture
//! Skips (prints a notice) without the gate, per the hardware-gating rule.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use grim_backend_cpu::storage::CpuStorage;
use grim_core::error::Result;
use grim_engine::pipeline_engine::{
    InprocVppTransport, TcpVppTransport, VirtualPipelineCoordinator, VirtualPipelinePlan,
    VppActivationTransport, VppTransfer,
};
use grim_kvtransport::TcpActivationTransport;
use grim_memory::KvBlockPool;
use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;

const LAYERS: usize = 8;
const CHUNKS: usize = 4;
const ROWS: usize = 64;
const HIDDEN: usize = 512;
/// Dense-update repetitions per layer call; sized so a debug build spends
/// low-single-digit ms per layer and overlap is measurable over transfer cost.
const WORK_REPS: usize = 300;

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

fn chunk_inputs(num_chunks: usize) -> Vec<Tensor> {
    (0..num_chunks)
        .map(|c| {
            let data: Vec<f32> = (0..ROWS * HIDDEN)
                .map(|i| (i % 97) as f32 + c as f32)
                .collect();
            make_cpu_tensor(data, vec![ROWS, HIDDEN])
        })
        .collect()
}

/// Synthetic transformer layer: fixed-cost dense probe over the activation
/// vector, renormalized per pass so the value stays bounded (no NaN drift),
/// then applied as a small deterministic shift. `black_box` keeps the probe
/// from being optimized away, which would turn this into a pure-transfer
/// measurement.
fn dense_layer(_layer_idx: usize, x: &Tensor, _pool: &mut KvBlockPool) -> Result<Tensor> {
    let mut v = x.to_vec_f32()?;
    let mut probe = 0.5f32;
    for _ in 0..WORK_REPS {
        // Renormalize every few elements: the probe recursion multiplies by
        // ~(cell+1) each step, so an unbounded pass overflows f32 to Inf and
        // the next multiply by a zero cell turns it into NaN. 8 steps bound
        // the growth at ~1e16, far under f32::MAX.
        for (k, cell) in v.iter().enumerate() {
            probe = probe.mul_add(*cell, probe);
            if k % 8 == 7 {
                probe = probe.rem_euclid(1.0);
            }
        }
        probe = probe.rem_euclid(1.0);
    }
    let probe = std::hint::black_box(probe);
    for cell in &mut v {
        *cell += probe * 1e-3;
    }
    Ok(make_cpu_tensor(v, x.shape().dims().to_vec()))
}

/// Wraps any transport and accumulates per-rank recv-block time — the
/// bubble proxy: a rank blocked in recv is a rank not computing.
struct TimedVppTransport {
    inner: Arc<dyn VppActivationTransport>,
    recv_wait: Mutex<Vec<Duration>>,
}

impl TimedVppTransport {
    fn new(inner: Arc<dyn VppActivationTransport>, num_ranks: usize) -> Self {
        Self {
            inner,
            recv_wait: Mutex::new(vec![Duration::ZERO; num_ranks]),
        }
    }

    fn recv_wait_snapshot(&self) -> Vec<Duration> {
        self.recv_wait.lock().unwrap().clone()
    }
}

impl VppActivationTransport for TimedVppTransport {
    fn send(&self, from_rank: usize, xfer: &VppTransfer, data: &[f32]) -> Result<()> {
        self.inner.send(from_rank, xfer, data)
    }

    fn recv(&self, for_rank: usize, xfer: &VppTransfer, elem_count: usize) -> Result<Vec<f32>> {
        let started = Instant::now();
        let out = self.inner.recv(for_rank, xfer, elem_count);
        let waited = started.elapsed();
        if let Ok(mut wait) = self.recv_wait.lock() {
            wait[for_rank] += waited;
        }
        out
    }
}

#[test]
fn vpp_bubble_ratio_benchmark() {
    if std::env::var("GRIM_RUN_GPU_TEST").as_deref() != Ok("1") {
        eprintln!(
            "[skipped: set GRIM_RUN_GPU_TEST=1 on a >=2 ROCm GPU box to run the VPP bubble benchmark]"
        );
        return;
    }

    let chunks = chunk_inputs(CHUNKS);
    let num_chunks = chunks.len();

    // Baseline: one rank owns every virtual stage (sequential V-traversal).
    let single_plan = VirtualPipelinePlan::plan(LAYERS, 1, &[0]).unwrap();
    let single = VirtualPipelineCoordinator::new(single_plan, 16, 2, 16);
    let started = Instant::now();
    for chunk in &chunks {
        single
            .forward_vpp(chunk.clone(), dense_layer)
            .expect("single-rank forward");
    }
    let single_total = started.elapsed();

    // VPP: two ranks, KV pools split across both ROCm devices.
    let vpp_plan = VirtualPipelinePlan::plan(LAYERS, 2, &[0, 1]).unwrap();
    let vpp = VirtualPipelineCoordinator::new(vpp_plan, 16, 2, 16);
    let transport = TimedVppTransport::new(InprocVppTransport::mesh(2), 2);
    let started = Instant::now();
    vpp.forward_vpp_multi_rank(&transport, chunks, dense_layer)
        .expect("multi-rank forward");
    let vpp_total = started.elapsed();
    let recv_wait = transport.recv_wait_snapshot();

    let wall = vpp_total.as_secs_f64();
    println!("\n=== VPP bubble-ratio benchmark (dual ROCm GPU, synthetic layers) ===");
    println!("chunks={num_chunks} rows={ROWS} hidden={HIDDEN} layers={LAYERS}");
    println!("single-rank total: {:.2?}", single_total);
    println!("2-rank VPP total:  {:.2?}", vpp_total);
    println!(
        "speedup: {:.2}x",
        single_total.as_secs_f64() / vpp_total.as_secs_f64()
    );
    for (rank, waited) in recv_wait.iter().enumerate() {
        println!(
            "rank {rank} bubble ratio (recv-block share): {:.1}%",
            waited.as_secs_f64() / wall * 100.0
        );
    }
    println!(
        "note: synthetic layer cost; wire through Engine + real 512K prefill for headline numbers"
    );

    // The schedule is async head-first: neither rank may sit blocked for the
    // whole run — a ~100% bubble on either rank means the overlap is broken.
    for (rank, waited) in recv_wait.iter().enumerate() {
        let bubble_pct = waited.as_secs_f64() / wall * 100.0;
        assert!(
            bubble_pct < 95.0,
            "rank {rank} bubble ratio {bubble_pct:.1}% — VPP-Async overlap is not engaging"
        );
    }
}

/// Cross-process shape check: the same schedule over the TCP transport must
/// produce chunk outputs identical to the single-rank path. Runs on CPU
/// (loopback), no GPU gate — it proves the multi-node transport path stays
/// bit-exact without owning hardware.
#[test]
fn vpp_benchmark_tcp_parity_smoke() {
    let plan = VirtualPipelinePlan::plan(LAYERS, 2, &[0, 1]).unwrap();
    let vpp = VirtualPipelineCoordinator::new(plan, 16, 2, 16);
    let chunks = chunk_inputs(2);

    let single_plan = VirtualPipelinePlan::plan(LAYERS, 1, &[0]).unwrap();
    let single = VirtualPipelineCoordinator::new(single_plan, 16, 2, 16);
    let expected: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| {
            single
                .forward_vpp(c.clone(), dense_layer)
                .unwrap()
                .to_vec_f32()
                .unwrap()
        })
        .collect();

    let mut tcp = TcpActivationTransport::bind(2).expect("bind");
    tcp.set_peer(0, tcp.local_addr(0).unwrap());
    tcp.set_peer(1, tcp.local_addr(1).unwrap());

    let actual = vpp
        .forward_vpp_multi_rank(&TcpVppTransport(tcp), chunks, dense_layer)
        .expect("tcp forward");

    for (chunk, want) in expected.iter().enumerate() {
        let got = actual[chunk].to_vec_f32().unwrap();
        assert_eq!(want, &got[..], "tcp chunk {chunk} must match single-rank");
    }
}
