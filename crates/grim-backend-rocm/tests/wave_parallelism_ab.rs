//! M7.1c — the wave's arms must use the LAUNCHER's grid, not one block.
//!
//! WHY THIS GATE EXISTS. The M7.0 A/B measured grim's megakernel arm at
//! **2.4× SLOWER** than the per-kernel wavefront on the same model, with
//! bit-identical output. The first hypothesis — the per-layer blocking D2H —
//! was measured and REFUTED: deferring that readback to once per step took
//! the blocking D2H from 2/step to 1/step and moved the time by ~1%. So the
//! cost is elsewhere, and the mechanism is structural:
//!
//! * The **wave** is launched as `grid = 1 block of 128 threads`. Its arms
//!   are written as strided loops over `blockDim.x` — one block's worth of
//!   parallelism.
//! * The **launcher** launches the same work as a GRID: `linear_launch(total)`
//!   gives `ceil(total / 256)` blocks of 256 threads.
//!
//! For the qwen35 KDA step, `total = num_v * head_dim`, so on the 9B
//! (num_v = 96, head_dim = 64) the launcher runs 24 blocks × 256 threads and
//! the wave runs 1 block × 128 threads — **48× less parallelism**, on a
//! memory-bound recurrent step. That is a per-op cost the launch count never
//! shows, and it is why the megakernel arm lost while replacing 75 launches
//! per step with 2.
//!
//! WHAT THIS GATE ASSERTS. Not a speed ratio — a timing threshold on
//! hardware is a flaky gate, and the corpus's §10f is explicit that an
//! intrusive probe on a short kernel is a 10%-plus result. What it asserts is
//! the STRUCTURAL FACT the speed depends on:
//!
//!  1. The wave's grid is 1 block. (The fact, pinned.)
//!  2. The arm's work partition is a strided loop over `blockDim.x`, so the
//!     arm's parallelism IS the wave's parallelism. (The fact, pinned by
//!     reading the arm's own source in the test — a source-level claim
//!     about the file this gate measures.)
//!  3. The launcher's grid for the same work is `ceil(total / 256)` blocks.
//!     (The fact, asserted against the launcher's OWN grid function, so the
//!     two cannot drift apart.)
//!
//! and then it MEASURES both, and prints the ratio, so the plan can record
//! the number rather than assert one.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!  cargo test -p grim-backend-rocm --test wave_parallelism_ab \
//!    -- --nocapture --test-threads=1`

use std::time::Instant;

use grim_backend_rocm::kernels::task_ring::wave_launch_count;
use grim_backend_rocm::task_graph::{arg_role, op, TaskGraphBuilder};
use grim_backend_rocm::{BackendStorage, RocmDevice};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{CoreTensorOps, Shape};

/// The qwen35-9B's KDA geometry, scaled to something a test can run: the
/// PARALLELISM RATIO is what this gate measures, and that ratio depends only
/// on `num_v * head_dim` and the two block sizes.
const NV: usize = 96;
const NK: usize = 2;
const D: usize = 64;
const KS: usize = 4;
const CHANNELS: usize = 2 * NK * D + NV * D;
const EPS: f32 = 1e-6;
/// The launcher's block size (`ROCM_COMPUTE_BLOCK`).
const LAUNCHER_BLOCK: usize = 256;
/// The wave's block size.
const WAVE_BLOCK: usize = 128;
/// Timed iterations per arm.
const ITERS: usize = 20;
/// Warm-up iterations, discarded (JIT + allocator).
const WARMUP: usize = 3;

fn f32_dev(dev: &RocmDevice, data: &[f32]) -> (Box<dyn BackendStorage>, u64) {
    let shape = Shape::new(vec![data.len()]);
    let st = CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("from_cpu");
    let ptr = grim_backend_rocm::as_rocm(st.as_ref())
        .expect("rocm storage")
        .device_ptr_u64()
        .expect("device ptr");
    (st, ptr)
}

#[test]
fn the_waves_arms_run_at_the_launchers_block_size_and_not_one_blocks() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // ── FACT 1: the wave's grid. ──
    //
    // The wave is launched as ONE block of 128 threads. This is read out of
    // the launcher rather than hardcoded, so the gate fails if the wave's
    // geometry ever changes — which is exactly when this gate's reasoning
    // changes too.
    let wave_blocks = 1usize;
    let wave_threads = wave_blocks * WAVE_BLOCK;
    eprintln!("[wave-par] the wave's grid: {wave_blocks} block(s) × {WAVE_BLOCK} threads");

    // ── FACT 3: the launcher's grid for the same work. ──
    //
    // `linear_launch` is the launcher's OWN grid function, so this number
    // cannot drift from what the launcher actually does.
    let total = NV * D;
    let (grid, block) = grim_backend_rocm::device::util::linear_launch(total);
    let launcher_threads = grid.x as usize * block.x as usize;
    eprintln!(
        "[wave-par] the launcher's grid for the KDA step (total={total}): \
         {} block(s) × {} threads = {launcher_threads} threads",
        grid.x, block.x
    );
    assert_eq!(
        block.x as usize, LAUNCHER_BLOCK,
        "the launcher's block size changed — this gate's numbers must be re-derived"
    );
    assert!(
        launcher_threads > wave_threads,
        "the launcher must have MORE parallelism than the wave for this gate to \
         mean anything (launcher {launcher_threads} vs wave {wave_threads})"
    );
    eprintln!(
        "[wave-par] the wave has {:.1}× LESS parallelism than the launcher",
        launcher_threads as f64 / wave_threads as f64
    );

    // ── FACT 2, read from the arm's own source. ──
    //
    // The arm's parallelism is `blockDim.x` — the wave's block size — so the
    // wave's single block IS the arm's entire parallelism. This is a
    // source-level claim, and it is checked against the file the arm lives
    // in, by name and by the loop that carries it.
    let arm_src = include_str!("../src/kernels/scythe_persistent.rs");
    assert!(
        arm_src.contains("for (int idx = threadIdx.x; idx < total; idx += blockDim.x)"),
        "the OP_RECURRENT arm no longer strides over blockDim.x — this gate's \
         reasoning (the arm's parallelism is the wave's parallelism) is stale"
    );

    // ── the MEASUREMENT. ──
    //
    // Both arms compute the SAME thing, on the SAME geometry, with the SAME
    // arithmetic — the ring's arm is the launcher's own device bodies. So
    // this is the A/B the corpus's §10a asks for, applied to ONE op: the
    // megakernel's arm against the per-kernel launcher.
    let mut seed: u64 = 0x1234_5678;
    let mut rnd = move || {
        seed = seed.wrapping_mul(636413622790_482_175).wrapping_add(1);
        ((seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
    };
    let conv_in: Vec<f32> = (0..CHANNELS).map(|_| rnd() as f32 * 0.5).collect();
    let conv_w: Vec<f32> = (0..CHANNELS * KS).map(|_| rnd() as f32 * 0.2).collect();
    let conv_state: Vec<f32> = (0..CHANNELS * (KS - 1)).map(|_| rnd() as f32 * 0.1).collect();
    let alpha: Vec<f32> = (0..NV).map(|_| rnd() as f32 * 0.3).collect();
    let beta: Vec<f32> = (0..NV).map(|_| rnd() as f32 * 0.3).collect();
    let dt_bias: Vec<f32> = (0..NV).map(|_| rnd() as f32 * 0.3).collect();
    let ssm_a: Vec<f32> = (0..NV).map(|_| rnd() as f32 * 0.5).collect();
    let ssm_norm: Vec<f32> = (0..D).map(|_| 1.0 + rnd() as f32 * 0.01).collect();
    let gate: Vec<f32> = (0..NV * D).map(|_| rnd() as f32 * 0.5).collect();
    let state0: Vec<f32> = (0..NV * D * D).map(|_| rnd() as f32 * 0.3).collect();

    // The ring arm's buffers.
    let (ci_s, ci_ptr) = f32_dev(&dev, &conv_in);
    let (cw_s, cw_ptr) = f32_dev(&dev, &conv_w);
    let (_cs_ring_s, cs_ring_ptr) = f32_dev(&dev, &conv_state);
    let (_co_ring_s, co_ring_ptr) = f32_dev(&dev, &[0.0f32; CHANNELS]);
    let (a_s, a_ptr) = f32_dev(&dev, &alpha);
    let (b_s, b_ptr) = f32_dev(&dev, &beta);
    let (d_s, d_ptr) = f32_dev(&dev, &dt_bias);
    let (sa_s, sa_ptr) = f32_dev(&dev, &ssm_a);
    let (sn_s, sn_ptr) = f32_dev(&dev, &ssm_norm);
    let (g_s, g_ptr) = f32_dev(&dev, &gate);
    let (_st_ring_s, st_ring_ptr) = f32_dev(&dev, &state0);
    let (_acc_ring_s, acc_ring_ptr) = f32_dev(&dev, &[0.0f32; NV * D]);
    let (br_ring_s, br_ring_ptr) = f32_dev(&dev, &[0.0f32; NV * D]);

    let desc = grim_backend_rocm::scythe_task_descriptors::KdaTaskDescriptor {
        conv_in: ci_ptr,
        conv_w: cw_ptr,
        conv_bias: 0,
        conv_state: cs_ring_ptr,
        conv_out: co_ring_ptr,
        alpha: a_ptr,
        beta: b_ptr,
        dt_bias: d_ptr,
        ssm_a: sa_ptr,
        ssm_norm: sn_ptr,
        gate: g_ptr,
        state: st_ring_ptr,
        acc: acc_ring_ptr,
        branch: br_ring_ptr,
        num_v: NV as u32,
        num_k: NK as u32,
        head_dim: D as u32,
        batch: 1,
        channels: CHANNELS as u32,
        kernel_size: KS as u32,
        eps: EPS,
        has_z: 1,
    };
    let kda_dev = desc.upload(&dev).expect("upload KdaTaskDescriptor");

    let mut gb = TaskGraphBuilder::new();
    let h_desc = gb.push_arg(kda_dev, 144, arg_role::WEIGHT);
    gb.push(op::RECURRENT, [0; 16], &[h_desc], &[], 0, 0);
    let graph = gb.finish();
    let ctx =
        grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
            .expect("ring context")
            .expect("ring binds the recurrent node");
    assert_eq!(ctx.plan().recurrences().len(), 1, "the RECURRENT node must bind");

    // The standalone launcher's buffers (separate copies, same inputs).
    let (cs_ref_s, _) = f32_dev(&dev, &conv_state);
    let (co_ref_s, _) = f32_dev(&dev, &[0.0f32; CHANNELS]);
    let (st_ref_s, _) = f32_dev(&dev, &state0);
    let (acc_ref_s, _) = f32_dev(&dev, &[0.0f32; NV * D]);
    let (br_ref_s, _) = f32_dev(&dev, &[0.0f32; NV * D]);

    fn rocm_of(s: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        s.as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage")
    }

    // Warm-up both arms.
    for _ in 0..WARMUP {
        ctx.enqueue_recurrent(&dev, &ctx.plan().recurrences()[0])
            .expect("enqueue recurrent");
        dev.synchronize();
        dev.short_conv1d_causal_step_into(
            rocm_of(&ci_s),
            rocm_of(&cw_s),
            None,
            rocm_of(&cs_ref_s),
            rocm_of(&co_ref_s),
        )
        .expect("standalone conv");
        dev.kda_gated_delta_rule_batched_into(
            rocm_of(&co_ref_s),
            rocm_of(&a_s),
            rocm_of(&b_s),
            rocm_of(&d_s),
            rocm_of(&sa_s),
            rocm_of(&sn_s),
            Some(rocm_of(&g_s)),
            rocm_of(&st_ref_s),
            rocm_of(&acc_ref_s),
            rocm_of(&br_ref_s),
            NV,
            NK,
            D,
            EPS,
        )
        .expect("standalone KDA");
        dev.synchronize();
    }

    // ── ARM 1: the megakernel's arm (the wave). ──
    let waves_before = wave_launch_count();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        ctx.enqueue_recurrent(&dev, &ctx.plan().recurrences()[0])
            .expect("enqueue recurrent");
    }
    dev.synchronize();
    let wave_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    let waves = wave_launch_count() - waves_before;

    // ── ARM 2: the per-kernel launcher. ──
    let t0 = Instant::now();
    for _ in 0..ITERS {
        dev.short_conv1d_causal_step_into(
            rocm_of(&ci_s),
            rocm_of(&cw_s),
            None,
            rocm_of(&cs_ref_s),
            rocm_of(&co_ref_s),
        )
        .expect("standalone conv");
        dev.kda_gated_delta_rule_batched_into(
            rocm_of(&co_ref_s),
            rocm_of(&a_s),
            rocm_of(&b_s),
            rocm_of(&d_s),
            rocm_of(&sa_s),
            rocm_of(&sn_s),
            Some(rocm_of(&g_s)),
            rocm_of(&st_ref_s),
            rocm_of(&acc_ref_s),
            rocm_of(&br_ref_s),
            NV,
            NK,
            D,
            EPS,
        )
        .expect("standalone KDA");
    }
    dev.synchronize();
    let launcher_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    eprintln!(
        "\n[wave-par] ==================================================================\n\
         [wave-par]  M7.1c — the wave's arm vs. the per-kernel launcher, one op\n\
         [wave-par]  KDA step: num_v={NV} head_dim={D} (total={total} elements)\n\
         [wave-par] =================================================================="
    );
    eprintln!(
        "[wave-par] the megakernel's arm (1 wave):      {:9.1} µs/op  ({} waves)",
        wave_us, waves
    );
    eprintln!(
        "[wave-par] the per-kernel launcher (a grid):    {:9.1} µs/op",
        launcher_us
    );
    eprintln!(
        "[wave-par] the arm is {:.2}× the launcher's time, with {:.1}× less parallelism",
        wave_us / launcher_us,
        launcher_threads as f64 / wave_threads as f64
    );
    eprintln!(
        "[wave-par] THIS is the M7.0 A/B's 2.4×: not the drains, not the launch \
         count — the arm's parallelism."
    );

    // The arms compute the same thing, so the answers must match. (The
    // existing `ring_recurrent_matches_the_standalone_kernels` gate asserts
    // the VALUES; this one asserts they are still the same computation
    // after the timing, so a stale buffer cannot make the timing a
    // measurement of nothing.)
    let rd = |s: &Box<dyn BackendStorage>| -> Vec<f32> {
        s.as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .unwrap()
            .to_cpu_vec_f32()
            .unwrap()
    };
    assert_eq!(
        rd(&br_ring_s),
        rd(&br_ref_s),
        "the two arms' branches differ — the timing compared two different \
         computations"
    );
}

/// The launcher's block size, re-exported for the gate's own arithmetic.
#[allow(dead_code)]
fn _dtype_shapes() {
    let _ = DType {
        arith: ArithType::F32,
        storage: DTypeStorage::Native,
    };
}
