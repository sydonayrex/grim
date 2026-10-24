//! M7.2 — the five candidate wave arms, MEASURED against the launcher.
//!
//! WHY THIS GATE EXISTS. The plan's M7.2 is "the wave arms for the launches
//! M7.1 named — the RoPE first, then the KV append, then the quantize, the
//! copies and the sigmoid-gate. Each arm is the launcher's own body, and
//! each gets a gate that the arm EQUALS the launcher (not 'close to')."
//!
//! But M7.0's A/B and M7.1's two measurements changed the question. The A/B
//! said the megakernel arm is 2.4× SLOWER on a hybrid-recurrent model; the
//! two follow-ups said WHY:
//!
//!  * M7.1c — the wave's arms run in ONE block of 128 threads, the
//!    launcher's in a grid of `total/256` blocks. The parallelism the wave
//!    gives up GROWS with the op (2× at 256 elements, 512× at 65536).
//!  * M7.1d — the wave's per-node dispatch (a descriptor write plus a head
//!    publish) costs MORE than the ONE launch it eliminates: 1.73–1.95× per
//!    op even fully amortised.
//!
//! So the arm's correctness gate ("the arm EQUALS the launcher") is
//! NECESSARY and NOT SUFFICIENT: an arm can be bit-identical and still be
//! slower, and a phase that lands five such arms makes the engine slower.
//! The corpus's §10a makes the same point about the one direct A/B in the
//! literature: "the losing megakernel has the HIGHER compute utilisation.
//! Higher SM%/RTCore% did not mean more work done."
//!
//! THIS GATE MEASURES EACH CANDIDATE ARM'S REAL GEOMETRY, so the decision
//! to land an arm is made on a number rather than on an assumption. The five
//! arms' shapes are the qwen35-9B's:
//!
//!  | arm         | elements                                     |
//!  |-------------|----------------------------------------------|
//!  | RoPE        | batch * heads * head_dim / 2 = 1 * 24 * 128/2 |
//!  | sigmoid-gate| batch * heads * head_dim = 1 * 24 * 128        |
//!  | copies      | the same, per slice                          |
//!  | quantize    | batch * hidden = 1 * 5120                     |
//!  | KV append   | batch * kv_dim = 1 * 4 * 128 = 512            |
//!
//! and the launcher is `linear_launch(total)` — a grid — for every one of
//! them. The measurement is the WAVE's per-op cost against the LAUNCHER's
//! per-op cost, at each shape, and it is the wave's ALREADY-MEASURED
//! dispatch cost plus the shape's own work.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!  cargo test -p grim-backend-rocm --test wave_arm_candidates_ab \
//!    -- --nocapture --test-threads=1`

use std::time::Instant;

use grim_backend_rocm::kernels::task_ring::TaskRingPlan;
use grim_backend_rocm::task_graph::{arg_role, op, TaskGraphBuilder};
use grim_backend_rocm::{BackendStorage, RocmDevice, as_rocm};
use grim_tensor::dtype::DType;
use grim_tensor::{CoreTensorOps, Shape};

/// The qwen35-9B's geometry, which is the geometry the arms would run at.
const HEADS: usize = 24;
const HEAD_DIM: usize = 128;
const KV_HEADS: usize = 4;
const HIDDEN: usize = 5120;

/// Timed iterations per arm.
const ITERS: usize = 50;
/// Warm-up iterations, discarded.
const WARMUP: usize = 5;

/// One candidate arm: a name and the number of elements it touches.
struct Arm {
    name: &'static str,
    elems: usize,
}

fn arms() -> Vec<Arm> {
    vec![
        Arm {
            name: "RoPE (per head, half width)",
            elems: HEADS * HEAD_DIM / 2,
        },
        Arm {
            name: "sigmoid-gate (attention out)",
            elems: HEADS * HEAD_DIM,
        },
        Arm {
            name: "copies (per slice)",
            elems: HEADS * HEAD_DIM,
        },
        Arm {
            name: "quantize (activation)",
            elems: HIDDEN,
        },
        Arm {
            name: "KV append (per tensor)",
            elems: KV_HEADS * HEAD_DIM,
        },
    ]
}

fn f32_dev(dev: &RocmDevice, data: &[f32]) -> (Box<dyn BackendStorage>, u64) {
    let shape = Shape::new(vec![data.len()]);
    let st = CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("from_cpu");
    let ptr = as_rocm(st.as_ref())
        .expect("rocm storage")
        .device_ptr_u64()
        .expect("device ptr");
    (st, ptr)
}

#[test]
fn the_candidate_arms_are_measured_against_the_launcher() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    eprintln!(
        "\n[arm-ab] ==================================================================\n\
         [arm-ab]  M7.2 — the candidate wave arms vs. the launcher\n\
         [arm-ab]  the qwen35-9B's geometry: {HEADS} heads x {HEAD_DIM} head_dim, \
         {KV_HEADS} kv heads, {HIDDEN} hidden\n\
         [arm-ab] =================================================================="
    );
    eprintln!(
        "[arm-ab] {:<32} {:>8} {:>14} {:>16} {:>9}",
        "arm", "elems", "launcher µs", "wave µs", "wave/launch"
    );

    let mut verdicts: Vec<(&str, bool)> = Vec::new();
    for arm in arms() {
        let n = arm.elems;
        let a_data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.11 - 0.9).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.07 - 0.75).collect();
        let (a_s, a_ptr) = f32_dev(&dev, &a_data);
        let (b_s, b_ptr) = f32_dev(&dev, &b_data);
        let (o_s, o_ptr) = f32_dev(&dev, &vec![0.0f32; n]);

        // The ring's plan: ONE add node over the arm's shape. The add is the
        // PROXY for the arm, and the proxy is honest here: every one of the
        // five arms is a grid-launched elementwise pass over the same kind of
        // element count, so the arm's cost is the SAME cost the add's is —
        // the dispatch plus the work. What differs between the arms is the
        // WORK's size, which is what the sweep varies.
        let mut gb = TaskGraphBuilder::new();
        let h_a = gb.push_arg(a_ptr, (n * 4) as u64, arg_role::ACT_IN);
        let h_b = gb.push_arg(b_ptr, (n * 4) as u64, arg_role::ACT_IN);
        let h_o = gb.push_arg(o_ptr, (n * 4) as u64, arg_role::ACT_OUT);
        gb.push(
            op::ELEMENTWISE,
            [n as u32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[h_a, h_b, h_o],
            &[],
            0,
            0,
        );
        let graph = gb.finish();
        let ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
            .expect("ring context")
            .expect("ring binds the add");
        let plan = TaskRingPlan {
            gemms: Vec::new(),
            norms: Vec::new(),
            adds: ctx.plan().adds().to_vec(),
            attns: Vec::new(),
            moes: Vec::new(),
            peers: Vec::new(),
            recurrences: Vec::new(),
            q8_dot4s: Vec::new(),
        };
        let (staged, count) = plan.stage_on_device(&dev).expect("stage plan");
        assert_eq!(count as usize, 1, "the one descriptor staged");

        // ── The LAUNCHER: one grid launch over the arm's shape. ──
        for _ in 0..WARMUP {
            dev.add_into(
                as_rocm(a_s.as_ref()).expect("a"),
                as_rocm(b_s.as_ref()).expect("b"),
                as_rocm(o_s.as_ref()).expect("o"),
            )
            .expect("launcher add");
        }
        dev.synchronize();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            dev.add_into(
                as_rocm(a_s.as_ref()).expect("a"),
                as_rocm(b_s.as_ref()).expect("b"),
                as_rocm(o_s.as_ref()).expect("o"),
            )
            .expect("launcher add");
        }
        dev.synchronize();
        let launcher_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        // ── The WAVE: the staged set, one wave for the whole set. ──
        //
        // The STAGED path is the only one that can amortise — M7.1d measured
        // the per-node path (write + publish per node) at 1.73-1.95× the
        // launcher's launch, and M7.1e measured the staged path converging
        // to ~1.9× as the set grows. So the staged path is the BEST case for
        // the wave, and this measures it at the arms' real shapes.
        let mut res_ctx = grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(
            &dev, &graph,
        )
        .expect("ring context")
        .expect("ring binds the add");
        res_ctx.launch_resident(&dev).expect("launch resident wave");
        for _ in 0..WARMUP {
            res_ctx
                .launch_staged_on_stream(&dev, dev.active_stream())
                .expect("launch staged");
            dev.synchronize();
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            res_ctx
                .launch_staged_on_stream(&dev, dev.active_stream())
                .expect("launch staged");
        }
        dev.synchronize();
        let wave_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        let _ = res_ctx.stop_wave(&dev);
        let _ = staged;

        let ratio = wave_us / launcher_us;
        verdicts.push((arm.name, ratio < 1.0));
        eprintln!(
            "[arm-ab] {:<32} {:>8} {:>14.1} {:>16.1} {:>9.2}",
            arm.name, n, launcher_us, wave_us, ratio
        );
    }

    eprintln!(
        "[arm-ab] the wave's arm is the launcher's own body, so the arm's cost is \
         the wave's DISPATCH plus the shape's WORK — and the dispatch is measured \
         at 1.73-1.95x the launcher's launch (M7.1d)."
    );
    let winners: Vec<&str> = verdicts
        .iter()
        .filter(|(_, wins)| *wins)
        .map(|(name, _)| *name)
        .collect();
    if winners.is_empty() {
        eprintln!(
            "[arm-ab] NO candidate arm is faster than the launcher at its real \
             geometry. Landing them would make the engine slower, so the \
             honest verdict is that M7.2's arms are not landed — and the \
             lever is the DISPATCH, not the arm count."
        );
    } else {
        eprintln!(
            "[arm-ab] the arms that beat the launcher at their real geometry: \
             {winners:?}"
        );
    }
}
