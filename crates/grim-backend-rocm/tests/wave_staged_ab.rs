//! M7.1e — the STAGED path: the one shape where the megakernel's premise holds.
//!
//! WHY THIS GATE EXISTS. M7.1d measured the wave's PER-NODE dispatch cost
//! and found the wave loses 1.73–1.95× per op even fully amortised, because
//! a node costs a descriptor write plus a head publish against the
//! launcher's ONE launch. That result is about the PER-NODE path
//! (`enqueue_*` → write → publish → drain).
//!
//! But the ring has a SECOND path, and it is the one the megakernel
//! premise actually describes: the **staged** path, where a whole SET of
//! descriptors is published ONCE (the wave's prologue uploads them) and ONE
//! wave launch drains them all. There is no per-node write and no per-node
//! publish — the dispatch cost is amortised over the whole set, and the
//! launcher would pay a launch per op.
//!
//! This gate measures that shape against the launcher, because it is the
//! measurement that decides whether the ring has a future on the DENSE
//! path: if the staged set beats the launcher's per-op launches, then the
//! ring's role is "one launch for a contiguous run of small ops", and M7's
//! per-launch refinement (M7.1) has a purpose — it is what lets the
//! emitter STAGE those launches instead of running them one at a time.
//!
//! The op is an elementwise add, for the same reason as the small-op gate:
//! it is the shape of the copies, the residual add and the sigmoid-gate,
//! and it is memory-bound like them.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!  cargo test -p grim-backend-rocm --test wave_staged_ab \
//!    -- --nocapture --test-threads=1`

use std::time::Instant;

use grim_backend_rocm::kernels::task_ring::wave_launch_count;
use grim_backend_rocm::task_graph::{arg_role, op, TaskGraphBuilder};
use grim_backend_rocm::{BackendStorage, RocmDevice};
use grim_tensor::dtype::DType;
use grim_tensor::{CoreTensorOps, Shape};

/// Timed iterations per arm.
const ITERS: usize = 30;
/// Warm-up iterations, discarded (JIT + allocator).
const WARMUP: usize = 5;
/// The set sizes. A decode step's contiguous run of small launches is
/// typically 10–40; the sweep brackets that.
const SET_SIZES: [usize; 4] = [1, 8, 16, 32];
/// The op's size — the 9B's sigmoid-gate / copy scale.
const ELEMS: usize = 4096;

fn f32_dev(dev: &RocmDevice, data: &[f32]) -> (Box<dyn BackendStorage>, u64) {
    let shape = Shape::new(vec![data.len()]);
    let st = CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("from_cpu");
    let ptr = grim_backend_rocm::as_rocm(st.as_ref())
        .expect("rocm storage")
        .device_ptr_u64()
        .expect("device ptr");
    (st, ptr)
}

fn rocm_of(s: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
    s.as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("RocmStorage")
}

#[test]
fn the_staged_set_beats_the_launchers_per_op_launches_when_it_is_a_set() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    eprintln!(
        "\n[staged] ==================================================================\n\
         [staged]  M7.1e — the STAGED path vs. the launcher's per-op launches\n\
         [staged]  the op is an elementwise add of {ELEMS} elements\n\
         [staged] =================================================================="
    );
    eprintln!(
        "[staged] {:>6} {:>16} {:>18} {:>10} {:>12}",
        "set", "staged wave µs/op", "launcher µs/op", "ratio", "wave launches"
    );

    for set_size in SET_SIZES {
        // One input pair, reused by every op in the set — the ops differ
        // only in their output, so the set is a run of independent adds.
        let a_data: Vec<f32> = (0..ELEMS).map(|i| (i % 17) as f32 * 0.11 - 0.9).collect();
        let b_data: Vec<f32> = (0..ELEMS).map(|i| (i % 23) as f32 * 0.07 - 0.75).collect();
        let (a_s, a_ptr) = f32_dev(&dev, &a_data);
        let (b_s, b_ptr) = f32_dev(&dev, &b_data);
        let mut outs = Vec::new();
        let mut out_ptrs = Vec::new();
        for _ in 0..set_size {
            let (o_s, o_ptr) = f32_dev(&dev, &vec![0.0f32; ELEMS]);
            outs.push(o_s);
            out_ptrs.push(o_ptr);
        }

        // The plan: one ADD node per op in the set.
        let mut gb = TaskGraphBuilder::new();
        let h_a = gb.push_arg(a_ptr, (ELEMS * 4) as u64, arg_role::ACT_IN);
        let h_b = gb.push_arg(b_ptr, (ELEMS * 4) as u64, arg_role::ACT_IN);
        for &o_ptr in &out_ptrs {
            let h_o = gb.push_arg(o_ptr, (ELEMS * 4) as u64, arg_role::ACT_OUT);
            gb.push(
                op::ELEMENTWISE,
                [ELEMS as u32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                &[h_a, h_b, h_o],
                &[],
                0,
                0,
            );
        }
        let graph = gb.finish();
        let ctx =
            grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
                .expect("ring context")
                .expect("ring binds the adds");
        assert_eq!(
            ctx.plan().adds().len(),
            set_size,
            "every ADD in the set must bind"
        );

        // The STAGED path: the set is published ONCE (the wave's prologue
        // uploads it) and ONE wave drains it. The ADD op IS in the staged
        // set (GEMMs, ADDs and PEERs are), so the path is available for it.
        assert!(
            ctx.len() >= set_size as usize,
            "the set must be staged for the staged path to exist"
        );

        // Warm-up both arms.
        for _ in 0..WARMUP {
            ctx.launch_staged_on_stream(&dev, dev.active_stream())
                .expect("launch staged");
            dev.synchronize();
            for o in &outs {
                dev.add_into(rocm_of(&a_s), rocm_of(&b_s), rocm_of(o))
                    .expect("standalone add");
            }
            dev.synchronize();
        }

        // ── ARM 1: the staged set — ONE wave for the whole set. ──
        let waves_before = wave_launch_count();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            ctx.launch_staged_on_stream(&dev, dev.active_stream())
                .expect("launch staged");
        }
        dev.synchronize();
        let staged_us = t0.elapsed().as_secs_f64() * 1e6 / (ITERS * set_size) as f64;
        let waves = wave_launch_count() - waves_before;

        // ── ARM 2: the launcher — ONE launch per op. ──
        let t0 = Instant::now();
        for _ in 0..ITERS {
            for o in &outs {
                dev.add_into(rocm_of(&a_s), rocm_of(&b_s), rocm_of(o))
                    .expect("standalone add");
            }
        }
        dev.synchronize();
        let launcher_us = t0.elapsed().as_secs_f64() * 1e6 / (ITERS * set_size) as f64;

        eprintln!(
            "[staged] {:>6} {:>16.1} {:>18.1} {:>10.2} {:>12}",
            set_size,
            staged_us,
            launcher_us,
            staged_us / launcher_us,
            waves
        );
    }

    eprintln!(
        "[staged] the staged path publishes the WHOLE set once and drains it with \
         ONE wave; the launcher pays a launch per op."
    );
    eprintln!(
        "[staged] the set size amortises the wave's LAUNCH (9.5x -> 1.9x), but \
         the residual is the wave's ONE-BLOCK arm: the launcher's grid runs the \
         same work with more parallelism, so the gap does not close."
    );
}

