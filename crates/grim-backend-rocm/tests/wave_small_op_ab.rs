//! M7.1d — does the megakernel's wave win ANY op?
//!
//! WHY THIS GATE EXISTS. The M7.0 A/B measured grim's megakernel arm at
//! **2.4× SLOWER** than the per-kernel wavefront on a hybrid-recurrent model,
//! with bit-identical output. The M7.1c sweep then measured the mechanism
//! down to the arm's PARALLELISM: the wave runs its arms in ONE block of 128
//! threads, the launcher in a grid of `total/256` blocks, so the wave loses at
//! every size and loses WORSE as the op grows (1.95× at 384 elements,
//! 13.86× at 6144).
//!
//! But the KDA step is a COMPUTE-heavy op. The ops M7.2 actually wants the
//! wave to take — the RoPE, the KV append, the quantize, the copies, the
//! sigmoid-gate — are MEMORY-bound and SMALL, and for those the launch
//! overhead the wave removes can be the larger term. So the question this
//! gate answers is the one that decides M7.2's arm list:
//!
//! **For a small, memory-bound op, is the wave's ONE launch + one block
//! cheaper than the launcher's ONE launch + a grid?**
//!
//! THE ANSWER, AND IT IS NOT THE OBVIOUS ONE. The wave's per-node cost is
//! not one launch: it is a descriptor WRITE (a kernel), a head PUBLISH (a
//! kernel), the wave's own LAUNCH (a kernel), the wave's poll-and-claim
//! round trip, and the wave's exit. The launcher pays ONE launch. So for a
//! single op the wave pays about three launches where the launcher pays one,
//! and it recovers them only by BATCHING many ops into one wave — which is
//! the whole point of a resident wave, and which the qwen35 path does not
//! currently do (it drains one node at a time).
//!
//! This gate measures the per-op cost of BOTH shapes, at the op sizes M7.2's
//! arms would actually have:
//!
//!  * ONE op through the wave (the current qwen35 shape: write, publish,
//!    launch a wave, drain) against ONE op through the launcher.
//!  * A BATCH of ops through ONE wave against the same batch through the
//!    launcher — the shape a resident wave would give.
//!
//! and it prints both. The second number is the one that decides whether
//! the megakernel direction has a future on this workload.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!  cargo test -p grim-backend-rocm --test wave_small_op_ab \
//!    -- --nocapture --test-threads=1`

use std::time::Instant;

use grim_backend_rocm::task_graph::{TaskGraphBuilder, arg_role, op};
use grim_backend_rocm::{BackendStorage, RocmDevice};
use grim_tensor::dtype::DType;
use grim_tensor::{CoreTensorOps, Shape};

/// The batch size for the amortised comparison.
const BATCH: usize = 16;
/// Timed iterations per arm.
const ITERS: usize = 30;
/// Warm-up iterations, discarded (JIT + allocator).
const WARMUP: usize = 5;
/// The op sizes M7.2's arms would have. The 9B's numbers:
/// the RoPE is over `batch * heads * head_dim` = 1 * 24 * 256 = 6144;
/// the sigmoid-gate over the attention output = 6144; the quantize over
/// `batch * hidden` = 5120; the KV append over `batch * kv_dim` = 1024;
/// the copies over the same. So the range that matters is 1K–8K elements,
/// and the sweep covers it.
const SIZES: [usize; 5] = [1024, 2048, 4096, 6144, 8192];

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
fn the_waves_per_op_cost_is_measured_against_the_launchers() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let dev = RocmDevice::try_new(0).expect("try_new");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    eprintln!(
        "\n[small-op] ==================================================================\n\
         [small-op]  M7.1d — the wave's per-op cost vs. the launcher's\n\
         [small-op]  the op is an elementwise add (the shape of the copies,\n\
         [small-op]  the residual add and the sigmoid-gate)\n\
         [small-op] =================================================================="
    );
    eprintln!(
        "[small-op] {:>8} {:>12} {:>12} {:>9} {:>12} {:>12} {:>9} {:>12} {:>9}",
        "elems",
        "1 wave µs",
        "1 launch µs",
        "ratio",
        "batch wave µs",
        "batch launch µs",
        "ratio",
        "RESIDENT µs",
        "ratio"
    );

    for n in SIZES {
        let a_data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.11 - 0.9).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.07 - 0.75).collect();
        let (a_s, a_ptr) = f32_dev(&dev, &a_data);
        let (b_s, b_ptr) = f32_dev(&dev, &b_data);
        let (o_s, o_ptr) = f32_dev(&dev, &vec![0.0f32; n]);

        // The ring's plan: ONE add node, m = n, k = 1.
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
        assert_eq!(ctx.plan().adds().len(), 1, "the ADD node must bind");

        // Warm-up both arms.
        for _ in 0..WARMUP {
            ctx.enqueue_add(&dev, &ctx.plan().adds()[0])
                .expect("enqueue add");
            dev.synchronize();
            dev.add_into(rocm_of(&a_s), rocm_of(&b_s), rocm_of(&o_s))
                .expect("standalone add");
            dev.synchronize();
        }

        // ── ONE op: the wave (write + publish + launch a wave + drain). ──
        let t0 = Instant::now();
        for _ in 0..ITERS {
            ctx.enqueue_add(&dev, &ctx.plan().adds()[0])
                .expect("enqueue add");
        }
        dev.synchronize();
        let one_wave_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        // ── ONE op: the launcher. ──
        let t0 = Instant::now();
        for _ in 0..ITERS {
            dev.add_into(rocm_of(&a_s), rocm_of(&b_s), rocm_of(&o_s))
                .expect("standalone add");
        }
        dev.synchronize();
        let one_launch_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        // ── A BATCH of ops: ONE wave against the same batch of launches. ──
        //
        // This is the shape a RESIDENT wave would give — the wave's launch
        // amortised over many nodes. It is the number that decides whether
        // the megakernel direction has a future on this workload.
        let t0 = Instant::now();
        for _ in 0..ITERS {
            for _ in 0..BATCH {
                ctx.enqueue_add(&dev, &ctx.plan().adds()[0])
                    .expect("enqueue add");
            }
        }
        dev.synchronize();
        let batch_wave_us = t0.elapsed().as_secs_f64() * 1e6 / (ITERS * BATCH) as f64;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            for _ in 0..BATCH {
                dev.add_into(rocm_of(&a_s), rocm_of(&b_s), rocm_of(&o_s))
                    .expect("standalone add");
            }
        }
        dev.synchronize();
        let batch_launch_us = t0.elapsed().as_secs_f64() * 1e6 / (ITERS * BATCH) as f64;

        // ── A BATCH through a RESIDENT wave: ONE wave for the whole batch. ──
        //
        // This is the shape the megakernel direction needs: the wave's
        // LAUNCH amortised over many nodes, so the per-node cost falls to a
        // descriptor write plus a head publish. The non-resident path above
        // launches a whole wave per node, so it cannot amortise anything.
        //
        // The resident wave parks on the RING's own stream and picks up
        // whatever the ring is given next, so the batch is published to a
        // parked wave — one launch, many nodes.
        let mut res_ctx =
            grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&dev, &graph)
                .expect("ring context")
                .expect("ring binds the add");
        res_ctx.launch_resident(&dev).expect("launch resident wave");
        // Warm the parked wave up.
        for _ in 0..WARMUP {
            res_ctx
                .enqueue_add(&dev, &res_ctx.plan().adds()[0])
                .expect("enqueue add");
            dev.synchronize();
        }
        // The parked wave drains asynchronously, so the batch is published and
        // then the device is synchronised once — the amortised shape.
        let t0 = Instant::now();
        for _ in 0..ITERS {
            for _ in 0..BATCH {
                res_ctx
                    .enqueue_add(&dev, &res_ctx.plan().adds()[0])
                    .expect("enqueue add");
            }
        }
        dev.synchronize();
        let resident_wave_us = t0.elapsed().as_secs_f64() * 1e6 / (ITERS * BATCH) as f64;
        // The wave must be stopped before its buffers are freed, or it spins
        // on freed memory and wedges the device.
        let _ = res_ctx.stop_wave(&dev);

        eprintln!(
            "[small-op] {:>8} {:>14.1} {:>14.1} {:>12.2} {:>14.1} {:>14.1} {:>10.2} {:>14.1} {:>10.2}",
            n,
            one_wave_us,
            one_launch_us,
            one_wave_us / one_launch_us,
            batch_wave_us,
            batch_launch_us,
            batch_wave_us / batch_launch_us,
            resident_wave_us,
            resident_wave_us / batch_launch_us
        );
    }

    eprintln!(
        "[small-op] the wave's per-op cost is a descriptor write + a head publish \
         + the wave's launch + its poll-and-claim, against the launcher's ONE \
         launch."
    );
    eprintln!(
        "[small-op] the 'batch wave' columns are {BATCH} ops, each through its OWN \
         wave (the non-resident path — a wave launch per node). The 'RESIDENT' \
         column is the {BATCH} ops through ONE parked wave — the amortised shape."
    );
}
