//! M6: the ring's CROSS-DEVICE half — the peer copy, the device-side
//! completion observation, and the boundary handoff that uses them.
//!
//! These gates need TWO devices, so they live in their own binary.
//!
//! ## What the hardware actually does (measured, 2026-10-20)
//!
//! * A peer **READ** is coherent: a device on one card sees another card's
//!   updates as they happen. This is what M6's completion observation relies
//!   on — the consumer polls a counter in the PRODUCER's memory.
//! * A peer **WRITE** into the consumer's own memory is NOT RELIABLE as a
//!   completion signal. MEASURED: the consumer's device spun on the written
//!   word for its whole 13.5 s budget and kept seeing the OLD value while
//!   the host read the NEW one — three runs. In a different launch order the
//!   same write WAS seen immediately. So the write's visibility to the reader
//!   is not guaranteed, and a design built on it is a design that sometimes
//!   hangs. An earlier version of this file put the completion flag in the
//!   CONSUMER's memory and spun on it; it was withdrawn for this reason.
//!
//! So the completion signal lives on the PRODUCER's side: the ring's tail is
//! a MONOTONIC step counter, and the consumer polls it over peer access. A
//! peer READ of the producer's memory is reliable — that is what
//!   `a_consumer_sees_the_producers_update_through_a_peer_read` gates, and
//!   what the two handoff gates exercise end to end.
//!
//! ## Why (the transport)
//!
//! `rocm-smi --showtopo` on this box: the link type between every GPU pair is
//! **PCIE** (weight 40, 2 hops) — there is no XGMI between these cards. A peer
//! write therefore travels over PCIe, and the writer's store is not promptly
//! reflected in the reader's cached copy of the line; a peer READ, by
//! contrast, goes to the producer's memory rather than to a locally cached
//! copy. That is the asymmetry the design above is built around, and it is
//! also why the handoff is opt-in: the peer grant it needs puts a PCIe peer
//! path into the production loop.
//!
//! ## An open cross-test interaction
//!
//! A device-1 test that runs BEFORE a device-0 ring test in one process makes
//! the later test HANG in its stream synchronisation. Either test alone
//! passes, and the pre-existing `ring_peer_copy_crosses_devices` (which also
//! uses device 1) does NOT poison the wait-node test — so the difference
//! between the two device-1 gates has not been isolated. The device-1 gates
//! are kept apart from the device-0 ring suite until it is. The interaction
//! is UNEXPLAINED, not dismissed — see the M6 note in
//! `plans/PLAN-multigpu-per-device-graphs.md`.

use grim_backend_rocm::task_graph::{arg_role, op, TaskGraphBuilder};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, MemoryOps, Shape};

/// A device-resident f32 buffer, and its device address.
fn f32_dev(dev: &grim_backend_rocm::RocmDevice, data: &[f32]) -> (Box<dyn BackendStorage>, u64) {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let storage: Box<dyn BackendStorage> = dev
        .from_cpu_bytes(
            &bytes,
            &Shape::new(vec![data.len()]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
        )
        .expect("f32 device tensor");
    let ptr = storage
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("f32 device tensor is RocmStorage")
        .device_ptr_u64()
        .expect("f32 device tensor has a device ptr");
    (storage, ptr)
}

/// A device-resident u32 buffer, and its device address.
fn u32_dev(dev: &grim_backend_rocm::RocmDevice, v: u32) -> (Box<dyn BackendStorage>, u64) {
    let storage: Box<dyn BackendStorage> = dev
        .from_cpu_bytes(
            &v.to_ne_bytes(),
            &Shape::new(vec![1]),
            DType {
                arith: ArithType::U32,
                storage: DTypeStorage::Native,
            },
        )
        .expect("u32 device scalar");
    let ptr = storage
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("u32 device scalar is RocmStorage")
        .device_ptr_u64()
        .expect("u32 device scalar has a device ptr");
    (storage, ptr)
}

/// A graph with one GEMM, over real device buffers on `dev`.
fn one_gemm_graph(
    dev: &grim_backend_rocm::RocmDevice,
) -> (
    grim_backend_rocm::task_graph::TaskGraph,
    Vec<Box<dyn BackendStorage>>,
) {
    let a = [1.0f32, 2.0];
    let b = [1.0f32, 0.0, 0.0, 1.0];
    let (a_s, a_p) = f32_dev(dev, &a);
    let (b_s, b_p) = f32_dev(dev, &b);
    let (c_s, c_p) = f32_dev(dev, &[0.0f32; 2]);
    let mut gb = TaskGraphBuilder::new();
    let h_i = gb.push_arg(a_p, 8, arg_role::ACT_IN);
    let h_w = gb.push_arg(b_p, 32, arg_role::WEIGHT);
    let h_o = gb.push_arg(c_p, 8, arg_role::ACT_OUT);
    gb.push(
        op::GEMM,
        [0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h_i, h_w, h_o],
        &[],
        0,
        0,
    );
    (gb.finish(), vec![a_s, b_s, c_s])
}

/// M6's completion observation works: the CONSUMER's device polls a counter in
/// the PRODUCER's memory over peer access and sees it advance.
///
/// This is the fact M6's whole handoff rests on, and it is the OPPOSITE of
/// what a peer write can do (see the gate below).
#[test]
fn a_consumer_sees_the_producers_update_through_a_peer_read() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let producer = grim_backend_rocm::RocmDevice::try_new(0).expect("dev 0");
    let consumer = grim_backend_rocm::RocmDevice::try_new(1).expect("dev 1");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // Both directions: the producer writes the consumer's memory, and the
    // consumer reads the producer's.
    grim_backend_rocm::peer_access::enable_peer_access(0, 1).expect("peer 0->1");
    grim_backend_rocm::peer_access::enable_peer_access(1, 0).expect("peer 1->0");

    let (graph, _keep) = one_gemm_graph(&producer);
    let ctx =
        grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&producer, &graph)
            .expect("ring context")
            .expect("ring binds the gemm");
    let tail_ptr = ctx.tail_ptr().expect("tail ptr");
    let (obs_s, obs_ptr) = u32_dev(&consumer, 0);
    let (last_s, last_ptr) = u32_dev(&consumer, 0);

    // The CONSUMER polls the producer's tail. It is launched BEFORE the
    // producer's work, so the consumer's FIRST read is of the pre-work value:
    // it has to see the advance, not a stale one. The host sleeps (rather than
    // synchronising — a device sync would invalidate the very cache line this
    // gate is about) so the first read definitely happens first.
    grim_backend_rocm::kernels::task_ring::launch_scythe_wait_counter(
        &consumer,
        tail_ptr,
        last_ptr,
        50_000_000,
        obs_ptr,
    )
    .expect("launch counter poll");
    std::thread::sleep(std::time::Duration::from_millis(200));

    ctx.enqueue_gemm(&producer, &ctx.plan().gemms[0])
        .expect("enqueue gemm");

    producer.synchronize();
    consumer.synchronize();

    let observed = obs_s.to_cpu_vec_u32().expect("read observed")[0];
    assert_ne!(
        observed, 0,
        "the consumer's device must see the producer's tail advance over peer access"
    );
    let last = last_s.to_cpu_vec_u32().expect("read last")[0];
    assert_eq!(
        last, observed,
        "the consumer's last-seen register must be advanced to the value it observed"
    );
}

/// M6 step 2: the boundary handoff with NO host event.
///
/// The producer's ring carries the boundary's PEER_COPY, and the CONSUMER's
/// DEVICE polls the producer's tail over peer access until it advances. No
/// host event and no host-visible synchronisation between the two.
#[test]
fn a_boundary_handoff_completes_with_no_host_event() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let producer = grim_backend_rocm::RocmDevice::try_new(0).expect("dev 0");
    let consumer = grim_backend_rocm::RocmDevice::try_new(1).expect("dev 1");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    grim_backend_rocm::peer_access::enable_peer_access(0, 1).expect("peer 0->1");
    grim_backend_rocm::peer_access::enable_peer_access(1, 0).expect("peer 1->0");

    const N: usize = 32;
    let src: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let (src_s, _src_ptr) = f32_dev(&producer, &src);
    let (dst_s, _dst_ptr) = f32_dev(&consumer, &[0.0f32; N]);

    let src_st = src_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("src is RocmStorage");
    let dst_st = dst_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("dst is RocmStorage");

    let handoff = grim_backend_rocm::peer_access::BoundaryHandoff::prepare(
        &producer,
        &consumer,
        src_st,
        dst_st,
        N as u32,
    )
    .expect("prepare handoff");

    // The producer's copy: ONE wave launch, on the producer's stream.
    handoff
        .launch_copy(&producer, producer.active_stream())
        .expect("launch copy");
    // The consumer's wait: ONE device-side poll, on the consumer's stream.
    // There is NO event between the two.
    handoff
        .launch_wait(&consumer, consumer.active_stream())
        .expect("launch wait");

    producer.synchronize();
    consumer.synchronize();

    let observed = handoff.observed().expect("observed");
    assert_ne!(
        observed, 0,
        "the consumer's device must have seen the producer's tail advance"
    );
    assert_eq!(
        handoff.tail_value().expect("tail"),
        observed,
        "the consumer observed the producer's tail"
    );
    let got: Vec<f32> = dst_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("dst is RocmStorage")
        .to_cpu_vec_f32()
        .expect("dst readback");
    assert_eq!(got, src, "the activation must land in the consumer's buffer");
}

/// M6: a REPEATED handoff must not reuse the previous handoff's signal.
///
/// The tail is MONOTONIC, so each handoff advances it past the value the
/// consumer last saw. A signal that merely changed could not be told apart
/// from the previous one, and the consumer would run before the new copy
/// landed.
#[test]
fn a_repeated_handoff_observes_a_new_signal_and_the_new_data() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let producer = grim_backend_rocm::RocmDevice::try_new(0).expect("dev 0");
    let consumer = grim_backend_rocm::RocmDevice::try_new(1).expect("dev 1");
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    grim_backend_rocm::peer_access::enable_peer_access(0, 1).expect("peer 0->1");
    grim_backend_rocm::peer_access::enable_peer_access(1, 0).expect("peer 1->0");

    const N: usize = 8;
    let src: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let (src_s, _src_ptr) = f32_dev(&producer, &src);
    let (dst_s, _dst_ptr) = f32_dev(&consumer, &[0.0f32; N]);

    let src_st = src_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("src is RocmStorage");
    let dst_st = dst_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("dst is RocmStorage");

    let handoff = grim_backend_rocm::peer_access::BoundaryHandoff::prepare(
        &producer,
        &consumer,
        src_st,
        dst_st,
        N as u32,
    )
    .expect("prepare handoff");

    let mut seen = Vec::new();
    for round in 0..3 {
        handoff
            .launch_copy(&producer, producer.active_stream())
            .expect("launch copy");
        handoff
            .launch_wait(&consumer, consumer.active_stream())
            .expect("launch wait");
        producer.synchronize();
        consumer.synchronize();
        let observed = handoff.observed().expect("observed");
        assert_ne!(
            observed, 0,
            "round {round}: the consumer must have observed the handoff"
        );
        seen.push(observed);
        let got: Vec<f32> = dst_s
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("dst is RocmStorage")
            .to_cpu_vec_f32()
            .expect("dst readback");
        assert_eq!(got, src, "round {round}: the activation must have landed");
    }

    // Each round must have produced a DIFFERENT signal: the same value twice
    // would mean the consumer consumed a stale signal.
    for w in seen.windows(2) {
        assert!(
            w[1] > w[0],
            "each handoff must advance the tail past the last (seen {seen:?})"
        );
    }
}

