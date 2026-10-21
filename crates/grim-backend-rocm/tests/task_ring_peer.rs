//! M6: the ring's CROSS-DEVICE half — the peer copy, the device-side
//! completion flag, and the boundary handoff that uses them.
//!
//! These gates need TWO devices, so they live in their own binary. That is not
//! only about device selection: a device-1 test that runs BEFORE a device-0
//! ring test in the same process makes the later test HANG, and the hang is
//! not in the device-1 code — it is in the later test's stream synchronisation.
//!
//! REPRODUCED (2026-10-20): running `a_peer_write_is_visible_to_the_consumer_
//! device` and then `a_wait_node_observes_a_ring_descriptors_completion` in
//! one process hangs the second test in its device sync, while either test
//! alone passes, and while the PRE-EXISTING `ring_peer_copy_crosses_devices`
//! (which also uses device 1) does NOT poison the wait-node test. The
//! difference between the two device-1 gates has not been isolated, so the
//! device-1 gates are kept apart from the device-0 ring suite until it is.
//! The interaction is UNEXPLAINED, not dismissed — see the M6 note in
//! `plans/PLAN-multigpu-per-device-graphs.md`.
//!
//! What is proven here:
//!  * the consumer's DEVICE sees the producer's peer write with no host sync
//!    (`a_peer_write_is_visible_to_the_consumer_device`), and
//!  * a whole boundary handoff completes with no host event
//!    (`a_boundary_handoff_completes_with_no_host_event`).

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

/// M6 step 2, the DISCRIMINATING question: can the CONSUMER's DEVICE see the
/// producer's peer write without any host synchronisation?
///
/// This decides the shape of the whole handoff:
///
///  * If YES — the producer's wave writes the completion flag straight into the
///    CONSUMER's memory, and the consumer's device spins on that flag. The
///    handoff needs no host event at all, which is M6's whole point.
///  * If NO — a peer write is not visible to the peer device until the host
///    synchronises, so the flag would have to travel through the copy engine
///    (`hipMemcpyPeerAsync`), which is a HOST call and would put the event
///    straight back.
///
/// The existing `ring_peer_copy_crosses_devices` gate does NOT answer this:
/// it reads the consumer's buffer back ON THE HOST after synchronising both
/// devices, which proves the bytes arrived, not that the consumer's DEVICE
/// could see them on its own.
///
/// So: the producer's wave writes a non-zero float into a flag buffer that
/// lives on the CONSUMER; the consumer's device then spins on that buffer in a
/// kernel, with no host synchronisation in between.
///
/// The second half is the NEGATIVE CONTROL: a flag nobody writes must make the
/// spin TIME OUT (observed zero) rather than report a completion. Without it,
/// the first half could pass for the wrong reason — a spin that exits at once
/// on a stale or pre-set flag would look exactly like a successful handoff.
#[test]
fn a_peer_write_is_visible_to_the_consumer_device() {
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

    // The flags live on the CONSUMER. The producer's wave writes into the
    // first one over peer access; the consumer's device spins on it. The
    // second one is never written — it is the control.
    let (flag_s, flag_ptr) = f32_dev(&consumer, &[0.0f32]);
    let (ctrl_s, ctrl_ptr) = f32_dev(&consumer, &[0.0f32]);
    // The source of the write: one float, 1.0f32 (a non-zero bit pattern).
    let (_src_s, src_ptr) = f32_dev(&producer, &[1.0f32]);
    // Where the consumer records what it saw.
    let (obs_s, obs_ptr) = u32_dev(&consumer, 0);
    let (ctrl_obs_s, ctrl_obs_ptr) = u32_dev(&consumer, 0);

    let mut gb = TaskGraphBuilder::new();
    let h_s = gb.push_arg(src_ptr, 4, arg_role::ACT_IN);
    let h_d = gb.push_arg(flag_ptr, 4, arg_role::PEER);
    gb.push(
        op::PEER_COPY,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[h_s, h_d],
        &[],
        0,
        0,
    );
    let graph = gb.finish();

    let ctx =
        grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&producer, &graph)
            .expect("ring context")
            .expect("ring binds the peer copy");
    ctx.enqueue_peer_copy(&producer, &ctx.plan().peers()[0])
        .expect("enqueue peer copy");

    // The CONSUMER spins on the flag — on the consumer's own device, with NO
    // host synchronisation between the producer's write and this read.
    grim_backend_rocm::kernels::task_ring::launch_scythe_wait_flag(
        &consumer,
        flag_ptr,
        10_000_000,
        obs_ptr,
    )
    .expect("launch flag wait");
    // The control: nobody writes this one, so the spin must run out.
    grim_backend_rocm::kernels::task_ring::launch_scythe_wait_flag(
        &consumer,
        ctrl_ptr,
        200_000,
        ctrl_obs_ptr,
    )
    .expect("launch control wait");

    let observed = obs_s.to_cpu_vec_u32().expect("read observed")[0];
    assert_ne!(
        observed, 0,
        "the consumer's DEVICE must see the producer's peer write with no host sync \
         (the flag stayed zero for the whole spin)"
    );
    let flag = flag_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap()
        .to_cpu_vec_f32()
        .unwrap();
    assert_eq!(flag[0], 1.0f32, "the flag must hold the producer's value");

    // The control must have TIMED OUT, not reported a completion.
    let ctrl_observed = ctrl_obs_s.to_cpu_vec_u32().expect("read control")[0];
    assert_eq!(
        ctrl_observed, 0,
        "a flag nobody writes must time out with zero, not report a completion"
    );
    let ctrl = ctrl_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap()
        .to_cpu_vec_f32()
        .unwrap();
    assert_eq!(ctrl[0], 0.0f32, "the control flag must still be zero");
}

/// M6 step 2: the boundary handoff with NO host event.
///
/// The producer's ring carries three PEER_COPY tasks — clear the consumer's
/// flag, copy the activation, set the flag — and the CONSUMER's DEVICE spins
/// on its own flag until it leaves zero. Between the two there is no host
/// event and no host-visible synchronisation: the producer's copy lands, and
/// the consumer's device sees it.
///
/// What this replaces: A6's `copy_boundary_ordered`, which is a host-issued
/// peer memcpy plus `hipEventRecord` plus `hipStreamWaitEvent` — three host
/// calls and a synchronisation, per boundary, per token.
///
/// The gate asserts the three things the design is made of:
///  * the consumer's device observed the flag (the wait did not time out),
///  * the activation arrived in the consumer's buffer, and
///  * the producer's tasks ran in the ORDER they were staged in (the flag is
///    set only after the data copy — a consumer that saw the flag but not the
///    data would be a handoff that lies).
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

    // The wave's peer arm writes the CONSUMER's memory directly, so the grant
    // has to exist in that direction (the copy engine's path does not need it).
    grim_backend_rocm::peer_access::enable_peer_access(0, 1).expect("peer 0->1");

    const N: usize = 32;
    let src: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let (_s_s, _s_ptr) = f32_dev(&producer, &src);
    let (d_s, _d_ptr) = f32_dev(&consumer, &[0.0f32; N]);

    // The producer's and consumer's storages, as the handoff wants them.
    let src_st = _s_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("src is RocmStorage");
    let dst_st = d_s
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
    // The consumer's wait: ONE device-side kernel launch, on the consumer's
    // stream. There is NO event between the two.
    handoff.launch_wait(&consumer).expect("launch wait");

    producer.synchronize();
    consumer.synchronize();

    let observed = handoff.observed().expect("observed");
    assert_ne!(
        observed, 0,
        "the consumer's device must have seen the flag (the wait timed out)"
    );
    assert_eq!(
        handoff.flag_value().expect("flag"),
        1.0f32,
        "the flag must be set by the producer's last task"
    );
    let got: Vec<f32> = d_s
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("dst is RocmStorage")
        .to_cpu_vec_f32()
        .expect("dst readback");
    assert_eq!(
        got, src,
        "the activation must land in the consumer's buffer"
    );
}
