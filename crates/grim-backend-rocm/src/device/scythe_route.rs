//! WI-SB6: production layer routing through the ScytheRing persistent dispatch wave - the "last mile" of scythe2.md §3.
//! `GRIM_SCYTHE_RING=1` reroutes F32 `matmul_op` GEMMs (the dense-layer op every decode step executes) from the rocBLAS.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, MemoryOps, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::pinned::RocmPinnedBuffer;
use crate::memory::storage::RocmStorage;
use crate::peer_access::{self, LinkType};
use crate::{Error, Result};

/// Ring capacity for the persistent-dispatch ring.
/// Power of two (ring index math), large enough that the host never laps the device.
const RING_CAPACITY: u32 = 32;

// Mirror of the opcode constant in kernels/scythe_persistent::KERNEL_SOURCE.
const OP_ROW_GEMM: u32 = 2; // B is [N, K] → C = A @ B^T (matmul_op convention)

/// `true` when `GRIM_SCYTHE_RING=1` is set — the SB6 production routing
/// gate. Read once per matmul; cheap env lookup relative to a GEMM.
pub fn ring_routing_enabled() -> bool {
    std::env::var_os("GRIM_SCYTHE_RING")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// One persistent channel per device ordinal: device-resident slot array plus the tail/head/stop scalars the wave polls.
/// Owned for the process lifetime; the head/tail counters are monotonic, so a channel is single-use.
#[allow(dead_code)]
pub struct RingChannel {
    /// Peer link type per device ordinal (only populated for known peers).
    pub peer_links: HashMap<usize, LinkType>,
    pub slots: Box<dyn BackendStorage>,
    pub tail: Box<dyn BackendStorage>,
    pub head: Box<dyn BackendStorage>,
    pub stop: Box<dyn BackendStorage>,
    /// Pinned 64-byte staging cell for the descriptor upload.
    pub staging: RocmPinnedBuffer<u8>,
    /// Pinned 4-byte cell for the head publish.
    pub head_cell: RocmPinnedBuffer<u8>,
    pub slots_dev: u64,
    pub head_dev: u64,
    /// Host-side monotonic head counter (device head is published from it).
    pub next_head: u32,
}

fn channels() -> &'static Mutex<HashMap<usize, Arc<Mutex<RingChannel>>>> {
    static CHANNELS: OnceLock<Mutex<HashMap<usize, Arc<Mutex<RingChannel>>>>> = OnceLock::new();
    CHANNELS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn channel_for(device: &RocmDevice) -> Result<Arc<Mutex<RingChannel>>> {
    let mut map = channels().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(chan) = map.get(&device.ordinal()) {
        return Ok(Arc::clone(chan));
    }

    // MG-1: discover peers and enable peer access to every reachable device.
    // This runs ONCE per device ring channel (cold-start cost, not per-op).
    let device_count = peer_access::enumerate_devices().unwrap_or(0);
    let mut peer_links = HashMap::new();
    for peer_ord in 0..device_count {
        if peer_ord == device.ordinal() {
            continue;
        }
        let link = match peer_access::peer_status(device.ordinal() as i32, peer_ord as i32) {
            Ok(status) => match status {
                peer_access::P2PStatus::P2P => LinkType::PeerDirect,
                peer_access::P2PStatus::Pcie => LinkType::PeerDirect,
                _ => LinkType::HostBounce,
            },
            Err(_) => LinkType::HostBounce,
        };
        if link != LinkType::HostBounce {
            let _ = peer_access::enable_peer_access(device.ordinal() as i32, peer_ord as i32);
        }
        peer_links.insert(peer_ord, link);
    }
    eprintln!(
        "[scythe-ring] device {} peer topology: {:?}",
        device.ordinal(),
        peer_links
    );
    let u32_dtype = DType {
        arith: ArithType::U32,
        storage: DTypeStorage::Native,
    };
    let scalar = |v: u32| -> Result<Box<dyn BackendStorage>> {
        let bytes = v.to_ne_bytes().to_vec();
        device
            .from_cpu_bytes(&bytes, &Shape::new(vec![1]), u32_dtype.clone())
            .map_err(|e| Error::Backend(format!("ring channel scalar alloc: {e}")))
    };
    // alloc_scythe_ring_bytes zeroes the slot array — status must start
    // PENDING(0) or the wave claims phantom descriptors.
    let slots = device.alloc_scythe_ring_bytes(RING_CAPACITY as usize * 64)?;
    let slots_dev = slots
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("ring channel slots have no device ptr".into()))?;
    let head = scalar(0)?;
    let head_dev = head
        .as_any()
        .downcast_ref::<RocmStorage>()
        .and_then(|rs| rs.device_ptr_u64())
        .ok_or_else(|| Error::Backend("ring channel head has no device ptr".into()))?;
    let chan = Arc::new(Mutex::new(RingChannel {
        peer_links,
        slots: Box::new(slots),
        tail: scalar(0)?,
        head,
        stop: scalar(0)?,
        staging: RocmPinnedBuffer::alloc(64)?,
        head_cell: RocmPinnedBuffer::alloc(4)?,
        slots_dev,
        head_dev,
        next_head: 0,
    }));
    map.insert(device.ordinal(), Arc::clone(&chan));
    Ok(chan)
}

/// Pack one GEMM `ScytheTaskDescriptor` into `cell`.
/// Byte layout (pinned by `test_task_descriptor_size` and the device-gated ring tests): opcode@0, m@4, n@8, k@12, input_ptr@16,.
///
/// `opcode` selects the B-matrix convention the kernel applies:
/// OP_COL_GEMM (1) — B is [K, N], computes C = A @ B.
/// OP_ROW_GEMM (2) — B is [N, K], computes C = A @ B^T (matmul_op convention).
/// Kept for the descriptor-packing ABI test; the dispatch path now writes
/// descriptors on-device via `grim_scythe_write_slot` (HIP-graph capturable).
#[allow(dead_code)]
fn pack_gemm_descriptor(
    cell: &mut [u8],
    opcode: u32,
    m: u32,
    n: u32,
    k: u32,
    input: u64,
    weight: u64,
    output: u64,
) {
    cell[..64].fill(0);
    cell[0..4].copy_from_slice(&opcode.to_ne_bytes());
    cell[4..8].copy_from_slice(&m.to_ne_bytes());
    cell[8..12].copy_from_slice(&n.to_ne_bytes());
    cell[12..16].copy_from_slice(&k.to_ne_bytes());
    cell[16..24].copy_from_slice(&input.to_ne_bytes());
    cell[24..32].copy_from_slice(&weight.to_ne_bytes());
    cell[32..40].copy_from_slice(&output.to_ne_bytes());
    // peer_ptr = 0, status = 0 (pending)
}

/// Route one F32 GEMM through the ring's persistent dispatch wave.
/// Computes the same `out[m,n] = Σ_k a[m,k]·b[k,n]` (b row-major) as the rocBLAS path in `matmul_op`.
pub fn route_gemm(
    device: &RocmDevice,
    stream: *mut c_void,
    a: &RocmStorage,
    b: &RocmStorage,
    out: &RocmStorage,
    m: usize,
    n: usize,
    k: usize,
) -> Result<*mut c_void> {
    let stream = if stream.is_null() {
        device.active_stream()
    } else {
        stream
    };
    let (m, n, k) = (
        u32::try_from(m).map_err(|_| Error::Shape("ring route: m exceeds u32".into()))?,
        u32::try_from(n).map_err(|_| Error::Shape("ring route: n exceeds u32".into()))?,
        u32::try_from(k).map_err(|_| Error::Shape("ring route: k exceeds u32".into()))?,
    );
    let a_ptr = a
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("ring route: a has no device ptr".into()))?;
    let b_ptr = b
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("ring route: b has no device ptr".into()))?;
    let out_ptr = out
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("ring route: out has no device ptr".into()))?;

    let chan = channel_for(device)?;
    let mut chan = chan.lock().unwrap_or_else(|e| e.into_inner());

    let slot = chan.next_head % RING_CAPACITY;
    let head_value = chan.next_head.wrapping_add(1);
    chan.next_head = head_value;

    // On-device descriptor write (Option A): grim_scythe_write_slot packs the
    // 64-byte descriptor into the ring slot entirely on-device — no host staging,
    // no H2D — so this enqueue is HIP-graph capturable. Replaces the legacy
    // pinned-staging + copy_scythe_descriptor_async path.
    crate::kernels::qkv_attention::launch_scythe_write_slot(
        device,
        chan.slots.as_ref(),
        slot as usize,
        OP_ROW_GEMM,
        m,
        n,
        k,
        a_ptr,
        b_ptr,
        out_ptr,
        0, // peer_ptr
    )?;

    // Publish the head so the wave sees the new descriptor. On-device kernel
    // write (graph-capturable) instead of the legacy pinned-staging + H2D.
    let head_store = chan
        .head
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| Error::Backend("ring route: head must be RocmStorage".into()))?;
    let head_ptr = head_store
        .device_ptr
        .ok_or_else(|| Error::Backend("ring route: head has no device ptr".into()))?;
    crate::kernels::qkv_attention::launch_scythe_publish_head(device, head_ptr, head_value)?;

    // One bounded wave consumes exactly this task.
    // Everything is ordered on the caller's stream, so back-to-back routed GEMMs serialize behind their predecessors'.
    device.launch_scythe_persistent_dispatch(
        chan.slots.as_ref(),
        RING_CAPACITY,
        chan.tail.as_ref(),
        chan.head.as_ref(),
        chan.stop.as_ref(),
        1,
        0,
    )?;
    Ok(stream)
}

// ─── MG-2 + MG-3: cross-device routing ────────────────────────────────────

/// Get (or create) the ring channel for a target device ordinal.
/// Used when routing ops cross-device: the descriptor goes to the
/// TARGET device's ring, not the caller's.
#[allow(dead_code)]
pub fn channel_for_ordinal(ordinal: usize) -> Result<Arc<Mutex<RingChannel>>> {
    let dev = RocmDevice::try_new(ordinal)?;
    channel_for(&dev)
}

/// Check if peer access is available between two device ordinals.
#[allow(dead_code)]
pub fn peer_link_type(src_ordinal: usize, dst_ordinal: usize) -> Option<LinkType> {
    let map = channels().lock().unwrap_or_else(|e| e.into_inner());
    let chan = map.get(&src_ordinal)?;
    chan.lock().unwrap_or_else(|e| e.into_inner()).peer_links.get(&dst_ordinal).copied()
}

/// MG-2: route one GEMM through a specific device's ring channel.
/// The caller specifies which device executes the GEMM; input/output
/// pointers use peer-access-enabled addresses when crossing devices.
#[allow(dead_code)]
pub fn route_gemm_to(
    target_ordinal: usize,
    stream: *mut c_void,
    a: &RocmStorage,
    b: &RocmStorage,
    out: &RocmStorage,
    m: usize,
    n: usize,
    k: usize,
) -> Result<*mut c_void> {
    let dev = RocmDevice::try_new(target_ordinal)?;
    route_gemm(&dev, stream, a, b, out, m, n, k)
}

/// MG-3: route an OP_COMMFUSE descriptor — copies src to both peer_dst
/// (remote device memory, via peer access) and local_out (local memory).
/// The persistent wave on the SOURCE device executes this inline.
#[allow(dead_code)]
pub fn route_commfuse(
    device: &RocmDevice,
    stream: *mut c_void,
    src: &RocmStorage,
    peer_dst: Option<u64>,
    local_out: Option<u64>,
    elem_count: usize,
) -> Result<*mut c_void> {
    let src_ptr = src
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("commfuse: src has no device ptr".into()))?;
    route_commfuse_ptrs(device, stream, src_ptr, peer_dst, local_out, elem_count)
}

/// Route OP_COMMFUSE directly using device pointers.
pub fn route_commfuse_ptrs(
    device: &RocmDevice,
    stream: *mut c_void,
    src_ptr: u64,
    peer_dst: Option<u64>,
    local_out: Option<u64>,
    elem_count: usize,
) -> Result<*mut c_void> {
    let stream = if stream.is_null() {
        device.active_stream()
    } else {
        stream
    };
    let peer_val = peer_dst.unwrap_or(0);
    let local_val = local_out.unwrap_or(0);

    let chan = channel_for(device)?;
    let mut chan = chan.lock().unwrap_or_else(|e| e.into_inner());

    let slot = chan.next_head % RING_CAPACITY;
    let head_value = chan.next_head.wrapping_add(1);
    chan.next_head = head_value;

    // On-device descriptor write (graph-capturable).
    crate::kernels::qkv_attention::launch_scythe_write_slot(
        device,
        chan.slots.as_ref(),
        slot as usize,
        5, // OP_COMMFUSE
        elem_count as u32,
        1,
        1,
        src_ptr,
        0, // weight_ptr unused
        local_val,
        peer_val,
    )?;

    let head_ptr = chan
        .head
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .and_then(|s| s.device_ptr)
        .ok_or_else(|| Error::Backend("commfuse: head has no device ptr".into()))?;
    crate::kernels::qkv_attention::launch_scythe_publish_head(device, head_ptr, head_value)?;

    device.launch_scythe_persistent_dispatch(
        chan.slots.as_ref(),
        RING_CAPACITY,
        chan.tail.as_ref(),
        chan.head.as_ref(),
        chan.stop.as_ref(),
        1,
        0,
    )?;
    Ok(stream)
}

/// MG-4: route an OP_PEER_REDUCE descriptor (all-reduce partials: result = local + peer).
/// Executed inline by the persistent wave on `device`.
#[allow(dead_code)]
pub fn route_peer_reduce(
    device: &RocmDevice,
    stream: *mut c_void,
    local: &RocmStorage,
    peer: &RocmStorage,
    out: &RocmStorage,
    elem_count: usize,
) -> Result<*mut c_void> {
    let stream = if stream.is_null() {
        device.active_stream()
    } else {
        stream
    };
    let local_ptr = local
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("peer_reduce: local has no device ptr".into()))?;
    let peer_ptr = peer
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("peer_reduce: peer has no device ptr".into()))?;
    let out_ptr = out
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("peer_reduce: out has no device ptr".into()))?;

    let chan = channel_for(device)?;
    let mut chan = chan.lock().unwrap_or_else(|e| e.into_inner());

    let slot = chan.next_head % RING_CAPACITY;
    let head_value = chan.next_head.wrapping_add(1);
    chan.next_head = head_value;

    // On-device descriptor write (graph-capturable).
    crate::kernels::qkv_attention::launch_scythe_write_slot(
        device,
        chan.slots.as_ref(),
        slot as usize,
        8, // OP_PEER_REDUCE
        elem_count as u32,
        1,
        1,
        local_ptr,
        0, // weight_ptr unused
        out_ptr,
        peer_ptr,
    )?;

    let head_ptr = chan
        .head
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .and_then(|s| s.device_ptr)
        .ok_or_else(|| Error::Backend("peer_reduce: head has no device ptr".into()))?;
    crate::kernels::qkv_attention::launch_scythe_publish_head(device, head_ptr, head_value)?;

    device.launch_scythe_persistent_dispatch(
        chan.slots.as_ref(),
        RING_CAPACITY,
        chan.tail.as_ref(),
        chan.head.as_ref(),
        chan.stop.as_ref(),
        1,
        0,
    )?;
    Ok(stream)
}

/// MG-4: route an OP_PEER_BROADCAST descriptor: writes root's src to remote peer dst.
#[allow(dead_code)]
pub fn route_peer_broadcast(
    device: &RocmDevice,
    stream: *mut c_void,
    src: &RocmStorage,
    peer_dst: &RocmStorage,
    elem_count: usize,
) -> Result<*mut c_void> {
    let stream = if stream.is_null() {
        device.active_stream()
    } else {
        stream
    };
    let src_ptr = src
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("peer_broadcast: src has no device ptr".into()))?;
    let dst_ptr = peer_dst
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("peer_broadcast: peer_dst has no device ptr".into()))?;

    let chan = channel_for(device)?;
    let mut chan = chan.lock().unwrap_or_else(|e| e.into_inner());

    let slot = chan.next_head % RING_CAPACITY;
    let head_value = chan.next_head.wrapping_add(1);
    chan.next_head = head_value;

    // On-device descriptor write (graph-capturable).
    crate::kernels::qkv_attention::launch_scythe_write_slot(
        device,
        chan.slots.as_ref(),
        slot as usize,
        9, // OP_PEER_BROADCAST
        elem_count as u32,
        1,
        1,
        src_ptr,
        0, // weight_ptr unused
        0, // output_ptr unused (peer broadcast writes to peer_dst directly)
        dst_ptr,
    )?;

    let head_ptr = chan
        .head
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .and_then(|s| s.device_ptr)
        .ok_or_else(|| Error::Backend("peer_broadcast: head has no device ptr".into()))?;
    crate::kernels::qkv_attention::launch_scythe_publish_head(device, head_ptr, head_value)?;

    device.launch_scythe_persistent_dispatch(
        chan.slots.as_ref(),
        RING_CAPACITY,
        chan.tail.as_ref(),
        chan.head.as_ref(),
        chan.stop.as_ref(),
        1,
        0,
    )?;
    Ok(stream)
}

/// MG-4: route an OP_PEER_GATHER descriptor: reads from remote peer src into local out.
#[allow(dead_code)]
pub fn route_peer_gather(
    device: &RocmDevice,
    stream: *mut c_void,
    peer_src: &RocmStorage,
    local_out: &RocmStorage,
    elem_count: usize,
) -> Result<*mut c_void> {
    let stream = if stream.is_null() {
        device.active_stream()
    } else {
        stream
    };
    let peer_ptr = peer_src
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("peer_gather: peer_src has no device ptr".into()))?;
    let out_ptr = local_out
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("peer_gather: local_out has no device ptr".into()))?;

    let chan = channel_for(device)?;
    let mut chan = chan.lock().unwrap_or_else(|e| e.into_inner());

    let slot = chan.next_head % RING_CAPACITY;
    let head_value = chan.next_head.wrapping_add(1);
    chan.next_head = head_value;

    // On-device descriptor write (graph-capturable).
    crate::kernels::qkv_attention::launch_scythe_write_slot(
        device,
        chan.slots.as_ref(),
        slot as usize,
        10, // OP_PEER_GATHER
        elem_count as u32,
        1,
        1,
        0, // input_ptr unused
        0, // weight_ptr unused
        out_ptr,
        peer_ptr,
    )?;

    let head_ptr = chan
        .head
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .and_then(|s| s.device_ptr)
        .ok_or_else(|| Error::Backend("peer_gather: head has no device ptr".into()))?;
    crate::kernels::qkv_attention::launch_scythe_publish_head(device, head_ptr, head_value)?;

    device.launch_scythe_persistent_dispatch(
        chan.slots.as_ref(),
        RING_CAPACITY,
        chan.tail.as_ref(),
        chan.head.as_ref(),
        chan.stop.as_ref(),
        1,
        0,
    )?;
    Ok(stream)
}

// ─── MG-5: Cross-device dependency tracking ───────────────────────────────

/// Record an event on `device`'s stream.
/// Returns the raw `hipEvent_t` handle.
#[allow(dead_code)]
pub fn record_event_on(device: &RocmDevice, stream: *mut c_void) -> Result<*mut c_void> {
    let stream = if stream.is_null() {
        device.active_stream()
    } else {
        stream
    };
    let _guard = crate::device::util::DeviceGuard::set(device.ordinal() as i32);
    let mut event: *mut c_void = std::ptr::null_mut();
    let rc = unsafe { crate::hipEventCreate(&mut event) };
    if rc != 0 {
        return Err(Error::Backend(format!("hipEventCreate failed {rc}")));
    }
    let rc = unsafe { crate::hipEventRecord(event, stream) };
    if rc != 0 {
        unsafe { crate::hipEventDestroy(event) };
        return Err(Error::Backend(format!("hipEventRecord failed {rc}")));
    }
    Ok(event)
}

/// Enqueue a stream wait on an event recorded on a potentially foreign device.
/// Enables cross-device zero-copy ordering without host sync when peer access is active.
#[allow(dead_code)]
pub fn stream_wait_event(
    waiting_device: &RocmDevice,
    waiting_stream: *mut c_void,
    event: *mut c_void,
) -> Result<()> {
    if event.is_null() {
        return Ok(());
    }
    let stream = if waiting_stream.is_null() {
        waiting_device.active_stream()
    } else {
        waiting_stream
    };
    let _guard = crate::device::util::DeviceGuard::set(waiting_device.ordinal() as i32);
    let rc = unsafe { crate::hipStreamWaitEvent(stream, event, 0) };
    if rc != 0 {
        return Err(Error::Backend(format!("hipStreamWaitEvent failed {rc}")));
    }
    Ok(())
}

/// Destroy a recorded event.
#[allow(dead_code)]
pub fn destroy_event(event: *mut c_void) {
    if !event.is_null() {
        unsafe {
            let _ = crate::hipEventDestroy(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_packing_matches_device_abi() {
        let mut cell = [0u8; 64];
        pack_gemm_descriptor(&mut cell, 1, 3, 5, 7, 0x1000, 0x2000, 0x3000);
        let u32_at = |off: usize| u32::from_ne_bytes(cell[off..off + 4].try_into().unwrap());
        let u64_at = |off: usize| u64::from_ne_bytes(cell[off..off + 8].try_into().unwrap());
        assert_eq!(u32_at(0), 1, "opcode 1 = OP_COL_GEMM");
        assert_eq!(u32_at(4), 3);
        assert_eq!(u32_at(8), 5);
        assert_eq!(u32_at(12), 7);
        assert_eq!(u64_at(16), 0x1000);
        assert_eq!(u64_at(24), 0x2000);
        assert_eq!(u64_at(32), 0x3000);
        assert_eq!(u64_at(40), 0, "peer unused");
        assert_eq!(u32_at(48), 0, "status pending");
    }

    #[test]
    fn routing_gate_defaults_off() {
        // The env var is absent in test runs: the gate must read OFF so the direct rocBLAS path stays the default everywhere.
        // (Read-only lookup - env mutation is unsafe in edition 2024.)
        assert!(
            !ring_routing_enabled(),
            "without GRIM_SCYTHE_RING=1 the gate must stay closed"
        );
    }
}
