//! WI-SB6: production layer routing through the ScytheRing persistent
//! dispatch wave — the "last mile" of scythe2.md §3.
//!
//! `GRIM_SCYTHE_RING=1` reroutes F32 `matmul_op` GEMMs (the dense-layer op
//! every decode step executes) from the rocBLAS direct path onto the
//! device-resident descriptor ring: the host writes one 64-byte
//! `ScytheTaskDescriptor` (opcode 1 = column-GEMM, `b` row-major `[k, n]` —
//! byte-identical semantics to the rocBLAS call in `matmul_op`), publishes
//! the head, and one bounded persistent wave consumes it without any
//! per-op hipModuleLaunchKernel GEMM dispatch.
//!
//! ## Execution mode
//!
//! Bounded batch-synchronous (the proven slice-1 semantics of
//! `ScytheRingExec::run_batch`): each routed GEMM publishes its descriptor,
//! stream-syncs the publish, then launches a wave with
//! `max_tasks = 1, resident = 0`. The sync-before-launch is required —
//! `hipMemcpyAsync` from pinned memory reads the host cell at EXECUTION
//! time, so the cell could otherwise be overwritten by the next call
//! before the copy runs. No eternal wave is alive in this mode, so the
//! eternal-kernel coexistence rules (no blocking hipMemcpy / pageable D2H
//! / per-call pinned alloc against a live wave) do not apply.
//!
//! This path is a BENCHMARK GATE (SB6: "production layer routing behind
//! benchmark gate"), not a default: the ring GEMM arm is a
//! 128-thread-per-CU reference kernel, and the per-op publish sync costs a
//! host round-trip the direct path avoids. `ring_vs_direct_decode`
//! quantifies both.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use grim_tensor::backend::BackendDevice;
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::pinned::RocmPinnedBuffer;
use crate::memory::storage::RocmStorage;
use crate::{Error, Result, HipMemcpyKind, hipMemcpyAsync, hipStreamSynchronize};

/// Ring capacity for the production channel. Power of two (ring index
/// math), large enough that the host never laps the device between the
/// per-op stream syncs (which bound in-flight work to one wave anyway).
const RING_CAPACITY: u32 = 8;

/// `true` when `GRIM_SCYTHE_RING=1` is set — the SB6 production routing
/// gate. Read once per matmul; cheap env lookup relative to a GEMM.
pub fn ring_routing_enabled() -> bool {
    std::env::var_os("GRIM_SCYTHE_RING")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// One persistent channel per device ordinal: device-resident slot array
/// plus the tail/head/stop scalars the wave polls. Owned for the process
/// lifetime; the head/tail counters are monotonic, so a channel is
/// single-use per ordinal (mirrors `ScytheRingExec`'s single-lifetime
/// resident contract, but bounded waves make it effectively unlimited).
struct RingChannel {
    slots: Box<dyn BackendStorage>,
    tail: Box<dyn BackendStorage>,
    head: Box<dyn BackendStorage>,
    stop: Box<dyn BackendStorage>,
    /// Pinned 64-byte staging cell for the descriptor upload.
    staging: RocmPinnedBuffer<u8>,
    /// Pinned 4-byte cell for the head publish.
    head_cell: RocmPinnedBuffer<u8>,
    slots_dev: u64,
    head_dev: u64,
    /// Host-side monotonic head counter (device head is published from it).
    next_head: u32,
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

/// Pack one opcode-1 (column-GEMM) `ScytheTaskDescriptor` into `cell`.
///
/// Byte layout (pinned by `test_task_descriptor_size` and the device-gated
/// ring tests): opcode@0, m@4, n@8, k@12, input_ptr@16, weight_ptr@24,
/// output_ptr@32, peer_ptr@40, status@48, padded to 64 under align(32).
fn pack_gemm_descriptor(
    cell: &mut [u8],
    m: u32,
    n: u32,
    k: u32,
    input: u64,
    weight: u64,
    output: u64,
) {
    cell[..64].fill(0);
    cell[0..4].copy_from_slice(&1u32.to_ne_bytes()); // opcode 1 = OP_COL_GEMM
    cell[4..8].copy_from_slice(&m.to_ne_bytes());
    cell[8..12].copy_from_slice(&n.to_ne_bytes());
    cell[12..16].copy_from_slice(&k.to_ne_bytes());
    cell[16..24].copy_from_slice(&input.to_ne_bytes());
    cell[24..32].copy_from_slice(&weight.to_ne_bytes());
    cell[32..40].copy_from_slice(&output.to_ne_bytes());
    // peer_ptr = 0, status = 0 (pending)
}

/// Route one F32 GEMM through the ring's persistent dispatch wave.
///
/// Computes the same `out[m,n] = Σ_k a[m,k]·b[k,n]` (b row-major) as the
/// rocBLAS path in `matmul_op`. Returns the stream the wave was launched
/// on — callers wrap it in a `RocmHandle` exactly like the direct path.
pub(crate) fn route_gemm(
    device: &RocmDevice,
    stream: *mut c_void,
    a: &RocmStorage,
    b: &RocmStorage,
    out: &RocmStorage,
    m: usize,
    n: usize,
    k: usize,
) -> Result<*mut c_void> {
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

    {
        let cell = chan.staging.as_mut_slice();
        pack_gemm_descriptor(cell, m, n, k, a_ptr, b_ptr, out_ptr);
    }
    let dst = chan.slots_dev + slot as u64 * 64;
    device.copy_scythe_descriptor_async(dst, chan.staging.as_ptr() as *const c_void, 64)?;

    // Publish the head so the wave sees the new descriptor. Async on the
    // same stream as the upload (ordered behind it), then a STREAM-scoped
    // sync: the pinned head cell must not be rewritten for the next op
    // until this copy has executed.
    chan.head_cell.as_mut_slice()[..4]
        .copy_from_slice(&head_value.to_ne_bytes());
    let _dev_guard = crate::device::util::DeviceGuard::set(device.ordinal() as i32);
    let rc = unsafe {
        hipMemcpyAsync(
            chan.head_dev as *mut c_void,
            chan.head_cell.as_ptr() as *const c_void,
            4,
            HipMemcpyKind::HostToDevice,
            stream,
        )
    };
    if rc != 0 {
        return Err(Error::Backend(format!(
            "ring route: head publish failed with hip status {rc}"
        )));
    }
    crate::device::helpers::check_hip("ring route: head publish sync", unsafe {
        hipStreamSynchronize(stream)
    })?;

    // One bounded wave consumes exactly this task. Everything is ordered on
    // the caller's stream, so back-to-back routed GEMMs serialize behind
    // their predecessors' waves. The launcher enqueues on the device active
    // stream (== `stream` here); the caller wraps the returned stream in a
    // RocmHandle exactly like the direct path.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_packing_matches_device_abi() {
        let mut cell = [0u8; 64];
        pack_gemm_descriptor(&mut cell, 3, 5, 7, 0x1000, 0x2000, 0x3000);
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
        // The env var is absent in test runs: the gate must read OFF so the
        // direct rocBLAS path stays the default everywhere. (Read-only
        // lookup — env mutation is unsafe in edition 2024.)
        assert!(
            !ring_routing_enabled(),
            "without GRIM_SCYTHE_RING=1 the gate must stay closed"
        );
    }
}
