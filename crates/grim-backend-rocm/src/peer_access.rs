//! Peer-access probe + P2P verdict for the multi-GPU head of §3.8.
//! Cycle 1 - the apparatus for asking "before I try a P2P memcpy / RCCL.

use std::borrow::Cow;
use std::ffi::c_void;

use grim_tensor::error::{Error, Result};
use grim_tensor::backend::BackendStorage;
use grim_tensor::MemoryOps;

use crate::device::helpers::check_hip;
use crate::{HipErrorT, hipSuccess};

// HIP symbols we call. We declare them locally rather than going through `crate::...` because the crate-root
// FFI declarations have drift between ROCm releases - re-declaring at the call site keeps `peer_access` self-contained.
unsafe extern "C" {
    fn hipGetDeviceCount(count: *mut i32) -> HipErrorT;
    fn hipGetDeviceProperties(prop: *mut c_void, device: i32) -> HipErrorT;
    fn hipDeviceCanAccessPeer(
        can_access_peer: *mut i32,
        device: i32,
        peer_device: i32,
    ) -> HipErrorT;
    fn hipDeviceEnablePeerAccess(peer_device: i32, flags: u32) -> HipErrorT;
    fn hipDeviceDisablePeerAccess(peer_device: i32) -> HipErrorT;
    fn hipMemcpyPeerAsync(
        dst: *mut c_void,
        dst_device: i32,
        src: *const c_void,
        src_device: i32,
        count: usize,
        stream: *mut c_void,
    ) -> HipErrorT;
}

/// Asynchronous Peer-to-Peer copy wrapping hipMemcpyPeerAsync directly (no RCCL feature dependency).
pub fn p2p_memcpy_async(
    dst: *mut c_void,
    dst_device: i32,
    src: *const c_void,
    src_device: i32,
    count: usize,
    stream: *mut c_void,
) -> Result<()> {
    if dst.is_null() || src.is_null() {
        return Err(Error::Backend("p2p_memcpy_async: null buffer".into()));
    }
    let res = unsafe {
        hipMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream)
    };
    if res == hipSuccess {
        Ok(())
    } else {
        Err(Error::Backend(format!(
            "hipMemcpyPeerAsync failed with status {res}"
        )))
    }
}

/// Host-visible boundary synchronisations issued by the A6 path.
///
/// One per boundary per token: the host records an event after the copy and
/// the consumer's stream waits on it. M6's handoff issues NONE of these — its
/// completion is observed by the consumer's DEVICE.
static BOUNDARY_HOST_SYNCS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// How many host-visible boundary synchronisations this process has issued.
pub fn boundary_host_sync_count() -> u64 {
    BOUNDARY_HOST_SYNCS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Is M6's ring handoff enabled?
///
/// OFF by default. The handoff replaces A6's host-issued peer copy + event
/// with a wave launch + a device-side wait, and it needs peer access granted
/// between the two devices (the wave's peer arm writes the consumer's memory
/// DIRECTLY, which is a different mechanism from the copy engine's
/// `hipMemcpyPeerAsync`). That grant changes the process's device topology in
/// a way whose interaction with the rest of the ring is not yet understood —
/// see the note in `plans/PLAN-multigpu-per-device-graphs.md` — so the
/// handoff is opt-in until it is.
pub fn ring_handoff_enabled() -> bool {
    std::env::var("GRIM_RING_HANDOFF").map(|v| v == "1").unwrap_or(false)
}

/// Enqueue a peer copy `src (on src_stream's device) -> dst`, then order
/// `dst_stream` after it. The copy must NOT be captured (it runs between two
/// graph launches in the replay driver), and `dst_stream` must not start the
/// consumer graph until the copy lands — hence the event rather than a device
/// sync, which would drain both queues every token.
pub fn copy_boundary_ordered(
    src: &crate::RocmStorage,
    dst: &crate::RocmStorage,
    src_stream: *mut c_void,
    dst_stream: *mut c_void,
    ev: *mut c_void,
) -> Result<()> {
    // A6's boundary synchronisation is a HOST-VISIBLE one: the host records an
    // event and the consumer's stream waits on it, so the host is in the loop
    // at every boundary of every token. Counted so M6's gate can MEASURE the
    // difference rather than assert it.
    BOUNDARY_HOST_SYNCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    use crate::device::util::DeviceGuard;
    let (src_dev, dst_dev) = (src.device_ordinal() as i32, dst.device_ordinal() as i32);
    let bytes = src.bytes().min(dst.bytes());
    let s = src
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("boundary src has no ptr".into()))?;
    let d = dst
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("boundary dst has no ptr".into()))?;
    let _guard = DeviceGuard::set(src_dev);
    p2p_memcpy_async(
        d as *mut c_void,
        dst_dev,
        s as *const c_void,
        src_dev,
        bytes,
        src_stream,
    )?;
    // SAFETY: ev created via hipEventCreate; src_stream valid.
    check_hip("hipEventRecord", unsafe { crate::hipEventRecord(ev, src_stream) })?;
    let _guard = DeviceGuard::set(dst_dev);
    // SAFETY: ev/dst_stream valid.
    check_hip("hipStreamWaitEvent", unsafe {
        crate::hipStreamWaitEvent(dst_stream, ev, 0)
    })?;
    Ok(())
}

/// Runtime verdict for a `(src, dst)` GPU pair. `Copy + PartialEq` so
/// it can move across thread boundaries without ceremony.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum P2PStatus {
    /// Direct peer DMA inferred (xGMI class).
    P2P,
    /// Peer enabled but on PCIe; usable but slow.
    Pcie,
    /// Peer access disabled or unreachable; caller must bounce
    /// through host pinned memory.
    Host,
}

impl Default for P2PStatus {
    /// Default to the most defensive option: bounce through host. Until
    /// `peer_status` succeeds we don't trust the runtime.
    fn default() -> Self {
        P2PStatus::Host
    }
}

/// HIP errors we tolerate silently. Sourced from ROCm headers; only the codes
/// that mean "the runtime already has this state" are mapped to soft-success.
const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: HipErrorT = 0xb16;

/// Enumerate HIP devices visible to the process.
/// Returns 0 on a GPU-less box.
pub fn enumerate_devices() -> Result<usize> {
    let mut count: i32 = 0;
    check_hip("hipGetDeviceCount", unsafe {
        hipGetDeviceCount(&mut count as *mut _)
    })?;
    Ok((count.max(0)) as usize)
}

/// Ask `hipDeviceCanAccessPeer`: returns `Ok(true)` only when the runtime can grant true peer class on `(src, dst)`.
/// A GPU-less box whose enumeration returned 0 yields `Ok(false)` rather than `Err(...)`, because "no peers".
pub fn peer_status(src: i32, dst: i32) -> Result<P2PStatus> {
    if src == dst {
        // Self-peer is trivially native.
        return Ok(P2PStatus::P2P);
    }
    let count = enumerate_devices()?;
    if count < 2 {
        // Single GPU: there are no inter-device links to enumerate.
        return Ok(P2PStatus::Host);
    }
    if !bounded(src, count as i32) || !bounded(dst, count as i32) {
        return Err(Error::Backend(format!(
            "peer_status: device out of range (src={}, dst={}, count={})",
            src, dst, count
        )));
    }
    let mut can_access: i32 = 0;
    let probe = unsafe { hipDeviceCanAccessPeer(&mut can_access, src, dst) };
    if probe != hipSuccess {
        return Err(Error::Backend(format!(
            "peer_status: hipDeviceCanAccessPeer returned code={}",
            probe
        )));
    }
    if can_access == 0 {
        return Ok(P2PStatus::Host);
    }
    // The runtime says peer is possible. We don't pre-flight PCIe vs xGMI from the
    // probe alone - that distinction collapses to `Pcie` for consumer hardware and `P2P` for Instinct.
    let gfn = unsafe { gcn_arch_for(src) };
    if is_instinct(gfn.as_ref()) {
        Ok(P2PStatus::P2P)
    } else {
        Ok(P2PStatus::Pcie)
    }
}

/// Enable peer access between `src` and `dst` (`hipDeviceEnablePeerAccess`).
/// Returns `Ok(true)` when peer access is granted, `Ok(false)` when the runtime already had it or.
pub fn enable_peer_access(src: i32, dst: i32) -> Result<bool> {
    if src == dst {
        return Ok(true);
    }
    let count = enumerate_devices()?;
    if count < 2 {
        return Ok(false);
    }
    if !bounded(src, count as i32) || !bounded(dst, count as i32) {
        return Err(Error::Backend(format!(
            "enable_peer_access: device out of range (src={}, dst={}, count={})",
            src, dst, count
        )));
    }
    // hipDeviceEnablePeerAccess / DisablePeerAccess operate on the CURRENT device, so we must pin the current device to `src` for the duration.
    // Without this, the grant lands on (or fails from) whatever device the caller last touched.
    let _guard = crate::device::util::DeviceGuard::set(src);
    // Disable first (best practice - peer access persists in the process
    // address space, so a fresh probe may collide with a stale grant).
    let _ = unsafe { hipDeviceDisablePeerAccess(dst) };
    let res = unsafe { hipDeviceEnablePeerAccess(dst, 0) };
    if res == hipSuccess || res == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED {
        Ok(true)
    } else {
        Err(Error::Backend(format!(
            "enable_peer_access: hipDeviceEnablePeerAccess returned code={}",
            res
        )))
    }
}

/// Peer link classification for inter-GPU communication paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkType {
    /// Peer direct DMA enabled via `hipDeviceEnablePeerAccess`.
    PeerDirect,
    /// Peer direct DMA unavailable; transfer requires host bounce.
    HostBounce,
    /// Cross-host or unreachable inter-device link.
    NoLink,
}

/// Matrix representation of peer-to-peer topology across GPU devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2PTopology {
    /// Total number of devices participating in the topology matrix.
    pub device_count: usize,
    /// 2D link matrix where `links[i][j]` describes the link from device `i` to device `j`.
    pub links: Vec<Vec<LinkType>>,
}

/// Build an N x N topology matrix for the provided slice of ROCm devices.
pub fn build_topology_matrix(devices: &[&crate::device::roc_device::RocmDevice]) -> P2PTopology {
    let count = devices.len();
    if count == 0 {
        return P2PTopology {
            device_count: 0,
            links: Vec::new(),
        };
    }

    let mut links = vec![vec![LinkType::NoLink; count]; count];
    for i in 0..count {
        for j in 0..count {
            if i == j {
                links[i][j] = LinkType::PeerDirect;
                continue;
            }
            let ord_i = devices[i].ordinal() as i32;
            let ord_j = devices[j].ordinal() as i32;

            let status = peer_status(ord_i, ord_j).unwrap_or(P2PStatus::Host);
            links[i][j] = match status {
                P2PStatus::P2P | P2PStatus::Pcie => {
                    let enabled = enable_peer_access(ord_i, ord_j).unwrap_or(false);
                    if enabled {
                        LinkType::PeerDirect
                    } else {
                        LinkType::HostBounce
                    }
                }
                P2PStatus::Host => LinkType::HostBounce,
            };
        }
    }

    P2PTopology {
        device_count: count,
        links,
    }
}

fn bounded(x: i32, n: i32) -> bool {
    x >= 0 && x < n
}

fn is_instinct(arch: &str) -> bool {
    arch.starts_with("gfx9")
}

/// Query `hipGetDeviceProperties` and surface the `gcnArchName` string.
/// Returns the `"gfx9999"` fallback on GPU-less boxes so the verdict branch isn't forced into `Other`.
unsafe fn gcn_arch_for(device: i32) -> Cow<'static, str> {
    let mut buf = vec![0u8; 8192];
    let r = unsafe { hipGetDeviceProperties(buf.as_mut_ptr() as *mut c_void, device) };
    if r != hipSuccess || buf.is_empty() {
        return Cow::Owned("gfx9999".to_string());
    }
    let mut i = 0;
    while i + 3 < buf.len() {
        if buf[i] == b'g' && buf[i + 1] == b'f' && buf[i + 2] == b'x' {
            let start = i;
            let mut end = start;
            while end < buf.len() && buf[end] != 0 {
                end += 1;
            }
            let s = std::str::from_utf8(&buf[start..end])
                .unwrap_or("gfx9999")
                .to_string();
            if s.starts_with("gfx0000") {
                return Cow::Owned("gfx9999".to_string());
            }
            return Cow::Owned(s);
        }
        i += 1;
    }
    Cow::Owned("gfx9999".to_string())
}

// ────────────────────────────────────────────────────────────────────────────
// M6: the segment-boundary handoff with NO host event.
//
// A6's boundary is a HOST-issued peer copy plus a cross-stream event: the
// copy is enqueued on the producing segment's stream, an event is recorded
// after it, and the consuming segment's stream waits on that event. Three
// host calls per boundary per token, and a host-visible synchronisation.
//
// M6 replaces all three with ONE wave launch and ONE device-side wait:
//
//   * The PRODUCER's ring carries three PEER_COPY tasks, in this order:
//       1. clear the consumer's flag (write 0.0),
//       2. copy the activation (src on the producer -> dst on the consumer),
//       3. set the consumer's flag (write 1.0).
//     One wave launch drains all three, in order.
//
//   * The CONSUMER's device spins on its own flag until it leaves zero. The
//     flag is in the CONSUMER's memory, written by the producer over peer
//     access — a direction MEASURED to be visible to the consumer's device
//     with no host sync (see the `a_peer_write_is_visible_to_the_consumer_
//     device` gate). The consumer never reads the producer's memory.
//
// The ORDER of the three tasks is what makes the flag safe. The producer
// clears the flag itself, at the START of its own step, so the consumer never
// has to reset anything: there is no window in which a late reset could
// clobber the next step's signal (the failure mode of a consumer-side reset).
//
// The consumer's wait is bounded and reports what it saw, so a producer that
// never ran shows up as a TIMEOUT rather than as a wedged device.
pub struct BoundaryHandoff {
    /// The producer's ring: it holds the three PEER_COPY tasks.
    ring: crate::kernels::task_ring::TaskRingContext,
    /// The producer's ring TAIL counter — the MONOTONIC step counter the
    /// consumer polls over peer access.
    ///
    /// It lives on the PRODUCER's side on purpose. MEASURED: a peer READ is
    /// coherent on this hardware (the consumer sees the producer's advance),
    /// but a peer WRITE is NOT — the consumer that read a line before the
    /// write keeps seeing the stale value, which is why an earlier
    /// flag-in-consumer-memory design was withdrawn.
    tail_ptr: u64,
    /// Where the consumer's wait kernel records what it observed.
    observed: Box<dyn grim_tensor::backend::BackendStorage>,
    /// The CONSUMER's "last seen" register: the producer's counter value it
    /// already consumed. The wait compares against it, so the consumer knows a
    /// NEW handoff landed rather than seeing the old one again.
    _last_seen: Box<dyn grim_tensor::backend::BackendStorage>,
}

/// How many times the consumer's wait kernel spins before it gives up.
///
/// Each spin costs a short sleep, so this is a few hundred milliseconds of
/// patience — long enough for a producer that is merely late, short enough
/// that a producer that never ran is reported rather than waited on.
pub const BOUNDARY_WAIT_SPINS: u32 = 2_000_000;

impl std::fmt::Debug for BoundaryHandoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundaryHandoff")
            .field("peers", &self.ring.plan().peers().len())
            .field("tail_ptr", &self.tail_ptr)
            .finish()
    }
}

impl BoundaryHandoff {
    /// Prepare a handoff from `src` (on the producer) to `dst` (on the
    /// consumer), copying `elems` floats.
    ///
    /// The peer grant must already be enabled producer -> consumer: the wave's
    /// peer arm writes the CONSUMER's memory directly, which is a different
    /// mechanism from `hipMemcpyPeerAsync` (the copy engine).
    pub fn prepare(
        producer: &crate::RocmDevice,
        consumer: &crate::RocmDevice,
        src: &crate::memory::storage::RocmStorage,
        dst: &crate::memory::storage::RocmStorage,
        elems: u32,
    ) -> Result<Self> {
        use crate::task_graph::{arg_role, op, TaskGraphBuilder};

        if elems == 0 {
            return Err(Error::Backend(
                "boundary handoff: a copy of zero elements is not a handoff".into(),
            ));
        }
        let src_ptr = src
            .device_ptr_u64()
            .ok_or_else(|| Error::Backend("boundary handoff: src has no device ptr".into()))?;
        let dst_ptr = dst
            .device_ptr_u64()
            .ok_or_else(|| Error::Backend("boundary handoff: dst has no device ptr".into()))?;

        // A one-element u32 device buffer on the given device.
        let u32_scalar = |dev: &crate::RocmDevice, v: u32| -> Result<Box<dyn grim_tensor::backend::BackendStorage>> {
            Ok(dev.from_cpu_bytes(
                &v.to_ne_bytes(),
                &grim_tensor::Shape::new(vec![1]),
                grim_tensor::dtype::DType {
                    arith: grim_tensor::ArithType::U32,
                    storage: grim_tensor::dtype::Storage::Native,
                },
            )?)
        };
        let ptr_of = |st: &Box<dyn grim_tensor::backend::BackendStorage>, what: &str| -> Result<u64> {
            st.as_any()
                .downcast_ref::<crate::memory::storage::RocmStorage>()
                .ok_or_else(|| Error::Backend(format!("boundary handoff: {what} is not RocmStorage")))?
                .device_ptr_u64()
                .ok_or_else(|| Error::Backend(format!("boundary handoff: {what} has no device ptr")))
        };

        // Where the consumer's wait kernel records what it observed, and the
        // consumer's "last seen" register: zero, so the producer's first
        // advance (a non-zero tail) differs from it.
        let observed = u32_scalar(consumer, 0)?;
        let observed_ptr = ptr_of(&observed, "observed")?;
        let last_seen = u32_scalar(consumer, 0)?;
        let last_seen_ptr = ptr_of(&last_seen, "last_seen")?;

        // The IR: three PEER_COPY nodes, in the order the wave must run them.
        let mut gb = TaskGraphBuilder::new();
        let h_src = gb.push_arg(src_ptr, elems as u64 * 4, arg_role::ACT_IN);
        let h_dst = gb.push_arg(dst_ptr, elems as u64 * 4, arg_role::PEER);
        // The activation copy. The COMPLETION is the ring's own tail counter:
        // the wave advances it as it drains, and the consumer polls it over
        // peer access. So there is no completion task to stage — the copy IS
        // the work, and the tail IS the signal.
        gb.push(
            op::PEER_COPY,
            [elems, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[h_src, h_dst],
            &[],
            0,
            0,
        );
        let graph = gb.finish();

        let ring = crate::kernels::task_ring::TaskRingContext::from_graph(producer, &graph)?
            .ok_or_else(|| {
                Error::Backend("boundary handoff: the ring did not bind the peer copies".into())
            })?;

        // The producer's tail counter: the MONOTONIC step counter the
        // consumer polls.
        let tail_ptr = ring.tail_ptr()?;

        // Resolve the WAIT kernel on the CONSUMER, eagerly and harmlessly:
        // the on-stream launch path only launches, so the entry has to be in
        // the resolved-function cache under its own name. A zero-spin launch
        // returns at once and still does the resolution.
        {
            let _guard = crate::device::util::DeviceGuard::set(consumer.ordinal as i32);
            crate::kernels::task_ring::launch_scythe_wait_counter(
                consumer,
                tail_ptr,
                last_seen_ptr,
                0,
                observed_ptr,
            )?;
        }

        Ok(Self {
            ring,
            tail_ptr,
            observed,
            _last_seen: last_seen,
        })
    }

    /// Launch the copy on the PRODUCER's stream.
    ///
    /// The wave lands in that stream's ORDER, so a caller that has already
    /// launched the producer's graph on the same stream gets the copy strictly
    /// after it. One host call; no host event.
    pub fn launch_copy(&self, producer: &crate::RocmDevice, producer_stream: *mut c_void) -> Result<()> {
        self.ring.launch_staged_on_stream(producer, producer_stream)?;
        Ok(())
    }

    /// Launch the CONSUMER's device-side wait on the CONSUMER's stream.
    ///
    /// The kernel spins until the flag leaves zero. It is launched BEFORE the
    /// consumer's graph on the same stream, so the graph starts only after the
    /// handoff landed. One host call; no host event.
    ///
    /// `stream` must be the consumer SEGMENT's stream — the consumer's graph
    /// runs on that one, and the wait has to be ordered before it.
    pub fn launch_wait(&self, consumer: &crate::RocmDevice, stream: *mut c_void) -> Result<()> {
        let obs_ptr = self
            .observed
            .as_any()
            .downcast_ref::<crate::memory::storage::RocmStorage>()
            .and_then(|r| r.device_ptr_u64())
            .ok_or_else(|| Error::Backend("boundary handoff: observed has no device ptr".into()))?;
        let last_seen_ptr = self
            ._last_seen
            .as_any()
            .downcast_ref::<crate::memory::storage::RocmStorage>()
            .and_then(|r| r.device_ptr_u64())
            .ok_or_else(|| Error::Backend("boundary handoff: last_seen has no device ptr".into()))?;
        crate::kernels::task_ring::launch_scythe_wait_counter_on_stream(
            consumer,
            self.tail_ptr,
            last_seen_ptr,
            BOUNDARY_WAIT_SPINS,
            obs_ptr,
            stream,
        )?;
        Ok(())
    }

    /// What the consumer's wait observed: zero means it timed out.
    pub fn observed(&self) -> Result<u32> {
        let r = self
            .observed
            .as_any()
            .downcast_ref::<crate::memory::storage::RocmStorage>()
            .ok_or_else(|| Error::Backend("boundary handoff: observed is not RocmStorage".into()))?;
        Ok(r.to_cpu_vec_u32()?.first().copied().unwrap_or(0))
    }

    /// The producer's tail counter, read on the host.
    pub fn tail_value(&self) -> Result<u32> {
        Ok(self.ring.tail_value()?)
    }
}


#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn bounded_helper_rejects_negative_or_oob() {
        assert!(bounded(0, 2));
        assert!(bounded(1, 2));
        assert!(!bounded(-1, 2));
        assert!(!bounded(2, 2));
    }

    #[test]
    fn is_instinct_helper_is_gxfx() {
        assert!(is_instinct("gfx908"));
        assert!(is_instinct("gfx942"));
        assert!(is_instinct("gfx940"));
        assert!(!is_instinct("gfx1036"));
        assert!(!is_instinct("gfx1200"));
        assert!(!is_instinct(""));
    }
}
