//! Phase-3 §3.8 — P2P routing + host-staging primitives.
//!
//! This module is the typed *decision + staging* surface. The bridge from
//! `RouteLink` verdict to the actual memcpy primitive lives on `RocmDevice`
//! (`copy_via_route`) — it calls `peer_status` → `to_route_link` → `copy_route`,
//! which dispatches `hipMemcpyPeerAsync` (PeerDirect) or a D2H→H2D
//! `hipMemcpyAsync` pair (HostBounce).
//!
//! Two primitives:
//!
//! - [`RouteLink`]: typed verdict — `PeerDirect` (xGMI-class) or
//!   `HostBounce` (peer-disabled, or PCIe-direct was too costly at
//!   this size).
//! - [`to_route_link`]: small-link classifier. The PCIe threshold is
//!   tunable because consumer PCIe direct P2P is fast-cheap for
//!   small calls but inferior to host pinned for large transfers
//!   (the host staging is amortised, the PCIe bandwidth is bounded).
//! - [`HostStagingBuffer`]: pinned host buffer for the host-bounce
//!   route. Allocates via `hipHostMalloc` on a running CUDA box; on a
//!   GPU-less box the call returns `Err` rather than allocating a
//!   non-pinned `Vec` (which would violate `rust-gpu-discipline`'s real
//!   pinned-memory requirement).
//!
//! Skill attribution:
//! - `rocm-multi-gpu-rccl` — host bounce explicit; small PCIe direct
//!   wins on latency, large host pinned wins on bandwidth.
//! - `rust-gpu-parallelism` — pinned host memory lifecycle: allocate
//!   once, reuse across bounces.
//! - `rust-gpu-discipline` Section 3 — no silent CPU fallback when
//!   peer access is enabled; pinned memory is real or it errors.
//! - `rust-ml-llm-architecture` — backend isolation: routing primitives
//!   live in the ROCm crate.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::peer_access::P2PStatus;

/// Typed verdict for how to execute one inter-device memcpy.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RouteLink {
    /// Use `hipMemcpyPeerAsync` directly (or `hipMemcpyAsync` D2D
    /// across devices when peer isn't requested).
    PeerDirect,
    /// Bounce through a single `HostStagingBuffer` — D2H to host
    /// pinned, then H2D to the destination.
    HostBounce,
}

impl Default for RouteLink {
    /// Default to defensive host bounce. Until
    /// `to_route_link(...)` runs we don't trust the link.
    fn default() -> Self {
        RouteLink::HostBounce
    }
}

/// Decide how to route a memcpy across two devices.
///
/// `pcie_threshold_bytes` is the inclusive upper bound on size under
/// `P2PStatus::Pcie` — transfers up to and including the threshold
/// stay PeerDirect, anything larger pivots to HostBounce. The default
/// is `u64::MAX` (treat every PCIe transfer as PeerDirect); a tuned
/// grim-server caller would override to a smaller value based on
/// observed PCIe bandwidth on the deployment box.
pub const fn to_route_link(status: P2PStatus, bytes: u64, pcie_threshold_bytes: u64) -> RouteLink {
    match status {
        // Native peer DMA (xGMI class): always PeerDirect.
        P2PStatus::P2P => RouteLink::PeerDirect,
        // Peer access denied or unreachable: always HostBounce.
        P2PStatus::Host => RouteLink::HostBounce,
        // Peer enabled but on PCIe. The decision is bytes vs threshold.
        P2PStatus::Pcie => {
            if bytes <= pcie_threshold_bytes {
                RouteLink::PeerDirect
            } else {
                RouteLink::HostBounce
            }
        }
    }
}

/// Pinned host allocation for the host-bounce path.
///
/// Allocated via `hipHostMalloc` on a running ROCm box; on a GPU-less
/// box the constructor returns `Err` (we never silently `Vec::new`
/// because that wouldn't be pinned and wouldn't satisfy the design
/// intent — a non-pinned bounce is slower and defeats the host bounce's
/// reason d'être).
/// Pinned host memory buffer allocated via `hipHostMalloc`.
///
/// # Safety
///
/// `HostStagingBuffer` wraps a raw pointer to pinned host memory allocated by
/// `hipHostMalloc`. The allocation is valid process-wide and can be accessed
/// from any thread (Send + Sync) because:
/// - The memory is page-locked and portable across CPU cores.
/// - `bytes()` and `bytes_mut()` return bounded slices that prevent out-of-bounds
///   access.
/// - The lifecycle is managed by the caller (freed via `hipHostFree` or drop).
///
/// Current enforcement: all live call paths into this type pass through the
/// staging cache mutex (`STAGING_CACHE`) in p2p_route.rs, plus
/// `AppState.engine: Mutex<Engine>` in grim-server. No concurrent access is
/// possible through the server's actual API today. Do NOT remove these locks or
/// add a second concurrent access path without auditing the buffer lifecycle
/// here first.
pub struct HostStagingBuffer {
    ptr: *mut c_void,
    size: usize,
}

// SAFETY: see the struct-level safety documentation above.
unsafe impl Send for HostStagingBuffer {}
unsafe impl Sync for HostStagingBuffer {}

impl HostStagingBuffer {
    /// Allocate `size` bytes of pinned host memory. Returns `Err` if
    /// the runtime refuses the size (zero-byte or out-of-quota) or
    /// if the runtime is unavailable.
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::Backend(
                "HostStagingBuffer::new: size must be > 0".into(),
            ));
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        crate::device::helpers::check_hip("hipHostMalloc", unsafe {
            hipHostMalloc(&mut ptr, size, 0)
        })?;
        Ok(Self { ptr, size })
    }

    /// Allocate only if `route` is `HostBounce`. Returns `None` when
    /// the caller asked for `PeerDirect` (no staging needed) or when
    /// the runtime refuses the size on a GPU-less box.
    pub fn for_route(route: RouteLink, size: usize) -> Option<Self> {
        match route {
            RouteLink::HostBounce => Self::new(size).ok(),
            RouteLink::PeerDirect => None,
        }
    }

    /// Backing site as a `*mut c_void` for `hipMemcpyAsync` /
    /// `hipMemcpyDtoHAsync`. `Null` only if allocation failed (in
    /// which case the `Result` was already returned from `new`).
    pub fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Allocated size in bytes (the rounded-up capacity; matches
    /// the size passed to `hipHostMalloc`).
    pub fn size(&self) -> usize {
        self.size
    }

    /// Host-side mutable view of the staging buffer. Bounded by
    /// `self.size`. `None` if `self.ptr` is null.
    pub fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        if self.ptr.is_null() {
            return None;
        }
        // SAFETY: `self.ptr` is a valid `hipHostMalloc`-aligned block
        // sized `self.size`. The lifetime is `&mut self`-bound, so we
        // don't leak aliasing. The `[u8]` representation is
        // unspecified for HIP allocations — pinning only requires the
        // host allocation to be page-locked, not for the contents to
        // be `u8`-readable without copying. This is fine for the
        // host-bounce path (no GPU view).
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                self.ptr as *mut u8,
                self.size,
            ))
        }
    }

    /// Host-side immutable view of the staging buffer.
    pub fn bytes(&self) -> Option<&[u8]> {
        if self.ptr.is_null() {
            return None;
        }
        unsafe { Some(std::slice::from_raw_parts(self.ptr as *const u8, self.size)) }
    }
}

impl Drop for HostStagingBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Free via the same FFI the allocation used. `hipHostFree`
            // tolerates being called with a `*mut c_void` returned by
            // `hipHostMalloc`; the runtime handles the size internally.
            let _ = unsafe { hipHostFree(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// HIP symbols we call directly. The crate root's `hipHostMalloc` already
// declares this; we shadow locally to keep the module self-contained.
use crate::{HipMemcpyKind, hipHostFree, hipHostMalloc, hipMemcpyAsync};

/// Performs inter-device copy routing either via direct peer copies or host bounce staging.
///
/// Host-bounce staging buffers are cached per stream: a D2H+H2D pair is
/// enqueued on `stream`, so the next bounce on the SAME stream is
/// queue-ordered after it and can safely reuse the pinned buffer. This
/// removes a `hipHostMalloc`/`hipHostFree` pair (~ms each) from every
/// bounced copy — the per-token cost of peer-denied TP routing.
pub fn copy_route(
    src_device: i32,
    dst_device: i32,
    src_ptr: *const c_void,
    dst_ptr: *mut c_void,
    len: usize,
    route: RouteLink,
    stream: *mut c_void,
) -> Result<()> {
    match route {
        RouteLink::PeerDirect => {
            crate::rccl::p2p_memcpy_async(dst_ptr, dst_device, src_ptr, src_device, len, stream)?;
        }
        RouteLink::HostBounce => {
            let staging = take_staging(stream, len)?;
            crate::device::helpers::check_hip("copy_route: D2H copy to staging", unsafe {
                hipMemcpyAsync(
                    staging.as_device_ptr(),
                    src_ptr,
                    len,
                    HipMemcpyKind::DeviceToHost,
                    stream,
                )
            })?;
            crate::device::helpers::check_hip("copy_route: H2D copy from staging", unsafe {
                hipMemcpyAsync(
                    dst_ptr,
                    staging.as_device_ptr(),
                    len,
                    HipMemcpyKind::HostToDevice,
                    stream,
                )
            })?;
        }
    }
    Ok(())
}

/// Cached pinned staging buffer, reused across host-bounce copies on one
/// stream. `None` until the first bounce; grown when a larger transfer
/// arrives. A bounce on a different stream gets a fresh one-shot buffer
/// (concurrent streams must not share staging without an event fence).
/// Cached pinned staging buffer, reused across host-bounce copies on one stream.
///
/// # Safety
///
/// `StagingCache` holds a raw HIP stream handle and a `HostStagingBuffer`. The
/// stream handle is valid process-wide and the buffer is Send+Sync (see its
/// safety documentation). The cache itself is safe to send between threads
/// because the stream handle is opaque and process-local. It is Sync because
/// all access to the cache is serialized by the `STAGING_CACHE` mutex — no
/// concurrent access to the buffer or stream occurs without holding that lock.
///
/// Current enforcement: the `STAGING_CACHE` static mutex serializes all access.
/// Do NOT remove this mutex or add a second concurrent access path without
/// adding internal synchronization here first.
struct StagingCache {
    stream: *mut c_void,
    buf: HostStagingBuffer,
}
// SAFETY: see the struct-level safety documentation above.
unsafe impl Send for StagingCache {}
unsafe impl Sync for StagingCache {}

static STAGING_CACHE: std::sync::Mutex<Option<StagingCache>> = std::sync::Mutex::new(None);

/// Staging memory pinned for the duration of one bounce. Holds the cache
/// lock so a concurrent regrow cannot free the buffer mid-enqueue; the lock
/// is only held across two async enqueue calls (microseconds), and
/// same-stream bounces serialize on the stream regardless.
struct StagingGuard<'a> {
    _lock: std::sync::MutexGuard<'a, Option<StagingCache>>,
    _one_shot: Option<HostStagingBuffer>,
    ptr: *mut c_void,
}

impl StagingGuard<'_> {
    fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

/// Obtain staging memory for a bounce on `stream`: the cached buffer when
/// the stream matches and capacity suffices, a regrown replacement when a
/// bigger same-stream transfer arrives, otherwise a one-shot allocation.
fn take_staging(stream: *mut c_void, len: usize) -> Result<StagingGuard<'static>> {
    let mut guard = STAGING_CACHE
        .lock()
        .map_err(|_| Error::Backend("staging cache mutex poisoned".into()))?;
    let reuse_cached = guard
        .as_ref()
        .is_some_and(|c| c.stream == stream && c.buf.size() >= len);
    if reuse_cached {
        // SAFETY: the lock is held for the whole lease, so the buffer cannot
        // be swapped out or freed underneath us; same-stream reuse is
        // additionally queue-ordered by `stream` itself.
        let ptr = unsafe { std::ptr::NonNull::new_unchecked(guard.as_ref().unwrap().buf.as_device_ptr()) }.as_ptr();
        return Ok(StagingGuard {
            _lock: guard,
            _one_shot: None,
            ptr,
        });
    }
    let fresh = HostStagingBuffer::new(len)?;
    let ptr = fresh.as_device_ptr();
    let same_stream_grow = guard
        .as_ref()
        .is_some_and(|c| c.stream == stream && c.buf.size() < len);
    if same_stream_grow || guard.is_none() {
        // First use, or a larger transfer on the same stream: (re)seed the
        // cache so subsequent bounces reuse this allocation.
        *guard = Some(StagingCache { stream, buf: fresh });
        Ok(StagingGuard {
            _lock: guard,
            _one_shot: None,
            ptr,
        })
    } else {
        // Different stream: one-shot buffer, freed when the leases drops.
        // The enqueued D2H+H2D pair completes before the free only because
        // the caller's stream drains; callers own that ordering (same
        // contract as the previous per-call allocation).
        Ok(StagingGuard {
            _lock: guard,
            _one_shot: Some(fresh),
            ptr,
        })
    }
}

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn to_route_link_classification_is_total() {
        // Every (status, bytes, threshold) combination yields exactly
        // one valid `RouteLink`; the qualifier functions are derivable.
        let cases: &[((P2PStatus, u64, u64), RouteLink)] = &[
            ((P2PStatus::P2P, 0, 0), RouteLink::PeerDirect),
            ((P2PStatus::P2P, 1024, 0), RouteLink::PeerDirect),
            ((P2PStatus::Host, 0, 0), RouteLink::HostBounce),
            ((P2PStatus::Host, 1024, u64::MAX), RouteLink::HostBounce),
            ((P2PStatus::Pcie, 0, u64::MAX), RouteLink::PeerDirect),
            ((P2PStatus::Pcie, 1024, u64::MAX), RouteLink::PeerDirect),
            ((P2PStatus::Pcie, 1024, 1023), RouteLink::HostBounce),
            ((P2PStatus::Pcie, 1024, 1024), RouteLink::PeerDirect),
        ];
        for (inp, want) in cases.iter().copied() {
            assert_eq!(to_route_link(inp.0, inp.1, inp.2), want, "case {:?}", inp);
        }
    }
}
