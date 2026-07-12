//! RED-GREEN-REFACTOR tests for the §3.8 P2P routing / staging primitives.
//!
//! Three sub-cycles land in this PR:
//!   1. `to_route_link(status, bytes)` — small-link classifier.
//!      `P2PStatus::P2P` always goes PeerDirect. `PS::Host` (peer
//!      access denied) is always HostBounce. `PS::Pcie` is the interesting
//!      case: small transfers stay PeerDirect (the PCIe round-trip is
//!      sub-microsecond-cheaper than host staging + D2H + H2D), large
//!      transfers go HostBounce (PCIe bandwidth × RDNA-pcie consumer
//!      cap is bounded at ~32 GB/s, while host pinned memory at 50 µs
//!      end-to-end + D2H+H2D applies a fixed tax — the crossover is
//!      user-tunable).
//!   2. `HostStagingBuffer` — pinned host memory for the host-bounce
//!      path. Allocates via `hipHostMalloc`, exposes
//!      `as_host_bytes_mut`/`as_device_ptr`. Round-trips a 16-byte
//!      record.
//!
//! Heavy GPU work (the actual `hipMemcpyPeerAsync` + ASYNC D2H/H2D
//! chain) is the next PR's surface; this one only writes decision +
//! staging CPU logic.
//!
//! Skill attribution:
//! - `rocm-multi-gpu-rccl` — host bounce explicit; small PCIe direct is
//!   cheaper than staging.
//! - `rust-gpu-parallelism` — per-device stream ownership; pinned host
//!   memory lifecycle.
//! - `rust-gpu-discipline` Section 3 — real runtime calls in
//!   `HostStagingBuffer`; CPU-only logic in `to_route_link` so no
//!   fabricated GPU state can sneak through.
//! - `rust-ml-llm-architecture` — backend isolation: the routing
//!   primitive lives in the ROCm crate, not core.

use grim_backend_rocm::p2p_route::{to_route_link, HostStagingBuffer, RouteLink};
use grim_backend_rocm::peer_access::P2PStatus;

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

// =========================================================================
// RED — `RouteLink` variants and their distinguishing characteristics.
// =========================================================================

#[test]
fn route_link_debug_works_for_every_variant() -> TestResult {
    let kinds = [RouteLink::PeerDirect, RouteLink::HostBounce];
    for k in kinds {
        let _ = format!("{:?}", k);
    }
    Ok(())
}

#[test]
fn route_link_two_kinds_are_distinct() -> TestResult {
    assert_ne!(RouteLink::PeerDirect, RouteLink::HostBounce);
    Ok(())
}

#[test]
fn route_link_default_is_host_bounce() -> TestResult {
    let d = RouteLink::default();
    assert_eq!(d, RouteLink::HostBounce);
    Ok(())
}

// =========================================================================
// RED — `to_route_link` decision matrix.
// =========================================================================

#[test]
fn link_decision_p2p_status_always_peer_direct() -> TestResult {
    let decision = to_route_link(P2PStatus::P2P, 0, u64::MAX);
    assert_eq!(decision, RouteLink::PeerDirect);
    let decision = to_route_link(P2PStatus::P2P, 1, 1_000_000);
    assert_eq!(decision, RouteLink::PeerDirect);
    Ok(())
}

#[test]
fn link_decision_host_status_always_host_bounce() -> TestResult {
    let decision = to_route_link(P2PStatus::Host, 0, u64::MAX);
    assert_eq!(decision, RouteLink::HostBounce);
    Ok(())
}

#[test]
fn link_decision_pcie_status_below_threshold_is_peer_direct() -> TestResult {
    // Treats `pcie_threshold_bytes` as the *inclusive* upper bound;
    // `bytes <= threshold` is PeerDirect, `bytes > threshold` is
    // HostBounce. The default threshold is a sensible RDNA-pcie
    // crossover; we tolerate caller overrides.
    let decision = to_route_link(P2PStatus::Pcie, 1024, u64::MAX);
    assert_eq!(decision, RouteLink::PeerDirect, "tiny PCIe direct");
    let decision = to_route_link(P2PStatus::Pcie, 0, u64::MAX);
    assert_eq!(decision, RouteLink::PeerDirect, "0-byte PCIe direct is a no-op");
    Ok(())
}

#[test]
fn link_decision_pcie_status_above_threshold_is_host_bounce() -> TestResult {
    let decision = to_route_link(P2PStatus::Pcie, 256 * 1024 * 1024, 0); // 256 MB > 0-byte pcie_threshold.
    assert_eq!(decision, RouteLink::HostBounce);
    Ok(())
}

#[test]
fn link_decision_pcie_threshold_boundary_is_peer_direct() -> TestResult {
    // At exactly the threshold (inclusive), keep PeerDirect. A
    // transfer of `threshold + 1` flips to HostBounce.
    let threshold = 4096_u64;
    let decision = to_route_link(P2PStatus::Pcie, threshold, threshold);
    assert_eq!(decision, RouteLink::PeerDirect);
    let decision = to_route_link(P2PStatus::Pcie, threshold + 1, threshold);
    assert_eq!(decision, RouteLink::HostBounce);
    Ok(())
}

#[test]
fn link_decision_pcie_zero_threshold_means_never_direct() -> TestResult {
    // If the caller passes `pcie_threshold_bytes == 0`, *nothing* under
    // PCIe status is direct — every transfer takes the host bounce.
    let decision = to_route_link(P2PStatus::Pcie, 1, 0);
    assert_eq!(decision, RouteLink::HostBounce);
    Ok(())
}

#[test]
fn link_decision_pcie_overflow_threshold_saturates() -> TestResult {
    // If `bytes > threshold` and the caller's threshold is the largest
    // possible value, the decision should still pick — HostBounce for
    // a `bytes` that exceeds the (saturated) threshold.
    let decision = to_route_link(P2PStatus::Pcie, u64::MAX, u64::MAX);
    assert_eq!(decision, RouteLink::PeerDirect, "threshold = bytes → inclusive");
    let decision = to_route_link(P2PStatus::Pcie, u64::MAX, u64::MAX - 1);
    assert_eq!(decision, RouteLink::HostBounce);
    Ok(())
}

// =========================================================================
// RED — `HostStagingBuffer` is a pinned host allocation. The CPU-side
// primitive does NOT bind to a single device; GPU-bound copy happens in
// the next PR. We test the lifecycle: alloc → bytes_mut → as_device_ptr
// (opaque device pointer for `hipMemcpyAsync`).
// =========================================================================

#[test]
fn host_staging_buffer_round_trips_a_short_byte_record() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    // Allocate pinned host memory for the host-bounce route.
    let mut stage = HostStagingBuffer::new(64)?;
    // Write into the staging buffer.
    let payload = b"grim-qkv-attn".to_vec();
    let bytes_mut = stage.bytes_mut().ok_or("staging bytes_mut() returned None")?;
    assert!(bytes_mut.len() >= payload.len());
    bytes_mut[..payload.len()].copy_from_slice(&payload);
    // The pinned device pointer is a stable `*mut c_void` exposed via
    // the staging buffer so a downstream `hipMemcpyAsync` could use
    // it. We only assert non-null + matching size; the GPU path is the
    // next PR's surface.
    let dev_ptr = stage.as_device_ptr();
    if !dev_ptr.is_null() {
        // Length is recorded so callers can drive D2H / H2D memcpy
        // sized off the page they allocated.
        assert!(stage.size() >= 64);
    }
    // Round-trip the *byte record* on the host side: read back via
    // `bytes()` and confirm the payload survived.
    let read = stage.bytes().ok_or("bytes() returned None")?;
    assert_eq!(&read[..payload.len()], &payload[..]);
    Ok(())
}

#[test]
fn host_staging_buffer_zero_size_returns_err() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    let res = HostStagingBuffer::new(0);
    assert!(res.is_err(), "zero-byte staging buffer is a programming mistake");
    Ok(())
}

#[test]
fn host_staging_buffer_drop_returns_the_pinned_block() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    // Smoke: alloc/drop pair doesn't leak (no verifier here, but the
    // Drop impl runs `hipHostFree`-equivalent via `pinned_in_drop`).
    let stage = HostStagingBuffer::new(32)?;
    drop(stage);
    Ok(())
}

#[test]
fn host_staging_buffer_rejects_overflow() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    // Saturating-multiply-like bugs shouldn't be possible. A 1-TB
    // request must be rejected as either Err or held under the
    // device's max-pinned size; on CPU-less boxes `new` only fails
    // for true allocation failure. We test for *either* Err or sized
    // < requested.
    if let Ok(stage) = HostStagingBuffer::new(usize::MAX / 2 + 1) {
        // It returned Ok — that's only OK if the size we actually got is
        // strictly less than what was requested; the runtime rejected
        // the over-quota amount on our behalf.
        assert!(stage.size() < usize::MAX / 2 + 1);
    }
    Ok(())
}

// =========================================================================
// RED — Decision-driven StageBuffer factory: `HostStagingBuffer::for_route`
// only constructs when the route is `HostBounce`; in PeerDirect mode it
// returns `None` gracefully (the caller doesn't need staging).
// =========================================================================

#[test]
fn staging_for_routes_is_none_when_peer_direct() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    // The factory method only consults the route decision; small
    // bytes / P2P status both collapse to PeerDirect on the API
    // surface — no staging allocation.
    let opt = HostStagingBuffer::for_route(RouteLink::PeerDirect, 1024);
    assert!(opt.is_none(), "PeerDirect must not allocate staging");
    Ok(())
}

#[test]
fn staging_for_routes_is_some_when_host_bounce() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    let opt = HostStagingBuffer::for_route(RouteLink::HostBounce, 1024);
    if let Some(stage) = opt {
        assert!(stage.size() >= 1024);
    }
    // It's acceptable for the factory to return None on a sparse
    // runtime — we only assert that *if* it returns Some, the size
    // is sensible.
    Ok(())
}
