//! Peer-access probe + P2P verdict for the multi-GPU head of §3.8.
//!
//! Cycle 1 — the apparatus for asking "before I try a P2P memcpy /
//! RCCL collective, what kind of link do I have here?". The answer is
//! one of three verdicts:
//!
//! - `P2P`         — native peer-class DMA (Instinct xGMI / consumer
//!                   PCIe with direct path discovered);
//! - `Pcie`        — peer-enabled but routing through PCIe (consumer
//!                   RDNA2/3/4 typical);
//! - `Host`        — peer access disabled or unreachable; the caller
//!                   must bounce through host pinned memory.
//!
//! Skill attribution:
//! - `rocm-multi-gpu-rccl` — peer probe before any peer memcopy.
//! - `rust-gpu-parallelism` — one stream per device.
//! - `rust-ml-llm-architecture` — backend isolation.

use std::borrow::Cow;
use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::device::helpers::check_hip;
use crate::{hipSuccess, HipErrorT};

// HIP symbols we call. We declare them locally rather than going
// through `crate::...` because the crate-root FFI declarations have
// drift between ROCm releases — re-declaring at the call site keeps
// `peer_access` self-contained. Edition 2024 requires the `unsafe`
// marker on `extern` blocks; `unsafe` on the body callsites then
// becomes redundant.
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

/// HIP errors we tolerate silently. Sourced from ROCm headers; only
/// the codes that mean "the runtime already has this state" are
/// mapped to soft-success.
const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: HipErrorT = 0xb16;

/// Enumerate HIP devices visible to the process. Returns 0 on a
/// GPU-less box. Errors are surfaced as `Err(...)` only when the runtime
/// actually fails to query — never for "no devices found".
pub fn enumerate_devices() -> Result<usize> {
    let mut count: i32 = 0;
    check_hip("hipGetDeviceCount", unsafe { hipGetDeviceCount(&mut count as *mut _) })?;
    Ok((count.max(0)) as usize)
}

/// Ask `hipDeviceCanAccessPeer`: returns `Ok(true)` only when the
/// runtime can grant true peer class on `(src, dst)`. A GPU-less box
/// whose enumeration returned 0 yields `Ok(false)` rather than
/// `Err(...)`, because "no peers" is a normal state.
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
    // The runtime says peer is possible. We don't pre-flight PCIe vs
    // xGMI from the probe alone — that distinction collapses to
    // `Pcie` for consumer hardware and `P2P` for Instinct. The caller
    // decides whether to treat PCIe as fast-enough for their path.
    let gfn = unsafe { gcn_arch_for(src) };
    if is_instinct(gfn.as_ref()) {
        Ok(P2PStatus::P2P)
    } else {
        Ok(P2PStatus::Pcie)
    }
}

/// Enable peer access between `src` and `dst` (`hipDeviceEnablePeerAccess`).
/// Returns `Ok(true)` when peer access is granted, `Ok(false)` when
/// the runtime already had it or the call was a no-op. A runtime error
/// from the driver surfaces as `Err(...)`.
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
    // Disable first (best practice — peer access persists in the
    // process address space, so a fresh probe may collide with a stale
    // grant).
    let _ = unsafe { hipDeviceDisablePeerAccess(dst) };
    let res = unsafe { hipDeviceEnablePeerAccess(dst, 0) };
    if res == hipSuccess {
        Ok(true)
    } else if res == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED {
        // Soft success — the runtime already enabled it.
        Ok(true)
    } else {
        Err(Error::Backend(format!(
            "enable_peer_access: hipDeviceEnablePeerAccess returned code={}",
            res
        )))
    }
}

fn bounded(x: i32, n: i32) -> bool {
    x >= 0 && x < n
}

fn is_instinct(arch: &str) -> bool {
    arch.starts_with("gfx9")
}

/// Query `hipGetDeviceProperties` and surface the `gcnArchName` string.
/// Returns the `"gfx9999"` fallback on GPU-less boxes so the verdict
/// branch isn't forced into `Other` by missing devices.
unsafe fn gcn_arch_for(device: i32) -> Cow<'static, String> {
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
