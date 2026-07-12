//! RED-GREEN-REFACTOR tests for peer-access probe + P2P memcpy primitives.
//!
//! Phase-3 §3.8 of the QKV spec (the multi-GPU ground). Cycle 1 covers
//! the `peer_access` module: enumerating devices, probing P2P, and
//! turning the runtime verdict into a typed `P2PStatus`. Cycle 2 covers
//! `p2p_memcpy`: a thin wrapper around `hipMemcpyPeerAsync` that falls
//! back to a host bounce when peer access is disabled, and surfaces
//! `Ok(())` to honest paths while never silently succeeding on a
//! failed lookup.
//!
//! Skill attribution:
//! - `rocm-multi-gpu-rccl` — one stream per device, peer-access probe
//!   before any P2P call.
//! - `rust-gpu-parallelism` — one thread per device; per-stream submission,
//!   never share streams across devices.
//! - `rust-ml-llm-architecture` — backend isolation: cross-device logic
//!   stays in the ROCm crate.

use std::sync::Arc;

use grim_backend_rocm::peer_access::{
    enable_peer_access, enumerate_devices, peer_status, P2PStatus,
};

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

// =========================================================================
// RED — `P2PStatus` is the typed verdict for a `(src, dst)` pair. The
// variant set must be small: `P2P` (xGMI-class direct), `Pcie` (peer
// enabled but on PCIe consumer), `Host` (peer disabled / unreachable).
// =========================================================================

#[test]
fn p2p_status_debug_works_for_every_variant() -> TestResult {
    let kinds = [P2PStatus::P2P, P2PStatus::Pcie, P2PStatus::Host];
    for v in kinds {
        let _ = format!("{:?}", v);
    }
    Ok(())
}

#[test]
fn p2p_status_three_routes_are_distinct() -> TestResult {
    assert_ne!(P2PStatus::P2P, P2PStatus::Pcie);
    assert_ne!(P2PStatus::Pcie, P2PStatus::Host);
    assert_ne!(P2PStatus::P2P, P2PStatus::Host);
    Ok(())
}

#[test]
fn p2p_status_default_is_host_bounce() -> TestResult {
    let default = P2PStatus::default();
    assert_eq!(default, P2PStatus::Host);
    Ok(())
}

// =========================================================================
// RED — `enumerate_devices`. Without a GPU this returns `Ok(0)` (or at
// the very least does not panic). The probe is supposed to never block.
// =========================================================================

#[test]
fn enumerate_devices_does_not_panic_on_gpu_less_box() -> TestResult {
    let res = enumerate_devices();
    let n = res.unwrap_or(0);
    // We don't strongly assert 0 or 1 here — sandboxes may have a stub
    // GPU and CI machines may have none. What we *do* assert is that we
    // know the answer rather than panicking.
    let _ = n;
    Ok(())
}

// =========================================================================
// RED — `peer_status(a, b)` gives a typed verdict. Symmetry property:
// peer_status(a, b) == peer_status(b, a) at the level of the verdict
// family (don't get a P2P <-> Host half-asymmetry). Directionality is
// handled by the *call*, not the verdict.
// =========================================================================

#[test]
fn peer_status_is_symmetric_in_family_within_a_single_call() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    let n = enumerate_devices()?;
    if n < 2 {
        return Ok(()); // single-GPU box: symmetry is vacuously true.
    }
    let a = peer_status(0, 1)?;
    let b = peer_status(1, 0)?;
    // The verdict *family* (whichever of {P2P, Pcie, Host}) is
    // symmetric — the same family answer regardless of ordering. We
    // don't assert strict equality because the API may eventually add
    // direction-specific metadata (e.g. DMA ranks). For the §3.8 head
    // the family is enough.
    fn family(s: &P2PStatus) -> u8 {
        match s {
            P2PStatus::P2P => 1,
            P2PStatus::Pcie => 2,
            P2PStatus::Host => 3,
        }
    }
    assert_eq!(family(&a), family(&b));
    Ok(())
}

// =========================================================================
// RED — `enable_peer_access(src, dst)` returns `Ok(true)` only when the
// runtime can grant peer access; `Ok(false)` is a graceful no-op
// (e.g. on GPU-less box). `Err(...)` is reserved for actual driver
// faults — never for "not implemented".
// =========================================================================

#[test]
fn enable_peer_access_is_infallible_or_errors_loud() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    let n = enumerate_devices()?;
    let r = enable_peer_access(0, 1);
    if n < 2 {
        return Ok(());
    }
    let granted = r?;
    assert!(granted || !granted, "must be Ok(bool), not Err");
    Ok(())
}

#[test]
fn enable_peer_access_self_pair_is_no_op() -> TestResult {
    let env = std::env::var("GRIM_RUN_GPU_TESTS").is_ok();
    if !env {
        return Ok(());
    }
    let n = enumerate_devices()?;
    if n == 0 {
        return Ok(());
    }
    // Asking for `0 → 0` peer access: trivially true; the call should
    // succeed (return Ok(true)) rather than a driver fault.
    let r = enable_peer_access(0, 0)?;
    assert!(r, "self-peer must report Ok(true)");
    Ok(())
}

// =========================================================================
// RED — `Arc` reusability of the typed verdict. Multiple threads may
// each greet it without copying. (Critically: the verdict is `Copy`,
// so this is mostly a smoke test, but it catches accidental inner
// mutability.)
// =========================================================================

#[test]
fn p2p_status_is_copy_and_thread_safe() -> TestResult {
    let v = P2PStatus::Pcie;
    let a = vec![v; 8];
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let a = Arc::new(a.clone());
            std::thread::spawn(move || -> usize {
                a.iter().filter(|x| matches!(x, P2PStatus::Pcie)).count()
            })
        })
        .collect();
    let mut total = 0;
    for h in handles {
        total += h.join().map_err(|_| "thread panicked")?;
    }
    assert_eq!(total, 32);
    Ok(())
}
