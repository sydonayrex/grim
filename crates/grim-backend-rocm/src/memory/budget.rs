//! Global ROCm residency policy for VRAM overflow.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Total number of allocations routed to the managed-memory fallback since
/// process start (WI-P3 instrumentation: makes the otherwise-silent
/// oversubscription path observable).
static MANAGED_FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Whether the one-time user-facing warning about the managed-memory fallback
/// has already been emitted (avoid warning spam on every oversubscribed alloc).
static MANAGED_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

/// How many allocations have been routed to `hipMallocManaged` (WI-P3).
pub fn managed_fallback_count() -> usize {
    MANAGED_FALLBACK_COUNT.load(Ordering::Relaxed)
}

/// Whether the one-time managed-fallback warning has been emitted (WI-P3).
pub fn managed_fallback_warned() -> bool {
    MANAGED_FALLBACK_WARNED.load(Ordering::Relaxed)
}

/// Reset the WI-P3 instrumentation (test hook).
pub fn reset_managed_fallback_instrumentation() {
    MANAGED_FALLBACK_COUNT.store(0, Ordering::Relaxed);
    MANAGED_FALLBACK_WARNED.store(false, Ordering::Relaxed);
}

/// Record that an allocation fell back to HIP managed memory and surface the
/// risk to the user once per process. AMD's SVM (the backing mechanism for
/// `hipMallocManaged`) evicts FIFO with no reuse awareness and — as of ROCm
/// 6.8.0 or later — migrates in 2 MiB fault granularity; under genuine
/// oversubscription this can collapse throughput to near zero. grim cannot
/// adopt the driver-level fix (it requires patching AMDGPU/TTM, not
/// upstreamed), so the honest remediation is: say it plainly, once, and point
/// at the lever (`GRIM_ROCM_VRAM_BUDGET_BYTES`).
pub fn note_managed_fallback(ordinal: usize, bytes: usize) {
    MANAGED_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
    if MANAGED_FALLBACK_WARNED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        eprintln!(
            "[grim-backend-rocm] WARNING: allocation of {bytes} bytes on device {ordinal} \
             fell back to HIP managed (host-backed) memory because the model did not fit \
             in VRAM (or GRIM_ROCM_MANAGED_ALLOCATIONS forces it). Managed memory can degrade \
             throughput severely under oversubscription: AMD SVM evicts FIFO without reuse \
             awareness and migrates at 2 MiB granularity on ROCm >= 6.8.0. If performance is \
             unexpectedly slow, reduce the model size or set GRIM_ROCM_VRAM_BUDGET_BYTES so \
             allocations stay in VRAM."
        );
    }
}

/// Decide whether a new allocation should use HIP managed memory.
///
/// `GRIM_ROCM_MANAGED_ALLOCATIONS=always` forces the host-backed tier.
/// `...=auto` uses the live free-memory watermark and an optional
/// `GRIM_ROCM_VRAM_BUDGET_BYTES` ceiling. The policy is intentionally global:
/// model weights, activations, gradients, and temporary kernel outputs all
/// pass through the same allocation seam.
pub fn use_managed_allocation(ordinal: usize, bytes: usize) -> bool {
    let mode = std::env::var("GRIM_ROCM_MANAGED_ALLOCATIONS").unwrap_or_default();
    if matches!(mode.as_str(), "1" | "true" | "always") {
        return true;
    }
    if mode != "auto" {
        return false;
    }
    let (free, total) = crate::vram_info(ordinal);
    let budget = std::env::var("GRIM_ROCM_VRAM_BUDGET_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| total.saturating_mul(9) / 10);
    // Reserve 64 MB cushion to prevent driver OOM locks.
    let effective_free = free.saturating_sub(64 * 1024 * 1024);
    should_use_managed("auto", effective_free, total, bytes as u64, budget)
}

/// Pure form of the residency decision, kept separate from HIP telemetry so
/// policy behavior can be tested without a ROCm device.
fn should_use_managed(mode: &str, free: u64, total: u64, bytes: u64, budget: u64) -> bool {
    match mode {
        "1" | "true" | "always" => true,
        "auto" => free < bytes || total.saturating_sub(free) > budget,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        managed_fallback_count, managed_fallback_warned, note_managed_fallback,
        reset_managed_fallback_instrumentation, should_use_managed,
    };

    #[test]
    fn forced_mode_uses_managed_memory() {
        assert!(should_use_managed(
            "always",
            u64::MAX,
            u64::MAX,
            1,
            u64::MAX
        ));
        assert!(should_use_managed("true", u64::MAX, u64::MAX, 1, u64::MAX));
    }

    #[test]
    fn disabled_mode_keeps_ordinary_allocations() {
        assert!(!should_use_managed("", 0, 1, 2, 0));
        assert!(!should_use_managed("never", 0, 1, 2, 0));
    }

    #[test]
    fn auto_mode_spills_when_request_does_not_fit() {
        assert!(should_use_managed("auto", 1, 100, 2, 100));
    }

    #[test]
    fn auto_mode_spills_when_budget_is_already_exceeded() {
        assert!(should_use_managed("auto", 40, 100, 1, 50));
        assert!(!should_use_managed("auto", 60, 100, 1, 50));
    }

    /// WI-P3: the managed fallback must be observable — note_managed_fallback
    /// bumps the counter and emits the user-facing warning exactly once.
    #[test]
    fn managed_fallback_is_observable_and_warns_once() {
        reset_managed_fallback_instrumentation();
        assert_eq!(
            managed_fallback_count(),
            0,
            "instrumentation must start clean"
        );
        assert!(!managed_fallback_warned(), "no warning before any fallback");

        note_managed_fallback(0, 4096);
        assert_eq!(managed_fallback_count(), 1);
        assert!(
            managed_fallback_warned(),
            "first fallback must surface the warning"
        );

        // Second fallback: counter keeps counting, warning does not repeat.
        note_managed_fallback(0, 8192);
        assert_eq!(managed_fallback_count(), 2);
        assert!(
            managed_fallback_warned(),
            "warning flag stays set after first emit"
        );
    }

    /// WI-P3 negative case: allocation that does NOT hit the managed path must
    /// not touch the instrumentation (no false-positive warning on normal
    /// small-model loads).
    #[test]
    fn non_managed_allocation_does_not_touch_instrumentation() {
        reset_managed_fallback_instrumentation();
        // should_use_managed("", ...) = ordinary allocation path.
        assert!(!should_use_managed("", 1000, 2000, 1, 2000));
        assert_eq!(
            managed_fallback_count(),
            0,
            "no fallback recorded for a fit"
        );
        assert!(!managed_fallback_warned(), "no warning for a fit");
    }
}
