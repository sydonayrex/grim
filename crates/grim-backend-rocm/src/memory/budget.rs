//! Global ROCm residency policy for VRAM overflow.

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
    should_use_managed("auto", free, total, bytes as u64, budget)
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
    use super::should_use_managed;

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
}
