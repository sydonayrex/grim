//! Device-feature probes — small functions that ask HIP runtime a
//! single yes/no question about a given ordinal. Each probe is a
//! tiny `hipDeviceGetAttribute` call wrapped in `unsafe`; we keep
//! them next to each other so callers can spot at a glance which
//! capabilities are query-able.
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` Action 2 — capability-probe layer.
//! - `rust-gpu-discipline` §4 — every probe returns `Result`-shaped
//!   data (bool here, since "is X available" is binary) and never
//!   silently fabricates a value.
//! - `rocm-profiling-perf` — XNACK-feasibility check is what unblocks
//!   the unified-memory memcpy fast path in `memcpy_with_xnack_fallback`.

use super::handles::{
    hipDeviceGetAttribute, hipSetDevice, HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
};

/// XNACK probe for unified memory availability. Returns true if the
/// device supports concurrent page faulting (so `hipMemAdvise`
/// paths can be used safely).
pub fn probe_xnack(device_ordinal: usize) -> bool {
    let mut val: i32 = 0;
    unsafe {
        hipSetDevice(device_ordinal as i32);
        let status = hipDeviceGetAttribute(
            &mut val,
            HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
            device_ordinal as i32,
        );
        status == 0 && val == 1
    }
}

#[cfg(test)]
mod probe_self_tests {
    use super::*;

    #[test]
    fn probe_xnack_returns_bool() {
        // We don't have a GPU in unit tests, so the call must
        // simply return SOMETHING without panicking. The hip
        // runtime returns hipErrorNoDevice on a host without HIP,
        // which our boolean wraps into false — but the function
        // signature must remain `bool` regardless.
        let _: bool = probe_xnack(0);
    }
}
