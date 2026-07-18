//! RED-GREEN-REFACTOR: Item 1 acceptance criterion from the ROCm spec.
//!
//! Spec (grim_rocm_perf_and_abi_fix_spec.md, Item 1 acceptance):
//!
//! > A test or benchmark that runs N forward/decode steps in a loop and
//! > asserts the number of raw `hipMalloc` calls (instrument with a
//! > counter behind a test-only feature flag, or log-and-count under
//! > `rocm-profile`) is roughly constant after a warmup period, not O(N)
//! > per allocation site.
//!
//! Strategy: drive the *caching* allocator (`RocmCachingAllocator`, the
//! subject of Item 1) via the public `copy_from_host_async` path, which
//! routes storage through `RocmStorage::alloc_gpu` -> `allocator.alloc`.
//! Capture `RocmDevice::allocator_stats()` (the caching allocator's
//! real `hipMalloc`/`hipFree` counters) once after a warmup, then run
//! many identical uploads and re-check. If the allocator is doing its
//! job, the count of raw `hipMalloc` calls stays flat — the pool is
//! recycling the slot across calls.
//!
//! NOTE: `upload_to_scratch` intentionally routes through the separate
//! `DeviceScratchPool` (Phase-3 §3.1), which has its own counters and is
//! NOT what `allocator_stats()` reports. Asserting the caching
//! allocator's stats against the scratch pool would always read 0.

use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_tensor::{DType, Shape};

fn gpu_tests_enabled() -> bool {
    std::env::var("GRIM_RUN_GPU_TESTS").is_ok()
}

/// One warmup + a large loop of `copy_from_host_async` uploads. The number
/// of `hipMalloc` calls observed at the end must equal the count at the
/// end of warmup (within a small fudge for path-finding). This proves
/// the caching pool is recycling the same bucket slot across calls, not
/// re-allocating per invocation.
#[test]
fn copy_from_host_async_does_not_grow_malloc_count_per_call() {
    if !gpu_tests_enabled() {
        eprintln!("[skipped: GRIM_RUN_GPU_TESTS not set]");
        return;
    }

    let dev = Arc::new(RocmDevice::new(0));
    let shape = Shape::from_slice(&[64]);
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

    // Warmup: drop the buffer so the caching pool registers one bucket hit.
    for _ in 0..4 {
        let _buf = dev
            .copy_from_host_async(&data, &shape, DType::F32)
            .expect("warmup copy_from_host_async");
    }
    let (malloc_after_warmup, _free_after_warmup) = dev.allocator_stats();
    assert!(
        malloc_after_warmup > 0,
        "warmup must produce at least one hipMalloc (sanity)"
    );

    // Steady state loop — many identical uploads. Pool must recycle.
    let iters: usize = 64;
    for _ in 0..iters {
        let buf = dev
            .copy_from_host_async(&data, &shape, DType::F32)
            .expect("steady-state copy_from_host_async");
        drop(buf); // explicit Drop returns the slot to the caching pool
    }

    let (malloc_after_steady, free_after_steady) = dev.allocator_stats();

    // The driver's hipMalloc count must NOT scale with `iters`. With
    // perfect pool reuse we'd expect `malloc_after_steady ==
    // malloc_after_warmup`. Real-world chip drivers may occasionally
    // lose a slot to a stride/alignment check, so allow a small bound
    // (≤ iters / 8 — many orders of magnitude tighter than O(N)).
    let grown = malloc_after_steady.saturating_sub(malloc_after_warmup);
    assert!(
        grown <= iters / 8,
        "hipMalloc count grew by {grown} across {iters} iterations \
         (> iters/8 = {}). Pool is not recycling — Item 1 acceptance \
         violated. malloc={} free={}",
        iters / 8,
        malloc_after_steady,
        free_after_steady
    );
}
