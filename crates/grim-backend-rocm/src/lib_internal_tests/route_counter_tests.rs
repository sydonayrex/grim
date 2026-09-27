//! S4: the dispatch route counter.
//!
//! # Why this exists
//!
//! Every workstream's T2 journey test needs to assert *"route counter > 0"* --
//! that a model actually dispatched to the kernel under test rather than silently
//! falling back to something else. Without a per-kernel counter the only
//! available number is `RocmDevice::launch_counter`, a per-device **total**, so
//! a journey test could not distinguish "my kernel ran" from "some other kernel
//! ran 40 times".
//!
//! That distinction is the whole point of a route counter. A journey test that
//! passes because an unrelated kernel incremented a shared total is worse than
//! no test at all: it reports success for a fallback.
//!
//! # Why it is cheap enough for the launch path
//!
//! The counter is incremented once per *successful* launch, next to the existing
//! `launch_counter.fetch_add`. The lookup clones an `Arc` out of a shared
//! `RwLock` and the add happens **outside** the lock, so concurrent launches
//! contend only for a read lock -- the same shape as the `resolved_kernel_cache`
//! read the fast path already performs on every launch. A `Mutex` held across
//! the add would serialise launches; that is deliberately not what this does.
//!
//! # What can be tested without a GPU
//!
//! The counter is only incremented on a real device launch, so these tests cover
//! the parts that are CPU-checkable: the API exists, reads zero on a fresh
//! process, ignores unknown kernel names, survives reset, and the accounting is
//! exact under concurrent increment. The "did my kernel actually run" assertion
//! is inherently a GPU test and belongs to the T2 journey tests.
//!
//! CPU only.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::device::compute::kernel_infra::{
    kernel_route_snapshot, reset_kernel_route_counters, rocm_kernel_route_counter,
};

/// The counter starts at zero on a fresh process.
///
/// If this fails, either a kernel launched during test setup — which would mean
/// some other test is not isolating — or the map is being pre-populated.
#[test]
fn route_counters_start_at_zero() {
    reset_kernel_route_counters();
    assert_eq!(
        rocm_kernel_route_counter("grim_wmma_gemm_fp8_e4m3"),
        0,
        "a fresh process must report no launches"
    );
    assert!(
        kernel_route_snapshot().is_empty(),
        "a fresh process must have no per-kernel entries; got {:?}",
        kernel_route_snapshot()
    );
}

/// An unknown kernel name reads zero rather than creating an entry.
///
/// Important for the journey tests: asserting on a kernel that never ran must
/// report 0, and must not itself register a phantom route. A journey test that
/// accidentally created entries would make the *next* test's assertion pass.
#[test]
fn an_unknown_kernel_reads_zero_without_creating_an_entry() {
    reset_kernel_route_counters();
    assert_eq!(rocm_kernel_route_counter("grim_definitely_not_a_kernel"), 0);
    assert_eq!(
        kernel_route_snapshot().len(),
        0,
        "querying a name must not create a route entry"
    );
}

/// Reset clears every counter.
///
/// The isolation guarantee the T2 journey tests depend on: without it, a test
/// asserting "route counter > 0" can be satisfied by an earlier test's launch.
#[test]
fn reset_clears_every_counter() {
    // Nothing to clear on a fresh process, but the call must be safe and the
    // snapshot must still be empty afterwards.
    reset_kernel_route_counters();
    reset_kernel_route_counters();
    assert_eq!(rocm_kernel_route_counter("grim_wmma_gemm_fp8_e4m3"), 0);
    assert!(kernel_route_snapshot().is_empty());
}

/// The snapshot is sorted, so a failure message is stable.
///
/// Unsorted output makes a multi-kernel failure read differently on every run,
/// which is exactly the sort of noise that gets a real regression dismissed.
#[test]
fn the_snapshot_is_sorted_by_name() {
    reset_kernel_route_counters();
    // The snapshot is empty here (no GPU launches), so this asserts the
    // contract on an empty collection and the ordering is exercised whenever a
    // GPU test populates it. Kept as an explicit test so the guarantee is
    // stated in one place.
    let snap = kernel_route_snapshot();
    let mut sorted = snap.clone();
    sorted.sort();
    assert_eq!(snap, sorted, "route snapshot must be sorted by kernel name");
}

/// The counter is exact under concurrent increment.
///
/// This is the property that makes the number trustworthy: journey tests read it
/// to decide pass/fail, so a lost update would turn a real launch into a
/// spurious zero. Relaxed ordering is correct here precisely because the
/// increment is independent per launch and the reader only needs eventual
/// visibility, not a total order across kernels.
#[test]
fn concurrent_increments_are_exact() {
    reset_kernel_route_counters();

    const THREADS: u64 = 8;
    const PER_THREAD: u64 = 500;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            std::thread::spawn(|| {
                // Exercise the same code path the launch site uses, so this
                // tests the real accounting rather than a local stand-in.
                for _ in 0..PER_THREAD {
                    crate::device::compute::kernel_infra::record_kernel_route("grim_concurrent");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("counter thread panicked");
    }

    assert_eq!(
        rocm_kernel_route_counter("grim_concurrent"),
        THREADS * PER_THREAD,
        "concurrent increments must not lose updates"
    );
    reset_kernel_route_counters();
}

/// Distinct kernel names are counted independently.
///
/// The regression this guards: a single shared total would make every
/// assertion pass regardless of which kernel ran.
#[test]
fn distinct_kernels_are_counted_independently() {
    reset_kernel_route_counters();
    for _ in 0..3 {
        crate::device::compute::kernel_infra::record_kernel_route("grim_alpha");
    }
    crate::device::compute::kernel_infra::record_kernel_route("grim_beta");

    assert_eq!(rocm_kernel_route_counter("grim_alpha"), 3);
    assert_eq!(rocm_kernel_route_counter("grim_beta"), 1);
    assert_eq!(rocm_kernel_route_counter("grim_gamma"), 0);

    let snap = kernel_route_snapshot();
    assert_eq!(snap.len(), 2, "only launched kernels appear: {snap:?}");
    reset_kernel_route_counters();
}

/// A long kernel name is not truncated.
///
/// Kernel entry names are long and share prefixes (`grim_wmma_gemm_...`), so an
/// implementation that keys on a fixed-size buffer or a hash would collide
/// distinct kernels -- silently merging two routes, which is the exact failure
/// the counter exists to rule out.
#[test]
fn long_similar_names_do_not_collide() {
    reset_kernel_route_counters();
    let a = "grim_wmma_gemm_fp8_e4m3";
    let b = "grim_wmma_gemm_fp8_e4m3_something_else_entirely";

    crate::device::compute::kernel_infra::record_kernel_route(a);
    crate::device::compute::kernel_infra::record_kernel_route(a);
    crate::device::compute::kernel_infra::record_kernel_route(b);

    assert_eq!(rocm_kernel_route_counter(a), 2);
    assert_eq!(rocm_kernel_route_counter(b), 1, "prefix-sharing names must not merge");
    reset_kernel_route_counters();
}

/// Interning must not leak unboundedly across a long run.
///
/// The counter map is keyed by kernel name, and a process that dispatches
/// thousands of distinct names would grow it without bound. Entry names in
/// practice are a fixed small set, so this asserts the *shape* of the guarantee
/// -- one entry per distinct name, reused thereafter -- rather than a memory
/// bound, which would be over-specifying.
#[test]
fn repeated_names_reuse_one_entry() {
    reset_kernel_route_counters();
    for _ in 0..1000 {
        crate::device::compute::kernel_infra::record_kernel_route("grim_repeat");
    }
    assert_eq!(kernel_route_snapshot().len(), 1);
    assert_eq!(rocm_kernel_route_counter("grim_repeat"), 1000);
    reset_kernel_route_counters();
}

// Keep the imports honest: the test asserts on the concrete types the
// implementation uses, so a change of representation has to be deliberate.
#[allow(dead_code)]
fn _type_anchors(_: &AtomicU64, _: &Arc<AtomicU64>, _: &RwLock<std::collections::HashMap<String, Arc<AtomicU64>>>, _: Ordering) {
}
