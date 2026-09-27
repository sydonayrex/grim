//! S4: the route counter is reachable from an integration test.
//!
//! This is a test about *visibility*, not behaviour, and it exists because the
//! counter was originally `pub` all the way down
//! (`device::compute::kernel_infra`) but absent from the crate root. An
//! integration test cannot reach a `pub` item that is not re-exported on the
//! root, so every T2 journey test -- the entire reason S4 exists -- would have
//! failed to compile against it.
//!
//! The behaviour is covered by `src/lib_internal_tests/route_counter_tests.rs`;
//! what is checked here is only that the door is open.

use grim_backend_rocm::{
    BLASLT_ROUTE, kernel_route_snapshot, record_kernel_route, reset_kernel_route_counters,
    rocm_kernel_route_counter,
};

#[test]
fn the_route_counter_is_reachable_and_coherent_from_outside_the_crate() {
    reset_kernel_route_counters();
    assert_eq!(rocm_kernel_route_counter("grim_probe"), 0);

    record_kernel_route("grim_probe");
    record_kernel_route("grim_probe");
    assert_eq!(rocm_kernel_route_counter("grim_probe"), 2);

    let snap = kernel_route_snapshot();
    assert_eq!(snap.len(), 1, "only the recorded kernel appears: {snap:?}");

    reset_kernel_route_counters();
    assert_eq!(rocm_kernel_route_counter("grim_probe"), 0);
}

/// The synthetic BLASLt route must be visible too, since detecting a silent
/// fallback to cuBLASLt is half the point of counting routes.
#[test]
fn the_blaslt_route_name_is_exported_and_does_not_collide() {
    assert_eq!(BLASLT_ROUTE, "blaslt_external");
    reset_kernel_route_counters();
    assert_eq!(rocm_kernel_route_counter(BLASLT_ROUTE), 0);
    record_kernel_route(BLASLT_ROUTE);
    assert_eq!(rocm_kernel_route_counter(BLASLT_ROUTE), 1);
    assert_eq!(
        rocm_kernel_route_counter("blaslt"),
        0,
        "a prefix of the BLASLt route must not alias it"
    );
    reset_kernel_route_counters();
}
