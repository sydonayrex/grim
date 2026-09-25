//! Runtime capability evidence for the optional hipBLASLt/rocBLASLt path.

use grim_backend_rocm::{
    gpu_test_enabled, probe_blaslt, select_blaslt_candidate, BlasLtSelection, RocmDevice,
};

#[test]
#[ignore]
fn gpu_probe_reports_blaslt_runtime_without_routing_decode() {
    if !gpu_test_enabled() {
        eprintln!("skipping: set GRIM_GPU_TEST=1");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skipping: no ROCm ordinal 0");
        return;
    }

    let probe = probe_blaslt();
    eprintln!("[blaslt-audit] {}", probe.describe());
    if !probe.available() {
        eprintln!("skipping BLASLt selection assertion: runtime unavailable");
        return;
    }
    assert!(
        probe.version.unwrap_or_default() > 0,
        "available BLASLt must report a positive version"
    );
    assert_eq!(
        select_blaslt_candidate(&probe, 1, 1024, 1024, true),
        BlasLtSelection::NotMeasured,
        "decode must not route to BLASLt"
    );
    assert_eq!(
        select_blaslt_candidate(&probe, 32, 4096, 4096, false),
        BlasLtSelection::NotMeasured,
        "unmeasured prefill must not route to BLASLt"
    );
}
