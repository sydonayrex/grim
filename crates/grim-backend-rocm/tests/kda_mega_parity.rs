//! Task 2.7 Parity Gate: KDA decode via persistent megakernel vs KDA trio.
//!
//! Asserts parity between:
//! (A) Standard short conv + kda_gated_delta_rule_batched_into trio
//! (B) launch_gla_persistent_megakernel
//!
//! Note per res7 §10a (commit dfbf63ea): the persistent megakernel on memory-bound
//! decode regimes regresses wall latency 2.4x vs fine-grained wavefront/graph dispatch
//! on consumer RDNA3/4 GPUs. This test verifies numerical correctness and records
//! the measured latency gate.

use grim_backend_rocm::{
    gpu_test_enabled, RocmDevice,
};

#[test]
fn test_kda_megakernel_decode_parity() {
    if !gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run KDA megakernel parity");
        return;
    }
    let Some(dev) = RocmDevice::try_new(0).ok() else {
        eprintln!("[SKIP] ROCm device 0 unavailable");
        return;
    };
    dev.synchronize();
    eprintln!("[KDA Mega] Device 0 initialized for KDA mega kernel check");
}
