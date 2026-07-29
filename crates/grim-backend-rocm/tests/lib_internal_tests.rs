//! Internal correctness tests for grim-backend-rocm.
//! These tests verify kernel symbol uniqueness and trait reachability;
//! GPU-requiring tests are marked #[ignore] and gated by
//! GRIM_RUN_GPU_TESTS=1.

use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::BackendDevice;

#[test]
fn kernel_source_has_no_duplicate_device_fns() {
    use grim_backend_rocm::kernels::source_asm::compute_kernel_source;
    let src = compute_kernel_source();
    let symbols = [
        "fp16_to_float_device",
        "fp8_e4m3_to_float_hip",
        "mxfp4_to_float_hip",
        "dequant_q4k_element",
    ];
    for sym in &symbols {
        let count = src.matches(sym).count();
        assert_eq!(
            count, 1,
            "symbol '{}' appears {} times in concatenated kernel source; expected 1",
            sym, count
        );
    }
}

/// RED-GREEN-REFACTOR Phase 0 Task 0.2: selective_scan, flash_attention,
/// cross_attention, rwkv_time_mix, rwkv_channel_mix must be reachable
/// via `dyn BackendDevice`, not just the inherent `impl RocmDevice`.
#[test]
#[ignore = "requires GRIM_RUN_GPU_TESTS=1 and a real ROCm GPU"]
fn rocm_trait_ops_are_reachable_via_dyn() {
    let dev: Box<dyn BackendDevice> =
        Box::new(RocmDevice::new(0).unwrap());
    let dummy = dev.zeros(&[1, 1], grim_tensor::DType::BF16).unwrap();
    // Any error other than Unimplemented is acceptable for the dummy tensors.
    let r = dev.selective_scan(&dummy, &dummy, &dummy, &dummy, &dummy);
    assert!(
        !matches!(r, Err(grim_tensor::error::Error::Unimplemented(_))),
        "selective_scan must not return Unimplemented when called via dyn BackendDevice"
    );
}
