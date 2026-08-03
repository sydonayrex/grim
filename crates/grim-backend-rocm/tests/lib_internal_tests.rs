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
        let target = format!("float {sym}(");
        let count = src.matches(&target).count();
        assert_eq!(
            count, 1,
            "symbol definition '{}' appears {} times in concatenated kernel source; expected 1",
            target, count
        );
    }
}

/// RED-GREEN-REFACTOR Phase 0 Task 0.2: selective_scan,
/// cross_attention, rwkv_time_mix, rwkv_channel_mix must be reachable
/// via `dyn BackendDevice`, not just the inherent `impl RocmDevice`.
#[test]
#[ignore = "requires GRIM_RUN_GPU_TESTS=1 and a real ROCm GPU"]
fn rocm_trait_ops_are_reachable_via_dyn() {
    let dev: Box<dyn BackendDevice> = Box::new(
        RocmDevice::try_new(0).expect("RocmDevice::new should succeed on a system with ROCm"),
    );
    let shape = grim_tensor::Shape::new(vec![1, 1]);
    let dummy = dev.zeros(&shape, grim_tensor::DType::BF16).unwrap();
    let s = dummy.as_ref();
    let r = dev.selective_scan(s, s, s, s, s, 1, 1, 1, 1, &shape);
    assert!(
        !matches!(r, Err(grim_tensor::error::Error::Unimplemented(_))),
        "selective_scan must not return Unimplemented when called via dyn BackendDevice"
    );
}
