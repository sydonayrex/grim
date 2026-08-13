//! Integration tests for CUDA catch-up plan features:
//! `CudaCaps`, empirical `CudaAutotuner`, `ShapeClass` op-identity classification, and resource gating.

use grim_backend_cuda::{CudaAutotuner, CudaCaps, CudaDevice, GemmOp, ShapeClass};
use grim_tensor::dtype::QuantFormat;

#[test]
fn test_cuda_caps_probing_and_hashing() {
    let caps = CudaCaps::probe_default(0, "NVIDIA GeForce RTX 4090".into(), 8, 9);
    assert_eq!(caps.device_name, "NVIDIA GeForce RTX 4090");
    assert_eq!(caps.shared_mem_per_block, 49152);
    assert!(caps.cache_key_hash() > 0);
    assert!(caps.supports_fp8_native());
    assert!(caps.supports_quant_format(QuantFormat::Q8_0));
    assert!(caps.supports_quant_format(QuantFormat::Fp8));
}

#[test]
fn test_cuda_autotuner_shape_class_routing() {
    let caps = CudaCaps::probe_default(0, "CUDA Test Device".into(), 8, 9);
    let autotuner = CudaAutotuner::new();

    // Decode shape (m=1)
    assert_eq!(ShapeClass::classify(1, 4096, 4096), ShapeClass::Decode);
    let decode_cfg = autotuner.search_tile_config(&caps, 1, 4096, 4096, None);
    assert_eq!(decode_cfg.block_m, 16);

    // TLOLog / LmHead wide-N shape
    let tlog_cfg = autotuner.search_tile_config(&caps, 16, 128000, 4096, Some(GemmOp::LmHead));
    assert_eq!(tlog_cfg.block_m, 16);
    assert_eq!(tlog_cfg.block_n, 64);

    // Prefill shape (m=64, n=64)
    let prefill_cfg = autotuner.search_tile_config(&caps, 64, 64, 4096, Some(GemmOp::Ffn));
    assert_eq!(prefill_cfg.block_m, 32);

}

#[test]
fn test_cuda_resource_limits_gating() {
    let caps = CudaCaps::probe_default(0, "CUDA Low Shared Mem Device".into(), 8, 9);
    assert!(caps.validate_resource_limits(16384, 256));
    assert!(!caps.validate_resource_limits(98304, 256)); // Exceeds 48KB shared memory limit
    assert!(!caps.validate_resource_limits(16384, 2048)); // Exceeds 1024 thread limit
}

#[test]
fn test_cuda_device_caps_and_fingerprint() {
    if let Ok(dev) = CudaDevice::new(0) {
        let caps = dev.caps();
        assert!(!caps.device_name.is_empty());
        assert!(dev.hw_fingerprint() > 0);
    }
}
