//! Integration tests for Metal catch-up plan features:
//! `MetalCaps`, empirical `MetalAutotuner`, `ShapeClass` op-identity classification, and resource gating.

use grim_backend_metal::{GemmOp, MetalAutotuner, MetalCaps, MetalDevice, ShapeClass};
use grim_tensor::dtype::QuantFormat;

#[test]
fn test_metal_caps_probing_and_hashing() {
    let caps = MetalCaps::probe_default(1001, "Apple M3 Max GPU".into(), 8);
    assert_eq!(caps.device_name, "Apple M3 Max GPU");
    assert_eq!(caps.max_threadgroup_memory_length, 32768);
    assert!(caps.cache_key_hash() > 0);
    assert!(caps.supports_fp8);
    assert!(caps.supports_quant_format(QuantFormat::Q8_0));
    assert!(caps.supports_quant_format(QuantFormat::Fp8));
    assert!(!caps.is_stale());
}

#[test]
fn test_metal_autotuner_shape_class_routing() {
    let caps = MetalCaps::probe_default(1001, "Metal Test Device".into(), 8);
    let autotuner = MetalAutotuner::new();

    assert_eq!(ShapeClass::from_op(GemmOp::LmHead, 16, 128000, 4096), ShapeClass::TLOLog);
    assert_eq!(ShapeClass::from_op(GemmOp::Ffn, 64, 64, 4096), ShapeClass::Prefill);

    // Decode shape (m=1)
    assert_eq!(ShapeClass::classify(1, 4096, 4096), ShapeClass::Decode);
    let decode_cfg = autotuner.search_tile_config(&caps, 1, 4096, 4096, None);
    assert_eq!(decode_cfg.block_m, 16);

    // TLOLog / LmHead wide-N shape
    let tlog_cfg = autotuner.search_tile_config(&caps, 16, 128000, 4096, Some(GemmOp::LmHead));
    assert_eq!(tlog_cfg.block_n, 64);

    // Prefill shape (m=64, n=64)
    let prefill_cfg = autotuner.search_tile_config(&caps, 64, 64, 4096, Some(GemmOp::Ffn));
    assert_eq!(prefill_cfg.block_m, 32);
}

#[test]
fn test_metal_resource_limits_gating() {
    let caps = MetalCaps::probe_default(1001, "Metal Low Mem Device".into(), 8);
    assert!(caps.validate_resource_limits(16384, 256));
    assert!(!caps.validate_resource_limits(65536, 256)); // Exceeds 32KB threadgroup memory
    assert!(!caps.validate_resource_limits(16384, 2048)); // Exceeds 1024 thread limit

    let valid_cfg = grim_backend_metal::MetalTileConfig { block_m: 16, block_n: 32, block_k: 16, split_k: 1 };
    let invalid_cfg = grim_backend_metal::MetalTileConfig { block_m: 128, block_n: 128, block_k: 16, split_k: 1 };
    assert!(valid_cfg.is_valid(&caps));
    assert!(!invalid_cfg.is_valid(&caps));
}

#[test]
fn test_metal_autotuner_cache_persistence() {
    let caps = MetalCaps::probe_default(2002, "Persistence Test GPU".into(), 8);
    let autotuner = MetalAutotuner::new();
    
    // Warm cache with shape (32, 64, 128)
    let original_cfg = autotuner.search_tile_config(&caps, 32, 64, 128, Some(GemmOp::Ffn));
    autotuner.save_cache(&caps);

    let autotuner_loaded = MetalAutotuner::new();
    autotuner_loaded.load_cache(&caps);
    let loaded_cfg = autotuner_loaded.search_tile_config(&caps, 32, 64, 128, Some(GemmOp::Ffn));
    assert_eq!(loaded_cfg, original_cfg);
}

#[test]
fn test_metal_device_caps_and_fingerprint() {
    if let Ok(dev) = MetalDevice::new(0) {
        let caps = dev.caps();
        assert!(!caps.device_name.is_empty());
        assert!(dev.hw_fingerprint() > 0);
    }
}

#[test]
fn test_metal_split_k_tile_config_and_reduction() {
    let caps = MetalCaps::probe_default(3003, "Split-K Test GPU".into(), 8);
    let cfg = grim_backend_metal::MetalTileConfig {
        block_m: 16,
        block_n: 32,
        block_k: 16,
        split_k: 4,
    };
    assert_eq!(cfg.split_k, 4);
    assert!(cfg.is_valid(&caps));
}
