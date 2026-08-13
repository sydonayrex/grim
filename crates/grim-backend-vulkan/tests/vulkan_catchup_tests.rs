//! Integration tests for Vulkan catch-up plan features:
//! `VulkanCaps`, empirical `VulkanAutotuner`, `ShapeClass` op-identity classification, and resource gating.

use grim_backend_vulkan::{GemmOp, ShapeClass, VulkanAutotuner, VulkanCaps, VulkanDevice};
use grim_tensor::dtype::QuantFormat;

#[test]
fn test_vulkan_caps_probing_and_hashing() {
    let caps = VulkanCaps::probe_default("AMD Radeon RX 7900 XTX".into(), 0x1002, 0x744c, 1);
    assert_eq!(caps.device_name, "AMD Radeon RX 7900 XTX");
    assert_eq!(caps.max_shared_memory_bytes, 32768);
    assert!(caps.cache_key_hash() > 0);
    assert!(caps.supports_quant_format(QuantFormat::Q8_0));
    assert!(caps.supports_quant_format(QuantFormat::Q4K));
}

#[test]
fn test_vulkan_autotuner_shape_class_routing() {
    let caps = VulkanCaps::probe_default("Vulkan Test Device".into(), 0x1002, 0x744c, 1);
    let autotuner = VulkanAutotuner::new();

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
fn test_vulkan_resource_limits_gating() {
    let caps = VulkanCaps::probe_default("Vulkan Low Shared Mem Device".into(), 0x1002, 0x744c, 1);
    assert!(caps.validate_resource_limits(16384, 256));
    assert!(!caps.validate_resource_limits(65536, 256)); // Exceeds 32KB shared memory
    assert!(!caps.validate_resource_limits(16384, 2048)); // Exceeds 1024 workgroup invocations
}

#[test]
fn test_vulkan_device_caps_and_fingerprint() {
    let dev = VulkanDevice::new();
    let caps = dev.caps();
    assert!(!caps.device_name.is_empty());
    assert!(dev.hw_fingerprint() > 0);
}
