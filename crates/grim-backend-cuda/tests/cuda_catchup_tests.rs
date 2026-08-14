//! Integration tests for CUDA catch-up plan features:
//! `CudaCaps`, empirical `CudaAutotuner`, `ShapeClass` op-identity classification, and resource gating.

use grim_backend_cuda::{CudaAutotuner, CudaCaps, CudaTileConfig, GemmOp, ShapeClass};
use grim_tensor::dtype::QuantFormat;

#[test]
fn test_cuda_caps_probing_and_hashing() {
    let caps1 = CudaCaps::probe_default(0, "NVIDIA GeForce RTX 4090".into(), 8, 9);
    assert_eq!(caps1.device_name, "NVIDIA GeForce RTX 4090");
    assert_eq!(caps1.shared_mem_per_block, 49152);
    assert!(caps1.mem_pitch > 0);
    assert!(caps1.cache_key_hash() > 0);
    assert!(caps1.supports_fp8_native());
    assert!(caps1.supports_quant_format(QuantFormat::Q8_0));
    assert!(caps1.supports_quant_format(QuantFormat::Fp8));
    assert!(!caps1.is_stale());

    let caps2 = CudaCaps::probe_default(0, "NVIDIA GeForce RTX 4090".into(), 8, 9);
    assert_eq!(caps1.epoch, caps2.epoch);
    assert!(!caps2.is_stale());
}

#[test]
fn test_cuda_fp8_capability_gating() {
    let old_caps = CudaCaps::probe_default(0, "NVIDIA GeForce GTX 1080".into(), 6, 1);
    assert!(!old_caps.supports_fp8_native());
    assert!(!old_caps.supports_quant_format(QuantFormat::Fp8));

    let ada_caps = CudaCaps::probe_default(0, "NVIDIA GeForce RTX 4090".into(), 8, 9);
    assert!(ada_caps.supports_fp8_native());
    assert!(ada_caps.supports_quant_format(QuantFormat::Fp8));
}

#[test]
fn test_cuda_autotuner_shape_class_routing() {
    let caps = CudaCaps::probe_default(0, "CUDA Test Device".into(), 8, 9);
    let autotuner = CudaAutotuner::new();

    assert_eq!(ShapeClass::from_op(GemmOp::LmHead, 16, 128000, 4096), ShapeClass::TLOLog);
    assert_eq!(ShapeClass::from_op(GemmOp::Ffn, 64, 64, 4096), ShapeClass::Prefill);

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
    let caps = CudaCaps::probe_default(0, "NVIDIA Test Device".into(), 8, 9);
    assert!(!caps.device_name.is_empty());
    assert!(caps.cache_key_hash() > 0);
}

#[test]
fn test_cuda_gemm_tile_config_op_identity() {
    let caps = CudaCaps::probe_default(0, "NVIDIA Test Device".into(), 8, 9);
    let autotuner = CudaAutotuner::new();
    autotuner.load_cache(&caps);

    // LmHead → TLOLog tile (block_m=16, block_n=64, block_k=64)
    let lm_head_cfg = autotuner.search_tile_config(&caps, 16, 128000, 4096, Some(GemmOp::LmHead));
    assert_eq!(lm_head_cfg.block_m, 16);
    assert_eq!(lm_head_cfg.block_n, 64);
    assert_eq!(lm_head_cfg.block_k, 64);

    // Decode path (m=1, no op) → block_m=16
    let decode_cfg = autotuner.search_tile_config(&caps, 1, 4096, 4096, None);
    assert_eq!(decode_cfg.block_m, 16);

    // Prefill (m=64, n=64, op=Ffn)
    let prefill_cfg = autotuner.search_tile_config(&caps, 64, 64, 4096, Some(GemmOp::Ffn));
    assert!(prefill_cfg.block_m >= 32);
    assert!(prefill_cfg.is_valid(&caps));

    // Resource gating: a config that exceeds shared memory should be filtered
    let over_cfg = CudaTileConfig { block_m: 128, block_n: 128, block_k: 128, split_k: 1 };
    assert!(!over_cfg.is_valid(&caps));
}

/// T2/T6 persistence round-trip: `save_cache` writes a non-empty JSON file and `load_cache`
/// restores the measured winner so a repeat shape on the same caps hits the cache.
#[test]
fn test_autotuner_cache_persistence_round_trip() {
    let caps = CudaCaps::probe_default(0, "CUDA Persist Device".into(), 8, 9);
    let autotuner = CudaAutotuner::new();

    // Miss -> compute + persist a TLOLog winner for this shape.
    let winner = autotuner.search_tile_config(&caps, 16, 128000, 4096, Some(GemmOp::LmHead));
    assert_eq!(winner.block_n, 64);
    autotuner.save_cache(&caps);

    // On-disk file must exist and be non-empty real JSON (not the old "{}" placeholder).
    let hash = caps.cache_key_hash();
    let path = std::path::PathBuf::from(format!(".autotune_cache/cuda_{hash:016x}.json"));
    let data = std::fs::read_to_string(&path).expect("save_cache must write a JSON file");
    assert!(!data.trim().is_empty(), "cache file must not be empty");
    assert!(
        !data.trim().eq("{}"),
        "cache file must contain real entries, not a placeholder"
    );

    // A fresh autotuner loading from disk should hit the persisted winner (no re-search).
    let loaded = CudaAutotuner::new();
    loaded.load_cache(&caps);
    let cfg = loaded.search_tile_config(&caps, 16, 128000, 4096, Some(GemmOp::LmHead));
    assert_eq!(cfg, winner, "loaded cache must restore the persisted winner");

    // Cleanup the test artifact.
    let _ = std::fs::remove_file(&path);
}
