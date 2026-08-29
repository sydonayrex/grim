use grim_backend_metal::kernels::{
    ATTENTION_MSL, GEMM_MSL, MATH_MSL, OPTIMIZER_MSL, QUANTIZATION_MSL, SPECULATIVE_MSL,
    load_unified_msl,
};

#[test]
fn test_metal_modular_kernel_sources_non_empty() {
    assert!(!MATH_MSL.is_empty(), "MATH_MSL must not be empty");
    assert!(!GEMM_MSL.is_empty(), "GEMM_MSL must not be empty");
    assert!(!ATTENTION_MSL.is_empty(), "ATTENTION_MSL must not be empty");
    assert!(!QUANTIZATION_MSL.is_empty(), "QUANTIZATION_MSL must not be empty");
    assert!(!OPTIMIZER_MSL.is_empty(), "OPTIMIZER_MSL must not be empty");
    assert!(!SPECULATIVE_MSL.is_empty(), "SPECULATIVE_MSL must not be empty");

    assert!(MATH_MSL.contains("grim_add"));
    assert!(MATH_MSL.contains("grim_rms_norm"));
    assert!(GEMM_MSL.contains("grim_matmul"));
    assert!(ATTENTION_MSL.contains("grim_qkv_attention"));
    assert!(QUANTIZATION_MSL.contains("grim_dequant_fp8"));
    assert!(OPTIMIZER_MSL.contains("grim_fused_adamw"));
    assert!(SPECULATIVE_MSL.contains("grim_speculative_acceptor"));
}

#[test]
fn test_metal_unified_bundle_integrity() {
    let unified = load_unified_msl();
    assert!(unified.contains("grim_matmul"));
    assert!(unified.contains("grim_qkv_attention"));
    assert!(unified.contains("grim_sage_attention"));
    assert!(unified.contains("grim_mla_decode"));
    assert!(unified.contains("grim_speculative_acceptor"));
}
