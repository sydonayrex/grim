use grim_backend_metal::caps::MetalCaps;

#[test]
fn test_metal_caps_modern_probes() {
    let caps = MetalCaps::probe_default(1001, "Apple M3 Max".into(), 9);
    assert!(caps.supports_fp16);
    assert!(caps.supports_bf16);
    assert!(caps.supports_fp8);
    assert!(caps.supports_simdgroup_matrix);
    assert!(caps.unified_memory);
}

#[test]
fn test_metal_msl_kernels_present() {
    let msl_source = include_str!("../src/kernels.msl");
    let required_kernels = [
        "grim_mla_decode",
        "grim_sage_attention",
        "grim_mrope",
        "grim_marlin_gemm",
        "grim_fused_linear_ce",
        "grim_fused_adamw",
        "grim_fused_lion",
        "grim_flash_decode_split_k",
        "grim_softmax_merge",
        "grim_qkv_attention_paged_dequant",
        "grim_speculative_acceptor",
        "grim_fused_dequant_gemm_q4k",
        "grim_fused_dequant_gemm_fp8",
    ];

    for kernel in required_kernels {
        assert!(
            msl_source.contains(kernel),
            "MSL shader source must contain kernel declaration for {}",
            kernel
        );
    }
}
