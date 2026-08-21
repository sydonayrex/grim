use grim_backend_vulkan::caps::VulkanCaps;
use grim_backend_vulkan::{VulkanKernel, binding_count, spirv_for};

#[test]
fn test_vulkan_caps_modern_probes() {
    let caps = VulkanCaps::probe_default("AMD Radeon RX 7900 XTX".into(), 0x1002, 0x7448, 2);
    assert!(caps.supports_subgroup_arithmetic);
    assert_eq!(caps.subgroup_size, 32);
    assert!(caps.supports_timeline_semaphores);
    assert!(caps.supports_external_memory_host);
    assert!(caps.supports_cooperative_matrix);
    assert!(caps.supports_fp32_atomic_add);
}

#[test]
fn test_vulkan_modern_kernels_spirv_embedded() {
    let kernels = [
        VulkanKernel::MlaDecode,
        VulkanKernel::SageAttention,
        VulkanKernel::FusedAdamw,
        VulkanKernel::FusedLion,
        VulkanKernel::Mrope,
        VulkanKernel::MarlinGemm,
        VulkanKernel::FusedLinearCe,
        VulkanKernel::FlashDecodeSplitK,
        VulkanKernel::SoftmaxMerge,
        VulkanKernel::QkvAttentionPagedDequant,
        VulkanKernel::SpeculativeAcceptor,
        VulkanKernel::RmsNorm,
        VulkanKernel::AddRmsNorm,
    ];

    for k in kernels {
        let spv = spirv_for(k);
        assert!(!spv.is_empty(), "SPIR-V blob for {:?} must not be empty", k);
        assert!(
            spv.len() % 4 == 0,
            "SPIR-V blob for {:?} must be 4-byte aligned",
            k
        );
        assert!(
            binding_count(k) >= 3,
            "Binding count for {:?} must be at least 3",
            k
        );
    }
}
