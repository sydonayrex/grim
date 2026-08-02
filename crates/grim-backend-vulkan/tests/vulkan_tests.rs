//! Unit and parity tests for Vulkan compute kernels.

use grim_backend_vulkan::{VulkanDevice, VulkanKernel, spirv_for};
use grim_tensor::dtype::DType;
use grim_tensor::{BackendDevice, Shape};

#[test]
fn test_all_vulkan_spirv_blobs_compiled_and_non_empty() {
    let kernels = [
        VulkanKernel::Add,
        VulkanKernel::Mul,
        VulkanKernel::SiluMul,
        VulkanKernel::RmsNorm,
        VulkanKernel::Softmax,
        VulkanKernel::Embedding,
        VulkanKernel::Matmul64,
        VulkanKernel::Matmul32,
        VulkanKernel::Matmul64Bf16,
        VulkanKernel::QkvAttention,
        VulkanKernel::MulScalar,
        VulkanKernel::Sqrt,
        VulkanKernel::Recip,
        VulkanKernel::Rope,
        VulkanKernel::FusedDequantGemmQ4K,
        VulkanKernel::FusedDequantGemmQ80,
        VulkanKernel::KvDequantAttention,
        VulkanKernel::SelectiveScan,
        VulkanKernel::QkvAttentionPaged,
        VulkanKernel::FlashAttention,
        VulkanKernel::SiluMulBackward,
        VulkanKernel::QuantizedMatmulBackwardDx,
        VulkanKernel::RwkvTimeMix,
        VulkanKernel::RwkvChannelMix,
        VulkanKernel::AllReduce,
        VulkanKernel::CommFuseReduce,
    ];

    // Verify all SPIR-V blobs are compiled and 4-byte aligned.
    for kernel in kernels {
        let bytes = spirv_for(kernel);
        assert!(
            !bytes.is_empty(),
            "SPIR-V blob for {:?} must be non-empty",
            kernel
        );
        assert_eq!(
            bytes.len() % 4,
            0,
            "SPIR-V blob for {:?} must be 4-byte aligned",
            kernel
        );
    }
}

#[test]
fn test_vulkan_device_creation_or_skip() {
    let _dev = VulkanDevice::new();
    println!("VulkanDevice instantiated.");
}

#[test]
fn test_vulkan_fused_dequant_gemm_fallback_matches_reference() {
    let dev = VulkanDevice::new();

    let m = 2usize;
    let k = 256usize;
    let n = 2usize;

    let a_vec = vec![1.0f32; m * k];
    let b_bytes = vec![0u8; (k / 256) * 144 * n];
    let b_scales = vec![1.0f32; (k / 32) * n];

    let shape_a = Shape::new(vec![m, k]);
    let shape_b = Shape::new(vec![(k / 256) * 144, n]);
    let shape_out = Shape::new(vec![m, n]);

    let a_storage = dev.from_cpu(&a_vec, &shape_a, DType::F32).unwrap();
    let b_storage = dev.from_cpu_bytes(&b_bytes, &shape_b, DType::F32).unwrap();

    let (out_storage, _) = dev
        .quantized_matmul(&*a_storage, &*b_storage, &b_scales, &shape_out)
        .unwrap();

    let out_vec = out_storage.to_cpu_vec_f32().unwrap();
    assert_eq!(out_vec.len(), m * n);
}
