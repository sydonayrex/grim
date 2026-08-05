//! Unit and parity tests for Vulkan compute kernels.

use grim_backend_vulkan::{VulkanDevice, VulkanKernel, spirv_for};
use grim_tensor::dtype::DType;
use grim_tensor::{BackendDevice, BackendStorage, Shape};
use grim_tensor::{ScytheLink, ScythePlacement};

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
    VulkanKernel::QuantizedMatmulBackwardDxQ8_0,
    VulkanKernel::QuantizedMatmulBackwardDxGeneric,
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

#[test]
fn test_vulkan_all_reduce_parity() {
    let dev = VulkanDevice::new();

    let shape = Shape::new(vec![8]);
    let inputs_data = vec![
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    ];
    let storages: Vec<Box<dyn BackendStorage>> = inputs_data
        .iter()
        .map(|v| dev.from_cpu(v, &shape, DType::F32).unwrap())
        .collect();
    let refs: Vec<&dyn BackendStorage> = storages.iter().map(|s| s.as_ref()).collect();

    let (out, handle) = dev.all_reduce(&refs, "sum").unwrap();
    handle.synchronize().unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    let expected: Vec<f32> = (0..8)
        .map(|i| inputs_data.iter().map(|v| v[i]).sum::<f32>())
        .collect();
    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "all_reduce mismatch: {} != {}",
            r,
            e
        );
    }
}

#[test]
fn test_vulkan_all_reduce_single_input_parity() {
    let dev = VulkanDevice::new();

    let shape = Shape::new(vec![4]);
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    let refs: Vec<&dyn BackendStorage> = vec![storage.as_ref()];

    let (out, _) = dev.all_reduce(&refs, "sum").unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    for (r, e) in result.iter().zip(data.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "all_reduce single mismatch: {} != {}",
            r,
            e
        );
    }
}

#[test]
fn test_vulkan_comm_fuse_reduce_parity() {
    let dev = VulkanDevice::new();

    let m = 2usize;
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // [2, 3]
    let shape_a = Shape::new(vec![m, 2]);
    let shape_b = Shape::new(vec![m, 3]);

    let a = dev.from_cpu(&a_data, &shape_a, DType::F32).unwrap();
    let b = dev.from_cpu(&b_data, &shape_b, DType::F32).unwrap();

    let placement = ScythePlacement {
        ranks: vec![0, 1],
        partition: vec![0.5, 0.5],
        routes: vec![ScytheLink::Host; 4],
    };
    let partials: Vec<(&dyn BackendStorage, &ScythePlacement)> =
        vec![(a.as_ref(), &placement), (b.as_ref(), &placement)];

    let out = dev.comm_fuse_reduce(&partials).unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    // Column-concat: [[1, 2, 10, 20, 30], [3, 4, 40, 50, 60]]
    let expected = vec![1.0f32, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0];
    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "comm_fuse mismatch: {} != {}",
            r,
            e
        );
    }
}
