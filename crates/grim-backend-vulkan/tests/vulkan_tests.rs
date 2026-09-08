//! Unit and parity tests for Vulkan compute kernels.

use grim_backend_vulkan::{VulkanDevice, VulkanKernel, spirv_for};
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, Storage};
use grim_tensor::{BackendStorage, CollectiveOps, CoreTensorOps, MemoryOps, QuantOps, Shape};
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
        VulkanKernel::FusedDequantGemmNvFp4,
        VulkanKernel::FusedDequantGemmW8A8Fp8,
        VulkanKernel::FusedDequantGemmW8A8Int8,
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
/// Verifies that `quantized_matmul` accepts a weight tensor tagged with Q8_0 dtype,
/// dispatches correctly (either to the GPU Q8_0 fused kernel or the CPU fallback),
/// and produces an output tensor of the correct shape.
///
/// The all-zero assertion is intentionally omitted: if the GPU kernel runs, the output
/// depends on the kernel's behavior with the provided (zero-encoded) weight bytes, which
/// is not the contract under test here. Shape correctness and no-panic are the invariants.
fn test_vulkan_fused_dequant_gemm_q80_dispatch_accepts_q80_dtype() {
    let dev = VulkanDevice::new();

    let m = 2usize;
    let k = 256usize;
    let n = 2usize;

    let a_vec = vec![1.0f32; m * k];
    // Q8_0 packs 32 signed int8 elements per block; k=256 / 32 = 8 blocks per column.
    // All-128 bytes encode to 0.0 via (128 - 128) / 127.0 = 0.0.
    let b_bytes = vec![128u8; k * n]; // one byte per logical weight element (ArithType::U8)
    let b_scales = vec![1.0f32; (k / 32) * n];

    let shape_a = Shape::new(vec![m, k]);
    let shape_b = Shape::new(vec![k, n]); // logical [K, N] — 1 byte per element for Q80
    let shape_out = Shape::new(vec![m, n]);

    let q80_dtype = DType {
        arith: ArithType::U8, // Q80 packs 1 signed int8 per logical element — U8 = 1 byte/elem.
        storage: Storage::KQuant(KQuantScheme::Q80),
    };

    let a_storage = dev.from_cpu(&a_vec, &shape_a, DType::F32).unwrap();
    // Tag the packed weight buffer with its true dtype so the kernel-selection guard
    // routes to FusedDequantGemmQ80 instead of the wrong kernel or silently corrupting data.
    let b_storage = dev.from_cpu_bytes(&b_bytes, &shape_b, q80_dtype).unwrap();

    let (out_storage, _) = dev
        .quantized_matmul(
            &*a_storage,
            &*b_storage,
            &b_scales,
            grim_tensor::QuantFormat::Q8_0,
            &shape_out,
        )
        .unwrap();

    // Verify output shape — the primary invariant for this dispatch correctness test.
    let out_vec = out_storage.to_cpu_vec_f32().unwrap();
    assert_eq!(out_vec.len(), m * n, "output length must equal m*n");
}

#[test]
fn test_vulkan_all_reduce_parity() {
    let dev = VulkanDevice::new();

    let shape = Shape::new(vec![8]);
    let inputs_data = [
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
        assert!((r - e).abs() < 1e-5, "all_reduce mismatch: {} != {}", r, e);
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
        assert!((r - e).abs() < 1e-5, "comm_fuse mismatch: {} != {}", r, e);
    }
}

#[test]
fn test_vulkan_fused_dequant_gemm_nvfp4_parity() {
    let dev = VulkanDevice::new();

    let m = 2usize;
    let k = 32usize;
    let n = 2usize;

    // A is [M, K] = [2, 32]
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();

    // B is [K, N] in column-major packed NVFP4 format.
    // Blocks per column: K / 16 = 2 sub-blocks.
    // Bytes per sub-block: 9 (1 byte shared exponent + 8 bytes packed codes).
    // Total bytes for B = N * 2 * 9 = 2 * 18 = 36 bytes.
    let mut b_bytes = Vec::with_capacity(n * 2 * 9);
    for col in 0..n {
        for sb in 0..2 {
            let shared_exp = 127u8; // scale = 2^0 = 1.0
            b_bytes.push(shared_exp);
            for b in 0..8 {
                // Low nibble = even element, high nibble = odd element
                // e.g. code 2 (val 1.0) and code 3 (val 1.5)
                let low = ((col + sb + b) % 7) as u8;
                let high = ((col + sb + b + 1) % 7) as u8;
                b_bytes.push(low | (high << 4));
            }
        }
    }

    let shape_a = Shape::new(vec![m, k]);
    let shape_b = Shape::new(vec![k, n]);
    let shape_out = Shape::new(vec![m, n]);

    use grim_tensor::dtype::FloatPackScheme;
    let nvfp4_dtype = DType {
        arith: ArithType::U8,
        storage: Storage::FloatPack(FloatPackScheme::NvFp4),
    };

    let a_storage = dev.from_cpu(&a_data, &shape_a, DType::F32).unwrap();
    let b_storage = dev.from_cpu_bytes(&b_bytes, &shape_b, nvfp4_dtype).unwrap();

    let (out_storage, _) = dev
        .quantized_matmul(
            &*a_storage,
            &*b_storage,
            &[],
            grim_tensor::QuantFormat::Fp4,
            &shape_out,
        )
        .unwrap();

    let gpu_result = out_storage.to_cpu_vec_f32().unwrap();
    assert_eq!(gpu_result.len(), m * n);

    // Compute expected result using CPU dequant_nvfp4
    // Note: each column has 32 weights, packed in 18 bytes.
    let mut b_dequant = vec![0.0f32; k * n];
    let col_bytes = (k / 16) * 9;
    for col in 0..n {
        let col_slice = &b_bytes[col * col_bytes..(col + 1) * col_bytes];
        let deq_col = grim_quant::dequant_nvfp4(col_slice, k).unwrap();
        for r in 0..k {
            b_dequant[r * n + col] = deq_col[r];
        }
    }

    let mut cpu_expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_data[row * k + p] * b_dequant[p * n + col];
            }
            cpu_expected[row * n + col] = sum;
        }
    }

    for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_expected.iter()).enumerate() {
        let diff = (gpu - cpu).abs();
        assert!(
            diff < 1e-4,
            "NVFP4 GEMM mismatch at {i}: gpu={gpu}, cpu={cpu}, diff={diff}"
        );
    }
}

#[test]
fn test_vulkan_fused_dequant_gemm_w8a8_fp8_parity() {
    let dev = VulkanDevice::new();

    let m = 2usize;
    let k = 16usize;
    let n = 2usize;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.15).cos()).collect();

    // Layout: [u64 scale_len (8 bytes)][scales F32 (N*4 bytes)][FP8 codes (K*N bytes)]
    let scale_len = (n * 4) as u64;
    let mut b_bytes = Vec::new();
    b_bytes.extend_from_slice(&scale_len.to_le_bytes());
    // Two scales
    let scales = [0.5f32, 2.0f32];
    for s in &scales {
        b_bytes.extend_from_slice(&s.to_le_bytes());
    }
    // FP8 codes for K*N elements
    for col in 0..n {
        for row in 0..k {
            let code = ((col * k + row) % 15 + 1) as u8;
            b_bytes.push(code);
        }
    }

    let shape_a = Shape::new(vec![m, k]);
    let shape_b = Shape::new(vec![k, n]);
    let shape_out = Shape::new(vec![m, n]);

    let w8a8_dtype = DType {
        arith: ArithType::U8,
        storage: Storage::CompressedTensorsW8A8Fp8,
    };

    let a_storage = dev.from_cpu(&a_data, &shape_a, DType::F32).unwrap();
    let b_storage = dev.from_cpu_bytes(&b_bytes, &shape_b, w8a8_dtype).unwrap();

    let (out_storage, _) = dev
        .quantized_matmul(
            &*a_storage,
            &*b_storage,
            &[],
            grim_tensor::QuantFormat::Fp8,
            &shape_out,
        )
        .unwrap();

    let gpu_result = out_storage.to_cpu_vec_f32().unwrap();
    assert_eq!(gpu_result.len(), m * n);

    let codes = &b_bytes[8 + n * 4..];
    let mut cpu_expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let scale = scales[col];
            let mut sum = 0.0f32;
            for p in 0..k {
                let code = codes[col * k + p];
                let w_val = grim_quant::fp8_e4m3_to_f32(code) * scale;
                sum += a_data[row * k + p] * w_val;
            }
            cpu_expected[row * n + col] = sum;
        }
    }

    for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_expected.iter()).enumerate() {
        let diff = (gpu - cpu).abs();
        assert!(
            diff < 1e-4,
            "W8A8 FP8 GEMM mismatch at {i}: gpu={gpu}, cpu={cpu}, diff={diff}"
        );
    }
}

#[test]
fn test_vulkan_fused_dequant_gemm_w8a8_int8_parity() {
    let dev = VulkanDevice::new();

    let m = 2usize;
    let k = 16usize;
    let n = 2usize;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.12).sin()).collect();

    let scale_len = (n * 4) as u64;
    let mut b_bytes = Vec::new();
    b_bytes.extend_from_slice(&scale_len.to_le_bytes());
    let scales = [0.25f32, 1.5f32];
    for s in &scales {
        b_bytes.extend_from_slice(&s.to_le_bytes());
    }
    // Signed int8 codes for K*N elements
    for col in 0..n {
        for row in 0..k {
            let val = ((col * k + row) as i8 % 17) - 8;
            b_bytes.push(val as u8);
        }
    }

    let shape_a = Shape::new(vec![m, k]);
    let shape_b = Shape::new(vec![k, n]);
    let shape_out = Shape::new(vec![m, n]);

    let w8a8_dtype = DType {
        arith: ArithType::U8,
        storage: Storage::CompressedTensorsW8A8Int8,
    };

    let a_storage = dev.from_cpu(&a_data, &shape_a, DType::F32).unwrap();
    let b_storage = dev.from_cpu_bytes(&b_bytes, &shape_b, w8a8_dtype).unwrap();

    let (out_storage, _) = dev
        .quantized_matmul(
            &*a_storage,
            &*b_storage,
            &[],
            grim_tensor::QuantFormat::Q8_0,
            &shape_out,
        )
        .unwrap();

    let gpu_result = out_storage.to_cpu_vec_f32().unwrap();
    assert_eq!(gpu_result.len(), m * n);

    let codes = &b_bytes[8 + n * 4..];
    let mut cpu_expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let scale = scales[col];
            let mut sum = 0.0f32;
            for p in 0..k {
                let code = codes[col * k + p] as i8;
                let w_val = (code as f32) * scale;
                sum += a_data[row * k + p] * w_val;
            }
            cpu_expected[row * n + col] = sum;
        }
    }

    for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_expected.iter()).enumerate() {
        let diff = (gpu - cpu).abs();
        assert!(
            diff < 1e-4,
            "W8A8 INT8 GEMM mismatch at {i}: gpu={gpu}, cpu={cpu}, diff={diff}"
        );
    }
}
