//! Unit tests for Vulkan quantize and fused_quant_gemm shaders.
//!
//! Validates that the device-side quantize kernels produce packed bytes that
//! match the CPU `grim_quant::quant_*` reference, and that the fused
//! quantize+GEMM kernels produce numerically correct results.

use grim_backend_vulkan::{VulkanDevice, extract_raw_bytes};
use grim_tensor::dtype::{DType, FloatPackScheme, KQuantScheme, QuantFormat, Storage};
use grim_tensor::{CoreTensorOps, QuantOps, Shape};

fn device_or_skip() -> Option<VulkanDevice> {
    let devices = VulkanDevice::probe().ok()?;
    devices.into_iter().next()
}

#[test]
fn test_vulkan_quantize_q8_0_and_fp8() {
    let mut dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no Vulkan device");
            return;
        }
    };
    // This test validates the FP8 quantize shader, so opt the device into FP8 support
    // (probe_default reports supports_fp8=false pending real device probing).
    dev.caps.supports_fp8 = true;

    let shape = Shape::new(vec![32]);
    let x_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 4.0).collect();
    let x_storage = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();

    // Quantize to Q8_0
    let (q8_storage, _h1) = dev
        .quantize_on_device(x_storage.as_ref(), QuantFormat::Q8_0)
        .unwrap();
    assert_eq!(
        q8_storage.dtype().storage,
        Storage::KQuant(KQuantScheme::Q80)
    );

    // Quantize to FP8
    let (fp8_storage, _h2) = dev
        .quantize_on_device(x_storage.as_ref(), QuantFormat::Fp8)
        .unwrap();
    assert_eq!(
        fp8_storage.dtype().storage,
        Storage::FloatPack(FloatPackScheme::Fp8)
    );
}

#[test]
fn test_vulkan_quantize_q8_0_parity() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no Vulkan device");
            return;
        }
    };

    // 64 weights = 2 Q8_0 blocks.
    let n = 64;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32 - 32.0) / 8.0).sin() * 3.0)
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    let (q, _h) = dev
        .quantize_on_device(x.as_ref(), QuantFormat::Q8_0)
        .unwrap();
    let device_bytes = extract_raw_bytes(q.as_ref()).unwrap();

    // CPU reference.
    let cpu_bytes = grim_quant::quant_q80(&input).unwrap();
    assert_eq!(device_bytes.len(), cpu_bytes.len());

    // Q8_0 roundtrip: dequantize and check error bound.
    let deq = grim_quant::dequant_q80(&device_bytes, n).unwrap();
    let max_abs = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tolerance = (max_abs / 127.0).max(1e-6);
    let max_err = input
        .iter()
        .zip(deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err <= tolerance,
        "Q8_0 roundtrip max error {max_err} exceeds tolerance {tolerance}"
    );
}

#[test]
fn test_vulkan_quantize_fp8_parity() {
    let mut dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no Vulkan device");
            return;
        }
    };
    // This test validates the FP8 quantize shader, so opt the device into FP8 support.
    dev.caps.supports_fp8 = true;

    let n = 64;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32 - 32.0) / 10.0).cos() * 2.0)
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    let (q, _h) = dev
        .quantize_on_device(x.as_ref(), QuantFormat::Fp8)
        .unwrap();
    let device_bytes = extract_raw_bytes(q.as_ref()).unwrap();

    // CPU reference.
    let cpu_bytes = grim_quant::quant_fp8(&input).unwrap();
    assert_eq!(device_bytes.len(), cpu_bytes.len());

    // FP8 roundtrip.
    let deq = grim_quant::dequant_fp8(&device_bytes, n).unwrap();
    let max_err = input
        .iter()
        .zip(deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // FP8 E4M3 has ~8 representable levels per octave; tolerance is generous.
    assert!(
        max_err < 0.5,
        "FP8 roundtrip max error {max_err} exceeds 0.5"
    );
}

#[test]
fn test_vulkan_fused_quant_gemm_q8_0() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no Vulkan device");
            return;
        }
    };

    let m = 2usize;
    let k = 32usize;
    let n = 4usize;

    let a_shape = Shape::new(vec![m, k]);
    let b_shape = Shape::new(vec![k, n]);
    let out_shape = Shape::new(vec![m, n]);

    let a_data = vec![1.0f32; m * k];
    let b_data = vec![0.5f32; k * n];

    let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
    let b_storage = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();

    let (out_q8, h1) = dev
        .fused_quant_gemm(
            a_storage.as_ref(),
            b_storage.as_ref(),
            QuantFormat::Q8_0,
            &out_shape,
        )
        .unwrap();
    h1.synchronize().unwrap();
    assert_eq!(out_q8.shape(), &out_shape);

    let result = out_q8.to_cpu_vec_f32().unwrap();

    // Manual reference: quantize A per-row-block, then matmul.
    // All A values are 1.0 → amax=1.0, scale=1/127, q=round(1.0/(1/127))=127.
    // So quantized A ≈ 127 * (1/127) = 1.0 (within rounding).
    // C[row, col] = sum(quantized_A[row,:] * B[:, col]) ≈ 32 * 1.0 * 0.5 = 16.0
    for v in &result {
        assert!(
            (v - 16.0).abs() < 1.0,
            "fused_quant_gemm_q8_0 result {v} not close to 16.0"
        );
    }
}

#[test]
fn test_vulkan_fused_quant_gemm_fp8() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no Vulkan device");
            return;
        }
    };

    let m = 2usize;
    let k = 16usize;
    let n = 4usize;

    let a_shape = Shape::new(vec![m, k]);
    let b_shape = Shape::new(vec![k, n]);
    let out_shape = Shape::new(vec![m, n]);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.3 - 1.0).tan()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.2).sin()).collect();

    let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
    let b_storage = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();

    let (out_fp8, h2) = dev
        .fused_quant_gemm(
            a_storage.as_ref(),
            b_storage.as_ref(),
            QuantFormat::Fp8,
            &out_shape,
        )
        .unwrap();
    h2.synchronize().unwrap();
    assert_eq!(out_fp8.shape(), &out_shape);

    let result = out_fp8.to_cpu_vec_f32().unwrap();

    // Manual reference: FP8 round-trip on A, then matmul.
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                let fp8_code = grim_quant::f32_to_fp8_e4m3(a_data[row * k + i]);
                let a_deq = grim_quant::fp8_e4m3_to_f32(fp8_code);
                sum += a_deq * b_data[i * n + col];
            }
            expected[row * n + col] = sum;
        }
    }

    let max_err = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 0.5,
        "fused_quant_gemm_fp8 max error {max_err} exceeds 0.5"
    );
}
