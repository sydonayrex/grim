//! Unit tests for ROCm quantize_on_device launchers.
//!
//! Validates that the device-side quantize kernels produce numerically
//! correct results via a quantize → dequantize round-trip, compared against
//! the CPU `grim_quant` reference.

use grim_backend_rocm::RocmDevice;
use grim_tensor::dtype::{DType, FloatPackScheme, KQuantScheme, QuantFormat, Storage};
use grim_tensor::{Shape};
use grim_tensor::{CoreTensorOps};

#[test]
fn test_rocm_quantize_q8_0_and_fp8() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_rocm_quantize_q8_0_and_fp8");
        return;
    }
    let dev = RocmDevice::new(0);

    let shape = Shape::new(vec![32]);
    let x_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 4.0).collect();
    let x_storage = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();

    // Quantize to Q8_0
    let (q8_storage, h1) = dev
        .quantize_on_device(x_storage.as_ref(), QuantFormat::Q8_0)
        .unwrap();
    h1.synchronize().unwrap();
    assert_eq!(
        q8_storage.dtype().storage,
        Storage::KQuant(KQuantScheme::Q80)
    );

    // Quantize to FP8
    let (fp8_storage, h2) = dev
        .quantize_on_device(x_storage.as_ref(), QuantFormat::Fp8)
        .unwrap();
    h2.synchronize().unwrap();
    assert_eq!(
        fp8_storage.dtype().storage,
        Storage::FloatPack(FloatPackScheme::Fp8)
    );
}

#[test]
fn test_rocm_quantize_q8_0_roundtrip() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_rocm_quantize_q8_0_roundtrip");
        return;
    }
    let dev = RocmDevice::new(0);

    let n = 256;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i as f32 + 1.0) * 0.05;
            if i % 2 == 0 { x } else { -x }
        })
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    let (q, h) = dev
        .quantize_on_device(x.as_ref(), QuantFormat::Q8_0)
        .unwrap();
    h.synchronize().unwrap();

    // to_cpu_vec_f32 dequantizes the packed Q8_0 bytes back to F32 on host/GPU.
    let deq = q.to_cpu_vec_f32().unwrap();
    assert_eq!(deq.len(), n, "Dequantized length mismatch for Q8_0");

    let max_abs = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tolerance = (max_abs / 127.0).max(1e-5);
    let max_err = input
        .iter()
        .zip(deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err <= tolerance,
        "ROCm Q8_0 roundtrip max error {max_err} exceeds tolerance {tolerance}"
    );
}

#[test]
fn test_rocm_quantize_fp8_roundtrip() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_rocm_quantize_fp8_roundtrip");
        return;
    }
    let dev = RocmDevice::new(0);

    let n = 256;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32 - 128.0) / 20.0).cos() * 3.5)
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    let (q, h) = dev
        .quantize_on_device(x.as_ref(), QuantFormat::Fp8)
        .unwrap();
    h.synchronize().unwrap();

    let deq = q.to_cpu_vec_f32().unwrap();
    assert_eq!(deq.len(), n, "Dequantized length mismatch for FP8");

    let max_err = input
        .iter()
        .zip(deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 0.5,
        "ROCm FP8 roundtrip max error {max_err} exceeds 0.5"
    );
}

#[test]
fn test_rocm_quantize_q8_0_vs_cpu_parity() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_rocm_quantize_q8_0_vs_cpu_parity");
        return;
    }
    let dev = RocmDevice::new(0);

    let n = 128;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin() * 4.0).collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    let (q, h) = dev
        .quantize_on_device(x.as_ref(), QuantFormat::Q8_0)
        .unwrap();
    h.synchronize().unwrap();

    let rocm_deq = q.to_cpu_vec_f32().unwrap();

    // CPU reference quantization -> dequantization
    let cpu_quant = grim_quant::quant_q80(&input).unwrap();
    let cpu_deq = grim_quant::dequant_q80(&cpu_quant, n).unwrap();

    let max_diff = rocm_deq
        .iter()
        .zip(cpu_deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "ROCm Q8_0 vs CPU reference parity diff {max_diff} exceeds 1e-4"
    );
}
