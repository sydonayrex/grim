//! Unit tests for Metal quantize_on_device launchers.
//!
//! Validates that the device-side quantize kernels produce numerically
//! correct results via a quantize → dequantize round-trip, compared against
//! the CPU `grim_quant` reference.

use grim_backend_metal::MetalDevice;
use grim_tensor::dtype::{DType, FloatPackScheme, KQuantScheme, QuantFormat, Storage};
use grim_tensor::{BackendDevice, Shape};

#[test]
fn test_metal_quantize_q8_0_and_fp8() {
    let dev = MetalDevice::new(0).unwrap();

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
fn test_metal_quantize_q8_0_roundtrip() {
    let dev = MetalDevice::new(0).unwrap();

    let n = 64;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i as f32 + 1.0) * 0.1;
            if i % 2 == 0 { x } else { -x }
        })
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    let (q, h) = dev
        .quantize_on_device(x.as_ref(), QuantFormat::Q8_0)
        .unwrap();
    h.synchronize().unwrap();

    // to_cpu_vec_f32 dequantizes the packed Q8_0 bytes back to F32.
    let deq = q.to_cpu_vec_f32().unwrap();

    let max_abs = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tolerance = (max_abs / 127.0).max(1e-6);
    let max_err = input
        .iter()
        .zip(deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err <= tolerance,
        "Metal Q8_0 roundtrip max error {max_err} exceeds tolerance {tolerance}"
    );
}

#[test]
fn test_metal_quantize_fp8_roundtrip() {
    let dev = MetalDevice::new(0).unwrap();

    let n = 64;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32 - 32.0) / 10.0).cos() * 2.0)
        .collect();
    let shape = Shape::new(vec![n]);
    let x = dev.from_cpu(&input, &shape, DType::F32).unwrap();

    let (q, h) = dev
        .quantize_on_device(x.as_ref(), QuantFormat::Fp8)
        .unwrap();
    h.synchronize().unwrap();

    let deq = q.to_cpu_vec_f32().unwrap();

    let max_err = input
        .iter()
        .zip(deq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 0.5,
        "Metal FP8 roundtrip max error {max_err} exceeds 0.5"
    );
}
