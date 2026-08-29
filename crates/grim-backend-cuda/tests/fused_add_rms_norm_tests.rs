//! Parity tests for the CUDA fused Add + RMSNorm kernel.
//!
//! Mirrors `grim-backend-rocm/tests/fused_add_rms_norm_tests.rs`: the fused kernel's
//! output is diffed against an independently-computed reference (add then rms_norm
//! separately), not a tautological length check.

use grim_backend_cuda::CudaDevice;
use grim_tensor::dtype::DType;
use grim_tensor::{CoreTensorOps, Shape};

/// Skip the test gracefully if no CUDA device is available.
fn device_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

#[test]
fn test_fused_add_rms_norm_parity() {
    let dev = match device_or_skip() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };

    let shape = Shape::new(vec![2, 4]); // 2 rows, row_len = 4
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let res_data = vec![0.5f32, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];
    let w_data = vec![1.0f32, 1.0, 1.0, 1.0];
    let eps = 1e-5f32;

    let x_s = dev.from_cpu(&x_data, &shape, DType::F32).unwrap();
    let res_s = dev.from_cpu(&res_data, &shape, DType::F32).unwrap();
    let w_s = dev
        .from_cpu(&w_data, &Shape::new(vec![4]), DType::F32)
        .unwrap();

    let (y_out, norm_out, handle) = dev
        .fused_add_rms_norm(x_s.as_ref(), res_s.as_ref(), w_s.as_ref(), eps, &shape)
        .unwrap();
    handle.synchronize().unwrap();

    let y_actual = y_out.to_cpu_vec_f32().unwrap();
    let norm_actual = norm_out.to_cpu_vec_f32().unwrap();

    // Expected y = x + res
    let y_expected: Vec<f32> = x_data
        .iter()
        .zip(res_data.iter())
        .map(|(a, b)| a + b)
        .collect();

    assert_eq!(y_actual.len(), y_expected.len());
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        assert!((a - e).abs() < 1e-5, "y mismatch: {} != {}", a, e);
    }

    // Reference standalone RMSNorm on y (add then rms_norm separately).
    let (add_s, h1) = dev.add(x_s.as_ref(), res_s.as_ref(), &shape).unwrap();
    h1.synchronize().unwrap();
    let (norm_expected_storage, h2) = dev
        .rms_norm(add_s.as_ref(), w_s.as_ref(), eps, &shape)
        .unwrap();
    h2.synchronize().unwrap();
    let norm_expected = norm_expected_storage.to_cpu_vec_f32().unwrap();

    assert_eq!(norm_actual.len(), norm_expected.len());
    for (a, e) in norm_actual.iter().zip(norm_expected.iter()) {
        assert!((a - e).abs() < 1e-4, "norm mismatch: {} != {}", a, e);
    }
}
