//! Golden mutation-resistant test for Jay MXFP4 and Magpie MXFP8 GPU dequantization GEMM.

use std::panic;
use grim_backend_rocm::RocmDevice;
use grim_quant::{f32_to_mxfp4_e2m1, mxfp4_e2m1_to_f32};
use grim_tensor::{
    dtype::{ArithType, DType, FloatPackScheme, Storage},
    BackendDevice, Shape,
};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

fn gpu_device() -> Option<RocmDevice> {
    if std::env::var(GPU_TEST_ENV).is_err() {
        return None;
    }
    match panic::catch_unwind(|| RocmDevice::new(0)) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

#[test]
fn test_jay_mxfp4_gpu_gemm_golden_mutation_resistant() -> TestResult {
    // Non-square asymmetric dimensions: M=2, K=128, N=64
    let (m, k, n) = (2usize, 128usize, 64usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.04).sin()).collect();

    let shared_exp = 127u8;
    let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos() * 2.0).collect();
    let b_codes: Vec<u8> = b_orig.iter().map(|&v| f32_to_mxfp4_e2m1(v, shared_exp)).collect();
    let b_dequant: Vec<f32> = b_codes.iter().map(|&c| mxfp4_e2m1_to_f32(c, shared_exp)).collect();

    // Ground-truth CPU reference: C = A @ B_dequant^T
    let mut expected_c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * b_dequant[ni * k + ki];
            }
            expected_c[mi * n + ni] = sum;
        }
    }

    if let Some(dev) = gpu_device() {
        let a_shape = Shape::from_slice(&[m, k]);
        let b_shape = Shape::from_slice(&[n, k]);
        let out_shape = Shape::from_slice(&[m, n]);

        let a_dev = BackendDevice::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
        let mxfp4_dtype = DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        };
        let b_dev = BackendDevice::from_cpu(&dev, &b_dequant, &b_shape, mxfp4_dtype)?;

        let (out, handle) = dev.quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], &out_shape)?;
        handle.synchronize()?;
        let actual_c = out.to_cpu_vec_f32()?;

        assert_eq!(actual_c.len(), expected_c.len());
        let mut max_err: f32 = 0.0;
        for (act, exp) in actual_c.iter().zip(expected_c.iter()) {
            let err = (act - exp).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 1e-3, "Jay MXFP4 GPU matmul max error {max_err} exceeds 1e-3 threshold");
    }

    Ok(())
}
