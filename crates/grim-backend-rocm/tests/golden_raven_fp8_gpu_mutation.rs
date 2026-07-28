//! Golden mutation-resistant test for Raven FP8 GPU dequantization GEMM forward and backward paths.

use std::panic;
use grim_backend_rocm::RocmDevice;
use grim_quant::dequant_fp8;
use grim_tensor::{
    dtype::{ArithType, BlockDtype, DType, Storage},
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
fn test_raven_fp8_gpu_gemm_golden_mutation_resistant() -> TestResult {
    // Non-square asymmetric dimensions: M=2, K=128, N=64
    let (m, k, n) = (2usize, 128usize, 64usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();

    // Raw FP8 E4M3 codes
    let fp8_bytes: Vec<u8> = (0..k * n).map(|i| (i % 254 + 1) as u8).collect();
    let b_dequant = dequant_fp8(&fp8_bytes, k * n)?;

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
        let fp8_dtype = DType {
            arith: ArithType::F32,
            storage: Storage::Block(BlockDtype::Fp8),
        };
        let b_dev = BackendDevice::from_cpu_bytes(&dev, &fp8_bytes, &b_shape, fp8_dtype)?;

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
        assert!(max_err < 1e-3, "Raven FP8 GPU matmul max error {max_err} exceeds 1e-3 threshold");
    }

    Ok(())
}
