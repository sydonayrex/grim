//! Golden mutation-resistant test for Crow Q4_K GPU dequantization GEMM forward and backward paths.

use grim_backend_rocm::RocmDevice;
use grim_format::convert::GpuDequant;
use grim_quant::{dequant_q4k, quant_q4k};
use grim_tensor::{
    BackendDevice, Shape,
    dtype::{ArithType, DType, KQuantScheme, Storage},
};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

fn gpu_device() -> Option<RocmDevice> {
    if std::env::var(GPU_TEST_ENV).is_err() {
        return None;
    }
    match panic::catch_unwind(|| {
        RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm")
    }) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

#[test]
fn test_q4k_gpu_gemm_golden_mutation_resistant() -> TestResult {
    // Non-square asymmetric dimensions: M=2, K=256, N=128
    let (m, k, n) = (2usize, 256usize, 128usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.015).cos().abs() * 8.0)
        .collect();

    let b_packed = quant_q4k(&b_orig).expect("quant_q4k");
    let b_dequant = dequant_q4k(&b_packed, b_orig.len()).expect("dequant_q4k");

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
        let q4k_dtype = DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        };
        let b_dev = BackendDevice::from_cpu_bytes(&dev, &b_packed, &b_shape, q4k_dtype)?;

        let (out, handle) = dev.quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &out_shape,
        )?;
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
        assert!(
            max_err < 1e-3,
            "Q4_K GPU matmul max error {max_err} exceeds 1e-3 threshold"
        );

        // --- GpuDequant path (used by grim-format conversion) ---
        // Routes `KQuant::Q4K` through `RocmDevice::dequantize` →
        // `dequantize_q4k_host` → the `grim_dequant_q4k` HIP kernel instead of
        // the CPU `grim_quant::dequant_q4k` fallback. Must remain bit-exact with
        // the CPU oracle `b_dequant` (computed above).
        let gpu_dequant = dev
            .dequantize(&Storage::KQuant(KQuantScheme::Q4K), &b_packed, b_orig.len())
            .expect("GpuDequant::dequantize Q4K");
        assert!(
            gpu_dequant.is_some(),
            "GpuDequant must handle Q4K (should not fall back to CPU)"
        );
        let gpu_dequant = gpu_dequant.expect("checked above");
        assert_eq!(gpu_dequant.len(), b_dequant.len());
        let dequant_max_err = b_dequant
            .iter()
            .zip(gpu_dequant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            dequant_max_err, 0.0,
            "Q4_K GpuDequant kernel deviates from CPU oracle (max_err={dequant_max_err:e})"
        );
    }

    Ok(())
}
