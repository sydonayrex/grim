//! Golden mutation-resistant test for Jay MXFP4 and Magpie MXFP8 GPU dequantization GEMM.

use grim_backend_rocm::RocmDevice;
use grim_quant::{f32_to_mxfp4_e2m1, mxfp4_e2m1_to_f32};
use grim_tensor::{
    Shape,
    dtype::{ArithType, DType, FloatPackScheme, Storage},
};
use std::panic;
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

#[test]
fn test_jay_mxfp4_gpu_gemm_golden_mutation_resistant() -> TestResult {
    // Non-square asymmetric dimensions: M=2, K=128, N=64
    let (m, k, n) = (2usize, 128usize, 64usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.04).sin()).collect();

    let shared_exp = 127u8;
    let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos() * 2.0).collect();
    let b_codes: Vec<u8> = b_orig
        .iter()
        .map(|&v| f32_to_mxfp4_e2m1(v, shared_exp))
        .collect();
    let b_dequant: Vec<f32> = b_codes
        .iter()
        .map(|&c| mxfp4_e2m1_to_f32(c, shared_exp))
        .collect();

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

        let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
        let mxfp4_dtype = DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        };

        // The MXFP4 kernel expects `B_codes` packed 2-per-byte (low nibble =
        // even element, high nibble = odd element) in flat `col*K+k` order.
        let mut b_packed = vec![0u8; (n * k) / 2];
        for j in 0..(n * k) {
            let code = b_codes[j] & 0x0F;
            if j % 2 == 0 {
                b_packed[j / 2] |= code;
            } else {
                b_packed[j / 2] |= code << 4;
            }
        }
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &b_shape, mxfp4_dtype)?;

        // One E8M0 exponent (shared_exp) per 32-element block, matching the
        // kernel's `block_idx = (col*K+k)/32` layout. Passed via `_b_scales`
        // so `quantized_matmul` uploads real exponents instead of a zero dummy.
        let num_blocks = (n * k) / 32;
        let b_exps: Vec<f32> = vec![shared_exp as f32; num_blocks];

        let (out, handle) = dev.quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &b_exps,
            grim_tensor::QuantFormat::Fp4Block16,
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
            "Jay MXFP4 GPU matmul max error {max_err} exceeds 1e-3 threshold"
        );
    }

    Ok(())
}
