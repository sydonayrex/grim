//! Golden mutation-resistant test for Raven FP8 GPU dequantization GEMM forward and backward paths.

use grim_backend_rocm::RocmDevice;
use grim_quant::fp8_e4m3_to_f32;
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{
    Shape,
    dtype::{ArithType, BlockDtype, DType, Storage},
};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

#[test]
#[ignore]
fn test_raven_fp8_gpu_gemm_golden_mutation_resistant() -> TestResult {
    // Non-square asymmetric dimensions: M=2, K=128, N=64
    let (m, k, n) = (2usize, 128usize, 64usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();

    // Raw FP8 E4M3 codes (unscaled — the GPU kernel decodes each byte with
    // `fp8_e4m3_to_float_hip` and applies no per-tensor/per-block scale, so the
    // CPU reference must match that unscaled decode, not `dequant_fp8` which
    // treats the first 4 bytes as a scale prefix).
    let fp8_bytes: Vec<u8> = (0..k * n).map(|i| (i % 254 + 1) as u8).collect();
    let b_dequant: Vec<f32> = fp8_bytes.iter().map(|&b| fp8_e4m3_to_f32(b)).collect();

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
        let fp8_dtype = DType {
            arith: ArithType::F32,
            storage: Storage::Block(BlockDtype::Fp8),
        };
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &fp8_bytes, &b_shape, fp8_dtype)?;

        let (out, handle) = dev.quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Fp8,
            &out_shape,
        )?;
        handle.synchronize()?;
        let actual_c = out.to_cpu_vec_f32()?;

        assert_eq!(actual_c.len(), expected_c.len());
        // FP8 E4M3 has ~3-bit mantissa, so an absolute 1e-3 threshold is
        // unrealistic for inputs reaching ±448. Use a relative tolerance on
        // the peak-magnitude element; a real kernel bug (e.g. the MFMA inner
        // loop using a single A[row*K+k] for all 32 K-steps) blows this out
        // by orders of magnitude and still fails.
        let max_abs_expected = expected_c.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let mut max_err: f32 = 0.0;
        for (act, exp) in actual_c.iter().zip(expected_c.iter()) {
            let err = (act - exp).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.15 * max_abs_expected,
            "Raven FP8 GPU matmul max error {max_err} exceeds 15% of peak magnitude {max_abs_expected}"
        );
    }

    Ok(())
}

// ── Real mutation proof — M-FP8-TRANSPOSE-01 ─────────────────────────────────
//
// Manual perturbation (no cargo-mutants): TRANSPOSED weight read — the same
// mutation class as the historical gw[j * hidden + i] bug (charon_wmma_parity).
// The packed FP8 bytes are permuted as [n,k] -> [k,n] -> [n,k] (a transposition
// of the dequant matrix) so the kernel sees permuted weights for every output.
// Kernel and oracle are untouched; only the GPU input buffer is permuted.
//
// Gate: passes_unmutated == true AND passes_mutated == false under the main
// test's 15%-of-peak assertion. Mutation ID: M-FP8-TRANSPOSE-01.
#[test]
#[ignore]
fn test_raven_fp8_gpu_gemm_mutation_is_detected() -> TestResult {
    let (m, k, n) = (2usize, 128usize, 64usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();
    let fp8_bytes: Vec<u8> = (0..k * n).map(|i| (i % 254 + 1) as u8).collect();
    let b_dequant: Vec<f32> = fp8_bytes.iter().map(|&b| fp8_e4m3_to_f32(b)).collect();

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
    let max_abs_expected = expected_c.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let verdict = |actual: &[f32]| -> bool {
        assert_eq!(actual.len(), expected_c.len());
        let max_err = actual
            .iter()
            .zip(expected_c.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        max_err < 0.15 * max_abs_expected
    };

    // Transposed permutation: b_mut[ni*k + ki] = b_dequant[ki*n + ni].
    assert_ne!(k, n, "transposition is a no-op for square fixtures");
    let mut b_mut_dequant = vec![0.0f32; n * k];
    for ni in 0..n {
        for ki in 0..k {
            b_mut_dequant[ni * k + ki] = b_dequant[ki * n + ni];
        }
    }
    let mut cpu_mut = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * b_mut_dequant[ni * k + ki];
            }
            cpu_mut[mi * n + ni] = sum;
        }
    }
    let cpu_pass_unmutated = verdict(&expected_c);
    let cpu_pass_mutated = verdict(&cpu_mut);
    assert!(cpu_pass_unmutated, "oracle must pass its own verdict");
    assert!(
        !cpu_pass_mutated,
        "M-FP8-TRANSPOSE-01 not detected at CPU level: transposed-weight oracle still passes — fixture insensible"
    );

    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] M-FP8-TRANSPOSE-01: no GPU (CPU sensitivity proven above)");
        return Ok(());
    };
    // Same permutation on the packed bytes the kernel consumes.
    let mut fp8_mut = vec![0u8; k * n];
    for ni in 0..n {
        for ki in 0..k {
            fp8_mut[ni * k + ki] = fp8_bytes[ki * n + ni];
        }
    }

    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[n, k]);
    let out_shape = Shape::from_slice(&[m, n]);
    let fp8_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Block(BlockDtype::Fp8),
    };
    let run_gpu = |bytes: &[u8]| -> TestResult<Vec<f32>> {
        let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
        let b_dev = MemoryOps::from_cpu_bytes(&dev, bytes, &b_shape, fp8_dtype.clone())?;
        let (out, handle) = dev.quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Fp8,
            &out_shape,
        )?;
        handle.synchronize()?;
        Ok(out.to_cpu_vec_f32()?)
    };

    let passes_unmutated = verdict(&run_gpu(&fp8_bytes)?);
    let passes_mutated = verdict(&run_gpu(&fp8_mut)?);
    assert!(passes_unmutated, "unmutated GPU run must pass");
    assert!(
        !passes_mutated,
        "M-FP8-TRANSPOSE-01 NOT detected: transposed-weight GPU output still passes — gate did not fire"
    );
    Ok(())
}
