//! Golden mutation-resistant test for Jay MXFP4 and Magpie MXFP8 GPU dequantization GEMM.

use grim_backend_rocm::RocmDevice;
use grim_quant::{f32_to_mxfp4_e2m1, mxfp4_e2m1_to_f32};
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{
    Shape,
    dtype::{ArithType, DType, FloatPackScheme, Storage},
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

// ── Real mutation proof — M-JAY-E8M0-01 ─────────────────────────────────────
//
// Manual perturbation (no cargo-mutants): flip ONE E8M0 exponent byte (the
// per-32-block shared-exponent scale for block 0: 127 -> 200, i.e. 2^73x scale
// on that block). The kernel is untouched. The true CPU oracle is untouched.
// Only the GPU's `b_scales` input is perturbed — equivalent to the kernel
// dropping/mangling the scale read for that block.
//
// Gate: passes_unmutated == true AND passes_mutated == false under the main
// test's max_err < 1e-3 assertion (the verdict helper is the same comparison,
// factored as a closure). Mutation ID: M-JAY-E8M0-01.
#[test]
#[ignore]
fn test_jay_mxfp4_gpu_gemm_mutation_is_detected() -> TestResult {
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
    let verdict = |actual: &[f32]| -> bool {
        assert_eq!(actual.len(), expected_c.len());
        let max_err = actual
            .iter()
            .zip(expected_c.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        max_err < 1e-3
    };

    // CPU-only sensitivity proof: recompute the oracle with block 0 rescaled
    // by 2^73 (what the mutated exponent means). Must fail the 1e-3 verdict.
    // b_dequant is stored [ni][ki] row-major; block 0 covers flattened
    // indices 0..32 (all of row ni=0, ki=0..32).
    let mut b_dequant_mut = b_dequant.clone();
    for idx in 0..32.min(b_dequant_mut.len()) {
        b_dequant_mut[idx] *= 2.0f32.powi(73);
    }
    let mut cpu_mut = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * b_dequant_mut[ni * k + ki];
            }
            cpu_mut[mi * n + ni] = sum;
        }
    }
    assert!(verdict(&expected_c), "oracle must pass its own verdict");
    assert!(
        !verdict(&cpu_mut),
        "M-JAY-E8M0-01 not detected at CPU level: rescaled-oracle still passes 1e-3 — fixture insensible"
    );

    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] M-JAY-E8M0-01: no GPU (CPU sensitivity proven above)");
        return Ok(());
    };

    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[n, k]);
    let out_shape = Shape::from_slice(&[m, n]);
    let mxfp4_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::FloatPack(FloatPackScheme::MxFp4),
    };

    let mut b_packed = vec![0u8; (n * k) / 2];
    for j in 0..(n * k) {
        let code = b_codes[j] & 0x0F;
        if j % 2 == 0 {
            b_packed[j / 2] |= code;
        } else {
            b_packed[j / 2] |= code << 4;
        }
    }

    let num_blocks = (n * k) / 32;
    let b_exps: Vec<f32> = vec![shared_exp as f32; num_blocks];
    let mut b_exps_mut = b_exps.clone();
    b_exps_mut[0] = 200.0; // flip ONE E8M0 exponent byte: 127 -> 200

    let run_gpu = |exps: &[f32]| -> TestResult<Vec<f32>> {
        let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &b_shape, mxfp4_dtype.clone())?;
        let (out, handle) = dev.quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            exps,
            grim_tensor::QuantFormat::Fp4Block16,
            &out_shape,
        )?;
        handle.synchronize()?;
        Ok(out.to_cpu_vec_f32()?)
    };

    let passes_unmutated = verdict(&run_gpu(&b_exps)?);
    let passes_mutated = verdict(&run_gpu(&b_exps_mut)?);
    assert!(passes_unmutated, "unmutated GPU run must pass");
    assert!(
        !passes_mutated,
        "M-JAY-E8M0-01 NOT detected: flipped-exponent GPU output still passes 1e-3 — gate did not fire"
    );
    Ok(())
}
