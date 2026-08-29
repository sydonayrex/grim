//! Parity test: `quantized_matmul` on a length-prefixed (framed) MXFP4
//! weight buffer must match the CPU dequant + GEMM oracle. This exercises
//! the device-side framing split in the ROCm MxFp4 dispatch arm.

use grim_backend_rocm::RocmDevice;
use grim_quant::{dequant_mxfp4, f32_to_mxfp4_e2m1, mxfp4_e2m1_to_f32};
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{
    QuantFormat, Shape,
    dtype::{ArithType, DType, FloatPackScheme, Storage},
};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

/// Like the framed parity test, but uses a **different E8M0 exponent per
/// 32-element block** (the real GGUF layout for MXFP4 experts). This exercises
/// the kernel's per-block exponent indexing — the original test only used a
/// single shared exponent, which would pass even if block exponents were
/// mis-indexed.
#[test]
fn mxfp4_framed_variable_exponent_parity() -> TestResult {
    if !grim_backend_rocm::gpu_test_enabled() {
        return Ok(());
    }
    let dev = RocmDevice::try_new(0)?;

    // Non-square expert-like shape: B is [out=64, in=96].
    let (m, k, n) = (3usize, 96usize, 64usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).sin()).collect();
    let b_orig: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.137 + 0.5).cos() * (1.0 + (i % 7) as f32))
        .collect();

    // Per-block (per 32 elements along K) exponents — VARIABLE across blocks.
    let exps_per_row = k / 32;
    let mut block_exps: Vec<u8> = Vec::with_capacity(n * exps_per_row);
    let mut rng = 12345u64;
    for _ in 0..(n * exps_per_row) {
        // cheap LCG; exponents in a plausible E8M0 range
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        block_exps.push(118u8 + (rng & 0x1F) as u8); // 118..149
    }

    // Pack codes even-low / odd-high in [n, k] row-major order.
    let mut codes = vec![0u8; n * k / 2];
    let mut exps = Vec::with_capacity(n * exps_per_row);
    for r in 0..n {
        for c in 0..k {
            let blk = c / 32;
            let e = block_exps[r * exps_per_row + blk];
            let code = f32_to_mxfp4_e2m1(b_orig[r * k + c], e);
            let byte_idx = r * (k / 2) + c / 2;
            if c % 2 == 0 {
                codes[byte_idx] |= code & 0x0F;
            } else {
                codes[byte_idx] |= code << 4;
            }
        }
    }
    for &e in &block_exps {
        exps.push(e);
    }

    let mut framed = Vec::with_capacity(16 + codes.len() + exps.len());
    framed.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    framed.extend_from_slice(&codes);
    framed.extend_from_slice(&(exps.len() as u64).to_le_bytes());
    framed.extend_from_slice(&exps);

    // CPU oracle: decode each element with its block exponent, then GEMM.
    let mut expected = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                let blk = ki / 32;
                let v = mxfp4_e2m1_to_f32(
                    f32_to_mxfp4_e2m1(b_orig[ni * k + ki], block_exps[ni * exps_per_row + blk]),
                    block_exps[ni * exps_per_row + blk],
                );
                sum += a_data[mi * k + ki] * v;
            }
            expected[mi * n + ni] = sum;
        }
    }

    let mxfp4_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::FloatPack(FloatPackScheme::MxFp4),
    };
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &Shape::from_slice(&[m, k]), DType::F32)?;
    let b_dev =
        MemoryOps::from_cpu_bytes(&dev, &framed, &Shape::from_slice(&[k, n]), mxfp4_dtype)?;
    let out_shape = Shape::from_slice(&[m, n]);
    let (out_dev, _h) = dev.quantized_matmul(
        a_dev.as_ref(),
        b_dev.as_ref(),
        &[],
        QuantFormat::Fp4,
        &out_shape,
    )?;
    dev.synchronize();

    let actual = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &v)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (a - v).abs();
        assert!(
            err < 1e-2,
            "GPU/CPU mismatch at {i}: actual={a}, expected={v}, err={err}"
        );
    }
    Ok(())
}

#[test]
fn mxfp4_framed_quantized_matmul_parity() -> TestResult {
    if !grim_backend_rocm::gpu_test_enabled() {
        return Ok(());
    }
    let dev = RocmDevice::try_new(0)?;

    // Non-square expert-like shape: B is [out=64, in=96] (per GGUF [out,in]).
    let (m, k, n) = (3usize, 96usize, 64usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).sin()).collect();
    let shared_exp = 129u8;
    let b_orig: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.03).cos() * 1.5).collect();

    // Codes packed even-low / odd-high (grim framing convention).
    let codes: Vec<u8> = b_orig
        .iter()
        .map(|&v| f32_to_mxfp4_e2m1(v, shared_exp))
        .collect();
    let mut packed = vec![0u8; (n * k) / 2];
    for (j, &c) in codes.iter().enumerate() {
        if j % 2 == 0 {
            packed[j / 2] |= c & 0x0F;
        } else {
            packed[j / 2] |= c << 4;
        }
    }
    let exps: Vec<u8> = vec![shared_exp; (n * k) / 32];

    // Frame: [u64 codes_len][codes][u64 exps_len][exps].
    let mut framed = Vec::with_capacity(16 + packed.len() + exps.len());
    framed.extend_from_slice(&(packed.len() as u64).to_le_bytes());
    framed.extend_from_slice(&packed);
    framed.extend_from_slice(&(exps.len() as u64).to_le_bytes());
    framed.extend_from_slice(&exps);

    // Host oracle via the shared dequantizer.
    let oracle_dequant = dequant_mxfp4(&framed, n * k)?;
    let b_dequant: Vec<f32> = b_orig
        .iter()
        .map(|&v| mxfp4_e2m1_to_f32(f32_to_mxfp4_e2m1(v, shared_exp), shared_exp))
        .collect();
    for (i, (&o, &d)) in oracle_dequant.iter().zip(b_dequant.iter()).enumerate() {
        assert!(
            (o - d).abs() < 1e-6,
            "host dequant mismatch at {i}: {o} vs {d}"
        );
    }
    let mut expected = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * oracle_dequant[ni * k + ki];
            }
            expected[mi * n + ni] = sum;
        }
    }

    let mxfp4_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::FloatPack(FloatPackScheme::MxFp4),
    };
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &Shape::from_slice(&[m, k]), DType::F32)?;
    // Framed weight stored with its logical [out, in] shape (as
    // `transpose_last_two` only relabels quantized ROCm tensors).
    let b_dev =
        MemoryOps::from_cpu_bytes(&dev, &framed, &Shape::from_slice(&[k, n]), mxfp4_dtype)?;
    let out_shape = Shape::from_slice(&[m, n]);
    let (out_dev, _h) = dev.quantized_matmul(
        a_dev.as_ref(),
        b_dev.as_ref(),
        &[],
        QuantFormat::Fp4, // unused: dispatch matches on storage dtype
        &out_shape,
    )?;
    dev.synchronize();

    let actual = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (a - e).abs();
        assert!(
            err < 1e-3,
            "GPU/CPU mismatch at {i}: actual={a}, expected={e}, err={err}"
        );
    }
    Ok(())
}
