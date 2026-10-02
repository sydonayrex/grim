//! WhiteRaven reaches the blocked WMMA GEMM through dtype dispatch.
//!
//! A `FloatPackScheme::Fp8Blocked16` tensor (16x16-blocked E4M3, same bytes a
//! loader would store) must route from `quantized_matmul` to
//! `grim_wmma_gemm_fp8_e4m3_blocked` for decode (m=1) and prefill (m>1),
//! with f32 activations converted through the dispatch's host path. The
//! oracle uses the QUANTIZED operands (fp8 codes decoded back), so it pins
//! kernel math, not the quantizer -- the same discipline as
//! tree_pie_dispatch. Bit-level layout agreement is gated separately by
//! whiteraven_blocked_parity (0 ulp vs the row-major kernel).

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, DType, MemoryOps, QuantOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

/// Dispatch case: f32 activations of shape [m, k], blocked fp8 B of shape
/// [n, k]. Returns the worst relative error against the quantized oracle.
fn dispatch_case(m: usize, n: usize, k: usize, seed: u64) -> TestResult<f64> {
    let Some(dev) = gpu_device() else {
        return Ok(0.0);
    };

    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    // O(1) fibers (E4M3 tops out at 448; keep well inside).
    let a: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();

    // Quantized oracle: dispatch converts f32 acts with f32_to_fp8_e4m3, so
    // match its rounding exactly; B codes decode with fp8_e4m3_to_f32.
    let aq: Vec<f64> = a
        .iter()
        .map(|&v| grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(v)) as f64)
        .collect();
    let wq: Vec<f64> = w
        .iter()
        .map(|&v| grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(v)) as f64)
        .collect();
    let mut oracle = vec![0.0f64; m * n];
    let mut scale = vec![0.0f64; m * n];
    for r in 0..m {
        for c in 0..n {
            for kk in 0..k {
                let t = aq[r * k + kk] * wq[c * k + kk];
                oracle[r * n + c] += t;
                scale[r * n + c] += t.abs();
            }
        }
    }

    let f32ty = || DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let a_t = MemoryOps::from_cpu_bytes(&dev, &a_bytes, &Shape::new(vec![m, k]), f32ty())
        .map_err(|e| format!("act h2d: {e}"))?;

    let codes: Vec<u8> = w.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let blocked = grim_quant::block_fp8_16x16(&codes, n, k).expect("block");
    let wr_ty = DType {
        arith: ArithType::U8,
        storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16),
    };
    let b_t = MemoryOps::from_cpu_bytes(&dev, &blocked, &Shape::new(vec![n, k]), wr_ty)
        .map_err(|e| format!("b blocked h2d: {e}"))?;

    let out_shape = Shape::new(vec![m, n]);
    let (out, _handle) = dev
        .quantized_matmul(
            &*a_t,
            &*b_t,
            &[],
            grim_tensor::QuantFormat::Fp8Blocked16,
            &out_shape,
        )
        .map_err(|e| format!("quantized_matmul: {e}"))?;
    dev.synchronize();

    let raw = grim_backend_rocm::as_rocm(out.as_ref())
        .map_err(|e| e.to_string())?
        .copy_to_host()?;
    let got: Vec<f64> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();
    assert_eq!(got.len(), m * n);
    let mut worst = 0.0f64;
    for i in 0..m * n {
        let rel = ((got[i] - oracle[i]) / scale[i].max(f64::MIN_POSITIVE)).abs();
        worst = worst.max(rel);
    }
    println!("whiteraven dispatch: m={m} n={n} k={k} worst_rel={worst:.3e}");
    Ok(worst)
}

#[test]
fn blocked_reaches_the_wmma_through_dispatch_at_decode_m1() -> TestResult {
    let worst = dispatch_case(1, 256, 128, 0xD15EA7C)?;
    assert!(worst < 1e-3, "dispatch math wrong: worst {worst:.3e}");
    Ok(())
}

#[test]
fn blocked_reaches_the_wmma_through_dispatch_at_prefill_m16() -> TestResult {
    let worst = dispatch_case(16, 256, 128, 0xD15EA7C ^ 0x10)?;
    assert!(worst < 1e-3, "dispatch math wrong: worst {worst:.3e}");
    Ok(())
}

#[test]
fn dispatch_refuses_ragged_k() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    // k % 16 != 0: rocwmma steps K by 16 with no tail guard.
    let (m, n, k) = (1usize, 16usize, 130usize);
    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let a_t =
        MemoryOps::from_cpu_bytes(&dev, &vec![0u8; m * k * 4], &Shape::new(vec![m, k]), f32ty)
            .map_err(|e| format!("act h2d: {e}"))?;
    let wr_ty = DType {
        arith: ArithType::U8,
        storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16),
    };
    // B bytes are unchecked content here: dispatch must refuse on geometry
    // before touching them.
    let b_t = MemoryOps::from_cpu_bytes(&dev, &vec![0u8; n * k], &Shape::new(vec![n, k]), wr_ty)
        .map_err(|e| format!("b h2d: {e}"))?;
    let out_shape = Shape::new(vec![m, n]);
    let r = dev.quantized_matmul(
        &*a_t,
        &*b_t,
        &[],
        grim_tensor::QuantFormat::Fp8Blocked16,
        &out_shape,
    );
    assert!(r.is_err(), "ragged k must be refused, not miscomputed");
    Ok(())
}

/// The capture-safe pair (`launch_quant_fp8_pad16` + blocked GEMM) must agree
/// bit-for-bit with the eager dispatch's host conversion.
///
/// Two independent fp8 encoders exist (the prologue kernel's RNE and the host
/// `f32_to_fp8_e4m3`), and graph decode uses the prologue while eager uses the
/// host. If they disagree, the same weights and activations give different
/// logits depending on which path ran — the divergence that no single-path
/// test can see, because each path is only ever compared to itself.
#[test]
fn device_act_quantizer_matches_the_host_converter() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let (m, k) = (16usize, 128usize);
    // Fixtures that straddle every rounding branch: ties (RNE must go to even),
    // saturation (>448), subnormal, sub-1e-3 flush-to-zero, NaN, +-0.
    let mut a: Vec<f32> = Vec::with_capacity(m * k);
    for i in 0..m * k {
        a.push(match i % 13 {
            0 => 0.0,
            1 => -0.0,
            2 => 448.0,
            3 => 1.0e4,  // saturate
            4 => 1.0e-4, // below min subnormal
            5 => f32::NAN,
            6 => -448.0,
            7 => 0.001953125,  // exactly 2^-9, the smallest subnormal
            8 => 0.0009765625, // exactly 2^-10, ties toward zero
            9 => 0.005859375,  // tie between two codes
            10 => -1.0 / 3.0,
            11 => 0.09375, // exact mid of two codes
            _ => (i as f32) * 0.03125 - 0.5,
        });
    }

    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter()
            .flat_map(|v| v.to_le_bytes().to_vec())
            .collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        f32ty,
    )
    .map_err(|e| format!("act h2d: {e}"))?;
    let a_rocm = grim_backend_rocm::as_rocm(a_t.as_ref()).map_err(|e| e.to_string())?;

    let a_rows = m.div_ceil(16) * 16;
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let pad_t = MemoryOps::from_cpu_bytes(
        &dev,
        &vec![0u8; a_rows * k],
        &Shape::new(vec![a_rows, k]),
        u8ty,
    )
    .map_err(|e| format!("pad h2d: {e}"))?;
    let pad_rocm = grim_backend_rocm::as_rocm(pad_t.as_ref()).map_err(|e| e.to_string())?;

    dev.launch_quant_fp8_pad16(a_rocm, pad_rocm, m, k)
        .map_err(|e| format!("launch_quant_fp8_pad16: {e}"))?;
    dev.synchronize();
    let got = pad_rocm.copy_to_host().map_err(|e| e.to_string())?;

    let mut mismatches = 0usize;
    for (i, &v) in a.iter().enumerate() {
        let want = grim_quant::f32_to_fp8_e4m3(v);
        let have = got[i];
        if want != have {
            mismatches += 1;
            println!("act quant mismatch at {i}: v={v:e} host={want:#04x} device={have:#04x}");
        }
    }
    // Pad rows must stay zero: the GEMM reads a whole 16-row tile, and stale
    // scratch would add a phantom activation row into the m=16 tail.
    for i in m * k..a_rows * k {
        assert_eq!(got[i], 0, "pad row {i} is not zero");
    }
    assert_eq!(
        mismatches,
        0,
        "device act quantizer disagrees with the host on {mismatches} of {} values",
        m * k
    );
    Ok(())
}
