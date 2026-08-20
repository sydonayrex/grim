//! CPU-vs-GPU parity test for the Q5_K fused dequant-GEMM kernel.
//!
//! Port of `q3k_gemm_cpu_gpu_parity.rs` verdicts to Q5_K (WI-P7): locks the
//! ROCm `dequant_q5k_element` layout to the authoritative CPU reference
//! `grim_quant::dequant_q5k` (matches llama.cpp `dequantize_row_q5_K`).
//! Q5_K block layout (176 bytes / 256 weights):
//!   d (2) @0, dmin (2) @2, scales (12) @4, qh (32) @16, qs (128) @48.
//!
//! Without `GRIM_GPU_TEST=1` the GPU half bails; the CPU-only element-wise
//! self-checks still run on CI (default crate lint set denies warnings).

use grim_backend_rocm::RocmDevice;
use grim_quant::dequant_q5k;
use grim_tensor::{
    BackendDevice, DType, KQuantScheme, Shape,
    dtype::{ArithType, Storage},
};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    match panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

fn f16_to_f32(b0: u8, b1: u8) -> f32 {
    let h = (b0 as u16) | ((b1 as u16) << 8);
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let res = (mant as f32) / 1024.0 * 0.000061_0351_5625;
        return if sign == 1 { -res } else { res };
    }
    if exp == 31 {
        return f32::INFINITY;
    }
    let res = (1.0 + (mant as f32) / 1024.0) * (2f32).powi((exp as i32) - 15);
    if sign == 1 { -res } else { res }
}

/// Port of the device `dequant_q5k_element` for host-side self-check. Reads a
/// 176-byte block_q5_K and returns the value at index `in_sb`. Must agree with
/// `grim_quant::dequant_q5k(&block, 256)[in_sb]`.
fn dequant_q5k_element_host(block: &[u8; 176], in_sb: usize) -> f32 {
    let d = f16_to_f32(block[0], block[1]);
    let dmin = f16_to_f32(block[2], block[3]);
    let scales = &block[4..16]; // 12 bytes
    let qh = &block[16..48]; // 32 bytes, 2 high bits/weight
    let qs = &block[48..176]; // 128 bytes, low 4 bits/weight

    // ggml layout: four 64-weight groups. Within group n, weights [0..32)
    // take the low nibble of qs[n*32 + l] with high bit qh[l] & (1 << 2n) and
    // scale sub-block 2n; weights [32..64) take the high nibble with bit
    // qh[l] & (1 << (2n+1)) and scale sub-block 2n+1.
    let n = in_sb / 64;
    let j = in_sb % 64;
    let l = j & 31;
    let hi = j >> 5;
    let is = 2 * n + hi; // 0..7

    // 6-bit scale unpacking (same as Q4_K): sc and m each 6 bits.
    let (sc_raw, m_raw) = if is < 4 {
        (scales[is] & 63, scales[is + 4] & 63)
    } else {
        let j = is - 4;
        (
            (scales[is + 4] & 0x0F) | ((scales[j] >> 6) << 4),
            (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4),
        )
    };

    let packed = qs[n * 32 + l];
    let q_low = if hi != 0 { packed >> 4 } else { packed & 0x0F };
    let msb = (qh[l] >> (2 * n + hi)) & 1;
    let q_code = (q_low as i32) | ((msb as i32) << 4);

    d * (sc_raw as f32) * (q_code as f32) - dmin * (m_raw as f32)
}

fn build_block(seed: u32) -> [u8; 176] {
    let mut b = [0u8; 176];
    for i in 0..2usize {
        b[i] = (i.wrapping_mul(7).wrapping_add(seed as usize)) as u8;
    }
    // d = 1.0 fp16 (00 3C), dmin = 0.0 fp16 (00 00)
    b[0] = 0x00;
    b[1] = 0x3C;
    b[2] = 0x00;
    b[3] = 0x00;
    for i in 4..16usize {
        b[i] = (i.wrapping_mul(11).wrapping_add(seed as usize)) as u8;
    }
    for i in 16..48usize {
        b[i] = (i.wrapping_mul(13).wrapping_add(seed as usize) ^ 0x55) as u8;
    }
    for i in 48..176usize {
        b[i] = (i.wrapping_mul(17).wrapping_add(seed as usize)) as u8;
    }
    b
}

#[test]
fn test_q5k_element_matches_cpu_reference_across_seeds() {
    for seed in 0..16u32 {
        let block = build_block(seed);
        let cpu_all = dequant_q5k(&block, 256).expect("dequant_q5k");
        let mut max_err: f32 = 0.0;
        for i in 0..256usize {
            let elem = dequant_q5k_element_host(&block, i);
            let err = (elem - cpu_all[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-6,
            "seed={seed}: element-wise vs CPU reference max_err {max_err} exceeds 1e-6"
        );
    }
}

#[test]
fn test_q5k_gpu_gemm_matches_cpu_dequant_reference() -> TestResult {
    // End-to-end forward + backward parity on a real AMD GPU. Without
    // GRIM_GPU_TEST=1 this bails green (CI-safe).
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let (m, k, n) = (4, 256, 16);
    let row_bytes = 176usize;

    let mut b_packed: Vec<u8> = Vec::with_capacity(n * row_bytes);
    let mut b_f32: Vec<f32> = Vec::with_capacity(k * n);
    for col in 0..n {
        let block = build_block(col as u32);
        b_packed.extend_from_slice(&block);
        b_f32.extend(dequant_q5k(&block, 256)?);
    }
    assert_eq!(b_packed.len(), n * row_bytes);

    let a_host: Vec<f32> = (0..(m * k) as u32)
        .map(|i| (i as f32 * 0.07).cos())
        .collect();

    let a_shape = Shape::from_slice(&[m, k]);
    let a_rocm = dev.from_cpu(&a_host, &a_shape, DType::F32)?;
    let b_shape = Shape::from_slice(&[n * row_bytes]);
    let b_rocm = dev.from_cpu_bytes(
        &b_packed,
        &b_shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q5K),
        },
    )?;
    let out_shape = Shape::from_slice(&[m, n]);
    let (c_rocm, _) = dev.quantized_matmul(
        a_rocm.as_ref(),
        b_rocm.as_ref(),
        &[],
        grim_tensor::QuantFormat::Q8_0,
        &out_shape,
    )?;
    let c_gpu = c_rocm.to_cpu_vec_f32()?;

    let mut c_ref = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_host[row * k + kk] * b_f32[col * k + kk];
            }
            c_ref[row * n + col] = acc;
        }
    }
    let max_abs = c_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err: f32 = c_ref
        .iter()
        .zip(c_gpu.iter())
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f32, f32::max);
    let rel = if max_abs == 0.0 { max_err } else { max_err / max_abs };
    assert!(
        rel < 2e-5,
        "Q5_K forward GEMM max_rel_err {rel:.3e} (max_abs {max_abs}) exceeds 2e-5"
    );

    // ── Backward: dX[m,k] = dY[m,n] @ B[k,n]^T ─────────────────────────────
    let dy_host: Vec<f32> = (0..(m * n) as u32)
        .map(|i| (i as f32 * 0.11).sin())
        .collect();
    let dy_shape = Shape::from_slice(&[m, n]);
    let dy_rocm = dev.from_cpu(&dy_host, &dy_shape, DType::F32)?;
    let dx_shape = Shape::from_slice(&[m, k]);
    let (dx_rocm, _) = dev.quantized_matmul_backward_dx(
        dy_rocm.as_ref(),
        b_rocm.as_ref(),
        &[],
        5, // bpw hint (~5 bits/weight)
        m,
        n,
        k,
        &dx_shape,
        None,
    )?;
    let dx_gpu = dx_rocm.to_cpu_vec_f32()?;

    let mut dx_ref = vec![0.0f32; m * k];
    for row in 0..m {
        for kk in 0..k {
            let mut acc = 0.0f32;
            for col in 0..n {
                acc += dy_host[row * n + col] * b_f32[col * k + kk];
            }
            dx_ref[row * k + kk] = acc;
        }
    }
    let bx = dx_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let derr: f32 = dx_ref
        .iter()
        .zip(dx_gpu.iter())
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f32, f32::max);
    let drel = if bx == 0.0 { derr } else { derr / bx };
    assert!(
        drel < 2e-5,
        "Q5_K backward dX max_rel_err {drel:.3e} (max_abs {bx}) exceeds 2e-5"
    );

    println!("[q5k parity] forward rel={rel:.3e}  backward rel={drel:.3e}");
    Ok(())
}