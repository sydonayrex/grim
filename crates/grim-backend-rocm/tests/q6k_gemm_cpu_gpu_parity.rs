//! CPU-vs-GPU parity test for the Q6_K fused dequant-GEMM kernel.
//!
//! Port of `q3k_gemm_cpu_gpu_parity.rs` verdicts to Q6_K (WI-P7): locks the
//! ROCm `dequant_q6k_element` layout to the authoritative CPU reference
//! `grim_quant::dequant_q6k` (matches llama.cpp `dequantize_row_q6_K`).
//! Q6_K block layout (210 bytes / 256 weights):
//!   ql (128) @0, qh (64) @128, scales (16, signed i8) @192, d (f16) @208.
//! Formula: `d * sc * (q - 32)` (no dmin/min term).
//!
//! Without `GRIM_GPU_TEST=1` the GPU half bails; the CPU-only element-wise
//! self-check still runs on CI.

use grim_backend_rocm::RocmDevice;
use grim_quant::dequant_q6k;
use grim_tensor::{
    CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape,
    dtype::{ArithType, Storage},
};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
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
        let res = (mant as f32) / 1024.0 * 0.000_061_035_156;
        return if sign == 1 { -res } else { res };
    }
    if exp == 31 {
        return f32::INFINITY;
    }
    let res = (1.0 + (mant as f32) / 1024.0) * (2f32).powi((exp as i32) - 15);
    if sign == 1 { -res } else { res }
}

/// Port of the device `dequant_q6k_element` for host-side self-check. Reads a
/// 210-byte block_q6_K and returns the value at index `in_sb`. Must agree with
/// `grim_quant::dequant_q6k(&block, 256)[in_sb]`.
fn dequant_q6k_element_host(block: &[u8; 210], in_sb: usize) -> f32 {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208]; // signed i8
    let d = f16_to_f32(block[208], block[209]);

    let n = in_sb / 128;
    let pos = in_sb % 128;
    let quarter = pos / 32; // 0..3
    let l = pos % 32;
    let is = l / 16; // 0 or 1
    let sc_idx = n * 8 + is + 2 * quarter;
    let sc = scales[sc_idx] as i8 as f32;

    let ql_offset = n * 64 + l + if quarter & 1 != 0 { 32 } else { 0 };
    let ql_byte = ql[ql_offset];
    let nibble = if quarter & 2 != 0 {
        ql_byte >> 4
    } else {
        ql_byte & 0x0F
    };

    let qh_byte = qh[n * 32 + l];
    let qh_bits = (qh_byte >> (2 * quarter)) & 0x03;

    let q_code = (nibble as i32) | ((qh_bits as i32) << 4);
    d * sc * ((q_code as f32) - 32.0)
}

fn build_block(seed: u32) -> [u8; 210] {
    let mut b = [0u8; 210];
    for (i, v) in b[..128].iter_mut().enumerate() {
        *v = (i.wrapping_mul(13).wrapping_add(seed as usize) ^ 0x33) as u8;
    }
    for (i, v) in b[128..192].iter_mut().enumerate() {
        *v = (i.wrapping_mul(7).wrapping_add(seed as usize)) as u8;
    }
    for (i, v) in b[192..208].iter_mut().enumerate() {
        // signed scales: wrap through 0..=255 so i8 cast stays varied
        *v = (i.wrapping_mul(5).wrapping_add(seed as usize * 3)) as u8;
    }
    b[208] = 0x00; // d = 1.0 fp16
    b[209] = 0x3C;
    b
}

#[test]
fn test_q6k_element_matches_cpu_reference_across_seeds() {
    for seed in 0..16u32 {
        let block = build_block(seed);
        let cpu_all = dequant_q6k(&block, 256).expect("dequant_q6k");
        let mut max_err: f32 = 0.0;
        for (i, &cpu_ref) in cpu_all.iter().enumerate() {
            let elem = dequant_q6k_element_host(&block, i);
            let err = (elem - cpu_ref).abs();
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
fn test_q6k_gpu_gemm_matches_cpu_dequant_reference() -> TestResult {
    // End-to-end forward + backward parity on a real AMD GPU. Without
    // GRIM_GPU_TEST=1 this bails green (CI-safe).
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let (m, k, n) = (4, 256, 16);
    let row_bytes = 210usize;

    let mut b_packed: Vec<u8> = Vec::with_capacity(n * row_bytes);
    let mut b_f32: Vec<f32> = Vec::with_capacity(k * n);
    for col in 0..n {
        let block = build_block(col as u32);
        b_packed.extend_from_slice(&block);
        b_f32.extend(dequant_q6k(&block, 256)?);
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
            storage: Storage::KQuant(KQuantScheme::Q6K),
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
    let rel = if max_abs == 0.0 {
        max_err
    } else {
        max_err / max_abs
    };
    assert!(
        rel < 2e-5,
        "Q6_K forward GEMM max_rel_err {rel:.3e} (max_abs {max_abs}) exceeds 2e-5"
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
        6, // bpw hint (~6 bits/weight)
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
        "Q6_K backward dX max_rel_err {drel:.3e} (max_abs {bx}) exceeds 2e-5"
    );

    println!("[q6k parity] forward rel={rel:.3e}  backward rel={drel:.3e}");
    Ok(())
}
