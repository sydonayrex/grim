//! CPU-vs-GPU parity test for the Q2_K fused dequant-GEMM kernel.
//!
//! Locks the ROCm `dequant_q2k_element` layout to the authoritative CPU
//! reference `grim_quant::dequant_q2k` (llama.cpp `dequantize_row_q2_K`,
//! ggml-quants.c:959, INTERLEAVED 2-bit codes). The GPU block_q2_K is
//! 84 bytes: scales[16] @0 (lo nibble = sc, hi nibble = m), qs[64] @16,
//! d (f16) @80, dmin (f16) @82. Weight w of sub-block `sub` lives at
//! qs[(sub/8)*32 + (w%16) + (sub%2)*16], low 2 bits of field
//! 2*((sub%8)/2) — the sequential 4-bytes-per-sub-block layout that used
//! to live here exists in no released GGUF (fixed 2026-10-04).
//!
//! Without `GRIM_RUN_GPU_TESTS=1` the GPU half bails, but the CPU-only
//! element-wise self-check still runs on CI.

use grim_backend_rocm::RocmDevice;
use grim_quant::dequant_q2k;
use grim_tensor::{
    CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape,
    dtype::{ArithType, Storage},
};
use std::panic;

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

/// Build a deterministic 84-byte Q2_K super-block. scales/qs cover all
/// nibble and 2-bit-field positions; d and dmin are 1.0 and 0.125 so the
/// CPU reference and the GPU element decoder exercise both terms.
fn build_block(seed: u32) -> [u8; 84] {
    let mut b = [0u8; 84];
    for (i, v) in b[0..16].iter_mut().enumerate() {
        // Ensure every nibble position varies: lo = sc (0..15), hi = m (0..15).
        *v = (i.wrapping_mul(19).wrapping_add(seed as usize).wrapping_mul(3)) as u8;
    }
    for (i, v) in b[16..80].iter_mut().enumerate() {
        *v = (i.wrapping_mul(11).wrapping_add(seed as usize).wrapping_mul(5)) as u8;
    }
    b[80] = 0x00; // d = 1.0 fp16
    b[81] = 0x3C;
    b[82] = 0x00; // dmin = 0.125 fp16
    b[83] = 0x30;
    b
}

/// Port of the device `dequant_q2k_element` for host-side self-check.
fn dequant_q2k_element_host(block: &[u8; 84], in_sb: usize) -> f32 {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16_to_f32(block[80], block[81]);
    let dmin = f16_to_f32(block[82], block[83]);

    let sub = in_sb / 16;
    let w = in_sb % 16;

    let sc = (scales[sub] & 0x0F) as f32;
    let m = (scales[sub] >> 4) as f32;

    let q_byte = (sub / 8) * 32 + (w % 16) + (sub % 2) * 16;
    let q_shift = 2 * ((sub % 8) / 2);
    let q_code = (qs[q_byte] >> q_shift) & 0x03;

    d * sc * q_code as f32 - dmin * m
}

#[test]
fn test_q2k_element_matches_cpu_reference_across_seeds() {
    for seed in 0..16u32 {
        let block = build_block(seed);
        let cpu_all = dequant_q2k(&block, 256).expect("dequant_q2k");
        let mut max_err: f32 = 0.0;
        for (i, &cpu_ref) in cpu_all.iter().enumerate() {
            let elem = dequant_q2k_element_host(&block, i);
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
#[ignore]
fn test_q2k_gpu_gemm_matches_cpu_dequant_reference() {
    // End-to-end parity on a real AMD GPU: the fused Q2_K GEMM against an
    // independent CPU reference built on `grim_quant::dequant_q2k`.
    let dev = match gpu_device() {
        Some(d) => d,
        None => return,
    };

    let (m, k, n) = (4, 256, 16);
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 84;

    let mut b_packed: Vec<u8> = Vec::with_capacity(n * row_bytes);
    let mut b_f32: Vec<f32> = Vec::with_capacity(k * n);
    for col in 0..n {
        let block = build_block(col as u32);
        b_packed.extend_from_slice(&block);
        b_f32.extend(dequant_q2k(&block, 256).expect("dequant_q2k"));
    }
    assert_eq!(b_packed.len(), n * row_bytes);

    let a_host: Vec<f32> = (0..(m * k) as u32)
        .map(|i| (i as f32 * 0.07).cos())
        .collect();

    let a_shape = Shape::from_slice(&[m, k]);
    let a_rocm = dev.from_cpu(&a_host, &a_shape, DType::F32).expect("A upload");
    let b_shape = Shape::from_slice(&[n * row_bytes]);
    let b_rocm = dev
        .from_cpu_bytes(
            &b_packed,
            &b_shape,
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::Q2K),
            },
        )
        .expect("B upload");
    let out_shape = Shape::from_slice(&[m, n]);
    let (c_rocm, _) = dev
        .quantized_matmul(
            a_rocm.as_ref(),
            b_rocm.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q8_0,
            &out_shape,
        )
        .expect("quantized_matmul");
    let c_gpu = c_rocm.to_cpu_vec_f32().expect("C readback");

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

    let amax = c_ref.iter().fold(0.0f32, |m2, v| m2.max(v.abs()));
    let max_err = c_gpu
        .iter()
        .zip(&c_ref)
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err <= amax * 2e-5 + 1e-4,
        "GPU Q2_K GEMM max_err {max_err} (amax {amax}) exceeds tolerance"
    );
}
