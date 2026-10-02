//! WhiteRaven blocked-B parity: `grim_wmma_gemm_fp8_e4m3_blocked` must compute
//! exactly what `grim_wmma_gemm_fp8_e4m3` computes.
//!
//! The blocked entry only changes HOW B bytes are fetched (one contiguous 256B
//! tile per fragment load instead of 16 K-strided 16B segments); the values
//! and the accumulation order are identical, so the outputs must be
//! bit-exact, not merely close. A tolerance here would launder a layout bug.
//!
//! Covers the M=1 decode shape (16-row pad, 1 live row) and the M=16 native
//! tile, plus N=48 (odd tile count, exercises the `valid_col1=false` path).

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

fn raw(t: &Box<dyn grim_tensor::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
    grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
}

fn read_f32(t: &Box<dyn grim_tensor::BackendStorage>) -> Vec<f32> {
    raw(t)
        .copy_to_host()
        .expect("d2h")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parity_case(m: usize, n: usize, k: usize, seed: u64) -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    assert_eq!(k % 16, 0);
    assert_eq!(n % 16, 0);

    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };

    // O(1) f32 fibers, encoded with the RNE E4M3 converter (per element --
    // NOT quant_fp8, whose output carries a 4-byte scale prefix).
    let a_f32: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let b_f32: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let a_codes: Vec<u8> = a_f32
        .iter()
        .map(|&v| grim_quant::f32_to_fp8_e4m3(v))
        .collect();
    let b_codes: Vec<u8> = b_f32
        .iter()
        .map(|&v| grim_quant::f32_to_fp8_e4m3(v))
        .collect();

    let u8ty = || DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    // A padded to whole 16-row tiles (launcher contract); pad rows are +0.0.
    let a_rows = m.div_ceil(16) * 16;
    let mut a_padded = vec![0u8; a_rows * k];
    for r in 0..m {
        a_padded[r * k..(r + 1) * k].copy_from_slice(&a_codes[r * k..(r + 1) * k]);
    }
    let a_t = MemoryOps::from_cpu_bytes(&dev, &a_padded, &Shape::new(vec![a_rows, k]), u8ty())
        .map_err(|e| format!("a h2d: {e}"))?;

    // Old path: row-major B.
    let b_old_t = MemoryOps::from_cpu_bytes(&dev, &b_codes, &Shape::new(vec![n, k]), u8ty())
        .map_err(|e| format!("b_old h2d: {e}"))?;
    // New path: 16x16-blocked B (flat shape: the layout is bytes, not [n,k]).
    let b_blocked = grim_quant::block_fp8_16x16(&b_codes, n, k).expect("block");
    let b_new_t = MemoryOps::from_cpu_bytes(&dev, &b_blocked, &Shape::new(vec![n * k]), u8ty())
        .map_err(|e| format!("b_new h2d: {e}"))?;

    let f32ty = || DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let out_old_t = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m, n]), f32ty())
        .map_err(|e| format!("out_old: {e}"))?;
    let out_new_t = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m, n]), f32ty())
        .map_err(|e| format!("out_new: {e}"))?;

    dev.launch_wmma_gemm_fp8_e4m3_for_ab(raw(&a_t), raw(&b_old_t), raw(&out_old_t), m, n, k)
        .map_err(|e| format!("old launch: {e}"))?;
    dev.launch_wmma_gemm_fp8_e4m3_blocked(raw(&a_t), raw(&b_new_t), raw(&out_new_t), m, n, k)
        .map_err(|e| format!("blocked launch: {e}"))?;
    dev.synchronize();

    let old = read_f32(&out_old_t);
    let new = read_f32(&out_new_t);
    assert_eq!(old.len(), m * n);
    assert_eq!(new.len(), m * n);
    let mut worst_bit = 0u32;
    for (i, (&a, &b)) in old.iter().zip(&new).enumerate() {
        let d = (a.to_bits() as i32)
            .wrapping_sub(b.to_bits() as i32)
            .unsigned_abs();
        worst_bit = worst_bit.max(d);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "m={m} n={n} k={k} output {i}: old={a:e} blocked={b:e}"
        );
    }
    println!("whiteraven-blocked parity: m={m} n={n} k={k} worst_ulp={worst_bit} (exact)");
    Ok(())
}

#[test]
fn blocked_matches_row_major_at_decode_m1() -> TestResult {
    parity_case(1, 256, 128, 0xB10C4ED)
}

#[test]
fn blocked_matches_row_major_at_native_m16() -> TestResult {
    parity_case(16, 256, 128, 0xB10C4ED ^ 0x1111)
}

#[test]
fn blocked_matches_row_major_with_ragged_n_tiles() -> TestResult {
    // N=48: 2 blocks cover tiles {0,1} and {2}; the last block takes the
    // valid_col1=false branch.
    parity_case(1, 48, 128, 0xB10C4ED ^ 0x4848)
}
