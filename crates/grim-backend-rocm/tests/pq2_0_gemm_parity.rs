//! Task 10.5 gate: Prism PQ2_0 dot4 GEMV parity.
//!
//! The GPU kernel `grim_dot4_pq2_0_q80_gemv` (geometry clone of the Q2_0
//! kernel at 128-weight blocks) must agree with the host oracle:
//! `grim_quant::dequant_pq2_0` → f32 matmul against the same q8_0 activations
//! decoded back to f32. The kernel's inner dot is pure integer MAC, so the
//! comparison is tight (float rounding only in the f16 scale multiplies).

use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, Shape};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed")).ok()
}

/// Pack one PQ2_0 block: `[f16 delta LE][32 bytes, 4 codes each]`.
fn pack_pq2_0_block(delta: f32, codes: &[u8; 128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(34);
    out.extend_from_slice(&delta.to_le_bytes()); // f16 bit pattern via f32->f16
    // f32 → f16 conversion (round-to-nearest is not needed; deltas are chosen
    // exactly representable in f16).
    let h = f32_to_f16_bits(delta);
    let bytes = out.split_off(0);
    let _ = bytes;
    let mut blk = h.to_le_bytes().to_vec();
    for b in 0..32 {
        let mut byte = 0u8;
        for i in 0..4 {
            byte |= (codes[b * 4 + i] & 0x3) << (i * 2);
        }
        blk.push(byte);
    }
    blk
}

fn f32_to_f16_bits(v: f32) -> u16 {
    // Exact for our test values (halves, powers of two, and small decimals
    // chosen to be f16-representable).
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let man = bits & 0x7f_ffff;
    if exp < -14 {
        return sign;
    }
    if exp > 15 {
        return sign | 0x7c00;
    }
    if exp >= -14 {
        let e = (exp + 15) as u16;
        let m = ((man >> 13) & 0x3ff) as u16;
        sign | (e << 10) | m
    } else {
        sign
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32 / 1024.0;
    if exp == 0 {
        sign * man * (2f32).powi(-14)
    } else {
        sign * (1.0 + man) * (2f32).powi(exp - 15)
    }
}

#[test]
#[ignore]
fn pq2_0_dot4_gemv_matches_host_oracle() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let n = 8usize;
    let k = 512usize; // 4 PQ2_0 blocks per row
    let m = 1usize;

    // Deterministic ternary-ish codes: {0,1,2} dominate; 3 appears rarely
    // (Ternary-2's fourth code encodes +2d) to pin the (q−1) law.
    let mut state = 0x243F6A8885A308D3u64;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };

    let mut packed = Vec::with_capacity(n * (k / 128) * 34);
    for ni in 0..n {
        for blk in 0..k / 128 {
            let delta = match (ni + blk) % 4 {
                0 => 0.5,
                1 => 0.25,
                2 => 0.125,
                _ => 1.0 / 32.0,
            };
            let mut codes = [0u8; 128];
            for c in &mut codes {
                let r = rand();
                *c = if r < 0.02 {
                    3
                } else if r < 0.34 {
                    0
                } else if r < 0.67 {
                    1
                } else {
                    2
                };
            }
            packed.extend(pack_pq2_0_block(delta, &codes));
        }
    }

    // q8_0 activations: [f16 delta][32 i8] blocks, deterministic.
    let mut a_q80 = Vec::with_capacity(m * (k / 32) * 34);
    let mut a_f32 = vec![0.0f32; m * k];
    for blk in 0..k / 32 {
        let delta = 1.0 / 64.0;
        a_q80.extend(f32_to_f16_bits(delta).to_le_bytes());
        for i in 0..32 {
            let q = (((rand() * 254.0) - 127.0).round() as i8).clamp(-127, 127);
            a_q80.push(q as u8);
            a_f32[blk * 32 + i] = q as f32 * f16_bits_to_f32(f32_to_f16_bits(delta));
        }
    }

    // CPU oracle: dequant_pq2_0 → f32 dot against dequantized activations.
    let oracle = grim_quant::dequant_pq2_0(&packed, n * k)?;
    let mut expected = vec![0.0f32; m * n];
    for (ni, row) in oracle.chunks(k).enumerate() {
        let mut acc = 0.0f32;
        for ki in 0..k {
            acc += row[ki] * a_f32[ki];
        }
        expected[ni] = acc;
    }

    // GPU: upload + launch.
    let b_dev = grim_backend_rocm::MemoryOps::from_cpu_bytes(
        &dev,
        &packed,
        &Shape::new(vec![n * (k / 128) * 34]),
        DType::U8,
    )?;
    let a_dev = grim_backend_rocm::MemoryOps::from_cpu_bytes(
        &dev,
        &a_q80,
        &Shape::new(vec![m * (k / 32) * 34]),
        DType::U8,
    )?;
    let out_dev = CoreTensorOps::zeros(&dev, &Shape::new(vec![m, n]), DType::F32)?;

    let b_s = grim_backend_rocm::device::util::as_rocm(b_dev.as_ref())?;
    let a_s = grim_backend_rocm::device::util::as_rocm(a_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;

    dev.launch_dot4_pq2_0_q80_gemv(a_s, b_s, out_s, m, n, k)?;
    dev.synchronize();

    let actual = out_dev.to_cpu_vec_f32()?;
    let mut worst = 0.0f64;
    for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        let denom = exp.abs().max(1e-6);
        let rel = ((act - exp).abs() / denom) as f64;
        worst = worst.max(rel);
        assert!(rel < 1e-4, "PQ2_0 GEMV mismatch at [{i}]: got={act}, oracle={exp}, rel={rel}");
    }
    println!("pq2_0 dot4 GEMV parity: worst rel err {worst:.2e} over {n}x{k}");
    Ok(())
}
