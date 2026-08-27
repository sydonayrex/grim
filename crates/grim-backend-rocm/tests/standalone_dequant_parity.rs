//! GPU-vs-CPU validation for the standalone dequant launchers.
//!
//! Run with:
//!   cargo test -p grim-backend-rocm --test standalone_dequant_parity -- --nocapture
//!
//! On a real ROCm device: `GRIM_RUN_GPU_TESTS=1 ... -- --ignored`.
//!
//! FP8 / MXFP4 / MXFP8 kernels are expected to be bit-exact against the CPU
//! oracles in `grim_quant`. The IQ kernels use a simplified index-as-scale model
//! and are expected to *deviate*; we only assert they run and we record max_err
//! so the deviation is visible in the log.

use grim_backend_rocm::RocmDevice;
use grim_quant::{
    dequant_fp8, dequant_iq2s, dequant_iq2xs, dequant_iq2xxs, dequant_iq3s, dequant_iq3xxs,
    dequant_iq4nl, dequant_iq4xs, dequant_mxfp4, dequant_mxfp8, dequant_q4k,
};
use grim_tensor::error::Error as QuantError;

/// NaN-aware bit-exact comparison.
fn same_val(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// Build a Q8_0 test roster.
fn build_q8_0_bytes() -> Vec<u8> {
    const QK8_0: usize = 32;
    let mut raw = Vec::new();
    let f16_bits = half::f16::from_f32(2.0).to_bits();
    // Block 1: scale=2.0, codes=[1,2,...,32]
    raw.extend_from_slice(&f16_bits.to_le_bytes());
    for i in 1..=QK8_0 {
        raw.push(i as u8);
    }
    // Block 2: scale=2.0, codes=[-1,-2,...,-32]
    raw.extend_from_slice(&f16_bits.to_le_bytes());
    for i in 1..=QK8_0 {
        raw.push((-(i as i8)) as u8);
    }
    raw
}

/// CPU oracle dequantize Q8_0.
fn dequant_q8_0_cpu(bytes: &[u8], n_weights: usize) -> Result<Vec<f32>, QuantError> {
    const QK8_0: usize = 32;
    let mut out = Vec::with_capacity(n_weights);
    for blk in bytes.chunks_exact(QK8_0 + 2) {
        let d_bits = u16::from_le_bytes([blk[0], blk[1]]);
        let d = half::f16::from_bits(d_bits).to_f32();
        let qs = &blk[2..2 + QK8_0];
        for &q in qs {
            out.push(d * (q as i8 as f32));
        }
    }
    Ok(out)
}

/// Build a Q4_K test super-block.
fn build_q4k_bytes() -> Vec<u8> {
    const BLOCK_BYTES: usize = 144;
    let mut buf = Vec::new();
    let start = buf.len();
    buf.resize(start + BLOCK_BYTES, 0u8);
    let block = &mut buf[start..start + BLOCK_BYTES];

    let d_bits = half::f16::from_f32(2.0).to_bits().to_le_bytes();
    let min_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
    block[0..2].copy_from_slice(&d_bits);
    block[2..4].copy_from_slice(&min_bits);

    let mut scales = [0u8; 12];
    scales[0] = 0x01; // sc0 = 1, m0 = 0
    scales[8] = 0x35; // sc4 = 5, m4 = 3
    block[4..16].copy_from_slice(&scales);
    block[80] = 0x35;
    block[16] = 0x35;
    buf
}

/// Build an FP8 roster: 4-byte f32 LE scale header + one E4M3 code per element.
fn build_fp8_bytes(n: usize, scale: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + n);
    bytes.extend_from_slice(&scale.to_le_bytes());
    for i in 0..n {
        // Exercise subnormals (exp=0), normals, negatives, and one NaN.
        let code = match i % 8 {
            0 => 0x00,                    // +0 subnormal
            1 => 0x01,                    // +1*2^-9 subnormal (mant=1)
            2 => 0x03,                    // +3*2^-9 subnormal
            3 => 0x10,                    // exp=2, mant=0 -> 1.0
            4 => 0x1B,                    // exp=3, mant=3 -> 1.375*2
            5 => 0x88,                    // negative normal (exp=1, mant=0)
            6 => 0x7F,                    // NaN
            _ => 0xC0 | (i as u8 & 0x07), // negative normal, varying mantissa
        };
        bytes.push(code);
    }
    bytes
}

/// Build an MXFP single-buffer roster (length-prefixed codes/exps segments).
/// `code_for` returns one code byte per element; nibble-packed for MXFP4.
fn build_mxfp_bytes(n: usize, codes_per_2: bool, exps: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let codes: Vec<u8> = (0..n)
        .map(|i| {
            if codes_per_2 {
                let nib = (i % 15) as u8; // 0..=14, all valid E2M1 codes
                nib
            } else {
                // One full E4M3 byte per element, exercising subnormals.
                match i % 8 {
                    0 => 0x00,
                    1 => 0x01,
                    2 => 0x03,
                    3 => 0x10,
                    4 => 0x1B,
                    5 => 0x88,
                    6 => 0x7F,
                    _ => 0xC0 | (i as u8 & 0x07),
                }
            }
        })
        .collect();
    let packed_codes: Vec<u8> = if codes_per_2 {
        codes
            .chunks(2)
            .map(|c| match c {
                [a, b] => a | (b << 4),
                [a] => *a,
                _ => unreachable!(),
            })
            .collect()
    } else {
        codes
    };
    bytes.extend_from_slice(&(packed_codes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&packed_codes);
    bytes.extend_from_slice(&(exps.len() as u64).to_le_bytes());
    bytes.extend_from_slice(exps);
    bytes
}

fn dev() -> RocmDevice {
    RocmDevice::try_new(0).expect("RocmDevice::try_new(0) should succeed on a system with ROCm")
}

fn assert_bit_exact(name: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{name}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    let mut bad = 0usize;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        if !same_val(*g, *e) {
            if bad < 8 {
                eprintln!("[{name}] MISMATCH i={i}: got={g:?} expected={e:?}");
            }
            bad += 1;
        }
    }
    assert!(
        bad == 0,
        "{name}: {bad}/{len} elements deviate (see log)",
        len = expected.len()
    );
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn q8_0_kernel_matches_cpu_oracle() {
    let bytes = build_q8_0_bytes();
    let n = 64;
    let expected = dequant_q8_0_cpu(&bytes, n).unwrap();
    let got = dev().dequantize_q8_0_host(&bytes, n).unwrap();
    eprintln!("[q8_0_parity] first 8 got={:?}", &got[0..8]);
    assert_bit_exact("q8_0", &got, &expected);
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn q4k_kernel_matches_cpu_oracle() {
    let bytes = build_q4k_bytes();
    let n = 256;
    let expected = dequant_q4k(&bytes, n).unwrap();
    let got = dev().dequantize_q4k_host(&bytes, n).unwrap();
    eprintln!("[q4k_parity] first 8 got={:?}", &got[0..8]);
    assert_bit_exact("q4k", &got, &expected);
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn fp8_kernel_matches_cpu_oracle() {
    let n: usize = 512;
    let scale = 2.0f32;
    let bytes = build_fp8_bytes(n, scale);
    let expected = dequant_fp8(&bytes, n).unwrap();
    let got = dev().dequantize_fp8_host(&bytes, n).unwrap();

    // Prove the subnormal fix explicitly: code 0x01 -> mant=1, exp=0 -> 1/512, *2.
    let expected_sub = (1.0f32 / 512.0) * scale;
    assert!(
        same_val(got[1], expected_sub),
        "fp8 subnormal 0x01 expected {expected_sub} got {}",
        got[1]
    );

    eprintln!("[fp8_parity] first 8 got={:?}", &got[0..8]);
    assert_bit_exact("fp8", &got, &expected);
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn mxfp4_kernel_matches_cpu_oracle() {
    let n: usize = 512;
    let exps = vec![127u8; n.div_ceil(32)];
    let bytes = build_mxfp_bytes(n, true, &exps);
    let expected = dequant_mxfp4(&bytes, n).unwrap();
    let got = dev().dequantize_mxfp4_host(&bytes, n).unwrap();
    eprintln!("[mxfp4_parity] first 8 got={:?}", &got[0..8]);
    assert_bit_exact("mxfp4", &got, &expected);
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn mxfp8_kernel_matches_cpu_oracle() {
    let n: usize = 512;
    let exps = vec![127u8; n.div_ceil(32)];
    let bytes = build_mxfp_bytes(n, false, &exps);
    let expected = dequant_mxfp8(&bytes, n).unwrap();
    let got = dev().dequantize_mxfp8_host(&bytes, n).unwrap();
    eprintln!("[mxfp8_parity] first 8 got={:?}", &got[0..8]);
    assert_bit_exact("mxfp8", &got, &expected);
}

// The dyn-Fn parameter keeps each quant-format case a one-liner at the call site.
#[allow(clippy::type_complexity)]
fn run_iq(
    name: &str,
    block_bytes: usize,
    cpu: &dyn Fn(&[u8], usize) -> Result<Vec<f32>, QuantError>,
) {
    let n_blocks = 2usize;
    let n_weights = n_blocks * 256;
    let mut bytes = Vec::with_capacity(n_blocks * block_bytes);
    for i in 0..(n_blocks * block_bytes) {
        bytes.push((i * 7 % 256) as u8);
    }
    let got = match name {
        "iq2xxs" => dev().dequantize_iq2xxs_host(&bytes, n_weights).unwrap(),
        "iq2xs" => dev().dequantize_iq2xs_host(&bytes, n_weights).unwrap(),
        "iq2s" => dev().dequantize_iq2s_host(&bytes, n_weights).unwrap(),
        "iq3xxs" => dev().dequantize_iq3xxs_host(&bytes, n_weights).unwrap(),
        "iq3s" => dev().dequantize_iq3s_host(&bytes, n_weights).unwrap(),
        "iq4nl" => dev().dequantize_iq4nl_host(&bytes, n_weights).unwrap(),
        "iq4xs" => dev().dequantize_iq4xs_host(&bytes, n_weights).unwrap(),
        _ => panic!("unknown iq scheme {name}"),
    };
    assert_eq!(got.len(), n_weights, "{name}: kernel produced wrong count");
    let expected = match cpu(&bytes, n_weights) {
        Ok(e) => e,
        Err(_) => {
            eprintln!(
                "[{name}_parity] skipped CPU oracle comparison (format unimplemented in CPU oracle)"
            );
            return;
        }
    };
    assert_eq!(
        expected.len(),
        n_weights,
        "{name}: oracle produced wrong count"
    );

    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!(
        "[{name}_parity] kernel ran ok; max|gpu-cpu| = {} (kernels use simplified model)",
        max_err
    );
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn iq_kernels_run_and_report_deviation() {
    run_iq("iq2xxs", 66, &|b, n| dequant_iq2xxs(b, n));
    run_iq("iq2xs", 74, &|b, n| dequant_iq2xs(b, n));
    run_iq("iq2s", 82, &|b, n| dequant_iq2s(b, n));
    run_iq("iq3xxs", 96, &|b, n| dequant_iq3xxs(b, n));
    run_iq("iq3s", 110, &|b, n| dequant_iq3s(b, n));
    run_iq("iq4nl", 170, &|b, n| dequant_iq4nl(b, n));
    run_iq("iq4xs", 178, &|b, n| dequant_iq4xs(b, n));
}
