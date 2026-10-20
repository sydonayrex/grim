//! Non-circular validity check for the Q*_K host references.
//!
//! The k-quant decode sweep (grim-backend-rocm `kquant_scheme_sweep`) compares
//! each device kernel against `grim_quant::dequant_*`. That comparison can only
//! mean something if the host reference is itself right, and for Q2_K/Q4_K/Q6_K
//! that was never established — the sweep's Q-family numbers were ambiguous
//! between "the kernel is wrong" and "the reference is wrong".
//!
//! Round-trip settles the reference half without a GPU and without trusting
//! either implementation: quantize known values, dequantize them back, and
//! require the reconstruction to sit inside the format's own quantization
//! error. A mis-indexed dequant (a wrong quarter-to-lane mapping, a swapped
//! sign plane, an off-by-one scale index) cannot agree with its own quantizer,
//! so it shows up as O(1) error rather than O(2^-bits).
//!
//! WHAT THIS DOES NOT PROVE: that the pair matches llama.cpp. Two mirrored
//! errors round-trip perfectly. It bounds self-consistency, which is the half
//! that was unverified — and it is the half that decides whether a kernel
//! disagreement means the kernel is broken.
//!
//! Measured (gfx1200 host, release, LCG seed 0x9E3779B97F4A7C15):
//!
//! ```text
//! Q6_K  6.562 bpw  rms_rel 7.5e-3  worst_rel 1.6e-2
//! Q4_K  4.500 bpw  rms_rel 1.7e-2  worst_rel 3.2e-2
//! Q2_K  2.625 bpw  rms_rel 7.8e-2  worst_rel 2.6e-1
//! ```
//!
//! Each lands where its bit-width predicts, so all three references are sound
//! and the sweep's Q2_K/Q4_K numbers are the dot4 GEMVs' accepted i32-accumulate
//! approximation (which dot_gemv_parity already gates at 0.15-0.5), not
//! reference error.

use grim_quant::{dequant_q2k, dequant_q4k, dequant_q6k, quant_q2k, quant_q4k, quant_q6k};

/// Deterministic LCG, so a failure reproduces from the test name alone.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn roundtrip(
    name: &str,
    n_weights: usize,
    quant: impl Fn(&[f32]) -> Vec<u8>,
    dequant: impl Fn(&[u8], usize) -> Vec<f32>,
    // Ceiling on RMS relative reconstruction error, sized from the format's
    // nominal resolution with headroom for its min/scale pairing:
    // 6.56 bpw ~2e-2, 4.5 bpw ~6e-2, 2.6 bpw ~2.5e-1.
    tol: f32,
) {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let src: Vec<f32> = (0..n_weights).map(|_| lcg(&mut seed)).collect();

    let bytes = quant(&src);
    let back = dequant(&bytes, n_weights);
    assert_eq!(
        back.len(),
        n_weights,
        "{name}: dequant returned {} weights, expected {n_weights}",
        back.len()
    );

    let scale = src.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    let mut sum_sq = 0.0f64;
    let mut worst = 0.0f32;
    // Not `mut`: the per-element assert below fires before any increment could,
    // so this counter is a formality kept for the message. `-D unused-mut` is on
    // workspace-wide, which is what broke this crate's test build.
    let nonfinite = 0usize;
    for (x, y) in src.iter().zip(back.iter()) {
        assert!(y.is_finite(), "{name}: non-finite reconstructed weight {y}");
        let e = (x - y).abs() / scale;
        worst = worst.max(e);
        sum_sq += (e as f64) * (e as f64);
    }
    assert_eq!(nonfinite, 0, "{name}: {nonfinite} non-finite weights");
    let rms = (sum_sq / n_weights as f64).sqrt() as f32;
    println!(
        "{name:<6} n={n_weights:<5} bytes={:<7} bpw={:.3}  rms_rel={rms:.3e}  worst_rel={worst:.3e}",
        bytes.len(),
        bytes.len() as f32 * 8.0 / n_weights as f32
    );
    assert!(
        rms <= tol,
        "{name}: quant->dequant round-trip RMS {rms:.3e} exceeds {tol:.3e}; the host \
         quantize/dequantize pair does not agree with itself, so it cannot serve \
         as a parity reference"
    );
}

/// A single super-block is the case where any per-block index error is fully
/// visible rather than averaged across neighbours.
#[test]
fn q6k_reference_roundtrip() {
    roundtrip("Q6_K", 256, |x| quant_q6k(x).unwrap(), |b, n| dequant_q6k(b, n).unwrap(), 2e-2);
}

#[test]
fn q4k_reference_roundtrip() {
    roundtrip("Q4_K", 256, |x| quant_q4k(x).unwrap(), |b, n| dequant_q4k(b, n).unwrap(), 6e-2);
}

#[test]
fn q2k_reference_roundtrip() {
    roundtrip("Q2_K", 256, |x| quant_q2k(x).unwrap(), |b, n| dequant_q2k(b, n).unwrap(), 25e-2);
}

/// Eight super-blocks: a per-block mistake that a 1-block test would catch must
/// still pass here, so this is a regression guard, not extra coverage.
#[test]
fn qk_reference_roundtrip_multi_block() {
    roundtrip("Q6_K", 2048, |x| quant_q6k(x).unwrap(), |b, n| dequant_q6k(b, n).unwrap(), 2e-2);
    roundtrip("Q4_K", 2048, |x| quant_q4k(x).unwrap(), |b, n| dequant_q4k(b, n).unwrap(), 6e-2);
    roundtrip("Q2_K", 2048, |x| quant_q2k(x).unwrap(), |b, n| dequant_q2k(b, n).unwrap(), 25e-2);
}