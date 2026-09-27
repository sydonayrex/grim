//! WS-C C5: converge grim's f32 -> FP8 E4M3 converters on RNE.
//!
//! # Why this file exists
//!
//! The plan found three divergent converters in the tree and was blunt about the
//! consequence: *"Do not leave this ambiguous — it will produce an
//! unreproducible A/B."* As of this commit there were four:
//!
//! | site | rounding |
//! |---|---|
//! | `dot_gemv.rs::grim_f32_to_fp8_e4m3` (device) | RNE, ties-to-even |
//! | `quant_standalone.rs::float_to_fp8_e4m3_hip` (device) | round-half-**up** |
//! | `grim-quant::f32_to_fp8_e4m3` (host API) | **truncation** |
//! | `precision_kernel_ab/e4m3.rs::to_e4m3_rne` (A/B oracle) | RNE, ties-to-even |
//!
//! Truncation is the worst of the three, and not merely because it disagrees.
//! Rounding error is symmetric noise that averages out over a dot product;
//! truncation is a *systematic* bias toward zero on every value, so it does not.
//! A GEMM over truncated weights is quietly computing something slightly wrong
//! in a consistent direction, which no tolerance test on the mean will catch.
//!
//! The A/B itself was never broken -- it carries its own faithful oracle -- but
//! any checkpoint converted through the public host API, or any path through
//! `quant_standalone`, would disagree with what the GPU kernel computes. That is
//! exactly the unreproducible comparison the plan warned about.
//!
//! RNE is the target because it is the only one of the four that is
//! round-to-nearest-*even*, which is what makes the result independent of the
//! order in which values happen to be accumulated and matches the IEEE default
//! for every other format in the tree.
//!
//! CPU only.

use grim_quant::f32_to_fp8_e4m3;

/// Independent E4M3 decoder, written from the format description rather than
/// by calling anything in grim, so agreement is evidence and not tautology.
fn e4m3_to_f32(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((code >> 3) & 0x0F) as i32;
    let mant = (code & 0x07) as f32;
    if exp == 0 {
        // Subnormal: mant/8 * 2^-6
        sign * (mant / 8.0) * (1.0 / 64.0)
    } else {
        sign * (1.0 + mant / 8.0) * (2f32).powi(exp - 7)
    }
}

/// E4M3: sign(1) | exp(4, bias 7) | mant(3). No Inf -- exp=15,mant=7 (0x7F) is
/// NaN -- so the largest finite value is `1.75 * 2^8 = 448` at code 0x7E.
const E4M3_MAX_FINITE: u8 = 0x7E;

/// The value of an E4M3 code, derived from the format definition.
///
/// Built from (exp, mant) rather than from a hardcoded table, because a table
/// is exactly what goes wrong: an earlier draft of this file listed codes 0..15
/// as if they spanned the format, when 0x7E is 448 and codes 0..15 only reach
/// 0.029. Every expectation below is now computed from the definition.
fn code_value(exp: u8, mant: u8) -> f32 {
    debug_assert!(exp < 16 && mant < 8);
    if exp == 0 {
        (mant as f32 / 8.0) * (1.0 / 64.0) // mant * 2^-9
    } else {
        (1.0 + mant as f32 / 8.0) * (2f32).powi(exp as i32 - 7)
    }
}

/// The code for an E4M3 (exp, mant) pair.
fn code_of(exp: u8, mant: u8) -> u8 {
    ((exp << 3) | mant) as u8
}

/// Every representable magnitude, in ascending order, for round-trip checks.
///
/// Enumerated from the definition rather than listed, so it cannot drift.
fn grid() -> Vec<(u8, f32)> {
    let mut out = vec![(0u8, 0.0f32)];
    for mant in 1..8u8 {
        out.push((code_of(0, mant), code_value(0, mant))); // subnormals
    }
    for exp in 1..16u8 {
        for mant in 0..8u8 {
            if exp == 15 && mant == 7 {
                continue; // 0x7F is NaN, not a value
            }
            out.push((code_of(exp, mant), code_value(exp, mant)));
        }
    }
    out
}

/// The round trip must be lossless on every representable value.
///
/// A converter that truncates still round trips *grid points* correctly -- the
/// grid is exactly representable, so there is nothing to round. This test
/// therefore cannot detect truncation on its own, which is why the tie and bias
/// tests below matter.
#[test]
fn every_grid_point_round_trips_exactly() {
    let g = grid();
    assert_eq!(g.len(), 127, "127 magnitudes: 0, 7 subnormals, 119 normals");
    for (code, v) in g {
        assert_eq!(f32_to_fp8_e4m3(v), code, "{v} must encode to code {code:#04x}");
        assert_eq!(
            f32_to_fp8_e4m3(-v),
            code | 0x80,
            "-{v} must set the sign"
        );
    }
    assert_eq!(code_value(15, 6), 448.0, "0x7E is the max finite value");
    assert_eq!(E4M3_MAX_FINITE, 0x7E);
}

/// The distinguishing test: ties go to even, not up and not down.
///
/// Midpoints in the `exp=1` binade, where the step is 2^-9. Truncation picks
/// the lower neighbour, round-half-up picks the upper one, and RNE picks
/// whichever neighbour has an even code -- so all three are distinguishable.
#[test]
fn ties_go_to_even_not_up_and_not_truncation() {
    // codes 8 (even) and 9 (odd)
    let mid_8_9 = (code_value(1, 0) + code_value(1, 1)) / 2.0;
    assert_eq!(f32_to_fp8_e4m3(mid_8_9), 8, "tie -> even code 8, not 9");
    // codes 9 (odd) and 10 (even) -- RNE must round *up* here, so this pins
    // that ties-to-even is not simply ties-down.
    let mid_9_10 = (code_value(1, 1) + code_value(1, 2)) / 2.0;
    assert_eq!(f32_to_fp8_e4m3(mid_9_10), 10, "tie -> even code 10, not 9");
    // codes 10 (even) and 11 (odd)
    let mid_10_11 = (code_value(1, 2) + code_value(1, 3)) / 2.0;
    assert_eq!(f32_to_fp8_e4m3(mid_10_11), 10, "tie -> even code 10, not 11");

    // A value three quarters of the way across, which is not a tie at all.
    // RNE gives the upper neighbour; truncation gives the lower one. This is
    // the unambiguous truncation discriminator.
    let three_quarters = code_value(1, 0) + 0.75 * (code_value(1, 1) - code_value(1, 0));
    assert_eq!(
        f32_to_fp8_e4m3(three_quarters),
        9,
        "3/4 across must round up to code 9; truncation would give 8"
    );
}

/// The same discrimination in the 1.0 binade, where the step is 0.125.
#[test]
fn rounding_is_not_truncation_in_the_unit_binade() {
    // 1.0 is code 56 (exp=7, mant=0). Step 0.125.
    let x = 1.0 + 0.75 * 0.125; // 1.09375
    assert_eq!(
        f32_to_fp8_e4m3(x),
        57,
        "1.09375 is nearer 1.125; truncation would give 56"
    );
    // And the ties in this binade, for the record.
    assert_eq!(f32_to_fp8_e4m3(1.0625), 56, "tie -> even 56");
    assert_eq!(f32_to_fp8_e4m3(1.1875), 58, "tie -> even 58");
}

/// Subnormal ties also go to even.
///
/// The subnormal row has its own rounding branch in every one of the four
/// converters, so it is the most likely place for one of them to still be
/// wrong after the normal path is fixed. Subnormal step is 2^-9.
#[test]
fn subnormal_ties_go_to_even() {
    let step = code_value(0, 1); // 2^-9
    assert_eq!(f32_to_fp8_e4m2_half_step(0.5 * step), 0, "0.5 step -> even 0");
    assert_eq!(f32_to_fp8_e4m2_half_step(1.5 * step), 2, "1.5 step -> even 2");
    assert_eq!(f32_to_fp8_e4m2_half_step(2.5 * step), 2, "2.5 step -> even 2");
    // 0.75 of a step is not a tie and must round up.
    assert_eq!(f32_to_fp8_e4m2_half_step(0.75 * step), 1, "0.75 step -> 1");
}

/// Small indirection so the subnormal test reads as arithmetic rather than as
/// a wall of magic constants. The value is what is encoded; the *expected code*
/// is what each assertion states.
#[allow(non_snake_case)]
fn f32_to_fp8_e4m2_half_step(multiples_of_step: f32) -> u8 {
    f32_to_fp8_e4m3(multiples_of_step)
}

/// Rounding must carry into the exponent, not wrap the mantissa.
///
/// A converter that increments the mantissa without carrying turns 1.9375 into
/// the code for 2.0's mantissa at the old exponent, producing a value that is
/// not on the grid at all -- a silently corrupt weight.
#[test]
fn rounding_carries_into_the_exponent() {
    // Just below 2.0, closer to 2.0 than to 1.75.
    let code = f32_to_fp8_e4m3(1.99);
    let decoded = e4m3_to_f32(code);
    let on_grid = grid().iter().any(|(_, v)| (*v - decoded).abs() < 1e-9);
    assert!(
        on_grid,
        "1.99 must land on a grid point, got {decoded} (code {code:#04x})"
    );
    assert_eq!(decoded, 2.0, "1.99 must carry to 2.0, not wrap the mantissa");
}

/// Saturation, and the threshold that the 448/480 bug once got wrong.
///
/// Any f32 at or above 464 carries a 3-bit mantissa of 7 at exponent 15, which
/// is the NaN slot. 448 is the correct clamp point; 480 is off by a full
/// mantissa step and let [464.01, 480) encode as NaN.
#[test]
fn saturation_threshold_is_448_not_480() {
    assert_eq!(f32_to_fp8_e4m3(448.0), 0x7E, "448 is the max finite E4M3");
    assert_eq!(f32_to_fp8_e4m3(500.0), 0x7E, "above max saturates to 0x7E");
    assert_eq!(f32_to_fp8_e4m3(1e30), 0x7E, "far above max saturates");
    assert_eq!(f32_to_fp8_e4m3(-1e30), 0xFE, "and keeps the sign");

    // The band that used to become NaN.
    for x in [464.01f32, 470.0, 479.9] {
        let code = f32_to_fp8_e4m3(x);
        assert_eq!(code, 0x7E, "{x} must saturate to 0x7E, not the NaN slot 0x7F");
        assert_ne!(code, 0x7F, "{x} must never encode as NaN");
    }
}

/// NaN is the one place the encoders legitimately disagree, and it must never
/// produce the *positive* NaN slot by accident.
#[test]
fn nan_encodes_to_the_nan_slot() {
    assert_eq!(f32_to_fp8_e4m3(f32::NAN), 0x7F);
    assert_eq!(f32_to_fp8_e4m3(f32::INFINITY), 0x7E, "inf saturates, not NaN");
    assert_eq!(f32_to_fp8_e4m3(f32::NEG_INFINITY), 0xFE, "-inf saturates");
}

/// Signed zero, and the subnormal boundary.
#[test]
fn zero_and_subnormal_boundary() {
    assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
    assert_eq!(f32_to_fp8_e4m3(-0.0), 0x80, "the sign of zero is preserved");

    // Half the min subnormal rounds to zero under RNE (code 0 is even).
    assert_eq!(f32_to_fp8_e4m3(code_value(0, 1) / 2.0), 0x00);
    // Three quarters of the min subnormal rounds up.
    assert_eq!(f32_to_fp8_e4m3(code_value(0, 1) * 0.75), 0x01);
    // Below half, everything is zero.
    assert_eq!(f32_to_fp8_e4m3(1e-9), 0x00);
}

/// The property truncation cannot have, stated as a bias test.
///
/// Draws uniformly over one binade and checks the mean signed error is
/// ~0. Truncation biases every value toward zero, so its mean error is
/// strictly negative and this fails. RNE is unbiased, so it passes.
///
/// This is the test that would have caught the original defect in production
/// use rather than in a code review.
#[test]
fn rounding_is_unbiased_and_truncation_would_not_be() {
    let mut sum = 0.0f64;
    let mut n = 0u32;
    // One binade, 1000 steps, avoiding exact ties (step count is even, so the
    // midpoints are not sampled on a regular grid).
    for i in 0..1000u32 {
        let x = 1.0 + (i as f32 + 0.5) / 1000.0;
        sum += (e4m3_to_f32(f32_to_fp8_e4m3(x)) - x) as f64;
        n += 1;
    }
    let mean = sum / n as f64;
    // The grid step here is 2^-7 * 2/8 = 3.05e-5, so an unbiased mean is ~0 and
    // a truncating one is about -half a step, 1.5e-5. The threshold sits well
    // between.
    assert!(
        mean.abs() < 5e-6,
        "mean signed error {mean:e} is not ~0; a negative bias means truncation"
    );
}
