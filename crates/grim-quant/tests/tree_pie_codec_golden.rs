//! Golden oracle for TreePie's E2M2 <-> FP16 pathway. CPU only.
//!
//! Covers the codec half of WS-A. The packer (A2) and the round trip (A3) live
//! in `tree_pie_pack.rs`; the GPU dot is A6 and is not reachable from here.
//!
//! # Why this file exists in this shape
//!
//! The FPE2M2 paper is internally inconsistent about the dequant constant, and
//! S3 in the plan records that as a blocker: the text says `S = (-1)^s * 2^15`
//! while the pseudocode uses `0x3c00`, which as an FP16 is **1.0**. Neither is
//! copied here. Both halves of the oracle are re-derived from the ExMy
//! definition, so this test pins a *derivation* rather than a constant
//! somebody transcribed.
//!
//! Byte-exactness is the point, exactly as in `golden_nutcracker_oracle.rs`. A
//! tolerance-based round trip would absorb a re-biasing of the exponent field
//! or an off-by-one in the subnormal ramp, which are the two mistakes most
//! likely here.
//!
//! # The derivation, stated once
//!
//! E2M2 is `sign(1) | exp(2) | mant(2)`, **bias 0**, and has no Inf/NaN -- the
//! encodings those would occupy are reclaimed as ordinary numbers. Applying the
//! standard IEEE subnormal rule with `b = 0`:
//!
//! ```text
//! e == 0 : value = (m / 2^M) * 2^(1-b) = (m / 4) * 2^1 = m / 2
//! e >= 1 : value = (1 + m/4) * 2^e
//! ```
//!
//! Note the subnormal scale factor is `2^(1-b) = 2`, **not** `1`. Dropping it
//! yields `0, .25, .5, .75, 2, 2.5, ...` -- a 2.67x hole between 0.75 and 2.0
//! and a grid no one would ship. With the factor the 16 magnitudes are
//!
//! ```text
//! 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14
//! ```
//!
//! which is continuous: the step doubles at each power of two, so no encoding
//! is wasted. Range 28:1.
//!
//! FP16 has a 5-bit exponent with bias 15, so the bias injection is
//! `exp_field = e + 15` and the 2-bit mantissa lands at bit offset 10 as
//! `m * 256` (2 mantissa bits scaled up to FP16's 10) -- i.e. the E2M2
//! mantissa drives exactly bits 8-9 of the FP16 mantissa, which is the
//! shift/AND/OR the format is chosen for. The `e == 0` row happens to be
//! *normal* in FP16 (0.5, 1.0, 1.5 all are), so it needs no special case in
//! the target format.

use grim_quant::tree_pie;

/// The 16 positive E2M2 magnitudes, derived from the ExMy definition.
///
/// Index is `(e << 2) | m`. Written out rather than computed in the test body
/// so a reader can check the grid against the format definition by eye, and so
/// the test still says something if the implementation's arithmetic changes.
const POSITIVE_GRID: [f32; 16] = [
    0.0,   // e=0 m=0  : +0
    0.5,   // e=0 m=1  : subnormal, (1/4)*2
    1.0,   // e=0 m=2  : subnormal, (2/4)*2
    1.5,   // e=0 m=3  : subnormal, (3/4)*2
    2.0,   // e=1 m=0  : 1.0 * 2^1
    2.5,   // e=1 m=1  : 1.25 * 2^1
    3.0,   // e=1 m=2  : 1.5 * 2^1
    3.5,   // e=1 m=3  : 1.75 * 2^1
    4.0,   // e=2 m=0
    5.0,   // e=2 m=1
    6.0,   // e=2 m=2
    7.0,   // e=2 m=3
    8.0,   // e=3 m=0
    10.0,  // e=3 m=1
    12.0,  // e=3 m=2
    14.0,  // e=3 m=3
];

/// Independently construct the FP16 bit pattern whose value is exactly `v`.
///
/// Built from `ilogb`/mantissa extraction rather than by calling anything in
/// grim, so agreement with the implementation is evidence and not tautology.
fn fp16_bits_of_exact(v: f32) -> u16 {
    assert!(v >= 0.0 && v.is_finite(), "oracle only handles finite non-negative");
    if v == 0.0 {
        return 0;
    }
    // v == 2^e * (1 + f) with 0 <= f < 1
    let e = v.log2().floor() as i32;
    let mantissa = (v / (2f32).powi(e) - 1.0) * 1024.0;
    let exp_field = e + 15;
    assert!(
        (0..32).contains(&exp_field),
        "v={v} needs exp_field {exp_field}, outside FP16's normal range"
    );
    assert!(
        (mantissa - mantissa.round()).abs() < 1e-3,
        "v={v} is not exactly representable in FP10 mantissa (mantissa {mantissa})"
    );
    (((exp_field as u16) << 10) | (mantissa.round() as u16 & 0x3FF)) & 0x7FFF
}

/// The expected FP16 bit pattern for E2M2 code `idx` (0..16, sign excluded).
///
/// This is the S3 resolution, expressed as code: take the magnitude from the
/// derived grid, then let [`fp16_bits_of_exact`] place it. The constant the
/// paper quotes is never involved.
fn expected_fp16(idx: u8) -> u16 {
    fp16_bits_of_exact(POSITIVE_GRID[idx as usize])
}

/// S3's RED, made GREEN: every one of the 32 encodings decodes to the exact
/// FP16 pattern the derivation requires.
#[test]
fn decode_matches_derived_oracle() {
    for idx in 0..16u8 {
        for sign in [0u8, 1u8] {
            let code = (sign << 4) | idx;
            let got = tree_pie::e2m2_to_fp16_bits(code);
            let want = expected_fp16(idx) | ((sign as u16) << 15);
            assert_eq!(
                got, want,
                "code {code:#04x} (idx {idx}, sign {sign}) decoded to {got:#06x}, \
                 derivation requires {want:#06x} for {}",
                POSITIVE_GRID[idx as usize]
            );
        }
    }
}

/// The grid itself, as a standalone check that the format is what the plan says.
///
/// E2M2 with bias 0 and 2 mantissa bits: the exponent runs 0..3 with no Inf, so
/// the largest magnitude is `(1 + 3/4) * 2^3 = 14`.
#[test]
fn the_e2m2_grid_is_the_one_the_format_defines() {
    // Largest: (1 + 3/4) * 2^3
    assert_eq!(POSITIVE_GRID[15], 14.0);
    // Smallest positive: subnormal (1/4)*2^1 = 0.5
    assert_eq!(POSITIVE_GRID[1], 0.5);
    // No Inf and no NaN encodings: 16 magnitudes, all finite, all distinct.
    assert_eq!(POSITIVE_GRID.iter().filter(|v| v.is_finite()).count(), 16);
    let mut sorted = POSITIVE_GRID;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(sorted.windows(2).all(|w| w[0] < w[1]), "grid must be strictly increasing");
    // Range 28:1.
    assert_eq!(POSITIVE_GRID[15] / POSITIVE_GRID[1], 28.0);
    // The property that distinguishes the correct subnormal rule from the
    // tempting wrong one: no encoding is wasted, i.e. the grid is *additively*
    // uniform within each exponent row. (Ratios are deliberately NOT uniform --
    // they vary from 4/3 to 1 in a float format, which is the whole point of it
    // being a float format.)
    //
    // Step size per row is 0.5 for e<=1, 1.0 for e=2, 2.0 for e=3.
    for (row, expected_step) in [(0usize, 0.5f32), (1, 0.5), (2, 1.0), (3, 2.0)] {
        for m in 0..3 {
            let lo = POSITIVE_GRID[row * 4 + m];
            let hi = POSITIVE_GRID[row * 4 + m + 1];
            assert!(
                (hi - lo - expected_step).abs() < 1e-6,
                "row e={row} step {m} to {}: {lo} -> {hi} should differ by {expected_step}",
                m + 1
            );
        }
    }
    // Cross-row boundaries must not skip a step either. 1.5 -> 2.0 is .5 (e0 to
    // e1) and 3.5 -> 4.0 is .5 (e1 to e2, i.e. *less* than e2's own 1.0 step),
    // both legal. A dropped `2^(1-b)` factor would make 0.75 -> 2.0 skip 1.25.
    assert!((POSITIVE_GRID[8] - POSITIVE_GRID[7] - 0.5).abs() < 1e-6);
    // Explicitly: the hole a wrong subnormal rule would introduce, and which
    // this grid does not have.
    assert_eq!(POSITIVE_GRID[3], 1.5);
    assert_eq!(POSITIVE_GRID[4], 2.0);
    assert!((POSITIVE_GRID[4] - POSITIVE_GRID[3] - 0.5).abs() < 1e-6);
}

/// S3's kill criterion, in the form that is actually checkable on CPU: the
/// decode must be bit-exact, because a tolerance-based decode would mean the
/// format is not reproducible and the A/B could not compare it against anything.
///
/// The paper reports round-trip error under 0.3%; this asserts the stronger
/// property that the grid is reproduced exactly, which is what makes that
/// number meaningful.
#[test]
fn the_decode_is_bit_exact_not_approximate() {
    for idx in 0..16u8 {
        let bits = tree_pie::e2m2_to_fp16_bits(idx);
        let back = f16_to_f32(bits);
        assert_eq!(
            back, POSITIVE_GRID[idx as usize],
            "idx {idx} must round-trip exactly, not approximately"
        );
    }
}

/// Minimal f16 -> f32 for the oracle's own use, so the test does not lean on
/// whatever fp16 helper the crate happens to expose (and cannot accidentally
/// test the implementation against itself).
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x3FF) as f32;
    if exp == 0 {
        sign * mant * (1.0 / 1024.0) * (2f32).powi(-14)
    } else {
        sign * (1.0 + mant / 1024.0) * (2f32).powi(exp - 15)
    }
}

/// f32 -> E2M2, round-to-nearest-even against the same derived grid.
///
/// The quantizer has to break ties toward even or the round trip is not
/// reproducible: a value exactly between two grid points would otherwise land
/// on whichever the comparison operator happened to prefer, and the two choices
/// differ by a full mantissa step.
#[test]
fn f32_to_e2m2_selects_the_nearest_grid_point_with_rne() {
    for idx in 0..16u8 {
        let v = POSITIVE_GRID[idx as usize];
        assert_eq!(
            tree_pie::f32_to_e2m2(v),
            idx,
            "grid point {v} (idx {idx}) must encode to itself"
        );
    }
    // Midpoints go to the even neighbour, in both directions.
    let mid_0_1 = 0.25f32; // between 0.0 (m=0, even) and 0.5 (m=1, odd)
    assert_eq!(tree_pie::f32_to_e2m2(mid_0_1), 0, "0.25 ties to even m=0");
    let mid_1_2 = 0.75f32; // between 0.5 (m=1, odd) and 1.0 (m=2, even)
    assert_eq!(tree_pie::f32_to_e2m2(mid_1_2), 2, "0.75 ties to even m=2");
    let mid_2_3 = 1.25f32; // between 1.0 (m=2, even) and 1.5 (m=3, odd)
    assert_eq!(tree_pie::f32_to_e2m2(mid_2_3), 2, "1.25 ties to even m=2");
    let mid_3_4 = 1.75f32; // between 1.5 (e0 m3) and 2.0 (e1 m0) -- the
                            // subnormal/normal boundary, where the even
                            // neighbour is m=0 of e=1
    assert_eq!(tree_pie::f32_to_e2m2(mid_3_4), 4, "1.75 ties to even m=0 of e=1");
    // Above the top of the grid, saturate rather than wrap. Wrapping would turn
    // an outlier into a wrong sign, which is the one failure a quantizer must
    // never have.
    assert_eq!(tree_pie::f32_to_e2m2(1000.0), 15, "saturate to the top");
    assert_eq!(tree_pie::f32_to_e2m2(-1000.0), 0x1F, "saturate, keeping the sign");
}

/// Negative zero is a real encodable value and must survive the round trip;
/// collapsing it to +0 would make a signed-zero-sensitive consumer see a
/// different number.
#[test]
fn negative_zero_round_trips() {
    let code = tree_pie::f32_to_e2m2(-0.0);
    assert_eq!(code & 0x0F, 0, "must be a zero magnitude");
    assert_eq!(code & 0x10, 0x10, "sign bit must be set");
    assert_eq!(tree_pie::e2m2_to_fp16_bits(code), 0x8000, "decodes to FP16 -0");
}
