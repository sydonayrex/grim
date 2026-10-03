//! ForestRaven codec: per-row absmax INT8 round-trip and framing validation.
//!
//! The happy path (quantize -> dequantize == int8 round of source) is
//! exercised end to end through `grim convert --format forestraven` in
//! `grim_container_whiteraven.rs`. This file pins the codec contract itself:
//! the absmax math per the article recipe, the all-zero-row edge (scale 1.0,
//! codes all zero -- no division by zero, no NaN scale), the clamp at both
//! ends, and the framing validation that makes truncated or mis-sized buffers
//! refuse instead of decoding a prefix.

use grim_quant::{dequant_forest, quant_forest_per_channel};

#[test]
fn per_row_scales_are_absmax_over_127() {
    // Two rows with different ranges: each row must use its OWN grid, so the
    // small row keeps full precision instead of inheriting the large row's
    // coarse scale. This is the whole point of per-channel over per-tensor.
    let w = vec![
        1.27, -1.27, 0.0, 0.635, // row 0: absmax 1.27 -> scale 0.01
        0.0127, -0.006, 0.0, 0.0127, // row 1: absmax 0.0127 -> scale 0.0001
    ];
    let (codes, scales) = quant_forest_per_channel(&w, 2, 4).expect("quant");
    assert_eq!(codes.len(), 8);
    assert_eq!(scales.len(), 8);

    let s0 = f32::from_le_bytes(scales[0..4].try_into().unwrap());
    let s1 = f32::from_le_bytes(scales[4..8].try_into().unwrap());
    assert_eq!(s0.to_bits(), (1.27f32 / 127.0).to_bits());
    assert_eq!(s1.to_bits(), (0.0127f32 / 127.0).to_bits());

    // Row 0 codes: 127, -127, 0, 63 (0.635/0.01 = 63.5 -> 64? No: 63.5 rounds
    // to 64 banker's? Rust round() rounds half away from zero -> 64).
    // Compute from the same formula rather than hardcoding the float print.
    let expect0: Vec<i8> = [1.27, -1.27, 0.0, 0.635]
        .iter()
        .map(|&v| (v / s0).round().clamp(-128.0, 127.0) as i8)
        .collect();
    let got0: Vec<i8> = codes[0..4].iter().map(|&b| b as i8).collect();
    assert_eq!(got0, expect0);
    // Row 1 uses its own fine grid: 0.0127 -> 127, not 1.
    assert_eq!(codes[4] as i8, 127, "small row must use its own scale");
    assert_eq!(codes[7] as i8, 127);
}

#[test]
fn all_zero_row_gets_unit_scale_and_zero_codes() {
    let w = vec![0.0f32; 8];
    let (codes, scales) = quant_forest_per_channel(&w, 2, 4).expect("quant");
    assert!(codes.iter().all(|&b| b == 0));
    for r in 0..2 {
        let s = f32::from_le_bytes(scales[r * 4..(r + 1) * 4].try_into().unwrap());
        assert_eq!(s.to_bits(), 1.0f32.to_bits(), "zero row scale must be exactly 1.0");
    }
    // And it decodes back to +0.0, not NaN.
    let mut blob = Vec::new();
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&(scales.len() as u64).to_le_bytes());
    blob.extend_from_slice(&scales);
    let deq = dequant_forest(&blob, 2, 4).expect("dequant");
    assert!(deq.iter().all(|&v| v.to_bits() == 0.0f32.to_bits()));
}

#[test]
fn codes_clamp_at_both_ends_without_wrapping() {
    // Under absmax the most-negative code is exactly -127 (-amax/scale), so
    // the -128 clamp end is defensive-only. What matters: negatives survive
    // the u8 storage round-trip with their sign -- a naive `as u8` cast
    // without the i8 intermediate would wrap them into large positives.
    let w = vec![10.0, -10.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0];
    let (codes, _) = quant_forest_per_channel(&w, 2, 4).expect("quant");
    assert_eq!(codes[0] as i8, 127);
    assert_eq!(codes[1] as i8, -127);
    assert_eq!(codes[0], 127u8);
    assert_eq!(codes[1], 129u8, "i8 -127 is u8 129");
    // And decoding the stored bytes recovers the sign.
    assert!((codes[1] as i8) < 0);
}

#[test]
fn framing_validation_refuses_every_truncation_shape() {
    let w: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
    let (codes, scales) = quant_forest_per_channel(&w, 4, 8).expect("quant");
    let mut blob = Vec::new();
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&(scales.len() as u64).to_le_bytes());
    blob.extend_from_slice(&scales);

    // Happy path first: valid blob decodes.
    let deq = dequant_forest(&blob, 4, 8).expect("valid blob must decode");
    assert_eq!(deq.len(), 32);

    // Every truncation shape refuses. A decoder that accepted a prefix would
    // rescale whole rows by whatever bytes follow -- finite, plausible, wrong.
    for len in [0, 7, 8, 8 + 16, blob.len() - 1] {
        assert!(
            dequant_forest(&blob[..len], 4, 8).is_err(),
            "truncated blob of {len} bytes must refuse"
        );
    }
    // Wrong geometry against a valid blob refuses: the codes length names n*k.
    assert!(dequant_forest(&blob, 2, 8).is_err());
    assert!(dequant_forest(&blob, 4, 4).is_err());
    // Trailing bytes refuse: slack past the scales segment is either a
    // writer bug or a spliced file, never data.
    let mut long = blob.clone();
    long.push(0);
    assert!(dequant_forest(&long, 4, 8).is_err());
    // Corrupted length prefix refuses rather than over-reading.
    let mut bad = blob.clone();
    bad[0] = 0xFF;
    assert!(dequant_forest(&bad, 4, 8).is_err());
}
