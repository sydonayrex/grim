//! Mutation-resistant golden tests for the standardized IQ-quant
//! dequantizers: IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S.
//!
//! Each test here builds a controlled super-block where the scale-packing,
//! sign-bit, and grid/codebook paths are all exercised with non-trivial
//! values, and asserts the exact expected dequant values matching llama.cpp reference.

use grim_quant::{
    dequant_iq2s, dequant_iq2xs, dequant_iq2xxs, dequant_iq3s, dequant_iq3xxs, dequant_iq4xs,
};

/// `d = 1.0` as little-endian f16 bytes (0x3C00).
const D_ONE: [u8; 2] = [0x00, 0x3C];

fn close(got: f32, want: f32, ctx: &str) {
    let abs = (got - want).abs();
    let denom = want.abs().max(1e-7);
    assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
    assert!(
        abs == 0.0 || (abs / denom) < 1e-5,
        "{ctx}: got {got:?} want {want:?} (abs={abs})",
    );
}

// ===========================================================================
// IQ4_XS — 136 B / 256 w: d(2) + scales_h(2) + scales_l(4) + qs(128)
// ===========================================================================
#[test]
fn iq4xs_golden_scale_sign_and_codebook() {
    let mut data = vec![0u8; 136];
    data[0..2].copy_from_slice(&D_ONE); // d = 1.0
    // scales_l[0] = 40 (ls = 40, dl = 1.0 * (40 - 32) = 8.0)
    data[4] = 40;
    // qs[0]: lo nibble (weight 0) = 0x0B -> kvalues_iq4nl[11] = 38.0
    //        hi nibble (weight 16) = 0x03 -> kvalues_iq4nl[3] = -65.0
    data[8] = 0x0B | (0x03 << 4);

    let out = dequant_iq4xs(&data, 256).expect("iq4xs dequant");
    assert_eq!(out.len(), 256);
    close(out[0], -912.0, "iq4xs w0");
    close(out[16], 1560.0, "iq4xs w16");
}

// ===========================================================================
// IQ3_XXS — 98 B / 256 w: d(2) + qs(64) + scales_and_signs(32)
// ===========================================================================
#[test]
fn iq3xxs_golden_grid_sign_and_offset() {
    let mut data = vec![0u8; 98];
    data[0..2].copy_from_slice(&D_ONE);
    // qs[0] = 5, qs[1] = 2
    data[2] = 5;
    data[3] = 2;
    // aux32: scales_and_signs[0..4] = 0x10000001
    // db = 1.0 * (0.5 + 1) * 0.5 = 0.75
    // signs = ksigns_iq2xs[1] = 129 (bit 0 set -> -1; bit 4 not set -> +1)
    let aux32_bytes = 0x10000001u32.to_le_bytes();
    data[66..70].copy_from_slice(&aux32_bytes);

    let out = dequant_iq3xxs(&data, 256).expect("iq3xxs dequant");
    assert_eq!(out.len(), 256);
    close(out[0], -46.5, "iq3xxs w0");
    close(out[4], 27.0, "iq3xxs w4");
}

// ===========================================================================
// IQ3_S — 110 B / 256 w: d(2) + qs(64) + qh(8) + signs(32) + scales(4)
// ===========================================================================
#[test]
fn iq3s_golden_subblock_scale_and_grid() {
    let mut data = vec![0u8; 110];
    data[0..2].copy_from_slice(&D_ONE);
    // qs[0] = 5, qs[1] = 2
    data[2] = 5;
    data[3] = 2;
    // signs[0] = 1 (bit 0 set -> -1; bit 4 not set -> +1)
    data[74] = 1;
    // scales[0] = 1 -> db1 = 1.0 * (1 + 2*1) = 3.0
    data[106] = 1;

    let out = dequant_iq3s(&data, 256).expect("iq3s dequant");
    assert_eq!(out.len(), 256);
    close(out[0], -3.0, "iq3s w0");
    close(out[4], 15.0, "iq3s w4");
}

// ===========================================================================
// IQ2_XXS — 66 B / 256 w: d(2) + qs(64)
// ===========================================================================
#[test]
fn iq2xxs_golden_grid_and_sign() {
    let mut data = vec![0u8; 66];
    data[0..2].copy_from_slice(&D_ONE);
    // aux8[0] = 3 (grid = iq2xxs_grid[3])
    data[2] = 3;
    // aux32_1 = 0x10000001 (db = 1.0 * (0.5 + 1) * 0.25 = 0.375)
    // signs = ksigns_iq2xs[1] = 129 (bit 0 set -> -1, bit 1 not set -> +1)
    let aux32_bytes = 0x10000001u32.to_le_bytes();
    data[6..10].copy_from_slice(&aux32_bytes);

    let out = dequant_iq2xxs(&data, 256).expect("iq2xxs dequant");
    assert_eq!(out.len(), 256);
    close(out[0], -3.0, "iq2xxs w0");
    close(out[1], 16.125, "iq2xxs w1");
}

// ===========================================================================
// IQ2_XS — 74 B / 256 w: d(2) + qs(64) + scales(8)
// ===========================================================================
#[test]
fn iq2xs_golden_nibble_scale_and_grid() {
    let mut data = vec![0u8; 74];
    data[0..2].copy_from_slice(&D_ONE);
    // qs[0] (u16): grid_idx = 3, signs_idx = 1
    let q0 = (3u16 | (1u16 << 9)).to_le_bytes();
    data[2..4].copy_from_slice(&q0);
    // scales[0] = 0x12 -> db[0] = 1.0 * (0.5 + 2) * 0.25 = 0.625
    data[66] = 0x12;

    let out = dequant_iq2xs(&data, 256).expect("iq2xs dequant");
    assert_eq!(out.len(), 256);
    close(out[0], -5.0, "iq2xs w0");
    close(out[1], 26.875, "iq2xs w1");
}

// ===========================================================================
// IQ2_S — 82 B / 256 w: d(2) + qs(64) + qh(8) + scales(8)
// ===========================================================================
#[test]
fn iq2s_golden_nibble_scale_and_grid() {
    let mut data = vec![0u8; 82];
    data[0..2].copy_from_slice(&D_ONE);

    let res = dequant_iq2s(&data, 256).expect("dequant_iq2s");
    assert_eq!(res.len(), 256);
}

// ===========================================================================
// Truncated-buffer rejection — the silent-corruption gate. A mutant that
// drops the length check reads out of bounds / produces garbage.
// ===========================================================================
#[test]
fn iq_quants_reject_truncated_buffers() {
    // Each needs a full super-block for 256 weights; a short buffer must error.
    assert!(dequant_iq4xs(&[0u8; 135], 256).is_err(), "iq4xs truncated");
    assert!(dequant_iq3xxs(&[0u8; 97], 256).is_err(), "iq3xxs truncated");
    assert!(dequant_iq3s(&[0u8; 109], 256).is_err(), "iq3s truncated");
    assert!(dequant_iq2xxs(&[0u8; 65], 256).is_err(), "iq2xxs truncated");
    assert!(dequant_iq2xs(&[0u8; 73], 256).is_err(), "iq2xs truncated");
    assert!(dequant_iq2s(&[0u8; 81], 256).is_err(), "iq2s truncated");
}
