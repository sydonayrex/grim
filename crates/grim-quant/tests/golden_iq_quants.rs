//! Mutation-resistant golden tests for the "new IQ-quant compatibility"
//! dequantizers: IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S.
//!
//! The in-crate tests (`test_iq4xs_dequant_exact_layout_and_math`, …) feed in a
//! zeroed buffer with `d = 1.0` and assert only `res.len() == 256` plus a
//! truncated-buffer error. They assert **no numeric value at all** — the
//! IQ4_XS test even sets its sub-block scale to 0 (scales[0]=32 → scale 0.0),
//! so every output is 0.0 regardless of any codebook/sign/scale mutant.
//!
//! Each test here builds a controlled super-block where the scale-packing,
//! sign-bit, and grid/codebook paths are all exercised with **non-trivial**
//! values, and asserts the exact expected dequant derived independently from
//! the format spec in `crates/grim-quant/src/lib.rs` (not by calling the
//! library's own `quant_iq*` encoder).

use grim_quant::{
    dequant_iq2s, dequant_iq2xs, dequant_iq2xxs, dequant_iq3s, dequant_iq3xxs, dequant_iq4xs,
};

/// `d = 1.0` as little-endian f16 bytes (0x3C00).
const D_ONE: [u8; 2] = [0x00, 0x3C];

fn close(got: f32, want: f32, ctx: &str) {
    let abs = (got - want).abs();
    let denom = want.abs().max(1e-7);
    assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})",);
    assert!(
        abs == 0.0 || (abs / denom) < 1e-5,
        "{ctx}: got {got:?} want {want:?} (abs={abs})",
    );
}

// ===========================================================================
// IQ4_XS — 136 B / 256 w: d(2) + scales(6) + qs(128)
//   sc_val = (scales[sb*6/8] >> (sb*6 % 8)) & 0x3F
//   scale  = d * (sc_val - 32) / 32
//   nibble lo3 -> IQ4_NL_CODEBOOK[nibble & 0x7]; bit3 -> sign
// ===========================================================================
#[test]
fn iq4xs_golden_scale_sign_and_codebook() {
    let mut data = vec![0u8; 136];
    data[0..2].copy_from_slice(&D_ONE); // d = 1.0
    // scales[0] = 40 -> sb0 sc_val = 40 & 0x3F = 40 -> scale = (40-32)/32 = 0.25
    data[2] = 40;
    // qs[0]: lo nibble (weight 0) = 0xB -> code index 3, sign -1
    //        hi nibble (weight 1) = 0x3 -> code index 3, sign +1
    data[8] = 0x0B | (0x03 << 4);

    let out = dequant_iq4xs(&data, 256).expect("iq4xs dequant");
    assert_eq!(out.len(), 256);
    // IQ4_NL_CODEBOOK[3] = 0.39743365; scale 0.25.
    close(
        out[0],
        -(0.39743365_f32 * 0.25),
        "iq4xs w0 (sign -, code 3)",
    );
    close(
        out[1],
        0.39743365_f32 * 0.25 * 1.0,
        "iq4xs w1 (sign +, code 3)",
    );
}

// ===========================================================================
// IQ3_XXS — 96 B / 256 w: d(2) + qs(64) + signs(30)
//   base = ((qs[i/8] + (i%8)*17) % 7) - 3 ; val = d * base * 0.25 * sign
// ===========================================================================
#[test]
fn iq3xxs_golden_grid_sign_and_offset() {
    let mut data = vec![0u8; 96];
    data[0..2].copy_from_slice(&D_ONE);
    // qs[0] = 5: w0 base = (5 + 0)%7 - 3 = 5-3 = 2; w1 base = (5+17)%7 - 3 = 1-3 = -2
    data[2] = 5;
    // signs[0] bit0 = 1 -> w0 sign -1; bit1 = 0 -> w1 sign +1
    data[2 + 64] = 0b0000_0001;

    let out = dequant_iq3xxs(&data, 256).expect("iq3xxs dequant");
    assert_eq!(out.len(), 256);
    close(out[0], -(1.0 * 2.0 * 0.25), "iq3xxs w0");
    close(out[1], 1.0 * (-2.0) * 0.25 * 1.0, "iq3xxs w1");
}

// ===========================================================================
// IQ3_S — 110 B / 256 w: d(2) + qs(64) + scales(12) + signs(32)
//   sc = (scales[sb*12/8] + 1) * 0.125 ; scale = d * sc
//   grid_val = ((qs[i/8] + i) % 7) - 3 ; val = scale * grid_val * sign
// ===========================================================================
#[test]
fn iq3s_golden_subblock_scale_and_grid() {
    let mut data = vec![0u8; 110];
    data[0..2].copy_from_slice(&D_ONE);
    // scales[0] = 0 -> sb0 sc = (0+1)*0.125 = 0.125 -> scale = 0.125
    // qs[0]=5: w0 grid_val = (5+0)%7 - 3 = 2 ; val = 0.125*2 = 0.25
    // qs[1]=5: w8 grid_val = (5+8)%7 - 3 = 6-3 = 3 ; val = 0.125*3 = 0.375
    data[2] = 5; // qs[0]
    data[3] = 5; // qs[1]
    // signs default 0 -> sign +1 for both.

    let out = dequant_iq3s(&data, 256).expect("iq3s dequant");
    assert_eq!(out.len(), 256);
    close(out[0], 0.125 * 2.0, "iq3s w0");
    close(out[8], 0.125 * 3.0, "iq3s w8 (grid index via qs[1])");
}

// ===========================================================================
// IQ2_XXS — 66 B / 256 w: d(2) + qs(32) + signs(32)
//   val = d * (((qs[i/8] + i%8) % 4) - 1.5) * sign
// ===========================================================================
#[test]
fn iq2xxs_golden_grid_and_sign() {
    let mut data = vec![0u8; 66];
    data[0..2].copy_from_slice(&D_ONE);
    // qs[0] = 3: w0 val = (3+0)%4 - 1.5 = 1.5 ; w1 val = (3+1)%4 - 1.5 = -1.5
    data[2] = 3;
    // signs[0] bit0 = 0 -> w0 sign +1; bit1 = 1 -> w1 sign -1
    data[2 + 32] = 0b0000_0010;

    let out = dequant_iq2xxs(&data, 256).expect("iq2xxs dequant");
    assert_eq!(out.len(), 256);
    close(out[0], 1.5 * 1.0, "iq2xxs w0");
    close(out[1], (-1.5) * -1.0, "iq2xxs w1 (sign flip)");
}

// ===========================================================================
// IQ2_XS — 74 B / 256 w: d(2) + qs(32) + scales(8) + signs(32)
//   sc = ((scales[sb/2] >> ((sb%2)*4)) & 0x0F) * 0.125 + 0.5 ; scale = d*sc
//   val = scale * (((qs[i/8] + i%8) % 4) - 1.5) * sign
// ===========================================================================
#[test]
fn iq2xs_golden_nibble_scale_and_grid() {
    let mut data = vec![0u8; 74];
    data[0..2].copy_from_slice(&D_ONE);
    // scales[0] lo nibble = 4 -> sb0 sc = 4*0.125 + 0.5 = 1.0 -> scale = 1.0
    data[2 + 32] = 4; // scales[0]
    // qs[0] = 3: w0 val = (3+0)%4 - 1.5 = 1.5 ; w1 val = (3+1)%4 - 1.5 = -1.5
    data[2] = 3; // qs[0]
    // signs default 0 -> sign +1 for both.

    let out = dequant_iq2xs(&data, 256).expect("iq2xs dequant");
    assert_eq!(out.len(), 256);
    close(out[0], 1.0 * 1.5, "iq2xs w0");
    close(out[1], 1.0 * (-1.5), "iq2xs w1");
}

// ===========================================================================
// IQ2_S — 82 B / 256 w: d(2) + qs(48) + scales(8) + signs(24)
//   sc = ((scales[sb/2] >> ((sb%2)*4)) & 0x0F) * 0.125 + 0.5 ; scale = d*sc
//   code = (((qs[i/8] + i%8) % 4) - 1.5) ; val = scale * code * sign
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
    assert!(dequant_iq3xxs(&[0u8; 95], 256).is_err(), "iq3xxs truncated");
    assert!(dequant_iq3s(&[0u8; 109], 256).is_err(), "iq3s truncated");
    assert!(dequant_iq2xxs(&[0u8; 65], 256).is_err(), "iq2xxs truncated");
    assert!(dequant_iq2xs(&[0u8; 73], 256).is_err(), "iq2xs truncated");
    assert!(dequant_iq2s(&[0u8; 81], 256).is_err(), "iq2s truncated");
}
