//! Golden oracle test for the Nutcracker numerical pathway, CPU only.
//!
//! Complements `nvfp4_nutcracker_dequant.rs` (which covers the *decode* side
//! against independent oracles). This file covers the full pathway in both
//! directions, at byte-exactness:
//!
//!   f32 --[quant_nutcracker]--> packed bytes --[dequant_nutcracker]--> f32
//!
//! The oracle below is a from-scratch reimplementation of both halves, written
//! from the format description rather than by porting grim's code. Byte-exact
//! comparison is the point: it catches a wire-format change, a re-biasing of
//! the exponent field, a swapped nibble order, or a changed selector table that
//! a tolerance-based round-trip test would silently absorb.
//!
//! # Format (recap)
//!
//! Per 16 values, 9 bytes:
//!   byte 0      : [ exp : 6 | sel : 2 ]   scale = 2^(exp - 31)
//!   bytes 1..=8 : 16 E2M1 codes, low nibble = even index
//!
//! `code & 0x7 == 0` (the two zero encodings) decodes to the block's special
//! value with the sign taken from the selector, not from the code's sign bit:
//!   sel 0 -> +5.0, 1 -> +2.5, 2 -> -5.0, 3 -> -2.5
//!
//! Packer: exponent is `ceil(log2(max/6)) + 31` so the block's largest
//! magnitude fits just inside E2M1's 6.0 ceiling; then all 4 selectors are
//! swept and the lowest squared-error one wins.

use grim_quant::{
    NUTCRACKER_SCALE_BIAS, NUTCRACKER_SEL_BITS, NUTCRACKER_SEL_MASK, NUTCRACKER_SUB_BLOCK,
    NUTCRACKER_SUB_BLOCK_BYTES, dequant_nutcracker, nutcracker_e2m1_to_f32, quant_nutcracker,
};

fn assert_eq_f32(got: f32, want: f32, ctx: &str) {
    // Bit-exact: both sides are exact powers-of-two scalings of a small
    // codebook, so any divergence is a real logic difference, not rounding.
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{ctx}: got {got:?}, want {want:?} (bits {:08x} vs {:08x})",
        got.to_bits(),
        want.to_bits()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Oracle — independent of the implementation
// ═══════════════════════════════════════════════════════════════════════════

/// OCP E2M1 magnitudes, sign-magnitude, exponent bias 1.
const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// OCP E2M1 codebook as the *format* defines it: codes with `code & 0x07 == 0`
/// are the repurposed zero and do NOT decode to E2M1's ±0 — they emit the
/// block's special value, with the sign taken from the selector. So this
/// helper is only meaningful for `code & 0x07 != 0`; see [`oracle_decode`].
fn oracle_e2m1_nonzero(code: u8) -> f32 {
    let mag = E2M1_MAG[(code & 0x07) as usize];
    if code & 0x08 != 0 { -mag } else { mag }
}

/// The four special values, indexed by selector.
fn oracle_special(sel: u8) -> f32 {
    match sel & 0x03 {
        0 => 5.0,
        1 => 2.5,
        2 => -5.0,
        _ => -2.5,
    }
}

fn oracle_scale(byte: u8) -> f32 {
    2.0f32.powi((byte >> NUTCRACKER_SEL_BITS) as i32 - NUTCRACKER_SCALE_BIAS)
}

/// Oracle decoder for one code within one sub-block.
fn oracle_decode(code: u8, scale_byte: u8) -> f32 {
    if code & 0x07 == 0 {
        oracle_special(scale_byte) * oracle_scale(scale_byte)
    } else {
        oracle_e2m1_nonzero(code) * oracle_scale(scale_byte)
    }
}

/// Nearest codeword to `v` (already in unscaled units) for a given selector.
///
/// Mirrors the implementation's search exactly: `0x0` is the seeded candidate
/// and codes `1..16` are compared with a strict `<`, so ties resolve toward the
/// lower code. Code `0x8` is skipped — it decodes to the same special value as
/// `0x0` and is a redundant duplicate.
fn oracle_nearest(v: f32, sel: u8) -> u8 {
    let mut best = 0u8;
    let mut best_d = (v - oracle_special(sel)).abs();
    // Skip 0x0 and 0x8: both decode to the special value, so 0x0 (the lower
    // code) is the only representative and it is already the seeded candidate.
    for code in 1u8..16 {
        if code & 0x07 == 0 {
            continue;
        }
        let d = (v - oracle_e2m1_nonzero(code)).abs();
        if d < best_d {
            best_d = d;
            best = code;
        }
    }
    best
}

/// Oracle packer for a single 16-value block. Returns the 9 bytes.
fn oracle_pack_block(block: &[f32]) -> Vec<u8> {
    let max_abs = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let exp: i32 = if max_abs == 0.0 {
        NUTCRACKER_SCALE_BIAS
    } else {
        let raw = (max_abs / 6.0).log2() + NUTCRACKER_SCALE_BIAS as f32;
        raw.ceil() as i32
    };
    let scale = 2.0f32.powi(exp - NUTCRACKER_SCALE_BIAS);

    // Sweep all selectors with a strict `<` so ties keep the lowest selector,
    // matching the implementation's `err < best_err`.
    let mut best_sel = 0u8;
    let mut best_err = f32::MAX;
    let mut best_codes: Vec<u8> = Vec::new();
    for sel in 0..4u8 {
        let codes: Vec<u8> = block
            .iter()
            .map(|&v| oracle_nearest(v / scale, sel))
            .collect();
        let mut err = 0.0f32;
        for (&v, &c) in block.iter().zip(codes.iter()) {
            let recon = oracle_decode(c, ((exp as u8) << NUTCRACKER_SEL_BITS) | sel);
            let d = v - recon;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_sel = sel;
            best_codes = codes;
        }
    }
    let (sel, codes) = (best_sel, best_codes);

    let mut out = vec![((exp as u8) << NUTCRACKER_SEL_BITS) | (sel & NUTCRACKER_SEL_MASK)];
    // Always a whole sub-block: 1 scale byte + 8 code bytes, even for a partial
    // tail. Slots past the block length are padding and are never decoded, so
    // the high nibble is zero-filled.
    for byte in 0..8 {
        let lo = codes.get(byte * 2).map(|c| c & 0x0F).unwrap_or(0);
        let hi = codes.get(byte * 2 + 1).map(|c| c & 0x0F).unwrap_or(0);
        out.push(lo | (hi << 4));
    }
    out
}

fn oracle_pack(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for block in data.chunks(NUTCRACKER_SUB_BLOCK) {
        out.extend_from_slice(&oracle_pack_block(block));
    }
    out
}

/// Deterministic pseudo-random f32 stream, so fixtures are reviewable and
/// stable across runs and machines.
fn lcg_stream(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Map to (-8, 8) with a couple of octaves of structure.
        let base = ((s >> 8) % 16_000) as f32 / 1000.0 - 8.0;
        let wobble = ((i % 13) as f32 - 6.0) * 0.11;
        out.push(base + wobble);
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Golden byte vectors — literal expected wire bytes
// ═══════════════════════════════════════════════════════════════════════════

/// Packing a hand-built block produces the expected wire bytes.
///
/// Inputs are chosen to be exactly representable, so the packer is forced to
/// emit a specific code per slot and the byte layout is pinned:
///
///   - index 0  -> +5.0  -> code 0x0 (the block special value at sel 0)
///   - index 1  -> +1.0  -> code 0x2
///   - index 2  -> +1.5  -> code 0x3
///   - ...
///   - index 7  -> +6.0  -> code 0x7
///   - index 8  -> -0.5  -> code 0x9
///   - ...
///   - index 15 ->  0.0  -> code 0x8 (E2M1 -0.0 is nearer than the +5.0 special)
///
/// Low nibble holds the even index, high nibble the odd. Block max is 6.0, so
/// the exponent must come out as 31 (scale 1.0) and the sweep picks sel 0 (the
/// special value is what makes index 0 exact).
#[test]
fn golden_scale_byte_and_code_packing_order() {
    // Every one of the 16 slots is assigned explicitly. The mapping is
    //   idx  0 -> +5.0   code 0x0  (the sel-0 special value)
    //   idx  1 -> +1.0   code 0x2
    //   idx  2 -> +1.5   code 0x3
    //   idx  3 -> +2.0   code 0x4
    //   idx  4 -> +3.0   code 0x5
    //   idx  5 -> +4.0   code 0x6
    //   idx  6 -> +6.0   code 0x7  (top E2M1 codeword)
    //   idx  7 ->  0.0   code 0x1  (nearest real codeword; no zero code exists)
    //   idx  8 -> -0.5   code 0x9
    //   idx  9 -> -1.0   code 0xA
    //   idx 10 -> -1.5   code 0xB
    //   idx 11 -> -2.0   code 0xC
    //   idx 12 -> -3.0   code 0xD
    //   idx 13 -> -4.0   code 0xE
    //   idx 14 -> -6.0   code 0xF
    //   idx 15 -> -2.0   code 0xC
    let inputs = vec![
        oracle_special(0),
        oracle_e2m1_nonzero(0x02),
        oracle_e2m1_nonzero(0x03),
        oracle_e2m1_nonzero(0x04),
        oracle_e2m1_nonzero(0x05),
        oracle_e2m1_nonzero(0x06),
        oracle_e2m1_nonzero(0x07),
        0.0,
        oracle_e2m1_nonzero(0x09),
        oracle_e2m1_nonzero(0x0A),
        oracle_e2m1_nonzero(0x0B),
        oracle_e2m1_nonzero(0x0C),
        oracle_e2m1_nonzero(0x0D),
        oracle_e2m1_nonzero(0x0E),
        oracle_e2m1_nonzero(0x0F),
        oracle_e2m1_nonzero(0x0C),
    ];
    assert_eq!(inputs.len(), NUTCRACKER_SUB_BLOCK);

    let packed = quant_nutcracker(&inputs).expect("pack");
    let expected = oracle_pack(&inputs);
    assert_eq!(
        &packed[..9],
        &expected[..],
        "wire bytes for a hand-built block"
    );

    // Pin the layout explicitly: byte 0 is the scale, then low-nibble-first pairs.
    // packed[0] is the scale byte; packed[1..9] are the 8 code bytes, each
    // holding two codes with the EVEN index in the LOW nibble.
    assert_eq!(
        packed[0] >> NUTCRACKER_SEL_BITS,
        31,
        "exp must be 31 (scale 1.0)"
    );
    assert_eq!(packed[0] & NUTCRACKER_SEL_MASK, 0, "sel must be 0");

    // Pairing consecutive indices low-nibble-first. Every input is exactly
    // representable, so the packer is forced to emit these codes:
    //   (0,1) 0x0,0x2 | (2,3) 0x3,0x4 | (4,5) 0x5,0x6 | (6,7) 0x7,0x1
    //   (8,9) 0x9,0xA | (10,11) 0xB,0xC | (12,13) 0xD,0xE | (14,15) 0xF,0xC
    // which is the byte sequence 20 43 65 17 A9 CB ED CF after the scale byte.
    // 0x7C = (31 << 2) | 0: exponent 31 (scale 1.0), selector 0 (+5.0).
    let expected_codes: [u8; 9] = [0x7C, 0x20, 0x43, 0x65, 0x17, 0xA9, 0xCB, 0xED, 0xCF];
    assert_eq!(
        &packed[..9],
        &expected_codes[..],
        "wire bytes: scale, then low-nibble-even code pairs"
    );
}

#[test]
fn golden_special_values_are_exactly_four() {
    // Pin the selector table. Changing it changes what every stored block
    // decodes to, so it is part of the format, not an implementation detail.
    assert_eq!(oracle_special(0), 5.0);
    assert_eq!(oracle_special(1), 2.5);
    assert_eq!(oracle_special(2), -5.0);
    assert_eq!(oracle_special(3), -2.5);
    // All multiples of 0.5, so decoded values stay on the FP4 grid.
    for sel in 0..4u8 {
        let v = oracle_special(sel);
        assert_eq!(v * 2.0, (v * 2.0).round(), "sel {sel} off-grid");
    }
}

#[test]
fn golden_scale_field_boundaries() {
    // exp = 0 -> 2^-31, exp = 63 -> 2^32. The 6-bit field must not be biased
    // differently at either end.
    assert_eq!(oracle_scale(0x00), 2.0f32.powi(-31));
    assert_eq!(oracle_scale(0xFF & !0x03), 2.0f32.powi(63 - 31));
    // The selector occupies the low 2 bits and must not perturb the scale.
    for sel in 0..4u8 {
        assert_eq!(
            oracle_scale(0x80 | sel),
            oracle_scale(0x80),
            "sel perturbs scale"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Packer byte-parity against the oracle
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn packer_is_byte_identical_to_the_oracle_on_structured_fixtures() {
    let fixtures: Vec<(&str, Vec<f32>)> = vec![
        ("all_zero", vec![0.0; 64]),
        (
            "full_codebook",
            (0..32).map(|i| oracle_decode(i as u8, 0x38)).collect(),
        ),
        ("top_gap", (0..16).map(|i| 4.2 + i as f32 * 0.11).collect()),
        ("bottom_heavy", {
            let mut v: Vec<f32> = (0..15).map(|i| 0.1 + i as f32 * 0.1).collect();
            v.push(2.0);
            v
        }),
        (
            "negatives",
            (0..16).map(|i| -(i as f32) * 0.37 - 0.2).collect(),
        ),
        (
            "tiny_magnitudes",
            (0..16).map(|i| (i as f32) * 1e-5).collect(),
        ),
        (
            "large_magnitudes",
            (0..16).map(|i| (i as f32) * 1e4).collect(),
        ),
        ("single_value", vec![3.7]),
        ("partial_tail", (0..19).map(|i| (i as f32) * 0.29).collect()),
    ];

    for (name, data) in fixtures {
        let got = quant_nutcracker(&data).unwrap_or_else(|e| panic!("{name}: pack failed: {e}"));
        let want = oracle_pack(&data);
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: byte length ({} values)",
            data.len()
        );
        if got != want {
            let gi: Vec<String> = got.iter().map(|b| format!("{b:02X}")).collect();
            let gi_s: String = gi.iter().take(18).cloned().collect::<Vec<_>>().join(" ");
            let wi_s: String = want
                .iter()
                .take(18)
                .map(|b| format!("{b:02X}"))
                .collect::<Vec<_>>()
                .join(" ");
            let first = got
                .iter()
                .zip(want.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            let blk = first / 9;
            panic!(
                "{name}: packed bytes differ at byte {first} (block {blk}, byte-in-block {})\n  impl:   {gi_s}\n  oracle: {wi_s}",
                first % 9,
            );
        }
    }
}

#[test]
fn packer_is_byte_identical_to_the_oracle_on_random_fixtures() {
    for seed in [0x0000_0001u32, 0xDEAD_BEEF, 0x1234_5678, 0x9E37_79B9] {
        for &n in &[16usize, 32, 64, 128, 257] {
            let data = lcg_stream(n, seed);
            let got = quant_nutcracker(&data).expect("pack");
            let want = oracle_pack(&data);
            assert_eq!(got, want, "seed {seed:#x} n {n}: bytes differ from oracle");
        }
    }
}

/// The selector sweep must actually choose — a packer that always emitted
/// sel 0 would still byte-match a degenerate oracle, so assert the oracle
/// itself is non-inert.
#[test]
fn oracle_selector_sweep_varies_across_block_shapes() {
    let gap: Vec<f32> = (0..16).map(|i| 4.2 + i as f32 * 0.11).collect();
    let low: Vec<f32> = {
        let mut v: Vec<f32> = (0..15).map(|i| 0.1 + i as f32 * 0.1).collect();
        v.push(2.0);
        v
    };
    let gs = oracle_pack_block(&gap)[0] & NUTCRACKER_SEL_MASK;
    let ls = oracle_pack_block(&low)[0] & NUTCRACKER_SEL_MASK;
    assert_eq!(gs, 0, "top-gap block should select +5.0");
    assert_eq!(ls, 1, "bottom-heavy block should select +2.5");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Decoder parity against the oracle
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn decoder_matches_the_oracle_over_the_whole_byte_space() {
    // Every scale byte (all 256, covering every exp x sel combination) crossed
    // with every code. 4096 cases — the complete decode input space for one
    // sub-block.
    for scale_byte in 0u8..=255 {
        for code in 0u8..16 {
            let got = nutcracker_e2m1_to_f32(code, scale_byte);
            let want = oracle_decode(code, scale_byte);
            assert_eq_f32(
                got,
                want,
                &format!("code {code:#04x} scale {scale_byte:#04x}"),
            );
        }
    }
}

#[test]
fn dequant_matches_the_oracle_across_random_packed_buffers() {
    for seed in [1u32, 0xBEEF, 0xFACE, 0x0BAD_C0DE] {
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        };
        for &n in &[16usize, 48, 256] {
            let blocks = n.div_ceil(NUTCRACKER_SUB_BLOCK);
            let mut packed = Vec::with_capacity(blocks * NUTCRACKER_SUB_BLOCK_BYTES);
            let mut want = Vec::with_capacity(n);
            for _ in 0..blocks {
                let scale_byte = (next() & 0xFF) as u8;
                let codes: [u8; 16] = std::array::from_fn(|_| (next() & 0x0F) as u8);
                packed.push(scale_byte);
                for pair in codes.chunks(2) {
                    packed.push((pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4));
                }
                for &c in &codes {
                    want.push(oracle_decode(c, scale_byte));
                }
            }
            let got = dequant_nutcracker(&packed, n).expect("dequant");
            assert_eq!(got.len(), n);
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq_f32(g, w, &format!("seed {seed:#x} n {n} elem {i}"));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. End-to-end pathway parity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_pathway_is_bit_identical_to_the_oracle() {
    // value -> packer -> bytes -> dequant -> value, with the bytes checked
    // against the oracle and the reconstruction checked against the oracle's
    // own decode of those bytes.
    for seed in [7u32, 0xC0FFEE, 0x5EED] {
        for &n in &[16usize, 64, 200] {
            let data = lcg_stream(n, seed);
            let got_bytes = quant_nutcracker(&data).expect("pack");
            let want_bytes = oracle_pack(&data);
            if got_bytes != want_bytes {
                let g: String = got_bytes.iter().map(|b| format!("{b:02X}")).collect();
                let w: String = want_bytes.iter().map(|b| format!("{b:02X}")).collect();
                let first = got_bytes
                    .iter()
                    .zip(want_bytes.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                panic!(
                    "seed {seed:#x} n {n}: packer bytes differ at byte {first} \
                     (block {}, in-block {})\n  impl:   {g}\n  oracle: {w}",
                    first / 9,
                    first % 9
                );
            }

            let got = dequant_nutcracker(&got_bytes, n).expect("dequant");
            for (i, &g) in got.iter().enumerate() {
                let block = i / NUTCRACKER_SUB_BLOCK;
                let scale_byte = got_bytes[block * NUTCRACKER_SUB_BLOCK_BYTES];
                let local = i % NUTCRACKER_SUB_BLOCK;
                let code_byte = got_bytes[block * NUTCRACKER_SUB_BLOCK_BYTES + 1 + local / 2];
                let code = if local % 2 == 0 {
                    code_byte & 0x0F
                } else {
                    code_byte >> 4
                };
                assert_eq_f32(
                    g,
                    oracle_decode(code, scale_byte),
                    &format!("seed {seed:#x} n {n} elem {i}"),
                );
            }
        }
    }
}

#[test]
fn pathway_cannot_represent_an_exact_zero() {
    // Documented format limitation, asserted so a future "fix" is deliberate:
    // the zero code is repurposed, so ±0 never survives a round trip.
    let data = vec![0.0f32; 16];
    let packed = quant_nutcracker(&data).expect("pack");
    let rec = dequant_nutcracker(&packed, 16).expect("dequant");
    // A zero input has no exact-zero codeword available, so it lands on the
    // nearest real one: +0.5 (code 0x1) when the block's special value is +5.0.
    let nearest_real = oracle_e2m1_nonzero(0x01);
    for (i, &v) in rec.iter().enumerate() {
        assert_ne!(v, 0.0, "elem {i} decoded to an exact zero");
        assert_eq_f32(v, nearest_real, &format!("elem {i}"));
    }
}

#[test]
fn pathway_beats_a_bare_e2m1_grid() {
    // The format's claim: the per-block selector reduces error versus plain
    // per-16 E2M1 with no special value. Measured, not asserted by fiat.
    let data = lcg_stream(512, 0xABCD_1234);
    let packed = quant_nutcracker(&data).expect("pack");
    let rec = dequant_nutcracker(&packed, data.len()).expect("dequant");
    let nut_sse: f32 = data.iter().zip(&rec).map(|(a, b)| (a - b) * (a - b)).sum();

    // Baseline: same exponent choice, zero code mapped to a real zero.
    let mut base_sse = 0.0f32;
    for block in data.chunks(NUTCRACKER_SUB_BLOCK) {
        let max_abs = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let exp: i32 = if max_abs == 0.0 {
            NUTCRACKER_SCALE_BIAS
        } else {
            ((max_abs / 6.0).log2() + NUTCRACKER_SCALE_BIAS as f32).ceil() as i32
        };
        let scale = 2.0f32.powi(exp - NUTCRACKER_SCALE_BIAS);
        for &v in block {
            let u = v / scale;
            // Codebook without the special value; 0x0 is a genuine zero.
            let mut best = 0.0f32;
            let mut best_d = (u - 0.0).abs();
            for c in 1u8..16 {
                let val = oracle_e2m1_nonzero(c);
                let d = (u - val).abs();
                if d < best_d {
                    best_d = d;
                    best = val;
                }
            }
            let d = v - best * scale;
            base_sse += d * d;
        }
    }
    assert!(
        nut_sse < base_sse,
        "Nutcracker SSE {nut_sse} should beat the bare E2M1 baseline {base_sse}"
    );
}

#[test]
fn malformed_buffers_are_rejected_loudly() {
    // 32 values need 18 bytes; 17 must error rather than read out of bounds.
    let err = dequant_nutcracker(&[0u8; 17], 32).expect_err("short buffer must error");
    assert!(
        format!("{err}").contains("expected 18 bytes"),
        "unhelpful error: {err}"
    );
    assert!(dequant_nutcracker(&[], 0).expect("empty").is_empty());
    // A partial trailing sub-block is legal and yields exactly n values.
    let packed = oracle_pack(&lcg_stream(20, 3));
    assert_eq!(dequant_nutcracker(&packed, 20).expect("partial").len(), 20);
}
