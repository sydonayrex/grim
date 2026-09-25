//! Golden tests for the two 9-bytes-per-16 formats: `dequant_nvfp4` (E4M3
//! block scale, zero code is a real zero) and `dequant_nutcracker`
//! (`[exp:6|sel:2]` block scale, zero code emits the block's special value).
//!
//! Both formats share a byte layout but not a decode, which is precisely why
//! these tests build buffers by hand against an independent oracle rather than
//! round-tripping through the packers. The failure this guards against is
//! real: grim previously decoded GGUF type-78 (true NVFP4) with an E8M0 scale.
//! Because E4M3 and E8M0 are both exactly 1 byte per 16 elements, every length
//! and allocation check passed and only the values were wrong.
//!
//! Independent oracles, not derived from the implementation:
//!   E2M1  = { 0, .5, 1, 1.5, 2, 3, 4, 6 } with sign-magnitude, bias 1
//!   E4M3  = OCP: bias 7, 3 mantissa bits, exp 0xF/mant 7 is NaN, max 448
//!   E8M0  = 2^(byte - 127)
//!   Nutcracker scale byte = [ exp:6 | sel:2 ], scale = 2^(exp - 31)

fn assert_close(got: f32, want: f32, ctx: &str) {
    let tol = 1e-5 * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol,
        "{ctx}: got {got}, want {want} (tol {tol})"
    );
}

/// OCP E2M1 codebook, decoded independently of the implementation.
fn oracle_e2m1(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if code & 0x8 != 0 { -1.0 } else { 1.0 };
    sign * MAG[(code & 0x7) as usize]
}

/// E8M0: a bare unsigned power of two.
fn oracle_e8m0_scale(byte: u8) -> f32 {
    2.0f32.powi(byte as i32 - 127)
}

/// OCP E4M3 ("FN") -> f32, written from the spec rather than ported.
///
/// The FN variant has **no infinities**: `exp == 0xF` is a normal binade
/// spanning [256, 448], and only `0x7F` / `0xFF` (mant == 7) are NaN. So
/// `0x78` is 256, not 448 — clamping the whole binade to 448 would be the
/// E4M3FN "all-ones-is-max" mistake.
fn oracle_e4m3_scale(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as f32;
    if exp == 0x0F {
        return if (b & 0x07) == 0x07 {
            f32::NAN
        } else {
            sign * (1.0 + mant / 8.0) * 256.0
        };
    }
    if exp == 0 {
        sign * (mant / 8.0) * 2.0f32.powi(-6)
    } else {
        sign * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
    }
}

fn oracle_nutcracker_value(code: u8, scale_byte: u8) -> f32 {
    let sel = scale_byte & 0x3;
    let scale = 2.0f32.powi((scale_byte >> 2) as i32 - 31);
    if code & 0x7 == 0 {
        let mag = if sel & 0x1 != 0 { 2.5 } else { 5.0 };
        return if sel & 0x2 != 0 { -mag } else { mag } * scale;
    }
    oracle_e2m1(code) * scale
}

/// Pack one 16-value sub-block: scale byte then 8 code bytes (low nibble = even).
fn pack_subblock(scale_byte: u8, codes: &[u8; 16]) -> Vec<u8> {
    let mut out = vec![scale_byte];
    for pair in codes.chunks(2) {
        out.push((pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4));
    }
    out
}

use grim_quant::{dequant_nutcracker, dequant_nvfp4};

fn all_codes() -> [u8; 16] {
    let mut c = [0u8; 16];
    for (i, slot) in c.iter_mut().enumerate() {
        *slot = i as u8;
    }
    c
}

// ── Nutcracker ────────────────────────────────────────────────────────────

#[test]
fn nutcracker_decodes_a_uniform_scale_subblock() {
    // exp = 31 -> scale 1.0, sel 0 -> special +5.0.
    let scale_byte = (31u8 << 2) | 0;
    let data = pack_subblock(scale_byte, &all_codes());
    let out = dequant_nutcracker(&data, 16).expect("nutcracker dequant");
    for (i, &code) in all_codes().iter().enumerate() {
        assert_close(
            out[i],
            oracle_nutcracker_value(code, scale_byte),
            &format!("code {code}"),
        );
    }
}

#[test]
fn nutcracker_distinguishes_per_subblock_exponents() {
    let codes = all_codes();
    let mut data = pack_subblock(30u8 << 2, &codes);
    data.extend(pack_subblock(32u8 << 2, &codes));
    let out = dequant_nutcracker(&data, 32).expect("nutcracker dequant");
    for (i, &code) in codes.iter().enumerate() {
        assert_close(
            out[i],
            oracle_nutcracker_value(code, 30u8.wrapping_shl(2)),
            "sub-block 0",
        );
        assert_close(
            out[16 + i],
            oracle_nutcracker_value(code, 32u8.wrapping_shl(2)),
            "sub-block 1",
        );
    }
}

#[test]
fn nutcracker_matches_independent_oracle_random_fixture() {
    // LCG so the fixture is deterministic and reviewable.
    let mut seed: u32 = 0x2545_F491;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    let codes: Vec<u8> = (0..256).map(|_| (next() & 0x0F) as u8).collect();
    let mut data = Vec::with_capacity(144);
    let mut want = Vec::with_capacity(256);
    for blk in 0..16 {
        let scale_byte = (next() & 0xFF) as u8;
        let c: [u8; 16] = codes[blk * 16..blk * 16 + 16].try_into().unwrap();
        data.extend(pack_subblock(scale_byte, &c));
        for &code in &c {
            want.push(oracle_nutcracker_value(code, scale_byte));
        }
    }
    let out = dequant_nutcracker(&data, 256).expect("nutcracker dequant");
    for (i, (&g, &w)) in out.iter().zip(want.iter()).enumerate() {
        assert_close(g, w, &format!("random fixture elem {i}"));
    }
}

#[test]
fn nutcracker_round_trips_through_the_packer() {
    let data: Vec<f32> = (0..512)
        .map(|i| (i as f32 / 511.0) * 9.0 - 4.5 + 0.2 * ((i % 5) as f32 - 2.0))
        .collect();
    let packed = grim_quant::quant_nutcracker(&data).expect("pack");
    let out = dequant_nutcracker(&packed, data.len()).expect("dequant");
    // Every value must be finite and within one E2M1 grid step of the original.
    let mut worst = 0.0f32;
    for (a, b) in data.iter().zip(out.iter()) {
        assert!(b.is_finite(), "non-finite reconstruction");
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst < 0.6,
        "worst absolute error {worst} exceeds one grid step"
    );
}

// ── NVFP4 (the real format) ───────────────────────────────────────────────

#[test]
fn nvfp4_decodes_e4m3_scales() {
    // E4M3 0x38 = 1.0, 0x3C = 1.5, 0x40 = 2.0.
    for (byte, want) in [(0x38u8, 1.0f32), (0x3C, 1.5), (0x40, 2.0)] {
        let data = pack_subblock(byte, &all_codes());
        let out = dequant_nvfp4(&data, 16).expect("nvfp4 dequant");
        for (i, &code) in all_codes().iter().enumerate() {
            assert_close(
                out[i],
                oracle_e2m1(code) * want,
                &format!("E4M3 {byte:#04x} code {code}"),
            );
        }
    }
}

#[test]
fn nvfp4_keeps_a_real_zero() {
    // The load-bearing difference from Nutcracker.
    let data = pack_subblock(0x38, &all_codes());
    let out = dequant_nvfp4(&data, 16).expect("nvfp4 dequant");
    assert_eq!(out[0], 0.0, "code 0x0 must decode to zero");
    assert_eq!(out[8], 0.0, "code 0x8 must decode to zero");
    assert!(!out.iter().any(|v| *v == 5.0), "NVFP4 has no special value");
}

#[test]
fn nvfp4_never_misreads_its_scale_as_e8m0() {
    // The regression that motivated a correct NVFP4: an E4M3 scale byte read
    // as E8M0 collapses the whole block to ~0 while every length check passes.
    let data = pack_subblock(0x3C, &all_codes()); // 1.5 in E4M3
    let out = dequant_nvfp4(&data, 16).expect("nvfp4 dequant");
    // 2^(0x3C - 127) is ~6.8e-21, so a correct decode (~1.5 * the E2M1
    // magnitude) exceeds the E8M0 misread by ~20 orders of magnitude.
    let e8m0_misread = oracle_e8m0_scale(0x3C);
    for (i, &code) in all_codes().iter().enumerate() {
        let mag = oracle_e2m1(code);
        let want = mag * 1.5;
        assert_close(out[i], want, &format!("code {code}"));
        if mag != 0.0 {
            // The misread is ~20 orders of magnitude below the correct value.
            // Compare absolute magnitudes so negative codes work too.
            let misread = mag * e8m0_misread;
            assert!(
                out[i].abs() / misread.abs() > 1e15,
                "code {code}: got {}, E8M0 misread would be {misread:e} (ratio {})",
                out[i],
                out[i].abs() / misread.abs()
            );
        }
    }
}

#[test]
fn nvfp4_and_nutcracker_disagree_on_the_same_buffer() {
    // Guards the scheme mix-up: same bytes, different values.
    let data = pack_subblock(
        0x38,
        &[
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x00,
        ],
    );
    let nv = dequant_nvfp4(&data, 16).expect("nvfp4");
    let nu = dequant_nutcracker(&data, 16).expect("nutcracker");
    assert_ne!(nv, nu, "the two decoders must not coincide");
    assert_eq!(nv[0], 0.5, "NVFP4 code 0x1 is +0.5");
    // 0x38 as Nutcracker = exp 14, sel 0 -> 2^(14-31).
    assert_close(nu[0], 0.5 * 2.0f32.powi(14 - 31), "nutcracker code 0x1");
}

#[test]
fn nvfp4_matches_independent_oracle_random_fixture() {
    let mut seed: u32 = 0x9E37_79B9;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    let mut data = Vec::with_capacity(144);
    let mut want = Vec::with_capacity(256);
    for _ in 0..16 {
        let scale_byte = (next() & 0xFF) as u8;
        let codes: [u8; 16] = std::array::from_fn(|_| (next() & 0x0F) as u8);
        // Skip the two E4M3 NaN encodings (0x7F/0xFF): their sub-block scale is
        // undefined, so a buffer containing one is malformed, not a fixture.
        if matches!(scale_byte, 0x7F | 0xFF) {
            continue;
        }
        data.extend(pack_subblock(scale_byte, &codes));
        let s = oracle_e4m3_scale(scale_byte);
        for &code in &codes {
            want.push(oracle_e2m1(code) * s);
        }
    }
    let n = want.len();
    assert!(n > 0 && n % 16 == 0, "fixture built {n} elements");
    let out = dequant_nvfp4(&data, n).expect("nvfp4 dequant");
    for (i, (&g, &w)) in out.iter().zip(want.iter()).enumerate() {
        assert_close(g, w, &format!("random fixture elem {i}"));
    }
}

#[test]
fn both_decoders_reject_truncated_buffers() {
    assert!(dequant_nvfp4(&[0u8; 8], 16).is_err());
    assert!(dequant_nutcracker(&[0u8; 8], 16).is_err());
}

#[test]
fn both_decoders_handle_empty_and_partial() {
    assert!(dequant_nvfp4(&[], 0).unwrap().is_empty());
    assert!(dequant_nutcracker(&[], 0).unwrap().is_empty());
    let data = pack_subblock(0x38, &all_codes());
    assert_eq!(dequant_nvfp4(&data, 10).unwrap().len(), 10);
    assert_eq!(dequant_nutcracker(&data, 10).unwrap().len(), 10);
}
