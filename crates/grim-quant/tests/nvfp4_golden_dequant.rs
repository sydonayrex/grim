//! Mutation-resistant golden tests for `dequant_nvfp4` and
//! `reframe_nvfp4_to_mxfp4`.
//!
//! `nvfp4_roundtrip.rs` does not exercise either function despite its name —
//! it round-trips `quant_fp4_block16`/`dequant_fp4_block16`, an unrelated
//! Jay-tier codec. This file constructs NVFP4 byte buffers **by hand**,
//! computes expected dequant values from the OCP E2M1 spec **independently**
//! (not by calling any of grim's own `f32_to_mxfp4_e2m1` / `quant_*`
//! functions, so an encode/decode bug can't cancel itself out), and asserts
//! exact expected values — following the convention in `golden_dequant.rs`.
//!
//! # OCP E2M1 codebook (derived from spec, 1 sign / 2 exp / 1 mantissa bit)
//! code -> |value|: 0->0.0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
//! (exp==0 is subnormal: value = mantissa * 0.5; exp!=0: (1 + mantissa*0.5) * 2^(exp-1))
//! Sign bit (code bit 3) negates. Final value is codebook value * 2^(e8m0_byte - 127).
//!
//! # NVFP4 packing (per grim's `dequant_nvfp4` doc comment)
//! Per 256-value super-block (144 bytes): 16 sub-blocks of 16 values each.
//! Per sub-block: 1 byte E8M0 shared exponent, then 8 bytes of packed E2M1
//! codes (2 per byte, low nibble = even index, high nibble = odd index).

use grim_quant::{dequant_mxfp4, dequant_nvfp4, reframe_nvfp4_to_mxfp4};

/// f32 comparison: bit-exact treated as exact, otherwise tight absolute
/// tolerance (values here are small hand-picked powers of two, so no
/// legitimate rounding should ever require a loose bound).
fn assert_close(got: f32, want: f32, ctx: &str) {
    let diff = (got - want).abs();
    assert!(
        diff < 1e-4,
        "{ctx}: got {got}, want {want} (diff {diff})"
    );
}

/// Independent oracle for the OCP E2M1 codebook, transcribed directly from
/// the spec — must NOT call grim's `mxfp4_e2m1_to_f32`.
fn oracle_e2m1(code: u8) -> f32 {
    let sign = (code >> 3) & 1 != 0;
    let exp = (code >> 1) & 3;
    let mant = (code & 1) as f32;
    let base = if exp == 0 {
        mant * 0.5
    } else {
        (1.0 + mant * 0.5) * 2f32.powi(exp as i32 - 1)
    };
    if sign { -base } else { base }
}

/// Independent oracle for E8M0 shared-exponent scale.
fn oracle_e8m0_scale(byte: u8) -> f32 {
    2f32.powi(byte as i32 - 127)
}

fn oracle_nvfp4_value(code: u8, exp_byte: u8) -> f32 {
    oracle_e2m1(code) * oracle_e8m0_scale(exp_byte)
}

/// Hand-pack one 16-value NVFP4 sub-block: 1 exponent byte + 8 code bytes
/// (2 nibbles per byte, low nibble first).
fn pack_subblock(exp_byte: u8, codes16: &[u8; 16]) -> Vec<u8> {
    let mut out = vec![exp_byte];
    for pair in codes16.chunks(2) {
        out.push(pair[0] | (pair[1] << 4));
    }
    out
}

// ===========================================================================
// dequant_nvfp4 — direct path. This is the function that actually ships
// (toolkit ingestion routes ModelOpt NVFP4 tensors through it via
// grim-format's `toolkit_to_storage`), so it gets the most scrutiny.
// ===========================================================================

#[test]
fn dequant_nvfp4_single_subblock_uniform_scale() {
    // Sub-block 0: exponent byte 127 (scale = 2^0 = 1.0), codes = [2,2,...]
    // (E2M1 code 2 -> codebook value 1.0), so every value should decode to
    // exactly 1.0. Remaining 15 sub-blocks of the super-block are all-zero
    // (exp=127, code=0) to keep the fixture legible.
    let mut data = pack_subblock(127, &[2u8; 16]);
    for _ in 0..15 {
        data.extend(pack_subblock(127, &[0u8; 16]));
    }
    assert_eq!(data.len(), 144);

    let out = dequant_nvfp4(&data, 256).expect("nvfp4 dequant");
    assert_eq!(out.len(), 256);
    for (i, &v) in out.iter().take(16).enumerate() {
        assert_close(v, 1.0, &format!("nvfp4 sub-block 0 elem {i}"));
    }
    for (i, &v) in out.iter().skip(16).enumerate() {
        assert_close(v, 0.0, &format!("nvfp4 zero-block elem {i}"));
    }
}

#[test]
fn dequant_nvfp4_distinguishes_per_subblock_exponents() {
    // Sub-block 0: exp=127 (scale 1.0), code=2 (codebook 1.0) -> values 1.0
    // Sub-block 1: exp=128 (scale 2.0), code=2 (codebook 1.0) -> values 2.0
    // These two sub-blocks fall in the SAME 32-element MXFP4 group, which is
    // exactly the boundary `reframe_nvfp4_to_mxfp4` mishandles below — this
    // test confirms the direct dequant path keeps them distinct.
    let mut data = pack_subblock(127, &[2u8; 16]);
    data.extend(pack_subblock(128, &[2u8; 16]));
    for _ in 0..14 {
        data.extend(pack_subblock(127, &[0u8; 16]));
    }
    assert_eq!(data.len(), 144);

    let out = dequant_nvfp4(&data, 256).expect("nvfp4 dequant");
    for (i, &v) in out.iter().take(16).enumerate() {
        assert_close(v, 1.0, &format!("nvfp4 sub-block 0 (elem {i})"));
    }
    for (i, &v) in out.iter().skip(16).take(16).enumerate() {
        assert_close(v, 2.0, &format!("nvfp4 sub-block 1 (elem {i})"));
    }
}

#[test]
fn dequant_nvfp4_matches_independent_oracle_random_fixture() {
    // Deterministic pseudo-random fixture (fixed LCG, not `rand`, to avoid a
    // new dev-dependency) covering all 16 sub-blocks with varied exponents
    // and codes, compared against the from-spec oracle rather than any of
    // grim's own encode functions.
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state
    };

    let mut data = Vec::with_capacity(144);
    let mut oracle = Vec::with_capacity(256);
    for _ in 0..16 {
        let exp_byte = (100 + (next() % 51)) as u8; // 100..=150
        let mut codes = [0u8; 16];
        for c in codes.iter_mut() {
            *c = (next() % 16) as u8;
        }
        data.extend(pack_subblock(exp_byte, &codes));
        for &c in &codes {
            oracle.push(oracle_nvfp4_value(c, exp_byte));
        }
    }
    assert_eq!(data.len(), 144);
    assert_eq!(oracle.len(), 256);

    let out = dequant_nvfp4(&data, 256).expect("nvfp4 dequant");
    for (i, (&got, &want)) in out.iter().zip(oracle.iter()).enumerate() {
        assert_close(got, want, &format!("nvfp4 random fixture elem {i}"));
    }
}

// ===========================================================================
// reframe_nvfp4_to_mxfp4 — the GPU-kernel bridge. This function maps two
// 16-element NVFP4 sub-blocks onto one 32-element MXFP4 group. That mapping
// is only lossless when both sub-blocks share the same E8M0 exponent. When
// they differ, the function must return an error rather than silently
// dropping one exponent (which would produce a 2x scaling error).
// ===========================================================================

#[test]
fn reframe_nvfp4_to_mxfp4_rejects_mismatched_exponents() {
    // Sub-block 0: exp=127 (scale 1.0), code=2 -> true value 1.0
    // Sub-block 1: exp=128 (scale 2.0), code=2 -> true value 2.0
    // Adjacent sub-blocks have different exponents, so lossless reframing to
    // MXFP4's 32-element groups is impossible. The function must error.
    let mut data = pack_subblock(127, &[2u8; 16]);
    data.extend(pack_subblock(128, &[2u8; 16]));
    for _ in 0..14 {
        data.extend(pack_subblock(127, &[0u8; 16]));
    }
    assert_eq!(data.len(), 144);

    let result = reframe_nvfp4_to_mxfp4(&data, 256);
    assert!(
        result.is_err(),
        "reframe_nvfp4_to_mxfp4 must return Err when adjacent sub-blocks have \
         different E8M0 exponents, got Ok"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("different E8M0 exponents"),
        "error message should mention mismatched exponents, got: {err_msg}"
    );
}

#[test]
fn reframe_nvfp4_to_mxfp4_succeeds_when_exponents_match() {
    // All sub-blocks share the same exponent (127 = scale 1.0), so the
    // reframing is lossless. Verify the reframed buffer decodes correctly
    // through dequant_mxfp4.
    let mut data = Vec::with_capacity(144);
    for sb in 0..16 {
        let code = (sb % 8) as u8; // varied codes, same exponent
        data.extend(pack_subblock(127, &[code; 16]));
    }
    assert_eq!(data.len(), 144);

    let reframed = reframe_nvfp4_to_mxfp4(&data, 256).expect("reframe should succeed");
    let out = dequant_mxfp4(&reframed, 256).expect("mxfp4 dequant");

    for (sb, chunk) in out.chunks(16).enumerate() {
        let code = (sb % 8) as u8;
        let expected = oracle_nvfp4_value(code, 127);
        for (i, &v) in chunk.iter().enumerate() {
            assert_close(v, expected, &format!("sub-block {sb} elem {i}"));
        }
    }
}
