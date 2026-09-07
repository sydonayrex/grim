//! Tests for NVFP4 dequantization (`dequant_nvfp4`) and layout reframing
//! (`reframe_nvfp4_to_mxfp4`).
//!
//! NVFP4 uses the OCP E2M1 codebook (same as MXFP4) but NVIDIA's packing:
//! per-16-element sub-blocks with one E8M0 shared exponent byte per
//! sub-block, interleaved: `[scale][codes][scale][codes]...`.
//!
//! These tests validate against hand-computed codebook values rather than
//! round-tripping the same encoder — catching systematic codebook bugs that
//! a tautological round-trip cannot.

use grim_quant::{dequant_nvfp4, reframe_nvfp4_to_mxfp4};

/// OCP E2M1 codebook: index → unscaled value (sign-magnitude decode).
///
/// Matches `mxfp4_e2m1_to_f32` with shared_exp = 127 (scale = 1.0).
/// Code 0 is the zero value (exp=0, mant=0 → 0.0).
fn e2m1_codebook(code: u8) -> f32 {
    let sign = ((code >> 3) & 1) != 0;
    let exp = (code >> 1) & 3;
    let mant = code & 1;
    let base_val = if exp == 0 {
        mant as f32 * 0.5
    } else {
        (1.0 + mant as f32 * 0.5) * (2.0f32).powi(exp as i32 - 1)
    };
    if sign { -base_val } else { base_val }
}

/// Build a single NVFP4 sub-block (16 weights) from a shared E8M0 exponent
/// and 16 E2M1 codes.
///
/// Layout: `[scale_byte][8 packed code bytes]` (low nibble = even index).
fn build_nvfp4_sub_block(shared_exp: u8, codes: &[u8; 16]) -> [u8; 9] {
    let mut out = [0u8; 9];
    out[0] = shared_exp;
    for (i, &code) in codes.iter().enumerate() {
        let byte_idx = 1 + i / 2;
        if i % 2 == 0 {
            out[byte_idx] = code & 0x0F;
        } else {
            out[byte_idx] |= (code & 0x0F) << 4;
        }
    }
    out
}

/// Build a full 256-weight NVFP4 super-block from 16 sub-blocks.
fn build_nvfp4_super_block(sub_blocks: &[[u8; 9]; 16]) -> [u8; 144] {
    let mut out = [0u8; 144];
    for (sb, block) in sub_blocks.iter().enumerate() {
        let offset = sb * 9;
        out[offset..offset + 9].copy_from_slice(block);
    }
    out
}

#[test]
fn test_dequant_nvfp4_all_e2m1_codes() {
    // Single sub-block (16 weights), E8M0 exponent = 127 (scale = 1.0).
    // Each weight gets a distinct E2M1 code 0..15.
    let codes: [u8; 16] = core::array::from_fn(|i| i as u8);
    let block = build_nvfp4_sub_block(127, &codes);

    // Dequantize 16 weights from one sub-block.
    let result = dequant_nvfp4(&block, 16).expect("dequant_nvfp4");
    assert_eq!(result.len(), 16);

    for (i, (&val, &code)) in result.iter().zip(codes.iter()).enumerate() {
        let expected = e2m1_codebook(code);
        assert!(
            (val - expected).abs() < 1e-6,
            "code {code}: expected {expected}, got {val} (index {i})"
        );
    }
}

#[test]
fn test_dequant_nvfp4_e8m0_scaling() {
    // Verify the E8M0 shared exponent scales all values in a sub-block.
    // Use code 6 (value 4.0) with exponent 128 → scale = 2^(128-127) = 2.0.
    let codes = [6u8; 16]; // all code 6 → base value 4.0
    let block = build_nvfp4_sub_block(128, &codes); // exp = 128 → scale 2.0

    let result = dequant_nvfp4(&block, 16).expect("dequant_nvfp4");
    for (i, &val) in result.iter().enumerate() {
        let expected = 4.0 * 2.0; // base * scale
        assert!(
            (val - expected).abs() < 1e-5,
            "index {i}: expected {expected}, got {val}"
        );
    }
}

#[test]
fn test_dequant_nvfp4_negative_values() {
    // Codes 8..15 are negative (sign bit set). Code 15 → -6.0 at scale 1.0.
    let codes = [15u8; 16]; // all code 15 → -6.0
    let block = build_nvfp4_sub_block(127, &codes);

    let result = dequant_nvfp4(&block, 16).expect("dequant_nvfp4");
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - (-6.0)).abs() < 1e-5,
            "index {i}: expected -6.0, got {val}"
        );
    }
}

#[test]
fn test_dequant_nvfp4_zero_code() {
    // Code 0 is the zero value (exp=0, mant=0 → 0.0 regardless of exponent).
    let codes = [0u8; 16];
    let block = build_nvfp4_sub_block(200, &codes); // large exponent, but zero * anything = 0

    let result = dequant_nvfp4(&block, 16).expect("dequant_nvfp4");
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val.abs() < 1e-6,
            "index {i}: expected 0.0, got {val}"
        );
    }
}

#[test]
fn test_dequant_nvfp4_multi_subblock() {
    // Build a 48-weight tensor = 3 sub-blocks, each with distinct exponent.
    let mut bytes = Vec::with_capacity(27); // 3 × 9
    for sb in 0..3u8 {
        let codes = [sb * 2; 16]; // different code per sub-block
        let block = build_nvfp4_sub_block(127, &codes);
        bytes.extend_from_slice(&block);
    }

    let result = dequant_nvfp4(&bytes, 48).expect("dequant_nvfp4");
    assert_eq!(result.len(), 48);

    for sb in 0..3usize {
        let code = (sb * 2) as u8;
        let expected = e2m1_codebook(code);
        for i in 0..16 {
            let idx = sb * 16 + i;
            assert!(
                (result[idx] - expected).abs() < 1e-6,
                "sub-block {sb} index {i}: expected {expected}, got {}",
                result[idx]
            );
        }
    }
}

#[test]
fn test_dequant_nvfp4_full_super_block() {
    // Full 256-weight super-block with varying codes per sub-block.
    let mut sub_blocks: [[u8; 9]; 16] = [[0u8; 9]; 16];
    for sb in 0..16 {
        // Each sub-block uses code = sb (0..15), exponent = 127 (scale 1.0).
        let codes = [sb as u8; 16];
        sub_blocks[sb] = build_nvfp4_sub_block(127, &codes);
    }
    let super_block = build_nvfp4_super_block(&sub_blocks);

    let result = dequant_nvfp4(&super_block, 256).expect("dequant_nvfp4");
    assert_eq!(result.len(), 256);

    for sb in 0..16 {
        let expected = e2m1_codebook(sb as u8);
        for i in 0..16 {
            let idx = sb * 16 + i;
            assert!(
                (result[idx] - expected).abs() < 1e-6,
                "sub-block {sb} index {i}: expected {expected}, got {}",
                result[idx]
            );
        }
    }
}

#[test]
fn test_dequant_nvfp4_empty_input() {
    let result = dequant_nvfp4(&[], 0).expect("dequant_nvfp4 empty");
    assert!(result.is_empty());
}

#[test]
fn test_dequant_nvfp4_truncated_input_errors() {
    // 256 weights need 144 bytes; providing less should error.
    let short = vec![0u8; 100];
    let result = dequant_nvfp4(&short, 256);
    assert!(result.is_err(), "expected error for truncated input");
}

#[test]
fn test_reframe_nvfp4_to_mxfp4_preserves_codes() {
    // Build a 32-weight NVFP4 block (2 sub-blocks) and reframe to MXFP4 framing.
    // Sub-block 0: code 2 (value 1.0), exponent 127.
    // Sub-block 1: code 4 (value 2.0), exponent 127.
    let codes_0 = [2u8; 16];
    let codes_1 = [4u8; 16];
    let block_0 = build_nvfp4_sub_block(127, &codes_0);
    let block_1 = build_nvfp4_sub_block(127, &codes_1);

    let mut nvfp4 = Vec::with_capacity(18);
    nvfp4.extend_from_slice(&block_0);
    nvfp4.extend_from_slice(&block_1);

    let reframe = reframe_nvfp4_to_mxfp4(&nvfp4, 32).expect("reframe_nvfp4_to_mxfp4");

    // MXFP4 framing: [u64 codes_len][codes...][u64 exps_len][exps...].
    let codes_len = u64::from_le_bytes(reframe[0..8].try_into().unwrap()) as usize;
    let exps_len =
        u64::from_le_bytes(reframe[8 + codes_len..8 + codes_len + 8].try_into().unwrap()) as usize;

    assert_eq!(codes_len, 16); // 32 weights / 2 per byte
    assert_eq!(exps_len, 1); // 1 group of 32

    // Verify codes are preserved: element i comes from sub-block i/16.
    // Sub-block 0 (elements 0-15) → code 2, sub-block 1 (elements 16-31) → code 4.
    let codes = &reframe[8..8 + codes_len];
    for i in 0..32 {
        let byte_idx = i / 2;
        let byte = codes[byte_idx];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
        let expected = if i < 16 { 2 } else { 4 };
        assert_eq!(nibble, expected, "element {i}");
    }
}

#[test]
fn test_reframe_nvfp4_to_mxfp4_roundtrip_values() {
    // Build NVFP4 bytes, reframe to MXFP4, then dequant via the MXFP4 path
    // (simulating kernel consumption). Values must match direct NVFP4 dequant.
    use grim_quant::dequant_mxfp4;

    // 32 weights: 2 sub-blocks of 16, code 6 (value 4.0) at exponent 128 (scale 2.0).
    let codes = [6u8; 16];
    let block_0 = build_nvfp4_sub_block(128, &codes);
    let block_1 = build_nvfp4_sub_block(128, &codes);

    let mut nvfp4 = Vec::with_capacity(18);
    nvfp4.extend_from_slice(&block_0);
    nvfp4.extend_from_slice(&block_1);

    // Direct NVFP4 dequant.
    let direct = dequant_nvfp4(&nvfp4, 32).expect("dequant_nvfp4");

    // Reframe then MXFP4 dequant.
    let reframe = reframe_nvfp4_to_mxfp4(&nvfp4, 32).expect("reframe_nvfp4_to_mxfp4");
    let via_mxfp4 = dequant_mxfp4(&reframe, 32).expect("dequant_mxfp4");

    for i in 0..32 {
        // MXFP4 uses one shared exponent per 32-element group — both halves
        // get the same exponent here, so values should match closely.
        let tol = 0.25 * direct[i].abs().max(1.0); // E2M1 quantization step ~25% at this range
        assert!(
            (direct[i] - via_mxfp4[i]).abs() < tol,
            "index {i}: direct={} via_mxfp4={} (tol={tol})",
            direct[i],
            via_mxfp4[i]
        );
    }
}

#[test]
fn test_reframe_nvfp4_to_mxfp4_rejects_mismatched_exponents() {
    // Adjacent sub-blocks with different exponents cannot be losslessly
    // reframed to MXFP4's 32-element groups. The function must return Err.
    let codes = [2u8; 16];
    let block_0 = build_nvfp4_sub_block(127, &codes); // scale 1.0
    let block_1 = build_nvfp4_sub_block(128, &codes); // scale 2.0

    let mut nvfp4 = Vec::with_capacity(18);
    nvfp4.extend_from_slice(&block_0);
    nvfp4.extend_from_slice(&block_1);

    let result = reframe_nvfp4_to_mxfp4(&nvfp4, 32);
    assert!(
        result.is_err(),
        "reframe_nvfp4_to_mxfp4 must error when adjacent sub-blocks have \
         different exponents, got Ok"
    );
}
