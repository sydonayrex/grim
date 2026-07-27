use grim_quant::{
    dequant_fp4, dequant_fp4_block16, dequant_fp8_block16, fp8_e4m3_to_f32,
};

fn close(got: f32, want: f32, ctx: &str) {
    let abs = (got - want).abs();
    let denom = want.abs().max(1e-7);
    assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
    assert!(
        abs == 0.0 || (abs / denom) < 1e-5,
        "{ctx}: got {got:?} want {want:?} (abs={abs})",
    );
}

const FP4_E2M1_LUT: [f32; 16] = [
    -1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125,
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,
];

// ===========================================================================
// FP4 (E2M1) — same layout as NF4: f32 scale prefix, then packed 4-bit nibbles.
// ===========================================================================

#[test]
fn fp4_golden_hand_constructed_buffer() {
    let scale = 2.0f32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&scale.to_le_bytes());
    // Need ≥8 bytes total for data_start=4 (scale then codes).
    // byte 0 = 0xF0: hi=0xF (15→0.875), lo=0x0 (0→-1.0)
    // byte 1 = 0x37: hi=0x3 (3→-0.625), lo=0x7 (7→-0.125)
    // bytes 3-4: pad to reach 8 bytes
    buf.push(0xF0);
    buf.push(0x37);
    buf.push(0x00);
    buf.push(0x00);

    let out = dequant_fp4(&buf, 6).expect("fp4 dequant");
    assert_eq!(out.len(), 6);

    close(out[0], FP4_E2M1_LUT[0xF] * scale, "fp4[0] hi=F");
    close(out[1], FP4_E2M1_LUT[0x0] * scale, "fp4[1] lo=0");
    close(out[2], FP4_E2M1_LUT[0x3] * scale, "fp4[2] hi=3");
    close(out[3], FP4_E2M1_LUT[0x7] * scale, "fp4[3] lo=7");
    close(out[4], FP4_E2M1_LUT[0x0] * scale, "fp4[4] pad=0");
    close(out[5], FP4_E2M1_LUT[0x0] * scale, "fp4[5] pad=0");
}

#[test]
fn fp4_golden_scale_only_no_codes_returns_empty() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&1.5f32.to_le_bytes());
    let out = dequant_fp4(&buf, 0).expect("fp4 zero values");
    assert!(out.is_empty());
}

#[test]
fn fp4_golden_empty_buffer_pads_to_num_values() {
    let out = dequant_fp4(&[], 3).expect("fp4 empty buffer pads");
    assert_eq!(out.len(), 3);
    for (i, &v) in out.iter().enumerate() {
        close(v, 0.0, &format!("fp4 pad[{i}]"));
    }
}

// ===========================================================================
// FP8 block16 — per-block E4M3 scale + 16 FP8 E4M3 codes.
// ===========================================================================
//
// Layout:
//   [0..4]   f32 global_scale (LE)
//   For each block of 16 values:
//     [0]     u8 E4M3 block_scale
//     [1..17] 16 × u8 E4M3 codes
// Dequant: out[i] = fp8_e4m3_to_f32(code[i]) * fp8_e4m3_to_f32(block_scale) * global_scale

#[test]
fn fp8_block16_golden_one_block_identity_scale() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&1.0f32.to_le_bytes()); // global_scale = 1.0
    buf.push(0b0_0111_000); // block_scale e4m3 = 1.0
    // codes: 1.0, 2.0, 0.75, -1.0, 1/512, -3/512, NaN placeholder
    buf.push(0b0_0111_000); //  1.0
    buf.push(0b0_1000_000); //  2.0
    buf.push(0b0_0110_100); //  0.75
    buf.push(0b1_0111_000); // -1.0
    buf.push(0b0_0000_001); //  1/512
    buf.push(0b1_0000_011); // -3/512
    // pad to 16 values
    for _ in 6..16 { buf.push(0); }

    let out = dequant_fp8_block16(&buf, 16).expect("fp8 block16");
    assert_eq!(out.len(), 16);

    close(out[0], 1.0, "fp8b16[0] = 1.0");
    close(out[1], 2.0, "fp8b16[1] = 2.0");
    close(out[2], 0.75, "fp8b16[2] = 0.75");
    close(out[3], -1.0, "fp8b16[3] = -1.0");
    close(out[4], 1.0 / 512.0, "fp8b16[4] = 1/512");
    close(out[5], -3.0 / 512.0, "fp8b16[5] = -3/512");
    for i in 6..16 {
        close(out[i], 0.0, &format!("fp8b16[{i}] = 0.0"));
    }
}

#[test]
fn fp8_block16_golden_two_blocks_with_non_trivial_scales() {
    let global_scale = 0.5f32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&global_scale.to_le_bytes());

    // Block 0: block_scale = 2.0 (E4M3 0b0_1000_000), codes = [1.0, 1.0, ...]
    buf.push(0b0_1000_000); // block_scale = 2.0
    for _ in 0..16 {
        buf.push(0b0_0111_000); // code = 1.0
    }

    // Block 1: block_scale = 0.5 (E4M3 0b0_0110_000), codes = [1.0, -1.0, ...]
    buf.push(0b0_0110_000); // block_scale = 0.5 (exp=6, mant=0 → (0/8+1)*2^-1 = 0.5)
    for i in 0..16 {
        if i % 2 == 0 {
            buf.push(0b0_0111_000); //  1.0
        } else {
            buf.push(0b1_0111_000); // -1.0
        }
    }

    let out = dequant_fp8_block16(&buf, 32).expect("fp8 block16 2 blocks");
    assert_eq!(out.len(), 32);

    // Block 0: global_scale=0.5, block_scale=2.0, code=1.0 → 0.5 * 2.0 * 1.0 = 1.0
    for i in 0..16 {
        close(out[i], 1.0, &format!("fp8b16 block0[{i}]"));
    }

    // Block 1: global_scale=0.5, block_scale=0.5, code=±1.0 → 0.5 * 0.5 * ±1.0 = ±0.25
    for i in 16..32 {
        let want = if (i - 16) % 2 == 0 { 0.25 } else { -0.25 };
        close(out[i], want, &format!("fp8b16 block1[{i}]"));
    }
}

#[test]
fn fp8_block16_golden_zero_values_returns_empty() {
    let out = dequant_fp8_block16(&[], 0).expect("fp8 block16 zero values");
    assert!(out.is_empty());
}

// ===========================================================================
// FP4 block16 — per-block E4M3 scale + packed 4-bit FP4 codes (8 bytes/block).
// ===========================================================================
//
// Layout:
//   [0..4]   f32 global_scale (LE)
//   For each block of 16 values:
//     [0]     u8 E4M3 block_scale
//     [1..9]  8 bytes × 2 nibbles each = 16 FP4 codes (hi nibble first)
// Dequant: out[i] = FP4_E2M1_LUT[nibble] * fp8_e4m3_to_f32(block_scale) * global_scale

#[test]
fn fp4_block16_golden_one_block_all_codes() {
    let global_scale = 1.0f32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&global_scale.to_le_bytes());

    // block_scale = 2.0 (E4M3 0b0_1000_000)
    buf.push(0b0_1000_000);

    // 8 bytes packed: 0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87
    // Each nibble maps to FP4_E2M1_LUT at that index.
    buf.push(0xF0); // hi=0xF,  lo=0x0
    buf.push(0xE1); // hi=0xE,  lo=0x1
    buf.push(0xD2); // hi=0xD,  lo=0x2
    buf.push(0xC3); // hi=0xC,  lo=0x3
    buf.push(0xB4); // hi=0xB,  lo=0x4
    buf.push(0xA5); // hi=0xA,  lo=0x5
    buf.push(0x96); // hi=0x9,  lo=0x6
    buf.push(0x87); // hi=0x8,  lo=0x7

    let out = dequant_fp4_block16(&buf, 16).expect("fp4 block16");
    assert_eq!(out.len(), 16);

    let scale = global_scale * fp8_e4m3_to_f32(0b0_1000_000); // = 1.0 * 2.0 = 2.0
    for i in 0..16 {
        let nibble = [0xF, 0x0, 0xE, 0x1, 0xD, 0x2, 0xC, 0x3,
                      0xB, 0x4, 0xA, 0x5, 0x9, 0x6, 0x8, 0x7][i];
        let want = FP4_E2M1_LUT[nibble as usize] * scale;
        close(out[i], want, &format!("fp4b16[{i}] nibble={nibble:#x}"));
    }
}

#[test]
fn fp4_block16_golden_two_blocks_partial_last_block() {
    let global_scale = 3.0f32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&global_scale.to_le_bytes());

    // Block 0: block_scale = 1.0, 16 values (8 bytes)
    buf.push(0b0_0111_000); // block_scale E4M3 = 1.0
    for _ in 0..8 { buf.push(0x00); } // all nibbles = 0 (FP4[0] = -1.0)
    // Block 1: block_scale = 1.0, 8 values only (partial, 4 bytes = 8 nibbles)
    buf.push(0b0_0111_000);
    buf.push(0xF0); // hi=0xF, lo=0x0 → out[16]=(-1.0), out[17]=(0.0 after scale...)

    let out = dequant_fp4_block16(&buf, 24).expect("fp4 block16 partial");
    assert_eq!(out.len(), 24);

    // Block 0: scale = 3.0 * 1.0 = 3.0, all codes = 0 → -1.0 * 3.0 = -3.0
    for i in 0..16 {
        close(out[i], -3.0, &format!("fp4b16 block0[{i}]"));
    }
    // Block 1: scale = 3.0 * 1.0 = 3.0, first nibble = 0xF → 0.875 * 3.0 = 2.625
    close(out[16], 0.875 * 3.0, "fp4b16 block1[16]");
    // second nibble = 0x0 → -1.0 * 3.0 = -3.0
    close(out[17], -3.0, "fp4b16 block1[17]");
    // remaining should be pad
    for i in 18..24 {
        close(out[i], 0.0, &format!("fp4b16 block1 pad[{i}]"));
    }
}

#[test]
fn fp4_block16_golden_zero_values_returns_empty() {
    let out = dequant_fp4_block16(&[], 0).expect("fp4 block16 zero values");
    assert!(out.is_empty());
}
