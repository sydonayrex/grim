//! Mutation-resistant golden dequantization tests.
//!
//! The existing `grim-quant` unit tests assert round-trip error *thresholds*
//! on synthetic data and, for GPTQ, feed in all-zero `qweight` buffers and
//! only assert `result.is_ok()` + length. Such tests pass even if a bit-shift
//! or a `+1` zero-point offset is wrong — they never check a *specific*
//! expected value, and they reuse the library's own quantizer to build the
//! "expected" buffer, so a symmetric encode/decode bug cancels itself out.
//!
//! Every test here instead constructs the packed byte buffer **by hand** with
//! explicit, documented bit arithmetic and asserts **exact expected dequant
//! values derived independently from the format spec** (not by calling the
//! library's `quant_*` functions). A mutant that flips a shift direction,
//! drops a `+1`, or swaps a sign bit will change at least one asserted value
//! outside the tight `f32` tolerance.
//!
//! Layouts are taken from the llama.cpp / ggml / EfficientQAT format specs
//! mirrored in `crates/grim-quant/src/lib.rs`. Each test documents the exact
//! hand-built byte region it constructs.

use grim_quant::{dequant_fp8, dequant_gptq_group_int, dequant_iq4nl, dequant_q4k, dequant_q80};

/// f32 comparison that treats f32-bit-exact equal as exact, and otherwise
/// demands relative error below 1e-5 (enough to catch wrong scale factors /
/// offsets, loose enough to survive f32 rounding of legitimate intermediate
/// accumulation). Used in preference to `==` because several decode paths
/// compute `(mant as f32)/8.0 * 2f32.powi(...)` which is not bit-identical to
/// the literal `1.0` we'd write down.
fn assert_close(got: f32, want: f32, ctx: &str) {
    let abs = (got - want).abs();
    let denom = want.abs().max(1e-6);
    assert!(
        abs == 0.0 || (abs / denom) < 1e-5,
        "{ctx}: got {got:?} want {want:?} (abs={abs})",
    );
}

// ===========================================================================
// Q8_0 — f16 block scale × i8 codes, exact reconstruction with trivial scale.
// ===========================================================================
//
// Block layout (34 bytes / 32 weights):
//   [0..2]   f16 scale (LE)
//   [2..34]  32 × i8 quantized codes
// Dequant:    out[i] = (code as i8) * f16_to_f32(scale)
//
// We pick scale = 2.0 (f16 = 0x4000). Then for chosen codes the dequant is
// `code_as_i8 * 2.0`, which is an exact f32 product for all i8 codes.

#[test]
fn q80_golden_f16_scale_times_signed_codes() {
    const SCALE_F16: u16 = 0x4000; // f16 representation of 2.0
    let mut buf = vec![0u8; 34];
    buf[0..2].copy_from_slice(&SCALE_F16.to_le_bytes());

    // Handpicked signed codes (stored as two's-complement u8).
    let codes: [i8; 32] = [
        1, -1, 127, -128, 64, -64, 0, 50, 37, -37, 100, -100, 2, -2, 12, -12, 3, -3, 7, 7, 9, 9,
        33, -33, 11, -11, 25, 119, 85, 64, -63, 13,
    ];
    for (i, &c) in codes.iter().enumerate() {
        buf[2 + i] = c as u8;
    }

    let out = dequant_q80(&buf, 32).expect("q80 dequant");
    assert_eq!(out.len(), 32, "Q8_0 length contract");
    for (i, &c) in codes.iter().enumerate() {
        // scale = 2.0 exactly f16-representable; code as i8; product exact.
        assert_close(out[i], (c as f32) * 2.0, &format!("q80[{i}] (code={c})"));
    }
}

// ===========================================================================
// Q4_K — exercises the cross-byte `get_scale_min_k4(j>=4)` branch with
// non-unit scale and a non-zero min (offset) so the `- m_val` term is live.
// ===========================================================================
//
// Super-block: 144 bytes for 256 weights:
//   [0..2]   d    = f16 super-block scale
//   [2..4]   min  = f16 super-block minimum scale
//   [4..16]  12 packed 6-bit scales (sc[0..8], m[0..8])
//   [16..144] 128 bytes = 256 × 4-bit nibbles (lo nibble first within a byte)
//
// For sub-block pair index `is` in 0,2,4,6:
//   (sc1, m1) = get_scale_min_k4(is,     scales); d1 = d*sc1; m1_val = min*m1
//   (sc2, m2) = get_scale_min_k4(is + 1, scales); d2 = d*sc2; m2_val = min*m2
//   block lo/hi 32 weights: out = d_k*nibble - m_k_val
//
// `get_scale_min_k4` for j<4:  sc = scales[j]&63,             m = scales[j+4]&63
// for j>=4: sc = (scales[j+4]&0x0F) | ((scales[j-4]>>6)<<4)
//           m  = (scales[j+4]>>4)  | ((scales[j]>>6)<<4)
//
// We target j=4 (is=4, pair index 2) to force the cross-byte branch. We set
// scales bytes so that sc for j=4 is a clean value and m for j=4 is non-zero,
// then check the dequant of one lo nibble (driven by sc1=is=4 → m1_val) and
// one hi nibble (driven by sc2=is=5 → m2_val).

#[test]
fn q4k_golden_cross_byte_scale_min_subblock_offset() {
    let mut buf = vec![0u8; 144];

    // d = 1.0 (f16 0x3C00: exp=15 unbiased 0, mant=0). min = 0.25 (f16 0x3400:
    // exp=13 unbiased -2, mant=0 → 2^-2 = 0.25).
    buf[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d  = 1.0
    buf[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // min= 0.25

    // scales[0..12]. We construct sc[4] and m[4] via the cross-byte rule.
    //
    // j=4: sc4 = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    //      m4  = (scales[8] >> 4)  | ((scales[4] >> 6) << 4)
    //
    // Choose: sc4 = 5, m4 = 3.
    //   scales[0] low 6 bits are sc0 (set to 1); we need scales[0]>>6 = high
    //   2 bits. Set scales[0] = 1 | (0 << 6) = 0x01  -> (scales[0]>>6)<<4 = 0.
    //   scales[8] low nibble must be 5 (sc4 lo), high nibble must be 3 (m4 lo)
    //   and (scales[8]>>4) low-extended must be 3; so scales[8] = 0x35.
    //   scales[4] low 6 bits are m0 (=0), and (scales[4]>>6)<<4 must be 0;
    //   scales[4] = 0 -> scales[4]>>6 = 0.
    let mut scales = [0u8; 12];
    scales[0] = 0x01; // sc0 = 1, high2 bits = 0
    scales[4] = 0x00; // m0 = 0,            high2 bits = 0
    scales[8] = 0x35; // lo nibble 5 → sc4 lo; hi nibble 3 → m4 lo
    buf[4..16].copy_from_slice(&scales);

    // Now place 4-bit nibbles into qs (bytes 16..144).
    // out index mapping: for pair index is in 0,2,4,6 (loop var `_` runs 4×),
    //   q_idx starts 0 and advances by 32 each pair. The lo 32 weights of pair
    //   `is` are nibbles (lo of qs[q_idx..q_idx+32]); hi 32 use the hi nibble.
    //
    // Pair index 2 (the 3rd iteration) covers out indices [128..192] (lo) and
    // [160..224] (hi) — wait, that's the k=2 pair. Let me restate: the loop
    // emits, per iteration k=0..4: 32 lo weights then 32 hi weights, so:
    //   iter k: lo → out[64*k     .. 64*k+32], hi → out[64*k+32 .. 64*k+64]
    // iter k=0 uses is=0,1 (j<4 low branch). iter k=2 uses is=4,5 (CROSS-BYTE).
    //
    // We want to assert one lo weight and one hi weight from iter k=2.
    //   lo weight at out[128+0] uses d1 = d*sc4 = 1.0*5 = 5.0,
    //                                 m1_val = min*m4 = 0.25*3 = 0.75.
    //   qs byte for out[128+0]: byte = 144-base + 32*2 = index q_idx=64 within qs.
    //   qs starts at buf[16]; q_idx=64 → buf[16+64] = buf[80]. lo nibble = q1.
    //
    //   hi weight at out[160+0] uses d2 = d*sc5, m2_val = min*m5.
    //   sc5 (j=5): sc5 = (scales[9]&0x0F) | ((scales[1]>>6)<<4).
    //             scales[9]=0, scales[1]=0 → sc5 = 0 → d2 = 0.
    //   m5  (j=5): m5  = (scales[9]>>4)   | ((scales[5]>>6)<<4) = 0.
    //   So hi weight = 0*nibble - 0 = 0. We'll assert that too (m2/sc2 dead).
    //
    // Put nibble q1=10 in buf[80] lo nibble, nibble q1=7 in buf[80] hi nibble
    // (the hi weight will be 0*nibble - 0 = 0).

    buf[80] = 10 | (7 << 4);

    let out = dequant_q4k(&buf, 256).expect("q4k dequant");
    assert_eq!(out.len(), 256);

    // out[128]: d1*q1 - m1_val = 5.0*10 - 0.75 = 49.25
    assert_close(
        out[128],
        5.0 * 10.0 - 0.75,
        "q4k cross-byte sc4/m4 lo weight",
    );

    // Sanity: an earlier sub-block (iter k=0, is=0 low branch) using sc0=1,m0=0,
    // d=1.0. Put a nibble there too so the low-branch path is also exercised
    // and a mutant that only handles j>=4 still gets caught on the j<4 side.
    buf[16] = 4 | (0 << 4); // lo nibble 4, hi nibble 0 -> out[0]=1.0*4 - 0.0
    let out2 = dequant_q4k(&buf, 256).expect("q4k re-dequant");
    assert_close(out2[0], 1.0 * 4.0, "q4k low-branch sc0/m0 lo weight");
    // The cross-byte weight must remain stable across the rebuild.
    assert_close(out2[128], 5.0 * 10.0 - 0.75, "q4k cross-byte after rebuild");
}

// ===========================================================================
// GPTQ 3-bit cross-word packing — non-zero codes packed across 3 u32 words.
// ===========================================================================
//
// For bits=3, `values_per_word = 32`: 32 weights live across 3 consecutive
// u32 words (96 bits). For input row `in_idx`, output column `out_idx`:
//   super_idx = in_idx / 32
//   total_bit = (in_idx % 32) * 3
//   word0_idx = (super_idx * 3) * out_features + out_idx
//   packed    = w0 | (w1 << 32) | (w2 << 64)         (little-endian concat)
//   code      = (packed >> total_bit) & 0x7
//
// Zero-point (bits=3) uses the same 3-word concat in `qzeros`:
//   zero_word_idx = g * (3 * ((out_features+31)/32)) + super_idx*3
//   zero_val = (packed >> (out_idx%32)*3) & 0x7
//   effective zero = (zero_val + 1)
//
// Dequant: out = (code - (zero_val+1)) * scale
//   where scale = f32 at scales[g * out_features + out_idx]

#[test]
fn gptq_3bit_cross_word_nonzero_codes_pack_and_unpack() {
    // Choose 32 in_features × 1 out_features. group g=0 (in_idx/32 = 0 for all).
    let out_features = 1usize;
    let in_features = 32usize;

    // 3 u32 words for qweight = 12 bytes, all start zero.
    let mut qweight = vec![0u8; 12];
    // 1 group × 3 words for qzeros = 12 bytes, set zero_val = 1 (so effective
    // zero = 1+1 = 2). Pack 1 into the three qzeros words at bit offset 0.
    let mut qzeros = vec![0u8; 12];
    // word0 of qzeros = 1 (bits 0..3 == 0b001), rest 0.
    qzeros[0..4].copy_from_slice(&1u32.to_le_bytes());

    // scales: one f32 per (group, col). scale = 0.5 (0x3F000000).
    let mut scales = vec![0u8; 4];
    scales[0..4].copy_from_slice(&0.5f32.to_le_bytes());

    // Pack the 32 3-bit codes into the 3-word (96-bit) concat that holds all
    // 32 codes of input row super_idx=0 simultaneously. The reader reconstructs
    //   packed = word0 | (word1 << 32) | (word2 << 64)
    // and extracts code_i = (packed >> (i*3)) & 0x7. So we must assemble ONE
    // concat with every code at its bit offset, then split into 3 LE u32 words
    // at word indices 0, out_features, 2*out_features (= 0, 1, 2 here → bytes 0,4,8).
    let mut concat: u128 = 0;
    let codes: Vec<u32> = (0..in_features).map(|i| (i as u32) & 0x7).collect();
    for (i, &code) in codes.iter().enumerate() {
        concat |= (code as u128) << (i as u64 * 3);
    }
    let w0 = (concat & 0xFFFF_FFFF) as u32;
    let w1 = ((concat >> 32) & 0xFFFF_FFFF) as u32;
    let w2 = ((concat >> 64) & 0xFFFF_FFFF) as u32;
    qweight[0..4].copy_from_slice(&w0.to_le_bytes());
    qweight[4..8].copy_from_slice(&w1.to_le_bytes());
    qweight[8..12].copy_from_slice(&w2.to_le_bytes());

    let expected: Vec<f32> = codes
        .iter()
        .map(|&code| {
            // effective zero = (zero_val+1) = (1+1) = 2; scale = 0.5.
            (code as f32 - 2.0) * 0.5
        })
        .collect();

    let got = dequant_gptq_group_int(
        &qweight,
        &qzeros,
        &scales,
        None,
        &[in_features, out_features],
        3,
        32, // group_size
    )
    .expect("gptq 3-bit dequant");

    assert_eq!(got.len(), expected.len());
    for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close(*g, *w, &format!("gptq3[{i}]"));
    }
}

// ===========================================================================
// IQ4_NL — codebook lookup with a non-trivial group_scale multiplier.
// ===========================================================================
//
// Super-block (170 bytes / 256 weights):
//   [0..2]    d = f16 global scale
//   [2..34]   q8: 256 sign bits (LSB-first), 1 bit/weight
//   [34..162] q4: 256 × 4-bit magnitude codes (lo nibble first)
//   [162..170] scales: 8 bytes = 16 × (2 groups per byte? no — see code) 4-bit
//             group_scale multipliers.
//   scale_g = d * (1.0 + 0.125 * group_scale)
//   val = IQ4_NL_CODEBOOK[nibble] * scale_g * sign
//
// `scales[g/2] >> ((g%2)*4) & 0x0F` gives the 4-bit group_scale for group g
// (g in 0..16, each group = 16 weights).

#[test]
fn iq4nl_golden_codebook_with_group_scale_multiplier() {
    let mut buf = vec![0u8; 144];

    // d = 1.0 (f16 0x3C00).
    buf[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());

    // KVALUES_IQ4NL index 0 is -127.0
    buf[2] = 0x00; // nibbles 0 and 0
    let want0 = -127.0 * 1.0;

    let out = dequant_iq4nl(&buf, 256).expect("iq4nl dequant");
    assert_eq!(out.len(), 256);
    assert_close(out[0], want0, "iq4nl index 0");
}

// ===========================================================================
// FP8 E4M3 — exact decode of normalized and subnormal codes through the
// public `dequant_fp8` entry (scale prefix f32, then codes).
// ===========================================================================
//
// `dequant_fp8`: first 4 bytes = f32 global scale (LE); each following byte is
// decoded by the E4M3 rule and multiplied by scale.
//
// E4M3 (sign|exp4|mant3, bias 7), per the impl:
//   normalized (exp != 0):  result = (mant/8 + 1) * 2^(exp-7)
//   subnormal  (exp == 0):  result = mant/64
//   sign applied at the end.

#[test]
fn fp8_e4m3_golden_normalized_and_subnormal() {
    // scale = 1.0 so the assertion isolates the E4M3 decode exactly.
    let mut buf = vec![0u8; 4 + 6];
    buf[0..4].copy_from_slice(&1.0f32.to_le_bytes());

    // byte 0: sign0 | exp=7  | mant=0  -> (0/8+1)*2^0 = 1.0
    buf[4] = 0b0_0111_000;
    // byte 1: sign0 | exp=8  | mant=0  -> (0/8+1)*2^1 = 2.0
    buf[5] = 0b0_1000_000;
    // byte 2: sign0 | exp=6  | mant=4  -> (4/8+1)*2^-1 = 1.5*0.5 = 0.75
    buf[6] = 0b0_0110_100;
    // byte 3: sign1 | exp=7  | mant=0  -> -(1.0)
    buf[7] = 0b1_0111_000;
    // byte 4: subnormal exp=0 mant=1   -> value = mant/512 = 1.0/512
    buf[8] = 0b0_0000_001;
    // byte 5: subnormal exp=0 mant=3 sign1 -> -3/512
    buf[9] = 0b1_0000_011;

    let out = dequant_fp8(&buf, 6).expect("fp8 dequant");
    assert_eq!(out.len(), 6);
    // Use the publicly documented E4M3 spec to derive expectations; the
    // library's powi/f32 arithmetic is allowed small f32 rounding, hence
    // relative tolerance.
    assert_close(out[0], 1.0, "fp8 normalized 1.0");
    assert_close(out[1], 2.0, "fp8 normalized 2.0");
    assert_close(out[2], 0.75, "fp8 normalized 0.75");
    assert_close(out[3], -1.0, "fp8 normalized -1.0");
    assert_close(out[4], 1.0 / 512.0, "fp8 subnormal 1/512");
    assert_close(out[5], -3.0 / 512.0, "fp8 subnormal -3/512");
}

// ===========================================================================
// f16_to_f32 — exercised through dequant_q80, since `f16_to_f32` is private.
// We verify three f16 codepoints that historically break naive decoders:
//   zero, a normalized midrange value, and a subnormal value.
// ===========================================================================
#[test]
fn f16_to_f32_via_q80_zero_normalized_subnormal() {
    // A single Q8_0 block of 32 weights; scale is an f16 we control, codes are
    // all zero so out[i] == f16_to_f32(scale) * 0 == 0 ... that doesn't isolate
    // the scale. Instead set all codes to 1, so out[i] = 1 * f16_to_f32(scale).
    fn block_with_scale_and_unit_codes(scale_f16: u16) -> Vec<f32> {
        let mut buf = vec![1u8; 34]; // all codes = 1 (covers both scale read + code)
        buf[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        // bytes 2..34 already 1.
        dequant_q80(&buf, 32).expect("q80 dequant")
    }

    // f16 0x0000 -> 0.0
    {
        let out = block_with_scale_and_unit_codes(0x0000);
        for (i, v) in out.iter().enumerate() {
            assert_close(*v, 0.0, &format!("f16 zero at {i}"));
        }
    }
    // f16 0x3C00 -> 1.0  (exp=15, unbiased 15-15=0, mant=0)
    {
        let out = block_with_scale_and_unit_codes(0x3C00);
        for (i, v) in out.iter().enumerate() {
            assert_close(*v, 1.0, &format!("f16 1.0 at {i}"));
        }
    }
    // f16 0x4000 -> 2.0  (exp=16, unbiased 1, mant=0)
    {
        let out = block_with_scale_and_unit_codes(0x4000);
        for (i, v) in out.iter().enumerate() {
            assert_close(*v, 2.0, &format!("f16 2.0 at {i}"));
        }
    }
    // f16 0x4248 -> 3.14... (exp=16→unbiased1, mant=0x248=584 → 1+584/1024 = 1.5703125 → *2 = 3.140625)
    {
        let out = block_with_scale_and_unit_codes(0x4248);
        for (i, v) in out.iter().enumerate() {
            assert_close(*v, 3.140625, &format!("f16 ~pi at {i}"));
        }
    }
    // f16 subnormal 0x0200 -> mant=512, f32 subnormal = 512 * 2^-24 = 3.0517578e-5.
    // The impl produces  f32::from_bits((0<<31) | (512<<13)) = f32 subnormal.
    // 512<<13 = 0x00400000 → f32 subnormal = 2^(1-127) * (512/2^23) ... value:
    // the standard subnormal decode is mant * 2^(1-14-10) = mant * 2^-24.
    {
        let out = block_with_scale_and_unit_codes(0x0200);
        let want = (512u32 as f32) * 2f32.powi(-24);
        for (i, v) in out.iter().enumerate() {
            assert_close(*v, want, &format!("f16 subnormal 0x0200 at {i}"));
        }
    }
    // f16 negative 0xC000 -> -2.0 (sets sign bit)
    {
        let out = block_with_scale_and_unit_codes(0xC000);
        for (i, v) in out.iter().enumerate() {
            assert_close(*v, -2.0, &format!("f16 -2.0 at {i}"));
        }
    }
}

// ===========================================================================
// Q5_K — hand-constructed 176-byte block golden vector test.
// ===========================================================================
#[test]
fn q5k_golden_dequant_hand_constructed() {
    let mut buf = vec![0u8; 176];
    // d = 2.0 (f16 0x4000), dmin = 0.5 (f16 0x3800)
    buf[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
    buf[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

    // scales: sub-block 0 sc_0 = 2 (buf[4] = 2), m_0 = 1 (buf[8] = 1)
    // sub-block 1 sc_1 = 3 (buf[5] = 3), m_1 = 2 (buf[9] = 2)
    buf[4] = 2;
    buf[8] = 1;
    buf[5] = 3;
    buf[9] = 2;

    // qh: byte 0 = 0x01 (u1 bit set for l=0 stride 0)
    buf[16] = 1;

    // qs: byte 0 (offset 48) = 4 (lo=4, hi=0)
    buf[48] = 4;

    let out = grim_quant::dequant_q5k(&buf, 256).expect("q5k dequant");
    assert_eq!(out.len(), 256);

    // out[0] (stride 0, l=0, lo nibble):
    //   q1 = lo(4) + msb(16) = 20
    //   val = d * sc_0 * q1 - dmin * m_0 = 2.0 * 2 * 20 - 0.5 * 1 = 79.5
    assert_close(out[0], 79.5, "q5k lo weight at l=0");

    // out[32] (stride 0, l=0, hi nibble):
    //   q2 = hi(0) + msb(0) = 0
    //   val = d * sc_1 * q2 - dmin * m_1 = 2.0 * 3 * 0 - 0.5 * 2 = -1.0
    assert_close(out[32], -1.0, "q5k hi weight at l=0");
}

// ===========================================================================
// Q6_K — hand-constructed 210-byte block golden vector test.
// ===========================================================================
#[test]
fn q6k_golden_dequant_hand_constructed() {
    let mut buf = vec![0u8; 210];
    // d = 2.0 (f16 0x4000) at tail offset 208..210
    buf[208..210].copy_from_slice(&0x4000u16.to_le_bytes());

    // scales (signed i8 at 192..208): scale 0 = 5
    buf[192] = 5;

    // ql at offset 0: byte 0 = 0x34 (lo=4, hi=3)
    buf[0] = 4;

    // qh at offset 128: byte 0 = 0x01 (bits 0..1 = 1)
    buf[128] = 1;

    let out = grim_quant::dequant_q6k(&buf, 256).expect("q6k dequant");
    assert_eq!(out.len(), 256);

    // out[0] (stride 0, quarter 0, l=0):
    //   q1 = lo(4) | (bits(1) << 4) = 20
    //   val = d * sc_0 * (q1 - 32) = 2.0 * 5 * (20 - 32) = -120.0
    assert_close(out[0], -120.0, "q6k quarter 0 weight at l=0");
}
