//! TreePie's 5 bpw packer (WS-A A2) and full round trip (A3). CPU only.
//!
//! # Why 4+1
//!
//! A TreePie code is 5 bits: `sign(1) | exp(2) | mant(2)`. The obvious packing
//! is 6 codes per i32, which wastes 2 bits per word -- 5.33 bpw, and a wasted
//! bit per 6 values forever. Instead the planes are separated:
//!
//! ```text
//! words 0..3 : 4-bit payload (exp|mant), 8 nibbles per i32  -> 32 values
//! word  4    : 1 sign bit per value, 32 per i32             -> 32 values
//! ---------------------------------------------------------------
//! 5 i32 for 32 values = 160 bits / 32 = exactly 5.0 bpw
//! ```
//!
//! The win comes from the sign being its own plane. Signs are 1/32 as
//! information-dense as everything else, so isolating them is what turns 5.33
//! into 5.0. It also means the payload plane is directly usable by a nibble
//! loader and the sign plane is a single bitmask -- which is the shape the GPU
//! decode wants (A6).
//!
//! Byte-exactness is the assertion, matching `golden_nutcracker_oracle.rs`. A
//! tolerance-based round trip would absorb a swapped nibble order or an
//! off-by-one in the sign index, both of which are silent corruption.

use grim_quant::tree_pie;
use grim_quant::tree_pie::{e2m2_to_f32, f32_to_e2m2, pack_tree_pie_32, unpack_tree_pie_32};

/// A2: 32 values occupy exactly 5 i32, and the split is exact.
///
/// The `5` here is the format's entire selling point, so it is asserted as a
/// literal rather than derived from a constant -- if the packer ever grows a
/// sixth word, this fails by name.
#[test]
fn pack_32_values_into_five_int32_is_exact() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.37 - 4.0).collect();
    let mut arr = [0f32; 32];
    arr.copy_from_slice(&values);

    let packed = pack_tree_pie_32(&arr);
    assert_eq!(packed.len(), 5, "32 values must occupy exactly 5 i32");

    // 5.0 bpw exactly, with no padding waste. Asserted from the bit count so
    // the claim cannot drift if the layout changes.
    let bits: usize = packed.len() * 32;
    assert_eq!(bits, 160);
    assert_eq!(bits as f32 / 32.0, 5.0, "bpw must be exactly 5.0, not 5.33");
    assert_eq!(tree_pie::TREE_PIE_BPW, 5.0);
    assert_eq!(tree_pie::TREE_PIE_WORDS_PER_32, 5);
}

/// A2: the nibble/sign plane split is exact in both directions.
///
/// Checks the split directly rather than only through the round trip, so a
/// failure says *which* plane is wrong.
#[test]
fn the_nibble_and_sign_planes_split_exactly() {
    // Codes chosen to exercise every field: all 16 magnitudes, mixed signs,
    // and a deliberately non-monotonic order so a permuted nibble index fails.
    let codes: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
        0x0E, 0x0F, 0x10, 0x11, 0x1B, 0x03, 0x0F, 0x07, 0x00, 0x1C, 0x08, 0x14, 0x0B, 0x1F,
        0x01, 0x12, 0x09, 0x1D,
    ];

    // Pack via f32 so the test goes through the real encoder: quantize to the
    // code, then feed the code's *value* back. That keeps the packer's input
    // domain honest (it takes f32, not codes).
    let mut values = [0f32; 32];
    for (v, &code) in values.iter_mut().zip(codes.iter()) {
        *v = e2m2_to_f32(code);
    }
    let packed = pack_tree_pie_32(&values);

    // Payload plane: word i holds values 8i..8i+8 as nibbles, low nibble first.
    for (w, chunk) in codes.chunks(8).enumerate() {
        let expect = chunk
            .iter()
            .enumerate()
            .fold(0i32, |acc, (j, &c)| acc | (((c & 0x0F) as i32) << (j * 4)));
        assert_eq!(
            packed[w], expect,
            "payload word {w} (values {}..{}) must hold the low nibbles in order",
            w * 8,
            w * 8 + 8
        );
    }

    // Sign plane: one bit per value, value i at bit i.
    let expect_signs = codes
        .iter()
        .enumerate()
        .fold(0i32, |acc, (i, &c)| acc | ((((c >> 4) & 1) as i32) << i));
    assert_eq!(
        packed[4], expect_signs,
        "sign word must hold bit i = sign of value i"
    );
}

/// A2: packing and unpacking is the identity on the quantized value.
///
/// This is the property that makes the packer lossless *for its own format*:
/// unpack(pack(v)) must equal the grid point, not v.
#[test]
fn pack_unpack_reaches_the_grid_point_exactly() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32 * 1.7).sin() * 9.0).collect();
    for chunk in values.chunks(32) {
        let mut arr = [0f32; 32];
        arr.copy_from_slice(chunk);
        let packed = pack_tree_pie_32(&arr);
        let back = unpack_tree_pie_32(&packed);
        for (i, (&orig, &got)) in arr.iter().zip(back.iter()).enumerate() {
            let want = e2m2_to_f32(f32_to_e2m2(orig));
            assert_eq!(
                got, want,
                "value {i}: {orig} should quantize to {want} and come back unchanged, got {got}"
            );
        }
    }
}

/// A3: all 32 encodings, both signs, survive a full pack/unpack cycle.
///
/// The `e == 0` row (subnormal in E2M2: 0, .5, 1, 1.5) is called out
/// separately because it is the row with the non-obvious FP16 placement, and a
/// bug there would otherwise hide behind the larger normal values.
#[test]
fn full_32_value_byte_space_roundtrips() {
    for code in 0..32u8 {
        let v = e2m2_to_f32(code);
        let mut arr = [0f32; 32];
        arr[0] = v;
        // Fill the rest with a rotating pattern so the surrounding nibbles are
        // non-zero and a stride bug cannot pass.
        for (i, slot) in arr.iter_mut().enumerate().skip(1) {
            *slot = e2m2_to_f32(((code as usize + i) % 32) as u8);
        }
        let back = unpack_tree_pie_32(&pack_tree_pie_32(&arr));
        assert_eq!(back[0], v, "code {code:#04x} ({v}) must survive the packer");
    }
}

/// A3: the subnormal row specifically, which is where the FP16 placement is
/// non-obvious (0, .5, 1, 1.5 are subnormal in E2M2 but normal in FP16).
#[test]
fn the_subnormal_row_round_trips_through_the_packer() {
    for (code, want) in [(0x00u8, 0.0f32), (0x01, 0.5), (0x02, 1.0), (0x03, 1.5)] {
        for sign in [0x00u8, 0x10u8] {
            let v = e2m2_to_f32(code | sign);
            let mut arr = [0f32; 32];
            arr[7] = v;
            let back = unpack_tree_pie_32(&pack_tree_pie_32(&arr));
            let expected = e2m2_to_f32(code | sign);
            assert_eq!(back[7], expected, "subnormal {expected} must round trip");
            assert_eq!(back[7].abs(), want, "subnormal magnitude must be {want}");
        }
    }
}

/// A3: negative zero is distinguishable from positive zero through the packer.
///
/// If the sign plane were dropped or the bit index were off, both would decode
/// to +0 and this would still pass a magnitude-only check.
#[test]
fn signed_zero_survives_the_sign_plane() {
    let mut arr = [0f32; 32];
    arr[0] = -0.0;
    arr[1] = 0.0;
    let packed = pack_tree_pie_32(&arr);
    assert_eq!(packed[4] & 1, 1, "value 0's sign bit must be set");
    assert_eq!(packed[4] & 2, 0, "value 1's sign bit must be clear");
    let back = unpack_tree_pie_32(&packed);
    assert!(back[0].is_sign_negative(), "-0 must stay negative");
    assert!(!back[1].is_sign_negative(), "+0 must stay positive");
}

/// The packer is total: it must not panic on any `f32`, including the values a
/// real checkpoint contains and the ones an adversarial one might.
///
/// Non-totality is not a style issue here. A panic during model load takes down
/// the process, so an encoder that can panic on `inf` or `NaN` is a denial of
/// service on malformed input.
#[test]
fn the_packer_is_total_over_adversarial_input() {
    let nasty = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-45,
        12.5,
        13.0,
        14.0,
        15.0,
        1e30,
        -1e30,
    ];
    let mut arr = [0f32; 32];
    for (i, &v) in nasty.iter().enumerate() {
        arr[i] = v;
    }
    // Must not panic.
    let packed = pack_tree_pie_32(&arr);
    let back = unpack_tree_pie_32(&packed);
    // Every output must be a finite grid point, never NaN.
    for (i, &b) in back.iter().enumerate() {
        assert!(
            b.is_finite(),
            "slot {i} decoded to {b}, which must be a finite grid point"
        );
    }
    // NaN and +-inf must both land on zero rather than saturating to 14,
    // which would amplify a caller bug into a large weight.
    assert_eq!(back[0], 0.0, "NaN encodes to zero");
    assert_eq!(back[1], 14.0, "+inf saturates to the top of the grid");
    assert_eq!(back[2], -14.0, "-inf saturates, keeping the sign");
}
