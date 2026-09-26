//! GGUF `Q2_0` (dtype tag 42) dequantization KATs.
//!
//! These replace a vacuous "all-zero block dequantizes to 256 zeros" check, which
//! passes under *any* layout. The geometry and math asserted here are
//! transcribed from ggml-org/llama.cpp at `f3f1a8f` — the runtime the
//! Qwen3.8-Flash-Next GSQ-RCO release is pinned to:
//!
//! ```c
//! #define QK2_0 64
//! typedef struct { ggml_half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;
//! // dequantize_row_q2_0:
//! const int q = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x03;
//! // 00=-1, 01=0, 10=+1, 11=+2
//! y[i*qk + j] = ((int)q - 1) * d;
//! ```

use grim_format::gguf::GgufDType;
use grim_quant::{
    BLOCK_BYTES_GSQ_RCO_3P5, BLOCK_SIZE_GSQ_RCO_3P5, dequant_gsq_rco_3p5, quantize_q2_0_block,
};
use grim_tensor::provider::TensorProvider;

fn f32_to_f16_le(v: f32) -> [u8; 2] {
    let s = v.to_bits();
    let sign = ((s >> 16) & 0x8000) as u16;
    let exp = ((s >> 23) & 0xff) as i32;
    let mant = (s & 0x007f_ffff) as u32;
    if exp == 0xff {
        let m = if mant != 0 { 0x200 } else { 0 };
        let e = if mant != 0 { 0x7e00 } else { 0x7c00 };
        return (sign | e | m as u16).to_le_bytes();
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return (sign | 0x7c00).to_le_bytes();
    }
    if e <= 0 {
        // Subnormal: quantize the 24-bit significand shifted into fp16 subnormal range.
        let full = mant | 0x0080_0000; // implicit leading 1
        let shift = (14 - e) as u32;
        let m = (full >> shift) as u16;
        return (sign | m).to_le_bytes();
    }
    (sign | ((e as u16) << 10) | ((mant >> 13) as u16)).to_le_bytes()
}

#[test]
fn geometry_is_64_weights_per_18_byte_block() {
    // 2.25 bits/weight. A 256-weight/72-byte reading is 4x wrong and silently
    // misaligns every subsequent block.
    assert_eq!(BLOCK_SIZE_GSQ_RCO_3P5, 64);
    assert_eq!(BLOCK_BYTES_GSQ_RCO_3P5, 18);
    let bits = BLOCK_BYTES_GSQ_RCO_3P5 * 8;
    assert_eq!(bits, BLOCK_SIZE_GSQ_RCO_3P5 * 9 / 4, "2.25 bits per weight");
}

#[test]
fn codebook_is_minus_one_zero_plus_one_plus_two() {
    // The single most important property: code 1 is ZERO, code 0 is -d.
    // Reading this as `-2..+1` (the older Q2_0 proposal) shifts every weight.
    let d = f32_to_f16_le(0.5);
    let mut block = [0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    block[0] = d[0];
    block[1] = d[1];
    // Byte 0 holds codes for j = 0..4; encode 0,1,2,3 in ascending bit order.
    block[2] = 0b11_10_01_00;

    let out = dequant_gsq_rco_3p5(&block, BLOCK_SIZE_GSQ_RCO_3P5).expect("dequant");
    assert_eq!(out.len(), 64);
    assert_eq!(out[0], -0.5, "code 0 -> -d");
    assert_eq!(out[1], 0.0, "code 1 -> 0");
    assert_eq!(out[2], 0.5, "code 2 -> +d");
    assert_eq!(out[3], 1.0, "code 3 -> +2d");
    // j=4 starts a fresh code byte, which is still zero => code 0 => -d.
    // This is what a decoder that packs 4 codes per byte gets wrong.
    assert_eq!(out[4], -0.5, "byte 1 restarts at j=4");
    assert_eq!(out[63], -0.5);
}

#[test]
fn negative_scale_flips_sign_of_every_code() {
    // Exercises bit 15 of the fp16 delta, which a truncated unpacker drops.
    let d = f32_to_f16_le(-0.25);
    let mut block = [0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    block[0] = d[0];
    block[1] = d[1];
    for byte in block[2..].iter_mut() {
        *byte = 0xff; // all codes = 3 -> +2d = -0.5
    }
    let out = dequant_gsq_rco_3p5(&block, BLOCK_SIZE_GSQ_RCO_3P5).expect("dequant");
    assert!(
        out.iter().all(|v| (*v - (-0.5)).abs() < 1e-6),
        "{:?}",
        out[0]
    );
}

#[test]
fn subnormal_and_tiny_scales_decode() {
    // fp16 subnormal deltas: a naive unpacker returns 0 or inf here.
    for d in [6.0e-8f32, 6.1e-5, 1.0e-3] {
        let hb = f32_to_f16_le(d);
        let mut block = [0u8; BLOCK_BYTES_GSQ_RCO_3P5];
        block[0] = hb[0];
        block[1] = hb[1];
        block[2] = 0b11_10_01_00;
        let out = dequant_gsq_rco_3p5(&block, BLOCK_SIZE_GSQ_RCO_3P5).expect("dequant");
        let got_d = -out[0];
        assert!(
            (got_d - d).abs() <= d * 1e-2,
            "d={d} did not round-trip: got {got_d}"
        );
        assert_eq!(out[1], 0.0);
    }
}

#[test]
fn block_stride_advances_by_18_not_72() {
    // Two blocks with different scales. Reading with a 72-byte stride would
    // pull the second block's header from the wrong offset.
    let d1 = f32_to_f16_le(0.5);
    let d2 = f32_to_f16_le(2.0);
    let mut buf = vec![0u8; BLOCK_BYTES_GSQ_RCO_3P5 * 2];
    buf[0] = d1[0];
    buf[1] = d1[1];
    buf[2] = 0b11_10_01_00;
    let o = BLOCK_BYTES_GSQ_RCO_3P5;
    buf[o] = d2[0];
    buf[o + 1] = d2[1];
    buf[o + 2] = 0b11_10_01_00;

    let out = dequant_gsq_rco_3p5(&buf, BLOCK_SIZE_GSQ_RCO_3P5 * 2).expect("dequant");
    assert_eq!(out.len(), 128);
    assert!((out[0] - -0.5).abs() < 1e-6, "block 0 d");
    assert!((out[64] - -2.0).abs() < 1e-6, "block 1 d");
    assert!((out[67] - 4.0).abs() < 1e-6, "block 1 code 3 -> +2d");
}

#[test]
fn quantize_dequant_roundtrip_recovers_the_four_level_grid() {
    // `d = max|x|`, so the codebook points are {-d, 0, +d, +2d} and +2d is
    // unreachable by construction. Only {-d, 0, +d} can round-trip exactly;
    // 0.25 exercises a non-grid value and must land within half a step.
    let mut vals = Vec::with_capacity(BLOCK_SIZE_GSQ_RCO_3P5);
    for j in 0..BLOCK_SIZE_GSQ_RCO_3P5 {
        vals.push(match j % 4 {
            0 => -1.0,
            1 => 0.0,
            2 => 1.0,
            _ => 0.25,
        });
    }
    let mut packed = vec![0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    quantize_q2_0_block(&vals, &mut packed).expect("quantize");

    let out = dequant_gsq_rco_3p5(&packed, BLOCK_SIZE_GSQ_RCO_3P5).expect("dequant");
    for (i, (&want, &got)) in vals.iter().zip(out.iter()).enumerate() {
        let tol = if want == 0.25 { 0.25 + 1e-6 } else { 1e-3 };
        assert!(
            (want - got).abs() <= tol,
            "index {i}: want {want}, got {got} (packed {packed:?})"
        );
    }
}

#[test]
fn quantize_dequant_max_error_is_half_the_largest_step() {
    // The grid is {-d, 0, d, 2d} with d = max|x|. The widest gap is 2d
    // (between -d and +d), so worst-case error is bounded by d.
    let n = BLOCK_SIZE_GSQ_RCO_3P5;
    let vals: Vec<f32> = (0..n)
        .map(|j| ((j as f32) / 7.0).sin() * 0.8 + ((j % 5) as f32 - 2.0) * 0.13)
        .collect();
    let mut packed = vec![0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    quantize_q2_0_block(&vals, &mut packed).expect("quantize");
    let out = dequant_gsq_rco_3p5(&packed, n).expect("dequant");

    let d = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (i, (&want, &got)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (want - got).abs() <= d + 1e-6,
            "index {i}: |{want} - {got}| exceeded one grid step d={d}"
        );
    }
}

#[test]
fn zero_weight_count_returns_empty_without_touching_data() {
    assert!(dequant_gsq_rco_3p5(&[], 0).expect("empty").is_empty());
}

#[test]
fn ragged_weight_count_is_rejected() {
    // GGUF rows are block-aligned; a ragged count means the caller mis-sized
    // the tensor and would otherwise read past the buffer.
    let buf = vec![0u8; BLOCK_BYTES_GSQ_RCO_3P5 * 2];
    let err = dequant_gsq_rco_3p5(&buf, 65).expect_err("65 is not a multiple of 64");
    assert!(format!("{err}").contains("multiple of block size"));
}

#[test]
fn short_buffer_is_rejected() {
    let buf = vec![0u8; BLOCK_BYTES_GSQ_RCO_3P5 - 1];
    let err = dequant_gsq_rco_3p5(&buf, BLOCK_SIZE_GSQ_RCO_3P5).expect_err("short buffer");
    assert!(format!("{err}").contains("too short"));
}

#[test]
fn all_zero_block_has_no_representable_scale_on_encode() {
    let vals = vec![0.0f32; BLOCK_SIZE_GSQ_RCO_3P5];
    let mut packed = vec![0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    let err = quantize_q2_0_block(&vals, &mut packed).expect_err("all-zero block");
    assert!(format!("{err}").contains("all zeros"));
}

// ---- Real checkpoint bytes --------------------------------------------

/// Read the first few tag-42 blocks of a real expert tensor out of the GSQ-RCO
/// checkpoint and assert they decode to finite, sane, non-degenerate weights.
///
/// This is the test that would have caught the wrong block layout. Under the
/// 256-weight/72-byte reading the same bytes decode as
/// `d * sub_scale * q - dmin` with one shared `dmin` across all sub-blocks,
/// which on real data yields NaN and `absmax` in the tens of thousands.
#[test]
fn real_checkpoint_blocks_decode_to_finite_sane_weights() {
    use grim_format::gguf::read_gguf;
    use std::io::BufReader;
    use std::path::Path;

    let path = "models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf";
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present");
        return;
    }

    let f = std::fs::File::open(path).expect("open checkpoint");
    let gguf = read_gguf(BufReader::new(f)).expect("parse gguf header");

    let t = gguf
        .tensors
        .iter()
        .find(|t| t.dtype == GgufDType::GsqRco3p5 && t.name.ends_with("ffn_down_exps.weight"))
        .expect("a tag-42 ffn_down_exps tensor must exist");

    let nblocks = (t.size_bytes as usize) / BLOCK_BYTES_GSQ_RCO_3P5;
    const NB: usize = 512;
    assert!(
        nblocks >= NB,
        "need at least {NB} blocks, tensor has {nblocks}"
    );

    // `get_packed` hands back the raw slice for this tensor without dequantizing.
    let provider = grim_format::tprov::GgufProvider::open(path).expect("open provider");
    let raw = provider
        .get_packed(&t.name)
        .expect("packed bytes for tag-42 tensor");

    let mut all = Vec::with_capacity(NB * BLOCK_SIZE_GSQ_RCO_3P5);
    for b in 0..NB {
        let s = b * BLOCK_BYTES_GSQ_RCO_3P5;
        all.extend(
            dequant_gsq_rco_3p5(
                &raw.bytes[s..s + BLOCK_BYTES_GSQ_RCO_3P5],
                BLOCK_SIZE_GSQ_RCO_3P5,
            )
            .expect("real block dequant"),
        );
    }

    assert!(
        all.iter().all(|v| v.is_finite()),
        "real Q2_0 weights must all be finite; got NaN/inf"
    );
    let absmax = all.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mean_abs = all.iter().map(|v| v.abs()).sum::<f32>() / all.len() as f32;
    assert!(
        absmax < 1.0,
        "Q2_0 expert weights should be small; absmax was {absmax} \
         (a wrong block layout produces values in the tens of thousands)"
    );
    assert!(
        mean_abs > 1e-5,
        "mean |w| was {mean_abs} — blocks decoded to near-constant garbage"
    );

    // Per-block 4-level grid: the distinct nonzero magnitudes in a single block
    // must be exactly {d, 2d}. This catches a layout that yields a plausible
    // range but the wrong internal structure.
    let first = &all[..BLOCK_SIZE_GSQ_RCO_3P5];
    let d = -first[0]
        .abs()
        .max(first.iter().fold(0.0f32, |m, v| m.max(v.abs())));
    let mags: std::collections::BTreeSet<u32> = first
        .iter()
        .filter(|v| v.abs() > 1e-9)
        .map(|v| (v.abs() / d).round() as u32)
        .collect();
    assert_eq!(
        mags,
        [1u32, 2]
            .into_iter()
            .collect::<std::collections::BTreeSet<u32>>(),
        "block 0 magnitudes (in units of d) should be exactly {{d, 2d}}, got {mags:?}"
    );
}
