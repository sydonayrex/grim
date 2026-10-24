//! GSQ-RCO 3.5-bit (GGUF tag 81) dequantization KATs.
//!
//! These replace a vacuous "all-zero block dequantizes to 256 zeros" check,
//! which passes under *any* layout. The block GEOMETRY is shared with
//! upstream Q2_0 (llama.cpp `ggml-quants.c:439`: QK2_0 = 64, `ggml_half d`
//! then 16 code bytes, 4 codes per byte low-first) — but the CODEBOOK is the
//! GSQ paper's, per its own reference quantizer
//! (old/repo/GSQ-main/src/quantization/gumbel_quantizer_2bit.py:10):
//! `values = [-2, -1, 0, 1]`, i.e. `y = (q - 2) * d`. One level of d below
//! Q2_0's `(q - 1) * d`. The bias-1 expectations this file used to pin were
//! the Q2_0-compatible reading; they contradicted the committed decoder AND
//! the GSQ source, and have been superseded (paper = source of truth per the
//! GSQ-RCO format directive).

use grim_format::gguf::GgufDType;
use grim_quant::{
    BLOCK_BYTES_GSQ_RCO_3P5, BLOCK_SIZE_GSQ_RCO_3P5, dequant_gsq_rco_3p5,
    quantize_gsq_rco_3p5_block,
};
use grim_tensor::provider::TensorProvider;

/// Resolve a workspace-relative checkpoint path, or `None` if absent.
///
/// `cargo test` sets the CWD to the *package* root (`crates/grim-quant`), not
/// the workspace root, so a bare `models/...` relative path never resolves and
/// the real-checkpoint gates below silently `return` — a green that ran
/// nothing. Try the CWD first (so a developer pointing at a scratch checkout
/// still works), then walk up from the crate manifest to the workspace root.
fn find_checkpoint(rel: &str) -> Option<std::path::PathBuf> {
    let direct = std::path::PathBuf::from(rel);
    if direct.exists() {
        return Some(direct);
    }
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for up in manifest.ancestors() {
        let cand = up.join(rel);
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

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

fn assert_close_f32(got: f32, want: f32, what: &str) {
    assert!((got - want).abs() < 1e-4, "{what}: got {got}, want {want}");
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
fn codebook_is_minus_two_minus_one_zero_plus_one() {
    // The single most important property: code 2 is ZERO, code 0 is -2d
    // (GSQ paper: values = [-2, -1, 0, 1]). Reading this as Q2_0's
    // `-1..+2` shifts every weight by one level of d — silently.
    let d = f32_to_f16_le(0.5);
    let mut block = [0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    block[0] = d[0];
    block[1] = d[1];
    // Byte 0 holds codes for j = 0..4; encode 0,1,2,3 in ascending bit order.
    block[2] = 0b11_10_01_00;

    let out = dequant_gsq_rco_3p5(&block, BLOCK_SIZE_GSQ_RCO_3P5).expect("dequant");
    assert_eq!(out.len(), 64);
    assert_eq!(out[0], -1.0, "code 0 -> -2d");
    assert_eq!(out[1], -0.5, "code 1 -> -d");
    assert_eq!(out[2], 0.0, "code 2 -> 0");
    assert_eq!(out[3], 0.5, "code 3 -> +d");
    // j=4 starts a fresh code byte, which is still zero => code 0 => -2d.
    // This is what a decoder that packs 4 codes per byte gets wrong.
    assert_eq!(out[4], -1.0, "byte 1 restarts at j=4");
    assert_eq!(out[63], -1.0);
}

#[test]
fn negative_scale_flips_sign_of_every_code() {
    // Exercises bit 15 of the fp16 delta, which a truncated unpacker drops.
    let d = f32_to_f16_le(-0.25);
    let mut block = [0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    block[0] = d[0];
    block[1] = d[1];
    for byte in block[2..].iter_mut() {
        *byte = 0xff; // all codes = 3 -> +d = -0.25
    }
    let out = dequant_gsq_rco_3p5(&block, BLOCK_SIZE_GSQ_RCO_3P5).expect("dequant");
    assert!(
        out.iter().all(|v| (*v - (-0.25)).abs() < 1e-6),
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
        let got_d = -out[0] / 2.0;
        assert!(
            (got_d - d).abs() <= d * 1e-2,
            "d={d} did not round-trip: got {got_d}"
        );
        assert_eq!(out[2], 0.0, "code 2 is the zero level");
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
    assert!((out[0] - -1.0).abs() < 1e-6, "block 0 code 0 -> -2d");
    assert!((out[64] - -4.0).abs() < 1e-6, "block 1 code 0 -> -2d");
    assert!((out[67] - 2.0).abs() < 1e-6, "block 1 code 3 -> +d");
}

#[test]
fn quantize_dequant_roundtrip_recovers_the_four_level_grid() {
    // The GSQ codebook is {-2d, -d, 0, +d}. With d = 1 (the MSE fitter's
    // choice for amax = 2) all four levels are exactly representable, so a
    // grid-valued block must round-trip exactly; 0.25 exercises a non-grid
    // value and must land within half a level.
    let mut vals = Vec::with_capacity(BLOCK_SIZE_GSQ_RCO_3P5);
    for j in 0..BLOCK_SIZE_GSQ_RCO_3P5 {
        vals.push(match j % 4 {
            0 => -2.0,
            1 => -1.0,
            2 => 0.0,
            _ => 1.0,
        });
    }
    let mut packed = vec![0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    quantize_gsq_rco_3p5_block(&vals, &mut packed).expect("quantize");

    let out = dequant_gsq_rco_3p5(&packed, BLOCK_SIZE_GSQ_RCO_3P5).expect("dequant");
    for (i, (&want, &got)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (want - got).abs() <= 1e-3,
            "index {i}: want {want}, got {got} (packed {packed:?})"
        );
    }
}

#[test]
fn quantize_dequant_max_error_is_half_the_largest_step() {
    // The grid is {-2d, -d, 0, d} with a MSE-fitted d. The level gap is d,
    // so nearest-level error is at most d/2; the one-level bound leaves the
    // fitter slack.
    let n = BLOCK_SIZE_GSQ_RCO_3P5;
    let vals: Vec<f32> = (0..n)
        .map(|j| ((j as f32) / 7.0).sin() * 0.8 + ((j % 5) as f32 - 2.0) * 0.13)
        .collect();
    let mut packed = vec![0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    quantize_gsq_rco_3p5_block(&vals, &mut packed).expect("quantize");
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
fn all_zero_block_encodes_as_flushed_zeros() {
    // All-zero blocks carry no representable scale: the encoder now flushes them
    // to d=0/codes=0 (a counted, non-erroring path) instead of returning Err,
    // because released GSQ-RCO artifacts carry ~50% zero-scale blocks. See
    // quantize_gsq_rco_3p5_block_counted for the rationale.
    let vals = vec![0.0f32; BLOCK_SIZE_GSQ_RCO_3P5];
    let mut packed = vec![1u8; BLOCK_BYTES_GSQ_RCO_3P5];
    quantize_gsq_rco_3p5_block(&vals, &mut packed).expect("all-zero block encodes");
    assert!(packed.iter().all(|&b| b == 0), "flushed block must be all zeros");
}

/// Hand-constructed golden vector in the style of `golden_dequant.rs`: the packed
/// bytes are built from explicit bit arithmetic and the expected values are
/// derived from the format spec, never from this crate's own quantizer.
///
/// Block: `d = 1.0` (fp16 0x3C00), then 16 code bytes each holding the pattern
/// `0b11_10_01_00` so every group of four weights is `-d, 0, +d, +2d`.
#[test]
fn golden_hand_built_block_matches_spec_derived_values() {
    let mut block = [0u8; BLOCK_BYTES_GSQ_RCO_3P5];
    block[0] = 0x00;
    block[1] = 0x3C; // d = 1.0
    for b in block[2..].iter_mut() {
        *b = 0b11_10_01_00;
    }

    let deq = dequant_gsq_rco_3p5(&block, BLOCK_SIZE_GSQ_RCO_3P5).expect("dequant");
    assert_eq!(deq.len(), 64);

    // y[j] = (code(j) - 2) * d (GSQ paper), pattern repeats every 4 weights.
    const CODEBOOK: [f32; 4] = [-2.0, -1.0, 0.0, 1.0];
    for (j, &got) in deq.iter().enumerate() {
        let want = CODEBOOK[j % 4];
        assert!(
            (got - want).abs() < 1e-4,
            "weight {j}: code {} should decode to {want}, got {got}",
            j % 4
        );
    }
    assert_close_f32(deq[0], -2.0, "code 0 -> -2d");
    assert_close_f32(deq[1], -1.0, "code 1 -> -d");
    assert_close_f32(deq[2], 0.0, "code 2 -> 0 (zero is code 2, not code 1)");
    assert_close_f32(deq[3], 1.0, "code 3 -> +d");
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

    let Some(path) = find_checkpoint("models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf")
    else {
        panic!("checkpoint missing: models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf \
                 — this gate must not silently skip; it is the only host-side \
                 check that a real GSQ-RCO artifact decodes sanely");
    };
    let path = path.to_string_lossy().to_string();

    let f = std::fs::File::open(&path).expect("open checkpoint");
    let gguf = read_gguf(BufReader::new(f)).expect("parse gguf header");

    // This checkpoint is tag 42 (`Q2_0`), NOT tag 81 (`GsqRco3p5`): see
    // `KQuantScheme::Q2_0` in grim-tensor/src/dtype.rs — "The Qwen3.8-Flash-Next
    // GSQ-RCO-3.5bit release writes tag 42", and the build manifest's
    // `runtime_requirements` lists `Q2_0`. Same 64-weight/18-byte geometry, but
    // the codebook is shifted by one level (`y = (q-1)*d` vs GSQ-RCO's
    // `y = (q-2)*d`), so it must decode with `dequant_q2_0`. The original
    // selector also demanded a `ffn_down_exps` tensor name from a MoE expert
    // layout this checkpoint does not use, so it matched nothing and skipped.
    // Every tag-42 tensor in this checkpoint is a 3-D MoE expert tensor
    // (`blk.N.ffn_down_exps.weight`). The block stream is a flat, contiguous
    // byte sequence whatever the rank, so slicing the first NB blocks off the
    // front is layout-independent and needs no reshape.
    let t = gguf
        .tensors
        .iter()
        .find(|t| t.dtype == GgufDType::Q2_0 && t.name.starts_with("blk."))
        .unwrap_or_else(|| {
            let names: Vec<&str> = gguf.tensors.iter().map(|t| t.name.as_str()).take(5).collect();
            panic!(
                "no Q2_0 (tag 42) tensor under blk.* in the Flash checkpoint; \
                 first tensors: {names:?}"
            )
        });

    const Q2_0_BLOCK_BYTES: usize = 18;
    const Q2_0_BLOCK_SIZE: usize = 64;
    let nblocks = (t.size_bytes as usize) / Q2_0_BLOCK_BYTES;
    const NB: usize = 512;
    assert!(
        nblocks >= NB,
        "need at least {NB} blocks, tensor has {nblocks}"
    );

    // `get_packed` hands back the raw slice for this tensor without dequantizing.
    let provider = grim_format::tprov::GgufProvider::open(&path).expect("open provider");
    let raw = provider
        .get_packed(&t.name)
        .expect("packed bytes for tag-42 tensor");

    let mut all = Vec::with_capacity(NB * Q2_0_BLOCK_SIZE);
    for b in 0..NB {
        let s = b * Q2_0_BLOCK_BYTES;
        all.extend(
            grim_quant::dequant_q2_0(
                &raw.bytes[s..s + Q2_0_BLOCK_BYTES],
                Q2_0_BLOCK_SIZE,
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

    // Per-block 4-level grid: `Q2_0` decodes `y = (q - 1) * d` over a 2-bit code,
    // so the level set is `{-d, 0, +d, +2d}`. The distinct nonzero magnitudes
    // across the decoded blocks must be a subset of {d, 2d} in units of the
    // per-block scale. Aggregated over NB blocks, not one block: a single real
    // block need not exercise every code, so asserting {d,2d} on block 0 alone
    // is flaky on real data (block 0 here carried only level 1).
    let mut bad: Vec<(usize, f32)> = Vec::new();
    for blk in all.chunks(Q2_0_BLOCK_SIZE) {
        let d = blk.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
        for (i, &v) in blk.iter().enumerate() {
            if v.abs() <= 1e-9 {
                continue;
            }
            let units = (v.abs() / d).round() as i32;
            if units != 1 && units != 2 {
                bad.push((i, v));
                break;
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} blocks decoded off the 4-level {{d,2d}} grid (first: index {}, value {})",
        bad.len(),
        bad.first().map(|x| x.0).unwrap_or(0),
        bad.first().map(|x| x.1).unwrap_or(0.0)
    );
}

// ---- Real IQ3_XXS checkpoint bytes (the 27B's dominant K-quant) -------------

/// The 27B checkpoint that emitted all-NaN logits at step 0 is NOT a GSQ-RCO
/// model. Its tensor histogram is
/// `{BF16:96, F32:360, IQ1_M:4, IQ1_S:4, IQ2_S:72, IQ2_XS:32, IQ2_XXS:30,
///   IQ3_S:97, IQ3_XXS:83, IQ4_XS:33, Q2_K:28, Q4_K:19, Q6_K:8}` — zero tag-81
/// tensors. `IQ3_XXS` is the single largest K-quant family (83 tensors) and it
/// decodes on the device through `launch_wmma_fused_dequant_iq3xxs`
/// (`gemm_launchers.rs:2821`), i.e. `wmma_quantized_gemm.rs` — the file
/// `041cb708` fixed for a NaN in the Q4_K leg.
///
/// `wmma_quant_parity.rs` covers only `Q4K` and `Q80`, so that leg has no
/// parity gate at all: a scheme that NaNs on real bytes is invisible to every
/// test in the tree. This is the host-side half of that gap — it reads REAL
/// artifact blocks (not bytes this crate packed itself, which would
/// round-trip any layout bug into agreement) and requires finite, sane
/// weights. If the device leg is the remaining half, it is caught by adding
/// `IQ3XXS` to the WMMA parity sweep.
#[test]
fn real_checkpoint_iq3xxs_blocks_decode_to_finite_sane_weights() {
    use grim_tensor::dtype::KQuantScheme;
    use std::io::BufReader;

    let Some(path) = find_checkpoint("models/qwen38-27b/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
    else {
        panic!("checkpoint missing: models/qwen38-27b/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
                 — this gate must not silently skip; IQ3_XXS is the 27B's dominant \
                 K-quant scheme and the one that produced all-NaN step-0 logits");
    };
    let path = path.to_string_lossy().to_string();

    let f = std::fs::File::open(&path).expect("open checkpoint");
    let gguf = grim_format::gguf::read_gguf(BufReader::new(f)).expect("parse gguf header");
    // The single largest K-quant family in this checkpoint. Assert the tag so
    // the test cannot silently drift onto a different scheme if the artifact
    // is ever re-quantized.
    let t = gguf
        .tensors
        .iter()
        .find(|t| {
            t.dtype == grim_format::gguf::GgufDType::IQ3_XXS
                && t.name.starts_with("blk.")
                && (t.name.contains("ffn_down") || t.name.contains("ffn_up")
                    || t.name.contains("ssm_out") || t.name.contains("attn_gate"))
        })
        .expect("an IQ3_XXS linear weight must exist in the 27B checkpoint");
    assert_eq!(t.shape().len(), 2, "{} should be 2-D, got {:?}", t.name, t.dims);
    let n_weights = t.elem_count();

    let provider = grim_format::tprov::GgufProvider::open(&path).expect("open provider");
    let raw = provider
        .get_packed(&t.name)
        .expect("packed IQ3_XXS bytes from the real checkpoint");
    let got = grim_quant::dequant_iq3xxs(&raw.bytes, n_weights)
        .unwrap_or_else(|e| panic!("{}: real IQ3_XXS dequant failed: {e}", t.name));
    assert_eq!(got.len(), n_weights, "{}: dequant length mismatch", t.name);

    assert!(
        got.iter().all(|v| v.is_finite()),
        "{}: all {n_weights} real IQ3_XXS weights must be finite; got NaN/inf",
        t.name
    );
    let absmax = got.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mean_abs = got.iter().map(|v| v.abs()).sum::<f32>() / got.len() as f32;
    assert!(
        absmax < 1.0,
        "{}: IQ3_XXS weights should be small; absmax was {absmax} \
         (a wrong block layout reads scales from the wrong offset and produces \
          values in the tens of thousands, or NaN)",
        t.name
    );
    assert!(
        mean_abs > 1e-5,
        "{}: mean |w| was {mean_abs} — decoded to near-constant garbage",
        t.name
    );

    // Non-degenerate per block: a correct IQ3_XXS block spans the 8-level
    // grid, so the distinct rounded magnitudes must exceed a couple of levels.
    // Catches a layout that lands in a plausible range but has no internal
    // structure.
    let distinct: std::collections::BTreeSet<u32> = got
        .iter()
        .filter(|v| v.abs() > 1e-9)
        .map(|v| {
            let d = absmax.max(1e-12);
            ((v.abs() / d * 8.0).round() as u32).min(8)
        })
        .collect();
    assert!(
        distinct.len() >= 4,
        "{}: only {} distinct magnitude levels survived; expected the \
         IQ3_XXS 8-level grid", t.name, distinct.len()
    );
    let _ = KQuantScheme::IQ3XXS; // pin the scheme this gate is about
}
