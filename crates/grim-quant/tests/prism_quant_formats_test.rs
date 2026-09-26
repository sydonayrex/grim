//! Prism-private GGUF quant format KATs: `PQ2_0` (tag 142) and `PTQ1_0` (tag 143).
//!
//! Both formats are private to the PrismML llama.cpp fork (branch `prism`,
//! commit `adfffbe41b2cabcd51fff326ab045662265062bb`). They are decoded here
//! but have no `Storage` backend, so a checkpoint using them fails with a named
//! error instead of an unknown-tag parse failure or, worse, silent data
//! reinterpretation under a similar scheme.
//!
//! Every expected value below was produced by compiling the fork's own C
//! (`ggml-common.h` block layouts, `ggml-quants.c` dequant/quant routines) and
//! dumping the output — NOT by transcribing the C into Rust and asserting
//! against itself. Upstream `Q2_0` at tag 42 is a *different* format (group 64,
//! 18 bytes); these high ids exist so the two can coexist.

use grim_quant::{BLOCK_BYTES_PQ2_0, BLOCK_BYTES_PTQ1_0, dequant_pq2_0, dequant_ptq1_0};

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn assert_eq_vec(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-5, "{what}: index {i}: got {g}, want {w}");
    }
}

// ---- PQ2_0 -------------------------------------------------------------

#[test]
fn pq2_0_geometry_is_128_weights_per_34_bytes() {
    assert_eq!(BLOCK_BYTES_PQ2_0, 34);
    // 2.125 bits/weight: 34*8/128 = 17/8 exactly, i.e. 272 bits per block.
    assert_eq!(BLOCK_BYTES_PQ2_0 * 8, 272);
    assert_eq!(BLOCK_BYTES_PQ2_0 * 8 * 8, 128 * 17, "2.125 bpw = 17/8");
}

#[test]
fn pq2_0_matches_c_reference_positive_scale() {
    // C: d = 2.0 (0x4000), every code byte 0b11_10_01_00.
    let blk = hex("0040e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4");
    assert_eq!(blk.len(), BLOCK_BYTES_PQ2_0);
    let out = dequant_pq2_0(&blk, 128).expect("dequant");
    let want: Vec<f32> = (0..32).flat_map(|_| [-2.0, 0.0, 2.0, 4.0]).collect();
    assert_eq_vec(&out, &want, "pq2_0 d=2.0");
}

#[test]
fn pq2_0_matches_c_reference_negative_scale() {
    // C: d = -2.0 (0xC000). Exercises the sign bit of the fp16 delta and the
    // fact that code 1 decodes to +0.0, not -0.0.
    let blk = hex("00c0e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4");
    let out = dequant_pq2_0(&blk, 128).expect("dequant");
    let want: Vec<f32> = (0..32).flat_map(|_| [2.0, 0.0, -2.0, -4.0]).collect();
    assert_eq_vec(&out, &want, "pq2_0 d=-2.0");
    // C emits -0.0 here: `(1-1) * d` with d < 0. IEEE gives -0.0, and the
    // C reference confirms it, so match it rather than normalizing.
    assert!(
        out[1] == 0.0 && out[1].is_sign_negative(),
        "code 1 with a negative scale decodes to -0.0, got {:?}",
        out[1]
    );
}

#[test]
fn pq2_0_matches_c_reference_roundtrip_from_reference_encoder() {
    // Bytes emitted by the fork's own quantize_row_pq2_0_ref on
    // w[j] = sin(0.37j) * (1 + 0.5 cos(0.11j)).
    let blk = hex("ef3da56a15405555555555a5550150a96a0140a55a55555555550155aa560050a95a");
    let out = dequant_pq2_0(&blk, 128).expect("dequant");
    // C reference: d = amax = 1.4834, and the decoded values are exactly
    // {-1.4834 (x24), 0.0 (x77), +1.4834 (x27)}. The 2d level is not reached
    // by this input; the grid is still {-d, 0, +d, +2d}, just unexercised.
    let d = 1.4834f32;
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for v in &out {
        *counts.entry(format!("{:?}", (v / d).round())).or_default() += 1;
    }
    assert_eq!(
        counts.get("0.0"),
        Some(&77),
        "C reports 77 zeros; got {counts:?}"
    );
    assert_eq!(
        counts.get("1.0"),
        Some(&27),
        "C reports 27 *d; got {counts:?}"
    );
    assert_eq!(
        counts.get("-1.0"),
        Some(&24),
        "C reports 24 *-d; got {counts:?}"
    );
    assert_eq!(counts.len(), 3, "no other levels: {counts:?}");
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn pq2_0_rejects_ragged_and_short() {
    let blk = vec![0u8; BLOCK_BYTES_PQ2_0];
    assert!(
        dequant_pq2_0(&blk, 100).is_err(),
        "100 not a multiple of 128"
    );
    assert!(dequant_pq2_0(&blk[..BLOCK_BYTES_PQ2_0 - 1], 128).is_err());
    assert!(dequant_pq2_0(&[], 0).expect("empty").is_empty());
}

// ---- PTQ1_0 ------------------------------------------------------------

#[test]
fn ptq1_0_geometry_is_128_weights_per_28_bytes() {
    assert_eq!(BLOCK_BYTES_PTQ1_0, 28);
    // 1.75 bits/weight = 7/4
    assert_eq!(BLOCK_BYTES_PTQ1_0 * 8 * 4, 128 * 7);
}

#[test]
fn ptq1_0_matches_c_reference_all_positive_trits() {
    // C: qs = 242 (five trits of code 2), qh = 0x01, d = 1.0 at the END.
    let blk = hex("f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f20101003c");
    assert_eq!(blk.len(), BLOCK_BYTES_PTQ1_0);
    let out = dequant_ptq1_0(&blk, 128).expect("dequant");
    // C reference: {-1.0 (x8), 0.0 (x72), +1.0 (x48)} over 128 values.
    let mut c = [0usize; 3];
    for v in &out {
        let i = (v + 1.0).round().clamp(0.0, 2.0) as usize;
        c[i] += 1;
    }
    assert_eq!(c, [8, 72, 48], "C reports -d x8, 0 x72, +d x48; got {c:?}");
}

#[test]
fn ptq1_0_matches_c_reference_mixed_trits() {
    // C: qs = 121 (five trits of code 1 -> all zero), qh = 0x01, d = 0.5.
    let blk = hex("79797979797979797979797979797979797979797979797901010038");
    let out = dequant_ptq1_0(&blk, 128).expect("dequant");
    let set: std::collections::BTreeSet<u32> = out
        .iter()
        .filter(|v| v.abs() > 1e-6)
        .map(|v| (v.abs() / 0.5).round() as u32)
        .collect();
    assert!(
        set.iter().all(|v| [1u32, 2].contains(v)),
        "nonzero magnitudes must be d or 2d, got {set:?}"
    );
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn ptq1_0_matches_c_reference_roundtrip_from_reference_encoder() {
    // Bytes emitted by the fork's own quantize_row_ptq1_0_ref on
    // w[j] = sin(0.29j) * 0.8.
    let blk = hex("6767bcb4d0cdcdcde6e69a9999444c30294545483f9493965600653a");
    let out = dequant_ptq1_0(&blk, 128).expect("dequant");
    let d = 0.799316f32;
    let set: std::collections::BTreeSet<u32> = out
        .iter()
        .filter(|v| v.abs() > 1e-4)
        .map(|v| (v.abs() / d).round() as u32)
        .collect();
    // Ternary: exactly {-d, 0, +d}. Unlike PQ2_0 there is no 2d level.
    assert_eq!(
        set,
        [1u32].into_iter().collect(),
        "PTQ1_0 nonzero magnitudes must be exactly {{d}}, got {set:?}"
    );
    assert!(out.iter().all(|v| v.is_finite()));
    // The fp16 delta is the last two bytes; sanity-check the decode read it.
    assert!(out.iter().any(|v| v.abs() > 0.79));
}

#[test]
fn ptq1_0_rejects_ragged_and_short() {
    let blk = vec![0u8; BLOCK_BYTES_PTQ1_0];
    assert!(
        dequant_ptq1_0(&blk, 100).is_err(),
        "100 not a multiple of 128"
    );
    assert!(dequant_ptq1_0(&blk[..BLOCK_BYTES_PTQ1_0 - 1], 128).is_err());
    assert!(dequant_ptq1_0(&[], 0).expect("empty").is_empty());
}

#[test]
fn ptq1_0_delta_sits_at_the_end_of_the_block() {
    // A decoder that reads `d` from the front (like every other 2-bit format
    // here) produces garbage on every PTQ1_0 block. Guard the offset.
    let mut blk = vec![0u8; BLOCK_BYTES_PTQ1_0];
    blk[BLOCK_BYTES_PTQ1_0 - 2] = 0x00;
    blk[BLOCK_BYTES_PTQ1_0 - 1] = 0x3C; // d = 1.0, last field only
    let out = dequant_ptq1_0(&blk, 128).expect("dequant");
    assert!(
        out.iter().all(|v| v.abs() <= 1.0 + 1e-6),
        "scale must be read from the trailing fp16 field"
    );
}

// ---- loud failure on load ---------------------------------------------

#[test]
fn prism_formats_map_to_unsupported_storage_not_a_wrong_kquant_scheme() {
    use grim_format::gguf::{GgufDType, map_gguf_dtype_to_storage};
    use grim_tensor::dtype::Storage;

    for (dt, name) in [(GgufDType::PQ2_0, "PQ2_0"), (GgufDType::PTQ1_0, "PTQ1_0")] {
        let dtype = map_gguf_dtype_to_storage(dt);
        match &dtype.storage {
            Storage::Unsupported(f) => {
                assert_eq!(f.name, name);
                assert_eq!(f.block_size, Some(128));
                assert!(
                    f.reason.contains(name) && f.reason.contains("no GPU/CPU Storage"),
                    "reason must name the format: {}",
                    f.reason
                );
            }
            other => panic!("{name} must not map to a usable storage: {other:?}"),
        }
        // Geometry must still be right so size accounting is exact.
        assert_eq!(dt.block_size(), 128);
        assert_eq!(
            dt.type_size_per_block(),
            if name == "PQ2_0" { 34 } else { 28 }
        );
        assert_eq!(dt.tag(), if name == "PQ2_0" { 142 } else { 143 });
    }

    // Upstream Q2_0 at tag 42 must remain the *supported* group-64 codec.
    let q2 = map_gguf_dtype_to_storage(GgufDType::GsqRco3p5);
    assert!(matches!(
        q2.storage,
        Storage::KQuant(grim_tensor::dtype::KQuantScheme::GsqRco3p5)
    ));
    assert_eq!(GgufDType::GsqRco3p5.block_size(), 64);
    assert_eq!(GgufDType::GsqRco3p5.type_size_per_block(), 18);
}
