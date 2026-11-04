//! `prism.hadamard.*` GGUF metadata contract tests.
//!
//! Every validation failure below must mirror the PrismML fork's
//! `load_hparams_hadamard` (llama-model.cpp:2090-2262) error strings so a bad
//! checkpoint fails identically in grim. The last test gates against the REAL
//! Bonsai-27B checkpoint (header-only mmap read).

use std::collections::HashMap;

use grim_format::gguf::GgufValue;
use grim_format::prism_hadamard::{parse_prism_hadamard, sign_vector_for};
use grim_format::tprov::GgufProvider;

/// Build a metadata map directly (the parser consumes the map, so no GGUF
/// byte-writing is needed here).
fn meta_with(kv: Vec<(&str, GgufValue)>) -> HashMap<String, GgufValue> {
    kv.into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

fn s(v: &str) -> GgufValue {
    GgufValue::String(v.to_string())
}

fn arr_i32(v: &[i32]) -> GgufValue {
    GgufValue::Array(v.iter().map(|&x| GgufValue::Int32(x)).collect())
}

fn arr_str(v: &[&str]) -> GgufValue {
    GgufValue::Array(v.iter().map(|&x| s(x)).collect())
}

/// A fully valid version-1 explicit-mode map (Bonsai-shaped, small widths).
fn valid_meta() -> HashMap<String, GgufValue> {
    meta_with(vec![
        ("prism.hadamard.version", GgufValue::Uint32(1)),
        ("prism.hadamard.block_size", GgufValue::Uint32(1024)),
        (
            "prism.hadamard.transform",
            s("normalized-sylvester-walsh-hadamard"),
        ),
        ("prism.hadamard.axis", s("input-last-dimension")),
        ("prism.hadamard.sign_mode", s("explicit")),
        ("prism.hadamard.sign_widths", arr_i32(&[2048, 4096])),
        (
            "prism.hadamard.sign_values",
            arr_i32(&vec![1i32; 2048].into_iter().chain(vec![-1i32; 4096]).collect::<Vec<_>>()),
        ),
        (
            "prism.hadamard.weight_names",
            arr_str(&["blk.0.attn_q.weight", "output.weight"]),
        ),
        (
            "prism.hadamard.inverse_weight_names",
            arr_str(&["token_embd.weight"]),
        ),
        ("prism.hadamard.gdn_v_grouped", GgufValue::Bool(true)),
    ])
}

fn has_output_weight(name: &str) -> bool {
    name == "output.weight"
}

#[test]
fn no_hadamard_keys_parses_to_none() {
    let meta = meta_with(vec![("general.architecture", s("qwen35"))]);
    let parsed = parse_prism_hadamard(&meta, &has_output_weight).unwrap();
    assert!(parsed.is_none(), "plain model must parse to Ok(None)");
}

#[test]
fn version_must_be_1_or_2() {
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.version".into(), GgufValue::Uint32(3));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(err.contains("unsupported prism.hadamard.version: 3"), "{err}");
}

#[test]
fn v2_tied_output_contract() {
    // version 2 without tied_output
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.version".into(), GgufValue::Uint32(2));
    meta.remove("prism.hadamard.tied_output");
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(
        err.contains("version 2 requires tied_output=true; version 1 forbids it"),
        "{err}"
    );

    // version 1 with tied_output=true
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.tied_output".into(), GgufValue::Bool(true));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(
        err.contains("version 2 requires tied_output=true; version 1 forbids it"),
        "{err}"
    );

    // tied_output=true + output.weight present -> forbidden
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.version".into(), GgufValue::Uint32(2));
    meta.insert("prism.hadamard.tied_output".into(), GgufValue::Bool(true));
    let has_output = |n: &str| n == "output.weight";
    let err = parse_prism_hadamard(&meta, &has_output).unwrap_err().to_string();
    assert!(
        err.contains("prism.hadamard.tied_output requires output.weight to be absent"),
        "{err}"
    );
}

#[test]
fn block_size_must_be_pow2() {
    for bad in [0u32, 1000, 3072] {
        let mut meta = valid_meta();
        meta.insert("prism.hadamard.block_size".into(), GgufValue::Uint32(bad));
        let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
        assert!(
            err.contains(&format!("invalid prism.hadamard.block_size: {bad}")),
            "{err}"
        );
    }
}

#[test]
fn transform_axis_sign_mode_whitelists() {
    let cases: &[(&str, &str)] = &[
        ("prism.hadamard.transform", "sylvester"),
        ("prism.hadamard.axis", "input-first-dimension"),
        ("prism.hadamard.sign_mode", "implicit"),
    ];
    for (key, bad) in cases {
        let mut meta = valid_meta();
        meta.insert(key.to_string(), s(bad));
        let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
        assert!(
            err.contains(&format!("unsupported {key}: {bad}")),
            "key={key} err={err}"
        );
    }
    // transform must equal the exact string
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.transform".into(), s("Normalized-Sylvester-Walsh-Hadamard"));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(err.contains("unsupported prism.hadamard.transform"), "{err}");
}

#[test]
fn weight_names_must_not_be_empty() {
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.weight_names".into(), arr_str(&[]));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(err.contains("prism.hadamard.weight_names is empty"), "{err}");
}

#[test]
fn explicit_sign_rules() {
    // explicit + empty widths
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.sign_widths".into(), arr_i32(&[]));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(
        err.contains("sign_mode is explicit but sign_widths is empty"),
        "{err}"
    );

    // width 20 with block 1024 (20 % 1024 != 0)
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.sign_widths".into(), arr_i32(&[20]));
    meta.insert("prism.hadamard.sign_values".into(), arr_i32(&vec![1i32; 20]));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(err.contains("invalid prism.hadamard sign width: 20"), "{err}");

    // width 0 / negative
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.sign_widths".into(), arr_i32(&[0, 2048]));
    meta.insert("prism.hadamard.sign_values".into(), arr_i32(&vec![1i32; 2048]));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(err.contains("invalid prism.hadamard sign width: 0"), "{err}");

    // duplicate width
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.sign_widths".into(), arr_i32(&[2048, 2048]));
    meta.insert("prism.hadamard.sign_values".into(), arr_i32(&vec![1i32; 4096]));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(err.contains("duplicate prism.hadamard sign width: 2048"), "{err}");

    // value 0
    let mut meta = valid_meta();
    let mut values = vec![1i32; 2048];
    values[7] = 0;
    meta.insert("prism.hadamard.sign_values".into(), arr_i32(&values));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(err.contains("sign values must be +/-1"), "{err}");

    // widths sum != values.len() (widths themselves all valid)
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.sign_widths".into(), arr_i32(&[2048]));
    meta.insert("prism.hadamard.sign_values".into(), arr_i32(&vec![1i32; 4096]));
    let err = parse_prism_hadamard(&meta, &has_output_weight).unwrap_err().to_string();
    assert!(err.contains("prism.hadamard.sign_values length mismatch"), "{err}");
}

#[test]
fn identity_sign_mode_needs_no_sign_tables() {
    let mut meta = valid_meta();
    meta.insert("prism.hadamard.sign_mode".into(), s("identity"));
    meta.remove("prism.hadamard.sign_widths");
    meta.remove("prism.hadamard.sign_values");
    let m = parse_prism_hadamard(&meta, &has_output_weight).unwrap().expect("some");
    assert_eq!(m.sign_widths, Vec::<i32>::new());
    assert_eq!(m.sign_values, Vec::<i32>::new());
}

#[test]
fn valid_meta_parses_and_lookups_are_exact_match() {
    let meta = valid_meta();
    let m = parse_prism_hadamard(&meta, &has_output_weight).unwrap().expect("some");
    assert_eq!(m.version, 1);
    assert_eq!(m.block_size, 1024);
    assert!(!m.tied_output);
    assert_eq!(m.sign_widths, vec![2048, 4096]);
    assert_eq!(m.weight_names.len(), 2);
    assert_eq!(m.inverse_weight_names, vec!["token_embd.weight"]);
    assert!(m.gdn_v_grouped);

    // exact-match lookup on the full tensor input dim
    assert_eq!(sign_vector_for(&m, 2048).unwrap().len(), 2048);
    assert_eq!(sign_vector_for(&m, 4096).unwrap().len(), 4096);
    // 2048+2048 is NOT a width group: groups are exact-match, not prefix
    assert!(sign_vector_for(&m, 4096 + 4096).is_none());
    // not divisible by block_size
    assert!(sign_vector_for(&m, 1023).is_none());
    // none at all in identity mode
    let none_meta = meta_with(vec![("prism.hadamard.version", GgufValue::Uint32(1))]);
    assert!(parse_prism_hadamard(&none_meta, &has_output_weight).is_err());
}

#[test]
fn bonsai_real_file_parses_exact() {
    // The real checkpoint pins every field. Skip when the file is absent.
    // cargo runs the test binary from the crate dir; resolve to the repo root.
    const MODEL: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/Bonsai-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
    );
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no Bonsai file");
        return;
    }
    let provider = GgufProvider::open(MODEL).unwrap();
    let has_weight = |n: &str| provider.tensors().contains_key(n);
    let m = parse_prism_hadamard(provider.metadata_map(), &has_weight)
        .unwrap()
        .expect("bonsai must carry hadamard");
    assert_eq!(m.version, 1);
    assert_eq!(m.block_size, 1024);
    assert_eq!(m.sign_widths, vec![5120, 6144, 17408]);
    assert_eq!(m.sign_values.len(), 28672);
    assert!(m.sign_values.iter().all(|&v| v == 1 || v == -1));
    assert_eq!(m.weight_names.len(), 401);
    assert_eq!(m.inverse_weight_names, vec!["token_embd.weight"]);
    assert!(m.gdn_v_grouped);
    // Per-tensor lookup by FULL input dim, exact match only:
    assert_eq!(sign_vector_for(&m, 5120).unwrap().len(), 5120);
    assert_eq!(sign_vector_for(&m, 6144).unwrap().len(), 6144);
    assert_eq!(sign_vector_for(&m, 17408).unwrap().len(), 17408);
    // width groups are exact-match, not prefix: a fused 10240 tensor would find no sign vector
    assert!(
        sign_vector_for(&m, 10240).is_none(),
        "width groups are exact-match, not prefix"
    );
}
