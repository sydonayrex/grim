//! How many architectures get `BertBlock`, and what the loader discards.
//!
//! ## Why this file exists
//!
//! `bert_reference_order.rs` records that `BertBlock` differs from
//! `bert.cpp`. That is only a live defect if something constructs it, so this
//! file measures the blast radius from `model_loader.rs` and pins it.
//!
//! It reads the loader as TEXT rather than calling it, because the routing is
//! a `match` over `ModelArchitecture` and the honest question -- "which
//! architectures reach `Bert::load_tp`" -- is answered by reading the arms.
//! A test that loaded six models would need six checkpoints on disk and would
//! still not fail if an arm were re-pointed without any of them being run.
//!
//! ## What it pins
//!
//! 1. `Bert::load_tp` is constructed in six production sites across two load
//!    paths, and the architectures those sites serve. Adding a seventh must
//!    change this count deliberately.
//! 2. `ModelArchitecture::Bert` is not routed at all, so a plain BERT
//!    checkpoint does not reach `BertBlock`. If someone routes it, that is a
//!    change to this surface and the count above moves.
//! 3. `modern_bert_cfg` and `nomic_bert_cfg` are built and then only logged;
//!    the `layer_norm_eps` the GGUF supplied is dropped, because
//!    `BertConfig` has no epsilon field at all. Two per load path.
//!
//! None of this is fixed here. Point 3 needs a `layer_norm_eps` on
//! `BertConfig` and a decision about which architecture's epsilon wins when
//! several share one arm, which is a design question rather than a rename.

use std::collections::BTreeSet;

fn loader() -> &'static str {
    include_str!("../../../grim-engine/src/model_loader.rs")
}

fn bert_rs() -> &'static str {
    include_str!("../src/bert.rs")
}

/// Every `Bert::load_tp(` in the loader, with the enclosing function.
fn bert_sites() -> Vec<(usize, String)> {
    loader()
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("Bert::load_tp("))
        .map(|(i, _)| (i + 1, enclosing_fn(i + 1)))
        .collect()
}

/// The nearest `fn` declaration at or above `line` (1-based).
fn enclosing_fn(line: usize) -> String {
    let lines: Vec<&str> = loader().lines().collect();
    for i in (0..line.min(lines.len())).rev() {
        let t = lines[i].trim_start();
        if t.starts_with("fn ") || t.starts_with("pub fn ") {
            return t
                .trim_start_matches("pub ")
                .trim_start_matches("fn ")
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("?")
                .to_string();
        }
    }
    "?".to_string()
}

#[test]
fn six_production_sites_construct_bert() {
    let sites = bert_sites();
    assert_eq!(
        sites.len(),
        6,
        "Bert::load_tp is now constructed in {} sites: {sites:?}\\n\\
         That is a deliberate routing change -- update the architecture list \\
         in this file and the plan's 4.9 entry with it.",
        sites.len()
    );
    // Every site must be in production code, not a test module: a count that
    // includes #[cfg(test)] copies is a count of the wrong thing.
    for (line, f) in &sites {
        assert!(
            !f.starts_with("test") && !f.contains("spec_"),
            "site at line {line} is inside {f}, not production code"
        );
    }
    // And they must span both load paths, or the safetensors path is untested
    // by any of this.
    let fns: BTreeSet<String> = sites.iter().map(|(_, f)| f.clone()).collect();
    assert!(
        fns.contains("load_model_with_providers") && fns.contains("load_model_from_config"),
        "expected both load paths to construct Bert, found {fns:?}"
    );
}

#[test]
fn model_architecture_bert_is_not_routed() {
    // A plain BERT checkpoint therefore does NOT reach BertBlock. Recorded
    // because bert.cpp is the one reference that actually applies
    // layer_out_norm, so this is the arm most worth checking next -- and it
    // is not this one.
    let src = loader();
    assert!(
        !src.contains("ModelArchitecture::Bert =>")
            && !src.contains("| ModelArchitecture::Bert "),
        "ModelArchitecture::Bert is now routed; if that is deliberate, BertBlock \\
         becomes live for plain BERT and the six sites above are no longer the \\
         whole blast radius"
    );
}

#[test]
fn the_loader_builds_two_configs_it_then_discards() {
    // `modern_bert_cfg` and `nomic_bert_cfg` carry a `layer_norm_eps` the
    // BertConfig passed to load_tp has no field for, and are only logged. The
    // GGUF's epsilon is read and dropped.
    let src = loader();
    for name in ["modern_bert_cfg", "nomic_bert_cfg"] {
        let uses: Vec<usize> = src
            .match_indices(name)
            .map(|(i, _)| i)
            .collect();
        // Two per load path: the declaration and the log. If a use appears
        // beyond those, the value is no longer discarded and the epsilon
        // question has been answered.
        assert_eq!(
            uses.len(),
            4,
            "{name} now has {} references, so it may be read rather than only \\
             logged: {uses:?}",
            uses.len()
        );
        let lines: Vec<&str> = src.lines().collect();
        for u in &uses {
            let ln = src[..*u].matches('\n').count();
            let text = lines[ln].trim();
            assert!(
                text.starts_with("let ") || text == name,
                "{name} is used at line {} beyond its declaration and the log: \\
                 {text:?}",
                ln + 1
            );
        }
    }
}

#[test]
fn bert_config_has_no_epsilon_field() {
    // The reason the two configs above are discarded: there is nowhere to
    // put the value. If a `layer_norm_eps` appears here, the loader should be
    // passing it and point 3 above is answered.
    let src = bert_rs();
    let start = src.find("pub struct BertConfig").expect("BertConfig must exist");
    let body_end = src[start..].find("\n}").expect("BertConfig must be closed");
    let body = &src[start..start + body_end];
    assert!(
        !body.contains("eps"),
        "BertConfig now has an epsilon field: {}",
        body.lines()
            .map(str::trim)
            .find(|l| l.contains("eps"))
            .unwrap_or("")
    );
    assert!(
        body.contains("pub max_seq_len"),
        "sanity: the field scan above is reading BertConfig, not something else"
    );
}
