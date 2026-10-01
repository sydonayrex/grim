//! `layer_out_norm` -- a third per-layer norm, applied AFTER the residual sum.
//!
//! ## The reference fact
//!
//! `bert.cpp:209` is the only place in the reference tree that applies it:
//!
//! ```text
//! // attentions bypass the intermediate layer
//! cur = ggml_add(ctx0, cur, ffn_inp);      // :206
//!
//! // output layer norm
//! cur = build_norm(cur, model.layers[il].layer_out_norm,
//!                            model.layers[il].layer_out_norm_b, LLM_NORM, il);  // :209
//! ```
//!
//! Note the position: it normalises the *sum*, not a branch. That is the
//! opposite of `attn_post_norm`, which `olmo2.cpp:157-162` applies to the
//! attention output *before* its add. Both are "post-attention" by name and
//! they are on opposite sides of the residual, so they cannot share a rule --
//! which is why a single `attn_post_norm` field cannot express this.
//!
//! ## The rest of the tree
//!
//! Nine references create `LLM_TENSOR_LAYER_OUT_NORM`, and only `bert`
//! applies it:
//!
//! * `jina-bert-v2`, `jina-bert-v3`, `nomic-bert`, `nomic-bert-moe` create it
//!   and have their own `build_arch_graph`; none calls `build_norm` with it.
//!   They do NOT inherit `llama_model_bert` -- every one derives from
//!   `llama_model_base` (`models.h`).
//! * `grok`, `bailingmoe2`, `bailingmoe3` create it and never reference it
//!   again -- dead upstream too.
//! * `mimo2.cpp:385` reads it, but as an MTP head-norm fallback:
//!   `layer.layer_out_norm ? layer.layer_out_norm : model.output_norm`, for
//!   the next-token head, not the main forward graph.
//! * `gemma3n`, `gemma4` do not have it.
//!
//! So the forward-graph scope is one architecture, and that is the fact worth
//! pinning: a grep says nine, and reproducing nine would be wrong.
//!
//! ## What this test does
//!
//! Reads the references and asserts the count, so the classification cannot
//! rot when llama.cpp changes. Newlines are squeezed before matching because
//! `build_norm(` calls wrap over several lines -- the same undercount that
//! made a related list read 12 instead of 30.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("old/repo/llama.cpp-master/src/models")
}

/// Every reference file, newline-squeezed.
fn flat_sources() -> Vec<(String, String)> {
    let dir = reference_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("cpp") {
            continue;
        }
        if let Ok(raw) = std::fs::read_to_string(&path) {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            out.push((name, raw.replace('\n', " ")));
        }
    }
    out
}

#[test]
fn the_reference_directory_is_readable() {
    assert!(
        reference_dir().exists(),
        "{} missing; this test cannot check anything without the reference",
        reference_dir().display()
    );
    assert!(
        flat_sources().len() > 100,
        "only {} reference files read; the parser is broken",
        flat_sources().len()
    );
}

#[test]
fn only_bert_applies_layer_out_norm_in_its_forward_graph() {
    let mut applied: BTreeSet<String> = BTreeSet::new();
    let mut created: BTreeSet<String> = BTreeSet::new();
    for (name, src) in flat_sources() {
        if src.contains("LLM_TENSOR_LAYER_OUT_NORM") {
            created.insert(name.clone());
        }
        // A call to build_norm that takes layer_out_norm. Everything else that
        // mentions the name is either a create_tensor or mimo2's ternary.
        let mut rest = src.as_str();
        while let Some(i) = rest.find("build_norm(") {
            let seg = &rest[i..];
            let end = seg.find(");").map_or(seg.len(), |e| e + 2);
            let call = &seg[..end];
            rest = &seg[end..];
            if call.contains("layer_out_norm") {
                applied.insert(name.clone());
            }
        }
    }
    assert!(
        created.len() >= 9,
        "only {} references create LAYER_OUT_NORM; expected at least 9",
        created.len()
    );
    assert_eq!(
        applied.iter().cloned().collect::<Vec<_>>(),
        vec!["bert".to_string()],
        "these apply layer_out_norm in a build_norm call: {applied:?}\n\
         The classification is wrong if this moved; re-read the reference and \
         update the doc comment with what changed."
    );
}

#[test]
fn the_bert_call_normalises_the_sum_not_a_branch() {
    // The position is the whole point, and it is the opposite of
    // attn_post_norm. Assert the shape rather than trust the comment: the
    // norm is applied to the value that already carries both residual terms.
    let src = flat_sources();
    let bert = src
        .iter()
        .find(|(n, _)| n == "bert")
        .map(|(_, s)| s.as_str())
        .expect("bert.cpp must be readable");
    // Anchor on the call, not a fragment inside it: the newline squeeze means
    // `cur = build_norm(cur, model.layers[il].layer_out_norm,` is one line.
    let i = bert
        .find("cur = build_norm(cur, model.layers[il].layer_out_norm,")
        .expect("bert calls build_norm with it");
    // The add must be the statement immediately before the norm -- that
    // adjacency is what makes this a norm of a SUM rather than of a branch.
    // Asserting the absence of a build_ffn in between is the substantive
    // part; a character count would be a reformat-detector, not a check.
    let add_at = bert[..i]
        .rfind("ggml_add")
        .expect("a residual add precedes the norm");
    let between = &bert[add_at..i];
    assert!(
        !between.contains("build_ffn"),
        "the add is not adjacent to the norm, so this is not asserting what \
         it claims: {between:?}"
    );
    // And the norm's own input is the summed `cur`, not a named branch like
    // `ffn_out` or `attn_out`.
    let call = &bert[i..i + 140];
    assert!(
        call.trim_start().starts_with("cur = build_norm(cur,"),
        "expected build_norm applied to `cur`; got: {}",
        &call[..60.min(call.len())]
    );
    // And it is LayerNorm, not RMS.
    let after = &bert[i..i + 200];
    assert!(
        after.contains("LLM_NORM,"),
        "bert's layer_out_norm must be LLM_NORM (LayerNorm)"
    );
}