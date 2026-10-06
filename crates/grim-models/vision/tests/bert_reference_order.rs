//! `BertBlock`'s residual order and norm kinds, against `bert.cpp`.
//!
//! ## The reference, per layer
//!
//! ```text
//! cur = build_attn(...)                       // :139
//! cur = ggml_add(ctx0, cur, inpL);            // :151  re-add the layer input
//! cur = build_norm(cur, attn_out_norm, ...);   // :154  LLM_NORM
//! if (attn_norm_2) {                          // :156
//!     cur = ggml_add(ctx0, cur, inpL);        // :157  re-add AGAIN
//!     cur = build_norm(cur, attn_norm_2, ...); // :158  LLM_NORM
//! }
//! ffn_inp = cur;
//! cur = build_ffn(cur, ...);                   // :164+
//! cur = ggml_add(ctx0, cur, ffn_inp);          // :206  "attentions bypass"
//! cur = build_norm(cur, layer_out_norm, ...);  // :209  LLM_NORM
//! ```
//!
//! Three things follow, and `BertBlock` gets all three wrong:
//!
//! 1. **Post-norm, with a double residual.** The attention output is added to
//!    `inpL` and *that sum* is normalised; then, when `attn_norm_2` exists
//!    (it is conditional -- `bert.cpp:156` guards it on a non-null tensor),
//!    `inpL` is added a second time and normalised again. `BertBlock` has one
//!    `attention_ln` over one add, so the second is always missing.
//! 2. **`layer_out_norm` normalises the sum** -- `x + attn(ln(x))` -- so it
//!    sees both residual terms. `BertBlock.output_ln` does see its sum, which
//!    is the one part that matches.
//! 3. **Every norm is `LLM_NORM` (LayerNorm).** `BertBlock` uses `RmsNorm`
//!    for both. LayerNorm subtracts the mean, so on any input with a
//!    non-zero mean the two disagree.
//!
//! ## Why this is a finding and not a fix
//!
//! `BertBlock` is in `grim-models-vision`, is not a `LlamaBlock`, and does not
//! go through the norm-kind machinery from `568c67b0`. Correcting it is a
//! rewrite of another crate's forward pass, which is a different piece of work
//! from anything in plan item 5. These tests pin the discrepancy so it cannot
//! be mistaken for a solved surface.

use grim_models_vision::bert::{BertBlock, BertConfig};

fn block(hidden: usize, inter: usize, heads: usize) -> BertBlock {
    let mut rng = grim_core::rng::SimpleRng::new(0x5eed);
    BertBlock::from_rng(
        &mut rng,
        &BertConfig {
            vocab_size: 16,
            hidden_size: hidden,
            num_heads: heads,
            num_layers: 1,
            intermediate_size: inter,
            max_seq_len: 8,
        },
    )
}

#[test]
fn bertblock_runs_and_is_finite() {
    // The baseline: the block is constructible and its forward is usable, so
    // the structural assertions below are about the SHAPE, not a crash.
    let h = 4usize;
    let b = block(h, 8, 2);
    let x = grim_backend_cpu::cpu_tensor(
        (0..h).map(|i| i as f32 * 0.5 - 0.75).collect::<Vec<f32>>(),
        grim_tensor::Shape::new(vec![1, h]),
    );
    let got = b
        .forward(&x)
        .expect("forward")
        .to_vec_f32()
        .expect("readable");
    assert_eq!(got.len(), h, "output must keep the hidden size");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "BertBlock produced a non-finite value: {got:?}"
    );
}

#[test]
fn bertblock_declares_two_per_layer_norms_and_the_reference_may_have_three() {
    // bert.cpp creates attn_out_norm and layer_out_norm UNCONDITIONALLY, and
    // attn_norm_2 only when the checkpoint provides it (:156). So the
    // reference has two per-layer norms always and a third conditionally.
    // Pinned as a count so the shape of the gap stays legible without
    // asserting weights that depend on a future rewrite.
    let src = include_str!("../src/bert.rs");
    let rms_fields = src
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("pub ") && l.contains(": RmsNorm"))
        .count();
    assert_eq!(
        rms_fields, 3,
        "BertBlock now declares {rms_fields} RmsNorm fields (was 3: \
         attention_ln, output_ln, embeddings_ln). Only two are PER-LAYER; \
         embeddings_ln is the module-level token-embedding norm. A third \
         per-layer norm would be attn_norm_2, and the kinds must change \
         with it."
    );
    for per_layer in ["attention_ln", "output_ln"] {
        assert!(
            src.contains(&format!("pub {per_layer}: RmsNorm")),
            "{per_layer} is no longer RmsNorm; see the kind test"
        );
    }
}

#[test]
fn bertblocks_norms_are_rms_which_the_reference_is_not() {
    // The defect itself, stated so it cannot be quietly forgotten. The
    // reference passes LLM_NORM at :154, :158 and :209; LayerNorm subtracts
    // the mean and RmsNorm does not, so the two disagree on any input with a
    // non-zero mean.
    //
    // When this is fixed, DELETE this test rather than invert it: the
    // replacement belongs in a suite that drives both kinds and compares
    // their arithmetic, which is what norm_kind.rs does for the Norm type.
    let src = include_str!("../src/bert.rs");
    assert!(
        src.contains("attention_ln: RmsNorm"),
        "BertBlock's attention_ln is no longer RmsNorm; the reference wants \\
         LLM_NORM (LayerNorm) at bert.cpp:154"
    );
    assert!(
        src.contains("output_ln: RmsNorm"),
        "BertBlock's output_ln is no longer RmsNorm; the reference wants \\
         LLM_NORM (LayerNorm) at bert.cpp:209"
    );
}

#[test]
fn the_reference_order_is_read_from_bert_cpp() {
    // An earlier version compared a local const to itself and so could not
    // fail; it did fail the first time, because the transcription was wrong.
    // The needles are now read out of bert.cpp, and the sequence is walked
    // positionally so a repeated needle cannot collapse onto its first match.
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../old/repo/llama.cpp-master/src/models/bert.cpp"
    ))
    .expect("the reference must be readable; a test that skips it proves nothing");
    let flat = src.replace('\n', " ");

    // In source order: :151 add, :154 norm, :156 guard, :157 add, :181 ffn,
    // :206 add, :209 norm. The add comes BEFORE the norm it feeds -- that is
    // what makes this post-norm, and getting it backwards is the defect.
    let seq = [
        ("first inpL add", "ggml_add(ctx0, cur, inpL)"),
        ("attn_out_norm applied", "attn_out_norm_b, LLM_NORM"),
        ("attn_norm_2 guard", "attn_norm_2 != nullptr"),
        ("second inpL add", "inpL);  // re-add the layer input"),
        ("ffn", "build_ffn(cur"),
        ("bypass add", "ggml_add(ctx0, cur, ffn_inp)"),
        ("layer_out_norm applied", "layer_out_norm_b, LLM_NORM"),
    ];
    // Positional: each step must be found AFTER the previous one, so a
    // repeated needle cannot collapse onto its first match.
    let after = |needle: &str, from: usize| {
        flat[from..]
            .find(needle)
            .map(|i| from + i)
            .unwrap_or_else(|| panic!("bert.cpp has no {needle:?} after offset {from}"))
    };
    let mut prev = 0;
    for (what, needle) in seq {
        let i = after(needle, prev);
        assert!(
            i > prev,
            "{what} ({needle:?}) must come after the previous step in bert.cpp"
        );
        prev = i;
    }

    // Both adds of inpL are real, and the second is the guarded one.
    assert_eq!(
        flat.matches("ggml_add(ctx0, cur, inpL)").count(),
        2,
        "bert.cpp adds inpL twice: once before attn_out_norm and once inside \
         the attn_norm_2 branch"
    );
    // Every norm in the file is LayerNorm.
    let calls: Vec<&str> = {
        let mut v = Vec::new();
        let mut rest = flat.as_str();
        while let Some(i) = rest.find("build_norm(") {
            let end = rest[i..].find(");").map_or(rest.len(), |e| i + e + 2);
            v.push(&rest[i..end]);
            rest = &rest[end..];
        }
        v
    };
    assert!(!calls.is_empty(), "bert.cpp has build_norm calls");
    for c in &calls {
        assert!(
            !c.contains("LLM_NORM_RMS"),
            "a bert norm uses LLM_NORM_RMS, which LayerNorm-matched tests \
             would miss: {c}"
        );
    }
}
