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
    let got = b.forward(&x).expect("forward").to_vec_f32().expect("readable");
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
fn the_reference_op_order_is_encoded_here_so_it_cannot_drift() {
    // A literal transcription of bert.cpp's per-layer order, so the target is
    // written down rather than inferred from grim's shape. The two additions
    // of inpL and the three norms are the parts most likely to be
    // "simplified", so both are counted explicitly.
    const REFERENCE_ORDER: &[&str] = &[
        "attn(inpL)",                 // :139
        "add(attn_out, inpL)",         // :151
        "norm(attn_out_norm)",         // :154  LLM_NORM
        "add(attn_out_norm_out, inpL)", // :157  only if attn_norm_2
        "norm(attn_norm_2)",           // :158  LLM_NORM
        "ffn",                         // :164
        "add(ffn_out, ffn_inp)",       // :206
        "norm(layer_out_norm)",        // :209  LLM_NORM
    ];
    assert_eq!(REFERENCE_ORDER.len(), 8);
    assert_eq!(
        REFERENCE_ORDER
            .iter()
            .filter(|s| s.starts_with("add(") && s.contains("inpL"))
            .count(),
        2,
        "bert.cpp adds inpL twice: :151 for attn_out_norm and :157 for \\
         attn_norm_2. If that changes upstream, this and the reference \\
         disagree and a human must resolve it."
    );
    assert_eq!(
        REFERENCE_ORDER
            .iter()
            .filter(|s| s.starts_with("norm("))
            .count(),
        3,
        "three norms per layer upstream"
    );
    // The FFN's input is the SECOND norm's output, not the first's. That is
    // the load-bearing detail for anyone porting this.
    let ffn_at = REFERENCE_ORDER.iter().position(|s| *s == "ffn").unwrap();
    assert!(
        REFERENCE_ORDER[..ffn_at]
            .iter()
            .filter(|s| s.starts_with("norm("))
            .count()
            == 2,
        "the FFN sees attn_norm_2's output, so two norms precede it"
    );
}
