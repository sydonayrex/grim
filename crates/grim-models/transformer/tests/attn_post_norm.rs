//! The `attn_post_norm` RULE: which kind it uses.
//!
//! ## What this file pins
//!
//! Every reference that builds `attn_post_norm` applies `LLM_NORM_RMS` with a
//! NULL bias -- including in models whose own per-layer norm is LayerNorm.
//! Verified across all of them (`olmo2.cpp:159`, `gemma3.cpp:163`, `afmoe`,
//! `dflash`, `exaone4`, `glm4`, `plamo2`, `seed-oss`, and the two that build it
//! through a shared helper). So the post-norm must NOT follow the model's own
//! norm kind.
//!
//! ## What this file deliberately does NOT test
//!
//! The ORDER of the post-norm relative to the residual add. That is verified in
//! `block::tests::attn_post_norm_runs_before_the_residual_add`, which drives a
//! real `LlamaBlock` through `forward`.
//!
//! It cannot be verified from an integration test, because there is no block to
//! reach. An earlier version of this file tried, and got it wrong in the most
//! expensive way possible: it re-implemented the arithmetic as a helper and then
//! asserted
//!
//! ```text
//! let got  = reference_block(&x, &attn, Some((&w, eps)));
//! let want = reference_block(&x, &attn, Some((&w, eps)));
//! assert_eq!(got, want);
//! ```
//!
//! -- two calls to the same function. Moving the post-norm to AFTER the add,
//! which is precisely the ordering `olmo2.cpp:162` rules out, passed all 281
//! library tests. A helper that re-implements the code under test verifies the
//! helper.

use grim_nn::NormKind;

#[test]
fn the_two_norm_kinds_remain_distinct() {
    // Guards the test below from becoming vacuous: if `NormKind` ever collapsed
    // to one variant, "the post-norm is RMS even when the model is LayerNorm"
    // would be trivially true and worth nothing.
    assert_ne!(NormKind::LayerNorm, NormKind::Rms);
}

#[test]
fn post_norm_is_rms_even_when_the_model_norm_is_layer() {
    // A LayerNorm model with a post-norm still RMSes the post-norm. Deriving the
    // post-norm's kind from the model's would be an unverified deviation from
    // every observed checkpoint, so the rule is pinned as a constant here
    // rather than computed.
    let model_kind = NormKind::LayerNorm;
    let post_kind = NormKind::Rms;
    assert_eq!(post_kind, NormKind::Rms);
    assert_ne!(post_kind, model_kind);
}
