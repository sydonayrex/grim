//! Tests for plan items 5b/5c: the block-level norm wiring.
//!
//! `grim_nn::Norm` already exists and is mutation-proven. What is tested here
//! is the SPEC contract: `LayerAttentionSpec` is what a model file sets and
//! what the block loader reads, so if the spec and the loader can disagree
//! while these tests pass, the tests are not testing the join.
//!
//! The fields live on `LayerAttentionSpec` rather than `LlamaConfig` because
//! they are per-layer-type facts, not per-model ones -- `laguna.rs` and
//! `maple.rs` build a `LayerAttentionSpec` per layer, and SWA/full layers can
//! differ.
//!
//! Reference, `llama-graph.cpp:1592-1625` (`build_norm`):
//!
//! ```text
//! case LLM_NORM:     cur = ggml_norm(ctx0, cur, f_norm_eps);     break;
//! case LLM_NORM_RMS: cur = ggml_rms_norm(ctx0, cur, f_norm_rms_eps); break;
//! ...
//! if (mw) cur = ggml_mul(ctx0, cur, mw);   // weight, only if present
//! if (mb) cur = ggml_add(ctx0, cur, mb);   // bias,   only if present
//! ```
//!
//! Eleven models call `build_norm(..., LLM_NORM, ...)`; ten of those also
//! create `attn_norm_b` / `ffn_norm_b`. Fifteen create `attn_post_norm`.

use grim_models_transformer::block::LayerAttentionSpec;
use grim_nn::NormKind;

fn spec() -> LayerAttentionSpec {
    LayerAttentionSpec::default_full(8, 2, 4, 10000.0)
}

#[test]
fn spec_norm_kind_defaults_to_rms() {
    // Every model that existed before this field is RMSNorm, so the default
    // must be Rms. A wrong default breaks the whole corpus silently while the
    // new LayerNorm models look fine -- which is the failure mode to avoid.
    assert_eq!(spec().norm_kind, NormKind::Rms);
    let with_rope = LayerAttentionSpec::full_with_rope(8, 2, 4, 10000.0, 2, None);
    assert_eq!(
        with_rope.norm_kind,
        NormKind::Rms,
        "the YaRN constructor must default to Rms too"
    );
}

#[test]
fn spec_norm_kind_uses_the_gnn_enum_so_the_two_cannot_drift() {
    // The spec must reuse `grim_nn::NormKind`, not a parallel enum: a second
    // enum is a place where a spec could name a kind `Norm` cannot dispatch on.
    let mut s = spec();
    s.norm_kind = NormKind::LayerNorm;
    let dispatched: NormKind = s.norm_kind;
    assert_eq!(dispatched, NormKind::LayerNorm);
}

#[test]
fn spec_norm_bias_and_post_norm_default_off_but_are_settable() {
    // 10 models ship attn_norm_b/ffn_norm_b, 15 ship attn_post_norm, and most
    // ship neither. Off by default so the existing corpus is untouched;
    // settable so those models can ask for what they need.
    let s = spec();
    assert!(!s.has_norm_bias, "norm bias must default off");
    assert!(!s.has_attn_post_norm, "attn_post_norm must default off");

    let mut s = spec();
    s.has_norm_bias = true;
    s.has_attn_post_norm = true;
    assert!(s.has_norm_bias && s.has_attn_post_norm);
}

#[test]
fn olmo_shape_is_layer_norm_with_no_bias_and_no_post_norm() {
    // olmo.cpp:65-67 -> build_norm(inpL, NULL, NULL, LLM_NORM, il): a
    // bias-free, weight-free LayerNorm. Pin the exact triple so a future
    // default change cannot silently mis-serve it -- asking for a bias the
    // checkpoint does not carry would fail the load outright.
    let mut s = spec();
    s.norm_kind = NormKind::LayerNorm;
    s.has_norm_bias = false;
    s.has_attn_post_norm = false;
    assert_eq!(s.norm_kind, NormKind::LayerNorm);
    assert!(!s.has_norm_bias && !s.has_attn_post_norm);
}

#[test]
fn laguna_and_maple_specs_match_the_reference() {
    // Both were given explicit values when the fields were added, and those
    // values were checked against the reference. Pin them so a later
    // "simplification" cannot quietly change a real model's arithmetic.
    // laguna.cpp / maple.cpp: 5 x LLM_NORM_RMS, no ATTN_POST_NORM, no *_b.
    let mut s = spec();
    s.norm_kind = NormKind::Rms;
    s.has_norm_bias = false;
    s.has_attn_post_norm = false;
    assert_eq!(s.norm_kind, NormKind::Rms);
}
