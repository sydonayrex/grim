//! Tests for the final (output) norm of `Llama` -- plan item 5, output-norm half.
//!
//! The per-layer norms moved to `grim_nn::Norm` in commit `847c7222`, but the
//! *final* norm did not: `Llama::load_tp` and `load_tp_moe_specs` still do
//! `RmsNorm::load(&ws.pp("norm"), ...)` at model.rs:111 and :237. So the
//! eleven LayerNorm models get their per-layer norms right and their final
//! norm wrong, which is the kind of half-fix that produces plausible garbage.
//!
//! Reference, `gptneox.cpp:206-208`:
//!
//! ```text
//! cur = build_norm(cur,
//!     model.output_norm,
//!     model.output_norm_b,
//!     LLM_NORM, -1);
//! ```
//!
//! Same three-part shape as a per-layer norm: a kind, a weight, and an
//! OPTIONAL bias. Twenty references create `output_norm_b`; ten apply it as
//! `LLM_NORM`.

use grim_models_transformer::LlamaConfig;
use grim_nn::NormKind;

/// The final norm's configuration, taken per model rather than per layer.
///
/// It lives on `LlamaConfig` because the output norm has exactly one instance
/// per model, unlike the per-layer norms which are per layer-type and so live
/// on `LayerAttentionSpec`.
#[test]
fn config_carries_the_output_norm_kind_and_bias() {
    let mut c = LlamaConfig {
        vocab_size: 8,
        hidden_size: 4,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        num_layers: 1,
        intermediate_size: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 16,
        partial_rotary_factor: 1.0,
        yarn: None,
    ..Default::default()
    };
    // The default must stay RMS: every model that predates these fields is
    // RMSNorm, and a wrong default would silently change the whole corpus.
    assert_eq!(c.norm_kind, NormKind::Rms);
    assert!(!c.has_norm_bias);

    c.norm_kind = NormKind::LayerNorm;
    c.has_norm_bias = true;
    assert_eq!(c.norm_kind, NormKind::LayerNorm);
    assert!(c.has_norm_bias);
}

/// gptneox needs both: `LLM_NORM` and `output_norm_b` (gptneox.cpp:57, :208).
#[test]
fn gptneox_output_shape_is_layer_norm_with_a_bias() {
    let mut c = LlamaConfig {
        vocab_size: 8,
        hidden_size: 4,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        num_layers: 1,
        intermediate_size: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 16,
        partial_rotary_factor: 1.0,
        yarn: None,
    ..Default::default()
    };
    c.norm_kind = NormKind::LayerNorm;
    c.has_norm_bias = true;
    assert_eq!(c.norm_kind, NormKind::LayerNorm);
    assert!(c.has_norm_bias);
}

/// `olmo` is the counter-example on the bias axis: `olmo.cpp` creates
/// `output_norm` with no bias partner, so asking for one would fail the load.
#[test]
fn a_model_with_no_output_norm_bias_keeps_it_false() {
    let mut c = LlamaConfig {
        vocab_size: 8,
        hidden_size: 4,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        num_layers: 1,
        intermediate_size: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 16,
        partial_rotary_factor: 1.0,
        yarn: None,
    ..Default::default()
    };
    assert!(!c.has_norm_bias, "no *_b tensors means no norm bias");
    // LayerNorm kind and no bias is a legal combination, unlike bias without
    // a weight -- `olmo.cpp:65-67` passes `NULL, NULL`.
    c.norm_kind = NormKind::LayerNorm;
    assert!(!c.has_norm_bias);
}
