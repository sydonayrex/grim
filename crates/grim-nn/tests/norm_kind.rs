//! RED tests for plan item 5: norm kind, norm bias, `attn_post_norm`.
//!
//! Reference: `old/repo/llama.cpp-master/src/llama-graph.cpp:1592-1625`
//! (`llm_graph_context::build_norm`):
//!
//! ```text
//! case LLM_NORM:     cur = ggml_norm(ctx0, cur, f_norm_eps);    break;
//! case LLM_NORM_RMS: cur = ggml_rms_norm(ctx0, cur, f_norm_rms_eps); break;
//! ...
//! if (mw) cur = ggml_mul(ctx0, cur, mw);   // weight, only if present
//! if (mb) cur = ggml_add(ctx0, cur, mb);   // bias,   only if present
//! ```
//!
//! Two consequences the current code cannot express, both observed in real
//! checkpoints:
//!
//! * `LLM_NORM` is mean-subtracted (`ggml_norm`); grim's `Llama` is
//!   `RmsNorm`. Eleven models in the plan audit call `build_norm(...,
//!   LLM_NORM, ...)`, and `phi2` is one of the two architectures the loader
//!   reaches with a *live* wrong-numerics bug.
//! * weight and bias are both optional. `olmo.cpp:65-67` passes
//!   `NULL, NULL, LLM_NORM` -- a bias-free, weight-free LayerNorm that
//!   `RmsNorm` cannot represent even with a weight.
//!
//! These tests pin the arithmetic against values computed by hand from the
//! reference formulas, NOT against grim's own output, so a shared
//! misreading cannot make both sides agree.

use grim_nn::modules::{NormKind, Norm};

/// ggml_norm: mean and variance over the last dim, no weight, no bias.
fn ref_layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().map(|v| (v - mean) * inv).collect()
}

/// ggml_rms_norm: sum of squares only, no mean subtraction.
fn ref_rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ss = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn norm_kind_routes_to_the_right_formula() {
    // A probe with a non-zero mean separates the two: LayerNorm removes it,
    // RMSNorm does not. A constant offset is therefore the discriminating
    // input, and a test that only used zero-mean data would pass either way.
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ln = Norm::new(NormKind::LayerNorm, 1e-5);
    let rms = Norm::new(NormKind::Rms, 1e-5);
    let got_ln = ln.apply(&x, None, None);
    let got_rms = rms.apply(&x, None, None);

    let want_ln = ref_layer_norm(&x, 1e-5);
    let want_rms = ref_rms_norm(&x, 1e-5);

    for i in 0..x.len() {
        assert!(
            close(got_ln[i], want_ln[i]),
            "LayerNorm[{i}]: got {} want {want_ln:?}",
            got_ln[i]
        );
        assert!(
            close(got_rms[i], want_rms[i]),
            "RmsNorm[{i}]: got {} want {want_rms:?}",
            got_rms[i]
        );
    }
    // The two must actually differ, or the test proves nothing.
    assert!(
        !close(got_ln[0], got_rms[0]),
        "LayerNorm and RmsNorm produced the same value on a non-zero-mean probe; \
         the test is not discriminating"
    );
}

#[test]
fn layer_norm_applies_weight_then_bias_when_present() {
    // build_norm: norm -> mul(weight) -> add(bias). Order matters: applying
    // the bias first would change the result, since LayerNorm is not
    // shift-invariant once a weight is present.
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let w = vec![2.0f32, 0.5, 1.5, 1.0];
    let b = vec![0.1f32, -0.2, 0.3, 0.0];
    let n = ref_layer_norm(&x, 1e-5);
    let want: Vec<f32> = (0..4).map(|i| n[i] * w[i] + b[i]).collect();

    let got = Norm::new(NormKind::LayerNorm, 1e-5).apply(&x, Some(&w), Some(&b));
    for i in 0..4 {
        assert!(close(got[i], want[i]), "weight+bias[{i}]: got {} want {}", got[i], want[i]);
    }
}

#[test]
fn layer_norm_without_weight_or_bias_is_plain_mean_variance_norm() {
    // olmo.cpp:65-67 -> build_norm(inpL, NULL, NULL, LLM_NORM, il).
    // The reference then skips both the mul and the add, so this is ggml_norm
    // alone. A missing weight must not silently become a weight of ones with
    // a different eps path, and must not fall back to RMSNorm.
    let x = vec![2.0f32, -1.0, 4.0, 0.5];
    let want = ref_layer_norm(&x, 1e-5);
    let got = Norm::new(NormKind::LayerNorm, 1e-5).apply(&x, None, None);
    for i in 0..x.len() {
        assert!(
            close(got[i], want[i]),
            "bias-free LayerNorm[{i}]: got {} want {}",
            got[i],
            want[i]
        );
    }
}

#[test]
fn norm_reports_whether_a_weight_or_bias_was_loaded() {
    // The block loader must be able to tell the ten models that ship
    // attn_norm_b/ffn_norm_b from olmo's NULL/NULL without a second load, so
    // presence is reported rather than assumed.
    let bare = Norm::new(NormKind::LayerNorm, 1e-5);
    assert!(!bare.has_weight() && !bare.has_bias(), "a bare Norm has neither");

    let mut wb = Norm::new(NormKind::Rms, 1e-5);
    wb.weight = Some(grim_backend_cpu::cpu_tensor(
        vec![1.0, 1.0],
        grim_tensor::Shape::new(vec![2]),
    ));
    assert!(wb.has_weight() && !wb.has_bias());
}
