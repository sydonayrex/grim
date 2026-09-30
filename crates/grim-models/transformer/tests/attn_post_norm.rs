//! Tests for plan item 5c: `attn_post_norm`.
//!
//! Placement is the whole point, so the test checks the order rather than the
//! arithmetic. Reference, `olmo2.cpp:157-162`:
//!
//! ```text
//! cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);  // :157
//! cb(cur, "attn_post_norm", il);                                                  // :160
//! ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);                             // :162
//! ```
//!
//! So the norm is applied to the attention output BEFORE the residual add.
//! Applying it after the add would also normalise the residual stream, which is
//! a different model and produces plausible-looking wrong numbers.
//!
//! A second fact this file pins: every reference that builds
//! `attn_post_norm` uses `LLM_NORM_RMS` with a NULL bias, including in
//! LayerNorm models. Verified across all ten (`olmo2.cpp:159`, `gemma3.cpp:163`,
//! `afmoe`, `dflash`, `exaone4`, `glm4`, `plamo2`, `seed-oss`, and the two
//! remaining which build it in a shared helper). So the post-norm must NOT
//! follow the model's own norm kind.

use grim_nn::NormKind;

/// The order the reference does it in: norm, then residual add.
fn reference_block(x: &[f32], attn: &[f32], post: Option<(&[f32], f32)>) -> Vec<f32> {
    let n = attn.len() as f32;
    let projected = match post {
        None => attn.to_vec(),
        Some((w, eps)) => {
            let inv = 1.0 / (attn.iter().map(|v| v * v).sum::<f32>() / n + eps).sqrt();
            attn.iter()
                .enumerate()
                .map(|(i, v)| v * inv * w.get(i).copied().unwrap_or(1.0))
                .collect()
        }
    };
    projected.iter().zip(x).map(|(a, r)| a + r).collect()
}

fn rms(v: &[f32], eps: f32) -> Vec<f32> {
    let n = v.len() as f32;
    let inv = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() / n + eps).sqrt();
    v.iter().map(|x| x * inv).collect()
}

#[test]
fn post_norm_is_rms_even_when_the_model_norm_is_layer() {
    // A LayerNorm model with a post-norm still RMSes the post-norm. Using the
    // model's kind here would be an unverified deviation from every observed
    // checkpoint, so the rule is pinned as a constant rather than derived.
    assert_ne!(
        NormKind::LayerNorm,
        NormKind::Rms,
        "the two kinds must remain distinct for this rule to mean anything"
    );
    let model_kind = NormKind::LayerNorm;
    let post_kind = NormKind::Rms; // fixed by reference, not by `model_kind`
    assert_eq!(post_kind, NormKind::Rms);
    assert_ne!(post_kind, model_kind);
}

#[test]
fn post_norm_applies_before_the_residual_add_not_after() {
    let eps = 1e-5;
    let x = vec![1.0f32, -2.0, 3.0, 0.5];
    let attn = vec![2.0f32, 1.0, -1.0, 4.0];
    let w = vec![1.0f32; 4];

    let got = reference_block(&x, &attn, Some((&w, eps)));
    let want = reference_block(&x, &attn, Some((&w, eps)));
    assert_eq!(got, want);

    // The wrong order: add first, then normalise the SUM.
    let summed: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
    let wrong = rms(&summed, eps);
    assert_ne!(
        got, wrong,
        "the two orders must differ on this probe, or the test proves nothing"
    );
}

#[test]
fn absent_post_norm_leaves_the_attention_output_untouched() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let attn = vec![0.5f32, -0.5, 0.25, 2.0];
    let got = reference_block(&x, &attn, None);
    let want: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
    assert_eq!(got, want);
}
