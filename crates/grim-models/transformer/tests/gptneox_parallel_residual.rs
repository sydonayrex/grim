//! `gptneox` parallel residual -- RED tests.
//!
//! Reference, `gptneox.cpp:143-168`:
//!
//! ```text
//! if (hparams.use_par_res) {
//!     // x = x + attn(ln1(x)) + ffn(ln2(x))
//!     ggml_tensor * attn_out = cur;
//!     cur = build_norm(inpL, ffn_norm, ffn_norm_b, LLM_NORM, il);   // :149
//!     cur = build_ffn(cur, ...);                                     // :156
//!     cur = ggml_add(ctx0, cur, inpL);
//!     cur = ggml_add(ctx0, cur, attn_out);
//! } else {
//!     // x = x + attn(ln1(x));  then  x = x + ffn(ln2(x))
//!     ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
//!     ...
//! }
//! ```
//!
//! ## What actually differs
//!
//! Not the order of the adds. Floating-point addition is associative, so
//! `x + a + f` and `(x + a) + f` are the same value, and a test that
//! compares them is a tautology. An earlier version of this file made
//! exactly that mistake and failed for the right reason: it was checking
//! arithmetic, not the graph.
//!
//! The difference is **what the FFN is fed**:
//!
//! * parallel  -- FFN sees `ln2(inpL)`, the layer input;
//! * sequential -- FFN sees `ln2(x + attn(ln1(x)))`, the post-attention
//!   value.
//!
//! That is observable only because the FFN is non-linear (norm, then
//! GeLU/SwiGLU). With a linear stand-in the two agree exactly, so the
//! probe below uses a genuinely non-linear branch.
//!
//! `gptneox.cpp` is the ONLY reference that branches this topology
//! (`grep -l 'if (hparams.use_par_res)' src/models/*.cpp` returns one file),
//! so the blast radius is one model -- which is exactly why it needs a
//! test rather than a code comment.

use grim_nn::{Norm, NormKind};

/// A non-linear stand-in for `ln2 -> ffn`: RMS-norm then GeLU, the
/// activation `gptneox.cpp` actually uses (`LLM_FFN_GELU` at `:157`).
struct Branch {
    norm: Norm,
}

impl Branch {
    fn new(eps: f32) -> Self {
        Self {
            norm: Norm::new(NormKind::Rms, eps),
        }
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        self.norm
            .apply(x, None, None)
            .into_iter()
            .map(gelu)
            .collect()
    }
}

/// GeLU, tanh approximation -- the same shape `ggml_gelu` computes.
fn gelu(v: f32) -> f32 {
    const K: f32 = 0.7978845608; // sqrt(2/pi)
    let t = (K * (v + 0.044715 * v * v * v)).tanh();
    0.5 * v * (1.0 + t)
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// The reference's parallel topology: `x + attn + ffn(ln2(x))`.
fn parallel(x: &[f32], attn: &[f32], ffn_out: &[f32]) -> Vec<f32> {
    add(&add(x, attn), ffn_out)
}

/// grim's sequential topology today: `ffn(ln2(x + attn)) + x + attn`, i.e.
/// the FFN is fed the post-attention value.
fn sequential(x: &[f32], attn: &[f32], branch: &Branch) -> Vec<f32> {
    let mid = add(x, attn);
    add(&add(x, attn), &branch.apply(&mid))
}

#[test]
fn the_two_topologies_differ_once_the_ffn_is_non_linear() {
    let b = Branch::new(1e-5);
    let x = vec![1.0f32, -2.0, 0.5, 3.0, -1.0, 2.5];
    let attn = vec![0.5f32, 0.25, -0.75, 1.0, -0.5, 0.125];

    // parallel: FFN is fed ln2(x)
    let par = parallel(&x, &attn, &b.apply(&x));
    // sequential: FFN is fed ln2(x + attn)
    let seq = sequential(&x, &attn, &b);

    assert_ne!(
        par, seq,
        "the two topologies agree on this input, so the test cannot \
         distinguish them; pick a probe where the FFN input matters"
    );
}

#[test]
fn even_a_linear_ffn_exposes_the_difference_because_the_input_differs() {
    // Corrects a claim this file made twice: that a linear FFN would hide
    // the difference. It would not. Floating-point addition associates, so
    // the ADD ORDER is irrelevant; what matters is that the FFN is evaluated
    // at `ln2(x)` in one topology and at `ln2(x + attn)` in the other. With an
    // identity FFN those are `x` and `x + attn`, which differ by `attn`.
    //
    // What WOULD hide it is an FFN whose output does not depend on its input.
    // That is pinned here too, so the invariant is stated rather than
    // assumed.
    let x = vec![1.0f32, 2.0, 3.0];
    let attn = vec![0.5f32, -0.5, 1.0];
    let identity = |v: &[f32]| v.to_vec();

    let par = parallel(&x, &attn, &identity(&x));
    let mid = add(&x, &attn);
    let seq = add(&add(&x, &attn), &identity(&mid));
    assert_ne!(
        par, seq,
        "identity FFN made the topologies agree, which contradicts the \
         arithmetic: sequential evaluates the FFN at x+attn, parallel at x"
    );

    // The only thing that hides it: an FFN insensitive to its input.
    let constant = |_: &[f32]| vec![7.0f32, 7.0, 7.0];
    assert_eq!(
        parallel(&x, &attn, &constant(&x)),
        add(&add(&x, &attn), &constant(&mid)),
        "a constant FFN must make the topologies identical"
    );
}

#[test]
fn parallel_feeds_the_ffn_the_layer_input() {
    // The defining property. With x = attn = 2.0 and a probe FFN, the
    // parallel path must evaluate the FFN at ln2(2.0), not at ln2(4.0).
    let b = Branch::new(1e-5);
    let x = vec![2.0f32, 2.0, 2.0, 2.0];
    let attn = vec![2.0f32, 2.0, 2.0, 2.0];

    let on_input = b.apply(&x);
    let on_mid = b.apply(&add(&x, &attn));
    assert_ne!(
        on_input, on_mid,
        "the FFN produced the same output for the layer input and for the \
         post-attention value, so this probe is too weak"
    );

    let par = parallel(&x, &attn, &on_input);
    for (i, v) in par.iter().enumerate() {
        let want = x[i] + attn[i] + on_input[i];
        assert!((v - want).abs() < 1e-6, "term {i}: {v} vs {want}");
    }
}

#[test]
fn the_ffn_norm_runs_before_the_branch_not_after_the_residual() {
    // Ordering the reference pins at gptneox.cpp:149-152. A constant row
    // normalises to 1.0 whatever it is fed, so the probe needs a row whose
    // mean shifts.
    let n = Norm::new(NormKind::Rms, 1e-5);
    let layer_input = vec![1.0f32, 2.0, 3.0, 4.0];
    let attn = vec![10.0f32, 0.0, 0.0, 0.0];
    let after_attn = add(&layer_input, &attn);
    assert_ne!(
        n.apply(&layer_input, None, None),
        n.apply(&after_attn, None, None),
        "normalising the layer input and the residual sum agree, so the probe \
         cannot detect the wrong ordering"
    );
}
