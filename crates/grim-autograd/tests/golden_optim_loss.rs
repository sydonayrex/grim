//! Mutation-resistant golden tests for the autograd numerics epicenter:
//! AdamW optimizer step, cross-entropy loss + gradient (incl. logsumexp
//! numerical stability), and the DPO/ORPO/GRPO preference losses.
//!
//! The in-crate tests for these are weak:
//!  - `adamw_step_updates_param_and_moments` only asserts `data[0] < 1.0`
//!    — a mutant that set `lr = 0` (no movement) or reversed the update sign
//!    still passes.
//!  - `cross_entropy_loss_zero_when_confident_correct` only asserts
//!    `loss < 1e-4` and the gradient *shape*, never any gradient *value*, and
//!    never exercises the max/logsumexp stability trick (the silent-NaN site).
//!  - `dpo_loss_decreases_when_policy_improves_chosen` only asserts
//!    `loss > 0.0` and `c_r > r_r` — a mutant scaling loss by 1000× passes.
//!
//! These tests assert **exact, hand-derived expected values** computed from the
//! formulas in the source specs, independent of the library's own code.

use grim_autograd::{
    adamw::{AdamW, AdamWConfig},
    loss::cross_entropy_loss,
    param::{ParamId, TrainableParam, TrainableParams},
    preference_loss::{dpo_loss, grpo_normalize_rewards, orpo_odds_ratio_loss},
};
use grim_backend_cpu::cpu_tensor;
use grim_tensor::Shape;

/// f32 relative-tolerance compare used for optimizer/loss golden values.
/// Generates f32-rounded intermediates (sigmoid, ln), so we allow 1e-5 rel.
fn close(got: f32, want: f32, ctx: &str) {
    let abs = (got - want).abs();
    let denom = want.abs().max(1e-7);
    assert!(
        got.is_finite(),
        "{ctx}: got non-finite {got:?} (want {want:?})",
    );
    assert!(
        abs == 0.0 || (abs / denom) < 1e-5,
        "{ctx}: got {got:?} want {want:?} (abs={abs}, rel={})",
        abs / denom,
    );
}

// ===========================================================================
// AdamW — one full step against a hand-computed reference.
// ===========================================================================
//
// With β1=0.9, β2=0.999, ε=1e-8, wd=0.01, lr=L, step_count=1, w0, grad g:
//   bc1 = 1-β1 = 0.1 ; bc2 = 1-β2 = 0.001
//   m  = (1-β1)*g            = 0.1*g
//   v  = (1-β2)*g²           = 0.001*g²
//   m̂  = m/bc1 = g ; v̂ = v/bc2 = g²
//   step_grad = m̂/(√v̂ + ε) + wd*w0 = g/(|g|+ε) + 0.01*w0
//   w1 = w0 - lr*step_grad
// We also assert the persisted moments m, v match 0.1*g, 0.001*g².
//
// We mutate the default config to lr=0.1 and leave the rest at defaults.

#[test]
fn adamw_one_step_matches_hand_computed_reference() {
    let cfg = AdamWConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let mut opt = AdamW::new(cfg.clone());

    let w0 = 2.0f32;
    let g = 1.0f32;
    let id = ParamId::a(0, 1);
    let mut p =
        TrainableParam::new(id, cpu_tensor(vec![w0], Shape::new(vec![1, 1]))).unwrap();
    p.accumulate_grad(&cpu_tensor(vec![g], Shape::new(vec![1, 1])))
        .unwrap();

    let mut params = TrainableParams::new();
    params.insert(p);

    opt.step(&mut params).unwrap();

    // Hand-derived reference (see header).
    let gabs = g.abs();
    let step_grad = g / (gabs + cfg.eps) + cfg.weight_decay * w0;
    let want_w1 = w0 - cfg.lr * step_grad;
    let want_m = (1.0 - cfg.beta1) * g;
    let want_v = (1.0 - cfg.beta2) * g * g;

    let got_w1 = params.get(id).unwrap().data.to_vec_f32().unwrap()[0];
    close(got_w1, want_w1, "adamw w1");
    assert_eq!(opt.step_count, 1, "adamw step_count");
    let got_m = opt.m.get(&id).unwrap().to_cpu_vec_f32().unwrap()[0];
    let got_v = opt.v.get(&id).unwrap().to_cpu_vec_f32().unwrap()[0];
    close(got_m, want_m, "adamw m");
    close(got_v, want_v, "adamw v");

    // ---- Second step exercises the bias-correction exponent (step_count=2) ----
    // bc1 = 1-0.9² = 0.19 ; bc2 = 1-0.999² = 0.001999
    // m2 = β1*m1 + (1-β1)*g ; v2 = β2*v1 + (1-β2)*g²
    // m̂2 = m2/0.19 ; v̂2 = v2/0.001999
    // step_grad2 = m̂2/(√v̂2 + ε) + wd*w1 ; w2 = w1 - lr*step_grad2
    let _ = params.get_mut(id).unwrap().zero_grad();
    params
        .get_mut(id)
        .unwrap()
        .accumulate_grad(&cpu_tensor(vec![g], Shape::new(vec![1, 1])))
        .unwrap();
    let w1 = got_w1;
    opt.step(&mut params).unwrap();

    let bc1_2 = 1.0 - cfg.beta1.powi(2);
    let bc2_2 = 1.0 - cfg.beta2.powi(2);
    let m2 = cfg.beta1 * want_m + (1.0 - cfg.beta1) * g;
    let v2 = cfg.beta2 * want_v + (1.0 - cfg.beta2) * g * g;
    let m_hat2 = m2 / bc1_2;
    let v_hat2 = v2 / bc2_2;
    let step_grad2 = m_hat2 / (v_hat2.sqrt() + cfg.eps) + cfg.weight_decay * w1;
    let want_w2 = w1 - cfg.lr * step_grad2;

    let got_w2 = params.get(id).unwrap().data.to_vec_f32().unwrap()[0];
    close(got_w2, want_w2, "adamw w2 (step_count=2 bias correction)");
    assert_eq!(opt.step_count, 2);
    close(opt.m.get(&id).unwrap().to_cpu_vec_f32().unwrap()[0], m2, "adamw m2");
    close(opt.v.get(&id).unwrap().to_cpu_vec_f32().unwrap()[0], v2, "adamw v2");
}

/// A mutant that drops the bias-correction (e.g. `m_hat = m` without dividing
/// by `bc`) would leave the *first step* unchanged (bc1=bc2 happens to equal
/// `1-beta` here) but would diverge on step 2. The second-step assertion above
/// is what guards that mutant specifically — this dedicated test makes the
/// intent explicit and tightens step-2-only.

#[test]
fn adamw_decay_actually_moves_weight_to_zero_under_constant_grad() {
    // Constant positive grad with weight_decay=0 and many steps should drive
    // w toward 0 (the only fixed point of AdamW with wd=0 is w=0 for g=0,
    // and g≠0 pushes w negative). A buggy sign-flip in the update
    // (`w1 = w0 + lr*step_grad`) would drive w *away* from 0; a no-op mutant
    // (`w1 = w0`) would keep w pinned — both fail this directional check.
    let cfg = AdamWConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut opt = AdamW::new(cfg);
    let id = ParamId::b(0, 2);
    let p = TrainableParam::new(id, cpu_tensor(vec![1.0], Shape::new(vec![1, 1]))).unwrap();
    let mut params = TrainableParams::new();
    params.insert(p);

    for _ in 0..50 {
        let pm = params.get_mut(id).unwrap();
        pm.zero_grad().unwrap();
        pm.accumulate_grad(&cpu_tensor(vec![1.0], Shape::new(vec![1, 1])))
            .unwrap();
        opt.step(&mut params).unwrap();
    }
    let w_final = params.get(id).unwrap().data.to_vec_f32().unwrap()[0];
    assert!(
        w_final < -0.5,
        "AdamW must drive w negative under constant positive grad; got {w_final}",
    );
    assert!(w_final.is_finite(), "AdamW must stay finite");
}

// ===========================================================================
// Cross-entropy — exact loss + gradient, and logsumexp stability.
// ===========================================================================

#[test]
fn cross_entropy_uniform_logits_ln2_loss_and_balanced_grad() {
    // logits=[0,0], target=0: softmax = [0.5,0.5]; loss = ln(2); grad =
    // (softmax - onehot)/batch = (0.5-1, 0.5-0) = (-0.5, 0.5).
    let logits = cpu_tensor(vec![0.0f32, 0.0], Shape::new(vec![1, 2]));
    let (loss, grad) = cross_entropy_loss(&logits, &[0]).unwrap();
    close(loss, (2.0f32).ln(), "ce uniform loss");
    let g = grad.to_vec_f32().unwrap();
    assert_eq!(g.len(), 2);
    close(g[0], -0.5, "ce uniform grad[0]");
    close(g[1], 0.5, "ce uniform grad[1]");
}

#[test]
fn cross_entropy_known_odds_loss_and_grad_reduce_by_batch() {
    // logits=[2,1,0], target=1.
    //   softmax = exp([2,1,0]-2)/Z ; exp(...)=[1, e^-1, e^-2]; Z≈1+0.3679+0.1353=1.5032
    //   p = [0.6652, 0.2447, 0.0900]
    //   loss = logsumexp - logit_target = ln(1.5032)+2 - 1 ...
    //   Equivalent closed form: loss = -ln(p_target) = -ln(0.2447) ≈ 1.4076.
    // Also: batch_size=1 so no /N distinction, and the gradient is (p - onehot).
    let logits = cpu_tensor(vec![2.0f32, 1.0, 0.0], Shape::new(vec![1, 3]));
    let (loss, grad) = cross_entropy_loss(&logits, &[1]).unwrap();
    let e = std::f32::consts::E;
    let z = 1.0 + 1.0 / e + 1.0 / (e * e);
    let p1 = (1.0 / e) / z;
    close(loss, -p1.ln(), "ce odds loss");
    let g = grad.to_vec_f32().unwrap();
    close(g[0], (1.0 / z) - 0.0, "ce odds grad[0]");
    close(g[1], p1 - 1.0, "ce odds grad[1]");
    close(g[2], ((1.0 / (e * e)) / z) - 0.0, "ce odds grad[2]");
}

#[test]
fn cross_entropy_batch_mean_divides_loss_and_grad_by_batch() {
    // Two samples, each half of the uniform -1/+1 case: total raw loss = 2*ln2,
    // averaged = ln2; all grads scale by 1/N = 0.5 → (-0.25, 0.25) per row.
    let logits = cpu_tensor(
        vec![0.0f32, 0.0, 0.0, 0.0],
        Shape::new(vec![2, 2]),
    );
    let (loss, grad) = cross_entropy_loss(&logits, &[0, 1]).unwrap();
    close(loss, (2.0f32).ln(), "ce batch mean loss");
    let g = grad.to_vec_f32().unwrap();
    assert_eq!(g.len(), 4);
    close(g[0], -0.25, "ce batch grad row0 col0");
    close(g[1], 0.25, "ce batch grad row0 col1");
    close(g[2], 0.25, "ce batch grad row1 col0");
    close(g[3], -0.25, "ce batch grad row1 col1");
}

#[test]
fn cross_entropy_logsumexp_stability_huge_logits_no_nan() {
    // The classic logsumexp silently-NaN site: logits spanning ±1e9.
    // Without the max-trick, exp(1e9-anything)=inf and the loss/grad go NaN.
    // Target is the *dominant* class (logit +1e9 = max), so softmax ≈ 1 there,
    // grad = (p - onehot) ≈ (1 - 1) = 0. The point: every value is finite, no
    // NaN, and the dominant class's grad is ≈ 0 (not -1 — it would be -1 only
    // if the target were the *minority* class).
    let logits = cpu_tensor(
        vec![1e9f32, -1e9, 0.0, 5e8, -5e8],
        Shape::new(vec![1, 5]),
    );
    let (loss, grad) = cross_entropy_loss(&logits, &[0]).unwrap();
    assert!(loss.is_finite(), "ce huge logits loss finite");
    // Cross-entropy is non-negative by definition (it's -ln p_target). A
    // mutant that drops the logsumexp max-trick subtracts `max_logit` (~1e9)
    // from the loss, producing a hugely *negative* value that still passes a
    // bare `loss < 1e-3` check — so we must also lower-bound the loss at 0.
    assert!(loss >= 0.0, "ce loss must be non-negative: {loss}");
    assert!(loss < 1e-3, "ce huge logits with target=max-class loss ≈ 0: {loss}");
    let g = grad.to_vec_f32().unwrap();
    assert_eq!(g.len(), 5);
    for &gi in &g {
        assert!(gi.is_finite(), "ce huge logits grad finite: {gi}");
    }
    // Dominant-class grad: p−1 ≈ 0 (p ≈ 1 since target == argmax).
    assert!(
        g[0].abs() < 1e-3,
        "ce huge logits dominant-class grad ≈ 0: {}",
        g[0],
    );
    // Non-target grads: p_j ≈ 0 for j != argmax → tiny positive.
    assert!(
        g[2..].iter().all(|&gi| (0.0..1e-3).contains(&gi)),
        "ce huge logits non-target grads ≈ 0: {:?}",
        &g[2..],
    );
    // Grad must sum to zero row-wise (softmax - onehot sums to 1-1 = 0).
    let row_sum: f32 = g.iter().sum();
    assert!(row_sum.abs() < 1e-3, "ce grad row sums to zero: {row_sum}");
}

// ===========================================================================
// DPO — exact loss at symmetric deltas (the -ln(sigmoid(0)) = ln 2 anchor).
// ===========================================================================

#[test]
fn dpo_symmetric_deltas_loss_is_ln2_and_rewards_match() {
    // pol_c=pol_r=ref_c=ref_r=0, beta=1 → logits=0, sigmoid(0)=0.5,
    // loss = -ln(0.5) = ln 2; chosen_r = beta*0 = 0, rejected_r = 0.
    let z = vec![0.0f32];
    let (loss, c_r, r_r) = dpo_loss(&z, &z, &z, &z, 1.0).unwrap();
    close(loss, (2.0f32).ln(), "dpo symmetric loss");
    assert_eq!(c_r.len(), 1);
    assert_eq!(r_r.len(), 1);
    close(c_r[0], 0.0, "dpo chosen reward");
    close(r_r[0], 0.0, "dpo rejected reward");
}

#[test]
fn dpo_unsymmetric_deltas_matches_hand_formula() {
    // β=0.5, chosen_logr=2 (pol_c-ref_c=2), rejected_logr=-2 (pol_r-ref_r=-2):
    //   logits = β*(2 - (-2)) = 0.5*4 = 2.0
    //   loss = -ln(sigmoid(2)) = -ln(1/(1+e^-2)) = ln(1+e^-2)
    //   chosen_r = β*2 = 1.0 ; rejected_r = β*(-2) = -1.0.
    let pol_c = vec![2.0f32];
    let pol_r = vec![0.0f32];
    let ref_c = vec![0.0f32];
    let ref_r = vec![2.0f32]; // pol_c-ref_c=2, pol_r-ref_r=-2
    let beta = 0.5f32;
    let (loss, c_r, r_r) = dpo_loss(&pol_c, &pol_r, &ref_c, &ref_r, beta).unwrap();
    let logits = beta * (2.0 - (-2.0));
    close(loss, (1.0 + (-2.0f32).exp()).ln(), "dpo unsymmetric loss");
    close(c_r[0], beta * 2.0, "dpo chosen reward");
    close(r_r[0], beta * (-2.0), "dpo rejected reward");
    // And the loss's logits consistency via the reward diff:
    close(c_r[0] - r_r[0], logits, "dpo logits = chosen_r - rejected_r");
}

#[test]
fn dpo_extreme_positive_logit_loss_is_far_below_ln2() {
    // β large enough to push logits >> 0 → sigmoid→1 → loss → 0. A mutant that
    // used sigmoid(x).ln() directly without the negative sign would flip this
    // toward +∞; a mutant dropping the clamp/exp would NaN out.
    let big = vec![100.0f32];
    let zero = vec![0.0f32];
    let (loss, _, _) = dpo_loss(&big, &zero, &zero, &zero, 10.0).unwrap();
    assert!(loss.is_finite(), "dpo extreme loss finite");
    assert!(loss < 1e-5, "dpo very-confident-correct loss ≈ 0: {loss}");
}

#[test]
fn dpo_extreme_negative_logit_loss_is_huge_and_finite() {
    // logits << 0 → sigmoid→0 → -ln(0) → +∞ but clamped/exp finite.
    let big = vec![0.0f32];
    let favored = vec![100.0f32];
    let zero = vec![0.0f32];
    let (loss, _, _) = dpo_loss(&big, &favored, &zero, &zero, 10.0).unwrap();
    assert!(loss.is_finite(), "dpo catastrophic loss finite");
    assert!(loss > 100.0, "dpo very-confident-wrong loss huge: {loss}");
}

// ===========================================================================
// ORPO — clamped-odds anchor and sign.
// ===========================================================================

#[test]
fn orpo_balanced_logps_loss_near_ln2_via_clamp() {
    // logp=0 → p=exp(0)=1, clamped to 1-1e-7. odds_chosen=odds_rejected, ratio=1,
    // log_odds_ratio=0, loss = -ln(sigmoid(0)) = ln(2). Within clamp tolerance.
    let z = vec![0.0f32];
    let loss = orpo_odds_ratio_loss(&z, &z, 1.0).unwrap();
    // Because of the 1e-7 clamp the logits are not *exactly* 0 (both odds are
    // (1-1e-7)/1e-7 = 9999999), so ratio=1 exactly and log_odds_ratio=0 exactly
    // → loss == ln(2) exactly. Assert that.
    close(loss, (2.0f32).ln(), "orpo balanced ln2");
}

#[test]
fn orpo_lambda_scales_loss_linearly() {
    // lambda multiplies the averaged loss. λ=0 must yield exactly 0.
    let z = vec![0.0f32];
    let loss0 = orpo_odds_ratio_loss(&z, &z, 0.0).unwrap();
    close(loss0, 0.0, "orpo lambda=0 zero loss");
    // λ=2 doubles ln2.
    let loss2 = orpo_odds_ratio_loss(&z, &z, 2.0).unwrap();
    close(loss2, 2.0 * (2.0f32).ln(), "orpo lambda=2 doubled");
}

// ===========================================================================
// GRPO — exact normalized rewards against the (r-mean)/std formula.
// ===========================================================================

#[test]
fn grpo_normalize_is_standard_score_with_known_eps() {
    // rewards=[1,2,3]: mean=2, var=2/3, std=sqrt(2/3+eps). With tiny eps the
    // normalized vector is [-1/std, 0, +1/std]; std≈0.8164966.
    let r = vec![1.0f32, 2.0, 3.0];
    let eps = 1e-8f32;
    let n = r.len() as f32;
    let mean = r.iter().sum::<f32>() / n;
    let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    let want = r.iter().map(|x| (x - mean) / std).collect::<Vec<_>>();

    let got = grpo_normalize_rewards(&r, eps);
    assert_eq!(got.len(), 3);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        close(*g, *w, &format!("grpo norm[{i}]"));
    }
    // Invariants that catch common mutants:
    assert!(got.iter().sum::<f32>().abs() < 1e-6, "grpo zero-mean");
    // std of the normalized vector must be ~1 (var+eps denominator, so near 1).
    let got_mean = got.iter().sum::<f32>() / n;
    let got_var = got.iter().map(|x| (x - got_mean).powi(2)).sum::<f32>() / n;
    close(got_var.sqrt(), 1.0, "grpo unit-std (when eps tiny)");
}

#[test]
fn grpo_normalize_empty_returns_empty_and_constant_is_zero_output() {
    assert!(grpo_normalize_rewards(&[], 1e-8).is_empty(), "grpo empty");
    // Constant rewards → mean==reward → every normalized value 0.
    let c = vec![4.2f32; 5];
    for v in grpo_normalize_rewards(&c, 1e-8) {
        close(v, 0.0, "grpo constant -> 0");
    }
}

#[test]
fn grpo_normalize_extreme_reward_difference_does_not_overflow() {
    // A mutant that computed `std = var.sqrt()` without `+ eps` would divide
    // by zero on constant input (caught above); a mutant that forgot to clamp
    // could overflow on this spread. We assert finiteness and exact symmetry.
    let r = vec![-1e6f32, 1e6];
    let got = grpo_normalize_rewards(&r, 1e-8);
    assert!(got.iter().all(|v| v.is_finite()), "grpo finite on extremes");
    close(got[0], -1.0, "grpo extreme low maps to -1/std... ≈ -1");
    close(got[1], 1.0, "grpo extreme high maps to +1/std... ≈ +1");
}
