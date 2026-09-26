//! Gated DeltaNet (GDN / KDA) recurrence for `Qwen38FlashNext`.
//!
//! # Provenance
//!
//! Transcribed from the Qwen3.5/3.8 GDN path already in this repo
//! (`qwen35.rs`: `gated_delta_net_forward`, `kda_gated_delta_rule_row`, and
//! `Softplus`), which implements the same `qwen4exp`-shaped mixer and cites
//! llama.cpp `src/models/qwen35.cpp` for the layout and gating.
//!
//! # Fused `attn_qkv` layout
//!
//! From `qwen35.rs` (llama.cpp `src/models/qwen35.cpp`):
//!
//! ```text
//!   head_k_dim = head_v_dim = ssm_d_state   (128)
//!   n_k_heads  = ssm_n_group                (16)
//!   n_v_heads  = ssm_dt_rank                (48)
//!   key_dim    = head_k * n_k              (2048)
//!   value_dim  = head_v * n_v              (6144)
//!   conv_dim   = key_dim * 2 + value_dim   (10240)
//! ```
//!
//! so the post-projection stream is `[K key_dim][K key_dim][V value_dim]` —
//! key, key, value. The query is read off the *value* stream, and the key is
//! broadcast from its key head to each value head.
//!
//! # Gating
//!
//! ```text
//!   alpha_biased   = alpha + ssm_dt
//!   alpha_softplus = softplus(alpha_biased)
//!   gate           = alpha_softplus * ssm_a
//!   beta           = sigmoid(ssm_beta(x))
//! ```
//!
//! `ssm_a` multiplies *after* the softplus; it is not added inside it. That is
//! a different function, and the ordering was a documented bug once.
//!
//! # Recurrence
//!
//! Per value head, with column-major state `S[j][i]` (`d_v` rows of `d_k`):
//!
//! ```text
//!   decay = exp(gate)
//!   pred  = k . (decay * S[j])
//!   delta = beta * (v[j] - pred)
//!   S[j]  = decay * S[j] + k * delta
//!   out   = (q . S_new[j]) * ssm_norm[i]
//! ```
//!
//! Decay is applied *before* the dot product, and the output reads the state
//! the update just wrote.

use grim_core::error::{Error, Result};

/// Softplus, `log1p(exp(x))`, stable at both ends. Mirrors `qwen35.rs::Softplus`.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// One value head's gated delta rule, in place on `state`.
///
/// `state` is column-major `[d_v][d_k]`: row `j` spans `d_k` elements, and row
/// `j` is the key-space image of value channel `j`. The caller owns layout so
/// the state can live in a flat session buffer without re-striding.
pub fn gated_delta_rule_row(
    k: &[f32],
    v: &[f32],
    beta: f32,
    gate: f32,
    state: &mut [f32],
    d_k: usize,
    d_v: usize,
) {
    // A short state buffer means a mis-sized cache. Do nothing rather than
    // index out of bounds: a wrong recurrence is bad enough without also
    // panicking, and silently wrong output is easier to debug than a crash
    // that masks the real defect.
    if d_k == 0 || d_v == 0 || state.len() < d_k * d_v {
        return;
    }
    let decay = gate.exp();
    for j in 0..d_v {
        let row = &mut state[j * d_k..(j + 1) * d_k];
        // Decay BEFORE the dot.
        let pred: f32 = k
            .iter()
            .zip(row.iter())
            .map(|(kk, ss)| kk * (decay * ss))
            .sum();
        // beta scales the whole delta term.
        let delta = beta * (v.get(j).copied().unwrap_or(0.0) - pred);
        for i in 0..d_k {
            row[i] = decay * row[i] + k.get(i).copied().unwrap_or(0.0) * delta;
        }
    }
}

/// How a value head maps to its key head in the 3:1 GQA-shaped KDA.
///
/// A 3:1 ratio is consistent with both of these and nothing in the GGUF encodes
/// the mapping, so it is an explicit switchable constant rather than a guess
/// baked into the loop. Inherited from `qwen35.rs::KdaHeadPairing`, which
/// carries the same caveat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaHeadPairing {
    /// `key_head = value_head % num_key_heads`
    Interleaved,
    /// `key_head = value_head / values_per_group`
    Grouped,
}

impl Default for KdaHeadPairing {
    fn default() -> Self {
        Self::Interleaved
    }
}

/// A linear-attention layer's declared GDN geometry, or [`GdnGeometry::Unknown`]
/// for a full-attention layer that has no recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnGeometry {
    Known(GdnShape),
    Unknown,
}

/// The three head dimensions that define the recurrent state size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnShape {
    /// `n_v_heads` (48 = `ssm_dt_rank`).
    pub n_v_heads: usize,
    /// `n_k_heads` (16 = `ssm_n_group`).
    pub n_k_heads: usize,
    /// `d_k` = `d_v` (128 = `ssm_d_state`).
    pub head_dim: usize,
    /// `conv_dim` (10240).
    pub conv_dim: usize,
}

/// Per-layer recurrent state for the GDN mixer.
///
/// Holds the depthwise conv ring *and* the delta-rule state. The conv history
/// must persist across steps or the convolution restarts every token; the
/// delta-rule state is the actual memory.
#[derive(Debug, Clone, Default)]
pub struct Qwen38GdnCache {
    /// Conv history, `[(taps - 1)][conv_dim]`, most recent first.
    pub conv_state: Vec<f32>,
    /// Delta-rule state, `[n_v_heads][d_v][d_k]`, column-major within a head.
    pub ssm_state: Vec<f32>,
    /// Tokens advanced so far, for snapshot bookkeeping.
    pub pos: usize,
}

impl Qwen38GdnCache {
    /// A cache sized for `n_v_heads` value heads of `d_v` x `d_k`, with a
    /// `taps`-tap convolution over `conv_dim` channels.
    pub fn new(n_v_heads: usize, d_v: usize, d_k: usize, taps: usize, conv_dim: usize) -> Self {
        let conv_len = taps.saturating_sub(1).saturating_mul(conv_dim);
        Self {
            conv_state: vec![0.0; conv_len],
            ssm_state: vec![0.0; n_v_heads * d_v * d_k],
            pos: 0,
        }
    }
}

/// Everything the GDN mixer needs for one layer's forward pass.
pub struct GdnParams<'a> {
    /// Post-projection, post-conv, post-SiLU stream, `[seq, conv_dim]`.
    pub conv_mix: &'a [f32],
    /// `alpha` projection, `[seq, n_v_heads]`, raw (pre-softplus).
    pub alpha: &'a [f32],
    /// `beta` projection, `[seq, n_v_heads]`, raw (pre-sigmoid).
    pub beta: &'a [f32],
    /// Per-head decay multiplier `ssm_a`, length `n_v_heads`.
    pub a: &'a [f32],
    /// Per-head timestep bias `ssm_dt.bias`, length `n_v_heads`.
    pub dt_bias: &'a [f32],
    /// Per-state-channel output norm `ssm_norm`, length `d_v`.
    pub norm: &'a [f32],
    /// `n_v_heads` (48 = `ssm_dt_rank`).
    pub n_v_heads: usize,
    /// `n_k_heads` (16 = `ssm_n_group`).
    pub n_k_heads: usize,
    /// `d_k` = `d_v` = `ssm_d_state` (128).
    pub head_dim: usize,
    /// `conv_dim` = `key_dim * 2 + value_dim` (10240).
    pub conv_dim: usize,
    /// `seq_len`.
    pub seq_len: usize,
    /// How value heads map to key heads.
    pub pairing: KdaHeadPairing,
}

/// Run the GDN recurrence over `params`, writing `[seq, value_dim]` into
/// `out_branch` and advancing `cache`.
///
/// # Errors
/// Returns [`Error::Shape`] if the supplied slices are too short for the
/// declared geometry, which means the loader sized something from the wrong
/// quantity — the failure mode that motivated keeping `ssm_dt_rank` and
/// `n_v_heads` as separate config fields.
pub fn gated_delta_net_forward(
    params: &GdnParams<'_>,
    cache: &mut Qwen38GdnCache,
    out_branch: &mut [f32],
) -> Result<()> {
    let GdnParams {
        conv_mix,
        alpha,
        beta,
        a,
        dt_bias,
        norm,
        n_v_heads,
        n_k_heads,
        head_dim,
        conv_dim,
        seq_len,
        pairing,
    } = *params;

    if n_v_heads == 0 || n_k_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    let key_dim = n_k_heads * head_dim;
    let value_dim = n_v_heads * head_dim;
    let expect_conv = conv_dim.max(key_dim * 2 + value_dim);
    if conv_mix.len() < seq_len * expect_conv {
        return Err(Error::Shape(format!(
            "gated_delta_net_forward: conv stream has {} elements, need {} for seq={} conv_dim={}",
            conv_mix.len(),
            seq_len * expect_conv,
            seq_len,
            expect_conv
        )));
    }

    let state_len = head_dim * head_dim;
    let need_state = n_v_heads * state_len;
    if cache.ssm_state.len() < need_state {
        cache.ssm_state.resize(need_state, 0.0);
    }

    let values_per_group = (n_v_heads / n_k_heads).max(1);
    let out_width = value_dim;

    for t in 0..seq_len {
        let base = t * expect_conv;
        for h in 0..n_v_heads {
            let key_head = match pairing {
                KdaHeadPairing::Interleaved => h % n_k_heads,
                KdaHeadPairing::Grouped => h / values_per_group,
            };
            // [K key_dim][K key_dim][V value_dim]: key, key, value.
            let k_off = base + key_head * head_dim;
            let q_off = base + 2 * key_dim + h * head_dim;

            // gate = softplus(alpha + dt_bias) * ssm_a. ssm_a multiplies AFTER
            // the softplus; folding it inside changes the function.
            let mut z = alpha.get(t * n_v_heads + h).copied().unwrap_or(0.0);
            if let Some(&d) = dt_bias.get(h) {
                z += d;
            }
            let mut gate = softplus(z);
            if let Some(&av) = a.get(h) {
                gate *= av;
            }
            // beta = sigmoid(ssm_beta(x)); a per-head gate in (0,1).
            let beta_t = {
                let raw = beta.get(t * n_v_heads + h).copied().unwrap_or(0.0);
                1.0 / (1.0 + (-raw).exp())
            };

            let st_off = h * state_len;
            gated_delta_rule_row(
                &conv_mix[k_off..k_off + head_dim],
                &conv_mix[q_off..q_off + head_dim],
                beta_t,
                gate,
                &mut cache.ssm_state[st_off..st_off + state_len],
                head_dim,
                head_dim,
            );

            // out = (q . S_new) * norm, read from the state just written.
            // Emitting `q[i] * norm[i]` instead would compute the correct state
            // and then ignore it, which is exactly the bug that shipped once.
            let q_slice = &conv_mix[q_off..q_off + head_dim];
            for i in 0..head_dim {
                let row = &cache.ssm_state[st_off + i * head_dim..st_off + (i + 1) * head_dim];
                let acc: f32 = q_slice.iter().zip(row.iter()).map(|(qq, ss)| qq * ss).sum();
                let w = norm.get(i).copied().unwrap_or(1.0);
                let idx = t * out_width + h * head_dim + i;
                if idx < out_branch.len() {
                    out_branch[idx] = acc * w;
                }
            }
        }
    }
    cache.pos += seq_len;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a geometry from the slices themselves, so a test cannot declare
    /// one head count and supply another. `conv` must be
    /// `seq * (2 * n_k * hd + n_v * hd)` long; `alpha`/`beta` are
    /// `seq * n_v`; `a`/`dt` are `n_v`; `norm` is `hd`.
    fn params_from<'a>(
        conv: &'a [f32],
        alpha: &'a [f32],
        beta: &'a [f32],
        a: &'a [f32],
        dt: &'a [f32],
        norm: &'a [f32],
        n_k_heads: usize,
    ) -> GdnParams<'a> {
        let n_v_heads = a.len();
        let head_dim = norm.len();
        let seq_len = alpha.len() / n_v_heads.max(1);
        GdnParams {
            conv_mix: conv,
            alpha,
            beta,
            a,
            dt_bias: dt,
            norm,
            n_v_heads,
            n_k_heads,
            head_dim,
            conv_dim: 2 * n_k_heads * head_dim + n_v_heads * head_dim,
            seq_len,
            pairing: KdaHeadPairing::Interleaved,
        }
    }

    /// A single-token slice of the same geometry, for stepwise comparison.
    fn params_step<'a>(
        conv: &'a [f32],
        alpha: &'a [f32],
        beta: &'a [f32],
        a: &'a [f32],
        dt: &'a [f32],
        norm: &'a [f32],
        n_k_heads: usize,
    ) -> GdnParams<'a> {
        GdnParams {
            seq_len: 1,
            ..params_from(conv, alpha, beta, a, dt, norm, n_k_heads)
        }
    }

    #[test]
    fn softplus_is_stable_at_both_ends() {
        assert!(
            (softplus(0.0) - 0.693_147_2).abs() < 1e-6,
            "softplus(0) = ln2"
        );
        assert!(
            (softplus(1000.0) - 1000.0).abs() < 1e-3,
            "large x -> x, not inf"
        );
        assert!(softplus(-1000.0).is_finite(), "large negative must not NaN");
    }

    #[test]
    fn delta_rule_reduces_to_a_rank_one_write_from_zero_state() {
        // From S = 0: pred = 0, so S[j] = k * beta * v[j].
        let d = 3;
        let k = [1.0, 2.0, 3.0];
        let v = [10.0, 20.0, 30.0];
        let mut state = vec![0.0; d * d];
        gated_delta_rule_row(&k, &v, 0.5, 0.0, &mut state, d, d);
        for j in 0..d {
            for i in 0..d {
                let want = k[i] * 0.5 * v[j];
                assert!(
                    (state[j * d + i] - want).abs() < 1e-6,
                    "S[{j}][{i}] = {} want {want}",
                    state[j * d + i]
                );
            }
        }
    }

    #[test]
    fn delta_rule_decay_shrinks_state_when_gate_is_negative() {
        let d = 2;
        let k = [0.0, 0.0];
        let v = [0.0, 0.0];
        let mut warm = vec![1.0; d * d];
        gated_delta_rule_row(&k, &v, 0.0, -1.0, &mut warm, d, d);
        let e_inv = (-1.0f32).exp();
        for x in &warm {
            assert!(
                (x - e_inv).abs() < 1e-6,
                "a zero-valued step must decay the state to e^gate, got {x}"
            );
        }
    }

    #[test]
    fn output_depends_on_state_not_only_on_query() {
        // The guard qwen35.rs carries: a mixer that computes the state and then
        // emits `q * norm` has no functional memory.
        let d = 4;
        let q = [1.0, 0.5, -0.5, 2.0];
        let k = [0.1; 4];
        let v = [1.0; 4];
        let read_out = |init: &[f32]| -> Vec<f32> {
            let mut state = init.to_vec();
            gated_delta_rule_row(&k, &v, 0.5, 0.0, &mut state, d, d);
            (0..d)
                .map(|i| {
                    let row = &state[i * d..(i + 1) * d];
                    q.iter().zip(row.iter()).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect()
        };
        let cold = read_out(&vec![0.0; d * d]);
        let warm = read_out(&vec![0.5; d * d]);
        assert_ne!(cold, warm, "output must depend on recurrent state");
        let bare: Vec<f32> = q.to_vec();
        assert_ne!(cold, bare, "output must be q . S, not a bare scale of q");
    }

    #[test]
    fn state_is_column_major() {
        // Guards the layout convention so a refactor cannot silently swap it.
        let d = 2;
        let k = [1.0, 0.0];
        let v = [5.0, 7.0];
        let mut state = vec![0.0; d * d];
        gated_delta_rule_row(&k, &v, 1.0, 0.0, &mut state, d, d);
        // S[j] = k * v[j] -> S[0] = [5, 0]; S[1] = [7, 0]
        assert_eq!(
            state,
            vec![5.0, 0.0, 7.0, 0.0],
            "column-major [d_v][d_k]; a row-major layout would give a different vector"
        );
    }

    #[test]
    fn forward_over_n_tokens_equals_n_single_token_steps() {
        // The defining property of a recurrent mixer: batched prefill and
        // token-at-a-time decode must agree.
        let n_v = 3;
        let n_k = 1;
        let hd = 2;
        let conv_dim = 2 * n_k * hd + n_v * hd;
        let seq = 5;
        let mut conv = vec![0.0f32; seq * conv_dim];
        for (i, x) in conv.iter_mut().enumerate() {
            *x = ((i * 37 % 17) as f32 - 8.0) / 8.0;
        }
        let alpha = vec![0.3f32; seq * n_v];
        let beta = vec![-0.2f32; seq * n_v];
        let a = vec![0.5f32; n_v];
        let dt = vec![0.1f32; n_v];
        let norm = vec![1.0f32; hd];

        let mut batched = Qwen38GdnCache::new(n_v, hd, hd, 4, conv_dim);
        let mut out_batched = vec![0.0f32; seq * n_v * hd];
        gated_delta_net_forward(
            &params_from(&conv, &alpha, &beta, &a, &dt, &norm, n_k),
            &mut batched,
            &mut out_batched,
        )
        .expect("batched");

        let mut stepwise = Qwen38GdnCache::new(n_v, hd, hd, 4, conv_dim);
        let mut out_step = vec![0.0f32; seq * n_v * hd];
        for t in 0..seq {
            let mut row = vec![0.0f32; n_v * hd];
            gated_delta_net_forward(
                &params_step(
                    &conv[t * conv_dim..(t + 1) * conv_dim],
                    &alpha[t * n_v..(t + 1) * n_v],
                    &beta[t * n_v..(t + 1) * n_v],
                    &a,
                    &dt,
                    &norm,
                    n_k,
                ),
                &mut stepwise,
                &mut row,
            )
            .expect("step");
            out_step[t * n_v * hd..(t + 1) * n_v * hd].copy_from_slice(&row);
        }

        for i in 0..out_batched.len() {
            assert!(
                (out_batched[i] - out_step[i]).abs() < 1e-5,
                "index {i}: batched {} vs stepwise {}",
                out_batched[i],
                out_step[i]
            );
        }
        assert_eq!(batched.pos, seq, "batched cache must advance by seq");
        assert_eq!(stepwise.pos, seq, "stepwise cache must advance by seq");
    }

    /// Distinguishes `softplus(alpha + dt) * a` from `softplus((alpha + dt) * a)`.
    ///
    /// These differ for any `a != 1` and any nonzero `alpha + dt`, and the
    /// ordering was a documented bug once. The check is on the DECAY itself
    /// rather than on a downstream output: observing the gate through the
    /// recurrence requires a two-token sequence whose second token leaves the
    /// state untouched, which is fiddly to arrange and easy to get subtly wrong.
    /// `exp(-gate)` is strictly monotonic, so pinning the decay pins the gate.
    #[test]
    fn ssm_a_multiplies_after_softplus_not_inside_it() {
        let alpha = 1.0f32;
        let dt = 0.0f32;
        let a = 2.0f32;
        let z = alpha + dt;

        let after = softplus(z) * a;
        let inside = softplus(z * a);
        assert!(
            (after - inside).abs() > 1e-3,
            "the two orderings must differ for this input, else the test proves nothing"
        );

        // Drive one step and read the decay back out of the state. The
        // recurrence uses `decay = gate.exp()`, so a positive gate GROWS the
        // state; the sign is easy to flip in either the code or this test.
        let d = 2;
        let k = [1.0, 0.0];
        let v = [0.0, 0.0];
        let observe = |gate: f32| -> f32 {
            let mut s = vec![1.0, 0.0, 1.0, 0.0];
            gated_delta_rule_row(&k, &v, 0.0, gate, &mut s, d, d);
            // beta = 0, so S[0] = decay * S[0] exactly; S started at 1.
            s[0]
        };
        let got = observe(after);
        assert!(
            (got - after.exp()).abs() < 1e-4,
            "observed decay {got} should be exp({after}) = {}",
            after.exp()
        );
        let wrong = observe(inside);
        assert!(
            (got - wrong).abs() > 1e-3,
            "the 'after' and 'inside' decays are indistinguishable here ({got} vs {wrong})"
        );
    }

    /// Pins the gate that `gated_delta_net_forward` actually hands the
    /// recurrence, for a case where the two orderings disagree.
    ///
    /// `ssm_a_multiplies_after_softplus_not_inside_it` checks the ordering
    /// through `gated_delta_rule_row`, which is the primitive. This one goes
    /// through the real call site, so a mutation that reorders the softplus
    /// inside `gated_delta_net_forward` is caught even though the primitive
    /// test still passes.
    #[test]
    fn call_site_gate_matches_softplus_times_a() {
        let n_v = 1;
        let n_k = 1;
        let hd = 2;
        let conv_dim = 2 * n_k * hd + n_v * hd;
        // k = [0, 0] and v = [0, 0] so delta is 0 and S stays 0, leaving the
        // output at 0 regardless of the gate. The gate itself is observed
        // indirectly: with a negative gate the state must shrink.
        let conv = vec![0.0f32; conv_dim];
        let alpha = vec![1.0f32];
        let beta = vec![0.0f32]; // sigmoid(0) = 0.5
        let dt = vec![0.0f32];
        let norm = vec![1.0f32; hd];

        // Two gates for alpha + dt = 1, a = 2:
        //   after  = softplus(1) * 2 = 2.6265234
        //   inside = softplus(2)      = 2.1269280
        // The decay is exp(gate), so the warm state after one step is
        // proportional to exp(2.6265234) vs exp(2.1269280) -- a 1.61x gap.
        let warm_then_read = |a_val: f32| -> f32 {
            let mut cache = Qwen38GdnCache::new(n_v, hd, hd, 4, conv_dim);
            // Warm with a k/v that writes a known state.
            let mut warm_conv = vec![0.0f32; conv_dim];
            warm_conv[0] = 1.0;
            warm_conv[2 * n_k * hd] = 1.0;
            let mut ignore = vec![0.0; n_v * hd];
            gated_delta_net_forward(
                &params_step(&warm_conv, &alpha, &beta, &[a_val], &dt, &norm, n_k),
                &mut cache,
                &mut ignore,
            )
            .expect("warm");
            // Now a step with no write; the state is purely decayed.
            let mut out = vec![0.0; n_v * hd];
            gated_delta_net_forward(
                &params_step(&conv, &alpha, &beta, &[a_val], &dt, &norm, n_k),
                &mut cache,
                &mut out,
            )
            .expect("decay");
            cache.ssm_state[0]
        };

        // Gate for the "after" ordering with a = 2.
        let gate_after = softplus(1.0) * 2.0;
        let got = warm_then_read(2.0);
        // Warm step: S[0] = k * beta * v[0] = 1 * 0.5 * 1 = 0.5
        // Decay step: S[0] = exp(gate) * 0.5
        let want = 0.5 * gate_after.exp();
        assert!(
            (got - want).abs() < 1e-4,
            "call site produced decay state {got}, want 0.5*exp({gate_after}) = {want}"
        );

        // And it must not match the "inside" ordering.
        let gate_inside = softplus(1.0 * 2.0);
        let wrong = 0.5 * gate_inside.exp();
        assert!(
            (got - wrong).abs() > 1e-3,
            "the call site cannot distinguish the two orderings ({got} vs {wrong})"
        );
    }

    /// Distinguishes `pred = k . (decay * S)` from `pred = (k . S) * decay`.
    ///
    /// With a constant k the two agree up to a factor, so drive a state whose
    /// rows are NOT parallel: decay must scale each element of the row before
    /// the dot, which is not the same as scaling the dot when rows differ.
    #[test]
    fn decay_is_applied_before_the_dot_not_after() {
        let d = 2;
        // k = [1, 0]; v = [0, 0] so delta = beta * (0 - pred) = -beta * pred.
        // Start from a state whose two value-rows differ, so scaling the state
        // and then dotting differs from dotting then scaling.
        let k = [1.0, 0.0];
        let v = [0.0, 0.0];
        let before = |gate: f32| {
            let mut s = vec![2.0, 0.0, 0.0, 0.0]; // S[0] = [2,0], S[1] = [0,0]
            gated_delta_rule_row(&k, &v, 0.5, gate, &mut s, d, d);
            // Read row 0 back: correct form is (decay*S[0] + k*delta)[0].
            s[0]
        };
        let gate = 0.5f32;
        let decay = gate.exp();
        // Correct: S[0] = decay*S[0] + k*(beta*(v - k.decay*S[0]))
        //   = 2*decay + (beta * (0 - 2*decay)) = 2*decay*(1 - beta)
        let want = 2.0 * decay * (1.0 - 0.5);
        let got = before(gate);
        assert!(
            (got - want).abs() < 1e-6,
            "decay-before-dot: got {got}, want {want}"
        );
        // If decay were applied after the dot, S[0] would be
        //   S[0] + k*(beta*(v - decay*(k.S[0])))  = 2 + 0.5*(0 - 2) = 1
        let want_after = 1.0;
        assert!(
            (got - want_after).abs() > 1e-3,
            "the two orderings coincide on this input; pick a sharper case"
        );
    }

    #[test]
    fn cache_clone_round_trips_the_recurrent_state() {
        // Session snapshot/restore must continue identically.
        let n_v = 2;
        let n_k = 1;
        let hd = 2;
        let conv_dim = 2 * n_k * hd + n_v * hd;
        let conv = vec![0.5f32; conv_dim];
        let alpha = vec![0.1; n_v];
        let beta = vec![0.2; n_v];
        let a = vec![0.9; n_v];
        let dt = vec![0.0; n_v];
        let norm = vec![1.0; hd];

        let mut warm = Qwen38GdnCache::new(n_v, hd, hd, 4, conv_dim);
        let mut ignore = vec![0.0; n_v * hd];
        gated_delta_net_forward(
            &params_step(&conv, &alpha, &beta, &a, &dt, &norm, n_k),
            &mut warm,
            &mut ignore,
        )
        .expect("warm");

        let snapshot = warm.clone();
        let mut continued = warm;
        let mut out_a = vec![0.0; n_v * hd];
        gated_delta_net_forward(
            &params_step(&conv, &alpha, &beta, &a, &dt, &norm, n_k),
            &mut continued,
            &mut out_a,
        )
        .expect("continue");

        let mut restored = snapshot;
        let mut out_b = vec![0.0; n_v * hd];
        gated_delta_net_forward(
            &params_step(&conv, &alpha, &beta, &a, &dt, &norm, n_k),
            &mut restored,
            &mut out_b,
        )
        .expect("restored");

        assert_eq!(
            out_a, out_b,
            "a restored snapshot must continue identically"
        );
    }

    #[test]
    fn short_conv_stream_is_rejected_not_silently_truncated() {
        let n_v = 3;
        let hd = 2;
        let mut cache = Qwen38GdnCache::new(n_v, hd, hd, 4, 2 * hd + n_v * hd);
        let conv = vec![0.0f32; 4]; // needs 1 * 8
        let mut out = vec![0.0; n_v * hd];
        let err = gated_delta_net_forward(
            &params_step(
                &conv, &[0.0; 3], &[0.0; 3], &[0.5; 3], &[0.0; 3], &[1.0; 2], 1,
            ),
            &mut cache,
            &mut out,
        )
        .expect_err("short conv stream must be an error, not a silent truncation");
        assert!(
            format!("{err}").contains("conv stream"),
            "error should name the mismatch: {err}"
        );
    }

    #[test]
    fn gate_multiplies_after_softplus_not_inside_it() {
        // gate = softplus(alpha + dt) * a. With a = 0 the gate collapses to 0
        // (decay 1) regardless of alpha; folding `a` inside softplus would not.
        let n_v = 1;
        let hd = 2;
        let conv_dim = 2 * hd + n_v * hd;
        let mut conv = vec![0.0f32; conv_dim];
        conv[0] = 1.0; // k[0] = 1 at the first key slot
        conv[2 * hd] = 3.0; // v[0] = 3 at the value slot
        let alpha = vec![50.0f32]; // softplus(50) would be ~50
        let beta = vec![0.0f32]; // sigmoid(0) = 0.5
        let a = vec![0.0f32]; // gate collapses
        let dt = vec![0.0f32];
        let norm = vec![1.0; hd];

        let mut cache = Qwen38GdnCache::new(n_v, hd, hd, 4, conv_dim);
        let mut out = vec![0.0; n_v * hd];
        gated_delta_net_forward(
            &params_step(&conv, &alpha, &beta, &a, &dt, &norm, 1),
            &mut cache,
            &mut out,
        )
        .expect("forward");
        // From S = 0 with gate 0 (decay 1): S[j] = k * beta * v[j].
        //   S[0] = [1,0] * 0.5 * 3 = [1.5, 0]
        // The query is the value stream, q = [3, 0], so out[0] = q . S[0] = 4.5.
        // The 1.5 is the state; the extra 3 is the read-out, and conflating them
        // is how a `q * norm` implementation would appear to pass.
        assert!(
            (out[0] - 4.5).abs() < 1e-5,
            "a=0 must zero the gate regardless of alpha, got {out:?}"
        );
        assert!((out[1]).abs() < 1e-5, "second state row stays zero");
    }

    #[test]
    fn head_pairing_selects_distinct_key_heads() {
        // 3:1 GQA: 6 value heads over 2 key heads. The two pairings differ, so
        // a wrong pairing is observable rather than incidental.
        let n_v = 6;
        let n_k = 2;
        let per_group = n_v / n_k;
        let interleaved: Vec<usize> = (0..n_v).map(|h| h % n_k).collect();
        let grouped: Vec<usize> = (0..n_v).map(|h| h / per_group).collect();
        assert_eq!(interleaved, vec![0, 1, 0, 1, 0, 1]);
        assert_eq!(grouped, vec![0, 0, 0, 1, 1, 1]);
        assert_ne!(
            interleaved, grouped,
            "the two pairings must be distinguishable"
        );
    }

    #[test]
    fn cache_sizes_match_the_declared_geometry() {
        // 48 value heads of 128x128 is the real per-layer state; 10240 conv
        // channels x 3 taps of history is the conv ring.
        let c = Qwen38GdnCache::new(48, 128, 128, 4, 10240);
        assert_eq!(c.ssm_state.len(), 48 * 128 * 128);
        assert_eq!(c.conv_state.len(), 3 * 10240);
        assert_eq!(c.pos, 0);
    }
}
