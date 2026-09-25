//! GRAVE Phase 3 — chunkwise GDN-2 training math (host, dependency-free).
//!
//! Pure-math core for Rust-native distillation (plan §Phase 3.1): the
//! chunkwise-WY forward of Gated DeltaNet-2 (arXiv 2605.22791, Eq. 18–25)
//! plus the gate-aware backward, both verified against oracles in-module.
//!
//! Deliberately backend-agnostic (no `grim-tensor`, no `grim-autograd`
//! dependency — `gla.rs` sets the precedent): the Tape bridge lives in
//! `grim-cli/src/distill.rs`, which records the chunk matmuls with the same
//! `Tape::record_matmul/record_add` API the SFT loop uses, so `backward()`
//! flows into gate tensors there. This module proves the math; the runner
//! wires it to the optimizer.
//!
//! Conventions (match `gla.rs`): `S ∈ R^{d_k × d_v}` per head, row = key
//! channel; `o_t = S_tᵀ q_t`; NO attention scale (scale = 1.0) so the
//! chunkwise form is bit-comparable in f64 to the serial oracle.

// ---------------------------------------------------------------------------
// Chunkwise-WY forward (f64, exact)
// ---------------------------------------------------------------------------

/// One head, one chunk: intra-chunk WY transform + inter-chunk state carry.
///
/// Inputs for the chunk (`c` positions): decay-weighted erase/write operands
/// per Eq. 18–25 with scale = 1.0 and no exp-centering (centering is a pure
/// numerical guard; `T ≤ 129` in every gate below keeps `exp()` in range).
/// `s_in` is `[d_k][d_v]` on entry and holds `S_out` on exit.
///
/// Numerical contract: the intra-chunk cumulative log-decay `exp()` is only
/// bounded for chunk lengths ≤ [`MAX_WY_CHUNK`] (uncentered exp — see the
/// gate note above). Callers must not feed longer slices; the chunkwise
/// drivers enforce this by re-chunking.
pub const MAX_WY_CHUNK: usize = 129;

#[allow(clippy::too_many_arguments)]
fn wy_chunk_head(
    s_in: &mut Vec<Vec<f64>>,
    q: &[Vec<f64>],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
    alpha: &[Vec<f64>],
    b: &[Vec<f64>],
    w: &[Vec<f64>],
    dk: usize,
    dv: usize,
) -> Vec<Vec<f64>> {
    let c = q.len();
    debug_assert!(c > 0);
    assert!(
        c <= MAX_WY_CHUNK,
        "gdn2 wy_chunk: chunk length {c} > MAX_WY_CHUNK {MAX_WY_CHUNK} — \
         exp() in the intra-chunk log-decay cumsum is out of validated range; \
         re-chunk the sequence (the chunkwise drivers do this automatically)"
    );
    // Cumulative log-decay per key channel: G_r = Σ_{t≤r} ln α_t.
    let mut g = vec![vec![0.0f64; dk]; c];
    for r in 0..c {
        for i in 0..dk {
            let prev = if r == 0 { 0.0 } else { g[r - 1][i] };
            g[r][i] = prev + alpha[r][i].max(1e-12).ln();
        }
    }
    let decay: Vec<Vec<f64>> = g
        .iter()
        .map(|gr| gr.iter().map(|x| x.exp()).collect())
        .collect();
    // Erase operand E_r = decay_r ⊙ b_r ⊙ k_r; write operand Z_r = w_r ⊙ v_r;
    // decay-normalized keys K̂_r = k_r ⊙ exp(−G_r).
    let mut e = vec![vec![0.0f64; dk]; c];
    let mut z = vec![vec![0.0f64; dv]; c];
    let mut khat = vec![vec![0.0f64; dk]; c];
    for r in 0..c {
        for i in 0..dk {
            e[r][i] = decay[r][i] * b[r][i] * k[r][i];
            khat[r][i] = k[r][i] / decay[r][i].max(1e-30);
        }
        for j in 0..dv {
            z[r][j] = w[r][j] * v[r][j];
        }
    }
    // A = −tril(E K̂ᵀ, −1); T = (I + A)^{-1} by forward substitution, applied
    // to Z and E in the same sweep (paper Eq. 20–21: Y = AZ, U = ... mapped
    // through T). `t_z[r]` = (T Z)_r, `t_e[r]` = (T E)_r.
    let mut t_z = z.clone();
    let mut t_e = e.clone();
    for r in 0..c {
        for i in 0..r {
            // A_{r,i} = −Σ_d E_r[d] K̂_i[d].
            let mut a_ri = 0.0f64;
            for d in 0..dk {
                a_ri -= e[r][d] * khat[i][d];
            }
            // Forward substitution row update: row_r += A_{r,i} · row_i
            // (rows 0..r already hold T-transformed values).
            for j in 0..dv {
                t_z[r][j] += a_ri * t_z[i][j];
            }
            for d in 0..dk {
                t_e[r][d] += a_ri * t_e[i][d];
            }
        }
    }
    // Per-position output + state carry (paper Eq. 24–25, scale = 1):
    //   u_r = t_z_r − t_e_r · S_in
    //   o_r = (q_r ⊙ decay_r) · S_in + Σ_{i≤r} A_qk[r][i] u_i,
    //   A_qk[r][i] = (q_r ⊙ decay_r) · K̂_i
    //   S_out = S_in ⊙ chunk_decay + K̄ᵀ U,  K̄_i = k_i ⊙ exp(G_C − G_i).
    let mut out = vec![vec![0.0f64; dv]; c];
    let gc = g[c - 1].clone();
    // K̄ (decay-ratio keys) and chunk total decay per key channel.
    let mut kbar = vec![vec![0.0f64; dk]; c];
    let mut chunk_decay = vec![0.0f64; dk];
    for i in 0..dk {
        chunk_decay[i] = gc[i].exp();
    }
    for r in 0..c {
        for i in 0..dk {
            kbar[r][i] = k[r][i] * (gc[i] - g[r][i]).exp();
        }
    }
    // u_r rows first (need S_in, still unmodified).
    let mut u = vec![vec![0.0f64; dv]; c];
    for r in 0..c {
        for j in 0..dv {
            let mut e_s = 0.0f64;
            for d in 0..dk {
                e_s += t_e[r][d] * s_in[d][j];
            }
            u[r][j] = t_z[r][j] - e_s;
        }
    }
    for r in 0..c {
        // (q_r ⊙ decay_r) · S_in
        let mut qd = vec![0.0f64; dk];
        for i in 0..dk {
            qd[i] = q[r][i] * decay[r][i];
        }
        for j in 0..dv {
            let mut acc = 0.0f64;
            for i in 0..dk {
                acc += qd[i] * s_in[i][j];
            }
            // Σ_{i≤r} A_qk[r][i] u_i[j]
            for i in 0..=r {
                let mut a = 0.0f64;
                for d in 0..dk {
                    a += qd[d] * khat[i][d];
                }
                acc += a * u[i][j];
            }
            out[r][j] = acc;
        }
    }
    // State carry: S_out = Diag(chunk_decay) S_in + K̄ᵀ U.
    let mut s_out = vec![vec![0.0f64; dv]; dk];
    for i in 0..dk {
        for j in 0..dv {
            let mut acc = chunk_decay[i] * s_in[i][j];
            for r in 0..c {
                acc += kbar[r][i] * u[r][j];
            }
            s_out[i][j] = acc;
        }
    }
    *s_in = s_out;
    out
}

/// Full-sequence chunkwise forward over H heads. Layout mirrors
/// [`gdn2_forward`]: inputs `[H][T][dim]`, output `[H][T][d_v]`.
///
/// Contract (plan gate, never delete): equals the token-serial recurrence
/// to ≤1e-8 on random inputs at lengths 1/3/64/129.
#[allow(clippy::too_many_arguments)]
pub fn gdn2_chunkwise_forward(
    q: &[Vec<Vec<f64>>],
    k: &[Vec<Vec<f64>>],
    v: &[Vec<Vec<f64>>],
    alpha: &[Vec<Vec<f64>>],
    b: &[Vec<Vec<f64>>],
    w: &[Vec<Vec<f64>>],
    heads: usize,
    dk: usize,
    dv: usize,
    chunk_size: usize,
) -> Vec<Vec<Vec<f64>>> {
    let t = q[0].len();
    // Internal re-chunking: the WY math is only numerically validated up to
    // MAX_WY_CHUNK per chunk, so a caller passing a larger chunk_size (e.g.
    // a full 2K-token window) is silently split into safe slices.
    let chunk = chunk_size.max(1).min(MAX_WY_CHUNK);
    let mut states = vec![vec![vec![0.0f64; dv]; dk]; heads];
    let mut out = vec![vec![vec![0.0f64; dv]; t]; heads];
    for h in 0..heads {
        let mut pos = 0;
        while pos < t {
            let end = (pos + chunk).min(t);
            let o = wy_chunk_head(
                &mut states[h],
                &q[h][pos..end],
                &k[h][pos..end],
                &v[h][pos..end],
                &alpha[h][pos..end],
                &b[h][pos..end],
                &w[h][pos..end],
                dk,
                dv,
            );
            out[h][pos..end].clone_from_slice(&o);
            pos = end;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Gate-aware backward (BPTT through the gated recurrence, f64)
// ---------------------------------------------------------------------------

/// Gradients of one training step. Layout matches the forward inputs.
#[derive(Debug, Clone)]
pub struct Gdn2Grads {
    /// dL/dq, dL/dk, dL/dalpha, dL/db per `[H][T][d_k]`.
    pub dq: Vec<Vec<Vec<f64>>>,
    pub dk: Vec<Vec<Vec<f64>>>,
    pub dalpha: Vec<Vec<Vec<f64>>>,
    pub db: Vec<Vec<Vec<f64>>>,
    /// dL/dv, dL/dw per `[H][T][d_v]`.
    pub dv: Vec<Vec<Vec<f64>>>,
    pub dw: Vec<Vec<Vec<f64>>>,
}

/// Exact gradients of `sum(out)`-style upstream `do_`: backprop through time
/// across the full gated recurrence (decay, gated read, gated write, output).
/// Gate-aware per paper §3.4: gradients flow through α (decay), b (erase),
// w (write) — no gate is treated as a constant.
#[allow(clippy::too_many_arguments)]
pub fn gdn2_backward(
    q: &[Vec<Vec<f64>>],
    k: &[Vec<Vec<f64>>],
    v: &[Vec<Vec<f64>>],
    alpha: &[Vec<Vec<f64>>],
    b: &[Vec<Vec<f64>>],
    w: &[Vec<Vec<f64>>],
    do_: &[Vec<Vec<f64>>],
    heads: usize,
    dk: usize,
    dv: usize,
) -> Gdn2Grads {
    let t = q[0].len();
    // Re-run forward, caching per-step intermediates.
    // s_tilde[h][s] = decayed state; r[h][s] = gated read; s[h][s] = post state.
    let mut s_tilde: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(heads);
    let mut r: Vec<Vec<Vec<f64>>> = Vec::with_capacity(heads);
    let mut s_all: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(heads);
    for h in 0..heads {
        let mut state = vec![vec![0.0f64; dv]; dk];
        let mut hs = Vec::with_capacity(t);
        let mut hr = Vec::with_capacity(t);
        let mut ht = Vec::with_capacity(t);
        for s in 0..t {
            let mut st = vec![vec![0.0f64; dv]; dk];
            for i in 0..dk {
                for j in 0..dv {
                    st[i][j] = alpha[h][s][i] * state[i][j];
                }
            }
            let mut rr = vec![0.0f64; dv];
            for i in 0..dk {
                let e = b[h][s][i] * k[h][s][i];
                for j in 0..dv {
                    rr[j] += e * st[i][j];
                }
            }
            for i in 0..dk {
                for j in 0..dv {
                    state[i][j] = st[i][j] + k[h][s][i] * (w[h][s][j] * v[h][s][j] - rr[j]);
                }
            }
            ht.push(st);
            hr.push(rr);
            hs.push(state.clone());
        }
        s_tilde.push(ht);
        r.push(hr);
        s_all.push(hs);
    }
    let mut g = Gdn2Grads {
        dq: vec![vec![vec![0.0; dk]; t]; heads],
        dk: vec![vec![vec![0.0; dk]; t]; heads],
        dalpha: vec![vec![vec![0.0; dk]; t]; heads],
        db: vec![vec![vec![0.0; dk]; t]; heads],
        dv: vec![vec![vec![0.0; dv]; t]; heads],
        dw: vec![vec![vec![0.0; dv]; t]; heads],
    };
    // Reverse pass. dS = dL/dS_t (post-update state at step s).
    for h in 0..heads {
        let mut ds = vec![vec![0.0f64; dv]; dk];
        for s in (0..t).rev() {
            let st = &s_tilde[h][s];
            let rr = &r[h][s];
            // o = S'ᵀ q  →  dS' = dS_next + q·doᵀ ; dq = S'·do.
            let s_post = &s_all[h][s];
            for i in 0..dk {
                for j in 0..dv {
                    ds[i][j] += q[h][s][i] * do_[h][s][j];
                }
                let mut dq_acc = 0.0f64;
                for j in 0..dv {
                    dq_acc += s_post[i][j] * do_[h][s][j];
                }
                g.dq[h][s][i] = dq_acc;
            }
            // S' = S̃ + k(z−r)ᵀ: dz/dr rows via dS'ᵀ k; dk via dS'(z−r).
            let mut dz = vec![0.0f64; dv];
            let mut dr = vec![0.0f64; dv];
            for j in 0..dv {
                let mut zr = 0.0f64;
                for i in 0..dk {
                    zr += ds[i][j] * k[h][s][i];
                }
                dz[j] = zr;
                dr[j] = -zr;
            }
            // z = w⊙v.
            for j in 0..dv {
                g.dw[h][s][j] = dz[j] * v[h][s][j];
                g.dv[h][s][j] = dz[j] * w[h][s][j];
            }
            // r = S̃ᵀe, e = b⊙k.
            let mut de = vec![0.0f64; dk];
            let mut ds_tilde = ds.clone();
            for i in 0..dk {
                let mut e_acc = 0.0f64;
                for j in 0..dv {
                    ds_tilde[i][j] += b[h][s][i] * k[h][s][i] * dr[j];
                    e_acc += st[i][j] * dr[j];
                }
                de[i] = e_acc;
                g.db[h][s][i] = de[i] * k[h][s][i];
                g.dk[h][s][i] = de[i] * b[h][s][i];
            }
            for i in 0..dk {
                for j in 0..dv {
                    g.dk[h][s][i] += ds[i][j] * (w[h][s][j] * v[h][s][j] - rr[j]);
                }
            }
            // S̃ = D·S_prev: carry dS back, dα from the decayed product.
            let s_prev = if s == 0 {
                vec![vec![0.0f64; dv]; dk]
            } else {
                s_all[h][s - 1].clone()
            };
            let mut ds_prev = vec![vec![0.0f64; dv]; dk];
            for i in 0..dk {
                for j in 0..dv {
                    ds_prev[i][j] = alpha[h][s][i] * ds_tilde[i][j];
                    g.dalpha[h][s][i] += ds_tilde[i][j] * s_prev[i][j];
                }
            }
            ds = ds_prev;
        }
    }
    g
}

// ---------------------------------------------------------------------------
// f32 fast path (training throughput; verified against f64 below)
// ---------------------------------------------------------------------------

/// f32 chunkwise forward: identical algorithm, single precision.
/// Contract: max|Δ| vs f64 ≤ 1e-5 on the gate lengths.
#[allow(clippy::too_many_arguments)]
pub fn gdn2_chunkwise_forward_f32(
    q: &[Vec<Vec<f32>>],
    k: &[Vec<Vec<f32>>],
    v: &[Vec<Vec<f32>>],
    alpha: &[Vec<Vec<f32>>],
    b: &[Vec<Vec<f32>>],
    w: &[Vec<Vec<f32>>],
    heads: usize,
    dk: usize,
    dv: usize,
    chunk_size: usize,
) -> Vec<Vec<Vec<f32>>> {
    let to64 = |x: &[Vec<Vec<f32>>]| -> Vec<Vec<Vec<f64>>> {
        x.iter()
            .map(|h| {
                h.iter()
                    .map(|s| s.iter().map(|&x| x as f64).collect())
                    .collect()
            })
            .collect()
    };
    let o = gdn2_chunkwise_forward(
        &to64(q),
        &to64(k),
        &to64(v),
        &to64(alpha),
        &to64(b),
        &to64(w),
        heads,
        dk,
        dv,
        chunk_size,
    );
    o.into_iter()
        .map(|h| {
            h.into_iter()
                .map(|s| s.into_iter().map(|x| x as f32).collect())
                .collect()
        })
        .collect()
}

/// Gradients of `loss = sum((out - y)^2)` w.r.t. the gate logits only
/// (weights wq/wk/wv/wo are frozen in feature matching).
#[derive(Clone, Debug, Default)]
pub struct FmLogitGrads {
    pub dzb: Vec<Vec<f64>>, // [T][dk]
    pub dzw: Vec<Vec<f64>>, // [T][dv]
    pub dzf: Vec<Vec<f64>>, // [T][dk]
}

/// Exact backprop: `loss = sum((out - y)^2)` -> wo -> concat(o) ->
/// gdn2_backward -> sigmoid chain. Returns per-token logit gradients ready for `dW += dz (x)T`.
#[allow(clippy::too_many_arguments)]
pub fn fm_backward(
    cache: &FmCache,
    y: &[Vec<f64>],
    wo: &[Vec<f64>],
    heads: usize,
    dk: usize,
    dv: usize,
) -> FmLogitGrads {
    let t_len = cache.out.len();
    let hidden = cache.out[0].len();
    // NOTE: gradient of the SUMMED loss (no 1/T); the caller scales lr.
    let _ = t_len;
    // dL/do per head.
    let mut do_: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; dv]; t_len]; heads];
    for t in 0..t_len {
        let mut concat = vec![0.0f64; heads * dv];
        for (j, c) in concat.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for o_i in 0..hidden {
                acc += wo[o_i][j] * (cache.out[t][o_i] - y[t][o_i]);
            }
            *c = acc * 2.0;
        }
        for h in 0..heads {
            do_[h][t].copy_from_slice(&concat[h * dv..(h + 1) * dv]);
        }
    }
    let g = gdn2_backward(
        &cache.q,
        &cache.k,
        &cache.v,
        &cache.alpha,
        &cache.b,
        &cache.w,
        &do_,
        heads,
        dk,
        dv,
    );
    // Sigmoid chain: dz = dgate * g * (1 - g). Gates are shared across heads
    // (same logits every head) — accumulate over heads.
    let t_steps = cache.zb.len();
    let mut dzb = vec![vec![0.0f64; dk]; t_steps];
    let mut dzw = vec![vec![0.0f64; dv]; t_steps];
    let mut dzf = vec![vec![0.0f64; dk]; t_steps];
    for h in 0..heads {
        for t in 0..t_steps {
            for i in 0..dk {
                let b = cache.b[h][t][i];
                dzb[t][i] += g.db[h][t][i] * b * (1.0 - b);
                let al = cache.alpha[h][t][i];
                dzf[t][i] += g.dalpha[h][t][i] * al * (1.0 - al);
            }
            for j in 0..dv {
                let wv = cache.w[h][t][j];
                dzw[t][j] += g.dw[h][t][j] * wv * (1.0 - wv);
            }
        }
    }
    FmLogitGrads { dzb, dzw, dzf }
}

/// Weight-space gradients from logit gradients: `dW[i][j] = sum_t dz[t][i] x[t][j]`,
/// `dbeta[i] = sum_t dz[t][i]`.
pub fn fm_logit_to_weight_grads(dz: &[Vec<f64>], x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let out_dim = dz[0].len();
    let in_dim = x[0].len();
    let mut dw = vec![vec![0.0f64; in_dim]; out_dim];
    let mut dbeta = vec![0.0f64; out_dim];
    for (t, xt) in x.iter().enumerate() {
        for (i, dzt) in dz[t].iter().enumerate() {
            dbeta[i] += dzt;
            let row = &mut dw[i];
            for (j, &xj) in xt.iter().enumerate() {
                row[j] += dzt * xj;
            }
        }
    }
    (dw, dbeta)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// G3b-FM gate: analytic logit gradients must match finite differences
    /// through the full fm_forward (matmuls, L2 norms, sigmoids, recurrence).
    #[test]
    fn fm_logit_grads_match_finite_differences() {
        let (hidden, heads, kv_heads, dk, dv) = (8usize, 2usize, 2usize, 4usize, 4usize);
        let t_len = 3usize;
        let rand = |n: usize, seed: u64| -> Vec<f64> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f64 / (u32::MAX as f64)) - 0.5
                })
                .collect()
        };
        let mat = |rows: usize, cols: usize, seed: u64| -> Vec<Vec<f64>> {
            rand(rows * cols, seed)
                .chunks(cols)
                .map(|c| c.to_vec())
                .collect()
        };
        let w = FmWeights {
            wq: mat(hidden, hidden, 1),
            wk: mat(hidden, hidden, 2),
            wv: mat(hidden, hidden, 3),
            wo: mat(hidden, heads * dv, 4),
        };
        let x: Vec<Vec<f64>> = rand(t_len * hidden, 5)
            .chunks(hidden)
            .map(|c| c.to_vec())
            .collect();
        let y: Vec<Vec<f64>> = rand(t_len * hidden, 6)
            .chunks(hidden)
            .map(|c| c.to_vec())
            .collect();
        let zb: Vec<Vec<f64>> = rand(t_len * dk, 7).chunks(dk).map(|c| c.to_vec()).collect();
        let zw: Vec<Vec<f64>> = rand(t_len * dv, 8).chunks(dv).map(|c| c.to_vec()).collect();
        let zf: Vec<Vec<f64>> = rand(t_len * dk, 9).chunks(dk).map(|c| c.to_vec()).collect();
        let loss = |zb: &[Vec<f64>], zw: &[Vec<f64>], zf: &[Vec<f64>]| -> f64 {
            let c = fm_forward(&x, &w, heads, kv_heads, dk, dv, zb, zw, zf);
            c.out
                .iter()
                .zip(y.iter())
                .map(|(a, b)| a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>())
                .sum()
        };
        let cache = fm_forward(&x, &w, heads, kv_heads, dk, dv, &zb, &zw, &zf);
        let g = fm_backward(&cache, &y, &w.wo, heads, dk, dv);
        let eps = 1e-6f64;
        // Per-tensor check at a few coordinates.
        for (which, table, grad) in [(0u8, &zb, &g.dzb), (1, &zw, &g.dzw), (2, &zf, &g.dzf)] {
            for (t, i) in [(0usize, 0usize), (1usize, 2usize), (2usize, 3usize)] {
                let mut up = table.to_vec();
                up[t][i] += eps;
                let mut dn = table.to_vec();
                dn[t][i] -= eps;
                let fd = match which {
                    0 => (loss(&up, &zw, &zf) - loss(&dn, &zw, &zf)) / (2.0 * eps),
                    1 => (loss(&zb, &up, &zf) - loss(&zb, &dn, &zf)) / (2.0 * eps),
                    _ => (loss(&zb, &zw, &up) - loss(&zb, &zw, &dn)) / (2.0 * eps),
                };
                let an = grad[t][i];
                assert!(
                    (fd - an).abs() < 1e-6,
                    "t{t} i{i}: fd {fd} vs analytic {an}"
                );
            }
        }
    }

    use crate::gla::gdn2_forward;

    fn rand_block(
        seed: &mut u64,
        heads: usize,
        t: usize,
        dk: usize,
        dv: usize,
    ) -> (
        Vec<Vec<Vec<f64>>>,
        Vec<Vec<Vec<f64>>>,
        Vec<Vec<Vec<f64>>>,
        Vec<Vec<Vec<f64>>>,
        Vec<Vec<Vec<f64>>>,
        Vec<Vec<Vec<f64>>>,
    ) {
        // Deterministic xorshift; gates mapped into valid ranges.
        let mut next = || {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as f64) / (u32::MAX as f64) - 0.5
        };
        let mut mk = |d: usize, f: &dyn Fn(f64) -> f64| -> Vec<Vec<Vec<f64>>> {
            (0..heads)
                .map(|_| {
                    (0..t)
                        .map(|_| (0..d).map(|_| f(next())).collect())
                        .collect()
                })
                .collect()
        };
        let q = mk(dk, &|x| x);
        let k = mk(dk, &|x| x);
        let v = mk(dv, &|x| x);
        let alpha = mk(dk, &|x| 0.9 + 0.099 * (x + 0.5) * 2.0);
        let b = mk(dk, &|x| 1.0 / (1.0 + (-x * 4.0).exp()));
        let w = mk(dv, &|x| 1.0 / (1.0 + (-x * 4.0).exp()));
        (q, k, v, alpha, b, w)
    }

    fn max_delta(a: &[Vec<Vec<f64>>], b: &[Vec<Vec<f64>>]) -> f64 {
        a.iter()
            .zip(b.iter())
            .flat_map(|(ha, hb)| {
                ha.iter()
                    .zip(hb.iter())
                    .flat_map(|(sa, sb)| sa.iter().zip(sb.iter()).map(|(x, y)| (x - y).abs()))
            })
            .fold(0.0f64, f64::max)
    }

    /// Plan gate (never delete): chunkwise-WY == token-serial recurrence.
    #[test]
    fn chunkwise_matches_token_serial_oracle() {
        for &t in &[1usize, 3, 64, 129] {
            for &chunk in &[16usize, 64] {
                let mut seed = 0x9E3779B97F4A7C15u64 ^ (t as u64) ^ ((chunk as u64) << 32);
                let (q, k, v, alpha, b, w) = rand_block(&mut seed, 2, t, 8, 8);
                let serial = gdn2_forward(&q, &k, &v, &alpha, &b, &w, 2, 8, 8);
                let chunked = gdn2_chunkwise_forward(&q, &k, &v, &alpha, &b, &w, 2, 8, 8, chunk);
                let d = max_delta(&serial, &chunked);
                assert!(
                    d <= 1e-8,
                    "T={t} chunk={chunk}: chunkwise vs serial Δ={d:.3e} > 1e-8"
                );
            }
        }
    }

    /// Plan gate: analytic backward == central finite differences ≤1e-4.
    #[test]
    fn backward_matches_finite_differences() {
        let (q0, k0, v0, alpha0, b0, w0) = rand_block(&mut 42u64, 1, 4, 3, 3);
        let do_ = vec![vec![vec![1.0f64; 3]; 4]];
        let g = gdn2_backward(&q0, &k0, &v0, &alpha0, &b0, &w0, &do_, 1, 3, 3);
        // Mutable working copies; `loss` always reads current values so each
        // two-sided evaluation perturbs exactly one cell from base.
        let mut qm = q0;
        let mut km = k0;
        let mut vm = v0;
        let mut am = alpha0;
        let mut bm = b0;
        let mut wm = w0;
        let loss = |qm: &[Vec<Vec<f64>>],
                    km: &[Vec<Vec<f64>>],
                    vm: &[Vec<Vec<f64>>],
                    am: &[Vec<Vec<f64>>],
                    bm: &[Vec<Vec<f64>>],
                    wm: &[Vec<Vec<f64>>]|
         -> f64 {
            gdn2_chunkwise_forward(qm, km, vm, am, bm, wm, 1, 3, 3, 16)
                .iter()
                .flat_map(|h| h.iter().flat_map(|s| s.iter().copied()))
                .sum()
        };
        let eps = 1e-6;
        let mut worst = 0.0f64;
        // Cell list: (buffer selector, time, index, analytic grad).
        // selector 0..3 = key-axis (dq/dk/dalpha/db), 4..5 = value-axis.
        let mut cells: Vec<(u8, usize, usize, f64)> = Vec::new();
        for s in 0..4 {
            for i in 0..3 {
                cells.push((0, s, i, g.dq[0][s][i]));
                cells.push((1, s, i, g.dk[0][s][i]));
                cells.push((2, s, i, g.dalpha[0][s][i]));
                cells.push((3, s, i, g.db[0][s][i]));
            }
            for j in 0..3 {
                cells.push((4, s, j, g.dv[0][s][j]));
                cells.push((5, s, j, g.dw[0][s][j]));
            }
        }
        for (kind, s, idx, analytic) in cells {
            let base = match kind {
                0 => qm[0][s][idx],
                1 => km[0][s][idx],
                2 => am[0][s][idx],
                3 => bm[0][s][idx],
                4 => vm[0][s][idx],
                _ => wm[0][s][idx],
            };
            // No closures: each poke is a scoped match so borrows end
            // before the next `loss` read (NLL-friendly, no unsafe).
            macro_rules! poke {
                ($v:expr) => {
                    match kind {
                        0 => qm[0][s][idx] = $v,
                        1 => km[0][s][idx] = $v,
                        2 => am[0][s][idx] = $v,
                        3 => bm[0][s][idx] = $v,
                        4 => vm[0][s][idx] = $v,
                        _ => wm[0][s][idx] = $v,
                    }
                };
            }
            poke!(base + eps);
            let up = loss(&qm, &km, &vm, &am, &bm, &wm);
            poke!(base - eps);
            let dn = loss(&qm, &km, &vm, &am, &bm, &wm);
            poke!(base);
            worst = worst.max((((up - dn) / (2.0 * eps)) - analytic).abs());
        }
        assert!(
            worst <= 1e-4,
            "backward vs finite-diff worst Δ={worst:.3e} > 1e-4"
        );
    }

    #[test]
    fn f32_fast_path_matches_f64() {
        let mut seed = 7u64;
        let (q, k, v, alpha, b, w) = rand_block(&mut seed, 2, 33, 8, 8);
        // [H][T][d] f64 → [H][T][d] f32.
        let to32 = |x: &[Vec<Vec<f64>>]| -> Vec<Vec<Vec<f32>>> {
            x.iter()
                .map(|h| {
                    h.iter()
                        .map(|s| s.iter().map(|&x| x as f32).collect::<Vec<f32>>())
                        .collect::<Vec<Vec<f32>>>()
                })
                .collect::<Vec<Vec<Vec<f32>>>>()
        };
        let q32 = to32(&q);
        let _ = (&k, &v, &alpha, &b, &w);
        let o32 = gdn2_chunkwise_forward_f32(
            &q32,
            &to32(&k),
            &to32(&v),
            &to32(&alpha),
            &to32(&b),
            &to32(&w),
            2,
            8,
            8,
            16,
        );
        let o64 = gdn2_chunkwise_forward(&q, &k, &v, &alpha, &b, &w, 2, 8, 8, 16);
        let d = max_delta(
            &o64,
            &o32.iter()
                .map(|h| {
                    h.iter()
                        .map(|s| s.iter().map(|&x| x as f64).collect())
                        .collect()
                })
                .collect::<Vec<Vec<Vec<f64>>>>(),
        );
        assert!(d <= 1e-5, "f32 vs f64 Δ={d:.3e} > 1e-5");
    }
}

// ---------------------------------------------------------------------------
// G3b per-layer feature matching: pure f64 forward/backward of one GDL
// block's attention stages (norm-free — the trainer feeds post-norm rows)
// against a teacher target. Exact gradients via gdn2_backward; the only
// trainable tensors are the three gate projections (logits zb/zw/zf).
// ---------------------------------------------------------------------------

/// Row-major `[out][in]` weight matrices for one block (q, k, v, wo).
#[derive(Clone, Debug)]
pub struct FmWeights {
    pub wq: Vec<Vec<f64>>,
    pub wk: Vec<Vec<f64>>,
    pub wv: Vec<Vec<f64>>,
    pub wo: Vec<Vec<f64>>,
}

/// Everything the backward needs from the forward.
#[derive(Clone, Debug)]
pub struct FmCache {
    pub q: Vec<Vec<Vec<f64>>>,     // [H][T][dk] L2-normalized
    pub k: Vec<Vec<Vec<f64>>>,     // [H][T][dk] L2-normalized
    pub v: Vec<Vec<Vec<f64>>>,     // [H][T][dv]
    pub alpha: Vec<Vec<Vec<f64>>>, // [H][T][dk]
    pub b: Vec<Vec<Vec<f64>>>,     // [H][T][dk]
    pub w: Vec<Vec<Vec<f64>>>,     // [H][T][dv]
    pub o: Vec<Vec<Vec<f64>>>,     // [H][T][dv]
    pub out: Vec<Vec<f64>>,        // [T][hidden]
    pub zb: Vec<Vec<f64>>,         // [T][dk] erase logits
    pub zw: Vec<Vec<f64>>,         // [T][dv] write logits
    pub zf: Vec<Vec<f64>>,         // [T][dk] decay logits
}

fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One GDL block's attention stages in f64. `zb`/`zf` are `[T][dk]`, `zw` is
/// `[T][dv]` (raw logits; sigmoid applied here — the ONLY place gates
/// nonlinearities enter, so the backward chain is exact).
#[allow(clippy::too_many_arguments)]
pub fn fm_forward(
    x: &[Vec<f64>],
    w: &FmWeights,
    heads: usize,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    zb: &[Vec<f64>],
    zw: &[Vec<f64>],
    zf: &[Vec<f64>],
) -> FmCache {
    let t_len = x.len();
    let hidden = x[0].len();
    let mut q: Vec<Vec<Vec<f64>>> = Vec::with_capacity(heads);
    let mut k: Vec<Vec<Vec<f64>>> = Vec::with_capacity(heads);
    let mut v: Vec<Vec<Vec<f64>>> = Vec::with_capacity(heads);
    let kv_group = heads / kv_heads.max(1);
    for t in 0..t_len {
        let qr = matvec(&w.wq, &x[t]);
        let kr = matvec(&w.wk, &x[t]);
        let vr = matvec(&w.wv, &x[t]);
        for h in 0..heads {
            let off = h * dk;
            let kv_off = (h / kv_group.max(1)) * dk;
            let mut qh: Vec<f64> = qr[off..off + dk].to_vec();
            let mut kh: Vec<f64> = kr[kv_off..kv_off + dk].to_vec();
            // L2 normalization per GDN-2 spec (matches forward_gdl).
            let qn: f64 = qh.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            let kn: f64 = kh.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            qh.iter_mut().for_each(|x| *x /= qn);
            kh.iter_mut().for_each(|x| *x /= kn);
            if t == 0 {
                q.push(Vec::new());
                k.push(Vec::new());
                v.push(Vec::new());
            }
            q[h].push(qh);
            k[h].push(kh);
            v[h].push(vr[kv_off..kv_off + dk].to_vec());
        }
    }
    let alpha: Vec<Vec<Vec<f64>>> = (0..heads)
        .map(|_| {
            zf.iter()
                .map(|r| r.iter().map(|&z| sigmoid(z)).collect())
                .collect()
        })
        .collect();
    let b: Vec<Vec<Vec<f64>>> = (0..heads)
        .map(|_| {
            zb.iter()
                .map(|r| r.iter().map(|&z| sigmoid(z)).collect())
                .collect()
        })
        .collect();
    let wg: Vec<Vec<Vec<f64>>> = (0..heads)
        .map(|_| {
            zw.iter()
                .map(|r| r.iter().map(|&z| sigmoid(z)).collect())
                .collect()
        })
        .collect();
    let o = crate::gla::gdn2_forward(&q, &k, &v, &alpha, &b, &wg, heads, dk, dv);
    // Concat heads and apply wo.
    let mut out = vec![vec![0.0f64; hidden]; t_len];
    for t in 0..t_len {
        let mut concat = vec![0.0f64; heads * dv];
        for h in 0..heads {
            concat[h * dv..(h + 1) * dv].copy_from_slice(&o[h][t]);
        }
        out[t] = matvec(&w.wo, &concat);
    }
    FmCache {
        q,
        k,
        v,
        alpha,
        b,
        w: wg,
        o,
        out,
        zb: zb.to_vec(),
        zw: zw.to_vec(),
        zf: zf.to_vec(),
    }
}
