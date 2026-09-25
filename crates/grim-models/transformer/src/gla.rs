//! GRAVE (Gated Recurrent Attention via Vector Engraving) — f64 reference oracle.
//!
//! Implements the Gated DeltaNet-2 recurrence (arXiv 2605.22791, Eq. 8–12) as the
//! single source of truth for every GRAVE kernel and graph path. Pure f64 host
//! math, no tensor-backend dependencies: this module is the contract the HIP
//! kernels must match (plan gate G4) and the degenerate-equivalence reference
//! for vanilla GDN.
//!
//! State orientation (paper §2): `S ∈ R^{d_k × d_v}` per head; output `o_t = S_tᵀ q_t`.
//! Per token, per head:
//!   1. decay first:  `S̃_t = D_t S_{t-1}`,  `D_t = Diag(α_t)` over key channels
//!   2. gated read:   `e_t = b_t ⊙ k_t`,  `r_t = S̃_tᵀ e_t`
//!   3. gated write:  `z_t = w_t ⊙ v_t`
//!   4. delta edit:   `S_t = S̃_t + k_t (z_t − r_t)ᵀ`
//! GDN-2 gates: `b_t = σ(B x_t)` (erase), `w_t = σ(W_w x_t)` (write),
//! `α_t = exp(−exp(a) ⊙ softplus(W_f x_t + δ))` (log-decay, key channels).
//! Vanilla GDN is the degenerate case `b_t = w_t = β_t·1`, `α_t = γ_t·1`.

/// One token step for one head, f64.
///
/// Taylor-Calibrated neutral operating point (arXiv 2606.16429, adapted to
/// GDN-2's decoupled write gate): decay `α = 2^(−1/d)` per key channel,
/// erase `b = σ(0) = 0.5`, write `w = σ(0) = 0.5`.
/// Returned as `[decay, erase, write]` — the same triple `forward_gdl` and
/// the distill sidecar use, so init is defined once.
pub fn gdl_gate_defaults(head_dim: usize) -> [f64; 3] {
    let d = head_dim.max(1) as f64;
    [(-2.0f64.ln() / d).exp(), 0.5, 0.5]
}

/// TC-inspired depth-scaled init (Taylor-Calibrate-lite, arXiv 2606.16429):
/// deeper converted layers get longer memory half-lives — early hybrid layers
/// integrate local structure (ShortConv already owns exact local memory), deep
/// layers carry global associations. Erase/write stay at the neutral σ(0)=0.5
/// operating point. Half-life doubles from 8 tokens (layer 0) to 64 tokens
/// (last layer of a 16-layer stack): half-life_l = 8 · 2^(1 + 3l/L).
/// Returns one `[decay, erase, write]` triple per GDL layer, in layer order.
pub fn tc_depth_scaled_gates(
    num_layers: usize,
    gdl_layers: &[usize],
    head_dim: usize,
) -> Vec<[f64; 3]> {
    let _ = (num_layers, head_dim);
    let n = gdl_layers.len().max(1);
    gdl_layers
        .iter()
        .enumerate()
        .map(|(ordinal, &_block_idx)| {
            // Fractional depth through the GDL stack (0 = shallow, 1 = deep).
            let frac = if n <= 1 {
                0.0
            } else {
                ordinal as f64 / (n - 1) as f64
            };
            let half_life = 8.0f64 * (2.0f64).powf(1.0 + 3.0 * frac).min(64.0);
            let decay = 0.5f64.powf(1.0 / half_life);
            [decay, 0.5, 0.5]
        })
        .collect()
}

/// Checkpoint sidecar for GRAVE gates: base GGUF + `*.grave.json`, no GGUF
/// rewrite (plan §Phase 3.6). JSON (serde_json is already a dependency).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraveSidecar {
    pub format: String,
    pub arch: String,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Per-layer `[decay, erase, write]` operating point (scalar init;
    /// per-channel gate projections land with the full-corpus run).
    pub layer_gates: Vec<[f64; 3]>,
    /// G3b: per-layer gate projection weights (grave-2). `None` = grave-1
    /// (static scalar triples only). Each projection is `[out][in]` weight
    /// rows plus an `[out]` bias; sigmoid(weight·x + bias) is the per-token
    /// per-channel gate (erase b, write w, decay f).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_projections: Option<Vec<LayerGateProjections>>,
}

/// One GDL layer's three gate projections (grave-2 sidecar payload).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerGateProjections {
    /// Erase gate: `[64][hidden]` weight + `[64]` bias -> sigmoid.
    pub b_weight: Vec<Vec<f32>>,
    pub b_bias: Vec<f32>,
    /// Write gate: `[64][hidden]` weight + `[64]` bias -> sigmoid.
    pub w_weight: Vec<Vec<f32>>,
    pub w_bias: Vec<f32>,
    /// Decay gate: `[64][hidden]` weight + `[64]` bias -> sigmoid.
    pub f_weight: Vec<Vec<f32>>,
    pub f_bias: Vec<f32>,
}

impl GraveSidecar {
    pub fn new(layers: usize, heads: usize, head_dim: usize, gates: &[[f64; 3]]) -> Self {
        assert_eq!(gates.len(), layers, "sidecar needs one triple per layer");
        Self {
            format: "grave-1".into(),
            arch: "lfm2grave".into(),
            layers,
            heads,
            head_dim,
            layer_gates: gates.to_vec(),
            layer_projections: None,
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        // Auto-bump: a sidecar carrying projections must claim grave-2 so
        // loaders can validate the payload instead of silently ignoring it.
        let mut out = self.clone();
        if out.layer_projections.is_some() && out.format == "grave-1" {
            out.format = "grave-2".into();
        }
        let json = serde_json::to_string_pretty(&out)
            .map_err(|e| format!("grave sidecar serialize: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("grave sidecar write: {e}"))?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| format!("grave sidecar read: {e}"))?;
        serde_json::from_str(&raw).map_err(|e| format!("grave sidecar parse: {e}"))
    }
}

/// One token step for one head, f64.
///
/// * `state`: `[d_k][d_v]`, mutated in place (decayed then engraved).
/// * Returns the output `o_t ∈ R^{d_v}`.
pub fn gdn2_step(
    state: &mut Vec<Vec<f64>>,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    alpha: &[f64], // decay per key channel, (0, 1]
    b: &[f64],     // erase gate per key channel, [0, 1]
    w: &[f64],     // write gate per value channel, [0, 1]
) -> Vec<f64> {
    let dk = state.len();
    let dv = state[0].len();
    debug_assert_eq!(q.len(), dk);
    debug_assert_eq!(k.len(), dk);
    debug_assert_eq!(alpha.len(), dk);
    debug_assert_eq!(b.len(), dk);
    debug_assert_eq!(v.len(), dv);
    debug_assert_eq!(w.len(), dv);

    // 1. decay first (key channels are the state rows).
    let mut s_tilde: Vec<Vec<f64>> = state
        .iter()
        .enumerate()
        .map(|(i, row)| row.iter().map(|&s| alpha[i] * s).collect())
        .collect();

    // 2. gated read along the erased key direction.
    let mut r = vec![0.0f64; dv];
    for i in 0..dk {
        let e = b[i] * k[i];
        for j in 0..dv {
            r[j] += e * s_tilde[i][j];
        }
    }

    // 3+4. gated write: engrave the correction k(z − r)ᵀ, accumulate the
    // output o_t = S_tᵀ q_t against the POST-update state.
    let mut o = vec![0.0f64; dv];
    for i in 0..dk {
        let k_i = k[i];
        for j in 0..dv {
            let delta = w[j] * v[j] - r[j];
            s_tilde[i][j] += k_i * delta;
            o[j] += q[i] * s_tilde[i][j];
        }
    }
    *state = s_tilde;
    o
}

/// Full sequence over H heads. Inputs are per-head slices laid out `[H][T][dim]`;
/// returns outputs `[H][T][d_v]`.
pub fn gdn2_forward(
    q: &[Vec<Vec<f64>>],
    k: &[Vec<Vec<f64>>],
    v: &[Vec<Vec<f64>>],
    alpha: &[Vec<Vec<f64>>],
    b: &[Vec<Vec<f64>>],
    w: &[Vec<Vec<f64>>],
    heads: usize,
    _d_k: usize,
    _d_v: usize,
) -> Vec<Vec<Vec<f64>>> {
    let t = q[0].len();
    let mut states = vec![vec![vec![0.0f64; _d_v]; _d_k]; heads];
    let mut out = vec![vec![vec![0.0f64; _d_v]; t]; heads];
    for h in 0..heads {
        for step in 0..t {
            out[h][step] = gdn2_step(
                &mut states[h],
                &q[h][step],
                &k[h][step],
                &v[h][step],
                &alpha[h][step],
                &b[h][step],
                &w[h][step],
            );
        }
    }
    out
}

/// Chunkwise step for one head over a chunk of size C, f64.
///
/// Implements Gated DeltaNet-2 chunkwise WY formulation (arXiv 2605.22791, Eq. 18–25).
/// State `S ∈ R^{d_k × d_v}` is updated in-place across chunks.
/// Returns chunk outputs `[C][d_v]`.
pub fn gdn2_chunkwise_step(
    state: &mut Vec<Vec<f64>>,
    q: &[Vec<f64>],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
    alpha: &[Vec<f64>],
    b: &[Vec<f64>],
    w: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let c = q.len();
    if c == 0 {
        return Vec::new();
    }
    let dk = state.len();
    let dv = state[0].len();

    // 1. Cumulative decay within chunk: γ_r = ∏_{i=0}^r α_i (Eq. 18)
    // gamma: [C][dk]
    let mut gamma = vec![vec![1.0f64; dk]; c];
    for d in 0..dk {
        let mut running = 1.0f64;
        for r in 0..c {
            running *= alpha[r][d];
            gamma[r][d] = running;
        }
    }
    let gamma_c = &gamma[c - 1];

    // 2. Normalized keys K_bar and erase factors E_bar (Eq. 19-20)
    // K_bar[r] = γ_r^{-1} ⊙ k_r
    // E_bar[r] = γ_r ⊙ (b_r ⊙ k_r)
    // Z[r] = w_r ⊙ v_r
    let mut k_bar = vec![vec![0.0f64; dk]; c];
    let mut e_bar = vec![vec![0.0f64; dk]; c];
    let mut z = vec![vec![0.0f64; dv]; c];
    for r in 0..c {
        for d in 0..dk {
            k_bar[r][d] = k[r][d] / gamma[r][d];
            e_bar[r][d] = gamma[r][d] * (b[r][d] * k[r][d]);
        }
        for j in 0..dv {
            z[r][j] = w[r][j] * v[r][j];
        }
    }

    // 3. Strictly lower triangular T = tril(E_bar @ K_bar^T, -1) (Eq. 21)
    let mut t_mat = vec![vec![0.0f64; c]; c];
    for r in 1..c {
        for s in 0..r {
            let mut dot = 0.0f64;
            for d in 0..dk {
                dot += e_bar[r][d] * k_bar[s][d];
            }
            t_mat[r][s] = dot;
        }
    }

    // 4. Invert (I + T): since T is strictly lower triangular, solve via forward substitution.
    // A = (I + T)^{-1}
    let mut a_mat = vec![vec![0.0f64; c]; c];
    for col in 0..c {
        for row in col..c {
            if row == col {
                a_mat[row][col] = 1.0;
            } else {
                let mut sum = 0.0f64;
                for k_idx in col..row {
                    sum += t_mat[row][k_idx] * a_mat[k_idx][col];
                }
                a_mat[row][col] = -sum;
            }
        }
    }

    // 5. WY auxiliaries: Y = A @ E_bar [C, dk], U = A @ Z [C, dv] (Eq. 22)
    let mut y_mat = vec![vec![0.0f64; dk]; c];
    let mut u_mat = vec![vec![0.0f64; dv]; c];
    for r in 0..c {
        for s in 0..c {
            let a_rs = a_mat[r][s];
            if a_rs != 0.0 {
                for d in 0..dk {
                    y_mat[r][d] += a_rs * e_bar[s][d];
                }
                for j in 0..dv {
                    u_mat[r][j] += a_rs * z[s][j];
                }
            }
        }
    }

    // delta_mem = U - Y @ S_init: [C, dv]
    let mut delta_mem = vec![vec![0.0f64; dv]; c];
    for r in 0..c {
        for j in 0..dv {
            let mut ys = 0.0f64;
            for d in 0..dk {
                ys += y_mat[r][d] * state[d][j];
            }
            delta_mem[r][j] = u_mat[r][j] - ys;
        }
    }

    // 6. Intrachunk attention matrix A_qk: (A_qk)_{rs} = 1_{r >= s} * q_r^T Diag(γ_r / γ_s) k_s (Eq. 25)
    // Q_gamma[r] = γ_r ⊙ q_r
    // Output block O = Q_gamma @ S_init + A_qk @ delta_mem (Eq. 24)
    let mut chunk_out = vec![vec![0.0f64; dv]; c];
    for r in 0..c {
        // term 1: (γ_r ⊙ q_r)^T @ S_init
        for j in 0..dv {
            let mut q_s = 0.0f64;
            for d in 0..dk {
                q_s += gamma[r][d] * q[r][d] * state[d][j];
            }
            chunk_out[r][j] = q_s;
        }
        // term 2: sum_{s <= r} A_qk[r][s] * delta_mem[s]
        for s in 0..=r {
            let mut a_qk_rs = 0.0f64;
            for d in 0..dk {
                let scale = gamma[r][d] / gamma[s][d];
                a_qk_rs += q[r][d] * scale * k[s][d];
            }
            for j in 0..dv {
                chunk_out[r][j] += a_qk_rs * delta_mem[s][j];
            }
        }
    }

    // 7. Update recurrent state: S_next = Diag(γ_C) @ S_init + K_tail^T @ delta_mem (Eq. 23)
    // row r of K_tail is (γ_C / γ_r) ⊙ k_r
    let mut next_state = vec![vec![0.0f64; dv]; dk];
    for d in 0..dk {
        for j in 0..dv {
            next_state[d][j] = gamma_c[d] * state[d][j];
        }
    }
    for r in 0..c {
        for d in 0..dk {
            let k_tail_rd = (gamma_c[d] / gamma[r][d]) * k[r][d];
            for j in 0..dv {
                next_state[d][j] += k_tail_rd * delta_mem[r][j];
            }
        }
    }
    *state = next_state;

    chunk_out
}

/// Full sequence forward using chunkwise WY formulation over H heads.
pub fn gdn2_chunkwise_forward(
    q: &[Vec<Vec<f64>>],
    k: &[Vec<Vec<f64>>],
    v: &[Vec<Vec<f64>>],
    alpha: &[Vec<Vec<f64>>],
    b: &[Vec<Vec<f64>>],
    w: &[Vec<Vec<f64>>],
    heads: usize,
    _d_k: usize,
    _d_v: usize,
    chunk_size: usize,
) -> Vec<Vec<Vec<f64>>> {
    let t = q[0].len();
    let mut states = vec![vec![vec![0.0f64; _d_v]; _d_k]; heads];
    let mut out = vec![vec![vec![0.0f64; _d_v]; t]; heads];
    // Re-chunk into validated slice lengths: the uncentered intra-chunk
    // exp() is only in range up to MAX_WY_CHUNK (see gla_train).
    let chunk_size = chunk_size.max(1).min(crate::gla_train::MAX_WY_CHUNK);

    for h in 0..heads {
        let mut step = 0;
        while step < t {
            let end = (step + chunk_size).min(t);
            let q_c = &q[h][step..end];
            let k_c = &k[h][step..end];
            let v_c = &v[h][step..end];
            let alpha_c = &alpha[h][step..end];
            let b_c = &b[h][step..end];
            let w_c = &w[h][step..end];

            let chunk_out = gdn2_chunkwise_step(&mut states[h], q_c, k_c, v_c, alpha_c, b_c, w_c);
            for (idx, row) in chunk_out.into_iter().enumerate() {
                out[h][step + idx] = row;
            }
            step = end;
        }
    }
    out
}

/// Vanilla gated delta recurrence (Gated DeltaNet, arXiv 2412.06464): scalar
/// decay γ_t and scalar write strength β_t. Used by the degenerate-equivalence
/// test: GDN-2 with `b = w = β·1`, `α = γ·1` must match this exactly.
pub fn gdn_step(
    state: &mut Vec<Vec<f64>>,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    gamma: f64,
    beta: f64,
) -> Vec<f64> {
    let dk = state.len();
    let dv = state[0].len();
    // decay
    for row in state.iter_mut() {
        for s in row.iter_mut() {
            *s *= gamma;
        }
    }
    // read
    let mut r = vec![0.0f64; dv];
    for (i, &k_i) in k.iter().enumerate() {
        for (j, r_j) in r.iter_mut().enumerate() {
            *r_j += k_i * state[i][j];
        }
    }
    // delta edit
    let mut o = vec![0.0f64; dv];
    for i in 0..dk {
        let k_i = k[i];
        for j in 0..dv {
            state[i][j] += k_i * beta * (v[j] - r[j]);
            o[j] += q[i] * state[i][j];
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G3b: grave-2 sidecar round-trips projection weights without loss and
    /// grave-1 files still parse (backward compat).
    #[test]
    fn grave2_sidecar_round_trip() {
        let layers = 2usize;
        let hidden = 8usize;
        let hd = 4usize;
        let mk = || crate::gla::LayerGateProjections {
            b_weight: vec![vec![0.25f32; hidden]; hd],
            b_bias: vec![-1.5f32; hd],
            w_weight: vec![vec![0.5f32; hidden]; hd],
            w_bias: vec![0.75f32; hd],
            f_weight: vec![vec![0.125f32; hidden]; hd],
            f_bias: vec![4.25f32; hd],
        };
        let mut sc = GraveSidecar::new(layers, 1, hd, &vec![[0.9, 0.5, 0.6]; layers]);
        sc.layer_projections = Some(vec![mk(), mk()]);
        let dir = std::env::temp_dir().join("grave2-rt-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("rt.grave.json");
        sc.save(&path).unwrap();
        let back = GraveSidecar::load(&path).unwrap();
        assert_eq!(back.format, "grave-2");
        let lps = back.layer_projections.expect("projections must survive");
        assert_eq!(lps.len(), layers);
        assert_eq!(lps[0].b_bias[0], -1.5);
        assert_eq!(lps[1].f_weight[hd - 1][hidden - 1], 0.125);
        let _ = std::fs::remove_file(&path);

        // grave-1 file (no projections field) still loads.
        let g1 = format!(
            r#"{{"format":"grave-1","arch":"lfm2grave","layers":1,"heads":1,"head_dim":64,"layer_gates":[[0.9,0.5,0.6]]}}"#
        );
        let p1 = dir.join("g1.grave.json");
        std::fs::write(&p1, g1).unwrap();
        let sc1 = GraveSidecar::load(&p1).unwrap();
        assert!(sc1.layer_projections.is_none());
        let _ = std::fs::remove_file(&p1);
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn gdn2_degenerates_to_vanilla_gdn() {
        // GDN-2 with b = w = β·1 and α = γ·1 must equal the vanilla gated
        // delta recurrence token-for-token (paper §3.1, Eq. 10 recovery).
        let (dk, dv, t) = (16usize, 16usize, 32usize);
        let (gamma, beta) = (0.93f64, 0.71f64);
        let mut s_a = vec![vec![0.0f64; dv]; dk];
        let mut s_b = vec![vec![0.0f64; dv]; dk];
        for step in 0..t {
            let q = rand_vec(dk, 100 + step as u64);
            let k = rand_vec(dk, 200 + step as u64);
            let v = rand_vec(dv, 300 + step as u64);
            let alpha = vec![gamma; dk];
            let b = vec![beta; dk];
            let w = vec![beta; dv];
            let o_a = gdn2_step(&mut s_a, &q, &k, &v, &alpha, &b, &w);
            let o_b = gdn_step(&mut s_b, &q, &k, &v, gamma, beta);
            for (x, y) in o_a.iter().zip(&o_b) {
                assert!((x - y).abs() < 1e-12, "output divergence at step {step}");
            }
        }
        for (ra, rb) in s_a.iter().zip(&s_b) {
            for (x, y) in ra.iter().zip(rb) {
                assert!((x - y).abs() < 1e-12, "state divergence");
            }
        }
    }

    #[test]
    fn delta_rule_overwrites_conflicting_association() {
        // Engrave v1 under key k with full write strength; present the same
        // key with a different value at β=1: the read must return ~v2 (the
        // delta edit erased the old association along that key direction).
        let (dk, dv) = (8usize, 8usize);
        let mut state = vec![vec![0.0f64; dv]; dk];
        // Unit-norm key: the delta edit scales by kᵀk, so exact single-step
        // overwrite requires |k|² == 1.
        let mut k = rand_vec(dk, 11);
        let norm: f64 = k.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in k.iter_mut() {
            *x /= norm;
        }
        let v1 = vec![1.0f64; dv];
        let v2 = vec![-1.0f64; dv];
        let ones = vec![1.0f64; dk];
        let q = k.clone();
        let _o1 = gdn2_step(&mut state, &q, &k, &v1, &ones, &ones, &ones);
        let o2 = gdn2_step(&mut state, &q, &k, &v2, &ones, &ones, &ones);
        let o3 = gdn2_step(&mut state, &q, &k, &v2, &ones, &ones, &ones);
        for (b, c) in o2.iter().zip(&o3) {
            assert!((b - v2[0]).abs() < 1e-9, "t2 must write v2, got {b}");
            assert!((c - v2[0]).abs() < 1e-9, "t3 must read v2, got {c}");
        }
    }

    #[test]
    fn decay_fades_old_associations() {
        // With β=1 writes and zero further input, aggressive decay must shrink
        // the read magnitude monotonically — the "forgetting" property.
        let (dk, dv) = (8usize, 8usize);
        let mut state = vec![vec![0.0f64; dv]; dk];
        let k = rand_vec(dk, 21);
        let v1 = vec![1.0f64; dv];
        let ones = vec![1.0f64; dk];
        let zeros_v = vec![0.0f64; dv];
        let alpha = vec![0.5f64; dk]; // aggressive decay
        let q = k.clone();
        let _ = gdn2_step(&mut state, &q, &k, &v1, &ones, &ones, &ones);
        let mut reads = Vec::new();
        for _ in 0..6 {
            let o = gdn2_step(&mut state, &q, &k, &zeros_v, &alpha, &ones, &ones);
            let mag: f64 = o.iter().map(|x| x.abs()).sum();
            reads.push(mag);
        }
        for w in reads.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "read magnitude must not grow: {reads:?}"
            );
        }
        assert!(reads[reads.len() - 1] < reads[0], "must fade: {reads:?}");
    }

    #[test]
    fn zero_erase_gate_leaves_orthogonal_rows_untouched() {
        // b = 0 on key channel i ⇒ that channel's erase contribution vanishes,
        // so state rows whose k_i = 0 are only decayed, never engraved.
        let (dk, dv) = (8usize, 8usize);
        // one-hot key on channel 0
        let mut k = vec![0.0f64; dk];
        k[0] = 1.0;
        let mut state = vec![vec![0.7f64; dv]; dk];
        let before = state.clone();
        let q = vec![0.0f64; dk]; // output unused here
        let v = rand_vec(dv, 32);
        let alpha = vec![1.0f64; dk]; // no decay
        let b = vec![0.0f64; dk]; // erase gate fully closed
        let w = vec![1.0f64; dv];
        let _ = gdn2_step(&mut state, &q, &k, &v, &alpha, &b, &w);
        // rows 1.. (k_i = 0, no decay) must be exactly untouched.
        for (i, (row, row_before)) in state.iter().zip(&before).enumerate() {
            if i == 0 {
                continue;
            }
            for (x, y) in row.iter().zip(row_before) {
                assert!((x - y).abs() < 1e-12, "row {i} must be untouched");
            }
        }
    }

    #[test]
    fn chunkwise_matches_token_serial_oracle() {
        // Plan Phase 2.2: chunkwise-parallel form == token-serial recurrence,
        // max|Δ| ≤ 1e-4 (matches < 1e-12 in f64) on random inputs at lengths 1, 3, 64, 129.
        let (heads, dk, dv) = (2usize, 8usize, 8usize);
        let test_lengths = [1usize, 3usize, 64usize, 129usize];
        let chunk_sizes = [16usize, 64usize];

        for &t in &test_lengths {
            let mut q = vec![vec![vec![0.0f64; dk]; t]; heads];
            let mut k = vec![vec![vec![0.0f64; dk]; t]; heads];
            let mut v = vec![vec![vec![0.0f64; dv]; t]; heads];
            let mut alpha = vec![vec![vec![0.0f64; dk]; t]; heads];
            let mut b = vec![vec![vec![0.0f64; dk]; t]; heads];
            let mut w = vec![vec![vec![0.0f64; dv]; t]; heads];

            for h in 0..heads {
                for step in 0..t {
                    let seed = ((h * 1000 + step) as u64).wrapping_mul(1234567);
                    q[h][step] = rand_vec(dk, seed + 1);
                    let mut k_row = rand_vec(dk, seed + 2);
                    let norm: f64 = k_row.iter().map(|x| x * x).sum::<f64>().sqrt();
                    for x in k_row.iter_mut() {
                        *x /= norm.max(1e-12);
                    }
                    k[h][step] = k_row;
                    v[h][step] = rand_vec(dv, seed + 3);
                    // alpha in (0.7, 0.99)
                    alpha[h][step] = rand_vec(dk, seed + 4)
                        .into_iter()
                        .map(|x| 0.85 + 0.14 * x)
                        .collect();
                    // b and w in (0.1, 0.9)
                    b[h][step] = rand_vec(dk, seed + 5)
                        .into_iter()
                        .map(|x| 0.5 + 0.4 * x)
                        .collect();
                    w[h][step] = rand_vec(dv, seed + 6)
                        .into_iter()
                        .map(|x| 0.5 + 0.4 * x)
                        .collect();
                }
            }

            let o_serial = gdn2_forward(&q, &k, &v, &alpha, &b, &w, heads, dk, dv);

            for &chunk_size in &chunk_sizes {
                let o_chunk =
                    gdn2_chunkwise_forward(&q, &k, &v, &alpha, &b, &w, heads, dk, dv, chunk_size);
                for h in 0..heads {
                    for step in 0..t {
                        for j in 0..dv {
                            let diff = (o_serial[h][step][j] - o_chunk[h][step][j]).abs();
                            assert!(
                                diff < 1e-11,
                                "T={t}, chunk={chunk_size}, head={h}, step={step}, j={j}: divergence {diff} >= 1e-11"
                            );
                        }
                    }
                }
            }
        }
    }
}
