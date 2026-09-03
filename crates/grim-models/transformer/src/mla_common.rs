//! Shared host-side math for Multi-head Latent Attention (MLA) models
//! (deepseek2 / deepseek32 / deepseek4 / kimi_k3).

/// Split q_flat `[seq, num_heads * (nope + rope_d)]` into per-head
/// `(q_nope, q_rope)` planes.
pub fn split_q_nope_rope(
    q_full: &[f32],
    seq_len: usize,
    num_heads: usize,
    nope: usize,
    rope_d: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total_q_head = nope + rope_d;
    let mut q_nope = vec![0.0f32; seq_len * num_heads * nope];
    let mut q_rope = vec![0.0f32; seq_len * num_heads * rope_d];
    for s in 0..seq_len {
        for h in 0..num_heads {
            let in_off = s * num_heads * total_q_head + h * total_q_head;
            let nope_off = s * num_heads * nope + h * nope;
            let rope_off = s * num_heads * rope_d + h * rope_d;
            q_nope[nope_off..nope_off + nope].copy_from_slice(&q_full[in_off..in_off + nope]);
            q_rope[rope_off..rope_off + rope_d]
                .copy_from_slice(&q_full[in_off + nope..in_off + total_q_head]);
        }
    }
    (q_nope, q_rope)
}

/// Split the kv-a projection `[seq, rank + rope_d]` into the compressed latent
/// `kv_a` rows and the shared rope key rows.
pub fn split_kv_latent(
    kv_latent: &[f32],
    seq_len: usize,
    rank: usize,
    rope_d: usize,
) -> (Vec<f32>, Vec<f32>) {
    let row = rank + rope_d;
    let mut kv_a = vec![0.0f32; seq_len * rank];
    let mut k_rope = vec![0.0f32; seq_len * rope_d];
    for s in 0..seq_len {
        let in_off = s * row;
        kv_a[s * rank..(s + 1) * rank].copy_from_slice(&kv_latent[in_off..in_off + rank]);
        k_rope[s * rope_d..(s + 1) * rope_d]
            .copy_from_slice(&kv_latent[in_off + rank..in_off + row]);
    }
    (kv_a, k_rope)
}

/// Apply neox-style rope with the MLA-standard theta.
pub fn apply_rope_on_latent(v: &mut [f32], positions: &[u32], num_heads: usize, head_dim: usize) {
    crate::qwen35::apply_rope_neox(v, positions, num_heads, head_dim, 10000.0);
}

/// Extract per-head key/value up-projections from the kv_b_proj weight
/// `[num_heads * (nope + v), rank]` (GGUF row-major) into
/// `(w_kc [nh, nope, rank], w_vc [nh, v, rank])`.
pub fn extract_kv_b_up_projs(
    kv_b_w: &[f32],
    num_heads: usize,
    nope: usize,
    v_dim: usize,
    rank: usize,
) -> (Vec<f32>, Vec<f32>) {
    let kv_b_head = nope + v_dim;
    let mut w_kc = vec![0.0f32; num_heads * nope * rank];
    let mut w_vc = vec![0.0f32; num_heads * v_dim * rank];
    for h in 0..num_heads {
        let hb = h * kv_b_head;
        for d in 0..nope {
            let src = (hb + d) * rank;
            let dst = (h * nope + d) * rank;
            w_kc[dst..dst + rank].copy_from_slice(&kv_b_w[src..src + rank]);
        }
        for d in 0..v_dim {
            let src = (hb + nope + d) * rank;
            let dst = (h * v_dim + d) * rank;
            w_vc[dst..dst + rank].copy_from_slice(&kv_b_w[src..src + rank]);
        }
    }
    (w_kc, w_vc)
}

/// Absorb the per-head key up-projection into the query:
/// `q_absorbed[s,h] = q_nope[s,h] @ w_kc[h]^T`,
/// since `q_nope · (w_kc c) == (q_nope w_kc) · c`.
pub fn absorb_query_wkc(
    q_nope: &[f32],
    w_kc: &[f32],
    seq_len: usize,
    num_heads: usize,
    nope: usize,
    rank: usize,
) -> Vec<f32> {
    let mut q_absorbed = vec![0.0f32; seq_len * num_heads * rank];
    for s in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..nope {
                let qv = q_nope[(s * num_heads + h) * nope + d];
                let wrow = &w_kc[(h * nope + d) * rank..(h * nope + d + 1) * rank];
                let dst =
                    &mut q_absorbed[(s * num_heads + h) * rank..(s * num_heads + h + 1) * rank];
                for (o, w) in dst.iter_mut().zip(wrow.iter()) {
                    *o += qv * w;
                }
            }
        }
    }
    q_absorbed
}

/// Pack compressed latent rows `[seq, rank + rope_d]` as `[normed c_kv || roped k_pe]`.
pub fn pack_latent_rows(
    kv_a_normed: &[f32],
    k_rope: &[f32],
    seq_len: usize,
    rank: usize,
    rope_d: usize,
) -> Vec<f32> {
    let row = rank + rope_d;
    let mut latent = vec![0.0f32; seq_len * row];
    for s in 0..seq_len {
        let dst = s * row;
        latent[dst..dst + rank].copy_from_slice(&kv_a_normed[s * rank..(s + 1) * rank]);
        latent[dst + rank..dst + row].copy_from_slice(&k_rope[s * rope_d..(s + 1) * rope_d]);
    }
    latent
}
