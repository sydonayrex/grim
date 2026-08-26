//! Reusable CPU reference-attention helpers for cache-aware decode.
//!
//! Several transformer families in this crate implement their attention as a
//! hand-rolled CPU loop with no KV-cache integration, so a single-token decode
//! step silently attended only over itself (the model's prior context was
//! invisible). These helpers give those loops a single, correct cache-append +
//! cache-aware causal-attention implementation so each model file wires it the
//! same way. [Group B fix.]
//!
//! The cache stores post-RoPE keys and raw values in one contiguous flat buffer
//! per layer: `(past_len, row_elems)` where `row_elems` is `num_heads*head_dim`
//! (MHA/MLA) or `num_kv_heads*head_dim` (GQA). `KvCache::current_k`/`current_v`
//! are scoped to the most-recently-appended slot and the paged variant is
//! block-addressed, so neither fits a naive full-history reference loop — hence
//! this per-layer buffer parked in `session.model_state` (mirroring `lfm2.rs`).

use grim_tensor::{Device, Shape, Tensor};

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;

/// Per-layer KV cache for a CPU reference-attention model.
#[derive(Clone, Default)]
pub struct RefKvCache {
    /// Post-RoPE keys, `(past_len, row_elems)`.
    pub k: Vec<f32>,
    /// Raw values (never RoPE'd), `(past_len, row_elems)`.
    pub v: Vec<f32>,
    /// Number of cached token positions.
    pub past_len: usize,
}

impl RefKvCache {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Append one call's post-RoPE `k` and raw `v` (each `[new_tokens, row_elems]`)
/// and return the full `(total_len, row_elems)` history slices to attend over.
pub fn append_and_get<'a>(
    cache: &'a mut RefKvCache,
    k_new: &Tensor,
    v_new: &[f32],
) -> Result<(&'a [f32], &'a [f32], usize)> {
    // Audit fix (grim-models M12): require an explicit row width — the old
    // `.unwrap_or(0).max(1)` fallback divided by 1 for malformed inputs and
    // mis-derived total_len from the raw element count.
    let dims = k_new.shape().dims();
    let Some(&row_elems) = dims.get(1).filter(|&&d| d > 0) else {
        return Err(grim_core::error::Error::Shape(format!(
            "append_and_get: k_new must be [tokens, row_elems] (rank>=2, nonzero width), got {:?}",
            dims
        )));
    };
    if v_new.len() != k_new.shape().elem_count() {
        return Err(grim_core::error::Error::Shape(format!(
            "append_and_get: v len {} != k elem count {}",
            v_new.len(),
            k_new.shape().elem_count()
        )));
    }
    cache.k.extend_from_slice(&k_new.to_vec_f32()?);
    cache.v.extend_from_slice(v_new);
    let total_len = cache.k.len() / row_elems;
    cache.past_len = total_len;
    Ok((&cache.k, &cache.v, total_len))
}

/// Cache-aware scaled-dot-product causal attention over the full K/V history.
///
/// * `q` is `(new_tokens, q_row_elems)` where `q_row_elems = num_heads*head_dim`.
/// * `k`/`v` are the full history `(total_len, kv_row_elems)`.
/// * `past_len` is the number of history tokens already present before this
///   call, so query index `t` (0-based within this call) attends over absolute
///   positions `0..=past_len+t` — i.e. it can see the current token and all
///   prior ones, which a stateless single-token forward could not.
/// * `kv_head` maps each query head to its KV head index for GQA.
#[allow(clippy::too_many_arguments)]
pub fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    new_tokens: usize,
    total_len: usize,
    past_len: usize,
    num_heads: usize,
    head_dim: usize,
    q_row_elems: usize,
    kv_row_elems: usize,
    kv_head: &[usize],
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; new_tokens * q_row_elems];

    for h in 0..num_heads {
        let kh = kv_head.get(h).copied().unwrap_or(h);
        for t in 0..new_tokens {
            let abs_t = past_len + t;
            let last = abs_t.min(total_len - 1);
            let mut scores = vec![f32::NEG_INFINITY; total_len];
            for t2 in 0..=last {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[t * q_row_elems + h * head_dim + d]
                        * k[t2 * kv_row_elems + kh * head_dim + d];
                }
                scores[t2] = dot * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut().take(last + 1) {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in scores.iter_mut().take(last + 1) {
                *s /= sum;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t2 in 0..=last {
                    acc += scores[t2] * v[t2 * kv_row_elems + kh * head_dim + d];
                }
                out[t * q_row_elems + h * head_dim + d] = acc;
            }
        }
    }
    out
}

/// Build a `(1, S, D)` 3-D view for `Rope`, which requires 3-D input.
pub fn as_3d(t: &Tensor, s: usize, d: usize) -> Tensor {
    Tensor::new(
        t.storage().clone(),
        Shape::new(vec![1, s, d]),
        t.dtype(),
        t.provenance().clone(),
        t.device().clone(),
    )
}

/// Relabel a 3-D `(1, S, D)` tensor back to 2-D `(S, D)`.
pub fn as_2d(t: &Tensor, s: usize, d: usize) -> Tensor {
    Tensor::new(
        t.storage().clone(),
        Shape::new(vec![s, d]),
        t.dtype(),
        t.provenance().clone(),
        t.device().clone(),
    )
}

/// Make a CPU f32 tensor (convenience re-export so callers don't reach into
/// the backend crate directly for the V buffer).
pub fn f32_tensor(data: Vec<f32>, shape: Shape) -> Tensor {
    cpu_tensor(data, shape)
}

/// Re-export the CPU device marker for callers that build tensors directly.
pub fn cpu() -> Device {
    Device::Cpu
}


#[cfg(test)]
mod audit_tests {
    use super::*;

    /// Audit gate (M12): malformed inputs to append_and_get must error, not
    /// divide by the `.max(1)` fallback and corrupt total_len.
    #[test]
    fn append_and_get_rejects_malformed_inputs() {
        use grim_tensor::Shape;
        let mut cache = RefKvCache::new();
        // Rank-1 k_new (no row width).
        let k_bad =
            cpu_tensor(vec![1.0f32, 2.0], Shape::new(vec![2]));
        let res = append_and_get(&mut cache, &k_bad, &[1.0, 2.0]);
        assert!(res.is_err(), "rank-1 k_new must error");

        // v length mismatch.
        let k_ok = cpu_tensor(vec![1.0f32, 2.0], Shape::new(vec![1, 2]));
        let res = append_and_get(&mut cache, &k_ok, &[1.0]);
        assert!(res.is_err(), "v/k length mismatch must error");

        // Well-formed call still round-trips.
        // One [1, 2] row appended → one cached token.
        let (_, _, total) =
            append_and_get(&mut cache, &cpu_tensor(vec![3.0f32, 4.0], Shape::new(vec![1, 2])), &[3.0, 4.0])
                .unwrap();
        assert_eq!(total, 1);
        assert_eq!(cache.past_len, 1);
    }
}
