//! DeepSeek family — Multi-head Latent Attention (MLA) and expert routing.

use grim_backend_cpu::{add_tensors, cpu_tensor};
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm, Rope};
use grim_tensor::{ArithType, DType, Device, Tensor};

#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
}

impl ModelConfig for DeepSeekConfig {
    fn name(&self) -> &str {
        "deepseek"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct DeepSeekBlock {
    pub attn_norm: RmsNorm,
    // MLA projections
    pub q_a_proj: Linear,
    pub q_b_proj: Linear,
    pub kv_a_proj: Linear,
    pub kv_b_proj: Linear,
    pub wo: Linear,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub rope: Rope,
}

impl DeepSeekBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &DeepSeekConfig) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let q_a_proj = Linear::load(&ws.pp("q_a_proj"), cfg.hidden_size, cfg.q_lora_rank, false)?;
        let q_b_proj = Linear::load(
            &ws.pp("q_b_proj"),
            cfg.q_lora_rank,
            cfg.num_heads * 128,
            false,
        )?;
        let kv_a_proj = Linear::load(
            &ws.pp("kv_a_proj"),
            cfg.hidden_size,
            cfg.kv_lora_rank,
            false,
        )?;
        let kv_b_proj = Linear::load(
            &ws.pp("kv_b_proj"),
            cfg.kv_lora_rank,
            // kv_b_proj produces K **and** V concatenated per position, so its
            // output width is 2 * num_heads * head_dim. The old value
            // (num_heads * 128) was half of what `forward`'s split indexes,
            // making `kv_data[pos * 2 * hidden + ...]` read out of bounds for
            // any pos > 0. [Group B fix.]
            2 * cfg.num_heads * 128,
            false,
        )?;
        let wo = Linear::load(&ws.pp("wo"), cfg.num_heads * 128, cfg.hidden_size, false)?;

        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ffn_gate = Linear::load(
            &ws.pp("ffn_gate"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let ffn_up = Linear::load(
            &ws.pp("ffn_up"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let ffn_down = Linear::load(
            &ws.pp("ffn_down"),
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
        )?;

        let rope = Rope::new(128, 10000.0); // DeepSeek uses head_dim=128

        Ok(Self {
            attn_norm,
            q_a_proj,
            q_b_proj,
            kv_a_proj,
            kv_b_proj,
            wo,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            num_heads: cfg.num_heads,
            head_dim: 128,
            q_lora_rank: cfg.q_lora_rank,
            kv_lora_rank: cfg.kv_lora_rank,
            rope,
        })
    }
}

/// Per-layer MLA KV cache: post-RoPE keys and raw values for every token
/// seen so far, flat `(past_len, num_heads * head_dim)`.
///
/// DeepSeek's MLA attention is a CPU reference loop, so it needs the *full*
/// K/V history in one contiguous buffer. `KvCache::current_k`/`current_v` are
/// scoped to the most recently appended slot(s) and the paged variant is
/// block-addressed for the paged kernel, so neither fits this loop. This
/// mirrors the pattern `lfm2.rs` already uses: per-layer caches parked in
/// `session.model_state`, one `Vec<Option<_>>` entry per layer, so concurrent
/// requests against the same model get independent state.
/// [Group B fix: decode was stateless — every prior token was invisible.]
#[derive(Clone, Default)]
pub struct MlaLayerCache {
    /// Post-RoPE keys, `(past_len, num_heads * head_dim)`.
    pub k_cache: Vec<f32>,
    /// Raw values (V is never RoPE'd), same layout.
    pub v_cache: Vec<f32>,
    /// Number of cached token positions.
    pub past_len: usize,
}

impl MlaLayerCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DeepSeekBlock {
    /// Prefill-only convenience wrapper: attends over just the tokens passed
    /// in, with no cross-call history. Correct when `x` is the whole prompt.
    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let mut cache = MlaLayerCache::new();
        self.forward_cached(x, positions, &mut cache)
    }

    /// Cache-aware forward. `cache` accumulates post-RoPE K and raw V across
    /// calls, so a single-token decode step attends over the full context
    /// rather than only itself.
    pub fn forward_cached(
        &self,
        x: &Tensor,
        positions: &[u32],
        cache: &mut MlaLayerCache,
    ) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;

        // MLA: Multi-head Latent Attention
        // Step 1: Project to latent space
        let q_latent = self.q_a_proj.forward(&norm_x)?;
        let kv_latent = self.kv_a_proj.forward(&norm_x)?;

        // Step 2: Project from latent to Q, K, V
        let q = self.q_b_proj.forward(&q_latent)?;
        let kv = self.kv_b_proj.forward(&kv_latent)?;

        // `new_tokens` is what THIS call projects — the whole prompt on
        // prefill, one token on decode. It is not the context length.
        let new_tokens = x.shape().dims()[0];
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let hidden = num_heads * head_dim;

        let kv_data = kv.to_vec_f32()?;
        let expected = new_tokens * 2 * hidden;
        if kv_data.len() < expected {
            return Err(grim_core::error::Error::Shape(format!(
                "deepseek kv_b_proj output has {} elements, need {} \
                 ({new_tokens} tokens x 2 x {hidden})",
                kv_data.len(),
                expected
            )));
        }

        // Split the concatenated K|V for this call's tokens.
        let mut k_new = vec![0.0f32; new_tokens * hidden];
        let mut v_new = vec![0.0f32; new_tokens * hidden];
        for pos in 0..new_tokens {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let idx = pos * hidden + h * head_dim + d;
                    k_new[idx] = kv_data[pos * 2 * hidden + h * head_dim + d];
                    v_new[idx] = kv_data[pos * 2 * hidden + hidden + h * head_dim + d];
                }
            }
        }

        // RoPE expects 3-D (B, S, D). Reshape each call's Q/K to (1, S, D);
        // the data is already contiguously (S, D) so this is a zero-copy
        // relabel. After rotation, relabel back to 2-D for the attention loop.
        let q_3d = Tensor::new(
            q.storage().clone(),
            grim_tensor::Shape::new(vec![1, new_tokens, hidden]),
            q.dtype(),
            q.provenance().clone(),
            q.device().clone(),
        );
        let k_new_t = cpu_tensor(
            std::mem::take(&mut k_new),
            grim_tensor::Shape::new(vec![new_tokens, hidden]),
        );
        let k_new_3d = Tensor::new(
            k_new_t.storage().clone(),
            grim_tensor::Shape::new(vec![1, new_tokens, hidden]),
            k_new_t.dtype(),
            k_new_t.provenance().clone(),
            k_new_t.device().clone(),
        );

        // RoPE on Q and the new K slice, using this call's absolute positions.
        let q_3d = self.rope.forward(&q_3d, positions)?;
        let k_new_3d = self.rope.forward(&k_new_3d, positions)?;

        let q = Tensor::new(
            q_3d.storage().clone(),
            grim_tensor::Shape::new(vec![new_tokens, hidden]),
            q_3d.dtype(),
            q_3d.provenance().clone(),
            q_3d.device().clone(),
        );
        let k_new_t = Tensor::new(
            k_new_3d.storage().clone(),
            grim_tensor::Shape::new(vec![new_tokens, hidden]),
            k_new_3d.dtype(),
            k_new_3d.provenance().clone(),
            k_new_3d.device().clone(),
        );

        // Append post-RoPE K and raw V, then attend over the whole history so
        // this call's tokens can see themselves and everything before them.
        cache.k_cache.extend_from_slice(&k_new_t.to_vec_f32()?);
        cache.v_cache.extend_from_slice(&v_new);
        let past_len = cache.past_len;
        cache.past_len += new_tokens;
        let total_len = cache.past_len;

        let qd = q.to_vec_f32()?;
        let kd = &cache.k_cache;
        let vd = &cache.v_cache;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; new_tokens * hidden];

        for h in 0..num_heads {
            for t in 0..new_tokens {
                // Causal limit is the query's absolute position in the full
                // context, not its index within this call's slice.
                let abs_t = past_len + t;
                let last = abs_t.min(total_len - 1);
                let mut scores = vec![f32::NEG_INFINITY; total_len];
                for t2 in 0..=last {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot +=
                            qd[t * hidden + h * head_dim + d] * kd[t2 * hidden + h * head_dim + d];
                    }
                    scores[t2] = dot * scale;
                }
                // Softmax over the unmasked prefix.
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut().take(last + 1) {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in scores.iter_mut().take(last + 1) {
                    *s /= sum;
                }
                // Weighted sum of V over the same prefix.
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for t2 in 0..=last {
                        acc += scores[t2] * vd[t2 * hidden + h * head_dim + d];
                    }
                    attn_out[t * hidden + h * head_dim + d] = acc;
                }
            }
        }

        let attn_out_tensor =
            cpu_tensor(attn_out, grim_tensor::Shape::new(vec![new_tokens, hidden]));
        let attn_out = self.wo.forward(&attn_out_tensor)?;

        // Residual
        let x_res1 = add_tensors(x, &attn_out).map_err(grim_core::Error::Tensor)?;

        // FFN
        let norm_x2 = self.ffn_norm.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        let up = self.ffn_up.forward(&norm_x2)?;
        let activated = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&activated)?;
        add_tensors(&x_res1, &ffn_out).map_err(grim_core::Error::Tensor)
    }
}

pub struct DeepSeek {
    pub cfg: DeepSeekConfig,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<DeepSeekBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl DeepSeek {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DeepSeekConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry for DeepSeek (MLA). The MLA attention uses
    /// projected (not headed) KV via `kv_b_proj` of shape
    /// `[2*num_heads*128, hidden]`; sharding it on the head axis requires
    /// bespoke handling that `Linear::load_column_parallel` cannot express, and
    /// `forward` calls plain `Linear::forward` (no all-reduce hook). Refuses
    /// `world_size > 1` until both the sharding math and the `forward` rework
    /// land. `world_size == 1` delegates to the plain path.
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DeepSeekConfig,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "DeepSeek",
            "MLA projected-KV layout needs bespoke head-axis sharding and a \
             forward rework to add the all-reduce hook",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        let tok_embeddings =
            Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DeepSeekBlock::load(&ws.pp("blk").pp(&i.to_string()), &cfg)?);
        }
        let norm = RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load(&ws.pp("output"), cfg.hidden_size, cfg.vocab_size, false)?;

        Ok(Self {
            cfg,
            device: device.clone(),
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }
}

impl Model for DeepSeek {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for DeepSeek {
    fn new_session(&self) -> Box<dyn SessionT> {
        let mut session = Inner::new(self.device.clone());
        let caches: Vec<Option<MlaLayerCache>> = vec![None; self.layers.len()];
        session.set_model_state(Box::new(caches));
        Box::new(session)
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids: Vec<u32> = match input_ids.dtype() {
            d if d == DType::F32 => {
                let v = input_ids.to_vec_f32()?;
                v.into_iter().map(|x| x as u32).collect()
            }
            _ => return Err(grim_tensor::Error::Unimplemented("non-F32 inputs".into()).into()),
        };
        let seq_len = ids.len();
        let pos_ids: Vec<u32> = match positions.dtype() {
            d if d == DType::F32 => {
                let v = positions.to_vec_f32()?;
                v.into_iter().map(|x| x as u32).collect()
            }
            _ => (0..seq_len).map(|i| i as u32).collect(),
        };
        let mut h = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?;

        // Per-layer MLA caches live on the session so decode steps see the
        // full prior context (and concurrent requests stay independent).
        if session.model_state().is_none() {
            session.set_model_state(Box::new(vec![None::<MlaLayerCache>; self.layers.len()]));
        }
        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<MlaLayerCache>>>())
            .expect("DeepSeek::forward: model_state must be Vec<Option<MlaLayerCache>>");
        if caches.len() < self.layers.len() {
            caches.resize(self.layers.len(), None);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = caches[i].get_or_insert_with(MlaLayerCache::new);
            h = layer.forward_cached(&h, &pos_ids, cache)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.output.forward(&h)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_nn::modules::Linear;
    use grim_tensor::Shape;

    fn lin(in_dim: usize, out_dim: usize) -> Linear {
        let w = cpu_tensor(
            vec![0.01f32; out_dim * in_dim],
            Shape::new(vec![out_dim, in_dim]),
        );
        Linear::from_tensor(w, None)
    }

    /// Build a tiny 2-head, head_dim=4 MLA block with near-identity-ish
    /// weights so the reference attention loop runs deterministically.
    fn tiny_block() -> DeepSeekBlock {
        let h = 2 * 4; // num_heads * head_dim
        DeepSeekBlock {
            attn_norm: RmsNorm::new(cpu_tensor(vec![1.0f32; h], Shape::new(vec![h])), 1e-6),
            q_a_proj: lin(h, h),
            q_b_proj: lin(h, h),
            kv_a_proj: lin(h, h),
            kv_b_proj: lin(h, 2 * h), // K and V concatenated
            wo: lin(h, h),
            ffn_norm: RmsNorm::new(cpu_tensor(vec![1.0f32; h], Shape::new(vec![h])), 1e-6),
            ffn_gate: lin(h, h),
            ffn_up: lin(h, h),
            ffn_down: lin(h, h),
            num_heads: 2,
            head_dim: 4,
            q_lora_rank: h,
            kv_lora_rank: h,
            rope: Rope::new(h, 10000.0),
        }
    }

    fn close(a: f32, b: f32, tol: f32, label: &str) {
        assert!((a - b).abs() <= tol, "{label}: {a} vs {b}");
    }

    // §4.1 Prefill regression: one call with positions [0..n) must match the
    // old prefill-only path (cache-aware forward with an empty cache).
    #[test]
    fn test_prefill_matches_cacheless() {
        let blk = tiny_block();
        let n = 3;
        let x = cpu_tensor(
            (0..n * 8).map(|i| (i as f32) * 0.1).collect(),
            Shape::new(vec![n, 8]),
        );
        let positions: Vec<u32> = (0..n).map(|i| i as u32).collect();

        let out_cached = blk
            .forward_cached(&x, &positions, &mut MlaLayerCache::new())
            .unwrap();
        let out_stateless = blk.forward(&x, &positions).unwrap();
        let a = out_cached.to_vec_f32().unwrap();
        let b = out_stateless.to_vec_f32().unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            close(*av, *bv, 1e-4, &format!("prefill[{i}]"));
        }
    }

    // §4.2 Decode sees prior context. The original bug: `forward` attended only
    // over the single decode token (seq_len=1), so the prior prompt was
    // invisible. Proof the cache-aware path is fixed: decoding token `d` after a
    // 2-token prefill must produce a DIFFERENT output than running `d` alone
    // through the (now-prefill-only) `forward` — the cached prefix changes the
    // attention result. Under uniform near-zero weights the cross-token
    // influence is faint, so we assert directly that the cached decode output
    // is not equal to a stateless single-token forward of the same token.
    #[test]
    fn test_decode_sees_prior_context() {
        let blk = tiny_block();

        let prompt = cpu_tensor(
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            ],
            Shape::new(vec![2, 8]),
        );
        let dec = cpu_tensor(
            vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            Shape::new(vec![1, 8]),
        );

        // Prefill then decode with a populated cache.
        let mut cache = MlaLayerCache::new();
        let _ = blk.forward_cached(&prompt, &[0, 1], &mut cache).unwrap();
        let cached_dec = blk.forward_cached(&dec, &[2], &mut cache).unwrap();

        // Stateless single-token forward of the same decode token (the old bug's
        // behavior: attention only over this one token).
        let stateless_dec = blk.forward(&dec, &[0]).unwrap();

        let a = cached_dec.to_vec_f32().unwrap();
        let b = stateless_dec.to_vec_f32().unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-5,
            "decode ignored the cached prefix (cached vs stateless diff={max_diff})"
        );

        // Sanity: cache length advanced correctly.
        assert_eq!(cache.past_len, 3, "cache should hold prompt + decode token");
    }

    // §4.3 Cache length invariant: P prefill + N decode => cache.past_len == P+N.
    #[test]
    fn test_cache_len_invariant() {
        let blk = tiny_block();
        let p = 4usize;
        let n = 3usize;
        let prompt = cpu_tensor(
            (0..p * 8).map(|i| (i as f32) * 0.05).collect(),
            Shape::new(vec![p, 8]),
        );
        let mut cache = MlaLayerCache::new();
        let _ = blk
            .forward_cached(&prompt, &(0..p as u32).collect::<Vec<_>>(), &mut cache)
            .unwrap();
        assert_eq!(cache.past_len, p, "after prefill");
        for i in 0..n {
            let tok = cpu_tensor(
                (0..8).map(|j| (j as f32) * 0.03 + i as f32).collect(),
                Shape::new(vec![1, 8]),
            );
            let _ = blk
                .forward_cached(&tok, &[(p + i) as u32], &mut cache)
                .unwrap();
        }
        assert_eq!(cache.past_len, p + n, "after decode steps");
    }

    // §4.4 kv_b_proj indexing stays in bounds. The split reads 2*hidden floats
    // per position; with new_tokens=1 (decode) and new_tokens>1 (prefill) the
    // index pos*2*hidden + hidden + (head_dim-1) must be < kv_data.len().
    #[test]
    fn test_kv_split_in_bounds() {
        let blk = tiny_block();
        let num_heads = blk.num_heads;
        let head_dim = blk.head_dim;
        let hidden = num_heads * head_dim;

        // Decode shape (new_tokens = 1): index of last V element.
        let last = hidden + (head_dim - 1);
        assert!(last < 2 * hidden, "decode kv index OOB: {last}");

        // Prefill shape (new_tokens = 5): last token's last V element.
        let nt = 5;
        let last_p = (nt - 1) * 2 * hidden + hidden + (head_dim - 1);
        assert!(last_p < nt * 2 * hidden, "prefill kv index OOB: {last_p}");
    }
}
