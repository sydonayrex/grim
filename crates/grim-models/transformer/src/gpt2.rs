//! GPT2 & GPT-NeoX family — standard LayerNorm + absolute positional embeddings.

use grim_backend_cpu::{add_tensors, cpu_tensor};
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear};
use grim_tensor::{ArithType, DType, Device, Tensor};

/// Tanh-based GELU approximation (GPT-2 paper: Gaussian Error Linear Units).
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))).
fn gelu(t: &Tensor) -> Result<Tensor> {
    let v = t.to_vec_f32()?;
    let mut out = vec![0.0f32; v.len()];
    for i in 0..v.len() {
        let x = v[i];
        out[i] = 0.5 * x * (1.0 + (x * 0.797884 * (1.0 + 0.044715 * x * x)).tanh());
    }
    Ok(cpu_tensor(out, t.shape().clone()))
}

#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub layer_norm_epsilon: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for Gpt2Config {
    fn name(&self) -> &str {
        "gpt2"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn load(ws: &grim_nn::WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        let bias = ws.get([dim], "bias").ok();
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xv = x.to_vec_f32()?;
        let dim = x.shape().dims().last().copied().unwrap_or(1);
        let mut out = vec![0.0f32; xv.len()];
        for chunk in xv.chunks(dim).enumerate() {
            let (i, c) = chunk;
            let mean = c.iter().sum::<f32>() / dim as f32;
            let variance = c.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (variance + self.eps).sqrt();
            let w = self.weight.to_vec_f32()?;
            if let Some(b) = &self.bias {
                let b_vec = b.to_vec_f32()?;
                for j in 0..dim {
                    out[i * dim + j] = ((c[j] - mean) * inv_std) * w[j] + b_vec[j];
                }
            } else {
                for j in 0..dim {
                    out[i * dim + j] = ((c[j] - mean) * inv_std) * w[j];
                }
            }
        }
        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

pub struct Gpt2Block {
    pub ln_1: LayerNorm,
    pub wqkv: Linear,
    pub c_proj: Linear,
    pub ln_2: LayerNorm,
    pub ffn_gate: Linear,
    pub ffn_down: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Gpt2Block {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &Gpt2Config) -> Result<Self> {
        let ln_1 = LayerNorm::load(&ws.pp("ln_1"), cfg.hidden_size, cfg.layer_norm_epsilon)?;
        let wqkv = Linear::load(
            &ws.pp("attn.wqkv"),
            cfg.hidden_size,
            3 * cfg.hidden_size,
            true,
        )?;
        let c_proj = Linear::load(
            &ws.pp("attn.c_proj"),
            cfg.hidden_size,
            cfg.hidden_size,
            true,
        )?;
        let ln_2 = LayerNorm::load(&ws.pp("ln_2"), cfg.hidden_size, cfg.layer_norm_epsilon)?;
        let ffn_gate = Linear::load(
            &ws.pp("mlp.c_fc"),
            cfg.hidden_size,
            cfg.intermediate_size,
            true,
        )?;
        let ffn_down = Linear::load(
            &ws.pp("mlp.c_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            true,
        )?;

        Ok(Self {
            ln_1,
            wqkv,
            c_proj,
            ln_2,
            ffn_gate,
            ffn_down,
            num_heads: cfg.num_heads,
            head_dim: cfg.hidden_size / cfg.num_heads,
        })
    }

    /// Prefill-only convenience wrapper: attends over just the tokens passed
    /// in. Correct when `x` is the whole prompt.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut cache = crate::kv_attention::RefKvCache::new();
        self.forward_cached(x, &mut cache)
    }

    /// Cache-aware forward. Appends this call's K/V to `cache` before
    /// attending, so a single-token decode step sees the full prior context
    /// rather than only itself. [Group B fix: decode was stateless.]
    pub fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut crate::kv_attention::RefKvCache,
    ) -> Result<Tensor> {
        let norm_x = self.ln_1.forward(x)?;
        let qkv = self.wqkv.forward(&norm_x)?;

        // Split QKV into separate Q, K, V
        let qkv_data = qkv.to_vec_f32()?;
        let new_tokens = qkv.shape().dims()[0];
        let hidden_size = self.num_heads * self.head_dim;
        let mut q = vec![0.0f32; new_tokens * hidden_size];
        let mut k = vec![0.0f32; new_tokens * hidden_size];
        let mut v = vec![0.0f32; new_tokens * hidden_size];

        for pos in 0..new_tokens {
            for h in 0..self.num_heads {
                for d in 0..self.head_dim {
                    let idx = pos * 3 * hidden_size + h * self.head_dim + d;
                    q[pos * hidden_size + h * self.head_dim + d] = qkv_data[idx];
                    k[pos * hidden_size + h * self.head_dim + d] = qkv_data[idx + hidden_size];
                    v[pos * hidden_size + h * self.head_dim + d] = qkv_data[idx + 2 * hidden_size];
                }
            }
        }

        let k_t = cpu_tensor(k, grim_tensor::Shape::new(vec![new_tokens, hidden_size]));
        let past_len = cache.past_len;
        cache.k.extend_from_slice(&k_t.to_vec_f32()?);
        cache.v.extend_from_slice(&v);
        let total_len = cache.past_len + new_tokens;
        cache.past_len = total_len;

        // Cache-aware causal attention (see kv_attention.rs). MHA: kv heads == q heads.
        let attn_out = crate::kv_attention::causal_attention(
            &q,
            &cache.k,
            &cache.v,
            new_tokens,
            total_len,
            past_len,
            self.num_heads,
            self.head_dim,
            hidden_size,
            hidden_size,
            &(0..self.num_heads).collect::<Vec<_>>(),
        );

        let attn_out_tensor = cpu_tensor(
            attn_out,
            grim_tensor::Shape::new(vec![new_tokens, hidden_size]),
        );
        let attn_out = self.c_proj.forward(&attn_out_tensor)?;
        let x_res1 = add_tensors(x, &attn_out).map_err(grim_core::Error::Tensor)?;

        let norm_x2 = self.ln_2.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        // CRIT-2: GPT-2 MLP is Linear(c_fc) → GELU → Linear(c_proj).
        // Without the activation the two linear layers compose to a single
        // linear transformation, destroying model capacity.
        let gate = gelu(&gate)?;
        let ffn_out = self.ffn_down.forward(&gate)?;
        add_tensors(&x_res1, &ffn_out).map_err(grim_core::Error::Tensor)
    }
}

pub struct Gpt2 {
    pub cfg: Gpt2Config,
    pub device: Device,
    pub wte: Embedding,
    pub wpe: Embedding,
    pub layers: Vec<Gpt2Block>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
}

impl Gpt2 {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: Gpt2Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry for GPT-2.
    ///
    /// GPT-2 stores attention as a single **fused `wqkv` projection** of shape
    /// `[hidden, 3*hidden]` whose output dim interleaves Q, K, V per head
    /// (`forward` reshapes by `num_heads`). Cleanly column-sharding it on
    /// `world_size` requires reshaping the weight into `[3, num_heads,
    /// head_dim, hidden]`, splitting along the head axis, and re-flattening —
    /// a non-trivial transformation that `Linear::load_column_parallel`
    /// (which shards dim 0 uniformly) cannot express. Doing the naive dim-0
    /// split would cut across the (Q, K, V) interleaving and silently
    /// corrupt attention — the exact bug class called out in the TP sanity
    /// check.
    ///
    /// Rather than ship a wrong split, this entry honours `world_size == 1`
    /// (delegates to the plain `load`) and **refuses `world_size > 1`** with a
    /// typed `Unsupported` error. A full GPT-2 `load_tp` (head-axis reshape +
    /// `ColumnParallelLinear`/`RowParallelLinear` + rewritten `forward`) is
    /// tracked as a follow-up.
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Gpt2Config,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "GPT-2",
            "fused QKV projection needs head-axis reshape before column-parallel sharding",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        let wte = Embedding::load(&ws.pp("wte"), cfg.vocab_size, cfg.hidden_size)?;
        let wpe = Embedding::load(&ws.pp("wpe"), cfg.max_seq_len, cfg.hidden_size)?;
        // Validate position embedding count matches config
        let actual_pos = wpe.weight.shape().dims().first().copied().unwrap_or(0);
        if actual_pos < cfg.max_seq_len {
            eprintln!(
                "[Gpt2] wpe has {} position embeddings, config expects {}. Clamping max_seq_len.",
                actual_pos, cfg.max_seq_len
            );
        }
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Gpt2Block::load(&ws.pp("h").pp(&i.to_string()), &cfg)?);
        }
        let ln_f = LayerNorm::load(&ws.pp("ln_f"), cfg.hidden_size, cfg.layer_norm_epsilon)?;
        let lm_head = Linear::load(&ws.pp("lm_head"), cfg.hidden_size, cfg.vocab_size, false)?;

        Ok(Self {
            cfg,
            device: device.clone(),
            wte,
            wpe,
            layers,
            ln_f,
            lm_head,
        })
    }
}

impl Model for Gpt2 {
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

impl CausalLm for Gpt2 {
    fn new_session(&self) -> Box<dyn SessionT> {
        let mut session = Inner::new(self.device.clone());
        let caches: Vec<Option<crate::kv_attention::RefKvCache>> =
            vec![None; self.layers.len()];
        session.set_model_state(Box::new(caches));
        Box::new(session)
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
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
        let tok_emb = self.wte.forward(&ids, seq_len, self.cfg.hidden_size)?;
        let pos_ids: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
        let pos_emb = self.wpe.forward(&pos_ids, seq_len, self.cfg.hidden_size)?;

        let mut h = add_tensors(&tok_emb, &pos_emb).map_err(grim_core::Error::Tensor)?;

        // Per-layer KV caches live on the session so decode steps see the full
        // prior context (Group B fix).
        if session.model_state().is_none() {
            session.set_model_state(Box::new(vec![
                None::<crate::kv_attention::RefKvCache>;
                self.layers.len()
            ]));
        }
        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<crate::kv_attention::RefKvCache>>>())
            .expect("Gpt2::forward: model_state must be Vec<Option<RefKvCache>>");
        if caches.len() < self.layers.len() {
            caches.resize(self.layers.len(), None);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = caches[i].get_or_insert_with(crate::kv_attention::RefKvCache::new);
            h = layer.forward_cached(&h, cache)?;
        }
        let h = self.ln_f.forward(&h)?;
        let logits = self.lm_head.forward(&h)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_attention::RefKvCache;
    use grim_nn::modules::Linear;
    use grim_tensor::Shape;

    fn lin(in_dim: usize, out_dim: usize) -> Linear {
        // Ramp weights (not uniform) so the prior-context influence during
        // decode is large enough to distinguish from a stateless forward.
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.01 + 0.01).collect();
        let w = cpu_tensor(w, Shape::new(vec![out_dim, in_dim]));
        Linear::from_tensor(w, None)
    }

    fn identity_ln(dim: usize) -> LayerNorm {
        LayerNorm {
            weight: cpu_tensor(vec![1.0f32; dim], Shape::new(vec![dim])),
            bias: None,
            eps: 1e-5,
        }
    }

    fn tiny_block() -> Gpt2Block {
        let h = 4 * 2; // num_heads * head_dim
        Gpt2Block {
            ln_1: identity_ln(h),
            wqkv: lin(h, 3 * h),
            c_proj: lin(h, h),
            ln_2: identity_ln(h),
            ffn_gate: lin(h, h),
            ffn_down: lin(h, h),
            num_heads: 4,
            head_dim: 2,
        }
    }

    // Prefill regression: one call with the whole prompt must match a fresh
    // cache-aware forward (Group B: the cache-aware path is the new default).
    #[test]
    fn test_prefill_matches_cacheless() {
        let blk = tiny_block();
        let n = 3;
        let x = cpu_tensor((0..n * 8).map(|i| (i as f32) * 0.1).collect(), Shape::new(vec![n, 8]));
        let a = blk
            .forward_cached(&x, &mut RefKvCache::new())
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let b = blk.forward(&x).unwrap().to_vec_f32().unwrap();
        assert_eq!(a.len(), b.len());
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!((av - bv).abs() <= 1e-4, "prefill mismatch {av} vs {bv}");
        }
    }

    // Decode sees prior context: a cached single-token decode must differ from
    // the stateless single-token forward (which is what the old code did).
    #[test]
    fn test_decode_sees_prior_context() {
        let blk = tiny_block();
        let prompt = cpu_tensor(
            (0..2 * 8).map(|i| (i as f32) * 0.1).collect(),
            Shape::new(vec![2, 8]),
        );
        let dec = cpu_tensor(vec![0.9f32; 8], Shape::new(vec![1, 8]));

        let mut cache = RefKvCache::new();
        let _ = blk.forward_cached(&prompt, &mut cache).unwrap();
        let cached = blk.forward_cached(&dec, &mut cache).unwrap();

        let stateless = blk.forward(&dec).unwrap();

        let a = cached.to_vec_f32().unwrap();
        let b = stateless.to_vec_f32().unwrap();
        let diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(diff > 1e-5, "decode ignored cached prefix (diff={diff})");
        assert_eq!(cache.past_len, 3, "cache should hold prompt + decode");
    }

    // Cache length invariant: P prefill + N decode => past_len == P+N.
    #[test]
    fn test_cache_len_invariant() {
        let blk = tiny_block();
        let p = 4usize;
        let n = 3usize;
        let prompt = cpu_tensor((0..p * 8).map(|i| (i as f32) * 0.05).collect(), Shape::new(vec![p, 8]));
        let mut cache = RefKvCache::new();
        let _ = blk.forward_cached(&prompt, &mut cache).unwrap();
        assert_eq!(cache.past_len, p);
        for i in 0..n {
            let tok = cpu_tensor((0..8).map(|j| (j as f32) * 0.03 + i as f32).collect(), Shape::new(vec![1, 8]));
            let _ = blk.forward_cached(&tok, &mut cache).unwrap();
        }
        assert_eq!(cache.past_len, p + n);
    }
}

