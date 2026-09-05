//! Gemma family — GeGLU activations, scale-norm normalization, and soft-capping.

use grim_backend_cpu::{add_tensors, cpu_tensor};
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm, Rope};
use grim_tensor::{ArithType, DType, Device, Tensor};

#[derive(Debug, Clone)]
pub struct GemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
}

impl ModelConfig for GemmaConfig {
    fn name(&self) -> &str {
        "gemma"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct GemmaBlock {
    pub attn_norm: RmsNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GemmaBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &GemmaConfig) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let wq = Linear::load(
            &ws.pp("wq"),
            cfg.hidden_size,
            cfg.num_heads * cfg.head_dim,
            false,
        )?;
        let wk = Linear::load(
            &ws.pp("wk"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            false,
        )?;
        let wv = Linear::load(
            &ws.pp("wv"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            false,
        )?;
        let wo = Linear::load(
            &ws.pp("wo"),
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_size,
            false,
        )?;

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

        let rope = Rope::new(cfg.head_dim, 10000.0); // Gemma typically uses 10000

        Ok(Self {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Prefill-only convenience wrapper (Group B fix): delegates to the
    /// cache-aware path with a fresh cache.
    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let mut cache = crate::kv_attention::RefKvCache::new();
        self.forward_cached(x, positions, &mut cache)
    }

    /// Cache-aware forward. Appends this call's post-RoPE K and raw V to
    /// `cache` before attending, so a single-token decode step sees the full
    /// prior context rather than only itself. [Group B fix: decode was
    /// stateless.]
    ///
    /// Legacy host-cache path kept for callers holding a `RefKvCache`;
    /// the model-level decode path uses the device-first [`GemmaBlock::forward_kv`].
    pub fn forward_cached(
        &self,
        x: &Tensor,
        positions: &[u32],
        cache: &mut crate::kv_attention::RefKvCache,
    ) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;
        let q = self.wq.forward(&norm_x)?;
        let k = self.wk.forward(&norm_x)?;
        let v = self.wv.forward(&norm_x)?;

        // Apply RoPE per head. The projections are (S, H*D); RoPE operates on
        // (B, S, D=head_dim) per head, so reshape to (1, S, H, D) and rotate
        // each head independently.
        let new_tokens = q.shape().dims()[0];
        let q_row = self.num_heads * self.head_dim;
        let kv_row = self.num_kv_heads * self.head_dim;
        let q_r = reshape_heads(&q, new_tokens, self.num_heads, self.head_dim)?;
        let k_r = reshape_heads(&k, new_tokens, self.num_kv_heads, self.head_dim)?;
        let q = apply_rope_per_head(&q_r, positions, &self.rope, self.num_heads, self.head_dim)?;
        let k = apply_rope_per_head(
            &k_r,
            positions,
            &self.rope,
            self.num_kv_heads,
            self.head_dim,
        )?;

        let k_t = cpu_tensor(
            k.to_vec_f32()?,
            grim_tensor::Shape::new(vec![new_tokens, kv_row]),
        );
        let v_vec = v.to_vec_f32()?;
        let past_len = cache.past_len;
        cache.k.extend_from_slice(&k_t.to_vec_f32()?);
        cache.v.extend_from_slice(&v_vec);
        let total_len = cache.past_len + new_tokens;
        cache.past_len = total_len;

        // GQA: map each query head to its KV head.
        let kv_head: Vec<usize> = (0..self.num_heads)
            .map(|h| (h * self.num_kv_heads) / self.num_heads)
            .collect();

        let qd = q.to_vec_f32()?;
        let attn_out = crate::kv_attention::causal_attention(
            &qd,
            &cache.k,
            &cache.v,
            new_tokens,
            total_len,
            past_len,
            self.num_heads,
            self.head_dim,
            q_row,
            kv_row,
            &kv_head,
                    None,
        );

        let attn_out_t = cpu_tensor(attn_out, grim_tensor::Shape::new(vec![new_tokens, q_row]));
        let attn_out = self.wo.forward(&attn_out_t)?;
        let x_res1 = add_tensors(x, &attn_out).map_err(grim_core::Error::Tensor)?;

        let norm_x2 = self.ffn_norm.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        let up = self.ffn_up.forward(&norm_x2)?;
        let activated = geglu(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&activated)?;
        add_tensors(&x_res1, &ffn_out).map_err(grim_core::Error::Tensor)
    }

    /// Device-first cache-aware forward: ONE `rope_2d_on_device` call per
    /// tensor (no per-head host loop), device-resident KV history via
    /// `concat_rows_on_device`, and fused `fused_attention_tensors` — no
    /// host roundtrip on GPU backends. The GeGLU activation stays host-side
    /// (gelu-tanh has no device kernel) and is re-uploaded once.
    pub fn forward_kv(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;
        let q = self.wq.forward(&norm_x)?;
        let k = self.wk.forward(&norm_x)?;
        let v = self.wv.forward(&norm_x)?;

        let new_tokens = q.shape().dims()[0];
        let q = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &q,
            self.num_heads,
            positions,
        )?;
        let k = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &k,
            self.num_kv_heads,
            positions,
        )?;

        // Device-side history: prev rows stay resident, only the new rows
        // are appended (D2D arena copy when the backend supports it).
        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let full_k = crate::shared_attention::concat_rows_on_device(prev_k, &k)?;
            let full_v = crate::shared_attention::concat_rows_on_device(prev_v, &v)?;
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k.clone(), v.clone()));
            (k.clone(), v.clone())
        };
        let kv_len = k_all.shape().dims()[0];

        // GPU-first fused attention; on backends that reject the kernel call
        // fall back to the host-history entry (scalar reference on CPU).
        let attn_tensor = match crate::shared_attention::fused_attention_tensors(
            &q,
            &k_all,
            &v_all,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            new_tokens,
            kv_len,
            None,
        ) {
            Ok(t) => t,
            Err(_) => crate::shared_attention::fused_or_scalar_attention(
                &q.to_vec_f32()?,
                &k_all.to_vec_f32()?,
                &v_all.to_vec_f32()?,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                new_tokens,
                None,
                x.device(),
            )?,
        };
        let attn_out = self.wo.forward(&attn_tensor)?;
        let x_res1 = grim_nn::modules::add_on_device(x, &attn_out)?;

        let norm_x2 = self.ffn_norm.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        let up = self.ffn_up.forward(&norm_x2)?;
        // GeGLU: gelu-tanh has no device kernel — host loop, re-uploaded once.
        let activated = grim_nn::modules::move_to_device(&geglu(&gate, &up)?, x.device())?;
        let ffn_out = self.ffn_down.forward(&activated)?;
        grim_nn::modules::add_on_device(&x_res1, &ffn_out).map_err(grim_core::Error::Tensor)
    }
}

pub struct Gemma {
    pub cfg: GemmaConfig,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<GemmaBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Gemma {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: GemmaConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry for Gemma.
    ///
    /// Gemma's attention layout (separate `wq`/`wk`/`wv`/`wo` + GQA
    /// `num_kv_heads`) is identical to Llama's, so the *sharding math* would
    /// reuse `plan_kv_head_sharding` cleanly. However, this module's `forward`
    /// and `GemmaBlock::forward` call plain `Linear::forward` directly — they
    /// do not go through `ColumnParallelLinear`/`RowParallelLinear`, so there
    /// is no all-reduce hook to sum the row-parallel `wo`/`ffn_down` partials
    /// across ranks. Shipping a load-side `load_tp` without reworking
    /// `forward` would load a sharded weight whose partial output is never
    /// reduced — silently wrong logits.
    ///
    /// Refuse `world_size > 1` with a typed `Unsupported` error until the
    /// `forward` rework lands. `world_size == 1` delegates to the plain path.
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: GemmaConfig,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "Gemma",
            "GemmaBlock::forward must be reworked to consume ColumnParallelLinear/RowParallelLinear \
             so the row-parallel partials get all-reduced",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        let tok_embeddings =
            Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(GemmaBlock::load(&ws.pp("blk").pp(&i.to_string()), &cfg)?);
        }
        let norm = RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        // Gemma uses tied embeddings: output projection uses token embedding weights transposed
        let output = Linear::from_tensor(tok_embeddings.weight.clone(), None);

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

impl Model for Gemma {
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

impl CausalLm for Gemma {
    fn new_session(&self) -> Box<dyn SessionT> {
        let mut session = Inner::new(self.device.clone());
        let caches: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
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

        // Per-layer device-resident KV caches live on the session so decode
        // steps see the full prior context (Group B fix); only the new K/V
        // rows are appended each step.
        if session.model_state().is_none() {
            session.set_model_state(Box::new(vec![
                None::<(Tensor, Tensor)>;
                self.layers.len()
            ]));
        }
        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<(Tensor, Tensor)>>>())
            .expect("Gemma::forward: model_state must be Vec<Option<(Tensor, Tensor)>>");
        if caches.len() < self.layers.len() {
            caches.resize(self.layers.len(), None);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_kv(&h, &pos_ids, &mut caches[i])?;
        }
        let h = self.norm.forward(&h)?;
        // Gemma uses tied embeddings via Linear layer
        let logits = self.output.forward(&h)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

fn geglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let g = gate.to_vec_f32()?;
    let u = up.to_vec_f32()?;
    let mut out = vec![0.0f32; g.len()];
    for i in 0..g.len() {
        // GELU approximation
        let x = g[i];
        let gelu = 0.5 * x * (1.0 + (x * 0.797884 * (1.0 + 0.044715 * x * x)).tanh());
        out[i] = gelu * u[i];
    }
    Ok(cpu_tensor(out, gate.shape().clone()))
}

/// Reshape a `(S, H*D)` projection into `(S, H, D)` for per-head RoPE.
/// The data is already in `(S, H, D)` row-major order; just relabel.
fn reshape_heads(x: &Tensor, s: usize, h: usize, d: usize) -> Result<Tensor> {
    crate::block::reshaped_view(x, &grim_tensor::Shape::new(vec![s, h, d]))
}

/// Apply `rope` to each head of a `(S, H, D)` tensor, returning `(S, H*D)`.
///
/// The shared `rope_2d_on_device` helper relabels to one `head_dim`-wide row
/// per head with positions repeated per head — mathematically identical to
/// rotating each head independently — and dispatches the device rope kernel
/// when available.
fn apply_rope_per_head(
    x: &Tensor,
    positions: &[u32],
    rope: &Rope,
    h: usize,
    d: usize,
) -> Result<Tensor> {
    let s = x.shape().dims()[0];
    let x2 = crate::block::reshaped_view(x, &grim_tensor::Shape::new(vec![s, h * d]))?;
    crate::shared_attention::rope_2d_on_device(rope, &x2, h, positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_attention::RefKvCache;
    use grim_nn::modules::{Linear, RmsNorm};
    use grim_tensor::{Shape, Tensor};

    fn lin(in_dim: usize, out_dim: usize) -> Linear {
        let w = cpu_tensor(
            vec![0.01f32; out_dim * in_dim],
            Shape::new(vec![out_dim, in_dim]),
        );
        Linear::from_tensor(w, None)
    }

    fn tiny_block() -> GemmaBlock {
        let h = 4 * 2; // num_heads * head_dim
        let kvh = 2 * 2; // num_kv_heads * head_dim
        GemmaBlock {
            attn_norm: RmsNorm::new(cpu_tensor(vec![1.0f32; h], Shape::new(vec![h])), 1e-6),
            wq: lin(h, h),
            wk: lin(h, kvh),
            wv: lin(h, kvh),
            wo: lin(h, h),
            ffn_norm: RmsNorm::new(cpu_tensor(vec![1.0f32; h], Shape::new(vec![h])), 1e-6),
            ffn_gate: lin(h, h),
            ffn_up: lin(h, h),
            ffn_down: lin(h, h),
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
            rope: Rope::new(2, 10000.0),
        }
    }

    fn t(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        cpu_tensor(data, Shape::new(shape))
    }

    // Prefill regression: whole-prompt call must match fresh-cache forward.
    #[test]
    fn test_prefill_matches_cacheless() {
        let blk = tiny_block();
        let n = 3;
        let x = t((0..n * 8).map(|i| (i as f32) * 0.1).collect(), vec![n, 8]);
        let pos: Vec<u32> = (0..n).map(|i| i as u32).collect();
        let a = blk
            .forward_cached(&x, &pos, &mut RefKvCache::new())
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let b = blk.forward(&x, &pos).unwrap().to_vec_f32().unwrap();
        assert_eq!(a.len(), b.len());
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!((av - bv).abs() <= 1e-4, "prefill mismatch {av} vs {bv}");
        }
    }

    // Decode sees prior context (GQA): cached single-token decode differs from
    // the stateless single-token forward.
    #[test]
    fn test_decode_sees_prior_context() {
        let blk = tiny_block();
        let prompt = t((0..2 * 8).map(|i| (i as f32) * 0.1).collect(), vec![2, 8]);
        let dec = t(vec![0.9f32; 8], vec![1, 8]);

        let mut cache = RefKvCache::new();
        let _ = blk.forward_cached(&prompt, &[0, 1], &mut cache).unwrap();
        let cached = blk.forward_cached(&dec, &[2], &mut cache).unwrap();
        let stateless = blk.forward(&dec, &[0]).unwrap();

        let a = cached.to_vec_f32().unwrap();
        let b = stateless.to_vec_f32().unwrap();
        let diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 1e-5, "decode ignored cached prefix (diff={diff})");
        assert_eq!(cache.past_len, 3, "cache should hold prompt + decode");
    }

    // Cache length invariant.
    #[test]
    fn test_cache_len_invariant() {
        let blk = tiny_block();
        let p = 4usize;
        let n = 3usize;
        let prompt = t((0..p * 8).map(|i| (i as f32) * 0.05).collect(), vec![p, 8]);
        let mut cache = RefKvCache::new();
        let _ = blk
            .forward_cached(&prompt, &(0..p as u32).collect::<Vec<_>>(), &mut cache)
            .unwrap();
        assert_eq!(cache.past_len, p);
        for i in 0..n {
            let tok = t(
                (0..8).map(|j| (j as f32) * 0.03 + i as f32).collect(),
                vec![1, 8],
            );
            let _ = blk
                .forward_cached(&tok, &[(p + i) as u32], &mut cache)
                .unwrap();
        }
        assert_eq!(cache.past_len, p + n);
    }
}
