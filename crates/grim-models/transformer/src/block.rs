//! Transformer block: pre-norm, GQA attention, SwiGLU FFN.

use std::sync::Arc;

use grim_core::error::Result;
use grim_nn::{Linear, RmsNorm, Rope};
use grim_tensor::{Device, Shape, Tensor};

use crate::model::LlamaConfig;

#[derive(Debug, Clone, Copy)]
pub struct LlamaConfigRefs {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
}

#[derive(Clone)]
pub struct LlamaBlock {
    pub attn_norm: RmsNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: RmsNorm,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub rope: Rope,
    pub(crate) _dev: Device,
    pub(crate) _cfg: LlamaConfigRefs,
}

impl LlamaBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &LlamaConfig) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let wq = Linear::load(
            &ws.pp("attn").pp("wq"),
            cfg.hidden_size,
            cfg.num_heads * cfg.head_dim,
            /*has_bias=*/false,
        )?;
        let wk = Linear::load(
            &ws.pp("attn").pp("wk"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            /*has_bias=*/false,
        )?;
        let wv = Linear::load(
            &ws.pp("attn").pp("wv"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            /*has_bias=*/false,
        )?;
        let wo = Linear::load(
            &ws.pp("attn").pp("wo"),
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_size,
            /*has_bias=*/false,
        )?;
        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let w_gate = Linear::load(
            &ws.pp("ffn").pp("w_gate"),
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/false,
        )?;
        let w_up = Linear::load(
            &ws.pp("ffn").pp("w_up"),
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/false,
        )?;
        let w_down = Linear::load(
            &ws.pp("ffn").pp("w_down"),
            cfg.intermediate_size,
            cfg.hidden_size,
            /*has_bias=*/false,
        )?;
        let device = wq.weight.device().clone();
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);
        Ok(Self {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
            rope,
            _dev: device,
            _cfg: LlamaConfigRefs {
                hidden_size: cfg.hidden_size,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                intermediate_size: cfg.intermediate_size,
            },
        })
    }

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let (out, _, _) = self.forward_with_kv(x, positions)?;
        Ok(out)
    }

    /// Like `forward` but also returns the K and V tensors (post-RoPE) so
    /// the caller can populate the KV cache (MAJ-1: Llama CPU path was
    /// not storing K/V, making the cache infrastructure dead code).
    pub fn forward_with_kv(&self, x: &Tensor, positions: &[u32]) -> Result<(Tensor, Tensor, Tensor)> {
        let _dims = x.shape().dims().to_vec();
        let hidden = self._cfg.hidden_size;

        let x_2d = x;

        let x_norm = self.attn_norm.forward(x_2d)?;
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;
        let attn_out = self.prefilled_self_attention(&q, &k, &v, positions)?;
        let attn_out = self.wo.forward(&attn_out)?;

        let x_res1_data = x_2d.to_vec_f32()?;
        let attn_data = attn_out.to_vec_f32()?;
        let mut added = vec![0.0f32; x_res1_data.len()];
        for i in 0..x_res1_data.len() {
            added[i] = x_res1_data[i] + attn_data[i];
        }

        // FFN: standard Llama uses a single shared expert for all tokens.
        // Process the full batch in one forward pass instead of per-token
        // CPU roundtrips (MAJ-4/MAJ-6: removed round-robin expert dispatch
        // and per-token device uploads).
        let x_norm = self.ffn_norm.forward(&x_2d)?;
        let gate = self.w_gate.forward(&x_norm)?;
        let up = self.w_up.forward(&x_norm)?;
        let gate_data = gate.to_vec_f32()?;
        let up_data = up.to_vec_f32()?;
        let mut silu_data = vec![0.0f32; gate_data.len()];
        for i in 0..gate_data.len() {
            let xv = gate_data[i];
            silu_data[i] = (xv / (1.0 + (-xv).exp())) * up_data[i];
        }
        let silu_storage = {
            let dev = grim_nn::modules::pick_device_for_storage_device(&self._dev);
            let storage = dev.from_cpu(&silu_data, &gate.shape().clone(), grim_tensor::DType::F32)?;
            Tensor::new(std::sync::Arc::from(storage), gate.shape().clone(), grim_tensor::DType::F32, grim_tensor::QuantProvenance::default(), self._dev.clone())
        };
        let ffn_out = self.w_down.forward(&silu_storage)?;
        let ffn_data = ffn_out.to_vec_f32()?;

        let mut out = vec![0.0f32; x_res1_data.len()];
        for i in 0..x_res1_data.len() {
            out[i] = added[i] + ffn_data[i];
        }
        let out = {
            let dev = grim_nn::modules::pick_device_for_storage_device(&self._dev);
            let storage = dev.from_cpu(&out, &x.shape().clone(), grim_tensor::DType::F32)?;
            Tensor::new(std::sync::Arc::from(storage), x.shape().clone(), grim_tensor::DType::F32, grim_tensor::QuantProvenance::default(), self._dev.clone())
        };
        Ok((out, k, v))
    }

    /// Apply RoPE to a multi-head tensor of shape (B, S, num_heads * head_dim)
    /// or (S, num_heads * head_dim) by reshaping to (B, S * num_heads, head_dim),
    /// repeating positions per-head, and reshaping back.
    fn apply_rope_multi_head(&self, x: &Tensor, positions: &[u32], num_heads: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, s, d) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else if dims.len() == 2 {
            (1, dims[0], dims[1])
        } else {
            return Err(grim_core::error::Error::Shape(format!("expected 2-D or 3-D tensor, got {dims:?}")));
        };
        let head_dim = self._cfg.head_dim;
        if d != num_heads * head_dim {
            return Err(grim_core::error::Error::Shape(format!(
                "expected last dim {num_heads}*{head_dim}={}, got {d}", num_heads * head_dim
            )));
        }
        // Reshape (B, S, num_heads * head_dim) -> (B, S * num_heads, head_dim)
        let data = x.to_vec_f32()?;
        let mut reshaped = vec![0.0f32; b * s * num_heads * head_dim];
        for bi in 0..b {
            for si in 0..s {
                for hi in 0..num_heads {
                    for di in 0..head_dim {
                        let src = (bi * s + si) * d + hi * head_dim + di;
                        let dst = (bi * s * num_heads + si * num_heads + hi) * head_dim + di;
                        reshaped[dst] = data[src];
                    }
                }
            }
        }
        let reshaped_tensor = {
            let dev = grim_nn::modules::pick_device_for_storage_device(&self._dev);
            let storage = dev.from_cpu(&reshaped, &Shape::new(vec![b, s * num_heads, head_dim]), grim_tensor::DType::F32)?;
            Tensor::new(std::sync::Arc::from(storage), Shape::new(vec![b, s * num_heads, head_dim]), grim_tensor::DType::F32, grim_tensor::QuantProvenance::default(), self._dev.clone())
        };
        // Repeat positions per-head. If positions is empty, default to
        // sequential positions (0..s) — callers that don't track position
        // (e.g. streaming block forward) still get valid RoPE.
        let mut ext_positions = Vec::with_capacity(s * num_heads);
        for si in 0..s {
            let pos = if si < positions.len() { positions[si] } else { si as u32 };
            for _ in 0..num_heads {
                ext_positions.push(pos);
            }
        }
        let rope_out = self.rope.forward(&reshaped_tensor, &ext_positions)?;
        // Reshape back (B, S * num_heads, head_dim) -> (B, S, num_heads * head_dim)
        let rope_data = rope_out.to_vec_f32()?;
        let mut result = vec![0.0f32; b * s * d];
        for bi in 0..b {
            for si in 0..s {
                for hi in 0..num_heads {
                    for di in 0..head_dim {
                        let src = (bi * s * num_heads + si * num_heads + hi) * head_dim + di;
                        let dst = (bi * s + si) * d + hi * head_dim + di;
                        result[dst] = rope_data[src];
                    }
                }
            }
        }
        let dev = grim_nn::modules::pick_device_for_storage_device(&self._dev);
        let storage = dev.from_cpu(&result, &x.shape().clone(), grim_tensor::DType::F32)?;
        Ok(Tensor::new(std::sync::Arc::from(storage), x.shape().clone(), grim_tensor::DType::F32, grim_tensor::QuantProvenance::default(), self._dev.clone()))
    }

    fn prefilled_self_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        positions: &[u32],
    ) -> Result<Tensor> {
        let cfg = &self._cfg;

        // Apply RoPE to Q and K. The rope.forward expects (B, S, D=head_dim)
        // but Q/K arrive as (B, S, num_heads * head_dim). Reshape to
        // (B, S * num_heads, head_dim), repeat positions per-head, apply
        // RoPE, then reshape back.
        let q = self.apply_rope_multi_head(q, positions, cfg.num_heads)?;
        let k = self.apply_rope_multi_head(k, positions, cfg.num_kv_heads)?;
        
        let qd = q.to_vec_f32()?;
        let kd = k.to_vec_f32()?;
        let vd = v.to_vec_f32()?;
        let num_head_dims = cfg.num_heads * cfg.head_dim;
        let total_tokens = qd.len() / num_head_dims;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let mut out = vec![0.0f32; total_tokens * num_head_dims];
        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        
        for h in 0..cfg.num_heads {
            let kvh = (h * cfg.num_kv_heads) / cfg.num_heads;
            for t in 0..total_tokens {
                let mut scores = vec![0.0f32; total_tokens];
                // CRIT-1: Causal masking - only attend to current and past tokens (t2 <= t)
                for t2 in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..cfg.head_dim {
                        dot += qd[t * num_head_dims + h * cfg.head_dim + d]
                            * kd[t2 * kv_stride + kvh * cfg.head_dim + d];
                    }
                    scores[t2] = dot * scale;
                }
                // Set future positions to -inf for softmax
                for t2 in (t + 1)..total_tokens {
                    scores[t2] = f32::NEG_INFINITY;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                for d in 0..cfg.head_dim {
                    let mut acc = 0.0f32;
                    // Only sum over valid (non-masked) positions
                    for t2 in 0..=t {
                        acc += scores[t2] * vd[t2 * kv_stride + kvh * cfg.head_dim + d];
                    }
                    out[t * num_head_dims + h * cfg.head_dim + d] = acc;
                }
            }
        }
        Ok({
            let dev = grim_nn::modules::pick_device_for_storage_device(&self._dev);
            let storage = dev.from_cpu(&out, &Shape::new(vec![total_tokens, num_head_dims]), grim_tensor::DType::F32)?;
            Tensor::new(std::sync::Arc::from(storage), Shape::new(vec![total_tokens, num_head_dims]), grim_tensor::DType::F32, grim_tensor::QuantProvenance::default(), self._dev.clone())
        })
    }
}
