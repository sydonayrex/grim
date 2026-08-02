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
            /*has_bias=*/ false,
        )?;
        let wk = Linear::load(
            &ws.pp("attn").pp("wk"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            /*has_bias=*/ false,
        )?;
        let wv = Linear::load(
            &ws.pp("attn").pp("wv"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            /*has_bias=*/ false,
        )?;
        let wo = Linear::load(
            &ws.pp("attn").pp("wo"),
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_size,
            /*has_bias=*/ false,
        )?;
        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let w_gate = Linear::load(
            &ws.pp("ffn").pp("w_gate"),
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/ false,
        )?;
        let w_up = Linear::load(
            &ws.pp("ffn").pp("w_up"),
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/ false,
        )?;
        let w_down = Linear::load(
            &ws.pp("ffn").pp("w_down"),
            cfg.intermediate_size,
            cfg.hidden_size,
            /*has_bias=*/ false,
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
    pub fn forward_with_kv(
        &self,
        x: &Tensor,
        positions: &[u32],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let _dims = x.shape().dims().to_vec();
        let hidden = self._cfg.hidden_size;

        let x_2d = x;

        let x_norm = self.attn_norm.forward(x_2d)?;
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;
        let attn_out = self.prefilled_self_attention(&q, &k, &v, positions)?;
        let attn_out = self.wo.forward(&attn_out)?;

        let added = grim_nn::modules::add_on_device(&x_2d, &attn_out)?;

        // FFN: standard Llama uses a single shared expert for all tokens.
        // Process the full batch in one forward pass on-device (zero CPU roundtrips).
        let x_norm = self.ffn_norm.forward(&x_2d)?;
        let gate = self.w_gate.forward(&x_norm)?;
        let up = self.w_up.forward(&x_norm)?;
        let silu_storage = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let ffn_out = self.w_down.forward(&silu_storage)?;

        let out = grim_nn::modules::add_on_device(&added, &ffn_out)?;
        Ok((out, k, v))
    }

    /// Apply RoPE to a multi-head tensor of shape (B, S, num_heads * head_dim)
    /// or (S, num_heads * head_dim) by reshaping to (B, S * num_heads, head_dim),
    /// repeating positions per-head, and reshaping back.
    pub(crate) fn apply_rope_multi_head(
        &self,
        x: &Tensor,
        positions: &[u32],
        num_heads: usize,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, s, d) = if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else if dims.len() == 2 {
            (1, dims[0], dims[1])
        } else {
            return Err(grim_core::error::Error::Shape(format!(
                "expected 2-D or 3-D tensor, got {dims:?}"
            )));
        };
        let head_dim = self._cfg.head_dim;
        if d != num_heads * head_dim {
            return Err(grim_core::error::Error::Shape(format!(
                "expected last dim {num_heads}*{head_dim}={}, got {d}",
                num_heads * head_dim
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
            let storage = dev.from_cpu(
                &reshaped,
                &Shape::new(vec![b, s * num_heads, head_dim]),
                grim_tensor::DType::F32,
            )?;
            Tensor::new(
                std::sync::Arc::from(storage),
                Shape::new(vec![b, s * num_heads, head_dim]),
                grim_tensor::DType::F32,
                grim_tensor::QuantProvenance::default(),
                self._dev.clone(),
            )
        };
        // Repeat positions per-head. If positions is empty, default to
        // sequential positions (0..s) — callers that don't track position
        // (e.g. streaming block forward) still get valid RoPE.
        let mut ext_positions = Vec::with_capacity(s * num_heads);
        for si in 0..s {
            let pos = if si < positions.len() {
                positions[si]
            } else {
                si as u32
            };
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
        Ok(Tensor::new(
            std::sync::Arc::from(storage),
            x.shape().clone(),
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::default(),
            self._dev.clone(),
        ))
    }

    pub(crate) fn prefilled_self_attention(
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
            let storage = dev.from_cpu(
                &out,
                &Shape::new(vec![total_tokens, num_head_dims]),
                grim_tensor::DType::F32,
            )?;
            Tensor::new(
                std::sync::Arc::from(storage),
                Shape::new(vec![total_tokens, num_head_dims]),
                grim_tensor::DType::F32,
                grim_tensor::QuantProvenance::default(),
                self._dev.clone(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::{DType, Device, Shape, Tensor};

    fn small_cfg() -> LlamaConfigRefs {
        LlamaConfigRefs {
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            intermediate_size: 64,
        }
    }

    fn make_linear(in_dim: usize, out_dim: usize) -> Linear {
        // Small weights to keep attention scores in a reasonable range for
        // softmax (large weights saturate softmax and make RoPE effects
        // invisible).
        let w = cpu_tensor(
            (0..out_dim * in_dim)
                .map(|i| (i as f32 * 0.001) - 0.05)
                .collect::<Vec<f32>>(),
            Shape::new(vec![out_dim, in_dim]),
        );
        Linear::from_tensor(w, None)
    }

    fn make_rmsnorm(dim: usize) -> RmsNorm {
        let w = cpu_tensor(
            (0..dim).map(|_| 1.0f32).collect::<Vec<f32>>(),
            Shape::new(vec![dim]),
        );
        RmsNorm {
            weight: w,
            eps: 1e-5,
        }
    }

    fn small_block() -> LlamaBlock {
        let cfg = small_cfg();
        let dev = Device::Cpu;
        let wq = make_linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim);
        let wk = make_linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim);
        let wv = make_linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim);
        let wo = make_linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
        let w_gate = make_linear(cfg.hidden_size, cfg.intermediate_size);
        let w_up = make_linear(cfg.hidden_size, cfg.intermediate_size);
        let w_down = make_linear(cfg.intermediate_size, cfg.hidden_size);
        let attn_norm = make_rmsnorm(cfg.hidden_size);
        let ffn_norm = make_rmsnorm(cfg.hidden_size);
        let rope = Rope::new(cfg.head_dim, 10000.0);
        LlamaBlock {
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
            _dev: dev,
            _cfg: cfg,
        }
    }

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
        let t = cpu_tensor(data, Shape::new(shape.to_vec()));
        t
    }

    /// CRIT-1: Causal mask — token at position i must not attend to positions > i.
    /// With a 3-token input, changing the 3rd token must not affect the output
    /// at position 0 or 1.
    #[test]
    fn test_causal_mask_no_future_leakage() {
        let block = small_block();
        let cfg = small_cfg();

        let x_data: Vec<f32> = (0..3 * cfg.hidden_size).map(|i| (i as f32) * 0.1).collect();
        let x = make_tensor(x_data.clone(), &[3, cfg.hidden_size]);

        let out1 = block.forward(&x, &[0, 1, 2]).unwrap();
        let out1_data = out1.to_vec_f32().unwrap();

        // Change the 3rd token — positions 0 and 1 should be unaffected
        let mut x_mod = x_data.clone();
        for i in (2 * cfg.hidden_size)..(3 * cfg.hidden_size) {
            x_mod[i] += 100.0;
        }
        let x2 = make_tensor(x_mod, &[3, cfg.hidden_size]);
        let out2 = block.forward(&x2, &[0, 1, 2]).unwrap();
        let out2_data = out2.to_vec_f32().unwrap();

        // Positions 0 and 1 must be identical (causal mask prevents future leakage)
        for i in 0..(2 * cfg.hidden_size) {
            assert!(
                (out1_data[i] - out2_data[i]).abs() < 1e-5,
                "Position {} leaked future token: {} vs {}",
                i,
                out1_data[i],
                out2_data[i]
            );
        }
    }

    /// CRIT-2: RoPE is applied — non-uniform position shifts produce
    /// different outputs for the same input embedding. Uses 3 tokens so
    /// attention depends on Q/K via relative positions (a uniform shift is
    /// invariant under RoPE, so it must not be uniform).
    #[test]
    fn test_rope_applied_in_forward() {
        let block = small_block();
        let cfg = small_cfg();

        let x_data: Vec<f32> = (0..3 * cfg.hidden_size).map(|i| (i as f32) * 0.1).collect();
        let x = make_tensor(x_data, &[3, cfg.hidden_size]);

        let out_pos0 = block.forward(&x, &[0, 1, 2]).unwrap();
        let out_pos10 = block.forward(&x, &[0, 2, 7]).unwrap();

        let v0 = out_pos0.to_vec_f32().unwrap();
        let v10 = out_pos10.to_vec_f32().unwrap();

        // Non-uniform position shift should change output via RoPE
        let diff: f32 = v0.iter().zip(v10.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "RoPE did not produce position-dependent output (diff={})",
            diff
        );
    }

    /// Direct test: Rope::forward with multi-token 3D input produces
    /// position-dependent output.
    #[test]
    fn test_rope_multi_token_position_dependent() {
        let rope = Rope::new(4, 10000.0);
        let data: Vec<f32> = (0..6 * 4).map(|i| (i as f32) * 0.1).collect();
        let x = make_tensor(data, &[1, 6, 4]);

        let y0 = rope.forward(&x, &[0, 1, 2, 3, 4, 5]).unwrap();
        let y1 = rope.forward(&x, &[10, 11, 12, 13, 14, 15]).unwrap();

        let v0 = y0.to_vec_f32().unwrap();
        let v1 = y1.to_vec_f32().unwrap();
        let diff: f32 = v0.iter().zip(v1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "RoPE multi-token diff={}", diff);
    }

    /// Direct test: apply_rope_multi_head produces position-dependent output.
    #[test]
    fn test_apply_rope_multi_head_position_dependent() {
        let block = small_block();
        let cfg = small_cfg();
        let data: Vec<f32> = (0..3 * cfg.hidden_size).map(|i| (i as f32) * 0.1).collect();
        let q = make_tensor(data, &[3, cfg.hidden_size]);

        let rope0 = block
            .apply_rope_multi_head(&q, &[0, 1, 2], cfg.num_heads)
            .unwrap();
        let rope10 = block
            .apply_rope_multi_head(&q, &[10, 11, 12], cfg.num_heads)
            .unwrap();

        let v0 = rope0.to_vec_f32().unwrap();
        let v10 = rope10.to_vec_f32().unwrap();
        let diff: f32 = v0.iter().zip(v10.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "apply_rope_multi_head diff={}", diff);
    }

    /// Debug: verify RoPE relative-encoding property. A uniform position shift
    /// must leave the attention output invariant (Q·K depends only on pos_q -
    /// pos_k), while a non-uniform shift must change it. This proves RoPE is
    /// actually applied in the block forward path.
    #[test]
    fn test_rope_relative_encoding_property() {
        let block = small_block();
        let cfg = small_cfg();
        let x_data: Vec<f32> = (0..3 * cfg.hidden_size).map(|i| (i as f32) * 0.1).collect();
        let x = make_tensor(x_data, &[3, cfg.hidden_size]);
        let x_norm = block.attn_norm.forward(&x).unwrap();
        let q = block.wq.forward(&x_norm).unwrap();
        let k = block.wk.forward(&x_norm).unwrap();
        let v = block.wv.forward(&x_norm).unwrap();

        // Q after RoPE must differ for different absolute positions
        let q0 = block
            .apply_rope_multi_head(&q, &[0, 1, 2], cfg.num_heads)
            .unwrap();
        let q10 = block
            .apply_rope_multi_head(&q, &[10, 11, 12], cfg.num_heads)
            .unwrap();
        let qd0 = q0.to_vec_f32().unwrap();
        let qd10 = q10.to_vec_f32().unwrap();
        let diff: f32 = qd0
            .iter()
            .zip(qd10.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "Q after RoPE diff={}", diff);

        // Uniform shift → attention output invariant (RoPE relative encoding)
        let out0 = block
            .prefilled_self_attention(&q, &k, &v, &[0, 1, 2])
            .unwrap();
        let out10 = block
            .prefilled_self_attention(&q, &k, &v, &[10, 11, 12])
            .unwrap();
        let od0 = out0.to_vec_f32().unwrap();
        let od10 = out10.to_vec_f32().unwrap();
        let odiff: f32 = od0
            .iter()
            .zip(od10.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            odiff < 1e-3,
            "Uniform shift should give identical attention (RoPE relative), diff={}",
            odiff
        );

        // Non-uniform shift → attention output differs
        let out_a = block
            .prefilled_self_attention(&q, &k, &v, &[0, 1, 2])
            .unwrap();
        let out_b = block
            .prefilled_self_attention(&q, &k, &v, &[0, 2, 5])
            .unwrap();
        let oa = out_a.to_vec_f32().unwrap();
        let ob = out_b.to_vec_f32().unwrap();
        let odiff2: f32 = oa.iter().zip(ob.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            odiff2 > 1e-3,
            "Non-uniform positions should change attention, diff={}",
            odiff2
        );
    }

    /// Debug: verify scores change with positions (kept as a regression guard
    /// against softmax saturation hiding RoPE effects).
    #[test]
    fn test_scores_position_dependent() {
        let block = small_block();
        let cfg = small_cfg();
        let x_data: Vec<f32> = (0..3 * cfg.hidden_size).map(|i| (i as f32) * 0.1).collect();
        let x = make_tensor(x_data, &[3, cfg.hidden_size]);
        let x_norm = block.attn_norm.forward(&x).unwrap();
        let q = block.wq.forward(&x_norm).unwrap();
        let k = block.wk.forward(&x_norm).unwrap();

        let num_head_dims = cfg.num_heads * cfg.head_dim;
        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        let compute_scores = |positions: &[u32]| -> (f32, f32) {
            let q_r = block
                .apply_rope_multi_head(&q, positions, cfg.num_heads)
                .unwrap();
            let k_r = block
                .apply_rope_multi_head(&k, positions, cfg.num_kv_heads)
                .unwrap();
            let qd = q_r.to_vec_f32().unwrap();
            let kd = k_r.to_vec_f32().unwrap();
            let h = 0;
            let kvh = 0;
            let t = 1;
            let s0 = (0..cfg.head_dim)
                .map(|d| {
                    qd[t * num_head_dims + h * cfg.head_dim + d]
                        * kd[0 * kv_stride + kvh * cfg.head_dim + d]
                })
                .sum::<f32>()
                * scale;
            let s1 = (0..cfg.head_dim)
                .map(|d| {
                    qd[t * num_head_dims + h * cfg.head_dim + d]
                        * kd[1 * kv_stride + kvh * cfg.head_dim + d]
                })
                .sum::<f32>()
                * scale;
            (s0, s1)
        };

        let (s0_a, s1_a) = compute_scores(&[0, 1, 2]);
        let (s0_b, s1_b) = compute_scores(&[0, 2, 5]);
        // s0 (q1·k0) differs because relative position differs (1 vs 2)
        assert!(
            (s0_a - s0_b).abs() > 1e-4,
            "s0 identical: a={}, b={}",
            s0_a,
            s0_b
        );
        // s1 (q1·k1) same because relative position identical (0 vs 0)
        assert!(
            (s1_a - s1_b).abs() < 1e-3,
            "s1 differs: a={}, b={}",
            s1_a,
            s1_b
        );
    }

    /// MAJ-1: forward_with_kv returns K/V tensors for KV cache population.
    #[test]
    fn test_forward_with_kv_returns_kv() {
        let block = small_block();
        let cfg = small_cfg();

        let x_data: Vec<f32> = (0..2 * cfg.hidden_size).map(|i| (i as f32) * 0.1).collect();
        let x = make_tensor(x_data, &[2, cfg.hidden_size]);

        let (out, k, v) = block.forward_with_kv(&x, &[0, 1]).unwrap();

        // K shape: [2, num_kv_heads * head_dim] = [2, 16]
        assert_eq!(k.shape().dims(), &[2, cfg.num_kv_heads * cfg.head_dim]);
        // V shape: same as K
        assert_eq!(v.shape().dims(), &[2, cfg.num_kv_heads * cfg.head_dim]);
        // Output shape matches input
        assert_eq!(out.shape().dims(), &[2, cfg.hidden_size]);
    }

    /// MAJ-3: Different positions produce different outputs (position tracking).
    /// Uses non-uniform spacing so RoPE relative encoding produces different
    /// attention scores (uniform shifts are invariant under RoPE).
    #[test]
    fn test_positions_affect_output() {
        let block = small_block();
        let cfg = small_cfg();

        let x_data: Vec<f32> = (0..3 * cfg.hidden_size).map(|i| (i as f32) * 0.1).collect();
        let x = make_tensor(x_data, &[3, cfg.hidden_size]);

        let out_0 = block.forward(&x, &[0, 1, 2]).unwrap();
        let out_5 = block.forward(&x, &[0, 2, 7]).unwrap();

        let v0 = out_0.to_vec_f32().unwrap();
        let v5 = out_5.to_vec_f32().unwrap();
        let diff: f32 = v0.iter().zip(v5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "Non-uniform positions produced identical output (diff={})",
            diff
        );
    }
}
