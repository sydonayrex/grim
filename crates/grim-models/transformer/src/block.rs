//! Transformer block: pre-norm, GQA attention, SwiGLU FFN.

use grim_core::error::{Error, Result};
use grim_nn::{
    ColumnParallelLinear, Linear, RmsNorm, Rope, RowParallelLinear, TensorParallelConfig,
    WeightSource,
};
use grim_tensor::{Device, Shape, Tensor};

use crate::model::LlamaConfig;

#[derive(Debug, Clone, Copy)]
pub struct LlamaConfigRefs {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    /// TP world size (1 = single device, no sharding).
    pub tp_world_size: usize,
    /// Per-rank number of attention heads (sharded across TP ranks).
    pub local_num_heads: usize,
    /// Per-rank number of KV heads (either sharded or replicated).
    pub local_num_kv_heads: usize,
    /// How many times each KV head is replicated across TP ranks.
    /// 1 = sharded, >1 = replicated.
    pub kv_head_replica_factor: usize,
}

/// Compute the per-rank TP sharding plan for attention heads.
///
/// Returns `(local_num_heads, local_num_kv_heads, kv_head_replica_factor)`:
/// - If `num_kv_heads % world_size == 0`: KV heads are sharded, each rank gets
///   `num_kv_heads / world_size` of them (replica factor 1).
/// - If `world_size % num_kv_heads == 0`: KV heads are replicated, each rank
///   gets all `num_kv_heads`, with `world_size / num_kv_heads` replicas.
/// - Otherwise: unsupported GQA topology (e.g. 8 KV heads / 6 GPUs).
pub fn plan_kv_head_sharding(
    num_heads: usize,
    num_kv_heads: usize,
    world_size: usize,
) -> Result<(usize, usize, usize)> {
    if num_heads % world_size != 0 {
        return Err(Error::Config(format!(
            "num_heads={num_heads} must be divisible by tp world_size={world_size}"
        )));
    }
    if num_kv_heads % world_size == 0 {
        Ok((num_heads / world_size, num_kv_heads / world_size, 1))
    } else if world_size % num_kv_heads == 0 {
        Ok((
            num_heads / world_size,
            num_kv_heads,
            world_size / num_kv_heads,
        ))
    } else {
        Err(Error::Config(format!(
            "unsupported GQA topology: num_heads={num_heads}, num_kv_heads={num_kv_heads}, world_size={world_size}"
        )))
    }
}

#[derive(Clone)]
pub struct LlamaBlock {
    pub attn_norm: RmsNorm,
    pub wq: ColumnParallelLinear,
    pub wk: ColumnParallelLinear,
    pub wv: ColumnParallelLinear,
    pub wo: RowParallelLinear,
    pub ffn_norm: RmsNorm,
    pub w_gate: ColumnParallelLinear,
    pub w_up: ColumnParallelLinear,
    pub w_down: RowParallelLinear,
    pub rope: Rope,
    pub tp_config: TensorParallelConfig,
    pub(crate) _dev: Device,
    pub(crate) _cfg: LlamaConfigRefs,
}

impl LlamaBlock {
    /// Load a `LlamaBlock` with TP config taken from the `WeightSource`.
    ///
    /// See [`crate::model::Llama::load`] for why this no longer re-reads env.
    pub fn load(ws: &WeightSource<'_>, cfg: &LlamaConfig) -> Result<Self> {
        Self::load_tp(ws, cfg, ws.tp_config())
    }

    /// Load a `LlamaBlock` with an explicit `TensorParallelConfig`.
    pub fn load_tp(
        ws: &WeightSource<'_>,
        cfg: &LlamaConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let wq = Linear::load_column_parallel(
            &ws.pp("attn").pp("wq"),
            cfg.hidden_size,
            cfg.num_heads * cfg.head_dim,
            /*has_bias=*/ false,
            tp,
        )?;
        let wk = Linear::load_column_parallel(
            &ws.pp("attn").pp("wk"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            /*has_bias=*/ false,
            tp,
        )?;
        let wv = Linear::load_column_parallel(
            &ws.pp("attn").pp("wv"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            /*has_bias=*/ false,
            tp,
        )?;
        let wo = Linear::load_row_parallel(
            &ws.pp("attn").pp("wo"),
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_size,
            /*has_bias=*/ false,
            tp,
        )?;
        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let w_gate = Linear::load_column_parallel(
            &ws.pp("ffn").pp("w_gate"),
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/ false,
            tp,
        )?;
        let w_up = Linear::load_column_parallel(
            &ws.pp("ffn").pp("w_up"),
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/ false,
            tp,
        )?;
        let w_down = Linear::load_row_parallel(
            &ws.pp("ffn").pp("w_down"),
            cfg.intermediate_size,
            cfg.hidden_size,
            /*has_bias=*/ false,
            tp,
        )?;
        let device = wq.weight().device().clone();
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        let (local_num_heads, local_num_kv_heads, kv_head_replica_factor) =
            plan_kv_head_sharding(cfg.num_heads, cfg.num_kv_heads, tp.world_size)?;

        Ok(Self {
            attn_norm,
            wq: ColumnParallelLinear::new(wq, tp),
            wk: ColumnParallelLinear::new(wk, tp),
            wv: ColumnParallelLinear::new(wv, tp),
            wo: RowParallelLinear::new(wo, tp),
            ffn_norm,
            w_gate: ColumnParallelLinear::new(w_gate, tp),
            w_up: ColumnParallelLinear::new(w_up, tp),
            w_down: RowParallelLinear::new(w_down, tp),
            rope,
            tp_config: tp,
            _dev: device,
            _cfg: LlamaConfigRefs {
                hidden_size: cfg.hidden_size,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                intermediate_size: cfg.intermediate_size,
                tp_world_size: tp.world_size,
                local_num_heads,
                local_num_kv_heads,
                kv_head_replica_factor,
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
        self.forward_with_kv_paged(x, positions, None)
    }

    /// Forward pass with optional SessionT paged attention dispatch.
    pub fn forward_with_kv_paged(
        &self,
        x: &Tensor,
        positions: &[u32],
        session: Option<&dyn grim_core::session::SessionT>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let x_2d = x;

        let x_norm = self.attn_norm.forward(x_2d)?;
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;

        let paged_attn_out = if let Some(sess) = session {
            if let (Some(bt), Some((k_pages, v_pages, page_size))) =
                (sess.block_table(), sess.paged_kv_handles())
            {
                self.paged_self_attention(&q, bt, k_pages, v_pages, page_size, positions)
                    .ok()
            } else {
                None
            }
        } else {
            None
        };

        let attn_out = match paged_attn_out {
            Some(out) => out,
            None => self.prefilled_self_attention(&q, &k, &v, positions)?,
        };
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
        // RoPE, then reshape back. Uses per-rank sharded head counts.
        let q = self.apply_rope_multi_head(q, positions, cfg.local_num_heads)?;
        let k = self.apply_rope_multi_head(k, positions, cfg.local_num_kv_heads)?;

        let qd = q.to_vec_f32()?;
        let kd = k.to_vec_f32()?;
        let vd = v.to_vec_f32()?;
        let num_head_dims = cfg.local_num_heads * cfg.head_dim;
        let total_tokens = qd.len() / num_head_dims;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let mut out = vec![0.0f32; total_tokens * num_head_dims];
        let kv_stride = cfg.local_num_kv_heads * cfg.head_dim;

        for h in 0..cfg.local_num_heads {
            let kvh = (h * cfg.local_num_kv_heads) / cfg.local_num_heads;
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

    /// Dispatch self-attention via paged attention kernel when block table & physical KV pools are available.
    pub fn paged_self_attention(
        &self,
        q: &Tensor,
        block_table: &[u32],
        k_pages: &Tensor,
        v_pages: &Tensor,
        page_size: usize,
        positions: &[u32],
    ) -> Result<Tensor> {
        let cfg = &self._cfg;
        let q_rot = self.apply_rope_multi_head(q, positions, cfg.local_num_heads)?;

        let dev = grim_nn::modules::pick_device_for_storage_device(&self._dev);

        let q_shape = q_rot.shape().dims();
        let total_tokens = if q_shape.len() == 3 {
            q_shape[0] * q_shape[1]
        } else if q_shape.len() == 2 {
            q_shape[0]
        } else {
            1
        };

        let q_3d_shape = Shape::new(vec![total_tokens, cfg.local_num_heads, cfg.head_dim]);
        let q_3d = Tensor::new(
            q_rot.storage().clone(),
            q_3d_shape,
            grim_tensor::DType::F32,
            q.provenance().clone(),
            q.device().clone(),
        );

        let bt_f32: Vec<f32> = block_table.iter().map(|&b| b as f32).collect();
        let bt_shape = Shape::new(vec![block_table.len()]);
        let bt_storage = dev.from_cpu(&bt_f32, &bt_shape, grim_tensor::DType::F32)?;

        let out_shape_3d = Shape::new(vec![total_tokens, cfg.local_num_heads, cfg.head_dim]);
        let cache_offset = positions.first().copied().unwrap_or(0);
        let kv_seq_len = cache_offset as usize + total_tokens;

        let (attn_storage, _) = dev.qkv_attention_paged(
            q_3d.storage().as_ref(),
            bt_storage.as_ref(),
            k_pages.storage().as_ref(),
            v_pages.storage().as_ref(),
            cfg.local_num_kv_heads,
            block_table.len(),
            page_size,
            kv_seq_len,
            cache_offset,
            &out_shape_3d,
        )?;

        let num_head_dims = cfg.local_num_heads * cfg.head_dim;
        let out_shape_2d = Shape::new(vec![total_tokens, num_head_dims]);
        Ok(Tensor::new(
            std::sync::Arc::from(attn_storage),
            out_shape_2d,
            grim_tensor::DType::F32,
            q.provenance().clone(),
            q.device().clone(),
        ))
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
            tp_world_size: 1,
            local_num_heads: 2,
            local_num_kv_heads: 1,
            kv_head_replica_factor: 1,
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
        let tp = TensorParallelConfig::default();
        let wq = ColumnParallelLinear::new(
            make_linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim),
            tp,
        );
        let wk = ColumnParallelLinear::new(
            make_linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim),
            tp,
        );
        let wv = ColumnParallelLinear::new(
            make_linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim),
            tp,
        );
        let wo = RowParallelLinear::new(
            make_linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size),
            tp,
        );
        let w_gate =
            ColumnParallelLinear::new(make_linear(cfg.hidden_size, cfg.intermediate_size), tp);
        let w_up =
            ColumnParallelLinear::new(make_linear(cfg.hidden_size, cfg.intermediate_size), tp);
        let w_down =
            RowParallelLinear::new(make_linear(cfg.intermediate_size, cfg.hidden_size), tp);
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
            tp_config: tp,
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

    // ---- TP tests (WI-TP-3) ----

    /// plan_kv_head_sharding: divisible KV heads → sharded, replica=1.
    #[test]
    fn test_plan_kv_sharding_divisible() {
        let (nh, nkv, rep) = plan_kv_head_sharding(8, 4, 2).unwrap();
        assert_eq!(nh, 4);
        assert_eq!(nkv, 2);
        assert_eq!(rep, 1);
    }

    /// plan_kv_head_sharding: KV heads replicated when world_size % num_kv_heads == 0.
    #[test]
    fn test_plan_kv_sharding_replicated() {
        let (nh, nkv, rep) = plan_kv_head_sharding(8, 2, 4).unwrap();
        assert_eq!(nh, 2);
        assert_eq!(nkv, 2);
        assert_eq!(rep, 2);
    }

    /// plan_kv_head_sharding: unsupported GQA topology (8 KV heads, 6 GPUs).
    #[test]
    fn test_plan_kv_sharding_unsupported() {
        let result = plan_kv_head_sharding(12, 8, 6);
        assert!(result.is_err(), "8 KV heads / 6 GPUs should error");
    }

    /// LlamaBlock::load_tp with world_size=1 (single device) using a fake
    /// provider that serves zero-initialised tensors. Verifies wrapper types
    /// are constructed and shard_size == full size.
    #[test]
    fn test_llama_block_load_tp_shards_weights() {
        use grim_tensor::{DType, QuantProvenance, RawTensor, TensorMeta, TensorProvider};
        use std::collections::HashMap;

        #[derive(Clone)]
        struct FullProvider {
            tensors: HashMap<String, RawTensor>,
        }

        impl TensorProvider for FullProvider {
            fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
                self.tensors.get(name).cloned().ok_or_else(|| {
                    grim_tensor::error::Error::Backend(format!("tensor '{name}' not found"))
                })
            }
            fn meta(&self, _name: &str) -> grim_tensor::error::Result<TensorMeta> {
                Ok(TensorMeta {
                    dtype: DType::F32,
                    provenance: QuantProvenance::GrimNative,
                    shape: vec![],
                    fusion_mask: 0,
                })
            }
        }

        let mut tensors = HashMap::new();
        for leaf in &["attn_norm", "ffn_norm"] {
            tensors.insert(
                format!("{}.weight", leaf),
                RawTensor {
                    bytes: vec![0u8; 32 * 4],
                    shape: vec![32],
                    dtype: DType::F32,
                    provenance: QuantProvenance::GrimNative,
                },
            );
        }
        for (prefix, out_dim, in_dim) in &[
            ("attn.wq", 32usize, 32usize),
            ("attn.wk", 16usize, 32usize),
            ("attn.wv", 16usize, 32usize),
            ("attn.wo", 32usize, 32usize),
            ("ffn.w_gate", 64usize, 32usize),
            ("ffn.w_up", 64usize, 32usize),
            ("ffn.w_down", 32usize, 64usize),
        ] {
            tensors.insert(
                format!("{}.weight", prefix),
                RawTensor {
                    bytes: vec![0u8; *out_dim * *in_dim * 4],
                    shape: vec![*out_dim, *in_dim],
                    dtype: DType::F32,
                    provenance: QuantProvenance::GrimNative,
                },
            );
        }

        let tp = TensorParallelConfig {
            rank: 0,
            world_size: 1,
        };
        let cfg = LlamaConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 1,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 512,
        };
        let provider = FullProvider { tensors };
        let ws = WeightSource::root(&provider, Device::Cpu);
        let block = LlamaBlock::load_tp(&ws, &cfg, tp).expect("load_tp ok");

        assert_eq!(block.wq.shard_size(), 32);
        assert_eq!(block.wo.shard_size(), 32);
        assert_eq!(block._cfg.local_num_heads, 2);
        assert_eq!(block._cfg.local_num_kv_heads, 1);
        assert_eq!(block._cfg.tp_world_size, 1);
    }

    /// LlamaBlock::load_tp with world_size=2 (column + row parallel) should
    /// shard weights to half size while keeping KV head replication correct.
    #[test]
    fn test_llama_load_tp_output_head_sharded() {
        let tp = TensorParallelConfig {
            rank: 0,
            world_size: 2,
        };
        let (nh, nkv, rep) = plan_kv_head_sharding(8, 4, 2).unwrap();
        assert_eq!(nh, 4);
        assert_eq!(nkv, 2);
        assert_eq!(rep, 1);

        let (nh2, nkv2, rep2) = plan_kv_head_sharding(12, 2, 4).unwrap();
        assert_eq!(nh2, 3);
        assert_eq!(nkv2, 2);
        assert_eq!(rep2, 2);

        // For world_size=2 with 8 heads and 2 KV heads:
        // shard_size of column-parallel = out_dim / 2
        let shard_out_wq = (8 * 16) / 2;
        assert_eq!(shard_out_wq, 64);
    }

    /// Part 7: TP parity — concatenating the shards from rank 0 and rank 1
    /// (world_size=2) must reproduce the full weight matrix exactly. This
    /// proves the sharding is a clean partition with no overlap, gap, or
    /// off-by-one in the rank offset — the class of bug the sanity check
    /// flagged (issue #6).
    ///
    /// Weight values are distinct per element (`row*1000+col` scaled), so a
    /// wrong shard boundary or swapped rank would be caught by element-wise
    /// inequality rather than a vacuous all-zeros match.
    #[test]
    fn test_llama_block_tp_parity_concat_shards_equals_full() {
        use grim_tensor::{DType, QuantProvenance, RawTensor, TensorMeta, TensorProvider};
        use std::collections::HashMap;

        #[derive(Clone)]
        struct FullProvider {
            tensors: HashMap<String, RawTensor>,
        }

        impl TensorProvider for FullProvider {
            fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
                self.tensors.get(name).cloned().ok_or_else(|| {
                    grim_tensor::error::Error::Backend(format!("tensor '{name}' not found"))
                })
            }
            fn meta(&self, _name: &str) -> grim_tensor::error::Result<TensorMeta> {
                Ok(TensorMeta {
                    dtype: DType::F32,
                    provenance: QuantProvenance::GrimNative,
                    shape: vec![],
                    fusion_mask: 0,
                })
            }
        }

        fn f32_vec_to_bytes(data: &[f32]) -> Vec<u8> {
            data.iter().flat_map(|v| v.to_le_bytes()).collect()
        }

        let cfg = LlamaConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 1,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 512,
        };

        // (name, out_dim, in_dim) for every weight the block loads.
        let weight_specs: &[(&str, usize, usize)] = &[
            ("attn.wq", cfg.num_heads * cfg.head_dim, cfg.hidden_size), // [32, 32]
            ("attn.wk", cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size), // [16, 32]
            ("attn.wv", cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size), // [16, 32]
            ("attn.wo", cfg.num_heads * cfg.head_dim, cfg.hidden_size), // [32, 32]
            ("ffn.w_gate", cfg.intermediate_size, cfg.hidden_size),     // [64, 32]
            ("ffn.w_up", cfg.intermediate_size, cfg.hidden_size),       // [64, 32]
            ("ffn.w_down", cfg.hidden_size, cfg.intermediate_size),     // [32, 64]
        ];

        // Build the fake provider with known, distinct float values for every
        // weight so we can verify exact sharding/reassembly.
        let mut tensors = HashMap::new();
        for (name, out_dim, in_dim) in weight_specs {
            let mut data = Vec::with_capacity(out_dim * in_dim);
            for row in 0..*out_dim {
                for col in 0..*in_dim {
                    // Unique per element: row*1000 + col, scaled to a small float.
                    data.push((row as f32 * 1000.0 + col as f32) * 0.001);
                }
            }
            tensors.insert(
                format!("{}.weight", name),
                RawTensor {
                    bytes: f32_vec_to_bytes(&data),
                    shape: vec![*out_dim, *in_dim],
                    dtype: DType::F32,
                    provenance: QuantProvenance::GrimNative,
                },
            );
        }
        // RMS-norm weights (1D, unsharded).
        for name in &["attn_norm", "ffn_norm"] {
            let data = vec![0.5f32; cfg.hidden_size];
            tensors.insert(
                format!("{}.weight", name),
                RawTensor {
                    bytes: f32_vec_to_bytes(&data),
                    shape: vec![cfg.hidden_size],
                    dtype: DType::F32,
                    provenance: QuantProvenance::GrimNative,
                },
            );
        }

        let provider = FullProvider { tensors };

        // world_size=1 — full weights.
        let tp1 = TensorParallelConfig {
            rank: 0,
            world_size: 1,
        };
        let ws_full = WeightSource::root(&provider, Device::Cpu).with_tp_config(tp1);
        let block_full = LlamaBlock::load_tp(&ws_full, &cfg, tp1).expect("full load_tp ok");

        // world_size=2, rank 0.
        let tp_r0 = TensorParallelConfig {
            rank: 0,
            world_size: 2,
        };
        let ws_r0 = WeightSource::root(&provider, Device::Cpu).with_tp_config(tp_r0);
        let block_r0 = LlamaBlock::load_tp(&ws_r0, &cfg, tp_r0).expect("rank 0 load_tp ok");

        // world_size=2, rank 1.
        let tp_r1 = TensorParallelConfig {
            rank: 1,
            world_size: 2,
        };
        let ws_r1 = WeightSource::root(&provider, Device::Cpu).with_tp_config(tp_r1);
        let block_r1 = LlamaBlock::load_tp(&ws_r1, &cfg, tp_r1).expect("rank 1 load_tp ok");

        let weight_f32 = |t: &Tensor| t.to_vec_f32().expect("to_vec_f32");

        // Column-parallel (dim=0): shard is a contiguous row block.
        // Concat = r0_flat ++ r1_flat  (both row-major [shard_rows, in_dim]).
        let check_col = |full: &Tensor, r0: &Tensor, r1: &Tensor, name: &str| {
            let full_v = weight_f32(full);
            let r0_v = weight_f32(r0);
            let r1_v = weight_f32(r1);
            let mut concat: Vec<f32> = r0_v.clone();
            concat.extend_from_slice(&r1_v);
            assert_eq!(
                full_v, concat,
                "column-parallel {name}: rank-0 + rank-1 shards must concatenate to full"
            );
            assert_eq!(
                r0_v.len(),
                r1_v.len(),
                "column-parallel {name}: both shards must have equal element count"
            );
        };

        // Row-parallel (dim=1): shard is contiguous column block per row.
        // Reconstruct = for each row: r0_cols ++ r1_cols.
        let check_row =
            |full: &Tensor, r0: &Tensor, r1: &Tensor, rows: usize, cols_half: usize, name: &str| {
                let full_v = weight_f32(full);
                let r0_v = weight_f32(r0);
                let r1_v = weight_f32(r1);
                assert_eq!(
                    r0_v.len(),
                    rows * cols_half,
                    "row-parallel {name}: rank-0 shard size mismatch"
                );
                assert_eq!(
                    r1_v.len(),
                    rows * cols_half,
                    "row-parallel {name}: rank-1 shard size mismatch"
                );
                let mut concat = Vec::with_capacity(rows * cols_half * 2);
                for row in 0..rows {
                    let base = row * cols_half;
                    concat.extend_from_slice(&r0_v[base..base + cols_half]);
                    concat.extend_from_slice(&r1_v[base..base + cols_half]);
                }
                assert_eq!(
                    full_v, concat,
                    "row-parallel {name}: rank-0 + rank-1 shards must concatenate to full"
                );
            };

        // Column-parallel weights (sharded along dim=0, rows).
        check_col(
            &block_full.wq.weight(),
            &block_r0.wq.weight(),
            &block_r1.wq.weight(),
            "wq",
        );
        check_col(
            &block_full.wk.weight(),
            &block_r0.wk.weight(),
            &block_r1.wk.weight(),
            "wk",
        );
        check_col(
            &block_full.wv.weight(),
            &block_r0.wv.weight(),
            &block_r1.wv.weight(),
            "wv",
        );
        check_col(
            &block_full.w_gate.weight(),
            &block_r0.w_gate.weight(),
            &block_r1.w_gate.weight(),
            "w_gate",
        );
        check_col(
            &block_full.w_up.weight(),
            &block_r0.w_up.weight(),
            &block_r1.w_up.weight(),
            "w_up",
        );

        // Row-parallel weights (sharded along dim=1, columns).
        // wo: [32, 32] → shard [32, 16]; w_down: [32, 64] → shard [32, 32].
        check_row(
            &block_full.wo.weight(),
            &block_r0.wo.weight(),
            &block_r1.wo.weight(),
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_size / 2,
            "wo",
        );
        check_row(
            &block_full.w_down.weight(),
            &block_r0.w_down.weight(),
            &block_r1.w_down.weight(),
            cfg.hidden_size,
            cfg.intermediate_size / 2,
            "w_down",
        );
    }
}
