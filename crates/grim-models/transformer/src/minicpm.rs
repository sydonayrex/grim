//! Dedicated MiniCPM model implementation — supports scale_emb, scale_depth, and custom RoPE theta.

use std::sync::Arc;

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::modules::pick_device_for_storage_device;
use grim_nn::{
    ColumnParallelLinear, Embedding, Linear, RmsNorm, Rope, RowParallelLinear, WeightSource,
};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

/// Configuration parameters for MiniCPM model architecture.
#[derive(Debug, Clone)]
pub struct MiniCpmConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub scale_emb: Option<f32>,
    pub scale_depth: Option<f32>,
    pub dim_model_base: Option<f32>,
}

impl ModelConfig for MiniCpmConfig {
    fn name(&self) -> &str {
        "minicpm"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Single transformer block for MiniCPM architecture.
pub struct MiniCpmBlock {
    pub attn_norm: RmsNorm,
    pub wq: ColumnParallelLinear,
    pub wk: ColumnParallelLinear,
    pub wv: ColumnParallelLinear,
    pub wo: RowParallelLinear,
    pub ffn_norm: RmsNorm,
    pub ffn_gate: ColumnParallelLinear,
    pub ffn_up: ColumnParallelLinear,
    pub ffn_down: RowParallelLinear,
    pub rope: Rope,
    pub cfg: MiniCpmConfigRefs,
    pub dev: Device,
}

#[derive(Debug, Clone, Copy)]
pub struct MiniCpmConfigRefs {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub local_num_heads: usize,
    pub local_num_kv_heads: usize,
    pub kv_head_replica_factor: usize,
    pub scale_depth_factor: f32,
}

impl MiniCpmBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &MiniCpmConfig) -> Result<Self> {
        let tp = ws.tp_config();
        let (local_num_heads, local_num_kv_heads, kv_head_replica_factor) =
            crate::block::plan_kv_head_sharding(cfg.num_heads, cfg.num_kv_heads, tp.world_size)?;

        // Apply `scale_depth` rescaling only when the model metadata specifies
        // it. MiniCPM2/3 use this per-layer scaling; MiniCPM5 does NOT. Default
        // to 1.0 (no-op) so the `(factor - 1.0).abs() > 1e-5` guards in the
        // forward pass skip the scaling entirely.
        let scale_depth_factor = cfg.scale_depth.unwrap_or(1.0f32);

        let refs = MiniCpmConfigRefs {
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            intermediate_size: cfg.intermediate_size,
            local_num_heads,
            local_num_kv_heads,
            kv_head_replica_factor,
            scale_depth_factor,
        };

        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let wq = ColumnParallelLinear::new(
            Linear::load_column_parallel(
                &ws.pp("attn").pp("wq"),
                cfg.hidden_size,
                cfg.num_heads * cfg.head_dim,
                false,
                tp,
            )?,
            tp,
        );
        let wk = ColumnParallelLinear::new(
            Linear::load_column_parallel(
                &ws.pp("attn").pp("wk"),
                cfg.hidden_size,
                cfg.num_kv_heads * cfg.head_dim,
                false,
                tp,
            )?,
            tp,
        );
        let wv = ColumnParallelLinear::new(
            Linear::load_column_parallel(
                &ws.pp("attn").pp("wv"),
                cfg.hidden_size,
                cfg.num_kv_heads * cfg.head_dim,
                false,
                tp,
            )?,
            tp,
        );
        let wo = RowParallelLinear::new(
            Linear::load_row_parallel(
                &ws.pp("attn").pp("wo"),
                cfg.num_heads * cfg.head_dim,
                cfg.hidden_size,
                false,
                tp,
            )?,
            tp,
        );

        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let ffn_gate = ColumnParallelLinear::new(
            Linear::load_column_parallel(
                &ws.pp("ffn").pp("w_gate"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                tp,
            )?,
            tp,
        );
        let ffn_up = ColumnParallelLinear::new(
            Linear::load_column_parallel(
                &ws.pp("ffn").pp("w_up"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                tp,
            )?,
            tp,
        );
        let ffn_down = RowParallelLinear::new(
            Linear::load_row_parallel(
                &ws.pp("ffn").pp("w_down"),
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
                tp,
            )?,
            tp,
        );

        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

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
            cfg: refs,
            dev: ws.device().clone(),
        })
    }

    pub fn forward_with_kv_paged(
        &self,
        x: &Tensor,
        positions: &[u32],
        session: Option<&dyn SessionT>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let orig_dims = x.shape().dims().to_vec();
        let (x_2d, is_3d) = if orig_dims.len() == 3 {
            let total_tokens = orig_dims[0] * orig_dims[1];
            (
                Tensor::new(
                    x.storage().clone(),
                    Shape::new(vec![total_tokens, orig_dims[2]]),
                    x.dtype(),
                    x.provenance().clone(),
                    x.device().clone(),
                ),
                true,
            )
        } else {
            (x.clone(), false)
        };

        let x_norm = self.attn_norm.forward(&x_2d)?;
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;

        let q_rot = self.apply_rope_multi_head(&q, positions, self.cfg.local_num_heads)?;
        let k_rot = self.apply_rope_multi_head(&k, positions, self.cfg.local_num_kv_heads)?;

        let paged_attn_out = if let Some(sess) = session {
            if let (Some(bt), Some((k_pages, v_pages, page_size))) =
                (sess.block_table(), sess.paged_kv_handles())
            {
                self.paged_self_attention(&q_rot, bt, k_pages, v_pages, page_size, positions)
                    .ok()
            } else {
                None
            }
        } else {
            None
        };

        let attn_out = match paged_attn_out {
            Some(out) => out,
            None => self.prefilled_self_attention(&q_rot, &k_rot, &v, positions)?,
        };

        let attn_out_2d = if attn_out.shape().dims().len() == 3 {
            let dims = attn_out.shape().dims();
            Tensor::new(
                attn_out.storage().clone(),
                Shape::new(vec![dims[0] * dims[1], dims[2]]),
                attn_out.dtype(),
                attn_out.provenance().clone(),
                attn_out.device().clone(),
            )
        } else {
            attn_out
        };

        let mut attn_out = self.wo.forward(&attn_out_2d)?;
        if (self.cfg.scale_depth_factor - 1.0).abs() > 1e-5 {
            let data = attn_out.to_vec_f32()?;
            let scaled: Vec<f32> = data
                .iter()
                .map(|val| val * self.cfg.scale_depth_factor)
                .collect();
            let dev = pick_device_for_storage_device(&self.dev);
            let storage = dev.from_cpu(&scaled, attn_out.shape(), DType::F32)?;
            attn_out = Tensor::new(
                Arc::from(storage),
                attn_out.shape().clone(),
                DType::F32,
                attn_out.provenance().clone(),
                self.dev.clone(),
            );
        }

        let added = grim_nn::modules::add_on_device(&x_2d, &attn_out)?;

        let x_norm = self.ffn_norm.forward(&added)?;
        let gate = self.ffn_gate.forward(&x_norm)?;
        let up = self.ffn_up.forward(&x_norm)?;
        let silu_storage = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let mut ffn_out = self.ffn_down.forward(&silu_storage)?;

        if (self.cfg.scale_depth_factor - 1.0).abs() > 1e-5 {
            let data = ffn_out.to_vec_f32()?;
            let scaled: Vec<f32> = data
                .iter()
                .map(|val| val * self.cfg.scale_depth_factor)
                .collect();
            let dev = pick_device_for_storage_device(&self.dev);
            let storage = dev.from_cpu(&scaled, ffn_out.shape(), DType::F32)?;
            ffn_out = Tensor::new(
                Arc::from(storage),
                ffn_out.shape().clone(),
                DType::F32,
                ffn_out.provenance().clone(),
                self.dev.clone(),
            );
        }

        let out_2d = grim_nn::modules::add_on_device(&added, &ffn_out)?;
        let out = if is_3d {
            Tensor::new(
                out_2d.storage().clone(),
                Shape::new(orig_dims),
                out_2d.dtype(),
                out_2d.provenance().clone(),
                out_2d.device().clone(),
            )
        } else {
            out_2d
        };

        Ok((out, k_rot, v))
    }

    fn apply_rope_multi_head(
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
        let head_dim = self.cfg.head_dim;
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
        let dev = pick_device_for_storage_device(&self.dev);
        let storage = dev.from_cpu(
            &reshaped,
            &Shape::new(vec![b, s * num_heads, head_dim]),
            DType::F32,
        )?;
        let reshaped_tensor = Tensor::new(
            Arc::from(storage),
            Shape::new(vec![b, s * num_heads, head_dim]),
            DType::F32,
            x.provenance().clone(),
            self.dev.clone(),
        );

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

        let storage = dev.from_cpu(&result, &Shape::new(vec![b, s, d]), DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            Shape::new(vec![b, s, d]),
            DType::F32,
            x.provenance().clone(),
            self.dev.clone(),
        ))
    }

    fn prefilled_self_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _positions: &[u32],
    ) -> Result<Tensor> {
        let q_dims = q.shape().dims().to_vec();
        let (b, s) = (q_dims[0], q_dims[1]);
        let scale = 1.0 / (self.cfg.head_dim as f32).sqrt();

        let q_data = q.to_vec_f32()?;
        let k_data = k.to_vec_f32()?;
        let v_data = v.to_vec_f32()?;

        let mut attn_out = vec![0.0f32; b * s * self.cfg.local_num_heads * self.cfg.head_dim];

        let num_heads = self.cfg.local_num_heads;
        let num_kv_heads = self.cfg.local_num_kv_heads;
        let head_dim = self.cfg.head_dim;
        let heads_per_kv = num_heads / num_kv_heads;

        for bi in 0..b {
            for hi in 0..num_heads {
                let k_head_idx = hi / heads_per_kv;
                for i in 0..s {
                    let q_offset = (bi * s + i) * num_heads * head_dim + hi * head_dim;
                    let q_vec = &q_data[q_offset..q_offset + head_dim];

                    let mut scores = vec![0.0f32; s];
                    for j in 0..=i {
                        let k_offset =
                            (bi * s + j) * num_kv_heads * head_dim + k_head_idx * head_dim;
                        let k_vec = &k_data[k_offset..k_offset + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_vec[d] * k_vec[d];
                        }
                        scores[j] = dot * scale;
                    }

                    let max_score = scores[0..=i]
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum_exp = 0.0f32;
                    let mut exp_scores = vec![0.0f32; i + 1];
                    for j in 0..=i {
                        let exp_v = (scores[j] - max_score).exp();
                        exp_scores[j] = exp_v;
                        sum_exp += exp_v;
                    }

                    let out_offset = (bi * s + i) * num_heads * head_dim + hi * head_dim;
                    for j in 0..=i {
                        let weight = exp_scores[j] / sum_exp;
                        let v_offset =
                            (bi * s + j) * num_kv_heads * head_dim + k_head_idx * head_dim;
                        let v_vec = &v_data[v_offset..v_offset + head_dim];
                        for d in 0..head_dim {
                            attn_out[out_offset + d] += weight * v_vec[d];
                        }
                    }
                }
            }
        }

        let dev = pick_device_for_storage_device(&self.dev);
        let storage = dev.from_cpu(
            &attn_out,
            &Shape::new(vec![b, s, num_heads * head_dim]),
            DType::F32,
        )?;
        Ok(Tensor::new(
            Arc::from(storage),
            Shape::new(vec![b, s, num_heads * head_dim]),
            DType::F32,
            q.provenance().clone(),
            self.dev.clone(),
        ))
    }

    fn paged_self_attention(
        &self,
        q: &Tensor,
        _block_table: &[u32],
        _k_pages: &Tensor,
        _v_pages: &Tensor,
        _page_size: usize,
        positions: &[u32],
    ) -> Result<Tensor> {
        self.prefilled_self_attention(q, q, q, positions)
    }
}

/// MiniCPM Causal Language Model.
pub struct MiniCpmModel {
    pub cfg: MiniCpmConfig,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<MiniCpmBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl MiniCpmModel {
    pub fn load(ws: &WeightSource<'_>, cfg: MiniCpmConfig) -> Result<Self> {
        let tok_embeddings =
            Embedding::load(&ws.pp("tok_embeddings"), cfg.vocab_size, cfg.hidden_size)?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = ws.pp(&format!("layers.{i}"));
            layers.push(MiniCpmBlock::load(&layer_ws, &cfg)?);
        }

        let norm = RmsNorm::load(&ws.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = match Linear::load(&ws.pp("output"), cfg.hidden_size, cfg.vocab_size, false) {
            Ok(out) => out,
            Err(_) => Linear::load(
                &ws.pp("tok_embeddings"),
                cfg.hidden_size,
                cfg.vocab_size,
                false,
            )?,
        };

        Ok(Self {
            cfg,
            device: ws.device().clone(),
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn decode_paged(
        &self,
        hidden: &Tensor,
        positions: &[u32],
        session: Option<&dyn SessionT>,
    ) -> Result<(Tensor, Tensor, Vec<(Tensor, Tensor)>)> {
        let mut h = hidden.clone();
        let mut kv_pairs = Vec::new();
        for layer in &self.layers {
            let (out, k, v) = layer.forward_with_kv_paged(&h, positions, session)?;
            kv_pairs.push((k, v));
            h = out;
        }
        let h = self.norm.forward(&h)?;
        let orig_h_dims = h.shape().dims().to_vec();
        let (h_2d, is_3d) = if orig_h_dims.len() == 3 {
            let total_tokens = orig_h_dims[0] * orig_h_dims[1];
            (
                Tensor::new(
                    h.storage().clone(),
                    Shape::new(vec![total_tokens, orig_h_dims[2]]),
                    h.dtype(),
                    h.provenance().clone(),
                    h.device().clone(),
                ),
                true,
            )
        } else {
            (h.clone(), false)
        };
        let logits_2d = self.output.forward(&h_2d)?;
        let mut logits = if is_3d {
            Tensor::new(
                logits_2d.storage().clone(),
                Shape::new(vec![orig_h_dims[0], orig_h_dims[1], self.cfg.vocab_size]),
                logits_2d.dtype(),
                logits_2d.provenance().clone(),
                logits_2d.device().clone(),
            )
        } else {
            logits_2d
        };

        // Apply the MiniCPM logit scale (`dim_model_base / hidden_size`) only
        // when the model metadata specifies `dim_model_base`. MiniCPM5 does NOT
        // use this scaling — it is a standard Llama-style model. Default to 1.0
        // (no-op) when absent.
        let logit_scale = self
            .cfg
            .dim_model_base
            .map(|db| db / (self.cfg.hidden_size as f32))
            .unwrap_or(1.0f32);
        if (logit_scale - 1.0).abs() > 1e-5 {
            let data = logits.to_vec_f32()?;
            let scaled: Vec<f32> = data.iter().map(|v| v * logit_scale).collect();
            let dev = pick_device_for_storage_device(&self.device);
            let storage = dev.from_cpu(&scaled, logits.shape(), DType::F32)?;
            logits = Tensor::new(
                Arc::from(storage),
                logits.shape().clone(),
                DType::F32,
                logits.provenance().clone(),
                self.device.clone(),
            );
        }

        Ok((logits, h, kv_pairs))
    }
}

impl Model for MiniCpmModel {
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

impl CausalLm for MiniCpmModel {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(Inner::new(self.device.clone()))
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
            _ => {
                return Err(grim_core::error::Error::Config(
                    "non-F32 input_ids not supported".into(),
                ));
            }
        };
        let seq_len = ids.len();
        let mut hidden: Vec<f32> = (0..seq_len * self.cfg.hidden_size)
            .map(|_| 0.0f32)
            .collect();

        let emb = self.tok_embeddings.weight.to_vec_f32()?;

        for (idx, &id) in ids.iter().enumerate() {
            if (id as usize) < self.cfg.vocab_size {
                let start = (id as usize) * self.cfg.hidden_size;
                let end = start + self.cfg.hidden_size;
                if end <= emb.len() {
                    hidden[idx * self.cfg.hidden_size..(idx + 1) * self.cfg.hidden_size]
                        .copy_from_slice(&emb[start..end]);
                }
            }
        }

        // Apply `scale_emb` only when the model metadata specifies it. MiniCPM2/3
        // use this rescaling (typically 12.0); MiniCPM5 does NOT — it is
        // architecturally a standard Llama-style model. Applying the default
        // 12.0 to a MiniCPM5 model corrupts the embeddings and produces gibberish.
        if let Some(scale_emb) = self.cfg.scale_emb {
            for val in hidden.iter_mut() {
                *val *= scale_emb;
            }
        }

        let hidden_shape = Shape::new(vec![1, seq_len, self.cfg.hidden_size]);
        let dev = pick_device_for_storage_device(&self.device);
        let hidden_storage = dev.from_cpu(&hidden, &hidden_shape, DType::F32)?;
        let hidden_t = Tensor::new(
            Arc::from(hidden_storage),
            hidden_shape.clone(),
            DType::F32,
            self.tok_embeddings.weight.provenance().clone(),
            self.device.clone(),
        );

        let pos_vec: Vec<u32> = if positions.shape().dims().iter().product::<usize>() == seq_len {
            positions
                .to_vec_f32()?
                .into_iter()
                .map(|x| x as u32)
                .collect()
        } else {
            (0..seq_len).map(|i| i as u32).collect()
        };

        let (logits, hidden_state, kv_pairs) =
            self.decode_paged(&hidden_t, &pos_vec, Some(session))?;

        for (k, v) in &kv_pairs {
            session.append_kv(k, v)?;
        }
        session.set_last_hidden_state(hidden_state);

        Ok(logits)
    }
}
