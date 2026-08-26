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
        session: Option<&mut dyn SessionT>,
        _cache: Option<&mut crate::block::LlamaLayerCache>,
        layer: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let orig_dims = x.shape().dims().to_vec();
        // Audit fix: these rank conversions must PHYSICALLY reshape the
        // storage — a hand-rolled Tensor::new relabel leaves the storage
        // 3-D and CPU matmul validates storage rank ("matmul expects 2-D
        // inputs" on every multi-token forward).
        let (x_2d, is_3d) = if orig_dims.len() == 3 {
            (
                crate::block::reshaped_view(
                    x,
                    &Shape::new(vec![orig_dims[0] * orig_dims[1], orig_dims[2]]),
                )?,
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
            if sess.has_paged_kv() {
                // The pages must hold POST-RoPE keys (the classic
                // LlamaLayerCache path caches k_rot and the dense attention
                // reads it directly — the pre-fix code appended the RAW k,
                // so every paged attention scored rotated queries against
                // un-rotated keys).
                // A FAILED append must skip the paged read: attending over
                // pages missing this call's K/V silently corrupts output;
                // falling back to the classic cache path is always correct.
                if sess.append_kv_layer(layer, &k_rot, &v).is_ok() {
                    if let (Some(bt), Some((k_pages, v_pages, page_size))) =
                        (sess.block_table(), sess.paged_kv_handles(layer))
                    {
                        self.paged_self_attention(
                            &q_rot,
                            bt,
                            &k_pages,
                            &v_pages,
                            page_size,
                            positions,
                        )
                        .ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
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
            crate::block::reshaped_view(
                &attn_out,
                &Shape::new(vec![dims[0] * dims[1], dims[2]]),
            )?
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
            crate::block::reshaped_view(&out_2d, &Shape::new(orig_dims))?
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
        // Accept both 3-D (B, S, H*D) and 2-D (S, H*D) producers: the block
        // converts its input to 2-D before the projections, so the classic
        // path hands this function a 2-D q. (Audit fix: the pre-fix code
        // indexed q_dims[1] as S unconditionally — garbage shapes for 2-D.)
        let (b, s) = match q_dims.len() {
            3 => (q_dims[0], q_dims[1]),
            2 => (1, q_dims[0]),
            _ => {
                return Err(grim_core::error::Error::Shape(format!(
                    "minicpm prefilled_self_attention: expected 2-D or 3-D q, got {q_dims:?}"
                )));
            }
        };
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

    /// Paged attention over the session KV pages. Audit fix (grim-models):
    /// this was a STUB that ignored the block table and pages entirely and
    /// computed `prefilled_self_attention(q, q, q)` — every engine-served
    /// MiniCPM token attended to garbage (its own query as keys AND values)
    /// instead of its history. Real implementation: gather the
    /// block-table-addressed history (post-RoPE K + raw V, appended by the
    /// caller BEFORE this call) and run offset-aware causal attention.
    #[allow(clippy::too_many_arguments)]
    fn paged_self_attention(
        &self,
        q: &Tensor,
        block_table: &[u32],
        k_pages: &Tensor,
        v_pages: &Tensor,
        page_size: usize,
        positions: &[u32],
    ) -> Result<Tensor> {
        use crate::kv_attention::causal_attention;
        let num_heads = self.cfg.local_num_heads;
        let num_kv_heads = self.cfg.local_num_kv_heads;
        let head_dim = self.cfg.head_dim;
        let q_dims = q.shape().dims().to_vec();
        let (b, s) = match q_dims.len() {
            3 => (q_dims[0], q_dims[1]),
            2 => (1, q_dims[0]),
            _ => {
                return Err(grim_core::error::Error::Shape(format!(
                    "minicpm paged_self_attention: expected 2-D or 3-D q, got {q_dims:?}"
                )));
            }
        };
        let kv_stride = num_kv_heads * head_dim;
        let cache_offset = positions.first().copied().unwrap_or(0) as usize;
        let kv_seq_len = cache_offset + s;

        let bt: Vec<usize> = block_table.iter().map(|&v| v as usize).collect();
        let k_flat = k_pages.to_vec_f32()?;
        let v_flat = v_pages.to_vec_f32()?;
        let k_hist = crate::shared_attention::gather_paged_history(
            &k_flat,
            &bt,
            page_size,
            kv_stride,
            kv_seq_len,
        )?;
        let v_hist = crate::shared_attention::gather_paged_history(
            &v_flat,
            &bt,
            page_size,
            kv_stride,
            kv_seq_len,
        )?;

        let q_data = q.to_vec_f32()?;
        let row_elems = num_heads * head_dim;
        let kv_head: Vec<usize> = (0..num_heads).map(|h| h * num_kv_heads / num_heads).collect();
        let mut out_total = Vec::with_capacity(b * s * row_elems);
        for bi in 0..b {
            let q_slice = &q_data[bi * s * row_elems..(bi + 1) * s * row_elems];
            let out = causal_attention(
                q_slice,
                &k_hist,
                &v_hist,
                s,
                kv_seq_len,
                cache_offset,
                num_heads,
                head_dim,
                row_elems,
                kv_stride,
                &kv_head,
            );
            out_total.extend_from_slice(&out);
        }

        let dev = pick_device_for_storage_device(&self.dev);
        let storage = dev.from_cpu(
            &out_total,
            &Shape::new(vec![b, s, row_elems]),
            DType::F32,
        )?;
        Ok(Tensor::new(
            Arc::from(storage),
            Shape::new(vec![b, s, row_elems]),
            DType::F32,
            q.provenance().clone(),
            self.dev.clone(),
        ))
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
        session: &mut dyn SessionT,
        _caches: Option<&mut [Option<crate::block::LlamaLayerCache>]>,
        _layer: usize,
    ) -> Result<(Tensor, Tensor, Vec<(Tensor, Tensor)>)> {
        let mut h = hidden.clone();
        let mut kv_pairs = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let (out, k, v) =
                layer.forward_with_kv_paged(&h, positions, Some(&mut *session), None, i)?;
            kv_pairs.push((k, v));
            h = out;
        }
        let h = self.norm.forward(&h)?;
        let orig_h_dims = h.shape().dims().to_vec();
        let (h_2d, is_3d) = if orig_h_dims.len() == 3 {
            (
                crate::block::reshaped_view(
                    &h,
                    &Shape::new(vec![orig_h_dims[0] * orig_h_dims[1], orig_h_dims[2]]),
                )?,
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
            // Audit fix (grim-models M13): out-of-vocabulary ids used to be
            // SILENTLY zeroed (an invisible no-op token); they now fail with
            // the offending id.
            if (id as usize) >= self.cfg.vocab_size {
                return Err(grim_core::error::Error::Config(format!(
                    "MiniCPM: token id {} out of range for vocab_size {}",
                    id as usize, self.cfg.vocab_size
                )));
            }
            let start = (id as usize) * self.cfg.hidden_size;
            let end = start + self.cfg.hidden_size;
            if end > emb.len() {
                return Err(grim_core::error::Error::Config(
                    "MiniCPM: embedding table smaller than vocab_size".into(),
                ));
            }
            hidden[idx * self.cfg.hidden_size..(idx + 1) * self.cfg.hidden_size]
                .copy_from_slice(&emb[start..end]);
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

        // Audit fix (grim-models M9): length mismatch is an error, not a
        // silent renumber from zero.
        let pos_count = positions.shape().dims().iter().product::<usize>();
        let pos_vec: Vec<u32> = if pos_count == seq_len {
            positions
                .to_vec_f32()?
                .into_iter()
                .map(|x| x as u32)
                .collect()
        } else {
            return Err(grim_core::error::Error::Shape(format!(
                "MiniCPM::forward: positions tensor has {} elements for {} input ids",
                pos_count, seq_len
            )));
        };

        let (logits, hidden_state, kv_pairs) =
            self.decode_paged(&hidden_t, &pos_vec, &mut *session, None, 0)?;

        for (k, v) in &kv_pairs {
            session.append_kv(k, v)?;
        }
        session.set_last_hidden_state(hidden_state);
        // Audit fix (grim-models): MiniCPM never advanced the session
        // position, so the engine's decode start_pos was stuck at 0 and
        // EVERY decode token ran at RoPE position 0 while the KV cache grew.
        session.advance_pos(seq_len);

        Ok(logits)
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use grim_core::session::{Inner, SessionT};
    use grim_nn::WeightSource;
    use grim_tensor::dtype::{Device as TDevice, QuantProvenance};
    use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
    use grim_tensor::Shape;

    /// In-memory provider so MiniCpmModel::load can run without a GGUF
    /// (mirrors moe_block's FullProvider pattern).
    #[derive(Clone)]
    struct FullProvider {
        tensors: std::collections::HashMap<String, RawTensor>,
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

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn tiny_model() -> MiniCpmModel {
        let (vocab, hidden, inter) = (64usize, 16usize, 16usize);
        let (heads, kv_heads, head_dim) = (2usize, 1usize, 8usize);
        let mut t = std::collections::HashMap::new();
        let mut put = |name: String, shape: Vec<usize>, data: Vec<f32>| {
            t.insert(name, RawTensor { bytes: f32_bytes(&data), shape, dtype: DType::F32, provenance: QuantProvenance::GrimNative });
        };
        let rnd: Vec<f32> = (0..vocab * hidden).map(|i| ((i % 37) as f32 * 0.02) - 0.35).collect();
        put("tok_embeddings.weight".into(), vec![vocab, hidden], rnd);
        let layer_names = [
            ("attn_norm.weight".to_string(), vec![hidden], vec![1.0f32; hidden]),
            ("ffn_norm.weight".to_string(), vec![hidden], vec![1.0f32; hidden]),
            ("attn.wq.weight".to_string(), vec![heads * head_dim, hidden],
                (0..heads * head_dim * hidden).map(|i| ((i % 11) as f32 * 0.01) - 0.05).collect()),
            ("attn.wk.weight".to_string(), vec![kv_heads * head_dim, hidden],
                (0..kv_heads * head_dim * hidden).map(|i| ((i % 7) as f32 * 0.01) - 0.03).collect()),
            ("attn.wv.weight".to_string(), vec![kv_heads * head_dim, hidden],
                (0..kv_heads * head_dim * hidden).map(|i| ((i % 13) as f32 * 0.01) - 0.06).collect()),
            ("attn.wo.weight".to_string(), vec![hidden, heads * head_dim],
                (0..hidden * heads * head_dim).map(|i| ((i % 17) as f32 * 0.01) - 0.08).collect()),
            ("ffn.w_gate.weight".to_string(), vec![inter, hidden],
                (0..inter * hidden).map(|i| ((i % 5) as f32 * 0.01) - 0.02).collect()),
            ("ffn.w_up.weight".to_string(), vec![inter, hidden],
                (0..inter * hidden).map(|i| ((i % 9) as f32 * 0.01) - 0.04).collect()),
            ("ffn.w_down.weight".to_string(), vec![hidden, inter],
                (0..hidden * inter).map(|i| ((i % 15) as f32 * 0.01) - 0.07).collect()),
        ];
        for (name, shape, data) in layer_names {
            put(format!("layers.0.{name}"), shape, data);
        }
        // norm.weight is legitimately ALL ONES (audit: must not be rejected).
        put("norm.weight".into(), vec![hidden], vec![1.0f32; hidden]);
        let out_w: Vec<f32> = (0..vocab * hidden).map(|i| ((i % 23) as f32 * 0.01) - 0.1).collect();
        put("output.weight".into(), vec![vocab, hidden], out_w);

        let provider = FullProvider { tensors: t };
        let ws = WeightSource::root(&provider, TDevice::Cpu);
        let cfg = MiniCpmConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            num_layers: 1,
            intermediate_size: inter,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            scale_emb: None,
            scale_depth: None,
            dim_model_base: None,
        };
        MiniCpmModel::load(&ws, cfg).expect("tiny minicpm load")
    }

    fn cpu_ids(ids: &[u32]) -> grim_tensor::Tensor {
        grim_backend_cpu::cpu_tensor(
            ids.iter().map(|&t| t as f32).collect::<Vec<f32>>(),
            Shape::new(vec![ids.len()]),
        )
    }

    /// Audit gate: MiniCPM must advance the session position — pre-fix it
    /// never did, so every engine decode token ran at RoPE position 0.
    #[test]
    fn minicpm_forward_advances_session_position() {
        let model = tiny_model();
        let mut sess = Inner::new(model.device.clone());
        let adapters: [grim_core::model::AdapterHandle; 0] = [];
        grim_core::CausalLm::forward(&model, &mut sess, &cpu_ids(&[3, 9]), &cpu_ids(&[0, 1]), &adapters)
            .expect("forward");
        assert_eq!(sess.current_pos(), 2, "session pos must advance by seq_len");
    }

    /// Audit gate (M13): out-of-vocabulary token ids must be a LOUD error,
    /// not silently-zeroed no-op tokens.
    #[test]
    fn minicpm_forward_rejects_out_of_vocab_ids() {
        let model = tiny_model();
        let mut sess = Inner::new(model.device.clone());
        let adapters: [grim_core::model::AdapterHandle; 0] = [];
        let res = grim_core::CausalLm::forward(
            &model,
            &mut sess,
            &cpu_ids(&[9999]), // vocab_size is 64 in the fixture
            &cpu_ids(&[0]),
            &adapters,
        );
        let err = res.expect_err("OOV id must error");
        assert!(
            err.to_string().contains("out of range"),
            "error should name the OOV token: {err}"
        );
    }

    /// Audit gate: the paged KV path must store POST-RoPE keys — paged and
    /// classic attention over identical input must agree. Pre-fix the paged
    /// append stored RAW keys while the query ran rotated, so the two paths
    /// diverged on any input with nonzero positions.
    #[test]
    fn minicpm_paged_matches_classic_attention() {
        let model = tiny_model();
        let ids = [3u32, 9u32];
        let pos = [0u32, 1];
        let adapters: [grim_core::model::AdapterHandle; 0] = [];

        // Classic path (per-layer cache in model_state).
        let mut classic = Inner::new(model.device.clone());
        let logits_classic =
            grim_core::CausalLm::forward(&model, &mut classic, &cpu_ids(&ids), &cpu_ids(&pos), &adapters)
                .expect("classic forward");

        // Paged path (PagedKvCache-backed session).
        let pool = std::sync::Arc::new(std::sync::Mutex::new(
            grim_memory::KvBlockPool::new(64, 1, 8),
        ));
        let mut kv = grim_memory::PagedKvCache::new(pool, 1, 8, grim_memory::BLOCK_SIZE);
        let backend = grim_nn::pick_device_for_storage_device(&model.device);
        kv.set_device(model.device.clone(), backend);
        let mut paged = Inner::with_kv(model.device.clone(), Box::new(kv));
        let logits_paged =
            grim_core::CausalLm::forward(&model, &mut paged, &cpu_ids(&ids), &cpu_ids(&pos), &adapters)
                .expect("paged forward");

        let a = logits_classic.to_vec_f32().unwrap();
        let b = logits_paged.to_vec_f32().unwrap();
        assert_eq!(a.len(), b.len());
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "paged path must match classic (post-RoPE key storage): max diff {max_diff}"
        );
    }
}
