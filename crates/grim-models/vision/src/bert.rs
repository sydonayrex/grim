//! BERT family — bidirectional encoder implementing the Encoder trait.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, Encoder, ModalityHint};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
}

impl ModelConfig for BertConfig {
    fn name(&self) -> &str {
        "bert"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct BertBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub attention_ln: RmsNorm,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub output_ln: RmsNorm,
    pub hidden_size: usize,
    pub num_heads: usize,
}

impl BertBlock {
    /// Build a randomly-initialized BertBlock. Suitable for unit tests.
    pub fn from_rng(rng: &mut grim_core::rng::SimpleRng, cfg: &BertConfig) -> Self {
        let h = cfg.hidden_size;
        let hid = cfg.intermediate_size;
        let mut mat = |out_dim: usize, in_dim: usize| -> Linear {
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.next_f32() * 0.02 - 0.01)
                .collect();
            let b: Vec<f32> = (0..out_dim).map(|_| 0.0f32).collect();
            Linear::from_tensor(
                cpu_tensor(w, Shape::new(vec![out_dim, in_dim])),
                Some(cpu_tensor(b, Shape::new(vec![out_dim]))),
            )
        };
        Self {
            wq: mat(h, h),
            wk: mat(h, h),
            wv: mat(h, h),
            wo: mat(h, h),
            attention_ln: RmsNorm {
                weight: cpu_tensor(vec![1.0; h], Shape::new(vec![h])),
                eps: 1e-12,
            },
            ffn_up: mat(hid, h),
            ffn_down: mat(h, hid),
            output_ln: RmsNorm {
                weight: cpu_tensor(vec![1.0; h], Shape::new(vec![h])),
                eps: 1e-12,
            },
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
        }
    }

    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &BertConfig) -> Result<Self> {
        let wq = Linear::load(
            &ws.pp("attention.self.query"),
            cfg.hidden_size,
            cfg.hidden_size,
            true,
        )?;
        let wk = Linear::load(
            &ws.pp("attention.self.key"),
            cfg.hidden_size,
            cfg.hidden_size,
            true,
        )?;
        let wv = Linear::load(
            &ws.pp("attention.self.value"),
            cfg.hidden_size,
            cfg.hidden_size,
            true,
        )?;
        let wo = Linear::load(
            &ws.pp("attention.output.dense"),
            cfg.hidden_size,
            cfg.hidden_size,
            true,
        )?;
        let attention_ln =
            RmsNorm::load(&ws.pp("attention.output.LayerNorm"), cfg.hidden_size, 1e-12)?;

        let ffn_up = Linear::load(
            &ws.pp("intermediate.dense"),
            cfg.hidden_size,
            cfg.intermediate_size,
            true,
        )?;
        let ffn_down = Linear::load(
            &ws.pp("output.dense"),
            cfg.intermediate_size,
            cfg.hidden_size,
            true,
        )?;
        let output_ln = RmsNorm::load(&ws.pp("output.LayerNorm"), cfg.hidden_size, 1e-12)?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            attention_ln,
            ffn_up,
            ffn_down,
            output_ln,
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let h = self.hidden_size;
        let n_heads = self.num_heads;
        let head_dim = h / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Project through Q, K, V.
        let q = self.wq.forward(x)?.to_vec_f32()?;
        let k = self.wk.forward(x)?.to_vec_f32()?;
        let v = self.wv.forward(x)?.to_vec_f32()?;

        // Multi-head self-attention.
        let mut attn_out = vec![0.0f32; seq_len * h];
        for head in 0..n_heads {
            for s in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[s * h + head * head_dim + d] * k[j * h + head * head_dim + d];
                    }
                    scores[j] = scale * dot;
                    if scores[j] > max_score {
                        max_score = scores[j];
                    }
                }
                let mut sum_exp = 0.0f32;
                for j in 0..seq_len {
                    scores[j] = (scores[j] - max_score).exp();
                    sum_exp += scores[j];
                }
                if sum_exp > 0.0 {
                    for j in 0..seq_len {
                        scores[j] /= sum_exp;
                    }
                }
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for j in 0..seq_len {
                        val += scores[j] * v[j * h + head * head_dim + d];
                    }
                    attn_out[s * h + head * head_dim + d] = val;
                }
            }
        }

        // Project attention output through W_O.
        let attn_tensor = cpu_tensor(attn_out, Shape::new(vec![seq_len, h]));
        let attn_res = self.wo.forward(&attn_tensor)?;

        // Residual + attention layer norm.
        let x_res1 = add_tensors(x, &attn_res)?;
        let norm_attn = self.attention_ln.forward(&x_res1)?;

        // FFN with GELU.
        let up = self.ffn_up.forward(&norm_attn)?;
        let gelu_up = gelu(&up)?;
        let ffn_out = self.ffn_down.forward(&gelu_up)?;
        let x_res2 = add_tensors(&norm_attn, &ffn_out)?;
        Ok(self.output_ln.forward(&x_res2)?)
    }
}

pub struct Bert {
    pub cfg: BertConfig,
    pub device: Device,
    pub word_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub token_type_embeddings: Embedding,
    pub embeddings_ln: RmsNorm,
    pub layers: Vec<BertBlock>,
}

impl Bert {
    /// Build a randomly-initialized BERT. Suitable for unit tests.
    pub fn from_rng(device: Device, cfg: BertConfig, rng: &mut grim_core::rng::SimpleRng) -> Self {
        let h = cfg.hidden_size;
        let mut emb = |vocab: usize| -> Embedding {
            let w: Vec<f32> = (0..vocab * h)
                .map(|_| rng.next_f32() * 0.02 - 0.01)
                .collect();
            Embedding {
                weight: cpu_tensor(w, Shape::new(vec![vocab, h])),
            }
        };
        let word_embeddings = emb(cfg.vocab_size);
        let position_embeddings = emb(cfg.max_seq_len);
        let token_type_embeddings = emb(2);
        let embeddings_ln = RmsNorm {
            weight: cpu_tensor(vec![1.0; h], Shape::new(vec![h])),
            eps: 1e-12,
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(BertBlock::from_rng(rng, &cfg));
        }
        Self {
            cfg,
            device,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_ln,
            layers,
        }
    }

    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: BertConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry for BERT. BERT is an encoder (`Model`, not
    /// `CausalLm`) and `BertBlock::forward` calls plain `Linear::forward` with
    /// no all-reduce hook. The serving engine's text-out path does not reach
    /// encoders, so TP here is low-leverage; refused until a `forward` rework
    /// and an actual encoder-consumer arrive.
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: BertConfig,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "BERT",
            "encoder-only BertBlock::forward calls plain Linear::forward with no \
             all-reduce hook",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        let word_embeddings = Embedding::load(
            &ws.pp("embeddings.word_embeddings"),
            cfg.vocab_size,
            cfg.hidden_size,
        )?;
        let position_embeddings = Embedding::load(
            &ws.pp("embeddings.position_embeddings"),
            cfg.max_seq_len,
            cfg.hidden_size,
        )?;
        let token_type_embeddings = Embedding::load(
            &ws.pp("embeddings.token_type_embeddings"),
            2,
            cfg.hidden_size,
        )?;
        let embeddings_ln = RmsNorm::load(&ws.pp("embeddings.LayerNorm"), cfg.hidden_size, 1e-12)?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(BertBlock::load(
                &ws.pp("encoder.layer").pp(&i.to_string()),
                &cfg,
            )?);
        }

        Ok(Self {
            cfg,
            device,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_ln,
            layers,
        })
    }
}

impl Model for Bert {
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

impl Encoder for Bert {
    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        let ids = input.to_vec_f32()?;
        // Round and validate token IDs instead of lossy truncation.
        let u_ids: Vec<u32> = ids
            .iter()
            .map(|&f| {
                let rounded = f.round();
                if rounded < 0.0 || rounded >= self.cfg.vocab_size as f32 {
                    panic!(
                        "Bert encode: token id {rounded} out of range [0, {})",
                        self.cfg.vocab_size
                    );
                }
                rounded as u32
            })
            .collect();
        let seq_len = u_ids.len();

        let w_emb = self
            .word_embeddings
            .forward(&u_ids, seq_len, self.cfg.hidden_size)?;
        let pos_ids: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
        let p_emb = self
            .position_embeddings
            .forward(&pos_ids, seq_len, self.cfg.hidden_size)?;
        let type_ids = vec![0u32; seq_len];
        let t_emb = self
            .token_type_embeddings
            .forward(&type_ids, seq_len, self.cfg.hidden_size)?;

        let mut h = add_tensors(&w_emb, &p_emb)?;
        h = add_tensors(&h, &t_emb)?;
        h = self.embeddings_ln.forward(&h)?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

impl CausalLm for Bert {
    fn new_session(&self) -> Box<dyn grim_core::session::SessionT> {
        Box::new(grim_core::session::Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids = input_ids.to_vec_f32()?;
        let u_ids: Vec<u32> = ids
            .iter()
            .map(|&f| {
                let rounded = f.round();
                if rounded < 0.0 || rounded >= self.cfg.vocab_size as f32 {
                    panic!(
                        "Bert forward: token id {rounded} out of range [0, {})",
                        self.cfg.vocab_size
                    );
                }
                rounded as u32
            })
            .collect();
        let seq_len = u_ids.len();

        let pos_vec: Vec<u32> = if positions.shape().dims().iter().product::<usize>() == seq_len {
            positions
                .to_vec_f32()?
                .iter()
                .map(|&f| f.round() as u32)
                .collect()
        } else {
            (0..seq_len).map(|i| i as u32).collect()
        };

        let w_emb = self
            .word_embeddings
            .forward(&u_ids, seq_len, self.cfg.hidden_size)?;
        let p_emb = self
            .position_embeddings
            .forward(&pos_vec, seq_len, self.cfg.hidden_size)?;
        let type_ids = vec![0u32; seq_len];
        let t_emb = self
            .token_type_embeddings
            .forward(&type_ids, seq_len, self.cfg.hidden_size)?;

        let mut h = add_tensors(&w_emb, &p_emb)?;
        h = add_tensors(&h, &t_emb)?;
        h = self.embeddings_ln.forward(&h)?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
        }

        // Apply LoRA adapters: h += (alpha / rank) * (h @ A^T) @ B^T
        // A: [rank, hidden], B: [hidden, rank], h: [seq_len, hidden]
        if !adapters.is_empty() {
            let hidden = self.cfg.hidden_size;
            let h_vec = h.to_vec_f32()?;
            for adapter in adapters {
                let rank = adapter.a.shape().dim(0).unwrap_or(1);
                let scale = adapter.alpha / rank as f32;
                let a_data = adapter.a.to_vec_f32()?;
                let b_data = adapter.b.to_vec_f32()?;
                let mut delta = vec![0.0f32; h_vec.len()];
                for s in 0..seq_len {
                    for r in 0..rank {
                        let mut temp = 0.0f32;
                        for hh in 0..hidden {
                            temp += h_vec[s * hidden + hh] * a_data[r * hidden + hh];
                        }
                        for hh in 0..hidden {
                            delta[s * hidden + hh] += scale * temp * b_data[hh * rank + r];
                        }
                    }
                }
                let delta_t = cpu_tensor(delta, h.shape().clone());
                h = add_tensors(&h, &delta_t)?;
            }
        }

        // Cache last hidden state in the session.
        session.set_last_hidden_state(h.clone());
        Ok(h)
    }
}

fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = grim_backend_cpu::CpuDevice::new();
    let (s, h) = grim_tensor::BackendDevice::add(
        &dev,
        a.storage().as_ref(),
        b.storage().as_ref(),
        a.shape(),
    )?;
    h.synchronize()?;
    Ok(Tensor::new(
        Arc::from(s),
        a.shape().clone(),
        DType::F32,
        a.provenance().clone(),
        a.device().clone(),
    ))
}

fn gelu(t: &Tensor) -> Result<Tensor> {
    let v = t.to_vec_f32()?;
    let mut out = vec![0.0f32; v.len()];
    for i in 0..v.len() {
        let x = v[i];
        out[i] = 0.5 * x * (1.0 + (x * 0.797884 * (1.0 + 0.044715 * x * x)).tanh());
    }
    Ok(cpu_tensor(out, t.shape().clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bert_cfg() -> BertConfig {
        BertConfig {
            vocab_size: 100,
            hidden_size: 16,
            num_heads: 4,
            num_layers: 2,
            intermediate_size: 32,
            max_seq_len: 64,
        }
    }

    #[test]
    fn test_bert_block_forward_shape() {
        // Verify BertBlock produces correct output shape after attention + FFN.
        let cfg = make_bert_cfg();
        let rng = &mut grim_core::rng::SimpleRng::new(0xDEADBEEF);
        let block = BertBlock::from_rng(rng, &cfg);
        let x = cpu_tensor(
            (0..4 * 16).map(|i| (i as f32) * 0.01).collect(),
            Shape::new(vec![4, 16]),
        );
        let out = block.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[4, 16]);
        let v = out.to_vec_f32().unwrap();
        assert!(v.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn test_bert_encode_runs() {
        // Smoke test: BERT encode produces (seq_len, hidden) output.
        let cfg = make_bert_cfg();
        let rng = &mut grim_core::rng::SimpleRng::new(0xC0FFEE);
        let bert = Bert::from_rng(Device::Cpu, cfg, rng);
        let input = cpu_tensor(vec![1.0, 5.0, 10.0, 2.0, 3.0], Shape::new(vec![5]));
        let out = bert.encode(&input).unwrap();
        assert_eq!(out.shape().dims(), &[5, 16]);
    }

    #[test]
    fn test_gelu_approx() {
        // Verify gelu produces finite outputs.
        let t = cpu_tensor(vec![-1.0, 0.0, 1.0, 2.0], Shape::new(vec![4]));
        let out = gelu(&t).unwrap();
        let v = out.to_vec_f32().unwrap();
        assert!(v.iter().all(|f| f.is_finite()));
        // gelu(0) should be ~0
        assert!((v[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_add_tensors_basic() {
        let a = cpu_tensor(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = cpu_tensor(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let c = add_tensors(&a, &b).unwrap();
        let v = c.to_vec_f32().unwrap();
        assert_eq!(v, vec![11.0, 22.0, 33.0]);
    }
}
