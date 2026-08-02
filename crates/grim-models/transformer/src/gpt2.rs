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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm_x = self.ln_1.forward(x)?;
        let qkv = self.wqkv.forward(&norm_x)?;

        // Split QKV into separate Q, K, V
        let qkv_data = qkv.to_vec_f32()?;
        let seq_len = qkv.shape().dims()[0];
        let hidden_size = self.num_heads * self.head_dim;
        let mut q = vec![0.0f32; seq_len * hidden_size];
        let mut k = vec![0.0f32; seq_len * hidden_size];
        let mut v = vec![0.0f32; seq_len * hidden_size];

        for pos in 0..seq_len {
            for h in 0..self.num_heads {
                for d in 0..self.head_dim {
                    let idx = pos * 3 * hidden_size + h * self.head_dim + d;
                    q[pos * hidden_size + h * self.head_dim + d] = qkv_data[idx];
                    k[pos * hidden_size + h * self.head_dim + d] = qkv_data[idx + hidden_size];
                    v[pos * hidden_size + h * self.head_dim + d] = qkv_data[idx + 2 * hidden_size];
                }
            }
        }

        // Apply causal attention computation
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * hidden_size];

        for h in 0..self.num_heads {
            for t in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];
                // Causal masking
                for t2 in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q[t * hidden_size + h * self.head_dim + d]
                            * k[t2 * hidden_size + h * self.head_dim + d];
                    }
                    scores[t2] = dot * scale;
                }
                // Mask future positions
                for t2 in (t + 1)..seq_len {
                    scores[t2] = f32::NEG_INFINITY;
                }
                // Softmax
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                // Weighted sum of V
                for d in 0..self.head_dim {
                    let mut acc = 0.0f32;
                    for t2 in 0..=t {
                        acc += scores[t2] * v[t2 * hidden_size + h * self.head_dim + d];
                    }
                    attn_out[t * hidden_size + h * self.head_dim + d] = acc;
                }
            }
        }

        let attn_out_tensor = cpu_tensor(
            attn_out,
            grim_tensor::Shape::new(vec![seq_len, hidden_size]),
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
        Box::new(Inner::new(self.device.clone()))
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
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = self.ln_f.forward(&h)?;
        let logits = self.lm_head.forward(&h)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}
