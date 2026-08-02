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
            cfg.num_heads * 128,
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

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;

        // MLA: Multi-head Latent Attention
        // Step 1: Project to latent space
        let q_latent = self.q_a_proj.forward(&norm_x)?;
        let kv_latent = self.kv_a_proj.forward(&norm_x)?;

        // Step 2: Project from latent to Q, K, V
        let q = self.q_b_proj.forward(&q_latent)?;
        let kv = self.kv_b_proj.forward(&kv_latent)?;

        // Split kv into k and v (half each)
        let seq_len = x.shape().dims()[0];
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let hidden = num_heads * head_dim;

        let _ = q.to_vec_f32()?; // Q is used after RoPE
        let kv_data = kv.to_vec_f32()?;

        // q_b_proj outputs [seq_len, num_heads * head_dim] = Q
        // kv_b_proj outputs [seq_len, num_heads * head_dim] = K and V concatenated
        let mut k = vec![0.0f32; seq_len * hidden];
        let mut v = vec![0.0f32; seq_len * hidden];

        for pos in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let idx = pos * hidden + h * head_dim + d;
                    k[idx] = kv_data[pos * 2 * hidden + h * head_dim + d];
                    v[idx] = kv_data[pos * 2 * hidden + hidden + h * head_dim + d];
                }
            }
        }

        // Create tensors for K and V
        let k_tensor = cpu_tensor(k, grim_tensor::Shape::new(vec![seq_len, hidden]));
        let v_tensor = cpu_tensor(v, grim_tensor::Shape::new(vec![seq_len, hidden]));

        // Apply RoPE to Q and K
        let q = self.rope.forward(&q, positions)?;
        let k = self.rope.forward(&k_tensor, positions)?;

        // Causal self-attention
        let qd = q.to_vec_f32()?;
        let kd = k.to_vec_f32()?;
        let vd = v_tensor.to_vec_f32()?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * hidden];

        for h in 0..num_heads {
            for t in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];
                // Causal masking
                for t2 in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot +=
                            qd[t * hidden + h * head_dim + d] * kd[t2 * hidden + h * head_dim + d];
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
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for t2 in 0..=t {
                        acc += scores[t2] * vd[t2 * hidden + h * head_dim + d];
                    }
                    attn_out[t * hidden + h * head_dim + d] = acc;
                }
            }
        }

        let attn_out_tensor = cpu_tensor(attn_out, grim_tensor::Shape::new(vec![seq_len, hidden]));
        let attn_out = self.wo.forward(&attn_out_tensor)?;

        // Residual
        let x_res1 = add_tensors(x, &attn_out).map_err(grim_core::Error::Tensor)?;

        // FFN
        let norm_x2 = self.ffn_norm.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        let up = self.ffn_up.forward(&norm_x2)?;
        let activated = silu_mul(&gate, &up)?;
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
        for layer in &self.layers {
            h = layer.forward(&h, &pos_ids)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.output.forward(&h)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let g = gate.to_vec_f32()?;
    let u = up.to_vec_f32()?;
    let mut out = vec![0.0f32; g.len()];
    for i in 0..g.len() {
        let silu = g[i] / (1.0 + (-g[i]).exp());
        out[i] = silu * u[i];
    }
    Ok(cpu_tensor(out, gate.shape().clone()))
}
