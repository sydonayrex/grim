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

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let norm_x = self.attn_norm.forward(x)?;
        let q = self.wq.forward(&norm_x)?;
        let k = self.wk.forward(&norm_x)?;
        let v = self.wv.forward(&norm_x)?;

        // Apply RoPE
        let q = self.rope.forward(&q, positions)?;
        let k = self.rope.forward(&k, positions)?;

        // Causal self-attention
        let attn_out = self.causal_self_attention(&q, &k, &v)?;
        let attn_out = self.wo.forward(&attn_out)?;
        let x_res1 = add_tensors(x, &attn_out).map_err(grim_core::Error::Tensor)?;

        let norm_x2 = self.ffn_norm.forward(&x_res1)?;
        let gate = self.ffn_gate.forward(&norm_x2)?;
        let up = self.ffn_up.forward(&norm_x2)?;
        let activated = geglu(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&activated)?;
        add_tensors(&x_res1, &ffn_out).map_err(grim_core::Error::Tensor)
    }

    fn causal_self_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let qd = q.to_vec_f32()?;
        let kd = k.to_vec_f32()?;
        let vd = v.to_vec_f32()?;
        let num_head_dims = self.num_heads * self.head_dim;
        let total_tokens = qd.len() / num_head_dims;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut out = vec![0.0f32; total_tokens * num_head_dims];
        let kv_stride = self.num_kv_heads * self.head_dim;

        for h in 0..self.num_heads {
            let kvh = (h * self.num_kv_heads) / self.num_heads;
            for t in 0..total_tokens {
                let mut scores = vec![0.0f32; total_tokens];
                // Causal masking: only attend to current and past tokens
                for t2 in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += qd[t * num_head_dims + h * self.head_dim + d]
                            * kd[t2 * kv_stride + kvh * self.head_dim + d];
                    }
                    scores[t2] = dot * scale;
                }
                // Mask future positions
                for t2 in (t + 1)..total_tokens {
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
                        acc += scores[t2] * vd[t2 * kv_stride + kvh * self.head_dim + d];
                    }
                    out[t * num_head_dims + h * self.head_dim + d] = acc;
                }
            }
        }
        Ok({
            let dev = grim_nn::modules::pick_device_for_storage_device(&q.device());
            let storage = dev.from_cpu(
                &out,
                &grim_tensor::Shape::new(vec![total_tokens, num_head_dims]),
                grim_tensor::DType::F32,
            )?;
            Tensor::new(
                std::sync::Arc::from(storage),
                grim_tensor::Shape::new(vec![total_tokens, num_head_dims]),
                grim_tensor::DType::F32,
                grim_tensor::QuantProvenance::default(),
                q.device().clone(),
            )
        })
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
