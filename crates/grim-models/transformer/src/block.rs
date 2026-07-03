//! Transformer block: pre-norm, GQA attention, SwiGLU FFN.

use grim_backend_cpu::{cpu_tensor, CpuDevice};
use grim_core::error::Result;
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{Shape, Tensor};

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
    pub(crate) _dev: CpuDevice,
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
            _dev: CpuDevice::new(),
            _cfg: LlamaConfigRefs {
                hidden_size: cfg.hidden_size,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                intermediate_size: cfg.intermediate_size,
            },
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _dims = x.shape().dims().to_vec();
        let hidden = self._cfg.hidden_size;

        let x_2d = x;

        let x_norm = self.attn_norm.forward(x_2d)?;
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;
        let attn_out = self.prefilled_self_attention(&q, &k, &v)?;
        let attn_out = self.wo.forward(&attn_out)?;

        let x_flat = x_2d;
        let x_res1_data = x_flat.to_vec_f32()?;
        let attn_data = attn_out.to_vec_f32()?;
        let mut added = vec![0.0f32; x_res1_data.len()];
        for i in 0..x_res1_data.len() {
            added[i] = x_res1_data[i] + attn_data[i];
        }

        let x_res1 = cpu_tensor(added.clone(), Shape::new(vec![x_res1_data.len() / hidden, hidden]));

        let x_norm2 = self.ffn_norm.forward(&x_res1)?;
        let gate = self.w_gate.forward(&x_norm2)?;
        let up = self.w_up.forward(&x_norm2)?;
        let gate_data = gate.to_vec_f32()?;
        let up_data = up.to_vec_f32()?;
        let mut silu_data = vec![0.0f32; gate_data.len()];
        for i in 0..gate_data.len() {
            let xv = gate_data[i];
            silu_data[i] = (xv / (1.0 + (-xv).exp())) * up_data[i];
        }
        let ffn_out = self.w_down.forward(&cpu_tensor(silu_data, gate.shape().clone()))?;
        let ffn_data = ffn_out.to_vec_f32()?;
        let mut out = vec![0.0f32; x_res1_data.len()];
        for i in 0..x_res1_data.len() {
            out[i] = added[i] + ffn_data[i];
        }
        Ok(cpu_tensor(out, x.shape().clone()))
    }

    fn prefilled_self_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self._cfg;
        let qd = q.to_vec_f32()?;
        let kd = k.to_vec_f32()?;
        let vd = v.to_vec_f32()?;
        // q, k, v are [B, H_head] where B = elem_count / head_dim
        let num_head_dims = cfg.num_heads * cfg.head_dim;
        let total_tokens = qd.len() / num_head_dims;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let mut out = vec![0.0f32; total_tokens * num_head_dims];
        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        for h in 0..cfg.num_heads {
            let kvh = (h * cfg.num_kv_heads) / cfg.num_heads;
            for t in 0..total_tokens {
                let mut scores = vec![0.0f32; total_tokens];
                for t2 in 0..total_tokens {
                    let mut dot = 0.0f32;
                    for d in 0..cfg.head_dim {
                        dot += qd[t * num_head_dims + h * cfg.head_dim + d]
                            * kd[t2 * kv_stride + kvh * cfg.head_dim + d];
                    }
                    scores[t2] = dot * scale;
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
                    for t2 in 0..total_tokens {
                        acc += scores[t2] * vd[t2 * kv_stride + kvh * cfg.head_dim + d];
                    }
                    out[t * num_head_dims + h * cfg.head_dim + d] = acc;
                }
            }
        }
        Ok(cpu_tensor(out, Shape::new(vec![total_tokens, num_head_dims])))
    }
}
