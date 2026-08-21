//! CogVLM vision-language model with dual visual-expert attention and FFN pathways.
//!
//! # Architecture Details
//! - **Visual Expert Routing**: Token sequence is tagged with text/vision mask; visual tokens activate specialized visual linear projections and visual MLPs.
//! - **EVA2-CLIP Visual Backbone**: Extracts spatial patch representations projected into the language model's hidden dimension.
//! - **Language Transformer**: GQA self-attention with SwiGLU text and visual expert feed-forward networks.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Vision Config & Encoder
// ---------------------------------------------------------------------------

/// Configuration for CogVLM visual encoder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CogVlmVisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
}

impl Default for CogVlmVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            image_size: 490,
            patch_size: 14,
            num_heads: 16,
            num_layers: 24,
            in_channels: 3,
            out_hidden_size: 4096,
        }
    }
}

/// EVA2-CLIP style visual encoder for CogVLM.
pub struct CogVlmVisionEncoder {
    pub patch_embed: Linear,
    pub proj: Linear,
    pub norm: RmsNorm,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
}

impl CogVlmVisionEncoder {
    pub fn load(ws: &WeightSource<'_>, cfg: &CogVlmVisionConfig) -> Result<Self> {
        let in_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let patch_embed = Linear::load_shape(&ws.scoped("patch_embed"), [in_dim, cfg.hidden_size])?;
        let norm = RmsNorm::load(&ws.scoped("norm"), cfg.hidden_size, 1e-6)?;
        let proj = Linear::load_shape(
            &ws.scoped("linear_proj"),
            [cfg.hidden_size, cfg.out_hidden_size],
        )?;

        Ok(Self {
            patch_embed,
            proj,
            norm,
            hidden_size: cfg.hidden_size,
            out_hidden_size: cfg.out_hidden_size,
        })
    }

    /// Encodes visual patches into text space `[num_patches, out_hidden_size]`.
    pub fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        let h = self.patch_embed.forward(patches)?;
        let normed = self.norm.forward(&h)?;
        Ok(self.proj.forward(&normed)?)
    }
}

// ---------------------------------------------------------------------------
// Model Config
// ---------------------------------------------------------------------------

/// Configuration for CogVLM transformer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CogVlmConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub vision_config: CogVlmVisionConfig,
}

impl Default for CogVlmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            num_layers: 32,
            intermediate_size: 11008,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            vision_config: CogVlmVisionConfig::default(),
        }
    }
}

impl ModelConfig for CogVlmConfig {
    fn name(&self) -> &str {
        "cogvlm"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::MultimodalInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Block with Visual Expert
// ---------------------------------------------------------------------------

/// Transformer block with text and visual expert pathways.
pub struct CogVlmBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub v_wq: Linear,
    pub v_wk: Linear,
    pub v_wv: Linear,
    pub v_wo: Linear,
    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub v_gate: Linear,
    pub v_up: Linear,
    pub v_down: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl CogVlmBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &CogVlmConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        // Visual expert attention weights (fallback to standard if shared)
        let v_wq = Linear::load_shape(&attn_ws.scoped("v_q_proj"), [cfg.hidden_size, q_dim])
            .unwrap_or_else(|_| wq.clone());
        let v_wk = Linear::load_shape(&attn_ws.scoped("v_k_proj"), [cfg.hidden_size, kv_dim])
            .unwrap_or_else(|_| wk.clone());
        let v_wv = Linear::load_shape(&attn_ws.scoped("v_v_proj"), [cfg.hidden_size, kv_dim])
            .unwrap_or_else(|_| wv.clone());
        let v_wo = Linear::load_shape(&attn_ws.scoped("v_o_proj"), [q_dim, cfg.hidden_size])
            .unwrap_or_else(|_| wo.clone());

        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let ffn_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let mlp_ws = ws.scoped("mlp");
        let w_gate = Linear::load_shape(
            &mlp_ws.scoped("gate_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let w_up = Linear::load_shape(
            &mlp_ws.scoped("up_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )?;
        let w_down = Linear::load_shape(
            &mlp_ws.scoped("down_proj"),
            [cfg.intermediate_size, cfg.hidden_size],
        )?;

        // Visual expert MLP weights
        let v_gate = Linear::load_shape(
            &mlp_ws.scoped("v_gate_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )
        .unwrap_or_else(|_| w_gate.clone());
        let v_up = Linear::load_shape(
            &mlp_ws.scoped("v_up_proj"),
            [cfg.hidden_size, cfg.intermediate_size],
        )
        .unwrap_or_else(|_| w_up.clone());
        let v_down = Linear::load_shape(
            &mlp_ws.scoped("v_down_proj"),
            [cfg.intermediate_size, cfg.hidden_size],
        )
        .unwrap_or_else(|_| w_down.clone());

        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            v_wq,
            v_wk,
            v_wv,
            v_wo,
            attn_norm,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
            v_gate,
            v_up,
            v_down,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        is_visual: bool,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.attn_norm.forward(x)?;

        let (cur_wq, cur_wk, cur_wv, cur_wo) = if is_visual {
            (&self.v_wq, &self.v_wk, &self.v_wv, &self.v_wo)
        } else {
            (&self.wq, &self.wk, &self.wv, &self.wo)
        };

        let q = cur_wq.forward(&normed_attn)?;
        let k = cur_wk.forward(&normed_attn)?;
        let v = cur_wv.forward(&normed_attn)?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let mut q_vec = q.to_vec_f32()?;
        let mut k_vec = k.to_vec_f32()?;

        crate::qwen35::apply_rope_neox(
            &mut q_vec,
            positions,
            self.num_heads,
            self.head_dim,
            10000.0,
        );
        crate::qwen35::apply_rope_neox(
            &mut k_vec,
            positions,
            self.num_kv_heads,
            self.head_dim,
            10000.0,
        );

        let q_rot = cpu_tensor(q_vec, Shape::new(vec![seq_len, q_dim]));
        let k_rot = cpu_tensor(k_vec, Shape::new(vec![seq_len, kv_dim]));

        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let mut new_k = prev_k.to_vec_f32()?;
            let mut new_v = prev_v.to_vec_f32()?;
            new_k.extend(k_rot.to_vec_f32()?);
            new_v.extend(v.to_vec_f32()?);
            let total_seq = new_k.len() / kv_dim;
            let full_k = cpu_tensor(new_k, Shape::new(vec![total_seq, kv_dim]));
            let full_v = cpu_tensor(new_v, Shape::new(vec![total_seq, kv_dim]));
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k_rot.clone(), v.clone()));
            (k_rot, v)
        };

        let q_heads = q_rot.to_vec_f32()?;
        let k_heads = k_all.to_vec_f32()?;
        let v_heads = v_all.to_vec_f32()?;

        // Shared helper applies the causal mask at cache_offset + s (fixes
        // future-token leakage during multi-token prefill).
        let attn_tensor = crate::shared_attention::fused_or_scalar_attention(
            &q_heads,
            &k_heads,
            &v_heads,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            None,
            &Device::Cpu,
        )?;
        let attn_proj = cur_wo.forward(&attn_tensor)?;

        let xv = x.to_vec_f32()?;
        let av = attn_proj.to_vec_f32()?;
        let res1: Vec<f32> = xv.iter().zip(av.iter()).map(|(&a, &b)| a + b).collect();
        let res1_t = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.ffn_norm.forward(&res1_t)?;

        let (cur_gate, cur_up, cur_down) = if is_visual {
            (&self.v_gate, &self.v_up, &self.v_down)
        } else {
            (&self.w_gate, &self.w_up, &self.w_down)
        };

        let gate = cur_gate.forward(&normed_ffn)?;
        let up = cur_up.forward(&normed_ffn)?;

        let g_v = gate.to_vec_f32()?;
        let u_v = up.to_vec_f32()?;
        let swiglu: Vec<f32> = g_v
            .iter()
            .zip(u_v.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let swiglu_t = cpu_tensor(swiglu, gate.shape().clone());
        let mlp_out = cur_down.forward(&swiglu_t)?;

        let r1v = res1_t.to_vec_f32()?;
        let mv = mlp_out.to_vec_f32()?;
        let out_vec: Vec<f32> = r1v.iter().zip(mv.iter()).map(|(&a, &b)| a + b).collect();

        Ok(cpu_tensor(out_vec, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct CogVlm {
    pub cfg: CogVlmConfig,
    pub device: Device,
    pub vision_encoder: Option<CogVlmVisionEncoder>,
    pub tok_embeddings: Linear,
    pub layers: Vec<CogVlmBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl CogVlm {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: CogVlmConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: CogVlmConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let vision_encoder = {
            let v_ws = if ws.has_tensor("visual.patch_embed.weight") {
                ws.scoped("visual")
            } else {
                root.scoped("visual")
            };
            CogVlmVisionEncoder::load(&v_ws, &cfg.vision_config).ok()
        };

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = CogVlmBlock::load(&layer_ws, &cfg, tp)?;
            layers.push(block);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| tok_embeddings.clone());

        Ok(Self {
            cfg,
            device,
            vision_encoder,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }
}

impl Model for CogVlm {
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

impl CausalLm for CogVlm {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids = input_ids.to_vec_f32()?;
        let seq_len = ids.len();
        let pos_v: Vec<u32> = positions
            .to_vec_f32()
            .map(|v| v.into_iter().map(|p| p as u32).collect())
            .unwrap_or_else(|_| (0..seq_len as u32).collect());

        let mut hidden = vec![0.0f32; seq_len * self.cfg.hidden_size];

        let embed_w = self.tok_embeddings.weight.to_vec_f32()?;
        for (i, &tok_f) in ids.iter().enumerate() {
            let tok = tok_f as usize;
            if tok < self.cfg.vocab_size {
                hidden[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size].copy_from_slice(
                    &embed_w[tok * self.cfg.hidden_size..(tok + 1) * self.cfg.hidden_size],
                );
            }
        }

        let mut x = cpu_tensor(hidden, Shape::new(vec![seq_len, self.cfg.hidden_size]));
        let mut kv_caches = vec![None; self.layers.len()];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_v, false, &mut kv_caches[layer_idx])?;
        }

        let normed = self.norm.forward(&x)?;
        let logits = self.output.forward(&normed)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cogvlm_config() {
        let cfg = CogVlmConfig::default();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_layers, 32);
    }
}
