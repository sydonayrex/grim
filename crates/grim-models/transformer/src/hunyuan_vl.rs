//! Hunyuan-VL multimodal vision-language model with compact ViT encoder and 4-section M-RoPE.
//!
//! # Architecture Details
//! - **Compact ViT Visual Backbone**: Patch extraction and spatial projection into language hidden dimension.
//! - **4-Section M-RoPE**: Rotary position frequencies split into `[2, 2, 2, 2]` for multidimensional spatio-temporal alignment.
//! - **Language Decoder**: Grouped Query Attention and SwiGLU activations.

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Tensor};

// ---------------------------------------------------------------------------
// Vision Config & Encoder
// ---------------------------------------------------------------------------

/// Configuration for Hunyuan-VL visual feature encoder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HunyuanVlVisionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
}

impl Default for HunyuanVlVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            patch_size: 16,
            num_channels: 3,
            intermediate_size: 128,
            out_hidden_size: 64,
        }
    }
}

/// Vision Transformer encoder for Hunyuan-VL.
pub struct HunyuanVlVisionEncoder {
    pub patch_embed: Linear,
    pub proj: Linear,
    pub norm: RmsNorm,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
}

impl HunyuanVlVisionEncoder {
    pub fn load(ws: &WeightSource<'_>, cfg: &HunyuanVlVisionConfig) -> Result<Self> {
        let in_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size;
        let patch_embed = Linear::load_shape(&ws.scoped("patch_embed"), [in_dim, cfg.hidden_size])?;
        let norm = RmsNorm::load(&ws.scoped("norm"), cfg.hidden_size, 1e-5)?;
        let proj = Linear::load_shape(&ws.scoped("proj"), [cfg.hidden_size, cfg.out_hidden_size])?;

        Ok(Self {
            patch_embed,
            proj,
            norm,
            hidden_size: cfg.hidden_size,
            out_hidden_size: cfg.out_hidden_size,
        })
    }

    /// Encodes visual patches into text hidden space.
    pub fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        let h = self.patch_embed.forward(patches)?;
        let normed = self.norm.forward(&h)?;
        Ok(self.proj.forward(&normed)?)
    }
}

// ---------------------------------------------------------------------------
// Model Config
// ---------------------------------------------------------------------------

/// Configuration for Hunyuan-VL model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HunyuanVlConfig {
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
    pub mrope_section: [usize; 4],
    pub image_token_id: usize,
    pub im_start_id: usize,
    pub im_end_id: usize,
    pub im_newline_id: usize,
    pub vision_config: HunyuanVlVisionConfig,
}

impl Default for HunyuanVlConfig {
    fn default() -> Self {
        Self {
            vocab_size: 120818,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            num_layers: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 4096,
            mrope_section: [2, 2, 2, 2],
            image_token_id: 5,
            im_start_id: 120118,
            im_end_id: 120119,
            im_newline_id: 120121,
            vision_config: HunyuanVlVisionConfig::default(),
        }
    }
}

impl ModelConfig for HunyuanVlConfig {
    fn name(&self) -> &str {
        "hunyuan_vl"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::MultimodalInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// Transformer block for Hunyuan-VL.
pub struct HunyuanVlBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl HunyuanVlBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &HunyuanVlConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

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

        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            attn_norm,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// GPU-first forward: Q/K/V, RoPE, KV-cache concat, attention and the
    /// SwiGLU MLP all run on the tensor's device. Host paths are only
    /// reached through the fused-kernel fallback guards.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.attn_norm.forward(x)?;

        let q = self.wq.forward(&normed_attn)?;
        let k = self.wk.forward(&normed_attn)?;
        let v = self.wv.forward(&normed_attn)?;

        let q = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &q,
            self.num_heads,
            positions,
        )?;
        let k = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &k,
            self.num_kv_heads,
            positions,
        )?;

        // Device-side history: prev rows stay resident, only the new rows
        // are appended (D2D arena copy when the backend supports it).
        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let full_k = crate::shared_attention::concat_rows_on_device(prev_k, &k)?;
            let full_v = crate::shared_attention::concat_rows_on_device(prev_v, &v)?;
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k.clone(), v.clone()));
            (k.clone(), v.clone())
        };
        let kv_len = k_all.shape().dims()[0];

        // Shared helper applies the causal mask at cache_offset + s (fixes
        // future-token leakage during multi-token prefill).
        let attn_tensor = crate::shared_attention::fused_attention_tensors(
            &q,
            &k_all,
            &v_all,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            kv_len,
            None,
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;

        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        let normed_ffn = self.ffn_norm.forward(&res1)?;
        let gate = self.w_gate.forward(&normed_ffn)?;
        let up = self.w_up.forward(&normed_ffn)?;
        let act = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let mlp_out = self.w_down.forward(&act)?;

        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct HunyuanVl {
    pub cfg: HunyuanVlConfig,
    pub device: Device,
    pub vision_encoder: Option<HunyuanVlVisionEncoder>,
    pub tok_embeddings: Linear,
    pub layers: Vec<HunyuanVlBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl HunyuanVl {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: HunyuanVlConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: HunyuanVlConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let vision_encoder = {
            let v_ws = if ws.has_tensor("visual.patch_embed.weight") {
                ws.scoped("visual")
            } else {
                root.scoped("visual")
            };
            HunyuanVlVisionEncoder::load(&v_ws, &cfg.vision_config).ok()
        };

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = HunyuanVlBlock::load(&layer_ws, &cfg, tp)?;
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

impl Model for HunyuanVl {
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

impl CausalLm for HunyuanVl {
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
        let ids_f32 = input_ids.to_vec_f32()?;
        let seq_len = ids_f32.len();
        let ids: Vec<u32> = ids_f32.iter().map(|&t| t as u32).collect();
        let pos_v: Vec<u32> = positions
            .to_vec_f32()
            .map(|v| v.into_iter().map(|p| p as u32).collect())
            .unwrap_or_else(|_| (0..seq_len as u32).collect());

        // GPU-first embedding gather: rows land on the weight's device; the
        // vocab×hidden table never crosses to host.
        let mut x = grim_nn::embedding_gather_on_device(
            &self.tok_embeddings.weight,
            &ids,
            seq_len,
            self.cfg.hidden_size,
        )?;

        let mut kv_caches = vec![None; self.layers.len()];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_v, &mut kv_caches[layer_idx])?;
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
    fn test_hunyuan_vl_config() {
        let cfg = HunyuanVlConfig::default();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.mrope_section, [2, 2, 2, 2]);
    }
}
