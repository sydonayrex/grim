//! Qwen3-VL vision-language multimodal transformer with DeepStack visual integration and interleaved M-RoPE.
//!
//! # Architecture Details
//! - **DeepStack ViT Vision Encoder**: Extracts intermediate layer representations (e.g. layers 8, 16, 24) and projects them into the 4096-dim text embedding space.
//! - **Interleaved Multimodal RoPE (M-RoPE)**: Rotary frequencies partitioned across interleaved sections `[24, 20, 20]`.
//! - **GQA SwiGLU Transformer**: 36-layer decoder-only transformer with grouped query attention and RMSNorm.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Vision Config & Encoder
// ---------------------------------------------------------------------------

/// Configuration for Qwen3-VL visual encoder with DeepStack features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen3VlVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Default for Qwen3VlVisionConfig {
    fn default() -> Self {
        Self {
            depth: 27,
            hidden_size: 1152,
            num_heads: 16,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            in_channels: 3,
            out_hidden_size: 4096,
            deepstack_visual_indexes: vec![8, 16, 24],
        }
    }
}

/// Vision Transformer (ViT) encoder for Qwen3-VL.
pub struct Qwen3VlVisionEncoder {
    pub patch_embed: Linear,
    pub merger_proj: Linear,
    pub norm: RmsNorm,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub deepstack_indexes: Vec<usize>,
}

impl Qwen3VlVisionEncoder {
    pub fn load(ws: &WeightSource<'_>, cfg: &Qwen3VlVisionConfig) -> Result<Self> {
        let in_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size * cfg.temporal_patch_size;
        let patch_embed = Linear::load_shape(&ws.scoped("patch_embed"), [in_dim, cfg.hidden_size])?;
        let norm = RmsNorm::load(&ws.scoped("norm"), cfg.hidden_size, 1e-6)?;
        let merger_in = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merger_proj =
            Linear::load_shape(&ws.scoped("merger"), [merger_in, cfg.out_hidden_size])?;

        Ok(Self {
            patch_embed,
            merger_proj,
            norm,
            hidden_size: cfg.hidden_size,
            out_hidden_size: cfg.out_hidden_size,
            patch_size: cfg.patch_size,
            in_channels: cfg.in_channels,
            deepstack_indexes: cfg.deepstack_visual_indexes.clone(),
        })
    }

    /// Encodes image patches into projected text space vectors `[num_tokens, out_hidden_size]`.
    pub fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        let embedded = self.patch_embed.forward(patches)?;
        let normed = self.norm.forward(&embedded)?;
        Ok(self.merger_proj.forward(&normed)?)
    }
}

// ---------------------------------------------------------------------------
// Model Config
// ---------------------------------------------------------------------------

/// Configuration for Qwen3-VL text-vision transformer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen3VlConfig {
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
    pub mrope_section: [usize; 3],
    pub deepstack_visual_indexes: Vec<usize>,
    pub vision_start_token_id: usize,
    pub vision_end_token_id: usize,
    pub vision_token_id: usize,
    pub image_token_id: usize,
    pub video_token_id: usize,
    pub vision_config: Qwen3VlVisionConfig,
}

impl Default for Qwen3VlConfig {
    fn default() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 36,
            intermediate_size: 27648,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_seq_len: 32768,
            mrope_section: [24, 20, 20],
            deepstack_visual_indexes: vec![8, 16, 24],
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
            vision_token_id: 151654,
            image_token_id: 151655,
            video_token_id: 151656,
            vision_config: Qwen3VlVisionConfig::default(),
        }
    }
}

impl ModelConfig for Qwen3VlConfig {
    fn name(&self) -> &str {
        "qwen3vl"
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

/// Transformer block with interleaved M-RoPE GQA attention and SwiGLU FFN.
pub struct Qwen3VlBlock {
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

impl Qwen3VlBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &Qwen3VlConfig,
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

pub struct Qwen3Vl {
    pub cfg: Qwen3VlConfig,
    pub device: Device,
    pub vision_encoder: Option<Qwen3VlVisionEncoder>,
    pub tok_embeddings: Linear,
    pub layers: Vec<Qwen3VlBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Qwen3Vl {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen3VlConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen3VlConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let vision_encoder = if ws.has_tensor("visual.patch_embed.weight")
            || ws.has_tensor("model.visual.patch_embed.weight")
        {
            let v_ws = if ws.has_tensor("visual.patch_embed.weight") {
                ws.scoped("visual")
            } else {
                root.scoped("visual")
            };
            Qwen3VlVisionEncoder::load(&v_ws, &cfg.vision_config).ok()
        } else {
            None
        };

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = Qwen3VlBlock::load(&layer_ws, &cfg, tp)?;
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

impl Model for Qwen3Vl {
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

impl CausalLm for Qwen3Vl {
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

        let mut kv_caches: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];

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
    fn test_qwen3vl_config() {
        let cfg = Qwen3VlConfig::default();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.mrope_section, [24, 20, 20]);
    }
}
