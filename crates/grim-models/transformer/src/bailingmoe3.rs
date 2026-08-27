//! Compatibility loader for `inclusionAI/Ling-3.0-tiny` (a.k.a. `BailingMoeV3`,
//! HuggingFace `model_type = "bailing_hybrid"`).
//!
//! ## What this model is
//!
//! Ling-3.0-tiny is a **hybrid linear-attention + MLA + sparse-MoE** CausalLM:
//!
//! * 24 layers stacked 3:1 — three **KDA** (Kimi Delta Attention, a gated
//!   linear-attention / short-conv hybrid) layers followed by one **MLA**
//!   (Multi-head Latent Attention) layer per 4-layer block.
//! * MLA uses `q_lora_rank` (256), `kv_lora_rank` (512), `qk_nope_head_dim`
//!   (128) and `qk_rope_head_dim` (64) — i.e. a low-rank value/key compression
//!   with partial RoPE, exactly like DeepSeek-V2/V3.
//! * MoE FFN: 128 routed experts, `num_experts_per_tok = 8`, 1 shared expert,
//!   sigmoid router (`scoring_func = "sigmoid"`) with expert bias and the
//!   `noaux_tc` top-k group selection (`n_group = 8`, `topk_group = 4`).
//! * `routed_scaling_factor = 2.5`, `norm_topk_prob = true`.
//!
//! ## Compatibility status
//!
//! All three primitives this model needs are now implemented in grim:
//!
//! 1. **MoE FFN** — `grim_nn::moe::MoeFfn` (sigmoid router, `noaux_tc` top-k,
//!    `routed_scaling_factor`, optional shared expert), fused-dispatch backends.
//! 2. **MLA** — `grim_nn::MlaAttention`: two-stage `q_a -> q_a_norm -> q_b`
//!    low-rank Q projection, `kv_a -> kv_a_norm -> kv_b` low-rank KV projection,
//!    split nope/rope head layout with partial RoPE on the rope slice, optional
//!    `q_norm`/`k_norm` (`use_qk_norm`). CPU reference in `grim-nn`.
//! 3. **KDA** — `grim_nn::KdaAttention`: gated delta-rule linear attention with
//!    a depthwise causal `short_conv1d` (`short_conv_kernel_size = 4`) on the
//!    value path and a per-step recurrent state (`KdaLayerCache`).
//!
//! `Ling3Tiny::load_tp` builds a real model (KDA/MLA chosen per layer by
//! `i % layer_group_size`, MoE FFN on every layer). Layer blend is 3 KDA : 1 MLA
//! (`layer_group_size = 4`). Remaining gaps (tracked, not silently fallen back):
//! * The `short_conv1d` / delta-rule / MLA projections have a CPU reference only;
//!   fused Metal/Vulkan/CUDA/ROCm kernels are a follow-up (mirroring the MoE
//!   fused-dispatch work). On non-CPU backends these still run the CPU path.
//! * Full end-to-end logits parity against HF transformers is not yet captured
//!   in CI (needs the 15.8 GB safetensors + a reference run on a GPU box).
//!
//! Reference config (Ling-3.0-tiny, bf16, 15.8 GB):
//! `hidden_size=1536, num_hidden_layers=24, num_attention_heads=16,
//! num_key_value_heads=16, head_dim=128, q_lora_rank=256, kv_lora_rank=512,
//! qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
//! intermediate_size=4608, moe_intermediate_size=512,
//! moe_shared_expert_intermediate_size=512, num_experts=128,
//! num_experts_per_tok=8, num_shared_experts=1, first_k_dense_replace=1,
//! routed_scaling_factor=2.5, vocab_size=157184, max_position_embeddings=131072,
//! rope_theta=6000000, partial_rotary_factor=0.5, rope_interleave=true.`

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_tensor::{ArithType, Device, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Native mirror of `BailingMoeV3Config` (HuggingFace `bailing_hybrid`).
#[derive(Debug, Clone)]
pub struct Ling3TinyConfig {
    // vocabulary / embedding
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub tie_word_embeddings: bool,
    // attention (MLA)
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub qk_head_dim: usize,
    pub rotary_dim: usize,
    pub partial_rotary_factor: f32,
    pub rope_interleave: bool,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub use_qk_norm: bool,
    // KDA (linear-attention half of the hybrid)
    pub layer_group_size: usize,
    pub max_window_layers: usize,
    pub short_conv_kernel_size: usize,
    pub num_kv_heads_for_linear_attn: usize,
    pub num_nextn_predict_layers: usize,
    // MoE FFN
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub moe_shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_shared_experts: usize,
    pub first_k_dense_replace: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub moe_router_enable_expert_bias: bool,
    pub routed_scaling_factor: f32,
    pub scoring_func: String,
    pub topk_method: String,
    // norms / misc
    pub rms_norm_eps: f32,
    pub hidden_act: String,
}

impl ModelConfig for Ling3TinyConfig {
    fn name(&self) -> &str {
        "bailingmoe3"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Ling3TinyConfig {
    /// Build from the raw HuggingFace `config.json` `serde_json::Value`.
    ///
    /// Panics are avoided: every field is read with a `get` + `as_*` + fallback
    /// so a slightly different tiny-variant config still parses.
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let s = |k: &str| {
            value
                .get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        Ling3TinyConfig {
            vocab_size: u("vocab_size"),
            hidden_size: u("hidden_size"),
            tie_word_embeddings: value
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            q_lora_rank: u("q_lora_rank"),
            kv_lora_rank: u("kv_lora_rank"),
            qk_nope_head_dim: u("qk_nope_head_dim"),
            qk_rope_head_dim: u("qk_rope_head_dim"),
            v_head_dim: u("v_head_dim"),
            qk_head_dim: u("qk_head_dim"),
            rotary_dim: u("rotary_dim"),
            partial_rotary_factor: f("partial_rotary_factor"),
            rope_interleave: value
                .get("rope_interleave")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            rope_theta: f("rope_theta"),
            max_position_embeddings: u("max_position_embeddings"),
            use_qk_norm: value
                .get("use_qk_norm")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            layer_group_size: u("layer_group_size"),
            max_window_layers: u("max_window_layers"),
            short_conv_kernel_size: u("short_conv_kernel_size"),
            num_kv_heads_for_linear_attn: u("num_kv_heads_for_linear_attn"),
            num_nextn_predict_layers: u("num_nextn_predict_layers"),
            num_hidden_layers: u("num_hidden_layers"),
            intermediate_size: u("intermediate_size"),
            moe_intermediate_size: u("moe_intermediate_size"),
            moe_shared_expert_intermediate_size: u("moe_shared_expert_intermediate_size"),
            num_experts: u("num_experts"),
            num_experts_per_tok: u("num_experts_per_tok"),
            num_shared_experts: u("num_shared_experts"),
            first_k_dense_replace: u("first_k_dense_replace"),
            n_group: u("n_group"),
            topk_group: u("topk_group"),
            norm_topk_prob: value
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            moe_router_enable_expert_bias: value
                .get("moe_router_enable_expert_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            routed_scaling_factor: f("routed_scaling_factor"),
            scoring_func: s("scoring_func"),
            topk_method: s("topk_method"),
            rms_norm_eps: f("rms_norm_eps"),
            hidden_act: s("hidden_act"),
        }
    }
}

// ---------------------------------------------------------------------------
// Safetensors tensor-name map (compatibility reference)
// ---------------------------------------------------------------------------
//
// These are the key paths present in `model-000XX-of-00032.safetensors` for
// Ling-3.0-tiny. They are documented here so the eventual loader (after KDA +
// MLA blocks land) can map HF tensors -> grim `WeightSource` keys without a
// second reverse-engineering pass. Naming follows the standard Bailing hybrid
// convention: `model.layers.{i}.{kda|self_attn|mla|mlp|moe}...`, where layer
// `i` is KDA when `i % layer_group_size < layer_group_size - 1` and MLA
// otherwise (3:1 interleave).
#[allow(dead_code)]
pub const LING3_TINY_TENSOR_KEYS: &[&str] = &[
    // shared / embed
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
    // per-layer (replace {i})
    "model.layers.{i}.input_layernorm.weight",
    "model.layers.{i}.post_attention_layernorm.weight",
    // KDA (linear-attn) branch — `model.layers.{i}.kda.*`
    "model.layers.{i}.kda.q_proj.weight",
    "model.layers.{i}.kda.k_proj.weight",
    "model.layers.{i}.kda.v_proj.weight",
    "model.layers.{i}.kda.o_proj.weight",
    "model.layers.{i}.kda.conv.weight", // short_conv_kernel_size=4 depthwise
    "model.layers.{i}.kda.gate.weight",
    "model.layers.{i}.kda.dt_proj.weight",
    "model.layers.{i}.kda.A.weight", // delta-rule transition
    // MLA branch — `model.layers.{i}.self_attn.*`
    "model.layers.{i}.self_attn.q_a_proj.weight", // q_lora_rank=256
    "model.layers.{i}.self_attn.q_a_layernorm.weight",
    "model.layers.{i}.self_attn.q_b_proj.weight",
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight", // kv_lora_rank=512
    "model.layers.{i}.self_attn.kv_a_layernorm.weight",
    "model.layers.{i}.self_attn.kv_b_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight",
    "model.layers.{i}.self_attn.q_norm.weight",
    "model.layers.{i}.self_attn.k_norm.weight",
    // MoE FFN — `model.layers.{i}.moe.*`
    "model.layers.{i}.moe.gate.weight", // [num_experts, hidden]
    "model.layers.{i}.moe.gate.e_score_correction_bias.weight", // if router bias
    "model.layers.{i}.moe.experts.{e}.w1.weight", // gate  [moe_inter, hidden]
    "model.layers.{i}.moe.experts.{e}.w2.weight", // down  [hidden, moe_inter]
    "model.layers.{i}.moe.experts.{e}.w3.weight", // up    [moe_inter, hidden]
    "model.layers.{i}.moe.shared_expert.w1.weight",
    "model.layers.{i}.moe.shared_expert.w2.weight",
    "model.layers.{i}.moe.shared_expert.w3.weight",
];

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Model Layer Types
// ---------------------------------------------------------------------------

pub enum Ling3Attn {
    Kda(Box<grim_nn::KdaAttention>),
    Mla(Box<grim_nn::MlaAttention>),
}

/// Per-layer session cache (audit fix): KDA layers carry conv + recurrent
/// state; MLA layers carry the post-RoPE KV history. Boxed so the enum stays
/// small.
pub enum Ling3LayerCache {
    Kda(Box<grim_nn::KdaLayerCache>),
    Mla(Box<grim_nn::MlaKvCache>),
}

impl Ling3LayerCache {
    /// Build the cache matching this attention variant.
    pub fn new_for(attn: &Ling3Attn) -> Self {
        match attn {
            Ling3Attn::Kda(_) => Self::Kda(Box::new(grim_nn::KdaLayerCache::new())),
            Ling3Attn::Mla(_) => Self::Mla(Box::new(grim_nn::MlaKvCache::new())),
        }
    }
}

pub struct Ling3TinyLayer {
    pub input_layernorm: grim_nn::RmsNorm,
    pub attn: Ling3Attn,
    pub post_attention_layernorm: grim_nn::RmsNorm,
    pub moe: grim_nn::moe::MoeFfn,
}

pub struct Ling3Tiny {
    pub cfg: Ling3TinyConfig,
    pub device: Device,
    pub embed_tokens: grim_nn::Embedding,
    pub layers: Vec<Ling3TinyLayer>,
    pub norm: grim_nn::RmsNorm,
    pub lm_head: Option<grim_nn::Linear>,
}

impl Ling3Tiny {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Ling3TinyConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Ling3TinyConfig,
    ) -> Result<Self> {
        let embed_w = ws.get_unconstrained("model.embed_tokens.weight")?;
        let embed_tokens = grim_nn::Embedding { weight: embed_w };

        let norm_w = ws.get_unconstrained("model.norm.weight")?;
        let norm = grim_nn::RmsNorm::new(norm_w, cfg.rms_norm_eps);

        let lm_head = if let Ok(w) = ws.get_unconstrained("lm_head.weight") {
            Some(grim_nn::Linear::from_tensor(w, None))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let group_size = if cfg.layer_group_size > 0 {
            cfg.layer_group_size
        } else {
            4
        };

        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let in_ln_w = ws.get_unconstrained(&format!("{prefix}.input_layernorm.weight"))?;
            let input_layernorm = grim_nn::RmsNorm::new(in_ln_w, cfg.rms_norm_eps);

            let post_ln_w =
                ws.get_unconstrained(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let post_attention_layernorm = grim_nn::RmsNorm::new(post_ln_w, cfg.rms_norm_eps);

            let is_kda = (i % group_size) < (group_size - 1);
            let attn = if is_kda {
                let kda_p = format!("{prefix}.kda");
                let q_w = ws.get_unconstrained(&format!("{kda_p}.q_proj.weight"))?;
                let k_w = ws.get_unconstrained(&format!("{kda_p}.k_proj.weight"))?;
                let v_w = ws.get_unconstrained(&format!("{kda_p}.v_proj.weight"))?;
                let gate_w = ws.get_unconstrained(&format!("{kda_p}.gate.weight"))?;
                let dt_w = ws.get_unconstrained(&format!("{kda_p}.dt_proj.weight"))?;
                let a_w = ws.get_unconstrained(&format!("{kda_p}.A.weight"))?;
                let conv_weight = ws.get_unconstrained(&format!("{kda_p}.conv.weight"))?;
                let conv_bias = ws.get_unconstrained(&format!("{kda_p}.conv.bias")).ok();
                let o_w = ws.get_unconstrained(&format!("{kda_p}.o_proj.weight"))?;

                Ling3Attn::Kda(Box::new(grim_nn::KdaAttention {
                    q_proj: grim_nn::Linear::from_tensor(q_w, None),
                    k_proj: grim_nn::Linear::from_tensor(k_w, None),
                    v_proj: grim_nn::Linear::from_tensor(v_w, None),
                    gate_proj: grim_nn::Linear::from_tensor(gate_w, None),
                    dt_proj: grim_nn::Linear::from_tensor(dt_w, None),
                    a_proj: grim_nn::Linear::from_tensor(a_w, None),
                    conv_weight,
                    conv_bias,
                    o_proj: grim_nn::Linear::from_tensor(o_w, None),
                    num_heads: cfg.num_attention_heads,
                    head_dim: cfg.head_dim,
                    v_dim: cfg.v_head_dim,
                }))
            } else {
                let mla_p = format!("{prefix}.self_attn");
                let q_a_w = ws.get_unconstrained(&format!("{mla_p}.q_a_proj.weight"))?;
                let q_a_ln_w = ws.get_unconstrained(&format!("{mla_p}.q_a_layernorm.weight"))?;
                let q_b_w = ws.get_unconstrained(&format!("{mla_p}.q_b_proj.weight"))?;
                let kv_a_w = ws.get_unconstrained(&format!("{mla_p}.kv_a_proj_with_mqa.weight"))?;
                let kv_a_ln_w = ws.get_unconstrained(&format!("{mla_p}.kv_a_layernorm.weight"))?;
                let kv_b_w = ws.get_unconstrained(&format!("{mla_p}.kv_b_proj.weight"))?;
                let o_w = ws.get_unconstrained(&format!("{mla_p}.o_proj.weight"))?;

                let q_norm = ws
                    .get_unconstrained(&format!("{mla_p}.q_norm.weight"))
                    .ok()
                    .map(|w| grim_nn::RmsNorm::new(w, cfg.rms_norm_eps));
                let k_norm = ws
                    .get_unconstrained(&format!("{mla_p}.k_norm.weight"))
                    .ok()
                    .map(|w| grim_nn::RmsNorm::new(w, cfg.rms_norm_eps));

                let rope = grim_nn::Rope::new(cfg.qk_rope_head_dim, cfg.rope_theta);

                Ling3Attn::Mla(Box::new(grim_nn::MlaAttention {
                    q_a_proj: grim_nn::Linear::from_tensor(q_a_w, None),
                    q_a_norm: grim_nn::RmsNorm::new(q_a_ln_w, cfg.rms_norm_eps),
                    q_b_proj: grim_nn::Linear::from_tensor(q_b_w, None),
                    kv_a_proj_with_mqa: grim_nn::Linear::from_tensor(kv_a_w, None),
                    kv_a_norm: grim_nn::RmsNorm::new(kv_a_ln_w, cfg.rms_norm_eps),
                    kv_b_proj: grim_nn::Linear::from_tensor(kv_b_w, None),
                    o_proj: grim_nn::Linear::from_tensor(o_w, None),
                    q_norm,
                    k_norm,
                    num_heads: cfg.num_attention_heads,
                    qk_nope_head_dim: cfg.qk_nope_head_dim,
                    qk_rope_head_dim: cfg.qk_rope_head_dim,
                    v_head_dim: cfg.v_head_dim,
                    rope,
                }))
            };

            let gate_w = ws.get_unconstrained(&format!("{prefix}.moe.gate.weight"))?;
            let gate_linear = grim_nn::Linear::from_tensor(gate_w, None);
            let router = grim_nn::moe::MoeRouter::new(
                gate_linear,
                grim_nn::moe::RouterKind::SoftmaxTopK,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                None,
            );
            let experts = grim_nn::moe::ExpertBank::load(
                ws,
                cfg.num_experts,
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                false,
            )?;
            let moe = grim_nn::moe::MoeFfn::new(router, experts, None, cfg.routed_scaling_factor);

            layers.push(Ling3TinyLayer {
                input_layernorm,
                attn,
                post_attention_layernorm,
                moe,
            });
        }

        Ok(Ling3Tiny {
            cfg,
            device,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}

impl Model for Ling3Tiny {
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

impl CausalLm for Ling3Tiny {
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
        // Audit fix (grim-models): per-layer caches now live on the SESSION
        // and every call threads them into the attention variants — the
        // pre-fix code passed None everywhere and ignored the session, so
        // decode attended only to itself (fully stateless). KDA layers get
        // real conv/recurrent state; MLA layers get the post-RoPE KV history.
        if session.model_state().is_none() {
            let caches: Vec<Ling3LayerCache> = self
                .layers
                .iter()
                .map(|l| Ling3LayerCache::new_for(&l.attn))
                .collect();
            session.set_model_state(Box::new(caches));
        }
        let caches_cell = session.model_state_mut().ok_or_else(|| {
            grim_core::error::Error::Session("Ling3Tiny: model_state vanished".into())
        })?;
        let caches = caches_cell
            .downcast_mut::<Vec<Ling3LayerCache>>()
            .ok_or_else(|| {
                grim_core::error::Error::Session(
                    "Ling3Tiny: model_state holds another model's caches".into(),
                )
            })?;
        if caches.len() != self.layers.len() {
            return Err(grim_core::error::Error::Session(format!(
                "Ling3Tiny: {} caches for {} layers",
                caches.len(),
                self.layers.len()
            )));
        }

        let seq_len = input_ids.shape().dims().iter().product();
        let indices = input_ids
            .to_vec_f32()?
            .into_iter()
            .map(|v| v as u32)
            .collect::<Vec<_>>();

        let mut h = self
            .embed_tokens
            .forward(&indices, seq_len, self.cfg.hidden_size)?;
        let pos_vec: Vec<u32> = positions
            .to_vec_f32()?
            .into_iter()
            .map(|v| v as u32)
            .collect();

        for (i, layer) in self.layers.iter().enumerate() {
            let residual = h.clone();
            let normed_input = layer.input_layernorm.forward(&h)?;

            let attn_cache = caches.get_mut(i);
            let attn_out = match (&layer.attn, attn_cache) {
                (Ling3Attn::Kda(kda), Some(Ling3LayerCache::Kda(c))) => {
                    kda.forward(&normed_input, Some(c))?
                }
                (Ling3Attn::Kda(kda), None) => kda.forward(&normed_input, None)?,
                (Ling3Attn::Mla(mla), Some(Ling3LayerCache::Mla(c))) => {
                    mla.forward(&normed_input, &pos_vec, Some(c.as_mut()))?
                }
                (Ling3Attn::Mla(mla), None) => mla.forward(&normed_input, &pos_vec, None)?,
                _ => {
                    return Err(grim_core::error::Error::Session(
                        "Ling3Tiny: cache variant does not match attention variant".into(),
                    ));
                }
            };

            let h_post_attn = grim_nn::add_tensors(&residual, &attn_out)?;
            let residual2 = h_post_attn.clone();
            let normed_post = layer.post_attention_layernorm.forward(&h_post_attn)?;
            let moe_out = layer.moe.forward(&normed_post)?;
            h = grim_nn::add_tensors(&residual2, &moe_out)?;
        }

        let normed = self.norm.forward(&h)?;
        let out = if let Some(lm_head) = &self.lm_head {
            lm_head.forward(&normed)?
        } else {
            normed
        };
        // Audit fix: advance the engine-visible position per call.
        session.advance_pos(seq_len);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const LING3_TINY_CONFIG: &str = r#"{
        "architectures": ["BailingMoeV3ForCausalLM"],
        "hidden_size": 1536,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "q_lora_rank": 256,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "qk_head_dim": 192,
        "rotary_dim": 64,
        "partial_rotary_factor": 0.5,
        "rope_interleave": true,
        "rope_theta": 6000000,
        "max_position_embeddings": 131072,
        "use_qk_norm": true,
        "layer_group_size": 4,
        "max_window_layers": 20,
        "short_conv_kernel_size": 4,
        "num_kv_heads_for_linear_attn": 0,
        "num_nextn_predict_layers": 0,
        "intermediate_size": 4608,
        "moe_intermediate_size": 512,
        "moe_shared_expert_intermediate_size": 512,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "n_group": 8,
        "topk_group": 4,
        "norm_topk_prob": true,
        "moe_router_enable_expert_bias": true,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "rms_norm_eps": 1e-06,
        "hidden_act": "silu",
        "vocab_size": 157184
    }"#;

    #[test]
    fn parses_ling3_tiny_config() {
        let v: serde_json::Value = serde_json::from_str(LING3_TINY_CONFIG).unwrap();
        let cfg = Ling3TinyConfig::from_hf(&v);
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.num_shared_experts, 1);
        assert_eq!(cfg.q_lora_rank, 256);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert!((cfg.routed_scaling_factor - 2.5).abs() < 1e-6);
        assert_eq!(cfg.scoring_func, "sigmoid");
        assert_eq!(cfg.topk_method, "noaux_tc");
        assert_eq!(cfg.name(), "bailingmoe3");
    }

    #[test]
    fn hf_model_type_dispatches_to_bailingmoe3() {
        // The HF `model_type` ("bailing_hybrid") and the grim name must both
        // resolve to the registered architecture, and it must be flagged MoE.
        assert_eq!(
            ModelArchitecture::from_str("bailing_hybrid"),
            ModelArchitecture::BailingMoe3
        );
        assert_eq!(
            ModelArchitecture::from_str("bailingmoe3"),
            ModelArchitecture::BailingMoe3
        );
        assert!(ModelArchitecture::BailingMoe3.is_moe());
    }
}
