//! Muse-Glimmer dense transformer — `CausalLm` implementation.
//!
//! Text backbone: Llama-style pre-norm + GQA attention with **hybrid**
//! sliding-window / full attention (layers in `attention_sliding_window_layer_ids`
//! attend only within `attention_sliding_window_size`; the rest attend to the
//! full causal prefix), **per-layer RoPE base** (`per_layer_rope_theta`), an
//! attention **`qk_scale_factor`** (multiplies `1/sqrt(head_dim)`), a per-layer
//! **`output_multiplier`** on the residual stream, and **`final_logit_softcapping`**
//! on the output logits (`softcap * tanh(logits/softcap)`).
//!
//! Vision: optional Muse-Glimmer temporal-patch ViT (`grim_models_vision::GlimmerVision`)
//! feeding a `GlimmerProjector` (Linear `vision_hidden → text_hidden`) whose
//! token embeddings may be merged into the text sequence.

use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
use grim_core::error::{Error, Result};
use grim_core::model::{
    AdapterHandle, CausalLm, ModalityHint, MultimodalCausalLm, MultimodalInputs,
};
use grim_core::rng::SimpleRng;
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_models_vision::{GlimmerVision, GlimmerVisionConfig};
use grim_nn::pick_device_for_storage_device;
use grim_nn::{
    ColumnParallelLinear, Embedding, Linear, RmsNorm, Rope, RowParallelLinear,
    TensorParallelConfig, WeightSource,
};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

use crate::block::{LlamaLayerCache, plan_kv_head_sharding};
use crate::multimodal::merge_multimodal_embeddings;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MuseGlimmerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    /// Per-layer RoPE base. `per_layer_rope_theta[layer]` if present,
    /// otherwise `base_rope_theta`.
    pub per_layer_rope_theta: Vec<f32>,
    pub base_rope_theta: f32,
    /// Layer indices that use sliding-window attention.
    pub sliding_window_layer_ids: Vec<usize>,
    /// Sliding window size for the layers above (0 disables the mask).
    pub sliding_window_size: usize,
    /// Attention score multiplier: `scale = qk_scale_factor / sqrt(head_dim)`.
    pub qk_scale_factor: f32,
    /// Per-layer residual-stream multiplier (len 0 or `num_layers`; missing
    /// layers default to 1.0).
    pub output_multiplier: Vec<f32>,
    /// `softcap * tanh(logits/softcap)` on final logits. `<= 0` disables.
    pub final_logit_softcapping: f32,
    pub max_seq_len: usize,
    /// Optional vision encoder config (temporal-patch ViT). `None` = text-only.
    pub vision: Option<GlimmerVisionConfig>,
}

impl MuseGlimmerConfig {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let per_layer_rope = value
            .get("per_layer_rope_theta")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();
        let sliding_layers = value
            .get("attention_sliding_window_layer_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default();
        let output_mult = value
            .get("output_multiplier")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        let vision_cfg = value.get("vision_config").map(|v| {
            let vu = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let vf = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
            grim_models_vision::GlimmerVisionConfig {
                image_temporal: if vu("image_temporal") > 0 {
                    vu("image_temporal")
                } else {
                    2
                },
                image_size: if vu("image_size") > 0 {
                    vu("image_size")
                } else {
                    336
                },
                patch_size: if vu("patch_size") > 0 {
                    vu("patch_size")
                } else {
                    14
                },
                temporal_patch_size: if vu("temporal_patch_size") > 0 {
                    vu("temporal_patch_size")
                } else {
                    2
                },
                in_channels: if vu("num_channels") > 0 {
                    vu("num_channels")
                } else {
                    3
                },
                hidden_size: if vu("hidden_size") > 0 {
                    vu("hidden_size")
                } else {
                    1024
                },
                num_heads: if vu("num_attention_heads") > 0 {
                    vu("num_attention_heads")
                } else {
                    16
                },
                num_layers: if vu("num_hidden_layers") > 0 {
                    vu("num_hidden_layers")
                } else {
                    24
                },
                intermediate_size: if vu("intermediate_size") > 0 {
                    vu("intermediate_size")
                } else {
                    4096
                },
                rms_norm_eps: if vf("layer_norm_eps") > 0.0 {
                    vf("layer_norm_eps")
                } else {
                    1e-5
                },
                merge_size: if vu("merge_size") > 0 {
                    vu("merge_size")
                } else {
                    2
                },
                use_vision_norm: true,
            }
        });

        let hidden_size = u("hidden_size");
        let num_heads = u("num_attention_heads");
        let raw_head_dim = u("head_dim");
        // MOD-2 fix: `hidden_size / num_heads` can evaluate to 0 (e.g. when
        // `hidden_size == 0` or is smaller than `num_heads`), and a 0
        // `head_dim` becomes a divisor later in the model and panics. Clamp
        // to a minimum of 1 so the value is always a valid, non-zero stride.
        let head_dim = if raw_head_dim > 0 {
            raw_head_dim
        } else if num_heads > 0 {
            hidden_size / num_heads
        } else {
            64
        };
        let head_dim = head_dim.max(1);

        MuseGlimmerConfig {
            vocab_size: u("vocab_size"),
            hidden_size,
            num_heads,
            num_kv_heads: u("num_key_value_heads"),
            head_dim,
            num_layers: u("num_hidden_layers"),
            intermediate_size: u("intermediate_size"),
            rms_norm_eps: f("rms_norm_eps"),
            per_layer_rope_theta: per_layer_rope,
            base_rope_theta: if f("rope_theta") > 0.0 {
                f("rope_theta")
            } else {
                10000.0
            },
            sliding_window_layer_ids: sliding_layers,
            sliding_window_size: u("sliding_window"),
            qk_scale_factor: if f("qk_scale_factor") > 0.0 {
                f("qk_scale_factor")
            } else {
                1.0
            },
            output_multiplier: output_mult,
            final_logit_softcapping: f("final_logit_softcapping"),
            max_seq_len: if u("max_position_embeddings") > 0 {
                u("max_position_embeddings")
            } else {
                8192
            },
            vision: vision_cfg,
        }
    }

    fn rope_theta_for(&self, layer: usize) -> f32 {
        self.per_layer_rope_theta
            .get(layer)
            .copied()
            .unwrap_or(self.base_rope_theta)
    }
    fn is_sliding_layer(&self, layer: usize) -> bool {
        self.sliding_window_size > 0 && self.sliding_window_layer_ids.contains(&layer)
    }
    fn output_multiplier_for(&self, layer: usize) -> f32 {
        self.output_multiplier.get(layer).copied().unwrap_or(1.0)
    }
}

#[allow(dead_code)]
pub const MUSE_GLIMMER_30B_TENSOR_KEYS: &[&str] = &[
    "tok_embeddings.weight",
    "norm.weight",
    "output.weight",
    "layers.{i}.attn_norm.weight",
    "layers.{i}.ffn_norm.weight",
    "layers.{i}.attn.wq.weight",
    "layers.{i}.attn.wk.weight",
    "layers.{i}.attn.wv.weight",
    "layers.{i}.attn.wo.weight",
    "layers.{i}.ffn.w_gate.weight",
    "layers.{i}.ffn.w_up.weight",
    "layers.{i}.ffn.w_down.weight",
];

impl ModelConfig for MuseGlimmerConfig {
    fn name(&self) -> &str {
        "muse-glimmer"
    }
    fn modality(&self) -> ModalityHint {
        if self.vision.is_some() {
            ModalityHint::MultimodalInTextOut
        } else {
            ModalityHint::TextInTextOut
        }
    }
    fn context_length(&self) -> u64 {
        self.max_seq_len as u64
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// GlimmerBlock — hybrid sliding-window/full attention
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct GlimmerConfigRefs {
    head_dim: usize,
    local_num_heads: usize,
    local_num_kv_heads: usize,
    sliding_window: Option<usize>,
    qk_scale_factor: f32,
}

#[derive(Clone)]
pub struct GlimmerBlock {
    pub attn_norm: RmsNorm,
    pub wq: ColumnParallelLinear,
    pub wk: ColumnParallelLinear,
    pub wv: ColumnParallelLinear,
    pub wo: RowParallelLinear,
    pub ffn_norm: RmsNorm,
    pub w_gate: ColumnParallelLinear,
    pub w_up: ColumnParallelLinear,
    pub w_down: RowParallelLinear,
    pub rope: Rope,
    pub tp_config: TensorParallelConfig,
    pub(crate) _cfg: GlimmerConfigRefs,
}

impl GlimmerBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &MuseGlimmerConfig, layer: usize) -> Result<Self> {
        Self::load_tp(ws, cfg, layer, ws.tp_config())
    }

    pub fn load_tp(
        ws: &WeightSource<'_>,
        cfg: &MuseGlimmerConfig,
        layer: usize,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let wq = Linear::load_column_parallel(
            &ws.pp("attn").pp("wq"),
            cfg.hidden_size,
            cfg.num_heads * cfg.head_dim,
            /*has_bias=*/ false,
            tp,
        )?;
        let wk = Linear::load_column_parallel(
            &ws.pp("attn").pp("wk"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            /*has_bias=*/ false,
            tp,
        )?;
        let wv = Linear::load_column_parallel(
            &ws.pp("attn").pp("wv"),
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            /*has_bias=*/ false,
            tp,
        )?;
        let wo = Linear::load_row_parallel(
            &ws.pp("attn").pp("wo"),
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_size,
            /*has_bias=*/ false,
            tp,
        )?;
        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let w_gate = Linear::load_column_parallel(
            &ws.pp("ffn").pp("w_gate"),
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/ false,
            tp,
        )?;
        let w_up = Linear::load_column_parallel(
            &ws.pp("ffn").pp("w_up"),
            cfg.hidden_size,
            cfg.intermediate_size,
            /*has_bias=*/ false,
            tp,
        )?;
        let w_down = Linear::load_row_parallel(
            &ws.pp("ffn").pp("w_down"),
            cfg.intermediate_size,
            cfg.hidden_size,
            /*has_bias=*/ false,
            tp,
        )?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta_for(layer));

        let (local_num_heads, local_num_kv_heads, _kv_head_replica_factor) =
            plan_kv_head_sharding(cfg.num_heads, cfg.num_kv_heads, tp.world_size)?;

        Ok(Self {
            attn_norm,
            wq: ColumnParallelLinear::new(wq, tp),
            wk: ColumnParallelLinear::new(wk, tp),
            wv: ColumnParallelLinear::new(wv, tp),
            wo: RowParallelLinear::new(wo, tp),
            ffn_norm,
            w_gate: ColumnParallelLinear::new(w_gate, tp),
            w_up: ColumnParallelLinear::new(w_up, tp),
            w_down: RowParallelLinear::new(w_down, tp),
            rope,
            tp_config: tp,
            _cfg: GlimmerConfigRefs {
                head_dim: cfg.head_dim,
                local_num_heads,
                local_num_kv_heads,
                sliding_window: if cfg.is_sliding_layer(layer) {
                    Some(cfg.sliding_window_size)
                } else {
                    None
                },
                qk_scale_factor: cfg.qk_scale_factor,
            },
        })
    }

    /// Decode one layer. Appends K/V to the (host-mirror) cache, runs hybrid
    /// causal (+ optional window) attention with `qk_scale_factor`, then the
    /// dense SwiGLU FFN. Returns `(out, k, v)` post-RoPE for the caller's
    /// KV population.
    pub fn forward_with_kv(
        &self,
        x_2d: &Tensor,
        positions: &[u32],
        mut cache: Option<&mut LlamaLayerCache>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self._cfg;
        let x_norm = self.attn_norm.forward(x_2d)?;
        let q = self.wq.forward(&x_norm)?;
        let k = self.wk.forward(&x_norm)?;
        let v = self.wv.forward(&x_norm)?;

        let q_rot = self.apply_rope_multi_head(&q, positions, cfg.local_num_heads)?;
        let k_rot = self.apply_rope_multi_head(&k, positions, cfg.local_num_kv_heads)?;

        let s = q_rot.shape().dims()[0];

        let v_vec = v.to_vec_f32()?;
        let (full_k, full_v) = match cache.as_deref_mut() {
            Some(c) => {
                c.k_cache.extend_from_slice(&k_rot.to_vec_f32()?);
                c.v_cache.extend_from_slice(&v_vec);
                let total = c.past_len + s;
                c.past_len = total;
                (c.k_cache.clone(), c.v_cache.clone())
            }
            None => (k_rot.to_vec_f32()?, v_vec),
        };
        let kv_len = match cache.as_ref() {
            Some(c) => c.past_len,
            None => s,
        };
        let past_len = kv_len - s;

        let attn_out = self.hybrid_attention(&q_rot, &full_k, &full_v, past_len, s, kv_len)?;
        let attn_out = reshaped_view(
            &attn_out,
            &Shape::new(vec![s, cfg.local_num_heads * cfg.head_dim]),
        )?;
        let attn_out = self.wo.forward(&attn_out)?;

        let added = grim_nn::modules::add_on_device(x_2d, &attn_out)?;

        let x_norm = self.ffn_norm.forward(&added)?;
        let gate = self.w_gate.forward(&x_norm)?;
        let up = self.w_up.forward(&x_norm)?;
        let silu_storage = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let ffn_out = self.w_down.forward(&silu_storage)?;
        let out = grim_nn::modules::add_on_device(&added, &ffn_out)?;

        Ok((out, k_rot, v))
    }

    /// RoPE a (B, S, num_heads*head_dim) tensor, staying on-device via a
    /// zero-copy (B, S*heads, head_dim) relabel + backend `rope` kernel.
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
            return Err(Error::Shape(format!(
                "expected 2-D or 3-D tensor, got {dims:?}"
            )));
        };
        if d != num_heads * self._cfg.head_dim {
            return Err(Error::Shape(format!(
                "expected last dim {n}*{hd}={exp}, got {d}",
                n = num_heads,
                hd = self._cfg.head_dim,
                exp = num_heads * self._cfg.head_dim
            )));
        }
        let rope_shape = Shape::new(vec![b, s * num_heads, self._cfg.head_dim]);
        let relabeled = Tensor::new(
            x.storage().clone(),
            rope_shape.clone(),
            x.dtype(),
            x.provenance().clone(),
            x.device().clone(),
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
        let dev = pick_device_for_storage_device(x.device());
        match dev.rope(
            relabeled.storage().as_ref(),
            &ext_positions,
            &self.rope.config,
            &rope_shape,
        ) {
            Ok((st, _h)) => {
                let rope_out = Tensor::new(
                    Arc::from(st),
                    rope_shape,
                    x.dtype(),
                    x.provenance().clone(),
                    x.device().clone(),
                );
                reshaped_view(
                    &rope_out,
                    &Shape::new(vec![b, s, num_heads * self._cfg.head_dim]),
                )
            }
            Err(_) => {
                let rope_out = self.rope.forward(&relabeled, &ext_positions)?;
                reshaped_view(
                    &rope_out,
                    &Shape::new(vec![b, s, num_heads * self._cfg.head_dim]),
                )
            }
        }
    }

    /// Hybrid attention on host mirrors. Causal always; when `sliding_window`
    /// is set for this layer, each query attends to at most the previous
    /// `window` keys. Scores are `scale = qk_scale_factor / sqrt(head_dim)`,
    /// no softcap at the attention stage (Muse-Glimmer only softcaps logits).
    #[allow(clippy::too_many_arguments)]
    fn hybrid_attention(
        &self,
        q_3d: &Tensor,
        full_k: &[f32],
        full_v: &[f32],
        _past_len: usize,
        q_len: usize,
        _kv_len: usize,
    ) -> Result<Tensor> {
        let cfg = &self._cfg;
        let qd = q_3d.to_vec_f32()?;
        // Shared helper keeps this layer's exact scale and per-layer sliding
        // window; causal limit is past_len + t.
        crate::shared_attention::fused_or_scalar_attention_scaled(
            &qd,
            full_k,
            full_v,
            cfg.local_num_heads,
            cfg.local_num_kv_heads,
            cfg.head_dim,
            q_len,
            cfg.sliding_window,
            cfg.qk_scale_factor / (cfg.head_dim as f32).sqrt(),
            self.wo.weight().device(),
        )
    }
}

/// Zero-copy relabel (B, S*H, D) → (B, S, H*D) when the flat element count and
/// storage rank match; otherwise physically reshape. Mirrors `block::reshaped_view`.
fn reshaped_view(x: &Tensor, shape: &Shape) -> Result<Tensor> {
    if x.shape().elem_count() != shape.elem_count() {
        return Err(Error::Shape(format!(
            "reshaped_view: element count mismatch {:?} vs {:?}",
            x.shape().dims(),
            shape.dims()
        )));
    }
    // Prefer a zero-copy relabel when storage rank matches the target rank.
    if x.storage().shape().dims().is_empty() || x.storage().shape().rank() == shape.rank() {
        return Ok(Tensor::new(
            x.storage().clone(),
            shape.clone(),
            x.dtype(),
            x.provenance().clone(),
            x.device().clone(),
        ));
    }
    let dev = pick_device_for_storage_device(x.device());
    let data = x.to_vec_f32()?;
    let st = dev.from_cpu(&data, shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(st),
        shape.clone(),
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    ))
}

// ---------------------------------------------------------------------------
// Projector
// ---------------------------------------------------------------------------

/// Projection of `GlimmerVision` patch embeddings into the text hidden space.
#[derive(Clone)]
pub struct GlimmerProjector {
    pub proj: Linear,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl GlimmerProjector {
    pub fn load(ws: &WeightSource<'_>, input_dim: usize, output_dim: usize) -> Result<Self> {
        let proj = Linear::load(&ws.pp("proj"), input_dim, output_dim, false)?;
        Ok(Self {
            proj,
            input_dim,
            output_dim,
        })
    }

    pub fn forward(&self, patch_features: &Tensor) -> Result<Tensor> {
        self.proj.forward(patch_features).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct MuseGlimmer {
    pub cfg: MuseGlimmerConfig,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<GlimmerBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
    pub(crate) vision: Option<GlimmerVision>,
    pub(crate) projector: Option<GlimmerProjector>,
}

impl MuseGlimmer {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: MuseGlimmerConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: MuseGlimmerConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let _num_layers = cfg.num_layers;
        let tok_embeddings =
            Embedding::load(&ws.pp("tok_embeddings"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(GlimmerBlock::load_tp(
                &ws.pp("layers").pp(&i.to_string()),
                &cfg,
                i,
                tp,
            )?);
        }
        let norm = RmsNorm::load(&ws.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = match Linear::load_column_parallel(
            &ws.pp("output"),
            cfg.hidden_size,
            cfg.vocab_size,
            /*has_bias=*/ false,
            tp,
        ) {
            Ok(o) => o,
            Err(_) => {
                let ws_unsharded = ws.with_tp_config(TensorParallelConfig::default());
                Linear::load(
                    &ws_unsharded.pp("output"),
                    cfg.hidden_size,
                    cfg.vocab_size,
                    false,
                )?
            }
        };

        let vision = if let Some(vcfg) = &cfg.vision {
            let vision_ws = ws.pp("vision");
            Some(GlimmerVision::load_tp(
                device.clone(),
                &vision_ws,
                vcfg.clone(),
                tp,
            )?)
        } else {
            None
        };
        let projector = match (&vision, &cfg.vision) {
            (Some(_), Some(vcfg)) => Some(GlimmerProjector::load(
                &ws.pp("vision").pp("proj"),
                vcfg.hidden_size,
                cfg.hidden_size,
            )?),
            _ => None,
        };

        Ok(Self {
            cfg,
            device: device.clone(),
            tok_embeddings,
            layers,
            norm,
            output,
            vision,
            projector,
        })
    }

    pub fn random(device: Device, cfg: MuseGlimmerConfig) -> Self {
        use grim_backend_cpu::cpu_tensor;
        let _num_layers = cfg.num_layers;
        let _dev = CpuDevice::new();
        let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_F00Du64);

        let embed_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let tok_embeddings = Embedding {
            weight: cpu_tensor(
                embed_data,
                Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
        };

        let mut linear = |out: usize, inp: usize| {
            let data: Vec<f32> = (0..out * inp)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            Linear::from_tensor(cpu_tensor(data, Shape::new(vec![out, inp])), None)
        };
        let rms = |dim: usize| RmsNorm {
            weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
            eps: cfg.rms_norm_eps,
        };

        let tp = TensorParallelConfig::default();
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer in 0..cfg.num_layers {
            let (local_num_heads, local_num_kv_heads, _kv_head_replica_factor) =
                plan_kv_head_sharding(cfg.num_heads, cfg.num_kv_heads, 1).unwrap();
            layers.push(GlimmerBlock {
                attn_norm: rms(cfg.hidden_size),
                wq: ColumnParallelLinear::new(
                    linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size),
                    tp,
                ),
                wk: ColumnParallelLinear::new(
                    linear(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
                    tp,
                ),
                wv: ColumnParallelLinear::new(
                    linear(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
                    tp,
                ),
                wo: RowParallelLinear::new(
                    linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim),
                    tp,
                ),
                ffn_norm: rms(cfg.hidden_size),
                w_gate: ColumnParallelLinear::new(
                    linear(cfg.intermediate_size, cfg.hidden_size),
                    tp,
                ),
                w_up: ColumnParallelLinear::new(linear(cfg.intermediate_size, cfg.hidden_size), tp),
                w_down: RowParallelLinear::new(linear(cfg.hidden_size, cfg.intermediate_size), tp),
                rope: Rope::new(cfg.head_dim, cfg.rope_theta_for(layer)),
                tp_config: tp,
                _cfg: GlimmerConfigRefs {
                    head_dim: cfg.head_dim,
                    local_num_heads,
                    local_num_kv_heads,
                    sliding_window: if cfg.is_sliding_layer(layer) {
                        Some(cfg.sliding_window_size)
                    } else {
                        None
                    },
                    qk_scale_factor: cfg.qk_scale_factor,
                },
            });
        }

        let norm = rms(cfg.hidden_size);
        let output = linear(cfg.vocab_size, cfg.hidden_size);
        Self {
            cfg: cfg.clone(),
            device,
            tok_embeddings,
            layers,
            norm,
            output,
            vision: None,
            projector: None,
        }
    }

    pub fn embed_token(&self, token: u32) -> Result<Tensor> {
        Ok(self
            .tok_embeddings
            .forward(&[token], 1, self.cfg.hidden_size)?)
    }

    pub fn decode(
        &self,
        hidden: &Tensor,
        positions: &[u32],
    ) -> Result<(Tensor, Tensor, Vec<(Tensor, Tensor)>)> {
        let mut h = hidden.clone();
        let mut kv_pairs = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let (attn_out, k, v) = layer.forward_with_kv(&h, positions, None)?;
            kv_pairs.push((k, v));
            h = attn_out;
            let om = self.cfg.output_multiplier_for(i);
            if om != 1.0 {
                h = scale_tensor(&h, om)?;
            }
        }
        let h = self.norm.forward(&h)?;
        let mut logits = self.output.forward(&h)?;
        logits = maybe_softcap(&logits, self.cfg.final_logit_softcapping)?;
        Ok((logits, h, kv_pairs))
    }

    /// Encode an image with the optional vision tower + projector, returning
    /// per-token projected embeddings of shape `(num_tokens, hidden_size)`.
    /// Errors when the model has no vision stack configured.
    pub fn encode_image_tokens(&self, image: &Tensor) -> Result<Tensor> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| Error::Config("MuseGlimmer has no vision encoder".into()))?;
        let proj = self
            .projector
            .as_ref()
            .ok_or_else(|| Error::Config("MuseGlimmer has no vision projector".into()))?;
        let feats = vision.encode_image(image)?;
        proj.forward(&feats)
    }
}

fn scale_tensor(t: &Tensor, s: f32) -> Result<Tensor> {
    let data = t.to_vec_f32()?;
    let scaled: Vec<f32> = data.into_iter().map(|x| x * s).collect();
    let dev = pick_device_for_storage_device(t.device());
    let st = dev.from_cpu(&scaled, t.shape(), DType::F32)?;
    Ok(Tensor::new(
        Arc::from(st),
        t.shape().clone(),
        DType::F32,
        t.provenance().clone(),
        t.device().clone(),
    ))
}

fn maybe_softcap(logits: &Tensor, softcap: f32) -> Result<Tensor> {
    if softcap <= 0.0 {
        return Ok(logits.clone());
    }
    let data = logits.to_vec_f32()?;
    let capped: Vec<f32> = data
        .into_iter()
        .map(|x| softcap * (x / softcap).tanh())
        .collect();
    let dev = pick_device_for_storage_device(logits.device());
    let st = dev.from_cpu(&capped, logits.shape(), DType::F32)?;
    Ok(Tensor::new(
        Arc::from(st),
        logits.shape().clone(),
        DType::F32,
        logits.provenance().clone(),
        logits.device().clone(),
    ))
}

impl Model for MuseGlimmer {
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

impl CausalLm for MuseGlimmer {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids = text_ids(input_ids)?;
        let seq_len = ids.len();
        let hidden: Vec<f32> = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?
            .to_vec_f32()?;
        let pos_vec = text_positions(positions, seq_len)?;
        self.forward_embeddings(session, &hidden, &pos_vec, adapters)
    }
}

impl MultimodalCausalLm for MuseGlimmer {
    fn forward_multimodal(
        &self,
        session: &mut dyn SessionT,
        inputs: &MultimodalInputs,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids = text_ids(&inputs.input_ids)?;
        let seq_len = ids.len();
        let mut hidden: Vec<f32> = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?
            .to_vec_f32()?;
        match (&inputs.image_patches, &inputs.image_placeholder_mask) {
            (Some(image), Some(placeholder_indices)) if !placeholder_indices.is_empty() => {
                let patch = self.encode_image_tokens(image)?.to_vec_f32()?;
                merge_multimodal_embeddings(
                    &mut hidden,
                    &patch,
                    placeholder_indices,
                    self.cfg.hidden_size,
                )?;
            }
            (Some(_), _) => {
                return Err(Error::Config(
                    "MuseGlimmer multimodal input has an image but no image_placeholder_mask"
                        .into(),
                ));
            }
            _ => {}
        }
        let pos_vec = text_positions(positions, seq_len)?;
        self.forward_embeddings(session, &hidden, &pos_vec, adapters)
    }
}

impl MuseGlimmer {
    /// Run the transformer core on pre-built (merged) embeddings. Shared by the
    /// text-only `CausalLm::forward` and the multimodal `forward_multimodal`.
    fn forward_embeddings(
        &self,
        session: &mut dyn SessionT,
        hidden: &[f32],
        pos_vec: &[u32],
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let seq_len = pos_vec.len();
        let hidden_shape = Shape::new(vec![seq_len, self.cfg.hidden_size]);
        let dev = pick_device_for_storage_device(&self.device);
        let hidden_storage = dev.from_cpu(hidden, &hidden_shape, DType::F32)?;
        let hidden_t = Tensor::new(
            Arc::from(hidden_storage),
            hidden_shape,
            DType::F32,
            self.tok_embeddings.weight.provenance().clone(),
            self.device.clone(),
        );
        if session.model_state().is_none() {
            session.set_model_state(Box::new(
                (0..self.layers.len())
                    .map(|_| Some(LlamaLayerCache::new()))
                    .collect::<Vec<_>>(),
            ));
        }
        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<LlamaLayerCache>>>())
            .ok_or_else(|| {
                Error::Session(
                    "MuseGlimmer::forward: session.model_state has wrong type for this operation"
                        .into(),
                )
            })?;

        let (logits, hidden_state, _) = {
            let mut h = hidden_t;
            for (i, layer) in self.layers.iter().enumerate() {
                let cache = caches[i].as_mut();
                let (attn_out, _k, _v) = layer.forward_with_kv(&h, pos_vec, cache)?;
                h = attn_out;
                let om = self.cfg.output_multiplier_for(i);
                if om != 1.0 {
                    h = scale_tensor(&h, om)?;
                }
            }
            let h = self.norm.forward(&h)?;
            let logits = self.output.forward(&h)?;
            let logits = maybe_softcap(&logits, self.cfg.final_logit_softcapping)?;
            (logits, h, Vec::<u32>::new())
        };
        session.set_last_hidden_state(hidden_state);
        let logits = if adapters.is_empty() {
            logits
        } else {
            crate::lora::apply_adapters_to_logits(&logits, adapters, self.cfg.hidden_size)?
        };
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

/// Decode `input_ids` (F32) into a `Vec<u32>` token sequence.
fn text_ids(input_ids: &Tensor) -> Result<Vec<u32>> {
    match input_ids.dtype() {
        d if d == DType::F32 => {
            let v = input_ids.to_vec_f32()?;
            Ok(v.into_iter().map(|x| x as u32).collect())
        }
        _ => Err(Error::Tensor(grim_tensor::Error::Unimplemented(
            "non-F32 input_ids not yet supported".into(),
        ))),
    }
}

/// Decode `positions` (F32) into `Vec<u32>`; falls back to `0..len` when it is
/// not element-count matched with the sequence.
fn text_positions(positions: &Tensor, seq_len: usize) -> Result<Vec<u32>> {
    if positions.shape().dims().iter().product::<usize>() == seq_len {
        Ok(positions
            .to_vec_f32()?
            .into_iter()
            .map(|x| x as u32)
            .collect())
    } else {
        Ok((0..seq_len).map(|i| i as u32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> MuseGlimmerConfig {
        MuseGlimmerConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            per_layer_rope_theta: vec![10000.0, 100000.0],
            base_rope_theta: 10000.0,
            sliding_window_layer_ids: vec![0],
            sliding_window_size: 4,
            qk_scale_factor: 1.0,
            output_multiplier: vec![1.0, 1.0],
            final_logit_softcapping: 20.0,
            max_seq_len: 32,
            vision: None,
        }
    }

    #[test]
    fn smoke_tiny_glimmer_logits() {
        use grim_core::session::Inner;
        let cfg = tiny_cfg();
        let model = MuseGlimmer::random(Device::Cpu, cfg);
        let tok = grim_backend_cpu::cpu_tensor(vec![1.0f32], Shape::new(vec![1]));
        let mut sess = Inner::new(model.device.clone());
        let logits = CausalLm::forward(&model, &mut sess, &tok, &tok, &[]).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 64]);
        let v = logits.to_vec_f32().unwrap();
        assert!(v.iter().any(|x| x.is_finite()));
        assert!(!v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn final_logit_softcap_bounds_logits() {
        let data: Vec<f32> = vec![-5.0, 5.0, 50.0];
        let t = grim_backend_cpu::cpu_tensor(data, Shape::new(vec![1, 3]));
        let capped = maybe_softcap(&t, 20.0).unwrap();
        let out = capped.to_vec_f32().unwrap();
        assert!(out[2] < 20.0, "50 must be capped under 20, got {}", out[2]);
        assert!((out[0].abs()).abs() < 5.0);
        let none = maybe_softcap(&t, 0.0).unwrap();
        assert_eq!(none.to_vec_f32().unwrap(), vec![-5.0, 5.0, 50.0]);
    }

    #[test]
    fn decode_kv_cache_grows() {
        let cfg = tiny_cfg();
        let model = MuseGlimmer::random(Device::Cpu, cfg);
        let hidden = grim_backend_cpu::cpu_tensor(vec![0.1f32; 1 * 32], Shape::new(vec![1, 32]));
        let (logits, _, kv) = model.decode(&hidden, &[0]).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 64]);
        assert_eq!(kv.len(), 2);
    }

    #[test]
    fn multimodal_forward_merges_vision_tokens() {
        use grim_core::model::MultimodalInputs;
        use grim_core::session::Inner;
        use grim_models_vision::{GlimmerVision, GlimmerVisionConfig};

        let mut cfg = tiny_cfg();
        cfg.vision = Some(GlimmerVisionConfig {
            image_temporal: 2,
            image_size: 8,
            patch_size: 4,
            temporal_patch_size: 1,
            in_channels: 3,
            hidden_size: 16,
            num_heads: 2,
            num_layers: 2,
            intermediate_size: 32,
            rms_norm_eps: 1e-5,
            merge_size: 2,
            use_vision_norm: true,
        });

        let mut model = MuseGlimmer::random(Device::Cpu, cfg.clone());
        let vision = GlimmerVision::random(Device::Cpu, cfg.vision.clone().unwrap());
        let proj = GlimmerProjector {
            proj: Linear::from_tensor(
                grim_backend_cpu::cpu_tensor(
                    (0..16 * 32).map(|i| ((i as f32) - 256.0) * 0.001).collect(),
                    Shape::new(vec![32, 16]),
                ),
                None,
            ),
            input_dim: 16,
            output_dim: 32,
        };
        model.vision = Some(vision);
        model.projector = Some(proj);
        assert_eq!(model.cfg.modality(), ModalityHint::MultimodalInTextOut);

        // Text sequence of 6 tokens with 4 image-token placeholder slots.
        let ids = grim_backend_cpu::cpu_tensor(
            vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 4.0],
            Shape::new(vec![6]),
        );
        let img = grim_backend_cpu::cpu_tensor(
            (0..3 * 2 * 8 * 8).map(|i| (i as f32) * 0.01).collect(),
            Shape::new(vec![3, 2, 8, 8]),
        );
        let inputs = MultimodalInputs {
            input_ids: ids,
            image_patches: Some(img),
            mel_frames: None,
            image_placeholder_mask: Some(vec![1, 2, 3, 4]),
            audio_placeholder_mask: None,
        };
        let pos = grim_backend_cpu::cpu_tensor(
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::new(vec![6]),
        );
        let mut sess = Inner::new(model.device.clone());
        let logits =
            MultimodalCausalLm::forward_multimodal(&model, &mut sess, &inputs, &pos, &[]).unwrap();
        assert_eq!(logits.shape().dims(), &[6, 64]);
        let v = logits.to_vec_f32().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn multimodal_forward_errors_without_vision() {
        use grim_core::model::MultimodalInputs;
        use grim_core::session::Inner;

        let cfg = tiny_cfg();
        let model = MuseGlimmer::random(Device::Cpu, cfg);
        let ids = grim_backend_cpu::cpu_tensor(vec![3.0f32, 0.0, 4.0], Shape::new(vec![3]));
        let img =
            grim_backend_cpu::cpu_tensor(vec![0.0f32; 3 * 2 * 8 * 8], Shape::new(vec![3, 2, 8, 8]));
        let inputs = MultimodalInputs {
            input_ids: ids,
            image_patches: Some(img),
            mel_frames: None,
            image_placeholder_mask: Some(vec![1]),
            audio_placeholder_mask: None,
        };
        let pos = grim_backend_cpu::cpu_tensor(vec![0.0f32, 1.0, 2.0], Shape::new(vec![3]));
        let mut sess = Inner::new(model.device.clone());
        match MultimodalCausalLm::forward_multimodal(&model, &mut sess, &inputs, &pos, &[]) {
            Err(Error::Config(_)) => {}
            other => panic!("expected Config error without vision, got {:?}", other),
        }
    }

    #[test]
    fn parses_muse_glimmer_30b_config() {
        let json_str = r#"{
            "architectures": ["MuseGlimmerForCausalLM"],
            "hidden_size": 7168,
            "num_hidden_layers": 48,
            "num_attention_heads": 56,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 18432,
            "rms_norm_eps": 1e-06,
            "rope_theta": 500000.0,
            "vocab_size": 256000,
            "final_logit_softcapping": 30.0
        }"#;
        let v: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let cfg = MuseGlimmerConfig::from_hf(&v);
        assert_eq!(cfg.hidden_size, 7168);
        assert_eq!(cfg.num_layers, 48);
        assert!((cfg.final_logit_softcapping - 30.0).abs() < 1e-6);
        assert_eq!(cfg.name(), "muse-glimmer");
    }
}
