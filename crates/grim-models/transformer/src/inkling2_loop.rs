//! Compatibility loader and native implementation for `thinkingmachines/Inkling-2-Mini`
//! (`model_type: "inkling2_loop"`, TML looped hybrid transformer).
//!
//! Reference semantics: HF `transformers.models.inkling2_loop.modeling_inkling2_loop`
//! (v5.24). Architecture: 24 layers, middle layers 6..18 run `loop_num_passes` times with
//! shared weights, block-residual attention pooling every `n_layers_per_attn_res_block`
//! layers, per-layer relative position bias + attention sinks, hybrid
//! sliding-window/full attention, short-convolution key/value mixing, sigmoid-routed
//! 128-expert MoE with softcapped SwiGLU, message gates across loop passes.
//!
//! Scope of this increment (correctness first):
//! - text tower only (vision tower + MTP head are NOT loaded)
//! - full-sequence recompute per forward (no KV cache, no conv-state reuse) — matches the
//!   reference with `cache=None` exactly; O(n^2) decode is the documented ceiling
//! - all math host-side f32; MoE experts dequantized on demand from NVFP4
//!
//! ponytail: once parity vs the HF reference is green, port attention/KV/sconv to the
//! device kernels and cache states; this host path is the parity oracle for that work.

use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::WeightSource;
use grim_quant::{mxfp4_e2m1_to_f32, nvfp4_e4m3_scale};
use grim_tensor::provider::RawTensor;
use grim_tensor::{ArithType, Device, Shape, Tensor};

// Config

/// Configuration for the `inkling2_loop` text tower. Defaults mirror the
/// `Inkling2LoopTextConfig` defaults in HF transformers 5.24.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Inkling2LoopConfig {
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub swa_head_dim: usize,
    pub sliding_window_size: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f32,
    /// true = "hybrid" (full attention), false = "hybrid_sliding"
    pub layer_is_full: Vec<bool>,
    pub rms_norm_eps: f32,
    pub sconv_kernel_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub route_scale: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub n_layers_per_attn_res_block: usize,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub final_logit_softcapping: f32,
    pub loop_layer_index_start: usize,
    pub loop_layer_index_end: usize,
    pub loop_num_passes: usize,
    pub loop_gate_rank: usize,
}

impl Default for Inkling2LoopConfig {
    fn default() -> Self {
        Self {
            vocab_size: 201024,
            unpadded_vocab_size: 200058,
            hidden_size: 2304,
            num_hidden_layers: 24,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            head_dim: 128,
            swa_num_attention_heads: 24,
            swa_num_key_value_heads: 8,
            swa_head_dim: 128,
            sliding_window_size: 1024,
            d_rel: 16,
            rel_extent: 1024,
            log_scaling_n_floor: 8192,
            log_scaling_alpha: 0.1,
            layer_is_full: Vec::new(),
            rms_norm_eps: 1e-6,
            sconv_kernel_size: 4,
            moe_intermediate_size: 1152,
            n_routed_experts: 128,
            num_experts_per_tok: 8,
            route_scale: 8.0,
            qk_l2_clip_threshold: 22.627_417,
            rel_l2_clip_threshold: 8.0,
            n_layers_per_attn_res_block: 6,
            mlp_gate_softcap: 4.0,
            mlp_up_softcap: 25.0,
            final_logit_softcapping: 30.0,
            loop_layer_index_start: 6,
            loop_layer_index_end: 18,
            loop_num_passes: 3,
            loop_gate_rank: 64,
        }
    }
}

impl ModelConfig for Inkling2LoopConfig {
    fn name(&self) -> &str {
        "inkling2_loop"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Inkling2LoopConfig {
    /// Parse an HF `config.json` value (top-level `inkling2_loop` config with a
    /// `text_config` sub-object, or a bare text-config value).
    pub fn from_hf(value: &serde_json::Value) -> Self {
        // text_config nests the real fields; fall back to the value itself.
        let text = value.get("text_config").unwrap_or(value);
        let us = |k: &str, default: usize| {
            text.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let fs = |k: &str, default: f32| {
            text.get(k)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(default)
        };

        let mut cfg = Self {
            vocab_size: us("vocab_size", 201024),
            unpadded_vocab_size: us("unpadded_vocab_size", 200058),
            hidden_size: us("hidden_size", 2304),
            num_hidden_layers: us("num_hidden_layers", 24),
            num_attention_heads: us("num_attention_heads", 24),
            num_key_value_heads: us("num_key_value_heads", 8),
            head_dim: us("head_dim", 128),
            swa_num_attention_heads: us("swa_num_attention_heads", 24),
            swa_num_key_value_heads: us("swa_num_key_value_heads", 8),
            swa_head_dim: us("swa_head_dim", 128),
            sliding_window_size: us("sliding_window_size", 1024),
            d_rel: us("d_rel", 16),
            rel_extent: us("rel_extent", 1024),
            log_scaling_n_floor: us("log_scaling_n_floor", 8192),
            log_scaling_alpha: fs("log_scaling_alpha", 0.1),
            layer_is_full: Vec::new(),
            rms_norm_eps: fs("rms_norm_eps", 1e-6),
            sconv_kernel_size: us("sconv_kernel_size", 4),
            moe_intermediate_size: us("intermediate_size", 1152),
            n_routed_experts: us("n_routed_experts", 128),
            num_experts_per_tok: us("num_experts_per_tok", 8),
            route_scale: fs("route_scale", 8.0),
            qk_l2_clip_threshold: fs("qk_l2_clip_threshold", 22.627_417),
            rel_l2_clip_threshold: fs("rel_l2_clip_threshold", 8.0),
            n_layers_per_attn_res_block: us("n_layers_per_attn_res_block", 6),
            mlp_gate_softcap: fs("mlp_gate_softcap", 4.0),
            mlp_up_softcap: fs("mlp_up_softcap", 25.0),
            final_logit_softcapping: fs("final_logit_softcapping", 30.0),
            loop_layer_index_start: us("loop_layer_index_start", 6),
            loop_layer_index_end: us("loop_layer_index_end", 18),
            loop_num_passes: us("loop_num_passes", 3),
            loop_gate_rank: us("loop_gate_rank", 64),
        };

        // loop_config sub-object carries the loop fields under different keys.
        if let Some(lc) = text.get("loop_config") {
            cfg.loop_layer_index_start = lc
                .get("core_start")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(cfg.loop_layer_index_start);
            cfg.loop_layer_index_end = lc
                .get("core_end")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(cfg.loop_layer_index_end);
            cfg.loop_num_passes = lc
                .get("num_passes")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(cfg.loop_num_passes);
            cfg.loop_gate_rank = lc
                .get("gate_rank")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(cfg.loop_gate_rank);
        }

        // layer_types: "hybrid" = full attention, "hybrid_sliding" = sliding.
        if let Some(types) = text.get("layer_types").and_then(|v| v.as_array()) {
            cfg.layer_is_full = types.iter().map(|t| t.as_str() == Some("hybrid")).collect();
        } else if let Some(local) = text.get("local_layer_ids").and_then(|v| v.as_array()) {
            // TML variant: local_layer_ids lists the sliding layers.
            let local: Vec<usize> = local
                .iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as usize)
                .collect();
            cfg.layer_is_full = (0..cfg.num_hidden_layers)
                .map(|i| !local.contains(&i))
                .collect();
        } else {
            // HF default rule: full attention every n_layers_per_attn_res_block-th layer.
            cfg.layer_is_full = (0..cfg.num_hidden_layers)
                .map(|i| (i + 1) % cfg.n_layers_per_attn_res_block == 0)
                .collect();
        }

        cfg.validate();
        cfg
    }

    fn validate(&self) {
        assert!(self.loop_num_passes >= 1, "loop_num_passes must be >= 1");
        assert!(
            self.loop_layer_index_start % self.n_layers_per_attn_res_block == 0
                && self.loop_layer_index_end % self.n_layers_per_attn_res_block == 0,
            "loop bounds must be multiples of n_layers_per_attn_res_block"
        );
        assert_eq!(
            self.layer_is_full.len(),
            self.num_hidden_layers,
            "layer_is_full must cover every layer"
        );
        assert!(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "num_attention_heads must be divisible by num_key_value_heads"
        );
    }
}

// Small host-side math helpers

/// RMS norm over each `dim`-wide row of `x`; `weight` is a per-channel gamma
/// broadcast across rows.
fn rms_norm_flat(x: &[f32], weight: Option<&[f32]>, eps: f32, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(dim) {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        match weight {
            Some(w) => out.extend(row.iter().zip(w).map(|(v, g)| v * inv * g)),
            None => out.extend(row.iter().map(|v| v * inv)),
        }
    }
    out
}

fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    rms_norm_flat(x, weight, eps, weight.map(|w| w.len()).unwrap_or(x.len()))
}

/// torch `F.linear`: W stored [out, in], returns out.
fn linear(x: &[f32], w: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0f32; out_dim];
    for (o, row) in out.iter_mut().enumerate() {
        let w_row = &w[o * in_dim..(o + 1) * in_dim];
        *row = x.iter().zip(w_row).map(|(a, b)| a * b).sum::<f32>();
    }
    out
}

fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `Inkling2LoopL2Clip`-family: per-head scaled L2 clip + optional log position scaling.
/// `states` is [seq][heads][dim].
fn scaled_log_l2_clip(
    states: &mut [Vec<Vec<f32>>],
    scale: &[f32],
    threshold: f32,
    positions: &[u32],
    log_scaling: bool,
    log_alpha: f32,
    log_floor: f64,
    eps: f32,
) {
    let (seq, heads, dim) = (states.len(), states[0].len(), states[0][0].len());
    for s in 0..seq {
        let pos_scale = if log_scaling {
            1.0 + log_alpha * (((positions[s] as f64 + 1.0) / log_floor).max(1.0).ln()) as f32
        } else {
            1.0
        };
        for h in 0..heads {
            let v = &mut states[s][h];
            for (d, item) in v.iter_mut().enumerate() {
                *item *= scale[h * dim + d] * pos_scale;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let factor = (threshold / (norm + eps)).min(1.0);
            for item in v.iter_mut() {
                *item *= factor;
            }
        }
    }
}

// Weight loads

fn load_f32(ws: &WeightSource<'_>, leaf: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = ws.get_f32(Shape::new(shape.to_vec()), leaf)?;
    t.to_vec_f32().map_err(grim_core::error::Error::from)
}

// Layer

/// One decoder layer. `is_full` selects the head geometry; `is_loop` adds the message
/// gates (controllers).

#[allow(dead_code)]
pub struct Inkling2LoopLayer {
    // attention
    wq: Vec<f32>,        // [nq*hd, H]
    wk: Vec<f32>,        // [nkv*hd, H]
    wv: Vec<f32>,        // [nkv*hd, H]
    o_gate: Vec<f32>,    // gate_attn [nq*hd, H]
    wo: Vec<f32>,        // wo_ud [H, nq*hd]
    sink: Vec<f32>,      // attn_sink_proj [nq, H]
    q_scale: Vec<f32>,   // q_l2_clip.scale [nq, hd]
    rel_scale: Vec<f32>, // rel_l2_clip.scale [nq, d_rel]
    rel_proj: Vec<f32>,  // rel_logits_proj.proj [d_rel, rel_extent]
    wr: Vec<f32>,        // wr_du [nq*d_rel, H]
    // k/v short convolutions: static-only (sliding) or static + dynamic (full)
    k_conv_static: Vec<f32>, // [K, nkv*hd] (sliding: conv1d weight transposed)
    v_conv_static: Vec<f32>,
    k_conv_proj: Option<Vec<f32>>, // [nkv*K, H]
    v_conv_proj: Option<Vec<f32>>,
    // post-sublayer static convs
    attn_sconv: Vec<f32>, // [K, H]
    mlp_sconv: Vec<f32>,
    // norms
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    v_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
    // attention-residual queries (shared, held on the model)
    // MoE
    router_w: Vec<f32>, // [E, H]
    // loop message gates (None on non-loop layers): (down, forget, gain) per q/k/v/mlp
    gates: Option<[MessageGate; 4]>,
    // per-layer flags
    is_full: bool,
    is_loop: bool,
    // geometry
    nq: usize,
    nkv: usize,
    hd: usize,
    d_rel: usize,
    rel_extent: usize,
    sliding_window: Option<usize>,
    hidden: usize,
    inter: usize,
    k: usize, // sconv kernel
}

struct MessageGate {
    down: Vec<f32>,   // [rank, 2H]
    forget: Vec<f32>, // [H, rank]
    gain: Vec<f32>,   // [H, rank]
}

impl MessageGate {
    fn load(ws: &WeightSource<'_>, cfg: &Inkling2LoopConfig, which: &str) -> Result<Self> {
        let cws = ws.scoped("controllers").scoped(which);
        Ok(Self {
            down: load_f32(
                &cws,
                "down_weight",
                &[cfg.loop_gate_rank, 2 * cfg.hidden_size],
            )?,
            forget: load_f32(
                &cws,
                "forget_weight",
                &[cfg.hidden_size, cfg.loop_gate_rank],
            )?,
            gain: load_f32(&cws, "gain_weight", &[cfg.hidden_size, cfg.loop_gate_rank])?,
        })
    }

    /// `(1 - forget) * h + gain * m`, reference `Inkling2LoopMessageGate::forward`.
    fn apply(&self, h: &[f32], m: &[f32], eps: f32) -> Vec<f32> {
        let rank = self.forget.len() / h.len();
        let mut gate_in = Vec::with_capacity(2 * h.len());
        gate_in.extend_from_slice(&rms_norm(h, None, eps));
        gate_in.extend_from_slice(&rms_norm(m, None, eps));
        let g = linear(&gate_in, &self.down, rank);
        let f = linear(&g, &self.forget, h.len());
        let a = linear(&g, &self.gain, h.len());
        h.iter()
            .zip(m)
            .zip(&f)
            .zip(&a)
            .map(|(((hv, mv), fv), av)| (1.0 - fv.tanh()) * hv + av.tanh() * mv)
            .collect()
    }
}

/// Depthwise causal short convolution, static taps, kernel `K`.
/// `weight` is [K, C] (tap 0 = current token); returns `(conv + residual) / sqrt(K + 1)`.
fn static_sconv(x: &[f32], weight: &[f32], k: usize, seq: usize, dim: usize) -> Vec<f32> {
    let norm = (k as f32 + 1.0).sqrt();
    let mut out = vec![0.0f32; seq * dim];
    for s in 0..seq {
        for tap in 0..k {
            if tap > s {
                continue;
            }
            let src = (s - tap) * dim;
            let w = &weight[tap * dim..(tap + 1) * dim];
            for (d, o) in out[s * dim..(s + 1) * dim].iter_mut().enumerate() {
                *o += x[src + d] * w[d];
            }
        }
        for d in 0..dim {
            out[s * dim + d] = (out[s * dim + d] + x[s * dim + d]) / norm;
        }
    }
    out
}

/// Dynamic short convolution (full-attention k/v): per-token taps
/// `static + weight_proj(dynamic_input)`, shared per key/value head.
/// `(conv + residual) / sqrt(2K + 1)`.
fn dynamic_sconv(
    x: &[f32],
    dynamic: &[f32],
    static_w: &[f32],
    proj: &[f32],
    k: usize,
    heads: usize,
    seq: usize,
    dim: usize,
    proj_input_dim: usize,
) -> Vec<f32> {
    let head_dim = dim / heads;
    let norm = (2.0 * k as f32 + 1.0).sqrt();
    // Per (token, head, tap) the reference adds one proj-derived scalar onto the
    // per-channel static weight: static_weight is [K, dim] (per-channel), the
    // weight_proj contribution is shared across the head's channels.
    let mut out = vec![0.0f32; seq * dim];
    for s in 0..seq {
        let dyn_row = &dynamic[s * proj_input_dim..(s + 1) * proj_input_dim];
        for h in 0..heads {
            // dynamic tap weights for this (token, head): [k]
            let mut taps = vec![0.0f32; k];
            let pw = &proj[(h * k) * proj_input_dim..(h * k + k) * proj_input_dim];
            for t in 0..k {
                taps[t] = dyn_row
                    .iter()
                    .zip(&pw[t * proj_input_dim..(t + 1) * proj_input_dim])
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            }
            for tap in 0..k {
                if tap > s {
                    continue;
                }
                let src = (s - tap) * dim + h * head_dim;
                let sw = &static_w[tap * dim + h * head_dim..tap * dim + (h + 1) * head_dim];
                for c in 0..head_dim {
                    out[s * dim + h * head_dim + c] += x[src + c] * (sw[c] + taps[tap]);
                }
            }
            for c in 0..head_dim {
                let idx = s * dim + h * head_dim + c;
                out[idx] = (out[idx] + x[idx]) / norm;
            }
        }
    }
    out
}

/// conv1d weight [C,1,K] row-major -> tap-major [K,C]
fn transpose_conv(bytes: Vec<f32>, channels: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; k * channels];
    for ci in 0..channels {
        for ki in 0..k {
            w[ki * channels + ci] = bytes[ci * k + ki];
        }
    }
    w
}

impl Inkling2LoopLayer {
    fn load(ws: &WeightSource<'_>, cfg: &Inkling2LoopConfig, idx: usize) -> Result<Self> {
        let lws = ws.scoped("layers").scoped(&idx.to_string());
        let aws = lws.scoped("attn");
        let is_full = cfg.layer_is_full[idx];
        let (nq, nkv, hd) = if is_full {
            (
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
            )
        } else {
            (
                cfg.swa_num_attention_heads,
                cfg.swa_num_key_value_heads,
                cfg.swa_head_dim,
            )
        };
        let hidden = cfg.hidden_size;
        let k = cfg.sconv_kernel_size;
        let conv_dim = nkv * hd;

        // wq_du [nq*hd, H], wk_dv/wv_dv [nkv*hd, H]
        let wq = load_f32(&aws, "wq_du.weight", &[nq * hd, hidden])?;
        let wk = load_f32(&aws, "wk_dv.weight", &[conv_dim, hidden])?;
        let wv = load_f32(&aws, "wv_dv.weight", &[conv_dim, hidden])?;
        let o_gate = load_f32(&aws, "gate_attn.weight", &[nq * hd, hidden])?;
        let wo = load_f32(&aws, "wo_ud.weight", &[hidden, nq * hd])?;
        let sink = load_f32(&aws, "attn_sink_proj.weight", &[nq, hidden])?;
        let q_scale = load_f32(&aws, "q_l2_clip.scale", &[nq, hd])?;
        let rel_scale = load_f32(&aws, "rel_l2_clip.scale", &[nq, cfg.d_rel])?;
        let rel_proj = load_f32(&aws, "rel_logits_proj.proj", &[cfg.d_rel, cfg.rel_extent])?;
        let wr = load_f32(&aws, "wr_du.weight", &[nq * cfg.d_rel, hidden])?;

        // k/v sconv: sliding = fused conv1d weight [C,1,K]; full = static_w [K,C] + proj [nkv*K, H]
        let (k_conv_static, v_conv_static, k_conv_proj, v_conv_proj) = if is_full {
            (
                load_f32(&aws.scoped("k_sconv"), "static_w", &[k, conv_dim])?,
                load_f32(&aws.scoped("v_sconv"), "static_w", &[k, conv_dim])?,
                Some(load_f32(
                    &aws.scoped("k_sconv"),
                    "weight_proj",
                    &[nkv * k, hidden],
                )?),
                Some(load_f32(
                    &aws.scoped("v_sconv"),
                    "weight_proj",
                    &[nkv * k, hidden],
                )?),
            )
        } else {
            (
                transpose_conv(
                    load_f32(&aws.scoped("k_sconv"), "weight", &[conv_dim, 1, k])?,
                    conv_dim,
                    k,
                ),
                transpose_conv(
                    load_f32(&aws.scoped("v_sconv"), "weight", &[conv_dim, 1, k])?,
                    conv_dim,
                    k,
                ),
                None,
                None,
            )
        };

        let attn_sconv = transpose_conv(
            load_f32(&lws.scoped("attn_sconv"), "weight", &[hidden, 1, k])?,
            hidden,
            k,
        );
        let mlp_sconv = transpose_conv(
            load_f32(&lws.scoped("mlp_sconv"), "weight", &[hidden, 1, k])?,
            hidden,
            k,
        );

        // norms: sliding layers share attn_norm for q/k/v; full layers have separate
        // q/k/v input norms (no gamma on the value norm).
        let (q_norm, k_norm, v_norm) = if is_full {
            (
                load_f32(&lws.scoped("q_input_norm"), "weight", &[hidden])?,
                load_f32(&lws.scoped("k_input_norm"), "weight", &[hidden])?,
                // v norm has no gamma (`use_value_norm_no_gamma`): loaded as None below.
                Vec::new(),
            )
        } else {
            let shared = load_f32(&lws.scoped("attn_norm"), "weight", &[hidden])?;
            (shared.clone(), shared, Vec::new())
        };
        let mlp_norm = load_f32(&lws.scoped("mlp_norm"), "weight", &[hidden])?;

        let router_w = load_f32(
            &lws.scoped("mlp").scoped("gate"),
            "weight",
            &[cfg.n_routed_experts, hidden],
        )?;

        let is_loop = cfg.loop_layer_index_start <= idx && idx < cfg.loop_layer_index_end;
        let gates = if is_loop {
            Some([
                MessageGate::load(&lws, cfg, "attn")?,
                MessageGate::load(&lws, cfg, "k")?,
                MessageGate::load(&lws, cfg, "v")?,
                MessageGate::load(&lws, cfg, "mlp")?,
            ])
        } else {
            None
        };

        Ok(Self {
            wq,
            wk,
            wv,
            o_gate,
            wo,
            sink,
            q_scale,
            rel_scale,
            rel_proj,
            wr,
            k_conv_static,
            v_conv_static,
            k_conv_proj,
            v_conv_proj,
            attn_sconv,
            mlp_sconv,
            q_norm,
            k_norm,
            v_norm,
            mlp_norm,
            router_w,
            gates,
            is_full,
            is_loop,
            nq,
            nkv,
            hd,
            d_rel: cfg.d_rel,
            rel_extent: if is_full {
                cfg.rel_extent
            } else {
                cfg.sliding_window_size
            },
            sliding_window: if is_full {
                None
            } else {
                Some(cfg.sliding_window_size)
            },
            hidden,
            inter: cfg.moe_intermediate_size,
            k,
        })
    }
}

// Model

pub struct Inkling2Loop {
    pub cfg: Inkling2LoopConfig,
    pub device: Device,
    embed: Vec<f32>,      // [vocab, H]
    embed_norm: Vec<f32>, // [H]
    layers: Vec<Inkling2LoopLayer>,
    attn_queries: Vec<Vec<f32>>, // per layer: [2304] or [3, 2304] (flattened)
    mlp_queries: Vec<Vec<f32>>,  // per layer: [2304]
    /// message_input query rows: [n_loop_layers, 4, H], assembled from
    /// middle_cycle_cross_pass_queries + middle_cycle_cross_pass_kv_queries.
    message_queries: Vec<Vec<f32>>,
    output_query: Vec<f32>, // [H]
    final_norm: Vec<f32>,   // [H]
    unembed: Vec<f32>,      // [unpadded_vocab, H]
    /// packed NVFP4 expert bytes + scales per layer (dequanted on demand)
    expert_w13: Vec<RawTensor>,
    expert_w13_scale: Vec<RawTensor>,
    expert_w2: Vec<RawTensor>,
    expert_w2_scale: Vec<RawTensor>,
    expert_denom: Vec<f32>,
}

/// NVFP4 expert plane dequant: packed U8 [E, out, in/2] + E4M3 scales [E, out, in/16],
/// one global denominator. Returns f32 [out, in] for expert `e`.
fn dequant_expert_plane(
    packed: &RawTensor,
    scales: &RawTensor,
    denom: f32,
    e: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let packed_per_row = in_dim / 2;
    let scales_per_row = in_dim / 16;
    let p_base = e * out_dim * packed_per_row;
    let s_base = e * out_dim * scales_per_row;
    let mut out = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        let prow =
            &packed.bytes[(p_base + o * packed_per_row)..(p_base + (o + 1) * packed_per_row)];
        let srow =
            &scales.bytes[(s_base + o * scales_per_row)..(s_base + (o + 1) * scales_per_row)];
        let orow = &mut out[o * in_dim..(o + 1) * in_dim];
        for i in 0..in_dim {
            let byte = prow[i / 2];
            let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            orow[i] = mxfp4_e2m1_to_f32(code, 127) * nvfp4_e4m3_scale(srow[i / 16]) * denom;
        }
    }
    out
}

impl Inkling2Loop {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: Inkling2LoopConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(device: Device, ws: &WeightSource<'_>, cfg: Inkling2LoopConfig) -> Result<Self> {
        let root = ws.scoped("model").scoped("llm");
        let hidden = cfg.hidden_size;

        let embed = load_f32(&root.scoped("embed"), "weight", &[cfg.vocab_size, hidden])?;
        let embed_norm = load_f32(&root.scoped("embed_norm"), "weight", &[hidden])?;
        let final_norm = load_f32(&root.scoped("norm"), "weight", &[hidden])?;
        let unembed = load_f32(
            &root.scoped("unembed"),
            "weight",
            &[cfg.unpadded_vocab_size, hidden],
        )?;

        // attention-residual query banks. attn_queries may be [H] (one query) or
        // [3, H] (full layers: q/k/v inputs); mlp_queries always [H].
        let mut attn_queries = Vec::with_capacity(cfg.num_hidden_layers);
        let mut mlp_queries = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let shape: Vec<usize> = if cfg.layer_is_full[i] {
                vec![3, hidden]
            } else {
                vec![hidden]
            };
            // checkpoint keys carry no `.weight` suffix on the query banks
            let arws = root.scoped("attn_res");
            attn_queries.push(load_f32(&arws, &format!("attn_queries.{i}"), &shape)?);
            mlp_queries.push(load_f32(&arws, &format!("mlp_queries.{i}"), &[hidden])?);
        }
        let output_query = load_f32(&root.scoped("attn_res"), "output_query", &[hidden])?;

        // Message-gate queries. HF's message_input.query is [n_loop, 4, H]; the TML
        // checkpoint ships two [2*n_loop, H] banks. Assembly order below is the
        // documented guess: cross_pass rows = (q, mlp) pairs, kv rows = (k, v) pairs.
        // ponytail: flip the pairing if HF parity shows message-attention mismatch.
        let n_loop = cfg.loop_layer_index_end - cfg.loop_layer_index_start;
        let cross = load_f32(
            &root,
            "middle_cycle_cross_pass_queries",
            &[2 * n_loop, hidden],
        )?;
        let cross_kv = load_f32(
            &root,
            "middle_cycle_cross_pass_kv_queries",
            &[2 * n_loop, hidden],
        )?;
        let mut message_queries = Vec::with_capacity(n_loop);
        for j in 0..n_loop {
            let q = cross[j * 2 * hidden..(j * 2 + 1) * hidden].to_vec();
            let mlp = cross[(j * 2 + 1) * hidden..(j * 2 + 2) * hidden].to_vec();
            let k = cross_kv[j * 2 * hidden..(j * 2 + 1) * hidden].to_vec();
            let v = cross_kv[(j * 2 + 1) * hidden..(j * 2 + 2) * hidden].to_vec();
            message_queries.push(vec![q, k, v, mlp].into_iter().flatten().collect());
        }

        // experts stay packed; dequant happens per selected expert at forward time
        let mut expert_w13 = Vec::with_capacity(cfg.num_hidden_layers);
        let mut expert_w13_scale = Vec::with_capacity(cfg.num_hidden_layers);
        let mut expert_w2 = Vec::with_capacity(cfg.num_hidden_layers);
        let mut expert_w2_scale = Vec::with_capacity(cfg.num_hidden_layers);
        let mut expert_denom = Vec::with_capacity(cfg.num_hidden_layers);
        let out13 = 2 * cfg.moe_intermediate_size;
        let out2 = hidden;
        for i in 0..cfg.num_hidden_layers {
            let ews = root
                .scoped("layers")
                .scoped(&i.to_string())
                .scoped("mlp")
                .scoped("experts");
            let w13 = ews.get_raw_packed("w13_weight")?;
            let w13s = ews.get_raw_packed("w13_weight_scale")?;
            let w2 = ews.get_raw_packed("w2_weight")?;
            let w2s = ews.get_raw_packed("w2_weight_scale")?;
            debug_assert_eq!(
                w13.shape,
                vec![cfg.n_routed_experts, out13, cfg.hidden_size / 2]
            );
            debug_assert_eq!(
                w2.shape,
                vec![cfg.n_routed_experts, out2, cfg.moe_intermediate_size / 2]
            );
            let denom = load_f32(&ews, "w13_weight_outer_scale_denominator", &[1])?[0];
            expert_w13.push(w13);
            expert_w13_scale.push(w13s);
            expert_w2.push(w2);
            expert_w2_scale.push(w2s);
            expert_denom.push(denom);
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = Inkling2LoopLayer::load(&root, &cfg, i)?;
            layers.push(layer);
        }

        Ok(Self {
            cfg,
            device,
            embed,
            embed_norm,
            layers,
            attn_queries,
            mlp_queries,
            message_queries,
            output_query,
            final_norm,
            unembed,
            expert_w13,
            expert_w13_scale,
            expert_w2,
            expert_w2_scale,
            expert_denom,
        })
    }

    /// Sigmoid top-k router (fp32, reference `Inkling2LoopTopKRouter`).
    fn route(&self, layer: usize, x: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let scores: Vec<f32> = self.layers[layer]
            .router_w
            .chunks(self.cfg.hidden_size)
            .map(|row| sigmoid(x.iter().zip(row).map(|(a, b)| a * b).sum::<f32>()))
            .collect();
        let k = self.cfg.num_experts_per_tok;
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap());
        idx.truncate(k);
        let mut weights: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in &mut weights {
            *w /= sum;
        }
        for w in &mut weights {
            *w *= self.cfg.route_scale;
        }
        (idx, weights)
    }

    /// Softcapped-SwiGLU MoE for one token; experts dequanted on demand.
    fn moe(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        let (experts, weights) = self.route(layer, x);
        let hidden = self.cfg.hidden_size;
        let inter = self.cfg.moe_intermediate_size;
        let mut out = vec![0.0f32; hidden];
        for (&e, &w) in experts.iter().zip(&weights) {
            let w13 = dequant_expert_plane(
                &self.expert_w13[layer],
                &self.expert_w13_scale[layer],
                self.expert_denom[layer],
                e,
                2 * inter,
                hidden,
            );
            let gate = linear(x, &w13[..inter * hidden], inter);
            let up = linear(x, &w13[inter * hidden..], inter);
            let mut h = vec![0.0f32; inter];
            for i in 0..inter {
                h[i] = softcap(gate[i], self.cfg.mlp_gate_softcap)
                    * sigmoid(gate[i])
                    * softcap(up[i], self.cfg.mlp_up_softcap);
            }
            let w2 = dequant_expert_plane(
                &self.expert_w2[layer],
                &self.expert_w2_scale[layer],
                self.expert_denom[layer],
                e,
                hidden,
                inter,
            );
            for (o, acc) in out.iter_mut().enumerate() {
                *acc += w * h
                    .iter()
                    .zip(&w2[o * inter..(o + 1) * inter])
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            }
        }
        out
    }

    /// Per-position non-causal block-state pooling (reference
    /// `Inkling2LoopBlockAttentionResidual`): at every position the learned queries
    /// attend over that position's own stacked block states + the partial state.
    /// `block_states` is [pos][state][H]; `partial` is seq*H. Returns nq vectors of
    /// seq*H.
    fn block_attention(
        &self,
        queries: &[f32],
        nq: usize,
        block_states: &[Vec<Vec<f32>>],
        partial: &[f32],
    ) -> Vec<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let seq = partial.len() / h;
        let scale = (h as f32).sqrt().recip();
        let mut out = vec![vec![0.0f32; seq * h]; nq];
        for s in 0..seq {
            let mut states: Vec<&[f32]> = block_states[s].iter().map(|v| v.as_slice()).collect();
            states.push(&partial[s * h..(s + 1) * h]);
            let normed: Vec<Vec<f32>> = states
                .iter()
                .map(|st| rms_norm(st, None, self.cfg.rms_norm_eps))
                .collect();
            for qi in 0..nq {
                let q = &queries[qi * h..(qi + 1) * h];
                let weights: Vec<f32> = normed
                    .iter()
                    .map(|k| scale * q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>())
                    .collect();
                let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = weights.iter().map(|w| (w - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (ki, k) in normed.iter().enumerate() {
                    let wk = exps[ki] / sum;
                    for (d, a) in out[qi][s * h..(s + 1) * h].iter_mut().enumerate() {
                        *a += wk * k[d];
                    }
                }
            }
        }
        out
    }

    /// One attention sublayer. `q_in`/`k_in`/`v_in` are the (already normed) inputs.
    fn attention(
        &self,
        layer: &Inkling2LoopLayer,
        _layer_idx: usize,
        q_in: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        seq: usize,
        positions: &[u32],
    ) -> Vec<f32> {
        let (nq, nkv, hd) = (layer.nq, layer.nkv, layer.hd);
        let groups = nq / nkv;

        // Q: linear + scaled/L2 clip (+ log scaling on full layers)
        let mut q: Vec<Vec<f32>> = (0..seq)
            .map(|s| {
                linear(
                    &q_in[s * layer.hidden..(s + 1) * layer.hidden],
                    &layer.wq,
                    nq * hd,
                )
            })
            .collect();
        {
            let mut view: Vec<Vec<Vec<f32>>> = q
                .iter()
                .map(|row| row.chunks(hd).map(|c| c.to_vec()).collect())
                .collect();
            scaled_log_l2_clip(
                &mut view,
                &layer.q_scale,
                self.cfg.qk_l2_clip_threshold,
                positions,
                layer.is_full,
                self.cfg.log_scaling_alpha,
                self.cfg.log_scaling_n_floor as f64,
                1e-6,
            );
            q = view
                .into_iter()
                .map(|per_seq| per_seq.into_iter().flatten().collect())
                .collect();
        }

        // K/V: linear + sconv + norm (L2 clip for K, unweighted RMS for V)
        let mut k = vec![0.0f32; seq * nkv * hd];
        let mut v = vec![0.0f32; seq * nkv * hd];
        for s in 0..seq {
            let row = s * layer.hidden;
            let kv_off = s * nkv * hd;
            let kr = linear(&k_in[row..row + layer.hidden], &layer.wk, nkv * hd);
            let vr = linear(&v_in[row..row + layer.hidden], &layer.wv, nkv * hd);
            k[kv_off..kv_off + nkv * hd].copy_from_slice(&kr);
            v[kv_off..kv_off + nkv * hd].copy_from_slice(&vr);
        }
        if layer.is_full {
            k = dynamic_sconv(
                &k,
                k_in,
                &layer.k_conv_static,
                layer.k_conv_proj.as_ref().unwrap(),
                layer.k,
                nkv,
                seq,
                nkv * hd,
                layer.hidden,
            );
            v = dynamic_sconv(
                &v,
                v_in,
                &layer.v_conv_static,
                layer.v_conv_proj.as_ref().unwrap(),
                layer.k,
                nkv,
                seq,
                nkv * hd,
                layer.hidden,
            );
        } else {
            k = static_sconv(&k, &layer.k_conv_static, layer.k, seq, nkv * hd);
            v = static_sconv(&v, &layer.v_conv_static, layer.k, seq, nkv * hd);
        }
        // K: per-head L2 clip; V: unweighted RMS norm
        let mut kh: Vec<Vec<Vec<f32>>> = (0..seq)
            .map(|s| {
                (0..nkv)
                    .map(|h| k[s * nkv * hd + h * hd..s * nkv * hd + (h + 1) * hd].to_vec())
                    .collect()
            })
            .collect();
        for s in 0..seq {
            for h in 0..nkv {
                let vrow = &mut kh[s][h];
                let norm = vrow.iter().map(|x| x * x).sum::<f32>().sqrt();
                let factor = (self.cfg.qk_l2_clip_threshold / (norm + 1e-6)).min(1.0);
                for x in vrow.iter_mut() {
                    *x *= factor;
                }
            }
        }
        let vh: Vec<Vec<Vec<f32>>> = (0..seq)
            .map(|s| {
                (0..nkv)
                    .map(|h| {
                        rms_norm(
                            &v[s * nkv * hd + h * hd..s * nkv * hd + (h + 1) * hd],
                            if layer.is_full {
                                // full layers: v_input_norm has no gamma
                                None
                            } else {
                                None
                            },
                            self.cfg.rms_norm_eps,
                        )
                    })
                    .collect()
            })
            .collect();

        // relative position bias + sinks + causal/sliding attention
        let window = layer.sliding_window;
        let mut attn = vec![0.0f32; seq * nq * hd];
        for s in 0..seq {
            // per-token rel states + sinks
            let h_row = &q_in[s * layer.hidden..(s + 1) * layer.hidden];
            let mut rel: Vec<Vec<f32>> = (0..nq)
                .map(|h| {
                    let mut r = linear(h_row, &layer.wr, nq * layer.d_rel)
                        [h * layer.d_rel..(h + 1) * layer.d_rel]
                        .to_vec();
                    // scale + clip (log scaling on full layers)
                    let pos_scale = if layer.is_full {
                        1.0 + self.cfg.log_scaling_alpha
                            * (((positions[s] as f64 + 1.0) / self.cfg.log_scaling_n_floor as f64)
                                .max(1.0)
                                .ln() as f32)
                    } else {
                        1.0
                    };
                    for (d, rv) in r.iter_mut().enumerate() {
                        *rv *= layer.rel_scale[h * layer.d_rel + d] * pos_scale;
                    }
                    let norm = r.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let factor = (self.cfg.rel_l2_clip_threshold / (norm + 1e-6)).min(1.0);
                    for rv in r.iter_mut() {
                        *rv *= factor;
                    }
                    r
                })
                .collect();
            // rel logits per head: [nq, rel_extent]
            let rel_logits: Vec<Vec<f32>> = rel
                .iter_mut()
                .map(|r| {
                    (0..layer.rel_extent)
                        .map(|e| {
                            r.iter()
                                .zip(&layer.rel_proj[e * layer.d_rel..(e + 1) * layer.d_rel])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                        })
                        .collect()
                })
                .collect();
            let sinks = linear(h_row, &layer.sink, nq);

            for h in 0..nq {
                let qh = &q[s][h * hd..(h + 1) * hd];
                let kv_h = h / groups;
                let scale = (hd as f32).recip();
                let mut best = f32::NEG_INFINITY;
                let mut logits = Vec::with_capacity(s + 1);
                for t in 0..=s {
                    let dist = positions[s] as i64 - positions[t] as i64;
                    if window.is_some() && dist >= window.unwrap() as i64 {
                        logits.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let mut z =
                        scale * qh.iter().zip(&kh[t][kv_h]).map(|(a, b)| a * b).sum::<f32>();
                    if dist >= 0 && (dist as usize) < layer.rel_extent {
                        z += rel_logits[h][dist as usize];
                    }
                    if z > best {
                        best = z;
                    }
                    logits.push(z);
                }
                // sink absorbs probability mass, then is dropped
                let sink_logit = sinks[h];
                if sink_logit > best {
                    best = sink_logit;
                }
                let mut denom = (sink_logit - best).exp();
                let mut exps = Vec::with_capacity(s + 1);
                for z in &logits {
                    let e = if *z == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (*z - best).exp()
                    };
                    denom += e;
                    exps.push(e);
                }
                let inv = 1.0 / denom;
                for (t, e) in exps.iter().enumerate() {
                    let w = e * inv;
                    if w == 0.0 {
                        continue;
                    }
                    for (d, a) in attn[s * nq * hd + h * hd..s * nq * hd + (h + 1) * hd]
                        .iter_mut()
                        .enumerate()
                    {
                        *a += w * vh[t][kv_h][d];
                    }
                }
            }
        }

        // gated output projection
        let mut out = vec![0.0f32; seq * layer.hidden];
        for s in 0..seq {
            let h_row = &q_in[s * layer.hidden..(s + 1) * layer.hidden];
            let flat = &attn[s * nq * hd..(s + 1) * nq * hd];
            let gate = linear(h_row, &layer.o_gate, nq * hd);
            let gated: Vec<f32> = flat
                .iter()
                .zip(&gate)
                .map(|(a, g)| a * sigmoid(*g))
                .collect();
            out[s * layer.hidden..(s + 1) * layer.hidden].copy_from_slice(&linear(
                &gated,
                &layer.wo,
                layer.hidden,
            ));
        }
        out
    }

    /// One decoder layer forward; returns (new_hidden, updated_block_states).
    fn run_layer(
        &self,
        layer_idx: usize,
        hidden: &[f32],
        block_states: &[Vec<Vec<f32>>],
        seq: usize,
        positions: &[u32],
        messages: Option<&[Vec<f32>; 4]>,
    ) -> Result<(Vec<f32>, Vec<Vec<Vec<f32>>>)> {
        let layer = &self.layers[layer_idx];
        let cfg = &self.cfg;
        let eps = cfg.rms_norm_eps;
        let block_residual = hidden.to_vec();

        // attention input: block-state pooling (all layers except layer 0)
        let qkv_in: Vec<Vec<f32>> = if layer_idx == 0 {
            let n = rms_norm(&hidden[..], Some(&layer.q_norm), eps);
            vec![n.clone(), n.clone(), n]
        } else {
            let nq = if layer.is_full { 3 } else { 1 };
            let pooled = self.block_attention(
                &self.attn_queries[layer_idx],
                nq,
                block_states,
                &block_residual,
            );
            if layer.is_full {
                pooled
            } else {
                vec![pooled[0].clone(), pooled[0].clone(), pooled[0].clone()]
            }
        };

        // normed q/k/v inputs (with message gates on loop layers after pass 1)
        let norm_q = rms_norm(&qkv_in[0], Some(&layer.q_norm), eps);
        let norm_k = if layer.is_full {
            rms_norm(&qkv_in[1], Some(&layer.k_norm), eps)
        } else {
            norm_q.clone()
        };
        let norm_v = if layer.is_full {
            // value norm has no gamma
            rms_norm(&qkv_in[2], None, eps)
        } else {
            norm_q.clone()
        };
        let (q_in, k_in, v_in) = if let Some(msgs) = messages {
            (
                layer.gates.as_ref().unwrap()[0].apply(&norm_q, &msgs[0], eps),
                layer.gates.as_ref().unwrap()[1].apply(&norm_k, &msgs[1], eps),
                layer.gates.as_ref().unwrap()[2].apply(&norm_v, &msgs[2], eps),
            )
        } else {
            (norm_q, norm_k, norm_v)
        };

        let mut h = self.attention(layer, layer_idx, &q_in, &k_in, &v_in, seq, positions);
        h = static_sconv(&h, &layer.attn_sconv, layer.k, seq, layer.hidden);

        // block-boundary bookkeeping: the boundary appends the pre-layer hidden state
        // to EVERY position's own stack, and skips the residual add of the attention
        // output onto it
        let is_boundary = layer_idx % cfg.n_layers_per_attn_res_block == 0;
        let hdim = layer.hidden;
        let mut new_block_states: Vec<Vec<Vec<f32>>> = block_states.to_vec();
        let mut residual: Vec<f32>;
        let mut h_f32: Vec<f32>;
        if is_boundary {
            for s in 0..seq {
                new_block_states[s].push(block_residual[s * hdim..(s + 1) * hdim].to_vec());
            }
            residual = h.clone();
            h_f32 = h;
        } else {
            residual = block_residual.clone();
            h_f32 = h.clone();
            for i in 0..residual.len() {
                h_f32[i] += residual[i];
            }
            residual = h_f32.clone();
        }

        // mlp input pooling
        let mut mlp_in =
            self.block_attention(&self.mlp_queries[layer_idx], 1, &new_block_states, &h_f32)[0]
                .clone();
        if let Some(msgs) = messages {
            mlp_in = layer.gates.as_ref().unwrap()[3].apply(&mlp_in, &msgs[3], eps);
        }
        let mlp_normed = rms_norm(&mlp_in, Some(&layer.mlp_norm), eps);
        let mut mlp_out = vec![0.0f32; mlp_normed.len()];
        for s in 0..seq {
            let mo = self.moe(
                layer_idx,
                &mlp_normed[s * layer.hidden..(s + 1) * layer.hidden],
            );
            mlp_out[s * layer.hidden..(s + 1) * layer.hidden].copy_from_slice(&mo);
        }
        mlp_out = static_sconv(&mlp_out, &layer.mlp_sconv, layer.k, seq, layer.hidden);
        for i in 0..residual.len() {
            mlp_out[i] += residual[i];
        }
        Ok((mlp_out, new_block_states))
    }
}

impl Model for Inkling2Loop {
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

impl CausalLm for Inkling2Loop {
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
        let seq = ids_f32.len();
        let ids: Vec<u32> = ids_f32.iter().map(|&t| t as u32).collect();
        let pos_v: Vec<u32> = positions
            .to_vec_f32()
            .map(|v| v.into_iter().map(|p| p as u32).collect())
            .unwrap_or_else(|_| (0..seq as u32).collect());
        let hidden = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;

        // embed + embed norm
        let mut x = vec![0.0f32; seq * hidden];
        for (s, &id) in ids.iter().enumerate() {
            let row = id as usize * hidden;
            if row + hidden > self.embed.len() {
                return Err(grim_core::error::Error::Backend(format!(
                    "inkling2_loop: token id {id} out of vocab range"
                )));
            }
            x[s * hidden..(s + 1) * hidden].copy_from_slice(&self.embed[row..row + hidden]);
        }
        for v in x.chunks_mut(hidden) {
            let nv = rms_norm(v, Some(&self.embed_norm), eps);
            v.copy_from_slice(&nv);
        }

        let loop_start = self.cfg.loop_layer_index_start;
        let loop_end = self.cfg.loop_layer_index_end;

        // per-position stacks of boundary hidden states: block_states[s] grows at
        // every block boundary
        let mut block_states: Vec<Vec<Vec<f32>>> = vec![Vec::new(); seq];
        for i in 0..loop_start {
            let (nx, nbs) = self.run_layer(i, &x, &block_states, seq, &pos_v, None)?;
            x = nx;
            block_states = nbs;
        }
        let pre_hidden = x.clone();
        let pre_blocks = block_states.clone();
        let n_pre_blocks = pre_blocks[0].len();

        // loop pass 1
        for i in loop_start..loop_end {
            let (nx, nbs) = self.run_layer(i, &x, &block_states, seq, &pos_v, None)?;
            x = nx;
            block_states = nbs;
        }

        // passes 2..N: restart from the pre-loop states, message-gated. The message
        // keys are the previous pass's own appended block states (dropping the first,
        // per the reference `num_pre_loop_blocks + 1:` slice) + the previous hidden.
        for _pass in 1..self.cfg.loop_num_passes {
            let prev_blocks: Vec<Vec<Vec<f32>>> = block_states
                .iter()
                .map(|bs| bs[n_pre_blocks + 1..].to_vec())
                .collect();
            let prev_hidden = x.clone();
            let mut cur_blocks = pre_blocks.clone();
            let mut cur = pre_hidden.clone();
            for (j, li) in (loop_start..loop_end).enumerate() {
                let msgs_raw =
                    self.block_attention(&self.message_queries[j], 4, &prev_blocks, &prev_hidden);
                let msgs = [
                    msgs_raw[0].clone(),
                    msgs_raw[1].clone(),
                    msgs_raw[2].clone(),
                    msgs_raw[3].clone(),
                ];
                let (nc, nbs) = self.run_layer(li, &cur, &cur_blocks, seq, &pos_v, Some(&msgs))?;
                cur = nc;
                cur_blocks = nbs;
            }
            x = cur;
            block_states = cur_blocks;
        }

        // post-loop layers
        for i in loop_end..self.cfg.num_hidden_layers {
            let (nx, nbs) = self.run_layer(i, &x, &block_states, seq, &pos_v, None)?;
            x = nx;
            block_states = nbs;
        }

        // final per-position pooling + norm + unembed + softcap (logits for ALL
        // positions, [seq, unpadded_vocab])
        let pooled = self.block_attention(&self.output_query, 1, &block_states, &x)[0].clone();
        let mut logits = Vec::with_capacity(seq * self.cfg.unpadded_vocab_size);
        for s in 0..seq {
            let normed = rms_norm(
                &pooled[s * hidden..(s + 1) * hidden],
                Some(&self.final_norm),
                eps,
            );
            let row = linear(&normed, &self.unembed, self.cfg.unpadded_vocab_size);
            logits.extend(
                row.into_iter()
                    .map(|l| softcap(l, self.cfg.final_logit_softcapping)),
            );
        }

        session.advance_pos(seq);
        Ok(grim_backend_cpu::cpu_tensor(
            logits,
            Shape::new(vec![seq, self.cfg.unpadded_vocab_size]),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const INKLING2_CONFIG: &str = r#"{
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "text_config": {
            "model_type": "inkling2_loop_text",
            "hidden_size": 2304,
            "num_hidden_layers": 24,
            "vocab_size": 201024,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "d_rel": 16,
            "rel_extent": 1024,
            "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
            "sconv_kernel_size": 4,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "intermediate_size": 1152,
            "route_scale": 8.0,
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64}
        }
    }"#;

    #[test]
    fn parses_inkling2_loop_config() {
        let v: serde_json::Value = serde_json::from_str(INKLING2_CONFIG).unwrap();
        let cfg = Inkling2LoopConfig::from_hf(&v);
        assert_eq!(cfg.name(), "inkling2_loop");
        assert_eq!(cfg.hidden_size, 2304);
        assert_eq!(cfg.loop_layer_index_start, 6);
        assert_eq!(cfg.loop_num_passes, 3);
        assert_eq!(cfg.layer_is_full.iter().filter(|f| **f).count(), 4);
        assert!(cfg.layer_is_full[5]);
        assert!(!cfg.layer_is_full[4]);
    }

    #[test]
    fn dispatches_inkling2_loop_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("inkling2_loop"),
            ModelArchitecture::Inkling2Loop
        );
        assert_eq!(ModelArchitecture::Inkling2Loop.as_str(), "inkling2_loop");
    }

    // ---- synthetic forward smoke test ----

    use grim_tensor::provider::{TensorMeta, TensorProvider};
    use grim_tensor::{DType as TDType, QuantProvenance};
    use std::collections::HashMap;

    struct MemProvider {
        tensors: HashMap<String, RawTensor>,
    }
    impl TensorProvider for MemProvider {
        fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing {name}")))
        }
        fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
            let t = self.get(name)?;
            Ok(TensorMeta {
                dtype: t.dtype,
                provenance: t.provenance,
                shape: t.shape,
                fusion_mask: 0,
            })
        }
        fn tensor_names(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }
    }

    fn raw_f32(shape: &[usize], seed: u32) -> RawTensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let v = (((s >> 8) as f32 / 8_388_608.0) - 1.0) * 0.05;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        RawTensor {
            bytes,
            shape: shape.to_vec(),
            dtype: TDType::F32,
            provenance: QuantProvenance::GrimNative,
        }
    }

    fn raw_u8(shape: &[usize], seed: u32) -> RawTensor {
        let n: usize = shape.iter().product();
        let mut s = seed | 1;
        let mut bytes = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            bytes.push((s >> 24) as u8);
        }
        // packed fp4: mask so e2m1 codes stay small (|v| <= 6)
        for b in bytes.iter_mut() {
            *b &= 0x07;
        }
        RawTensor {
            bytes,
            shape: shape.to_vec(),
            dtype: TDType::U8,
            provenance: QuantProvenance::GrimNative,
        }
    }

    /// NVFP4 scale plane with a fixed tiny E4M3 scale byte (2^-6).
    fn raw_scale(shape: &[usize]) -> RawTensor {
        let n: usize = shape.iter().product();
        RawTensor {
            bytes: vec![0x00; n],
            shape: shape.to_vec(),
            dtype: TDType::U8,
            provenance: QuantProvenance::GrimNative,
        }
    }

    fn small_cfg() -> Inkling2LoopConfig {
        let mut cfg = Inkling2LoopConfig::default();
        cfg.vocab_size = 64;
        cfg.unpadded_vocab_size = 64;
        cfg.hidden_size = 16;
        cfg.num_hidden_layers = 6;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 4;
        cfg.swa_num_attention_heads = 4;
        cfg.swa_num_key_value_heads = 2;
        cfg.swa_head_dim = 4;
        cfg.sliding_window_size = 4;
        cfg.d_rel = 2;
        cfg.rel_extent = 8;
        cfg.log_scaling_n_floor = 8;
        cfg.sconv_kernel_size = 2;
        cfg.moe_intermediate_size = 16;
        cfg.n_routed_experts = 2;
        cfg.num_experts_per_tok = 2;
        cfg.route_scale = 1.0;
        cfg.n_layers_per_attn_res_block = 2;
        cfg.loop_layer_index_start = 2;
        cfg.loop_layer_index_end = 4;
        cfg.loop_num_passes = 2;
        cfg.loop_gate_rank = 4;
        cfg.layer_is_full = vec![false, true, false, true, false, true];
        cfg
    }

    fn build_provider(cfg: &Inkling2LoopConfig) -> MemProvider {
        let h = cfg.hidden_size;
        let mut t: HashMap<String, RawTensor> = HashMap::new();
        let mut ins = |name: String, r: RawTensor| {
            t.insert(name, r);
        };
        ins(
            "model.llm.embed.weight".into(),
            raw_f32(&[cfg.vocab_size, h], 1),
        );
        ins("model.llm.embed_norm.weight".into(), raw_f32(&[h], 2));
        ins("model.llm.norm.weight".into(), raw_f32(&[h], 3));
        ins(
            "model.llm.unembed.weight".into(),
            raw_f32(&[cfg.unpadded_vocab_size, h], 4),
        );
        ins("model.llm.attn_res.output_query".into(), raw_f32(&[h], 5));
        ins(
            "model.llm.middle_cycle_cross_pass_queries".into(),
            raw_f32(&[4, h], 6),
        );
        ins(
            "model.llm.middle_cycle_cross_pass_kv_queries".into(),
            raw_f32(&[4, h], 7),
        );
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let k = cfg.sconv_kernel_size;
        let e = cfg.n_routed_experts;
        let inter = cfg.moe_intermediate_size;
        for i in 0..cfg.num_hidden_layers {
            let full = cfg.layer_is_full[i];
            let (lq, lkv, lhd) = if full {
                (nq, nkv, hd)
            } else {
                (
                    cfg.swa_num_attention_heads,
                    cfg.swa_num_key_value_heads,
                    cfg.swa_head_dim,
                )
            };
            let conv_dim = lkv * lhd;
            let p = move |s: u32, shape: &[usize]| raw_f32(shape, s + i as u32 * 100);
            let base = format!("model.llm.layers.{i}");
            ins(format!("{base}.attn.wq_du.weight"), p(10, &[lq * lhd, h]));
            ins(format!("{base}.attn.wk_dv.weight"), p(11, &[conv_dim, h]));
            ins(format!("{base}.attn.wv_dv.weight"), p(12, &[conv_dim, h]));
            ins(
                format!("{base}.attn.gate_attn.weight"),
                p(13, &[lq * lhd, h]),
            );
            ins(format!("{base}.attn.wo_ud.weight"), p(14, &[h, lq * lhd]));
            ins(
                format!("{base}.attn.attn_sink_proj.weight"),
                p(15, &[lq, h]),
            );
            ins(format!("{base}.attn.q_l2_clip.scale"), p(16, &[lq, lhd]));
            ins(
                format!("{base}.attn.rel_l2_clip.scale"),
                p(17, &[lq, cfg.d_rel]),
            );
            ins(
                format!("{base}.attn.rel_logits_proj.proj"),
                p(18, &[cfg.d_rel, cfg.rel_extent]),
            );
            ins(
                format!("{base}.attn.wr_du.weight"),
                p(19, &[lq * cfg.d_rel, h]),
            );
            if full {
                ins(
                    format!("{base}.attn.k_sconv.static_w"),
                    p(20, &[k, conv_dim]),
                );
                ins(
                    format!("{base}.attn.k_sconv.weight_proj"),
                    p(21, &[lkv * k, h]),
                );
                ins(
                    format!("{base}.attn.v_sconv.static_w"),
                    p(22, &[k, conv_dim]),
                );
                ins(
                    format!("{base}.attn.v_sconv.weight_proj"),
                    p(23, &[lkv * k, h]),
                );
                ins(format!("{base}.q_input_norm.weight"), p(24, &[h]));
                ins(format!("{base}.k_input_norm.weight"), p(25, &[h]));
            } else {
                ins(
                    format!("{base}.attn.k_sconv.weight"),
                    p(20, &[conv_dim, 1, k]),
                );
                ins(
                    format!("{base}.attn.v_sconv.weight"),
                    p(22, &[conv_dim, 1, k]),
                );
                ins(format!("{base}.attn_norm.weight"), p(24, &[h]));
            }
            ins(format!("{base}.attn_sconv.weight"), p(26, &[h, 1, k]));
            ins(format!("{base}.mlp_sconv.weight"), p(27, &[h, 1, k]));
            ins(format!("{base}.mlp_norm.weight"), p(28, &[h]));
            ins(format!("{base}.mlp.gate.weight"), p(29, &[e, h]));
            ins(
                format!("{base}.mlp.experts.w13_weight"),
                raw_u8(&[e, 2 * inter, h / 2], 30),
            );
            ins(
                format!("{base}.mlp.experts.w13_weight_scale"),
                raw_scale(&[e, 2 * inter, h / 16]),
            );
            ins(
                format!("{base}.mlp.experts.w2_weight"),
                raw_u8(&[e, h, inter / 2], 32),
            );
            ins(
                format!("{base}.mlp.experts.w2_weight_scale"),
                raw_scale(&[e, h, inter / 16]),
            );
            ins(
                format!("{base}.mlp.experts.w13_weight_outer_scale_denominator"),
                RawTensor {
                    bytes: 1.0f32.to_le_bytes().to_vec(),
                    shape: vec![1],
                    dtype: TDType::F32,
                    provenance: QuantProvenance::GrimNative,
                },
            );
            let qshape = if full { vec![3, h] } else { vec![h] };
            ins(
                format!("model.llm.attn_res.attn_queries.{i}"),
                p(34, &qshape),
            );
            ins(format!("model.llm.attn_res.mlp_queries.{i}"), p(35, &[h]));
            if cfg.loop_layer_index_start <= i && i < cfg.loop_layer_index_end {
                for which in ["attn", "k", "v", "mlp"] {
                    ins(
                        format!("{base}.controllers.{which}.down_weight"),
                        p(40, &[cfg.loop_gate_rank, 2 * h]),
                    );
                    ins(
                        format!("{base}.controllers.{which}.forget_weight"),
                        p(41, &[h, cfg.loop_gate_rank]),
                    );
                    ins(
                        format!("{base}.controllers.{which}.gain_weight"),
                        p(42, &[h, cfg.loop_gate_rank]),
                    );
                }
            }
        }
        drop(ins);
        MemProvider { tensors: t }
    }

    #[test]
    fn synthetic_forward_smoke() {
        let cfg = small_cfg();
        let provider = build_provider(&cfg);
        let ws = WeightSource::root(&provider, Device::Cpu);
        let model = Inkling2Loop::load(Device::Cpu, &ws, cfg).expect("load");
        let ids: Vec<f32> = (3u32..9).map(|t| t as f32).collect();
        let input = grim_backend_cpu::cpu_tensor(ids, Shape::new(vec![6]));
        let pos = grim_backend_cpu::cpu_tensor(
            (0u32..6).map(|p| p as f32).collect::<Vec<_>>(),
            Shape::new(vec![6]),
        );
        let mut sess = model.new_session();
        let logits = model
            .forward(sess.as_mut(), &input, &pos, &[])
            .expect("forward");
        let out = logits.to_vec_f32().expect("logits f32");
        assert_eq!(out.len(), 6 * 64);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite logits");
        // deterministic: same input -> same output
        let mut sess2 = model.new_session();
        let out2 = model
            .forward(sess2.as_mut(), &input, &pos, &[])
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(out, out2);
    }
}
