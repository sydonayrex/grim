//! Google Gemma-4 text stack (E2B / E4B / 26B-A4B).
//!
//! Reference: `old/repo/llama.cpp-master/src/models/gemma4.cpp` (the graph) and
//! `old/repo/llama.cpp-master/conversion/gemma.py` (the HF -> GGUF spec).
//!
//! Gemma 4 differs from Gemma 2/3 in five ways that the forward pass has to
//! get right:
//!
//! 1. **Two attention kinds per model.** A block is either sliding-window
//!    attention (window 512, head_dim 256, rope theta 10k) or full attention
//!    (head_dim 512, "proportional" rope theta 1M). The kinds are interleaved
//!    (`layer_types` / `attention.sliding_window_pattern`).
//! 2. **"Proportional" RoPE on the full-attention blocks.** The HF->GGUF
//!    converter emits a `rope_freqs` vector that pins the unrotated
//!    dimensions to identity with a 1e30 factor
//!    (`conversion/gemma.py:752-762`); ggml then applies it as a NeoX
//!    half-split with `n_dims = head_dim` (`ggml/src/ggml-cpu/ops.cpp:5979`,
//!    `:6063`). Only `head_dim * partial_rotary_factor / 2` pairs move.
//! 3. **Per-layer ("per layer input") embeddings.** Every block mixes in a
//!    256-wide slice of a per-layer token embedding table, projected through
//!    the block's own gate and projection matrices
//!    (`src/models/gemma4.cpp:367-389`).
//! 4. **Shared KV layers.** The tail `num_kv_shared_layers` blocks do not own
//!    a KV cache: they attend over the K/V of an earlier block
//!    (`src/llama-model.cpp:2726-2733`).
//! 5. **No attention scaling.** `f_attention_scale = 1.0`
//!    (`src/models/gemma4.cpp:11`) - the softmax sees the raw `q . k`.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Embedding, Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, RopeConfig, Shape, Tensor};

/// Which attention a Gemma-4 block uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Gemma4LayerType {
    /// Sliding-window attention: window 512, head_dim 256, plain rope theta 10k.
    Sliding,
    /// Full attention: head_dim 512, "proportional" rope theta 1M.
    Full,
}

// Config

/// Configuration for the Google Gemma-4 text stack.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Query head count. Gemma 4 keeps the same head count for both attention kinds.
    pub num_attention_heads: usize,
    /// Key/value head count of the sliding-window blocks.
    pub num_key_value_heads: usize,
    /// Key/value head count of the full-attention blocks.
    pub num_key_value_heads_full: usize,
    /// Head width of the sliding-window blocks.
    pub head_dim: usize,
    /// Head width of the full-attention blocks.
    pub head_dim_full: usize,
    /// Feed-forward width. The loader overrides it per block from the tensor
    /// shapes, so the "double wide MLP" variant needs no separate flag.
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    /// RoPE base of the sliding-window blocks.
    pub rope_theta: f32,
    /// RoPE base of the full-attention blocks.
    pub rope_theta_full: f32,
    /// Number of rotary pairs the full-attention blocks actually rotate.
    ///
    /// The HF config states this as a fraction of the head
    /// (`partial_rotary_factor`); the GGUF stores the resulting `rope_freqs`
    /// vector instead, whose leading entries are the live pairs. The loader
    /// resolves both into this one count.
    pub n_rot_pairs_full: usize,
    /// Sliding window of the sliding-window blocks.
    pub sliding_window: usize,
    pub final_logit_softcapping: Option<f32>,
    pub max_position_embeddings: usize,
    /// Width of the per-layer ("per layer input") embedding. 0 = the model has none.
    pub n_embd_per_layer: usize,
    /// Number of tail blocks that reuse an earlier block's KV cache.
    pub num_kv_shared_layers: usize,
    /// Per-block attention kind.
    pub layer_types: Vec<Gemma4LayerType>,
    /// Softmax scale of the attention. Gemma 4 uses 1.0 (no `1/sqrt(d)`).
    pub attention_scale: f32,
}

impl Default for Gemma4Config {
    fn default() -> Self {
        Self {
            vocab_size: 262144,
            hidden_size: 2560,
            num_hidden_layers: 42,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            num_key_value_heads_full: 2,
            head_dim: 256,
            head_dim_full: 512,
            intermediate_size: 10240,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_theta_full: 1000000.0,
            n_rot_pairs_full: 64,
            sliding_window: 512,
            final_logit_softcapping: Some(30.0),
            max_position_embeddings: 131072,
            n_embd_per_layer: 256,
            num_kv_shared_layers: 18,
            layer_types: Vec::new(),
            attention_scale: 1.0,
        }
    }
}

impl ModelConfig for Gemma4Config {
    fn name(&self) -> &str {
        "gemma4"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Gemma4Config {
    /// The first block index that owns a KV cache. Blocks at or after this index
    /// reuse the K/V of an earlier block.
    ///
    /// `n_layer_kv_from_start = n_layer_all - n_kv_shared_layers`
    /// (`src/models/gemma4.cpp:10`).
    pub fn n_layer_kv_from_start(&self) -> usize {
        self.num_hidden_layers.saturating_sub(self.num_kv_shared_layers)
    }

    /// The attention kind of block `il`. Defaults to sliding when the checkpoint
    /// did not declare a pattern.
    pub fn layer_type(&self, il: usize) -> Gemma4LayerType {
        self.layer_types
            .get(il)
            .copied()
            .unwrap_or(Gemma4LayerType::Sliding)
    }

    /// Head width of block `il`.
    pub fn layer_head_dim(&self, il: usize) -> usize {
        match self.layer_type(il) {
            Gemma4LayerType::Sliding => self.head_dim,
            Gemma4LayerType::Full => self.head_dim_full,
        }
    }

    /// Key/value head count of block `il`.
    pub fn layer_num_kv_heads(&self, il: usize) -> usize {
        match self.layer_type(il) {
            Gemma4LayerType::Sliding => self.num_key_value_heads,
            Gemma4LayerType::Full => self.num_key_value_heads_full,
        }
    }

    /// The block whose K/V cache block `il` reads, or `None` when `il` owns its own.
    ///
    /// The reference maps a shared block onto
    /// `n_layer_kv_from_start - (is_swa ? 2 : 1)` (`src/llama-model.cpp:2730`),
    /// i.e. the last sliding block and the last full block below the split.
    pub fn kv_source(&self, il: usize) -> Option<usize> {
        let start = self.n_layer_kv_from_start();
        if il < start || start < 2 {
            return None;
        }
        Some(start - match self.layer_type(il) {
            Gemma4LayerType::Sliding => 2,
            Gemma4LayerType::Full => 1,
        })
    }

    /// Number of rotary pairs of the full-attention "proportional" rope.
    ///
    /// The converter builds `n_rot_full = head_dim_full * partial_rotary_factor / 2`
    /// live pairs and pins the rest (`conversion/gemma.py:757-760`).
    pub fn n_rot_pairs(&self) -> usize {
        self.n_rot_pairs_full.min(self.head_dim_full / 2)
    }
}

// RoPE

/// Gemma-4's "proportional" RoPE, used by the full-attention blocks.
///
/// ggml's NeoX mode pairs element `j` of a head with element `j + n_dims/2` and
/// rotates them by `pos * freq_base^(-2j/n_dims)`, where `n_dims` is the FULL
/// head width (`ggml/src/ggml-cpu/ops.cpp:6063`, `:5979`). The converter's
/// `rope_freqs` vector pins every pair beyond `partial_rotary_factor` to the
/// identity with a 1e30 divisor, so only the first `n_pairs` pairs move.
///
/// This runs on the host: the device rope kernels derive the pair count and the
/// partner offset from a single `rotary_dim`, which cannot express "rotate 64
/// pairs of a 512-wide head against the +256 half".
#[derive(Debug, Clone, Copy)]
pub struct Gemma4ProportionalRope {
    pub base: f32,
    pub head_dim: usize,
    pub n_pairs: usize,
}

impl Gemma4ProportionalRope {
    /// Rotate `x` (`[steps, num_heads * head_dim]`) on the host.
    pub fn apply(&self, x: &Tensor, num_heads: usize, positions: &[u32]) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (steps, width) = (dims[0], dims[1]);
        if width != num_heads * self.head_dim {
            return Err(grim_core::error::Error::Shape(format!(
                "gemma4 proportional rope: expected width {} ({} heads x {}), got {}",
                num_heads * self.head_dim,
                num_heads,
                self.head_dim,
                width
            )));
        }
        let half = self.head_dim / 2;
        let mut data = x.to_vec_f32()?;
        for t in 0..steps {
            let pos = positions.get(t).copied().unwrap_or(t as u32) as f32;
            for h in 0..num_heads {
                let row = t * width + h * self.head_dim;
                for j in 0..self.n_pairs {
                    let theta = pos * self.base.powf(-2.0 * j as f32 / self.head_dim as f32);
                    let (sin_t, cos_t) = theta.sin_cos();
                    let a = data[row + j];
                    let b = data[row + half + j];
                    data[row + j] = a * cos_t - b * sin_t;
                    data[row + half + j] = a * sin_t + b * cos_t;
                }
            }
        }
        Ok(cpu_tensor(data, x.shape().clone()))
    }
}

// Feed forward

/// Gemma-4's gated MLP: `down(gelu_tanh(gate) * up)`.
///
/// `LLM_FFN_GELU` + `LLM_FFN_PAR` in the reference
/// (`src/models/gemma4.cpp:351-356`) is ggml's `geglu_split(gate, up)`, and
/// ggml's GELU op is the tanh approximation - the same
/// `gelu_pytorch_tanh` the config names.
pub struct Gemma4Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Gemma4Mlp {
    pub fn load(ws: &WeightSource<'_>, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let gate_proj = Linear::load_shape(&ws.scoped("gate_proj"), [in_dim, hidden_dim])?;
        let up_proj = Linear::load_shape(&ws.scoped("up_proj"), [in_dim, hidden_dim])?;
        let down_proj = Linear::load_shape(&ws.scoped("down_proj"), [hidden_dim, in_dim])?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        tsig: &dyn Fn(&str, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        tsig("mlp_gate", &gate)?;
        let up = self.up_proj.forward(x)?;
        tsig("mlp_up", &up)?;
        let act = grim_nn::modules::gelu_tanh_mul_on_device(&gate, &up)?;
        tsig("mlp_act", &act)?;
        let out = self.down_proj.forward(&act)?;
        tsig("mlp_down", &out)?;
        Ok(out)
    }
}

/// An RMS norm that names itself when it fails.
///
/// A Gemma-4 block applies eight different norms; a bare
/// "rms_norm: weight size mismatch" does not say which one, and the shapes
/// are the whole diagnosis.
fn rms_named(x: &Tensor, norm: &RmsNorm, what: &str) -> Result<Tensor> {
    norm.forward(x).map_err(|e| {
        grim_core::error::Error::Backend(format!(
            "gemma4 {what}: {e} (input {:?}, weight {:?})",
            x.shape().dims(),
            norm.weight.shape().dims()
        ))
    })
}

// Block

/// One Gemma-4 decoder block.
pub struct Gemma4Block {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    /// `input_layernorm` - the pre-attention norm.
    pub attn_norm: RmsNorm,
    /// `self_attn.q_norm` - per-head RMS norm of the query.
    pub q_norm: RmsNorm,
    /// `self_attn.k_norm` - per-head RMS norm of the key.
    pub k_norm: RmsNorm,
    /// Per-head RMS norm of the value, WITHOUT a weight
    /// (`ggml_rms_norm(ctx0, Vcur, eps)`, `src/models/gemma4.cpp:256`).
    pub v_norm: RmsNorm,
    /// `post_attention_layernorm` - the post-attention norm.
    pub attn_post_norm: RmsNorm,
    /// `pre_feedforward_layernorm` - the pre-FFN norm.
    pub ffn_norm: RmsNorm,
    /// `post_feedforward_layernorm` - the post-FFN norm.
    pub ffn_post_norm: RmsNorm,
    /// `post_per_layer_input_norm` - the norm of the per-layer injection.
    pub per_layer_post_norm: Option<RmsNorm>,
    pub mlp: Gemma4Mlp,
    /// `per_layer_input_gate` - projects the residual stream down to the
    /// per-layer embedding width.
    pub per_layer_inp_gate: Option<Linear>,
    /// `per_layer_projection` - projects the per-layer embedding back up.
    pub per_layer_proj: Option<Linear>,
    /// `layer_scalar` - the block's output scale.
    pub out_scale: Option<f32>,
    pub layer_type: Gemma4LayerType,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// The sliding-window rope (NeoX half-split over the whole head).
    pub rope_swa: Option<Rope>,
    /// The "proportional" rope of the full-attention blocks.
    pub rope_full: Option<Gemma4ProportionalRope>,
    /// The attention window: `Some(512)` for sliding blocks, `None` for full ones.
    pub window: Option<usize>,
    /// Softmax scale. Gemma 4 uses 1.0 - no `1/sqrt(head_dim)`.
    pub attention_scale: f32,
}

/// The K/V a block produced, so a later block can reuse it.
#[derive(Clone)]
pub struct Gemma4BlockKv {
    /// Post-RoPE keys, `[tokens, num_kv_heads * head_dim]`.
    pub k: Vec<f32>,
    /// Raw values, `[tokens, num_kv_heads * head_dim]`.
    pub v: Vec<f32>,
}

impl Gemma4Block {
    /// Load one block. `intermediate_size` is taken from the checkpoint's own
    /// tensor shape so the "double wide MLP" variant needs no flag.
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &Gemma4Config,
        il: usize,
        device: &Device,
    ) -> Result<Self> {
        let layer_type = cfg.layer_type(il);
        let head_dim = cfg.layer_head_dim(il);
        let num_kv_heads = cfg.layer_num_kv_heads(il);
        let num_heads = cfg.num_attention_heads;

        let attn_ws = ws.scoped("self_attn");
        // The head count and the key/value head count are read back from the
        // checkpoint: a block's own q/k widths are the authority, not the config.
        let q_dim = attn_ws
            .scoped("q_proj")
            .meta("weight")
            .and_then(|m| m.shape.first().copied())
            .unwrap_or(num_heads * head_dim);
        let kv_dim = attn_ws
            .scoped("k_proj")
            .meta("weight")
            .and_then(|m| m.shape.first().copied())
            .unwrap_or(num_kv_heads * head_dim);
        let num_heads = if head_dim > 0 { q_dim / head_dim } else { num_heads };
        let num_kv_heads = if head_dim > 0 { kv_dim / head_dim } else { num_kv_heads };

        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let q_norm = RmsNorm::load(&attn_ws.scoped("q_norm"), head_dim, cfg.rms_norm_eps)?;
        let k_norm = RmsNorm::load(&attn_ws.scoped("k_norm"), head_dim, cfg.rms_norm_eps)?;
        // The value norm carries no weight; a ones-weight RMS norm is the same
        // computation and keeps the device kernel in play.
        let v_norm = RmsNorm::new(
            grim_nn::modules::move_to_device(
                &cpu_tensor(vec![1.0f32; head_dim], Shape::new(vec![head_dim])),
                device,
            )?,
            cfg.rms_norm_eps,
        );
        let attn_post_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let ffn_norm = RmsNorm::load(
            &ws.scoped("pre_feedforward_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let ffn_post_norm = RmsNorm::load(
            &ws.scoped("post_feedforward_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        // The feed-forward width comes from the checkpoint: the tail blocks of
        // a "double wide MLP" model are twice as wide as the head blocks.
        let mlp_ws = ws.scoped("mlp");
        let intermediate_size = mlp_ws
            .scoped("gate_proj")
            .meta("weight")
            .and_then(|m| m.shape.first().copied())
            .unwrap_or(cfg.intermediate_size);
        let mlp = Gemma4Mlp::load(&mlp_ws, cfg.hidden_size, intermediate_size)?;

        let per_layer_post_norm = if cfg.n_embd_per_layer > 0 {
            Some(RmsNorm::load(
                &ws.scoped("post_per_layer_input_norm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?)
        } else {
            None
        };
        let per_layer_inp_gate = if cfg.n_embd_per_layer > 0 {
            Some(Linear::load_shape(
                &ws.scoped("per_layer_input_gate"),
                [cfg.hidden_size, cfg.n_embd_per_layer],
            )?)
        } else {
            None
        };
        let per_layer_proj = if cfg.n_embd_per_layer > 0 {
            Some(Linear::load_shape(
                &ws.scoped("per_layer_projection"),
                [cfg.n_embd_per_layer, cfg.hidden_size],
            )?)
        } else {
            None
        };

        // `layer_scalar` is a single-element tensor; the model multiplies the
        // block output by it.
        //
        // TWO names: the GGUF conversion appends ".weight" to the checkpoint's
        // `layer_scalar` (`conversion/gemma.py: the filter_tensors rule`), so
        // the GGUF carries `layer_scalar.weight` while the safetensors
        // checkpoint still carries the bare `layer_scalar`. Asking for only the
        // first finds nothing in the safetensors file and the block silently
        // loses its output scale.
        let out_scale = match ws.meta("layer_scalar.weight").or_else(|| ws.meta("layer_scalar")) {
            Some(_) => {
                let t = ws
                    .get_unconstrained("layer_scalar.weight")
                    .or_else(|_| ws.get_unconstrained("layer_scalar"))?;
                t.to_vec_f32()?.first().copied()
            }
            None => None,
        };

        // The sliding rope is a plain NeoX half-split over the whole head. The
        // full-attention rope is the "proportional" one.
        let (rope_swa, rope_full, window) = match layer_type {
            Gemma4LayerType::Sliding => (
                Some(Rope::from_config(RopeConfig {
                    dim: head_dim,
                    base: cfg.rope_theta,
                    rotary_dim: head_dim,
                    yarn: None,
                    // NeoX half-split, matching the reference's rope type
                    // (`src/llama-model.cpp:3091`).
                    interleaved: false,
                })),
                None,
                Some(cfg.sliding_window),
            ),
            Gemma4LayerType::Full => (
                None,
                Some(Gemma4ProportionalRope {
                    base: cfg.rope_theta_full,
                    head_dim,
                    n_pairs: cfg.n_rot_pairs(),
                }),
                None,
            ),
        };

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            attn_norm,
            q_norm,
            k_norm,
            v_norm,
            attn_post_norm,
            ffn_norm,
            ffn_post_norm,
            per_layer_post_norm,
            mlp,
            per_layer_inp_gate,
            per_layer_proj,
            out_scale,
            layer_type,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_swa,
            rope_full,
            window,
            attention_scale: cfg.attention_scale,
        })
    }

    /// Rotate `x` (`[steps, heads * head_dim]`) with this block's rope.
    fn rope(&self, x: &Tensor, num_heads: usize, positions: &[u32]) -> Result<Tensor> {
        if let Some(rope) = &self.rope_swa {
            crate::shared_attention::rope_2d_on_device(rope, x, num_heads, positions)
        } else if let Some(rope) = &self.rope_full {
            rope.apply(x, num_heads, positions)
        } else {
            Ok(x.clone())
        }
    }

    /// Per-head RMS norm: `x` is `[tokens, heads * head_dim]`, the norm covers
    /// one head's `head_dim` columns.
    fn norm_heads(x: &Tensor, heads: usize, head_dim: usize, norm: &RmsNorm) -> Result<Tensor> {
        let steps = x.shape().dims()[0];
        let flat = crate::block::reshaped_view(x, &Shape::new(vec![steps * heads, head_dim]))?;
        let normed = rms_named(&flat, norm, "head_norm")?;
        crate::block::reshaped_view(&normed, x.shape())
    }

    /// Forward a block that owns its KV cache.
    ///
    /// `per_layer` carries this block's slice of the per-layer embedding table
    /// (`[tokens, n_embd_per_layer]`, host-resident). `kv_prev` is the
    /// history of the earlier tokens: the decode path feeds ONE token at a
    /// time and the session carries the rest, exactly like every other
    /// family in this crate.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        per_layer: Option<&[f32]>,
        kv_prev: Option<&Gemma4BlockKv>,
        tsig: &dyn Fn(&str, &Tensor) -> Result<()>,
    ) -> Result<(Tensor, Gemma4BlockKv)> {
        let steps = x.shape().dims()[0];
        let device = x.device().clone();

        // --- attention ---
        let normed = rms_named(x, &self.attn_norm, "attn_norm")?;
        tsig("attn_norm", &normed)?;
        let q = self.wq.forward(&normed)?;
        let k = self.wk.forward(&normed)?;
        let v = self.wv.forward(&normed)?;
        tsig("wq", &q)?;
        tsig("wk", &k)?;
        tsig("wv", &v)?;

        let q = Self::norm_heads(&q, self.num_heads, self.head_dim, &self.q_norm)?;
        let k = Self::norm_heads(&k, self.num_kv_heads, self.head_dim, &self.k_norm)?;
        let v = Self::norm_heads(&v, self.num_kv_heads, self.head_dim, &self.v_norm)?;
        tsig("q_norm", &q)?;
        tsig("k_norm", &k)?;
        tsig("v_norm", &v)?;

        let q = self.rope(&q, self.num_heads, positions)?;
        let k = self.rope(&k, self.num_kv_heads, positions)?;
        tsig("q_rope", &q)?;
        tsig("k_rope", &k)?;

        let q_host = q.to_vec_f32()?;
        let k_new = k.to_vec_f32()?;
        let v_new = v.to_vec_f32()?;

        // The attention set is the whole history plus the new rows; the
        // helper derives the causal offset from `kv_len - steps`.
        let (k_hist, v_hist) = match kv_prev {
            Some(prev) => {
                let mut k = prev.k.clone();
                k.extend_from_slice(&k_new);
                let mut v = prev.v.clone();
                v.extend_from_slice(&v_new);
                (k, v)
            }
            None => (k_new.clone(), v_new.clone()),
        };

        // The attention itself runs on the host: the device kernel hardcodes
        // the `1/sqrt(head_dim)` softmax scale and Gemma 4 uses 1.0.
        let attn = crate::shared_attention::fused_or_scalar_attention_scaled(
            &q_host,
            &k_hist,
            &v_hist,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            steps,
            self.window,
            self.attention_scale,
            &device,
        )?;
        tsig("attn_raw", &attn)?;
        let attn = self.wo.forward(&attn)?;
        tsig("wo", &attn)?;
        let attn = rms_named(&attn, &self.attn_post_norm, "attn_post_norm")?;
        let attn_out = grim_nn::modules::add_on_device(x, &attn)?;
        tsig("attn_out", &attn_out)?;

        // --- feed forward ---
        let ffn_in = rms_named(&attn_out, &self.ffn_norm, "ffn_norm")?;
        tsig("ffn_in", &ffn_in)?;
        let mlp = self.mlp.forward(&ffn_in, tsig)?;
        tsig("mlp", &mlp)?;
        let mlp = rms_named(&mlp, &self.ffn_post_norm, "ffn_post_norm")?;
        let mut out = grim_nn::modules::add_on_device(&attn_out, &mlp)?;
        tsig("ffn_out", &out)?;

        // --- per-layer embedding injection ---
        out = self.inject_per_layer(out, steps, per_layer, &device, tsig)?;

        // --- block output scale ---
        if let Some(scale) = self.out_scale {
            out = grim_nn::modules::mul_scalar_on_device(&out, scale)?;
        }

        Ok((
            out,
            Gemma4BlockKv {
                k: k_hist,
                v: v_hist,
            },
        ))
    }

    /// Forward a block that reuses an earlier block's KV cache.
    ///
    /// Such a block computes its own query but attends over the borrowed keys
    /// and values (`src/models/gemma4.cpp:269-274`).
    pub fn forward_shared(
        &self,
        x: &Tensor,
        positions: &[u32],
        per_layer: Option<&[f32]>,
        kv: &Gemma4BlockKv,
        _tsig: &dyn Fn(&str, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let steps = x.shape().dims()[0];
        let device = x.device().clone();

        let normed = rms_named(x, &self.attn_norm, "attn_norm")?;
        let q = self.wq.forward(&normed)?;
        let q = Self::norm_heads(&q, self.num_heads, self.head_dim, &self.q_norm)?;
        let q = self.rope(&q, self.num_heads, positions)?;
        let q_host = q.to_vec_f32()?;

        let attn = crate::shared_attention::fused_or_scalar_attention_scaled(
            &q_host,
            &kv.k,
            &kv.v,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            steps,
            self.window,
            self.attention_scale,
            &device,
        )?;
        let attn = self.wo.forward(&attn)?;
        let attn = rms_named(&attn, &self.attn_post_norm, "attn_post_norm")?;
        let attn_out = grim_nn::modules::add_on_device(x, &attn)?;

        let ffn_in = rms_named(&attn_out, &self.ffn_norm, "ffn_norm")?;
        let mlp = self.mlp.forward(&ffn_in, _tsig)?;
        let mlp = rms_named(&mlp, &self.ffn_post_norm, "ffn_post_norm")?;
        let mut out = grim_nn::modules::add_on_device(&attn_out, &mlp)?;

        out = self.inject_per_layer(out, steps, per_layer, &device, _tsig)?;

        if let Some(scale) = self.out_scale {
            out = grim_nn::modules::mul_scalar_on_device(&out, scale)?;
        }
        Ok(out)
    }

    /// Mix this block's slice of the per-layer embedding into the residual
    /// stream (`src/models/gemma4.cpp:367-389`).
    fn inject_per_layer(
        &self,
        out: Tensor,
        steps: usize,
        per_layer: Option<&[f32]>,
        device: &Device,
        tsig: &dyn Fn(&str, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let (Some(slice), Some(gate), Some(proj), Some(norm)) = (
            per_layer,
            &self.per_layer_inp_gate,
            &self.per_layer_proj,
            &self.per_layer_post_norm,
        ) else {
            return Ok(out);
        };
        let pe_in = out.clone();
        let g = gate.forward(&out)?;
        tsig("pl_gate", &g)?;
        let g = grim_nn::modules::gelu_on_device(&g)?;
        tsig("pl_gelu", &g)?;
        let width = self.per_layer_width();
        let pl = grim_nn::modules::move_to_device(
            &cpu_tensor(slice.to_vec(), Shape::new(vec![steps, width])),
            device,
        )?;
        let g = mul_on_device(&g, &pl)?;
        let p = proj.forward(&g)?;
        tsig("pl_proj", &p)?;
        let p = rms_named(&p, norm, "per_layer_post_norm")?;
        let out = grim_nn::modules::add_on_device(&pe_in, &p)?;
        tsig("pl_out", &out)?;
        Ok(out)
    }

    /// Width of the per-layer embedding slice this block consumes.
    ///
    /// `per_layer_input_gate` maps the residual stream (hidden) DOWN to the
    /// per-layer width, so its output width is dim 0 of the stored
    /// `[out, in]` weight.
    fn per_layer_width(&self) -> usize {
        self.per_layer_inp_gate
            .as_ref()
            .and_then(|l| l.weight.shape().dim(0).ok())
            .unwrap_or(0)
    }
}

/// Elementwise product of two equally shaped tensors.
fn mul_on_device(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = grim_nn::modules::pick_device_for_tensor(a);
    let (out, _handle) = dev.mul(&**a.storage(), &**b.storage(), a.shape())?;
    Ok(Tensor::new(
        std::sync::Arc::from(out),
        a.shape().clone(),
        a.dtype(),
        grim_tensor::QuantProvenance::default(),
        a.device().clone(),
    ))
}

// Per-layer embeddings

/// The per-layer ("per layer input") embedding table, resolved per block.
///
/// The reference computes it as
/// `inp_per_layer = rms_norm(per_layer_model_proj @ embd) + per_layer_tok_embd[token]`
/// and scales it by `1/sqrt(2)` (`src/models/gemma4.cpp:440-498`).
struct PerLayerInputs {
    /// Layer-major: the slice of block `l` starts at `l * tokens * width`.
    data: Vec<f32>,
    tokens: usize,
    width: usize,
    n_layers: usize,
}

impl PerLayerInputs {
    fn layer(&self, il: usize) -> Option<&[f32]> {
        if il >= self.n_layers {
            return None;
        }
        let start = il * self.tokens * self.width;
        Some(&self.data[start..start + self.tokens * self.width])
    }
}


/// Re-index the per-layer embedding from `[tokens, n_layers * width]` to
/// layer-major `[n_layers][tokens][width]`.
///
/// The projection's output feature index is `layer * width + component`
/// (the reference reshapes it to `[n_embd_per_layer, n_layer, n_tokens]`,
/// `src/models/gemma4.cpp:486`), so block `l` owns the columns
/// `l * width .. (l + 1) * width` of every token row.
fn layer_major_per_layer(flat: &[f32], tokens: usize, n_layers: usize, width: usize) -> Vec<f32> {
    let stride = n_layers * width;
    let mut data = vec![0.0f32; n_layers * tokens * width];
    for t in 0..tokens {
        let src = &flat[t * stride..(t + 1) * stride];
        for l in 0..n_layers {
            let dst = l * tokens * width + t * width;
            data[dst..dst + width].copy_from_slice(&src[l * width..(l + 1) * width]);
        }
    }
    data
}

// Model

/// The per-request KV history of a Gemma-4 model.
///
/// The decode path feeds ONE token at a time and the model carries the rest,
/// so every KV-owning block keeps its post-RoPE keys and raw values in token
/// order. The blocks that share a KV cache read the history of their source
/// block instead of keeping their own.
#[derive(Default)]
struct Gemma4KvState {
    layers: Vec<Option<Gemma4BlockKv>>,
}

/// The Gemma-4 text model.
pub struct Gemma4 {
    pub cfg: Gemma4Config,
    pub device: Device,
    /// The token embedding table; also the (tied) output head.
    pub tok_embeddings: Embedding,
    /// The per-layer token embedding table, `[vocab, n_embd_per_layer * n_layers]`.
    pub per_layer_tok_embd: Option<Embedding>,
    /// Projects the residual stream into the per-layer embedding space.
    pub per_layer_model_proj: Option<Linear>,
    /// Norm of the per-layer projection.
    pub per_layer_proj_norm: Option<RmsNorm>,
    pub layers: Vec<Gemma4Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Gemma4 {
    /// Load a Gemma-4 model from a weight source.
    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: Gemma4Config,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");
        // The token embedding is row-major `[vocab, hidden]` - the embedding
        // layout, not the `[out, in]` one a `Linear` reads.
        let tok_embeddings = Embedding::load(
            &root.scoped("embed_tokens"),
            cfg.vocab_size,
            cfg.hidden_size,
        )?;

        let (per_layer_tok_embd, per_layer_model_proj, per_layer_proj_norm) =
            if cfg.n_embd_per_layer > 0 {
                let width = cfg.n_embd_per_layer * cfg.num_hidden_layers;
                let embd =
                    Embedding::load(&root.scoped("embed_tokens_per_layer"), cfg.vocab_size, width)?;
                let proj = Linear::load_shape(
                    &root.scoped("per_layer_model_projection"),
                    [cfg.hidden_size, width],
                )?;
                let norm = RmsNorm::load(
                    &root.scoped("per_layer_projection_norm"),
                    cfg.n_embd_per_layer,
                    cfg.rms_norm_eps,
                )?;
                (Some(embd), Some(proj), Some(norm))
            } else {
                (None, None, None)
            };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for il in 0..cfg.num_hidden_layers {
            let layer_ws = root.scoped("layers").scoped(&il.to_string());
            layers.push(Gemma4Block::load(&layer_ws, &cfg, il, &device)?);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        // Gemma 4 ties the output head to the token embedding
        // (`src/models/gemma4.cpp:49-55`).
        let output = match ws.scoped("lm_head").meta("weight") {
            Some(_) => Linear::load_shape(
                &ws.scoped("lm_head"),
                [cfg.vocab_size, cfg.hidden_size],
            )?,
            None => Linear::from_tensor(tok_embeddings.weight.clone(), None),
        };

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            per_layer_tok_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            layers,
            norm,
            output,
        })
    }

    /// Build the per-layer embedding inputs for a batch of tokens.
    fn build_per_layer_inputs(
        &self,
        h: &Tensor,
        ids: &[u32],
        tokens: usize,
    ) -> Result<Option<PerLayerInputs>> {
        let (Some(embd), Some(proj), Some(norm)) = (
            self.per_layer_tok_embd.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) else {
            return Ok(None);
        };
        let n_pl = self.cfg.n_embd_per_layer;
        let n_layers = self.cfg.num_hidden_layers;

        // The per-layer token embedding, scaled by sqrt(n_embd_per_layer).
        let gathered = embd.forward(ids, tokens, n_pl * n_layers)?;
        let gathered = grim_nn::modules::mul_scalar_on_device(&gathered, (n_pl as f32).sqrt())?;

        // The projection of the residual stream, scaled by 1/sqrt(hidden).
        let projected = proj.forward(h)?;
        let projected = grim_nn::modules::mul_scalar_on_device(
            &projected,
            1.0 / (self.cfg.hidden_size as f32).sqrt(),
        )?;

        // RMS norm over the per-layer width, then add, then scale by 1/sqrt(2).
        let flat =
            crate::block::reshaped_view(&projected, &Shape::new(vec![tokens * n_layers, n_pl]))?;
        let normed = rms_named(&flat, norm, "per_layer_proj_norm")?;
        let normed =
            crate::block::reshaped_view(&normed, &Shape::new(vec![tokens, n_pl * n_layers]))?;
        let sum = grim_nn::modules::add_on_device(&normed, &gathered)?;
        let sum = grim_nn::modules::mul_scalar_on_device(&sum, 1.0 / 2.0f32.sqrt())?;

        // Re-index to layer-major so each block can take a contiguous slice.
        let data = layer_major_per_layer(&sum.to_vec_f32()?, tokens, n_layers, n_pl);
        Ok(Some(PerLayerInputs {
            data,
            tokens,
            width: n_pl,
            n_layers,
        }))
    }
}

impl Model for Gemma4 {
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

impl CausalLm for Gemma4 {
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
        let pos_f32 = positions.to_vec_f32()?;
        let pos_u32: Vec<u32> = pos_f32.iter().map(|&p| p as u32).collect();
        let ids_f32 = input_ids.to_vec_f32()?;
        let tokens = ids_f32.len();
        let ids: Vec<u32> = ids_f32.iter().map(|&t| t as u32).collect();

        // The token embedding is scaled by sqrt(hidden_size)
        // (`src/models/gemma4.cpp:164`).
        let mut h = self.tok_embeddings.forward(&ids, tokens, self.cfg.hidden_size)?;
        h = grim_nn::modules::mul_scalar_on_device(&h, (self.cfg.hidden_size as f32).sqrt())?;

        let per_layer = self.build_per_layer_inputs(&h, &ids, tokens)?;

        // The KV history lives on the session: the decode path feeds one token
        // at a time, and the blocks that share a KV cache read the history of
        // their source block instead of producing their own.
        if session.model_state().is_none() {
            session.set_model_state(Box::new(Gemma4KvState {
                layers: (0..self.layers.len()).map(|_| None).collect(),
            }));
        }
        let state = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Gemma4KvState>())
            .ok_or_else(|| {
                grim_core::error::Error::Backend(
                    "Gemma4::forward: model_state must be Gemma4KvState".into(),
                )
            })?;
        if state.layers.len() < self.layers.len() {
            state.layers.resize(self.layers.len(), None);
        }

        // Per-layer trace, off by default. A device fault or a wrong device
        // primitive corrupts the logits silently, and the only way to learn
        // WHICH layer diverged is to say so on the way in.
        let trace = std::env::var("GRIM_TRACE_LAYERS").is_ok();
        // GRIM_TRACE_OPS=<name> narrows the trace to ONE point. Every point is
        // a D2H readback, and a readback SYNCHRONIZES the stream - so a
        // full trace hides a missing dependency. One point says whether that
        // single sync is enough.
        let trace_op: Option<String> = std::env::var("GRIM_TRACE_OPS").ok();
        let tsig = |name: &str, t: &Tensor| -> Result<()> {
            if trace && trace_op.as_deref().map_or(true, |want| want == name) {
                let sig: Vec<f32> = t.to_vec_f32()?.into_iter().take(4).collect();
                eprintln!("[gemma4-trace]   {name} = {sig:?}");
            }
            Ok(())
        };

        for il in 0..self.layers.len() {
            let layer = &self.layers[il];
            let slice = per_layer.as_ref().and_then(|p| p.layer(il));
            let out = match self.cfg.kv_source(il) {
                Some(src) => {
                    // The source block has already run in this pass, so its
                    // history includes the tokens of THIS step.
                    let kv = state.layers[src].as_ref().ok_or_else(|| {
                        grim_core::error::Error::Backend(format!(
                            "gemma4: block {il} reuses the KV of block {src}, which has not run yet"
                        ))
                    })?;
                    layer.forward_shared(&h, &pos_u32, slice, kv, &tsig)?
                }
                None => {
                    let prev = state.layers[il].as_ref();
                    let (out, kv) = layer.forward(&h, &pos_u32, slice, prev, &tsig)?;
                    state.layers[il] = Some(kv);
                    out
                }
            };
            if trace {
                // A short prefix signature: enough to tell "this layer is
                // already wrong" from "this layer is fine".
                let sig: Vec<f32> = out.to_vec_f32()?.into_iter().take(4).collect();
                eprintln!(
                    "[gemma4-trace] layer {il} ({}) out[0..4]={sig:?}",
                    match layer.layer_type {
                        Gemma4LayerType::Sliding => "swa",
                        Gemma4LayerType::Full => "full",
                    }
                );
            }
            h = out;
        }
        session.advance_pos(tokens);

        let normed = rms_named(&h, &self.norm, "output_norm")?;
        session.set_last_hidden_state(normed.clone());
        let logits = self.output.forward(&normed)?;

        // Final logit softcapping (`src/models/gemma4.cpp:426-430`).
        if let Some(cap) = self.cfg.final_logit_softcapping {
            let mut v = logits.to_vec_f32()?;
            for x in v.iter_mut() {
                *x = cap * (*x / cap).tanh();
            }
            Ok(cpu_tensor(v, logits.shape().clone()))
        } else {
            Ok(logits)
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// A block that owns its KV cache must not be confused with one that
    /// borrows it: the borrow map has to point at the last sliding / last full
    /// block below the split.
    #[test]
    fn kv_source_map_matches_the_reference() {
        // E4B: 42 blocks, 18 shared -> the split is at 24.
        let mut cfg = Gemma4Config::default();
        cfg.num_hidden_layers = 42;
        cfg.num_kv_shared_layers = 18;
        cfg.layer_types = (0..42)
            .map(|il| {
                if il % 6 == 5 {
                    Gemma4LayerType::Full
                } else {
                    Gemma4LayerType::Sliding
                }
            })
            .collect();
        assert_eq!(cfg.n_layer_kv_from_start(), 24);
        // Blocks below the split borrow.
        assert_eq!(cfg.kv_source(23), None);
        assert_eq!(cfg.kv_source(24), Some(22));
        assert_eq!(cfg.kv_source(29), Some(23));
        assert_eq!(cfg.kv_source(41), Some(23));
        // The borrowed blocks have the same attention kind as their source.
        assert_eq!(cfg.layer_type(22), cfg.layer_type(24));
        assert_eq!(cfg.layer_type(23), cfg.layer_type(29));
    }

    /// The "proportional" rope rotates only the live pairs, against the
    /// `head_dim/2` half, and leaves the other dimensions untouched.
    #[test]
    fn proportional_rope_moves_only_the_live_pairs() {
        let head_dim = 8usize;
        // One live pair: dimension 0 rotates against dimension 4 (the
        // `head_dim/2` half), every other dimension stays put.
        let n_pairs = 1usize;
        let rope = Gemma4ProportionalRope {
            base: 10000.0,
            head_dim,
            n_pairs,
        };
        let x = cpu_tensor(
            (0..head_dim).map(|i| i as f32).collect(),
            Shape::new(vec![1, head_dim]),
        );
        let out = rope.apply(&x, 1, &[0]).unwrap().to_vec_f32().unwrap();
        // Position 0: the rotation angle is 0, so the pair is unchanged.
        for i in 0..head_dim {
            assert_eq!(out[i], i as f32, "dimension {i} must be untouched");
        }
        // Position 1: the first pair rotates, the rest does not.
        let out = rope.apply(&x, 1, &[1]).unwrap().to_vec_f32().unwrap();
        let theta = (10000.0f32).powf(-2.0 * 0.0 / head_dim as f32);
        let (s, c) = theta.sin_cos();
        assert!((out[0] - (0.0 * c - 4.0 * s)).abs() < 1e-5);
        assert!((out[4] - (0.0 * s + 4.0 * c)).abs() < 1e-5);
        for i in 1..head_dim {
            if i != 4 {
                assert_eq!(out[i], i as f32, "dimension {i} must be untouched");
            }
        }
    }


    /// The per-layer embedding is re-indexed to layer-major, so block `l` takes
    /// the columns `l * width .. (l + 1) * width` of every token row.
    #[test]
    fn per_layer_embedding_is_reindexed_to_layer_major() {
        let tokens = 2usize;
        let n_layers = 3usize;
        let width = 2usize;
        // Row t holds the values t*100 + column, and the column of
        // (layer l, component d) is l * width + d.
        let flat: Vec<f32> = (0..tokens)
            .flat_map(|t| (0..n_layers * width).map(move |i| (t * 100 + i) as f32))
            .collect();
        let data = layer_major_per_layer(&flat, tokens, n_layers, width);
        assert_eq!(data.len(), n_layers * tokens * width);
        for l in 0..n_layers {
            for t in 0..tokens {
                for d in 0..width {
                    let got = data[l * tokens * width + t * width + d];
                    let want = (t * 100 + l * width + d) as f32;
                    assert_eq!(got, want, "layer {l}, token {t}, component {d}");
                }
            }
        }
    }

    /// The live rotary pair count is clamped to the head: a checkpoint cannot
    /// rotate more pairs than the head has.
    #[test]
    fn n_rot_pairs_is_clamped_to_the_head() {
        let mut cfg = Gemma4Config::default();
        cfg.head_dim_full = 512;
        cfg.n_rot_pairs_full = 64;
        assert_eq!(cfg.n_rot_pairs(), 64);
        cfg.n_rot_pairs_full = 1000;
        assert_eq!(cfg.n_rot_pairs(), 256);
    }
}
