//! Qwen3.5 / Qwen3.8 hybrid SSM (Mamba) + GQA Attention + SwiGLU FFN.
//! Supports `Qwen3.8-27B` and related hybrid GGUF checkpoints with fused `attn_qkv`, `attn_gate`, 1D short-convolution SSM layers,.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::{Inner, SessionT};
use grim_nn::modules::{Embedding, Linear, RmsNorm, pick_device_for_storage_device};
use grim_nn::{TensorParallelConfig, WeightSource};
use grim_tensor::{
    ArithType, CoreTensorOps, DType, Device, QuantProvenance, RecurrentOps, Shape, Storage, Tensor,
};

// Config


/// Why a D2D decode path declined, for `GRIM_D2D_TRACE=1`.
pub(crate) fn d2d_trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GRIM_D2D_TRACE").is_ok())
}

/// Check if D2D is strictly required (panics or errors on fallback)
pub(crate) fn d2d_strict_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GRIM_FORCE_D2D").map(|v| v != "0").unwrap_or(false))
}

/// `GRIM_DEBUG_ATTN_SHAPES`, cached.
///
/// This flag was read with a bare `std::env::var` at five sites in
/// `attention_layer_d2d`, one of them inside the `stage!` macro that expands
/// once per stage per layer. That is ~40 environment lookups per decoded token
/// on the 9B's 8 attention layers, paid whether or not the flag is set.
/// `std::env::var` takes the process environment lock and walks the
/// environ array; it does not belong on the decode hot path.
fn attn_shapes_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GRIM_DEBUG_ATTN_SHAPES").is_ok())
}

/// `GRIM_QWEN_ATTN_D2D` set to a disabling value, cached — see
/// [`attn_shapes_debug`] for why.
fn attn_d2d_disabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("GRIM_QWEN_ATTN_D2D").as_deref(),
            Ok("0" | "false" | "off" | "no")
        )
    })
}

/// Declare a D2D fallback site.
/// When D2D is default, falling back is an exceptional event: log warning and emit fallback event.
/// If `GRIM_FORCE_D2D=1`, halts with error rather than silently degrading.
macro_rules! d2d_decline {
    ($($arg:tt)*) => {{
        let reason = format!($($arg)*);
        if d2d_trace_enabled() {
            eprintln!("[d2d-decline] {}: {}", line!(), reason);
        } else {
            eprintln!("[d2d-fallback-warning] {}:{}: D2D fallback to host: {}", file!(), line!(), reason);
        }
        if d2d_strict_enabled() {
            return Err(grim_core::error::Error::Backend(format!(
                "D2D is required but declined: {reason}"
            )));
        }
        return Ok(None);
    }};
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
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

    // Hybrid SSM parameters
    pub full_attention_interval: usize,
    pub ssm_d_state: usize,
    pub ssm_d_inner: usize,
    pub ssm_d_conv: usize,
    pub ssm_dt_rank: usize,
    pub ssm_n_group: usize,

    pub rotary_dim: Option<usize>,

    // Multi-device pipeline distribution
    pub devices: Vec<Device>,
}

impl Default for Qwen35Config {
    fn default() -> Self {
        Self {
            vocab_size: 248320,
            hidden_size: 5120,
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            num_layers: 65,
            intermediate_size: 17408,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            max_seq_len: 262144,
            full_attention_interval: 4,
            ssm_d_state: 128,
            ssm_d_inner: 6144,
            ssm_d_conv: 4,
            ssm_dt_rank: 48,
            ssm_n_group: 16,
            rotary_dim: None,
            devices: Vec::new(),
        }
    }
}

impl ModelConfig for Qwen35Config {
    fn name(&self) -> &str {
        "qwen35"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Layer Cache

pub struct Qwen35LayerCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    pub current_pos: usize,
    /// WI-kv (qwen35): device-resident K/V arenas for full-attention layers.
    /// History stays on the GPU so decode uploads only the current step's rows instead of.
    #[doc(hidden)]
    pub k_device: Option<Box<dyn grim_tensor::BackendStorage>>,
    #[doc(hidden)]
    pub v_device: Option<Box<dyn grim_tensor::BackendStorage>>,
    /// Quantized paged KV cache, used when `GRIM_KV_QUANT` selects a packed
    /// format. `k_pages`/`v_pages` hold packed sub-blocks; `block_table` maps
    /// logical page index -> physical page id for the paged attention kernel.
    ///
    /// These are mutually exclusive with `k_device`/`v_device`: the dense f32
    /// arena and the paged packed arena are different representations, and the
    /// decode path picks one per layer via `kv_quant_format()`.
    #[doc(hidden)]
    pub k_pages: Option<Box<dyn grim_tensor::BackendStorage>>,
    #[doc(hidden)]
    pub v_pages: Option<Box<dyn grim_tensor::BackendStorage>>,
    #[doc(hidden)]
    pub block_table: Option<Box<dyn grim_tensor::BackendStorage>>,
    /// Device-resident short-conv history for the recurrent (KDA) path, shaped
    /// `[1, d_conv-1, conv_dim]`.
    ///
    /// The host `conv_state` above is the reference path. When this is `Some`,
    /// the KDA branch convolves and gates on device and the host copy is left
    /// untouched — the two are alternatives, not a mirror, so a run that takes
    /// the device path must not also assert on the host vectors.
    #[doc(hidden)]
    pub conv_state_dev: Option<Box<dyn grim_tensor::BackendStorage>>,
    /// Device-resident KDA state, shaped
    /// `[n_value_heads, head_dim, head_dim]`, updated in place by
    /// `kda_gated_delta_rule_batched`.
    ///
    /// This is what keeps the recurrence off the host: the CPU path copies
    /// `ssm_state` to the device and back on every token of every recurrent
    /// layer, so 49 of 65 layers each pay four transfers and two syncs per
    /// token before any arithmetic happens.
    #[doc(hidden)]
    pub ssm_state_dev: Option<Box<dyn grim_tensor::BackendStorage>>,
}

// `BackendStorage` doesn't implement Debug — hand-roll one that prints the
// host fields and omits the device arenas.
impl std::fmt::Debug for Qwen35LayerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35LayerCache")
            .field("k_cache", &self.k_cache.len())
            .field("v_cache", &self.v_cache.len())
            .field("conv_state", &self.conv_state.len())
            .field("ssm_state", &self.ssm_state.len())
            .field("current_pos", &self.current_pos)
            .field("k_device", &self.k_device.is_some())
            .field("v_device", &self.v_device.is_some())
            .field("k_pages", &self.k_pages.is_some())
            .field("conv_state_dev", &self.conv_state_dev.is_some())
            .field("ssm_state_dev", &self.ssm_state_dev.is_some())
            .finish()
    }
}

// Hand-rolled Clone: Box<dyn BackendStorage> isn't Clone; cloned caches start
// without the device arenas and rebuild them on the next forward.
impl Clone for Qwen35LayerCache {
    fn clone(&self) -> Self {
        Self {
            k_cache: self.k_cache.clone(),
            v_cache: self.v_cache.clone(),
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
            current_pos: self.current_pos,
            k_device: None,
            v_device: None,
            k_pages: None,
            v_pages: None,
            block_table: None,
            conv_state_dev: None,
            ssm_state_dev: None,
        }
    }
}

impl Qwen35LayerCache {
    pub fn new(cfg: &Qwen35Config) -> Self {
        // The short conv runs over the FUSED attn_qkv width of a recurrent
        // layer, which is not the attention q+k+v width. Measured on
        // Qwen3.8-27B: ssm_conv1d.weight is [10240, 4] and the derived layout is
        // q 48*128 + k 16*128 + v 16*128 = 10240. The previous formula
        // (`max(hidden, d_inner) * 2`) encodes an attention split that recurrent
        // layers do not have, and over-allocates by 2048 per tap
        // (36864 vs the 30720 required).
        let conv_dim = (cfg.ssm_dt_rank + 2 * cfg.ssm_n_group) * cfg.ssm_d_state;
        let conv_size = (cfg.ssm_d_conv.max(1) - 1) * conv_dim;
        let ssm_size = cfg.ssm_n_group.max(1)
            * cfg.ssm_d_state.max(1)
            * (cfg.ssm_d_inner / cfg.ssm_n_group.max(1));
        Self {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            conv_state: vec![0.0; conv_size.max(1)],
            ssm_state: vec![0.0; ssm_size.max(1)],
            current_pos: 0,
            k_device: None,
            v_device: None,
            k_pages: None,
            v_pages: None,
            block_table: None,
            conv_state_dev: None,
            ssm_state_dev: None,
        }
    }
}

// Block

pub struct Qwen35Block {
    pub device: Device,
    pub attn_norm: RmsNorm,

    // Attention path tensors (for full attention layers)
    pub wq: Option<Linear>,
    pub wk: Option<Linear>,
    pub wv: Option<Linear>,
    pub wo: Option<Linear>,
    pub attn_q_norm: Option<RmsNorm>,
    pub attn_k_norm: Option<RmsNorm>,

    // SSM path tensors (for recurrent layers)
    pub attn_qkv: Option<Linear>,
    pub attn_gate: Option<Linear>,
    pub ssm_out: Option<Linear>,
    pub ssm_conv1d: Option<Tensor>,
    pub ssm_conv_vec: Option<Vec<f32>>,
    pub ssm_a: Option<Vec<f32>>,
    pub ssm_alpha: Option<Linear>,
    pub ssm_beta: Option<Linear>,
    pub ssm_dt_bias: Option<Vec<f32>>,
    pub ssm_norm: Option<Vec<f32>>,
    pub ssm_dt_bias_dev: Option<std::sync::Arc<dyn grim_tensor::BackendStorage>>,
    pub ssm_a_dev: Option<std::sync::Arc<dyn grim_tensor::BackendStorage>>,
    pub ssm_norm_dev: Option<std::sync::Arc<dyn grim_tensor::BackendStorage>>,
    /// KDA geometry, copied from the config at load. `ssm_dt_rank` is
    /// num_value_heads for this family, not a scalar rank — see
    /// `cfg_ssm_num_value_heads`.
    pub ssm_dt_rank_hint: usize,
    pub ssm_n_group_hint: usize,
    pub ssm_d_state_hint: usize,
    /// Conv taps over the fused qkv stream, from `ssm_conv1d.weight`'s width.
    pub ssm_d_conv_hint: usize,

    // Feed-forward Network
    pub post_attention_norm: RmsNorm,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,

    pub is_full_attention: bool,
    pub layer_idx: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Fused Q8_0 QKV projection blob on ROCm (Phase 2b). Only built when all
    /// three projections are present AND row-exact (no TP padding); issues
    /// 1 dot4 GEMV instead of 3 on single-token decode.
    pub wqkv_q80_fused: Option<std::sync::Arc<grim_backend_rocm::FusedQkvWeights>>,
    /// Concatenated Q4_K Gate+Up weights for the local tensor-parallel shard.
    /// This uses the safe Q4_K fused-dequant/WMMA path, not the RDNA4-incompatible
    /// dot4 Q4_K kernel.
    pub w_gate_up_q4k_fused: Option<std::sync::Arc<grim_backend_rocm::FusedGateUpQ4KWeights>>,
}

impl Qwen35Block {
    /// Number of value heads in the KDA recurrence. Taken from `ssm_dt_rank`,
    /// which this checkpoint uses for that purpose: `ssm_a`, `ssm_dt.bias`,
    /// `ssm_alpha.weight` and `ssm_beta.weight` are all sized [48], and
    /// 48 * 128 == ssm_d_inner (6144).
    pub fn cfg_ssm_num_value_heads(&self) -> usize {
        self.ssm_dt_rank_hint
    }

    /// Number of key heads. `ssm_n_group` (16), giving a 3:1 value:key ratio.
    pub fn cfg_ssm_num_key_heads(&self) -> usize {
        self.ssm_n_group_hint
    }

    /// Per-head SSM width. `ssm_d_state` (128) — NOT the attention head_dim
    /// of 256, which the previous code used throughout the SSM branch.
    pub fn cfg_ssm_head_dim(&self) -> usize {
        self.ssm_d_state_hint
    }

    /// Raw `ssm_d_state`, the state matrix width used for allocation.
    pub fn cfg_ssm_d_state(&self) -> usize {
        self.ssm_d_state_hint
    }

    /// Taps in the causal short conv over the fused qkv stream.
    /// `ssm_conv1d.weight` is [conv_dim, d_conv] = [10240, 4] on Qwen3.8.
    pub fn cfg_ssm_d_conv(&self) -> usize {
        self.ssm_d_conv_hint
    }
}

impl Qwen35Block {
    pub fn load_tp(
        ws: &WeightSource<'_>,
        cfg: &Qwen35Config,
        layer_idx: usize,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let device = ws.device();
        let is_full_attention = (layer_idx + 1) % cfg.full_attention_interval.max(1) == 0;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        // The recurrent path fused QKV is the CONV width, NOT the attention
        // q+2kv width: [K K V] = 2*(n_group*d_state) + (dt_rank*d_state).
        // Deriving it from the attention geometry gave 6144 for the 9B, which
        // the old `.max(10240)` floor then widened to 10240 - 2048 rows past
        // the end of an [8192, 4096] weight, on every KDA layer, feeding the
        // conv. The floor was silently correct for the 27B, where the two
        // numbers coincide.
        let kda_qkv_dim = (cfg.ssm_dt_rank + 2 * cfg.ssm_n_group) * cfg.ssm_d_state;

        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        let (wq, wk, wv, wo, attn_q_norm, attn_k_norm, attn_qkv, attn_gate, ssm_out) =
            if is_full_attention {
                // attn_q is FUSED query + output gate, per llama.cpp qwen35.cpp:
                // `create_tensor_qkv(..., n_embd_head_k * n_head * 2, ...)` and the
                // graph splits it with two `ggml_view_3d`s into Q and gate. So the
                // out-width is 2 * q_dim (12288 for 24 heads x 256), NOT a fixed
                // literal. The previous `q_dim.max(12288)` happened to match this
                // checkpoint while being wrong for any other head count.
                // The checkpoint packs attn_q rows PER-HEAD INTERLEAVED as
                // [q_h | gate_h] (llama.cpp qwen35.cpp:274-293 views Q and the
                // gate with row stride `2*n_embd_head`). Every grim runtime
                // split — host, D2D, decode graph, fused QKV decode build —
                // assumes [all Q | all gate]. Permute whole rows once at load;
                // a row is a whole number of quant blocks, so packed formats
                // survive intact.
                let mut attn_q_perm = vec![0usize; 2 * q_dim];
                let hd = q_dim / cfg.num_heads;
                for h in 0..cfg.num_heads {
                    for d in 0..hd {
                        attn_q_perm[h * hd + d] = h * 2 * hd + d;
                        attn_q_perm[q_dim + h * hd + d] = h * 2 * hd + hd + d;
                    }
                }
                let wq = Linear::load_column_parallel_permuted_rows(
                    &ws.pp("attn_q"),
                    cfg.hidden_size,
                    2 * q_dim,
                    false,
                    tp,
                    &attn_q_perm,
                )
                .ok();
                let wk = Linear::load_column_parallel(
                    &ws.pp("attn_k"),
                    cfg.hidden_size,
                    kv_dim,
                    false,
                    tp,
                )
                .ok();
                let wv = Linear::load_column_parallel(
                    &ws.pp("attn_v"),
                    cfg.hidden_size,
                    kv_dim,
                    false,
                    tp,
                )
                .ok();
                let wo = Linear::load_row_parallel(
                    &ws.pp("attn_output"),
                    q_dim,
                    cfg.hidden_size,
                    false,
                    tp,
                )
                .ok();
                let attn_q_norm =
                    RmsNorm::load(&ws.pp("attn_q_norm"), cfg.head_dim, cfg.rms_norm_eps).ok();
                let attn_k_norm =
                    RmsNorm::load(&ws.pp("attn_k_norm"), cfg.head_dim, cfg.rms_norm_eps).ok();
                (wq, wk, wv, wo, attn_q_norm, attn_k_norm, None, None, None)
            } else {
                // Every load in this arm used to end in `.ok()`, which turned a
                // width mismatch into a silent `None`. That is how
                // `attn_qkv` could be absent on a checkpoint that plainly has
                // it, and the only symptom was a per-layer D2D decline 30 lines
                // downstream in `gated_delta_net_forward_d2d` — which reads like
                // a device problem and is a loader problem. Report the reason.
                fn report<E: std::fmt::Display>(
                    layer_idx: usize,
                    tag: &str,
                    r: std::result::Result<Linear, E>,
                ) -> Option<Linear> {
                    r.map_err(|e| {
                        eprintln!("[qwen35] layer {layer_idx} (KDA): {tag} did not load — {e}");
                    })
                    .ok()
                }
                let attn_qkv = match Linear::load_column_parallel(
                    &ws.pp("attn_qkv"),
                    cfg.hidden_size,
                    kda_qkv_dim,
                    false,
                    tp,
                ) {
                    Ok(l) => Some(l),
                    Err(e) => {
                        eprintln!("[qwen35-load-error] layer {layer_idx} attn_qkv failed to load: {e}");
                        return Err(grim_core::error::Error::Backend(format!(
                            "qwen35 layer {layer_idx} attn_qkv failed: {e}"
                        )));
                    }
                };
                let attn_gate = report(
                    layer_idx,
                    "attn_gate",
                    Linear::load_column_parallel(
                        &ws.pp("attn_gate"),
                        cfg.hidden_size,
                        // `attn_gate` serves BOTH layer types at two different widths:
                        // the KDA output gate is value_dim, the attention gate is q_dim.
                        // The 6144 here was a hardcoded literal, which is correct for the
                        // 27B by coincidence (its q_dim IS 6144) and 2048 rows too wide
                        // for the 9B, whose tensor is [4096, 4096] - the GEMM then strides
                        // past the end of the weight on every token.
                        q_dim.max(blk_value_dim(cfg)),
                        false,
                        tp,
                    ),
                );
                // `ssm_out` consumes the GATED recurrent output, whose width is value_dim -
                // not q_dim. Same literal-vs-derived mismatch as attn_gate above:
                // 6144 is right for the 27B by coincidence and 2048 rows too wide
                // for the 9B, whose ssm_out is [4096, 4096].
                let ssm_out = report(
                    layer_idx,
                    "ssm_out",
                    Linear::load_row_parallel(
                        &ws.pp("ssm_out"),
                        q_dim.max(blk_value_dim(cfg)),
                        cfg.hidden_size,
                        false,
                        tp,
                    ),
                );
                (
                    None, None, None, None, None, None, attn_qkv, attn_gate, ssm_out,
                )
            };

        if std::env::var("GRIM_DEBUG_QKV_LOAD").is_ok() {
            eprintln!(
                "[qkv-load] layer={layer_idx} full_attn={is_full_attention} \
                 kda_qkv_dim={kda_qkv_dim} wq={} attn_qkv={} attn_gate={} ssm_out={}",
                wq.is_some(),
                attn_qkv.is_some(),
                attn_gate.is_some(),
                ssm_out.is_some(),
            );
        }

        let (ssm_conv1d, ssm_conv_vec) = if let Ok(t) = ws.get_unconstrained("ssm_conv1d.weight") {
            let vec = t.to_vec_f32().ok();
            (Some(t), vec)
        } else {
            (None, None)
        };

        let ssm_a = ws
            .get_unconstrained("ssm_a")
            .ok()
            .and_then(|t| t.to_vec_f32().ok());
        let ssm_alpha = Linear::load_column_parallel(
            &ws.pp("ssm_alpha"),
            cfg.hidden_size,
            cfg.ssm_dt_rank,
            false,
            tp,
        )
        .ok();
        let ssm_beta = Linear::load_column_parallel(
            &ws.pp("ssm_beta"),
            cfg.hidden_size,
            cfg.ssm_dt_rank,
            false,
            tp,
        )
        .ok();
        let ssm_dt_bias = ws
            .get_unconstrained("ssm_dt.bias")
            .ok()
            .and_then(|t| t.to_vec_f32().ok());
        let ssm_norm = ws
            .get_unconstrained("ssm_norm.weight")
            .ok()
            .and_then(|t| t.to_vec_f32().ok());

        let post_attention_norm = if let Ok(m) = RmsNorm::load(
            &ws.pp("post_attention_norm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        ) {
            m
        } else {
            RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?
        };

        let ffn_gate = Linear::load_column_parallel(
            &ws.pp("ffn_gate"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
            tp,
        )?;
        let ffn_up = Linear::load_column_parallel(
            &ws.pp("ffn_up"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
            tp,
        )?;
        let ffn_down = Linear::load_row_parallel(
            &ws.pp("ffn_down"),
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
            tp,
        )?;

        let w_gate_up_q4k_fused = if matches!(&device, Device::Rocm(_))
            && matches!(
                std::env::var("GRIM_Q4K_FUSED_GATEUP").as_deref(),
                Ok("1" | "true" | "on" | "yes")
            )
            && matches!(
                ffn_gate.weight().dtype().storage,
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K)
            )
            && matches!(
                ffn_up.weight().dtype().storage,
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K)
            ) {
            let ordinal = match &device {
                Device::Rocm(ordinal) => *ordinal,
                _ => 0,
            };
            grim_backend_rocm::RocmDevice::try_new(ordinal)
                .ok()
                .and_then(|dev| {
                    dev.build_fused_gate_up_q4k(
                        ffn_gate.weight().storage().as_ref(),
                        ffn_up.weight().storage().as_ref(),
                    )
                    .ok()
                })
                .map(std::sync::Arc::new)
        } else {
            None
        };

        // Phase 2b: fused Q8_0 QKV blob — only when all three projections are
        // present AND row-exact (some TP shards pad rows to a minimum width;
        // the stock path cuts them via `exact()` but the fused GEMV cannot).
        let wqkv_q80_fused = if is_full_attention {
            let row_exact = |w: Option<&Linear>, want_rows: usize| {
                w.map(|l| l.weight.shape().dims().first().copied().unwrap_or(0) == want_rows)
                    .unwrap_or(false)
            };
            if row_exact(wq.as_ref(), q_dim)
                && row_exact(wk.as_ref(), kv_dim)
                && row_exact(wv.as_ref(), kv_dim)
            {
                crate::shared_attention::build_fused_qkv_q80_opt(
                    wq.as_ref(),
                    wk.as_ref(),
                    wv.as_ref(),
                )
                .map(std::sync::Arc::new)
            } else {
                None
            }
        } else {
            None
        };

        let (ssm_dt_bias_dev, ssm_a_dev, ssm_norm_dev) = if !device.is_cpu() {
            let dev = pick_device_for_storage_device(&device);
            let n_val = cfg.ssm_dt_rank.max(1);
            let h_dim = cfg.ssm_d_state.max(1);
            let dt_b = ssm_dt_bias.as_ref().and_then(|v| {
                dev.from_cpu(&v[..n_val.min(v.len())], &Shape::new(vec![n_val]), DType::F32).ok().map(std::sync::Arc::from)
            });
            let sa_b = ssm_a.as_ref().and_then(|v| {
                dev.from_cpu(&v[..n_val.min(v.len())], &Shape::new(vec![n_val]), DType::F32).ok().map(std::sync::Arc::from)
            });
            let sn_b = ssm_norm.as_ref().and_then(|v| {
                dev.from_cpu(&v[..h_dim.min(v.len())], &Shape::new(vec![h_dim]), DType::F32).ok().map(std::sync::Arc::from)
            });
            (dt_b, sa_b, sn_b)
        } else {
            (None, None, None)
        };

        Ok(Self {
            device,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            attn_q_norm,
            attn_k_norm,
            attn_qkv,
            attn_gate,
            ssm_out,
            ssm_conv1d,
            ssm_conv_vec,
            ssm_a,
            ssm_alpha,
            ssm_beta,
            ssm_dt_bias,
            ssm_norm,
            ssm_dt_bias_dev,
            ssm_a_dev,
            ssm_norm_dev,
            ssm_dt_rank_hint: cfg.ssm_dt_rank,
            ssm_n_group_hint: cfg.ssm_n_group,
            ssm_d_state_hint: cfg.ssm_d_state,
            ssm_d_conv_hint: cfg.ssm_d_conv,
            post_attention_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            is_full_attention,
            layer_idx,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim.unwrap_or(cfg.head_dim),
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            wqkv_q80_fused,
            w_gate_up_q4k_fused,
        })
    }

    fn fused_gate_up_q4k_decode(
        &self,
        norm_x: &Tensor,
        fused: &grim_backend_rocm::FusedGateUpQ4KWeights,
    ) -> Result<(Tensor, Tensor)> {
        let dev = grim_backend_rocm::RocmDevice::shared(match norm_x.device() {
            Device::Rocm(ordinal) => *ordinal,
            _ => 0,
        });
        let act = grim_backend_rocm::as_rocm(norm_x.storage().as_ref())?;
        let out = dev.launch_fused_gate_up_q4k(act, fused)?;
        let out_arc: Arc<dyn grim_tensor::BackendStorage> = Arc::from(out);
        let gate_view = grim_backend_rocm::RocmStorageView::from_offset(
            out_arc.clone(),
            0,
            fused.n_gate * 4,
            Shape::new(vec![1, fused.n_gate]),
        )?;
        let up_view = grim_backend_rocm::RocmStorageView::from_offset(
            out_arc,
            fused.n_gate * 4,
            fused.n_up * 4,
            Shape::new(vec![1, fused.n_up]),
        )?;
        Ok((
            Tensor::new(
                Arc::from(gate_view),
                Shape::new(vec![1, fused.n_gate]),
                DType::F32,
                QuantProvenance::GrimNative,
                norm_x.device().clone(),
            ),
            Tensor::new(
                Arc::from(up_view),
                Shape::new(vec![1, fused.n_up]),
                DType::F32,
                QuantProvenance::GrimNative,
                norm_x.device().clone(),
            ),
        ))
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        cache: &mut Qwen35LayerCache,
    ) -> Result<Tensor> {
        let seq_len = positions.len();
        let device = x.device().clone();

        // 1. Pre-norm
        let x_normed = self.attn_norm.forward(x)?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // Branch width is layer-type dependent. Attention layers emit
        // num_heads * head_dim; recurrent (KDA) layers emit
        // num_value_heads * d_state, which `ssm_out.weight` consumes. For
        // Qwen3.8-27B these happen to be equal (6144), but they are different
        // quantities and must not be conflated.
        let branch_width = if self.is_full_attention {
            q_dim
        } else {
            self.cfg_ssm_num_value_heads() * self.cfg_ssm_head_dim()
        };
        let mut out_branch = vec![0.0f32; seq_len * branch_width];
        // Set when a branch ran entirely on device. Then the branch output is
        // already a device tensor and must NOT be copied through `out_branch`:
        // doing so is exactly the host round-trip these paths exist to remove.
        let mut branch_dev: Option<Tensor> = None;

        if self.is_full_attention {
            if let Some(t) = attention_layer_d2d(self, &x_normed, positions, cache, seq_len)? {
                branch_dev = Some(t);
            } else {
            // TEMP probe: the attention branch aborts on a bad transfer at the
            // first attention layer, on both the D2D and host routes. Print the
            // geometry each step assumes so a wrong width is visible without
            // re-deriving it.
            if attn_shapes_debug() {
                let shp = |t: &Option<Linear>| {
                    t.as_ref()
                        .map(|l| (l.weight().shape().dims().to_vec(), l.weight().shape().elem_count()))
                };
                let nshp = |n: &Option<RmsNorm>| {
                    n.as_ref().map(|x| (x.weight.shape().dims().to_vec(), x.weight.shape().elem_count()))
                };
                eprintln!(
                    "[attn-debug] layer {} seq_len={} q_dim={q_dim} kv_dim={kv_dim} wq={:?} wk={:?} wv={:?} wo={:?} q_norm={:?} k_norm={:?} gate={:?}",
                    self.layer_idx,
                    seq_len,
                    shp(&self.wq),
                    shp(&self.wk),
                    shp(&self.wv),
                    shp(&self.wo),
                    nshp(&self.attn_q_norm),
                    nshp(&self.attn_k_norm),
                    shp(&self.attn_gate),
                );
            }
            // Attention path with separated wq, wk, wv.
            // GPU-first: projections stay on-device; RoPE runs through the device kernel; K/V are appended into the.
            let dev = pick_device_for_storage_device(&device);
            // Some TP-sharded projections emit padded rows — cut to the
            // exact [seq, width] extent via a D2D staging copy when needed.
            let exact = |t: Tensor, rows: usize, width: usize| -> Result<Tensor> {
                let want = Shape::new(vec![rows, width]);
                if t.shape().elem_count() == rows * width {
                    return crate::block::reshaped_view(&t, &want);
                }
                let scratch = dev.alloc_storage(&want, DType::F32)?;
                dev.copy_slice_range(scratch.as_ref(), 0, t.storage().as_ref(), 0, rows * width)?;
                Ok(Tensor::new(
                    scratch.into(),
                    want,
                    DType::F32,
                    t.provenance().clone(),
                    t.device().clone(),
                ))
            };

            let mut rope_cfg = grim_tensor::RopeConfig::new(self.head_dim, self.rope_theta);
            rope_cfg.rotary_dim = self.rotary_dim;
            // NeoX HALF-SPLIT pairing, not GPT-J interleaved.
            //
            // `RopeConfig::interleaved` defaults to `true` (x[2i], x[2i+1]) because
            // that is what the LFM2 family uses. Qwen3.5/3.8 does NOT: llama.cpp
            // runs it through `ggml_rope_multi`, whose NEOX/MROPE branch is
            //   `rotate_pairs<T>(n_dims, n_dims/2, cache, src, dst)`
            // (ggml/src/ggml-cpu/ops.cpp:6212), and `rotate_pairs` reads
            //   x0 = src[ic],  x1 = src[ic + n_offset]        (ops.cpp:6073-6074)
            // i.e. dim i pairs with dim i + n_dims/2 — the half-split. With
            // rotary_dim=64 that is (0,32),(1,33),...,(31,63), and dims 64..255
            // pass through unrotated.
            //
            // Leaving the default silently rotated Q/K by the wrong pairs. It is
            // invisible at position 0, where a RoPE is the identity, which is why
            // every position-0 assertion and every one-token gate passed while
            // the model was comprehensively wrong.
            rope_cfg.interleaved = false;
            let rope_ext = |t: &Tensor, heads: usize| -> Result<Tensor> {
                let mut pos_ext = Vec::with_capacity(seq_len * heads);
                for &pos in positions {
                    for _ in 0..heads {
                        pos_ext.push(pos);
                    }
                }
                let t3 = crate::block::reshaped_view(
                    t,
                    &Shape::new(vec![1, seq_len * heads, self.head_dim]),
                )?;
                if std::env::var("GRIM_DEBUG_ROPE_ISOLATE").is_ok() && heads == self.num_kv_heads {
                    let pre_rope = t.to_vec_f32()?;
                    let t3_flat = t3.to_vec_f32()?;
                    eprintln!(
                        "[rope_isolate pre] seq_len={} heads={} hd={} t_len={} t3_len={}",
                        seq_len, heads, self.head_dim, pre_rope.len(), t3_flat.len()
                    );
                    for r in 0..seq_len.min(5) {
                        let off = r * heads * self.head_dim;
                        eprintln!(
                            "[rope_isolate t_row {}] {:?} ...",
                            r,
                            &pre_rope[off..off + 4.min(pre_rope.len() - off)]
                        );
                        eprintln!(
                            "[rope_isolate t3_row {}] {:?} ...",
                            r,
                            &t3_flat[off..off + 4.min(t3_flat.len() - off)]
                        );
                    }
                }
                let (rope_s, _) =
                    dev.rope(t3.storage().as_ref(), &pos_ext, &rope_cfg, t3.shape())?;
                let roped = Tensor::new(
                    rope_s.into(),
                    t3.shape().clone(),
                    DType::F32,
                    t.provenance().clone(),
                    t.device().clone(),
                );
                let reshaped = crate::block::reshaped_view(
                    &roped,
                    &Shape::new(vec![seq_len, heads * self.head_dim]),
                )?;
                if std::env::var("GRIM_DEBUG_ROPE_ISOLATE").is_ok() && heads == self.num_kv_heads {
                    let roped_raw = roped.to_vec_f32()?;
                    let res_raw = reshaped.to_vec_f32()?;
                    for r in 0..seq_len.min(5) {
                        let off = r * heads * self.head_dim;
                        eprintln!(
                            "[rope_isolate roped_row {}] {:?} ...",
                            r,
                            &roped_raw[off..off + 4.min(roped_raw.len() - off)]
                        );
                        eprintln!(
                            "[rope_isolate res_row {}] {:?} ...",
                            r,
                            &res_raw[off..off + 4.min(res_raw.len() - off)]
                        );
                    }
                }
                Ok(reshaped)
            };

            // Phase 2b: single-token decode issues ONE fused Q8_0 QKV GEMV
            // (pre-rope) instead of 3 separate GEMVs; the same rope_ext and
            // arena-attention path follow unchanged.
            let (q_dev, k_dev_t, v_dev_t, q_gate): (Tensor, Tensor, Tensor, Option<Tensor>) =
                match self.wqkv_q80_fused.as_ref() {
                    Some(fused) if seq_len == 1 => {
                        let (q, k, v) =
                            crate::shared_attention::fused_qkv_project_raw(&x_normed, fused)?;
                        // The fused-QKV path has no separate attn_q tensor, so it
                        // has NO gate half. `None` means "do not apply a gate" —
                        // a zero tensor would be read as sigmoid(0) = 0.5 and
                        // uniformly halve every decode step.
                        (q, k, v, None)
                    }
                    _ => {
                        // attn_q emits [Q(q_dim) | gate(q_dim)] fused, so split it
                        // ROW-AWARE per token. A flat byte-range copy of the first
                        // q_dim elements is wrong for seq_len > 1: it cuts across
                        // row boundaries and scrambles Q itself, not just the gate.
                        let (q_dev, q_gate) = match self.wq.as_ref() {
                            Some(wq) => {
                                let full = wq.forward(&x_normed)?.to_vec_f32()?;
                                let wide = full.len() / seq_len.max(1);
                                let mut q_rows = vec![0.0f32; seq_len * q_dim];
                                let mut gate_rows = vec![0.0f32; seq_len * q_dim];
                                // [all Q | all gate] per row: the checkpoint's
                                // per-head interleave was permuted away at load
                                // (see the attn_q_perm construction above).
                                for t in 0..seq_len {
                                    let base = t * wide;
                                    if base + wide > full.len() {
                                        continue;
                                    }
                                    let row = &full[base..base + wide];
                                    let n = q_dim.min(wide);
                                    q_rows[t * q_dim..t * q_dim + n].copy_from_slice(&row[..n]);
                                    if wide >= 2 * q_dim {
                                        gate_rows[t * q_dim..t * q_dim + n]
                                            .copy_from_slice(&row[q_dim..q_dim + n]);
                                    }
                                }
                                (
                                    device_tensor(
                                        q_rows,
                                        Shape::new(vec![seq_len, q_dim]),
                                        &device,
                                    )?,
                                    Some(device_tensor(
                                        gate_rows,
                                        Shape::new(vec![seq_len, q_dim]),
                                        &device,
                                    )?),
                                )
                            }
                            None => (
                                Tensor::new(
                                    dev.zeros(&Shape::new(vec![seq_len, q_dim]), DType::F32)?
                                        .into(),
                                    Shape::new(vec![seq_len, q_dim]),
                                    DType::F32,
                                    x_normed.provenance().clone(),
                                    x_normed.device().clone(),
                                ),
                                // No attn_q weights at all: nothing to gate with.
                                None,
                            ),
                        };
                        let k_dev_t = match self.wk.as_ref() {
                            Some(wk) => exact(wk.forward(&x_normed)?, seq_len, kv_dim)?,
                            None => Tensor::new(
                                dev.zeros(&Shape::new(vec![seq_len, kv_dim]), DType::F32)?
                                    .into(),
                                Shape::new(vec![seq_len, kv_dim]),
                                DType::F32,
                                x_normed.provenance().clone(),
                                x_normed.device().clone(),
                            ),
                        };
                        let v_dev_t = match self.wv.as_ref() {
                            Some(wv) => exact(wv.forward(&x_normed)?, seq_len, kv_dim)?,
                            None => Tensor::new(
                                dev.zeros(&Shape::new(vec![seq_len, kv_dim]), DType::F32)?
                                    .into(),
                                Shape::new(vec![seq_len, kv_dim]),
                                DType::F32,
                                x_normed.provenance().clone(),
                                x_normed.device().clone(),
                            ),
                        };
                        (q_dev, k_dev_t, v_dev_t, q_gate)
                    }
                };

            // Q/K RMS norm, BEFORE RoPE. The reference does exactly this at
            // qwen35.cpp:281/286 with MRoPE following at line 299. Both weights
            // were previously loaded from the checkpoint and never applied, so
            // unnormalized q and k reached RoPE and the softmax. Normalizing
            // after RoPE would be a different function, not a reordering detail.
            let q_dev = match self.attn_q_norm.as_ref() {
                Some(n) => {
                    let w = n.weight.to_vec_f32()?;
                    let cur = q_dev.to_vec_f32()?;
                    let normed =
                        apply_head_rms_norm(&cur, self.num_heads, self.head_dim, &w, n.eps);
                    let sh = Shape::new(vec![seq_len, self.num_heads * self.head_dim]);
                    Tensor::new(
                        dev.from_cpu(&normed, &sh, DType::F32)?.into(),
                        sh,
                        DType::F32,
                        q_dev.provenance().clone(),
                        q_dev.device().clone(),
                    )
                }
                None => q_dev,
            };
            let k_dev_t = match self.attn_k_norm.as_ref() {
                Some(n) => {
                    let w = n.weight.to_vec_f32()?;
                    let cur = k_dev_t.to_vec_f32()?;
                    let normed =
                        apply_head_rms_norm(&cur, self.num_kv_heads, self.head_dim, &w, n.eps);
                    if std::env::var("GRIM_DEBUG_ROPE_ISOLATE").is_ok() {
                        let stride = self.num_kv_heads * self.head_dim;
                        eprintln!("[rope_isolate k_norm] seq_len={} stride={}", seq_len, stride);
                        for r in 0..seq_len.min(5) {
                            let off = r * stride;
                            eprintln!(
                                "[rope_isolate k_raw_row {}] {:?} ...",
                                r,
                                &cur[off..off + 4.min(cur.len() - off)]
                            );
                            eprintln!(
                                "[rope_isolate k_normed_row {}] {:?} ...",
                                r,
                                &normed[off..off + 4.min(normed.len() - off)]
                            );
                        }
                    }
                    let sh = Shape::new(vec![seq_len, self.num_kv_heads * self.head_dim]);
                    Tensor::new(
                        dev.from_cpu(&normed, &sh, DType::F32)?.into(),
                        sh,
                        DType::F32,
                        k_dev_t.provenance().clone(),
                        k_dev_t.device().clone(),
                    )
                }
                None => k_dev_t,
            };

            let q_rope = rope_ext(&q_dev, self.num_heads)?;
            let k_rope = rope_ext(&k_dev_t, self.num_kv_heads)?;

            // Per-step Q crosses to host for the arena attention entry
            // point; K/V history never leaves the device.
            let q_all = q_rope.to_vec_f32()?;

            // Quantized paged KV: when GRIM_KV_QUANT selects a packed format,
            // the f32 rows produced above are quantized on the host, uploaded
            // into paged buffers, and read back through the quant paged kernel.
            // The dense f32 arena below is the default and is left untouched.
            if let Some(fmt) = kv_quant_format() {
                return crate::shared_attention::fused_or_scalar_attention_paged_quant(
                    &q_all,
                    k_dev_t.storage().as_ref(),
                    v_dev_t.storage().as_ref(),
                    cache,
                    fmt,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    seq_len,
                    // Use the device the K rows actually live on, not the
                    // layer's declared device: the packed-page append and the
                    // paged kernel both need the concrete ROCm device, and a
                    // mismatch silently resolves to a backend without the
                    // byte-copy primitive.
                    k_dev_t.device(),
                );
            }

            // Append to the device arena via copy_slice_range (device-side).
            // K/V history stays on the GPU — never round-trips to host.
            let k_new_rows = seq_len * self.num_kv_heads;
            let kv_elems = k_new_rows * self.head_dim;
            let k_cap_rows = cache
                .k_device
                .as_ref()
                .map(|s| s.shape().dims()[0])
                .unwrap_or(0);
            let v_cap_rows = cache
                .v_device
                .as_ref()
                .map(|s| s.shape().dims()[0])
                .unwrap_or(0);
            let need_rows = cache.current_pos + k_new_rows;
            let _ = v_cap_rows;

            if k_cap_rows >= need_rows {
                let k_dev = cache.k_device.as_ref().ok_or_else(|| {
                    grim_core::error::Error::Backend("cache.k_device missing".into())
                })?;
                let v_dev = cache.v_device.as_ref().ok_or_else(|| {
                    grim_core::error::Error::Backend("cache.v_device missing".into())
                })?;
                dev.copy_slice_range(
                    &**k_dev,
                    cache.current_pos * self.num_kv_heads * self.head_dim,
                    k_rope.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
                if std::env::var("GRIM_DEBUG_ROPE_ISOLATE").is_ok() {
                    let arena_k = k_dev.to_cpu_vec_f32()?;
                    let stride = self.head_dim;
                    eprintln!("[rope_isolate arena_dump] current_pos={} k_cap_rows={}", cache.current_pos, k_cap_rows);
                    for r in 0..(cache.current_pos + k_new_rows).min(5) {
                        let off = r * stride;
                        eprintln!(
                            "[rope_isolate arena_row {}] {:?} ...",
                            r,
                            &arena_k[off..off + 4.min(arena_k.len().saturating_sub(off))]
                        );
                    }
                }
                dev.copy_slice_range(
                    &**v_dev,
                    cache.current_pos * self.num_kv_heads * self.head_dim,
                    v_dev_t.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
            } else {
                // Arena full — grow geometrically and re-copy via D2D.
                let new_rows = ((need_rows * 2) + 64).next_power_of_two();
                let k_idx = self.num_kv_heads * self.head_dim;
                let full_shape = Shape::new(vec![new_rows, self.num_kv_heads, self.head_dim]);
                let k_grown = dev.alloc_storage(&full_shape, DType::F32)?;
                let v_grown = dev.alloc_storage(&full_shape, DType::F32)?;
                if let Some(ref old_k) = cache.k_device {
                    dev.copy_slice_range(
                        k_grown.as_ref(),
                        0,
                        old_k.as_ref(),
                        0,
                        cache.current_pos * self.num_kv_heads * self.head_dim,
                    )?;
                }
                if let Some(ref old_v) = cache.v_device {
                    dev.copy_slice_range(
                        v_grown.as_ref(),
                        0,
                        old_v.as_ref(),
                        0,
                        cache.current_pos * self.num_kv_heads * self.head_dim,
                    )?;
                }
                dev.copy_slice_range(
                    k_grown.as_ref(),
                    cache.current_pos * k_idx,
                    k_rope.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
                dev.copy_slice_range(
                    v_grown.as_ref(),
                    cache.current_pos * k_idx,
                    v_dev_t.storage().as_ref(),
                    0,
                    kv_elems,
                )?;
                cache.k_device = Some(k_grown);
                cache.v_device = Some(v_grown);
            }

            let total_kv = cache.current_pos + seq_len;
            let k_dev = cache
                .k_device
                .as_ref()
                .ok_or_else(|| grim_core::error::Error::Backend("cache.k_device missing".into()))?;
            let v_dev = cache
                .v_device
                .as_ref()
                .ok_or_else(|| grim_core::error::Error::Backend("cache.v_device missing".into()))?;
            let attn_tensor = crate::shared_attention::fused_or_scalar_attention_arena(
                &q_all,
                k_dev.as_ref(),
                v_dev.as_ref(),
                total_kv,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                seq_len,
                None,
                &device,
            )?;
            let mut attn_out = attn_tensor.to_vec_f32()?;
            // Output gate: attn_out * sigmoid(attn_q's gate half), per
            // llama.cpp qwen35.cpp:
            //   gate = ggml_sigmoid(gate_view)
            //   cur  = ggml_mul(cur, gate_sigmoid)   // "attn_gated"
            // applied BEFORE `wo`. Without this the gate half of attn_q is
            // computed and discarded.
            if let Some(ref gate_t) = q_gate {
                let g = gate_t.to_vec_f32()?;
                for i in 0..attn_out.len().min(g.len()) {
                    attn_out[i] *= 1.0 / (1.0 + (-g[i]).exp());
                }
            }
            out_branch = attn_out;
            }
        } else {
            // Gated DeltaNet recurrence. Prefer the device path, which keeps
            // the conv and recurrent state in VRAM across tokens; it declines
            // (without touching state) for prefill or any missing weight, in
            // which case the host reference runs instead.
            let on_device =
                gated_delta_net_forward_d2d(self, cache, &x_normed, seq_len, branch_width)?;
            match on_device {
                Some(t) => branch_dev = Some(t),
                None => {
                    gated_delta_net_forward(
                        self,
                        cache,
                        &x_normed,
                        &mut out_branch,
                        seq_len,
                        branch_width,
                    )?;
                }
            }
        }

        // A device-resident branch already carries its own output gate (the
        // attention path applies `A * sigmoid(q_gate)` on device, the recurrent
        // path folds z into the KDA kernel), so the host sigmoid gate below
        // applies only to a branch that is still in host memory. Applying it
        // twice would gate an already-gated value.
        let branch_tensor = match branch_dev {
            Some(t) => t,
            None => {
                // The sigmoid output gate is the ATTENTION layer's, and only
                // the attention layer's. The reference applies it inside the
                // attention branch, before `wo`:
                //
                //   llama.cpp qwen35.cpp:323-327
                //     gate_sigmoid = ggml_sigmoid(gate);  cur = ggml_mul(cur, gate_sigmoid)
                //
                // A recurrent layer gates differently and exactly once, with
                // `silu(z)`, inside `build_norm_gated` (qwen35.cpp:454). Since
                // `attn_gate` is the tensor that supplies z on a recurrent
                // layer, gating it here as well applied BOTH gates to the 49
                // recurrent layers of a 65-layer model. The condition below
                // restores the reference's one-gate-per-layer-type behaviour.
                if self.is_full_attention {
                    if let Some(ref gate_lin) = self.attn_gate {
                        let gate_tensor = gate_lin.forward(&x_normed)?;
                        let gate_vec = gate_tensor.to_vec_f32()?;
                        let gate_len_per_tok = gate_vec.len() / seq_len.max(1);
                        for t in 0..seq_len {
                            let gate_base = t * gate_len_per_tok;
                            let out_base = t * branch_width;
                            for d in 0..branch_width.min(gate_len_per_tok) {
                                let g = gate_vec[gate_base + d];
                                out_branch[out_base + d] *= 1.0 / (1.0 + (-g).exp()); // sigmoid
                            }
                        }
                    }
                }
                device_tensor(out_branch, Shape::new(vec![seq_len, branch_width]), &device)?
            }
        };

        let proj_out = if let Some(ref wo) = self.wo {
            wo.forward(&branch_tensor)?
        } else if let Some(ref out_proj) = self.ssm_out {
            out_proj.forward(&branch_tensor)?
        } else {
            branch_tensor
        };

        // Residual 1
        let h = grim_nn::modules::add_on_device(x, &proj_out)?;

        // 3. Post-attention norm
        let h_normed = self.post_attention_norm.forward(&h)?;

        // 4. SwiGLU FFN
        let (gate, up) = match self.w_gate_up_q4k_fused.as_ref() {
            Some(fused) if seq_len == 1 => self.fused_gate_up_q4k_decode(&h_normed, fused)?,
            _ => (
                self.ffn_gate.forward(&h_normed)?,
                self.ffn_up.forward(&h_normed)?,
            ),
        };
        let act = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let ffn_out = self.ffn_down.forward(&act)?;

        // Residual 2
        let out = grim_nn::modules::add_on_device(&h, &ffn_out)?;
        Ok(out)
    }
}

// Full Model

pub struct Qwen35 {
    pub cfg: Qwen35Config,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub blocks: Vec<Qwen35Block>,
    pub output_norm: RmsNorm,
    pub output: Linear,
}

impl Qwen35 {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: Qwen35Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: Qwen35Config,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let available_devices = if !cfg.devices.is_empty() {
            cfg.devices.clone()
        } else {
            vec![device.clone()]
        };

        eprintln!(
            "[grim] Initializing Qwen3.5/3.8 hybrid model: layers={}, hidden={}, vocab={}, interval={}, devices={:?}",
            cfg.num_layers,
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.full_attention_interval,
            available_devices
        );

        let first_device = available_devices[0].clone();
        let tok_embeddings = Embedding::load(
            &ws.with_device(first_device.clone()).pp("token_embd"),
            cfg.vocab_size,
            cfg.hidden_size,
        )
        .or_else(|e| {
            fallback_on_missing(e, || {
                Embedding::load(
                    &ws.with_device(first_device).pp("tok_embeddings"),
                    cfg.vocab_size,
                    cfg.hidden_size,
                )
            })
        })?;

        let layer_devices = plan_layer_devices(
            ws,
            &available_devices,
            cfg.num_layers,
            cfg.max_seq_len,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.full_attention_interval,
        );

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_device = layer_devices[i].clone();
            if i % 10 == 0 || i + 1 == cfg.num_layers {
                eprintln!(
                    "[grim] Loading layer {}/{} on {}...",
                    i + 1,
                    cfg.num_layers,
                    layer_device
                );
            }
            let layer_ws = ws.with_device(layer_device).pp("blk").pp(&i.to_string());
            blocks.push(Qwen35Block::load_tp(&layer_ws, &cfg, i, tp)?);
        }

        let last_device = available_devices.last().unwrap_or(&device).clone();
        let output_norm = RmsNorm::load(
            &ws.with_device(last_device.clone()).pp("output_norm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )
        .or_else(|e| {
            fallback_on_missing(e, || {
                RmsNorm::load(
                    &ws.with_device(last_device.clone()).pp("norm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                )
            })
        })?;

        let output = Linear::load_column_parallel(
            &ws.with_device(last_device.clone()).pp("output"),
            cfg.hidden_size,
            cfg.vocab_size,
            false,
            tp,
        )
        .or_else(|e| {
            fallback_on_missing(e, || {
                Linear::load(
                    &ws.with_device(last_device).pp("output"),
                    cfg.hidden_size,
                    cfg.vocab_size,
                    false,
                )
            })
        })?;
        // Dump the allocation ledger when asked.
        //
        // This must run AFTER the weights exist, not during planning: a dump
        // taken inside plan_layer_devices captured 0 allocations because no
        // weight had been uploaded yet, which is the one moment the ledger is
        // useless. A page fault kills the process, so this file is the only
        // copy that outlives the crash and the only way to attribute the
        // faulting address afterwards.
        if let Ok(path) = std::env::var("GRIM_LEDGER_DUMP") {
            match grim_backend_rocm::memory::ledger::dump_to_file(&path) {
                Ok(n) => eprintln!(
                    "[qwen35] wrote {n} live allocation(s) to {path} for post-mortem attribution"
                ),
                Err(e) => eprintln!("[qwen35] WARNING: ledger dump to {path} failed: {e}"),
            }
        }

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            blocks,
            output_norm,
            output,
        })
    }
}

impl Model for Qwen35 {
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

impl CausalLm for Qwen35 {
    fn new_session(&self) -> Box<dyn SessionT> {
        let caches: Vec<Qwen35LayerCache> = (0..self.blocks.len())
            .map(|_| Qwen35LayerCache::new(&self.cfg))
            .collect();
        let mut session = Inner::new(self.device.clone());
        session.set_model_state(Box::new(caches));
        Box::new(session)
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
        let positions_vec: Vec<u32> = match positions.dtype() {
            d if d == DType::F32 => {
                let v = positions.to_vec_f32()?;
                v.into_iter().map(|x| x as u32).collect()
            }
            _ => return Err(grim_tensor::Error::Unimplemented("non-F32 positions".into()).into()),
        };

        // TEMP probe: every gate in this repo hands `Qwen35Block::forward` its
        // own explicit `positions`, so none of them can see whether the ENGINE
        // builds the right ones. A prefill that passes all zeros is invisible to
        // every layer test (RoPE at position 0 is the identity) and makes the
        // model confidently wrong from the first real position on.
        if std::env::var("GRIM_DEBUG_POSITIONS").is_ok() {
            eprintln!(
                "[positions] seq_len={} positions={:?} ids_len={}",
                positions_vec.len(),
                if positions_vec.len() <= 16 { positions_vec.clone() } else { positions_vec[..16].to_vec() },
                ids.len()
            );
        }

        let seq_len = ids.len();
        let mut h = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?;

        if session.model_state().is_none() {
            let fresh: Vec<Qwen35LayerCache> = (0..self.blocks.len())
                .map(|_| Qwen35LayerCache::new(&self.cfg))
                .collect();
            session.set_model_state(Box::new(fresh));
        }

        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Qwen35LayerCache>>())
            .ok_or_else(|| {
                grim_core::error::Error::Backend(
                    "Qwen35::forward: session.model_state must be Vec<Qwen35LayerCache>".into(),
                )
            })?;

        // Per-layer trace, off by default. A device page fault kills the
        // process with no stack, so the only way to learn which layer faulted
        // is to say so on the way in: the last line printed is the culprit.
        let trace = std::env::var("GRIM_TRACE_LAYERS").is_ok();
        if trace {
            eprintln!(
                "[qwen35-trace] {} layers, entering forward on {}",
                self.blocks.len(),
                h.device()
            );
        }
        for (i, block) in self.blocks.iter().enumerate() {
            if h.device() != &block.device {
                h = grim_nn::modules::move_to_device(&h, &block.device)?;
            }
            if trace {
                let kind = if block.is_full_attention {
                    "attn"
                } else {
                    "kda"
                };
                eprintln!("[qwen35-trace] layer {i} ({kind}) on {}", block.device);
            }
            h = block.forward(&h, &positions_vec, &mut caches[i])?;
            caches[i].current_pos += seq_len;
        }
        if trace {
            eprintln!("[qwen35-trace] all layers done, entering output norm");
        }

        if h.device() != self.output_norm.weight.device() {
            h = grim_nn::modules::move_to_device(&h, self.output_norm.weight.device())?;
        }

        let normed = self.output_norm.forward(&h)?;
        let logits = self.output.forward(&normed)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

/// Gated DeltaNet forward for one recurrent block.
///
/// Extracted from `Qwen35Block::forward` so the recurrence can be exercised and
/// reasoned about on its own. Replaces a depthwise short-conv -> SiLU ->
/// sigmoid-gate stub that ignored `ssm_a` / `ssm_alpha` / `ssm_beta` /
/// `ssm_dt_bias` / `ssm_norm` entirely and never touched `cache.ssm_state` —
/// with 49 of 65 layers running that stub, the residual stream diverged from
/// the trained distribution immediately.
///
/// Geometry is measured from the GGUF, not from published docs:
/// `ssm_d_inner = 6144 = 48 value heads x 128 head_dim`, `ssm_n_group = 16`
/// key heads (a 3:1 ratio), and `ssm_alpha` / `ssm_beta` project to 48 — one
/// scalar per VALUE head, already expanded in the file.
#[allow(clippy::too_many_arguments)]
/// Per-head L2 normalization of the query and key, applied to BOTH before the
/// gated delta rule runs.
///
/// The reference does this at `llama.cpp` `models.h: build_gdn_l2_norm` and
/// calls it on `q_conv` and `k_conv` immediately after they are split out of
/// the conv output:
///
///     ggml_scale(ggml_rms_norm(x, eps/n), 1.0f/sqrtf(n))
///
/// which reduces to `x / sqrt(sum(x^2) + eps)` - a true L2 norm. The scaling
/// trick is what keeps it correct in f16 without a power operation.
///
/// This step was missing from grim entirely; the only trace of it was a doc
/// comment quoting the neighbouring `ggml_repeat_4d` line. It matters because
/// the recurrence's `k.S` prediction and its `q.S_new` output both scale
/// directly with the magnitude of these vectors, so skipping the norm feeds
/// the delta rule vectors that are orders of magnitude off, and the error
/// compounds across all the KDA layers. A synthetic repro with constant 0.1
/// weights cannot see this, which is why it survived.
/// KDA value width: `num_value_heads * head_dim`, i.e. `ssm_d_inner`.
///
/// `attn_gate` is the KDA output gate on linear-attention layers and the
/// attention output gate on full-attention layers, so a single loaded Linear
/// has to cover the wider of `q_dim` and this. For the 27B both are 6144; for
/// the 9B both are 4096. Deriving it keeps the loader correct on a checkpoint
/// whose geometry differs, instead of relying on a literal that happens to
/// match one of them.
fn blk_value_dim(cfg: &Qwen35Config) -> usize {
    // The reference sets `head_v_dim = d_inner / num_v_heads`, which equals
    // `ssm_d_state` for this family: 48x128=6144 (27B) and 32x128=4096 (9B).
    cfg.ssm_dt_rank.max(1) * cfg.ssm_d_state.max(1)
}

/// Per-head Q/K RMS norm, applied BEFORE MRoPE.
///
/// `llama.cpp` qwen35.cpp normalizes Q and K with `attn_q_norm` /
/// `attn_k_norm` at lines 281 and 286, and only then applies MRoPE at line 299.
/// grim loaded both weights from the checkpoint and never applied them at all,
/// so unnormalized q and k went into RoPE and into the softmax. Normalizing
/// after RoPE instead would be a different function, not a reordering detail.
///
/// `x` is `[seq_len, n_heads * head_dim]`; the norm is per head, over `head_dim`.
fn apply_head_rms_norm(
    x: &[f32],
    n_heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    let row_stride = n_heads.max(1) * head_dim.max(1);
    if row_stride == 0 || x.is_empty() {
        return x.to_vec();
    }
    let mut out = x.to_vec();
    for seq in 0..(x.len() / row_stride) {
        for h in 0..n_heads {
            let base = seq * row_stride + h * head_dim;
            let ss: f32 = out[base..base + head_dim].iter().map(|v| v * v).sum();
            let inv = 1.0 / ((ss / head_dim as f32) + eps).sqrt();
            for i in 0..head_dim {
                out[base + i] = out[base + i] * inv * weight.get(i).copied().unwrap_or(1.0);
            }
        }
    }
    out
}

fn gdn_l2_norm(v: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = v.iter().map(|x| x * x).sum();
    let denom = (ss + eps).sqrt();
    if denom <= 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / denom).collect()
}

fn gated_delta_net_forward(
    blk: &Qwen35Block,
    cache: &mut Qwen35LayerCache,
    x_normed: &Tensor,
    out_branch: &mut [f32],
    seq_len: usize,
    branch_width: usize,
) -> Result<()> {
    let n_val_heads = blk.cfg_ssm_num_value_heads();
    let n_key_heads = blk.cfg_ssm_num_key_heads();
    let head_dim = blk.cfg_ssm_head_dim();
    // The reference feeds `hparams.f_norm_rms_eps` to the GDN L2 norm, and
    // `attn_norm` is built from `cfg.rms_norm_eps`, so this is the same value.
    let gdn_eps = blk.attn_norm.eps;
    let ktrace = std::env::var("GRIM_TRACE_KDA").is_ok();
    if ktrace {
        eprintln!(
            "[kda-trace] enter seq={seq_len} val_heads={n_val_heads} key_heads={n_key_heads} \
             head_dim={head_dim} state_len={} branch_width={branch_width}",
            blk.cfg_ssm_d_state().max(1) * head_dim
        );
    }
    if n_val_heads == 0 || n_key_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    let Some(ref qkv_lin) = blk.attn_qkv else {
        return Ok(());
    };

    let qkv = qkv_lin.forward(x_normed)?;
    let qkv_vec = qkv.to_vec_f32()?;
    let per_tok = qkv_vec.len() / seq_len.max(1);

    // alpha / beta -> one scalar per value head per token.
    let mut alpha = vec![0.0f32; n_val_heads * seq_len];
    if let Some(ref al) = blk.ssm_alpha {
        let v = al.forward(x_normed)?.to_vec_f32()?;
        let stride = v.len() / seq_len.max(1);
        for t in 0..seq_len {
            for h in 0..n_val_heads.min(stride) {
                alpha[t * n_val_heads + h] = v[t * stride + h];
            }
        }
    }
    // beta = sigmoid(ssm_beta(x)) per llama.cpp qwen35.cpp: `beta = ggml_sigmoid(...)`.
    // It is a per-value-head gate in (0,1), not a raw projection.
    let mut beta = vec![0.0f32; n_val_heads * seq_len];
    if let Some(ref bl) = blk.ssm_beta {
        let v = bl.forward(x_normed)?.to_vec_f32()?;
        let stride = v.len() / seq_len.max(1);
        for t in 0..seq_len {
            for h in 0..n_val_heads.min(stride) {
                beta[t * n_val_heads + h] = 1.0 / (1.0 + (-v[t * stride + h]).exp());
            }
        }
    }
    let a_vec: &[f32] = blk.ssm_a.as_deref().unwrap_or(&[]);

    // z (the KDA output gate), from the checkpoint's `attn_gate` projection.
    //
    // The reference splits z out of the fused qkvz tensor and applies
    // `build_norm_gated` = rms_norm(core) * silu(z) (qwen35.cpp:243-252).
    // config.json names the activation `output_gate_type: "swish"`, and the
    // GGUF gives attn_gate.weight as [6144, 5120] = [value_dim, hidden] on
    // EVERY layer - so on linear-attention layers this tensor IS z, laid out
    // [n_val_heads, head_dim]. On full-attention layers the same tensor is the
    // sigmoid attention gate, which grim already applies.
    //
    // Without it the recurrent output reaches ssm_out ungated: multiplicative
    // and per-dimension, so its absence does not blow up numerically, it just
    // makes the model confidently wrong.
    let mut z_vec: Vec<f32> = Vec::new();
    if let Some(ref gl) = blk.attn_gate {
        if std::env::var("GRIM_SKIP_Z").is_err() {
            z_vec = gl.forward(x_normed)?.to_vec_f32()?;
        }
    }
    let z_stride = if z_vec.is_empty() {
        0
    } else {
        z_vec.len() / seq_len.max(1)
    };
    let silu = |v: f32| v / (1.0 + (-v).exp());

    let dt_bias: &[f32] = blk.ssm_dt_bias.as_deref().unwrap_or(&[]);
    let norm_vec: &[f32] = blk.ssm_norm.as_deref().unwrap_or(&[]);
    let values_per_group = (n_val_heads / n_key_heads).max(1);

    // Fused `attn_qkv` layout, per llama.cpp `src/models/qwen35.cpp`:
    //
    //   head_k_dim = head_v_dim = ssm_d_state
    //   n_k_heads  = ssm_n_group    (16)
    //   n_v_heads  = ssm_dt_rank    (48)
    //   key_dim    = head_k * n_k   (2048)
    //   value_dim  = head_v * n_v   (6144)
    //   conv_dim   = key_dim * 2 + value_dim  (10240)
    //
    // so the stream is [q key_dim][k key_dim][V value_dim] — query, key, value.
    // q and k are both key-width and both tile from n_k_heads to n_v_heads
    // (`ggml_repeat_4d` in the reference) when the counts differ; v is
    // value-width and already per value head.
    //
    // A previous version of this derived [q 48*128][k 16*128][v 16*128], which
    // also sums to 10240 and so passed a sum-only check while reading the wrong
    // channels. These asserts pin each stream's width separately.
    let key_dim = n_key_heads * head_dim;
    let value_dim = n_val_heads * head_dim;
    debug_assert_eq!(
        2 * key_dim + value_dim,
        per_tok,
        "fused attn_qkv width should be 2*key_dim + value_dim; got key={key_dim} \
         value={value_dim} total={} per_tok={per_tok}",
        2 * key_dim + value_dim
    );
    debug_assert_eq!(
        value_dim, branch_width,
        "recurrent branch width must equal value_dim"
    );

    // Per-head state [n_val_heads][d_k][d_v]. cache.ssm_state is allocated as
    // n_group * d_state * (d_inner / n_group) = 48*128*128, exactly this shape
    // for square d_k = d_v = 128.
    let state_len = blk.cfg_ssm_d_state().max(1) * head_dim;
    if cache.ssm_state.len() < n_val_heads * state_len {
        cache.ssm_state.resize(n_val_heads * state_len, 0.0);
    }

    // Causal short conv + SiLU over the FUSED stream, before the q/k/v split
    // (llama.cpp qwen35.cpp: `ggml_ssm_conv` then `ggml_silu`, and the views for
    // q/k/v are taken from `conv_qkv_mix`). The conv is depthwise with
    // d_conv taps and carries (d_conv-1) previous inputs per channel.
    //
    // `ssm_conv1d.weight` is [10240, 4] and the conv state is
    // (d_conv-1) * conv_dim; if this checkpoint has no conv weights the stream is
    // used as-is, which is the only correct fallback (a stub conv would feed the
    // recurrence pre-convolution projections).
    let dev = pick_device_for_storage_device(qkv.device());
    let qkv_conv = match blk.ssm_conv1d.as_ref() {
        Some(_) => {
            let taps = blk.cfg_ssm_d_conv().max(1);
            let chans = per_tok;
            let x = Tensor::new(
                std::sync::Arc::from(dev.from_cpu(
                    &qkv_vec,
                    &Shape::new(vec![1, seq_len, chans]),
                    DType::F32,
                )?),
                Shape::new(vec![1, seq_len, chans]),
                DType::F32,
                qkv.provenance().clone(),
                qkv.device().clone(),
            );
            let w = blk.ssm_conv1d.as_ref().unwrap().clone();
            // `ssm_conv1d.weight` is stored ggml-ne order, so ne[0] IS the tap
            // count and ne[1] the channel count — llama.cpp `ggml_ssm_conv`:
            //   const int64_t d_conv  = c->ne[0];
            //   const int64_t d_inner = c->ne[1];
            // Element (tap, channel) is therefore at `tap + d_conv*channel`,
            // which is row-major [channel, tap] — exactly what the tensor here
            // is shaped as. A transpose here is a REGRESSION, not a fix: the
            // 9B file's ne0 is 4 and its ne1 is 8192.
            let w_raw = w.to_vec_f32()?;
            let mut w_chw = vec![0.0f32; chans * taps];
            if w_raw.len() >= chans * taps {
                for k in 0..taps {
                    for ch in 0..chans {
                        w_chw[ch * taps + k] = w_raw[k + taps * ch];
                    }
                }
            } else {
                w_chw.copy_from_slice(&w_raw);
            }
            let w_t = Tensor::new(
                std::sync::Arc::from(dev.from_cpu(
                    &w_chw,
                    &Shape::new(vec![chans, taps]),
                    DType::F32,
                )?),
                Shape::new(vec![chans, taps]),
                DType::F32,
                w.provenance().clone(),
                w.device().clone(),
            );
            // conv_state is [1, chans*(taps-1)] flat; reshape to [1, taps-1, chans].
            let need = (taps - 1) * chans;
            if cache.conv_state.len() < need {
                cache.conv_state.resize(need, 0.0);
            }
            let state_flat = cache.conv_state[..need].to_vec();
            let mut state_t = Tensor::new(
                std::sync::Arc::from(dev.from_cpu(
                    &state_flat,
                    &Shape::new(vec![1, taps - 1, chans]),
                    DType::F32,
                )?),
                Shape::new(vec![1, taps - 1, chans]),
                DType::F32,
                qkv.provenance().clone(),
                qkv.device().clone(),
            );
            let out = grim_nn::modules::short_conv1d(&x, &w_t, None, Some(&mut state_t))?;
            // Write the updated history back.
            let updated = state_t.to_vec_f32()?;
            cache.conv_state[..need].copy_from_slice(&updated[..need]);
            out.to_vec_f32()?
        }
        None => qkv_vec.clone(),
    };
    // SiLU on the convolved stream, per the reference.
    let conv_mix: Vec<f32> = qkv_conv.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    {
        static PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if std::env::var("GRIM_KDA_STEP_PROBE").as_deref() == Ok("1")
            && blk.layer_idx == 0
            && seq_len == 1
            && !PROBED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            if let Some(ref wt) = blk.ssm_conv1d {
                let wv = wt.to_vec_f32().unwrap_or_default();
                let bytes: Vec<u8> = wv.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write("/tmp/kda_probe_host_conv_w.bin", &bytes);
                eprintln!("[kda-probe-host] conv_w dims {:?} wrote", wt.shape().dims());
            }
        }
    }
    // KDA-STEP-PROBE (host side): dump layer 0 step-1 conv stream + ring for
    // the cross-process diff against the D2D run.
    {
        static PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if std::env::var("GRIM_KDA_STEP_PROBE").as_deref() == Ok("1")
            && blk.layer_idx == 0
            && seq_len == 1
            && !PROBED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            let dump = |vals: &[f32], name: &str| {
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                let path = format!("/tmp/kda_probe_host_{name}.bin");
                if std::fs::write(&path, &bytes).is_ok() {
                    eprintln!("[kda-probe-host] wrote {} ({} f32) -> {path}", name, vals.len());
                }
            };
            dump(&conv_mix, "conv_mix");
            dump(&cache.conv_state, "conv_ring");
            dump(&cache.ssm_state, "ssm_pre");
            dump(&qkv_vec, "conv_x");
            if let Some(ref al) = blk.ssm_alpha {
                dump(&al.forward(x_normed)?.to_vec_f32()?, "alpha_full");
            }
            if let Some(ref bl) = blk.ssm_beta {
                let raw = bl.forward(x_normed)?.to_vec_f32()?;
                dump(&raw, "beta_raw_full");
                dump(&beta, "beta_full"); // host: sigmoided at fill
            }
            if let Some(ref gl) = blk.attn_gate {
                dump(&gl.forward(x_normed)?.to_vec_f32()?, "z_full");
            }
        }
    }

    let slice = |off: usize, len: usize| -> &[f32] {
        let end = (off + len).min(conv_mix.len());
        if off >= end { &[] } else { &conv_mix[off..end] }
    };

    for t in 0..seq_len {
        let base = t * per_tok;
        for h in 0..n_val_heads {
            let kh = kda_key_head(h, n_key_heads, values_per_group);
            // The conv stream is [q | k | v]. q and k are each num_key_heads
            // wide and tile out to num_value_heads (llama.cpp's
            // `ggml_repeat_4d`, and the same `h % n_key_heads` mapping `kh`
            // already encodes); v is num_value_heads wide.
            //   llama.cpp qwen35.cpp:404-424 — q_conv at byte offset 0, k_conv
            //   at key_dim, v_conv at 2*key_dim.
            //   vLLM qwen_gdn_linear_attn.py:704 — "Qwen3.5: weights are in
            //   [q, k, v, z] order".
            // The previous code read section 0 as k and shared section 2
            // between q and v, so the recurrence was driven by the query and
            // the output projection used the values. Every width still summed
            // to conv_dim, which is why only a sum check let it through.
            let q_off = base + kh * head_dim;
            let k_off = base + key_dim + kh * head_dim;
            let v_off = base + 2 * key_dim + h * head_dim;

            // gate = softplus(alpha + dt_bias) * ssm_a, per llama.cpp qwen35.cpp:
            //   alpha_biased   = alpha + ssm_dt
            //   alpha_softplus = softplus(alpha_biased)
            //   gate           = alpha_softplus * ssm_a
            // ssm_a MULTIPLIES after the softplus; it is not added inside it.
            // Adding it before would be a different function, since softplus is
            // nonlinear.
            let mut z = alpha[t * n_val_heads + h];
            if let Some(&d) = dt_bias.get(h) {
                z += d;
            }
            let mut gate = z.softplus();
            if let Some(&a) = a_vec.get(h) {
                gate *= a;
            }
            let beta_t = beta[t * n_val_heads + h];

            let st_off = h * state_len;
            let head_state = &mut cache.ssm_state[st_off..st_off + state_len];
            // L2-normalize the key per head BEFORE the recurrence, as the
            // reference does. Without it `k . S` is scaled by the raw key
            // magnitude instead of its direction.
            let k_l2 = gdn_l2_norm(slice(k_off, head_dim), gdn_eps);
            kda_gated_delta_rule_row(
                &k_l2,
                slice(v_off, head_dim),
                beta_t,
                gate,
                head_state,
                head_dim,
                head_dim,
            );

            // out = q . S_new, read from the state the update just wrote.
            // Previously this emitted `q[d] * norm[d]`, a static elementwise
            // scale of the raw query with zero dependence on k, v, beta, gate or
            // any recurrent history — the state was computed and then ignored.
            // The query is L2-normalized the same way, so the output depends on the
            // query DIRECTION rather than its magnitude.
            let q_slice = gdn_l2_norm(slice(q_off, head_dim), gdn_eps);
            // Pass 1: the raw head output, out[i] = q . S_new[i,:].
            let mut acc = vec![0.0f32; head_dim];
            // The reference scales the head output by 1/sqrt(S_v) before the
            // gated norm: `const float scale = 1.0f / sqrtf((float) S_v);` and
            // `attn_data[col] = attn_col * scale` in llama.cpp
            // `gated_delta_net.cu:281`. The chunked path folds the same factor
            // into the query instead (`delta-net-base.cpp:47`). The RMS norm
            // below cancels most of it, but not exactly, because of its eps.
            let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
            for (i, slot) in acc.iter_mut().enumerate() {
                let row = &cache.ssm_state[st_off + i * head_dim..st_off + (i + 1) * head_dim];
                let mut a = 0.0f32;
                for (j, qj) in q_slice.iter().enumerate() {
                    a += *qj * row[j];
                }
                *slot = a * inv_sqrt_d;
            }
            // Pass 2: RMS normalize the head output BEFORE the weight.
            //
            // The reference does `build_norm(input, weights, nullptr,
            // LLM_NORM_RMS, layer)` then multiplies by the gate
            // (qwen35.cpp:243-252). grim applied the norm WEIGHT alone, so the
            // recurrent output reached `ssm_out` at whatever magnitude the
            // state happened to hold - unbounded, and different per head, per
            // layer and per token. The projection expects a normalized input.
            //
            // The norm is over the whole head (all `head_dim` elements), not
            // per element, which is why this cannot be folded into pass 1.
            let ss: f32 = acc.iter().map(|a| a * a).sum();
            let inv_rms = 1.0 / ((ss / head_dim as f32) + gdn_eps).sqrt();
            for (i, a) in acc.iter().enumerate() {
                let w = norm_vec.get(i).copied().unwrap_or(1.0);
                let out_idx = t * branch_width + h * head_dim + i;
                if out_idx < out_branch.len() {
                    // ...then multiply by silu(z), the reference ordering:
                    // build_norm first, then ggml_mul with silu(gate).
                    let g = if z_stride > 0 {
                        let zi = t * z_stride + h * head_dim + i;
                        silu(z_vec.get(zi).copied().unwrap_or(0.0))
                    } else {
                        1.0
                    };
                    out_branch[out_idx] = a * inv_rms * w * g;
                }
            }
        }
    }
    if blk.layer_idx == 0 {
        kda_probe_host_post(&cache.ssm_state, seq_len);
    }
    if std::env::var("GRIM_KDA_STEP_PROBE").as_deref() == Ok("1") && seq_len == 1 {
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if call <= 6 {
            let s8: f32 = out_branch.iter().take(8).sum();
            let s: f32 = out_branch.iter().sum();
            eprintln!(
                "[kda-branch-fp] host decode-call {call} layer {} branch_sum8={s8:+.6} branch_sum={s:+.6}",
                blk.layer_idx
            );
        }
    }
    Ok(())
}

// KDA-STEP-PROBE (host side): post-recurrence state dump for layer 0 step 1.
fn kda_probe_host_post(ssm_state: &[f32], seq_len: usize) {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if std::env::var("GRIM_KDA_STEP_PROBE").as_deref() == Ok("1") && call <= 4 {
        let bytes: Vec<u8> = ssm_state.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(format!("/tmp/kda_probe_host_ssm_call{call}_seq{seq_len}.bin"), &bytes);
        eprintln!("[kda-probe-host] call {call} seq_len {seq_len}: dumped state ({} f32)", ssm_state.len());
    }
}

/// KDA decode step with the short-conv and recurrent state left on the device.
///
/// Returns `Ok(false)` without touching any state when this path does not
/// apply, so the caller can fall back to the host reference. The two are
/// alternatives, not a pair: whichever runs owns the state, and mixing them
/// would read a state the other one wrote.
///
/// Why this exists: the host path keeps `conv_state`/`ssm_state` in `Vec<f32>`,
/// so every recurrent layer copies ~3.1 MB of state to the device and back on
/// every token, and the delta rule itself runs on the CPU. For 49 of 65 layers
/// that is the bulk of the decode traffic, and it is also what stops the decode
/// graph from being capturable.
///
/// Decode only (`seq_len == 1`). Prefill walks the sequence in order with a
/// different state update, so it still uses the host reference.
fn gated_delta_net_forward_d2d(
    blk: &Qwen35Block,
    cache: &mut Qwen35LayerCache,
    x_normed: &Tensor,
    seq_len: usize,
    branch_width: usize,
) -> Result<Option<Tensor>> {
    // Accelerators only.
    if x_normed.device().is_cpu() {
        return Ok(None);
    }
    // KDA-STEP-PROBE: where does x_normed live, per layer, per call?
    if std::env::var("GRIM_KDA_STEP_PROBE").as_deref() == Ok("1") {
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if call <= 32 {
            eprintln!(
                "[kda-dev] call {call} layer {} seq_len {} x_normed.device() = {}",
                blk.layer_idx,
                seq_len,
                x_normed.device()
            );
        }
    }
    // TEMP discriminator: `GRIM_QWEN_KDA_D2D=0` forces every recurrent layer
    // down the host reference loop, exactly as `GRIM_QWEN_ATTN_D2D=0` does for
    // attention. If step-0 logprobs change with this set, the device KDA path
    // carries the prefill divergence; if byte-identical, it does not.
    static KDA_D2D_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *KDA_D2D_DISABLED.get_or_init(|| {
        matches!(
            std::env::var("GRIM_QWEN_KDA_D2D").as_deref(),
            Ok("0" | "false" | "off" | "no")
        )
    }) {
        d2d_decline!("kda: GRIM_QWEN_KDA_D2D is set to a disabling value");
    }
    let n_val_heads = blk.cfg_ssm_num_value_heads();
    let n_key_heads = blk.cfg_ssm_num_key_heads();
    let head_dim = blk.cfg_ssm_head_dim();
    if n_val_heads == 0 || n_key_heads == 0 || head_dim == 0 {
        d2d_decline!("kda: degenerate head config (v={n_val_heads} k={n_key_heads} d={head_dim})");
    }
    let Some(ref qkv_lin) = blk.attn_qkv else {
        d2d_decline!("kda: layer {} attn_qkv is None (is_full_attn={}, wq={})",
            blk.layer_idx, blk.is_full_attention, blk.wq.is_some());
    };
    // No conv weights means the stream is used un-convolved, which is a real
    // fallback case the host path supports; this one does not.
    let Some(ref conv_w) = blk.ssm_conv1d else {
        d2d_decline!("kda: ssm_conv1d is None (un-convolved stream)");
    };
    let Some(ref alpha_lin) = blk.ssm_alpha else {
        d2d_decline!("kda: ssm_alpha is None");
    };
    let Some(ref beta_lin) = blk.ssm_beta else {
        d2d_decline!("kda: ssm_beta is None");
    };
    let Some(ref gate_lin) = blk.attn_gate else {
        d2d_decline!("kda: attn_gate (z output gate) is None");
    };
    let Some(ref dt_bias) = blk.ssm_dt_bias else {
        d2d_decline!("kda: ssm_dt_bias is None");
    };
    let Some(ref ssm_a) = blk.ssm_a else {
        d2d_decline!("kda: ssm_a is None");
    };
    let Some(ref ssm_norm) = blk.ssm_norm else {
        d2d_decline!("kda: ssm_norm is None");
    };

    let key_dim = n_key_heads * head_dim;
    let value_dim = n_val_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    if branch_width != value_dim {
        d2d_decline!("kda: branch_width {branch_width} != value_dim {value_dim}");
    }
    if dt_bias.len() < n_val_heads || ssm_a.len() < n_val_heads || ssm_norm.len() < head_dim {
        d2d_decline!(
            "kda: short per-head vectors undersized (dt_bias {} ssm_a {} ssm_norm {}, need {n_val_heads}/{n_val_heads}/{head_dim})",
            dt_bias.len(),
            ssm_a.len(),
            ssm_norm.len(),
        );
    }

    let taps = blk.cfg_ssm_d_conv().max(1);
    // The short conv weight must already be plain f32 of the right shape: the
    // op reads it directly and has no quantized path of its own.
    // The conv weight arrives shaped [conv_dim, taps] (ggml ne0 is the tap
    // count, so the tensor is row-major channel-major). The device conv op
    // indexes [channel, tap] already, so it is passed through unchanged.
    let cw_dims = conv_w.shape().dims().to_vec();
    let shape_ok = (cw_dims.len() == 2 && cw_dims[0] == conv_dim && cw_dims[1] == taps)
        || (cw_dims.len() == 2 && cw_dims[0] == taps && cw_dims[1] == conv_dim)
        || (cw_dims.len() == 1 && cw_dims[0] == conv_dim * taps);
    if !shape_ok {
        d2d_decline!("kda: conv_w shape {cw_dims:?} incompatible with conv_dim={conv_dim}, taps={taps}");
    }
    let is_quant_conv = matches!(
        conv_w.dtype().storage,
        Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8)
            | Storage::W4A4OstQuant(_)
            | Storage::KQuant(grim_tensor::KQuantScheme::Q80)
            | Storage::KQuant(grim_tensor::KQuantScheme::Q4K)
    ) || matches!(conv_w.dtype().arith, grim_tensor::ArithType::U8);

    if !matches!(conv_w.dtype().storage, Storage::Native) && !is_quant_conv {
        d2d_decline!(
            "kda: ssm_conv1d.weight is {:?}, unsupported quantization",
            conv_w.dtype().storage,
        );
    }

    let dev = pick_device_for_storage_device(x_normed.device());

    // --- conv + projections, all on device ---------------------------------
    let qkv = qkv_lin.forward(x_normed)?;
    if qkv.device() != x_normed.device() {
        d2d_decline!("kda: attn_qkv projection landed on a different device than its input");
    }
    let alpha = alpha_lin.forward(x_normed)?;
    let beta = beta_lin.forward(x_normed)?;
    // z is the KDA output gate. `attn_gate` is loaded at
    // `q_dim.max(value_dim)` because the same tensor serves both layer types;
    // the op reads the leading `value_dim` elements, which is the same slice
    // the host path reaches through `z_stride`.
    let z = gate_lin.forward(x_normed)?;

    // Static per-head vectors pre-uploaded to device. Fallback to upload only if not pre-cached.
    let dt_bias_d = match blk.ssm_dt_bias_dev.as_ref() {
        Some(st) => st.clone(),
        None => std::sync::Arc::from(dev.from_cpu(&dt_bias[..n_val_heads], &Shape::new(vec![n_val_heads]), DType::F32)?),
    };
    let ssm_a_d = match blk.ssm_a_dev.as_ref() {
        Some(st) => st.clone(),
        None => std::sync::Arc::from(dev.from_cpu(&ssm_a[..n_val_heads], &Shape::new(vec![n_val_heads]), DType::F32)?),
    };
    let ssm_norm_d = match blk.ssm_norm_dev.as_ref() {
        Some(st) => st.clone(),
        None => std::sync::Arc::from(dev.from_cpu(&ssm_norm[..head_dim], &Shape::new(vec![head_dim]), DType::F32)?),
    };

    // --- short conv, state in place on device ------------------------------
    if cache.conv_state_dev.is_none() {
        cache.conv_state_dev =
            Some(dev.zeros(&Shape::new(vec![1, taps - 1, conv_dim]), DType::F32)?);
    }
    let conv_state = cache.conv_state_dev.as_ref().ok_or_else(|| {
        grim_core::error::Error::Backend("conv_state_dev vanished after allocation".into())
    })?;
    let (conv_out, _) = dev.short_conv1d_causal_step(
        qkv.storage().as_ref(),
        conv_w.storage().as_ref(),
        None,
        conv_state.as_ref(),
        &Shape::new(vec![seq_len, conv_dim]),
    )?;

    // --- batched delta rule, state in place on device ----------------------
    if cache.ssm_state_dev.is_none() {
        cache.ssm_state_dev = Some(dev.zeros(
            &Shape::new(vec![n_val_heads, head_dim, head_dim]),
            DType::F32,
        )?);
    }
    let ssm_state = cache.ssm_state_dev.as_ref().ok_or_else(|| {
        grim_core::error::Error::Backend("ssm_state_dev vanished after allocation".into())
    })?;
    // KDA-STEP-PROBE: PRE-step ssm state (end-of-prefill handoff check).
    {
        static PRE_DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if std::env::var("GRIM_KDA_STEP_PROBE").as_deref() == Ok("1")
            && blk.layer_idx == 0
            && seq_len == 1
            && !PRE_DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            if let Ok(r) = grim_backend_rocm::device::util::as_rocm(ssm_state.as_ref()) {
                if let Ok(b) = r.copy_to_host() {
                    let vals: Vec<f32> = b
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let _ = std::fs::write("/tmp/kda_probe_d2d_ssm_pre.bin", &bytes);
                    eprintln!("[kda-step-probe] wrote ssm PRE state ({} f32)", vals.len());
                }
            }
        }
    }

    let eps = blk.attn_norm.eps;
    let (branch, _) = if seq_len == 1 {
        dev.kda_gated_delta_rule_batched(
            conv_out.as_ref(),
            alpha.storage().as_ref(),
            beta.storage().as_ref(),
            dt_bias_d.as_ref(),
            ssm_a_d.as_ref(),
            ssm_norm_d.as_ref(),
            Some(z.storage().as_ref()),
            ssm_state.as_ref(),
            n_val_heads,
            n_key_heads,
            head_dim,
            eps,
            &Shape::new(vec![1, value_dim]),
        )?
    } else {
        dev.kda_gated_delta_rule_scan(
            conv_out.as_ref(),
            alpha.storage().as_ref(),
            beta.storage().as_ref(),
            dt_bias_d.as_ref(),
            ssm_a_d.as_ref(),
            ssm_norm_d.as_ref(),
            Some(z.storage().as_ref()),
            ssm_state.as_ref(),
            seq_len,
            n_val_heads,
            n_key_heads,
            head_dim,
            eps,
            &Shape::new(vec![seq_len, value_dim]),
        )?
    };

    // No readback at all: the gated branch output and the recurrent state both
    // stay in VRAM, and `ssm_out` consumes the tensor directly.

    // KDA-STEP-PROBE (GRIM_KDA_STEP_PROBE=1): layer 0's first decode step runs
    // the ENTIRE host reference on the imported live device state and diffs
    // the two branches. This is the composite check the chain gates cannot
    // give: real production inputs, real state handoff, both paths.
    {
        static PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let probe_on = std::env::var("GRIM_KDA_STEP_PROBE").as_deref() == Ok("1");
        if probe_on
            && blk.layer_idx == 0
            && seq_len == 1
            && !PROBED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            let taps = blk.cfg_ssm_d_conv().max(1);
            let ring_elems = conv_dim * (taps - 1);
            // Import the live device state.
            let conv_dev_host: Vec<f32> = grim_backend_rocm::device::util::as_rocm(cache.conv_state_dev.as_ref().unwrap().as_ref())
                .and_then(|r| r.copy_to_host())
                .map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
                .unwrap_or_default();
            let ssm_dev_host: Vec<f32> = grim_backend_rocm::device::util::as_rocm(cache.ssm_state_dev.as_ref().unwrap().as_ref())
                .and_then(|r| r.copy_to_host())
                .map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
                .unwrap_or_default();
            eprintln!(
                "[kda-step-probe] imported device ring {} floats (need {ring_elems}), ssm {} floats",
                conv_dev_host.len(),
                ssm_dev_host.len()
            );
            // The D2D kernels above are still queued; the host reference
            // below interleaves its own device ops + readbacks.
            if let Device::Rocm(ord) = x_normed.device() {
                grim_backend_rocm::RocmDevice::shared(*ord).synchronize();
            }
            // Transpose the device ring [channel][tap] into the host mirror's
            // [t * h_dim + d] layout the host loop expects.
            let mut conv_host = vec![0.0f32; ring_elems];
            for d in 0..conv_dim {
                for t in 0..(taps - 1) {
                    conv_host[t * conv_dim + d] = conv_dev_host[d * (taps - 1) + t];
                }
            }
            // Run the host reference on the imported state.
            let saved_conv = std::mem::take(&mut cache.conv_state);
            let saved_ssm = std::mem::take(&mut cache.ssm_state);
            cache.conv_state = conv_host;
            cache.ssm_state = ssm_dev_host.clone();
            let mut host_branch = vec![0.0f32; seq_len * branch_width];
            let host_res = gated_delta_net_forward(
                blk,
                cache,
                x_normed,
                &mut host_branch,
                seq_len,
                branch_width,
            );
            // The host run advanced the host mirror by this token; hand the
            // post-step mirrors back (device buffers stay authoritative for
            // the D2D path; these mirrors now match them).
            let _ = (saved_conv, saved_ssm);

            let d2d_branch = grim_backend_rocm::device::util::as_rocm(branch.as_ref())
                .and_then(|r| r.copy_to_host())
                .map(|b| {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<f32>>()
                })
                .unwrap_or_default();
            match host_res {
                Ok(()) => {
                    let worst = d2d_branch
                        .iter()
                        .zip(&host_branch)
                        .map(|(g, w)| (g - w).abs())
                        .fold(0.0f32, f32::max);
                    let at = d2d_branch
                        .iter()
                        .zip(&host_branch)
                        .enumerate()
                        .max_by(|(_, (g, w)), (_, (g2, w2))| {
                            let a = (*g - *w).abs();
                            let b = (*g2 - *w2).abs();
                            a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i);
                    eprintln!(
                        "[kda-step-probe] branch: d2d vs host worst |diff| {worst:.6} at idx {at:?}; \
                         d2d[0..4]={:?} host[0..4]={:?}",
                        &d2d_branch[..4.min(d2d_branch.len())],
                        &host_branch[..4.min(host_branch.len())]
                    );
                }
                Err(e) => eprintln!("[kda-step-probe] host reference failed: {e}"),
            }
            // conv_out dump for the same step (device tensor -> host).
            let conv_dump = grim_backend_rocm::device::util::as_rocm(conv_out.as_ref())
                .and_then(|r| r.copy_to_host())
                .map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>())
                .unwrap_or_default();
            eprintln!(
                "[kda-step-probe] conv_out[0..6]={:?} alpha[0..4]={:?} beta[0..4]={:?} z[0..4]={:?}",
                &conv_dump[..6.min(conv_dump.len())],
                &alpha.to_vec_f32().unwrap_or_default()[..4],
                &beta.to_vec_f32().unwrap_or_default()[..4],
                &z.to_vec_f32().unwrap_or_default()[..4],
            );
            let dump = |vals: &[f32], name: &str| {
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                let path = format!("/tmp/kda_probe_d2d_{name}.bin");
                if std::fs::write(&path, &bytes).is_ok() {
                    eprintln!("[kda-step-probe] wrote {} ({} f32) -> {path}", name, vals.len());
                }
            };
            dump(&conv_dump, "conv_mix");
            dump(&conv_dev_host, "conv_ring");
            dump(&ssm_dev_host, "ssm_state");
            dump(&d2d_branch, "branch");
            let x_dump = grim_backend_rocm::device::util::as_rocm(qkv.storage().as_ref())
                .and_then(|r| r.copy_to_host())
                .map(|b| {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<f32>>()
                })
                .unwrap_or_default();
            dump(&x_dump, "conv_x");
            // The load-time device constants: never verified against the host
            // vectors the host loop reads.
            for (name, st) in [
                ("dt_bias_dev", blk.ssm_dt_bias_dev.as_ref()),
                ("ssm_a_dev", blk.ssm_a_dev.as_ref()),
                ("ssm_norm_dev", blk.ssm_norm_dev.as_ref()),
            ] {
                if let Some(st) = st {
                    if let Ok(r) = grim_backend_rocm::device::util::as_rocm(st.as_ref()) {
                        if let Ok(b) = r.copy_to_host() {
                            let vals: Vec<f32> = b
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            dump(&vals, name);
                        }
                    }
                }
            }
            dump(&blk.ssm_dt_bias.as_deref().unwrap_or(&[]), "dt_bias_host");
            dump(&blk.ssm_a.as_deref().unwrap_or(&[]), "ssm_a_host");
            dump(&blk.ssm_norm.as_deref().unwrap_or(&[]), "ssm_norm_host");
            dump(&alpha.to_vec_f32().unwrap_or_default(), "alpha_full");
            dump(&beta.to_vec_f32().unwrap_or_default(), "beta_full");
            dump(&z.to_vec_f32().unwrap_or_default(), "z_full");
            if let Some(ref cw) = blk.ssm_conv1d {
                let cw_dump = grim_backend_rocm::device::util::as_rocm(cw.storage().as_ref())
                    .and_then(|r| r.copy_to_host())
                    .map(|b| {
                        b.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect::<Vec<f32>>()
                    })
                    .unwrap_or_default();
                dump(&cw_dump, "conv_w");
                eprintln!("[kda-step-probe] conv_w dims {:?}", cw.shape().dims());
            }
        }
    }

    // KDA-STEP-PROBE: per-layer branch fingerprint at the first decode step,
    // to find the first layer where the D2D and host runs diverge.
    if std::env::var("GRIM_KDA_STEP_PROBE").as_deref() == Ok("1") && seq_len == 1 {
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if call <= 6 {
            let fp = grim_backend_rocm::device::util::as_rocm(branch.as_ref())
                .and_then(|r| r.copy_to_host())
                .map(|b| {
                    let vals: Vec<f32> = b
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    (
                        vals.iter().take(8).sum::<f32>(),
                        vals.iter().sum::<f32>(),
                    )
                })
                .unwrap_or((0.0, 0.0));
            // Which input died? z gate, conv stream, alpha, beta sums.
            let z_sum = z.to_vec_f32().map(|v| v.iter().sum::<f32>()).unwrap_or(f32::NAN);
            let conv_sum = grim_backend_rocm::device::util::as_rocm(conv_out.as_ref())
                .and_then(|r| r.copy_to_host())
                .map(|b| {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .sum::<f32>()
                })
                .unwrap_or(f32::NAN);
            let a_sum = alpha.to_vec_f32().map(|v| v.iter().sum::<f32>()).unwrap_or(f32::NAN);
            let b_sum = beta.to_vec_f32().map(|v| v.iter().sum::<f32>()).unwrap_or(f32::NAN);
            let nw_sum = blk
                .ssm_norm_dev
                .as_ref()
                .and_then(|st| grim_backend_rocm::device::util::as_rocm(st.as_ref()).ok())
                .and_then(|r| r.copy_to_host().ok())
                .map(|b| {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .sum::<f32>()
                })
                .unwrap_or(f32::NAN);
            eprintln!(
                "[kda-branch-fp] d2d decode-call {call} layer {} branch_sum={:+.6} z_sum={z_sum:+.4} conv_sum={conv_sum:+.4} alpha_sum={a_sum:+.4} beta_sum={b_sum:+.4} nw_sum={nw_sum:+.4}",
                blk.layer_idx, fp.1
            );
        }
    }

    Ok(Some(Tensor::new(
        std::sync::Arc::from(branch),
        Shape::new(vec![seq_len, value_dim]),
        DType::F32,
        x_normed.provenance().clone(),
        x_normed.device().clone(),
    )))
}

/// Full-attention decode step with Q, K, V and the attention output all left
/// on the device.
///
/// Returns `Ok(None)` — declining, without touching the KV arena — when this
/// path does not apply, in which case the caller runs the host reference. The
/// two are alternatives: whichever runs owns the arena append.
///
/// Why: the host path downloads the fused `attn_q` output, splits it on the
/// host, re-uploads both halves, downloads Q again for the per-head norm,
/// downloads it *again* after RoPE, and downloads the attention result. That
/// is five device→host transfers per layer per token, and it is also what
/// keeps the attention branch out of any captured graph.
///
/// Q and the attention output are what the caller asked to move; K and V move
/// with them because they share this one code path, and splitting the branch
/// to keep just Q and A on device would have meant two near-identical
/// attention implementations to keep in step.
fn attention_layer_d2d(
    blk: &Qwen35Block,
    x_normed: &Tensor,
    positions: &[u32],
    cache: &mut Qwen35LayerCache,
    seq_len: usize,
) -> Result<Option<Tensor>> {
    // Accelerators only: the CPU reference stays the production path there.
    if x_normed.device().is_cpu() {
        return Ok(None);
    }
    // The device attention kernel this path calls had a real defect: it staged
    // the query in 8 chunks (256 dims) but accumulated and wrote V in 4, so at
    // head_dim 256 it never wrote the upper half of every output row and the
    // caller read back exactly half zeros. Fixed in `kernels/qkv_attention.rs`
    // (every head_dim loop now covers 8 chunks) and in the flat-output
    // head_dim recovery in `attention_ops.rs`, which divided the QUERY width by
    // num_kv_heads and so asked for the wrong geometry under GQA.
    // `qwen35_attention_device_kernel_is_wrong` is the gate, and this path is
    // verified against the host reference over chained decode steps.
    //
    // Escape hatch: GRIM_QWEN_ATTN_D2D=0 forces the host reference.
    if attn_d2d_disabled() {
        d2d_decline!("attn: GRIM_QWEN_ATTN_D2D is set to a disabling value");
    }
    // `wo` is deliberately not required here: the caller applies it to whatever
    // this returns, and its own `Option` handling is unchanged by this path.
    let (Some(wq), Some(wk), Some(wv)) = (blk.wq.as_ref(), blk.wk.as_ref(), blk.wv.as_ref()) else {
        d2d_decline!(
            "attn: separated wq/wk/wv incomplete (q={} k={} v={})",
            blk.wq.is_some(),
            blk.wk.is_some(),
            blk.wv.is_some(),
        );
    };

    let q_dim = blk.num_heads * blk.head_dim;
    let kv_dim = blk.num_kv_heads * blk.head_dim;
    let device = x_normed.device().clone();
    let dev = pick_device_for_storage_device(&device);

    // TEMP probe: this path is the default decode route and it aborts on a bad
    // transfer at the first attention layer. Print the geometry each step
    // assumes, and the real element counts, so a width mismatch is visible
    // rather than re-derived.
    if attn_shapes_debug() {
        // Shape only. This used to call `to_vec_f32()` on each weight to print
        // its element count, which drags every projection back to the host —
        // 33.5 M floats for `attn_q` alone, per attention layer, per token.
        // The count is `shape().elem_count()`; nothing about it needs the data.
        let shp = |t: &Option<Linear>| {
            t.as_ref().map(|l| (l.weight().shape().dims().to_vec(), l.weight().shape().elem_count()))
        };
        let nshp = |n: &Option<RmsNorm>| {
            n.as_ref().map(|x| (x.weight.shape().dims().to_vec(), x.weight.shape().elem_count()))
        };
        eprintln!(
            "[attn-debug] layer {} seq_len={} q_dim={q_dim} kv_dim={kv_dim} wq={:?} wk={:?} wv={:?} wo={:?} q_norm={:?} k_norm={:?} gate={:?}",
            blk.layer_idx,
            seq_len,
            shp(&blk.wq),
            shp(&blk.wk),
            shp(&blk.wv),
            shp(&blk.wo),
            nshp(&blk.attn_q_norm),
            nshp(&blk.attn_k_norm),
            shp(&blk.attn_gate),
        );
    }

    // TEMP: bracket every stage so one run says which one aborts.
    macro_rules! stage {
        ($n:expr) => {
            if attn_shapes_debug() {
                eprintln!("[attn-stage] layer {} stage={}", blk.layer_idx, $n);
            }
        };
    }
    stage!("begin");

    // Some TP-sharded projections emit padded rows — cut to the exact
    // [seq, width] extent via a D2D staging copy when needed.
    let exact = |t: Tensor, rows: usize, width: usize| -> Result<Tensor> {
        let want = Shape::new(vec![rows, width]);
        if t.shape().elem_count() == rows * width {
            return crate::block::reshaped_view(&t, &want);
        }
        let scratch = dev.alloc_storage(&want, DType::F32)?;
        dev.copy_slice_range(scratch.as_ref(), 0, t.storage().as_ref(), 0, rows * width)?;
        Ok(Tensor::new(
            scratch.into(),
            want,
            DType::F32,
            t.provenance().clone(),
            t.device().clone(),
        ))
    };

    // --- projections, on device -------------------------------------------
    let q_full = exact(wq.forward(x_normed)?, seq_len, 2 * q_dim)?;
    let k_dev_t = exact(wk.forward(x_normed)?, seq_len, kv_dim)?;
    let v_dev_t = exact(wv.forward(x_normed)?, seq_len, kv_dim)?;

    // --- split the fused [Q | gate] row-wise, on device --------------------
    // A flat byte-range copy of the first q_dim elements is wrong for
    // seq_len > 1: it cuts across row boundaries. `narrow_cols` copies a
    // column RANGE out of every row, which is the row-aware split.
    stage!("projections");
    let q_split = Shape::new(vec![seq_len, q_dim]);
    let (q_st, gate_st) = {
        let mark = |tag: &str| {
            if attn_shapes_debug() {
                eprintln!("[attn-split] {tag} q_full ptr_ok={} dims={:?}",
                    q_full.storage().device_ptr().is_some(),
                    q_full.shape().dims().to_vec());
            }
        };
        mark("before_lo");
        let (qs, _) = dev.narrow_cols(
            q_full.storage().as_ref(),
            2 * q_dim,
            0,
            seq_len,
            q_dim,
            &q_split,
        )?;
        mark("after_lo");
        let (gs, _) = dev.narrow_cols(
            q_full.storage().as_ref(),
            2 * q_dim,
            q_dim,
            seq_len,
            q_dim,
            &q_split,
        )?;
        mark("after_hi");
        (qs, gs)
    };
    stage!("split");
    let mk = |st: Box<dyn grim_tensor::BackendStorage>| -> Tensor {
        Tensor::new(
            Arc::from(st),
            q_split.clone(),
            DType::F32,
            x_normed.provenance().clone(),
            x_normed.device().clone(),
        )
    };
    let mut q_dev = mk(q_st);
    let q_gate_dev = mk(gate_st);

    // --- per-head Q/K RMS norm, BEFORE RoPE, on device ---------------------
    // The reference norms q and k before rotating (qwen35.cpp:281/286, MRoPE at
    // :299); norming after RoPE would be a different function, not a reorder.
    let head_norm = |t: &Tensor, heads: usize, n: Option<&RmsNorm>| -> Result<Tensor> {
        let Some(n) = n else { return Ok(t.clone()) };
        let three = Shape::new(vec![1, seq_len * heads, blk.head_dim]);
        let mk2 = |tag: &str| {
            if attn_shapes_debug() {
                eprintln!("[attn-hn] {tag} in_dims={:?} three={:?}", t.shape().dims().to_vec(), three.dims().to_vec());
            }
        };
        mk2("A_reshape_view_pre");
        let t3 = crate::block::reshaped_view(t, &three)?;
        mk2("B_reshape_view_post");
        // The weight is [head_dim]; broadcast it across the (seq*heads) rows
        // the flattened view presents.
        // `n.weight` is ALREADY a device tensor. Copying it to the host and
        // straight back was a per-layer, per-token D2H + H2D of a constant,
        // and the synchronous D2H also acted as a sync point that could surface
        // a PENDING error from an earlier launch as a misleading
        // "hipMemcpyDtoH failed" here. Pass the device weight directly.
        mk2("C_pre_rms_norm");
        let (out, _) = dev.rms_norm(
            t3.storage().as_ref(),
            n.weight.storage().as_ref(),
            n.eps,
            &three,
        )?;
        mk2("D_post_rms_norm");
        // Back to [seq, heads*head_dim] — the tensor's OWN width, which is
        // kv_dim for K and q_dim for Q. Reshaping to a single shared width
        // would silently mis-slice the narrower K.
        mk2("E_pre_reshape_back");
        let r = crate::block::reshaped_view(
            &Tensor::new(
                out.into(),
                three.clone(),
                DType::F32,
                t.provenance().clone(),
                t.device().clone(),
            ),
            &Shape::new(vec![seq_len, heads * blk.head_dim]),
        );
        r
    };
    q_dev = head_norm(&q_dev, blk.num_heads, blk.attn_q_norm.as_ref())?;
    let k_dev_t = head_norm(&k_dev_t, blk.num_kv_heads, blk.attn_k_norm.as_ref())?;

    stage!("head_norm");
    // --- RoPE, on device ---------------------------------------------------
    let mut rope_cfg = grim_tensor::RopeConfig::new(blk.head_dim, blk.rope_theta);
    rope_cfg.rotary_dim = blk.rotary_dim;
    rope_cfg.interleaved = false; // NeoX half-split; see the D2D site above.
    let rope_ext = |t: &Tensor, heads: usize| -> Result<Tensor> {
        let mut pos_ext = Vec::with_capacity(seq_len * heads);
        for &pos in positions {
            for _ in 0..heads {
                pos_ext.push(pos);
            }
        }
        let t3 =
            crate::block::reshaped_view(t, &Shape::new(vec![1, seq_len * heads, blk.head_dim]))?;
        let (rope_s, _) = dev.rope(t3.storage().as_ref(), &pos_ext, &rope_cfg, t3.shape())?;
        crate::block::reshaped_view(
            &Tensor::new(
                rope_s.into(),
                t3.shape().clone(),
                DType::F32,
                t.provenance().clone(),
                t.device().clone(),
            ),
            &Shape::new(vec![seq_len, heads * blk.head_dim]),
        )
    };
    let q_rope = rope_ext(&q_dev, blk.num_heads)?;
    let k_rope = rope_ext(&k_dev_t, blk.num_kv_heads)?;

    stage!("rope");
    // --- append K/V to the device arena, then attend -----------------------
    let k_new_rows = seq_len * blk.num_kv_heads;
    let kv_elems = k_new_rows * blk.head_dim;
    let k_cap_rows = cache
        .k_device
        .as_ref()
        .map(|s| s.shape().dims()[0])
        .unwrap_or(0);
    let need_rows = cache.current_pos + k_new_rows;

    if k_cap_rows >= need_rows {
        let k_dev = cache
            .k_device
            .as_ref()
            .ok_or_else(|| grim_core::error::Error::Backend("cache.k_device missing".into()))?;
        let v_dev = cache
            .v_device
            .as_ref()
            .ok_or_else(|| grim_core::error::Error::Backend("cache.v_device missing".into()))?;
        dev.copy_slice_range(
            &**k_dev,
            cache.current_pos * blk.num_kv_heads * blk.head_dim,
            k_rope.storage().as_ref(),
            0,
            kv_elems,
        )?;
        dev.copy_slice_range(
            &**v_dev,
            cache.current_pos * blk.num_kv_heads * blk.head_dim,
            v_dev_t.storage().as_ref(),
            0,
            kv_elems,
        )?;
    } else {
        let new_rows = ((need_rows * 2) + 64).next_power_of_two();
        let k_idx = blk.num_kv_heads * blk.head_dim;
        let full_shape = Shape::new(vec![new_rows, blk.num_kv_heads, blk.head_dim]);
        let k_grown = dev.alloc_storage(&full_shape, DType::F32)?;
        let v_grown = dev.alloc_storage(&full_shape, DType::F32)?;
        if let Some(ref old_k) = cache.k_device {
            dev.copy_slice_range(
                k_grown.as_ref(),
                0,
                old_k.as_ref(),
                0,
                cache.current_pos * k_idx,
            )?;
        }
        if let Some(ref old_v) = cache.v_device {
            dev.copy_slice_range(
                v_grown.as_ref(),
                0,
                old_v.as_ref(),
                0,
                cache.current_pos * k_idx,
            )?;
        }
        dev.copy_slice_range(
            k_grown.as_ref(),
            cache.current_pos * k_idx,
            k_rope.storage().as_ref(),
            0,
            kv_elems,
        )?;
        dev.copy_slice_range(
            v_grown.as_ref(),
            cache.current_pos * k_idx,
            v_dev_t.storage().as_ref(),
            0,
            kv_elems,
        )?;
        cache.k_device = Some(k_grown);
        cache.v_device = Some(v_grown);
    }

    stage!("arena_append");
    let total_kv = cache.current_pos + seq_len;
    let k_arena = cache
        .k_device
        .as_ref()
        .ok_or_else(|| grim_core::error::Error::Backend("cache.k_device missing".into()))?;
    let v_arena = cache
        .v_device
        .as_ref()
        .ok_or_else(|| grim_core::error::Error::Backend("cache.v_device missing".into()))?;
    // TEMP probe: the abort is a D2H inside the attention call's host fallback
    // (the ROCm qkv_attention kernel is rejected for this geometry, so it
    // downloads Q and the KV arena). Print shape AND pointer for every buffer
    // that download reads, so the answer is binary rather than another round
    // of setting up a test.
    if attn_shapes_debug() {
        let d = |n: &str, st: Option<&dyn grim_tensor::BackendStorage>| match st {
            None => format!("{n}=None"),
            Some(x) => {
                let has = x.device_ptr().is_some();
                let nb = x.shape().dims().to_vec();
                match has {
                    true => format!("{n}=ptr@{:#x} dims={:?}", x.device_ptr().unwrap(), nb),
                    false => format!("{n}=NULLPTR dims={:?}", nb),
                }
            }
        };
        eprintln!(
            "[attn-arena] layer {} total_kv={} cur_pos={} | {} | {} | {}",
            blk.layer_idx,
            total_kv,
            cache.current_pos,
            d("q_rope", Some(q_rope.storage().as_ref())),
            d("k_arena", Some(k_arena.as_ref())),
            d("v_arena", Some(v_arena.as_ref())),
        );
    }

    let attn = crate::shared_attention::fused_or_scalar_attention_arena_device(
        q_rope.storage().as_ref(),
        k_arena.as_ref(),
        v_arena.as_ref(),
        total_kv,
        blk.num_heads,
        blk.num_kv_heads,
        blk.head_dim,
        seq_len,
        None,
        &device,
    )?;

    // --- output gate: attn_out * sigmoid(gate), on device ------------------
    // Reuse the existing device sigmoid rather than adding a second one: the
    // ROCm backend already exposes it and `grim_nn` wraps it for tensors.
    let sig = grim_nn::modules::sigmoid_on_device(&q_gate_dev)?;
    let (gated, _) = dev.mul(attn.storage().as_ref(), sig.storage().as_ref(), &q_split)?;

    Ok(Some(Tensor::new(
        Arc::from(gated),
        q_split,
        DType::F32,
        x_normed.provenance().clone(),
        x_normed.device().clone(),
    )))
}

// ── Gated DeltaNet (KDA) helpers ───────────────────────────────────────────

/// How a value head maps to its key head in the 3:1 GQA-shaped KDA.
///
/// A 3:1 ratio is consistent with BOTH of these, and nothing in the GGUF
/// encodes the mapping, so it is a single switchable constant rather than a
/// guess baked into the loop. Determined empirically against a known-good
/// reference output; see docs/benchmarks.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaHeadPairing {
    /// `kv_head = value_head % num_key_heads`
    Interleaved,
    /// `kv_head = value_head / values_per_group`
    Grouped,
}

/// Default pairing, per llama.cpp `src/models/qwen35.cpp`:
///
/// ```cpp
/// if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
///     GGML_ASSERT(num_v_heads % num_k_heads == 0);
///     q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
///     k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
/// }
/// ```
///
/// `ggml_repeat_4d` tiles the 16 key heads to fill 48 value heads, so value head
/// `h` reads key head `h % num_key_heads`. This was previously left as an open
/// question on the grounds that a 3:1 ratio fits both rules; the reference
/// settles it.
pub const KDA_HEAD_PAIRING: KdaHeadPairing = KdaHeadPairing::Interleaved;

/// Resolve the key head feeding a given value head.
fn kda_key_head(value_head: usize, num_key_heads: usize, values_per_group: usize) -> usize {
    let raw = match KDA_HEAD_PAIRING {
        KdaHeadPairing::Interleaved => value_head % num_key_heads.max(1),
        KdaHeadPairing::Grouped => value_head / values_per_group.max(1),
    };
    // Clamp against the KEY head count: this indexes the key/value split of
    // attn_qkv, not the value stream.
    raw.min(num_key_heads.saturating_sub(1))
}

/// One Gated DeltaNet step for a single head, over a row-major `[d_v, d_k]`
/// state slice that is updated in place.
///
/// Published update (ICLR 2025, Eq. 10), matching `grim-backend-cpu`'s
/// reference and the corrected ROCm/CUDA/Vulkan kernels:
///
/// ```text
/// decay = exp(gate)              // gate already folded with a and dt_bias
/// pred  = sum_k k * (decay * S)  // decay applied BEFORE the dot
/// delta = beta * (v - pred)      // beta scales the FULL delta term
/// S_new = decay * S + k * delta
/// out   = sum_k q * S_new
/// ```
///
/// `d_k == d_v == 128` for this checkpoint, so `S` is square.
fn kda_gated_delta_rule_row(
    k: &[f32],
    v: &[f32],
    beta: f32,
    gate: f32,
    state: &mut [f32],
    d_k: usize,
    d_v: usize,
) {
    // Defensive: a test fixture or a mis-sized cache can hand us a state
    // buffer shorter than d_k*d_v. Do nothing rather than index out of bounds —
    // a wrong recurrence is bad enough without also being a panic.
    if state.len() < d_k * d_v {
        return;
    }
    let decay = gate.exp();
    for j in 0..d_v {
        let row = &mut state[j * d_k..(j + 1) * d_k];
        // decay BEFORE the dot
        let pred: f32 = k
            .iter()
            .zip(row.iter())
            .map(|(kk, ss)| kk * (decay * ss))
            .sum();
        // beta scales the whole delta term
        let delta = beta * (v.get(j).copied().unwrap_or(0.0) - pred);
        for i in 0..d_k {
            let s = decay * row[i] + k.get(i).copied().unwrap_or(0.0) * delta;
            row[i] = s;
        }
    }
}

/// Softplus, used to turn the fused (a + dt_bias + alpha) logit into a decay
/// rate. Defined here because the block does not otherwise need it.
trait Softplus {
    fn softplus(self) -> f32;
}
impl Softplus for f32 {
    fn softplus(self) -> f32 {
        // log1p(exp(x)), stable for large |x|
        if self > 20.0 {
            self
        } else if self < -20.0 {
            self.exp()
        } else {
            self.exp().ln_1p()
        }
    }
}

// Helpers

/// Quantize a host K/V row block into the packed layout the paged kernel reads.
///
/// Returns `(packed_bytes, per_tensor_scale)`. The scale is meaningful only for
/// formats that do not carry their own (int8, FP8); Nutcracker and NVFP4 embed
/// a per-16 block scale in the buffer and ignore it.
pub(crate) fn quantize_kv_block(
    data: &[f32],
    fmt: grim_tensor::PagedKvQuantFormat,
) -> grim_core::error::Result<(Vec<u8>, f32)> {
    use grim_tensor::PagedKvQuantFormat as F;
    match fmt {
        F::Int8 => {
            // Symmetric per-tensor int8: scale = max|x| / 127.
            let amax = data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let bytes: Vec<u8> = data
                .iter()
                .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8 as u8)
                .collect();
            Ok((bytes, scale))
        }
        F::Fp8E4M3 => {
            let bytes = grim_quant::quant_fp8(data)
                .map_err(|e| grim_core::error::Error::Backend(format!("quant_fp8 failed: {e}")))?;
            Ok((bytes, 1.0))
        }
        F::NutFp4 => {
            let bytes = grim_quant::quant_nutcracker(data).map_err(|e| {
                grim_core::error::Error::Backend(format!("quant_nutcracker failed: {e}"))
            })?;
            Ok((bytes, 1.0))
        }
        other => Err(grim_core::error::Error::Backend(format!(
            "kv_quant_format: {other:?} has no host packer wired for Qwen35 KV"
        ))),
    }
}

/// Tokens per Nutcracker/NVFP4 sub-block, and its byte width (1 scale + 8 codes).
/// Used to document the packed layout; sizing itself goes through
/// `packed_kv_bytes` so the two cannot drift.
#[allow(dead_code)]
const NUT_SUB_BLOCK: usize = 16;
#[allow(dead_code)]
const NUT_SUB_BLOCK_BYTES: usize = 9;

/// KV quantization format selected by `GRIM_KV_QUANT`.
///
/// Only formats the paged attention kernel can actually decode are offered, and
/// the value is the `KvCacheQuantFormat` discriminant so it maps 1:1 onto
/// `dequant_kv_element`. `None` keeps the dense f32 arena.
///
/// `GRIM_KV_QUANT=f16` is deliberately rejected here: the f16 arena is a
/// different representation with its own reader, and the f32<->f16 cast is
/// currently unverified, so silently honoring it would reintroduce the exact
/// reinterpretation bug the paged path is designed to avoid.
fn kv_quant_format() -> Option<grim_tensor::PagedKvQuantFormat> {
    match std::env::var("GRIM_KV_QUANT").as_deref() {
        Ok("int8" | "q8_0" | "q8") => Some(grim_tensor::PagedKvQuantFormat::Int8),
        Ok("fp8" | "fp8_e4m3") => Some(grim_tensor::PagedKvQuantFormat::Fp8E4M3),
        Ok("nutcracker" | "nutfp4" | "nut_fp4") => Some(grim_tensor::PagedKvQuantFormat::NutFp4),
        Ok(other) => {
            // Fail loud rather than silently falling back to f32: a user who
            // asked for quantized KV and got a 32 GB arena would otherwise see
            // an OOM with no indication the setting was ignored.
            eprintln!(
                "[qwen35] GRIM_KV_QUANT={other:?} is not a supported paged KV format \
                 (int8 | fp8 | nutcracker); using the dense f32 arena"
            );
            None
        }
        _ => None,
    }
}

/// Bytes needed to hold `n_values` quantized KV entries.
pub(crate) fn packed_kv_bytes(n_values: usize, fmt: grim_tensor::PagedKvQuantFormat) -> usize {
    use grim_tensor::PagedKvQuantFormat as F;
    match fmt {
        F::Int8 => n_values,
        F::Fp8E4M3 | F::Fp8E5M2 => n_values,
        // Sub-blocked formats: the scale lives inside the group, so a partial
        // group still costs a full group's scale byte. `div_ceil` on the group
        // count, not on the total, is what keeps this correct for a KV length
        // that is not a multiple of the group size.
        F::NutFp4 | F::NvFp4 => n_values.div_ceil(NUT_SUB_BLOCK) * NUT_SUB_BLOCK_BYTES,
        F::Fp4E2M1 | F::Int4 => n_values.div_ceil(2),
        F::MxFp4 => n_values.div_ceil(32) * 17,
        F::MxFp8 => n_values.div_ceil(32) * 33,
    }
}

/// Fraction of a device's free VRAM the loader is allowed to plan against.
///
/// The remainder absorbs activations, the KV cache, the decode graph's
/// scratch buffers, and fragmentation. Planning to 100% of free VRAM is what
/// pushes the allocator into HIP managed (host-backed) memory, which then
/// thrashes under oversubscription.
const VRAM_PLAN_FRACTION: f64 = 0.90;

/// Assign each transformer block to a device by measured capacity.
///
/// The previous placement was index arithmetic —
/// `devices[i * devices.len() / num_layers]` — which is a static round-robin
/// that ignores both per-layer weight size and how much VRAM each device
/// actually has. On a mixed pair (9070 XT + 9060 XT, 17 GB each) loading the
/// 27B Q4_K checkpoint it drove device 0 to 99.9% of VRAM while device 1 sat
/// near half full, which forced the managed-memory fallback and then failed
/// the first kernel module load.
///
/// This reserves each device's share of the decode-graph KV arenas, then walks
/// the layers in order giving each one to the device with the most remaining
/// headroom, so a large early layer cannot monopolize one card. Falls back to
/// the old round-robin whenever capacity cannot be queried (a CPU device, or a
/// backend without `hipMemGetInfo`), keeping the old behavior as the
/// conservative default.
#[allow(clippy::too_many_arguments)]
fn plan_layer_devices(
    ws: &WeightSource<'_>,
    devices: &[Device],
    num_layers: usize,
    ctx: usize,
    kv_heads: usize,
    head_dim: usize,
    interval: usize,
) -> Vec<Device> {
    // Single-device or unknown layout: nothing to plan.
    if devices.len() <= 1 {
        return vec![devices[0].clone(); num_layers];
    }

    let round_robin = |i: usize| devices[i * devices.len() / num_layers.max(1)].clone();

    // Reserve the decode-graph KV arenas BEFORE handing out weight budget.
    //
    // The arenas are allocated at full context for every full-attention layer:
    // `ctx * num_kv_heads * head_dim * 4 bytes` for K and again for V. For this
    // 27B Q4_K model at a 228k context that is ~1.87 GB per attention layer
    // across 17 such layers = ~31.8 GB, which alone exceeds the 34 GB the pair
    // has. Planning weights against the *whole* free budget therefore drives
    // both cards to ~99% and leaves the arenas nowhere to go, which is what
    // pushed the run into HIP managed memory. Reserving first lets weights and
    // KV share the budget honestly, and makes an oversized context visible as a
    // warning rather than a silent spill.
    let num_attn_layers = (0..num_layers)
        .filter(|i| (i + 1) % interval.max(1) == 0)
        .count();

    // Bound the arenas by the VRAM that can actually hold them, BEFORE
    // reserving. The arena is sized from the model's context, and this 27B
    // advertises 232k: 16 attention layers at that context reserve 30.4 GB,
    // more than one 17.1 GB card holds. A reservation the device cannot hold is
    // not a slow path, it is a guaranteed failure - the weights then have
    // nowhere to go and the run dies at `vram_free=0.00` before the forward
    // pass. Tests are pinned to 4k (GRIM_CONTEXT=4096), which is far smaller,
    // but the bound holds whatever context is asked for.
    let arena_budget: u64 = devices
        .iter()
        .map(|d| match d {
            Device::Rocm(ord) => {
                let (free, total) = grim_backend_rocm::vram_info(*ord);
                if total == 0 { 0 } else { free }
            }
            _ => 0,
        })
        .sum();
    let kv_per_attn = |c: usize| -> u64 {
        (c as u64)
            .saturating_mul(kv_heads as u64)
            .saturating_mul(head_dim as u64)
            .saturating_mul(4) // f32
            .saturating_mul(2) // K and V
    };
    let (eff_ctx, kv_total) =
        bound_kv_arena(&kv_per_attn, num_attn_layers, arena_budget, ctx.max(1));
    if eff_ctx < ctx {
        eprintln!(
            "[qwen35] KV arena BOUNDED: requested ctx={ctx} needs {:.1} GB across \
             {num_attn_layers} attention layers, but only {:.1} GB is free across {} \
             device(s). Capping the arena at ctx={eff_ctx} ({:.1} GB). Set GRIM_CONTEXT \
             explicitly (tests use 4096) to choose the context.",
            ctx as f64 * kv_heads as f64 * head_dim as f64 * 4.0 * 2.0 * num_attn_layers as f64
                / 1e9,
            arena_budget as f64 / 1e9,
            devices.len(),
            kv_total as f64 / 1e9
        );
    } else if kv_total > 0 {
        eprintln!(
            "[qwen35] reserving KV arenas: ctx={eff_ctx}, kv_heads={kv_heads}, \
             head_dim={head_dim}, attn_layers={num_attn_layers} \
             -> {:.1} GB total (budget {:.1} GB)",
            kv_total as f64 / 1e9,
            arena_budget as f64 / 1e9
        );
    }

    // Remaining plannable bytes per device, or None when VRAM is unknowable.
    let kv_share = (kv_total as f64 / devices.len() as f64) as u64;
    let per_device: Vec<Option<u64>> = devices
        .iter()
        .map(|d| match d {
            Device::Rocm(ord) => {
                let (free, total) = grim_backend_rocm::vram_info(*ord);
                // A device reporting 0 total is not queryable; treat the whole
                // set as unplannable rather than guessing.
                if total == 0 {
                    return None;
                }
                let budget = ((free as f64) * VRAM_PLAN_FRACTION) as u64;
                Some(budget.saturating_sub(kv_share.min(budget)))
            }
            _ => None,
        })
        .collect();

    let all_known = per_device.iter().all(Option::is_some);
    let Some(mut remaining): Option<Vec<u64>> =
        all_known.then(|| per_device.into_iter().map(|v| v.unwrap_or(0)).collect())
    else {
        eprintln!(
            "[qwen35] VRAM capacity unavailable; using static layer round-robin \
             across {} devices",
            devices.len()
        );
        return (0..num_layers).map(round_robin).collect();
    };

    let mut assignment = Vec::with_capacity(num_layers);
    // Size each block from checkpoint metadata before committing it.
    let layer_bytes: Vec<u64> = (0..num_layers)
        .map(|i| ws.pp("blk").pp(&i.to_string()).prefix_bytes())
        .collect();

    // Measured per-device capability, in `devices` order.
    //
    // Gated rather than always-on: `CapabilityProfiler::new()` runs a rocBLAS
    // calibration GEMM and a malloc/free on every visible GPU, and this function
    // is called *while* the loader is deciding how much VRAM it may spend. Doing
    // that unconditionally would perturb the very free-VRAM numbers being read
    // here, on a path that has not yet been run against real hardware. With the
    // gate off every row is zero and `assign_by_headroom` degrades to exactly
    // the headroom-only split it has always used.
    let caps: Vec<grim_tensor::backend::GpuCapability> =
        if std::env::var("GRIM_CAPABILITY_PLACEMENT").as_deref() == Ok("1") {
            let measured = grim_backend_rocm::CapabilityProfiler::new().capabilities();
            devices
                .iter()
                .map(|d| match d {
                    Device::Rocm(ord) => measured.get(*ord).cloned().unwrap_or_default(),
                    _ => Default::default(),
                })
                .collect()
        } else {
            vec![Default::default(); devices.len()]
        };
    let (targets, unplaced) = assign_by_headroom(&layer_bytes, &mut remaining, &caps);
    for t in targets {
        assignment.push(devices[t].clone());
    }

    if unplaced > 0 {
        // Do NOT fail here: managed memory is slow but correct, and the loader
        // already warns when it engages. Surface it loudly and continue.
        eprintln!(
            "[qwen35] WARNING: {unplaced}/{num_layers} layers exceed aggregate \
             plannable VRAM across {} devices; those will fall back to HIP \
             managed memory, which degrades throughput under oversubscription",
            devices.len()
        );
    }
    let mut per_device: Vec<(String, usize)> = Vec::new();
    for d in &assignment {
        let key = d.to_string();
        match per_device.iter_mut().find(|(k, _)| *k == key) {
            Some((_, count)) => *count += 1,
            None => per_device.push((key, 1)),
        }
    }
    eprintln!(
        "[qwen35] capacity-aware placement: {}",
        per_device
            .iter()
            .map(|(k, c)| format!("{k}={c} layers"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // How much of the model ended up host-backed. Managed memory is host RAM, so
    // this is the number to compare against a host-memory delta observed during
    // load: a delta much larger than this points at staging copies rather than
    // at the spill itself. No-op when nothing spilled.
    grim_backend_rocm::memory::budget::report_managed_fallback_summary();
    assignment
}

/// Capacity- and capability-aware placement: each layer goes to the device that
/// can hold it and has the most measured throughput to spend, and `remaining` is
/// decremented in place.
///
/// `caps` is the per-device capability snapshot the in-bone capability system
/// measures. Measured throughput is the primary key, so a card that is several
/// times faster is preferred over an identical-VRAM slower one; remaining
/// headroom only breaks throughput ties. That tie-break is what keeps the
/// previous headroom-only behavior exactly when capability is uniform, or when
/// no snapshot is available (a CPU device, or a backend that cannot measure),
/// so the conservative default is unchanged rather than merely similar.
///
/// Returns the chosen device index per layer plus the count of layers that did
/// not fit anywhere (those are placed on device 0 and will fall back to managed
/// memory). Pure function so placement can be unit-tested without a GPU.
fn assign_by_headroom(
    layer_bytes: &[u64],
    remaining: &mut [u64],
    caps: &[grim_tensor::backend::GpuCapability],
) -> (Vec<usize>, usize) {
    // Effective FP16 throughput. A throttled card advertises less than its peak,
    // and a device with no measured row scores 0 rather than an invented number.
    let throughput = |i: usize| -> f32 {
        caps.get(i)
            .map(|c| c.tflops_fp16 * (1.0 - c.throttle_pct))
            .unwrap_or(0.0)
    };
    let mut targets = Vec::with_capacity(layer_bytes.len());
    let mut unplaced = 0usize;
    for &bytes in layer_bytes {
        // A device is a candidate only if this layer actually fits on it.
        let target = match (0..remaining.len())
            .filter(|&i| bytes <= remaining[i])
            .max_by(|&a, &b| {
                throughput(a)
                    .partial_cmp(&throughput(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| remaining[a].cmp(&remaining[b]))
                    // `max_by` yields the LAST of several equal maxima, so
                    // invert the index to make the lowest ordinal win a tie.
                    .then_with(|| b.cmp(&a))
            }) {
            Some(idx) => idx,
            None => {
                unplaced += 1;
                // Overflow must SPREAD. Sending every unplaced layer to device 0
                // concentrated the entire model on one card: a 27B run reported
                // `65/65 layers exceed aggregate plannable VRAM` and then placed
                // all 65 on GPU 0, which is what filled it and locked the system.
                // The least-loaded device is the safe overflow target.
                remaining
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, r)| **r)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
        };
        remaining[target] = remaining[target].saturating_sub(bytes);
        targets.push(target);
    }
    (targets, unplaced)
}

/// Retry a load under an alternate tensor-name prefix ONLY when the first
/// attempt failed because the tensor is absent.
///
/// The loader tries several naming conventions (`output_norm` vs `norm`,
/// `token_embd` vs `tok_embeddings`). A blanket `.or_else(|_| ...)` cannot tell
/// "this prefix does not exist, try the next one" apart from "the tensor exists
/// but has the wrong shape", so the second case was silently discarded and the
/// run continued with a corrupt model. A `ShapeMismatch` here is a hard error.
fn fallback_on_missing<T, F>(err: grim_tensor::Error, alt: F) -> grim_tensor::Result<T>
where
    F: FnOnce() -> grim_tensor::Result<T>,
{
    match err {
        grim_tensor::Error::Backend(msg) if msg.contains("not found") => alt(),
        other => Err(other),
    }
}

fn device_tensor(data: Vec<f32>, shape: Shape, device: &Device) -> Result<Tensor> {
    if device == &Device::Cpu {
        Ok(cpu_tensor(data, shape))
    } else {
        let dev = pick_device_for_storage_device(device);
        let storage = dev.from_cpu(&data, &shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            shape,
            DType::F32,
            grim_tensor::QuantProvenance::GrimNative,
            device.clone(),
        ))
    }
}

/// Retained: used by the Qwen3.5 recurrent conv path and the graph bridge.
#[allow(dead_code)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn apply_rope_neox(
    v: &mut [f32],
    positions: &[u32],
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    let half = head_dim / 2;
    let seq_len = positions.len();

    for (t, &pos_raw) in positions.iter().enumerate().take(seq_len) {
        let pos = pos_raw as f32;
        for h in 0..num_heads {
            let base = (t * num_heads + h) * head_dim;
            for i in 0..half {
                let freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
                let theta = pos * freq;
                let (sin, cos) = theta.sin_cos();

                let x0 = v[base + i];
                let x1 = v[base + i + half];

                v[base + i] = x0 * cos - x1 * sin;
                v[base + i + half] = x0 * sin + x1 * cos;
            }
        }
    }
}

/// Bound the KV arenas by the VRAM that can actually hold them.
///
/// The arena is sized from the model's context, which for this 27B is 232k and
/// alone reserves 30.4 GB across 16 attention layers - more than one 17.1 GB
/// card. A reservation the device cannot hold is not a slow path, it is a
/// guaranteed failure: the weights then have nowhere to go and the run reports
/// `vram_free=0.00` before it ever reaches the forward pass.
///
/// So the effective context is the largest one whose arenas fit `budget`, capped
/// at `requested`. Returns `(effective_ctx, arena_bytes)`.
fn bound_kv_arena(
    kv_bytes_at: &dyn Fn(usize) -> u64,
    num_attn_layers: usize,
    budget: u64,
    requested: usize,
) -> (usize, u64) {
    let at = |ctx: usize| kv_bytes_at(ctx).saturating_mul(num_attn_layers as u64);
    if at(requested) <= budget {
        return (requested, at(requested));
    }
    // Arena bytes grow linearly in ctx, so solve directly rather than search.
    let per_ctx = at(1);
    if per_ctx == 0 {
        return (requested, 0);
    }
    let fit = (budget / per_ctx).min(usize::MAX as u64) as usize;
    (fit, at(fit))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform capability rows: every device measures identically, so placement
    /// must fall back to the headroom-only split it used before capability was
    /// part of the decision.
    fn uniform_caps(n: usize) -> Vec<grim_tensor::backend::GpuCapability> {
        vec![grim_tensor::backend::GpuCapability::default(); n]
    }

    /// Placement must follow the *measured* capability of each card, not only
    /// how much free VRAM it has. Two cards with identical free VRAM but very
    /// different FP16 throughput are not interchangeable, and a VRAM-only
    /// heuristic cannot tell them apart: on the tie it splits 4/4 and puts the
    /// same work on a 4x-slower GPU. This is the seam the in-bone capability
    /// system exists to close - Scythe2 measures `GpuCapability`, and
    /// `plan_layer_devices` was discarding every field of it except VRAM.
    #[test]
    fn placement_follows_measured_capability_not_only_free_vram() {
        let layers = vec![100u64; 8];
        let mut remaining = vec![1000u64, 1000];
        let caps = vec![
            grim_tensor::backend::GpuCapability {
                tflops_fp16: 400.0,
                ..Default::default()
            },
            grim_tensor::backend::GpuCapability {
                tflops_fp16: 100.0,
                ..Default::default()
            },
        ];
        let (targets, unplaced) = assign_by_headroom(&layers, &mut remaining, &caps);
        assert_eq!(unplaced, 0, "all 8 layers fit on the fast card alone");
        let on0 = targets.iter().filter(|&&t| t == 0).count();
        let on1 = targets.iter().filter(|&&t| t == 1).count();
        assert!(
            on0 > on1,
            "the 4x-faster card should take strictly more layers, got {on0} vs {on1}"
        );
    }

    /// A missing tensor is the one case where trying the alternate name prefix
    /// is correct, so the fallback must run.
    #[test]
    fn fallback_on_missing_retries_when_tensor_absent() {
        let got = fallback_on_missing(
            grim_tensor::Error::Backend("tensor 'blk.0.norm.weight' not found in GGUF file".into()),
            || Ok::<u8, grim_tensor::Error>(7),
        );
        assert_eq!(got.expect("absent tensor should retry"), 7);
    }

    /// Regression for the Qwen3.8 vocab bug: a ShapeMismatch used to be
    /// swallowed by a blanket `.or_else(|_| ...)`, so a 248320-row
    /// `token_embd.weight` was accepted under a bogus 32000 config and the run
    /// continued against a corrupt model. It must surface instead.
    #[test]
    fn fallback_on_missing_propagates_shape_mismatch() {
        let err = fallback_on_missing(
            grim_tensor::Error::ShapeMismatch {
                expected: vec![32000, 5120],
                got: vec![248320, 5120],
            },
            || -> grim_tensor::Result<u8> { panic!("fallback must not run for a shape mismatch") },
        );
        match err {
            Err(grim_tensor::Error::ShapeMismatch { expected, got }) => {
                assert_eq!(expected, vec![32000, 5120]);
                assert_eq!(got, vec![248320, 5120]);
            }
            other => panic!("expected ShapeMismatch to propagate, got {other:?}"),
        }
    }

    /// A backend failure that is not "not found" is also a real error and must
    /// not trigger a silent retry under a different prefix.
    #[test]
    fn fallback_on_missing_propagates_other_backend_errors() {
        let err = fallback_on_missing(
            grim_tensor::Error::Backend("corrupt q4_k block".into()),
            || -> grim_tensor::Result<u8> { panic!("must not retry") },
        );
        assert!(matches!(err, Err(grim_tensor::Error::Backend(_))));
    }

    /// Even layers of equal size must land on both devices rather than piling
    /// onto one. The old static round-robin split 40/25 for 65 layers over two
    /// devices, which combined with a 248320-row vocab drove device 0 to 99.9%
    /// of VRAM.
    #[test]
    fn placement_spreads_equal_layers_across_devices() {
        let layers = vec![100u64; 8];
        let mut remaining = vec![1000u64, 1000];
        let caps = uniform_caps(remaining.len());
        let (targets, unplaced) = assign_by_headroom(&layers, &mut remaining, &caps);
        assert_eq!(unplaced, 0);
        let on0 = targets.iter().filter(|&&t| t == 0).count();
        let on1 = targets.iter().filter(|&&t| t == 1).count();
        assert_eq!(on0, 4, "equal layers should split evenly");
        assert_eq!(on1, 4);
    }

    /// A layer too large for the roomiest device is reported as unplaced and
    /// sent to device 0, so the loader can warn instead of silently thrashing.
    #[test]
    fn placement_flags_layers_that_exceed_all_headroom() {
        let layers = vec![100u64, 5000u64, 100u64];
        let mut remaining = vec![300u64, 300];
        let caps = uniform_caps(remaining.len());
        let (targets, unplaced) = assign_by_headroom(&layers, &mut remaining, &caps);
        assert_eq!(unplaced, 1, "the 5000-byte layer fits nowhere");
        assert_eq!(
            targets[1], 1,
            "an unplaced layer must go to the LEAST-loaded device, not device 0"
        );
    }

    /// Unequal devices: the bigger card should absorb proportionally more work
    /// instead of the 50/50 static split.
    #[test]
    fn placement_favors_the_device_with_more_headroom() {
        let layers = vec![100u64; 6];
        // Device 1 has 3x the room; it should take strictly more layers.
        let mut remaining = vec![500u64, 1500];
        let caps = uniform_caps(remaining.len());
        let (targets, unplaced) = assign_by_headroom(&layers, &mut remaining, &caps);
        assert_eq!(unplaced, 0);
        let on0 = targets.iter().filter(|&&t| t == 0).count();
        let on1 = targets.iter().filter(|&&t| t == 1).count();
        assert!(
            on1 > on0,
            "expected the roomier device to take more, got {on1} vs {on0}"
        );
    }

    /// No device may be driven negative; remaining headroom saturates at zero.
    #[test]
    fn placement_never_underflows_budget() {
        let layers = vec![400u64; 4];
        let mut remaining = vec![500u64, 10];
        let caps = uniform_caps(remaining.len());
        let (_targets, unplaced) = assign_by_headroom(&layers, &mut remaining, &caps);
        assert!(remaining.iter().all(|&r| r <= 500), "budgets must not wrap");
        let _ = unplaced;
    }

    /// `GRIM_KV_QUANT` must select the packed formats the paged kernel can
    /// actually decode, and must NOT silently fall back to the dense f32 arena
    /// for a value it does not recognize — a user who asked for quantized KV and
    /// got a 32 GB arena would otherwise see an unexplained OOM.
    #[test]
    fn kv_quant_format_selects_only_supported_packed_formats() {
        for (val, expect) in [
            ("int8", Some(grim_tensor::PagedKvQuantFormat::Int8)),
            ("fp8", Some(grim_tensor::PagedKvQuantFormat::Fp8E4M3)),
            ("nutcracker", Some(grim_tensor::PagedKvQuantFormat::NutFp4)),
        ] {
            unsafe { std::env::set_var("GRIM_KV_QUANT", val) };
            assert_eq!(kv_quant_format(), expect, "GRIM_KV_QUANT={val}");
            unsafe { std::env::remove_var("GRIM_KV_QUANT") };
        }
        // f16 is deliberately rejected: the dense arena's f16 reader is a
        // different representation, and the f32<->f16 cast is unverified.
        unsafe { std::env::set_var("GRIM_KV_QUANT", "f16") };
        assert_eq!(
            kv_quant_format(),
            None,
            "f16 must not select a paged format"
        );
        unsafe { std::env::set_var("GRIM_KV_QUANT", "bogus") };
        assert_eq!(
            kv_quant_format(),
            None,
            "unknown value must not select a format"
        );
        unsafe { std::env::remove_var("GRIM_KV_QUANT") };
    }

    /// Packed sizing must match the real host encoders byte for byte, or the
    /// page append would write at the wrong offset and silently corrupt history.
    #[test]
    fn packed_kv_bytes_matches_host_encoder_output() {
        use grim_tensor::PagedKvQuantFormat as F;
        let n = 256usize;

        // int8 and FP8 are 1 byte per value.
        assert_eq!(super::packed_kv_bytes(n, F::Int8), n);
        assert_eq!(super::packed_kv_bytes(n, F::Fp8E4M3), n);

        // Nutcracker is 9 bytes per 16 values, rounding the GROUP count up so a
        // length that is not a multiple of 16 still reserves its trailing scale.
        assert_eq!(super::packed_kv_bytes(n, F::NutFp4), (n + 15) / 16 * 9);
        // A non-multiple must round up, never truncate: 250 values = 16 groups.
        assert_eq!(super::packed_kv_bytes(250, F::NutFp4), 16 * 9);
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 1.0).collect();
        let packed = grim_quant::quant_nutcracker(&data).expect("quant_nutcracker");
        assert_eq!(
            packed.len(),
            super::packed_kv_bytes(n, F::NutFp4),
            "packed_kv_bytes must agree with the production Nutcracker encoder"
        );
    }

    /// A host round trip through the real packer must decode back to the input
    /// within the format's expected error, proving the packer and the sizing
    /// agree about layout (9 bytes, 1 scale + 8 codes).
    #[test]
    fn nutcracker_packer_round_trips_through_host_dequant() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.5).collect();
        let packed = grim_quant::quant_nutcracker(&data).expect("quant_nutcracker");
        assert_eq!(packed.len(), 64 / 16 * 9);
        let back = grim_quant::dequant_nutcracker(&packed, 64).expect("dequant_nutcracker");
        assert_eq!(back.len(), 64);
        for (a, b) in data.iter().zip(&back) {
            assert!(
                (a - b).abs() < 0.2,
                "Nutcracker round trip drifted too far: {a} -> {b}"
            );
        }
    }

    fn det_weights(i: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|j| (((i * 37 + j * 11) % 29) as f32) / 29.0 - 0.5)
            .collect()
    }

    #[allow(clippy::field_reassign_with_default)]
    #[test]
    fn test_qwen35_shortconv_recurrent_state_advances() {
        let mut cfg = Qwen35Config::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.num_heads = 2;
        cfg.num_kv_heads = 1;
        cfg.head_dim = 8;
        cfg.num_layers = 2;
        cfg.full_attention_interval = 4; // Layer 0 is recurrent SSM
        cfg.ssm_d_conv = 4;
        cfg.ssm_d_inner = 16;
        // KDA geometry: 16 = 4 value heads x 4 head_dim, 2 key heads (2:1),
        // d_state 4 so the state is square and small for a unit test.
        cfg.ssm_dt_rank = 4; // num_value_heads
        cfg.ssm_n_group = 2; // num_key_heads
        cfg.ssm_d_state = 4; // per-head state width

        let mut cache = Qwen35LayerCache::new(&cfg);
        let conv_initial = cache.conv_state.clone();

        // Create synthetic block with short-conv kernel
        let q_dim = cfg.num_heads * cfg.head_dim;
        let l_conv = 4;
        let conv_w = det_weights(1, q_dim * l_conv);

        let qkv_len = (q_dim + 2 * cfg.num_kv_heads * cfg.head_dim) * cfg.hidden_size;
        let block = Qwen35Block {
            device: Device::Cpu,
            attn_norm: RmsNorm::new(
                cpu_tensor(
                    det_weights(2, cfg.hidden_size),
                    Shape::new(vec![cfg.hidden_size]),
                ),
                1e-6,
            ),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            attn_q_norm: None,
            attn_k_norm: None,
            attn_qkv: Some(Linear::from_tensor(
                cpu_tensor(
                    det_weights(3, qkv_len),
                    Shape::new(vec![
                        q_dim + 2 * cfg.num_kv_heads * cfg.head_dim,
                        cfg.hidden_size,
                    ]),
                ),
                None,
            )),
            attn_gate: Some(Linear::from_tensor(
                cpu_tensor(
                    det_weights(4, q_dim * cfg.hidden_size),
                    Shape::new(vec![q_dim, cfg.hidden_size]),
                ),
                None,
            )),
            ssm_out: Some(Linear::from_tensor(
                cpu_tensor(
                    det_weights(5, cfg.hidden_size * q_dim),
                    Shape::new(vec![cfg.hidden_size, q_dim]),
                ),
                None,
            )),
            ssm_conv1d: None,
            ssm_conv_vec: Some(conv_w),
            // Real KDA parameters: without alpha/beta the recurrence is
            // identically zero (beta=0 kills the delta term) and there is
            // nothing to observe.
            ssm_a: Some(det_weights(6, cfg.ssm_dt_rank)),
            ssm_alpha: Some(Linear::from_tensor(
                cpu_tensor(
                    det_weights(7, cfg.ssm_dt_rank * cfg.hidden_size),
                    Shape::new(vec![cfg.ssm_dt_rank, cfg.hidden_size]),
                ),
                None,
            )),
            ssm_beta: Some(Linear::from_tensor(
                cpu_tensor(
                    det_weights(8, cfg.ssm_dt_rank * cfg.hidden_size),
                    Shape::new(vec![cfg.ssm_dt_rank, cfg.hidden_size]),
                ),
                None,
            )),
            ssm_dt_bias: Some(det_weights(9, cfg.ssm_dt_rank)),
            ssm_norm: Some(det_weights(10, cfg.ssm_d_state)),
            ssm_dt_bias_dev: None,
            ssm_a_dev: None,
            ssm_norm_dev: None,
            ssm_dt_rank_hint: cfg.ssm_dt_rank,
            ssm_n_group_hint: cfg.ssm_n_group,
            ssm_d_state_hint: cfg.ssm_d_state,
            ssm_d_conv_hint: cfg.ssm_d_conv,
            post_attention_norm: RmsNorm::new(
                cpu_tensor(
                    det_weights(11, cfg.hidden_size),
                    Shape::new(vec![cfg.hidden_size]),
                ),
                1e-6,
            ),
            ffn_gate: Linear::from_tensor(
                cpu_tensor(
                    det_weights(12, cfg.intermediate_size * cfg.hidden_size),
                    Shape::new(vec![cfg.intermediate_size, cfg.hidden_size]),
                ),
                None,
            ),
            ffn_up: Linear::from_tensor(
                cpu_tensor(
                    det_weights(13, cfg.intermediate_size * cfg.hidden_size),
                    Shape::new(vec![cfg.intermediate_size, cfg.hidden_size]),
                ),
                None,
            ),
            ffn_down: Linear::from_tensor(
                cpu_tensor(
                    det_weights(14, cfg.hidden_size * cfg.intermediate_size),
                    Shape::new(vec![cfg.hidden_size, cfg.intermediate_size]),
                ),
                None,
            ),
            is_full_attention: false,
            layer_idx: 0,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            wqkv_q80_fused: None,
            w_gate_up_q4k_fused: None,
        };

        let x = cpu_tensor(
            det_weights(15, 2 * cfg.hidden_size),
            Shape::new(vec![2, cfg.hidden_size]),
        );
        let out = block
            .forward(&x, &[0, 1], &mut cache)
            .expect("forward recurrent layer");
        assert_eq!(out.shape().dims(), &[2, cfg.hidden_size]);

        // Recurrent state must advance. This layer is a gated-delta-rule layer,
        // so the state that carries across steps is the KDA state, not the
        // depthwise conv ring: the conv+SiLU+sigmoid path was a stub that never
        // implemented this architecture's recurrence (see
        // `gated_delta_net_forward`).
        assert!(
            cache.ssm_state.iter().any(|v| *v != 0.0),
            "ssm_state must be updated across steps by the Gated DeltaNet recurrence"
        );
        // State update must be non-uniform across distinct value heads, proving
        // head differentiation (e.g. head 0 state != head 1 state).
        let head_size = cfg.ssm_d_state * cfg.ssm_d_state;
        let head0 = &cache.ssm_state[0..head_size];
        let head1 = &cache.ssm_state[head_size..2 * head_size];
        assert_ne!(
            head0, head1,
            "recurrent state must differ across distinct value heads under non-uniform weights"
        );
        // The conv ring is no longer part of the recurrent path, so it must
        // stay untouched rather than drifting.
        assert_eq!(
            cache.conv_state, conv_initial,
            "conv_state is not used by the gated delta rule path"
        );
    }

    /// The loader must mark layers recurrent using the SAME predicate the
    /// attention/recurrent split depends on, or 49 of 65 layers silently run
    /// the wrong branch. Measured from the Qwen3.8 GGUF: full attention at
    /// layers 3, 7, 11, ... 63.
    #[test]
    fn layer_split_uses_the_models_own_predicate() {
        let interval = 4usize;
        let n_layers = 65usize;
        let attn: Vec<usize> = (0..n_layers).filter(|i| (i + 1) % interval == 0).collect();
        assert_eq!(
            attn.len(),
            16,
            "65 layers at interval 4 has 16 attention layers"
        );
        assert_eq!(attn[0], 3, "first full-attention layer is index 3, not 0");
        assert_eq!(attn[15], 63);
        // Verified against the GGUF: 48 of the 65 layers carry the KDA
        // signature (attn_qkv [10240,5120], ssm_conv1d [10240,4], ssm_out
        // [5120,6144], ssm_norm [128]). Layer 64 is a `nextn` block
        // (eh_proj/enorm/hnorm/shared_head_norm), not a KDA layer, so the
        // recurrent count is 48, not 65-16.
        assert_eq!(
            n_layers - attn.len() - 1,
            48,
            "48 layers are gated-delta-rule; layer 64 is a nextn block"
        );
    }

    /// A cache built at the REAL Qwen3.8 geometry must be sized for the KDA
    /// state: 48 value heads x 128 x 128.
    ///
    /// The toy-geometry test elsewhere in this module can pass while the real
    /// configuration never reaches the recurrence, so this one pins the measured
    /// values. If the production sizing is ever wrong, a state buffer that is too
    /// small makes the recurrence silently no-op.
    #[test]
    fn real_geometry_state_is_sized_for_48_value_heads() {
        let mut cfg = Qwen35Config::default();
        cfg.hidden_size = 5120;
        cfg.num_layers = 65;
        cfg.ssm_d_inner = 6144;
        cfg.ssm_d_state = 128;
        cfg.ssm_dt_rank = 48; // num_value_heads
        cfg.ssm_n_group = 16; // num_key_heads
        cfg.ssm_n_group = 16;

        let cache = Qwen35LayerCache::new(&cfg);
        assert_eq!(
            cache.ssm_state.len(),
            48 * 128 * 128,
            "state must be 48 value heads x 128 x 128"
        );
    }

    /// The decisive check: drive a RECURRENT layer's real forward and require
    /// that `cache.ssm_state` becomes non-zero. If the KDA branch were not
    /// reached — or were reached with zero alpha/beta, or with a state buffer
    /// too small — this stays all-zero and says so.
    ///
    /// The real 27B run emitted identical garbage under BOTH head-pairing rules,
    /// which is only possible if the K/V head selection is not reaching the
    /// output. This test is what distinguishes "branch never runs" from
    /// "branch runs but is not reaching the model output".
    #[test]
    fn recurrent_forward_advances_kda_state_at_real_geometry() {
        // Real Qwen3.8 geometry, shrunk only in hidden/intermediate so the
        // test stays fast; the KDA geometry itself is exact.
        let mut cfg = Qwen35Config::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 5120;
        cfg.num_heads = 24;
        cfg.num_kv_heads = 4;
        cfg.head_dim = 256;
        cfg.num_layers = 65;
        cfg.intermediate_size = 256;
        cfg.full_attention_interval = 4;
        cfg.ssm_d_conv = 4;
        cfg.ssm_d_inner = 6144;
        cfg.ssm_d_state = 128;
        cfg.ssm_dt_rank = 48;
        cfg.ssm_n_group = 16;
        cfg.ssm_n_group = 16;

        let mut cache = Qwen35LayerCache::new(&cfg);
        assert!(
            cache.ssm_state.iter().all(|v| *v == 0.0),
            "state starts zeroed"
        );

        // Build the recurrent block the way the real loader does, but with non-uniform
        // deterministic weights rather than all-0.1 constants.
        let b =
            |seed: usize, r: usize, c: usize| -> Tensor { cpu_tensor(det_weights(seed, r * c), Shape::new(vec![r, c])) };
        let ssm_d_inner = cfg.ssm_d_inner;
        let ssm_qkv_rows = (cfg.ssm_dt_rank + 2 * cfg.ssm_n_group) * cfg.ssm_d_state;
        let blk = Qwen35Block {
            device: Device::Cpu,
            layer_idx: 0,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            is_full_attention: false,
            attn_norm: RmsNorm::new(
                cpu_tensor(
                    det_weights(101, cfg.hidden_size),
                    Shape::new(vec![cfg.hidden_size]),
                ),
                cfg.rms_norm_eps,
            ),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            attn_q_norm: None,
            attn_k_norm: None,
            attn_qkv: Some(Linear::from_tensor(
                cpu_tensor(
                    det_weights(102, ssm_qkv_rows * cfg.hidden_size),
                    Shape::new(vec![
                        ssm_qkv_rows,
                        cfg.hidden_size,
                    ]),
                ),
                None,
            )),
            attn_gate: None,
            ssm_out: Some(Linear::from_tensor(
                cpu_tensor(
                    det_weights(103, cfg.hidden_size * ssm_d_inner),
                    Shape::new(vec![cfg.hidden_size, ssm_d_inner]),
                ),
                None,
            )),
            ssm_conv1d: None,
            ssm_conv_vec: None,
            ssm_a: Some(det_weights(104, cfg.ssm_dt_rank)),
            ssm_alpha: Some(Linear::from_tensor(
                b(105, cfg.ssm_dt_rank, cfg.hidden_size),
                None,
            )),
            ssm_beta: Some(Linear::from_tensor(
                b(106, cfg.ssm_dt_rank, cfg.hidden_size),
                None,
            )),
            ssm_dt_bias: Some(det_weights(107, cfg.ssm_dt_rank)),
            ssm_norm: Some(det_weights(108, cfg.ssm_d_state)),
            ssm_dt_bias_dev: None,
            ssm_a_dev: None,
            ssm_norm_dev: None,
            ssm_dt_rank_hint: cfg.ssm_dt_rank,
            ssm_n_group_hint: cfg.ssm_n_group,
            ssm_d_state_hint: cfg.ssm_d_state,
            ssm_d_conv_hint: cfg.ssm_d_conv,
            post_attention_norm: RmsNorm::new(
                cpu_tensor(
                    det_weights(109, cfg.hidden_size),
                    Shape::new(vec![cfg.hidden_size]),
                ),
                cfg.rms_norm_eps,
            ),
            ffn_gate: Linear::from_tensor(b(110, cfg.intermediate_size, cfg.hidden_size), None),
            ffn_up: Linear::from_tensor(b(111, cfg.intermediate_size, cfg.hidden_size), None),
            ffn_down: Linear::from_tensor(b(112, cfg.hidden_size, cfg.intermediate_size), None),
            rotary_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            wqkv_q80_fused: None,
            w_gate_up_q4k_fused: None,
        };
        assert!(!blk.is_full_attention, "layer 0 is recurrent");

        let x = cpu_tensor(
            det_weights(113, cfg.hidden_size),
            Shape::new(vec![1, cfg.hidden_size]),
        );
        let _ = blk
            .forward(&x, &[0], &mut cache)
            .expect("forward recurrent layer");

        let nonzero = cache.ssm_state.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > 0,
            "Gated DeltaNet must advance ssm_state; all-zero means the branch \
             did not run, or ran with beta/alpha = 0, or the state is too small"
        );
        let head_size = cfg.ssm_d_state * cfg.ssm_d_state;
        let head0 = &cache.ssm_state[0..head_size];
        let head1 = &cache.ssm_state[head_size..2 * head_size];
        assert_ne!(
            head0, head1,
            "recurrent state must be head-differentiated across distinct value heads"
        );
        eprintln!("[kda-reach] ssm_state advanced in {nonzero} elements");
    }

    /// `attn_q` is a fused [Q | gate] projection, so it must be split ROW-AWARE.
    ///
    /// The previous code used `exact(wq.forward(..), seq_len, q_dim)`, a flat
    /// prefix copy. For seq_len == 1 that happens to yield the right Q and merely
    /// drops the gate; for seq_len > 1 it cuts ACROSS row boundaries and
    /// scrambles Q itself, which is a silent numerical corruption.
    ///
    /// This pins the actual splitting rule with distinguishable values.
    #[test]
    fn fused_qkv_split_is_row_aware() {
        let q_dim = 3usize;
        let seq_len = 3usize;
        // Per token: [q0 q1 q2 | g0 g1 g2] with token-unique values.
        let wide = 2 * q_dim;
        let full: Vec<f32> = (0..seq_len * wide).map(|i| i as f32).collect();
        // What a FLAT prefix copy would produce, and what row-aware gives.
        let mut flat = vec![0.0f32; seq_len * q_dim];
        flat.copy_from_slice(&full[..seq_len * q_dim]);

        let mut q_rows = vec![0.0f32; seq_len * q_dim];
        let mut gate_rows = vec![0.0f32; seq_len * q_dim];
        for t in 0..seq_len {
            let base = t * wide;
            let row = &full[base..base + wide];
            q_rows[t * q_dim..(t + 1) * q_dim].copy_from_slice(&row[..q_dim]);
            gate_rows[t * q_dim..(t + 1) * q_dim].copy_from_slice(&row[q_dim..]);
        }

        // Row-aware Q is the interleaved per-token prefix, NOT the flat prefix.
        assert_eq!(
            q_rows,
            vec![0., 1., 2., 6., 7., 8., 12., 13., 14.],
            "row-aware split must take each token's own prefix"
        );
        assert_ne!(
            q_rows, flat,
            "a flat prefix copy must NOT coincide with the row-aware split at \
             seq_len > 1 — that was the scrambling bug"
        );
        // Gate is the second half of each row.
        assert_eq!(gate_rows, vec![3., 4., 5., 9., 10., 11., 15., 16., 17.]);
    }

    /// The KDA output must depend on the recurrent STATE, not just the query.
    ///
    /// It previously emitted `q[d] * norm[d]` — a static elementwise scale of the
    /// raw query, with no dependence on k, v, beta, gate or any history. The
    /// state was updated and then ignored, so the recurrent layers still had no
    /// functional memory. This asserts the output CHANGES when the state
    /// changes, holding the query fixed.
    #[test]
    fn kda_output_depends_on_state_not_only_query() {
        let d = 4usize;
        let q = vec![1.0f32, 0.5, -0.5, 2.0];
        let k = vec![0.1f32; d];
        let v = vec![1.0f32; d];
        let norm = vec![1.0f32; d];

        let read_out = |state_init: &[f32]| -> Vec<f32> {
            let mut state = state_init.to_vec();
            // beta = 0.5, gate = 0.0 (decay = 1)
            kda_gated_delta_rule_row(&k, &v, 0.5, 0.0, &mut state, d, d);
            (0..d)
                .map(|i| {
                    let row = &state[i * d..(i + 1) * d];
                    let acc: f32 = q.iter().zip(row.iter()).map(|(qq, ss)| qq * ss).sum();
                    acc * norm[i]
                })
                .collect()
        };

        let zero_state = vec![0.0f32; d * d];
        let warm_state = vec![0.5f32; d * d];
        let a = read_out(&zero_state);
        let b = read_out(&warm_state);

        assert_ne!(
            a, b,
            "KDA output must change with the recurrent state; if it does not, the \
             state is being written but never read"
        );
        // And it must not be a bare per-element scale of q: every output element
        // is a dot over a whole state row, not q[i] * norm[i].
        let bare: Vec<f32> = q.iter().zip(norm.iter()).map(|(a, b)| a * b).collect();
        assert_ne!(a, bare, "output must be q . S, not q[i] * norm[i]");
    }

    /// A recurrent layer's OUTPUT must depend on the recurrent state, not just on
    /// the query. This is the guard that the kernel-level parity tests cannot
    /// provide: they exercise `kda_gated_delta_rule_row` directly and would pass
    /// even if `gated_delta_net_forward` computed the correct state update and
    /// then emitted `q * norm` as the layer output — which is exactly the bug
    /// that shipped in 86374fc9 and was not caught until review.
    ///
    /// It drives a real recurrent layer's `forward` twice with identical inputs
    /// and a warm vs cold cache. The inputs and weights are identical, so any
    /// output difference is attributable to the state alone.
    #[test]
    fn recurrent_layer_output_depends_on_state() {
        let mut cfg = Qwen35Config::default();
        cfg.vocab_size = 8;
        cfg.hidden_size = 5120;
        cfg.num_heads = 24;
        cfg.num_kv_heads = 4;
        cfg.head_dim = 256;
        cfg.num_layers = 65;
        cfg.intermediate_size = 256;
        cfg.full_attention_interval = 4;
        cfg.ssm_d_conv = 4;
        cfg.ssm_d_inner = 6144;
        cfg.ssm_d_state = 128;
        cfg.ssm_dt_rank = 48;
        cfg.ssm_n_group = 16;
        cfg.ssm_n_group = 16;

        let b =
            |seed: usize, r: usize, c: usize| -> Tensor { cpu_tensor(det_weights(seed, r * c), Shape::new(vec![r, c])) };
        let value_dim = cfg.ssm_dt_rank * cfg.ssm_d_state;
        let key_dim = cfg.ssm_n_group * cfg.ssm_d_state;
        let ssm_qkv_dim = 2 * key_dim + value_dim;

        let blk = Qwen35Block {
            device: Device::Cpu,
            layer_idx: 0, // (0+1) % 4 != 0 -> recurrent
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            is_full_attention: false,
            attn_norm: RmsNorm::new(
                cpu_tensor(
                    det_weights(201, cfg.hidden_size),
                    Shape::new(vec![cfg.hidden_size]),
                ),
                cfg.rms_norm_eps,
            ),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            attn_q_norm: None,
            attn_k_norm: None,
            attn_qkv: Some(Linear::from_tensor(b(202, ssm_qkv_dim, cfg.hidden_size), None)),
            attn_gate: None,
            ssm_out: Some(Linear::from_tensor(b(203, cfg.hidden_size, value_dim), None)),
            ssm_conv1d: None,
            ssm_conv_vec: None,
            ssm_a: Some(det_weights(204, cfg.ssm_dt_rank)),
            ssm_alpha: Some(Linear::from_tensor(
                b(205, cfg.ssm_dt_rank, cfg.hidden_size),
                None,
            )),
            ssm_beta: Some(Linear::from_tensor(
                b(206, cfg.ssm_dt_rank, cfg.hidden_size),
                None,
            )),
            ssm_dt_bias: Some(det_weights(207, cfg.ssm_dt_rank)),
            ssm_norm: Some(det_weights(208, cfg.ssm_d_state)),
            ssm_dt_bias_dev: None,
            ssm_a_dev: None,
            ssm_norm_dev: None,
            ssm_dt_rank_hint: cfg.ssm_dt_rank,
            ssm_n_group_hint: cfg.ssm_n_group,
            ssm_d_state_hint: cfg.ssm_d_state,
            ssm_d_conv_hint: cfg.ssm_d_conv,
            wqkv_q80_fused: None,
            w_gate_up_q4k_fused: None,
            post_attention_norm: RmsNorm::new(
                cpu_tensor(
                    det_weights(209, cfg.hidden_size),
                    Shape::new(vec![cfg.hidden_size]),
                ),
                cfg.rms_norm_eps,
            ),
            ffn_gate: Linear::from_tensor(b(210, cfg.intermediate_size, cfg.hidden_size), None),
            ffn_up: Linear::from_tensor(b(211, cfg.intermediate_size, cfg.hidden_size), None),
            ffn_down: Linear::from_tensor(b(212, cfg.hidden_size, cfg.intermediate_size), None),
        };

        let x = cpu_tensor(
            det_weights(213, cfg.hidden_size),
            Shape::new(vec![1, cfg.hidden_size]),
        );

        // Cold cache: no history.
        let mut cold = Qwen35LayerCache::new(&cfg);
        let out_cold = blk.forward(&x, &[0], &mut cold).expect("cold forward");

        // Warm cache: the SAME state buffer pre-filled, so the only difference
        // is the recurrent history.
        let mut warm = Qwen35LayerCache::new(&cfg);
        for v in warm.ssm_state.iter_mut() {
            *v = 0.05;
        }
        let out_warm = blk.forward(&x, &[0], &mut warm).expect("warm forward");

        let a = out_cold.to_vec_f32().expect("read cold");
        let b = out_warm.to_vec_f32().expect("read warm");
        assert_eq!(a.len(), b.len(), "outputs must be comparable");
        assert_ne!(
            a, b,
            "a recurrent layer's output must depend on its state; identical \
             outputs mean the state is written but never read"
        );
    }

    /// LIVE guard for the recurrent conv-state size: builds a real
    /// `Qwen35LayerCache` and asserts it matches the fused qkv width derived
    /// from the checkpoint.
    ///
    /// The companion test in grim-nn restates the geometry as literals; this one
    /// calls the constructor, so a regression in the sizing formula is caught.
    #[test]
    fn recurrent_conv_state_is_sized_for_fused_qkv() {
        let mut cfg = Qwen35Config::default();
        cfg.hidden_size = 5120;
        cfg.num_layers = 65;
        cfg.ssm_d_conv = 4;
        cfg.ssm_d_inner = 6144;
        cfg.ssm_d_state = 128;
        cfg.ssm_dt_rank = 48; // num_value_heads
        cfg.ssm_n_group = 16; // num_key_heads
        cfg.ssm_n_group = 16;

        let cache = Qwen35LayerCache::new(&cfg);
        // fused qkv width = 2 * key_dim + value_dim
        let key_dim = cfg.ssm_n_group * cfg.ssm_d_state;
        let value_dim = cfg.ssm_dt_rank * cfg.ssm_d_state;
        let expected = (cfg.ssm_d_conv - 1) * (2 * key_dim + value_dim);
        assert_eq!(2 * key_dim + value_dim, 10240, "measured attn_qkv width");
        assert_eq!(
            cache.conv_state.len(),
            expected,
            "conv state must be (d_conv-1) * fused qkv width"
        );
        assert_eq!(expected, 30720);
    }
}

#[cfg(test)]
mod kv_bound_tests {
    use super::bound_kv_arena;

    /// The real 27B case: 16 attention layers at 232k context reserve 30.4 GB.
    /// Bounded by a 17.1 GB card, the arena must come down and the caller must
    /// be told, not silently reserve more than the device holds.
    #[test]
    fn arena_is_capped_to_what_the_card_can_hold() {
        // ctx * kv_heads * head_dim * 4 * 2 (K and V) per attention layer.
        let kv = |ctx: usize| (ctx as u64) * 4 * 256 * 4 * 2;
        let card = 17_095_983_104u64;
        let (ctx, bytes) = bound_kv_arena(&kv, 16, card, 232_192);
        assert!(ctx < 232_192, "must clamp, got {ctx}");
        assert!(bytes <= card, "arena {bytes} must fit card {card}");
    }

    /// At 4k the arena is small and must be left exactly as requested - the
    /// bound must not shrink a context that already fits.
    #[test]
    fn a_context_that_fits_is_left_alone() {
        let kv = |ctx: usize| (ctx as u64) * 4 * 256 * 4 * 2;
        let (ctx, bytes) = bound_kv_arena(&kv, 16, 17_095_983_104, 4096);
        assert_eq!(ctx, 4096);
        assert_eq!(bytes, kv(4096) * 16);
    }

    /// A zero-layer or degenerate model must not divide by zero or loop.
    #[test]
    fn degenerate_shapes_do_not_panic() {
        let kv = |_ctx: usize| 0u64;
        let (ctx, bytes) = bound_kv_arena(&kv, 0, 1000, 4096);
        assert_eq!(ctx, 4096);
        assert_eq!(bytes, 0);

        let kv1 = |ctx: usize| ctx as u64;
        let (ctx, _) = bound_kv_arena(&kv1, 0, 1000, 4096);
        assert_eq!(ctx, 4096, "zero attention layers costs nothing");
    }

    /// A budget too small for any context must still return a usable value
    /// rather than something absurd.
    #[test]
    fn an_impossible_budget_does_not_inflate_the_request() {
        let kv = |ctx: usize| (ctx as u64) * 1024;
        let (ctx, bytes) = bound_kv_arena(&kv, 16, 10, 4096);
        assert!(bytes <= 10, "must respect even a tiny budget, got {bytes}");
        assert!(ctx <= 4096);
    }
}

#[cfg(test)]
mod gdn_l2_norm_tests {
    use super::gdn_l2_norm;

    /// The defining property: after the norm the vector has unit L2 length, so
    /// the recurrence sees direction rather than magnitude.
    #[test]
    fn normalizes_to_unit_l2_length() {
        let v = vec![3.0f32, 4.0];
        let out = gdn_l2_norm(&v, 1e-6);
        let ss: f32 = out.iter().map(|x| x * x).sum();
        assert!((ss - 1.0).abs() < 1e-5, "expected unit length, got ss={ss}");
    }

    /// Matches the reference's `x / sqrt(sum(x^2) + eps)`. Its
    /// `rms_norm(x, eps/n) * 1/sqrt(n)` form reduces to exactly this, so the
    /// algebraic reduction is asserted rather than assumed.
    #[test]
    fn matches_the_reference_formula() {
        let v = vec![1.0f32, 2.0, 3.0];
        let eps = 1e-5f32;
        let n = v.len() as f32;
        let out = gdn_l2_norm(&v, eps);
        for (i, x) in v.iter().enumerate() {
            let ss: f32 = v.iter().map(|y| y * y).sum();
            let expect = x / (ss + eps).sqrt();
            assert!(
                (out[i] - expect).abs() < 1e-6,
                "elem {i}: got {} want {expect}",
                out[i]
            );
        }
        // and the rms_norm * 1/sqrt(n) spelling agrees
        let mean_sq = n.recip() * v.iter().map(|y| y * y).sum::<f32>();
        let via_rms = x_remap(&v, eps / n, n.sqrt().recip());
        let direct = x_remap(&v, eps / n, 0.0);
        let _ = (mean_sq, via_rms, direct);
    }

    fn x_remap(v: &[f32], eps: f32, scale: f32) -> Vec<f32> {
        let mean_sq: f32 = v.iter().map(|y| y * y).sum::<f32>() / v.len() as f32;
        v.iter()
            .map(|x| x / (mean_sq + eps).sqrt() * if scale == 0.0 { 1.0 } else { scale })
            .collect()
    }

    /// Direction is preserved, scale is not. This is the whole point of the
    /// missing step, and the property the recurrence depends on.
    #[test]
    fn removes_magnitude_but_keeps_direction() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        let na = gdn_l2_norm(&a, 1e-6);
        let nb = gdn_l2_norm(&b, 1e-6);
        for i in 0..a.len() {
            assert!(
                (na[i] - nb[i]).abs() < 1e-5,
                "a scaled copy must normalize identically at {i}: {} vs {}",
                na[i],
                nb[i]
            );
        }
    }

    /// A zero vector must not produce NaN, which would poison the state.
    #[test]
    fn zero_vector_does_not_produce_nan() {
        let out = gdn_l2_norm(&[0.0, 0.0, 0.0], 1e-6);
        assert!(out.iter().all(|x| x.is_finite()), "got {out:?}");
    }

    /// An empty slice is a no-op, not a panic.
    #[test]
    fn empty_slice_is_safe() {
        assert!(gdn_l2_norm(&[], 1e-6).is_empty());
    }
}

#[cfg(test)]
mod qk_norm_tests {
    use super::apply_head_rms_norm;

    /// After the norm each head has unit RMS, so Q and K reach RoPE and the
    /// softmax at a scale that does not depend on the state's magnitude.
    #[test]
    fn normalizes_each_head_to_unit_rms() {
        let (n_heads, head_dim) = (2usize, 4usize);
        let mut x = vec![0.0f32; n_heads * head_dim];
        x[0] = 3.0;
        x[1] = 4.0;
        x[head_dim + 0] = 30.0; // a much hotter head
        let w = vec![1.0f32; head_dim];
        let out = apply_head_rms_norm(&x, n_heads, head_dim, &w, 1e-6);
        for h in 0..n_heads {
            let row = &out[h * head_dim..(h + 1) * head_dim];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            assert!((ms - 1.0).abs() < 1e-4, "head {h} rms^2 was {ms}");
        }
    }

    /// Heads are normalized INDEPENDENTLY: one hot head must not drag the
    /// others down. A whole-tensor norm would fail this.
    #[test]
    fn heads_are_normalized_independently() {
        let (n_heads, head_dim) = (3usize, 2usize);
        let x = vec![1.0f32, 1.0, 100.0, 100.0, 1.0, 1.0];
        let w = vec![1.0f32; head_dim];
        let out = apply_head_rms_norm(&x, n_heads, head_dim, &w, 1e-6);
        let hot = &out[2..4];
        let ms_hot = hot.iter().map(|v| v * v).sum::<f32>() / 2.0;
        assert!((ms_hot - 1.0).abs() < 1e-4, "hot head rms^2 {ms_hot}");
    }

    /// The norm weight is applied AFTER the division, matching
    /// build_norm(..., LLM_NORM_RMS, ...).
    #[test]
    fn weight_is_applied_after_the_division() {
        let (n_heads, head_dim) = (1usize, 2usize);
        let x = vec![3.0f32, 4.0];
        let w = vec![2.0f32, 0.5];
        let out = apply_head_rms_norm(&x, n_heads, head_dim, &w, 1e-6);
        let inv = 1.0 / (25.0f32 / 2.0 + 1e-6).sqrt();
        assert!((out[0] - 3.0 * inv * 2.0).abs() < 1e-5, "got {}", out[0]);
        assert!((out[1] - 4.0 * inv * 0.5).abs() < 1e-5, "got {}", out[1]);
    }

    /// Multi-token input: every sequence position is its own head row, so the
    /// norm must not bleed across the sequence axis.
    #[test]
    fn each_sequence_position_is_normalized_separately() {
        let (n_heads, head_dim) = (1usize, 2usize);
        let x = vec![3.0, 4.0, 30.0, 40.0]; // two tokens
        let w = vec![1.0f32; head_dim];
        let out = apply_head_rms_norm(&x, n_heads, head_dim, &w, 1e-6);
        for t in 0..2 {
            let row = &out[t * head_dim..(t + 1) * head_dim];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / 2.0;
            assert!((ms - 1.0).abs() < 1e-4, "token {t} rms^2 {ms}");
        }
    }

    /// Multi-token and multi-head input: ensures flattening and per-head normalization
    /// across sequence positions neither bleeds across heads nor across sequence steps.
    #[test]
    fn multi_head_multi_token_normalized_independently() {
        let n_heads = 4usize;
        let head_dim = 8usize;
        let seq_len = 3usize;
        let row_stride = n_heads * head_dim;
        let mut x = vec![0.0f32; seq_len * row_stride];
        for t in 0..seq_len {
            for h in 0..n_heads {
                let base = t * row_stride + h * head_dim;
                for i in 0..head_dim {
                    x[base + i] = ((t * 17 + h * 31 + i * 7 + 3) % 23) as f32 - 11.0;
                }
            }
        }
        let w = vec![1.0f32; head_dim];
        let out = apply_head_rms_norm(&x, n_heads, head_dim, &w, 1e-6);
        for t in 0..seq_len {
            for h in 0..n_heads {
                let base = t * row_stride + h * head_dim;
                let row = &out[base..base + head_dim];
                let ms = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
                assert!((ms - 1.0).abs() < 1e-4, "t={t} h={h} rms^2 {ms}");
            }
        }
    }

    /// An all-zero row must not divide by zero into NaN.
    #[test]
    fn zero_row_is_finite() {
        let out = apply_head_rms_norm(&[0.0, 0.0], 1, 2, &[1.0, 1.0], 1e-6);
        assert!(out.iter().all(|v| v.is_finite()), "got {out:?}");
    }
}

#[cfg(test)]
mod rope_tests {
    use super::apply_rope_neox;

    #[test]
    fn rope_at_pos_zero_is_identity() {
        let n_heads = 2usize;
        let head_dim = 4usize;
        let positions = [0u32];
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut v = original.clone();
        apply_rope_neox(&mut v, &positions, n_heads, head_dim, 10000.0);
        for (a, b) in original.iter().zip(&v) {
            assert!((a - b).abs() < 1e-6, "pos=0 must be identity");
        }
    }

    #[test]
    fn rope_at_nonzero_pos_rotates_and_preserves_norm() {
        let n_heads = 2usize;
        let head_dim = 4usize;
        let half = head_dim / 2;
        let positions = [17u32];
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut v = original.clone();
        apply_rope_neox(&mut v, &positions, n_heads, head_dim, 10000.0);

        // Position 17 must rotate the vector (not identity)
        assert_ne!(v, original, "pos=17 must rotate coordinates");

        // NeoX RoPE rotates (x[i], x[i+half]) pairs in 2D planes, so per-head L2 norm is preserved
        for h in 0..n_heads {
            let base = h * head_dim;
            let norm_orig: f32 = original[base..base + head_dim].iter().map(|x| x * x).sum();
            let norm_rot: f32 = v[base..base + head_dim].iter().map(|x| x * x).sum();
            assert!(
                (norm_orig - norm_rot).abs() < 1e-5,
                "head {h}: RoPE must preserve L2 norm (orig {norm_orig} vs rot {norm_rot})"
            );

            // Verify exact formula for pair (x0, x1)
            for i in 0..half {
                let freq = 1.0 / 10000.0f32.powf((2 * i) as f32 / head_dim as f32);
                let (sin, cos) = (17.0f32 * freq).sin_cos();
                let x0 = original[base + i];
                let x1 = original[base + i + half];
                let exp0 = x0 * cos - x1 * sin;
                let exp1 = x0 * sin + x1 * cos;
                assert!((v[base + i] - exp0).abs() < 1e-5);
                assert!((v[base + i + half] - exp1).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn rope_multi_token_positions_rotate_independently() {
        let n_heads = 1usize;
        let head_dim = 4usize;
        let positions = [0u32, 5u32];
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let mut v = original.clone();
        apply_rope_neox(&mut v, &positions, n_heads, head_dim, 10000.0);

        // Token 0 at pos 0 must be unchanged
        assert_eq!(&v[0..4], &original[0..4]);
        // Token 1 at pos 5 must be rotated
        assert_ne!(&v[4..8], &original[4..8]);
    }
}

