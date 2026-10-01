//! Xing4.0 (XingChen-AGI `Xing4.0-29B-A4B`) architecture: DeepSeek-V3-style
//! latent attention (MLA), `noaux_tc` sigmoid MoE routing, and Manifold
//! Hyper-Connections (MHC) — a **multi-stream residual** with a Sinkhorn
//! doubly-stochastic stream combiner.
//!
//! # Architecture
//!
//! * **MLA** (`q_lora_rank 768` → `q_a_proj/q_a_layernorm/q_b_proj`, `kv_lora_rank 512`
//!   with shared-MQA `kv_a_proj_with_mqa`, per-head `kv_b_proj`) — absorbed-latent
//!   attention, identical in shape to `deepseek32`.
//! * **MoE**: 64 routed experts (`moe_intermediate 1024`), 4 active per token,
//!   `scoring_func = sigmoid` with a learned `e_score_correction_bias` that only
//!   affects *selection* (`topk(sigmoid(logits) + bias)`), never the combine
//!   weights (`sigmoid(logits)` gathered, then `norm_topk_prob`).
//!   1 always-on shared expert; `first_k_dense_replace = 2` dense SwiGLU layers.
//! * **MHC** (`hc_mult = 4`): the residual state carries **4 parallel hidden
//!   streams** instead of one. Each block runs two hyper-connection modules
//!   (`attn_hc`, `ffn_hc`) that emit
//!   - `pre`  — sigmoid gate collapsing the streams into the block input,
//!   - `post` — `2*sigmoid` gate writing the block output into every stream,
//!   - `comb` — an `hc_mult × hc_mult` Sinkhorn-normalized (doubly stochastic)
//!     matrix mixing the streams.
//!
//!   The layer update is `streams = post ⊗ f(streams) + comb @ streams`; the model
//!   head collapses the streams with a mean before `norm` + `lm_head`.
//!
//! # Quantization arms
//!
//! Every projection goes through `Linear`, so all storage formats the provider
//! serves are load- and run-able: F32/BF16/F16 reference, Q4_K/Q5_K/Q6_K/Q8_0
//! (GGUF K-quant), FP8 (block-scaled, `weight_scale_inv` siblings folded by
//! `grim-format`), MXFP4/MXFP8, W4A16/WNA16/AWQ/OSTQuant/CompressedTensors.
//! The latent-absorbed MLA decode kernel declines quantized `kv_b_proj` and
//! falls back to the documented host path, which dequantizes through `Linear`.
//!
//! The `num_nextn_predict_layers` (MTP) head at `model.layers.{num_layers}` is
//! intentionally **not** loaded — it is a speculative-decoding head, not part of
//! the primary forward, matching the reference implementation's
//! `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.40.*"]`.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::pick_device_for_tensor;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{
    ArithType, AttentionOps, BackendStorage, CoreTensorOps, DType, Device, ElementwiseOps,
    QuantProvenance, RopeConfig, Shape, Tensor,
};
use std::sync::{Arc, OnceLock};

// Config

/// Configuration for the Xing4.0 architecture (`Xing4.0-29B-A4B` defaults).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Xing40Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    // MLA
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    /// YaRN rope scaling declared by the checkpoint
    /// (`xing4_0.rope.scaling.*`).
    pub rope_yarn: Option<grim_tensor::YaRNParams>,
    pub v_head_dim: usize,
    // MoE
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub routed_scaling_factor: f32,
    /// `topk_method = noaux_tc` — the correction bias steers selection only.
    pub noaux_tc_routing: bool,
    // Manifold hyper-connections
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub mhc_h_res_clamp_min: f32,
    pub mhc_h_res_clamp_max: f32,
    // Norm / rope
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl Default for Xing40Config {
    fn default() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 3584,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 64,
            num_layers: 40,
            intermediate_size: 9216,
            kv_lora_rank: 512,
            q_lora_rank: Some(768),
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            rope_yarn: None,
            moe_intermediate_size: 1024,
            n_routed_experts: 64,
            n_shared_experts: 1,
            num_experts_per_tok: 4,
            first_k_dense_replace: 2,
            routed_scaling_factor: 2.0,
            noaux_tc_routing: true,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            mhc_h_res_clamp_min: -30.0,
            mhc_h_res_clamp_max: 30.0,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 262144,
        }
    }
}

impl ModelConfig for Xing40Config {
    fn name(&self) -> &str {
        "xing4_0"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// GPU fallback guard

/// `Ok(None)` marks "backend lacks the kernel — use the host fallback";
/// other errors are real failures and propagate.
fn or_host_fallback<T>(r: std::result::Result<T, grim_tensor::Error>) -> Result<Option<T>> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e) if grim_nn::is_kernel_unimplemented(&e) => Ok(None),
        Err(e) => Err(grim_core::error::Error::from(e)),
    }
}

// Manifold hyper-connection (MHC)

/// Gates emitted by one hyper-connection application, all host-side scalars.
/// A device-resident F32 tensor from host data, tagged with `device`.
///
/// Used for the constants the hyper-connection materializes itself (the all-ones
/// unweighted-norm vector), which must live wherever the model lives or the norm
/// kernel gets a host pointer.
fn const_tensor(data: Vec<f32>, shape: Shape, device: &Device) -> Result<Tensor> {
    if device.is_cpu() {
        return Ok(cpu_tensor(data, shape));
    }
    let dev = grim_nn::modules::pick_device_for_storage_device(device);
    let storage = dev.from_cpu(&data, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        QuantProvenance::default(),
        device.clone(),
    ))
}

/// Seed the `hc` hyper-connection streams from a device-resident embedding by
/// writing the same `[seq, hidden]` block into each stream's column range.
pub fn seed_streams_device(
    x0: &Tensor,
    hc: usize,
    hidden: usize,
    seq_len: usize,
    full: &Shape,
    device: &Device,
) -> Result<Tensor> {
    let dev = grim_nn::modules::pick_device_for_storage_device(device);
    let mut dst = dev.zeros(full, DType::F32)?;
    for h in 0..hc {
        dev.write_cols(
            dst.as_mut(),
            hc * hidden,
            h * hidden,
            x0.storage().as_ref(),
            seq_len,
            hidden,
        )?;
    }
    Ok(Tensor::new(
        Arc::from(dst),
        full.clone(),
        DType::F32,
        QuantProvenance::default(),
        device.clone(),
    ))
}

/// Mean the `hc` hyper-connection streams into the single `[seq, hidden]` model
/// output, on device. Each stream is scaled by `1/hc` and accumulated, so the
/// streams never leave the device.

/// Env-gated stage fingerprints inside the MLA module (GRIM_XING_TRACE=1):
/// rms + first values per named tensor, so a host/device divergence can be
/// bisected to one stage instead of the module's final output.
fn xing_trace_stage(name: &str, t: &Tensor) {
    if std::env::var_os("GRIM_XING_TRACE").is_none() {
        return;
    }
    if let Ok(v) = t.to_vec_f32() {
        let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
        eprintln!("[xing-mla] {name}: len {} rms {rms:.5e} head {:?}", v.len(), &v[..v.len().min(4)]);
    }
}

fn mean_collapse(
    streams: &Tensor,
    hc: usize,
    hidden: usize,
    seq_len: usize,
    plane: &Shape,
) -> Result<Tensor> {
    if streams.device().is_cpu() {
        let v = streams.to_vec_f32()?;
        let mut out = vec![0.0f32; seq_len * hidden];
        for s in 0..seq_len {
            for h in 0..hc {
                let src = (s * hc + h) * hidden;
                let dst = s * hidden;
                for d in 0..hidden {
                    out[dst + d] += v[src + d] / hc as f32;
                }
            }
        }
        return Ok(cpu_tensor(out, plane.clone()));
    }
    let dev = pick_device_for_tensor(streams);
    let flat = hc * hidden;
    let inv = 1.0f32 / hc as f32;
    let mut acc: Option<Box<dyn BackendStorage>> = None;
    for h in 0..hc {
        let (src, _) = dev.narrow_cols(
            streams.storage().as_ref(),
            flat,
            h * hidden,
            seq_len,
            hidden,
            plane,
        )?;
        let (scaled, _) = dev.mul_scalar(src.as_ref(), inv, plane)?;
        acc = Some(match acc {
            None => scaled,
            Some(a) => dev.add(a.as_ref(), scaled.as_ref(), plane)?.0,
        });
    }
    let acc = acc.ok_or_else(|| Error::Config("Xing40: hc_mult must be >= 1".into()))?;
    Ok(wrap_like(streams, acc, plane.clone()))
}

/// The ROCm ordinal backing a tensor, or a clear error naming the backend.
fn rocm_ordinal(t: &Tensor) -> Result<usize> {
    match t.device() {
        Device::Rocm(o) => Ok(*o),
        other => Err(Error::Backend(format!(
            "Xing40: device-resident hyper-connections require a ROCm device, got {other:?}"
        ))),
    }
}

/// Wrap a backend op result back into a `Tensor`, inheriting dtype/provenance/device.
fn wrap_like(reference: &Tensor, storage: Box<dyn BackendStorage>, shape: Shape) -> Tensor {
    Tensor::new(
        Arc::from(storage),
        shape,
        reference.dtype(),
        reference.provenance().clone(),
        reference.device().clone(),
    )
}

/// Device-resident MHC gates.
///
/// `pre` / `post` are `[hc, seq]` and `comb` is `[hc*hc, seq]` — all
/// stream-major with the token index last, so each per-stream weight vector is a
/// contiguous row slice and the write-back needs no gather.
pub struct Xing40HcGatesD2D {
    /// `[hc, seq]` collapse gate, stream-major.
    pub pre: Box<dyn BackendStorage>,
    /// `[hc, seq]` stream-write gate, stream-major.
    pub post: Box<dyn BackendStorage>,
    /// `[hc * hc, seq]` Sinkhorn combiner, stream-major.
    pub comb: Box<dyn BackendStorage>,
    hc: usize,
    seq: usize,
}

impl Xing40HcGatesD2D {
    /// One `[1, seq]` weight row out of a `[rows, seq]` gate tensor.
    fn weight_row(
        dev: &Arc<dyn grim_tensor::BackendDevice>,
        which: &dyn BackendStorage,
        idx: usize,
        seq: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        Ok(dev
            .narrow_rows(which, idx, 1, seq, &Shape::new(vec![1, seq]))?
            .0)
    }

    /// `pre[h, :]` — the collapse weight of stream `h`.
    fn pre_row(
        &self,
        dev: &Arc<dyn grim_tensor::BackendDevice>,
        h: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        Self::weight_row(dev, self.pre.as_ref(), h, self.seq)
    }

    /// `post[h, :]` — the write weight of stream `h`.
    fn post_row(
        &self,
        dev: &Arc<dyn grim_tensor::BackendDevice>,
        h: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        Self::weight_row(dev, self.post.as_ref(), h, self.seq)
    }

    /// `comb[h, i, :]` — how much of source stream `i` lands in stream `h`.
    fn comb_row(
        &self,
        dev: &Arc<dyn grim_tensor::BackendDevice>,
        h: usize,
        i: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        // The kernel emits comb_out[k*seq + t] with k in the projection's flat
        // comb order — the same k the host path indexes as comb[t*hc*hc +
        // dst*hc + src] (see `write_back`, which the reference-dump bisect
        // pinned at 4.5e-3). So the weight (dst=h <- src=i) lives at row
        // h*hc + i. The previous i*hc + h reading here was the transposition:
        // a Sinkhorn output is doubly stochastic, so its transpose normalizes
        // equally well and row sums CANNOT distinguish the two — the direct
        // write-back comparison with identical inputs did (1.46e-1 vs 8.5e-7
        // for the exact post gate).
        Self::weight_row(dev, self.comb.as_ref(), h * self.hc + i, self.seq)
    }
}

#[derive(Debug, Clone)]
pub struct Xing40HcGates {
    /// Stream-collapse weights `[seq, hc_mult]`: `sigmoid(w * scale + b)`.
    pub pre: Vec<f32>,
    /// Stream-write weights `[seq, hc_mult]`: `2 * sigmoid(w * scale + b)`.
    pub post: Vec<f32>,
    /// Doubly stochastic stream combiner `[seq, hc_mult, hc_mult]`.
    pub comb: Vec<f32>,
}

/// Lay out the GGUF split MLA up-projection banks.
///
/// The banks are stored per head:
/// * `k_b` `[num_heads, rank, nope]` — the key up-projection, with its last two
///   dims transposed relative to the absorbed path's `[nope, rank]` view.
/// * `v_b` `[num_heads, v_head, rank]` — already in `[v_head, rank]` order.
///
/// Returns three views of the same numbers:
/// * `w_kc` `[num_heads, nope, rank]` and `w_vc` `[num_heads, v_head, rank]` —
///   the host absorbed-MLA path's inputs.
/// * `kv_b` `[num_heads * (nope + v_head), rank]` — the single matrix the
///   latent-absorbed decode kernel indexes directly: head stride
///   `(nope + v_head) * rank`, value rows at offset `nope * rank`. This is the
///   same `[out, in]` layout the safetensors container ships as a single
///   `kv_b_proj.weight`, so both containers reach the kernel identically.
pub(crate) fn assemble_split_mla_banks(
    k_b: &[f32],
    v_b: &[f32],
    nh: usize,
    rank: usize,
    nope: usize,
    vd: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kv_b_head = nope + vd;
    assert_eq!(k_b.len(), nh * rank * nope, "attn_k_b element count");
    assert_eq!(v_b.len(), nh * vd * rank, "attn_v_b element count");

    // attn_k_b: [nh, rank, nope] -> w_kc[h] row-major [nope, rank]
    let mut w_kc = vec![0.0f32; nh * nope * rank];
    for h in 0..nh {
        for d in 0..nope {
            for r in 0..rank {
                w_kc[(h * nope + d) * rank + r] = k_b[(h * rank + r) * nope + d];
            }
        }
    }

    // attn_v_b: [nh, vd, rank] -> w_vc[h] row-major [vd, rank]
    let w_vc = v_b[..nh * vd * rank].to_vec();

    // Reassemble [nh * (nope + vd), rank]: per head, `nope` key rows then `vd`
    // value rows, each already [dim, rank].
    let mut kv_b = vec![0.0f32; nh * kv_b_head * rank];
    for h in 0..nh {
        let k_dst = h * kv_b_head * rank;
        kv_b[k_dst..k_dst + nope * rank]
            .copy_from_slice(&w_kc[h * nope * rank..(h + 1) * nope * rank]);
        let v_dst = k_dst + nope * rank;
        kv_b[v_dst..v_dst + vd * rank].copy_from_slice(&w_vc[h * vd * rank..(h + 1) * vd * rank]);
    }

    (w_kc, w_vc, kv_b)
}

/// Manifold hyper-connection module (`attn_hc` / `ffn_hc`).
///
/// Weights (all excluded from quantization in the checkpoint):
/// * `hc_fn`   `[(2 + hc) * hc, hidden * hc]` — pre/post/comb projection
/// * `hc_base` `[(2 + hc) * hc]` — per-output bias
/// * `hc_scale` `[3]` — pre / post / comb logit scales
pub struct Xing40HyperConnection {
    /// Pub for the stage-isolation oracle (`mhc_device_stage_probe_binary`):
    /// it must run norm and projection separately to split a device divergence.
    pub hc_fn: Linear,
    hc_base: Vec<f32>,
    hc_scale: [f32; 3],
    /// Unweighted RMSNorm over the flattened stream vector (`hidden * hc_mult`).
    /// Pub for the same oracle.
    pub input_norm: RmsNorm,
    hc_mult: usize,
    hidden_size: usize,
    sinkhorn_iters: usize,
    eps: f32,
    clamp_min: f32,
    clamp_max: f32,
    /// Device copies of `hc_base` / `hc_scale`, keyed by ordinal (built once).
    dev_base: OnceLock<Option<(usize, Box<dyn BackendStorage>)>>,
    dev_scale: OnceLock<Option<(usize, Box<dyn BackendStorage>)>>,
}

impl Xing40HyperConnection {
    /// Load `hc_fn` / `hc_base` / `hc_scale` from either container.
    ///
    /// * GGUF: flat `hc_{attn,ffn}_{fn,base,scale}.weight` — a single 2D
    ///   `[mix, hidden*hc]` projection plus 1D bias/scale vectors.
    /// * safetensors: nested `attn_hc.hc_fn` / `.hc_base` / `.hc_scale` with
    ///   **no** `.weight` suffix (the released export lists them in
    ///   `modules_to_not_convert`).
    pub fn load(ws: &WeightSource<'_>, cfg: &Xing40Config, gguf_tag: &str) -> Result<Self> {
        let device = ws.device();
        let hc = cfg.hc_mult;
        let hidden = cfg.hidden_size;
        let flat = hidden * hc;
        let mix = (2 + hc) * hc;

        // `hc_*` tensors are never quantized.
        let (hc_fn, hc_base, scale_v) = if ws.has_tensor(&format!("hc_{gguf_tag}_fn.weight")) {
            (
                Linear::load_shape(&ws.scoped(&format!("hc_{gguf_tag}_fn")), [flat, mix])?,
                ws.get([mix], &format!("hc_{gguf_tag}_base.weight"))?
                    .to_vec_f32()?,
                ws.get([3], &format!("hc_{gguf_tag}_scale.weight"))?
                    .to_vec_f32()?,
            )
        } else {
            // safetensors: nested `<tag>_hc.hc_fn` with no `.weight` suffix.
            let src = ws.scoped(&format!("{gguf_tag}_hc"));
            (
                Linear::from_tensor(src.get([mix, flat], "hc_fn")?, None),
                src.get([mix], "hc_base")?.to_vec_f32()?,
                src.get([3], "hc_scale")?.to_vec_f32()?,
            )
        };
        let hc_scale = [
            *scale_v.first().unwrap_or(&1.0),
            *scale_v.get(1).unwrap_or(&1.0),
            *scale_v.get(2).unwrap_or(&1.0),
        ];

        // The MHC input norm is *unweighted* RMSNorm — an all-ones vector of the
        // flattened stream width is exactly that, and keeps the reduction on-device.
        // It must be resident wherever the model lives: the norm kernel dereferences
        // the weight on the device.
        let input_norm = RmsNorm::new(
            const_tensor(vec![1.0f32; flat], Shape::new(vec![flat]), &device)?,
            cfg.rms_norm_eps,
        );

        Ok(Self {
            hc_fn,
            hc_base,
            hc_scale,
            input_norm,
            hc_mult: hc,
            hidden_size: hidden,
            sinkhorn_iters: cfg.hc_sinkhorn_iters,
            eps: cfg.hc_eps,
            clamp_min: cfg.mhc_h_res_clamp_min,
            clamp_max: cfg.mhc_h_res_clamp_max,
            dev_base: OnceLock::new(),
            dev_scale: OnceLock::new(),
        })
    }

    /// Synthesize a near-identity module (unit `hc_base`/`hc_scale`, zero
    /// `hc_fn`) — used by tests and the `random` constructor.
    pub fn identity(cfg: &Xing40Config, device: &Device) -> Self {
        let hc = cfg.hc_mult;
        let flat = cfg.hidden_size * hc;
        let mix = (2 + hc) * hc;
        let hc_fn = Linear::from_tensor(
            cpu_tensor(vec![0.0f32; mix * flat], Shape::new(vec![mix, flat])),
            None,
        );
        let input_norm = RmsNorm::new(
            cpu_tensor(vec![1.0f32; flat], Shape::new(vec![flat])),
            cfg.rms_norm_eps,
        );
        let _ = device;
        Self {
            hc_fn,
            hc_base: vec![0.0; mix],
            hc_scale: [1.0, 1.0, 1.0],
            input_norm,
            hc_mult: hc,
            hidden_size: cfg.hidden_size,
            sinkhorn_iters: cfg.hc_sinkhorn_iters,
            eps: cfg.hc_eps,
            clamp_min: cfg.mhc_h_res_clamp_min,
            clamp_max: cfg.mhc_h_res_clamp_max,
            dev_base: OnceLock::new(),
            dev_scale: OnceLock::new(),
        }
    }

    /// Unweighted RMSNorm + the gate projection, kept on-device (the big matmul),
    /// returning the `[seq, (2+hc)*hc]` projection for the tiny host-side gate math.
    /// Pub for the stage-isolation oracle (`mhc_device_stage_probe_binary`).
    pub fn project(&self, streams: &Tensor) -> Result<Vec<f32>> {
        let normed = self.input_norm.forward(streams)?;
        Ok(self.hc_fn.forward(&normed)?.to_vec_f32()?)
    }

    /// Gate bias and logit scales, for the oracle that checks the LOADED
    /// wiring against the reference's own dumped `hc_attn_base` / `hc_scale`.
    pub fn gate_params(&self) -> (&[f32], &[f32; 3]) {
        (&self.hc_base, &self.hc_scale)
    }

    /// The loaded `hc_fn` weight, for the oracle that pins WHICH matrix
    /// (attn or ffn) actually landed in this module.
    pub fn hc_fn_weight(&self) -> &Tensor {
        &self.hc_fn.weight
    }

    /// Reference gate math: sigmoid gates + clamped Sinkhorn combiner.
    /// Operates on the `[seq, mix]` projection produced by [`Self::project`].
    pub fn gates_from_projection(&self, proj: &[f32], seq_len: usize) -> Xing40HcGates {
        let hc = self.hc_mult;
        let mix = (2 + hc) * hc;
        let mut pre = vec![0.0f32; seq_len * hc];
        let mut post = vec![0.0f32; seq_len * hc];
        let mut comb = vec![0.0f32; seq_len * hc * hc];

        let (pre_scale, post_scale, comb_scale) =
            (self.hc_scale[0], self.hc_scale[1], self.hc_scale[2]);

        for s in 0..seq_len {
            let row = &proj[s * mix..(s + 1) * mix];
            // pre = sigmoid(w * scale + b); post = 2 * sigmoid(w * scale + b)
            for h in 0..hc {
                let pre_w = row[h];
                let pre_b = self.hc_base[h];
                pre[s * hc + h] = sigmoid(pre_w * pre_scale + pre_b);
                let post_w = row[hc + h];
                let post_b = self.hc_base[hc + h];
                post[s * hc + h] = 2.0 * sigmoid(post_w * post_scale + post_b);
            }
            // comb logits: [hc, hc], clamped, then Sinkhorn-normalized to a
            // doubly stochastic matrix (rows and columns sum to 1).
            let comb_off = 2 * hc;
            for o in 0..hc {
                for i in 0..hc {
                    let w = row[comb_off + o * hc + i];
                    let b = self.hc_base[comb_off + o * hc + i];
                    let v = (w * comb_scale + b).clamp(self.clamp_min, self.clamp_max);
                    comb[(s * hc * hc) + o * hc + i] = v;
                }
            }
            // Softmax over the flattened row, then alternate row/col normalization.
            let comb_off_idx = s * hc * hc;
            let mut max_v = f32::NEG_INFINITY;
            for k in 0..hc * hc {
                max_v = max_v.max(comb[comb_off_idx + k]);
            }
            for k in 0..hc * hc {
                comb[comb_off_idx + k] = (comb[comb_off_idx + k] - max_v).exp();
            }
            for _ in 0..self.sinkhorn_iters {
                // Row normalize.
                for o in 0..hc {
                    let mut sum = 0.0f32;
                    for i in 0..hc {
                        sum += comb[comb_off_idx + o * hc + i];
                    }
                    let denom = sum + self.eps;
                    for i in 0..hc {
                        comb[comb_off_idx + o * hc + i] /= denom;
                    }
                }
                // Column normalize.
                for i in 0..hc {
                    let mut sum = 0.0f32;
                    for o in 0..hc {
                        sum += comb[comb_off_idx + o * hc + i];
                    }
                    let denom = sum + self.eps;
                    for o in 0..hc {
                        comb[comb_off_idx + o * hc + i] /= denom;
                    }
                }
            }
        }

        Xing40HcGates { pre, post, comb }
    }

    /// Collapse the `hc_mult` streams into the single block input:
    /// `collapsed[s, d] = Σ_h pre[s, h] * streams[s, h, d]`.
    pub fn collapse(&self, streams_v: &[f32], seq_len: usize, gates: &Xing40HcGates) -> Vec<f32> {
        let hc = self.hc_mult;
        let hidden = self.hidden_size;
        let mut collapsed = vec![0.0f32; seq_len * hidden];
        for s in 0..seq_len {
            for h in 0..hc {
                let p = gates.pre[s * hc + h];
                let src = (s * hc + h) * hidden;
                let dst = s * hidden;
                for d in 0..hidden {
                    collapsed[dst + d] += p * streams_v[src + d];
                }
            }
        }
        collapsed
    }

    /// Device-resident `hc_base` / `hc_scale`, materialized once per ordinal.
    ///
    /// These are 24 + 3 floats of static checkpoint constants; caching them
    /// keeps the per-layer gate math free of any host involvement.
    fn dev_param(&self, ordinal: usize, base: bool) -> Result<&dyn BackendStorage> {
        let cell = if base {
            &self.dev_base
        } else {
            &self.dev_scale
        };
        let slot = cell.get_or_init(|| {
            let rocm = grim_backend_rocm::RocmDevice::shared(ordinal);
            let mix = (2 + self.hc_mult) * self.hc_mult;
            let (data, len): (&[f32], usize) = if base {
                (self.hc_base.as_slice(), mix)
            } else {
                (self.hc_scale.as_slice(), 3)
            };
            rocm.from_cpu(data, &Shape::new(vec![len]), DType::F32)
                .ok()
                .map(|b| (ordinal, b))
        });
        match slot {
            Some((ord, b)) if *ord == ordinal => Ok(b.as_ref()),
            _ => Err(Error::Backend(
                "Xing40: could not materialize MHC gate parameters on device".into(),
            )),
        }
    }

    /// Device-resident MHC gates for the token-major `[seq, hc * hidden]` stream
    /// state. Nothing here touches the host: the unweighted RMSNorm over the
    /// flattened stream vector, the `hc_fn` projection, and the fused
    /// sigmoid + Sinkhorn gate math (`grim_mhc_gates`) all stay on device.
    pub fn gates_d2d(&self, streams: &Tensor) -> Result<Xing40HcGatesD2D> {
        self.gates_d2d_with_proj(streams).map(|(g, _)| g)
    }

    /// Oracle entry point: [`Self::gates_d2d`] plus the raw `hc_fn` projection
    /// that produced the gates. A parity gate has to compare the LOGITS the
    /// kernel consumed against the reference's own `hc_mixes`, and this is the
    /// alternative to making the module's fields public.
    pub fn gates_d2d_with_proj(&self, streams: &Tensor) -> Result<(Xing40HcGatesD2D, Vec<f32>)> {
        let ordinal = rocm_ordinal(streams)?;
        let rocm = grim_backend_rocm::RocmDevice::shared(ordinal);
        let hc = self.hc_mult;
        let seq = streams.shape().dims()[0];

        // The projection consumes the flattened per-token stream vector, so the
        // state must be token-major `[seq, hc * hidden]` here.
        let normed = self.input_norm.forward(streams)?;
        let proj = self.hc_fn.forward(&normed)?;
        let gates = rocm.mhc_gates_into(
            proj.storage().as_ref(),
            self.dev_param(ordinal, true)?,
            self.dev_param(ordinal, false)?,
            seq,
            hc,
            self.sinkhorn_iters,
            self.eps,
            self.clamp_min,
            self.clamp_max,
        )?;
        let proj_host = proj.to_vec_f32()?;
        Ok((Xing40HcGatesD2D {
            pre: gates.pre,
            post: gates.post,
            comb: gates.comb,
            hc,
            seq,
        }, proj_host))
    }

    /// Collapse the `hc_mult` streams into the single block input:
    /// `collapsed[s, d] = Σ_h pre[h, s] * streams[s, h * hidden + d]`, `[seq, hidden]`.
    pub fn collapse_d2d(&self, streams: &Tensor, g: &Xing40HcGatesD2D) -> Result<Tensor> {
        let dev = pick_device_for_tensor(streams);
        let hidden = self.hidden_size;
        let flat = self.hc_mult * hidden;
        let plane = Shape::new(vec![g.seq, hidden]);
        let mut acc: Option<Box<dyn BackendStorage>> = None;
        for h in 0..self.hc_mult {
            let (src, _) = dev.narrow_cols(
                streams.storage().as_ref(),
                flat,
                h * hidden,
                g.seq,
                hidden,
                &plane,
            )?;
            let w = g.pre_row(&dev, h)?;
            let (scaled, _) = dev.row_scale(src.as_ref(), w.as_ref(), g.seq, hidden, &plane)?;
            acc = Some(match acc {
                None => scaled,
                Some(a) => dev.add(a.as_ref(), scaled.as_ref(), &plane)?.0,
            });
        }
        let acc = acc.ok_or_else(|| Error::Config("Xing40: hc_mult must be >= 1".into()))?;
        Ok(wrap_like(streams, acc, plane))
    }

    /// Write the block output back into the multi-stream state:
    /// `out[s, h*hidden + d] = post[h, s] * y[s, d] + Σ_i comb[h, i, s] * streams[s, i*hidden + d]`.
    ///
    /// The destination is allocated once and each stream's column block is
    /// written in place, so the whole update is `hc * (hc + 2)` device ops with
    /// no host round-trip.
    pub fn update_d2d(&self, streams: &Tensor, y: &Tensor, g: &Xing40HcGatesD2D) -> Result<Tensor> {
        let dev = pick_device_for_tensor(streams);
        let hc = self.hc_mult;
        let hidden = self.hidden_size;
        let flat = hc * hidden;
        let plane = Shape::new(vec![g.seq, hidden]);
        let full = Shape::new(vec![g.seq, flat]);
        let mut dst = dev.zeros(&full, DType::F32)?;
        for h in 0..hc {
            let mut mixed: Option<Box<dyn BackendStorage>> = None;
            for i in 0..hc {
                let (src, _) = dev.narrow_cols(
                    streams.storage().as_ref(),
                    flat,
                    i * hidden,
                    g.seq,
                    hidden,
                    &plane,
                )?;
                let w = g.comb_row(&dev, h, i)?;
                let (scaled, _) = dev.row_scale(src.as_ref(), w.as_ref(), g.seq, hidden, &plane)?;
                mixed = Some(match mixed {
                    None => scaled,
                    Some(m) => dev.add(m.as_ref(), scaled.as_ref(), &plane)?.0,
                });
            }
            let mixed =
                mixed.ok_or_else(|| Error::Config("Xing40: hc_mult must be >= 1".into()))?;
            let pw = g.post_row(&dev, h)?;
            let (gated, _) =
                dev.row_scale(y.storage().as_ref(), pw.as_ref(), g.seq, hidden, &plane)?;
            let (out_h, _) = dev.add(gated.as_ref(), mixed.as_ref(), &plane)?;
            dev.write_cols(
                dst.as_mut(),
                flat,
                h * hidden,
                out_h.as_ref(),
                g.seq,
                hidden,
            )?;
        }
        Ok(wrap_like(streams, dst, full))
    }

    /// Full forward: project the (possibly device-resident) stream state and
    ///
    /// The stream math is host-side (the established `softplus_mul_on_device`
    /// pattern): the tensors involved are `[seq, hc*hidden]`, while the projection
    /// itself stays on-device.
    pub fn forward(&self, streams: &Tensor, seq_len: usize) -> Result<(Xing40HcGates, Vec<f32>)> {
        let proj = self.project(streams)?;
        let gates = self.gates_from_projection(&proj, seq_len);
        let streams_v = streams.to_vec_f32()?;
        let collapsed = self.collapse(&streams_v, seq_len, &gates);
        Ok((gates, collapsed))
    }

    /// Write a block result back into the stream state:
    /// `new[s, o, d] = post[s, o] * y[s, d] + Σ_i comb[s, o, i] * streams[s, i, d]`.
    pub fn write_back(
        &self,
        streams_v: &[f32],
        y_v: &[f32],
        seq_len: usize,
        gates: &Xing40HcGates,
    ) -> Vec<f32> {
        let hc = self.hc_mult;
        let hidden = self.hidden_size;
        let mut out = vec![0.0f32; seq_len * hc * hidden];
        for s in 0..seq_len {
            for o in 0..hc {
                let post = gates.post[s * hc + o];
                // post ⊗ y — the block output written into every stream.
                let y_row = &y_v[s * hidden..(s + 1) * hidden];
                let out_row = &mut out[(s * hc + o) * hidden..(s * hc + o + 1) * hidden];
                for d in 0..hidden {
                    out_row[d] = post * y_row[d];
                }
                // comb @ streams — mix the incoming streams into stream `o`.
                for i in 0..hc {
                    let c = gates.comb[(s * hc * hc) + o * hc + i];
                    if c == 0.0 {
                        continue;
                    }
                    let src = (s * hc + i) * hidden;
                    for d in 0..hidden {
                        out_row[d] += c * streams_v[src + d];
                    }
                }
            }
        }
        out
    }
}

// MLA Attention Block

pub struct Xing40Mla {
    pub q_a_proj: Option<Linear>,
    pub q_a_layernorm: Option<RmsNorm>,
    pub q_b_proj: Option<Linear>,
    pub q_proj_direct: Option<Linear>,
    pub kv_a_proj: Linear,
    pub kv_a_layernorm: RmsNorm,
    pub kv_b_proj: Linear,
    pub o_proj: Linear,
    pub rope: Rope,
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Absorbed per-head key up-projection `w_kc[h]`, row-major
    /// `[num_heads, qk_nope_head_dim, kv_lora_rank]`.
    pub w_kc: Vec<f32>,
    /// Per-head value up-projection `w_vc[h]`, row-major
    /// `[num_heads, v_head_dim, kv_lora_rank]`.
    pub w_vc: Vec<f32>,
    /// Device-resident `[num_heads, qk_nope_head_dim, kv_lora_rank]` copy of
    /// `w_kc`, for the D2D per-head query absorb. Built once per ordinal.
    w_kc_dev: OnceLock<Option<(usize, Arc<dyn BackendStorage>)>>,
    /// Device-resident `[num_heads, kv_lora_rank, v_head_dim]` copy of `w_vc`
    /// (already transposed for `matmul`), for the D2D value up-projection.
    w_vc_t_dev: OnceLock<Option<(usize, Arc<dyn BackendStorage>)>>,
}

impl Xing40Mla {
    /// Load the MLA projections from either container.
    ///
    /// * safetensors: nested `self_attn.{q_a_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj}`
    ///   with `kv_b_proj` a flat `[num_heads*(nope+v), kv_lora_rank]` matrix.
    /// * GGUF: flat `attn_{q_a, q_b, kv_a_mqa, k_b, v_b, output}.weight`, with
    ///   `attn_k_b` / `attn_v_b` already split per head as
    ///   `[num_heads, kv_lora_rank, nope]` / `[num_heads, v_head_dim, kv_lora_rank]`.
    pub fn load(ws: &WeightSource<'_>, cfg: &Xing40Config) -> Result<Self> {
        let q_dim = cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
        let kv_lora_out = cfg.kv_lora_rank + cfg.qk_rope_head_dim;

        // RopeConfig defaults to GPT-J INTERLEAVED pairing; Xing4.0's MLA is a
        // DeepSeek-V3-style family, which the reference rotates NeoX half-split
        // (same defect c1352622 fixed for Qwen3.5). YaRN from the checkpoint
        // (`rope.scaling.type = yarn`) rides along.
        let rope = {
            // INTERLEAVED (ggml ROPE_TYPE_NORM = 0 — the reference logs exactly
            // this). An earlier `rc.interleaved = false` here claimed
            // DeepSeek-style NeoX half-split and was refuted by measurement:
            // the reference's own k_pe tensors agree with interleaved to ~1e-3
            // and disagree with NeoX by 29%.
            let mut rc = grim_tensor::RopeConfig::new(cfg.qk_rope_head_dim, cfg.rope_theta);
            rc.interleaved = true;
            rc.yarn = cfg.rope_yarn;
            Rope::from_config(rc)
        };
        let out = if ws.has_tensor("attn_k_b.weight") {
            // ---- GGUF container ----
            let q_a_proj = Linear::load_shape(
                &ws.scoped("attn_q_a"),
                [cfg.hidden_size, cfg.q_lora_rank.unwrap_or(0)],
            )
            .ok();
            let q_a_layernorm = cfg
                .q_lora_rank
                .and_then(|r| RmsNorm::load(&ws.scoped("attn_q_a_norm"), r, cfg.rms_norm_eps).ok());
            let q_b_proj = Linear::load_shape(
                &ws.scoped("attn_q_b"),
                [cfg.q_lora_rank.unwrap_or(0), q_dim],
            )
            .ok();
            let kv_a_proj =
                Linear::load_shape(&ws.scoped("attn_kv_a_mqa"), [cfg.hidden_size, kv_lora_out])?;
            let kv_a_layernorm = RmsNorm::load(
                &ws.scoped("attn_kv_a_norm"),
                cfg.kv_lora_rank,
                cfg.rms_norm_eps,
            )?;
            let o_proj = Linear::load_shape(
                &ws.scoped("attn_output"),
                [cfg.num_heads * cfg.v_head_dim, cfg.hidden_size],
            )?;

            // Per-head key/value up-projections come pre-split from GGUF:
            //   attn_k_b [num_heads, kv_lora_rank, nope]
            //   attn_v_b [num_heads, v_head_dim, kv_lora_rank]
            let (w_kc, w_vc, kv_b) = Xing40Mla::load_split_up_projs(ws, cfg)?;
            let kv_b_proj = Xing40Mla::split_kv_b_linear(kv_b, cfg, &ws.device())?;
            Self {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
                q_proj_direct: None,
                kv_a_proj,
                kv_a_layernorm,
                kv_b_proj,
                o_proj,
                rope,
                num_heads: cfg.num_heads,
                qk_nope_head_dim: cfg.qk_nope_head_dim,
                qk_rope_head_dim: cfg.qk_rope_head_dim,
                v_head_dim: cfg.v_head_dim,
                w_kc,
                w_vc,
                w_kc_dev: OnceLock::new(),
                w_vc_t_dev: OnceLock::new(),
            }
        } else {
            // ---- safetensors container ----
            // `ws` arrives already scoped to the attention module by the caller.
            let (q_a_proj, q_a_layernorm, q_b_proj, q_proj_direct) = if let Some(q_rank) =
                cfg.q_lora_rank
            {
                let qa_r = Linear::load_shape(&ws.scoped("q_a_proj"), [cfg.hidden_size, q_rank]);
                let qn_r = RmsNorm::load(&ws.scoped("q_a_layernorm"), q_rank, cfg.rms_norm_eps);
                let qb_r = Linear::load_shape(&ws.scoped("q_b_proj"), [q_rank, q_dim]);
                let (qa, qn, qb) = (qa_r.ok(), qn_r.ok(), qb_r.ok());
                if qa.is_some() && qn.is_some() && qb.is_some() {
                    (qa, qn, qb, None)
                } else {
                    (
                        None,
                        None,
                        None,
                        Some(Linear::load_shape(
                            &ws.scoped("q_proj"),
                            [cfg.hidden_size, q_dim],
                        )?),
                    )
                }
            } else {
                (
                    None,
                    None,
                    None,
                    Some(Linear::load_shape(
                        &ws.scoped("q_proj"),
                        [cfg.hidden_size, q_dim],
                    )?),
                )
            };

            let kv_a_proj = Linear::load_shape(
                &ws.scoped("kv_a_proj_with_mqa"),
                [cfg.hidden_size, kv_lora_out],
            )?;
            let kv_a_layernorm = RmsNorm::load(
                &ws.scoped("kv_a_layernorm"),
                cfg.kv_lora_rank,
                cfg.rms_norm_eps,
            )?;
            let kv_b_proj = Linear::load_shape(
                &ws.scoped("kv_b_proj"),
                [
                    cfg.kv_lora_rank,
                    cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
                ],
            )?;
            let o_proj = Linear::load_shape(
                &ws.scoped("o_proj"),
                [cfg.num_heads * cfg.v_head_dim, cfg.hidden_size],
            )?;

            let kv_b_w = kv_b_proj.weight.to_vec_f32()?;
            let (w_kc, w_vc) = crate::mla_common::extract_kv_b_up_projs(
                &kv_b_w,
                cfg.num_heads,
                cfg.qk_nope_head_dim,
                cfg.v_head_dim,
                cfg.kv_lora_rank,
            );

            Self {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
                q_proj_direct,
                kv_a_proj,
                kv_a_layernorm,
                kv_b_proj,
                o_proj,
                rope,
                num_heads: cfg.num_heads,
                qk_nope_head_dim: cfg.qk_nope_head_dim,
                qk_rope_head_dim: cfg.qk_rope_head_dim,
                v_head_dim: cfg.v_head_dim,
                w_kc,
                w_vc,
                w_kc_dev: OnceLock::new(),
                w_vc_t_dev: OnceLock::new(),
            }
        };
        // NOTE: deliberately NOT rebuilt here. An earlier version re-assigned
        // `out.rope = Rope::new(...)` "to be sure", which threw away the YaRN
        // config built above and replaced it with a fresh default (yarn: None).
        // Every consumer of `self.rope.config` — the host rope, the D2D rope,
        // AND the kq_scale mscale — then ran plain RoPE with mscale 1.0 no
        // matter what the loader parsed, while host and device agreed with each
        // other (both got None) and every internal parity gate stayed green.
        // The reference comparison caught it: attention diverged 60x above the
        // arithmetic noise floor at blk.0 and compounded to ~100% by layer 9.
        // The checkpoint's own `rope` (base, theta, YaRN) is the one built above.
        Ok(out)
    }

    /// Read the GGUF per-head `attn_k_b` / `attn_v_b` banks and lay them out three ways.
    ///
    /// See [`assemble_split_mla_banks`] for the layout; this only does the I/O.
    fn load_split_up_projs(
        ws: &WeightSource<'_>,
        cfg: &Xing40Config,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let nh = cfg.num_heads;
        let rank = cfg.kv_lora_rank;
        let nope = cfg.qk_nope_head_dim;
        let vd = cfg.v_head_dim;
        let k_b = ws.get([nh, rank, nope], "attn_k_b.weight")?.to_vec_f32()?;
        let v_b = ws.get([nh, vd, rank], "attn_v_b.weight")?.to_vec_f32()?;
        Ok(assemble_split_mla_banks(&k_b, &v_b, nh, rank, nope, vd))
    }

    /// Wrap the reassembled `[num_heads * (nope + v_head), rank]` matrix as the
    /// `kv_b_proj` the latent-absorbed decode kernel expects.
    ///
    /// The kernel reads `kv_b_proj.weight`'s raw storage, so this must be the
    /// `[out, in]` layout — the same layout the safetensors container delivers
    /// natively as a single `kv_b_proj.weight` tensor. `w_t` is left aliasing
    /// `weight` rather than being transposed: nothing calls `Linear::forward` on
    /// it (the host path uses `w_kc`/`w_vc`), and a transpose here would
    /// allocate and fill a second `[rank, nh * (nope + vd)]` buffer for nothing.
    fn split_kv_b_linear(kv_b: Vec<f32>, cfg: &Xing40Config, device: &Device) -> Result<Linear> {
        let shape = Shape::new(vec![
            cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            cfg.kv_lora_rank,
        ]);
        let weight = const_tensor(kv_b, shape, device)?;
        Ok(Linear {
            w_t: weight.clone(),
            weight,
            bias: None,
            quant_format: None,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {

        let seq_len = x.shape().dims()[0];

        // On a device the whole MLA forward is device-resident: the host path below
        // builds CPU activations and then calls CPU-device ops against the
        // device-resident weights, so it cannot run here at all. `forward_d2d_mla`
        // picks the fused decode kernel for `seq_len == 1` and the query-blocked
        // prefill kernel otherwise.
        if x.device() != &Device::Cpu {
            if let Some(out) = self.forward_d2d_mla(x, positions, kv_cache)? {
                return Ok(out);
            }
        }

        // 1. Q projection (LoRA-compressed).
        let q_full = if let (Some(qa), Some(qn), Some(qb)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_lat = qa.forward(x)?;
            let q_lat_normed = qn.forward(&q_lat)?;
            qb.forward(&q_lat_normed)?
        } else if let Some(ref q_direct) = self.q_proj_direct {
            q_direct.forward(x)?
        } else {
            return Err(Error::Config("Xing40: no valid Q projection".into()));
        };
        let q_full_v = q_full.to_vec_f32()?;

        let (q_nope_v, mut q_rope_v) = crate::mla_common::split_q_nope_rope(
            &q_full_v,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
        );

        // The checkpoint's OWN rope — base and YaRN — not a hardcoded theta.
        // Plain RoPE here while the D2D path rotated with YaRN is what made the
        // host reference disagree with llama.cpp from blk.0 onward.
        crate::qwen35::apply_rope_yarn_interleaved(
            &mut q_rope_v,
            positions,
            self.num_heads,
            self.qk_rope_head_dim,
            self.rope.config.base,
            self.rope.config.yarn.as_ref(),
        );

        // 2. KV latent projection.
        let kv_latent = self.kv_a_proj.forward(x)?;
        let kv_latent_v = kv_latent.to_vec_f32()?;
        let kv_rank = self.kv_a_layernorm.weight.shape().dims()[0];

        let (kv_a_v, mut k_rope_v) = crate::mla_common::split_kv_latent(
            &kv_latent_v,
            seq_len,
            kv_rank,
            self.qk_rope_head_dim,
        );

        let kv_a_t = cpu_tensor(kv_a_v, Shape::new(vec![seq_len, kv_rank]));
        let kv_a_normed = self.kv_a_layernorm.forward(&kv_a_t)?;

        crate::qwen35::apply_rope_yarn_interleaved(
            &mut k_rope_v,
            positions,
            1,
            self.qk_rope_head_dim,
            self.rope.config.base,
            self.rope.config.yarn.as_ref(),
        );

        // 3. Absorb w_kc into the query so attention runs in latent space.
        let kv_a_normed_v = kv_a_normed.to_vec_f32()?;
        let rank = kv_rank;
        let nope = self.qk_nope_head_dim;
        let rope_d = self.qk_rope_head_dim;
        let vd = self.v_head_dim;
        let nh = self.num_heads;

        if std::env::var_os("GRIM_XING_TRACE").is_some() {
            let qn = cpu_tensor(
                q_nope_v.clone(),
                Shape::new(vec![seq_len, nh * nope]),
            );
            xing_trace_stage("host q_nope", &qn);
        }
        let q_absorbed =
            crate::mla_common::absorb_query_wkc(&q_nope_v, &self.w_kc, seq_len, nh, nope, rank);
        if std::env::var_os("GRIM_XING_TRACE").is_some() {
            let qa = cpu_tensor(
                q_absorbed.clone(),
                Shape::new(vec![seq_len, nh * rank]),
            );
            xing_trace_stage("host q_absorbed", &qa);
        }

        // 4. Packed latent rows: [normed c_kv || roped k_pe].
        let latent_new =
            crate::mla_common::pack_latent_rows(&kv_a_normed_v, &k_rope_v, seq_len, rank, rope_d);

        // 5. Append to the device-resident latent KV cache.
        let row = rank + rope_d;
        let cache_dev = grim_nn::modules::pick_device_for_storage_device(x.device());
        let new_latent_st =
            cache_dev.from_cpu(&latent_new, &Shape::new(vec![seq_len, row]), DType::F32)?;
        let new_latent = Tensor::new(
            Arc::from(new_latent_st),
            Shape::new(vec![seq_len, row]),
            DType::F32,
            QuantProvenance::default(),
            x.device().clone(),
        );
        let latent_all = match kv_cache.as_ref() {
            Some((prev_latent, _unused)) => {
                crate::shared_attention::concat_rows_on_device(prev_latent, &new_latent)?
            }
            None => new_latent,
        };
        let total_kv_len = latent_all.shape().dims()[0];
        xing_trace_stage("d2d latent_all", &latent_all);
        *kv_cache = Some((
            latent_all.clone(),
            cpu_tensor(Vec::new(), Shape::new(vec![0, 0])),
        ));

        // llama.cpp: kq_scale = mscale^2 / sqrt(n_embd_head_k), where
        // mscale folds the YaRN attention factor (0.1*ln(factor)+1 = 1.415 for
        // factor 64, so the squared factor is ~2x). Ignoring it scales every
        // attention logit wrong.
        let mscale = self
            .rope
            .config
            .yarn
            .map(|y| y.attention_factor)
            .unwrap_or(1.0f32);
        let scale = mscale * mscale / ((nope + rope_d) as f32).sqrt();

        // 6a. GPU decode fast path (decode-only kernel: one launch per head).
        if seq_len == 1 && x.device() != &Device::Cpu {
            if let Some(attn_t) = self.gpu_absorbed_decode(
                &q_absorbed,
                &q_rope_v,
                &latent_all,
                rank,
                total_kv_len,
                scale,
                x.device(),
            )? {
                return Ok(self.o_proj.forward(&attn_t)?);
            }
        }

        // 6b. Scalar latent-space reference path with causal masking - the documented FALLBACK.
        let latent_all_v = latent_all.to_vec_f32()?;
        let cache_offset = total_kv_len - seq_len;
        let row = rank + rope_d;
        let mut attn_out = vec![0.0f32; seq_len * nh * vd];

        for s in 0..seq_len {
            let causal_limit = cache_offset + s;
            for h in 0..nh {
                let q_abs = &q_absorbed[(s * nh + h) * rank..(s * nh + h + 1) * rank];
                let q_rp = &q_rope_v[(s * nh + h) * rope_d..(s * nh + h + 1) * rope_d];

                let mut scores = vec![0.0f32; causal_limit + 1];
                for (t, score) in scores.iter_mut().enumerate().take(causal_limit + 1) {
                    let lb = t * row;
                    let dot_c: f32 = q_abs
                        .iter()
                        .zip(&latent_all_v[lb..lb + rank])
                        .map(|(a, b)| a * b)
                        .sum();
                    let dot_r: f32 = q_rp
                        .iter()
                        .zip(&latent_all_v[lb + rank..lb + row])
                        .map(|(a, b)| a * b)
                        .sum();
                    *score = (dot_c + dot_r) * scale;
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
                let weights: Vec<f32> = scores
                    .iter()
                    .map(|e| (e - max_score).exp() / (sum_exp + 1e-12))
                    .collect();

                let mut attn_latent = vec![0.0f32; rank];
                for t in 0..=causal_limit {
                    let w = weights[t];
                    for (o, l) in attn_latent
                        .iter_mut()
                        .zip(&latent_all_v[t * row..t * row + rank])
                    {
                        *o += w * l;
                    }
                }

                for d in 0..vd {
                    let wrow = &self.w_vc[(h * vd + d) * rank..(h * vd + d + 1) * rank];
                    attn_out[(s * nh + h) * vd + d] = attn_latent
                        .iter()
                        .zip(wrow.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                }
            }
        }

        let attn_tensor = cpu_tensor(
            attn_out,
            Shape::new(vec![seq_len, self.num_heads * self.v_head_dim]),
        );
        Ok(self.o_proj.forward(&attn_tensor)?)
    }

    /// Device-resident prefill path via `BackendDevice::mla_absorbed_prefill`.
    ///
    /// Everything stays on the GPU: the Q projection and its per-head
    /// `W_UK` absorb, the RoPE on both the query's and the cache's rope slice, the
    /// `kv_a` norm, the packed-latent cache append, the latent-space causal
    /// attention, and the per-head `W_UV` up-projection.
    ///
    /// Returns `Ok(None)` — the documented fallback — when the backend lacks the
    /// prefill kernel, so the caller can use the host reference.
    #[allow(clippy::too_many_arguments)]
    fn forward_d2d_mla(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Option<Tensor>> {
        let ordinal = match x.device() {
            Device::Rocm(o) => *o,
            _ => return Ok(None),
        };
        let nh = self.num_heads;
        let nope = self.qk_nope_head_dim;
        let rope_d = self.qk_rope_head_dim;
        let vd = self.v_head_dim;
        let rank = self.kv_a_layernorm.weight.shape().dims()[0];
        let seq_len = x.shape().dims()[0];
        let dev = pick_device_for_tensor(x);
        let row = rank + rope_d;
        let q_stride = nope + rope_d; // per-head width inside q_full

        // 1. Q projection (LoRA-compressed), device-resident.
        let q_full = if let (Some(qa), Some(qn), Some(qb)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_lat = qa.forward(x)?;
            let q_lat_normed = qn.forward(&q_lat)?;
            qb.forward(&q_lat_normed)?
        } else if let Some(ref q_direct) = self.q_proj_direct {
            q_direct.forward(x)?
        } else {
            return Err(Error::Config("Xing40: no valid Q projection".into()));
        };

        // 2. Split the per-head rope slice out of q_full and rope it in place.
        //    `q_rope` is laid out [seq, nh * rope_d] == [seq, nh, rope_d].
        let q_rope_shape = Shape::new(vec![seq_len, nh * rope_d]);
        let mut q_rope = dev.zeros(&q_rope_shape, DType::F32)?;
        for h in 0..nh {
            let src = dev
                .narrow_cols(
                    q_full.storage().as_ref(),
                    nh * q_stride,
                    h * q_stride + nope,
                    seq_len,
                    rope_d,
                    &Shape::new(vec![seq_len, rope_d]),
                )?
                .0;
            let Some(roped) = or_host_fallback(dev.rope(
                src.as_ref(),
                positions,
                // The checkpoint's OWN rope: base and the YaRN scaling, or the
                // device path rotates different frequencies than the host
                // reference (Xing4.0 ships YaRN factor 64).
                &RopeConfig {
                    dim: rope_d,
                    base: self.rope.config.base,
                    rotary_dim: rope_d,
                    yarn: self.rope.config.yarn,
                    // INTERLEAVED — see the note at rope construction.
                    interleaved: true,
                },
                &Shape::new(vec![1, seq_len, rope_d]),
            ))?
            else {
                return Ok(None);
            };
            dev.write_cols(
                q_rope.as_mut(),
                nh * rope_d,
                h * rope_d,
                roped.0.as_ref(),
                seq_len,
                rope_d,
            )?;
        }
        let q_rope = wrap_like(x, q_rope, q_rope_shape.clone());

        // 3. Absorb W_UK into the query: per head,
        //    q_absorbed[:, h] = q_nope[:, h] @ w_kc[h]  ([nope, rank]).
        let (Some(w_kc_dev), Some(w_vc_dev)) =
            (self.w_kc_device(ordinal)?, self.w_vc_device(ordinal)?)
        else {
            return Ok(None);
        };
        let q_abs_shape = Shape::new(vec![seq_len, nh * rank]);
        let mut q_absorbed = dev.zeros(&q_abs_shape, DType::F32)?;
        for h in 0..nh {
            let q_nope_h = dev
                .narrow_cols(
                    q_full.storage().as_ref(),
                    nh * q_stride,
                    h * q_stride,
                    seq_len,
                    nope,
                    &Shape::new(vec![seq_len, nope]),
                )?
                .0;
            if h == 0 && std::env::var_os("GRIM_XING_TRACE").is_some() {
                let probe = dev
                    .narrow_cols(
                        q_full.storage().as_ref(),
                        nh * q_stride,
                        0,
                        seq_len,
                        nope,
                        &Shape::new(vec![seq_len, nope]),
                    )?
                    .0;
                xing_trace_stage(
                    "d2d q_nope_h0",
                    &wrap_like(x, probe, Shape::new(vec![seq_len, nope])),
                );
            }
            let w_h = dev
                .narrow_rows(
                    w_kc_dev,
                    h * rank,
                    rank,
                    nope,
                    &Shape::new(vec![rank, nope]),
                )?
                .0;
            let absorbed = dev
                .matmul(
                    q_nope_h.as_ref(),
                    w_h.as_ref(),
                    &Shape::new(vec![seq_len, rank]),
                )?
                .0;
            dev.write_cols(
                q_absorbed.as_mut(),
                nh * rank,
                h * rank,
                absorbed.as_ref(),
                seq_len,
                rank,
            )?;
        }
        let q_absorbed = wrap_like(x, q_absorbed, q_abs_shape.clone());
        xing_trace_stage("d2d q_absorbed", &q_absorbed);

        // 4. KV latent projection: norm c_kv, rope k_pe, pack [c_kv || k_pe].
        let kv_latent = self.kv_a_proj.forward(x)?;
        let c_kv = dev
            .narrow_cols(
                kv_latent.storage().as_ref(),
                row,
                0,
                seq_len,
                rank,
                &Shape::new(vec![seq_len, rank]),
            )?
            .0;
        let c_kv =
            self.kv_a_layernorm
                .forward(&wrap_like(x, c_kv, Shape::new(vec![seq_len, rank])))?;
        let k_pe = dev
            .narrow_cols(
                kv_latent.storage().as_ref(),
                row,
                rank,
                seq_len,
                rope_d,
                &Shape::new(vec![seq_len, rope_d]),
            )?
            .0;
        let Some((k_pe, _k_pe_handle)) = or_host_fallback(dev.rope(
            k_pe.as_ref(),
            positions,
            // Checkpoint rope (base + YaRN), matching the host reference.
            // INTERLEAVED — same pairing as the q_pe rotation above and the
            // host path; the reference logs rope type 0 (ggml ROPE_TYPE_NORM).
            // This site alone read `interleaved: false` (NeoX half-split),
            // which is invisible at position 0 — no rotation — and grew with
            // position and input magnitude: the real-prompt chain read
            // dev-vs-host 1.8e-1 at blk.0 while the random-input repro read
            // 1.2e-2, both localized here by the stage bisect.
            &RopeConfig {
                dim: rope_d,
                base: self.rope.config.base,
                rotary_dim: rope_d,
                yarn: self.rope.config.yarn,
                interleaved: true,
            },
            &Shape::new(vec![1, seq_len, rope_d]),
        ))?
        else {
            return Ok(None);
        };

        let latent_new_shape = Shape::new(vec![seq_len, row]);
        let mut latent_new = dev.zeros(&latent_new_shape, DType::F32)?;
        dev.write_cols(
            latent_new.as_mut(),
            row,
            0,
            c_kv.storage().as_ref(),
            seq_len,
            rank,
        )?;
        dev.write_cols(
            latent_new.as_mut(),
            row,
            rank,
            k_pe.as_ref(),
            seq_len,
            rope_d,
        )?;
        let latent_new = wrap_like(x, latent_new, latent_new_shape);

        // 5. Append to the latent KV cache.
        let latent_all = match kv_cache.as_ref() {
            Some((prev_latent, _)) => {
                crate::shared_attention::concat_rows_on_device(prev_latent, &latent_new)?
            }
            None => latent_new,
        };
        let total_kv_len = latent_all.shape().dims()[0];
        xing_trace_stage("host latent_all", &latent_all);
        let cache_offset = total_kv_len - seq_len;
        *kv_cache = Some((
            latent_all.clone(),
            cpu_tensor(Vec::new(), Shape::new(vec![0, 0])),
        ));

        // 6. Latent-space attention.
        //
        //    Decode keeps the fused `mla_absorbed_decode` kernel: it applies W_UV
        //    in-kernel, so the per-head GEMM loop below is skipped entirely for the
        //    single-token case where launch count dominates. That kernel hardcodes
        //    1/sqrt(rank + rope_d) as the softmax scale, so pre-scale the query by
        //    the ratio that reconciles it with the model's 1/sqrt(nope + rope_d) —
        //    the same correction the host decode path applies.
        // Same scale the host path uses: kq_scale = mscale^2 / sqrt(nope +
        // rope_d), with mscale folding the YaRN attention factor (1.415 for
        // Xing4.0's factor 64). Dropping mscale here rescales every logit.
        let mscale = self
            .rope
            .config
            .yarn
            .map(|y| y.attention_factor)
            .unwrap_or(1.0f32);
        let scale = mscale * mscale / ((nope + rope_d) as f32).sqrt();
        if seq_len == 1 && rank <= 512 && !self.kv_b_proj.weight.dtype().is_quantized() {
            let kernel_scale = 1.0f32 / ((rank + rope_d) as f32).sqrt();
            let ratio = scale / kernel_scale;
            let qa_s = dev
                .mul_scalar(q_absorbed.storage().as_ref(), ratio, &q_abs_shape)?
                .0;
            let qr_s = dev
                .mul_scalar(q_rope.storage().as_ref(), ratio, &q_rope_shape)?
                .0;
            let attn_shape = Shape::new(vec![1, nh * vd]);
            let mut attn = dev.zeros(&attn_shape, DType::F32)?;
            let fused = or_host_fallback(dev.mla_absorbed_decode(
                qa_s.as_ref(),
                qr_s.as_ref(),
                latent_all.storage().as_ref(),
                Some(self.kv_b_proj.weight.storage().as_ref()),
                attn.as_mut(),
                nh,
                rank,
                rope_d,
                vd,
                total_kv_len,
                nope * rank,
                (nope + vd) * rank,
            ))?;
            if let Some(_handle) = fused {
                let attn = wrap_like(x, attn, attn_shape);
                return Ok(Some(self.o_proj.forward(&attn)?));
            }
        }

        // 6b. Prefill / wide-rank path: the causal query-blocked prefill kernel
        //     emits the normalized latent, and W_UV is applied as per-head GEMMs.
        let out_latent_shape = Shape::new(vec![seq_len, nh * rank]);
        let mut out_latent = dev.zeros(&out_latent_shape, DType::F32)?;
        or_host_fallback(dev.mla_absorbed_prefill(
            q_absorbed.storage().as_ref(),
            q_rope.storage().as_ref(),
            latent_all.storage().as_ref(),
            out_latent.as_mut(),
            seq_len,
            nh,
            rank,
            rope_d,
            cache_offset,
            total_kv_len,
            scale,
        ))?;

        // 7. Per-head W_UV up-projection: attn[:, h] = latent[:, h] @ w_vc_t[h].
        //    One GEMM per head over the whole query block, rather than a fused
        //    `vd * rank` matmul inside every (query, head) attention block.
        let attn_shape = Shape::new(vec![seq_len, nh * vd]);
        let mut attn = dev.zeros(&attn_shape, DType::F32)?;
        for h in 0..nh {
            let latent_h = dev
                .narrow_cols(
                    out_latent.as_ref(),
                    nh * rank,
                    h * rank,
                    seq_len,
                    rank,
                    &Shape::new(vec![seq_len, rank]),
                )?
                .0;
            let w_h = dev
                .narrow_rows(w_vc_dev, h * vd, vd, rank, &Shape::new(vec![vd, rank]))?
                .0;
            let projected = dev
                .matmul(
                    latent_h.as_ref(),
                    w_h.as_ref(),
                    &Shape::new(vec![seq_len, vd]),
                )?
                .0;
            dev.write_cols(
                attn.as_mut(),
                nh * vd,
                h * vd,
                projected.as_ref(),
                seq_len,
                vd,
            )?;
        }
        let attn = wrap_like(x, attn, attn_shape);

        Ok(Some(self.o_proj.forward(&attn)?))
    }

    /// Device-resident `w_kc` as `[num_heads, nope, rank]`, built once per ordinal.
    fn w_kc_device(&self, ordinal: usize) -> Result<Option<&dyn BackendStorage>> {
        let slot = self.w_kc_dev.get_or_init(|| {
            let nh = self.num_heads;
            let rocm = grim_backend_rocm::RocmDevice::shared(ordinal);
            let nope = self.qk_nope_head_dim;
            let rank = self.kv_a_layernorm.weight.shape().dims()[0];
            // w_kc is [nh][nope][rank] (row-major); store [nh][rank][nope].
            let mut t = vec![0.0f32; nh * rank * nope];
            for h in 0..nh {
                for d in 0..nope {
                    for r in 0..rank {
                        t[(h * rank + r) * nope + d] = self.w_kc[(h * nope + d) * rank + r];
                    }
                }
            }
            rocm.from_cpu(&t, &Shape::new(vec![nh, rank, nope]), DType::F32)
                .ok()
                .map(|b| (ordinal, Arc::from(b)))
        });
        Ok(match slot {
            Some((ord, b)) if *ord == ordinal => Some(b.as_ref()),
            _ => None,
        })
    }

    /// Device-resident `w_vc` as `[num_heads, v_head_dim, kv_lora_rank]` — the
    /// natural weight layout `matmul` already wants, so no transpose.
    fn w_vc_device(&self, ordinal: usize) -> Result<Option<&dyn BackendStorage>> {
        let slot = self.w_vc_t_dev.get_or_init(|| {
            let nh = self.num_heads;
            let vd = self.v_head_dim;
            let rank = self.kv_a_layernorm.weight.shape().dims()[0];
            // `matmul` takes its right operand in natural weight `[N, K]` layout
            // and computes `A @ Bᵀ`, and `w_vc[h]` is already `[vd, rank]`, so
            // it uploads as-is — transposing here would be a silent no-op that
            // only shows up as scrambled per-head blocks downstream.
            let rocm = grim_backend_rocm::RocmDevice::shared(ordinal);
            rocm.from_cpu(&self.w_vc, &Shape::new(vec![nh, vd, rank]), DType::F32)
                .ok()
                .map(|b| (ordinal, Arc::from(b)))
        });
        Ok(match slot {
            Some((ord, b)) if *ord == ordinal => Some(b.as_ref()),
            _ => None,
        })
    }

    /// GPU decode path via `BackendDevice::mla_absorbed_decode` (decode-only).
    /// Returns `Ok(None)` — the documented fallback — when the backend lacks the
    /// kernel, the rank exceeds the kernel's staging width, or `kv_b_proj` is
    /// quantized (the kernel reads the per-head `w_vc` blocks straight out of the
    /// device weight, which a packed storage does not expose). The host path then
    /// runs and dequantizes through `Linear`.
    #[allow(clippy::too_many_arguments)]
    fn gpu_absorbed_decode(
        &self,
        q_absorbed: &[f32],
        q_rope: &[f32],
        latent_all: &Tensor,
        rank: usize,
        total_kv_len: usize,
        scale: f32,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let nh = self.num_heads;
        let nope = self.qk_nope_head_dim;
        let rope_d = self.qk_rope_head_dim;
        let vd = self.v_head_dim;

        if rank > 512 {
            return Ok(None);
        }
        if self.kv_b_proj.weight.dtype().is_quantized() {
            return Ok(None);
        }

        let dev = grim_nn::modules::pick_device_for_storage_device(device);

        // The kernel applies a fixed 1/sqrt(rank + rope_d); pre-scale q so the
        // effective softmax scale stays the model's 1/sqrt(nope + rope_d).
        let kernel_scale = 1.0f32 / ((rank + rope_d) as f32).sqrt();
        let ratio = scale / kernel_scale;
        let q_abs_scaled: Vec<f32> = q_absorbed.iter().map(|v| v * ratio).collect();
        let q_rope_scaled: Vec<f32> = q_rope.iter().map(|v| v * ratio).collect();

        let Some(qa_all) =
            or_host_fallback(dev.from_cpu(&q_abs_scaled, &Shape::new(vec![nh, rank]), DType::F32))?
        else {
            return Ok(None);
        };
        let Some(qr_all) = or_host_fallback(dev.from_cpu(
            &q_rope_scaled,
            &Shape::new(vec![nh, rope_d]),
            DType::F32,
        ))?
        else {
            return Ok(None);
        };
        let out_shape = Shape::new(vec![1, nh * vd]);
        let Some(out_all) = or_host_fallback(dev.zeros(&out_shape, DType::F32))? else {
            return Ok(None);
        };

        let kv_st = latent_all.storage().as_ref();
        let w_src = self.kv_b_proj.weight.storage().as_ref();

        if or_host_fallback(dev.mla_absorbed_decode(
            qa_all.as_ref(),
            qr_all.as_ref(),
            kv_st,
            Some(w_src),
            out_all.as_ref(),
            nh,
            rank,
            rope_d,
            vd,
            total_kv_len,
            nope * rank,
            (nope + vd) * rank,
        ))?
        .is_none()
        {
            return Ok(None);
        }

        Ok(Some(Tensor::new(
            Arc::from(out_all),
            out_shape,
            DType::F32,
            QuantProvenance::default(),
            device.clone(),
        )))
    }
}

// Expert / MLP

pub struct Xing40Expert {
    pub w1: Linear,
    pub w3: Linear,
    pub w2: Linear,
}

/// SwiGLU expert projection triple (`gate_proj` / `up_proj` / `down_proj` in the
/// HF export, `w1` / `w3` / `w2` in GGUF exports) — both naming conventions are
/// probed so the model loads from safetensors and GGUF alike.
impl Xing40Expert {
    pub fn load(
        ws: &WeightSource<'_>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let load = |hf: &str, gguf: &str, in_dim: usize, out_dim: usize| -> Result<Linear> {
            Linear::load_shape(&ws.scoped(hf), [in_dim, out_dim])
                .or_else(|_| Linear::load_shape(&ws.scoped(gguf), [in_dim, out_dim]))
                .map_err(Error::from)
        };
        let w1 = load("gate_proj", "w1", hidden_size, intermediate_size)?;
        let w3 = load("up_proj", "w3", hidden_size, intermediate_size)?;
        let w2 = load("down_proj", "w2", intermediate_size, hidden_size)?;
        Ok(Self { w1, w3, w2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;
        let swiglu_t = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        Ok(self.w2.forward(&swiglu_t)?)
    }
}

impl Xing40Expert {
    /// Dense-layer loader with the same HF/GGUF naming probe.
    fn load_dense(ws: &WeightSource<'_>, hidden: usize, inter: usize) -> Result<Self> {
        Self::load(ws, hidden, inter)
    }
}

// MoE

pub struct Xing40Moe {
    pub gate: Linear,
    /// `mlp.gate.e_score_correction_bias` — `noaux_tc` selection bias
    /// (all-zero on freshly initialized checkpoints; trained on released ones).
    pub correction_bias: Option<Vec<f32>>,
    pub experts: Vec<Xing40Expert>,
    pub shared_experts: Option<Xing40Expert>,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    pub noaux_tc_routing: bool,
    /// Device-resident `[n_routed_experts]` copy of `correction_bias`, so the
    /// D2D route can apply it without a host round-trip. `None` on CPU.
    pub correction_bias_dev: Option<Tensor>,
    /// Charon grouped-dispatch resident weight cache.
    pub charon_cache: crate::shared_moe::CharonCache,
}

impl Xing40Moe {
    /// Load the MoE from either container.
    ///
    /// * GGUF: 3D expert banks `ffn_{gate,up,down}_exps.weight` (llama.cpp's
    ///   `[n_experts, ...]` layout), router `ffn_gate_inp.weight`, shared expert
    ///   `ffn_{gate,up,down}_shexp.weight`, bias `exp_probs_b.bias`.
    /// * safetensors: per-expert `mlp.experts.{i}.{gate,up,down}_proj.weight`,
    ///   router `mlp.gate.weight`, bias `mlp.gate.e_score_correction_bias`.
    pub fn load(ws: &WeightSource<'_>, cfg: &Xing40Config) -> Result<Self> {
        let (gate, correction_bias, experts, shared_experts) =
            if ws.has_tensor("ffn_gate_exps.weight") {
                // ---- GGUF container ----
                let gate = Linear::load_shape(
                    &ws.scoped("ffn_gate_inp"),
                    [cfg.hidden_size, cfg.n_routed_experts],
                )?;
                let bias = ws
                    .scoped("exp_probs_b")
                    .get_unconstrained("bias")
                    .ok()
                    .and_then(|t| t.to_vec_f32().ok());
                let bank = grim_nn::moe::ExpertBank::load(
                    ws,
                    cfg.n_routed_experts,
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                    false,
                )?;
                let experts: Vec<Xing40Expert> = (0..cfg.n_routed_experts)
                    .map(|i| Xing40Expert {
                        w1: bank.gate[i].clone(),
                        w3: bank.up[i].clone(),
                        w2: bank.down[i].clone(),
                    })
                    .collect();
                let shared = if cfg.n_shared_experts > 0 && ws.has_tensor("ffn_gate_shexp.weight") {
                    Some(Xing40Expert {
                        w1: Linear::load_shape(
                            &ws.scoped("ffn_gate_shexp"),
                            [
                                cfg.hidden_size,
                                cfg.moe_intermediate_size * cfg.n_shared_experts,
                            ],
                        )?,
                        w3: Linear::load_shape(
                            &ws.scoped("ffn_up_shexp"),
                            [
                                cfg.hidden_size,
                                cfg.moe_intermediate_size * cfg.n_shared_experts,
                            ],
                        )?,
                        w2: Linear::load_shape(
                            &ws.scoped("ffn_down_shexp"),
                            [
                                cfg.moe_intermediate_size * cfg.n_shared_experts,
                                cfg.hidden_size,
                            ],
                        )?,
                    })
                } else {
                    None
                };
                (gate, bias, experts, shared)
            } else {
                // ---- safetensors container ----
                let gate_ws = ws.scoped("gate");
                let gate = Linear::load_shape(&gate_ws, [cfg.hidden_size, cfg.n_routed_experts])?;
                let bias = gate_ws
                    .get_unconstrained("e_score_correction_bias")
                    .ok()
                    .and_then(|t| t.to_vec_f32().ok());

                let mut experts = Vec::with_capacity(cfg.n_routed_experts);
                let exp_ws = ws.scoped("experts");
                for e in 0..cfg.n_routed_experts {
                    experts.push(Xing40Expert::load(
                        &exp_ws.scoped(&e.to_string()),
                        cfg.hidden_size,
                        cfg.moe_intermediate_size,
                    )?);
                }
                let shared = if cfg.n_shared_experts > 0 {
                    Some(Xing40Expert::load(
                        &ws.scoped("shared_experts"),
                        cfg.hidden_size,
                        cfg.moe_intermediate_size * cfg.n_shared_experts,
                    )?)
                } else {
                    None
                };
                (gate, bias, experts, shared)
            };

        // Device-resident copy of the correction bias so the D2D route can read
        // it on the device; the host vector stays for the host reference path.
        let correction_bias_dev = match (&correction_bias, ws.device()) {
            (Some(b), dev) if !dev.is_cpu() && b.len() == cfg.n_routed_experts => {
                const_tensor(b.clone(), Shape::new(vec![cfg.n_routed_experts]), &dev).ok()
            }
            _ => None,
        };

        Ok(Self {
            gate,
            correction_bias,
            experts,
            shared_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            routed_scaling_factor: cfg.routed_scaling_factor,
            noaux_tc_routing: cfg.noaux_tc_routing,
            correction_bias_dev,
            charon_cache: crate::shared_moe::CharonCache::new(),
        })
    }

    /// Per-token routing for `noaux_tc` sigmoid gating: select on
    /// `sigmoid(logit) + bias`, combine on the raw `sigmoid(logit)`, then
    /// `norm_topk_prob` normalize and apply `routed_scaling_factor`.
    fn route_sigmoid(
        &self,
        logits_v: &[f32],
        seq_len: usize,
    ) -> Vec<crate::shared_moe::TokenRouting> {
        let num_exp = self.experts.len();
        let k = self.num_experts_per_tok.min(num_exp);
        let bias = self
            .correction_bias
            .as_ref()
            .map(|b| b.as_slice())
            .unwrap_or(&[]);
        let mut out = Vec::with_capacity(seq_len);
        for s in 0..seq_len {
            let row = &logits_v[s * num_exp..(s + 1) * num_exp];
            let scores: Vec<f32> = row.iter().map(|&l| sigmoid(l)).collect();
            let mut indexed: Vec<(usize, f32)> = (0..num_exp)
                .map(|i| {
                    let sel = if self.noaux_tc_routing {
                        scores[i] + bias.get(i).copied().unwrap_or(0.0)
                    } else {
                        scores[i]
                    };
                    (i, sel)
                })
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk = &indexed[..k];
            let raw: Vec<f32> = topk.iter().map(|(i, _)| scores[*i]).collect();
            let sum: f32 = raw.iter().sum();
            let entry: Vec<(usize, f32)> = topk
                .iter()
                .zip(raw.iter())
                .map(|((i, _), w)| (*i, w / (sum + 1e-20)))
                .collect();
            out.push(entry);
        }
        out
    }

    /// GPU-first MoE forward: routing (a small `[seq, n_experts]` host pull) then
    /// the shared Charon grouped dispatch, which dequantizes packed expert banks.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.gate.forward(x)?;
        let seq_len = x.shape().dims()[0];
        // Gate-logits D2H is deferred behind a closure so the bias-aware D2D
        // dispatch below never performs it. Only the two fallbacks call this.
        let host_logits = || logits.to_vec_f32();

        if seq_len == 0 {
            return Ok(x.clone());
        }

        // Device-resident path: the gate logits and the routing decision both
        // stay on the GPU (route_mode 2 = sigmoid + correction bias, which is
        // exactly Xing4.0's noaux_tc). Falls through to the host reference only
        // when the backend cannot run the dispatch.
        if x.device() != &Device::Cpu {
            let dev = grim_nn::modules::pick_device_for_storage_device(x.device());
            let experts: Vec<crate::shared_moe::MoeExpert> = self
                .experts
                .iter()
                .map(|e| crate::shared_moe::MoeExpert {
                    gate: e.w1.clone(),
                    up: e.w3.clone(),
                    down: e.w2.clone(),
                })
                .collect();
            let shared_expert =
                self.shared_experts
                    .as_ref()
                    .map(|e| crate::shared_moe::MoeExpert {
                        gate: e.w1.clone(),
                        up: e.w3.clone(),
                        down: e.w2.clone(),
                    });
            if let Ok(Some(out)) = crate::shared_moe::fused_moe_dispatch_from_logits_with_bias(
                dev.as_ref(),
                x,
                &logits,
                self.correction_bias_dev.as_ref(),
                &experts,
                shared_expert.as_ref(),
                self.num_experts_per_tok,
                self.routed_scaling_factor,
                2, // route_mode: sigmoid + e_score_correction_bias
                true, // norm_weights: xing4_0.expert_weights_norm = true
                &self.charon_cache,
            ) {
                return Ok(out);
            }

            // Precomputed-routing fallback: still D2D for the expert math, but
            // the routing table comes from the host reference. Only this branch
            // pays for the gate-logits D2H; the bias-aware path above never does.
            let routings = self.route_sigmoid(&host_logits()?, seq_len);
            if let Ok(out) = crate::shared_moe::fused_moe_dispatch(
                dev.as_ref(),
                x,
                &experts,
                shared_expert.as_ref(),
                &routings,
                self.routed_scaling_factor,
                &self.charon_cache,
            ) {
                return Ok(out);
            }
        }

        // Host reference path (CPU device, or GPU backends missing a primitive).
        let xv = x.to_vec_f32()?;
        let hidden = x.shape().dims()[1];
        let mut out = vec![0.0f32; seq_len * hidden];
        let routings = self.route_sigmoid(&host_logits()?, seq_len);
        for s in 0..seq_len {
            let token_x = cpu_tensor(
                xv[s * hidden..(s + 1) * hidden].to_vec(),
                Shape::new(vec![1, hidden]),
            );
            for (exp_idx, w) in &routings[s] {
                let exp_out = self.experts[*exp_idx].forward(&token_x)?.to_vec_f32()?;
                let w = w * self.routed_scaling_factor;
                for d in 0..hidden {
                    out[s * hidden + d] += w * exp_out[d];
                }
            }
        }
        let mut out_t = cpu_tensor(out, x.shape().clone());
        if let Some(ref shared) = self.shared_experts {
            let sh_out = shared.forward(x)?;
            out_t = grim_nn::modules::add_on_device(&out_t, &sh_out)?;
        }
        Ok(out_t)
    }
}

fn sigmoid(v: f32) -> f32 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}

// Block

pub struct Xing40Block {
    pub attn_norm: RmsNorm,
    pub attn_hc: Xing40HyperConnection,
    pub self_attn: Xing40Mla,
    pub ffn_norm: RmsNorm,
    pub ffn_hc: Xing40HyperConnection,
    pub mlp: Option<Xing40Expert>,
    pub moe: Option<Xing40Moe>,
    pub hc_mult: usize,
    pub hidden_size: usize,
}

impl Xing40Block {
    pub fn load(ws: &WeightSource<'_>, cfg: &Xing40Config, is_dense: bool) -> Result<Self> {
        // GGUF flattens the sub-module names (`attn_norm`, `hc_attn_*`); the
        // safetensors export nests them (`input_layernorm`, `attn_hc.*`).
        let (attn_norm, ffn_norm) = if ws.has_tensor("attn_norm.weight") {
            (
                RmsNorm::load(&ws.scoped("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?,
                RmsNorm::load(&ws.scoped("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?,
            )
        } else {
            (
                RmsNorm::load(
                    &ws.scoped("input_layernorm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                )?,
                RmsNorm::load(
                    &ws.scoped("post_attention_layernorm"),
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                )?,
            )
        };
        let attn_hc = Xing40HyperConnection::load(ws, cfg, "attn")?;
        let ffn_hc = Xing40HyperConnection::load(ws, cfg, "ffn")?;
        // GGUF keeps the MLA tensors at the layer root (`attn_q_a` …);
        // safetensors nests them under `self_attn.`.
        let attn_ws = if ws.has_tensor("attn_q_b.weight") {
            ws.with_tp_config(ws.tp_config())
        } else {
            ws.scoped("self_attn")
        };
        let self_attn = Xing40Mla::load(&attn_ws, cfg)?;

        let (mlp, moe) = if is_dense {
            // GGUF: flat `ffn_{gate,up,down}.weight`; safetensors: `mlp.*_proj`.
            let mlp = if ws.has_tensor("ffn_gate.weight") {
                Xing40Expert {
                    w1: Linear::load_shape(
                        &ws.scoped("ffn_gate"),
                        [cfg.hidden_size, cfg.intermediate_size],
                    )?,
                    w3: Linear::load_shape(
                        &ws.scoped("ffn_up"),
                        [cfg.hidden_size, cfg.intermediate_size],
                    )?,
                    w2: Linear::load_shape(
                        &ws.scoped("ffn_down"),
                        [cfg.intermediate_size, cfg.hidden_size],
                    )?,
                }
            } else {
                Xing40Expert::load_dense(&ws.scoped("mlp"), cfg.hidden_size, cfg.intermediate_size)?
            };
            (Some(mlp), None)
        } else {
            // GGUF keeps the router + banks at the layer root; safetensors nests
            // them under `mlp.`.
            let moe_ws = if ws.has_tensor("ffn_gate_exps.weight") {
                ws.with_tp_config(ws.tp_config())
            } else {
                ws.scoped("mlp")
            };
            let moe = Xing40Moe::load(&moe_ws, cfg)?;
            (None, Some(moe))
        };

        Ok(Self {
            attn_norm,
            attn_hc,
            self_attn,
            ffn_norm,
            ffn_hc,
            mlp,
            moe,
            hc_mult: cfg.hc_mult,
            hidden_size: cfg.hidden_size,
        })
    }

    /// One decoder layer over the `[seq, hc_mult, hidden]` stream state.
    /// Flattened on-device as `[seq, hc_mult * hidden]` (stream-major) so the MHC
    /// projection is a single matmul; the state is written back per the MHC update.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // On a device backend the whole block runs D2D: the multi-stream state
        // stays resident and neither the gate math nor the collapse/write-back
        // round-trips through the host. The CPU backend keeps the host
        // reference path — there the data is already in host memory, so that is
        // not a device->host fallback.
        match x.device() {
            Device::Rocm(_) => self.forward_d2d(x, positions, kv_cache),
            _ => self.forward_host(x, positions, kv_cache),
        }
    }

    /// Oracle entry points: `forward` dispatches on the input's device, but a
    /// parity gate must run BOTH paths on the same input. These expose them.
    pub fn forward_device_for_oracle(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        self.forward_d2d(x, positions, kv_cache)
    }

    pub fn forward_host_for_oracle(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        self.forward_host(x, positions, kv_cache)
    }

    /// Device-resident block forward. Requires the device hyper-connection
    /// primitives (`narrow_cols` / `write_cols` / `row_scale` + `grim_mhc_gates`).
    fn forward_d2d(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let trace_nan = std::env::var_os("GRIM_XING_TRACE").is_some();
        let nan_stage = |name: &str, t: &Tensor| {
            if !trace_nan { return; }
            if let Ok(v) = t.to_vec_f32() {
                let nan = v.iter().filter(|x| x.is_nan()).count();
                eprintln!("[xing-trace] {name}: len {} nan {nan} head {:?}", v.len(), &v[..v.len().min(4)]);
            }
        };
        nan_stage("block_in", x);
        // 1. attn_hc: collapse the multi-stream state into the single attention input.
        let attn_gates = self.attn_hc.gates_d2d(x)?;
        let collapsed = self.attn_hc.collapse_d2d(x, &attn_gates)?;
        let collapsed = self.attn_norm.forward(&collapsed)?;

        nan_stage("attn_in", &collapsed);
        // 2. Self-attention on the collapsed stream.
        let attn_out = self.self_attn.forward(&collapsed, positions, kv_cache)?;
        nan_stage("attn_out", &attn_out);
        // 3. Write the attention result back into the streams:
        //    streams = post ⊗ attn_out + comb @ streams.
        let streams = self.attn_hc.update_d2d(x, &attn_out, &attn_gates)?;
        nan_stage("streams_after_attn", &streams);
        // 4. ffn_hc: collapse for the feed-forward.
        let ffn_gates = self.ffn_hc.gates_d2d(&streams)?;
        let collapsed = self.ffn_hc.collapse_d2d(&streams, &ffn_gates)?;
        let collapsed = self.ffn_norm.forward(&collapsed)?;
        nan_stage("ffn_in", &collapsed);
        // 5. Dense SwiGLU or sparse MoE.
        let ffn_out = if let Some(ref mlp) = self.mlp {
            mlp.forward(&collapsed)?
        } else if let Some(ref moe) = self.moe {
            moe.forward(&collapsed)?
        } else {
            collapsed.clone()
        };
        nan_stage("ffn_out", &ffn_out);

        // 6. Write the FFN result back into the streams.
        let out = self.ffn_hc.update_d2d(&streams, &ffn_out, &ffn_gates)?;
        nan_stage("block_out", &out);
        Ok(out)
    }

    /// Host reference block forward (CPU backend).
    fn forward_host(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        // 1. attn_hc: collapse the multi-stream state into the single attention input.
        let (attn_gates, collapsed_v) = self.attn_hc.forward(x, seq_len)?;
        let collapsed = cpu_tensor(collapsed_v, Shape::new(vec![seq_len, self.hidden_size]));
        let collapsed = self.attn_norm.forward(&collapsed)?;

        // 2. Self-attention on the collapsed stream.
        let attn_out = self.self_attn.forward(&collapsed, positions, kv_cache)?;

        // 3. Write the attention result back into the streams:
        //    streams = post ⊗ attn_out + comb @ streams.
        let x_v = x.to_vec_f32()?;
        let attn_v = attn_out.to_vec_f32()?;
        let streams_v = self.attn_hc.write_back(&x_v, &attn_v, seq_len, &attn_gates);
        let hc = self.hc_mult;
        let streams_shape = Shape::new(vec![seq_len, hc * self.hidden_size]);
        let streams = self.upload_streams(&streams_v, streams_shape.clone(), x.device())?;

        // 4. ffn_hc: collapse for the feed-forward.
        let (ffn_gates, collapsed_v) = self.ffn_hc.forward(&streams, seq_len)?;
        let collapsed = cpu_tensor(collapsed_v, Shape::new(vec![seq_len, self.hidden_size]));
        let collapsed = self.ffn_norm.forward(&collapsed)?;

        // 5. Dense SwiGLU or sparse MoE.
        let ffn_out = if let Some(ref mlp) = self.mlp {
            mlp.forward(&collapsed)?
        } else if let Some(ref moe) = self.moe {
            moe.forward(&collapsed)?
        } else {
            collapsed.clone()
        };

        // 6. Write the FFN result back into the streams.
        let streams_v = streams.to_vec_f32()?;
        let ffn_v = ffn_out.to_vec_f32()?;
        let next_v = self
            .ffn_hc
            .write_back(&streams_v, &ffn_v, seq_len, &ffn_gates);
        self.upload_streams(&next_v, streams_shape, x.device())
    }

    fn upload_streams(&self, v: &[f32], shape: Shape, device: &Device) -> Result<Tensor> {
        if device.is_cpu() {
            return Ok(cpu_tensor(v.to_vec(), shape));
        }
        let dev = grim_nn::modules::pick_device_for_storage_device(device);
        let st = dev.from_cpu(v, &shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(st),
            shape,
            DType::F32,
            QuantProvenance::default(),
            device.clone(),
        ))
    }
}

// Model & Session

pub struct Xing40 {
    pub cfg: Xing40Config,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<Xing40Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Xing40 {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: Xing40Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Xing40Config,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        // GGUF stores the decoder stack under `blk.{i}.*` with flat
        // `token_embd` / `output_norm` / `output`; safetensors uses
        // `model.layers.{i}.*` with `model.embed_tokens` / `model.norm`.
        let gguf = ws.has_tensor("blk.0.attn_norm.weight");
        let (root, layer_root, tok_leaf, norm_leaf) = if gguf {
            // `with_tp_config` re-derives an owned WeightSource at the same prefix.
            (
                ws.with_tp_config(ws.tp_config()),
                ws.scoped("blk"),
                "token_embd",
                "output_norm",
            )
        } else {
            (
                ws.scoped("model"),
                ws.scoped("model").scoped("layers"),
                "embed_tokens",
                "norm",
            )
        };

        // Both containers store the embedding row-major as `[vocab, hidden]`
        // (the GGUF `token_embd.weight` is `[131072, 3584]`), which is
        // `load_shape`'s `[in_dim, out_dim] = [hidden, vocab]`.
        //
        // `Embedding::load` applies the format contract: keep the table packed
        // when the backend has a gather kernel for the scheme (Q4_K, IQ3_S),
        // materialize f32 otherwise. Loading through `Linear::load_shape` kept
        // the table packed unconditionally, and the gather then read IQ3_S
        // block bytes as f32 (logits full of 3.4e38, NaN from token 0).
        let emb_ws = root.scoped(tok_leaf);
        let emb = grim_nn::modules::Embedding::load(&emb_ws, cfg.vocab_size, cfg.hidden_size)?;
        let tok_embeddings = Linear {
            weight: emb.weight.clone(),
            bias: None,
            w_t: emb.weight.clone(),
            quant_format: None,
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = layer_root.scoped(&i.to_string());
            let is_dense = i < cfg.first_k_dense_replace;
            layers.push(Xing40Block::load(&layer_ws, &cfg, is_dense)?);
        }

        let norm = RmsNorm::load(&root.scoped(norm_leaf), cfg.hidden_size, cfg.rms_norm_eps)?;
        // `output.weight` (GGUF) and `lm_head.weight` (safetensors) are both
        // `[vocab, hidden]`.
        let output = Linear::load_shape(&ws.scoped("output"), [cfg.hidden_size, cfg.vocab_size])
            .or_else(|_| {
                Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            })
            .unwrap_or_else(|_| tok_embeddings.clone());

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }
}

impl Model for Xing40 {
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

impl CausalLm for Xing40 {
    fn new_session(&self) -> Box<dyn SessionT> {
        let mut session = grim_core::session::Inner::new(self.device.clone());
        // The per-layer LATENT kv caches must persist across decode steps;
        // the generation-state gate caught them being rebuilt per call,
        // which made every decode step attend only itself.
        session.set_model_state(Box::new(Vec::<Option<(Tensor, Tensor)>>::new()));
        Box::new(session)
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

        let x0 = grim_nn::embedding_gather_on_device(
            &self.tok_embeddings.weight,
            &ids,
            seq_len,
            self.cfg.hidden_size,
        )?;
        if std::env::var_os("GRIM_XING_TRACE").is_some() {
            if let Ok(v) = x0.to_vec_f32() {
                let nan = v.iter().filter(|x| x.is_nan()).count();
                let max = v.iter().cloned().fold(0f32, f32::max);
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                eprintln!("[xing-trace] x0 embedding: len {} nan {nan} rms {rms:.4e} max {max:.4e} head {:?}", v.len(), &v[..v.len().min(4)]);
                let r0: Vec<f32> = v[..v.len().min(3584)].to_vec();
                let m0 = r0.iter().cloned().fold(0f32, f32::max);
                let r0r = (r0.iter().map(|x| x * x).sum::<f32>() / r0.len() as f32).sqrt();
                eprintln!("[xing-trace] x0 token0 row: rms {r0r:.4e} max {m0:.4e} head {:?}", &r0[..4]);
            }
            eprintln!("[xing-trace] tok_emb dtype {:?} shape {:?} device {:?}", self.tok_embeddings.weight.dtype(), self.tok_embeddings.weight.shape(), self.tok_embeddings.weight.device());
        }

        // Seed the hc_mult streams with the embedding (all streams start equal).
        let hc = self.cfg.hc_mult;
        let hidden = self.cfg.hidden_size;
        let shape = Shape::new(vec![seq_len, hc * hidden]);
        let plane = Shape::new(vec![seq_len, hidden]);
        let mut x = if self.device.is_cpu() {
            let x0_v = x0.to_vec_f32()?;
            let mut streams_v = vec![0.0f32; seq_len * hc * hidden];
            for s in 0..seq_len {
                for h in 0..hc {
                    let src = s * hidden;
                    let dst = (s * hc + h) * hidden;
                    streams_v[dst..dst + hidden].copy_from_slice(&x0_v[src..src + hidden]);
                }
            }
            cpu_tensor(streams_v, shape)
        } else {
            // Broadcast the embedding across the hc streams with device
            // column writes — no embedding round-trip, no host-side seeding.
            seed_streams_device(&x0, hc, hidden, seq_len, &shape, &self.device)?
        };

        // The caches live in the SESSION so decode steps attend the whole
        // history; a fresh vec here would drop the prefill state between
        // engine calls (caught by full_model_generation_state_consistency:
        // incremental-vs-teacher-forced rel ~1.0 at every generated position).
        if session.model_state().is_none() {
            session.set_model_state(Box::new(Vec::<Option<(Tensor, Tensor)>>::new()));
        }
        let kv_caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<(Tensor, Tensor)>>>())
            .ok_or_else(|| {
                Error::Backend(
                    "Xing40::forward: session.model_state must be Vec<Option<(Tensor, Tensor)>>"
                        .into(),
                )
            })?;
        if kv_caches.len() != self.layers.len() {
            *kv_caches = vec![None; self.layers.len()];
        }
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_v, &mut kv_caches[i])?;
        }
        // Collapse the streams with a mean, then norm + lm_head.
        let collapsed = mean_collapse(&x, hc, hidden, seq_len, &plane)?;
        let normed = self.norm.forward(&collapsed)?;
        let logits = self.output.forward(&normed)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_shape(hc: usize, hidden: usize, seq: usize) -> Shape {
        Shape::new(vec![seq, hc * hidden])
    }

    /// Model-level gate 1: the D2D stream seed must equal a CPU broadcast of
    /// the embedding (the reference's
    /// `inputs_embeds.unsqueeze(2).expand(-1, -1, hc, -1)`). No layer gate
    /// reaches this function, and a wrong seed corrupts the first forward pass
    /// while every block still agrees host-vs-device.
    #[test]
    #[ignore = "needs a ROCm device; GRIM_RUN_GPU_TESTS=1 -- --ignored"]
    fn seed_streams_matches_a_cpu_broadcast() {
        if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (hc, hidden, seq) = (4usize, 64usize, 5usize);
        let mut x0 = vec![0f32; seq * hidden];
        let mut s = 0x1234_5678_9abc_def0u64;
        for v in x0.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f32 / 8388608.0 - 1.0) * 0.03;
        }
        // In production x0 is device-resident (the embedding gather lands
        // there) and the seed's column writes are a D2D op; mirror that.
        let x0_shape = Shape::new(vec![seq, hidden]);
        let dev0 = grim_backend_rocm::RocmDevice::shared(0);
        use grim_tensor::CoreTensorOps;
        let x0_st = dev0.from_cpu(&x0, &x0_shape, DType::F32).expect("upload x0");
        let x0_t = Tensor::new(
            Arc::from(x0_st),
            x0_shape,
            DType::F32,
            QuantProvenance::default(),
            Device::Rocm(0),
        );
        let full = Shape::new(vec![seq, hc * hidden]);
        let seeded =
            seed_streams_device(&x0_t, hc, hidden, seq, &full, &Device::Rocm(0)).expect("seed");
        let got = seeded.to_vec_f32().expect("read seeded");
        let mut expect = vec![0f32; seq * hc * hidden];
        for si in 0..seq {
            for h in 0..hc {
                let src = si * hidden;
                let dst = (si * hc + h) * hidden;
                expect[dst..dst + hidden].copy_from_slice(&x0[src..src + hidden]);
            }
        }
        let diff: f32 = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("[xing-oracle] seed broadcast: max|dev-broadcast| = {diff:.3e}");
        assert!(diff == 0.0, "seed broadcast diverges: {diff}");
    }

    /// Model-level gate 2: `mean_collapse` must equal a plain per-(token,
    /// channel) mean over the hc streams (the reference's
    /// `hidden_states.mean(dim=2)`). Nothing in the block oracle reaches it.
    #[test]
    #[ignore = "needs a ROCm device; GRIM_RUN_GPU_TESTS=1 -- --ignored"]
    fn mean_collapse_matches_a_plain_mean() {
        if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
            return;
        }
        let (hc, hidden, seq) = (4usize, 128usize, 6usize);
        let mut streams = vec![0f32; seq * hc * hidden];
        let mut s = 0xDEAD_BEEF_1234_5678u64;
        for v in streams.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f32 / 8388608.0 - 1.0) * 0.05;
        }
        let plane = Shape::new(vec![seq, hidden]);
        let got = mean_collapse(
            &cpu_tensor(streams.clone(), full_shape(hc, hidden, seq)),
            hc,
            hidden,
            seq,
            &plane,
        )
        .expect("collapse");
        let got = got.to_vec_f32().expect("read");
        let mut expect = vec![0f32; seq * hidden];
        for si in 0..seq {
            for h in 0..hc {
                let src = (si * hc + h) * hidden;
                for d in 0..hidden {
                    expect[si * hidden + d] += streams[src + d] / hc as f32;
                }
            }
        }
        let diff: f32 = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("[xing-oracle] mean collapse: max|dev-mean| = {diff:.3e}");
        assert!(diff < 1e-6, "mean collapse diverges: {diff}");
    }

    /// Model-level gate 3: the head's Q6_K `lm_head` (n = 131072) on device
    /// against a host dequantized matmul on the same hidden state. The block
    /// oracle never touches the head, and the IQ3_S WMMA GEMM was wrong at
    /// real shapes - the output projection deserves the same treatment.
    #[test]
    #[ignore = "needs XING_GGUF + a ROCm device; GRIM_RUN_GPU_TESTS=1 -- --ignored"]
    fn lm_head_device_matches_a_host_dequant_matmul() {
        if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
            return;
        }
        let Ok(path) = std::env::var("XING_GGUF") else {
            return;
        };
        use grim_format::tprov::GgufProvider;
        let prov = GgufProvider::open(path.as_str()).expect("open gguf");
        // Two copies: the host reference dequantizes on CPU, the device arm
        // needs a ROCm-resident weight (a CPU weight cannot feed a matmul).
        let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu);
        let head = Linear::load_shape(&ws_cpu.pp("output"), [3584, 131072]).expect("load lm_head cpu");
        let ws_dev = grim_nn::WeightSource::root(&prov, Device::Rocm(0));
        let head_dev =
            Linear::load_shape(&ws_dev.pp("output"), [3584, 131072]).expect("load lm_head dev");
        let hidden = 3584usize;
        let mut x = vec![0f32; hidden];
        let mut s = 0x5A17_9E37u64;
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f32 / 8388608.0 - 1.0) * 0.5;
        }
        let x_cpu = cpu_tensor(x.clone(), Shape::new(vec![1, hidden]));
        let want = head.forward(&x_cpu).expect("host lm_head").to_vec_f32().expect("read host");

        let x_shape = Shape::new(vec![1, hidden]);
        use grim_tensor::{CoreTensorOps, DType, QuantProvenance};
        let dev = grim_backend_rocm::RocmDevice::shared(0);
        let x_dev = Tensor::new(
            Arc::from(dev.from_cpu(&x, &x_shape, DType::F32).expect("upload")),
            x_shape,
            DType::F32,
            QuantProvenance::default(),
            Device::Rocm(0),
        );
        let got = head_dev.forward(&x_dev).expect("device lm_head").to_vec_f32().expect("read dev");
        assert_eq!(got.len(), want.len(), "logits length mismatch");
        let scale = want
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        let mut max_abs = 0f32;
        for (g, w) in got.iter().zip(&want) {
            max_abs = max_abs.max((g - w).abs());
        }
        let rel = max_abs / scale;
        eprintln!("[xing-oracle] lm_head: max_abs {max_abs:.3e} rel {rel:.3e}");
        assert!(rel < 1e-3, "lm_head device vs host diverges (rel {rel:.3e})");
    }

    /// The GGUF split-bank reassembly must produce the same matrix the
    /// safetensors container ships as one `kv_b_proj.weight`, because the
    /// latent-absorbed decode kernel indexes both identically.
    #[test]
    fn split_mla_bank_reassembly_matches_safetensors_layout() {
        // Xing4.0's real per-layer dimensions, so a layout mistake here is the
        // real one rather than an artefact of tiny test dims.
        let (nh, rank, nope, vd) = (32usize, 512usize, 128usize, 128usize);

        // A "ground truth" kv_b_proj.weight in the safetensors [out, in] layout.
        let kv_b_true: Vec<f32> = (0..nh * (nope + vd) * rank)
            .map(|i| ((i % 977) as f32) * 0.001 - 0.5)
            .collect();

        // The safetensors arm derives its host up-projections from exactly this.
        let (kc_true, vc_true) =
            crate::mla_common::extract_kv_b_up_projs(&kv_b_true, nh, nope, vd, rank);

        // Now express the same weights the way GGUF stores them: k_b transposed
        // to [nh, rank, nope], v_b as [nh, vd, rank].
        let mut k_b = vec![0.0f32; nh * rank * nope];
        let mut v_b = vec![0.0f32; nh * vd * rank];
        for h in 0..nh {
            for d in 0..nope {
                for r in 0..rank {
                    k_b[(h * rank + r) * nope + d] = kc_true[(h * nope + d) * rank + r];
                }
            }
            for d in 0..vd {
                for r in 0..rank {
                    v_b[(h * vd + d) * rank + r] = vc_true[(h * vd + d) * rank + r];
                }
            }
        }

        let (kc, vc, kv_b) = assemble_split_mla_banks(&k_b, &v_b, nh, rank, nope, vd);

        assert_eq!(kc, kc_true, "w_kc must match the safetensors derivation");
        assert_eq!(vc, vc_true, "w_vc must match the safetensors derivation");
        assert_eq!(
            kv_b, kv_b_true,
            "reassembled kv_b must be bit-identical to the safetensors kv_b_proj.weight"
        );

        // And the kernel's own indexing must land on the value rows of each head:
        // head stride (nope + vd) * rank, value offset nope * rank.
        for h in 0..nh {
            let base = h * (nope + vd) * rank;
            assert_eq!(
                &kv_b[base + nope * rank..base + (nope + vd) * rank],
                &vc_true[h * vd * rank..(h + 1) * vd * rank],
                "head {h}: kernel value-window must be that head's w_vc rows"
            );
            assert_eq!(
                &kv_b[base..base + nope * rank],
                &kc_true[h * nope * rank..(h + 1) * nope * rank],
                "head {h}: kernel key-window must be that head's w_kc rows"
            );
        }
    }

    /// A non-square bank pair must not silently transpose: `attn_k_b` is
    /// `[nh, rank, nope]` while `attn_v_b` is `[nh, v_head, rank]`, and the
    /// element counts coincide, so a wrong orientation would pass a count check.
    #[test]
    fn split_mla_banks_handle_asymmetric_head_dims() {
        let (nh, rank, nope, vd) = (3usize, 5usize, 7usize, 2usize);
        let k_b: Vec<f32> = (0..nh * rank * nope).map(|i| i as f32).collect();
        let v_b: Vec<f32> = (0..nh * vd * rank).map(|i| -(i as f32)).collect();

        let (kc, vc, kv_b) = assemble_split_mla_banks(&k_b, &v_b, nh, rank, nope, vd);

        // w_kc[h][d][r] == k_b[h][r][d]  (the transpose that defines this bank)
        for h in 0..nh {
            for d in 0..nope {
                for r in 0..rank {
                    assert_eq!(
                        kc[(h * nope + d) * rank + r],
                        k_b[(h * rank + r) * nope + d]
                    );
                }
            }
            for d in 0..vd {
                for r in 0..rank {
                    assert_eq!(vc[(h * vd + d) * rank + r], v_b[(h * vd + d) * rank + r]);
                }
            }
        }
        assert_eq!(kv_b.len(), nh * (nope + vd) * rank);
        // The first head's key block must be the transpose, not a copy.
        assert_eq!(kv_b[0], k_b[0]); // d=0, r=0
        assert_eq!(kv_b[1], k_b[nope]); // d=0, r=1
    }

    #[test]
    fn test_xing40_config() {
        let cfg = Xing40Config::default();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.n_routed_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 4);
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.q_lora_rank, Some(768));
        assert_eq!(cfg.first_k_dense_replace, 2);
        assert!(cfg.noaux_tc_routing);
    }

    #[test]
    fn sinkhorn_combiner_is_doubly_stochastic() {
        // A zero-projection module makes the comb logits constant, so the
        // Sinkhorn iterations must still produce a doubly stochastic matrix.
        let cfg = Xing40Config::default();
        let hc = cfg.hc_mult;
        let mix = (2 + hc) * hc;
        let seq_len = 3;
        let proj = vec![0.0f32; seq_len * mix];
        let module = Xing40HyperConnection::identity(&cfg, &Device::Cpu);
        let gates = module.gates_from_projection(&proj, seq_len);
        for s in 0..seq_len {
            for o in 0..hc {
                let mut rowsum = 0.0f32;
                for i in 0..hc {
                    rowsum += gates.comb[(s * hc * hc) + o * hc + i];
                }
                assert!((rowsum - 1.0).abs() < 1e-4, "row {o} sums to {rowsum}");
            }
            for i in 0..hc {
                let mut colsum = 0.0f32;
                for o in 0..hc {
                    colsum += gates.comb[(s * hc * hc) + o * hc + i];
                }
                assert!((colsum - 1.0).abs() < 1e-4, "col {i} sums to {colsum}");
            }
        }
    }

    #[test]
    fn post_gate_is_bounded_by_two() {
        // post = 2 * sigmoid(...) ∈ (0, 2) by construction.
        let cfg = Xing40Config::default();
        let hc = cfg.hc_mult;
        let mix = (2 + hc) * hc;
        let proj: Vec<f32> = (0..mix).map(|i| (i as f32 - 8.0) * 3.0).collect();
        let module = Xing40HyperConnection::identity(&cfg, &Device::Cpu);
        let gates = module.gates_from_projection(&proj, 1);
        for h in 0..hc {
            let p = gates.post[h];
            assert!(p > 0.0 && p < 2.0, "post[{h}] = {p} out of (0, 2)");
            let pr = gates.pre[h];
            assert!(pr > 0.0 && pr < 1.0, "pre[{h}] = {pr} out of (0, 1)");
        }
    }

    #[test]
    fn noaux_tc_bias_steers_selection_not_weights() {
        // With `noaux_tc`, a large bias promotes an expert into the top-k, but
        // the combine weight is still the raw sigmoid of its logit.
        let cfg = Xing40Config::default();
        let num_exp = 4;
        let k = 2;
        let tiny_expert = || Xing40Expert {
            w1: Linear::from_tensor(cpu_tensor(vec![0.0f32; 1], Shape::new(vec![1, 1])), None),
            w3: Linear::from_tensor(cpu_tensor(vec![0.0f32; 1], Shape::new(vec![1, 1])), None),
            w2: Linear::from_tensor(cpu_tensor(vec![0.0f32; 1], Shape::new(vec![1, 1])), None),
        };
        let moe = Xing40Moe {
            gate: Linear::from_tensor(
                cpu_tensor(
                    vec![0.0f32; cfg.hidden_size * num_exp],
                    Shape::new(vec![cfg.hidden_size, num_exp]),
                ),
                None,
            ),
            // Expert 3 has the lowest logit but a large correction bias.
            correction_bias: Some(vec![0.0, 0.0, 0.0, 100.0]),
            experts: (0..num_exp).map(|_| tiny_expert()).collect(),
            shared_experts: None,
            num_experts_per_tok: k,
            routed_scaling_factor: 1.0,
            noaux_tc_routing: true,
            correction_bias_dev: None,
            charon_cache: crate::shared_moe::CharonCache::new(),
        };
        // logits: sigmoid values ascending so expert 3 is last without the bias.
        let logits = vec![-2.0, -1.0, 0.0, 1.0];
        let routings = moe.route_sigmoid(&logits, 1);
        let selected: Vec<usize> = routings[0].iter().map(|(i, _)| *i).collect();
        assert!(
            selected.contains(&3),
            "bias must promote expert 3, selected {selected:?}"
        );
        let sum: f32 = routings[0].iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights must sum to 1, got {sum}");
    }
}
