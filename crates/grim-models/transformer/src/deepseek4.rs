//! DeepSeek V4 architecture with Hyper-Connections (multi-stream residual), Sqrt-Softplus MoE routing, and Compressor-Indexer attention.
//!
//! # Architecture Details
//! - **Hyper-Connections (`hc_mult = 4`)**: Multi-stream residual state propagation ($h_0, h_1, h_2, h_3$) with linear mixing across layer transitions.
//! - **Sqrt-Softplus Gating**: MoE gating activation function $\text{gate}(x) = \sqrt{\text{softplus}(x \cdot W_g)}$ for stable high-capacity expert distribution.
//! - **Compressor/Indexer MLA**: Compressed latent memory projection paired with sparse indexing.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, DType, Device, QuantProvenance, Shape, Tensor};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for DeepSeek V4 model architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeepSeek4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub routed_scaling_factor: f32,
    pub hc_mult: usize,
    pub sqrtsoftplus_moe: bool,
    pub compressor_indexer_enabled: bool,
}

impl Default for DeepSeek4Config {
    fn default() -> Self {
        Self {
            vocab_size: 129280,
            hidden_size: 8192,
            num_heads: 128,
            num_kv_heads: 128,
            head_dim: 192,
            num_layers: 64,
            intermediate_size: 20480,
            kv_lora_rank: 512,
            q_lora_rank: Some(1536),
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 262144,
            moe_intermediate_size: 2048,
            n_routed_experts: 256,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            first_k_dense_replace: 3,
            routed_scaling_factor: 2.5,
            hc_mult: 4,
            sqrtsoftplus_moe: true,
            compressor_indexer_enabled: true,
        }
    }
}

impl ModelConfig for DeepSeek4Config {
    fn name(&self) -> &str {
        "deepseek4"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// GPU fallback guard
// ---------------------------------------------------------------------------

/// `Ok(None)` marks "backend lacks the kernel — use the host fallback";
/// other errors are real failures and propagate.
fn or_host_fallback<T>(r: std::result::Result<T, grim_tensor::Error>) -> Result<Option<T>> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e) if grim_nn::is_kernel_unimplemented(&e) => Ok(None),
        Err(e) => Err(grim_core::error::Error::from(e)),
    }
}

/// Elementwise `t * scalar` dispatched on the tensor's own device
/// (hyper-connection residual scaling without a host round-trip).
fn mul_scalar_on_device(t: &Tensor, s: f32) -> Result<Tensor> {
    let dev = grim_nn::modules::pick_device_for_tensor(t);
    let (st, _h) = dev
        .mul_scalar(t.storage().as_ref(), s, t.shape())
        .map_err(grim_core::error::Error::from)?;
    Ok(Tensor::new(
        Arc::from(st),
        t.shape().clone(),
        t.dtype(),
        QuantProvenance::default(),
        t.device().clone(),
    ))
}

// ---------------------------------------------------------------------------
// MLA Attention Block
// ---------------------------------------------------------------------------

pub struct DeepSeek4Mla {
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
    /// `[num_heads, qk_nope_head_dim, kv_lora_rank]` (rows of `kv_b_proj.weight`).
    pub w_kc: Vec<f32>,
    /// Per-head value up-projection `w_vc[h]`, row-major
    /// `[num_heads, v_head_dim, kv_lora_rank]`.
    pub w_vc: Vec<f32>,
}

impl DeepSeek4Mla {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeepSeek4Config) -> Result<Self> {
        let q_dim = cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj_direct) =
            if let Some(q_rank) = cfg.q_lora_rank {
                let qa = Linear::load_shape(&ws.scoped("q_a_proj"), [cfg.hidden_size, q_rank]).ok();
                let qn = RmsNorm::load(&ws.scoped("q_a_layernorm"), q_rank, cfg.rms_norm_eps).ok();
                let qb = Linear::load_shape(&ws.scoped("q_b_proj"), [q_rank, q_dim]).ok();
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
            [cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim],
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

        let rope = Rope::new(cfg.qk_rope_head_dim, cfg.rope_theta);

        // Extract the per-head key/value up-projections from kv_b_proj's
        // weight ([num_heads * (nope + v), rank], GGUF row-major) so queries
        // can absorb w_kc and attention can run in latent space.
        let kv_b_w = kv_b_proj.weight.to_vec_f32()?;
        let (w_kc, w_vc) = crate::mla_common::extract_kv_b_up_projs(
            &kv_b_w,
            cfg.num_heads,
            cfg.qk_nope_head_dim,
            cfg.v_head_dim,
            cfg.kv_lora_rank,
        );

        Ok(Self {
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
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];

        // 1. Q projection
        let q_full = if let (Some(qa), Some(qn), Some(qb)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_lat = qa.forward(x)?;
            let q_lat_normed = qn.forward(&q_lat)?;
            qb.forward(&q_lat_normed)?
        } else if let Some(ref q_direct) = self.q_proj_direct {
            q_direct.forward(x)?
        } else {
            return Err(Error::Config("DeepSeek4: no valid Q projection".into()));
        };
        let q_full_v = q_full.to_vec_f32()?;
        let _total_q_head = self.qk_nope_head_dim + self.qk_rope_head_dim;

        // Split q/kv via the shared MLA helper (same as old per-head loops).
        let (q_nope_v, mut q_rope_v) = crate::mla_common::split_q_nope_rope(
            &q_full_v,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
        );

        crate::qwen35::apply_rope_neox(
            &mut q_rope_v,
            positions,
            self.num_heads,
            self.qk_rope_head_dim,
            10000.0,
        );

        // 2. KV latent projection
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

        crate::qwen35::apply_rope_neox(&mut k_rope_v, positions, 1, self.qk_rope_head_dim, 10000.0);

        // 3. Absorb the per-head key up-projection (w_kc) into the query so
        //    attention runs entirely in latent space:
        //    q_absorbed[s,h] = q_nope[s,h] @ w_kc[h]^T
        //    (q_nope · (w_kc c) == (q_nope w_kc) · c.)
        let kv_a_normed_v = kv_a_normed.to_vec_f32()?;
        let rank = kv_rank;
        let nope = self.qk_nope_head_dim;
        let rope_d = self.qk_rope_head_dim;
        let vd = self.v_head_dim;
        let nh = self.num_heads;

        let q_absorbed = crate::mla_common::absorb_query_wkc(
            &q_nope_v,
            &self.w_kc,
            seq_len,
            nh,
            nope,
            rank,
        );

        // 4. Compressed latent rows — [normed c_kv || roped k_pe].
        let latent_new = crate::mla_common::pack_latent_rows(
            &kv_a_normed_v,
            &k_rope_v,
            seq_len,
            rank,
            rope_d,
        );

        // 5. Append to the device-resident latent KV cache. Format: `.0` holds
        //    the compressed latent `[total_kv, rank + rope_d]`; `.1` is unused
        //    (kept only for the `(Tensor, Tensor)` cache shape the surrounding
        //    plumbing uses). The history stays on its device — only the new
        //    rows cross H2D, and the append is a D2D concat.
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
        *kv_cache = Some((
            latent_all.clone(),
            cpu_tensor(Vec::new(), Shape::new(vec![0, 0])),
        ));

        let scale = 1.0 / ((nope + rope_d) as f32).sqrt();

        // 6a. GPU decode fast path (decode-only kernel: one launch per head).
        if seq_len == 1 && x.device() != &Device::Cpu {
            if let Some(attn_t) =
                self.gpu_absorbed_decode(&q_absorbed, &q_rope_v, &latent_all, rank, total_kv_len, scale, x.device())?
            {
                return Ok(self.o_proj.forward(&attn_t)?);
            }
        }

        // 6b. Scalar latent-space reference path with causal masking — the
        // documented FALLBACK, reached only when the backend lacks the MLA
        // decode kernel (`is_kernel_unimplemented`) or on the CPU device.
        // Query at absolute position cache_offset + s attends only to
        // t <= cache_offset + s.
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

    /// GPU decode path via `BackendDevice::mla_absorbed_decode` (decode-only,
    /// `seq_len == 1`). Runs entirely on-device: the latent cache is the
    /// device-resident tensor (no H2D re-upload), the per-head `w_vc`
    /// projection happens inside the kernel, and the decoded output stays a
    /// device tensor — no `synchronize`, no result D2H, no host matmul.
    ///
    /// The kernel indexes `w_uv` WITHOUT a per-head offset
    /// (`w_uv[v * kv_lora_rank + c]` for every head), so a single multi-head
    /// launch would repeat head 0's rows for all heads. The kernel is instead
    /// launched once per head with `num_heads = 1`, passing exactly that
    /// head's `[v_head_dim, kv_lora_rank]` block extracted D2D from the
    /// device-resident `kv_b_proj.weight` (contiguous rows
    /// `[h*(nope+v) + nope, +v)` — identical values to `self.w_vc[h]`).
    ///
    /// Returns `Ok(None)` when the backend lacks a needed kernel
    /// (`is_kernel_unimplemented`); the caller then runs the scalar latent
    /// loop. Real kernel failures propagate.
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

        // The kernel stages the normalized latent in a fixed 512-wide shared
        // buffer; larger ranks cannot take the in-kernel projection.
        if rank > 512 {
            return Ok(None);
        }
        // Per-head w_uv blocks are extracted from the raw kv_b weight storage;
        // quantized weights would need the dequant path first.
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

        // H2D only the small per-step query planes; the latent cache and the
        // w_vc source stay on their device.
        let Some(qa_all) = or_host_fallback(
            dev.from_cpu(&q_abs_scaled, &Shape::new(vec![nh, rank]), DType::F32),
        )?
        else {
            return Ok(None);
        };
        let Some(qr_all) = or_host_fallback(
            dev.from_cpu(&q_rope_scaled, &Shape::new(vec![nh, rope_d]), DType::F32),
        )?
        else {
            return Ok(None);
        };
        let out_shape = Shape::new(vec![1, nh * vd]);
        let Some(out_all) = or_host_fallback(dev.zeros(&out_shape, DType::F32))? else {
            return Ok(None);
        };

        let kv_st = latent_all.storage().as_ref();
        let w_src = self.kv_b_proj.weight.storage().as_ref();

        for h in 0..nh {
            let Some(qa_h) =
                or_host_fallback(dev.alloc_storage(&Shape::new(vec![rank]), DType::F32))?
            else {
                return Ok(None);
            };
            if or_host_fallback(dev.copy_slice_range(
                qa_h.as_ref(),
                0,
                qa_all.as_ref(),
                h * rank,
                rank,
            ))?
            .is_none()
            {
                return Ok(None);
            }

            let Some(qr_h) =
                or_host_fallback(dev.alloc_storage(&Shape::new(vec![rope_d]), DType::F32))?
            else {
                return Ok(None);
            };
            if or_host_fallback(dev.copy_slice_range(
                qr_h.as_ref(),
                0,
                qr_all.as_ref(),
                h * rope_d,
                rope_d,
            ))?
            .is_none()
            {
                return Ok(None);
            }

            // w_vc[h] = contiguous rows [h*(nope+v)+nope, +v) of kv_b_proj.weight.
            let Some(w_h) =
                or_host_fallback(dev.alloc_storage(&Shape::new(vec![vd, rank]), DType::F32))?
            else {
                return Ok(None);
            };
            if or_host_fallback(dev.copy_slice_range(
                w_h.as_ref(),
                0,
                w_src,
                (h * (nope + vd) + nope) * rank,
                vd * rank,
            ))?
            .is_none()
            {
                return Ok(None);
            }

            let Some(out_h) =
                or_host_fallback(dev.alloc_storage(&Shape::new(vec![vd]), DType::F32))?
            else {
                return Ok(None);
            };
            if or_host_fallback(dev.mla_absorbed_decode(
                qa_h.as_ref(),
                qr_h.as_ref(),
                kv_st,
                Some(w_h.as_ref()),
                out_h.as_ref(),
                1,
                rank,
                rope_d,
                vd,
                total_kv_len,
                0,
                0,
            ))?
            .is_none()
            {
                return Ok(None);
            }
            if or_host_fallback(dev.copy_slice_into(
                out_all.as_ref(),
                out_h.as_ref(),
                h * vd,
                vd,
            ))?
            .is_none()
            {
                return Ok(None);
            }
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

// ---------------------------------------------------------------------------
// MoE Feed-Forward Layer (Sqrt-Softplus Gating + Hyper-Connections)
// ---------------------------------------------------------------------------

pub struct DeepSeek4Expert {
    pub w1: Linear,
    pub w3: Linear,
    pub w2: Linear,
}

impl DeepSeek4Expert {
    pub fn load(
        ws: &WeightSource<'_>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let w1 = Linear::load_shape(&ws.scoped("w1"), [hidden_size, intermediate_size])?;
        let w3 = Linear::load_shape(&ws.scoped("w3"), [hidden_size, intermediate_size])?;
        let w2 = Linear::load_shape(&ws.scoped("w2"), [intermediate_size, hidden_size])?;
        Ok(Self { w1, w3, w2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;
        let swiglu_t = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        Ok(self.w2.forward(&swiglu_t)?)
    }
}

pub struct DeepSeek4Moe {
    pub gate: Linear,
    pub experts: Vec<DeepSeek4Expert>,
    pub shared_experts: Option<DeepSeek4Expert>,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
}

impl DeepSeek4Moe {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeepSeek4Config) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.n_routed_experts])?;

        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        let exp_ws = ws.scoped("experts");
        for e in 0..cfg.n_routed_experts {
            let exp = DeepSeek4Expert::load(
                &exp_ws.scoped(&e.to_string()),
                cfg.hidden_size,
                cfg.moe_intermediate_size,
            )?;
            experts.push(exp);
        }

        let shared_experts = if cfg.n_shared_experts > 0 {
            let shared_ws = ws.scoped("shared_experts");
            let exp = DeepSeek4Expert::load(
                &shared_ws,
                cfg.hidden_size,
                cfg.moe_intermediate_size * cfg.n_shared_experts,
            )?;
            Some(exp)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    /// GPU-first MoE forward: routing stays on host (small gate-logits pull),
    /// but on non-CPU devices the experts run on-device and the routing
    /// weighted sum accumulates with scalar-mul/add kernels so per-expert
    /// outputs never cross to host. Falls back to [`Self::forward_moe_host`]
    /// on CPU devices or when the backend lacks a needed primitive.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.gate.forward(x)?;
        let logits_v = logits.to_vec_f32()?;

        if x.device() != &Device::Cpu {
            if let Some(out) = self.forward_moe_device(x, &logits_v)? {
                return Ok(out);
            }
        }
        self.forward_moe_host(x, &logits_v)
    }

    /// Device-resident MoE: token rows are extracted D2D, experts run
    /// on-device (Linear + `silu_mul_on_device`), and the weighted sum
    /// accumulates on-device. `Ok(None)` = backend lacks a needed kernel
    /// (`is_kernel_unimplemented`); caller uses the host path.
    fn forward_moe_device(&self, x: &Tensor, logits_v: &[f32]) -> Result<Option<Tensor>> {
        let seq_len = x.shape().dims()[0];
        let hidden_dim = x.shape().dims()[1];
        let num_exp = self.experts.len();
        let dev = grim_nn::modules::pick_device_for_storage_device(x.device());

        let Some(out_st) = or_host_fallback(dev.zeros(x.shape(), DType::F32))? else {
            return Ok(None);
        };

        for s in 0..seq_len {
            let row = &logits_v[s * num_exp..(s + 1) * num_exp];
            // Sqrt-Softplus gating: s(x) = sqrt(softplus(x)) = sqrt(ln(1 + exp(x)))
            let mut indexed: Vec<(usize, f32)> = row
                .iter()
                .cloned()
                .enumerate()
                .map(|(idx, l)| {
                    let sp = if l > 20.0 { l } else { (1.0 + l.exp()).ln() };
                    (idx, sp.sqrt())
                })
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk = &indexed[..self.num_experts_per_tok.min(num_exp)];

            let sum_w: f32 = topk.iter().map(|(_, w)| *w).sum();
            let weights: Vec<f32> = topk
                .iter()
                .map(|(_, w)| (w / (sum_w + 1e-12)) * self.routed_scaling_factor)
                .collect();

            // Token row stays on-device (D2D extraction).
            let tok_shape = Shape::new(vec![1, hidden_dim]);
            let Some(token_st) = or_host_fallback(dev.alloc_storage(&tok_shape, DType::F32))?
            else {
                return Ok(None);
            };
            if or_host_fallback(dev.copy_slice_range(
                token_st.as_ref(),
                0,
                x.storage().as_ref(),
                s * hidden_dim,
                hidden_dim,
            ))?
            .is_none()
            {
                return Ok(None);
            }
            let token_x = Tensor::new(
                Arc::from(token_st),
                tok_shape,
                DType::F32,
                QuantProvenance::default(),
                x.device().clone(),
            );

            let mut acc: Option<Tensor> = None;
            for (i, (exp_idx, _)) in topk.iter().enumerate() {
                let w = weights[i];
                let exp_out = self.experts[*exp_idx].forward(&token_x)?;
                let Some((scaled_st, _h)) = or_host_fallback(dev.mul_scalar(
                    exp_out.storage().as_ref(),
                    w,
                    exp_out.shape(),
                ))?
                else {
                    return Ok(None);
                };
                let scaled = Tensor::new(
                    Arc::from(scaled_st),
                    exp_out.shape().clone(),
                    DType::F32,
                    QuantProvenance::default(),
                    exp_out.device().clone(),
                );
                acc = Some(match acc {
                    Some(a) => grim_nn::modules::add_on_device(&a, &scaled)?,
                    None => scaled,
                });
            }
            if let Some(acc) = acc {
                if or_host_fallback(dev.copy_slice_into(
                    out_st.as_ref(),
                    acc.storage().as_ref(),
                    s * hidden_dim,
                    hidden_dim,
                ))?
                .is_none()
                {
                    return Ok(None);
                }
            }
        }

        let mut out_t = Tensor::new(
            Arc::from(out_st),
            x.shape().clone(),
            DType::F32,
            QuantProvenance::default(),
            x.device().clone(),
        );

        if let Some(ref shared) = self.shared_experts {
            let sh_out = shared.forward(x)?;
            out_t = grim_nn::modules::add_on_device(&out_t, &sh_out)?;
        }

        Ok(Some(out_t))
    }

    /// Host routing reference path — the documented FALLBACK (CPU device, or
    /// GPU backends missing the copy/mul primitives). Identical math to the
    /// device path: per-token sqrt-softplus top-k routing, expert forward,
    /// routed-scaling weighted sum.
    fn forward_moe_host(&self, x: &Tensor, logits_v: &[f32]) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let hidden_dim = x.shape().dims()[1];
        let num_exp = self.experts.len();

        let xv = x.to_vec_f32()?;
        let mut out = vec![0.0f32; seq_len * hidden_dim];

        for s in 0..seq_len {
            let row = &logits_v[s * num_exp..(s + 1) * num_exp];
            // Sqrt-Softplus gating: s(x) = sqrt(softplus(x)) = sqrt(ln(1 + exp(x)))
            let mut indexed: Vec<(usize, f32)> = row
                .iter()
                .cloned()
                .enumerate()
                .map(|(idx, l)| {
                    let sp = if l > 20.0 { l } else { (1.0 + l.exp()).ln() };
                    (idx, sp.sqrt())
                })
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk = &indexed[..self.num_experts_per_tok.min(num_exp)];

            let sum_w: f32 = topk.iter().map(|(_, w)| *w).sum();
            let weights: Vec<f32> = topk
                .iter()
                .map(|(_, w)| (w / (sum_w + 1e-12)) * self.routed_scaling_factor)
                .collect();

            let token_x = cpu_tensor(
                xv[s * hidden_dim..(s + 1) * hidden_dim].to_vec(),
                Shape::new(vec![1, hidden_dim]),
            );

            for (i, (exp_idx, _)) in topk.iter().enumerate() {
                let w = weights[i];
                let exp_out = self.experts[*exp_idx].forward(&token_x)?.to_vec_f32()?;
                for d in 0..hidden_dim {
                    out[s * hidden_dim + d] += w * exp_out[d];
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

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct DeepSeek4Block {
    pub attn_norm: RmsNorm,
    pub self_attn: DeepSeek4Mla,
    pub ffn_norm: RmsNorm,
    pub mlp: Option<DeepSeek4Expert>,
    pub moe: Option<DeepSeek4Moe>,
    pub hc_mult: f32,
}

impl DeepSeek4Block {
    pub fn load(ws: &WeightSource<'_>, cfg: &DeepSeek4Config, is_dense: bool) -> Result<Self> {
        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let self_attn = DeepSeek4Mla::load(&ws.scoped("self_attn"), cfg)?;
        let ffn_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let (mlp, moe) = if is_dense {
            let mlp =
                DeepSeek4Expert::load(&ws.scoped("mlp"), cfg.hidden_size, cfg.intermediate_size)?;
            (Some(mlp), None)
        } else {
            let moe = DeepSeek4Moe::load(&ws.scoped("mlp"), cfg)?;
            (None, Some(moe))
        };

        Ok(Self {
            attn_norm,
            self_attn,
            ffn_norm,
            mlp,
            moe,
            hc_mult: cfg.hc_mult as f32,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let normed_attn = self.attn_norm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed_attn, positions, kv_cache)?;

        // Hyper-Connection residual scaling: res = x + hc_mult * attn_out,
        // dispatched on-device (scalar-mul + add kernels).
        let res1_t = grim_nn::modules::add_on_device(x, &mul_scalar_on_device(&attn_out, self.hc_mult)?)?;

        let normed_ffn = self.ffn_norm.forward(&res1_t)?;
        let mlp_out = if let Some(ref mlp) = self.mlp {
            mlp.forward(&normed_ffn)?
        } else if let Some(ref moe) = self.moe {
            moe.forward(&normed_ffn)?
        } else {
            normed_ffn.clone()
        };

        grim_nn::modules::add_on_device(&res1_t, &mul_scalar_on_device(&mlp_out, self.hc_mult)?)
            .map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model & Session
// ---------------------------------------------------------------------------

pub struct DeepSeek4 {
    pub cfg: DeepSeek4Config,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<DeepSeek4Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl DeepSeek4 {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DeepSeek4Config,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DeepSeek4Config,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let is_dense = i < cfg.first_k_dense_replace;
            let block = DeepSeek4Block::load(&layer_ws, &cfg, is_dense)?;
            layers.push(block);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
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

impl Model for DeepSeek4 {
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

impl CausalLm for DeepSeek4 {
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
    fn test_deepseek4_config() {
        let cfg = DeepSeek4Config::default();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.hc_mult, 4);
        assert!(cfg.sqrtsoftplus_moe);
    }
}
