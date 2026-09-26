//! NVIDIA Nemotron 3.5 Lightning 30B A3B (nemotron_h_moe / nemotron_h).
//! Hybrid Mamba-2 + periodic GQA Attention + MoE (128 routed + 1 shared expert) architecture.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::{Inner, SessionT};
use grim_nn::modules::{Embedding, Linear, RmsNorm, Rope, pick_device_for_storage_device};
use grim_nn::moe::{MoeRouter, NonGatedExpertBank, NonGatedSharedExpert, RouterKind};
use grim_nn::{TensorParallelConfig, WeightSource};
use grim_tensor::shape::Shape;
use grim_tensor::{ArithType, DType, Device, Tensor};

/// Layer block type in Nemotron 3.5 Lightning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronHBlockType {
    Mamba,
    MoE,
    Attention,
}

/// Configuration for Nemotron 3.5 Lightning (nemotron_h_moe).
#[derive(Debug, Clone)]
pub struct NemotronHMoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,

    // Layer schedule: 53 layers (52 base + 1 MTP).
    pub layers_block_type: Vec<NemotronHBlockType>,

    // Attention parameters
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize, // GGUF rope.dimension_count (e.g. 84, or 0 if un-rotated)

    // Mamba-2 SSM parameters
    pub ssm_d_state: usize, // 128
    pub ssm_d_inner: usize, // 4096
    pub ssm_d_conv: usize,  // 4
    pub ssm_dt_rank: usize, // 64
    pub ssm_n_group: usize, // 8

    // MoE parameters
    pub num_routed_experts: usize,                // 128
    pub num_experts_per_tok: usize,               // 6
    pub expert_feed_forward_length: usize,        // 1856
    pub expert_shared_feed_forward_length: usize, // 3712
    pub expert_weights_scale: f32,                // 2.5
    pub expert_weights_norm: bool,                // true
}

impl NemotronHMoeConfig {
    /// Width of the `xBC` slice inside `ssm_in` output (= conv input width: inner + 2 * groups * state).
    pub fn ssm_conv_dim(&self) -> usize {
        self.ssm_d_inner + 2 * self.ssm_n_group * self.ssm_d_state
    }

    /// Total width of `ssm_in` output: inner (gate) + conv_dim (xBC) + dt_rank (dt).
    pub fn ssm_in_dim(&self) -> usize {
        self.ssm_d_inner + self.ssm_conv_dim() + self.ssm_dt_rank
    }

    /// Derive default 53-block schedule from attention layers:
    /// Attention at layers 5, 12, 19, 26, 33, 42, 52; remainder interleaves Mamba and MoE.
    pub fn default_schedule_53() -> Vec<NemotronHBlockType> {
        let attn_indices = [5, 12, 19, 26, 33, 42, 52];
        let mut schedule = Vec::with_capacity(53);
        let mut toggle = false;
        for i in 0..53 {
            if attn_indices.contains(&i) {
                schedule.push(NemotronHBlockType::Attention);
            } else {
                if toggle {
                    schedule.push(NemotronHBlockType::MoE);
                } else {
                    schedule.push(NemotronHBlockType::Mamba);
                }
                toggle = !toggle;
            }
        }
        schedule
    }

    /// Helper to derive schedule from head_count_kv array or tensor presence.
    pub fn schedule_from_kv_heads(
        kv_heads: &[u32],
        total_layers: usize,
    ) -> Vec<NemotronHBlockType> {
        let mut schedule = Vec::with_capacity(total_layers);
        let mut toggle = false;
        for i in 0..total_layers {
            if let Some(&k) = kv_heads.get(i) {
                if k > 0 {
                    schedule.push(NemotronHBlockType::Attention);
                    continue;
                }
            }
            if toggle {
                schedule.push(NemotronHBlockType::MoE);
            } else {
                schedule.push(NemotronHBlockType::Mamba);
            }
            toggle = !toggle;
        }
        schedule
    }
}

impl Default for NemotronHMoeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 2688,
            num_layers: 53,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 4096,
            layers_block_type: Self::default_schedule_53(),
            num_attention_heads: 32,
            num_key_value_heads: 2,
            head_dim: 128,
            rope_dim: 84,
            ssm_d_state: 128,
            ssm_d_inner: 4096,
            ssm_d_conv: 4,
            ssm_dt_rank: 64,
            ssm_n_group: 8,
            num_routed_experts: 128,
            num_experts_per_tok: 6,
            expert_feed_forward_length: 1856,
            expert_shared_feed_forward_length: 3712,
            expert_weights_scale: 2.5,
            expert_weights_norm: true,
        }
    }
}

impl ModelConfig for NemotronHMoeConfig {
    fn name(&self) -> &str {
        "nemotron_h_moe"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ── Per-layer Session State Cache ──────────────────────────────────────────

/// Recurrent / KV state per layer across decode steps.
pub struct NemotronHLayerCache {
    pub conv_state: Vec<f32>, // (d_conv-1) * ssm_conv_dim
    pub ssm_state: Vec<f32>,  // d_state * ssm_d_inner
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub k_device: Option<Box<dyn grim_tensor::BackendStorage>>,
    pub v_device: Option<Box<dyn grim_tensor::BackendStorage>>,
    pub current_pos: usize,
}

impl NemotronHLayerCache {
    pub fn new(cfg: &NemotronHMoeConfig) -> Self {
        let conv_state = vec![0.0f32; (cfg.ssm_d_conv - 1) * cfg.ssm_conv_dim()];
        let ssm_state = vec![0.0f32; cfg.ssm_d_state * cfg.ssm_d_inner];
        Self {
            conv_state,
            ssm_state,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            k_device: None,
            v_device: None,
            current_pos: 0,
        }
    }
}

// ── Layers ─────────────────────────────────────────────────────────────────

pub struct NemotronHMambaLayer {
    pub ssm_in: Linear,
    pub ssm_out: Linear,
    pub ssm_conv_w: Tensor,
    pub ssm_conv_b: Tensor,
    pub ssm_a: Tensor,
    pub ssm_d: Tensor,
    pub ssm_dt_b: Tensor,
    pub ssm_norm: Tensor, // [8, 512] or [512, 8]
}

impl NemotronHMambaLayer {
    pub fn load(ws: &WeightSource<'_>, cfg: &NemotronHMoeConfig) -> Result<Self> {
        let ssm_in = Linear::load(&ws.pp("ssm_in"), cfg.hidden_size, cfg.ssm_in_dim(), false)?;
        let ssm_out = Linear::load(&ws.pp("ssm_out"), cfg.ssm_d_inner, cfg.hidden_size, false)?;
        let ssm_conv_w = ws.get([cfg.ssm_conv_dim(), cfg.ssm_d_conv], "ssm_conv1d.weight")?;
        let ssm_conv_b = ws.get([cfg.ssm_conv_dim()], "ssm_conv1d.bias")?;
        let ssm_a = ws.get([cfg.ssm_dt_rank, 1], "ssm_a")?;
        let ssm_d = ws.get([cfg.ssm_dt_rank, 1], "ssm_d")?;
        let ssm_dt_b = ws.get([cfg.ssm_dt_rank], "ssm_dt.bias")?;

        let group_size = cfg.ssm_d_inner / cfg.ssm_n_group;
        let ssm_norm = ws
            .get([cfg.ssm_n_group, group_size], "ssm_norm.weight")
            .or_else(|_| ws.get([group_size, cfg.ssm_n_group], "ssm_norm.weight"))?;

        Ok(Self {
            ssm_in,
            ssm_out,
            ssm_conv_w,
            ssm_conv_b,
            ssm_a,
            ssm_d,
            ssm_dt_b,
            ssm_norm,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cfg: &NemotronHMoeConfig,
        cache: &mut NemotronHLayerCache,
        _device: &Device,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dim(0)?;
        let ssm_proj = self.ssm_in.forward(x)?;
        let ssm = ssm_proj.to_vec_f32()?;
        let in_dim = cfg.ssm_in_dim();
        let conv_dim = cfg.ssm_conv_dim();
        let d_conv = cfg.ssm_d_conv;
        let d_state = cfg.ssm_d_state;
        let n_ssm_head = cfg.ssm_dt_rank;
        let head_dim_ssm = cfg.ssm_d_inner / n_ssm_head;

        // Conv1D with state buffer
        let mut buffer = Vec::with_capacity((d_conv - 1 + seq_len) * conv_dim);
        buffer.extend_from_slice(&cache.conv_state);
        for t in 0..seq_len {
            let src_base = t * in_dim + cfg.ssm_d_inner;
            buffer.extend_from_slice(&ssm[src_base..src_base + conv_dim]);
        }

        let conv_w = self.ssm_conv_w.to_vec_f32()?;
        let conv_b = self.ssm_conv_b.to_vec_f32()?;
        let mut xbc = vec![0.0f32; seq_len * conv_dim];
        for t in 0..seq_len {
            for i1 in 0..conv_dim {
                let mut s = conv_b[i1];
                for i0 in 0..d_conv {
                    s += conv_w[i1 * d_conv + i0] * buffer[(t + i0) * conv_dim + i1];
                }
                xbc[t * conv_dim + i1] = silu(s);
            }
        }

        let new_len = (d_conv - 1) * conv_dim;
        let off = buffer.len() - new_len;
        cache.conv_state.clear();
        cache.conv_state.extend_from_slice(&buffer[off..]);

        // Selective Scan
        let a_vec = self.ssm_a.to_vec_f32()?;
        let d_vec = self.ssm_d.to_vec_f32()?;
        let dt_b = self.ssm_dt_b.to_vec_f32()?;
        let dt_src = cfg.ssm_d_inner + conv_dim;
        let mut y = vec![0.0f32; seq_len * cfg.ssm_d_inner];

        for t in 0..seq_len {
            let mut dt = vec![0.0f32; n_ssm_head];
            for h in 0..n_ssm_head {
                dt[h] = (1.0_f32 + (ssm[t * in_dim + dt_src + h] + dt_b[h]).exp()).ln();
            }
            let xb = &xbc[t * conv_dim..];
            for h in 0..n_ssm_head {
                let d_a = (dt[h] * a_vec[h]).exp();
                for j in 0..head_dim_ssm {
                    let i_dim = j + h * head_dim_ssm;
                    let x_dt = xb[i_dim] * dt[h];
                    let so = i_dim * d_state;
                    let mut acc = 0.0f32;
                    for k in 0..d_state {
                        let bk = xb[cfg.ssm_d_inner + k];
                        let s_new = cache.ssm_state[so + k] * d_a + bk * x_dt;
                        cache.ssm_state[so + k] = s_new;
                        let ck = xb[cfg.ssm_d_inner + d_state + k];
                        acc += s_new * ck;
                    }
                    y[t * cfg.ssm_d_inner + i_dim] = acc + d_vec[h] * x_dt;
                }
            }
        }

        // Zamba2 Gated Group RMSNorm:
        // y' = y * SiLU(gate)
        // normalized per group (n_groups=8, group_size=512) and multiplied by ssm_norm.weight
        let group_size = cfg.ssm_d_inner / cfg.ssm_n_group;
        let norm_w = self.ssm_norm.to_vec_f32()?;
        let mut gated_normed = vec![0.0f32; seq_len * cfg.ssm_d_inner];

        for t in 0..seq_len {
            let z_row = &ssm[t * in_dim..t * in_dim + cfg.ssm_d_inner];
            let y_row = &y[t * cfg.ssm_d_inner..(t + 1) * cfg.ssm_d_inner];

            for g in 0..cfg.ssm_n_group {
                let start = g * group_size;
                let end = start + group_size;
                let mut ss = 0.0f32;
                for i in start..end {
                    let val = y_row[i] * silu(z_row[i]);
                    gated_normed[t * cfg.ssm_d_inner + i] = val;
                    ss += val * val;
                }
                let rms = (ss / group_size as f32 + cfg.rms_norm_eps).sqrt();
                let inv_rms = 1.0 / rms;

                for (j, i) in (start..end).enumerate() {
                    let w = norm_w.get(g * group_size + j).copied().unwrap_or(1.0);
                    gated_normed[t * cfg.ssm_d_inner + i] =
                        gated_normed[t * cfg.ssm_d_inner + i] * inv_rms * w;
                }
            }
        }

        let normed_t = cpu_tensor(gated_normed, Shape::new(vec![seq_len, cfg.ssm_d_inner]));
        let out = self.ssm_out.forward(&normed_t)?;
        Ok(out)
    }
}

pub struct NemotronHAttentionLayer {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub rope: Option<Rope>,
}

impl NemotronHAttentionLayer {
    pub fn load(ws: &WeightSource<'_>, cfg: &NemotronHMoeConfig) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let wq = Linear::load(&ws.pp("attn_q"), cfg.hidden_size, q_dim, false)?;
        let wk = Linear::load(&ws.pp("attn_k"), cfg.hidden_size, kv_dim, false)?;
        let wv = Linear::load(&ws.pp("attn_v"), cfg.hidden_size, kv_dim, false)?;
        let wo = Linear::load(&ws.pp("attn_output"), q_dim, cfg.hidden_size, false)?;

        let rope = if cfg.rope_dim > 0 {
            Some(Rope::new(cfg.head_dim, cfg.rope_theta))
        } else {
            None
        };

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            rope,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cfg: &NemotronHMoeConfig,
        cache: &mut NemotronHLayerCache,
        positions: &[u32],
        device: &Device,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dim(0)?;
        let h_dim = cfg.head_dim;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;

        let q_t = self.wq.forward(x)?;
        let k_t = self.wk.forward(x)?;
        let v_t = self.wv.forward(x)?;

        let (q_roped, k_roped) = if let Some(rope) = &self.rope {
            let q_3d =
                crate::block::reshaped_view(&q_t, &Shape::new(vec![1, seq_len * n_heads, h_dim]))?;
            let k_3d =
                crate::block::reshaped_view(&k_t, &Shape::new(vec![1, seq_len * n_kv, h_dim]))?;
            let qr = rope.forward(&q_3d, &expand_positions(positions, n_heads))?;
            let kr = rope.forward(&k_3d, &expand_positions(positions, n_kv))?;
            (qr, kr)
        } else {
            (q_t, k_t)
        };

        let dev = pick_device_for_storage_device(device);
        let row_elems = n_kv * h_dim;

        let attn_out = match crate::block::cache_append_kv(
            dev.as_ref(),
            &mut cache.k_device,
            &mut cache.v_device,
            k_roped.storage().as_ref(),
            v_t.storage().as_ref(),
            cache.current_pos,
            seq_len,
            row_elems,
        ) {
            Ok((k_st, v_st, total)) => {
                cache.current_pos = total;
                let out_shape = Shape::new(vec![seq_len, n_heads * h_dim]);
                match dev.qkv_attention(
                    q_roped.storage().as_ref(),
                    k_st,
                    v_st,
                    n_kv,
                    total,
                    (total - seq_len) as u32,
                    None,
                    &out_shape,
                    None,
                    None,
                ) {
                    Ok((st, _h)) => Tensor::new(
                        std::sync::Arc::from(st),
                        out_shape,
                        DType::F32,
                        grim_tensor::QuantProvenance::default(),
                        device.clone(),
                    ),
                    Err(_) => {
                        let mut hk = k_st.to_cpu_vec_f32()?;
                        hk.truncate(total * row_elems);
                        let mut hv = v_st.to_cpu_vec_f32()?;
                        hv.truncate(total * row_elems);
                        crate::shared_attention::fused_or_scalar_attention(
                            &q_roped.to_vec_f32()?,
                            &hk,
                            &hv,
                            n_heads,
                            n_kv,
                            h_dim,
                            seq_len,
                            None,
                            &Device::Cpu,
                        )?
                    }
                }
            }
            Err(_) => {
                let k_vec = k_roped.to_vec_f32()?;
                let v_vec = v_t.to_vec_f32()?;
                cache.k_cache.extend_from_slice(&k_vec);
                cache.v_cache.extend_from_slice(&v_vec);
                cache.current_pos += seq_len;
                crate::shared_attention::fused_or_scalar_attention(
                    &q_roped.to_vec_f32()?,
                    &cache.k_cache,
                    &cache.v_cache,
                    n_heads,
                    n_kv,
                    h_dim,
                    seq_len,
                    None,
                    &Device::Cpu,
                )?
            }
        };

        Ok(self.wo.forward(&attn_out)?)
    }
}

pub struct NemotronHMoELayer {
    pub router: MoeRouter,
    pub experts: NonGatedExpertBank,
    pub shared_expert: Option<NonGatedSharedExpert>,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
}

impl NemotronHMoELayer {
    pub fn load(ws: &WeightSource<'_>, cfg: &NemotronHMoeConfig) -> Result<Self> {
        let gate = Linear::load(
            &ws.pp("ffn_gate_inp"),
            cfg.hidden_size,
            cfg.num_routed_experts,
            false,
        )?;
        let correction_bias = ws.get([cfg.num_routed_experts], "exp_probs_b.bias").ok();

        let router = MoeRouter::new(
            gate,
            RouterKind::SigmoidTopKWithBias,
            cfg.num_experts_per_tok,
            cfg.num_routed_experts,
            correction_bias,
        );

        let experts = NonGatedExpertBank::load(
            ws,
            cfg.num_routed_experts,
            cfg.hidden_size,
            cfg.expert_feed_forward_length,
            false,
        )?;

        let shared_expert = NonGatedSharedExpert::load(
            ws,
            cfg.hidden_size,
            cfg.expert_shared_feed_forward_length,
            false,
        )
        .ok();

        Ok(Self {
            router,
            experts,
            shared_expert,
            expert_weights_scale: cfg.expert_weights_scale,
            expert_weights_norm: cfg.expert_weights_norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (indices, raw_weights) = self.router.route(x)?;
        let batch = indices.len();
        let hidden = x.shape().dims().last().copied().unwrap_or(0);

        let mut out_vec = vec![0.0f32; batch * hidden];

        for t in 0..batch {
            let experts = &indices[t];
            let mut w = raw_weights[t].clone();

            if self.expert_weights_norm {
                let sum: f32 = w.iter().sum();
                if sum > 0.0 {
                    for val in w.iter_mut() {
                        *val /= sum;
                    }
                }
            }

            let xt = slice_row(x, t)?;
            let mut routed = vec![0.0f32; hidden];

            for (rank, &e) in experts.iter().enumerate() {
                let y = self.experts.expert_forward(e, &xt)?;
                let yv = y.to_vec_f32()?;
                for (i, v) in yv.iter().enumerate() {
                    routed[i] += w[rank] * v;
                }
            }

            for (i, v) in routed.iter().enumerate() {
                out_vec[t * hidden + i] += self.expert_weights_scale * v;
            }

            if let Some(sh) = &self.shared_expert {
                let s = sh.forward(&xt)?;
                let sv = s.to_vec_f32()?;
                for (i, v) in sv.iter().enumerate() {
                    out_vec[t * hidden + i] += v;
                }
            }
        }

        Ok(cpu_tensor(out_vec, Shape::new(vec![batch, hidden])))
    }
}

pub enum NemotronHMixer {
    Mamba(NemotronHMambaLayer),
    Attention(NemotronHAttentionLayer),
    MoE(NemotronHMoELayer),
}

pub struct NemotronHBlock {
    pub attn_norm: RmsNorm,
    pub mixer: NemotronHMixer,
    pub block_type: NemotronHBlockType,
}

impl NemotronHBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &NemotronHMoeConfig,
        b_type: NemotronHBlockType,
    ) -> Result<Self> {
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let mixer = match b_type {
            NemotronHBlockType::Mamba => NemotronHMixer::Mamba(NemotronHMambaLayer::load(ws, cfg)?),
            NemotronHBlockType::Attention => {
                NemotronHMixer::Attention(NemotronHAttentionLayer::load(ws, cfg)?)
            }
            NemotronHBlockType::MoE => NemotronHMixer::MoE(NemotronHMoELayer::load(ws, cfg)?),
        };
        Ok(Self {
            attn_norm,
            mixer,
            block_type: b_type,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cfg: &NemotronHMoeConfig,
        cache: &mut NemotronHLayerCache,
        positions: &[u32],
        device: &Device,
    ) -> Result<Tensor> {
        let normed = self.attn_norm.forward(x)?;
        let mixer_out = match &self.mixer {
            NemotronHMixer::Mamba(m) => m.forward(&normed, cfg, cache, device)?,
            NemotronHMixer::Attention(a) => a.forward(&normed, cfg, cache, positions, device)?,
            NemotronHMixer::MoE(moe) => moe.forward(&normed)?,
        };

        let xv = x.to_vec_f32()?;
        let mv = mixer_out.to_vec_f32()?;
        let mut out = xv;
        for (o, m) in out.iter_mut().zip(mv.iter()) {
            *o += m;
        }
        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

// ── Model ──────────────────────────────────────────────────────────────────

pub struct NemotronHMoe {
    pub cfg: NemotronHMoeConfig,
    pub device: Device,
    pub embedding: Embedding,
    pub blocks: Vec<NemotronHBlock>,
    pub output_norm: RmsNorm,
    pub lm_head: Linear,
}

impl NemotronHMoe {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: NemotronHMoeConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: NemotronHMoeConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let embedding = Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let output_norm = RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let lm_head = Linear::load(&ws.pp("output"), cfg.hidden_size, cfg.vocab_size, false)?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let b_type = cfg
                .layers_block_type
                .get(i)
                .copied()
                .unwrap_or(NemotronHBlockType::Mamba);
            let b_ws = ws.pp(&format!("blk.{i}"));
            blocks.push(NemotronHBlock::load(&b_ws, &cfg, b_type)?);
        }

        Ok(Self {
            cfg,
            device,
            embedding,
            blocks,
            output_norm,
            lm_head,
        })
    }

    pub fn forward_cpu(
        &self,
        caches: &mut [NemotronHLayerCache],
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let mut h = self
            .embedding
            .forward(input_ids, seq_len, self.cfg.hidden_size)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let cache = &mut caches[i];
            h = block.forward(&h, &self.cfg, cache, positions, &self.device)?;
        }

        let normed = self.output_norm.forward(&h)?;
        Ok(self.lm_head.forward(&normed)?)
    }
}

impl Model for NemotronHMoe {
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

impl CausalLm for NemotronHMoe {
    fn new_session(&self) -> Box<dyn SessionT> {
        let caches: Vec<NemotronHLayerCache> = (0..self.cfg.num_layers)
            .map(|_| NemotronHLayerCache::new(&self.cfg))
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
        let ids: Vec<u32> = input_ids.to_vec_u32()?;
        let positions_vec: Vec<u32> = positions.to_vec_u32()?;
        let caches: &mut Vec<NemotronHLayerCache> = match session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<NemotronHLayerCache>>())
        {
            Some(c) => c,
            None => {
                let fresh: Vec<NemotronHLayerCache> = (0..self.cfg.num_layers)
                    .map(|_| NemotronHLayerCache::new(&self.cfg))
                    .collect();
                session.set_model_state(Box::new(fresh));
                session
                    .model_state_mut()
                    .and_then(|s| s.downcast_mut::<Vec<NemotronHLayerCache>>())
                    .ok_or_else(|| {
                        grim_core::error::Error::Backend(
                            "NemotronHMoe::forward: model_state downcast after init".into(),
                        )
                    })?
            }
        };

        let logits = self.forward_cpu(caches, &ids, &positions_vec)?;
        session.advance_pos(ids.len());
        Ok(logits)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn expand_positions(positions: &[u32], heads: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(positions.len() * heads);
    for &p in positions {
        for _ in 0..heads {
            out.push(p);
        }
    }
    out
}

fn slice_row(x: &Tensor, row: usize) -> Result<Tensor> {
    let hidden = x.shape().dims().last().copied().unwrap_or(0);
    let v = x.to_vec_f32()?;
    let start = row * hidden;
    let end = start + hidden;
    Ok(cpu_tensor(
        v[start..end].to_vec(),
        Shape::new(vec![1, hidden]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nemotron_schedule_defaults() {
        let schedule = NemotronHMoeConfig::default_schedule_53();
        assert_eq!(schedule.len(), 53);
        // Attention layers must be at 5, 12, 19, 26, 33, 42, 52
        let expected_attn = [5, 12, 19, 26, 33, 42, 52];
        for (i, &block_type) in schedule.iter().enumerate() {
            if expected_attn.contains(&i) {
                assert_eq!(
                    block_type,
                    NemotronHBlockType::Attention,
                    "layer {i} should be Attention"
                );
            }
        }
    }

    #[test]
    fn test_nemotron_schedule_from_kv_heads() {
        let mut kv_heads = vec![0u32; 53];
        for &idx in &[5, 12, 19, 26, 33, 42, 52] {
            kv_heads[idx] = 2;
        }
        let schedule = NemotronHMoeConfig::schedule_from_kv_heads(&kv_heads, 53);
        assert_eq!(schedule.len(), 53);
        assert_eq!(schedule[5], NemotronHBlockType::Attention);
        assert_eq!(schedule[12], NemotronHBlockType::Attention);
        assert_eq!(schedule[52], NemotronHBlockType::Attention);
        assert_ne!(schedule[0], NemotronHBlockType::Attention);
    }

    #[test]
    fn test_nemotron_relu2_non_gated_expert() {
        // Test relu(u)^2 activation
        let up = vec![2.0f32, -3.0f32, 0.5f32];
        let relu2: Vec<f32> = up
            .into_iter()
            .map(|val| {
                let r = val.max(0.0);
                r * r
            })
            .collect();
        assert_eq!(relu2, vec![4.0f32, 0.0f32, 0.25f32]);
    }

    #[test]
    fn test_nemotron_hmoe_synthetic_forward() {
        // Construct a tiny synthetic 3-layer NemotronHMoe (Mamba, MoE, Attention)
        let cfg = NemotronHMoeConfig {
            vocab_size: 64,
            hidden_size: 16,
            num_layers: 3,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 32,
            layers_block_type: vec![
                NemotronHBlockType::Mamba,
                NemotronHBlockType::MoE,
                NemotronHBlockType::Attention,
            ],
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rope_dim: 4,
            ssm_d_state: 8,
            ssm_d_inner: 16,
            ssm_d_conv: 4,
            ssm_dt_rank: 4,
            ssm_n_group: 2,
            num_routed_experts: 4,
            num_experts_per_tok: 2,
            expert_feed_forward_length: 8,
            expert_shared_feed_forward_length: 8,
            expert_weights_scale: 2.5,
            expert_weights_norm: true,
        };

        let dev = Device::Cpu;
        let mut session = Inner::new(dev.clone());
        let caches: Vec<NemotronHLayerCache> = (0..cfg.num_layers)
            .map(|_| NemotronHLayerCache::new(&cfg))
            .collect();
        session.set_model_state(Box::new(caches));

        // Create synthetic weights
        let emb = Embedding {
            weight: cpu_tensor(vec![0.01f32; 64 * 16], Shape::new(vec![64, 16])),
        };
        let out_norm = RmsNorm::new(cpu_tensor(vec![1.0f32; 16], Shape::new(vec![16])), 1e-5);
        let lm_head = Linear::from_tensor(
            cpu_tensor(vec![0.01f32; 64 * 16], Shape::new(vec![64, 16])),
            None,
        );

        let mut blocks = Vec::with_capacity(3);

        // Layer 0: Mamba
        let mamba = NemotronHMambaLayer {
            ssm_in: Linear::from_tensor(
                cpu_tensor(
                    vec![0.01f32; cfg.ssm_in_dim() * 16],
                    Shape::new(vec![cfg.ssm_in_dim(), 16]),
                ),
                None,
            ),
            ssm_out: Linear::from_tensor(
                cpu_tensor(vec![0.01f32; 16 * 16], Shape::new(vec![16, 16])),
                None,
            ),
            ssm_conv_w: cpu_tensor(
                vec![0.01f32; cfg.ssm_conv_dim() * 4],
                Shape::new(vec![cfg.ssm_conv_dim(), 4]),
            ),
            ssm_conv_b: cpu_tensor(
                vec![0.0f32; cfg.ssm_conv_dim()],
                Shape::new(vec![cfg.ssm_conv_dim()]),
            ),
            ssm_a: cpu_tensor(vec![-1.0f32; 4], Shape::new(vec![4, 1])),
            ssm_d: cpu_tensor(vec![1.0f32; 4], Shape::new(vec![4, 1])),
            ssm_dt_b: cpu_tensor(vec![0.0f32; 4], Shape::new(vec![4])),
            ssm_norm: cpu_tensor(vec![1.0f32; 16], Shape::new(vec![2, 8])),
        };
        blocks.push(NemotronHBlock {
            attn_norm: RmsNorm::new(cpu_tensor(vec![1.0f32; 16], Shape::new(vec![16])), 1e-5),
            mixer: NemotronHMixer::Mamba(mamba),
            block_type: NemotronHBlockType::Mamba,
        });

        // Layer 1: MoE
        let gate = Linear::from_tensor(
            cpu_tensor(vec![0.1f32; 4 * 16], Shape::new(vec![4, 16])),
            None,
        );
        let router = MoeRouter::new(gate, RouterKind::SigmoidTopKWithBias, 2, 4, None);
        let up_lins: Vec<Linear> = (0..4)
            .map(|_| {
                Linear::from_tensor(
                    cpu_tensor(vec![0.05f32; 8 * 16], Shape::new(vec![8, 16])),
                    None,
                )
            })
            .collect();
        let down_lins: Vec<Linear> = (0..4)
            .map(|_| {
                Linear::from_tensor(
                    cpu_tensor(vec![0.05f32; 16 * 8], Shape::new(vec![16, 8])),
                    None,
                )
            })
            .collect();
        let experts = NonGatedExpertBank::from_linears(up_lins, down_lins);
        let shared = NonGatedSharedExpert {
            up: Linear::from_tensor(
                cpu_tensor(vec![0.05f32; 8 * 16], Shape::new(vec![8, 16])),
                None,
            ),
            down: Linear::from_tensor(
                cpu_tensor(vec![0.05f32; 16 * 8], Shape::new(vec![16, 8])),
                None,
            ),
        };
        let moe = NemotronHMoELayer {
            router,
            experts,
            shared_expert: Some(shared),
            expert_weights_scale: 2.5,
            expert_weights_norm: true,
        };
        blocks.push(NemotronHBlock {
            attn_norm: RmsNorm::new(cpu_tensor(vec![1.0f32; 16], Shape::new(vec![16])), 1e-5),
            mixer: NemotronHMixer::MoE(moe),
            block_type: NemotronHBlockType::MoE,
        });

        // Layer 2: Attention
        let attn = NemotronHAttentionLayer {
            wq: Linear::from_tensor(
                cpu_tensor(vec![0.05f32; 16 * 16], Shape::new(vec![16, 16])),
                None,
            ),
            wk: Linear::from_tensor(
                cpu_tensor(vec![0.05f32; 8 * 16], Shape::new(vec![8, 16])),
                None,
            ),
            wv: Linear::from_tensor(
                cpu_tensor(vec![0.05f32; 8 * 16], Shape::new(vec![8, 16])),
                None,
            ),
            wo: Linear::from_tensor(
                cpu_tensor(vec![0.05f32; 16 * 16], Shape::new(vec![16, 16])),
                None,
            ),
            rope: Some(Rope::new(4, 10000.0)),
        };
        blocks.push(NemotronHBlock {
            attn_norm: RmsNorm::new(cpu_tensor(vec![1.0f32; 16], Shape::new(vec![16])), 1e-5),
            mixer: NemotronHMixer::Attention(attn),
            block_type: NemotronHBlockType::Attention,
        });

        let model = NemotronHMoe {
            cfg,
            device: dev,
            embedding: emb,
            blocks,
            output_norm: out_norm,
            lm_head,
        };

        let input_ids = cpu_tensor(vec![1.0f32, 5.0f32, 10.0f32], Shape::new(vec![3]));
        let positions = cpu_tensor(vec![0.0f32, 1.0f32, 2.0f32], Shape::new(vec![3]));

        let logits = model
            .forward(&mut session, &input_ids, &positions, &[])
            .expect("forward pass should succeed");
        assert_eq!(logits.shape().dims(), &[3, 64]);
        let logits_vec = logits.to_vec_f32().expect("logits to vec");
        assert_eq!(logits_vec.len(), 3 * 64);
        assert!(!logits_vec.iter().any(|v| v.is_nan() || v.is_infinite()));
    }
}
