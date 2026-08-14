//! Falcon-H1: hybrid Mamba-2 + GQA attention + SwiGLU FFN.
//!
//! Self-contained CPU implementation. Follows `llama.cpp` reference:
//! - `src/models/mamba-base.cpp` (build_mamba2_layer, lines 193–295)
//! - `src/models/falcon-h1.cpp` (FalconH1 model, lines 593–700)
//!
//! Differences from a plain transformer:
//! - `token_embd` is reused as the LM head (tied output).
//! - `ffn_norm.weight` has no `.weight` suffix in the GGUF — loaded as a bare
//!   tensor and wrapped in `RmsNorm` directly.
//! - RoPE pairing is NEOX (rotates `(i, i+half)` pairs), per `Rope::forward`
//!   in `grim-nn`.
//!
//! Pattern mirrors `lfm2.rs`: `FalconH1Config: ModelConfig`, `FalconH1Model:
//! Model + CausalLm`. The loader constructs the type via `FalconH1Model::load`.
use grim_backend_cpu::cpu_tensor;
use grim_core::Result;
use grim_core::model::{CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::{Inner, SessionT};
use grim_nn::modules::{
    Embedding, Linear, RmsNorm, Rope, pick_device_for_storage_device, require_single_device,
};
use grim_nn::{TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

/// Falcon-H1 model config — what the loader passes in.
///
/// Derived from the GGUF metadata of `NeVe-Cascade-S-90M-Q8_0.gguf`.
#[derive(Clone, Debug)]
pub struct FalconH1Config {
    pub vocab_size: usize,
    pub hidden_size: usize,       // 512
    pub num_heads: usize,         // 8
    pub num_kv_heads: usize,      // 2
    pub head_dim: usize,          // 64
    pub num_layers: usize,        // 24
    pub intermediate_size: usize, // 768
    pub rms_norm_eps: f32,
    pub rope_theta: f32, // 10000.0

    // SSM (Mamba-2) per-block hyper-params.
    pub ssm_d_state: usize, // 64
    pub ssm_d_inner: usize, // 768
    pub ssm_d_conv: usize,  // 4
    pub ssm_dt_rank: usize, // 24
    pub ssm_n_group: usize, // 1
}

impl FalconH1Config {
    pub fn head_group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
    /// Width of the `xBC` slice inside `ssm_in` output (= conv input width).
    pub fn ssm_conv_dim(&self) -> usize {
        self.ssm_d_inner + 2 * self.ssm_n_group * self.ssm_d_state
    }
    /// Width of `ssm_in` output: z(d_inner) + xBC(ssm_conv_dim) + dt(dt_rank).
    pub fn ssm_in_dim(&self) -> usize {
        self.ssm_d_inner + self.ssm_conv_dim() + self.ssm_dt_rank
    }
}

impl ModelConfig for FalconH1Config {
    fn name(&self) -> &str {
        "falcon-h1"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Per-layer state kept across decode steps.
pub struct FalconH1LayerCache {
    pub conv_state: Vec<f32>, // (d_conv-1) * ssm_conv_dim
    pub ssm_state: Vec<f32>,  // d_state * ssm_d_inner
    pub k_cache: Vec<f32>,    // past_tokens * (num_kv_heads * head_dim)
    pub v_cache: Vec<f32>,    // past_tokens * (num_kv_heads * head_dim)
    pub current_pos: usize,
}

impl FalconH1LayerCache {
    pub fn new(cfg: &FalconH1Config) -> Self {
        // conv_state holds (d_conv-1) past rows of the xBC slice, each
        // `ssm_conv_dim` wide — matches the conv buffer column width.
        let conv_state = vec![0.0f32; (cfg.ssm_d_conv - 1) * cfg.ssm_conv_dim()];
        let ssm_state = vec![0.0f32; cfg.ssm_d_state * cfg.ssm_d_inner];
        Self {
            conv_state,
            ssm_state,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            current_pos: 0,
        }
    }
}

/// One Falcon-H1 block.
pub struct FalconH1Block {
    pub attn_norm: RmsNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: RmsNorm,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub rope: Rope,

    pub ssm_in: Linear,
    pub ssm_out: Linear,
    pub ssm_conv_w: Tensor, // [ssm_d_inner, ssm_d_conv]
    pub ssm_conv_b: Tensor, // [ssm_d_inner]
    pub ssm_a: Tensor,      // [n_ssm_head]      — exp(-A * dt) per head
    pub ssm_d: Tensor,      // [n_ssm_head]      — D residual per head
    pub ssm_dt_b: Tensor,   // [n_ssm_head]      — dt bias
}

/// The Falcon-H1 model: embedding + per-block tensors + output norm + tied LM head.
pub struct FalconH1Model {
    pub cfg: FalconH1Config,
    pub device: Device,
    pub embedding: Embedding,
    pub blocks: Vec<FalconH1Block>,
    pub output_norm: RmsNorm,
    pub lm_head: Linear, // tied from `token_embd`
}

impl FalconH1Model {
    /// Construct a new model from a loaded weight source. Mirrors `Lfm2::load_tp`.
    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: FalconH1Config,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        require_single_device(tp, "FalconH1", "CPU-only scaffold")
            .map_err(grim_core::Error::Unimplemented)?;
        let embedding = Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for li in 0..cfg.num_layers {
            blocks.push(load_block(&ws.pp("blk").pp(&li.to_string()), &cfg)?);
        }
        let output_norm = RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let lm_head = Linear::from_tensor(embedding.weight.clone(), None);
        Ok(Self {
            cfg,
            device,
            embedding,
            blocks,
            output_norm,
            lm_head,
        })
    }
}

/// Load one block. GGUF tensor names inside `blk.{i}.` are:
/// `attn_norm.weight`, `ffn_norm` (bare weight, no `.weight` suffix),
/// `attn_q/wk/wv/wo`, `ffn_gate/up/down`, `ssm_in/out.weight`,
/// `ssm_conv1d.{weight,bias}`, `ssm_a`, `ssm_d`, `ssm_dt.bias`.
fn load_block(ws: &WeightSource<'_>, cfg: &FalconH1Config) -> Result<FalconH1Block> {
    let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
    let wq = Linear::load(
        &ws.pp("attn_q"),
        cfg.hidden_size,
        cfg.num_heads * cfg.head_dim,
        false,
    )?;
    let wk = Linear::load(
        &ws.pp("attn_k"),
        cfg.hidden_size,
        cfg.num_kv_heads * cfg.head_dim,
        false,
    )?;
    let wv = Linear::load(
        &ws.pp("attn_v"),
        cfg.hidden_size,
        cfg.num_kv_heads * cfg.head_dim,
        false,
    )?;
    let wo = Linear::load(
        &ws.pp("attn_output"),
        cfg.hidden_size,
        cfg.hidden_size,
        false,
    )?;

    // ffn_norm is a bare weight tensor (no `.weight` suffix in GGUF).
    let ffn_norm_weight = ws.get([cfg.hidden_size], "ffn_norm")?;
    let ffn_norm = RmsNorm {
        weight: ffn_norm_weight,
        eps: cfg.rms_norm_eps,
    };
    let w_gate = Linear::load(
        &ws.pp("ffn_gate"),
        cfg.hidden_size,
        cfg.intermediate_size,
        false,
    )?;
    let w_up = Linear::load(
        &ws.pp("ffn_up"),
        cfg.hidden_size,
        cfg.intermediate_size,
        false,
    )?;
    let w_down = Linear::load(
        &ws.pp("ffn_down"),
        cfg.intermediate_size,
        cfg.hidden_size,
        false,
    )?;

    let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

    let ssm_in = Linear::load(&ws.pp("ssm_in"), cfg.hidden_size, cfg.ssm_in_dim(), false)?;
    let ssm_out = Linear::load(&ws.pp("ssm_out"), cfg.ssm_d_inner, cfg.hidden_size, false)?;
    let ssm_conv_w = ws.get([cfg.ssm_conv_dim(), cfg.ssm_d_conv], "ssm_conv1d.weight")?;
    let ssm_conv_b = ws.get([cfg.ssm_conv_dim()], "ssm_conv1d.bias")?;
    // Scalar-per-head tensors arrive as `[n, 1]` → flatten to `n` in forward.
    let ssm_a = ws.get([cfg.ssm_dt_rank, 1], "ssm_a")?;
    let ssm_d = ws.get([cfg.ssm_dt_rank, 1], "ssm_d")?;
    let ssm_dt_b = ws.get([cfg.ssm_dt_rank], "ssm_dt.bias")?;

    Ok(FalconH1Block {
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        ffn_norm,
        w_gate,
        w_up,
        w_down,
        rope,
        ssm_in,
        ssm_out,
        ssm_conv_w,
        ssm_conv_b,
        ssm_a,
        ssm_d,
        ssm_dt_b,
    })
}

// ---- Trait impls ----------------------------------------------------------

impl Model for FalconH1Model {
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

impl CausalLm for FalconH1Model {
    fn new_session(&self) -> Box<dyn SessionT> {
        let caches: Vec<FalconH1LayerCache> = (0..self.cfg.num_layers)
            .map(|_| FalconH1LayerCache::new(&self.cfg))
            .collect();
        let mut session = Inner::new(self.device.clone());
        // Store caches in model_state; the trait forward downcasts them.
        session.set_model_state(Box::new(caches));
        Box::new(session)
    }
    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[grim_core::model::AdapterHandle],
    ) -> Result<Tensor> {
        let ids: Vec<u32> = match input_ids.dtype() {
            d if d == DType::F32 => input_ids
                .to_vec_f32()?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
            _ => {
                return Err(grim_tensor::Error::Unimplemented(
                    "FalconH1 only accepts F32 input ids".into(),
                )
                .into());
            }
        };
        let positions_vec: Vec<u32> = match positions.dtype() {
            d if d == DType::F32 => positions
                .to_vec_f32()?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
            _ => {
                return Err(grim_tensor::Error::Unimplemented(
                    "FalconH1 only accepts F32 positions".into(),
                )
                .into());
            }
        };
        let caches: &mut Vec<FalconH1LayerCache> = match session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<FalconH1LayerCache>>())
        {
            Some(c) => c,
            None => {
                let fresh: Vec<FalconH1LayerCache> = (0..self.cfg.num_layers)
                    .map(|_| FalconH1LayerCache::new(&self.cfg))
                    .collect();
                session.set_model_state(Box::new(fresh));
                session
                    .model_state_mut()
                    .and_then(|s| s.downcast_mut::<Vec<FalconH1LayerCache>>())
                    .expect("FalconH1::forward: model_state downcast after init")
            }
        };
        self.forward_cpu(caches, &ids, &positions_vec)
    }
}

impl FalconH1Model {
    /// CPU forward: `(input_ids, positions) -> logits [seq_len, vocab_size]`.
    pub fn forward_cpu(
        &self,
        caches: &mut Vec<FalconH1LayerCache>,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let seq_len = input_ids.len();
        let hidden = self
            .embedding
            .forward(input_ids, seq_len, cfg.hidden_size)?;
        let device = hidden.device().clone();
        let mut h = hidden.to_vec_f32()?;

        for (li, block) in self.blocks.iter().enumerate() {
            h = forward_block_cpu(&mut caches[li], block, cfg, &h, positions)?;
        }

        // output_norm + tied LM head
        let normed = self.output_norm.forward(&{
            let shape = Shape::new(vec![h.len() / cfg.hidden_size, cfg.hidden_size]);
            device_tensor(h.clone(), shape, &device)?
        })?;
        Ok(self.lm_head.forward(&normed)?)
    }
}

// ---- Forward math ---------------------------------------------------------

/// Block: `inpL -> inpSA = inpL + (attn_out + ssm_out) -> ffn(inpSA) + inpSA`.
fn forward_block_cpu(
    cache: &mut FalconH1LayerCache,
    b: &FalconH1Block,
    cfg: &FalconH1Config,
    h: &[f32],
    positions: &[u32],
) -> Result<Vec<f32>> {
    let seq_len = positions.len();
    let device = b.attn_norm.weight.device();

    let normed = b
        .attn_norm
        .forward(&wrap(h, seq_len, cfg.hidden_size, device)?)?;
    let normed = normed.to_vec_f32()?;

    let q =
        b.wq.forward(&wrap(&normed, seq_len, cfg.hidden_size, device)?)?
            .to_vec_f32()?;
    let k =
        b.wk.forward(&wrap(&normed, seq_len, cfg.hidden_size, device)?)?
            .to_vec_f32()?;
    let v =
        b.wv.forward(&wrap(&normed, seq_len, cfg.hidden_size, device)?)?
            .to_vec_f32()?;

    let attn_out = gqa_attn_with_cache(b, cfg, &q, &k, &v, positions, seq_len, cache)?;
    let ssm_out = mamba2_layer_cpu(b, cfg, &normed, cache)?;

    let mut inp_sa = vec![0.0f32; seq_len * cfg.hidden_size];
    let mut layer_out = vec![0.0f32; seq_len * cfg.hidden_size];
    for i in 0..(seq_len * cfg.hidden_size) {
        layer_out[i] = attn_out[i] + ssm_out[i];
        inp_sa[i] = h[i] + layer_out[i];
    }

    // FFN on inp_sa.
    let ffh = b
        .ffn_norm
        .forward(&wrap(&inp_sa, seq_len, cfg.hidden_size, device)?)?
        .to_vec_f32()?;
    let gate = b
        .w_gate
        .forward(&wrap(&ffh, seq_len, cfg.hidden_size, device)?)?
        .to_vec_f32()?;
    let up = b
        .w_up
        .forward(&wrap(&ffh, seq_len, cfg.hidden_size, device)?)?
        .to_vec_f32()?;
    let mut act = vec![0.0f32; gate.len()];
    for i in 0..gate.len() {
        act[i] = silu(gate[i]) * up[i];
    }
    let ffn_out = b
        .w_down
        .forward(&wrap(&act, seq_len, cfg.intermediate_size, device)?)?
        .to_vec_f32()?;
    let mut out = inp_sa;
    for i in 0..(seq_len * cfg.hidden_size) {
        out[i] += ffn_out[i];
    }
    Ok(out)
}

/// GQA attention with RoPE and a growing KV cache stored in `cache`.
fn gqa_attn_with_cache(
    b: &FalconH1Block,
    cfg: &FalconH1Config,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[u32],
    seq_len: usize,
    cache: &mut FalconH1LayerCache,
) -> Result<Vec<f32>> {
    let h_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv = cfg.num_kv_heads;
    let group = cfg.head_group();

    // RoPE via the Rope module (NEOX pairing, reshaped to (1, S*heads, head_dim)).
    let q_3d = q_reshaped(q, seq_len, n_heads, h_dim);
    let k_3d = q_reshaped(k, seq_len, n_kv, h_dim);
    let q_roped = b
        .rope
        .forward(&q_3d, &expand_positions(positions, n_heads))?;
    let k_roped = b.rope.forward(&k_3d, &expand_positions(positions, n_kv))?;
    let q_roped = q_roped.to_vec_f32()?;
    let k_roped = k_roped.to_vec_f32()?;

    cache.k_cache.extend_from_slice(&k_roped);
    cache.v_cache.extend_from_slice(v);
    let total = cache.k_cache.len() / (n_kv * h_dim);

    let scale = 1.0 / (h_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * n_heads * h_dim];

    for t in 0..seq_len {
        let q_pos_t = positions[t] as usize;
        for h in 0..n_heads {
            let kvh = h / group;
            let q_base = (t * n_heads + h) * h_dim;
            let mut max_logit = f32::NEG_INFINITY;
            let mut logits = vec![0.0f32; total];
            for t2 in 0..total {
                // KV position of slot t2 (monotonic, append-in-order).
                let kv_pos = cache.current_pos + t2;
                if kv_pos > q_pos_t {
                    logits[t2] = f32::NEG_INFINITY;
                } else {
                    let k_base = (t2 * n_kv + kvh) * h_dim;
                    let mut dot = 0.0f32;
                    for d in 0..h_dim {
                        dot += q_roped[q_base + d] * cache.k_cache[k_base + d];
                    }
                    let l = dot * scale;
                    logits[t2] = l;
                    if l > max_logit {
                        max_logit = l;
                    }
                }
            }
            let mut sum = 0.0f32;
            for l in logits.iter_mut() {
                if l.is_finite() {
                    *l = (*l - max_logit).exp();
                    sum += *l;
                } else {
                    *l = 0.0;
                }
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            let out_base = (t * n_heads + h) * h_dim;
            for t2 in 0..total {
                let w = logits[t2] * inv;
                let v_base = (t2 * n_kv + kvh) * h_dim;
                for d in 0..h_dim {
                    out[out_base + d] += w * cache.v_cache[v_base + d];
                }
            }
        }
    }
    Ok(out)
}

/// Faithful Mamba-2 selective scan forward.
///
/// h_normed: `[seq_len, hidden_size]` (768-wide ssm_d_inner, but input here is
/// `hidden_size=512` from attn_norm) → `ssm_in.forward` projects 512 → 1688.
fn mamba2_layer_cpu(
    b: &FalconH1Block,
    cfg: &FalconH1Config,
    h_normed: &[f32],
    cache: &mut FalconH1LayerCache,
) -> Result<Vec<f32>> {
    let seq_len = h_normed.len() / cfg.hidden_size;
    let device = b.attn_norm.weight.device();

    let in_dim = cfg.ssm_in_dim();
    let ssm = b
        .ssm_in
        .forward(&wrap(h_normed, seq_len, cfg.hidden_size, device)?)?
        .to_vec_f32()?;

    let conv_dim = cfg.ssm_conv_dim();
    let d_conv = cfg.ssm_d_conv;
    let d_state = cfg.ssm_d_state;
    let n_ssm_head = cfg.ssm_dt_rank;
    let head_dim_ssm = cfg.ssm_d_inner / n_ssm_head;

    // Buffer for conv: (d_conv-1) past rows + seq_len current xBC rows, width conv_dim.
    let mut buffer: Vec<f32> = Vec::with_capacity((d_conv - 1 + seq_len) * conv_dim);
    buffer.extend_from_slice(&cache.conv_state);
    for t in 0..seq_len {
        let src_base = t * in_dim + cfg.ssm_d_inner;
        buffer.extend_from_slice(&ssm[src_base..src_base + conv_dim]);
    }

    let conv_w = b.ssm_conv_w.to_vec_f32()?;
    let conv_b = b.ssm_conv_b.to_vec_f32()?;
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
    // conv_state <- last (d_conv-1) raw xBC rows.
    let new_len = (d_conv - 1) * conv_dim;
    let off = buffer.len() - new_len;
    cache.conv_state.clear();
    cache.conv_state.extend_from_slice(&buffer[off..]);

    // dt bias added, then softplus.
    let a_vec = b.ssm_a.to_vec_f32()?;
    let d_vec = b.ssm_d.to_vec_f32()?;
    let dt_b = b.ssm_dt_b.to_vec_f32()?;
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

    // swiglu: out = silu(z) * y, z = ssm_in[:, 0..ssm_d_inner]
    let mut swiglu = vec![0.0f32; seq_len * cfg.ssm_d_inner];
    for t in 0..seq_len {
        for i in 0..cfg.ssm_d_inner {
            swiglu[t * cfg.ssm_d_inner + i] =
                silu(ssm[t * in_dim + i]) * y[t * cfg.ssm_d_inner + i];
        }
    }

    let out = b
        .ssm_out
        .forward(&wrap(&swiglu, seq_len, cfg.ssm_d_inner, device)?)?
        .to_vec_f32()?;
    Ok(out)
}

// ---- scalar helpers -------------------------------------------------------

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn wrap(data: &[f32], seq_len: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let shape = Shape::new(vec![seq_len, dim]);
    Ok(device_tensor(data.to_vec(), shape, device)?)
}

fn device_tensor(data: Vec<f32>, shape: Shape, device: &Device) -> Result<Tensor> {
    if device == &Device::Cpu {
        return Ok(cpu_tensor(data, shape));
    }
    let dev = pick_device_for_storage_device(device);
    let storage = dev.from_cpu(&data, &shape, DType::F32)?;
    Ok(Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative.into(),
        device.clone(),
    ))
}

fn q_reshaped(x: &[f32], seq_len: usize, n_heads: usize, h_dim: usize) -> Tensor {
    cpu_tensor(x.to_vec(), Shape::new(vec![1, seq_len * n_heads, h_dim]))
}

fn expand_positions(positions: &[u32], repeats: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(positions.len() * repeats);
    for &p in positions {
        for _ in 0..repeats {
            out.push(p);
        }
    }
    out
}
