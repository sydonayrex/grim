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
    /// Device-resident key arena (shape `[cap, num_kv_heads * head_dim]`),
    /// valid rows `0..current_pos`. Grown geometrically by
    /// `block.rs::cache_append_kv`; host mirrors below only advance on the
    /// host fallback path.
    pub k_device: Option<Box<dyn grim_tensor::BackendStorage>>,
    /// Device-resident value arena, same layout as `k_device`.
    pub v_device: Option<Box<dyn grim_tensor::BackendStorage>>,
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
            k_device: None,
            v_device: None,
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
        let logits = self.forward_cpu(caches, &ids, &positions_vec)?;
        // Audit fix (grim-models): FalconH1 never advanced the session
        // position — the engine's decode start_pos stayed at 0, so every
        // decode token ran at RoPE position 0 while its KV cache grew.
        session.advance_pos(ids.len());
        Ok(logits)
    }
}

impl FalconH1Model {
    /// CPU forward: `(input_ids, positions) -> logits [seq_len, vocab_size]`.
    pub fn forward_cpu(
        &self,
        caches: &mut [FalconH1LayerCache],
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

    let q_t =
        b.wq.forward(&wrap(&normed, seq_len, cfg.hidden_size, device)?)?;
    let k_t =
        b.wk.forward(&wrap(&normed, seq_len, cfg.hidden_size, device)?)?;
    let v_t =
        b.wv.forward(&wrap(&normed, seq_len, cfg.hidden_size, device)?)?;

    let attn_tensor =
        gqa_attn_with_cache(b, cfg, &q_t, &k_t, &v_t, positions, seq_len, cache)?;
    let attn_out = attn_tensor.to_vec_f32()?;
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

/// GQA attention with RoPE and a growing KV cache. Q/K/V arrive as device
/// tensors from `wq`/`wk`/`wv` and never round-trip through the host on the
/// primary path: RoPE stays on-device, new K/V rows are appended to the
/// device arena D2D (`block.rs::cache_append_kv`), and the fused attention
/// kernel reads the arena. Host mirrors + host attention remain the
/// `Unimplemented` fallback for backends without the device primitives.
fn gqa_attn_with_cache(
    b: &FalconH1Block,
    cfg: &FalconH1Config,
    q_t: &Tensor,
    k_t: &Tensor,
    v_t: &Tensor,
    positions: &[u32],
    seq_len: usize,
    cache: &mut FalconH1LayerCache,
) -> Result<Tensor> {
    let h_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv = cfg.num_kv_heads;
    let device = b.attn_norm.weight.device();

    // RoPE via the Rope module (NEOX pairing, reshaped to (1, S*heads, head_dim)).
    let q_3d = crate::block::reshaped_view(
        q_t,
        &Shape::new(vec![1, seq_len * n_heads, h_dim]),
    )?;
    let k_3d = crate::block::reshaped_view(
        k_t,
        &Shape::new(vec![1, seq_len * n_kv, h_dim]),
    )?;
    let q_roped = b
        .rope
        .forward(&q_3d, &expand_positions(positions, n_heads))?;
    let k_roped = b.rope.forward(&k_3d, &expand_positions(positions, n_kv))?;

    let dev = pick_device_for_storage_device(device);
    let row_elems = n_kv * h_dim;

    match crate::block::cache_append_kv(
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
                Ok((st, _h)) => Ok(Tensor::new(
                    std::sync::Arc::from(st),
                    out_shape,
                    DType::F32,
                    grim_tensor::QuantProvenance::default(),
                    device.clone(),
                )),
                // Arena already holds this step's rows — fetch them to host
                // once for the scalar kernel instead of re-extending mirrors.
                // The arena buffer is capacity-sized; only `total` rows are
                // valid, and the scalar kernel derives kv_len from the vec
                // length — truncate.
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
                        device,
                    )
                }
            }
        }
        Err(
            grim_core::Error::Unimplemented(_)
            | grim_core::Error::Tensor(grim_tensor::Error::Unimplemented(_)),
        ) => {
            // Legacy host-mirror path: extend the host caches and run the
            // scalar attention loop.
            let q_roped_v = q_roped.to_vec_f32()?;
            let k_roped_v = k_roped.to_vec_f32()?;
            let v_v = v_t.to_vec_f32()?;
            cache.k_cache.extend_from_slice(&k_roped_v);
            cache.v_cache.extend_from_slice(&v_v);
            cache.current_pos += seq_len;
            crate::shared_attention::fused_or_scalar_attention(
                &q_roped_v,
                &cache.k_cache,
                &cache.v_cache,
                n_heads,
                n_kv,
                h_dim,
                seq_len,
                None,
                device,
            )
        }
        Err(e) => Err(e),
    }
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
    let ssm_t = b.ssm_in.forward(&wrap(h_normed, seq_len, cfg.hidden_size, device)?)?;
    let ssm = ssm_t.to_vec_f32()?;

    let conv_dim = cfg.ssm_conv_dim();
    let d_conv = cfg.ssm_d_conv;
    let d_state = cfg.ssm_d_state;
    let n_ssm_head = cfg.ssm_dt_rank;
    let head_dim_ssm = cfg.ssm_d_inner / n_ssm_head;

    // xBC rows (post-conv + silu), width conv_dim. Decode (seq == 1) runs the
    // `short_conv1d_causal_step` kernel on device; prefill keeps the host
    // loop (per-token kernel launches lose at prefill lengths).
    let mut xbc = vec![0.0f32; seq_len * conv_dim];
    let mut conv_done = false;
    if seq_len == 1 {
        match ssm_conv_step_device(b, cfg, &ssm_t, &ssm, cache, device) {
            Ok(row) => {
                xbc[..conv_dim].copy_from_slice(&row);
                conv_done = true;
            }
            Err(ConvStepError::Unimplemented) => {}
            Err(ConvStepError::Fatal(e)) => return Err(e),
        }
    }
    if !conv_done {
        // Buffer for conv: (d_conv-1) past rows + seq_len current xBC rows, width conv_dim.
        let mut buffer: Vec<f32> = Vec::with_capacity((d_conv - 1 + seq_len) * conv_dim);
        buffer.extend_from_slice(&cache.conv_state);
        for t in 0..seq_len {
            let src_base = t * in_dim + cfg.ssm_d_inner;
            buffer.extend_from_slice(&ssm[src_base..src_base + conv_dim]);
        }

        let conv_w = b.ssm_conv_w.to_vec_f32()?;
        let conv_b = b.ssm_conv_b.to_vec_f32()?;
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
    }

    // dt bias added, then softplus.
    let a_vec = b.ssm_a.to_vec_f32()?;
    let d_vec = b.ssm_d.to_vec_f32()?;
    let dt_b = b.ssm_dt_b.to_vec_f32()?;
    let dt_src = cfg.ssm_d_inner + conv_dim;
    let mut y = vec![0.0f32; seq_len * cfg.ssm_d_inner];

    // WI-D dispatch: decode-step scan via `selective_scan_headed` when the
    // backend wires it. Until then every backend hits Unimplemented and the
    // host loop below runs — the always-correct fallback.
    let mut scan_done = false;
    if seq_len == 1 {
        match ssm_scan_step_device(
            cfg, &xbc, &ssm, &a_vec, &d_vec, &dt_b, cache, device,
        ) {
            Ok(Some(y_row)) => {
                y[..cfg.ssm_d_inner].copy_from_slice(&y_row);
                scan_done = true;
            }
            Ok(None) => {}
            Err(e) => return Err(e),
        }
    }
    if !scan_done {
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

/// WI-D dispatch: one decode-step scan via `BackendDevice::selective_scan_headed`.
///
/// Builds the kernel inputs from the host row slices (x / B / C from the
/// post-conv xBC row, dt post-softplus), uploads state, and syncs the updated
/// state back into the host mirror. Returns `Ok(None)` when the backend does
/// not implement the kernel yet — every backend today — so the host loop
/// stays the single source of truth until real kernel impls land.
#[allow(clippy::too_many_arguments)]
fn ssm_scan_step_device(
    cfg: &FalconH1Config,
    xbc_row: &[f32],
    ssm_row: &[f32],
    a_vec: &[f32],
    d_vec: &[f32],
    dt_b: &[f32],
    cache: &mut FalconH1LayerCache,
    device: &Device,
) -> Result<Option<Vec<f32>>> {
    let d_inner = cfg.ssm_d_inner;
    let d_state = cfg.ssm_d_state;
    let n_heads = cfg.ssm_dt_rank;
    let head_dim_ssm = d_inner / n_heads;
    let dt_src = d_inner + cfg.ssm_conv_dim();
    let dev = pick_device_for_storage_device(device);
    let mut inner = || -> Result<Vec<f32>> {
        // dt post-softplus (per head, tiny — host math).
        let dt: Vec<f32> = (0..n_heads)
            .map(|h| (1.0_f32 + (ssm_row[dt_src + h] + dt_b[h]).exp()).ln())
            .collect();
        let flat = |v: &[f32]| v.to_vec();
        let x_st = dev.from_cpu(
            &flat(&xbc_row[..d_inner]),
            &Shape::new(vec![d_inner]),
            DType::F32,
        )?;
        let dt_st = dev.from_cpu(&dt, &Shape::new(vec![n_heads]), DType::F32)?;
        let a_st = dev.from_cpu(&flat(a_vec), &Shape::new(vec![n_heads]), DType::F32)?;
        let d_st = dev.from_cpu(&flat(d_vec), &Shape::new(vec![n_heads]), DType::F32)?;
        let b_st = dev.from_cpu(
            &flat(&xbc_row[d_inner..d_inner + d_state]),
            &Shape::new(vec![d_state]),
            DType::F32,
        )?;
        let c_st = dev.from_cpu(
            &flat(&xbc_row[d_inner + d_state..d_inner + 2 * d_state]),
            &Shape::new(vec![d_state]),
            DType::F32,
        )?;
        let state_st = dev.from_cpu(
            &cache.ssm_state,
            &Shape::new(vec![d_inner * d_state]),
            DType::F32,
        )?;

        let (y_st, _h) = dev.selective_scan_headed(
            x_st.as_ref(),
            dt_st.as_ref(),
            a_st.as_ref(),
            b_st.as_ref(),
            c_st.as_ref(),
            d_st.as_ref(),
            state_st.as_ref(),
            n_heads,
            d_state,
            head_dim_ssm,
            &Shape::new(vec![d_inner]),
        )?;
        // State was updated in place — sync the host mirror.
        let mut state_new = state_st.to_cpu_vec_f32()?;
        state_new.truncate(d_inner * d_state);
        cache.ssm_state.copy_from_slice(&state_new);
        let mut y = y_st.to_cpu_vec_f32()?;
        y.truncate(d_inner);
        Ok(y)
    };
    match inner() {
        Ok(y) => Ok(Some(y)),
        Err(
            grim_core::Error::Unimplemented(_)
            | grim_core::Error::Tensor(grim_tensor::Error::Unimplemented(_)),
        ) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Decode-step conv via the `short_conv1d_causal_step` kernel.
///
/// Slices the current xBC row off the `ssm_in` output on device (columns
/// `[d_inner, d_inner+conv_dim)` are one contiguous flat range), convolves
/// against the device weight/bias, and returns the post-silu row. The scan
/// itself stays host-side at this stage, so the result crosses D2H once.
///
/// Layout note: `cache.conv_state` is time-major (`[t][d]`); the kernel wants
/// channel-major (`[d][t]`) — rearranged per call (host data, no round-trip).
/// The host mirror still slides here; the device arena lands in WI-C.
enum ConvStepError {
    Unimplemented,
    Fatal(grim_core::Error),
}

fn ssm_conv_step_device(
    b: &FalconH1Block,
    cfg: &FalconH1Config,
    ssm_t: &Tensor,
    ssm_row: &[f32],
    cache: &mut FalconH1LayerCache,
    device: &Device,
) -> std::result::Result<Vec<f32>, ConvStepError> {
    let conv_dim = cfg.ssm_conv_dim();
    let d_conv = cfg.ssm_d_conv;
    let kc = d_conv - 1;
    let dev = pick_device_for_storage_device(device);
    let mut inner = || -> Result<Vec<f32>> {
        let ssm_st = ssm_t.storage().as_ref();
        let xbc_st = dev.alloc_storage(&Shape::new(vec![conv_dim]), DType::F32)?;
        dev.copy_slice_range(
            xbc_st.as_ref(),
            0,
            ssm_st,
            cfg.ssm_d_inner,
            conv_dim,
        )?;

        let mut state_cm = vec![0.0f32; kc * conv_dim];
        for t in 0..kc {
            for d in 0..conv_dim {
                state_cm[d * kc + t] = cache.conv_state[t * conv_dim + d];
            }
        }
        let state_st =
            dev.from_cpu(&state_cm, &Shape::new(vec![kc * conv_dim]), DType::F32)?;

        let (out_st, _h) = dev.short_conv1d_causal_step(
            xbc_st.as_ref(),
            b.ssm_conv_w.storage().as_ref(),
            Some(b.ssm_conv_b.storage().as_ref()),
            state_st.as_ref(),
            &Shape::new(vec![conv_dim]),
        )?;
        let mut sums = out_st.to_cpu_vec_f32()?;
        sums.truncate(conv_dim);
        for v in sums.iter_mut() {
            *v = silu(*v);
        }

        // Slide the host mirror: drop the oldest raw row, append the current
        // raw xBC row (pre-conv, pre-silu — matches the prefill path).
        cache.conv_state.copy_within(conv_dim.., 0);
        let tail = cache.conv_state.len() - conv_dim;
        cache.conv_state[tail..]
            .copy_from_slice(&ssm_row[cfg.ssm_d_inner..cfg.ssm_d_inner + conv_dim]);
        Ok(sums)
    };
    match inner() {
        Ok(v) => Ok(v),
        Err(
            grim_core::Error::Unimplemented(_)
            | grim_core::Error::Tensor(grim_tensor::Error::Unimplemented(_)),
        ) => Err(ConvStepError::Unimplemented),
        Err(e) => Err(ConvStepError::Fatal(e)),
    }
}

// ---- scalar helpers -------------------------------------------------------

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn wrap(data: &[f32], seq_len: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let shape = Shape::new(vec![seq_len, dim]);
    device_tensor(data.to_vec(), shape, device)
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
        grim_tensor::QuantProvenance::GrimNative,
        device.clone(),
    ))
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

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::provider::{RawTensor, TensorMeta};
    use std::collections::HashMap;

    /// Deterministic LCG so parity failures are reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|i| self.next_f32() * scale + i as f32 * 1e-3).collect()
        }
    }

    struct MemProvider(HashMap<String, (Vec<u8>, Vec<usize>)>);
    impl grim_tensor::TensorProvider for MemProvider {
        fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
            let (bytes, shape) = self.0.get(name).cloned().ok_or_else(|| {
                grim_tensor::error::Error::Backend(format!("missing: {name}"))
            })?;
            Ok(RawTensor {
                bytes,
                shape,
                dtype: DType::F32,
                provenance: grim_tensor::QuantProvenance::GrimNative,
            })
        }
        fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
            let (_, shape) = self.0.get(name).cloned().ok_or_else(|| {
                grim_tensor::error::Error::Backend(format!("missing: {name}"))
            })?;
            Ok(TensorMeta {
                dtype: DType::F32,
                provenance: grim_tensor::QuantProvenance::GrimNative,
                shape,
                fusion_mask: 0,
            })
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    fn tiny_cfg() -> FalconH1Config {
        FalconH1Config {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            num_layers: 1,
            intermediate_size: 48,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            ssm_d_state: 8,
            ssm_d_inner: 24,
            ssm_d_conv: 4,
            ssm_dt_rank: 4,
            ssm_n_group: 1,
        }
    }

    fn make_provider(cfg: &FalconH1Config, rng: &mut Lcg) -> MemProvider {
        let hs = cfg.hidden_size;
        let mut m: HashMap<String, (Vec<u8>, Vec<usize>)> = HashMap::new();
        let put = |m: &mut HashMap<_, _>, name: &str, shape: Vec<usize>, rng: &mut Lcg| {
            let n: usize = shape.iter().product();
            m.insert(name.to_string(), (f32_bytes(&rng.vec(n, 0.3)), shape));
        };
        put(&mut m, "attn_norm.weight", vec![hs], rng);
        put(&mut m, "ffn_norm", vec![hs], rng);
        put(&mut m, "attn_q.weight", vec![cfg.num_heads * cfg.head_dim, hs], rng);
        put(&mut m, "attn_k.weight", vec![cfg.num_kv_heads * cfg.head_dim, hs], rng);
        put(&mut m, "attn_v.weight", vec![cfg.num_kv_heads * cfg.head_dim, hs], rng);
        put(&mut m, "attn_output.weight", vec![hs, hs], rng);
        put(&mut m, "ffn_gate.weight", vec![cfg.intermediate_size, hs], rng);
        put(&mut m, "ffn_up.weight", vec![cfg.intermediate_size, hs], rng);
        put(&mut m, "ffn_down.weight", vec![hs, cfg.intermediate_size], rng);
        put(&mut m, "ssm_in.weight", vec![cfg.ssm_in_dim(), hs], rng);
        put(&mut m, "ssm_out.weight", vec![hs, cfg.ssm_d_inner], rng);
        let conv_dim = cfg.ssm_conv_dim();
        put(&mut m, "ssm_conv1d.weight", vec![conv_dim, cfg.ssm_d_conv], rng);
        put(&mut m, "ssm_conv1d.bias", vec![conv_dim], rng);
        put(&mut m, "ssm_a", vec![cfg.ssm_dt_rank, 1], rng);
        put(&mut m, "ssm_d", vec![cfg.ssm_dt_rank, 1], rng);
        put(&mut m, "ssm_dt.bias", vec![cfg.ssm_dt_rank], rng);
        MemProvider(m)
    }

    fn small_block() -> (FalconH1Block, FalconH1Config) {
        let cfg = tiny_cfg();
        let provider = make_provider(&cfg, &mut Lcg(0xF00D));
        let ws = WeightSource::root(&provider, Device::Cpu);
        let block = load_block(&ws, &cfg).expect("load_block");
        (block, cfg)
    }

    /// Host-reference attention: rope on device, then pull Q/K to host, grow
    /// host caches, run the scalar kernel. Mirrors the pre-WI-A algorithm.
    fn ref_attn_step(
        b: &FalconH1Block,
        cfg: &FalconH1Config,
        q_t: &Tensor,
        k_t: &Tensor,
        v_t: &Tensor,
        positions: &[u32],
        seq_len: usize,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
    ) -> Vec<f32> {
        let q_3d = crate::block::reshaped_view(
            q_t,
            &Shape::new(vec![1, seq_len * cfg.num_heads, cfg.head_dim]),
        )
        .unwrap();
        let k_3d = crate::block::reshaped_view(
            k_t,
            &Shape::new(vec![1, seq_len * cfg.num_kv_heads, cfg.head_dim]),
        )
        .unwrap();
        let q_r = b
            .rope
            .forward(&q_3d, &expand_positions(positions, cfg.num_heads))
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let k_r = b
            .rope
            .forward(&k_3d, &expand_positions(positions, cfg.num_kv_heads))
            .unwrap()
            .to_vec_f32()
            .unwrap();
        k_cache.extend_from_slice(&k_r);
        v_cache.extend_from_slice(&v_t.to_vec_f32().unwrap());
        crate::shared_attention::fused_or_scalar_attention(
            &q_r,
            k_cache,
            v_cache,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            seq_len,
            None,
            &Device::Cpu,
        )
        .unwrap()
        .to_vec_f32()
        .unwrap()
    }

    /// WI-A gate: 8-step decode through the device-arena path must match the
    /// host-reference algorithm (rope + host cache + scalar attention) to
    /// atol 1e-5, and the arena must actually be populated.
    #[test]
    fn test_gqa_arena_matches_host_reference() {
        let (block, cfg) = small_block();
        let hs = cfg.hidden_size;
        let mut rng = Lcg(0xBEEF);

        let mut arena_cache = FalconH1LayerCache::new(&cfg);
        let mut ref_k: Vec<f32> = Vec::new();
        let mut ref_v: Vec<f32> = Vec::new();

        for step in 0..8u32 {
            let h: Vec<f32> = rng.vec(hs, 0.5);
            let normed = block
                .attn_norm
                .forward(&wrap(&h, 1, hs, &Device::Cpu).unwrap())
                .unwrap();
            let q_t = block.wq.forward(&normed).unwrap();
            let k_t = block.wk.forward(&normed).unwrap();
            let v_t = block.wv.forward(&normed).unwrap();

            let device_out = gqa_attn_with_cache(
                &block, &cfg, &q_t, &k_t, &v_t, &[step], 1, &mut arena_cache,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap();
            let ref_out = ref_attn_step(
                &block, &cfg, &q_t, &k_t, &v_t, &[step], 1, &mut ref_k, &mut ref_v,
            );

            assert_eq!(device_out.len(), ref_out.len());
            for (i, (a, b)) in device_out.iter().zip(&ref_out).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "step {step} elem {i}: device {a} vs ref {b}"
                );
            }
        }

        // Arena populated and position tracking matches the reference cache.
        assert!(arena_cache.k_device.is_some());
        assert!(arena_cache.v_device.is_some());
        assert_eq!(arena_cache.current_pos, 8);
        assert_eq!(ref_k.len(), 8 * cfg.num_kv_heads * cfg.head_dim);
    }

    /// WI-B gate: single-token decode (device `short_conv1d_causal_step`)
    /// must match multi-token prefill (host conv loop) — causal equivalence
    /// of the conv + scan stack, including state evolution.
    #[test]
    fn test_ssm_decode_matches_prefill() {
        let (block, cfg) = small_block();
        let hs = cfg.hidden_size;
        let seq = 3;
        let mut rng = Lcg(0x5EED);
        let h: Vec<f32> = rng.vec(seq * hs, 0.5);

        // Path A: one prefill call (host conv loop).
        let mut cache_a = FalconH1LayerCache::new(&cfg);
        let out_a = mamba2_layer_cpu(&block, &cfg, &h, &mut cache_a).unwrap();

        // Path B: three decode calls (device conv kernel at seq == 1).
        let mut cache_b = FalconH1LayerCache::new(&cfg);
        let mut out_b = Vec::new();
        for t in 0..seq {
            out_b.extend(
                mamba2_layer_cpu(&block, &cfg, &h[t * hs..(t + 1) * hs], &mut cache_b)
                    .unwrap(),
            );
        }

        assert_eq!(out_a.len(), out_b.len());
        // 1e-4 (not tighter): prefill matmuls run [3,in] batches vs [1,in]
        // per decode step — fp32 accumulation order differs by ~1 ulp and
        // propagates through conv + scan.
        for (i, (a, b)) in out_a.iter().zip(&out_b).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "token {i}: prefill {a} vs decode {b}"
            );
        }
        for (i, (a, b)) in cache_a.conv_state.iter().zip(&cache_b.conv_state).enumerate() {
            assert!((a - b).abs() < 1e-4, "conv_state[{i}]: {a} vs {b}");
        }
        for (i, (a, b)) in cache_a.ssm_state.iter().zip(&cache_b.ssm_state).enumerate() {
            assert!((a - b).abs() < 1e-4, "ssm_state[{i}]: {a} vs {b}");
        }
    }

    /// WI-A guard: on the device path the host mirrors must stay empty —
    /// if the fallback branch ever runs silently on a kernel-capable backend,
    /// mirrors would grow while the arena also grows.
    #[test]
    fn test_gqa_device_path_leaves_mirrors_empty() {
        let (block, cfg) = small_block();
        let hs = cfg.hidden_size;
        let mut rng = Lcg(0xCAFE);

        let mut cache = FalconH1LayerCache::new(&cfg);
        for step in 0..4u32 {
            let h: Vec<f32> = rng.vec(hs, 0.5);
            let normed = block
                .attn_norm
                .forward(&wrap(&h, 1, hs, &Device::Cpu).unwrap())
                .unwrap();
            let q_t = block.wq.forward(&normed).unwrap();
            let k_t = block.wk.forward(&normed).unwrap();
            let v_t = block.wv.forward(&normed).unwrap();
            gqa_attn_with_cache(&block, &cfg, &q_t, &k_t, &v_t, &[step], 1, &mut cache)
                .unwrap();
        }
        assert!(cache.k_device.is_some(), "arena must be allocated");
        assert!(
            cache.k_cache.is_empty() && cache.v_cache.is_empty(),
            "host mirrors must stay empty on the device path"
        );
        assert_eq!(cache.current_pos, 4);
    }
}
