//! Mamba / Mamba2 state-space model (SSM) and RWKV stateful sequence architectures.

use std::any::Any;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{
    AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig, SsmState, StatefulSequence,
};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

pub mod configs;
pub mod mamba2;
pub mod rwkv;

pub use configs::*;
pub use mamba2::{Mamba2, Mamba2Block, Mamba2State};
pub use rwkv::{Rwkv, RwkvConfig};

#[derive(Debug, Clone)]
pub struct MambaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub d_conv: usize,
    pub num_layers: usize,
    pub conv_kernel: usize,
    pub rms_norm_eps: f32,
}

impl ModelConfig for MambaConfig {
    fn name(&self) -> &str {
        "mamba"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Debug)]
pub struct MambaState {
    /// `(d_inner, d_state)` per batch row. Flattened as one big vec; layout
    /// `[batch, d_inner * d_state]`.
    pub h: Vec<f32>,
    pub batch: usize,
    pub d_inner: usize,
    pub d_state: usize,
    /// Tokens already advanced (pos cursor). Cheap to snapshot for
    /// speculative-decode rollback (§5.3).
    pub pos: usize,
}

impl MambaState {
    pub fn new(batch: usize, d_inner: usize, d_state: usize) -> Self {
        Self {
            h: vec![0.0; batch * d_state * d_inner],
            batch,
            d_inner,
            d_state,
            pos: 0,
        }
    }
}

impl SsmState for MambaState {
    fn clone_snapshot(&self) -> Result<Box<dyn SsmState>> {
        Ok(Box::new(self.clone()))
    }
    fn restore_snapshot(&mut self, snap: &dyn SsmState) -> Result<()> {
        let other = snap
            .as_any()
            .downcast_ref::<MambaState>()
            .ok_or_else(|| Error::Session("snapshot downcast failed".into()))?;
        if self.batch != other.batch
            || self.d_inner != other.d_inner
            || self.d_state != other.d_state
        {
            return Err(Error::Session("snapshot shape mismatch".into()));
        }
        self.h.copy_from_slice(&other.h);
        self.pos = other.pos;
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// One Mamba block: pre-norm → in_proj → conv1d (skipped in v1) →
/// selective SSM scan → out_proj.
#[derive(Clone)]
pub struct MambaBlock {
    pub norm: RmsNorm,
    pub in_proj: Linear,
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub b_param: Vec<f32>,
    pub d_param: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub out_proj: Linear,
    pub d_state: usize,
    pub d_inner: usize,
    pub d_conv: usize,
    pub device: Device,
}

impl MambaBlock {
    pub fn random(cfg: &MambaConfig, rng: &mut grim_core::rng::SimpleRng) -> Self {
        let in_proj_weight: Vec<f32> = (0..(2 * cfg.d_inner) * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let in_proj = Linear::from_tensor(
            cpu_tensor(
                in_proj_weight,
                Shape::new(vec![2 * cfg.d_inner, cfg.hidden_size]),
            ),
            None,
        );
        let out_proj_weight: Vec<f32> = (0..cfg.hidden_size * cfg.d_inner)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let out_proj = Linear::from_tensor(
            cpu_tensor(
                out_proj_weight,
                Shape::new(vec![cfg.hidden_size, cfg.d_inner]),
            ),
            None,
        );
        let conv: Vec<f32> = (0..cfg.d_inner * cfg.conv_kernel)
            .map(|_| (rng.next_f32() - 0.5) * 0.5)
            .collect();
        let a_log: Vec<f32> = (0..cfg.d_inner * cfg.d_state)
            .map(|_| (rng.next_f32() - 0.5) * 0.5)
            .collect();
        let b_param: Vec<f32> = (0..cfg.d_inner * cfg.d_state)
            .map(|_| (rng.next_f32() - 0.5) * 0.5)
            .collect();
        let d_param: Vec<f32> = (0..cfg.d_inner).map(|_| 1.0).collect();
        let dt_bias: Vec<f32> = (0..cfg.d_inner).map(|_| 0.0).collect();
        Self {
            norm: RmsNorm {
                weight: cpu_tensor(
                    vec![1.0; cfg.hidden_size],
                    Shape::new(vec![cfg.hidden_size]),
                ),
                eps: cfg.rms_norm_eps,
            },
            in_proj,
            conv,
            a_log,
            b_param,
            d_param,
            dt_bias,
            out_proj,
            d_state: cfg.d_state,
            d_inner: cfg.d_inner,
            d_conv: cfg.d_conv,
            device: Device::Cpu,
        }
    }

    /// Forward one step using existing state. Selective scan updated in place.
    ///
    /// When `self.device` is `Device::Rocm`, dispatches to the JIT-compiled
    /// `grim_selective_scan` HIP kernel (Phase 2 — mambo5.md Item 11).
    /// Falls back to the CPU scan loop for `Device::Cpu`.
    pub fn step_block(&self, x: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        // GPU dispatch path: Mamba selective scan HIP kernel.
        if let Device::Rocm(ordinal) = self.device {
            #[cfg(feature = "rocm")]
            {
                if let Ok(result) = self.step_block_gpu(x, state, ordinal) {
                    return Ok(result);
                }
            }
            #[cfg(not(feature = "rocm"))]
            {
                let _ = ordinal;
            }
            // Fall through to CPU fallback on any GPU dispatch failure.
        }
        self.step_block_cpu(x, state)
    }

    /// GPU dispatch path for Mamba selective scan via `BackendDevice::selective_scan`.
    #[cfg(feature = "rocm")]
    fn step_block_gpu(&self, x: &Tensor, state: &mut MambaState, ordinal: usize) -> Result<Tensor> {
        use grim_backend_rocm::RocmDevice;
        use grim_tensor::BackendDevice;

        let dev = RocmDevice::try_new(ordinal)?;
        let h_in = x.shape().dims().last().copied().unwrap_or(0);
        let xd = x.to_vec_f32()?;
        if xd.is_empty() {
            return Err(Error::Shape("empty Mamba input".into()));
        }
        if self.b_param.is_empty() {
            // MOD-1 fix: `b_param` (the SSM B matrix) must never be aliased to
            // `a_log` (the A log-weights). Substitifying them produces a
            // completely different, silent recurrence. Fail loudly instead.
            return Err(Error::Unimplemented(
                "Mamba step_block_gpu: b_param is empty; refusing to alias a_log as B".into(),
            ));
        }
        // MOD-4 fix: take the first `h_in` elements (batch=1 token) instead of
        // replicating `xd[0]` across the whole vector.
        let x_flat: Vec<f32> = xd.iter().take(h_in).copied().collect();

        // Upload input, weights, and SSM params to GPU.
        let x_gpu = dev.from_cpu(&x_flat, &Shape::new(vec![1, h_in]), grim_tensor::DType::F32)?;
        let a_gpu = dev.from_cpu(
            &self.a_log,
            &Shape::new(vec![self.d_inner, self.d_state]),
            grim_tensor::DType::F32,
        )?;
        let b_gpu = dev.from_cpu(
            &self.b_param,
            &Shape::new(vec![self.d_inner, self.d_state]),
            grim_tensor::DType::F32,
        )?;
        let c_gpu = dev.from_cpu(
            &self.d_param,
            &Shape::new(vec![self.d_inner]),
            grim_tensor::DType::F32,
        )?;
        let d_gpu = dev.from_cpu(
            &self.dt_bias,
            &Shape::new(vec![self.d_inner]),
            grim_tensor::DType::F32,
        )?;
        let state_gpu = dev.from_cpu(
            &state.h,
            &Shape::new(vec![1, self.d_inner * self.d_state]),
            grim_tensor::DType::F32,
        )?;

        let out_shape = Shape::new(vec![1, self.d_inner]);
        let (scan_out, _) = dev.selective_scan(
            x_gpu.as_ref(),
            a_gpu.as_ref(),
            b_gpu.as_ref(),
            c_gpu.as_ref(),
            d_gpu.as_ref(),
            state_gpu.as_ref(),
            1,
            self.d_state,
            self.d_inner,
            1,
            &out_shape,
        )?;
        let scan_data = scan_out.to_cpu_vec_f32()?;
        let state_data = state_gpu.to_cpu_vec_f32()?;

        // Update state from GPU output (kernel writes back in-place).
        state.h.copy_from_slice(&state_data);
        state.pos += 1;

        // Build output token and project out. (Audit fix: this vec was sized
        // `h_in` but written for `d_inner` entries — out of bounds whenever
        // d_inner > hidden.)
        let mut out = vec![0.0f32; self.d_inner];
        for n in 0..self.d_inner {
            out[n] = scan_data.get(n).copied().unwrap_or(0.0);
        }
        let out_t = cpu_tensor(out, Shape::new(vec![1, self.d_inner]));
        let residual = self.out_proj.forward(&out_t)?;
        Ok(residual)
    }

    /// CPU fallback path for Mamba selective scan.
    fn step_block_cpu(&self, x: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        // Step-wise selective SSM scan.
        //
        // Audit fix (grim-models): this path previously skipped `in_proj`
        // entirely and sliced the RAW hidden vector as if it were the xz
        // pair, then indexed past its end — an out-of-bounds panic for any
        // config with `2 * d_inner > hidden_size` (i.e. every real Mamba
        // shape). Block contract: norm(hidden) → in_proj → xz of length
        // 2*d_inner (scan input x = xz[..d_inner], gate z = xz[d_inner..])
        // → selective scan → out_proj back to hidden.
        let x_norm = self.norm.forward(x)?;
        let xz_t = self.in_proj.forward(&x_norm)?;
        let xz = xz_t.to_vec_f32()?;
        if xz.len() < 2 * self.d_inner {
            return Err(Error::Shape(format!(
                "mamba step_block_cpu: in_proj produced {} elements, need 2*d_inner={}",
                xz.len(),
                2 * self.d_inner
            )));
        }
        let x_flat: Vec<f32> = xz[..self.d_inner].to_vec();

        // Consistency with the GPU path (MOD-1): a missing `ssm_b` tensor
        // zero-fills B at load, which silently turns the SSM into a
        // zero-input recurrence. The GPU path refuses; the CPU path must
        // refuse identically instead of degrading quietly.
        if self.b_param.is_empty() {
            return Err(Error::Unimplemented(
                "Mamba step_block_cpu: b_param is empty; refusing to run with a zero B matrix".into(),
            ));
        }

        // MOD-3 fix: proper discretized SSM recurrence. The previous code used a
        // placeholder `xz_data[s] * (state.pos as f32 * 0.01)` term that has no
        // basis in the selective-scan math (it grew linearly with position and
        // indexed the wrong tensor). The correct update is
        //   h_new = A[n,s] * h + B[n,s] * x_n
        // where A = a_log, B = b_param, and x_n is the input to channel `n`
        // (mirroring the GPU kernel `h_new = a * h_prev + x_n * b_row[s]`).
        for (n, &x_n) in x_flat.iter().take(state.d_inner).enumerate() {
            for s in 0..state.d_state {
                let a = self.a_log[n * state.d_state + s];
                let b = self
                    .b_param
                    .get(n * state.d_state + s)
                    .copied()
                    .unwrap_or(0.0);
                let h_idx = n * state.d_state + s;
                let new_h = a * state.h[h_idx] + b * x_n;
                state.h[h_idx] = new_h;
            }
        }
        state.pos += 1;

        // Build an output token by summing state over s, gated by z, then
        // project back to hidden width via out_proj.
        let mut out = vec![0.0f32; self.d_inner];
        for n in 0..self.d_inner {
            let mut acc = 0.0f32;
            for s in 0..self.d_state {
                acc += state.h[n * self.d_state + s];
            }
            out[n] = acc + xz[self.d_inner + n] * self.d_param[n];
        }
        let out_t = cpu_tensor(out, Shape::new(vec![1, self.d_inner]));
        let residual = self.out_proj.forward(&out_t)?;
        Ok(residual)
    }
}

pub struct Mamba {
    pub cfg: MambaConfig,
    pub device: Device,
    pub tok_embeddings: grim_nn::Embedding,
    pub layers: Vec<MambaBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Mamba {
    pub fn random(device: Device, cfg: MambaConfig) -> Self {
        let mut rng = grim_core::rng::SimpleRng::new(0xCAFE_F00D_BEEF_DEADu64);
        let embed_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let tok_embeddings = grim_nn::Embedding {
            weight: cpu_tensor(
                embed_data,
                Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let mut block = MambaBlock::random(&cfg, &mut rng);
            block.device = device.clone();
            layers.push(block);
        }
        let norm = RmsNorm {
            weight: cpu_tensor(
                vec![1.0; cfg.hidden_size],
                Shape::new(vec![cfg.hidden_size]),
            ),
            eps: cfg.rms_norm_eps,
        };
        let output_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let output = Linear::from_tensor(
            cpu_tensor(
                output_data,
                Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
            None,
        );
        Self {
            cfg: cfg.clone(),
            device,
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }

    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: MambaConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Tensor-parallel load entry for Mamba. Mamba is a state-space model:
    /// the recurrent SSM path has no row-parallel all-reduce semantics (there
    /// is no matmul whose partial outputs sum across ranks — the state
    /// evolves *per-token*), so naive column/row sharding of the `in_proj` /
    /// `out_proj` matrices is mathematically wrong rather than merely
    /// unfinished. A safe `load_tp` needs a bespoke SSM sharding plan. Refuses
    /// `world_size > 1` until then.
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: MambaConfig,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "Mamba",
            "the SSM recurrent path has no row-parallel all-reduce semantics; \
             sharding in_proj/out_proj needs a bespoke plan",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        let tok_embeddings =
            grim_nn::Embedding::load(&ws.pp("token_embd"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(MambaBlock::load(
                &ws.pp("blk").pp(&i.to_string()),
                &cfg,
                device.clone(),
            )?);
        }
        let norm = RmsNorm::load(&ws.pp("output_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load(&ws.pp("output"), cfg.hidden_size, cfg.vocab_size, false)?;
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

impl MambaBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &MambaConfig, device: Device) -> Result<Self> {
        let norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let in_proj = Linear::load(&ws.pp("ssm_in"), cfg.hidden_size, 2 * cfg.d_inner, false)?;
        let out_proj = Linear::load(&ws.pp("ssm_out"), cfg.d_inner, cfg.hidden_size, false)?;

        let conv = ws
            .get([cfg.d_inner, cfg.conv_kernel], "ssm_conv1d.weight")?
            .to_vec_f32()?;
        let a_log = ws.get([cfg.d_inner, cfg.d_state], "ssm_a")?.to_vec_f32()?;
        let b_param = ws
            .get([cfg.d_inner, cfg.d_state], "ssm_b")
            .and_then(|t| t.to_vec_f32())
            .unwrap_or_default();
        let d_param = ws.get([cfg.d_inner], "ssm_d")?.to_vec_f32()?;
        let dt_bias = ws.get([cfg.d_inner], "ssm_dt.bias")?.to_vec_f32()?;

        Ok(Self {
            norm,
            in_proj,
            conv,
            a_log,
            b_param,
            d_param,
            dt_bias,
            out_proj,
            d_state: cfg.d_state,
            d_inner: cfg.d_inner,
            d_conv: cfg.d_conv,
            device,
        })
    }
}

impl Model for Mamba {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
}
impl StatefulSequence for Mamba {
    fn init_state(&self, batch: usize) -> Box<dyn SsmState> {
        // Instantiate using the state pool representation or fall back to MambaState (§5.1)
        Box::new(MambaState::new(batch, self.cfg.d_inner, self.cfg.d_state))
    }

    fn step(&self, state: &mut dyn SsmState, input: &Tensor) -> Result<Tensor> {
        let ms: &mut MambaState = state
            .as_any_mut()
            .downcast_mut::<MambaState>()
            .ok_or_else(|| Error::Session("state downcast".into()))?;

        // Apply token embedding — input is token IDs, not hidden states.
        let ids = input.to_vec_f32()?;
        let h = self.tok_embeddings.forward(
            &ids.iter().map(|x| *x as u32).collect::<Vec<u32>>(),
            input.shape().dims()[0],
            self.cfg.hidden_size,
        )?;

        // Map (input -> step through each layer with shared SSM state).
        let mut h = h;
        for layer in &self.layers {
            h = layer.step_block(&h, ms)?;
        }

        let h = self.norm.forward(&h)?;
        let logits = self.output.forward(&h)?;
        Ok(logits)
    }
}

impl CausalLm for Mamba {
    fn new_session(&self) -> Box<dyn grim_core::session::SessionT> {
        Box::new(grim_core::session::Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        // Audit fix (grim-models): this used to call `init_state(1)` fresh on
        // EVERY forward and discard it — a stateful SSM driven through the
        // engine was completely context-free after its first token (each
        // decode step saw one token from zeroed state; prefill worked, every
        // subsequent token was garbage). The state now lives on the session
        // and advances across calls, mirroring the KV-cache contract.
        if session.model_state().is_none() {
            session.set_model_state(Box::new(self.init_state(1)));
        }
        let cell = session
            .model_state_mut()
            .ok_or_else(|| Error::Session("mamba: session model_state vanished".into()))?;
        let boxed_state = cell.downcast_mut::<Box<dyn SsmState>>().ok_or_else(|| {
            Error::Session("mamba: session model_state holds another model's state".into())
        })?;
        let logits = self.step(boxed_state.as_mut(), input_ids)?;
        let seq_len = input_ids.shape().elem_count();
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use grim_core::model::CausalLm;
    use grim_core::session::{Inner, SessionT};

    /// Audit gate (grim-models): CausalLm::forward must thread the SSM state
    /// across calls via session.model_state — the pre-fix code re-initialized
    /// a fresh state on EVERY call, so engine decode was context-free after
    /// the first token. The second call's logits through one session must
    /// equal an explicit init→step→step reference, and the session position
    /// must advance per call.
    #[test]
    fn mamba_forward_keeps_state_across_calls() {
        let cfg = MambaConfig {
            vocab_size: 64,
            hidden_size: 16,
            d_state: 4,
            d_inner: 32,
            d_conv: 4,
            num_layers: 2,
            conv_kernel: 4,
            rms_norm_eps: 1e-5,
        };
        let model = Mamba::random(Device::Cpu, cfg);
        let tok = |v: f32| cpu_tensor(vec![v], Shape::new(vec![1]));

        let mut sess = Inner::new(Device::Cpu);
        let _first = CausalLm::forward(&model, &mut sess, &tok(1.0), &tok(0.0), &[]).unwrap();
        let second_with_state = CausalLm::forward(&model, &mut sess, &tok(2.0), &tok(0.0), &[])
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(
            sess.current_pos(),
            2,
            "session position must advance once per forward call"
        );

        // Reference: explicitly carried state (init once, step twice).
        let mut state = model.init_state(1);
        let _ = model.step(state.as_mut(), &tok(1.0)).unwrap();
        let ref_second = model
            .step(state.as_mut(), &tok(2.0))
            .unwrap()
            .to_vec_f32()
            .unwrap();

        assert_eq!(
            second_with_state, ref_second,
            "CausalLm::forward must thread SSM state across calls — a mismatch \
             means the session state was reset between calls"
        );
    }
}

// ---------------------------------------------------------------------------
// Numeric reference tests (audit follow-up): the SSM recurrences were
// previously tested only for boundedness/state-threading, never for value
// correctness against an independent recomputation of the documented math.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod numeric_reference_tests {
    use super::*;
    use grim_core::rng::SimpleRng;

    fn f64_matvec(w: &[f32], x: &[f64], out_dim: usize, in_dim: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; out_dim];
        for i in 0..out_dim {
            let mut acc = 0.0f64;
            for j in 0..in_dim {
                acc += w[i * in_dim + j] as f64 * x[j];
            }
            out[i] = acc;
        }
        out
    }

    fn f64_rmsnorm(x: &[f64], w: &[f32], eps: f32) -> Vec<f64> {
        let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (mean_sq + eps as f64).sqrt();
        x.iter()
            .zip(w)
            .map(|(&v, &wi)| v * inv * wi as f64)
            .collect()
    }

    /// Mamba-1 selective scan (CPU path) vs an independent f64 recomputation
    /// of the documented update h' = A·h + B·x, output = Σ_s h' + z·D,
    /// threaded across TWO steps (so state carry is value-checked too).
    #[test]
    fn mamba1_scan_two_steps_match_f64_reference() {
        let cfg = MambaConfig {
            vocab_size: 64,
            hidden_size: 8,
            d_state: 4,
            d_inner: 16,
            d_conv: 4,
            num_layers: 1,
            conv_kernel: 4,
            rms_norm_eps: 1e-5,
        };
        let model = Mamba::random(Device::Cpu, cfg.clone());
        let block = &model.layers[0];
        let eps = cfg.rms_norm_eps;
        let w_in = block.in_proj.weight.to_vec_f32().unwrap();
        let w_out = block.out_proj.weight.to_vec_f32().unwrap();
        let w_norm = block.norm.weight.to_vec_f32().unwrap();

        let mut state = MambaState::new(1, cfg.d_inner, cfg.d_state);
        // Two different input tokens.
        let inputs = [vec![0.3f32; 8], vec![-0.7f32; 8]];
        let mut h: Vec<f64> = vec![0.0; cfg.d_inner * cfg.d_state];
        let mut f64_inputs: Vec<Vec<f64>> = Vec::new();
        let mut got_outputs: Vec<Vec<f32>> = Vec::new();
        for x in &inputs {
            let xt = cpu_tensor(x.clone(), Shape::new(vec![1, 8]));
            got_outputs.push(block.step_block(&xt, &mut state).unwrap().to_vec_f32().unwrap());
            f64_inputs.push(x.iter().map(|&v| v as f64).collect());
        }

        for (step, x64) in f64_inputs.iter().enumerate() {
            let xn = f64_rmsnorm(x64, &w_norm, eps);
            let xz = f64_matvec(&w_in, &xn, 2 * cfg.d_inner, cfg.hidden_size);
            let x_scan = &xz[..cfg.d_inner];
            let z = &xz[cfg.d_inner..];
            for n in 0..cfg.d_inner {
                for s in 0..cfg.d_state {
                    let idx = n * cfg.d_state + s;
                    h[idx] = block.a_log[idx] as f64 * h[idx]
                        + block.b_param[idx] as f64 * x_scan[n];
                }
            }
            let mut out = vec![0.0f64; cfg.d_inner];
            for n in 0..cfg.d_inner {
                out[n] = (0..cfg.d_state).map(|s| h[n * cfg.d_state + s]).sum::<f64>()
                    + z[n] * block.d_param[n] as f64;
            }
            let res = f64_matvec(&w_out, &out, cfg.hidden_size, cfg.d_inner);
            for (r, g) in res.iter().zip(&got_outputs[step]) {
                assert!(
                    (r - *g as f64).abs() < 1e-4,
                    "step {step}: reference {r} vs impl {g}"
                );
            }
        }
        // And the state must NOT have been reset between steps (the second
        // step's reference used the carried h) — implicitly proven by the
        // step-1 comparison above.
    }

    /// Mamba-2 SSD recurrence vs an independent f64 recomputation:
    /// decay = -exp(A_log)·softplus(dt_bias), h' = decay·h + dt·B·x,
    /// y = Σ_s h'·C + D·x, gated by SiLU(z).
    #[test]
    fn mamba2_ssd_two_steps_match_f64_reference() {
        use crate::mamba2::{Mamba2Block, Mamba2State};

        let cfg = crate::configs::Mamba2Config {
            vocab_size: 64,
            hidden_size: 8,
            d_state: 4,
            d_inner: 8,
            d_conv: 4,
            num_heads: 2,
            num_layers: 1,
            rms_norm_eps: 1e-5,
        };
        let mut rng = SimpleRng::new(0xABCD_1234u64);
        let block = Mamba2Block::random(&cfg, &mut rng);
        let eps = cfg.rms_norm_eps;
        let w_in = block.in_proj.weight.to_vec_f32().unwrap();
        let w_out = block.out_proj.weight.to_vec_f32().unwrap();
        let w_norm = block.norm.weight.to_vec_f32().unwrap();

        let head_dim = cfg.d_inner / cfg.num_heads;
        let group_state = 1 * cfg.d_state; // n_groups = 1
        let mut state = Mamba2State::new(1, cfg.num_heads, cfg.d_state);
        let inputs = [vec![0.4f32; 8], vec![-0.6f32; 8]];
        let mut h: Vec<f64> = vec![0.0; cfg.num_heads * cfg.d_state];
        let mut got_outputs: Vec<Vec<f32>> = Vec::new();
        for x in &inputs {
            let xt = cpu_tensor(x.clone(), Shape::new(vec![1, 8]));
            got_outputs.push(block.step_block(&xt, &mut state).unwrap().to_vec_f32().unwrap());
            let xn = f64_rmsnorm(&x.iter().map(|&v| v as f64).collect::<Vec<_>>(), &w_norm, eps);
            let xzbc = f64_matvec(
                &w_in,
                &xn,
                2 * cfg.d_inner + 2 * group_state,
                cfg.hidden_size,
            );
            let x_scan = &xzbc[..cfg.d_inner];
            let z = &xzbc[cfg.d_inner..2 * cfg.d_inner];
            let b_flat = &xzbc[2 * cfg.d_inner..2 * cfg.d_inner + group_state];
            let c_flat = &xzbc[2 * cfg.d_inner + group_state..];

            let mut y = vec![0.0f64; cfg.d_inner];
            for head in 0..cfg.num_heads {
                let a = -(block.a_log[head] as f64).exp();
                let dt_x = block.dt_bias[head] as f64;
                let dt = if dt_x > 20.0 { dt_x } else { dt_x.exp().ln_1p() };
                let decay = a * dt;
                for s in 0..cfg.d_state {
                    let mut acc = 0.0f64;
                    for d in 0..head_dim {
                        let x_idx = head * head_dim + d;
                        let h_idx = head * cfg.d_state + s;
                        let new_h =
                            decay * h[h_idx] + dt * b_flat[s] as f64 * x_scan[x_idx];
                        h[h_idx] = new_h;
                        acc += new_h * c_flat[s] as f64;
                    }
                    for d in 0..head_dim {
                        let x_idx = head * head_dim + d;
                        y[x_idx] += acc + block.d_param[head] as f64 * x_scan[x_idx];
                    }
                }
            }
            // SiLU z-gate.
            for (yi, &zi) in y.iter_mut().zip(z.iter()) {
                *yi *= zi / (1.0 + (-zi).exp());
            }
            let res = f64_matvec(&w_out, &y, cfg.hidden_size, cfg.d_inner);
            let got = got_outputs.last().unwrap();
            for (r, g) in res.iter().zip(got) {
                assert!((r - *g as f64).abs() < 1e-4, "reference {r} vs impl {g}");
            }
        }
        // State must have been carried (both steps used the same h).
        assert_eq!(state.pos, 2, "mamba2 state must advance once per step");
    }
}
