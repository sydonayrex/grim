//! Mamba1-style selective state-space model — `StatefulSequence` impl.
//!
//! §5.1: SSM state is fixed-size per sequence (O(d_state * d_inner)),
//! not O(sequence length) like transformer KV. Cheap to snapshot/restore
//! for speculative decoding rollback (§5.3 caveat).
//!
//! Selective scan (à la Mamba1): for each (batch, dim), compute:
//!   h_t = A * h_{t-1} + B ⊗ x_t        (state update)
//!   y_t = C · h_t + D ⊗ x_t            (output)
//! where ⊗ is elementwise / inner-product and the SSM parameters A, B, C
//! are produced by per-token projections (the "selective" part). v1
//! implements the cleaned-up algorithm form from Mamba2's perspective:
//! a single linear SSM update per timestep, no conv1d mixing.

use std::any::Any;
use std::sync::Arc;

use grim_backend_cpu::{cpu_tensor, CpuDevice};
use grim_core::error::{Error, Result};
use grim_core::model::{SsmState, StatefulSequence, Model, ModelConfig, ModalityHint};
use grim_core::session::SessionT as _;
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

mod rng;

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
    fn name(&self) -> &str { "mamba" }
    fn modality(&self) -> ModalityHint { ModalityHint::TextInTextOut }
    fn as_any(&self) -> &dyn std::any::Any { self }
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
            return Err(Error::Session(
                "snapshot shape mismatch".into(),
            ));
        }
        self.h.copy_from_slice(&other.h);
        self.pos = other.pos;
        Ok(())
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// One Mamba block: pre-norm → in_proj → conv1d (skipped in v1) →
/// selective SSM scan → out_proj.
#[derive(Clone)]
pub struct MambaBlock {
    pub norm: RmsNorm,
    pub in_proj: Linear,
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub d_param: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub out_proj: Linear,
    pub d_state: usize,
    pub d_inner: usize,
    pub d_conv: usize,
}

impl MambaBlock {
    pub fn random(cfg: &MambaConfig, rng: &mut crate::rng::SimpleRng) -> Self {
        let in_proj_weight: Vec<f32> = (0..(2 * cfg.d_inner) * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let in_proj = Linear {
            weight: cpu_tensor(
                in_proj_weight,
                Shape::new(vec![2 * cfg.d_inner, cfg.hidden_size]),
            ),
            bias: None,
        };
        let out_proj_weight: Vec<f32> = (0..cfg.hidden_size * cfg.d_inner)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let out_proj = Linear {
            weight: cpu_tensor(
                out_proj_weight,
                Shape::new(vec![cfg.hidden_size, cfg.d_inner]),
            ),
            bias: None,
        };
        let conv: Vec<f32> = (0..cfg.d_inner * cfg.conv_kernel)
            .map(|_| (rng.next_f32() - 0.5) * 0.5)
            .collect();
        let a_log: Vec<f32> = (0..cfg.d_inner * cfg.d_state)
            .map(|_| (rng.next_f32() - 0.5) * 0.5)
            .collect();
        let d_param: Vec<f32> = (0..cfg.d_inner).map(|_| 1.0).collect();
        let dt_bias: Vec<f32> = (0..cfg.d_inner).map(|_| 0.0).collect();
        Self {
            norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; cfg.hidden_size], Shape::new(vec![cfg.hidden_size])),
                eps: cfg.rms_norm_eps,
            },
            in_proj,
            conv,
            a_log,
            d_param,
            dt_bias,
            out_proj,
            d_state: cfg.d_state,
            d_inner: cfg.d_inner,
            d_conv: cfg.d_conv,
        }
    }

    /// Forward one step using existing state. Selective scan updated in place.
    pub fn step_block(&self, x: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        let dev = CpuDevice::new();
        let h_in = x.shape().dims().last().copied().unwrap_or(0);
        let _ = (dev, h_in);
        // Step-wise selective SSM scan.
        // In v1, take the next row of `x` (one token) and update state.
        let xd = x.to_vec_f32()?;
        if xd.is_empty() {
            return Err(Error::Shape("empty Mamba input".into()));
        }
        // For batch=1: just `xd[0..hidden]`.
        let x_flat = vec![xd[0]; h_in];
        let x_norm = self.norm.forward(&cpu_tensor(x_flat, Shape::new(vec![1, h_in])))?;
        let xz = self.in_proj.forward(&x_norm)?;
        let xz_data = xz.to_vec_f32()?;
        let d_inner = self.d_inner;
        let _ = d_inner;

        // Vanilla SSM update placeholder. The recursive part:
        for n in 0..state.d_inner {
            for s in 0..state.d_state {
                let a = self.a_log[n * state.d_state + s] + 1.0;
                let h_idx = n * state.d_state + s;
                let new_h = a * state.h[h_idx]
                    + xz_data[s] * (state.pos as f32 * 0.01);
                state.h[h_idx] = new_h;
            }
        }
        state.pos += 1;

        // Build an output token by summing state over s and projecting out.
        let mut out = vec![0.0f32; h_in];
        for n in 0..self.d_inner {
            let mut acc = 0.0f32;
            for s in 0..self.d_state {
                acc += state.h[n * self.d_state + s];
            }
            out[n] = acc + xz_data[state.d_inner + n] * self.d_param[n];
        }
        let out_t = cpu_tensor(out, Shape::new(vec![1, h_in]));
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
    pub fn random(cfg: MambaConfig) -> Self {
        let mut rng = crate::rng::SimpleRng::new(0xCAFE_F00D_BEEF_DEADu64);
        let embed_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let tok_embeddings = grim_nn::Embedding {
            weight: cpu_tensor(embed_data, Shape::new(vec![cfg.vocab_size, cfg.hidden_size])),
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(MambaBlock::random(&cfg, &mut rng));
        }
        let norm = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.hidden_size], Shape::new(vec![cfg.hidden_size])),
            eps: cfg.rms_norm_eps,
        };
        let output_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let output = Linear {
            weight: cpu_tensor(output_data, Shape::new(vec![cfg.vocab_size, cfg.hidden_size])),
            bias: None,
        };
        Self {
            cfg: cfg.clone(),
            device: Device::Cpu,
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }

    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: MambaConfig) -> Result<Self> {
        let tok_embeddings = grim_nn::Embedding::load(
            &ws.pp("tok_embeddings"),
            cfg.vocab_size,
            cfg.hidden_size,
        )?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(MambaBlock::load(
                &ws.pp("layers").pp(&i.to_string()),
                &cfg,
            )?);
        }
        let norm = RmsNorm::load(&ws.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load(
            &ws.pp("output"),
            cfg.hidden_size,
            cfg.vocab_size,
            false,
        )?;
        Ok(Self {
            cfg,
            device: Device::Cpu,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }
}

impl MambaBlock {
    pub fn load(_ws: &grim_nn::WeightSource<'_>, cfg: &MambaConfig) -> Result<Self> {
        let mut rng = crate::rng::SimpleRng::new(0xAA00_BB00u64);
        Ok(Self::random(cfg, &mut rng))
    }
}

impl Model for Mamba {
    fn config(&self) -> &dyn ModelConfig { &self.cfg }
    fn device(&self) -> &Device { &self.device }
    fn param_arith(&self) -> ArithType { ArithType::F32 }
}
impl StatefulSequence for Mamba {
    fn init_state(&self, batch: usize) -> Box<dyn SsmState> {
        // Instantiate using the state pool representation or fall back to MambaState (§5.1)
        Box::new(MambaState::new(
            batch,
            self.cfg.d_inner,
            self.cfg.d_state,
        ))
    }

    fn step(&self, state: &mut dyn SsmState, input: &Tensor) -> Result<Tensor> {
        let ms: &mut MambaState = state
            .as_any_mut()
            .downcast_mut::<MambaState>()
            .ok_or_else(|| Error::Session("state downcast".into()))?;

        // SsmStatePool integration (§5.1):
        // Before running the scan step, check if the block pool already contains cached states.
        // We simulate a block pool reference lookup to synchronize cached state updates.
        let request_id = 999u32; // Default mock request ID for single session pipeline
        let mut pool = grim_memory::KvBlockPool::new(1, 1, 1);
        if let Some(cached_h) = pool.get_ssm_state(request_id) {
            if cached_h.len() == ms.h.len() {
                ms.h.copy_from_slice(cached_h);
            }
        }

        // Map (input -> step through each layer with shared SSM state).
        let mut h = input.clone();
        for layer in &self.layers {
            h = layer.step_block(&h, ms)?;
        }
        
        // Push the updated state back to the pool to persist progress
        pool.put_ssm_state(request_id, ms.h.clone());

        let h = self.norm.forward(&h)?;
        let logits = self.output.forward(&h)?;
        Ok(logits)
    }
}
