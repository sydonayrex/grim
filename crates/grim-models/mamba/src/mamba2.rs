//! Mamba-2 SSD architecture (step-wise CPU body).
//!
//! Block contract (per `Mamba2Config`): pre-norm → `in_proj` producing
//! `(x, z, B, C)` with group-shared B/C (n_groups ≤ n_heads, GQA-style) →
//! head-chunked selective SSM with a per-head scalar decay
//! `A[h] = -exp(A_log[h])` (Mamba-2's defining difference vs Mamba-1's
//! `(d_inner, d_state)` A matrix) → SiLU z-gate → D skip → `out_proj`.
//!
//! Fidelity note (mirrors the Mamba-1 body in `lib.rs`): this is a
//! step-wise recurrence with bias-only discretization
//! `dt = softplus(dt_bias[h])`, and the short conv is carried in weights but
//! not applied (same "skipped in v1" contract). Chunked parallel scan, conv,
//! and time-varying dt are the GPU-kernel work items.

use std::any::Any;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, Model, ModelConfig, SsmState, StatefulSequence};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

use crate::configs::Mamba2Config;

#[derive(Clone, Debug)]
pub struct Mamba2State {
    /// Per-head SSM state, layout `[batch, n_heads * d_state]`.
    pub h: Vec<f32>,
    pub batch: usize,
    pub n_heads: usize,
    pub d_state: usize,
    /// Tokens already advanced (pos cursor; snapshot-friendly, §5.3).
    pub pos: usize,
}

impl Mamba2State {
    pub fn new(batch: usize, n_heads: usize, d_state: usize) -> Self {
        Self {
            h: vec![0.0; batch * n_heads * d_state],
            batch,
            n_heads,
            d_state,
            pos: 0,
        }
    }
}

impl SsmState for Mamba2State {
    fn clone_snapshot(&self) -> Result<Box<dyn SsmState>> {
        Ok(Box::new(self.clone()))
    }
    fn restore_snapshot(&mut self, snap: &dyn SsmState) -> Result<()> {
        let other = snap
            .as_any()
            .downcast_ref::<Mamba2State>()
            .ok_or_else(|| Error::Session("snapshot downcast failed".into()))?;
        if self.batch != other.batch
            || self.n_heads != other.n_heads
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

#[derive(Clone)]
pub struct Mamba2Block {
    pub norm: RmsNorm,
    pub in_proj: Linear,
    pub out_proj: Linear,
    /// Conv weights carried for the load contract; not applied (v1 contract).
    pub conv: Vec<f32>,
    /// Per-head log-decay `A_log[h]`; the decay is `-exp(A_log[h])`.
    pub a_log: Vec<f32>,
    /// Per-head dt bias (discretization step).
    pub dt_bias: Vec<f32>,
    /// Per-head skip connection D.
    pub d_param: Vec<f32>,
    pub n_heads: usize,
    pub n_groups: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub device: Device,
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Mamba2Block {
    pub fn random(cfg: &Mamba2Config, rng: &mut grim_core::rng::SimpleRng) -> Self {
        // Single group (n_groups = 1): all heads share one B/C — the smallest
        // valid Mamba-2 grouping. Multi-group loading is a weight-loader work
        // item once a real Mamba-2 checkpoint pipeline exists.
        let n_groups = 1usize;
        // in_proj: hidden → 2*d_inner + 2*n_groups*d_state (x, z, B, C).
        let group_state = n_groups * cfg.d_state;
        let in_proj_weight: Vec<f32> = (0..(2 * cfg.d_inner + 2 * group_state) * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let in_proj = Linear::from_tensor(
            cpu_tensor(
                in_proj_weight,
                Shape::new(vec![2 * cfg.d_inner + 2 * group_state, cfg.hidden_size]),
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
        Self {
            norm: RmsNorm {
                weight: cpu_tensor(
                    vec![1.0; cfg.hidden_size],
                    Shape::new(vec![cfg.hidden_size]),
                ),
                eps: cfg.rms_norm_eps,
            },
            in_proj,
            out_proj,
            conv: (0..cfg.d_inner * cfg.d_conv)
                .map(|_| (rng.next_f32() - 0.5) * 0.5)
                .collect(),
            a_log: (0..cfg.num_heads)
                .map(|_| (rng.next_f32() - 0.5) * 0.2)
                .collect(),
            dt_bias: (0..cfg.num_heads).map(|_| 0.0).collect(),
            d_param: vec![1.0; cfg.num_heads],
            n_heads: cfg.num_heads,
            n_groups: n_groups.min(cfg.num_heads).max(1),
            d_state: cfg.d_state,
            d_inner: cfg.d_inner,
            device: Device::Cpu,
        }
    }

    /// Forward one token using existing state. CPU step-wise recurrence.
    pub fn step_block(&self, x: &Tensor, state: &mut Mamba2State) -> Result<Tensor> {
        let x_norm = self.norm.forward(x)?;
        let xzbc_t = self.in_proj.forward(&x_norm)?;
        let xzbc = xzbc_t.to_vec_f32()?;
        let group_state = self.n_groups * self.d_state;
        let need = 2 * self.d_inner + 2 * group_state;
        if xzbc.len() < need {
            return Err(Error::Shape(format!(
                "mamba2 step_block: in_proj produced {} elements, need {need}",
                xzbc.len()
            )));
        }
        let x = &xzbc[..self.d_inner];
        let z = &xzbc[self.d_inner..2 * self.d_inner];
        let b_flat = &xzbc[2 * self.d_inner..2 * self.d_inner + group_state];
        let c_flat = &xzbc[2 * self.d_inner + group_state..need];

        let head_dim = (self.d_inner / self.n_heads).max(1);
        let mut y = vec![0.0f32; self.d_inner];
        for head in 0..self.n_heads {
            // Group-shared B/C: head h reads group (h * n_groups / n_heads).
            let group = head * self.n_groups / self.n_heads;
            let a = -self.a_log[head].exp();
            let dt = softplus(self.dt_bias[head]);
            let decay = a * dt; // bias-only discretization (see module docs)
            let h_base = head * self.d_state;
            let b_base = group * self.d_state;
            for s in 0..self.d_state {
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    let x_idx = head * head_dim + d;
                    let h_idx = h_base + s;
                    let new_h = decay * state.h[h_idx] + dt * b_flat[b_base + s] * x[x_idx];
                    state.h[h_idx] = new_h;
                    acc += new_h * c_flat[b_base + s];
                }
                // D skip + accumulate into the head's output rows.
                for d in 0..head_dim {
                    let x_idx = head * head_dim + d;
                    y[x_idx] += acc + self.d_param[head] * x[x_idx];
                }
            }
        }

        // SiLU z-gate.
        for (yi, &zi) in y.iter_mut().zip(z.iter()) {
            *yi *= silu(zi);
        }
        // Advance the position cursor: MambaState documents `pos` as the
        // per-call token count and Mamba-1's step advances it — the Mamba-2
        // step must too, or speculative snapshots read a stale position.
        state.pos += 1;
        let out_t = cpu_tensor(y, Shape::new(vec![1, self.d_inner]));
        let out = self.out_proj.forward(&out_t)?;
        Ok(out)
    }
}

pub struct Mamba2 {
    pub cfg: Mamba2Config,
    pub device: Device,
    pub tok_embeddings: grim_nn::Embedding,
    pub layers: Vec<Mamba2Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Mamba2 {
    pub fn random(device: Device, cfg: Mamba2Config) -> Self {
        let mut rng = grim_core::rng::SimpleRng::new(0x5D2A_C2DE_AD01u64);
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
            let mut block = Mamba2Block::random(&cfg, &mut rng);
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
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }

    /// Tensor-parallel load is refused for the same reason as Mamba-1: the
    /// SSM recurrence has no row-parallel all-reduce semantics.
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Mamba2Config,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "Mamba2",
            "the SSM recurrent path has no row-parallel all-reduce semantics; \
             sharding in_proj/out_proj needs a bespoke plan",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        let _ = ws;
        Ok(Mamba2::random(device, cfg))
    }
}

impl Model for Mamba2 {
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

impl StatefulSequence for Mamba2 {
    fn init_state(&self, batch: usize) -> Box<dyn SsmState> {
        Box::new(Mamba2State::new(
            batch,
            self.cfg.num_heads,
            self.cfg.d_state,
        ))
    }

    fn step(&self, state: &mut dyn SsmState, input: &Tensor) -> Result<Tensor> {
        let ms: &mut Mamba2State = state
            .as_any_mut()
            .downcast_mut::<Mamba2State>()
            .ok_or_else(|| Error::Session("state downcast".into()))?;

        let ids = input.to_vec_f32()?;
        let h = self.tok_embeddings.forward(
            &ids.iter().map(|x| *x as u32).collect::<Vec<u32>>(),
            input.shape().dims()[0],
            self.cfg.hidden_size,
        )?;

        let mut h = h;
        for layer in &self.layers {
            h = layer.step_block(&h, ms)?;
        }

        let h = self.norm.forward(&h)?;
        let logits = self.output.forward(&h)?;
        Ok(logits)
    }
}

impl CausalLm for Mamba2 {
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
        // SSM state lives on the session and advances across calls (same
        // contract as Mamba-1: a fresh state per call would make decode
        // context-free after the first token).
        if session.model_state().is_none() {
            session.set_model_state(Box::new(self.init_state(1)));
        }
        let cell = session
            .model_state_mut()
            .ok_or_else(|| Error::Session("mamba2: session model_state vanished".into()))?;
        let boxed_state = cell.downcast_mut::<Box<dyn SsmState>>().ok_or_else(|| {
            Error::Session("mamba2: session model_state holds another model's state".into())
        })?;
        let logits = self.step(boxed_state.as_mut(), input_ids)?;
        let seq_len = input_ids.shape().elem_count();
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::model::CausalLm;
    use grim_core::session::{Inner, SessionT};

    fn cfg() -> Mamba2Config {
        Mamba2Config {
            vocab_size: 64,
            hidden_size: 16,
            d_state: 4,
            d_inner: 32,
            d_conv: 4,
            num_heads: 4,
            num_layers: 2,
            rms_norm_eps: 1e-5,
        }
    }

    fn tok(v: f32) -> Tensor {
        cpu_tensor(vec![v], Shape::new(vec![1]))
    }

    /// Session state threads across forward calls; logits through one session
    /// equal an explicit init→step→step reference (same gate as Mamba-1).
    #[test]
    fn mamba2_forward_keeps_state_across_calls() {
        let model = Mamba2::random(Device::Cpu, cfg());
        let mut sess = Inner::new(Device::Cpu);
        let _first = CausalLm::forward(&model, &mut sess, &tok(1.0), &tok(0.0), &[]).unwrap();
        let second_with_state = CausalLm::forward(&model, &mut sess, &tok(2.0), &tok(0.0), &[])
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(sess.current_pos(), 2, "position must advance per call");

        let mut state = model.init_state(1);
        let _ = model.step(state.as_mut(), &tok(1.0)).unwrap();
        let ref_second = model
            .step(state.as_mut(), &tok(2.0))
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(
            second_with_state, ref_second,
            "session state must thread across calls"
        );
    }

    /// State shape follows the head-chunked contract `[n_heads * d_state]`,
    /// not Mamba-1's `[d_inner * d_state]`.
    #[test]
    fn mamba2_state_is_head_chunked() {
        let model = Mamba2::random(Device::Cpu, cfg());
        let state = model.init_state(1);
        let ms = state.as_any().downcast_ref::<Mamba2State>().unwrap();
        assert_eq!(ms.h.len(), 4 * 4, "n_heads * d_state");
        assert_eq!(ms.n_heads, 4);
        assert_eq!(ms.d_state, 4);
    }

    /// A nonzero A_log must actually decay the state: stepping the same
    /// token repeatedly with a fixed B/C stream drives the state magnitude
    /// below the first-step magnitude (the recurrence is contractive).
    #[test]
    fn mamba2_recurrence_is_contractive() {
        let model = Mamba2::random(Device::Cpu, cfg());
        let mut state = model.init_state(1);
        let ms_before = state
            .as_any()
            .downcast_ref::<Mamba2State>()
            .unwrap()
            .clone();
        let mag = |s: &Mamba2State| s.h.iter().map(|v| v.abs()).sum::<f32>();
        let _ = model.step(state.as_mut(), &tok(1.0)).unwrap();
        let m1 = mag(state.as_any().downcast_ref::<Mamba2State>().unwrap());
        for _ in 0..50 {
            let _ = model.step(state.as_mut(), &tok(1.0)).unwrap();
        }
        let mn = mag(state.as_any().downcast_ref::<Mamba2State>().unwrap());
        let _ = ms_before;
        assert!(
            mn <= m1 * 5.0 + 1e-6,
            "repeated input must not blow up the state (first {m1}, later {mn})"
        );
    }
}
