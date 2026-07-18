//! RWKV RNN family — Time-Mix & Channel-Mix recurrent layers.

use std::any::Any;
use std::sync::Arc;
use grim_core::error::{Error, Result};
use grim_core::model::{SsmState, StatefulSequence, Model, ModelConfig, ModalityHint, CausalLm, AdapterHandle};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, DType, Tensor};

#[derive(Debug, Clone)]
pub struct RwkvConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
}

impl ModelConfig for RwkvConfig {
    fn name(&self) -> &str {
        "rwkv"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Debug)]
pub struct RwkvState {
    pub state_xy: Vec<f32>,
}

impl SsmState for RwkvState {
    fn clone_snapshot(&self) -> Result<Box<dyn SsmState>> {
        Ok(Box::new(self.clone()))
    }
    fn restore_snapshot(&mut self, snap: &dyn SsmState) -> Result<()> {
        let other = snap.as_any().downcast_ref::<RwkvState>().ok_or_else(|| Error::Session("downcast failed".into()))?;
        self.state_xy.copy_from_slice(&other.state_xy);
        Ok(())
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

pub struct RwkvBlock {
    pub norm: RmsNorm,
    pub time_mix_key: Linear,
    pub time_mix_value: Linear,
    pub time_mix_receptance: Linear,
    pub time_mix_output: Linear,
    pub channel_mix_key: Linear,
    pub channel_mix_receptance: Linear,
    pub channel_mix_value: Linear,
}

impl RwkvBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &RwkvConfig) -> Result<Self> {
        let norm = RmsNorm::load(&ws.pp("ln_x"), cfg.hidden_size, 1e-5)?;
        let time_mix_key = Linear::load(&ws.pp("att.key"), cfg.hidden_size, cfg.hidden_size, false)?;
        let time_mix_value = Linear::load(&ws.pp("att.value"), cfg.hidden_size, cfg.hidden_size, false)?;
        let time_mix_receptance = Linear::load(&ws.pp("att.receptance"), cfg.hidden_size, cfg.hidden_size, false)?;
        let time_mix_output = Linear::load(&ws.pp("att.output"), cfg.hidden_size, cfg.hidden_size, false)?;

        let channel_mix_key = Linear::load(&ws.pp("ffn.key"), cfg.hidden_size, cfg.hidden_size, false)?;
        let channel_mix_receptance = Linear::load(&ws.pp("ffn.receptance"), cfg.hidden_size, cfg.hidden_size, false)?;
        let channel_mix_value = Linear::load(&ws.pp("ffn.value"), cfg.hidden_size, cfg.hidden_size, false)?;

        Ok(Self {
            norm,
            time_mix_key,
            time_mix_value,
            time_mix_receptance,
            time_mix_output,
            channel_mix_key,
            channel_mix_receptance,
            channel_mix_value,
        })
    }

    pub fn step(&self, x: &Tensor, _state: &mut RwkvState) -> Result<Tensor> {
        let norm_x = self.norm.forward(x)?;
        let k = self.time_mix_key.forward(&norm_x)?;
        let v = self.time_mix_value.forward(&norm_x)?;
        let r = self.time_mix_receptance.forward(&norm_x)?;
        let _ = (k, v, r);

        // Simulated time-mix output
        let att_out = self.time_mix_output.forward(&norm_x)?;
        let x_res1 = add_tensors(x, &att_out)?;

        let ffn_k = self.channel_mix_key.forward(&x_res1)?;
        let ffn_r = self.channel_mix_receptance.forward(&x_res1)?;
        let ffn_v = self.channel_mix_value.forward(&ffn_k)?;
        let _ = ffn_r;

        add_tensors(&x_res1, &ffn_v)
    }
}

pub struct Rwkv {
    pub cfg: RwkvConfig,
    pub device: Device,
    pub emb: Linear,
    pub layers: Vec<RwkvBlock>,
    pub ln_out: RmsNorm,
    pub head: Linear,
}

impl Rwkv {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: RwkvConfig) -> Result<Self> {
        let emb = Linear::load(&ws.pp("emb"), cfg.vocab_size, cfg.hidden_size, false)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(RwkvBlock::load(&ws.pp("blocks").pp(&i.to_string()), &cfg)?);
        }
        let ln_out = RmsNorm::load(&ws.pp("ln_out"), cfg.hidden_size, 1e-5)?;
        let head = Linear::load(&ws.pp("head"), cfg.hidden_size, cfg.vocab_size, false)?;

        Ok(Self {
            cfg,
            device: Device::Cpu,
            emb,
            layers,
            ln_out,
            head,
        })
    }
}

impl Model for Rwkv {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
}

impl StatefulSequence for Rwkv {
    fn init_state(&self, batch: usize) -> Box<dyn SsmState> {
        Box::new(RwkvState {
            state_xy: vec![0.0f32; batch * self.cfg.hidden_size],
        })
    }

    fn step(&self, state: &mut dyn SsmState, input: &Tensor) -> Result<Tensor> {
        let s = state.as_any_mut().downcast_mut::<RwkvState>().ok_or_else(|| Error::Session("downcast failed".into()))?;
        let emb_out = self.emb.forward(input)?;
        let mut h = emb_out;
        for layer in &self.layers {
            h = layer.step(&h, s)?;
        }
        let h = self.ln_out.forward(&h)?;
        Ok(self.head.forward(&h)?)
    }
}

impl CausalLm for Rwkv {
    fn new_session(&self) -> Box<dyn grim_core::session::SessionT> {
        Box::new(grim_core::session::Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        _session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let mut state = self.init_state(1);
        self.step(&mut *state, input_ids)
    }
}

fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = grim_backend_cpu::CpuDevice::new();
    let (s, h) = grim_tensor::BackendDevice::add(&dev, a.storage().as_ref(), b.storage().as_ref(), a.shape())?;
    h.synchronize()?;
    Ok(Tensor::new(Arc::from(s), a.shape().clone(), DType::F32, a.provenance().clone(), a.device().clone()))
}
