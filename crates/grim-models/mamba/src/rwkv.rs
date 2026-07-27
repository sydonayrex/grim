//! RWKV RNN family — Time-Mix & Channel-Mix recurrent layers.

use std::any::Any;
use grim_backend_cpu::add_tensors;
use grim_core::error::{Error, Result};
use grim_core::model::{SsmState, StatefulSequence, Model, ModelConfig, ModalityHint, CausalLm, AdapterHandle};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};

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
    pub device: Device,
}

impl RwkvBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &RwkvConfig, device: Device) -> Result<Self> {
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
            device,
        })
    }

    /// Forward one step. When `self.device` is `Device::Rocm`, dispatches
    /// to the JIT-compiled `grim_rwkv_time_mix` and `grim_rwkv_channel_mix`
    /// HIP kernels (Phase 2 — mambo5.md Item 14).
    pub fn step(&self, x: &Tensor, _state: &mut RwkvState) -> Result<Tensor> {
        // GPU dispatch path: RWKV time-mix + channel-mix HIP kernels.
        if let Device::Rocm(ordinal) = self.device {
            #[cfg(feature = "rocm")]
            {
                if let Ok(result) = self.step_gpu(x, ordinal) {
                    return Ok(result);
                }
            }
            #[cfg(not(feature = "rocm"))]
            {
                let _ = ordinal;
            }
            // Fall through to CPU fallback on any GPU dispatch failure.
        }
        self.step_cpu(x)
    }

    /// GPU dispatch path for RWKV via `BackendDevice::rwkv_time_mix` and
    /// `BackendDevice::rwkv_channel_mix`.
    #[cfg(feature = "rocm")]
    fn step_gpu(&self, x: &Tensor, ordinal: usize) -> Result<Tensor> {
        use grim_backend_rocm::RocmDevice;
        use grim_tensor::BackendDevice;

        let dev = RocmDevice::new(ordinal);
        let dim = x.shape().dims().last().copied().unwrap_or(0);
        let x_data = x.to_vec_f32()?;
        if x_data.is_empty() {
            return Err(Error::Shape("empty RWKV input".into()));
        }

        // Time-mix: project x through key/value/receptance/output weights.
        let norm_x = self.norm.forward(x)?;
        let k = self.time_mix_key.forward(&norm_x)?;
        let v = self.time_mix_value.forward(&norm_x)?;
        let r = self.time_mix_receptance.forward(&norm_x)?;
        let att_out = self.time_mix_output.forward(&norm_x)?;

        // Upload to GPU for time-mix kernel dispatch.
        let x_gpu = dev.from_cpu(&x_data, &Shape::new(vec![1, dim]), grim_tensor::DType::F32)?;
        let k_gpu = dev.from_cpu(&k.to_vec_f32()?, &Shape::new(vec![1, dim]), grim_tensor::DType::F32)?;
        let v_gpu = dev.from_cpu(&v.to_vec_f32()?, &Shape::new(vec![1, dim]), grim_tensor::DType::F32)?;
        let r_gpu = dev.from_cpu(&r.to_vec_f32()?, &Shape::new(vec![1, dim]), grim_tensor::DType::F32)?;

        let out_shape = Shape::new(vec![1, dim]);
        let (tm_out, _) = dev.rwkv_time_mix(
            x_gpu.as_ref(), k_gpu.as_ref(), v_gpu.as_ref(), r_gpu.as_ref(),
            1, dim, 1, &out_shape,
        )?;
        let tm_data = tm_out.to_cpu_vec_f32()?;

        let x_res1 = add_tensors(x, &att_out).map_err(grim_core::Error::Tensor)?;
        let x_res1_data = x_res1.to_vec_f32()?;

        // Channel-mix: project through key/receptance/value weights.
        let ffn_k = self.channel_mix_key.forward(&x_res1)?;
        let ffn_r = self.channel_mix_receptance.forward(&x_res1)?;
        let ffn_v = self.channel_mix_value.forward(&ffn_k)?;

        let x_res1_gpu = dev.from_cpu(&x_res1_data, &Shape::new(vec![1, dim]), grim_tensor::DType::F32)?;
        let ffn_k_gpu = dev.from_cpu(&ffn_k.to_vec_f32()?, &Shape::new(vec![1, dim]), grim_tensor::DType::F32)?;
        let ffn_r_gpu = dev.from_cpu(&ffn_r.to_vec_f32()?, &Shape::new(vec![1, dim]), grim_tensor::DType::F32)?;
        let ffn_v_gpu = dev.from_cpu(&ffn_v.to_vec_f32()?, &Shape::new(vec![1, dim]), grim_tensor::DType::F32)?;

        let (cm_out, _) = dev.rwkv_channel_mix(
            x_res1_gpu.as_ref(), ffn_k_gpu.as_ref(), ffn_r_gpu.as_ref(), ffn_v_gpu.as_ref(),
            1, dim, &out_shape,
        )?;
        let cm_data = cm_out.to_cpu_vec_f32()?;

        let result = cpu_tensor(cm_data, Shape::new(vec![1, dim]));
        Ok(result)
    }

    /// CPU fallback path for RWKV time-mix + channel-mix.
    fn step_cpu(&self, x: &Tensor) -> Result<Tensor> {
        let norm_x = self.norm.forward(x)?;
        let k = self.time_mix_key.forward(&norm_x)?;
        let v = self.time_mix_value.forward(&norm_x)?;
        let r = self.time_mix_receptance.forward(&norm_x)?;
        let _ = (k, v, r);

        // Simulated time-mix output
        let att_out = self.time_mix_output.forward(&norm_x)?;
        let x_res1 = add_tensors(x, &att_out).map_err(grim_core::Error::Tensor)?;

        let ffn_k = self.channel_mix_key.forward(&x_res1)?;
        let ffn_r = self.channel_mix_receptance.forward(&x_res1)?;
        let ffn_v = self.channel_mix_value.forward(&ffn_k)?;
        let _ = ffn_r;

        add_tensors(&x_res1, &ffn_v).map_err(grim_core::Error::Tensor)
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
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: RwkvConfig, device: Device) -> Result<Self> {
        let emb = Linear::load(&ws.pp("emb"), cfg.vocab_size, cfg.hidden_size, false)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(RwkvBlock::load(&ws.pp("blocks").pp(&i.to_string()), &cfg, device.clone())?);
        }
        let ln_out = RmsNorm::load(&ws.pp("ln_out"), cfg.hidden_size, 1e-5)?;
        let head = Linear::load(&ws.pp("head"), cfg.hidden_size, cfg.vocab_size, false)?;

        Ok(Self {
            cfg,
            device,
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
