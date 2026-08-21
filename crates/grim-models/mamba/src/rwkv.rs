//! RWKV RNN family — Time-Mix & Channel-Mix recurrent layers.

use crate::cpu_tensor;
use grim_backend_cpu::add_tensors;
use grim_core::error::{Error, Result};
use grim_core::model::{
    AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig, SsmState, StatefulSequence,
};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};
use std::any::Any;

/// Validate that a loaded weight tensor has the expected shape, returning a
/// descriptive `Error::Shape` when it does not.
fn assert_weight_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    let dims = tensor.shape().dims();
    if dims != expected {
        return Err(Error::Shape(format!(
            "RWKV weight '{}' has shape {:?}, expected {:?}",
            name, dims, expected
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct RwkvConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f64,
}

impl Default for RwkvConfig {
    fn default() -> Self {
        Self {
            vocab_size: 0,
            hidden_size: 0,
            num_layers: 0,
            rms_norm_eps: 1e-5,
        }
    }
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
        let other = snap
            .as_any()
            .downcast_ref::<RwkvState>()
            .ok_or_else(|| Error::Session("downcast failed".into()))?;
        self.state_xy.copy_from_slice(&other.state_xy);
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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
        let norm = RmsNorm::load(&ws.pp("ln_x"), cfg.hidden_size, cfg.rms_norm_eps as f32)?;
        let time_mix_key =
            Linear::load(&ws.pp("att.key"), cfg.hidden_size, cfg.hidden_size, false)?;
        let time_mix_value =
            Linear::load(&ws.pp("att.value"), cfg.hidden_size, cfg.hidden_size, false)?;
        let time_mix_receptance = Linear::load(
            &ws.pp("att.receptance"),
            cfg.hidden_size,
            cfg.hidden_size,
            false,
        )?;
        let time_mix_output = Linear::load(
            &ws.pp("att.output"),
            cfg.hidden_size,
            cfg.hidden_size,
            false,
        )?;

        let channel_mix_key =
            Linear::load(&ws.pp("ffn.key"), cfg.hidden_size, cfg.hidden_size, false)?;
        let channel_mix_receptance = Linear::load(
            &ws.pp("ffn.receptance"),
            cfg.hidden_size,
            cfg.hidden_size,
            false,
        )?;
        let channel_mix_value =
            Linear::load(&ws.pp("ffn.value"), cfg.hidden_size, cfg.hidden_size, false)?;

        assert_weight_shape(&norm.weight, &[cfg.hidden_size], "ln_x")?;
        assert_weight_shape(
            &time_mix_key.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "att.key",
        )?;
        assert_weight_shape(
            &time_mix_value.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "att.value",
        )?;
        assert_weight_shape(
            &time_mix_receptance.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "att.receptance",
        )?;
        assert_weight_shape(
            &time_mix_output.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "att.output",
        )?;
        assert_weight_shape(
            &channel_mix_key.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "ffn.key",
        )?;
        assert_weight_shape(
            &channel_mix_receptance.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "ffn.receptance",
        )?;
        assert_weight_shape(
            &channel_mix_value.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "ffn.value",
        )?;

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

        let dev = RocmDevice::try_new(ordinal)?;
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
        let k_gpu = dev.from_cpu(
            &k.to_vec_f32()?,
            &Shape::new(vec![1, dim]),
            grim_tensor::DType::F32,
        )?;
        let v_gpu = dev.from_cpu(
            &v.to_vec_f32()?,
            &Shape::new(vec![1, dim]),
            grim_tensor::DType::F32,
        )?;
        let r_gpu = dev.from_cpu(
            &r.to_vec_f32()?,
            &Shape::new(vec![1, dim]),
            grim_tensor::DType::F32,
        )?;
        let w_gpu = dev.from_cpu(
            &att_out.to_vec_f32()?,
            &Shape::new(vec![1, dim]),
            grim_tensor::DType::F32,
        )?;

        let out_shape = Shape::new(vec![1, dim]);
        let (tm_out, _) = dev.rwkv_time_mix(
            x_gpu.as_ref(),
            w_gpu.as_ref(),
            k_gpu.as_ref(),
            v_gpu.as_ref(),
            r_gpu.as_ref(),
            1,
            dim,
            1,
            &out_shape,
        )?;
        let tm_data = tm_out.to_cpu_vec_f32()?;

        // Use time-mix output (tm_data) in the residual, not att_out.
        // [P1-32 fix: use tm_data in residual.]
        let x_res1 = add_tensors(x, &cpu_tensor(tm_data, Shape::new(vec![1, dim]))).map_err(grim_core::Error::Tensor)?;
        let x_res1_data = x_res1.to_vec_f32()?;

        // Channel-mix: project through key/receptance/value weights.
        // ffn_v must use its own weight (channel_mix_value), not ffn_k's.
        // [P1-32 fix: ffn_v uses channel_mix_value weight.]
        let ffn_k = self.channel_mix_key.forward(&x_res1)?;
        let ffn_r = self.channel_mix_receptance.forward(&x_res1)?;
        let ffn_v = self.channel_mix_value.forward(&x_res1)?;

        let x_res1_gpu = dev.from_cpu(
            &x_res1_data,
            &Shape::new(vec![1, dim]),
            grim_tensor::DType::F32,
        )?;
        let ffn_k_gpu = dev.from_cpu(
            &ffn_k.to_vec_f32()?,
            &Shape::new(vec![1, dim]),
            grim_tensor::DType::F32,
        )?;
        let ffn_r_gpu = dev.from_cpu(
            &ffn_r.to_vec_f32()?,
            &Shape::new(vec![1, dim]),
            grim_tensor::DType::F32,
        )?;
        let ffn_v_gpu = dev.from_cpu(
            &ffn_v.to_vec_f32()?,
            &Shape::new(vec![1, dim]),
            grim_tensor::DType::F32,
        )?;

        let (cm_out, _) = dev.rwkv_channel_mix(
            x_res1_gpu.as_ref(),
            ffn_k_gpu.as_ref(),
            ffn_r_gpu.as_ref(),
            ffn_v_gpu.as_ref(),
            1,
            dim,
            &out_shape,
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

        let k_vec = k.to_vec_f32()?;
        let v_vec = v.to_vec_f32()?;
        let r_vec = r.to_vec_f32()?;
        let dim = k_vec.len();

        let mut time_mix_in = vec![0.0f32; dim];
        for i in 0..dim {
            let sig_r = 1.0 / (1.0 + (-r_vec[i]).exp());
            time_mix_in[i] = sig_r * (k_vec[i] * v_vec[i]);
        }

        let att_out = self
            .time_mix_output
            .forward(&cpu_tensor(time_mix_in, Shape::new(vec![1, dim])))?;
        // Use time-mix output in residual.
        // [P1-32 fix: use tm_data in residual.]
        let x_res1 = add_tensors(x, &att_out).map_err(grim_core::Error::Tensor)?;

        let _ffn_k = self.channel_mix_key.forward(&x_res1)?;
        let ffn_r = self.channel_mix_receptance.forward(&x_res1)?;
        // ffn_v must use its own weight, not ffn_k's.
        // [P1-32 fix: ffn_v uses channel_mix_value weight.]
        let ffn_v = self.channel_mix_value.forward(&x_res1)?;

        let ffn_r_vec = ffn_r.to_vec_f32()?;
        let ffn_v_vec = ffn_v.to_vec_f32()?;
        let mut ffn_out = vec![0.0f32; ffn_v_vec.len()];
        for i in 0..ffn_v_vec.len() {
            let sig_r = 1.0 / (1.0 + (-ffn_r_vec[i]).exp());
            ffn_out[i] = sig_r * ffn_v_vec[i];
        }

        let ffn_t = cpu_tensor(ffn_out, Shape::new(vec![1, dim]));
        add_tensors(&x_res1, &ffn_t).map_err(grim_core::Error::Tensor)
    }
}

pub struct Rwkv {
    pub cfg: RwkvConfig,
    pub device: Device,
    /// Embedding table: [vocab_size, hidden_size] stored as flat Vec.
    /// Used as a gather (token_id -> row), NOT a Linear matrix multiply.
    pub emb: Vec<f32>,
    pub emb_shape: (usize, usize), // (vocab_size, hidden_size)
    pub layers: Vec<RwkvBlock>,
    pub ln_out: RmsNorm,
    pub head: Linear,
}

impl Rwkv {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: RwkvConfig, device: Device) -> Result<Self> {
        Self::load_tp(ws, cfg, device, ws.tp_config())
    }

    /// Tensor-parallel load entry for RWKV. RWKV is a recurrent (time-mix)
    /// model: like Mamba, the recurrent path has no row-parallel all-reduce
    /// semantics, so column/row sharding of the time-/channel-mix matrices
    /// would change the recurrence rather than parallelise a matmul. A safe
    /// `load_tp` needs a bespoke RWKV sharding plan. Refuses `world_size > 1`
    /// until then.
    pub fn load_tp(
        ws: &grim_nn::WeightSource<'_>,
        cfg: RwkvConfig,
        device: Device,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "RWKV",
            "the recurrent time/channel-mix path has no row-parallel all-reduce semantics; \
             sharding needs a bespoke plan",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        let emb_weight = ws.pp("emb").get(Shape::new(vec![cfg.vocab_size, cfg.hidden_size]), "weight")?
            .to_vec_f32()?;
        // RWKV embedding is [vocab_size, hidden_size] — use as a gather table,
        // NOT a Linear matrix multiply.
        // [P1-32 fix: emb is an embedding table, not a Linear.]
        if emb_weight.len() != cfg.vocab_size * cfg.hidden_size {
            return Err(Error::Shape(format!(
                "RWKV emb: expected {} elements, got {}",
                cfg.vocab_size * cfg.hidden_size,
                emb_weight.len()
            )));
        }
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(RwkvBlock::load(
                &ws.pp("blocks").pp(&i.to_string()),
                &cfg,
                device.clone(),
            )?);
        }
        let ln_out = RmsNorm::load(&ws.pp("ln_out"), cfg.hidden_size, cfg.rms_norm_eps as f32)?;
        let head = Linear::load(&ws.pp("head"), cfg.hidden_size, cfg.vocab_size, false)?;

        let vocab_size = cfg.vocab_size;
        let hidden_size = cfg.hidden_size;
        assert_weight_shape(&ln_out.weight, &[hidden_size], "ln_out")?;
        assert_weight_shape(&head.weight, &[vocab_size, hidden_size], "head")?;

        Ok(Self {
            cfg,
            device,
            emb: emb_weight,
            emb_shape: (vocab_size, hidden_size),
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
        let s = state
            .as_any_mut()
            .downcast_mut::<RwkvState>()
            .ok_or_else(|| Error::Session("downcast failed".into()))?;
        // Embedding gather: for each token ID, look up the corresponding row
        // in the embedding table. NOT a Linear matrix multiply.
        // [P1-32 fix: emb is an embedding gather, not Linear forward.]
        let input_ids = input.to_vec_f32()?;
        let emb_rows: Vec<Vec<f32>> = input_ids
            .iter()
            .map(|&id| {
                let idx = id as usize * self.emb_shape.1;
                self.emb[idx..idx + self.emb_shape.1].to_vec()
            })
            .collect();
        let emb_out = cpu_tensor(
            emb_rows.iter().flatten().cloned().collect::<Vec<f32>>(),
            Shape::new(vec![input_ids.len(), self.emb_shape.1]),
        );
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
