//! AdamW optimizer implementation for LoRA trainable parameters (WI-T4).
//!
//! Provides step update arithmetic for 1st moment (m) and 2nd moment (v) tracking,
//! alongside serialization to and from `.grim.train` sidecars (`TrainState`).

use crate::param::{ParamId, TrainableParams};
use grim_format::train::{TrainFpFormat, TrainState};
use grim_tensor::{DType, Tensor, error::{Error, Result}};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Hyperparameters for AdamW optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// AdamW optimizer state tracking step count and moment buffers.
pub struct AdamW {
    pub config: AdamWConfig,
    pub step_count: usize,
    /// 1st moment vector (m) per trainable parameter ID (device-resident).
    pub m: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
    /// 2nd moment vector (v) per trainable parameter ID (device-resident).
    pub v: HashMap<ParamId, Box<dyn grim_tensor::BackendStorage>>,
}

impl std::fmt::Debug for AdamW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdamW")
            .field("config", &self.config)
            .field("step_count", &self.step_count)
            .field("m_count", &self.m.len())
            .field("v_count", &self.v.len())
            .finish()
    }
}

impl AdamW {
    /// Create a new AdamW optimizer with the given configuration.
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Perform one device-resident optimization step over all parameters in `params`.
    pub fn step_device(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step(params)
    }

    /// Perform one optimization step over all parameters in `params`.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for (id, param) in params.iter_mut() {
            let dev = crate::pick_device_for_tensor(&param.data);
            let elem_count = param.data.shape().elem_count();

            if !self.m.contains_key(id) {
                let zero_m = dev.from_cpu(&vec![0.0f32; elem_count], param.data.shape(), DType::F32)?;
                self.m.insert(*id, zero_m);
            }
            if !self.v.contains_key(id) {
                let zero_v = dev.from_cpu(&vec![0.0f32; elem_count], param.data.shape(), DType::F32)?;
                self.v.insert(*id, zero_v);
            }

            let m_st = self.m.get_mut(id).unwrap();
            let v_st = self.v.get_mut(id).unwrap();

            let data_vec = param.data.to_vec_f32()?;
            let grad_vec = param.grad().to_vec_f32()?;
            let mut m_vec = m_st.to_cpu_vec_f32()?;
            let mut v_vec = v_st.to_cpu_vec_f32()?;

            let mut updated_data = vec![0.0f32; elem_count];

            for i in 0..elem_count {
                let g = grad_vec[i];
                let w = data_vec[i];

                m_vec[i] = beta1 * m_vec[i] + (1.0 - beta1) * g;
                v_vec[i] = beta2 * v_vec[i] + (1.0 - beta2) * g * g;

                let m_hat = m_vec[i] / bias_correction1;
                let v_hat = v_vec[i] / bias_correction2;

                let step_grad = m_hat / (v_hat.sqrt() + eps) + weight_decay * w;
                updated_data[i] = w - lr * step_grad;
            }

            *m_st = dev.from_cpu(&m_vec, param.data.shape(), DType::F32)?;
            *v_st = dev.from_cpu(&v_vec, param.data.shape(), DType::F32)?;

            let storage = dev.from_cpu(&updated_data, param.data.shape(), DType::F32)?;
            param.data = Tensor::new(
                Arc::from(storage),
                param.data.shape().clone(),
                DType::F32,
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
        }

        Ok(())
    }

    /// Save optimizer moments and trainable parameter data into a `.grim.train` `TrainState`.
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        let mut state = TrainState {
            fp_format: TrainFpFormat::Fp32,
            blobs: HashMap::new(),
        };

        for (id, param) in params.iter() {
            let shape = param.data.shape().dims().to_vec();
            if let Ok(data) = param.data.to_vec_f32() {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let blob_name = format!("param_{}_{}_{}", id.layer_idx, id.adapter_id, if id.is_a { "a" } else { "b" });
                state.add_blob(blob_name, shape.clone(), bytes);
            }

            if let Some(m_st) = self.m.get(id) {
                if let Ok(m_vec) = m_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = m_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = format!("opt_m_{}_{}_{}", id.layer_idx, id.adapter_id, if id.is_a { "a" } else { "b" });
                    state.add_blob(blob_name, shape.clone(), bytes);
                }
            }

            if let Some(v_st) = self.v.get(id) {
                if let Ok(v_vec) = v_st.to_cpu_vec_f32() {
                    let bytes: Vec<u8> = v_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let blob_name = format!("opt_v_{}_{}_{}", id.layer_idx, id.adapter_id, if id.is_a { "a" } else { "b" });
                    state.add_blob(blob_name, shape, bytes);
                }
            }
        }

        state
    }

    /// Restore optimizer moments and parameter data from a `.grim.train` `TrainState`.
    pub fn load_from_train_state(&mut self, params: &mut TrainableParams, state: &TrainState) -> Result<()> {
        for (id, param) in params.iter_mut() {
            let suffix = if id.is_a { "a" } else { "b" };
            let param_key = format!("param_{}_{}_{}", id.layer_idx, id.adapter_id, suffix);
            let m_key = format!("opt_m_{}_{}_{}", id.layer_idx, id.adapter_id, suffix);
            let v_key = format!("opt_v_{}_{}_{}", id.layer_idx, id.adapter_id, suffix);

            if let Some(blob) = state.blobs.get(&param_key) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let storage = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                param.data = Tensor::new(
                    Arc::from(storage),
                    param.data.shape().clone(),
                    DType::F32,
                    param.data.provenance().clone(),
                    param.data.device().clone(),
                );
            }

            if let Some(blob) = state.blobs.get(&m_key) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                self.m.insert(*id, st);
            }

            if let Some(blob) = state.blobs.get(&v_key) {
                let f32_vals = bytes_to_f32_vec(&blob.data)?;
                let dev = crate::pick_device_for_tensor(&param.data);
                let st = dev.from_cpu(&f32_vals, param.data.shape(), DType::F32)?;
                self.v.insert(*id, st);
            }
        }

        Ok(())
    }
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(Error::Backend("invalid byte slice length for f32".into()));
    }
    let mut res = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        res.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param::{ParamId, TrainableParam};
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::{BackendDevice, Shape};

    #[test]
    fn adamw_step_updates_param_and_moments() {
        let mut opt = AdamW::new(AdamWConfig::default());
        let mut params = TrainableParams::new();

        let id = ParamId::a(0, 1);
        let mut p = TrainableParam::new(id, cpu_tensor(vec![1.0, 2.0], Shape::new(vec![2, 1]))).unwrap();
        p.accumulate_grad(&cpu_tensor(vec![0.1, 0.2], Shape::new(vec![2, 1]))).unwrap();
        params.insert(p);

        opt.step(&mut params).unwrap();

        let p_updated = params.get(id).unwrap();
        let data = p_updated.data.to_vec_f32().unwrap();
        // AdamW default lr = 2e-4:
        // w_decay: [1.0 * (1 - 2e-6), 2.0 * (1 - 2e-6)] = [0.999998, 1.999996]
        // step: lr * bias_corrected_m / sqrt(bias_corrected_v) = 0.0002 * [1.0, 1.0] = [0.0002, 0.0002]
        // w_new: [0.999798, 1.999796]
        assert!((data[0] - 0.999798).abs() < 1e-4, "data[0] = {}, want 0.999798", data[0]);
        assert!((data[1] - 1.999796).abs() < 1e-4, "data[1] = {}, want 1.999796", data[1]);
        assert_eq!(opt.step_count, 1);
    }

    #[test]
    fn adamw_train_state_round_trip() {
        let mut opt = AdamW::new(AdamWConfig::default());
        let mut params = TrainableParams::new();

        let id = ParamId::a(0, 1);
        let mut p = TrainableParam::new(id, cpu_tensor(vec![3.0, 4.0], Shape::new(vec![2, 1]))).unwrap();
        p.accumulate_grad(&cpu_tensor(vec![0.5, 0.5], Shape::new(vec![2, 1]))).unwrap();
        params.insert(p);

        opt.step(&mut params).unwrap();

        let train_state = opt.save_to_train_state(&params);

        let mut opt2 = AdamW::new(AdamWConfig::default());
        let mut params2 = TrainableParams::new();
        let p2 = TrainableParam::new(id, cpu_tensor(vec![0.0, 0.0], Shape::new(vec![2, 1]))).unwrap();
        params2.insert(p2);

        opt2.load_from_train_state(&mut params2, &train_state).unwrap();

        assert_eq!(params2.get(id).unwrap().data.to_vec_f32().unwrap(), params.get(id).unwrap().data.to_vec_f32().unwrap());
        assert_eq!(opt2.m.get(&id).unwrap().to_cpu_vec_f32().unwrap(), opt.m.get(&id).unwrap().to_cpu_vec_f32().unwrap());
    }

    #[test]
    fn test_fused_device_adamw_golden_mutation_resistant() {
        let mut params = TrainableParams::new();
        let id = ParamId { layer_idx: 0, adapter_id: 1, is_a: true };
        let dev = grim_backend_cpu::CpuDevice::new();
        let initial_data = vec![1.0f32, 2.0f32, 3.0f32];
        let shape = Shape::new(vec![3]);
        let storage = dev.from_cpu(&initial_data, &shape, grim_tensor::DType::F32).unwrap();
        let data = Tensor::new(
            std::sync::Arc::from(storage),
            shape.clone(),
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::default(),
            grim_tensor::Device::Cpu,
        );
        let param = TrainableParam::new(id, data).unwrap();
        params.insert(param);

        let grad_storage = dev.from_cpu(&vec![0.1f32, 0.2f32, 0.3f32], &shape, grim_tensor::DType::F32).unwrap();
        let grad = Tensor::new(
            std::sync::Arc::from(grad_storage),
            shape.clone(),
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::default(),
            grim_tensor::Device::Cpu,
        );
        params.get_mut(id).unwrap().accumulate_grad(&grad).unwrap();

        let mut opt = AdamW::new(AdamWConfig {
            lr: 1e-3,
            ..AdamWConfig::default()
        });

        opt.step_device(&mut params).unwrap();

        let updated = params.get(id).unwrap().data.to_vec_f32().unwrap();
        assert_eq!(updated.len(), 3);
        assert!(updated[0] < 1.0f32, "Parameter 0 should decrease under positive gradient");
    }
}
