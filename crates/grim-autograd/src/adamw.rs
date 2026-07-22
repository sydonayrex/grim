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
#[derive(Debug, Clone)]
pub struct AdamW {
    pub config: AdamWConfig,
    pub step_count: usize,
    /// 1st moment vector (m) per trainable parameter ID.
    pub m: HashMap<ParamId, Vec<f32>>,
    /// 2nd moment vector (v) per trainable parameter ID.
    pub v: HashMap<ParamId, Vec<f32>>,
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
            let data_vec = param.data.to_vec_f32()?;
            let grad_vec = param.grad().to_vec_f32()?;
            let elem_count = data_vec.len();

            let m_vec = self.m.entry(*id).or_insert_with(|| vec![0.0f32; elem_count]);
            let v_vec = self.v.entry(*id).or_insert_with(|| vec![0.0f32; elem_count]);

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

            let dev = crate::pick_device_for_tensor(&param.data);
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

            if let Some(m_vec) = self.m.get(id) {
                let bytes: Vec<u8> = m_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                let blob_name = format!("opt_m_{}_{}_{}", id.layer_idx, id.adapter_id, if id.is_a { "a" } else { "b" });
                state.add_blob(blob_name, shape.clone(), bytes);
            }

            if let Some(v_vec) = self.v.get(id) {
                let bytes: Vec<u8> = v_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                let blob_name = format!("opt_v_{}_{}_{}", id.layer_idx, id.adapter_id, if id.is_a { "a" } else { "b" });
                state.add_blob(blob_name, shape, bytes);
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
                self.m.insert(*id, bytes_to_f32_vec(&blob.data)?);
            }

            if let Some(blob) = state.blobs.get(&v_key) {
                self.v.insert(*id, bytes_to_f32_vec(&blob.data)?);
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
    use grim_tensor::Shape;

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
        assert!(data[0] < 1.0);
        assert!(data[1] < 2.0);
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
        assert_eq!(opt2.m.get(&id).unwrap(), opt.m.get(&id).unwrap());
    }
}
