//! LOMO (Low-Memory Optimization) and AdaLomo optimizers.
//!
//! LOMO fuses backward computation with parameter updates to achieve zero
//! gradient memory overhead during LLM pretraining and full-parameter finetuning.
//! AdaLomo adds adaptive learning rate scaling with minimal second-moment state.

use crate::param::{ParamId, TrainableParam, TrainableParams};
use grim_format::train::TrainState;
use grim_tensor::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the LOMO optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LomoConfig {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub clip_grad_norm: Option<f32>,
}

impl Default for LomoConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.0,
            weight_decay: 0.0,
            clip_grad_norm: Some(1.0),
        }
    }
}

/// LOMO: Low-Memory Optimization.
/// Updates parameters in-place using SGD with optional momentum and weight decay.
#[derive(Debug, Clone)]
pub struct Lomo {
    pub config: LomoConfig,
    /// Momentum buffers per parameter: m_t = beta * m_{t-1} + g_t
    pub momentum_buffers: HashMap<ParamId, Vec<f32>>,
    pub step_count: usize,
}

impl Lomo {
    pub fn new(config: LomoConfig) -> Self {
        Self {
            config,
            momentum_buffers: HashMap::new(),
            step_count: 0,
        }
    }

    /// Perform in-place update on a parameter slice given its gradient slice.
    pub fn update_param(
        &mut self,
        id: ParamId,
        param_slice: &mut [f32],
        grad_slice: &[f32],
    ) -> Result<()> {
        if param_slice.len() != grad_slice.len() {
            return Err(Error::Shape(format!(
                "Lomo::update_param: param len {} != grad len {}",
                param_slice.len(),
                grad_slice.len()
            )));
        }

        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        let beta = self.config.momentum;

        if beta > 0.0 {
            let buf = self
                .momentum_buffers
                .entry(id)
                .or_insert_with(|| vec![0.0f32; param_slice.len()]);

            for i in 0..param_slice.len() {
                let mut g = grad_slice[i];
                if wd > 0.0 {
                    g += wd * param_slice[i];
                }
                buf[i] = beta * buf[i] + g;
                param_slice[i] -= lr * buf[i];
            }
        } else {
            for i in 0..param_slice.len() {
                let mut g = grad_slice[i];
                if wd > 0.0 {
                    g += wd * param_slice[i];
                }
                param_slice[i] -= lr * g;
            }
        }

        Ok(())
    }

    /// Single-param step (fused streaming update).
    pub fn step_param(&mut self, id: ParamId, param: &mut TrainableParam) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let mut data = param.data.to_vec_f32()?;
        let grad = param.grad().to_vec_f32()?;
        self.update_param(id, &mut data, &grad)?;
        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape().clone();
        let updated = dev.from_cpu(&data, &shape, param.data.dtype())?;
        param.data = grim_tensor::Tensor::new(
            std::sync::Arc::from(updated),
            shape,
            param.data.dtype(),
            param.data.provenance().clone(),
            param.data.device().clone(),
        );
        param.zero_grad()?;
        Ok(())
    }

    /// Perform a full step on all parameters in `TrainableParams`.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let mut data = param.data.to_vec_f32()?;
            let grad = param.grad().to_vec_f32()?;
            self.update_param(*id, &mut data, &grad)?;
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape().clone();
            let updated = dev.from_cpu(&data, &shape, param.data.dtype())?;
            param.data = grim_tensor::Tensor::new(
                std::sync::Arc::from(updated),
                shape,
                param.data.dtype(),
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
            param.zero_grad()?;
        }

        Ok(())
    }

    /// Export optimizer state to `TrainState`.
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        crate::adamw::save_param_data_only(params, self.step_count)
    }

    /// Restore optimizer state from `TrainState`.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        crate::adamw::load_param_data_only(params, state)
    }
}

/// Configuration for AdaLomo optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaLomoConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub clip_grad_norm: Option<f32>,
}

impl Default for AdaLomoConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            clip_grad_norm: Some(1.0),
        }
    }
}

/// AdaLomo: Adaptive Low-Memory Optimization.
/// Tracks running variance v_t to provide adaptive per-element updates with minimal footprint.
#[derive(Debug, Clone)]
pub struct AdaLomo {
    pub config: AdaLomoConfig,
    /// Second moment variance buffers: v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    pub exp_avg_sq: HashMap<ParamId, Vec<f32>>,
    pub step_count: usize,
}

impl AdaLomo {
    pub fn new(config: AdaLomoConfig) -> Self {
        Self {
            config,
            exp_avg_sq: HashMap::new(),
            step_count: 0,
        }
    }

    /// Update a single parameter slice given its gradient.
    pub fn update_param(
        &mut self,
        id: ParamId,
        param_slice: &mut [f32],
        grad_slice: &[f32],
    ) -> Result<()> {
        if param_slice.len() != grad_slice.len() {
            return Err(Error::Shape(format!(
                "AdaLomo::update_param: param len {} != grad len {}",
                param_slice.len(),
                grad_slice.len()
            )));
        }

        let lr = self.config.lr;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;

        let v_buf = self
            .exp_avg_sq
            .entry(id)
            .or_insert_with(|| vec![0.0f32; param_slice.len()]);

        let step = self.step_count.max(1) as f32;
        let bias_correction2 = 1.0 - beta2.powf(step);

        for i in 0..param_slice.len() {
            let g = grad_slice[i];
            v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * (g * g);

            let v_hat = (v_buf[i] / bias_correction2).max(0.0);
            let denom = v_hat.sqrt() + eps;
            let step_update = g / denom;

            if wd > 0.0 {
                param_slice[i] -= lr * wd * param_slice[i];
            }
            param_slice[i] -= lr * step_update;
        }

        Ok(())
    }

    /// Single-param step (fused streaming update).
    pub fn step_param(&mut self, id: ParamId, param: &mut TrainableParam) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let mut data = param.data.to_vec_f32()?;
        let grad = param.grad().to_vec_f32()?;
        self.update_param(id, &mut data, &grad)?;
        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape().clone();
        let updated = dev.from_cpu(&data, &shape, param.data.dtype())?;
        param.data = grim_tensor::Tensor::new(
            std::sync::Arc::from(updated),
            shape,
            param.data.dtype(),
            param.data.provenance().clone(),
            param.data.device().clone(),
        );
        param.zero_grad()?;
        Ok(())
    }

    /// Perform a full step on all parameters in `TrainableParams`.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let mut data = param.data.to_vec_f32()?;
            let grad = param.grad().to_vec_f32()?;
            self.update_param(*id, &mut data, &grad)?;
            let dev = crate::pick_device_for_tensor(&param.data);
            let shape = param.data.shape().clone();
            let updated = dev.from_cpu(&data, &shape, param.data.dtype())?;
            param.data = grim_tensor::Tensor::new(
                std::sync::Arc::from(updated),
                shape,
                param.data.dtype(),
                param.data.provenance().clone(),
                param.data.device().clone(),
            );
            param.zero_grad()?;
        }

        Ok(())
    }

    /// Export state to `TrainState`.
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        crate::adamw::save_param_data_only(params, self.step_count)
    }

    /// Restore state from `TrainState`.
    pub fn load_from_train_state(
        &mut self,
        params: &mut TrainableParams,
        state: &TrainState,
    ) -> Result<()> {
        self.step_count = state.step as usize;
        crate::adamw::load_param_data_only(params, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::injection::LoRAInjectionPoint;

    #[test]
    fn test_lomo_analytical_step() {
        let mut lomo = Lomo::new(LomoConfig {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.0,
            clip_grad_norm: None,
        });

        let mut param = vec![1.0, 2.0, 3.0];
        let grad = vec![0.5, -0.5, 1.0];

        let id = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);
        lomo.update_param(id, &mut param, &grad).unwrap();

        assert!((param[0] - 0.95).abs() < 1e-6);
        assert!((param[1] - 2.05).abs() < 1e-6);
        assert!((param[2] - 2.90).abs() < 1e-6);
    }

    #[test]
    fn test_adalomo_convergence() {
        let mut adalomo = AdaLomo::new(AdaLomoConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            clip_grad_norm: None,
        });

        let mut param = vec![10.0f32];
        let id = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);

        for _ in 0..50 {
            let grad = vec![param[0]];
            adalomo.step_count += 1;
            adalomo.update_param(id, &mut param, &grad).unwrap();
        }

        assert!(
            param[0].abs() < 6.0,
            "param should decrease toward 0: got {}",
            param[0]
        );
    }
}
