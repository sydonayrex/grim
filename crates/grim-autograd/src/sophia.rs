//! Sophia (Second-order Clipped Stochastic Optimization).
//!
//! Uses a diagonal Hessian estimate with elementwise clipping:
//! theta_{t+1} = theta_t - eta * clip(m_t / max(h_t, gamma), rho) - eta * lambda * theta_t.

use crate::param::{ParamId, TrainableParam, TrainableParams};
use grim_format::train::TrainState;
use grim_tensor::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Sophia optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SophiaConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub rho: f32,
    pub gamma: f32,
    pub weight_decay: f32,
    pub hessian_update_interval: usize,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.96,
            beta2: 0.99,
            rho: 0.04,
            gamma: 1e-2,
            weight_decay: 0.1,
            hessian_update_interval: 10,
        }
    }
}

/// Sophia Optimizer state per parameter.
#[derive(Debug, Clone)]
pub struct SophiaState {
    pub exp_avg: Vec<f32>,
    pub hessian: Vec<f32>,
}

/// Sophia Optimizer.
#[derive(Debug, Clone)]
pub struct Sophia {
    pub config: SophiaConfig,
    pub states: HashMap<ParamId, SophiaState>,
    pub step_count: usize,
}

impl Sophia {
    pub fn new(config: SophiaConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            step_count: 0,
        }
    }

    /// Update the diagonal Hessian estimate for a parameter.
    pub fn update_hessian(&mut self, id: ParamId, hessian_diag: &[f32]) -> Result<()> {
        let beta2 = self.config.beta2;
        let state = self.states.entry(id).or_insert_with(|| SophiaState {
            exp_avg: vec![0.0f32; hessian_diag.len()],
            hessian: vec![0.0f32; hessian_diag.len()],
        });

        if state.hessian.len() != hessian_diag.len() {
            return Err(Error::Shape(format!(
                "Sophia::update_hessian: state len {} != diag len {}",
                state.hessian.len(),
                hessian_diag.len()
            )));
        }

        for i in 0..hessian_diag.len() {
            state.hessian[i] = beta2 * state.hessian[i] + (1.0 - beta2) * hessian_diag[i];
        }

        Ok(())
    }

    /// Step update a single parameter with current gradient and momentum clipping.
    pub fn update_param(&mut self, id: ParamId, param: &mut [f32], grad: &[f32]) -> Result<()> {
        if param.len() != grad.len() {
            return Err(Error::Shape(format!(
                "Sophia::update_param: param len {} != grad len {}",
                param.len(),
                grad.len()
            )));
        }

        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let rho = self.config.rho;
        let gamma = self.config.gamma;
        let wd = self.config.weight_decay;

        let state = self.states.entry(id).or_insert_with(|| SophiaState {
            exp_avg: vec![0.0f32; param.len()],
            hessian: vec![0.0f32; param.len()],
        });

        for i in 0..param.len() {
            let g = grad[i];
            state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g;

            let h = state.hessian[i].max(gamma);
            let raw_step = state.exp_avg[i] / h;
            let clipped_step = raw_step.clamp(-rho, rho);

            if wd > 0.0 {
                param[i] -= lr * wd * param[i];
            }
            param[i] -= lr * clipped_step;
        }

        Ok(())
    }

    /// Single param step.
    pub fn step_param(&mut self, id: ParamId, param: &mut TrainableParam) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let mut data = param.data.to_vec_f32()?;
        let grad = param.grad().to_vec_f32()?;
        if let Some(state) = self.states.get_mut(&id) {
            if state.hessian.iter().all(|&x| x == 0.0) {
                for (h, &gv) in state.hessian.iter_mut().zip(grad.iter()) {
                    *h = gv.abs();
                }
            }
        }
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

    /// Step over TrainableParams with gradients.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let mut data = param.data.to_vec_f32()?;
            let grad = param.grad().to_vec_f32()?;
            if let Some(state) = self.states.get_mut(id) {
                if state.hessian.iter().all(|&x| x == 0.0) {
                    for (h, &gv) in state.hessian.iter_mut().zip(grad.iter()) {
                        *h = gv.abs();
                    }
                }
            }
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

    /// Export to TrainState.
    pub fn save_to_train_state(&self, params: &TrainableParams) -> TrainState {
        crate::adamw::save_param_data_only(params, self.step_count)
    }

    /// Restore from TrainState.
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
    fn test_sophia_clipping_bound() {
        let mut sophia = Sophia::new(SophiaConfig {
            lr: 1.0,
            beta1: 0.0,
            beta2: 0.99,
            rho: 0.05,
            gamma: 1e-4,
            weight_decay: 0.0,
            hessian_update_interval: 10,
        });

        let mut param = vec![1.0];
        let grad = vec![1000.0];
        let id = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);

        sophia.update_hessian(id, &[0.001]).unwrap();
        sophia.update_param(id, &mut param, &grad).unwrap();

        assert!(
            (param[0] - 0.95).abs() < 1e-5,
            "param should be clipped to 0.95, got {}",
            param[0]
        );
    }
}
