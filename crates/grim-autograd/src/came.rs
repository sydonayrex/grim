//! CAME (Confidence-guided Adaptive Memory Efficient Optimization).
//!
//! Factored second-moment matrix tracking (row & column sums) with an
//! instability/confidence matrix for stable and memory-efficient LLM training.

use crate::param::{ParamId, TrainableParam, TrainableParams};
use grim_format::train::TrainState;
use grim_tensor::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for CAME optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub beta3: f32,
    pub eps1: f32,
    pub eps2: f32,
    pub weight_decay: f32,
    pub clip_threshold: f32,
}

impl Default for CameConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.9999,
            eps1: 1e-30,
            eps2: 1e-16,
            weight_decay: 0.01,
            clip_threshold: 1.0,
        }
    }
}

/// Factored 2D matrix state for CAME.
#[derive(Debug, Clone)]
pub struct CameMatrixState {
    pub exp_avg_sq_row: Vec<f32>,
    pub exp_avg_sq_col: Vec<f32>,
    pub exp_avg_res: Vec<f32>,
    pub exp_avg: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

/// 1D vector state fallback for biases and layer norms.
#[derive(Debug, Clone)]
pub struct CameVectorState {
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Vec<f32>,
    pub exp_avg_res: Vec<f32>,
}

/// CAME Optimizer.
#[derive(Debug, Clone)]
pub struct Came {
    pub config: CameConfig,
    pub matrix_states: HashMap<ParamId, CameMatrixState>,
    pub vector_states: HashMap<ParamId, CameVectorState>,
    pub step_count: usize,
}

impl Came {
    pub fn new(config: CameConfig) -> Self {
        Self {
            config,
            matrix_states: HashMap::new(),
            vector_states: HashMap::new(),
            step_count: 0,
        }
    }

    /// Perform update on a 2D matrix parameter `(rows, cols)`.
    pub fn update_matrix(
        &mut self,
        id: ParamId,
        param: &mut [f32],
        grad: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if param.len() != rows * cols || grad.len() != rows * cols {
            return Err(Error::Shape(format!(
                "Came::update_matrix: expected size {}x{} = {}, got param {}, grad {}",
                rows,
                cols,
                rows * cols,
                param.len(),
                grad.len()
            )));
        }

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let beta3 = self.config.beta3;
        let eps1 = self.config.eps1;
        let eps2 = self.config.eps2;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;

        let state = self
            .matrix_states
            .entry(id)
            .or_insert_with(|| CameMatrixState {
                exp_avg_sq_row: vec![0.0f32; rows],
                exp_avg_sq_col: vec![0.0f32; cols],
                exp_avg_res: vec![0.0f32; rows * cols],
                exp_avg: vec![0.0f32; rows * cols],
                rows,
                cols,
            });

        // 1. Factored second moments
        let mut r_grad = vec![0.0f32; rows];
        let mut c_grad = vec![0.0f32; cols];

        for r in 0..rows {
            let mut sum_sq = 0.0f32;
            for c in 0..cols {
                let g = grad[r * cols + c];
                sum_sq += g * g;
            }
            r_grad[r] = sum_sq / (cols as f32);
        }

        for c in 0..cols {
            let mut sum_sq = 0.0f32;
            for r in 0..rows {
                let g = grad[r * cols + c];
                sum_sq += g * g;
            }
            c_grad[c] = sum_sq / (rows as f32);
        }

        for (r, slot) in state.exp_avg_sq_row.iter_mut().enumerate() {
            *slot = beta2 * *slot + (1.0 - beta2) * r_grad[r];
        }
        for (c, slot) in state.exp_avg_sq_col.iter_mut().enumerate() {
            *slot = beta2 * *slot + (1.0 - beta2) * c_grad[c];
        }

        let mean_r: f32 = (state.exp_avg_sq_row.iter().sum::<f32>() / (rows as f32)).max(1e-12);

        // 2. Factored variance & update matrix M_t
        let mut m_mat = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let r_val = state.exp_avg_sq_row[r];
            for c in 0..cols {
                let c_val = state.exp_avg_sq_col[c];
                let v = (r_val * c_val / mean_r).max(0.0);
                let g = grad[r * cols + c];
                let idx = r * cols + c;
                m_mat[idx] = g / (v.sqrt() + eps1);
            }
        }

        // 3. Confidence matrix U_t on residual (M_t - exp_avg_res)
        for i in 0..rows * cols {
            let diff = m_mat[i] - state.exp_avg[i];
            state.exp_avg_res[i] = beta3 * state.exp_avg_res[i] + (1.0 - beta3) * (diff * diff);

            state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * m_mat[i];

            let u_hat = state.exp_avg_res[i].max(0.0).sqrt() + eps2;
            let step_m = state.exp_avg[i] / u_hat;

            if wd > 0.0 {
                param[i] -= lr * wd * param[i];
            }
            param[i] -= lr * step_m;
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
        let len = data.len();
        let (r, c) = if len >= 64 && len % 32 == 0 {
            (len / 32, 32)
        } else if len >= 16 && len % 4 == 0 {
            (len / 4, 4)
        } else {
            (len, 1)
        };

        if c > 1 {
            self.update_matrix(id, &mut data, &grad, r, c)?;
        } else {
            let state = self
                .vector_states
                .entry(id)
                .or_insert_with(|| CameVectorState {
                    exp_avg: vec![0.0f32; len],
                    exp_avg_sq: vec![0.0f32; len],
                    exp_avg_res: vec![0.0f32; len],
                });
            let lr = self.config.lr;
            let beta1 = self.config.beta1;
            let beta2 = self.config.beta2;
            let eps1 = self.config.eps1;

            for i in 0..len {
                let grad_val = grad[i];
                state.exp_avg_sq[i] =
                    beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * (grad_val * grad_val);
                let denom = state.exp_avg_sq[i].sqrt() + eps1;
                state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * (grad_val / denom);
                data[i] -= lr * state.exp_avg[i];
            }
        }

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

    /// Step over TrainableParams.
    pub fn step(&mut self, params: &mut TrainableParams) -> Result<()> {
        self.step_count += 1;

        for (id, param) in params.iter_mut() {
            if param.is_frozen() {
                param.zero_grad()?;
                continue;
            }
            let mut data = param.data.to_vec_f32()?;
            let grad = param.grad().to_vec_f32()?;
            let len = data.len();
            let (r, c) = if len >= 64 && len % 32 == 0 {
                (len / 32, 32)
            } else if len >= 16 && len % 4 == 0 {
                (len / 4, 4)
            } else {
                (len, 1)
            };

            if c > 1 {
                self.update_matrix(*id, &mut data, &grad, r, c)?;
            } else {
                let state = self
                    .vector_states
                    .entry(*id)
                    .or_insert_with(|| CameVectorState {
                        exp_avg: vec![0.0f32; len],
                        exp_avg_sq: vec![0.0f32; len],
                        exp_avg_res: vec![0.0f32; len],
                    });
                let lr = self.config.lr;
                let beta1 = self.config.beta1;
                let beta2 = self.config.beta2;
                let eps1 = self.config.eps1;

                for i in 0..len {
                    let grad_val = grad[i];
                    state.exp_avg_sq[i] =
                        beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * (grad_val * grad_val);
                    let denom = state.exp_avg_sq[i].sqrt() + eps1;
                    state.exp_avg[i] =
                        beta1 * state.exp_avg[i] + (1.0 - beta1) * (grad_val / denom);
                    data[i] -= lr * state.exp_avg[i];
                }
            }

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
    fn test_came_matrix_update_step() {
        let mut came = Came::new(CameConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            beta3: 0.999,
            eps1: 1e-12,
            eps2: 1e-8,
            weight_decay: 0.0,
            clip_threshold: 1.0,
        });

        let mut param = vec![1.0; 8];
        let grad = vec![0.5; 8];
        let id = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);

        came.update_matrix(id, &mut param, &grad, 2, 4).unwrap();

        for &val in &param {
            assert!(val < 1.0, "parameter should decrease from 1.0, got {val}");
        }
    }
}
