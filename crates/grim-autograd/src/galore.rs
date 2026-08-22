//! GaLore (Gradient Low-Rank Projection) Optimizer.
//!
//! Projects full-rank matrix gradients G in R^{m x n} into low-rank subspace
//! G_{low} = P^T G (r << min(m, n)), maintaining AdamW momentum/variance states
//! purely in the low-rank subspace and reconstructing updates via alpha * P * U.

use crate::param::{ParamId, TrainableParam, TrainableParams};
use grim_format::train::TrainState;
use grim_tensor::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for GaLore optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaLoreConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub rank: usize,
    pub update_proj_gap: usize,
    pub scale: f32,
}

impl Default for GaLoreConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            rank: 16,
            update_proj_gap: 200,
            scale: 1.0,
        }
    }
}

/// 2D Matrix GaLore projection state.
#[derive(Debug, Clone)]
pub struct GaLoreMatrixState {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    /// Orthogonal projection matrix P: (rows, rank)
    pub proj_p: Vec<f32>,
    /// Low-rank 1st moment: (rank, cols)
    pub exp_avg: Vec<f32>,
    /// Low-rank 2nd moment: (rank, cols)
    pub exp_avg_sq: Vec<f32>,
    pub step: usize,
}

/// Standalone GaLore Optimizer.
#[derive(Debug, Clone)]
pub struct GaLoreOptimizer {
    pub config: GaLoreConfig,
    pub matrix_states: HashMap<ParamId, GaLoreMatrixState>,
    pub vector_states: HashMap<ParamId, (Vec<f32>, Vec<f32>)>,
    pub step_count: usize,
}

impl GaLoreOptimizer {
    pub fn new(config: GaLoreConfig) -> Self {
        Self {
            config,
            matrix_states: HashMap::new(),
            vector_states: HashMap::new(),
            step_count: 0,
        }
    }

    /// Update projection matrix P via approximate truncated SVD / power iteration on G.
    fn refresh_subspace(p_mat: &mut [f32], grad: &[f32], rows: usize, cols: usize, rank: usize) {
        let mut ggt = vec![0.0f32; rows * rows];
        for i in 0..rows {
            for j in 0..=i {
                let mut sum = 0.0f32;
                for k in 0..cols {
                    sum += grad[i * cols + k] * grad[j * cols + k];
                }
                ggt[i * rows + j] = sum;
                ggt[j * rows + i] = sum;
            }
        }

        for c in 0..rank {
            for r in 0..rows {
                p_mat[r * rank + c] = if r == c % rows { 1.0 } else { 0.01 };
            }
        }

        for _iter in 0..3 {
            let mut next_p = vec![0.0f32; rows * rank];
            for r in 0..rows {
                for c in 0..rank {
                    let mut sum = 0.0f32;
                    for k in 0..rows {
                        sum += ggt[r * rows + k] * p_mat[k * rank + c];
                    }
                    next_p[r * rank + c] = sum;
                }
            }

            for c in 0..rank {
                for prev in 0..c {
                    let mut dot = 0.0f32;
                    for r in 0..rows {
                        dot += next_p[r * rank + c] * next_p[r * rank + prev];
                    }
                    for r in 0..rows {
                        next_p[r * rank + c] -= dot * next_p[r * rank + prev];
                    }
                }
                let mut norm_sq = 0.0f32;
                for r in 0..rows {
                    norm_sq += next_p[r * rank + c] * next_p[r * rank + c];
                }
                let norm = norm_sq.sqrt().max(1e-12);
                for r in 0..rows {
                    p_mat[r * rank + c] = next_p[r * rank + c] / norm;
                }
            }
        }
    }

    /// Step update on a 2D matrix parameter with low-rank projection.
    pub fn update_matrix(
        &mut self,
        id: ParamId,
        param: &mut [f32],
        grad: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let rank = self.config.rank.min(rows).min(cols).max(1);
        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;
        let gap = self.config.update_proj_gap.max(1);
        let alpha = (self.config.scale / (rank as f32)).max(1e-4);

        let state = self.matrix_states.entry(id).or_insert_with(|| {
            let mut p_mat = vec![0.0f32; rows * rank];
            Self::refresh_subspace(&mut p_mat, grad, rows, cols, rank);
            GaLoreMatrixState {
                rows,
                cols,
                rank,
                proj_p: p_mat,
                exp_avg: vec![0.0f32; rank * cols],
                exp_avg_sq: vec![0.0f32; rank * cols],
                step: 0,
            }
        });

        state.step += 1;

        if state.step % gap == 0 {
            Self::refresh_subspace(&mut state.proj_p, grad, rows, cols, rank);
        }

        // 1. Project gradient to low-rank subspace: G_low = P^T * G in R^{rank x cols}
        let mut g_low = vec![0.0f32; rank * cols];
        for r_idx in 0..rank {
            for c_idx in 0..cols {
                let mut sum = 0.0f32;
                for row in 0..rows {
                    sum += state.proj_p[row * rank + r_idx] * grad[row * cols + c_idx];
                }
                g_low[r_idx * cols + c_idx] = sum;
            }
        }

        // 2. Low-rank AdamW step
        let step_f = state.step as f32;
        let bc1 = 1.0 - beta1.powf(step_f);
        let bc2 = 1.0 - beta2.powf(step_f);

        let mut u_low = vec![0.0f32; rank * cols];
        for i in 0..rank * cols {
            let g = g_low[i];
            state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g;
            state.exp_avg_sq[i] = beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * (g * g);

            let m_hat = state.exp_avg[i] / bc1;
            let v_hat = (state.exp_avg_sq[i] / bc2).max(0.0);
            u_low[i] = m_hat / (v_hat.sqrt() + eps);
        }

        // 3. Project update back to full-rank: Delta = alpha * P * U_low in R^{rows x cols}
        for r in 0..rows {
            for c in 0..cols {
                let mut recon = 0.0f32;
                for r_idx in 0..rank {
                    recon += state.proj_p[r * rank + r_idx] * u_low[r_idx * cols + c];
                }
                let idx = r * cols + c;
                if wd > 0.0 {
                    param[idx] -= lr * wd * param[idx];
                }
                param[idx] -= lr * alpha * recon;
            }
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
                .or_insert_with(|| (vec![0.0f32; len], vec![0.0f32; len]));
            let lr = self.config.lr;
            let beta1 = self.config.beta1;
            let beta2 = self.config.beta2;
            let eps = self.config.eps;

            for i in 0..len {
                let gv = grad[i];
                state.0[i] = beta1 * state.0[i] + (1.0 - beta1) * gv;
                state.1[i] = beta2 * state.1[i] + (1.0 - beta2) * (gv * gv);
                data[i] -= lr * (state.0[i] / (state.1[i].sqrt() + eps));
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
                    .or_insert_with(|| (vec![0.0f32; len], vec![0.0f32; len]));
                let lr = self.config.lr;
                let beta1 = self.config.beta1;
                let beta2 = self.config.beta2;
                let eps = self.config.eps;

                for i in 0..len {
                    let gv = grad[i];
                    state.0[i] = beta1 * state.0[i] + (1.0 - beta1) * gv;
                    state.1[i] = beta2 * state.1[i] + (1.0 - beta2) * (gv * gv);
                    data[i] -= lr * (state.0[i] / (state.1[i].sqrt() + eps));
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
    fn test_galore_low_rank_projection_step() {
        let mut galore = GaLoreOptimizer::new(GaLoreConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            rank: 2,
            update_proj_gap: 10,
            scale: 1.0,
        });

        let mut param = vec![1.0; 4 * 4];
        let grad = vec![0.5; 4 * 4];
        let id = ParamId::new(0, 0, LoRAInjectionPoint::QProj, true);

        galore.update_matrix(id, &mut param, &grad, 4, 4).unwrap();

        for &val in &param {
            assert!(val < 1.0, "parameter should decrease from 1.0, got {val}");
        }
    }
}
