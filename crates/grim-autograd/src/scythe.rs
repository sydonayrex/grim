//! SCYTHE: FORGE (fused tile-wise backward) + SCALE (stateless column-norm) + OASIS low-rank adapter optimizer.
//!
//! - FORGE: Consumes U and V gradients tile-wise in register tiles (e.g. 64 rows), avoiding
//!   ever materializing full [d_out, r] or [d_in, r] gradient tensors in memory.
//! - SCALE: Eliminates persistent second-moment EMA (Fisher/Adam v) for singular values Σ,
//!   using instantaneous column-wise RMS normalization.
//! - OASIS: Provides online low-rank subspace projection for intermediate activation tensors.

use grim_backend_cpu::cpu_tensor;
use grim_quant::soul_eater::subspace_newton_schulz_step;
use grim_tensor::{Result, Shape, Tensor};
use std::collections::HashMap;

use crate::scale::column_rms;

/// Tile chunk row size for FORGE streaming backward updates.
pub const U_TILE_ROWS: usize = 64;

/// SCYTHE low-rank adapter representation.
#[derive(Debug, Clone)]
pub struct ScytheAdapter {
    /// Output basis matrix U [d_out, r], semi-orthogonal U^T * U = I_r.
    pub u: Tensor,
    /// Input basis matrix V [d_in, r], semi-orthogonal V^T * V = I_r.
    pub v: Tensor,
    /// Diagonal singular values Σ [r].
    pub sigma: Tensor,
    /// Scaling factor (alpha / r).
    pub scale: f32,
    /// Low-rank dimension.
    pub rank: usize,
}

impl ScytheAdapter {
    /// Instantiate a new SCYTHE adapter for dimensions [d_out, d_in] and rank `r`.
    pub fn new(d_out: usize, d_in: usize, r: usize, alpha: f32) -> Result<Self> {
        let mut u_data = vec![0.0f32; d_out * r];
        let mut v_data = vec![0.0f32; d_in * r];

        for i in 0..d_out {
            for j in 0..r {
                u_data[i * r + j] = (((i + 1) * 17 + (j + 1) * 31) % 100) as f32 / 100.0 - 0.5;
            }
        }
        for i in 0..d_in {
            for j in 0..r {
                v_data[i * r + j] = (((i + 1) * 13 + (j + 1) * 29) % 100) as f32 / 100.0 - 0.5;
            }
        }

        let _ = subspace_newton_schulz_step(&mut u_data, d_out, r, 10);
        let _ = subspace_newton_schulz_step(&mut v_data, d_in, r, 10);

        let u = cpu_tensor(u_data, Shape::new(vec![d_out, r]));
        let v = cpu_tensor(v_data, Shape::new(vec![d_in, r]));
        let sigma = cpu_tensor(vec![1.0f32; r], Shape::new(vec![r]));
        let scale = alpha / (r as f32);

        Ok(Self {
            u,
            v,
            sigma,
            scale,
            rank: r,
        })
    }

    /// Compute forward pass: Y = (scale) * (X * V) * diag(Σ) * U^T.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let u_slice = self.u.to_vec_f32()?;
        let v_slice = self.v.to_vec_f32()?;
        let sig_slice = self.sigma.to_vec_f32()?;
        let x_slice = x.to_vec_f32()?;

        let x_dims = x.shape().dims();
        let batch_tokens = if x_dims.len() == 1 {
            1
        } else {
            x_dims[..x_dims.len() - 1].iter().product()
        };
        let d_in = x_dims[x_dims.len() - 1];
        let d_out = self.u.shape().dims()[0];
        let r = self.rank;

        // Step 1: X_V = X * V -> [batch_tokens, r]
        let mut xv = vec![0.0f32; batch_tokens * r];
        for b in 0..batch_tokens {
            let x_row = &x_slice[b * d_in..(b + 1) * d_in];
            for k in 0..r {
                let mut sum = 0.0f32;
                for j in 0..d_in {
                    sum += x_row[j] * v_slice[j * r + k];
                }
                xv[b * r + k] = sum;
            }
        }

        // Step 2: Scale by Σ
        let mut xv_sig = vec![0.0f32; batch_tokens * r];
        for b in 0..batch_tokens {
            for k in 0..r {
                xv_sig[b * r + k] = xv[b * r + k] * sig_slice[k];
            }
        }

        // Step 3: Y = (xv_sig) * U^T * scale -> [batch_tokens, d_out]
        let mut y = vec![0.0f32; batch_tokens * d_out];
        for b in 0..batch_tokens {
            let xv_row = &xv_sig[b * r..(b + 1) * r];
            for i in 0..d_out {
                let mut sum = 0.0f32;
                for k in 0..r {
                    sum += xv_row[k] * u_slice[i * r + k];
                }
                y[b * d_out + i] = sum * self.scale;
            }
        }

        let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(d_out);
        Ok(cpu_tensor(y, Shape::new(out_dims)))
    }
}

/// Hyperparameters for SCYTHE optimizer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScytheConfig {
    pub lr_basis: f32,
    pub lr_sigma: f32,
    pub beta: f32,
}

impl Default for ScytheConfig {
    fn default() -> Self {
        Self {
            lr_basis: 2e-4,
            lr_sigma: 1e-3,
            beta: 0.9,
        }
    }
}

/// Fused memory-efficient SCYTHE optimizer.
pub struct ScytheOptimizer {
    pub config: ScytheConfig,
    pub m_u: HashMap<String, Vec<f32>>,
    pub m_v: HashMap<String, Vec<f32>>,
}

impl ScytheOptimizer {
    pub fn new(lr_basis: f32, lr_sigma: f32, beta: f32) -> Self {
        Self {
            config: ScytheConfig {
                lr_basis,
                lr_sigma,
                beta,
            },
            m_u: HashMap::new(),
            m_v: HashMap::new(),
        }
    }

    pub fn with_config(config: ScytheConfig) -> Self {
        Self {
            config,
            m_u: HashMap::new(),
            m_v: HashMap::new(),
        }
    }

    /// Step parameter registry (dispatches column RMS updates and momentum step across trainable params).
    pub fn step(&mut self, params: &mut crate::param::TrainableParams) -> Result<()> {
        for (&id, param) in params.iter_mut() {
            self.step_param(id, param)?;
        }
        Ok(())
    }

    pub fn step_param(
        &mut self,
        id: crate::param::ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let g_vec = param.grad().to_vec_f32()?;
        let mut d_vec = param.data.to_vec_f32()?;
        let d = d_vec.len();
        let norms = column_rms(&g_vec, 1, d, 1e-8);

        let m_entry = self
            .m_u
            .entry(format!("{:?}", id))
            .or_insert_with(|| vec![0.0f32; d]);

        for i in 0..d {
            let update_dir = g_vec[i] / norms[i];
            m_entry[i] = self.config.beta * m_entry[i] + (1.0 - self.config.beta) * update_dir;
            d_vec[i] -= self.config.lr_basis * m_entry[i];
        }

        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape().clone();
        let new_storage = dev.from_cpu(&d_vec, &shape, grim_tensor::DType::F32)?;
        param.data = grim_tensor::Tensor::new(
            std::sync::Arc::from(new_storage),
            shape,
            grim_tensor::DType::F32,
            param.data.provenance().clone(),
            param.data.device().clone(),
        );
        param.zero_grad()?;
        Ok(())
    }

    pub fn save_to_train_state(
        &self,
        params: &crate::param::TrainableParams,
    ) -> grim_format::train::TrainState {
        crate::adamw::save_param_data_only(params, 0)
    }

    pub fn load_from_train_state(
        &mut self,
        params: &mut crate::param::TrainableParams,
        state: &grim_format::train::TrainState,
    ) -> Result<()> {
        crate::adamw::load_param_data_only(params, state)
    }

    /// Fused tile-wise backward + parameter update step (FORGE + SCALE).
    /// Consumes gradients in 64-row tiles without ever materializing full [d_out, r] or [d_in, r] buffers.
    pub fn fused_step(
        &mut self,
        name: &str,
        adapter: &mut ScytheAdapter,
        out_grad: &Tensor,
        x: &Tensor,
    ) -> Result<()> {
        self.fused_step_with_oasis(name, adapter, out_grad, x, None)
    }

    /// Fused backward + optimizer step with optional OASIS activation subspace projection.
    pub fn fused_step_with_oasis(
        &mut self,
        name: &str,
        adapter: &mut ScytheAdapter,
        out_grad: &Tensor,
        x: &Tensor,
        mut oasis: Option<&mut crate::oasis::OasisSubspace>,
    ) -> Result<()> {
        let d_out = adapter.u.shape().dims()[0];
        let d_in = adapter.v.shape().dims()[0];
        let r = adapter.rank;
        let scale = adapter.scale;

        let g_out_slice = out_grad.to_vec_f32()?;
        let x_raw = x.to_vec_f32()?;
        let x_dims = x.shape().dims();
        let batch_tokens = if x_dims.len() == 1 {
            1
        } else {
            x_dims[..x_dims.len() - 1].iter().product()
        };

        // If OASIS is enabled, update basis and reconstruct through low-rank coordinates
        let x_slice = if let Some(subspace) = oasis.as_deref_mut() {
            subspace.update_basis(&x_raw, batch_tokens);
            let proj = subspace.project(&x_raw, batch_tokens);
            subspace.reconstruct(&proj, batch_tokens)
        } else {
            x_raw
        };

        let u_slice = adapter.u.to_vec_f32()?;
        let v_slice = adapter.v.to_vec_f32()?;
        let sig_slice = adapter.sigma.to_vec_f32()?;

        // 1. Recompute forward intermediate X_V and X_V_Sig
        let mut xv = vec![0.0f32; batch_tokens * r];
        let mut xv_sig = vec![0.0f32; batch_tokens * r];
        for b in 0..batch_tokens {
            let x_row = &x_slice[b * d_in..(b + 1) * d_in];
            for k in 0..r {
                let mut sum = 0.0f32;
                for j in 0..d_in {
                    sum += x_row[j] * v_slice[j * r + k];
                }
                xv[b * r + k] = sum;
                xv_sig[b * r + k] = sum * sig_slice[k];
            }
        }

        // 2. Accumulate g_sigma (length r, negligible memory)
        let mut g_sigma = vec![0.0f32; r];
        for b in 0..batch_tokens {
            let g_row = &g_out_slice[b * d_out..(b + 1) * d_out];
            let xv_row = &xv[b * r..(b + 1) * r];
            for k in 0..r {
                let mut u_proj = 0.0f32;
                for i in 0..d_out {
                    u_proj += g_row[i] * u_slice[i * r + k];
                }
                g_sigma[k] += u_proj * xv_row[k] * scale;
            }
        }

        // 3. Tile-wise stream computation & momentum folding for U
        let m_u_entry = self
            .m_u
            .entry(name.to_string())
            .or_insert_with(|| vec![0.0f32; d_out * r]);

        let mut updated_u = u_slice.clone();

        for tile_start in (0..d_out).step_by(U_TILE_ROWS) {
            let tile_end = (tile_start + U_TILE_ROWS).min(d_out);
            let tile_rows = tile_end - tile_start;

            let mut g_u_tile = vec![0.0f32; tile_rows * r];
            for b in 0..batch_tokens {
                let g_row = &g_out_slice[b * d_out..(b + 1) * d_out];
                let xv_sig_row = &xv_sig[b * r..(b + 1) * r];
                for local_i in 0..tile_rows {
                    let global_i = tile_start + local_i;
                    let g_val = g_row[global_i] * scale;
                    for k in 0..r {
                        g_u_tile[local_i * r + k] += g_val * xv_sig_row[k];
                    }
                }
            }

            // Fold momentum & update into tile
            for local_i in 0..tile_rows {
                let global_i = tile_start + local_i;
                for k in 0..r {
                    let idx = global_i * r + k;
                    let grad = g_u_tile[local_i * r + k];
                    m_u_entry[idx] =
                        self.config.beta * m_u_entry[idx] + (1.0 - self.config.beta) * grad;
                    updated_u[idx] -= self.config.lr_basis * m_u_entry[idx];
                }
            }
        }

        // Orthogonalize U
        let _ = subspace_newton_schulz_step(&mut updated_u, d_out, r, 5);
        adapter.u = cpu_tensor(updated_u, Shape::new(vec![d_out, r]));

        // 4. Tile-wise stream computation & momentum folding for V
        let m_v_entry = self
            .m_v
            .entry(name.to_string())
            .or_insert_with(|| vec![0.0f32; d_in * r]);

        let mut updated_v = v_slice.clone();

        // G_U_Sig = G_out * U * diag(Σ) * scale -> [batch_tokens, r]
        let mut g_xv_sig = vec![0.0f32; batch_tokens * r];
        for b in 0..batch_tokens {
            let g_row = &g_out_slice[b * d_out..(b + 1) * d_out];
            for k in 0..r {
                let mut sum = 0.0f32;
                for i in 0..d_out {
                    sum += g_row[i] * u_slice[i * r + k];
                }
                g_xv_sig[b * r + k] = sum * sig_slice[k] * scale;
            }
        }

        for tile_start in (0..d_in).step_by(U_TILE_ROWS) {
            let tile_end = (tile_start + U_TILE_ROWS).min(d_in);
            let tile_rows = tile_end - tile_start;

            let mut g_v_tile = vec![0.0f32; tile_rows * r];
            for b in 0..batch_tokens {
                let x_row = &x_slice[b * d_in..(b + 1) * d_in];
                let g_xv_row = &g_xv_sig[b * r..(b + 1) * r];
                for local_j in 0..tile_rows {
                    let global_j = tile_start + local_j;
                    let x_val = x_row[global_j];
                    for k in 0..r {
                        g_v_tile[local_j * r + k] += x_val * g_xv_row[k];
                    }
                }
            }

            for local_j in 0..tile_rows {
                let global_j = tile_start + local_j;
                for k in 0..r {
                    let idx = global_j * r + k;
                    let grad = g_v_tile[local_j * r + k];
                    m_v_entry[idx] =
                        self.config.beta * m_v_entry[idx] + (1.0 - self.config.beta) * grad;
                    updated_v[idx] -= self.config.lr_basis * m_v_entry[idx];
                }
            }
        }

        // Orthogonalize V
        let _ = subspace_newton_schulz_step(&mut updated_v, d_in, r, 5);
        adapter.v = cpu_tensor(updated_v, Shape::new(vec![d_in, r]));

        // 5. Update Σ using SCALE column RMS normalization (no FIM EMA)
        let sig_norms = column_rms(&g_sigma, 1, r, 1e-8);
        let mut updated_sig = sig_slice;
        for k in 0..r {
            let norm = sig_norms[k];
            let update_dir = g_sigma[k] / norm;
            updated_sig[k] -= self.config.lr_sigma * update_dir;
            if updated_sig[k] < 0.0 {
                updated_sig[k] = 0.0;
            }
        }
        adapter.sigma = cpu_tensor(updated_sig, Shape::new(vec![r]));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scythe_adapter_forward_shape() {
        let adapter = ScytheAdapter::new(32, 16, 4, 1.0).unwrap();
        let x = cpu_tensor(vec![1.0f32; 16], Shape::new(vec![1, 16]));
        let y = adapter.forward(&x).unwrap();

        assert_eq!(y.shape().dims(), &[1, 32]);
    }

    #[test]
    fn test_scythe_optimizer_step_loss_reduction() {
        let mut adapter = ScytheAdapter::new(8, 8, 4, 1.0).unwrap();
        let mut opt = ScytheOptimizer::new(0.05, 0.05, 0.9);

        let x = cpu_tensor(vec![0.5f32; 8], Shape::new(vec![1, 8]));
        let target = vec![1.0f32; 8];

        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for step in 0..20 {
            let y = adapter.forward(&x).unwrap();
            let y_slice = y.to_vec_f32().unwrap();

            let mut loss = 0.0f32;
            let mut g_out = vec![0.0f32; 8];
            for i in 0..8 {
                let diff = y_slice[i] - target[i];
                loss += diff * diff;
                g_out[i] = 2.0 * diff; // MSE gradient
            }

            if step == 0 {
                initial_loss = loss;
            }
            if step == 19 {
                final_loss = loss;
            }

            let g_tensor = cpu_tensor(g_out, Shape::new(vec![1, 8]));
            opt.fused_step("test_layer", &mut adapter, &g_tensor, &x)
                .unwrap();
        }

        assert!(
            final_loss < initial_loss,
            "Scythe optimizer step should decrease MSE loss: init={}, final={}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_scythe_with_oasis_convergence() {
        let mut adapter = ScytheAdapter::new(16, 16, 8, 1.0).unwrap();
        let mut opt = ScytheOptimizer::new(0.05, 0.05, 0.9);
        let mut oasis = crate::oasis::OasisSubspace::new(16, 4, 0.95);

        let x = cpu_tensor(vec![0.5f32; 16], Shape::new(vec![1, 16]));
        let target = vec![1.0f32; 16];

        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for step in 0..50 {
            let y = adapter.forward(&x).unwrap();
            let y_slice = y.to_vec_f32().unwrap();

            let mut loss = 0.0f32;
            let mut g_out = vec![0.0f32; 16];
            for i in 0..16 {
                let diff = y_slice[i] - target[i];
                loss += diff * diff;
                g_out[i] = 2.0 * diff;
            }

            if step == 0 {
                initial_loss = loss;
            }
            if step == 49 {
                final_loss = loss;
            }

            let g_tensor = cpu_tensor(g_out, Shape::new(vec![1, 16]));
            opt.fused_step_with_oasis("oasis_layer", &mut adapter, &g_tensor, &x, Some(&mut oasis))
                .unwrap();
        }

        assert!(
            final_loss < initial_loss,
            "Scythe+OASIS step must reduce MSE loss: init={}, final={}",
            initial_loss,
            final_loss
        );
    }
}
