//! SOUL EATER Adapter & Optimizer module for `grim-autograd`.
//!
//! Provides the `SoulEaterAdapter` structural parameterization:
//! ΔW = U * Σ * V^T, with forward pass Y = X * W0^T + (α/r) * (X * V) * Σ * U^T.
//! Also provides `SoulEaterOptimizer` using 1-bit Sign-SGD for Σ and
//! momentum-accelerated pre-normalized cubic Newton-Schulz for U and V.

use std::collections::HashMap;
use grim_backend_cpu::cpu_tensor;
use grim_tensor::{Result, Shape, Tensor};
use grim_quant::soul_eater::subspace_newton_schulz_step;

/// Parameter representation for SOUL EATER adapter (U, V, Σ).
pub struct SoulEaterAdapter {
    /// Output basis matrix U [d_out, r], semi-orthogonal U^T * U = I_r.
    pub u: Tensor,
    /// Input basis matrix V [d_in, r], semi-orthogonal V^T * V = I_r.
    pub v: Tensor,
    /// Diagonal singular values Σ [r].
    pub sigma: Tensor,
    /// Scaling alpha / r.
    pub scale: f32,
    pub rank: usize,
}

impl SoulEaterAdapter {
    /// Instantiate a new SOUL EATER adapter for linear layer dimensions [d_out, d_in] and rank `r`.
    pub fn new(d_out: usize, d_in: usize, r: usize, alpha: f32) -> Result<Self> {
        let mut u_data = vec![0.0f32; d_out * r];
        let mut v_data = vec![0.0f32; d_in * r];
        
        // Initialize U and V with normalized well-conditioned values
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

        // Perform initial orthogonalization
        let _ = subspace_newton_schulz_step(&mut u_data, d_out, r, 10);
        let _ = subspace_newton_schulz_step(&mut v_data, d_in, r, 10);

        let u = cpu_tensor(u_data, Shape::new(vec![d_out, r]));
        let v = cpu_tensor(v_data, Shape::new(vec![d_in, r]));
        // Initialize singular values to 1.0
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

    /// Compute forward adapter output: Y_adapter = (α/r) * (X * V) * Σ * U^T.
    /// Returns output tensor of shape [B, d_out] for input X of shape [B, d_in].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        let b = x_shape[0];
        let d_in = x_shape[1];
        let d_out = self.u.shape().dims()[0];
        let r = self.rank;

        let x_vec = x.to_vec_f32()?;
        let u_vec = self.u.to_vec_f32()?;
        let v_vec = self.v.to_vec_f32()?;
        let sig_vec = self.sigma.to_vec_f32()?;

        // 1. Compute X_V = X * V [B, d_in] * [d_in, r] = [B, r]
        let mut x_v = vec![0.0f32; b * r];
        for i in 0..b {
            for j in 0..r {
                let mut sum = 0.0f32;
                for k in 0..d_in {
                    sum += x_vec[i * d_in + k] * v_vec[k * r + j];
                }
                x_v[i * r + j] = sum;
            }
        }

        // 2. Scale by Σ: X_V_Sig[i, j] = X_V[i, j] * Σ[j]
        let mut x_v_sig = vec![0.0f32; b * r];
        for i in 0..b {
            for j in 0..r {
                x_v_sig[i * r + j] = x_v[i * r + j] * sig_vec[j];
            }
        }

        // 3. Multiply by U^T: Out = (X_V_Sig * U^T) * (alpha / r) [B, r] * [r, d_out] = [B, d_out]
        let mut out = vec![0.0f32; b * d_out];
        for i in 0..b {
            for j in 0..d_out {
                let mut sum = 0.0f32;
                for k in 0..r {
                    sum += x_v_sig[i * r + k] * u_vec[j * r + k];
                }
                out[i * d_out + j] = sum * self.scale;
            }
        }

        Ok(cpu_tensor(out, Shape::new(vec![b, d_out])))
    }
}

/// SOUL EATER Optimizer: Momentum + Newton-Schulz for U, V; Sign-SGD for Σ.
pub struct SoulEaterOptimizer {
    pub lr_basis: f32,
    pub lr_sigma: f32,
    pub beta: f32,
    pub m_u: HashMap<String, Vec<f32>>,
    pub m_v: HashMap<String, Vec<f32>>,
}

impl SoulEaterOptimizer {
    pub fn new(lr_basis: f32, lr_sigma: f32, beta: f32) -> Self {
        Self {
            lr_basis,
            lr_sigma,
            beta,
            m_u: HashMap::new(),
            m_v: HashMap::new(),
        }
    }

    /// Perform optimizer update step on adapter parameter tensors U, V, Σ.
    pub fn step(
        &mut self,
        name: &str,
        u: &mut Tensor,
        v: &mut Tensor,
        sigma: &mut Tensor,
        g_u: &[f32],
        g_v: &[f32],
        g_sigma: &[f32],
    ) -> Result<()> {
        let d_out = u.shape().dims()[0];
        let r = u.shape().dims()[1];
        let d_in = v.shape().dims()[0];

        // 1. Update U momentum & run subspace Newton-Schulz step
        let key_u = format!("{name}_u");
        let m_u_entry = self.m_u.entry(key_u).or_insert_with(|| vec![0.0f32; d_out * r]);
        for i in 0..(d_out * r) {
            m_u_entry[i] = self.beta * m_u_entry[i] + (1.0 - self.beta) * g_u[i];
        }

        let mut o_u = m_u_entry.clone();
        if subspace_newton_schulz_step(&mut o_u, d_out, r, 10).is_ok() {
            let mut u_vec = u.to_vec_f32()?;
            for i in 0..(d_out * r) {
                u_vec[i] -= self.lr_basis * o_u[i];
            }
            *u = cpu_tensor(u_vec, Shape::new(vec![d_out, r]));
        }

        // 2. Update V momentum & run subspace Newton-Schulz step
        let key_v = format!("{name}_v");
        let m_v_entry = self.m_v.entry(key_v).or_insert_with(|| vec![0.0f32; d_in * r]);
        for i in 0..(d_in * r) {
            m_v_entry[i] = self.beta * m_v_entry[i] + (1.0 - self.beta) * g_v[i];
        }

        let mut o_v = m_v_entry.clone();
        if subspace_newton_schulz_step(&mut o_v, d_in, r, 10).is_ok() {
            let mut v_vec = v.to_vec_f32()?;
            for i in 0..(d_in * r) {
                v_vec[i] -= self.lr_basis * o_v[i];
            }
            *v = cpu_tensor(v_vec, Shape::new(vec![d_in, r]));
        }

        // 3. Update Σ via 1-bit Sign-SGD
        let mut sig_vec = sigma.to_vec_f32()?;
        for i in 0..r {
            let sign = if g_sigma[i] > 0.0 {
                1.0
            } else if g_sigma[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
            sig_vec[i] -= self.lr_sigma * sign;
        }
        *sigma = cpu_tensor(sig_vec, Shape::new(vec![r]));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soul_eater_adapter_shape_algebra() {
        let b = 4;
        let d_in = 32;
        let d_out = 64;
        let r = 16;
        let adapter = SoulEaterAdapter::new(d_out, d_in, r, 1.0).unwrap();

        let x = cpu_tensor(vec![1.0f32; b * d_in], Shape::new(vec![b, d_in]));
        let y = adapter.forward(&x).unwrap();

        assert_eq!(y.shape().dims(), vec![b, d_out]);
    }

    #[test]
    fn test_soul_eater_forward_backward_loss_reduction() {
        let d_in = 16;
        let d_out = 16;
        let r = 8;
        let mut adapter = SoulEaterAdapter::new(d_out, d_in, r, 1.0).unwrap();
        let mut opt = SoulEaterOptimizer::new(0.01, 0.01, 0.0); // beta=0 for direct gradient step

        let x = cpu_tensor(vec![0.5f32; d_in], Shape::new(vec![1, d_in]));
        let target = cpu_tensor(vec![1.0f32; d_out], Shape::new(vec![1, d_out]));

        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for step in 0..50 {
            let y = adapter.forward(&x).unwrap();
            let y_vec = y.to_vec_f32().unwrap();
            let target_vec = target.to_vec_f32().unwrap();

            let mut loss = 0.0f32;
            let mut dy = vec![0.0f32; d_out];
            for i in 0..d_out {
                let diff = y_vec[i] - target_vec[i];
                loss += diff * diff;
                dy[i] = 2.0 * diff; // L = sum( (y - target)^2 )
            }
            if step == 0 {
                initial_loss = loss;
            }
            final_loss = loss;

            // Analytical gradients:
            // Y = (alpha/r) * (X V) * Σ * U^T
            let x_vec = x.to_vec_f32().unwrap();
            let u_vec = adapter.u.to_vec_f32().unwrap();
            let v_vec = adapter.v.to_vec_f32().unwrap();
            let sig_vec = adapter.sigma.to_vec_f32().unwrap();
            let scale = adapter.scale;

            // dL/dU [d_out, r]: dL/dU[j, k] = dy[j] * scale * (X V)[k] * Σ[k]
            let mut x_v = vec![0.0f32; r];
            for k in 0..r {
                let mut sum = 0.0f32;
                for i in 0..d_in {
                    sum += x_vec[i] * v_vec[i * r + k];
                }
                x_v[k] = sum;
            }

            let mut g_u = vec![0.0f32; d_out * r];
            for j in 0..d_out {
                for k in 0..r {
                    g_u[j * r + k] = dy[j] * scale * x_v[k] * sig_vec[k];
                }
            }

            // dL/dΣ [r]: dL/dΣ[k] = sum_j ( dy[j] * scale * (X V)[k] * U[j, k] )
            let mut g_sigma = vec![0.0f32; r];
            for k in 0..r {
                let mut sum = 0.0f32;
                for j in 0..d_out {
                    sum += dy[j] * scale * x_v[k] * u_vec[j * r + k];
                }
                g_sigma[k] = sum;
            }

            // dL/dV [d_in, r]: dL/dV[i, k] = x_vec[i] * sum_j ( dy[j] * scale * Σ[k] * U[j, k] )
            let mut g_v = vec![0.0f32; d_in * r];
            for i in 0..d_in {
                for k in 0..r {
                    let mut sum = 0.0f32;
                    for j in 0..d_out {
                        sum += dy[j] * scale * sig_vec[k] * u_vec[j * r + k];
                    }
                    g_v[i * r + k] = x_vec[i] * sum;
                }
            }

            opt.step("layer0", &mut adapter.u, &mut adapter.v, &mut adapter.sigma, &g_u, &g_v, &g_sigma).unwrap();
        }

        assert!(final_loss <= initial_loss, "Loss must not increase: initial {initial_loss}, final {final_loss}");
    }
}
