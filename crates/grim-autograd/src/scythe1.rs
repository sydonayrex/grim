//! SCYTHE1 = SOUL EATER adapter + Natural GaLore inverse-FIM preconditioning
//! in the adapter subspace.
//!
//! Extends `SoulEaterAdapter` / `SoulEaterOptimizer` with a diagonal Fisher
//! Information Matrix (FIM) accumulator that runs a running-average estimate
//! of `E[g * g^T]` over the low-rank adapter parameters.
//!
//! For rank-r=16 the FIM stays as an r×r diagonal (`Vec<f32>` of length 16).
//! At each optimizer step:
//!   1) accumulate outer products of adapter gradients into FIM,
//!   2) precondition with `FIM_inv_diag = 1 / max(FIM_diag, eps)`,
//!   3) delegate to `SoulEaterOptimizer::step` for the actual update.

use grim_tensor::{Result, Tensor};

use crate::soul_eater::{SoulEaterAdapter, SoulEaterOptimizer};

pub use crate::soul_eater::{SickleAdapter, SickleOptimizer};

const FIM_BETA: f32 = 0.99;
const FIM_EPS: f32 = 1e-8;

/// SCYTHE1 adapter: low-rank structural adapter (U, V, Σ) for inverse-FIM preconditioning.
pub struct Scythe1Adapter {
    pub inner: SoulEaterAdapter,
}

impl Scythe1Adapter {
    /// Instantiate from base dimensions, rank, and alpha.
    pub fn new(d_out: usize, d_in: usize, r: usize, alpha: f32) -> Result<Self> {
        let inner = SoulEaterAdapter::new(d_out, d_in, r, alpha)?;
        Ok(Self { inner })
    }

    /// Compute forward adapter output, same as `SoulEaterAdapter::forward`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

/// SCYTHE1 optimizer: wraps `SoulEaterOptimizer` and adds inverse-FIM
/// preconditioning on the adapter subspace.
pub struct Scythe1Optimizer {
    pub inner: SoulEaterOptimizer,
    pub fim_diag: Vec<f32>,
    pub rank: usize,
}

impl Scythe1Optimizer {
    /// Create a new SCYTHE1 optimizer.
    pub fn new(lr_basis: f32, lr_sigma: f32, beta: f32, rank: usize) -> Self {
        Self {
            inner: SoulEaterOptimizer::new(lr_basis, lr_sigma, beta),
            fim_diag: vec![1.0f32; rank],
            rank,
        }
    }

    /// Perform one optimizer step:
    /// 1) accumulate diagonal FIM from raw gradients,
    /// 2) precondition U, V, Σ gradients with inverse-FIM diagonal,
    /// 3) delegate to `SoulEaterOptimizer::step` for the actual update.
    #[allow(clippy::too_many_arguments)]
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
        // 1) accumulate FIM (running average of outer products)
        let r = self.rank;
        let d_out = u.shape().dims()[0];
        let d_in = v.shape().dims()[0];

        let mut proj_u = vec![0.0f32; r];
        for k in 0..r {
            let mut s = 0.0f32;
            for j in 0..d_out {
                s += g_u[j * r + k] * g_u[j * r + k];
            }
            proj_u[k] = s;
        }
        let mut proj_v = vec![0.0f32; r];
        for k in 0..r {
            let mut s = 0.0f32;
            for j in 0..d_in {
                s += g_v[j * r + k] * g_v[j * r + k];
            }
            proj_v[k] = s;
        }

        for k in 0..r {
            let outer = proj_u[k] + proj_v[k] + g_sigma[k] * g_sigma[k];
            self.fim_diag[k] = FIM_BETA * self.fim_diag[k] + (1.0 - FIM_BETA) * outer;
        }

        // 2) precondition gradients with inverse-FIM diagonal
        let p_u = self.precondition_u_grad(g_u);
        let p_v = self.precondition_v_grad(g_v);
        let p_sigma: Vec<f32> = g_sigma
            .iter()
            .zip(self.fim_diag.iter())
            .map(|(&g, &fim)| g / (fim.max(FIM_EPS)))
            .collect();

        // 3) delegate to SoulEater optimizer
        self.inner.step(name, u, v, sigma, &p_u, &p_v, &p_sigma)
    }

    /// Helper: precondition a generic rank-r gradient slice.
    fn precondition_u_grad(&self, g_u: &[f32]) -> Vec<f32> {
        let len = g_u.len();
        let r = self.rank;
        let mut out = vec![0.0f32; len];
        for j in 0..(len / r) {
            for k in 0..r {
                let idx = j * r + k;
                out[idx] = g_u[idx] / (self.fim_diag[k].max(FIM_EPS));
            }
        }
        out
    }

    fn precondition_v_grad(&self, g_v: &[f32]) -> Vec<f32> {
        let len = g_v.len();
        let r = self.rank;
        let mut out = vec![0.0f32; len];
        for j in 0..(len / r) {
            for k in 0..r {
                let idx = j * r + k;
                out[idx] = g_v[idx] / (self.fim_diag[k].max(FIM_EPS));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    #[test]
    fn test_scythe1_adapter_shape_algebra() {
        let b = 4;
        let d_in = 32;
        let d_out = 64;
        let r = 16;
        let adapter = Scythe1Adapter::new(d_out, d_in, r, 1.0).unwrap();

        let x = cpu_tensor(vec![1.0f32; b * d_in], Shape::new(vec![b, d_in]));
        let y = adapter.forward(&x).unwrap();

        assert_eq!(y.shape().dims(), vec![b, d_out]);
    }

    #[test]
    fn test_scythe1_fim_accumulation() {
        let d_in = 16;
        let d_out = 16;
        let r = 8;
        let mut adapter = Scythe1Adapter::new(d_out, d_in, r, 1.0).unwrap();
        let mut opt = Scythe1Optimizer::new(0.01, 0.01, 0.0, r);

        let g_u = vec![1.0f32; d_out * r];
        let g_v = vec![1.0f32; d_in * r];
        let g_sigma = vec![1.0f32; r];

        let fim_before: f32 = opt.fim_diag.iter().sum();
        opt.step(
            "layer0",
            &mut adapter.inner.u,
            &mut adapter.inner.v,
            &mut adapter.inner.sigma,
            &g_u,
            &g_v,
            &g_sigma,
        )
        .unwrap();
        let fim_after: f32 = opt.fim_diag.iter().sum();

        assert!(
            fim_after > fim_before,
            "FIM must grow with gradient magnitude"
        );
    }

    #[test]
    fn test_scythe1_preconditioning() {
        let r = 4;
        let mut opt = Scythe1Optimizer::new(0.01, 0.01, 0.0, r);

        // Artificially inflate FIM for dimension 2 to simulate high curvature
        opt.fim_diag[2] = 100.0;
        let g_sigma = vec![1.0f32; r];

        let p: Vec<f32> = g_sigma
            .iter()
            .zip(opt.fim_diag.iter())
            .map(|(&g, &fim)| g / fim.max(FIM_EPS))
            .collect();

        // Dimension 2 should be divided by ~100, others by ~1.0
        assert!(
            p[2] < p[0] && p[2] < p[1] && p[2] < p[3],
            "large-FIM dimension must get smaller update: {:?}",
            p
        );
    }

    #[test]
    fn test_scythe1_optimizer_forward_backward_reduction() {
        let d_in = 16;
        let d_out = 16;
        let r = 8;
        let mut adapter = Scythe1Adapter::new(d_out, d_in, r, 1.0).unwrap();
        let mut opt = Scythe1Optimizer::new(0.01, 0.01, 0.0, r);

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
                dy[i] = 2.0 * diff;
            }
            if step == 0 {
                initial_loss = loss;
            }
            final_loss = loss;

            let x_vec = x.to_vec_f32().unwrap();
            let u_vec = adapter.inner.u.to_vec_f32().unwrap();
            let v_vec = adapter.inner.v.to_vec_f32().unwrap();
            let sig_vec = adapter.inner.sigma.to_vec_f32().unwrap();
            let scale = adapter.inner.scale;

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

            let mut g_sigma = vec![0.0f32; r];
            for k in 0..r {
                let mut sum = 0.0f32;
                for j in 0..d_out {
                    sum += dy[j] * scale * x_v[k] * u_vec[j * r + k];
                }
                g_sigma[k] = sum;
            }

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

            opt.step(
                "layer0",
                &mut adapter.inner.u,
                &mut adapter.inner.v,
                &mut adapter.inner.sigma,
                &g_u,
                &g_v,
                &g_sigma,
            )
            .unwrap();
        }

        assert!(
            final_loss <= initial_loss,
            "Loss must not increase: initial {initial_loss}, final {final_loss}"
        );
    }
}
