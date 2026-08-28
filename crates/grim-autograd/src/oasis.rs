//! OASIS: Online Activation Subspace Projection for Low-Rank Activation Compression.
//!
//! Repurposed from OASIS (arXiv:2604.09406) with unbiased-linear-safety (arXiv:2605.01255).
//! Projects intermediate linear layer forward activations onto an online low-rank basis subspace
//! matrix Q [d, p], saving activation memory during training without loss of first-order gradient fidelity.
//! Subspace updates use warm-started power iteration (O(d·p)) rather than costly per-step SVD.

use grim_quant::soul_eater::subspace_newton_schulz_step;

/// Online activation subspace tracker and projection.
#[derive(Debug, Clone)]
pub struct OasisSubspace {
    /// Semi-orthonormal basis [d, p], row-major (d = activation dimension, p = subspace rank).
    pub basis: Vec<f32>,
    /// Full activation dimension.
    pub d: usize,
    /// Low-rank subspace rank (e.g. 128 for d=4096).
    pub p: usize,
    /// EMA decay for subspace tracking (e.g. 0.95).
    pub ema_decay: f32,
}

impl OasisSubspace {
    /// Create a new OasisSubspace with initial orthogonalized random basis.
    pub fn new(d: usize, p: usize, ema_decay: f32) -> Self {
        let mut basis = vec![0.0f32; d * p];
        for i in 0..d {
            for j in 0..p {
                basis[i * p + j] = (((i + 1) * 23 + (j + 1) * 37) % 100) as f32 / 100.0 - 0.5;
            }
        }
        let _ = subspace_newton_schulz_step(&mut basis, d, p, 10);

        Self {
            basis,
            d,
            p,
            ema_decay,
        }
    }

    /// Project a [b, d] flat activation slice into its [b, p] low-rank coordinate representation.
    /// out[b*p + j] = sum_k act[b*d + k] * basis[k*p + j]
    pub fn project(&self, act: &[f32], b: usize) -> Vec<f32> {
        let d = self.d;
        let p = self.p;
        let mut out = vec![0.0f32; b * p];

        for row in 0..b {
            let act_row = &act[row * d..(row + 1) * d];
            for j in 0..p {
                let mut sum = 0.0f32;
                for k in 0..d {
                    sum += act_row[k] * self.basis[k * p + j];
                }
                out[row * p + j] = sum;
            }
        }

        out
    }

    /// Reconstruct approximate [b, d] activation from [b, p] coordinates: X_hat = X_proj * Basis^T.
    pub fn reconstruct(&self, proj: &[f32], b: usize) -> Vec<f32> {
        let d = self.d;
        let p = self.p;
        let mut out = vec![0.0f32; b * d];

        for row in 0..b {
            let proj_row = &proj[row * p..(row + 1) * p];
            for k in 0..d {
                let mut sum = 0.0f32;
                for j in 0..p {
                    sum += proj_row[j] * self.basis[k * p + j];
                }
                out[row * d + k] = sum;
            }
        }

        out
    }

    /// Update the basis from incoming activations using warm-started power iteration (O(d·p)).
    pub fn update_basis(&mut self, act: &[f32], b: usize) {
        if b == 0 {
            return;
        }
        let d = self.d;
        let p = self.p;

        // 1. Compute projection coordinates Y = act * basis [b, p]
        let proj = self.project(act, b);

        // 2. Power iteration update: new_basis = act^T * Y [d, p]
        let mut new_basis = vec![0.0f32; d * p];
        for row in 0..b {
            let act_row = &act[row * d..(row + 1) * d];
            let proj_row = &proj[row * p..(row + 1) * p];
            for k in 0..d {
                let a_val = act_row[k];
                for j in 0..p {
                    new_basis[k * p + j] += a_val * proj_row[j];
                }
            }
        }

        // 3. EMA blend into basis: basis = ema_decay * basis + (1 - ema_decay) * (new_basis / b)
        let scale = (1.0 - self.ema_decay) / (b as f32);
        for k in 0..d {
            for j in 0..p {
                let idx = k * p + j;
                self.basis[idx] = self.ema_decay * self.basis[idx] + scale * new_basis[idx];
            }
        }

        // 4. Re-orthogonalize basis
        let _ = subspace_newton_schulz_step(&mut self.basis, d, p, 3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oasis_subspace_known_projection() {
        // d = 4, p = 2
        // basis = canonical projection to first 2 dims:
        // [ [1, 0],
        //   [0, 1],
        //   [0, 0],
        //   [0, 0] ]
        let mut basis = vec![0.0f32; 4 * 2];
        basis[0 * 2 + 0] = 1.0;
        basis[1 * 2 + 1] = 1.0;

        let subspace = OasisSubspace {
            basis,
            d: 4,
            p: 2,
            ema_decay: 0.95,
        };

        let act = vec![
            1.0f32, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
        ];

        let proj = subspace.project(&act, 2);
        assert_eq!(proj.len(), 4);
        assert!((proj[0] - 1.0).abs() < 1e-6);
        assert!((proj[1] - 2.0).abs() < 1e-6);
        assert!((proj[2] - 5.0).abs() < 1e-6);
        assert!((proj[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_oasis_subspace_update_and_reconstruction() {
        let mut subspace = OasisSubspace::new(16, 4, 0.9);
        let act = vec![1.5f32; 2 * 16];

        subspace.update_basis(&act, 2);

        let proj = subspace.project(&act, 2);
        assert_eq!(proj.len(), 2 * 4);

        let recon = subspace.reconstruct(&proj, 2);
        assert_eq!(recon.len(), 2 * 16);
    }
}
