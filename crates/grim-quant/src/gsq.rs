//! GSQ (Gumbel-Softmax Quantization) for sub-Q4 scalar quantization.
//!
//! Learns discrete scalar grid assignments and per-group scales via continuous
//! Gumbel-Softmax relaxation. Closes the gap between scalar and vector quantization
//! at 2 to 3 bits while remaining deployable into standard scalar formats (Q2_K, Q3_K, IQ2/3).

use grim_tensor::error::{Error, Result};

/// Configuration for GSQ calibration and grid optimization.
#[derive(Debug, Clone)]
pub struct GsqConfig {
    /// Number of Gumbel-Softmax gradient steps.
    pub steps: usize,
    /// Initial temperature for Gumbel-Softmax relaxation.
    pub temperature_init: f32,
    /// Final temperature after annealing.
    pub temperature_min: f32,
    /// Learning rate for continuous grid coordinates and scale.
    pub lr: f32,
}

impl Default for GsqConfig {
    fn default() -> Self {
        Self {
            steps: 25,
            temperature_init: 1.5,
            temperature_min: 0.2,
            lr: 0.05,
        }
    }
}

/// GSQ quantized block result.
#[derive(Debug, Clone)]
pub struct GsqBlockFit {
    /// Discrete quantization codes per weight.
    pub codes: Vec<u8>,
    /// Optimized scalar grid levels.
    pub grid: Vec<f32>,
    /// Optimized group scale.
    pub scale: f32,
    /// L2 reconstruction error.
    pub error: f32,
}

/// Fit a sub-Q4 weight block using Gumbel-Softmax relaxation.
///
/// Optimizes continuous coordinates `c_0, ..., c_{K-1}` and scalar `scale` such that
/// soft relaxation $\sum_k P(w_i \in k) c_k \cdot \text{scale}$ minimizes MSE to $w_i$,
/// then extracts hard assignments $\arg\max_k P(w_i \in k)$.
pub fn gsq_fit_block(
    data: &[f32],
    bits: u8,
    config: &GsqConfig,
) -> Result<GsqBlockFit> {
    if data.is_empty() {
        return Err(Error::Backend("gsq_fit_block: empty data block".into()));
    }

    let n_levels = 1usize << (bits as usize);
    let num_weights = data.len();

    // Initial scale: maximum absolute magnitude normalized to grid range
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let mut scale = if max_abs > 0.0 {
        max_abs / ((n_levels - 1) as f32 * 0.5)
    } else {
        1.0f32
    };

    // Initial grid: uniform spacing centered around zero
    let half_levels = (n_levels - 1) as f32 * 0.5;
    let mut grid: Vec<f32> = (0..n_levels)
        .map(|k| k as f32 - half_levels)
        .collect();

    // Optimization loop via Gumbel-Softmax gradient steps
    for step in 0..config.steps {
        let progress = step as f32 / config.steps.max(1) as f32;
        let tau = config.temperature_init * (config.temperature_min / config.temperature_init).powf(progress);

        // Compute assignment logits: -||w_i - scale * c_k||^2
        let mut d_grid = vec![0.0f32; n_levels];
        let mut d_scale = 0.0f32;

        for &w in data {
            let mut logits = vec![0.0f32; n_levels];
            let mut max_l = f32::NEG_INFINITY;
            for (k, &c) in grid.iter().enumerate() {
                let diff = w - scale * c;
                let logit = -(diff * diff);
                logits[k] = logit;
                if logit > max_l {
                    max_l = logit;
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            let mut probs = vec![0.0f32; n_levels];
            for k in 0..n_levels {
                let p = ((logits[k] - max_l) / tau).exp();
                probs[k] = p;
                sum_exp += p;
            }
            if sum_exp > 0.0 {
                for k in 0..n_levels {
                    probs[k] /= sum_exp;
                }
            }

            // Reconstruction and residual
            let mut recon = 0.0f32;
            for k in 0..n_levels {
                recon += probs[k] * scale * grid[k];
            }
            let err = recon - w;

            // Gradients w.r.t grid points and scale
            for k in 0..n_levels {
                d_grid[k] += err * probs[k] * scale;
                d_scale += err * probs[k] * grid[k];
            }
        }

        // Apply parameter updates
        let inv_n = 1.0 / num_weights as f32;
        for k in 0..n_levels {
            grid[k] -= config.lr * (d_grid[k] * inv_n);
        }
        scale -= (config.lr * 0.5) * (d_scale * inv_n);
        scale = scale.max(1e-6);

        // Maintain monotonic order of grid points
        grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    // Hard discretization step (argmax assignment)
    let mut codes = Vec::with_capacity(num_weights);
    let mut total_sq_err = 0.0f32;

    for &w in data {
        let mut best_k = 0u8;
        let mut best_diff = f32::MAX;
        for (k, &c) in grid.iter().enumerate() {
            let recon = scale * c;
            let diff = (w - recon).abs();
            if diff < best_diff {
                best_diff = diff;
                best_k = k as u8;
            }
        }
        codes.push(best_k);
        let recon = scale * grid[best_k as usize];
        total_sq_err += (w - recon) * (w - recon);
    }

    Ok(GsqBlockFit {
        codes,
        grid,
        scale,
        error: total_sq_err / num_weights as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gsq_2bit_reconstruction() {
        // Synthesize heavy-tailed distribution block
        let data: Vec<f32> = (0..64)
            .map(|i| {
                let x = (i as f32 - 32.0) / 8.0;
                x * x.abs() // heavy tails
            })
            .collect();

        let config = GsqConfig {
            steps: 20,
            ..Default::default()
        };

        let fit = gsq_fit_block(&data, 2, &config).expect("gsq fit succeeds");
        assert_eq!(fit.codes.len(), 64);
        assert_eq!(fit.grid.len(), 4);
        for &code in &fit.codes {
            assert!(code < 4);
        }
        // Reconstruction error should be finite and positive
        assert!(fit.error > 0.0 && fit.error.is_finite());
    }
}
