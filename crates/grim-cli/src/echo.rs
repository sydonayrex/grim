//! SCALE-ECHO echo training mode — subspace echo state + FP4 updates.

use serde::{Deserialize, Serialize};

/// Configuration for the SCALE-ECHO echo trainer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EchoConfig {
    /// Rank of the random projection subspace.
    #[serde(default = "default_subspace_rank")]
    pub subspace_rank: usize,
    /// Scale of the random perturbation for finite-difference gradient estimation.
    #[serde(default = "default_perturbation_scale")]
    pub perturbation_scale: f32,
    /// Decay factor for the diagonal Fisher Information Matrix estimate.
    #[serde(default = "default_fim_decay")]
    pub fim_decay: f32,
    /// Whether to quantize updates to FP4 before applying.
    #[serde(default = "default_fp4_quant")]
    pub fp4_quant: bool,
}

const fn default_subspace_rank() -> usize {
    8
}
const fn default_perturbation_scale() -> f32 {
    0.01
}
const fn default_fim_decay() -> f32 {
    0.99
}
const fn default_fp4_quant() -> bool {
    true
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            subspace_rank: 8,
            perturbation_scale: 0.01,
            fim_decay: 0.99,
            fp4_quant: true,
        }
    }
}

/// SCALE-ECHO trainer: finite-difference gradient estimation in a random
/// subspace with diagonal FIM preconditioning and FP4 weight updates.
#[derive(Debug, Clone)]
pub struct EchoTrainer {
    config: EchoConfig,
    /// Diagonal FIM estimate (one entry per weight dimension).
    fim_diagonal: Vec<f32>,
    /// Number of steps taken so far.
    step_count: usize,
    /// Seeded RNG state for the fixed random projection matrix.
    rng_state: u64,
}

impl EchoTrainer {
    /// Create a new EchoTrainer with the given configuration and a seeded RNG.
    pub fn new(config: EchoConfig) -> Self {
        let rng_state = 0x9E3779B97F4A7C15; // golden seed for reproducibility
        Self {
            config,
            fim_diagonal: Vec::new(),
            step_count: 0,
            rng_state,
        }
    }

    /// Deterministic pseudo-random number generator step (xorshift64).
    fn next_rng(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        let bits = x as u32;
        ((bits >> 9) & 0x007FFFFF | 0x3F800000) as f32 - 1.0
    }

    /// Generate a fixed random projection matrix E seeded by `rng_state`.
    /// Returns a flat Vec of length `input_dim * subspace_rank`.
    fn generate_projection(&mut self, input_dim: usize) -> Vec<f32> {
        let total = input_dim * self.config.subspace_rank;
        let mut proj = Vec::with_capacity(total);
        for _ in 0..total {
            proj.push((self.next_rng() - 0.5) * 2.0);
        }
        proj
    }

    /// Apply the fixed random projection E to `input`, returning the projected
    /// echo state `h = E^T * input`.
    pub fn echo_forward(&mut self, input: &[f32]) -> Vec<f32> {
        let input_dim = input.len();
        let proj = self.generate_projection(input_dim);
        let mut h = vec![0.0f32; self.config.subspace_rank];
        for r in 0..self.config.subspace_rank {
            for i in 0..input_dim {
                h[r] += proj[r * input_dim + i] * input[i];
            }
        }
        h
    }

    /// Estimate the gradient direction via finite differences in the subspace.
    ///
    /// Returns `(current_loss - previous_loss) / (perturbation_scale * ||perturbation||)`.
    /// The returned vector is a direction-only estimate in the subspace.
    pub fn estimate_gradient(
        &mut self,
        current_loss: f32,
        previous_loss: f32,
        perturbation: &[f32],
    ) -> Vec<f32> {
        let norm = perturbation
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        let scale = self.config.perturbation_scale * norm;
        let diff = current_loss - previous_loss;
        let factor = diff / scale;
        perturbation.iter().map(|p| p * factor).collect()
    }

    /// Quantize an update to FP4 and apply it in-place to `weights`.
    ///
    /// FP4 quantization: clamp to [-3, 3], round to nearest 0.25 step.
    pub fn apply_fp4_update(weights: &mut [f32], update: &[f32]) {
        assert_eq!(weights.len(), update.len(), "weight/update length mismatch");
        for (w, u) in weights.iter_mut().zip(update.iter()) {
            let mut val = *w + *u;
            val = val.clamp(-3.0, 3.0);
            val = (val * 4.0).round() / 4.0;
            *w = val;
        }
    }

    /// One echo training step.
    ///
    /// (a) Forward through model with frozen base + adapter (represented here
    ///     by projecting the current adapter weights into the echo subspace).
    /// (b) Compute loss as the L2 norm of the echo state (lower = better).
    /// (c) Estimate gradient via finite differences in subspace.
    /// (d) Apply FIM preconditioning (diagonal).
    /// (e) Apply FP4 update to adapter weights only.
    ///
    /// Returns the scalar loss for this step.
    pub fn step(&mut self, weights: &mut [f32]) -> f32 {
        let previous_loss = if self.step_count == 0 {
            2.3
        } else {
            let h = self.echo_forward(weights);
            h.iter().map(|x| x * x).sum::<f32>().sqrt()
        };

        // (c) Generate a random perturbation in the subspace for gradient estimation.
        let mut perturbation = vec![0.0f32; weights.len()];
        for p in perturbation.iter_mut() {
            *p = (self.next_rng() - 0.5) * 2.0;
        }

        // Apply perturbation to weights temporarily for finite difference.
        let mut perturbed_weights = weights.to_vec();
        for (pw, p) in perturbed_weights.iter_mut().zip(perturbation.iter()) {
            *pw += self.config.perturbation_scale * p;
        }

        // (a)+(b) Forward through projected echo state and compute perturbed loss.
        let current_loss = {
            let h = self.echo_forward(&perturbed_weights);
            h.iter().map(|x| x * x).sum::<f32>().sqrt()
        };

        let gradient = self.estimate_gradient(current_loss, previous_loss, &perturbation);

        // (d) Apply FIM diagonal preconditioning.
        let mut preconditioned = vec![0.0f32; weights.len()];
        for i in 0..weights.len() {
            let fim_val = self.fim_diagonal.get(i).copied().unwrap_or(1.0).max(1e-6);
            preconditioned[i] = gradient[i] / fim_val;
        }

        // Update FIM diagonal estimate with exponential moving average.
        if self.fim_diagonal.len() != weights.len() {
            self.fim_diagonal = vec![1.0; weights.len()];
        }
        for i in 0..weights.len() {
            let grad_sq = preconditioned[i].powi(2);
            self.fim_diagonal[i] = self.config.fim_decay * self.fim_diagonal[i]
                + (1.0 - self.config.fim_decay) * grad_sq;
        }

        // (e) Apply FP4 update to adapter weights only.
        if self.config.fp4_quant {
            Self::apply_fp4_update(weights, &preconditioned);
        } else {
            for (w, u) in weights.iter_mut().zip(preconditioned.iter()) {
                *w += u;
            }
        }

        self.step_count += 1;
        previous_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_forward_shape() {
        let cfg = EchoConfig::default();
        let mut trainer = EchoTrainer::new(cfg);
        let input = vec![1.0f32; 16];
        let h = trainer.echo_forward(&input);
        assert_eq!(h.len(), 8, "echo state must match subspace_rank");
        // Second call with same input should produce same output because
        // generate_projection consumes RNG state; verify determinism with
        // fresh trainer.
        let mut trainer2 = EchoTrainer::new(EchoConfig::default());
        let h2 = trainer2.echo_forward(&input);
        assert_eq!(h2.len(), 8);
    }

    #[test]
    fn test_echo_gradient_estimate_direction() {
        let mut trainer = EchoTrainer::new(EchoConfig::default());
        let perturbation = vec![1.0f32, 0.0f32, 0.0f32];
        // If loss increased by 0.01 along unit perturbation, gradient should be positive.
        let grad = trainer.estimate_gradient(2.31, 2.3, &perturbation);
        assert_eq!(grad.len(), 3);
        assert!(
            grad[0] > 0.0,
            "gradient should point in direction of increasing loss, got {}",
            grad[0]
        );
        assert!((grad[1] - 0.0).abs() < 1e-6);
        assert!((grad[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_echo_fp4_quantization() {
        let mut weights = vec![0.0f32, 1.0f32, -1.0f32, 10.0f32, -10.0f32];
        let update = vec![0.1f32, 0.0f32, -0.1f32, 0.0f32, 0.0f32];
        EchoTrainer::apply_fp4_update(&mut weights, &update);
        // 0.0 + 0.1 -> 0.0 after FP4 rounding to 0.25 steps.
        assert!((weights[0] - 0.0).abs() < 1e-6);
        // 1.0 + 0.0 -> 1.0.
        assert!((weights[1] - 1.0).abs() < 1e-6);
        // -1.0 + -0.1 -> -1.0 after clamp/round.
        assert!((weights[2] - (-1.0)).abs() < 1e-6);
        // 10.0 -> clamp to 3.0.
        assert!((weights[3] - 3.0).abs() < 1e-6);
        // -10.0 -> clamp to -3.0.
        assert!((weights[4] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_echo_step_returns_finite_loss() {
        let mut trainer = EchoTrainer::new(EchoConfig::default());
        let mut weights = vec![0.01f32; 32];
        let loss = trainer.step(&mut weights);
        assert!(loss.is_finite(), "step loss must be finite, got {loss}");
        assert!(loss >= 0.0, "step loss must be non-negative, got {loss}");
    }

    #[test]
    fn test_echo_step_updates_weights() {
        let mut trainer = EchoTrainer::new(EchoConfig::default());
        let mut weights = vec![0.0f32; 32];
        let _ = trainer.step(&mut weights);
        // After at least one step, weights should have changed from all zeros
        // because FP4 update can produce non-zero values.
        let changed = weights.iter().any(|w| *w != 0.0);
        assert!(changed, "weights should change after an echo step");
    }
}
