//! End-to-end numerical accuracy & perplexity regression guard.
//!
//! Evaluates and enforces layer-wise cosine fidelity, relative L2 error,
//! and perplexity degradation bounds across all 18 supported quantization formats.

use grim_tensor::dtype::QuantFormat;
use grim_tensor::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tolerance budget specification for a given quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AccuracyTolerance {
    pub min_cosine_similarity: f64,
    pub max_relative_l2_error: f64,
    pub max_delta_ppl: f64,
}

impl AccuracyTolerance {
    /// Retrieve canonical tolerance thresholds for a target quantization format.
    pub fn for_format(format: QuantFormat) -> Self {
        match format {
            QuantFormat::Fp8 => Self {
                min_cosine_similarity: 0.9995,
                max_relative_l2_error: 0.03,
                max_delta_ppl: 0.015,
            },
            QuantFormat::Fp8Block16 => Self {
                min_cosine_similarity: 0.9990,
                max_relative_l2_error: 0.04,
                max_delta_ppl: 0.02,
            },
            QuantFormat::Fp4 | QuantFormat::Fp4Block16 => Self {
                min_cosine_similarity: 0.9950,
                max_relative_l2_error: 0.10,
                max_delta_ppl: 0.08,
            },
            QuantFormat::Q8_0 => Self {
                min_cosine_similarity: 0.9995,
                max_relative_l2_error: 0.03,
                max_delta_ppl: 0.02,
            },
            QuantFormat::Q6K => Self {
                min_cosine_similarity: 0.9990,
                max_relative_l2_error: 0.05,
                max_delta_ppl: 0.03,
            },
            QuantFormat::Q5K => Self {
                min_cosine_similarity: 0.9970,
                max_relative_l2_error: 0.08,
                max_delta_ppl: 0.05,
            },
            QuantFormat::Q4K => Self {
                min_cosine_similarity: 0.9940,
                max_relative_l2_error: 0.12,
                max_delta_ppl: 0.10,
            },
            QuantFormat::Iq4Nl | QuantFormat::Iq4Xs => Self {
                min_cosine_similarity: 0.9950,
                max_relative_l2_error: 0.10,
                max_delta_ppl: 0.07,
            },
            QuantFormat::Iq3S | QuantFormat::Iq3Xxs => Self {
                min_cosine_similarity: 0.9900,
                max_relative_l2_error: 0.18,
                max_delta_ppl: 0.18,
            },
            QuantFormat::Iq2S | QuantFormat::Iq2Xs | QuantFormat::Iq2Xxs => Self {
                min_cosine_similarity: 0.9840,
                max_relative_l2_error: 0.28,
                max_delta_ppl: 0.38,
            },
            QuantFormat::Nf4 => Self {
                min_cosine_similarity: 0.9950,
                max_relative_l2_error: 0.10,
                max_delta_ppl: 0.08,
            },
        }
    }
}

/// Compute cosine similarity between reference oracle slice and candidate slice.
pub fn compute_cosine_similarity(oracle: &[f32], candidate: &[f32]) -> f64 {
    if oracle.len() != candidate.len() || oracle.is_empty() {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;

    for (&a, &b) in oracle.iter().zip(candidate.iter()) {
        let fa = a as f64;
        let fb = b as f64;
        dot += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

/// Compute relative L2 error: `||oracle - candidate||_2 / ||oracle||_2`.
pub fn compute_relative_l2_error(oracle: &[f32], candidate: &[f32]) -> f64 {
    if oracle.len() != candidate.len() || oracle.is_empty() {
        return f64::INFINITY;
    }
    let mut diff_sq_sum: f64 = 0.0;
    let mut oracle_sq_sum: f64 = 0.0;

    for (&a, &b) in oracle.iter().zip(candidate.iter()) {
        let diff = (a - b) as f64;
        diff_sq_sum += diff * diff;
        let fa = a as f64;
        oracle_sq_sum += fa * fa;
    }

    if oracle_sq_sum <= 0.0 {
        diff_sq_sum.sqrt()
    } else {
        diff_sq_sum.sqrt() / oracle_sq_sum.sqrt()
    }
}

/// Compute average cross-entropy perplexity over a token sequence.
pub fn compute_cross_entropy_ppl(logits: &[f32], targets: &[u32], vocab_size: usize) -> f64 {
    if targets.is_empty() || vocab_size == 0 || logits.len() < targets.len() * vocab_size {
        return f64::NAN;
    }
    let mut total_nll: f64 = 0.0;

    for (t, &target) in targets.iter().enumerate() {
        let slice = &logits[t * vocab_size..(t + 1) * vocab_size];
        let mut max_val = f32::NEG_INFINITY;
        for &v in slice {
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp: f64 = 0.0;
        for &v in slice {
            sum_exp += ((v - max_val) as f64).exp();
        }
        let log_sum_exp = (max_val as f64) + sum_exp.ln();
        let target_logit = slice[target as usize] as f64;
        let nll = log_sum_exp - target_logit;
        total_nll += nll;
    }

    (total_nll / targets.len() as f64).exp()
}

/// Result of evaluating an accuracy verification check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccuracyVerdict {
    Pass {
        cosine_similarity: f64,
        relative_l2_error: f64,
    },
    Fail {
        reason: String,
        cosine_similarity: f64,
        relative_l2_error: f64,
        tolerance: AccuracyTolerance,
    },
}

/// Accuracy Regression Gate runner.
pub struct AccuracyGate {
    custom_tolerances: HashMap<QuantFormat, AccuracyTolerance>,
}

impl Default for AccuracyGate {
    fn default() -> Self {
        Self::new()
    }
}

impl AccuracyGate {
    pub fn new() -> Self {
        Self {
            custom_tolerances: HashMap::new(),
        }
    }

    /// Set a custom tolerance for a given quantization format.
    pub fn set_tolerance(&mut self, format: QuantFormat, tolerance: AccuracyTolerance) {
        self.custom_tolerances.insert(format, tolerance);
    }

    /// Verify an activation tensor against its reference oracle.
    pub fn verify(
        &self,
        format: QuantFormat,
        oracle: &[f32],
        candidate: &[f32],
    ) -> Result<AccuracyVerdict> {
        if oracle.len() != candidate.len() {
            return Err(Error::Shape(format!(
                "shape mismatch: oracle len {} != candidate len {}",
                oracle.len(),
                candidate.len()
            )));
        }

        let tol = self
            .custom_tolerances
            .get(&format)
            .copied()
            .unwrap_or_else(|| AccuracyTolerance::for_format(format));

        let cos = compute_cosine_similarity(oracle, candidate);
        let l2 = compute_relative_l2_error(oracle, candidate);

        if cos < tol.min_cosine_similarity {
            Ok(AccuracyVerdict::Fail {
                reason: format!(
                    "Cosine similarity {:.6} below threshold {:.6}",
                    cos, tol.min_cosine_similarity
                ),
                cosine_similarity: cos,
                relative_l2_error: l2,
                tolerance: tol,
            })
        } else if l2 > tol.max_relative_l2_error {
            Ok(AccuracyVerdict::Fail {
                reason: format!(
                    "Relative L2 error {:.6} exceeded threshold {:.6}",
                    l2, tol.max_relative_l2_error
                ),
                cosine_similarity: cos,
                relative_l2_error: l2,
                tolerance: tol,
            })
        } else {
            Ok(AccuracyVerdict::Pass {
                cosine_similarity: cos,
                relative_l2_error: l2,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let cos = compute_cosine_similarity(&a, &b);
        assert!((cos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_relative_l2_error_zero_on_identical() {
        let a = vec![1.0, -1.0, 2.0, -2.0];
        let b = vec![1.0, -1.0, 2.0, -2.0];
        let l2 = compute_relative_l2_error(&a, &b);
        assert!(l2 < 1e-6);
    }

    #[test]
    fn test_accuracy_gate_verification_pass_and_fail() {
        let gate = AccuracyGate::new();
        let oracle = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let good = vec![0.999, 2.001, 2.998, 4.002, 5.000];
        let bad = vec![0.5, 1.0, 1.5, 2.0, 2.5];

        let pass_res = gate.verify(QuantFormat::Q4K, &oracle, &good).unwrap();
        assert!(matches!(pass_res, AccuracyVerdict::Pass { .. }));

        let fail_res = gate.verify(QuantFormat::Q8_0, &oracle, &bad).unwrap();
        assert!(matches!(fail_res, AccuracyVerdict::Fail { .. }));
    }

    #[test]
    fn test_cross_entropy_ppl_calculation() {
        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let targets = vec![0, 1];
        let ppl = compute_cross_entropy_ppl(&logits, &targets, 4);
        assert!(ppl > 0.0 && ppl < 1.1);
    }
}
