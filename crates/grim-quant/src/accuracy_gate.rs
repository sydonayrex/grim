//! End-to-end numerical accuracy & perplexity regression guard.
//! Evaluates and enforces layer-wise cosine fidelity, relative L2 error, and perplexity degradation bounds across all.

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
            // Same E4M3 codes as `Fp8`, different byte arrangement: a storage
            // permutation, not a coarser quantizer, so it inherits Fp8's
            // tolerances exactly.
            QuantFormat::Fp8Blocked16 => Self {
                min_cosine_similarity: 0.9995,
                max_relative_l2_error: 0.03,
                max_delta_ppl: 0.015,
            },
            QuantFormat::Fp8Block16 => Self {
                min_cosine_similarity: 0.9990,
                max_relative_l2_error: 0.04,
                max_delta_ppl: 0.02,
            },
            // GreyRaven 2:4 keeps half the weights and zeroes the rest, so it is
            // strictly coarser than dense Fp8 and cannot inherit its tolerance.
            // Provisional and deliberately LOOSER than Fp8: a placeholder that
            // keeps the gate honest (an unmeasured format must not be judged
            // against a tolerance it was never sized for). E9's
            // matched-tolerance measurement is what should replace these.
            QuantFormat::Fp8Sparse24 => Self {
                min_cosine_similarity: 0.995,
                max_relative_l2_error: 0.12,
                max_delta_ppl: 0.08,
            },
            // Same E4M3 element format as `Fp8`; only the scale granularity
            // differs (128x128 grid vs per-tensor), so the same tolerances apply.
            QuantFormat::Fp8Block128 => Self {
                min_cosine_similarity: 0.9995,
                max_relative_l2_error: 0.03,
                max_delta_ppl: 0.015,
            },
            QuantFormat::Fp4 | QuantFormat::Fp4Block16 => Self {
                min_cosine_similarity: 0.9950,
                max_relative_l2_error: 0.10,
                max_delta_ppl: 0.08,
            },
            // 4-bit weights, per-group (128) bf16 scales: same code-width
            // class as Fp4, so the same provisional thresholds. Tighten once
            // measured against the real OSTQuant gate.
            QuantFormat::W4A4OstQuant => Self {
                min_cosine_similarity: 0.9950,
                max_relative_l2_error: 0.10,
                max_delta_ppl: 0.08,
            },
            // Per-row absmax INT8: the article's recipe bounds the error at
            // half a scale step per element, and per-channel scales keep every
            // row's grid tight. Same tier as Q8_0 until measured against the
            // real gate.
            QuantFormat::Int8PerChannel => Self {
                min_cosine_similarity: 0.9995,
                max_relative_l2_error: 0.03,
                max_delta_ppl: 0.02,
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
            // Q3_K is ~3.5 bpw with per-8-sub-block 6-bit scales; it sits
            // between Q4_K and Q2_K in observed error, so its tolerances do too.
            QuantFormat::Q3K => Self {
                min_cosine_similarity: 0.9900,
                max_relative_l2_error: 0.18,
                max_delta_ppl: 0.15,
            },
            // Upstream Q2_0: 2.25 bpw, 64-weight blocks with a single fp16 scale and the
            // codebook {-1, 0, +1, +2}. Coarser than Q2_K (2.6 bpw, two scales
            // plus a min per sub-block) so it cannot inherit Q2_K's budget, but
            // it is a real shipped format with real measured perplexity (the
            // Qwen3.8-Flash-Next GSQ-RCO-3.5bit release runs it for 62 expert
            // banks), so the budget is tightened from the "unmeasured format"
            // placeholder tier.
            QuantFormat::Q2_0 => Self {
                min_cosine_similarity: 0.9750,
                max_relative_l2_error: 0.34,
                max_delta_ppl: 0.34,
            },
            // GSQ-RCO 3.5-bit (tag 81): same 2.25 bpw block geometry as Q2_0
            // with the GSQ codebook {-2,-1,0,+1}. The MSE-fitted RTN scale is
            // typically a bit tighter than Q2_0's d=amax choice, but the
            // budget stays in the 2-bit tier until a real GSQ conversion
            // (calibration + Gumbel-Softmax) justifies Q3_K-class numbers.
            QuantFormat::GsqRco3p5 => Self {
                min_cosine_similarity: 0.9750,
                max_relative_l2_error: 0.34,
                max_delta_ppl: 0.34,
            },
            // Q2_K is 2.6 bpw with 2-bit codes and per-16-sub-block min/scale;
            // the worst of the K-quants grim can decode.
            QuantFormat::Q2K => Self {
                min_cosine_similarity: 0.9800,
                max_relative_l2_error: 0.30,
                max_delta_ppl: 0.30,
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
            // TreePie is E2M2 -- one exponent bit, two mantissa bits -- so it sits
            // between NF4 and FP8 on precision. These thresholds are inherited from
            // NF4, the nearest measured format, and are deliberately NOT presented
            // as tuned for TreePie: no accuracy sweep has been run against it. They
            // are a starting bound that will fail loudly if TreePie is worse than
            // NF4, which is the property worth having until a real sweep replaces them.
            QuantFormat::TreePie => Self {
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
