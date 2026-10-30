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
            // HW-tiled GreyRaven: same codes as Fp8Sparse24, but tile-coupled
            // patterns (shared across 16 rows and K-halves) prune less
            // optimally than independent groups. Same provisional tier until
            // E9 measures the coupled gate.
            QuantFormat::Fp8Sparse24Hw => Self {
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
            // IQ1_S/IQ1_M: 1.56/1.75 bpw lattice formats. The tightest tier
            // grim decodes; budgets reflect that honestly until measured
            // against a real IQ1 checkpoint oracle.
            QuantFormat::Iq1S | QuantFormat::Iq1M => Self {
                min_cosine_similarity: 0.9500,
                max_relative_l2_error: 0.45,
                max_delta_ppl: 0.50,
            },
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

/// Dense-KV-arena representations whose accuracy the gate certifies.
///
/// Mirrors the ROCm backend's `KvArenaFormat`, but defined here so grim-quant
/// keeps no backend dependency (the model/CLI layer maps one to the other).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvArenaKind {
    F32,
    F16,
    Int8,
    Int4,
    /// WS-3A: asymmetric 2-bit keys / 4-bit values.
    Int2V4,
}

/// Accuracy budget for a dense KV-arena representation, expressed as the
/// change measured against the f32 arena on the SAME corpus and window.
///
/// Deliberately separate from [`AccuracyTolerance`]: that one is sized for
/// weight quantization and its `max_delta_ppl` is an absolute delta, whereas a
/// KV arena is judged on the relative drift of an already-quantized model, and
/// additionally on step-0 logprob parity (the KV representation is the only
/// thing that changed, so the first decode step's logprobs isolate it).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KvArenaTolerance {
    /// Max `|ppl(mode) - ppl(f32)| / ppl(f32)` on the pinned corpus/window.
    pub max_relative_ppl_delta: f64,
    /// Max `|logprob(mode)[i] - logprob(f32)[i]|` at decode step 0.
    pub max_step0_logprob_abs_delta: f64,
    /// True while the budget is a provisional bound rather than a measured one.
    /// A provisional budget still fails loudly; it just has not been tuned by a
    /// sweep, and says so.
    pub provisional: bool,
}

impl KvArenaTolerance {
    pub fn for_kind(kind: KvArenaKind) -> Self {
        match kind {
            // The reference arm: no drift is permitted, and none can occur.
            KvArenaKind::F32 => Self {
                max_relative_ppl_delta: 0.0,
                max_step0_logprob_abs_delta: 0.0,
                provisional: false,
            },
            // f16 is a rounding of the f32 arena: ~3 significant decimal
            // digits. The PPL bound is that structural floor. The step-0 bound
            // is MEASURED, not guessed: on LFM2.5-350M-Q8_0 over 768 wikitext
            // tokens the f16 arena moved PPL by 0.0999% and the top-16 step-0
            // logprobs by at most 0.115 (argmax unchanged). 0.115 is larger
            // than f16 storage rounding alone predicts, so this bound is a
            // recorded measurement with margin — NOT a claim that the f16 arena
            // is a pure rounding. See `kv_arena_ppl_gate`.
            KvArenaKind::F16 => Self {
                max_relative_ppl_delta: 0.001,
                max_step0_logprob_abs_delta: 0.25,
                provisional: false,
            },
            // int8 per-(row, kv_head): symmetric 8-bit, one scale per
            // head-vector, dequant `(code-128)/127 * amax`.
            //
            // The step-0 bound is a RECORDED MEASUREMENT with margin, the same
            // rule the f16 entry uses. On Qwen3.5-9B-Q4_K_M (pinned corpus,
            // 3x256-token windows, `kv_arena_ppl_gate`) int8 moved PPL by
            // 0.0837% and the step-0 top-16 logprobs by 0.8597 (argmax
            // unchanged); 2.0 is 2.3x that, matching f16's 2.2x margin over its
            // own 0.115. The 0.02 this used to carry was set before anything
            // had been measured, and it was TIGHTER than f16's 0.25 for a
            // strictly coarser format — an inversion, not a finding.
            //
            // Note the step-0 probe reads row 0, whose attention has a single
            // KV entry: it is the arena's worst case by construction (the
            // stored V comes back verbatim, nothing averages out). The
            // decode-like row of the same probe measures 0.0997 for int8 and
            // 0.0097 for f16, a ~10x ladder per format step.
            KvArenaKind::Int8 => Self {
                max_relative_ppl_delta: 0.02,
                max_step0_logprob_abs_delta: 2.0,
                provisional: true,
            },
            // int4: coarser (15 levels); the same tier as the Q4K weight budget.
            // Measured 3.3890 at step 0 on the same checkpoint (the decode-like
            // row: 0.5097); 8.0 is 2.4x that. int4 additionally MOVES THE
            // ARGMAX at row 0, which the gate's structural check catches
            // separately — this budget does not, and should not, excuse it.
            KvArenaKind::Int4 => Self {
                max_relative_ppl_delta: 0.08,
                max_step0_logprob_abs_delta: 8.0,
                provisional: true,
            },
            // K2V4: the KEY side has only THREE levels, so it is strictly
            // coarser than int4's 15 and cannot inherit its budget. Unmeasured
            // against a real model — provisional, and deliberately loose so it
            // fails loudly only on gross corruption rather than on the 2-bit
            // coarseness it was designed for. It must stay above int4's
            // measured 3.389, which the 0.60 it used to carry did not: it was
            // below a FINER format's measurement.
            KvArenaKind::Int2V4 => Self {
                max_relative_ppl_delta: 0.25,
                max_step0_logprob_abs_delta: 16.0,
                provisional: true,
            },
        }
    }
}

/// Result of certifying one dense-KV-arena representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KvArenaVerdict {
    Pass {
        relative_ppl_delta: f64,
        step0_logprob_abs_delta: f64,
    },
    Fail {
        reason: String,
        relative_ppl_delta: f64,
        step0_logprob_abs_delta: f64,
        tolerance: KvArenaTolerance,
    },
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
    kv_arena_tolerances: HashMap<KvArenaKind, KvArenaTolerance>,
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
            kv_arena_tolerances: HashMap::new(),
        }
    }

    /// Set a custom tolerance for a given quantization format.
    pub fn set_tolerance(&mut self, format: QuantFormat, tolerance: AccuracyTolerance) {
        self.custom_tolerances.insert(format, tolerance);
    }

    /// Override the KV-arena budget for one representation (used once a sweep
    /// has measured a tighter or looser bound than the provisional default).
    pub fn set_kv_arena_tolerance(&mut self, kind: KvArenaKind, tolerance: KvArenaTolerance) {
        self.kv_arena_tolerances.insert(kind, tolerance);
    }

    /// Certify a dense-KV-arena representation against the f32 arena.
    ///
    /// `baseline_ppl` is the f32-arm perplexity; `candidate_ppl` the mode under
    /// test. `step0_logprob_f32` / `step0_logprob_mode` are the decode-step-0
    /// next-token logprobs for the two arms (same length, same positions).
    ///
    /// Both arms must be produced on the same corpus, window and context, so
    /// the only variable is the arena representation.
    pub fn verify_kv_arena(
        &self,
        kind: KvArenaKind,
        baseline_ppl: f64,
        candidate_ppl: f64,
        step0_logprob_f32: &[f32],
        step0_logprob_mode: &[f32],
    ) -> Result<KvArenaVerdict> {
        if step0_logprob_f32.len() != step0_logprob_mode.len() {
            return Err(Error::Shape(format!(
                "kv-arena step-0 logprob length mismatch: {} != {}",
                step0_logprob_f32.len(),
                step0_logprob_mode.len()
            )));
        }
        if !(baseline_ppl.is_finite() && baseline_ppl > 0.0) {
            return Err(Error::Shape(format!(
                "kv-arena baseline ppl is not a positive finite number: {baseline_ppl}"
            )));
        }
        if !candidate_ppl.is_finite() {
            return Err(Error::Shape(format!(
                "kv-arena candidate ppl is not finite: {candidate_ppl}"
            )));
        }

        let tol = self
            .kv_arena_tolerances
            .get(&kind)
            .copied()
            .unwrap_or_else(|| KvArenaTolerance::for_kind(kind));

        let relative_ppl_delta = (candidate_ppl - baseline_ppl).abs() / baseline_ppl;
        let step0_logprob_abs_delta = step0_logprob_f32
            .iter()
            .zip(step0_logprob_mode)
            .map(|(a, b)| (*a - *b).abs() as f64)
            .fold(0.0f64, f64::max);

        if relative_ppl_delta > tol.max_relative_ppl_delta {
            return Ok(KvArenaVerdict::Fail {
                reason: format!(
                    "relative PPL delta {:.6} exceeded budget {:.6}{}",
                    relative_ppl_delta,
                    tol.max_relative_ppl_delta,
                    if tol.provisional { " (provisional)" } else { "" }
                ),
                relative_ppl_delta,
                step0_logprob_abs_delta,
                tolerance: tol,
            });
        }
        if step0_logprob_abs_delta > tol.max_step0_logprob_abs_delta {
            return Ok(KvArenaVerdict::Fail {
                reason: format!(
                    "step-0 logprob max-abs delta {:.6} exceeded budget {:.6}{}",
                    step0_logprob_abs_delta,
                    tol.max_step0_logprob_abs_delta,
                    if tol.provisional { " (provisional)" } else { "" }
                ),
                relative_ppl_delta,
                step0_logprob_abs_delta,
                tolerance: tol,
            });
        }
        Ok(KvArenaVerdict::Pass {
            relative_ppl_delta,
            step0_logprob_abs_delta,
        })
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

    #[test]
    fn kv_arena_tolerance_is_ordered_by_precision() {
        // A coarser representation may never be given a tighter budget than a
        // finer one — the ordering is the property that makes the ladder
        // meaningful.
        //
        // Both columns are checked. The step-0 column was NOT, and it carried
        // exactly the inversion this test exists to catch: int8's bound was
        // 0.02 against f16's 0.25, i.e. an 8-bit arena held to 1/12th of the
        // f16 arena's tolerance. Nothing failed, because nothing looked.
        let f32t = KvArenaTolerance::for_kind(KvArenaKind::F32);
        let f16t = KvArenaTolerance::for_kind(KvArenaKind::F16);
        let i8t = KvArenaTolerance::for_kind(KvArenaKind::Int8);
        let i4t = KvArenaTolerance::for_kind(KvArenaKind::Int4);
        let i2v4t = KvArenaTolerance::for_kind(KvArenaKind::Int2V4);
        assert!(f32t.max_relative_ppl_delta <= f16t.max_relative_ppl_delta);
        assert!(f16t.max_relative_ppl_delta <= i8t.max_relative_ppl_delta);
        assert!(i8t.max_relative_ppl_delta <= i4t.max_relative_ppl_delta);
        assert!(i4t.max_relative_ppl_delta <= i2v4t.max_relative_ppl_delta);
        assert!(f32t.max_step0_logprob_abs_delta <= f16t.max_step0_logprob_abs_delta);
        assert!(f16t.max_step0_logprob_abs_delta <= i8t.max_step0_logprob_abs_delta);
        assert!(i8t.max_step0_logprob_abs_delta <= i4t.max_step0_logprob_abs_delta);
        assert!(i4t.max_step0_logprob_abs_delta <= i2v4t.max_step0_logprob_abs_delta);
        // The reference arm is exact, and f16 is a rounding, not a tunable.
        assert_eq!(f32t.max_relative_ppl_delta, 0.0);
        assert!(!f16t.provisional, "f16's bound is structural, not provisional");
        assert!(i8t.provisional && i4t.provisional);
    }

    #[test]
    fn kv_arena_step0_budget_admits_the_measured_ladder() {
        // The budgets are MEASURED envelopes, so the measurement they were
        // derived from must fit inside them with room to spare. These are the
        // Qwen3.5-9B-Q4_K_M / `kv_arena_ppl_gate` step-0 top-16 max-abs deltas
        // at PROBE_LEN = 32 (see the plan's Gate-G2 table): f16 0.0152,
        // int8 0.8597, int4 3.3890 — a ~10x ladder per format step at the
        // decode-like row, and the numbers here are the row-0 worst case.
        let i8t = KvArenaTolerance::for_kind(KvArenaKind::Int8);
        let i4t = KvArenaTolerance::for_kind(KvArenaKind::Int4);
        let i2v4t = KvArenaTolerance::for_kind(KvArenaKind::Int2V4);
        assert!(
            i8t.max_step0_logprob_abs_delta > 0.8597,
            "int8's measured step-0 drift no longer fits its budget"
        );
        assert!(
            i4t.max_step0_logprob_abs_delta > 3.3890,
            "int4's measured step-0 drift no longer fits its budget"
        );
        // A budget that admits everything is not a budget: a wrong scale axis
        // (the failure class this column exists to catch) moves the top-16 by
        // tens of nats or sends it non-finite.
        assert!(i2v4t.max_step0_logprob_abs_delta < 64.0);
    }

    #[test]
    fn kv_arena_gate_passes_small_drift_and_fails_gross_drift() {
        let gate = AccuracyGate::new();
        let base = 20.0f64;
        let step0_f32 = [1.0f32, -2.0, 0.5];

        // f16: 0.05% ppl drift, step-0 within 2e-3 -> pass.
        let v = gate
            .verify_kv_arena(
                KvArenaKind::F16,
                base,
                base * 1.0005,
                &step0_f32,
                &[1.0005, -2.0004, 0.5004],
            )
            .unwrap();
        assert!(matches!(v, KvArenaVerdict::Pass { .. }), "got {v:?}");

        // int4 with a 50% ppl blow-up -> fail, and the verdict names the budget.
        let v = gate
            .verify_kv_arena(KvArenaKind::Int4, base, base * 1.5, &step0_f32, &step0_f32)
            .unwrap();
        match v {
            KvArenaVerdict::Fail { reason, tolerance, .. } => {
                assert!(reason.contains("relative PPL delta"), "reason: {reason}");
                assert!(tolerance.provisional, "int4 budget is provisional");
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn kv_arena_gate_flags_step0_desync_even_when_ppl_is_flat() {
        // A representation can leave aggregate PPL untouched while corrupting
        // the first step (e.g. a mis-scaled sink row). The step-0 arm is what
        // catches that; a PPL-only gate would pass it.
        let gate = AccuracyGate::new();
        let v = gate
            .verify_kv_arena(
                KvArenaKind::F16,
                20.0,
                20.0,
                &[1.0f32, -2.0],
                &[1.0, -1.0], // 1.0 abs delta, way over 2e-3
            )
            .unwrap();
        match v {
            KvArenaVerdict::Fail { reason, .. } => {
                assert!(reason.contains("step-0 logprob"), "reason: {reason}")
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn kv_arena_gate_rejects_bad_inputs() {
        let gate = AccuracyGate::new();
        // Length mismatch is a harness bug, not a verdict.
        assert!(
            gate.verify_kv_arena(KvArenaKind::F16, 20.0, 20.0, &[1.0], &[1.0, 2.0])
                .is_err()
        );
        // A non-positive baseline would make the relative delta meaningless.
        assert!(
            gate.verify_kv_arena(KvArenaKind::F16, 0.0, 20.0, &[], &[])
                .is_err()
        );
    }
}
