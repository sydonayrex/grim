//! Concrete `ConfidenceHead` impl: per-position confidence derived from
//! a softmax-entropy heuristic over the base logits.
//!
//! §5.3.2: the verifier uses these per-position scores to decide how
//! many draft tokens to verify. Real implementations train a learned
//! predictor against the target's accept/reject history. This structural
//! impl uses a deterministic entropy-based estimate: sharper peaks (low
//! entropy) → high confidence; flat distributions → low confidence.

use std::sync::Arc;

use crate::confidence_head::ConfidenceHead;
use crate::draft_backbone::DraftBlock;

/// Confidence from softmax entropy: `confidence = exp(-entropy)`.
pub struct EntropyConfidenceHead;

impl ConfidenceHead for EntropyConfidenceHead {
    fn score(&self, draft_block: &DraftBlock) -> Vec<f32> {
        let logits = match draft_block.base_logits.to_vec_f32() {
            Ok(v) => v,
            Err(_) => return vec![0.0; draft_block.tokens.len()],
        };
        let shape = draft_block.base_logits.shape().dims().to_vec();
        if shape.len() != 2 {
            return vec![0.0; draft_block.tokens.len()];
        }
        let (positions, vocab) = (shape[0], shape[1]);
        let mut out = Vec::with_capacity(positions);
        for t in 0..positions {
            let row = &logits[t * vocab..(t + 1) * vocab];
            // Numerically stable softmax.
            let mut max = f32::NEG_INFINITY;
            for v in row {
                if *v > max {
                    max = *v;
                }
            }
            let mut sum = 0.0f32;
            let mut probs = vec![0.0f32; vocab];
            for v in 0..vocab {
                probs[v] = (row[v] - max).exp();
                sum += probs[v];
            }
            for v in 0..vocab {
                probs[v] /= sum;
            }
            // Entropy.
            let mut h = 0.0f32;
            for v in 0..vocab {
                if probs[v] > 1e-9 {
                    h -= probs[v] * probs[v].ln();
                }
            }
            // exp(-H) ∈ (0, 1], with higher = more confident.
            out.push({
                if h.is_finite() {
                    (-h).exp()
                } else {
                    1.0
                }
            });
        }
        out
    }
}

impl From<EntropyConfidenceHead> for Arc<dyn ConfidenceHead> {
    fn from(c: EntropyConfidenceHead) -> Self {
        Arc::new(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::Shape;

    #[test]
    fn confidence_high_for_peaked_distribution() {
        let head = EntropyConfidenceHead;
        let mut logits = vec![0.0f32; 2 * 16];
        // Position 0: peaked at id 5.
        logits[0 * 16 + 5] = 10.0;
        // Position 1: flat (low confidence).
        for v in 0..16 {
            logits[1 * 16 + v] = 0.0;
        }
        let block = DraftBlock {
            tokens: vec![5, 0],
            base_logits: grim_backend_cpu::cpu_tensor(logits, Shape::new(vec![2, 16])),
            confidence: Vec::new(),
        };
        let scores = head.score(&block);
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "peaked distribution must yield higher confidence than flat"
        );
        // Peaked → near 1.0; flat → significantly lower.
        assert!(scores[0] > 0.5);
        assert!(scores[1] < scores[0]);
    }

    #[test]
    fn confidence_in_unit_interval() {
        let head = EntropyConfidenceHead;
        let mut logits = vec![0.0f32; 3 * 8];
        for i in 0..8 {
            logits[i] = i as f32;
            logits[8 + i] = i as f32 * 0.5;
            logits[16 + i] = -(i as f32);
        }
        let block = DraftBlock {
            tokens: vec![0, 0, 0],
            base_logits: grim_backend_cpu::cpu_tensor(logits, Shape::new(vec![3, 8])),
            confidence: Vec::new(),
        };
        let scores = head.score(&block);
        assert_eq!(scores.len(), 3);
        for s in scores {
            assert!((0.0..=1.0).contains(&s));
        }
    }
}
