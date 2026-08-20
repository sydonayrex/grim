//! Packed-step training driver (salamander.md P2 wire-in).
//!
//! The varlen collator primitives (`grim_autograd::collate`) exist but were
//! never wired into a training loop. This module is the driver: it groups
//! variable-length samples under a token budget and runs one optimizer step
//! per **packed group** instead of per sample.
//!
//! Leakage-free by construction: each segment gets a fresh single-sequence
//! causal forward (the existing streaming path), so no block-diagonal
//! attention mask tensor and no `qkv_attention` trait change are needed.
//! Gradients accumulate across segments of a group exactly as gradient
//! accumulation already does; the only change is *when* the optimizer steps
//! and how the reported loss is weighted (by token count, not by sample).

use grim_autograd::collate::TokenSequence;
use grim_tensor::error::{Error, Result};

/// Configuration for one packed step.
#[derive(Debug, Clone, Copy)]
pub struct PackedStepConfig {
    /// Maximum total tokens per packed group (sum of segment lengths).
    pub max_tokens_per_group: usize,
    /// Maximum number of segments per packed group.
    pub max_seqs_per_group: usize,
}

impl Default for PackedStepConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_group: 8192,
            max_seqs_per_group: 16,
        }
    }
}

/// Statistics reported by [`packed_step`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PackedStepStats {
    /// Number of packed groups executed (= optimizer steps taken by caller).
    pub num_groups: usize,
    /// Number of sequences processed across all groups.
    pub sequences_packed: usize,
    /// Total tokens processed (sum of segment lengths).
    pub tokens_processed: usize,
    /// Token-weighted mean loss across all segments.
    pub mean_loss: f32,
}

/// Greedy first-fit grouping of sequences under a token budget.
///
/// Unlike `VarLenCollator::collate_1d_packed` (which stops at the first
/// sequence that overflows the budget), this packs the full input: sequences
/// that do not fit the current group start a new one. Sequences longer than
/// `max_tokens_per_group` still get their own group so nothing is dropped.
///
/// Returns groups of indices into `sequences`.
pub fn group_sequences(
    sequences: &[TokenSequence],
    cfg: &PackedStepConfig,
) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::with_capacity(cfg.max_seqs_per_group);
    let mut current_tokens = 0usize;

    for (i, seq) in sequences.iter().enumerate() {
        let len = seq.len().max(1);
        let fits_tokens = current_tokens + len <= cfg.max_tokens_per_group || current.is_empty();
        let fits_seqs = current.len() < cfg.max_seqs_per_group;
        if fits_tokens && fits_seqs {
            current.push(i);
            current_tokens += len;
        } else {
            groups.push(std::mem::take(&mut current));
            current.push(i);
            current_tokens = len;
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Run one packed step over `sequences`.
///
/// `forward_backward` is invoked once per **segment** (single sequence) and
/// must run the caller's forward + backward (gradients accumulate in the
/// caller's `TrainableParams`) and return `(loss, tokens)` for that segment.
/// The caller performs the optimizer step once per returned group — use
/// [`PackedStepStats::num_groups`] — so N short samples cost one step instead
/// of N.
pub fn packed_step<F>(sequences: &[TokenSequence], cfg: &PackedStepConfig, mut forward_backward: F) -> Result<PackedStepStats>
where
    F: FnMut(&TokenSequence) -> Result<f32>,
{
    if sequences.is_empty() {
        return Err(Error::Backend("packed_step: no sequences".into()));
    }
    let groups = group_sequences(sequences, cfg);

    let mut stats = PackedStepStats {
        num_groups: groups.len(),
        ..Default::default()
    };
    let mut loss_sum = 0.0f32;
    let mut token_sum = 0usize;

    for group in &groups {
        for &idx in group {
            let seq = &sequences[idx];
            let loss = forward_backward(seq)?;
            let tokens = seq.len().max(1);
            loss_sum += loss * tokens as f32;
            token_sum += tokens;
            stats.sequences_packed += 1;
            stats.tokens_processed += tokens;
        }
    }
    stats.mean_loss = if token_sum > 0 {
        loss_sum / token_sum as f32
    } else {
        0.0
    };
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(len: usize) -> TokenSequence {
        TokenSequence::new((0..len as u32).collect())
    }

    #[test]
    fn groups_respect_token_budget() {
        let seqs: Vec<TokenSequence> = vec![seq(300), seq(300), seq(300), seq(10)];
        let cfg = PackedStepConfig {
            max_tokens_per_group: 700,
            max_seqs_per_group: 8,
        };
        let groups = group_sequences(&seqs, &cfg);
        // 300+300 fits (600 <= 700); +300 would overflow -> new group.
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], vec![0, 1]);
        assert_eq!(groups[1], vec![2, 3]);
        for g in &groups {
            let total: usize = g.iter().map(|&i| seqs[i].len()).sum();
            assert!(total <= 700 || g.len() == 1);
        }
    }

    #[test]
    fn groups_respect_seq_cap() {
        let seqs: Vec<TokenSequence> = (0..5).map(|_| seq(10)).collect();
        let cfg = PackedStepConfig {
            max_tokens_per_group: 1000,
            max_seqs_per_group: 2,
        };
        let groups = group_sequences(&seqs, &cfg);
        assert_eq!(groups.len(), 3);
        assert!(groups.iter().all(|g| g.len() <= 2));
    }

    #[test]
    fn oversized_sequence_gets_own_group_and_is_not_dropped() {
        let seqs = vec![seq(10), seq(5000), seq(10)];
        let cfg = PackedStepConfig {
            max_tokens_per_group: 100,
            max_seqs_per_group: 8,
        };
        let groups = group_sequences(&seqs, &cfg);
        assert_eq!(groups.len(), 3);
        let flat: Vec<usize> = groups.concat();
        assert_eq!(flat, vec![0, 1, 2], "no sequence may be dropped");
    }

    #[test]
    fn packed_step_weights_loss_by_tokens() {
        let seqs = vec![seq(100), seq(300)];
        let cfg = PackedStepConfig::default();
        let stats = packed_step(&seqs, &cfg, |s| Ok(s.len() as f32)).unwrap();
        assert_eq!(stats.num_groups, 1);
        assert_eq!(stats.sequences_packed, 2);
        assert_eq!(stats.tokens_processed, 400);
        // loss per segment == its token count, so the token-weighted mean is
        // (100*100 + 300*300) / 400 = 250.
        assert!((stats.mean_loss - 250.0).abs() < 1e-3);
    }

    #[test]
    fn packed_step_propagates_errors() {
        let seqs = vec![seq(4)];
        let cfg = PackedStepConfig::default();
        let res = packed_step(&seqs, &cfg, |_| {
            Err(Error::Backend("boom".into()))
        });
        assert!(res.is_err());
    }

    #[test]
    fn empty_input_errors() {
        let cfg = PackedStepConfig::default();
        assert!(packed_step(&[], &cfg, |_| Ok(0.0)).is_err());
    }
}
