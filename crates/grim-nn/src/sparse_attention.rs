//! WI-P1 — sparse-attention selection core (GLM-DSA / DeepSeek-V3.2
//! lightning-indexer shape).
//!
//! The DeepSeek-V3.2-Exp checkpoint header (verified 2026-08-19) carries the
//! sparse-attention fields `index_head_dim` (128), `index_n_heads` (64) and
//! `index_topk` (2048): a per-head learned indexer key scores history
//! positions and the top-k are selected per query. This module implements the
//! deterministic selection core — score-history-then-top-k — as a reusable
//! primitive, the same way `moe::MoeRouter` is the reusable selection
//! primitive for expert routing.
//!
//! Scope boundary (per the WI-P1 plan's no-guessing rule): the *trained*
//! indexer weight tensors and their GGUF/safetensors names must be verified
//! against a real checkpoint before a model loader wires them in; this module
//! only provides the structural selection with an injectable scorer, plus the
//! config fields a checkpoint header defines. Device parity of an actual
//! sparse-attention mask inside Llama's attention is the checkpoint-gated
//! follow-up (UNVERIFIED until a real GLM-DSA/DeepSeek-V3.2 checkpoint with
//! indexer tensors is loaded).

use grim_tensor::error::{Error, Result};

/// Sparse-attention selector configuration — field names and semantics match
/// the DeepSeek-V3.2-Exp checkpoint header keys (`index_head_dim`,
/// `index_n_heads`, `index_topk`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseAttentionConfig {
    /// Dimensionality of each head's indexer key (DeepSeek-V3.2: 128).
    pub index_head_dim: usize,
    /// Number of indexer heads (DeepSeek-V3.2: 64; GQA-style mapping to the
    /// attention heads is left to the caller).
    pub index_n_heads: usize,
    /// Number of history positions selected per query (DeepSeek-V3.2: 2048).
    pub index_topk: usize,
}

/// Supplies the per-position indexer score for one indexer head. A real model
/// would back this with the trained lightning-indexer weights; tests inject a
/// hand-computed scorer to pin the selection math.
pub trait IndexerScorer {
    /// Historical length available for selection (<= current causal position).
    fn history_len(&self) -> usize;
    /// Score of history `position` under indexer `head`.
    fn score_history(&self, head: usize, position: usize) -> f32;
}

/// Deterministic top-k history selection (the "lightning indexer" selection
/// core). Selection is per head, ranked by score descending, ties broken by
/// position ascending (lower position wins) — deterministic for a fixed scorer.
#[derive(Debug, Clone)]
pub struct SparseAttentionSelector {
    cfg: SparseAttentionConfig,
}

impl SparseAttentionSelector {
    pub fn new(cfg: SparseAttentionConfig) -> Result<Self> {
        if cfg.index_head_dim == 0 {
            return Err(Error::Backend(
                "sparse-attention index_head_dim must be > 0".into(),
            ));
        }
        if cfg.index_n_heads == 0 {
            return Err(Error::Backend(
                "sparse-attention index_n_heads must be > 0".into(),
            ));
        }
        if cfg.index_topk == 0 {
            return Err(Error::Backend(
                "sparse-attention index_topk must be > 0".into(),
            ));
        }
        Ok(Self { cfg })
    }

    pub fn config(&self) -> &SparseAttentionConfig {
        &self.cfg
    }

    /// Select the top-k history positions for `head` given `scorer`. Returns
    /// positions ranked by score desc; when `top_k` exceeds history length the
    /// result clamps to the available history. `history_len` is capped at
    /// `scorer.history_len()` (callers handle the causal window).
    pub fn select(&self, head: usize, scorer: &dyn IndexerScorer) -> Result<Vec<usize>> {
        if head >= self.cfg.index_n_heads {
            return Err(Error::Backend(format!(
                "sparse-attention head {head} out of range (index_n_heads={})",
                self.cfg.index_n_heads
            )));
        }
        let len = scorer.history_len();
        let mut scored: Vec<(f32, usize)> = (0..len)
            .map(|pos| (scorer.score_history(head, pos), pos))
            .collect();
        // Rank by score desc; ties by position asc (normalize the sign).
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        Ok(scored
            .into_iter()
            .take(self.cfg.index_topk.min(len))
            .map(|(_, pos)| pos)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstScorer {
        len: usize,
        base: f32,
    }
    impl IndexerScorer for ConstScorer {
        fn history_len(&self) -> usize {
            self.len
        }
        fn score_history(&self, _head: usize, position: usize) -> f32 {
            self.base + position as f32
        }
    }

    #[test]
    fn select_ranks_by_score_desc() {
        let sel = SparseAttentionSelector::new(SparseAttentionConfig {
            index_head_dim: 8,
            index_n_heads: 2,
            index_topk: 3,
        })
        .unwrap();
        let scores = ConstScorer { len: 4, base: 0.0 }; // scores 0,1,2,3
        assert_eq!(sel.select(1, &scores).unwrap(), vec![3, 2, 1]);
    }

    #[test]
    fn head_out_of_range_errors() {
        let sel = SparseAttentionSelector::new(SparseAttentionConfig {
            index_head_dim: 8,
            index_n_heads: 2,
            index_topk: 3,
        })
        .unwrap();
        assert!(sel.select(2, &ConstScorer { len: 3, base: 0.0 }).is_err());
    }
}
