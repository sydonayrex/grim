//! WI-P1 — SparseAttentionSelector primitive tests (GLM-DSA / DeepSeek
//! lightning-indexer-shaped selection core).
//!
//! The selection shape is grounded in the real DeepSeek-V3.2-Exp checkpoint
//! header (fetched 2026-08-19): `index_head_dim: 128`, `index_n_heads: 64`,
//! `index_topk: 2048` — a per-head learned indexer key that scores history
//! positions and takes the top-k. This module implements the deterministic
//! selection core (score-history-then-top-k) with hand-computed references;
//! the trained indexer weight tensors and their GGUF names are the
//! checkpoint-gated follow-up (must be verified against a real checkpoint
//! before wiring, per the plan's no-guessing rule).

use grim_nn::sparse_attention::{IndexerScorer, SparseAttentionSelector, SparseAttentionConfig};

/// Hand-computed dot-product scorer: score = q·k for one head.
struct DotScorer {
    q: [f32; 4],
    keys: Vec<[f32; 4]>,
}

impl IndexerScorer for DotScorer {
    fn score_history(&self, _head: usize, position: usize) -> f32 {
        let k = self.keys[position];
        self.q[0] * k[0] + self.q[1] * k[1] + self.q[2] * k[2] + self.q[3] * k[3]
    }
    fn history_len(&self) -> usize {
        self.keys.len()
    }
}

#[test]
fn selector_topk_returns_highest_scoring_positions_in_order() {
    // q = [1,0,0,0]; keys have distinct first-coordinates:
    let scorer = DotScorer {
        q: [1.0, 0.0, 0.0, 0.0],
        keys: vec![
            [3.0, 0.0, 0.0, 0.0], // position 0 → score 3.0
            [1.0, 0.0, 0.0, 0.0], // position 1 → score 1.0
            [5.0, 0.0, 0.0, 0.0], // position 2 → score 5.0
            [2.0, 0.0, 0.0, 0.0], // position 3 → score 2.0
        ],
    };
    let cfg = SparseAttentionConfig {
        index_head_dim: 4,
        index_n_heads: 1,
        index_topk: 2,
    };
    let sel = SparseAttentionSelector::new(cfg).expect("valid config");
    let top = sel.select(0, &scorer).expect("select");
    // top-k = {2 (5.0), 0 (3.0)} sorted by score desc, then position asc.
    assert_eq!(top, vec![2, 0], "top-k must rank by score");
}

#[test]
fn selector_clamps_topk_to_history_len() {
    let scorer = DotScorer {
        q: [1.0, 0.0, 0.0, 0.0],
        keys: vec![[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
    };
    let cfg = SparseAttentionConfig {
        index_head_dim: 4,
        index_n_heads: 1,
        index_topk: 10, // > history
    };
    let sel = SparseAttentionSelector::new(cfg).expect("valid config");
    let top = sel.select(0, &scorer).expect("select");
    assert_eq!(top, vec![1, 0], "top-k clamps to available history, ranked by score");
}

#[test]
fn selector_rejects_bad_config() {
    assert!(SparseAttentionSelector::new(SparseAttentionConfig {
        index_head_dim: 0,
        index_n_heads: 2,
        index_topk: 8,
    })
    .is_err());
    assert!(SparseAttentionSelector::new(SparseAttentionConfig {
        index_head_dim: 128,
        index_n_heads: 0,
        index_topk: 8,
    })
    .is_err());
    assert!(SparseAttentionSelector::new(SparseAttentionConfig {
        index_head_dim: 128,
        index_n_heads: 2,
        index_topk: 0,
    })
    .is_err());
}

#[test]
fn selector_empty_history_yields_empty_selection() {
    let scorer = DotScorer {
        q: [1.0, 0.0, 0.0, 0.0],
        keys: vec![],
    };
    let cfg = SparseAttentionConfig {
        index_head_dim: 4,
        index_n_heads: 1,
        index_topk: 8,
    };
    let sel = SparseAttentionSelector::new(cfg).expect("valid config");
    assert!(sel.select(0, &scorer).expect("select").is_empty());
}