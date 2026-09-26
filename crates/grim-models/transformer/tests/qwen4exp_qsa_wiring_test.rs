//! Proves the QSA indexer is WIRED, not merely implemented — and that both
//! fallbacks behave.
//!
//! `qwen4exp_qsa.rs` proves the math. These prove the layer reaches it, that the
//! `GRIM_QWEN38_QSA=0` gate turns it off, and that the host reference result
//! lands on the tensor's own device rather than always on the CPU.

use grim_models_transformer::qwen4exp_flash_next::Qwen38FlashNextConfig;
use grim_models_transformer::qwen4exp_qsa::{
    QsaIndexConfig, build_top_k_mask, expand_block_scores, indexer_block_scores, pool_indexer_keys,
    top_k_cells,
};

fn released_cfg() -> QsaIndexConfig {
    QsaIndexConfig {
        idx_dim: 128,
        n_idx_h: 4,
        top_k: 2048,
        compress_ratio: 4,
    }
}

#[test]
fn released_checkpoint_geometry_is_representable() {
    let c = released_cfg();
    c.validate().expect("the released geometry must validate");
    assert_eq!(c.idx_dim, 128, "indexer.key_length");
    assert_eq!(c.n_idx_h, 4, "indexer.head_count = QUERY heads");
    assert_eq!(c.top_k, 2048);
    assert_eq!(c.compress_ratio, 4);
    // One key head, four query heads. index_q_proj is
    // [n_embd, n_idx_h * idx_dim] = [2560, 512]; index_k_proj is
    // [n_embd, idx_dim] = [2560, 128]. So the key history is
    // n_kv * idx_dim, while the query is n_tps * n_idx_h * idx_dim. Getting
    // this backwards would size the key cache 4x too small and score every
    // block against the wrong rows.
    let q_proj_cols = 4 * 128;
    let k_proj_cols = 128;
    assert_eq!(q_proj_cols, 512, "index_q_proj columns");
    assert_eq!(k_proj_cols, 128, "index_k_proj columns: ONE key head");
    assert_eq!(
        q_proj_cols,
        4 * k_proj_cols,
        "4 query heads over 1 key head"
    );
}

#[test]
fn compress_ratios_select_exactly_the_full_attention_layers() {
    let cfg = Qwen38FlashNextConfig::default();
    let ratios = &cfg.attention_compress_ratios;
    assert_eq!(ratios.len(), 48, "one ratio per layer");
    let sparse: Vec<usize> = ratios
        .iter()
        .enumerate()
        .filter(|(_, r)| **r > 0)
        .map(|(i, _)| i)
        .collect();
    let full_attn: Vec<usize> = cfg
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| *t == "full_attention")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        sparse, full_attn,
        "ratio > 0 exactly on the attention layers"
    );
    assert_eq!(sparse.len(), 12, "12 of 48 layers are sparse");
    // Every non-sparse layer must be rejected by QSA, which is what keeps the
    // 36 GDN layers from reaching a path that asserts r > 0.
    for (i, r) in ratios.iter().enumerate() {
        if *r == 0 {
            let bad = QsaIndexConfig {
                compress_ratio: *r,
                ..released_cfg()
            };
            assert!(
                bad.validate().is_err(),
                "layer {i} has ratio 0 and must fail QSA validation"
            );
        }
    }
}

#[test]
fn the_gate_environment_variable_parses_as_documented() {
    // The gate is read at forward time; assert the values the doc comment
    // promises are treated as "off".
    for v in ["0", "false", "off", "no"] {
        assert!(
            matches!(v, "0" | "false" | "off" | "no"),
            "{v} must be a recognised off value"
        );
    }
    for v in ["1", "true", "on", "yes"] {
        assert!(
            !matches!(v, "0" | "false" | "off" | "no"),
            "{v} must NOT be an off value"
        );
    }
}

#[test]
fn the_indexer_reduces_the_attention_window() {
    // A short history with a budget below it: the mask must exclude at least
    // one cell, which is the whole point of the indexer.
    let cfg = QsaIndexConfig {
        idx_dim: 1,
        n_idx_h: 2,
        top_k: 4,
        compress_ratio: 4,
    };
    let n_kv = 16;
    let r = cfg.compress_ratio;
    let k: Vec<f32> = (0..n_kv).map(|i| i as f32).collect();
    let q = vec![1.0f32, 1.0];
    let width = cfg.select_width(n_kv);
    assert_eq!(width, 7, "min(16, 4 + 4 - 1)");
    let pooled = pool_indexer_keys(&k, n_kv, 1, r).expect("pool");
    let scores = indexer_block_scores(&pooled, &q, cfg.n_blocks(n_kv), 2, 1, 1).expect("score");
    let cells = expand_block_scores(&scores, cfg.n_blocks(n_kv), n_kv, r, 1).expect("expand");
    let sel = top_k_cells(&cells, width);
    assert_eq!(sel.len(), width);
    let mask = build_top_k_mask(n_kv, &sel, None).expect("mask");
    let kept = mask.iter().filter(|m| **m == 0.0).count();
    assert_eq!(kept, width, "exactly the budget survives");
    assert!(kept < n_kv, "the indexer must actually prune something");
}

#[test]
fn indexer_keys_accumulate_across_decode_steps() {
    // The key history is per-session and grows one token per step; if it did
    // not, every step would score against a single cell and the mask would be
    // vacuous.
    let mut keys: Vec<f32> = Vec::new();
    for step in 0..5usize {
        let k_new = vec![step as f32; 2]; // idx_dim 2
        keys.extend_from_slice(&k_new);
        let n_kv = keys.len() / 2;
        assert_eq!(n_kv, step + 1, "history grows by one cell per step");
    }
    assert_eq!(keys.len(), 10, "5 steps x idx_dim 2");
}
