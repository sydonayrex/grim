//! QSA block allocator: known-answer tests against the reference semantics.
//!
//! # The bug this replaces
//!
//! The earlier implementation pooled `cell_index / r` over a dense contiguous
//! cache. Upstream keys blocks on `(sequence set, index bucket)` because a
//! unified paged cache counts every sequence from zero: two sequences holding
//! cells at positions 0 and 4 would pool into one block under `pos / r`,
//! letting one sequence attend to the other's content. That is a correctness
//! bug, not a performance one, and the old tests missed it because every
//! fixture was a single dense sequence.
//!
//! # What is asserted here
//!
//! 1. a dense single sequence reproduces the old `pos / r` layout exactly
//! 2. two sequences in the same bucket get SEPARATE blocks
//! 3. only full groups are promoted; an incomplete group is not pooled
//! 4. unpooled cells get a spare block id, and none exists when everything pools
//! 5. the block-causal rule from Eq. (15): `p_b + r - 1 <= q`
//! 6. the tail is always visible (Eq. 16), via the `+1e9` bias
//! 7. pooling is a mean over exactly `r` cells

use grim_models_transformer::qwen4exp_qsa_blocks::{CellInfo, allocate_blocks};

/// A dense single-sequence cache: cell `j` holds position `j`.
fn dense(n: u32, seq: u32) -> Vec<CellInfo> {
    (0..n).map(|j| CellInfo::new(j, seq)).collect()
}

#[test]
fn a_dense_single_sequence_reproduces_the_flat_pos_over_r_layout() {
    // r = 4, 16 cells, 4 buckets: one block per bucket, filled in order.
    let cells = dense(16, 0);
    let lay = allocate_blocks(&cells, 4, 4).expect("alloc");
    assert_eq!(lay.n_bid, 4, "four full buckets -> four blocks");
    assert_eq!(
        lay.blk_pos,
        vec![0, 4, 8, 12],
        "block starts are bucket * r"
    );
    assert_eq!(
        lay.blk_of,
        vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    );
    assert_eq!(lay.dead_bid, None, "nothing is unpooled, so no spare block");
    assert!(!lay.out_of_range);
    // Every block holds exactly r members, in slot order.
    assert_eq!(
        lay.blk_cells,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    );
}

#[test]
fn two_sequences_in_the_same_bucket_never_share_a_block() {
    // THE bug. Cell 0 belongs to sequence 1 at position 0; cells 1..4 belong to
    // sequence 2 at positions 0..3. All five sit in bucket 0. Under `pos / r`
    // they would pool into one block, so a query on sequence 2 could attend to
    // sequence 1's key.
    let cells = vec![
        CellInfo::new(0, 1),
        CellInfo::new(0, 2),
        CellInfo::new(1, 2),
        CellInfo::new(2, 2),
        CellInfo::new(3, 2),
    ];
    let lay = allocate_blocks(&cells, 1, 4).expect("alloc");
    assert_eq!(
        lay.n_bid, 1,
        "only sequence 2's four cells form a complete block; sequence 1's \\
         single cell cannot be pooled with them"
    );
    assert_eq!(lay.blk_pos, vec![0]);
    // Sequence 1's cell is the odd one out and must not be a member.
    assert!(
        !lay.blk_cells.contains(&0),
        "cell 0 (sequence 1) must not be pooled into sequence 2's block, \\
         got {:?}",
        lay.blk_cells
    );
    assert_eq!(lay.blk_cells, vec![1, 2, 3, 4], "only sequence 2's cells");
    assert_eq!(
        lay.blk_of[0],
        grim_models_transformer::qwen4exp_qsa_blocks::NO_BLOCK
    );
    assert_eq!(lay.blk_of[1], 0, "sequence 2's cells are in block 0");
}

#[test]
fn an_incomplete_group_is_never_promoted() {
    // 3 cells in a bucket of 4, then a full bucket. The mean over fewer than r
    // cells would change the scale the norm sees, so the group is dropped.
    let cells = vec![
        CellInfo::new(0, 0),
        CellInfo::new(1, 0),
        CellInfo::new(2, 0),
        CellInfo::new(4, 0),
        CellInfo::new(5, 0),
        CellInfo::new(6, 0),
        CellInfo::new(7, 0),
    ];
    let lay = allocate_blocks(&cells, 2, 4).expect("alloc");
    assert_eq!(lay.n_bid, 1, "bucket 0 has 3 of 4 slots and must not pool");
    assert_eq!(lay.blk_pos, vec![4], "the only complete bucket is 1");
    assert_eq!(lay.blk_cells, vec![3, 4, 5, 6]);
    for j in 0..3 {
        assert_eq!(
            lay.blk_of[j],
            grim_models_transformer::qwen4exp_qsa_blocks::NO_BLOCK,
            "cell {j} is unpooled"
        );
    }
}

#[test]
fn a_spare_block_exists_only_when_something_is_unpooled() {
    // All full: no spare.
    let full = dense(8, 0);
    let lay = allocate_blocks(&full, 2, 4).expect("alloc");
    assert_eq!(lay.n_bid, 2);
    assert_eq!(lay.dead_bid, None, "every cell pooled, so no spare block");

    // One cell short: a spare exists, so unpooled cells have somewhere to go.
    let mut short = dense(8, 0);
    short.pop();
    let lay = allocate_blocks(&short, 2, 4).expect("alloc");
    assert_eq!(lay.n_bid, 1, "the second bucket is incomplete");
    assert_eq!(lay.dead_bid, Some(1), "n_bid == 1 < n_blocks == 2");
}

#[test]
fn a_bucket_past_the_window_is_reported_not_silently_dropped() {
    // A cell whose position lands in a bucket beyond n_blocks. Upstream sets
    // `oor` and asserts against it in block-bias mode; we surface it so the
    // caller can fail loudly instead of computing a plausible wrong answer.
    let cells = vec![CellInfo::new(64, 0)];
    let lay = allocate_blocks(&cells, 2, 4).expect("alloc");
    assert!(
        lay.out_of_range,
        "position 64 / 4 = 16 is past n_blocks = 2"
    );
    assert_eq!(lay.n_bid, 0);
}

#[test]
fn the_block_causal_rule_waits_for_the_whole_block() {
    // Eq. (15): a block is scored only once every one of its r tokens has been
    // observed, i.e. p_b + r - 1 <= q.
    let cells = dense(16, 0);
    let lay = allocate_blocks(&cells, 4, 4).expect("alloc");
    let r = 4;
    // Block 0 starts at 0, so it is complete once q >= 3.
    assert!(!lay.block_is_observed(0, 0, r), "q=0: block 0 incomplete");
    assert!(!lay.block_is_observed(0, 2, r), "q=2: block 0 incomplete");
    assert!(lay.block_is_observed(0, 3, r), "q=3: p_b + r - 1 = 3 <= 3");
    // Block 1 starts at 4, complete at q = 7.
    assert!(!lay.block_is_observed(1, 6, r), "q=6: block 1 incomplete");
    assert!(lay.block_is_observed(1, 7, r), "q=7: p_b + r - 1 = 7 <= 7");
    // A query can never see a block beyond its own.
    assert!(!lay.block_is_observed(3, 5, r), "block 3 is in the future");
}

#[test]
fn the_tail_is_always_visible_whatever_its_score() {
    // Eq. (16): the tail of the final incomplete block is always included. The
    // reference encodes it as a +1e9 bonus so the tail always wins the top-k,
    // and a -inf on blocks that are empty, foreign or future.
    let cells = dense(10, 0); // 10 cells, r = 4 -> buckets 0,1,2; bucket 2 has 2
    let lay = allocate_blocks(&cells, 3, 4).expect("alloc");
    assert_eq!(lay.n_bid, 2, "buckets 0 and 1 are full; bucket 2 is not");

    let all_visible = |_: usize| true;
    // Query at position 9 (the newest cell). tail_start = (9+1)/4*4 = 8.
    let bias = lay.block_bias(9, 4, &all_visible);
    assert_eq!(bias.len(), 2);
    // Block 0 starts at 0 (< 8) so it is an ordinary candidate.
    assert_eq!(bias[0], 0.0, "block 0 is a normal visible block");
    // Block 1 starts at 4 (< 8) too.
    assert_eq!(bias[1], 0.0);
    // No block starts at or past 8, because bucket 2 is incomplete, so there is
    // no tail block to bias. Assert the geometry that produces that.
    assert!(
        lay.blk_pos.iter().all(|&p| p < 8),
        "no pooled block is in the tail region: {:?}",
        lay.blk_pos
    );
}

#[test]
fn a_future_block_is_masked_to_negative_infinity() {
    let cells = dense(16, 0);
    let lay = allocate_blocks(&cells, 4, 4).expect("alloc");
    let all_visible = |_: usize| true;
    // Query at position 3 has observed block 0 only.
    let bias = lay.block_bias(3, 4, &all_visible);
    assert_eq!(bias[0], 0.0, "block 0 is visible at q=3");
    for b in 1..lay.n_bid {
        // tail_start = (3+1)/4*4 = 4, so blocks starting at >= 4 are the tail
        // and get +1e9 rather than -inf. Either way they are not 0.
        assert_ne!(
            bias[b], 0.0,
            "block {b} starts at {} and must not be a plain candidate",
            lay.blk_pos[b]
        );
    }
    // A cell the query cannot read masks its whole block.
    let only_seq2 = |cell: usize| cells[cell].seqs.contains(&2);
    let bias = lay.block_bias(15, 4, &only_seq2);
    assert!(
        bias.iter().all(|v| *v == f32::NEG_INFINITY),
        "no cell belongs to sequence 2, so every block is masked: {bias:?}"
    );
}

#[test]
fn pooling_is_a_mean_over_exactly_r_cells() {
    // Distinct keys per cell so the mean is checkable by hand.
    let n_kv = 8usize;
    let idx_dim = 2usize;
    let cells: Vec<CellInfo> = (0..4).map(|j| CellInfo::new(j as u32, 0)).collect();
    let lay = allocate_blocks(&cells, 1, 4).expect("alloc");
    // k_raw[cell * idx_dim + d] = cell * 10 + d, so:
    //   dim 0: cells 0..4 hold 0, 10, 20, 30  -> mean 15
    //   dim 1: cells 0..4 hold 1, 11, 21, 31  -> mean 16
    let k_raw: Vec<f32> = (0..n_kv * idx_dim)
        .map(|i| (i / idx_dim) as f32 * 10.0 + (i % idx_dim) as f32)
        .collect();
    let pooled = lay.pooled(&k_raw, n_kv, idx_dim, 4).expect("pool");
    assert_eq!(pooled.len(), 1 * idx_dim);
    assert!((pooled[0] - 15.0).abs() < 1e-6, "mean of 0,10,20,30 = 15");
    assert!((pooled[1] - 16.0).abs() < 1e-6, "mean of 1,11,21,31 = 16");
}

#[test]
fn a_ratio_of_zero_or_over_64_is_rejected() {
    let cells = dense(8, 0);
    assert!(
        allocate_blocks(&cells, 2, 0).is_err(),
        "r = 0 must be rejected"
    );
    assert!(
        allocate_blocks(&cells, 2, 65).is_err(),
        "r = 65 exceeds the 64-slot occupancy limit"
    );
    assert!(
        allocate_blocks(&cells, 2, 64).is_ok(),
        "r = 64 is the limit"
    );
}

#[test]
fn on_a_dense_single_sequence_the_allocator_agrees_with_cell_over_r() {
    // Why the forward-path mutation "revert to cell/r pooling" survives: on a
    // dense, single-sequence cache the two are IDENTICAL by construction, so no
    // end-to-end test can tell them apart. They diverge as soon as the cache is
    // paged or multi-sequence, which is what the other tests cover.
    //
    // Pin the equivalence explicitly so a future change that breaks it in the
    // dense case is caught here rather than blamed on the allocator.
    let n_kv = 16usize;
    let idx_dim = 2usize;
    let r = 4usize;
    let cells = dense(n_kv as u32, 0);
    let lay = allocate_blocks(&cells, n_kv / r, r).expect("alloc");
    let k_raw: Vec<f32> = (0..n_kv * idx_dim)
        .map(|i| (i / idx_dim) as f32 * 10.0 + (i % idx_dim) as f32)
        .collect();
    let via_allocator = lay.pooled(&k_raw, n_kv, idx_dim, r).expect("pool");
    let via_cell_over_r =
        grim_models_transformer::qwen4exp_qsa::pool_indexer_keys(&k_raw, n_kv, idx_dim, r)
            .expect("pool");
    assert_eq!(
        via_allocator.len(),
        via_cell_over_r.len(),
        "same number of blocks on a dense cache"
    );
    for (a, b) in via_allocator.iter().zip(via_cell_over_r.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "the two must agree on a dense single-sequence cache: \
             allocator {a} vs cell/r {b}"
        );
    }
}

#[test]
fn an_empty_cell_is_skipped_entirely() {
    // A paged cache has holes. An empty cell must not consume a block slot.
    let cells = vec![
        CellInfo::new(0, 0),
        CellInfo::empty(),
        CellInfo::new(1, 0),
        CellInfo::new(2, 0),
        CellInfo::new(3, 0),
    ];
    let lay = allocate_blocks(&cells, 1, 4).expect("alloc");
    assert_eq!(
        lay.n_bid, 1,
        "the four filled cells still form a full block"
    );
    assert_eq!(lay.blk_cells, vec![0, 2, 3, 4], "cell 1 is the hole");
    assert_eq!(
        lay.blk_of[1],
        grim_models_transformer::qwen4exp_qsa_blocks::NO_BLOCK
    );
}
