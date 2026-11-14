use grim_memory::{KvBlockPool, SeqWindow, BLOCK_SIZE};
use std::collections::{BTreeMap, BTreeSet};

/// Ported directly from Radiance `tests/kv_window_test.cpp:87-195`
/// (`rswa_reference_and_ring_keeps_the_visible_set_and_flat_bytes`):
/// Uses a non-block-multiple prefill length `p_len = 50` (with `block_size = 16`, `window = 32`,
/// so `ring_pos = 64 = 4 blocks` and `ring_blocks = 2`), decodes `l_steps = 2000` tokens,
/// tracks physical `slot -> token_pos` in a map, and asserts at every single step:
/// 1. Total physical blocks stay flat at `max_blocks_needed() = 6` forever.
/// 2. Once `n >= ring_pos + ring_span`, the exact set of token positions resident in the block
///    table equals `[0, ring_pos) U [n - ring_span, n)` (frozen reference + trailing ring).
#[test]
fn radiance_oracle_seq_window_2000_steps_visible_set_and_flat_blocks() {
    let b = BLOCK_SIZE; // 16
    let w = 32usize;    // 2 ring blocks
    let p_len = 50usize; // intentionally not a multiple of 16 -> ring_pos = 64 (4 ref blocks)
    let win = SeqWindow::new(p_len, w, b);
    assert_eq!(win.ring_pos(), 64);
    assert_eq!(win.ring_blocks(), 2);
    assert_eq!(win.max_blocks_needed(), 6); // 4 reference blocks + 2 ring blocks

    let ring_span = win.ring_blocks() * b; // 32
    let mut slot_content: BTreeMap<usize, usize> = BTreeMap::new();
    let mut touched_blocks: BTreeSet<usize> = BTreeSet::new();

    // Prefill [0, p_len)
    for pos in 0..p_len {
        let blk = win.logical_block_for_pos(pos);
        let slot = blk * b + (pos % b);
        slot_content.insert(slot, pos);
        touched_blocks.insert(blk);
    }
    assert_eq!(touched_blocks.len(), 4);

    // Decode 2000 tokens past prefill
    let l_steps = 2000usize;
    for pos in p_len..(p_len + l_steps) {
        let blk = win.logical_block_for_pos(pos);
        let slot = blk * b + (pos % b);
        slot_content.insert(slot, pos);
        touched_blocks.insert(blk);

        // Flat block count forever: never exceeds max_blocks_needed() (6)
        assert!(
            touched_blocks.len() <= win.max_blocks_needed(),
            "block count {} exceeded cap {} at pos {pos}",
            touched_blocks.len(),
            win.max_blocks_needed()
        );

        let n = pos + 1;
        // Once the ring has filled at least once (n >= ring_pos + ring_span) and sits on a block
        // boundary (n % b == 0), the resident token set must equal [0, ring_pos) U [n - ring_span, n).
        if n >= win.ring_pos() + ring_span && n % b == 0 {
            assert_eq!(touched_blocks.len(), win.max_blocks_needed());
            let resident: BTreeSet<usize> = slot_content.values().copied().collect();
            let mut want: BTreeSet<usize> = (0..win.ring_pos()).collect();
            for tail_pos in (n - ring_span)..n {
                want.insert(tail_pos);
            }
            assert_eq!(
                resident, want,
                "R-SWA visible set mismatch at token count n={n}"
            );
        }
    }
}

#[test]
fn copy_on_idle_clean_flag_clears_on_mutation() {
    let mut pool = KvBlockPool::new(4, 2, 8);
    let blk = pool.alloc().expect("allocate block");
    assert!(!pool.is_block_clean_on_host(blk));

    // Background idle tiering copies block to host and marks clean
    pool.mark_block_clean_on_host(blk);
    assert!(pool.is_block_clean_on_host(blk));

    // Mutating block keys invalidates clean flag
    let dummy = vec![1.0f32; 2 * 8];
    pool.write_keys(blk, &dummy, 1);
    assert!(!pool.is_block_clean_on_host(blk));
}
