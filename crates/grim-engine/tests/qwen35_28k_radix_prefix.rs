//! Phase D: Engine-level `28k + 128` Radix Prefix Caching harness + mandatory
//! negative tests (partial-block clamping, fingerprint/dtype fail-closed,
//! pinned-anchor eviction protection).

use grim_memory::{
    hash_tokens_u64, DiskPrefixStore, PrefixCheckpoint, PrefixCheckpointMeta, PrefixKvDtype,
    RadixTree, BLOCK_SIZE,
};

/// Negative Gate 1: Partial-block / unaligned prefix match must clamp to the
/// last `BLOCK_SIZE`-aligned anchored boundary (never matching partial blocks).
#[test]
fn negative_partial_block_clamps_to_block_boundary() {
    let mut tree = RadixTree::new(BLOCK_SIZE);
    // Insert 2 full blocks (32 tokens) anchored with recurrent state 42 on block 1.
    let prefix: Vec<u32> = (100..132).collect();
    tree.insert(&prefix, &[10, 11]);
    tree.attach_recurrent_state(11, 42);

    // Query with 25 matching tokens (1 full block = 16 + 9 partial) + mismatch
    let mut query: Vec<u32> = (100..125).collect();
    query.push(9999);
    let (blocks, matched, anchor) = tree.match_prefix_with_anchor(&query);
    assert_eq!(
        matched % BLOCK_SIZE,
        0,
        "matched tokens must be BLOCK_SIZE-aligned, got {matched}"
    );
    assert_eq!(blocks.len(), 1);
    assert_eq!(matched, BLOCK_SIZE);
    assert_eq!(anchor, None, "unanchored block 0 must not fabricate an anchor");

    // Query matching all 32 tokens + 5-token suffix -> hits anchor 42 at 32
    let mut query2 = prefix.clone();
    query2.extend_from_slice(&[500, 501, 502, 503, 504]);
    let (blocks2, matched2, anchor2) = tree.match_prefix_with_anchor(&query2);
    assert_eq!(blocks2, vec![10, 11]);
    assert_eq!(matched2, 32);
    assert_eq!(anchor2, Some(42));
}

/// Negative Gate 2: Model-fingerprint mismatch and KV-dtype mismatch must fail
/// closed (`Ok(None)`) without corrupting or consuming the stored checkpoint.
#[test]
fn negative_model_fingerprint_and_dtype_mismatch_fail_closed() {
    let dir = std::env::temp_dir().join(format!(
        "grim_phase_d_neg_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let store = DiskPrefixStore::new(&dir, 1024 * 1024).unwrap();
    let th = hash_tokens_u64(&[10, 20, 30, 40]);
    let ckpt = PrefixCheckpoint {
        meta: PrefixCheckpointMeta {
            model_fingerprint: 0xDEAD_BEEF_CAFE_0001,
            kv_dtype: PrefixKvDtype::Q8_1,
            prefix_len: 28_672,
            num_attn_layers: 16,
            num_gdn_layers: 48,
        },
        kv_bytes: vec![7u8; 512],
        recurrent_bytes: vec![3u8; 256],
    };
    store.save(th, &ckpt).unwrap();

    // Wrong model fingerprint -> None
    assert!(store
        .load(0xDEAD_BEEF_CAFE_0002, PrefixKvDtype::Q8_1, th)
        .unwrap()
        .is_none());
    // Wrong KV dtype (F16 vs Q8_1) -> None
    assert!(store
        .load(0xDEAD_BEEF_CAFE_0001, PrefixKvDtype::F16, th)
        .unwrap()
        .is_none());
    // Exact match -> Some
    let ok = store
        .load(0xDEAD_BEEF_CAFE_0001, PrefixKvDtype::Q8_1, th)
        .unwrap();
    assert_eq!(ok.as_ref(), Some(&ckpt));

    let _ = std::fs::remove_dir_all(&dir);
}

/// Negative Gate 3: Eviction under active use must NEVER evict a pinned anchor.
#[test]
fn negative_eviction_under_active_pin_preserves_anchor() {
    let mut tree = RadixTree::new(BLOCK_SIZE);
    let p1: Vec<u32> = (1..=16).collect();
    let p2: Vec<u32> = (101..=116).collect();
    tree.insert(&p1, &[1]);
    tree.attach_recurrent_state(1, 77);
    tree.insert(&p2, &[2]);
    tree.attach_recurrent_state(2, 88);

    // Release active request refcounts (both blocks now ref_count == 0);
    // pin block 1 for 60s while block 2 remains unpinned.
    tree.remove(&[1]);
    tree.remove(&[2]);
    tree.pin_blocks(&[1], 60);
    assert!(tree.is_pinned(1));
    assert!(!tree.is_pinned(2));

    let evicted = tree.evict_coldest_leaf();
    assert_eq!(evicted, Some(2), "eviction must pick unpinned block 2, never pinned block 1");
    assert!(tree.contains_block(1), "pinned anchor block 1 must survive eviction");
    // Second eviction with only pinned block 1 left must return None.
    assert_eq!(tree.evict_coldest_leaf(), None);
}
