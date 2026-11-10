//! CPU-only tests for astrum P10 conversation checkpoint cache.
use grim_memory::conversation_checkpoint::ConversationCheckpointCache;

#[test]
fn save_restore_prefix_and_byte_equal() {
    let mut c = ConversationCheckpointCache::new(4);
    let key = vec![1u32, 2, 3, 4];
    let state = vec![0xAAu8, 0xBB, 0xCC, 0xDD];
    c.save(key.clone(), state.clone(), vec![10, 11]);

    // mutate tokens beyond boundary (extend) -> prefix still matches
    let mut tokens = key.clone();
    tokens.extend_from_slice(&[5, 6, 7]);
    let r = c.restore_if_prefix_matches(&tokens).expect("prefix hit");
    assert_eq!(r.key, key);
    assert_eq!(r.state_bytes, state, "recurrent state must be byte-equal");
    assert_eq!(r.kv_block_ids, vec![10, 11]);
}

#[test]
fn prefix_mismatch_returns_none() {
    let mut c = ConversationCheckpointCache::new(4);
    c.save(vec![1, 2, 3], vec![9, 9], vec![]);
    assert!(c.restore_if_prefix_matches(&[1, 2, 4, 5]).is_none());
    assert!(c.restore_if_prefix_matches(&[9, 9, 9]).is_none());
}

#[test]
fn cap_evicts_oldest_lru() {
    let mut c = ConversationCheckpointCache::new(4);
    for i in 0u32..5 {
        c.save(vec![i, 100], vec![i as u8], vec![i as usize]);
    }
    assert_eq!(c.len(), 4);
    // oldest (i=0) evicted
    assert!(c.restore_if_prefix_matches(&[0, 100, 9]).is_none());
    // newest still present
    assert!(c.restore_if_prefix_matches(&[4, 100, 9]).is_some());
}

#[test]
fn restore_marks_mru_and_longest_prefix_wins() {
    let mut c = ConversationCheckpointCache::new(4);
    c.save(vec![1, 2], vec![1], vec![]);
    c.save(vec![1, 2, 3, 4], vec![2], vec![]);
    // tokens match both prefixes; longest wins
    let r = c.restore_if_prefix_matches(&[1, 2, 3, 4, 9]).unwrap();
    assert_eq!(r.key, vec![1, 2, 3, 4]);
    assert_eq!(r.state_bytes, vec![2]);
}

#[test]
fn exact_key_resave_refreshes_without_growing() {
    let mut c = ConversationCheckpointCache::new(4);
    c.save(vec![7, 8], vec![1], vec![]);
    c.save(vec![7, 8], vec![2], vec![42]);
    assert_eq!(c.len(), 1);
    let r = c.restore_if_prefix_matches(&[7, 8]).unwrap();
    assert_eq!(r.state_bytes, vec![2]);
    assert_eq!(r.kv_block_ids, vec![42]);
}