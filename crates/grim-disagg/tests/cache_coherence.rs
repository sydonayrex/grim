use grim_disagg::coherence::{InvalidationMsg, CacheCoherenceManager};

#[test]
fn test_invalidation_msg_roundtrip() {
    let msg = InvalidationMsg {
        prefix_hash: 0xDEAD_BEEF_CAFE_BABE,
        origin_node: 42,
        timestamp: 1725000000,
    };

    let encoded = msg.encode();
    assert_eq!(encoded.len(), 20);

    let decoded = InvalidationMsg::decode(&encoded).expect("decode should succeed");
    assert_eq!(decoded.prefix_hash, msg.prefix_hash);
    assert_eq!(decoded.origin_node, msg.origin_node);
    assert_eq!(decoded.timestamp, msg.timestamp);
}

#[test]
fn test_invalidation_msg_corrupted_length() {
    let bad_bytes = vec![0u8; 19]; // Invalid length (must be 20)
    assert!(InvalidationMsg::decode(&bad_bytes).is_err());
}

#[test]
fn test_cache_coherence_manager_invalidation_propagation() {
    let mut node0 = CacheCoherenceManager::new_standalone(0);
    let mut node1 = CacheCoherenceManager::new_standalone(1);

    // Both nodes cache prefix [10, 20, 30]
    let tokens = [10u32, 20, 30];
    node0.insert_prefix(&tokens, 100);
    node1.insert_prefix(&tokens, 200);

    assert_eq!(node0.lookup_prefix(&tokens), Some(100));
    assert_eq!(node1.lookup_prefix(&tokens), Some(200));

    // Node 0 invalidates prefix and creates invalidation message
    let msg = node0.invalidate_prefix(&tokens);

    // Node 1 receives message and applies invalidation
    node1.handle_invalidation(&msg);

    assert_eq!(node0.lookup_prefix(&tokens), None);
    assert_eq!(node1.lookup_prefix(&tokens), None);
}
