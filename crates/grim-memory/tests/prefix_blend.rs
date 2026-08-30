use grim_memory::{KvBlockPool, BLOCK_SIZE};

#[test]
fn test_blend_reuses_partial_block() {
    let mut pool = KvBlockPool::new(16, 4, 64);
    let tokens: Vec<u32> = (0..32).collect();
    let (blocks, _) = pool.find_or_share_prefix_tokens(&tokens).unwrap();
    assert_eq!(blocks.len(), 2);

    // Diverge at token 22 (mid-block 1).
    let (matched, matched_tokens, blended) = pool.match_prefix_blending(&tokens[..22]);
    assert_eq!(matched_tokens, BLOCK_SIZE); // block 0 fully reused (16 tokens)
    assert!(blended, "partial block match should enable blending");
    assert_eq!(matched.len(), 1);
}

#[test]
fn test_blend_no_match_returns_not_blended() {
    let pool = KvBlockPool::new(16, 4, 64);
    let tokens: Vec<u32> = (100..120).collect();
    let (matched, matched_tokens, blended) = pool.match_prefix_blending(&tokens);
    assert_eq!(matched.len(), 0);
    assert_eq!(matched_tokens, 0);
    assert!(!blended);
}
