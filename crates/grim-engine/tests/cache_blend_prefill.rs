//! Integration test for Task 10.7: CacheBlend prefill layout and blend_kv_rope_kernel verification.

use grim_engine::cache_blend::{CacheBlendEngine, CachedChunkEntry, ChunkKvStore};
use grim_backend_rocm::kernels::blend_kv_rope::{
    BlendConfig, blend_kv_rope_cpu, blend_kv_rope_transform_cpu,
};

#[test]
fn test_cache_blend_prefill_and_rope_rotation() {
    let chunk_size = 64usize;
    let blend = CacheBlendEngine::new(chunk_size);
    let mut store = ChunkKvStore::new();

    // Populate ChunkKvStore with harvested chunks from prior requests
    store.insert_chunk(CachedChunkEntry {
        chunk_hash: 1001,
        original_base_pos: 0,
        token_len: chunk_size,
        block_ids: vec![10, 11],
        k_data: None,
        v_data: None,
    });
    store.insert_chunk(CachedChunkEntry {
        chunk_hash: 1002,
        original_base_pos: chunk_size,
        token_len: chunk_size,
        block_ids: vec![12, 13],
        k_data: None,
        v_data: None,
    });
    store.insert_chunk(CachedChunkEntry {
        chunk_hash: 2001,
        original_base_pos: 128,
        token_len: chunk_size,
        block_ids: vec![20, 21],
        k_data: None,
        v_data: None,
    });

    // Prompt with 4 chunks: chunks 0 and 1 are cached prefix; chunk 2 is cached non-prefix; chunk 3 missed
    let hashes = vec![1001, 1002, 2001, 9999];
    let layout = blend.plan_stitched_prompt(&hashes, chunk_size, |h| {
        store.lookup(h).map(|e| e.block_ids.clone())
    });

    assert_eq!(layout.total_tokens, 256);
    assert_eq!(layout.cached_token_count, 192);
    assert_eq!(layout.cached_segments.len(), 3);
    assert_eq!(layout.compute_ranges.len(), 1);
    assert_eq!(layout.compute_ranges[0], (192, 256));

    // Test blend_kv_rope on 50% prefix match + 50% divergent suffix
    let cfg = BlendConfig {
        block_size: 16,
        num_heads: 4,
        head_dim: 16,
        divergence_token: 8,
        rotary_dim: 16,
        interleaved: false,
    };
    let total_len = cfg.block_size * cfg.num_heads * cfg.head_dim;
    let k_src = vec![1.5f32; total_len];
    let v_src = vec![2.5f32; total_len];

    let mut k_dst = vec![0.0f32; total_len];
    let mut v_dst = vec![0.0f32; total_len];
    let prefix_len = cfg.divergence_token * cfg.num_heads * cfg.head_dim;
    k_dst[..prefix_len].fill(9.0);
    v_dst[..prefix_len].fill(9.5);

    blend_kv_rope_cpu(&cfg, &k_src, &v_src, &mut k_dst, &mut v_dst, 0, 10000.0)
        .expect("blend_kv_rope_cpu succeeds");

    // Prefix preserved
    for &x in &k_dst[..prefix_len] {
        assert_eq!(x, 9.0);
    }
    for &x in &v_dst[..prefix_len] {
        assert_eq!(x, 9.5);
    }

    // Suffix tokens 8..16 rotated and finite
    for &x in &k_dst[prefix_len..] {
        assert!(x.is_finite());
        assert_ne!(x, 0.0);
    }
    for &x in &v_dst[prefix_len..] {
        assert_eq!(x, 2.5); // V is copied verbatim
    }

    // Verify two-position unrotate(old_pos) -> rotate(new_pos) transform:
    // If a chunk was cached post-RoPE at old_pos, unrotating to base and re-rotating to new_pos
    // matches direct rotation from pre-RoPE raw keys to new_pos.
    let raw_k = vec![1.234f32; total_len];
    let mut post_rope_at_old = vec![0.0f32; total_len];
    let mut dummy_v = vec![0.0f32; total_len];
    let old_pos = 64usize;
    let new_pos = 128usize;
    let rope_theta = 10000.0f32;

    let full_cfg = BlendConfig {
        block_size: 16,
        num_heads: 4,
        head_dim: 16,
        divergence_token: 0,
        rotary_dim: 16,
        interleaved: false,
    };

    // 1. Direct rotation from pre-RoPE raw keys to new_pos
    let mut direct_at_new = vec![0.0f32; total_len];
    blend_kv_rope_cpu(
        &full_cfg,
        &raw_k,
        &v_src,
        &mut direct_at_new,
        &mut dummy_v,
        new_pos,
        rope_theta,
    )
    .unwrap();

    // 2. Initial rotation to old_pos
    blend_kv_rope_cpu(
        &full_cfg,
        &raw_k,
        &v_src,
        &mut post_rope_at_old,
        &mut dummy_v,
        old_pos,
        rope_theta,
    )
    .unwrap();

    // 3. Two-position transform: unrotate(old_pos) -> rotate(new_pos)
    let mut re_rotated_to_new = vec![0.0f32; total_len];
    blend_kv_rope_transform_cpu(
        &full_cfg,
        &post_rope_at_old,
        &v_src,
        &mut re_rotated_to_new,
        &mut dummy_v,
        Some(old_pos),
        new_pos,
        rope_theta,
    )
    .unwrap();

    // Verify bit-exact/numerical equivalence between direct rotation and unrotate->rotate transform
    let mut max_diff = 0.0f32;
    for (&d, &r) in direct_at_new.iter().zip(re_rotated_to_new.iter()) {
        max_diff = max_diff.max((d - r).abs());
    }
    assert!(
        max_diff < 1e-5,
        "Two-position RoPE unrotate->rotate transform failed: max_diff = {max_diff} >= 1e-5"
    );
}
