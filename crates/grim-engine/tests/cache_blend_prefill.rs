//! Integration test for Task 10.7: CacheBlend prefill layout and blend_kv_rope_kernel verification.

use grim_engine::cache_blend::CacheBlendEngine;
use grim_backend_rocm::kernels::blend_kv_rope::{BlendConfig, blend_kv_rope_cpu};

#[test]
fn test_cache_blend_prefill_and_rope_rotation() {
    let chunk_size = 64usize;
    let blend = CacheBlendEngine::new(chunk_size);

    // Prompt with 4 chunks: chunks 0 and 1 are cached prefix; chunk 2 is cached non-prefix; chunk 3 missed
    let hashes = vec![1001, 1002, 2001, 9999];
    let layout = blend.plan_stitched_prompt(&hashes, chunk_size, |h| {
        match h {
            1001 => Some(vec![10, 11]),
            1002 => Some(vec![12, 13]),
            2001 => Some(vec![20, 21]),
            _ => None,
        }
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
}
