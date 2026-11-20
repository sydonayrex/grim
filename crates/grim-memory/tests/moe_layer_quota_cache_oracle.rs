use grim_memory::moe_budget::{ForesightSwapWindow, LruResidencyTracker, LayerQuotaExpertCache};

/// On a 48-layer MoE prefill where each layer activates 8 experts per chunk across 2 chunks,
/// global LRU with capacity 192 (4 slots/layer) thrashes every layer when chunk 0 exceeds 192
/// total admissions (0% reuse on chunk 1), whereas per-layer quota `[l*4, (l+1)*4)` retains 4/8
/// experts per layer (50.0% hit rate on chunk 1).
#[test]
fn oracle_per_layer_quota_eliminates_cross_layer_prefill_thrashing() {
    let num_layers = 48;
    let capacity = 192; // 4 slots per layer
    let mut global_lru = LruResidencyTracker::new(capacity);
    let mut quota_cache = LayerQuotaExpertCache::new(capacity, num_layers, 4);

    // Chunk 0: each layer routes experts 0..8
    for l in 0..num_layers {
        for e in 0..8 {
            global_lru.admit(l, e);
            quota_cache.admit(l, e, &[]);
        }
    }

    // Chunk 1: check how many of the 8 experts in early layers (0..24) survived from chunk 0
    let mut global_early_hits = 0usize;
    let mut quota_early_hits = 0usize;
    for l in 0..24 {
        for e in 0..8 {
            if global_lru.lookup(l, e).is_some() {
                global_early_hits += 1;
            }
            if quota_cache.lookup(l, e).is_some() {
                quota_early_hits += 1;
            }
        }
    }

    assert_eq!(
        global_early_hits, 0,
        "global LRU evicts 100% of early layers 0..24 when cycle > capacity"
    );
    assert_eq!(
        quota_early_hits,
        24 * 4,
        "per-layer quota must retain exactly q=4 experts in every early layer"
    );
}

/// Adaptive 64-step window switches to LFU under Zipfian hot-expert traffic + one-shot scan noise.
#[test]
fn oracle_adaptive_window_switches_to_lfu_under_scan_plus_hot_set() {
    let mut cache = LayerQuotaExpertCache::new(4, 1, 4);
    for _ in 0..32 {
        cache.admit(0, 0, &[]);
        cache.admit(0, 1, &[]);
        cache.admit(0, 2, &[]);
    }
    for step in 0..64 {
        let hot = step % 3;
        cache.admit(0, hot, &[]);
        cache.admit(0, 100 + step, &[]);
    }
    assert!(
        cache.is_using_lfu(),
        "shadow tracker must switch policy to LFU when hot experts outperform recency under scan noise"
    );
    assert!(cache.lookup(0, 0).is_some());
    assert!(cache.lookup(0, 1).is_some());
    assert!(cache.lookup(0, 2).is_some());
}

/// 1. 2-hit streak filter promotes an expert only on its second consecutive demand.
/// 2. Window-depth protection refuses to evict an expert demanded within the next `depth=4` steps.
#[test]
fn oracle_foresight_swap_two_hit_streak_and_lookahead_guard() {
    let mut fs = ForesightSwapWindow::new(4);
    assert!(!fs.record_demand_and_should_promote(0, 7), "1st hit must not promote");
    assert!(fs.record_demand_and_should_promote(0, 7), "2nd consecutive hit must promote");
    assert!(!fs.record_demand_and_should_promote(0, 9), "interleaved miss resets other expert");

    let mut cache = LayerQuotaExpertCache::new(3, 1, 4);
    cache.admit(0, 10, &[]);
    cache.admit(0, 20, &[]);
    cache.admit(0, 30, &[]);
    let (_slot, evicted) = cache.admit(0, 40, &[10]);
    assert_eq!(evicted, Some((0, 20)), "lookahead guard must protect expert 10 from eviction");
    assert!(cache.lookup(0, 10).is_some());
}
