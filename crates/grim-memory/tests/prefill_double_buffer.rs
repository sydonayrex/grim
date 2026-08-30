use grim_memory::moe_budget::{OffloadMoeCache, TimeRange};

#[test]
fn test_prefill_overlap_hides_pcie() {
    let cache = OffloadMoeCache::new(/*prefill_overlap=*/true, 4);
    assert!(cache.is_overlap_enabled());

    let (load_0, gemm_0) = cache.schedule_layer(0, &[0, 1]);
    let (load_1, gemm_1) = cache.schedule_layer(1, &[2, 3]);

    // Layer 1 loads concurrently while Layer 0 executes its GEMM
    assert!(load_1.overlaps(&gemm_0), "layer 1 load must overlap with layer 0 gemm");
    assert!(gemm_1.start >= gemm_0.end, "layer 1 gemm must start after layer 0 gemm finishes");
    assert!(load_0.end <= gemm_0.start, "layer 0 gemm waits for layer 0 load");
}

#[test]
fn test_time_range_overlap_math() {
    let t1 = TimeRange { start: 10.0, end: 20.0 };
    let t2 = TimeRange { start: 15.0, end: 25.0 };
    let t3 = TimeRange { start: 20.0, end: 30.0 };
    let t4 = TimeRange { start: 25.0, end: 35.0 };

    assert!(t1.overlaps(&t2));
    assert!(t2.overlaps(&t1));
    assert!(!t1.overlaps(&t4));
    assert!(t2.overlaps(&t3));
}
