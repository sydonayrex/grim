use grim_disagg::bloom::BloomFilter;

#[test]
fn test_bloom_filter_false_positive_rate_under_target() {
    let expected_items = 10_000;
    let target_fp_rate = 0.01; // 1%
    let mut bf = BloomFilter::new(expected_items, target_fp_rate);

    for i in 0..expected_items {
        bf.insert(&(i as u64).to_le_bytes());
    }

    // All inserted items must test positive
    for i in 0..expected_items {
        assert!(bf.might_contain(&(i as u64).to_le_bytes()));
    }

    // Test non-members to verify empirical false positive rate
    let test_queries = 100_000;
    let mut false_positives = 0;
    for i in expected_items..(expected_items + test_queries) {
        if bf.might_contain(&(i as u64).to_le_bytes()) {
            false_positives += 1;
        }
    }

    let empirical_fp_rate = (false_positives as f64) / (test_queries as f64);
    println!("Empirical FP rate: {:.4} (target: {})", empirical_fp_rate, target_fp_rate);
    assert!(
        empirical_fp_rate <= 0.015,
        "empirical FP rate {} exceeded bound",
        empirical_fp_rate
    );
}

#[test]
fn test_bloom_filter_empty() {
    let bf = BloomFilter::new(100, 0.01);
    assert!(!bf.might_contain(b"nonexistent"));
}
