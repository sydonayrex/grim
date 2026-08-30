use grim_disagg::bloom::BloomFilter;
use grim_disagg::lookup::LookupClient;

#[test]
fn test_lookup_client_skips_rtt_on_negative() {
    let mut remote_bloom = BloomFilter::new(100, 0.01);
    // Insert token prefix [1, 2, 3]
    let mut key = Vec::new();
    for &t in &[1u32, 2, 3] {
        key.extend_from_slice(&t.to_le_bytes());
    }
    remote_bloom.insert(&key);

    let client = LookupClient::new(remote_bloom, "127.0.0.1:9099".to_string());

    // Non-existent prefix [9, 9, 9] should return false/None immediately without network call
    assert!(!client.might_have_prefix(&[9, 9, 9]));
    // Existing prefix [1, 2, 3] returns true
    assert!(client.might_have_prefix(&[1, 2, 3]));
}
