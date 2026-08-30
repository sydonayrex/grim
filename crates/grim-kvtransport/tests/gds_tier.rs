use grim_kvtransport::{GdsTier, gds_ffi::HipFileLib};
use tempfile::tempdir;

#[test]
fn test_gds_probe_availability_no_panic() {
    // Probing should be completely safe and never panic
    let available = HipFileLib::probe_available();
    println!("GDS libhipfile availability: {}", available);
}

#[test]
fn test_gds_tier_lifecycle_and_roundtrip() {
    let dir = tempdir().unwrap();
    let tier = GdsTier::new(dir.path()).expect("GdsTier initialization should succeed");

    let block_id = 42;
    let block_data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.25).collect();

    tier.demote_block(block_id, &block_data).expect("demote_block should succeed");

    let mut restored = vec![0.0f32; 1024];
    tier.promote_block(block_id, &mut restored).expect("promote_block should succeed");

    assert_eq!(restored, block_data);
}

#[test]
fn test_gds_tier_nonexistent_block_returns_error() {
    let dir = tempdir().unwrap();
    let tier = GdsTier::new(dir.path()).unwrap();
    let mut buf = vec![0.0f32; 128];
    assert!(tier.promote_block(99999, &mut buf).is_err());
}
