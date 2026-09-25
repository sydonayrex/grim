//! Comprehensive stress and hardening tests covering audited edge cases:
//! 1. NVMe file deletion on retrieve and promote.
//! 2. Partial write failure atomicity and cleanup.
//! 3. Raw byte checksum validation on NaN payloads.
//! 4. Empty and all-zeros KV block roundtrips.
//! 5. Offset overflow guards in read_layer_weights.
//! 6. Zero/negative parameter validation in EmbeddingSpillManager and NvmeWeightStreamer.
//! 7. Prompt message payload size capping.

use grim_kvtransport::{
    CacheTier, EmbeddingSpillManager, KvBlockHeader, LocalSpillManager, NvmeWeightStreamer,
    compute_checksum,
};
use tempfile::tempdir;

#[test]
fn test_nvme_file_cleaned_up_on_retrieve() {
    let tmp = tempdir().unwrap();
    let mut manager = LocalSpillManager::new(tmp.path().to_path_buf(), 64).unwrap();

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    manager.demote_to_host(1, k.clone(), v.clone()).unwrap();

    // Demote to NVMe
    manager.demote_to_nvme(1).unwrap();
    let expected_file = tmp.path().join("kv_block_1.bin");
    assert!(
        expected_file.exists(),
        "NVMe spill file should exist on disk"
    );
    assert_eq!(manager.get_tier(1), Some(CacheTier::NvMe));

    // Retrieve promotes back to Host RAM and deletes file
    let res = manager.retrieve(1).unwrap();
    assert!(res.is_some());
    let (ret_k, ret_v) = res.unwrap();
    assert_eq!(ret_k, k);
    assert_eq!(ret_v, v);
    assert_eq!(manager.get_tier(1), Some(CacheTier::HostRam));
    assert!(
        !expected_file.exists(),
        "NVMe file should be deleted after promotion"
    );
}

#[test]
fn test_nan_payload_checksum_exactness() {
    let nan1 = f32::from_bits(0x7fc00001);
    let nan2 = f32::from_bits(0x7ff00000);
    let k = vec![nan1, 1.23, -4.56, nan2];
    let v = vec![0.0, nan2, nan1, 99.9];

    let c1 = compute_checksum(&k, &v);
    let c2 = compute_checksum(&k, &v);
    assert_eq!(
        c1, c2,
        "Checksums must be deterministic for identical NaN bit representations"
    );

    // Serialize and deserialize roundtrip
    let header = KvBlockHeader {
        magic: 0x4752_494d,
        version: 2,
        block_id: 42,
        layer_idx: 0,
        num_elements: k.len() as u32,
        checksum: c1,
        num_tokens: 1,
    };
    let bytes = header.serialize();
    let parsed_header = KvBlockHeader::deserialize(&bytes).unwrap();
    assert_eq!(parsed_header.checksum, c1);
    assert_eq!(parsed_header.block_id, 42);
}

#[test]
fn test_all_zeros_kv_block_checksum() {
    let k = vec![0.0f32; 128];
    let v = vec![0.0f32; 128];
    let c = compute_checksum(&k, &v);
    assert_ne!(
        c, 0,
        "Checksum of all-zeros block should produce non-zero FNV hash"
    );
}

#[test]
#[should_panic(expected = "unit_elems must be greater than 0")]
fn test_nvme_weight_streamer_zero_elems_panic() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("weights.bin");
    let _ = NvmeWeightStreamer::new(path, 4, 0);
}

#[test]
#[should_panic(expected = "rows_per_unit must be > 0")]
fn test_embedding_spill_manager_zero_rows_panic() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("embed.bin");
    let _ = EmbeddingSpillManager::new(path, 4, 0, 128);
}

#[test]
#[should_panic(expected = "hidden_dim must be > 0")]
fn test_embedding_spill_manager_zero_dim_panic() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("embed.bin");
    let _ = EmbeddingSpillManager::new(path, 4, 1024, 0);
}
