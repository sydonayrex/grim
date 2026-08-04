use grim_format::gguf::GrimMetadata;

#[test]
fn metadata_roundtrip_preserves_new_fields() {
    let v2 = grim_format::gguf::GrimMetadataV2 {
        rotation_id: Some("gyrot".into()),
        rotation_inverse: Some(vec![1, 2, 3]),
        recon_method: Some("serq".into()),
        recon_rank: Some(4),
        kv_method: Some("rotatekv".into()),
        kv_bpw: Some(3.25),
    };
    let mut meta = GrimMetadata::default();
    meta.set_v2(v2.clone());
    let restored = GrimMetadata::from_json(&meta.to_json());
    let restored_v2 = restored.v2();
    assert_eq!(restored_v2.rotation_id, v2.rotation_id);
    assert_eq!(restored_v2.recon_rank, v2.recon_rank);
    assert_eq!(restored_v2.kv_bpw, v2.kv_bpw);
}
