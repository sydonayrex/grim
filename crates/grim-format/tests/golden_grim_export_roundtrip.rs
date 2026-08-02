use grim_format::format::{GrimFile, GrimHeader, GrimTensorEntry};
use grim_format::gguf::{GrimMetadata, GrimRocmlProfile};
use grim_format::tprov::GrimProvider;
use grim_tensor::provider::TensorProvider;
use std::collections::HashMap;
use std::io::Cursor;

#[test]
fn golden_grim_export_round_trips_f32_tensor() {
    let tensor_name = "golden.f32.weight";

    // Hand-construct 64 F32 values that fill exactly one Wave64 segment (256 B).
    let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.125 - 4.0).collect();
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(payload.len(), 256);

    let entry = GrimTensorEntry {
        name: tensor_name.into(),
        shape: vec![8, 8],
        base_bitwidth: 32,
        payload_offset: 0,
        payload_size: 256,
        outlier_count: 0,
        outlier_offset: 0,
        ..Default::default()
    };

    let metadata = GrimMetadata {
        magic: Some("grim-v1".into()),
        quant_version: Some(1),
        rocml_profile: GrimRocmlProfile::Rdna3,
        wavefront_size: 64,
        target_gcn: Some("gfx1100".into()),
        ..Default::default()
    };

    let grim_file = GrimFile {
        header: GrimHeader::new(1, 0),
        metadata,
        tensors: vec![entry],
        tensors_by_name: HashMap::new(),
        kv_blobs: HashMap::new(),
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("golden.grim");

    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        let written = grim_file.write(&mut cursor).expect("write");
        let last = &written[0];
        let needed = (last.payload_offset + last.payload_size) as usize;
        if buf.len() < needed {
            buf.resize(needed, 0);
        }
        // Splice the known payload into the right position.
        let offset = last.payload_offset as usize;
        buf[offset..offset + 256].copy_from_slice(&payload);
    }
    std::fs::write(&path, &buf).expect("write .grim");

    let provider = GrimProvider::open(path.to_str().expect("utf8")).expect("open");
    let raw = provider.get(tensor_name).expect("get golden tensor");

    assert_eq!(raw.shape, vec![8, 8]);
    assert_eq!(raw.bytes.len(), 256);

    let got_values: Vec<f32> = raw
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for (i, (&got, &want)) in got_values.iter().zip(values.iter()).enumerate() {
        let abs = (got - want).abs();
        assert!(abs == 0.0, "golden[{}]: got {} want {}", i, got, want,);
    }
}
