use grim_format::torch::PthProvider;
use grim_tensor::provider::TensorProvider;

#[test]
fn test_pth_zip_reader_synthetic() {
    let mut zip_bytes = Vec::new();

    // Construct a minimal uncompressed ZIP entry for "archive/data.pkl"
    let filename = b"archive/data.pkl";
    // Construct fake pickle bytes with two strings: "layer1.weight", "layer1.bias"
    let mut pkl_data = Vec::new();
    pkl_data.push(0x80); // PROTO
    pkl_data.push(4);
    // SHORT_BINUNICODE "layer1.weight"
    let s1 = b"layer1.weight";
    pkl_data.push(0x8c);
    pkl_data.push(s1.len() as u8);
    pkl_data.extend_from_slice(s1);
    // SHORT_BINUNICODE "layer1.bias"
    let s2 = b"layer1.bias";
    pkl_data.push(0x8c);
    pkl_data.push(s2.len() as u8);
    pkl_data.extend_from_slice(s2);
    pkl_data.push(0x2e); // STOP

    // Local file header for data.pkl
    zip_bytes.extend_from_slice(b"PK\x03\x04");
    zip_bytes.extend_from_slice(&20u16.to_le_bytes()); // version
    zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // flags
    zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // comp = stored
    zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // time
    zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // date
    zip_bytes.extend_from_slice(&0u32.to_le_bytes()); // crc
    zip_bytes.extend_from_slice(&(pkl_data.len() as u32).to_le_bytes()); // comp size
    zip_bytes.extend_from_slice(&(pkl_data.len() as u32).to_le_bytes()); // uncomp size
    zip_bytes.extend_from_slice(&(filename.len() as u16).to_le_bytes());
    zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // extra len
    zip_bytes.extend_from_slice(filename);
    zip_bytes.extend_from_slice(&pkl_data);

    // Add "archive/data/0" tensor bytes
    let t0_name = b"archive/data/0";
    let t0_data = vec![0u8; 64]; // 16 f32 values
    zip_bytes.extend_from_slice(b"PK\x03\x04");
    zip_bytes.extend_from_slice(&20u16.to_le_bytes());
    zip_bytes.extend_from_slice(&0u16.to_le_bytes());
    zip_bytes.extend_from_slice(&0u16.to_le_bytes());
    zip_bytes.extend_from_slice(&0u16.to_le_bytes());
    zip_bytes.extend_from_slice(&0u16.to_le_bytes());
    zip_bytes.extend_from_slice(&0u32.to_le_bytes());
    zip_bytes.extend_from_slice(&(t0_data.len() as u32).to_le_bytes());
    zip_bytes.extend_from_slice(&(t0_data.len() as u32).to_le_bytes());
    zip_bytes.extend_from_slice(&(t0_name.len() as u16).to_le_bytes());
    zip_bytes.extend_from_slice(&0u16.to_le_bytes());
    zip_bytes.extend_from_slice(t0_name);
    zip_bytes.extend_from_slice(&t0_data);

    let provider =
        PthProvider::load_from_bytes(&zip_bytes).expect("PthProvider parse should succeed");
    let names = provider.tensor_names();
    assert!(names.contains(&"layer1.weight".to_string()));

    let tensor = provider
        .get("layer1.weight")
        .expect("Get layer1.weight should succeed");
    assert_eq!(tensor.bytes.len(), 64);
}
