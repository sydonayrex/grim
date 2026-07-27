use grim_format::gguf::{GgufDType, GGUF_MAGIC, GGUF_VERSION};
use grim_format::tprov::GgufProvider;
use grim_tensor::provider::TensorProvider;

/// GGUF stores dimensions outer-first; `GgufTensorInfo::shape()` reverses them
/// to inner-first.  These helpers accept the **logical** (inner-first) shape
/// and reverse for the on-disk layout.
fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn push_tensor(buf: &mut Vec<u8>, name: &str, logical_dims: &[u64], dtype_tag: u32, offset: u64) {
    push_string(buf, name);
    buf.extend_from_slice(&(logical_dims.len() as u32).to_le_bytes());
    for &d in logical_dims.iter().rev() {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&dtype_tag.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

fn build_gguf(tensors: &[(&str, &[u64], u32, u64)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count = 0
    for &(name, dims, dtype, offset) in tensors {
        push_tensor(&mut buf, name, dims, dtype, offset);
    }
    buf
}

/// Write a hand-constructed GGUF v3 file to a temp location and return the path.
fn write_gguf_bytes(bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("grim_golden_gguf");
    std::fs::create_dir_all(&dir).unwrap();
    let name = format!("golden_{:016x}.gguf", randish());
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// A deterministic pseudo-random u64 for unique temp filenames.
fn randish() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    d.as_nanos() as u64
}

/// Golden end-to-end: hand-construct GGUF with one F32 tensor, open via
/// GgufProvider, read tensor bytes, verify exact values.
#[test]
fn golden_gguf_provider_round_trips_hand_constructed_f32_tensor() {
    // Build a GGUF with 1 tensor "test.weight", shape [2, 3], F32.
    let mut header = build_gguf(&[("test.weight", &[2, 3], GgufDType::F32 as u32, 0)]);

    // Compute data_start: align header length up to 32.
    let info_end = header.len() as u64;
    let data_start = (info_end + 31) & !31;

    // Pad up to data_start.
    header.resize(data_start as usize, 0x00);

    // Write 6 f32 values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].
    let values: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for v in &values {
        header.extend_from_slice(&v.to_le_bytes());
    }

    let path = write_gguf_bytes(&header);
    let provider = GgufProvider::open(path.to_str().unwrap()).unwrap();
    let raw = provider.get("test.weight").unwrap();

    assert_eq!(raw.shape, vec![2, 3], "shape read back");
    assert_eq!(raw.bytes.len(), 24, "6 × f32 = 24 bytes");

    let got: Vec<f32> = raw
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got, &values[..], "actual f32 payload values");
}

/// Golden test: GgufProvider::meta returns correct shape + dtype for a tensor.
#[test]
fn golden_gguf_provider_meta_returns_expected_metadata() {
    let mut header = build_gguf(&[("foo.bar", &[4, 8], GgufDType::F32 as u32, 0)]);
    let info_end = header.len() as u64;
    let data_start = (info_end + 31) & !31;
    header.resize(data_start as usize, 0x00);
    // Write 32 zeros (4*8*4 bytes)
    header.extend_from_slice(&[0u8; 32]);

    let path = write_gguf_bytes(&header);
    let provider = GgufProvider::open(path.to_str().unwrap()).unwrap();
    let meta = provider.meta("foo.bar").unwrap();

    assert_eq!(meta.shape, vec![4, 8], "meta shape");
    assert!(matches!(meta.dtype.storage, grim_tensor::Storage::Native), "meta dtype should be native");
}

/// Golden test: GgufProvider::get rejects unknown tensor name.
#[test]
fn golden_gguf_provider_rejects_unknown_tensor() {
    let header = build_gguf(&[]); // no tensors
    let path = write_gguf_bytes(&header);
    let provider = GgufProvider::open(path.to_str().unwrap()).unwrap();
    let result = provider.get("nonexistent");
    assert!(result.is_err(), "unknown tensor must return Err");
}

/// Golden test: GgufProvider loads a Q8_0 tensor and the raw bytes
/// match the expected packed size. (We don't verify dequant values here
/// since that's covered by golden_dequant in grim-quant.)
#[test]
fn golden_gguf_provider_q80_tensor_size() {
    // 64 params of Q8_0 → 2 blocks × 34 bytes = 68 bytes.
    let mut header = build_gguf(&[("q.weight", &[64], GgufDType::Q8_0 as u32, 0)]);
    let info_end = header.len() as u64;
    let data_start = (info_end + 31) & !31;
    header.resize(data_start as usize, 0x00);
    header.extend_from_slice(&[0xABu8; 68]); // 2 blocks of Q8_0 padding

    let path = write_gguf_bytes(&header);
    let provider = GgufProvider::open(path.to_str().unwrap()).unwrap();
    let raw = provider.get("q.weight").unwrap();
    assert_eq!(raw.bytes.len(), 68, "Q8_0 64 params = 68 raw bytes");
}
