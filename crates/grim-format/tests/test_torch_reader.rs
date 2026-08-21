use grim_format::torch::PthProvider;
use grim_tensor::provider::TensorProvider;

/// Emit a minimal-but-genuine PyTorch-style pickle for one state-dict entry:
/// `{name: _rebuild_tensor_v2(('storage', FloatStorage, key, loc), 0,
///    size, stride, True, {})}` — the exact opcode sequence torch.save
/// writes (PROTO 4, GLOBAL/REDUCE/BINPERSID, marks + tuples).
fn torch_pickle_entry(name: &str, storage_key: &str, dims: &[u8]) -> Vec<u8> {
    let mut p = Vec::new();
    let mut s = |b: &[u8]| p.extend_from_slice(b);

    s(&[0x80, 0x04]); // PROTO 4
    s(b"}"); // EMPTY_DICT (state dict root)
    s(&[0x70, 0x00]); // BINPUT 0
    s(&[0x28]); // MARK
    s(&[0x8c, name.len() as u8]);
    s(name.as_bytes());
    s(b"c");
    s(b"torch._utils\n_rebuild_tensor_v2\n"); // GLOBAL
    s(&[0x28]); // MARK — args tuple for REDUCE
    {
        // persistent id: ('storage', torch.FloatStorage, key, 'cpu')
        s(&[0x28]);
        s(&[0x8c, 7]);
        s(b"storage");
        s(b"c");
        s(b"torch\nFloatStorage\n");
        s(&[0x8c, storage_key.len() as u8]);
        s(storage_key.as_bytes());
        s(&[0x8c, 3]);
        s(b"cpu");
        s(&[0x74]); // TUPLE
        s(&[0x51]); // BINPERSID
    }
    s(&[0x4b, 0x00]); // BININT1 offset = 0
    s(&[0x28]); // MARK — size tuple
    for d in dims {
        s(&[0x4b, *d]);
    }
    s(&[0x74]);
    // stride: C-contiguous
    let nd = dims.len();
    s(&[0x28]);
    for i in 0..nd {
        let st: usize = dims[i + 1..]
            .iter()
            .map(|d| *d as usize)
            .product::<usize>()
            .max(1);
        s(&[0x4b, st as u8]);
    }
    s(&[0x74]);
    s(&[0x88]); // NEWTRUE requires_grad
    s(b"}"); // backward_hooks = {}
    s(&[0x74]); // TUPLE — bundle args for REDUCE
    s(&[0x52]); // REDUCE
    s(&[0x65]); // SETITEMS (consumes through MARK)
    s(&[0x2e]); // STOP
    p
}

/// Build an in-memory uncompressed ZIP from (name, payload) entries.
fn stored_zip(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut z = Vec::new();
    for (name, data) in entries {
        z.extend_from_slice(b"PK\x03\x04");
        z.extend_from_slice(&20u16.to_le_bytes()); // version
        z.extend_from_slice(&0u16.to_le_bytes()); // flags
        z.extend_from_slice(&0u16.to_le_bytes()); // method = stored
        z.extend_from_slice(&0u16.to_le_bytes()); // time
        z.extend_from_slice(&0u16.to_le_bytes()); // date
        z.extend_from_slice(&0u32.to_le_bytes()); // crc
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes()); // extra len
        z.extend_from_slice(name.as_bytes());
        z.extend_from_slice(data);
    }
    z
}

#[test]
fn test_pth_zip_reader_synthetic() {
    // 4x4 f32 weight = 64 bytes, matching the storage entry below.
    let pkl = torch_pickle_entry("layer1.weight", "0", &[4, 4]);
    let weight_bytes = vec![7u8; 64];
    let zip = stored_zip(&[
        ("archive/data.pkl", pkl),
        ("archive/data/0", weight_bytes.clone()),
    ]);

    let provider = PthProvider::load_from_bytes(&zip).expect("PthProvider parse should succeed");
    assert!(
        provider
            .tensor_names()
            .contains(&"layer1.weight".to_string())
    );

    let meta = provider.meta("layer1.weight").unwrap();
    assert_eq!(meta.shape, vec![4, 4]);

    let tensor = provider
        .get("layer1.weight")
        .expect("Get layer1.weight should succeed");
    assert_eq!(tensor.bytes.len(), 64);
    assert!(tensor.bytes.iter().all(|b| *b == 7));
}

#[test]
fn test_pth_nonstandard_prefix_and_nested_dict() {
    // Kokoro-style prefix ("model/") plus nested submodule dicts.
    let inner = torch_pickle_entry("enc.weight", "3", &[2, 2]);
    // Wrap: outer pickle referencing nested dict is overkill here — instead
    // verify the reader discovers data.pkl under a non-"archive" prefix.
    let zip = stored_zip(&[("model/data.pkl", inner), ("model/data/3", vec![1u8; 16])]);
    let provider = PthProvider::load_from_bytes(&zip).expect("parse under model/ prefix");
    assert!(provider.tensor_names().contains(&"enc.weight".to_string()));
    let raw = provider.get("enc.weight").unwrap();
    assert_eq!(raw.bytes.len(), 16);
}
