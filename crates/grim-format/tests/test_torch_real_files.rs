//! Acid tests: parse the real reference checkpoints under `models/audio/`.
//!
//! Skipped silently when the checkpoint files are absent (CI without model
//! downloads); they are the ground truth for the pickle reader on this host.

use grim_format::torch::PthProvider;
use grim_tensor::provider::TensorProvider;
use std::path::Path;

const KOKORO_PTH: &str = "../../models/audio/Kokoro-82m/kokoro-v1_0.pth";
const VOCOS_PT: &str = "../../models/audio/MeanVC2/vocos.pt";

fn dump(provider: &PthProvider, label: &str) {
    let mut names = provider.tensor_names();
    names.sort();
    println!("[{label}] {} tensors", names.len());
    for n in names.iter().take(10) {
        let meta = provider.meta(n).unwrap();
        println!("  {n} {:?} {:?}", meta.shape, meta.dtype);
    }
}

#[test]
fn real_kokoro_pth_parses() {
    if !Path::new(KOKORO_PTH).exists() {
        eprintln!("skipping: {KOKORO_PTH} not present");
        return;
    }
    let p = PthProvider::load_from_file(KOKORO_PTH).expect("kokoro pth should parse");
    dump(&p, "kokoro");
    // The Kokoro state dict has hundreds of named parameters; a heuristic
    // scanner finds a handful, a real pickle VM finds them all with correct
    // multi-dim shapes.
    assert!(
        p.tensor_names().len() > 100,
        "expected >100 tensors in Kokoro state dict, got {}",
        p.tensor_names().len()
    );
    // Spot-check a known StyleTTS2 parameter shape from config.json:
    // text encoder embedding is [n_token=178? style/hidden dims apply].
    // At minimum every tensor must have a plausible non-empty byte payload
    // whose length matches its shape product at 4 bytes/elem for F32.
    for name in p.tensor_names() {
        let meta = p.meta(&name).unwrap();
        let elems: usize = meta.shape.iter().product();
        let raw = p.get(&name).unwrap();
        assert_eq!(
            raw.bytes.len(),
            elems * 4,
            "tensor {name} shape {:?} vs {} bytes",
            meta.shape,
            raw.bytes.len()
        );
    }
}

#[test]
fn real_vocos_pt_parses() {
    if !Path::new(VOCOS_PT).exists() {
        eprintln!("skipping: {VOCOS_PT} not present");
        return;
    }
    let p = PthProvider::load_from_file(VOCOS_PT).expect("vocos pt should parse");
    dump(&p, "vocos");
    assert!(!p.tensor_names().is_empty(), "vocos.pt yielded no tensors");
}
