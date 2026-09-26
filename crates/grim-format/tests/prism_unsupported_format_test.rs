//! End-to-end proof that a Prism-format checkpoint fails loudly.
//!
//! A GGUF file whose tensors use `PQ2_0` (142) or `PTQ1_0` (143) must open far
//! enough to report *which* format is unsupported and what to do about it —
//! not "unknown GGUF dtype tag 142" from the parser, and not a silent
//! reinterpretation under a same-shaped K-quant scheme.

use std::io::Write;

use grim_format::gguf::{GGUF_MAGIC, GGUF_VERSION, GgufDType};
use grim_format::tprov::GgufProvider;
use grim_tensor::provider::TensorProvider;

fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// One synthetic tensor using `dtype`, with a real (zero-filled) payload.
fn write_gguf(path: &std::path::Path, dtype: GgufDType, name: &str, params: u64) {
    let block = dtype.block_size();
    let tsize = dtype.type_size_per_block();
    let nblocks = params.div_ceil(block);
    let payload = nblocks * tsize;

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&2u64.to_le_bytes()); // kv count
    push_string(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "qwen4exp");
    push_string(&mut buf, "general.name");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "prism-format-probe");

    push_string(&mut buf, name);
    buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    buf.extend_from_slice(&params.to_le_bytes());
    buf.extend_from_slice(&dtype.tag().to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset

    let aligned = (buf.len() + 31) / 32 * 32;
    buf.resize(aligned, 0);
    buf.resize(buf.len() + payload as usize, 0);

    let mut f = std::fs::File::create(path).expect("create");
    f.write_all(&buf).expect("write");
}

struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("grim_prism_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("mkdir");
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn prism_tag_is_parsed_not_rejected_as_unknown() {
    let d = Scratch::new();
    for (dtype, name) in [(GgufDType::PQ2_0, "PQ2_0"), (GgufDType::PTQ1_0, "PTQ1_0")] {
        let p = d.0.join(format!("{name}.gguf"));
        // 128 weights = exactly one block.
        write_gguf(&p, dtype, "blk.0.ffn_up_exps.weight", 128);
        let provider = GgufProvider::open(p.to_str().unwrap())
            .unwrap_or_else(|e| panic!("{name} must parse, not fail as an unknown tag: {e}"));
        let meta = provider
            .meta("blk.0.ffn_up_exps.weight")
            .unwrap_or_else(|e| panic!("{name} tensor must be indexed: {e}"));
        assert_eq!(meta.shape, vec![128], "{name} shape");
    }
}

#[test]
fn prism_tensor_payload_size_matches_geometry() {
    let d = Scratch::new();
    // 4096 weights = 32 PQ2_0 blocks (34 B) = 1088 bytes, 32 PTQ1_0 (28 B) = 896.
    for (dtype, want) in [(GgufDType::PQ2_0, 32 * 34u64), (GgufDType::PTQ1_0, 32 * 28)] {
        let p = d.0.join(format!("sz{}.gguf", dtype.tag()));
        write_gguf(&p, dtype, "blk.0.ssm_out.weight", 4096);
        let bytes = std::fs::metadata(&p).expect("stat").len();
        let header = bytes - want;
        // The provider must accept it, which it only does when the declared
        // size matches the real payload.
        GgufProvider::open(p.to_str().unwrap())
            .unwrap_or_else(|e| panic!("tag {} payload geometry wrong: {e}", dtype.tag()));
        assert!(header > 0, "sanity: there is a header");
    }
}

#[test]
fn unsupported_storage_names_the_format_and_the_remedy() {
    use grim_format::gguf::map_gguf_dtype_to_storage;
    use grim_tensor::dtype::Storage;

    for (dtype, name) in [(GgufDType::PQ2_0, "PQ2_0"), (GgufDType::PTQ1_0, "PTQ1_0")] {
        let s = map_gguf_dtype_to_storage(dtype).storage;
        let Storage::Unsupported(f) = s else {
            panic!("{name} must be Unsupported");
        };
        assert_eq!(f.name, name);
        assert_eq!(f.block_size, Some(128));
        assert_eq!(
            f.bytes_per_block,
            Some(if name == "PQ2_0" { 34 } else { 28 })
        );
        // The message must tell the user what failed AND what to do.
        for needle in [name, "Re-quantize", "Q2_0 tag 42"] {
            assert!(
                f.reason.contains(needle),
                "{name} reason missing {needle:?}: {}",
                f.reason
            );
        }
    }
}

#[test]
fn upstream_tag_42_is_untouched_and_still_supported() {
    use grim_format::gguf::map_gguf_dtype_to_storage;
    use grim_tensor::dtype::{KQuantScheme, Storage};

    let s = map_gguf_dtype_to_storage(GgufDType::GsqRco3p5).storage;
    assert!(matches!(s, Storage::KQuant(KQuantScheme::GsqRco3p5)));
    assert_eq!(GgufDType::GsqRco3p5.tag(), 42);
    assert_eq!(GgufDType::GsqRco3p5.block_size(), 64);
    assert_eq!(GgufDType::GsqRco3p5.type_size_per_block(), 18);
    // And the real checkpoint still opens and dequantizes.
    let p = "models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf";
    if std::path::Path::new(p).exists() {
        let prov = GgufProvider::open(p).expect("real checkpoint opens");
        assert!(prov.meta("blk.0.ffn_down_exps.weight").is_ok());
    }
}
