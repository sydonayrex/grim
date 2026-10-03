//! WhiteRaven must survive the `.grim` container: tag 670 in, blocked-FP8 out.
//!
//! The chain under test is the one a real user hits:
//!
//!   tag-670 .gguf  --convert_to_grim-->  .grim  --GrimProvider-->  blocked FP8
//!
//! Every link has a way to fail silently:
//!
//! - The writer used to dequantize every source tensor to F32 and repack,
//!   destroying the blocked layout and misreading the codes as Q80 samples.
//! - The reader resolved dtype from `base_bitwidth` alone, so even a correct
//!   writer produced a file the reader would hand back as Q80.
//! - The scheme was only recoverable if `grim.quant_overrides` was written by
//!   the writer AND honored by the reader. Neither end did that for `.grim`.
//!
//! This test pins all three: pass-through bytes, a serialized override naming
//! tag 670, and a reader that honors it.

use std::io::Write;

use grim_format::gguf::{GGUF_MAGIC, GGUF_VERSION, GgufDType};
use grim_format::tprov::GrimProvider;
use grim_format::GgufProvider;
use grim_tensor::provider::TensorProvider;

fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// A one-tensor GGUF carrying `dtype`. Payload is caller-supplied and must
/// match the dtype's declared geometry.
fn write_gguf(path: &std::path::Path, dtype: GgufDType, name: &str, dims: &[u64], payload: &[u8]) {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    push_string(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "qwen4exp");
    push_string(&mut buf, "general.name");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "grim-container-probe");

    push_string(&mut buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&dtype.tag().to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let aligned = (buf.len() + 31) / 32 * 32;
    buf.resize(aligned, 0);
    buf.extend_from_slice(payload);

    let mut f = std::fs::File::create(path).expect("create");
    f.write_all(&buf).expect("write");
}

/// One directory per TEST, not per process (see `prism_unsupported_format_test`).
struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new() -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("grim_container_{}_{}", std::process::id(), n));
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
fn whiteraven_loads_through_the_grim_container_round_trip() {
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let want: Vec<f32> = (0..n * k)
        .map(|i| ((i * 37) % 61) as f32 * 0.03125 - 0.9)
        .collect();
    let codes: Vec<u8> = want.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let blocked = grim_quant::block_fp8_16x16(&codes, n, k).expect("block");

    // 1. Source: tag-670 GGUF. NOTE the dims are [k, n] -- GGUF stores
    //    fastest-first; `shape()` reverses it. Getting this backwards produces
    //    a plausible, wrong matmul rather than an error.
    let src = scratch.0.join("src.gguf");
    write_gguf(&src, GgufDType::WhiteRaven, "blk.weight", &[k as u64, n as u64], &blocked);

    // Sanity: the GGUF reader already resolves the tag.
    let prov = GgufProvider::open(src.to_str().unwrap()).expect("open gguf");
    let raw = prov.get("blk.weight").expect("get");
    assert_eq!(
        raw.dtype.storage,
        grim_tensor::dtype::Storage::FloatPack(
            grim_tensor::dtype::FloatPackScheme::Fp8Blocked16
        ),
        "the GGUF reader must resolve tag 670 before the writer even runs"
    );

    // 2. Convert to .grim.
    let out = scratch.0.join("out.grim");
    grim_format::convert_to_grim(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "gfx1200",
        8.0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("convert to grim");

    // 3. Read it back through GrimProvider.
    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open grim");
    let meta = gp.meta("blk.weight").expect("meta");
    assert_eq!(
        meta.dtype.storage,
        grim_tensor::dtype::Storage::FloatPack(
            grim_tensor::dtype::FloatPackScheme::Fp8Blocked16
        ),
        "GrimProvider must honor the quant_override the writer records"
    );
    assert_eq!(meta.shape, vec![n, k]);

    let raw2 = gp.get("blk.weight").expect("get from grim");
    assert_eq!(
        raw2.bytes, blocked,
        "the .grim payload must be the SAME blocked codes, not a requantized copy"
    );

    // 4. And they must still decode to the source weights.
    let deq = grim_quant::dequant_fp8_blocked16(&raw2.bytes, n, k).expect("dequant");
    for (i, (&w, &g)) in want.iter().zip(&deq).enumerate() {
        let q = grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(w));
        assert_eq!(g.to_bits(), q.to_bits(), "element {i} diverged through .grim");
    }

    // 5. The file itself must serialize the override, or the chain works today
    //    only because nobody re-reads the JSON.
    let bytes = std::fs::read(&out).expect("read file");
    // Header is magic(5) | metadata_len(8) | num_tensors(4) = 17 bytes.
    let meta_len = u64::from_le_bytes(bytes[5..13].try_into().unwrap()) as usize;
    let json: serde_json::Value =
        serde_json::from_slice(&bytes[17..17 + meta_len]).expect("metadata JSON parses");
    let overrides = json
        .get("quant_overrides")
        .and_then(|v| v.as_array())
        .expect("file must carry quant_overrides");
    let found = overrides.iter().any(|o| {
        o.get("tensor_name").and_then(|v| v.as_str()) == Some("blk.weight")
            && o.get("override_dtype").and_then(|v| v.as_u64()) == Some(670)
    });
    assert!(found, "quant_overrides must name tag 670 for blk.weight");
}
