//! The WhiteRaven loader, end to end: a file carrying tag 670 must load to a
//! blocked-FP8 tensor whose dequantized values are the original f32 weights.
//!
//! The tag table test (`raven_crow_tags`) proves the tag resolves. This proves
//! the *bytes* survive the trip: a blocked tensor and a row-major one hold the
//! same bytes in different orders, so a loader that resolves the right storage
//! but reads the payload row-major yields finite, plausible, wrong weights and
//! no error at all. Only an end-to-end decode catches that.
//!
//! Built the same way as `prism_unsupported_format_test`, so the file really
//! goes through `GgufProvider::open` rather than a hand-built DType.

use std::io::Write;

use grim_format::gguf::{GGUF_MAGIC, GGUF_VERSION, GgufDType};
use grim_format::tprov::GgufProvider;
use grim_tensor::provider::TensorProvider;

fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// A one-tensor GGUF file whose single tensor uses `dtype`, payload `bytes`.
fn write_gguf(path: &std::path::Path, dtype: GgufDType, name: &str, dims: &[u64], payload: &[u8]) {
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
    push_string(&mut buf, "whiteraven-loader-probe");

    push_string(&mut buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&dtype.tag().to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Payload region starts 32-byte aligned; tensor offsets are relative to it.
    let aligned = (buf.len() + 31) / 32 * 32;
    buf.resize(aligned, 0);
    buf.extend_from_slice(payload);

    let mut f = std::fs::File::create(path).expect("create probe");
    f.write_all(&buf).expect("write probe");
}

/// One directory per TEST, not per process: keyed on pid alone, the tests race
/// under cargo's parallel threads (same reasoning as `prism_unsupported_format_test`).
struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new() -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("grim_wr_loader_{}_{}", std::process::id(), n));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("mkdir");
        Self(p)
    }
    fn path(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_tag_670_tensor_loads_as_blocked_fp8_and_decodes_to_its_weights() {
    // 32x64 so the 16x16 blocking is exercised on real multi-block geometry.
    let (n, k) = (32usize, 64usize);
    let want: Vec<f32> = (0..n * k)
        .map(|i| ((i * 37) % 61) as f32 * 0.03125 - 0.9)
        .collect();

    // Produce the payload through grim's own producer, not by hand: a
    // hand-blocked buffer would not catch a producer that disagrees with the
    // loader about the arrangement.
    let codes: Vec<u8> = want.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let blocked = grim_quant::block_fp8_16x16(&codes, n, k).expect("block");
    assert_eq!(blocked.len(), n * k, "blocked is a permutation, not a compression");

    let scratch = Scratch::new();
    let path = scratch.path("wr.gguf");
    write_gguf(&path, GgufDType::WhiteRaven, "blk.weight", &[n as u64, k as u64], &blocked);
    let provider = GgufProvider::open(path.to_str().expect("path")).expect("open");

    // The dtype must be blocked FP8, not row-major FP8.
    let meta = provider.meta("blk.weight").expect("meta");
    assert_eq!(
        meta.dtype.storage,
        grim_tensor::dtype::Storage::FloatPack(
            grim_tensor::dtype::FloatPackScheme::Fp8Blocked16
        ),
        "tag 670 must load as blocked FP8"
    );
    assert_eq!(meta.dtype.arith, grim_tensor::ArithType::U8);

    // And the bytes must decode back to the original f32, in row-major order.
    let raw = provider.get("blk.weight").expect("get");
    assert_eq!(raw.bytes, blocked, "payload must survive the file round-trip verbatim");
    let deq = grim_quant::dequant_fp8_blocked16(&raw.bytes, n, k).expect("dequant");
    assert_eq!(deq.len(), want.len());
    for (i, (&w, &g)) in want.iter().zip(&deq).enumerate() {
        // The tensor really is fp8, so compare against the quantized value, not
        // the f32 source.
        let q = grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(w));
        assert_eq!(
            g.to_bits(),
            q.to_bits(),
            "element {i}: blocked load decoded {g:e}, expected {q:e} -- a row-major \
             read of these bytes would produce finite, plausible, wrong weights here"
        );
    }
}

#[test]
fn blocked_fp8_is_a_permutation_not_a_compression_and_ragged_n_is_the_kernels_refusal() {
    // n=24 is 16-misaligned; k=32 stays clean so this isolates the N dimension.
    let scratch = Scratch::new();
    let path = scratch.path("ragged.gguf");
    let payload = vec![0u8; 24 * 32];
    write_gguf(&path, GgufDType::WhiteRaven, "blk.weight", &[24, 32], &payload);

    let provider = GgufProvider::open(path.to_str().expect("path")).expect("open");
    let meta = provider.meta("blk.weight").expect("meta");
    assert_eq!(
        meta.dtype.storage,
        grim_tensor::dtype::Storage::FloatPack(
            grim_tensor::dtype::FloatPackScheme::Fp8Blocked16
        )
    );

    // The byte count is n*k whether or not the shape tiles evenly -- blocking
    // permutes, it does not compress. An earlier draft of this test asserted
    // `expected_bytes != n*k` on the theory that ragged shapes change the
    // layout; they do not, and asserting it would have pinned a false claim.
    let dt = grim_tensor::DType {
        arith: meta.dtype.arith,
        storage: meta.dtype.storage,
    };
    assert_eq!(dt.expected_bytes(24 * 32), 24 * 32);

    // So the 16x16 constraint cannot live in the file format: the file is
    // well-formed and every byte is there. It lives in the kernel contract,
    // which refuses ragged K -- `whiteraven_dispatch`'s `dispatch_refuses_
    // ragged_k` is where that is pinned. Assert here only that the fixture is
    // really ragged, so that test cannot silently become vacuous.
    assert_eq!(32 % 16, 0, "k stays clean so only N is at fault");
    assert_ne!(24 % 16, 0, "this fixture must actually be N-misaligned");
}

#[test]
fn forestraven_gguf_direct_refuses_pointing_at_grim_while_the_tag_has_meaning() {
    // ForestRaven is the asymmetric case: tag 672 maps to a REAL storage
    // (`Block(Int8PerChannel)` -- the tag has meaning), but GGUF-direct reads
    // cannot serve it (row-scaled framing has no fixed block geometry, so the
    // reader cannot size the payload). The file must refuse at `get` naming
    // the .grim path -- not hand back a truncated prefix, and not claim the
    // tag is unknown.
    //
    // `meta` stays truthful (reports the real storage): the refusal belongs
    // to the byte path, not the type mapping.
    let scratch = Scratch::new();
    let path = scratch.path("forestraven.gguf");
    // Payload content is irrelevant: the refusal happens before any byte is
    // read. Write n*k zero bytes to keep the file well-formed.
    write_gguf(
        &path,
        GgufDType::ForestRaven,
        "w.weight",
        &[64, 32],
        &vec![0u8; 32 * 64],
    );
    let provider = GgufProvider::open(path.to_str().expect("path")).expect("open");
    let meta = provider.meta("w.weight").expect("meta");
    assert_eq!(
        meta.dtype.storage,
        grim_tensor::dtype::Storage::Block(
            grim_tensor::dtype::BlockDtype::Int8PerChannel
        ),
        "meta must report the tag's true meaning"
    );
    let err = provider
        .get("w.weight")
        .err()
        .expect("GGUF-direct get of a row-scaled blob must refuse, not truncate");
    let msg = format!("{err}");
    assert!(
        msg.contains("672") && msg.contains(".grim"),
        "refusal must name the tag and the remedy, got: {msg}"
    );
}

/// The regression that motivated declaring every geometry explicitly.
///
/// `type_size_per_block`'s fallback is `_ => 0`, and the reader derives a
/// tensor's length from it. A grim-native tag missing from that match does not
/// error -- it declares a zero-byte tensor, which loads fine and hands every
/// consumer an empty buffer. The absence is invisible at compile time and at
/// load time, and only shows up as a GPU fault or a garbage matmul much later.
#[test]
fn no_declared_format_sizes_its_tensor_to_zero_bytes() {
    for d in [
        GgufDType::WhiteCrow,
        GgufDType::Raven,
        GgufDType::WhiteRaven,
        GgufDType::GreyRaven,
        GgufDType::ForestRaven,
    ] {
        assert!(
            d.type_size_per_block() > 0,
            "{} sizes its tensor to 0 bytes; the reader will hand consumers an \
             empty buffer instead of an error",
            d.display_name()
        );
        assert!(
            d.block_size() > 0,
            "{} has a zero block_size, which makes the byte count divide by zero \
             in the reader's `params * type_size / block_size` path",
            d.display_name()
        );
    }
}
