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
fn the_other_series_tags_load_as_unsupported_naming_the_format_and_the_remedy() {
    // Convention (matching `prism_unsupported_format_test`): the file opens and
    // the tensor is returned, but its storage is `Unsupported` carrying a reason
    // that names the format and says what to do. Refusing inside the provider
    // would be worse -- it would turn "this format has no loader" into "this
    // file is broken", which is a different bug and sends the user to the wrong
    // place.
    // Raven (669) and WhiteCrow (660) resolved to real loads once their
    // QuantFormats landed; only these two are still deliberately refused.
    for (dtype, tag_name) in [
        (GgufDType::ForestRaven, "forestraven"),
        (GgufDType::GreyRaven, "greyraven"),
    ] {
        let scratch = Scratch::new();
        let path = scratch.path(&format!("{tag_name}.gguf"));
        // Size the payload from the declared geometry, not from n*k: WhiteCrow
        // is framed and occupies fewer bytes than its element count, and a
        // fixture that wrote n*k would leave undeclared slack in the payload
        // region -- a malformed file that still happens to read back the prefix
        // we asked for.
        let elems = 32u64 * 64;
        let nbytes =
            (elems / dtype.block_size() as u64) * dtype.type_size_per_block() as u64;
        write_gguf(
            &path,
            dtype,
            "w.weight",
            &[32, 64],
            &vec![0u8; nbytes as usize],
        );
        let provider = GgufProvider::open(path.to_str().expect("path")).expect("open");
        let meta = provider.meta("w.weight").expect("meta");

        let grim_tensor::dtype::Storage::Unsupported(f) = meta.dtype.storage else {
            panic!(
                "{} must not resolve to a real storage -- a packed kernel payload read \
                 as a same-shaped standard format decodes to plausible garbage",
                dtype.display_name()
            );
        };
        assert_eq!(f.name, dtype.display_name());
        assert_eq!(f.block_size, Some(dtype.block_size() as usize));
        assert_eq!(
            f.bytes_per_block,
            Some(dtype.type_size_per_block() as usize),
            "{} must report the geometry the loader would have to honour",
            dtype.display_name()
        );
        // The message must say WHAT failed and WHAT to do about it.
        for needle in [dtype.display_name(), "loader", "WhiteRaven"] {
            assert!(
                f.reason.contains(needle),
                "{} reason missing {needle:?}: {}",
                dtype.display_name(),
                f.reason
            );
        }
        // And the payload is read at the length the geometry declares -- and
        // that length is nonzero. The reason a format has no loader must never
        // be "it sized to nothing": that is indistinguishable, from the
        // consumer's side, from a file that was truncated on copy.
        //
        // Derived from the geometry rather than hardcoded: WhiteCrow's framed
        // group-128 payload is 16 * 67 = 1072 bytes for 2048 weights, not 2048.
        // Writing a n*k buffer here and asserting `len() == n*k` would have
        // "passed" for every bare-code format while quietly mis-asserting the
        // one format whose geometry differs.
        let expected = (32u64 * 64 / dtype.block_size() as u64) * dtype.type_size_per_block() as u64;
        assert!(expected > 0, "{} geometry declares zero bytes", dtype.display_name());
        let raw = provider.get("w.weight").expect("get");
        assert_eq!(
            raw.bytes.len() as u64,
            expected,
            "{} payload must be read at its declared geometry",
            dtype.display_name()
        );
    }
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
