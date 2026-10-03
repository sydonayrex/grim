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

#[test]
fn convert_with_format_whiteraven_produces_a_blocked_fp8_grim_file() {
    // Same fixture, but the SOURCE is a plain F32 checkpoint and WhiteRaven
    // is requested as the conversion target. This is the path `grim convert
    // --format whiteraven` drives; the weights have to come out quantized
    // AND blocked, and the override must be written.
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let want: Vec<f32> = (0..n * k)
        .map(|i| ((i * 37) % 61) as f32 * 0.03125 - 0.9)
        .collect();
    let payload: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();

    let src = scratch.0.join("f32.gguf");
    write_gguf(&src, GgufDType::F32, "blk.weight", &[k as u64, n as u64], &payload);

    let out = scratch.0.join("wr.grim");
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
        Some("whiteraven".to_string()),
        None,
        None,
    )
    .expect("convert --format whiteraven");

    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open grim");
    let meta = gp.meta("blk.weight").expect("meta");
    assert_eq!(
        meta.dtype.storage,
        grim_tensor::dtype::Storage::FloatPack(
            grim_tensor::dtype::FloatPackScheme::Fp8Blocked16
        ),
        "a --format whiteraven conversion must tag the tensor WhiteRaven"
    );

    let raw = gp.get("blk.weight").expect("get");
    assert_eq!(raw.bytes.len(), n * k, "blocked FP8 is one byte per weight");

    // The dequantized values must equal the ORIGINAL f32 weights rounded
    // through E4M3 -- not a requantized-through-F32 artifact.
    let deq = grim_quant::dequant_fp8_blocked16(&raw.bytes, n, k).expect("dequant");
    for (i, (&w, &g)) in want.iter().zip(&deq).enumerate() {
        let q = grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(w));
        assert_eq!(g.to_bits(), q.to_bits(), "element {i} should be the E4M3 round of the F32 weight");
    }
}

#[test]
fn convert_with_format_raven_produces_a_bare_fp8_payload() {
    // Raven is dense FP8 in one row-major pass: a 32x64 F32 checkpoint in,
    // n*k E4M3 codes out, tagged 669 with tag-derived bare-code storage.
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let want: Vec<f32> = (0..n * k)
        .map(|i| ((i * 17) % 53) as f32 * 0.0625 - 1.5)
        .collect();
    let payload: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();

    let src = scratch.0.join("f32_raven.gguf");
    write_gguf(&src, GgufDType::F32, "blk.weight", &[k as u64, n as u64], &payload);

    let out = scratch.0.join("raven.grim");
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
        Some("raven".to_string()),
        None,
        None,
    )
    .expect("convert --format raven");

    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open grim");
    let meta = gp.meta("blk.weight").expect("meta");
    assert_eq!(
        meta.dtype.storage,
        grim_tensor::dtype::Storage::FloatPack(
            grim_tensor::dtype::FloatPackScheme::Fp8
        ),
        "--format raven must tag FloatPack(Fp8)"
    );

    let raw = gp.get("blk.weight").expect("get");
    assert_eq!(raw.bytes.len(), n * k, "Raven payload is bare codes");

    // NOTE the decode convention: the ROCm Fp8 path (dot4 GEMV / MFMA) loads
    // the payload as bare codes one-per-weight, which is also what
    // `expected_bytes` says the layout is. `grim_quant::dequant_fp8` is the
    // OTHER convention (4-byte per-tensor scale prefix) used by the legacy
    // `quant_fp8` producer -- it cannot be asked to decode a bare payload
    // because it folds the first four codes into the scale. Decode per-code:
    for (i, (&w, &g)) in want.iter().zip(raw.bytes.iter()).enumerate() {
        let got = grim_quant::fp8_e4m3_to_f32(g);
        let q = grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(w));
        assert_eq!(got.to_bits(), q.to_bits(), "element {i} diverged");
    }
}

#[test]
fn convert_with_format_whitecrow_produces_the_ostquant_blob() {
    // WhiteCrow: W4A4 OSTQuant group-128. Needs K%128==0 -- 32x128 here.
    let scratch = Scratch::new();
    let (n, k) = (32usize, 128usize);
    let want: Vec<f32> = (0..n * k)
        .map(|i| ((i * 11) % 41) as f32 * 0.125 - 2.5)
        .collect();
    let payload: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();

    let src = scratch.0.join("f32_crow.gguf");
    write_gguf(&src, GgufDType::F32, "blk.weight", &[k as u64, n as u64], &payload);

    let out = scratch.0.join("crow.grim");
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
        Some("whitecrow".to_string()),
        None,
        None,
    )
    .expect("convert --format whitecrow");

    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open grim");
    let meta = gp.meta("blk.weight").expect("meta");
    assert!(
        matches!(
            meta.dtype.storage,
            grim_tensor::dtype::Storage::W4A4OstQuant(_)
        ),
        "--format whitecrow must tag W4A4OstQuant, got {:?}",
        meta.dtype.storage
    );

    let raw = gp.get("blk.weight").expect("get");
    // Framed triple stream: [u64 qw_len][qw][u64 sc_len][sc][u64 zr_len][zr].
    // 32x128 at group-128 with 2-bit packed scale + 1-bit packed zero.
    let qw_len = u64::from_le_bytes(raw.bytes[0..8].try_into().unwrap()) as usize;
    let expected_qw = n * (k / 8) * 4;
    assert_eq!(qw_len, expected_qw, "qw segment at the expected length");
    let sc_len = u64::from_le_bytes(raw.bytes[8 + qw_len..16 + qw_len].try_into().unwrap()) as usize;
    let expected_sc = n * (k / 128) * 2;
    assert_eq!(sc_len, expected_sc, "sc segment at the expected length");
    let zr_len =
        u64::from_le_bytes(raw.bytes[16 + qw_len + sc_len..24 + qw_len + sc_len].try_into().unwrap())
            as usize;
    let expected_zr = n * (k / 128);
    assert_eq!(zr_len, expected_zr, "zr segment at the expected length");
    assert_eq!(raw.bytes.len(), 24 + qw_len + sc_len + zr_len);
}

#[test]
fn unknown_format_fails_before_any_packing() {
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let weights: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.001).collect();
    let src = scratch.0.join("f32_any.gguf");
    let payload: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_gguf(&src, GgufDType::F32, "blk.weight", &[k as u64, n as u64], &payload);

    let out = scratch.0.join("any.grim");
    let err = grim_format::convert_to_grim(
        src.to_str().unwrap(),
        out.to_str().unwrap(),
        "gfx1200",
        8.0,
        0,
        None,
        None,
        None,
        None,
        Some("typo_of_whitecrow".to_string()),
        None,
        None,
    )
    .expect_err("an unknown format name must fail at the boundary");
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown"),
        "error must name the problem, got: {msg}"
    );
    assert!(
        !out.exists(),
        "an early --format failure must not leave a partial file behind"
    );
}

#[test]
fn convert_with_format_greyraven_prunes_packs_and_loads() {
    // GreyRaven is LOSSY by construction: 2:4 magnitude prune, then E4M3.
    // The decoded model must equal the pruned model, not the dense source.
    // Asserting dense equality here would pin a false claim; asserting nothing
    // about the values would let a dense passthrough slip by.
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let want: Vec<f32> = (0..n * k)
        .map(|i| ((i * 37) % 61) as f32 * 0.03125 - 0.9)
        .collect();
    let payload: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();

    let src = scratch.0.join("f32_grey.gguf");
    write_gguf(&src, GgufDType::F32, "blk.weight", &[k as u64, n as u64], &payload);

    let out = scratch.0.join("grey.grim");
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
        Some("greyraven".to_string()),
        None,
        None,
    )
    .expect("convert --format greyraven");

    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open grim");
    let meta = gp.meta("blk.weight").expect("meta");
    assert_eq!(
        meta.dtype.storage,
        grim_tensor::dtype::Storage::Block(
            grim_tensor::dtype::BlockDtype::Fp8Sparse24
        ),
        "--format greyraven must tag Block(Fp8Sparse24)"
    );
    assert_eq!(meta.shape, vec![n, k]);

    let raw = gp.get("blk.weight").expect("get");
    let expected_len = grim_quant::grey_raven::packed_bytes_for(n * k);
    assert_eq!(
        raw.bytes.len(),
        expected_len,
        "payload must match the 4.75bpw geometry, not n*k"
    );
    assert!(
        expected_len < n * k,
        "a 2:4 payload that is not smaller than dense is not pruned"
    );

    // Host reference: magnitude prune (top-2 |v| per consecutive group of 4,
    // ties by ascending slot -- the same rule `sparsify_2_4_flat` uses), then
    // E4M3 round the survivors. Flat consecutive groups coincide with K-groups
    // for row-major [n, k] with k % 4 == 0.
    let mut pruned = want.clone();
    for g in 0..(n * k / 4) {
        let base = g * 4;
        let mut order = [0usize, 1, 2, 3];
        order.sort_by(|&a, &b| {
            want[base + b]
                .abs()
                .partial_cmp(&want[base + a].abs())
                .unwrap()
                .then_with(|| a.cmp(&b))
        });
        pruned[base + order[2]] = 0.0;
        pruned[base + order[3]] = 0.0;
    }
    let zeros = pruned.iter().filter(|&&v| v == 0.0).count();
    assert_eq!(
        zeros,
        n * k / 2,
        "every group of 4 must lose exactly 2 weights"
    );

    let deq = grim_quant::grey_raven::dequant_grey_raven(&raw.bytes, n * k)
        .expect("dequant");
    assert_eq!(deq.len(), n * k);
    for (i, (&w, &g)) in pruned.iter().zip(&deq).enumerate() {
        if w == 0.0 {
            assert_eq!(
                g.to_bits(),
                0.0f32.to_bits(),
                "element {i}: pruned weights must decode to exactly +0.0"
            );
        } else {
            let q = grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(w));
            assert_eq!(
                g.to_bits(),
                q.to_bits(),
                "element {i}: survivor must be the E4M3 round, got {g:e} want {q:e}"
            );
        }
    }
}

#[test]
fn convert_with_format_greyraven_leaves_1d_tensors_untouched() {
    // 1D norms carry no K axis to group along -- they must take the uniform
    // path, not fail and not get tagged sparse.
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let weights: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.001).collect();
    let norms: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.01).collect();

    let src = scratch.0.join("mixed_grey.gguf");
    write_mixed(&src, &weights, &norms, n, k);

    let out = scratch.0.join("mixed_grey.grim");
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
        Some("greyraven".to_string()),
        None,
        None,
    )
    .expect("convert");

    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open");
    let w_meta = gp.meta("blk.weight").expect("meta weight");
    assert_eq!(
        w_meta.dtype.storage,
        grim_tensor::dtype::Storage::Block(
            grim_tensor::dtype::BlockDtype::Fp8Sparse24
        ),
        "2D K-aligned weight must become GreyRaven"
    );
    let n_meta = gp.meta("blk.norm.weight").expect("meta norm");
    assert!(
        !matches!(
            n_meta.dtype.storage,
            grim_tensor::dtype::Storage::Block(
                grim_tensor::dtype::BlockDtype::Fp8Sparse24
            )
        ),
        "1D norm must NOT be tagged sparse -- there is no K axis to group"
    );
}

#[test]
fn convert_with_format_whiteraven_leaves_non_conforming_tensors_untouched() {
    // 1D tensors (norm gains) carry no 2D shape to block -- the converter
    // must pack them with the uniform path rather than fail or silently
    // mis-shape them. The two records in one file exercise both arms.
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let weights: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.001).collect();
    let norms: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.01).collect();

    // Each tensor needs its OWN payload offset. Writing 0 for both is not a
    // caught error: both tensors then read from the start of the payload
    // region, the second tensor's "weights" are the first tensor's, and the
    // only symptom is a plausible, wrong tensor on the far side.
    let src = scratch.0.join("mixed.gguf");
    write_mixed(&src, &weights, &norms, n, k);

    let out = scratch.0.join("mixed.grim");
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
        Some("whiteraven".to_string()),
        None,
        None,
    )
    .expect("convert");

    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open");
    let w_meta = gp.meta("blk.weight").expect("meta weight");
    assert_eq!(
        w_meta.dtype.storage,
        grim_tensor::dtype::Storage::FloatPack(
            grim_tensor::dtype::FloatPackScheme::Fp8Blocked16
        ),
        "2D 16-aligned weight must become WhiteRaven"
    );
    let n_meta = gp.meta("blk.norm.weight").expect("meta norm");
    assert!(
        !matches!(
            n_meta.dtype.storage,
            grim_tensor::dtype::Storage::FloatPack(
                grim_tensor::dtype::FloatPackScheme::Fp8Blocked16
            )
        ),
        "1D norm must NOT be tagged WhiteRaven -- it cannot hold the layout"
    );
}

/// A two-tensor GGUF with correct per-tensor payload offsets.
fn write_mixed(path: &std::path::Path, weights: &[f32], norms: &[f32], n: usize, k: usize) {
    let wbytes: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    let nbytes: Vec<u8> = norms.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    push_string(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "qwen4exp");
    push_string(&mut buf, "general.name");
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut buf, "mixed");

    let tensor_section = |buf: &mut Vec<u8>, name: &str, dims: &[u64], dtype: GgufDType, offset: u64| {
        push_string(buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&dtype.tag().to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    };
    tensor_section(&mut buf, "blk.weight", &[k as u64, n as u64], GgufDType::F32, 0);
    tensor_section(&mut buf, "blk.norm.weight", &[n as u64], GgufDType::F32, wbytes.len() as u64);

    let aligned = (buf.len() + 31) / 32 * 32;
    buf.resize(aligned, 0);
    buf.extend_from_slice(&wbytes);
    buf.extend_from_slice(&nbytes);

    let mut f = std::fs::File::create(path).expect("create");
    f.write_all(&buf).expect("write");
}
