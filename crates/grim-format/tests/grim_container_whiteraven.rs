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
    // GreyRaven-HW is LOSSY by construction: tile-coupled 2:4 prune, then
    // E4M3 in hardware order. The decoded model must equal the pruned model,
    // not the dense source -- and the coupling (shared patterns across 16
    // rows and K-halves) must hold, or the sidx words cannot select it.
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
            grim_tensor::dtype::BlockDtype::Fp8Sparse24Hw
        ),
        "--format greyraven must tag Block(Fp8Sparse24Hw)"
    );
    assert_eq!(meta.shape, vec![n, k]);

    let raw = gp.get("blk.weight").expect("get");
    // 2 row-tiles x 2 windows x 260 B.
    assert_eq!(raw.bytes.len(), 2 * 2 * 260);

    let deq = grim_quant::grey_raven::dequant_grey_raven_hw(&raw.bytes, n, k)
        .expect("dequant");
    assert_eq!(deq.len(), n * k);
    // Every survivor is the E4M3 round of the SOURCE at its position (the
    // packer encodes, never invents), and pruned cells are exactly +0.0.
    // Implementation-independent: holds regardless of which pattern won.
    for (i, (&w, &g)) in want.iter().zip(&deq).enumerate() {
        if g == 0.0 {
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
                "element {i}: survivor must be the E4M3 round of source, got {g:e} want {q:e}"
            );
        }
    }
    // 2:4 structure: every live K-group keeps exactly 2.
    for r in 0..n {
        for g in 0..k / 4 {
            let nz = (0..4).filter(|s| deq[r * k + 4 * g + s] != 0.0).count();
            // An all-zero source group decodes all-zero (0 survivors, not 2):
            // the packer cannot invent mass. Otherwise exactly 2.
            let src_nz = (0..4).filter(|s| want[r * k + 4 * g + s] != 0.0).count();
            if src_nz == 0 {
                assert_eq!(nz, 0);
            } else {
                assert_eq!(nz, 2, "row {r} group {g} must keep exactly 2");
            }
        }
    }
    // Coupling: groups {p, p+4} of each 32-window share survivor slots
    // (hardware forced sharing). Verify slot sets match within each pair,
    // skipping rows where either group is all-zero in source.
    for r in (0..n).step_by(16) {
        for w0 in 0..k / 32 {
            for p in 0..4 {
                for rr in r..(r + 16).min(n) {
                    let a = (0..4)
                        .filter(|s| deq[rr * k + 4 * (8 * w0 + p) + s] != 0.0)
                        .collect::<Vec<_>>();
                    let b = (0..4)
                        .filter(|s| deq[rr * k + 4 * (8 * w0 + p + 4) + s] != 0.0)
                        .collect::<Vec<_>>();
                    let sa = (0..4).any(|s| want[rr * k + 4 * (8 * w0 + p) + s] != 0.0);
                    let sb = (0..4).any(|s| want[rr * k + 4 * (8 * w0 + p + 4) + s] != 0.0);
                    if sa && sb {
                        assert_eq!(
                            a, b,
                            "tile row {rr} window {w0} pair {p}: groups must share slots"
                        );
                    }
                }
            }
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
            grim_tensor::dtype::BlockDtype::Fp8Sparse24Hw
        ),
        "2D weight must become GreyRaven-HW"
    );
    let n_meta = gp.meta("blk.norm.weight").expect("meta norm");
    assert!(
        !matches!(
            n_meta.dtype.storage,
            grim_tensor::dtype::Storage::Block(
                grim_tensor::dtype::BlockDtype::Fp8Sparse24Hw
            )
        ),
        "1D norm must NOT be tagged sparse -- there is no K axis to group"
    );
}

#[test]
fn convert_with_format_forestraven_produces_row_scaled_int8() {
    // ForestRaven: per-output-row absmax INT8 in the framed blob. The decoded
    // model must equal the per-row int8 round of the source -- dense (no
    // pruning, unlike GreyRaven), but quantized.
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let mut want: Vec<f32> = (0..n * k)
        .map(|i| ((i * 13) % 29) as f32 * 0.0625 - 0.8)
        .collect();
    // Pin the zero-row edge through the full convert path: an all-zero row
    // must get scale 1.0 and decode back to +0.0, not NaN.
    for c in 0..k {
        want[5 * k + c] = 0.0;
    }
    let payload: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();

    let src = scratch.0.join("f32_forest.gguf");
    write_gguf(&src, GgufDType::F32, "blk.weight", &[k as u64, n as u64], &payload);

    let out = scratch.0.join("forest.grim");
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
        Some("forestraven".to_string()),
        None,
        None,
    )
    .expect("convert --format forestraven");

    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open grim");
    let meta = gp.meta("blk.weight").expect("meta");
    assert_eq!(
        meta.dtype.storage,
        grim_tensor::dtype::Storage::Block(
            grim_tensor::dtype::BlockDtype::Int8PerChannel
        ),
        "--format forestraven must tag Block(Int8PerChannel)"
    );
    assert_eq!(meta.shape, vec![n, k]);

    let raw = gp.get("blk.weight").expect("get");
    // Framed blob: 8 + n*k + 8 + 4n.
    assert_eq!(raw.bytes.len(), 16 + n * k + 4 * n);
    let qw_len = u64::from_le_bytes(raw.bytes[0..8].try_into().unwrap()) as usize;
    assert_eq!(qw_len, n * k, "codes segment is one byte per weight");
    let sc_len =
        u64::from_le_bytes(raw.bytes[8 + qw_len..16 + qw_len].try_into().unwrap()) as usize;
    assert_eq!(sc_len, 4 * n, "scales segment is one fp32 per row");

    // Per-row scales are absmax/127; the zero row gets exactly 1.0.
    for r in 0..n {
        let s = f32::from_le_bytes(
            raw.bytes[16 + qw_len + r * 4..20 + qw_len + r * 4]
                .try_into()
                .unwrap(),
        );
        let amax = want[r * k..(r + 1) * k]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let expect = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        assert_eq!(
            s.to_bits(),
            expect.to_bits(),
            "row {r} scale must be absmax/127"
        );
    }

    // Decode equals the per-row int8 round, bit-exact.
    let deq = grim_quant::dequant_forest(&raw.bytes, n, k).expect("dequant");
    assert_eq!(deq.len(), n * k);
    for r in 0..n {
        let amax = want[r * k..(r + 1) * k]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        for c in 0..k {
            let q = ((want[r * k + c] / s).round().clamp(-128.0, 127.0) as i8) as f32 * s;
            // +0.0 and -0.0 compare equal but have different bits; the zero
            // row must decode to exactly +0.0 (0.0 * 1.0), never -0.0.
            if want[r * k + c] == 0.0 && amax == 0.0 {
                assert_eq!(
                    deq[r * k + c].to_bits(),
                    0.0f32.to_bits(),
                    "zero-row element must be exactly +0.0"
                );
            } else {
                assert_eq!(
                    deq[r * k + c].to_bits(),
                    q.to_bits(),
                    "element [{r},{c}] must be the int8 round"
                );
            }
        }
    }
}

#[test]
fn convert_with_format_forestraven_leaves_1d_tensors_untouched() {
    // 1D norms stay high-precision: the future dot4 kernel wants 2D weights,
    // and quantizing gains to int8 buys nothing.
    let scratch = Scratch::new();
    let (n, k) = (32usize, 64usize);
    let weights: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.001).collect();
    let norms: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.01).collect();

    let src = scratch.0.join("mixed_forest.gguf");
    write_mixed(&src, &weights, &norms, n, k);

    let out = scratch.0.join("mixed_forest.grim");
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
        Some("forestraven".to_string()),
        None,
        None,
    )
    .expect("convert");

    let gp = GrimProvider::open(out.to_str().unwrap()).expect("open");
    let w_meta = gp.meta("blk.weight").expect("meta weight");
    assert_eq!(
        w_meta.dtype.storage,
        grim_tensor::dtype::Storage::Block(
            grim_tensor::dtype::BlockDtype::Int8PerChannel
        ),
        "2D weight must become ForestRaven"
    );
    let n_meta = gp.meta("blk.norm.weight").expect("meta norm");
    assert!(
        !matches!(
            n_meta.dtype.storage,
            grim_tensor::dtype::Storage::Block(
                grim_tensor::dtype::BlockDtype::Int8PerChannel
            )
        ),
        "1D norm must NOT be tagged ForestRaven"
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
