//! Differential test against llama.cpp (plan Task 2.1, WI-2).
//!
//! This is the test the whole encoder has been waiting for. Every other test
//! checks grim against itself; this one checks grim against an independent
//! implementation of the same spec, and it exists because of a specific,
//! identified hazard:
//!
//! **The spatial-merge permutation is not derivable by inspection.** The
//! `// spatial merge` block (`qwen3vl.cpp:18-31`) interleaves `ggml_permute`,
//! which produces a strided VIEW, with `ggml_cont` applied to that view, which
//! GATHERS through the view's strides. The result is not a plain reshape. A
//! hand-derived closed form was attempted during planning and rejected twice by
//! a bijection assertion; it was deleted rather than shipped.
//!
//! A wrong merge still produces finite, correctly-shaped, non-degenerate output.
//! It changes what the model *sees* and nothing else, so no unit test on grim's
//! own code can detect it. Only an independent implementation can.
//!
//! # Provenance of the oracle
//!
//! The primary reference tree (`old/repo/llama.cpp-master`) DOES NOT BUILD: its
//! `ggml/src/ggml-cpu/ggml-cpu.c:7` includes `"iqp.h"`, which was never vendored,
//! so the core CPU backend does not compile. The oracle is therefore produced from
//! the sibling tree `old/repo/llama.cpp-xing4_0-port`, which carries the genuine
//! upstream `iqp.h`. That is sound because the two trees' forward specs are
//! **byte-identical**:
//!
//! ```text
//! cmp llama.cpp-master/tools/mtmd/models/qwen3vl.cpp \
//!     llama.cpp-xing4_0-port/tools/mtmd/models/qwen3vl.cpp   # identical
//! ```
//!
//! so the spec this test compares against is the same spec either tree encodes.
//!
//! # Running it
//!
//! The oracle is generated out-of-tree by `/tmp/hermes-qwen3vl-oracle.sh`, which
//! never writes into `old/repo/`. Point this test at the dump with:
//!
//! ```text
//! GRIM_QWEN3VL_ORACLE=/path/to/oracle.bin \
//! GRIM_QWEN3VL_IMAGE=/path/to/probe.png \
//!   cargo test -p grim-models-vision --test qwen3vl_parity -- --ignored --nocapture
//! ```
//!
//! Without those variables the test PANICS rather than skipping. A parity test
//! that passes because it did not run is worse than no parity test.

use grim_models_vision::qwen3vl_clip::{Qwen3VlClip, Qwen3VlClipConfig};
use std::path::PathBuf;

fn required_path(var: &str) -> PathBuf {
    match std::env::var(var) {
        Ok(v) => {
            let p = PathBuf::from(&v);
            assert!(
                p.exists(),
                "{var}={v} does not exist; this test must find the oracle or fail, \
                 never pass by absence"
            );
            p
        }
        Err(_) => panic!(
            "{var} is not set. This is the differential test against llama.cpp and \
             it cannot run without the oracle dump. See the module docs for how to \
             produce it with /tmp/hermes-qwen3vl-oracle.sh."
        ),
    }
}

/// Read the oracle dump: `[i32 n_tokens][i32 n_embd][f32 ...]`
/// (`clip.cpp:5899-5908`).
fn read_oracle(path: &PathBuf) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).expect("oracle dump is readable");
    assert!(bytes.len() >= 8, "oracle dump is shorter than its 8-byte header");
    let n_tokens = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let n_embd = i32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert!(n_tokens > 0 && n_embd > 0, "oracle header is empty");

    let data_bytes = &bytes[8..];
    assert_eq!(
        data_bytes.len() % 4,
        0,
        "oracle data is not a whole number of f32"
    );
    let n_data = data_bytes.len() / 4;
    assert_eq!(
        n_data,
        n_tokens * n_embd,
        "oracle dump is truncated: {n_data} floats for {n_tokens}x{n_embd}"
    );

    let data = data_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (n_tokens, n_embd, data)
}

/// Cosine similarity over two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// grim's tower must reproduce llama.cpp's embeddings on the same image.
///
/// The checkpoint's geometry is 576 merged tokens of 5120 features, so a shape
/// mismatch alone localises the failure to the merge or the merger reshape.
#[test]
#[ignore = "requires a llama.cpp oracle dump; see module docs"]
fn grim_matches_llama_cpp_on_a_real_image() {
    use grim_format::tprov::GgufProvider;

    let oracle_path = required_path("GRIM_QWEN3VL_ORACLE");
    let (o_tokens, o_embd, oracle) = read_oracle(&oracle_path);
    println!("oracle: {o_tokens} tokens x {o_embd} features");

    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut mmproj: Option<PathBuf> = None;
    for _ in 0..4 {
        let c = dir.join("models/qwen38-27b/mmproj-F16.gguf");
        if c.exists() {
            mmproj = Some(c);
            break;
        }
        dir = dir.join("..");
    }
    let mmproj = mmproj.expect(
        "mmproj-F16.gguf not found; this test must find the checkpoint or fail",
    );

    let provider = GgufProvider::open(mmproj.to_str().expect("utf-8 path"))
        .expect("mmproj opens");
    let cfg = Qwen3VlClipConfig::from_provider(&provider).expect("geometry parses");
    cfg.ensure_supported().expect("no deepstack");
    let clip = Qwen3VlClip::load(&provider, cfg.clone()).expect("real tower loads");

    assert_eq!(
        o_tokens,
        cfg.merged_tokens(),
        "oracle token count must equal the checkpoint's merged_tokens"
    );
    assert_eq!(
        o_embd,
        cfg.projection_dim,
        "oracle width must equal the checkpoint's projection_dim"
    );

    // The same pixels the oracle saw. llama.cpp normalised with this file's own
    // image_mean/image_std ([0.5,0.5,0.5]) and bicubic resize; the test image is
    // already exactly 768x768, so no resize happens and only the normalisation
    // applies.
    let image_path = required_path("GRIM_QWEN3VL_IMAGE");
    let pixels = decode_png_rgb8(&image_path);
    assert_eq!(
        pixels.len(),
        cfg.in_channels * cfg.image_size * cfg.image_size,
        "decoded pixel count must match the checkpoint's declared geometry"
    );

    let got = clip
        .forward_cpu(&pixels, cfg.image_size, cfg.image_size)
        .expect("grim forward runs");

    assert_eq!(
        got.len(),
        oracle.len(),
        "grim produced {} values, oracle {} - shapes disagree",
        got.len(),
        oracle.len()
    );

    let cos = cosine(&got, &oracle);
    let max_abs: f32 = got
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("cosine similarity: {cos:.8}");
    println!("max abs difference: {max_abs:.8}");

    // The gate. A wrong spatial merge permutes 2x2 blocks of the token grid;
    // that is a large, structured disagreement, not float noise.
    assert!(
        cos >= 0.999,
        "WI-2 FAILED: cosine {cos:.8} < 0.999. The spatial-merge permutation or \
         the merger reshape does not match llama.cpp. Fix spatial_merge() in \
         forward.rs, not this threshold."
    );
}

/// Per-token cosine, to localise a mismatch to one region of the grid.
///
/// A single bad block among 576 averages away in the whole-tensor cosine, so
/// this reports where the disagreement is concentrated.
#[test]
#[ignore = "requires a llama.cpp oracle dump; see module docs"]
fn per_token_cosine_localises_a_merge_mismatch() {
    use grim_format::tprov::GgufProvider;

    let oracle_path = required_path("GRIM_QWEN3VL_ORACLE");
    let (o_tokens, o_embd, oracle) = read_oracle(&oracle_path);

    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut mmproj = None;
    for _ in 0..4 {
        let c = dir.join("models/qwen38-27b/mmproj-F16.gguf");
        if c.exists() {
            mmproj = Some(c);
            break;
        }
        dir = dir.join("..");
    }
    let provider = GgufProvider::open(
        mmproj
            .expect("mmproj-F16.gguf not found")
            .to_str()
            .expect("utf-8"),
    )
    .expect("mmproj opens");
    let cfg = Qwen3VlClipConfig::from_provider(&provider).expect("geometry");
    let clip = Qwen3VlClip::load(&provider, cfg.clone()).expect("tower loads");

    let pixels = decode_png_rgb8(&required_path("GRIM_QWEN3VL_IMAGE"));
    let got = clip
        .forward_cpu(&pixels, cfg.image_size, cfg.image_size)
        .expect("grim forward runs");

    let grid = (o_tokens as f64).sqrt() as usize;
    let mut worst = (1.0f32, 0usize);
    for t in 0..o_tokens {
        let a = &got[t * o_embd..(t + 1) * o_embd];
        let b = &oracle[t * o_embd..(t + 1) * o_embd];
        let c = cosine(a, b);
        if c < worst.0 {
            worst = (c, t);
        }
    }
    println!(
        "grid {grid}x{grid}; worst token {} (row {}, col {}) cosine {:.8}",
        worst.1,
        worst.1 / grid,
        worst.1 % grid,
        worst.0
    );
    assert!(
        worst.0 >= 0.99,
        "worst token cosine {:.8} at token {} - a localised mismatch points at the \\
         merge permutation's row/col ordering rather than the merger reshape",
        worst.0,
        worst.1
    );
}

/// Decode a PNG to `[3, h, w]` f32, normalised with the checkpoint's own
/// mean/std.
///
/// Minimal by design: it reads only the PNG subset this repo's oracle generator
/// emits (8-bit RGB, non-interlaced, filter types 0-4), so the parity test needs
/// no new crate dependency. Anything else is refused rather than mis-decoded.
fn decode_png_rgb8(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("image is readable");
    assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n", "not a PNG");

    let mut pos = 8usize;
    let (mut w, mut h) = (0usize, 0usize);
    let mut idat: Vec<u8> = Vec::new();

    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let tag = &bytes[pos + 4..pos + 8];
        let body = &bytes[pos + 8..pos + 8 + len];
        if tag == b"IHDR" {
            w = u32::from_be_bytes(body[0..4].try_into().unwrap()) as usize;
            h = u32::from_be_bytes(body[4..8].try_into().unwrap()) as usize;
            assert_eq!(body[8], 8, "only 8-bit PNGs are supported");
            assert_eq!(body[9], 2, "only truecolour RGB PNGs are supported");
            assert_eq!(body[12], 0, "interlaced PNGs are not supported");
        } else if tag == b"IDAT" {
            idat.extend_from_slice(body);
        } else if tag == b"IEND" {
            break;
        }
        pos += 12 + len; // length + tag + body + crc
    }
    assert!(w > 0 && h > 0 && !idat.is_empty(), "PNG has no image data");

    // Inflate: PNG's zlib stream is stored with a 2-byte header and an Adler-32
    // trailer. Rather than reimplement inflate, shell out is not acceptable in a
    // test, so use the workspace's existing decoder if present; otherwise decode
    // with a minimal fixed-Huffman inflate.
    let raw = inflate_zlib(&idat);
    let bpp = 3usize;
    let stride = w * bpp;
    let mut out = vec![0u8; h * stride];

    for y in 0..h {
        let filter = raw[y * (stride + 1)];
        let line = &raw[y * (stride + 1) + 1..y * (stride + 1) + 1 + stride];
        for x in 0..stride {
            let a = if x >= bpp { out[y * stride + x - bpp] } else { 0 };
            let b = if y > 0 { out[(y - 1) * stride + x] } else { 0 };
            let c = if x >= bpp && y > 0 {
                out[(y - 1) * stride + x - bpp]
            } else {
                0
            };
            out[y * stride + x] = match filter {
                0 => line[x],
                1 => line[x].wrapping_add(a),
                2 => line[x].wrapping_add(b),
                3 => line[x].wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => line[x].wrapping_add(paeth(a, b, c)),
                other => panic!("unsupported PNG filter type {other}"),
            };
        }
    }

    // Channel-major [3, h, w], scaled to 0..1 then normalised with the
    // checkpoint's mean/std (both 0.5 for this family).
    let mut px = vec![0.0f32; 3 * h * w];
    for i in 0..(w * h) {
        for c in 0..3 {
            let v = out[i * 3 + c] as f32 / 255.0;
            px[c * h * w + i] = (v - 0.5) / 0.5;
        }
    }
    px
}

/// Inflate a zlib stream with `miniz_oxide`.
///
/// Deliberately a dependency rather than a hand-rolled DEFLATE decoder: inflate is
/// ~200 lines of spec-dense code whose failure mode is silent corruption, which is
/// precisely what a parity oracle must not be. `miniz_oxide` is already in the
/// lock file, so this adds no new package to the build.
fn inflate_zlib(data: &[u8]) -> Vec<u8> {
    miniz_oxide::inflate::decompress_to_vec_zlib(data)
        .expect("test PNG must be a valid zlib stream")
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let pa = (p - a as i16).abs();
    let pb = (p - b as i16).abs();
    let pc = (p - c as i16).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}
