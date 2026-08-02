//! Mutation-resistant golden tests for the GGUF loader's size/offset/dtype
//! math — the silent-corruption gates the in-crate tests skip.
//!
//! The existing `test_read_gguf_binary_parsing_and_metadata` parses a single
//! F32 tensor and asserts only `name`/`dims`/`offset`. It never checks:
//!   - `size_bytes` for any K-quant (a wrong `type_size_per_block` constant
//!     silently mis-sizes every tensor read → `UnexpectedEof` or garbage
//!     weights far downstream).
//!   - `data_start` 32-byte alignment (an off-by-one misaligns every tensor's
//!     byte offset).
//!   - the `I16/I32/I64 → DType::F32` integer-to-float *promotion* in
//!     `map_gguf_dtype_to_storage` (silent integer-as-float reinterpretation).
//!
//! These tests build a minimal GGUF byte stream **by hand** (mirroring the
//! on-disk layout, not the library's own writer) and assert exact expected
//! `size_bytes`/`data_start` values derived independently from the ggml
//! `type_size`/`type_traits` tables.

use grim_format::gguf::{
    GGUF_MAGIC, GGUF_VERSION, GgufDType, map_gguf_dtype_to_grim, map_gguf_dtype_to_storage,
    read_gguf,
};
use grim_tensor::{ArithType, DType, Storage, dtype::KQuantScheme};
use std::io::{Cursor, Read, Seek, SeekFrom};

// ---- GGUF value type tags (from read_gguf_value_with_tag) — kept for
// reference; this test currently builds only tensor-info records. ----

/// Append a GGUF string: u64 length + UTF-8 bytes.
fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Append a tensor-info record: name, n_dims, dims[..], dtype tag, offset.
fn push_tensor(buf: &mut Vec<u8>, name: &str, dims: &[u64], dtype_tag: u32, offset: u64) {
    push_string(buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&dtype_tag.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// Build a minimal GGUF v3 stream with no metadata and `tensors` tensor infos.
fn build_gguf(
    tensors: &[(
        /*name*/ &str,
        /*dims*/ &[u64],
        /*dtype*/ u32,
        /*offset*/ u64,
    )],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count = 0
    for &(name, dims, dtype, offset) in tensors {
        push_tensor(&mut buf, name, dims, dtype, offset);
    }
    buf
}

#[test]
fn gguf_size_bytes_f32_and_f16_match_exact_byte_count() {
    // F32: block_size=1, type_size=4.  16 params → 64 bytes.
    // F16: block_size=1, type_size=2.  16 params → 32 bytes.
    let buf = build_gguf(&[
        ("a", &[16], GgufDType::F32 as u32, 0),
        ("b", &[16], GgufDType::F16 as u32, 64),
    ]);
    let g = read_gguf(Cursor::new(buf)).expect("read gguf");
    assert_eq!(g.tensors.len(), 2);
    assert_eq!(g.tensors[0].size_bytes, 16 * 4, "F32 16 params size");
    assert_eq!(g.tensors[1].size_bytes, 16 * 2, "F16 16 params size");
}

#[test]
fn gguf_size_bytes_q8_0_uses_34_byte_blocks() {
    // Q8_0: block_size=32, type_size_per_block=2(scale f16)+32(i8)=34.
    //   32 params → 1 block → 34 bytes.
    //   64 params → 2 blocks → 68 bytes.
    //   33 params → ceil via integer floor? The formula is (params*ts)/bs:
    //     (33*34)/32 = 1122/32 = 35 (floor). We assert the FORMULA result
    //     so a mutant that flips `2+32`→`2+16` is caught regardless.
    let buf = build_gguf(&[
        ("q8.32", &[32], GgufDType::Q8_0 as u32, 0),
        ("q8.64", &[64], GgufDType::Q8_0 as u32, 34),
        ("q8.33", &[33], GgufDType::Q8_0 as u32, 68),
    ]);
    let g = read_gguf(Cursor::new(buf)).expect("read gguf");
    assert_eq!(g.tensors[0].size_bytes, 34, "Q8_0 32 params = 1 block");
    assert_eq!(g.tensors[1].size_bytes, 68, "Q8_0 64 params = 2 blocks");
    assert_eq!(
        g.tensors[2].size_bytes,
        (33 * 34) / 32,
        "Q8_0 33 params = formula"
    );
}

#[test]
fn gguf_size_bytes_k_quant_superblock_constants_are_exact() {
    // K-quants use block_size=256. The per-block byte constants are the
    // critical silent-corruption gate: a wrong constant mis-sizes every
    // K-quant tensor. Values from ggml type_size table:
    //   Q2_K=84, Q3_K=108, Q4_K=144, Q5_K=176, Q6_K=210, Q8_K=252.
    // Each at 256 params (one super-block) must equal the per-block constant.
    let cases: &[(GgufDType, u64)] = &[
        (GgufDType::Q2K, 84),
        (GgufDType::Q3K, 108),
        (GgufDType::Q4K, 144),
        (GgufDType::Q5K, 176),
        (GgufDType::Q6K, 210),
        (GgufDType::Q8K, 252),
    ];
    // Fixed tensor-info table; offsets are cumulative sizes (not validated by
    // the size_bytes assertion, only used so the stream is well-formed).
    let t: &[(&str, &[u64], u32, u64)] = &[
        ("q2k", &[256], GgufDType::Q2K as u32, 0),
        ("q3k", &[256], GgufDType::Q3K as u32, 84),
        ("q4k", &[256], GgufDType::Q4K as u32, 192),
        ("q5k", &[256], GgufDType::Q5K as u32, 336),
        ("q6k", &[256], GgufDType::Q6K as u32, 512),
        ("q8k", &[256], GgufDType::Q8K as u32, 722),
    ];
    let buf = build_gguf(t);
    let g = read_gguf(Cursor::new(buf)).expect("read gguf");
    for (i, &(dt, want_ts)) in cases.iter().enumerate() {
        assert_eq!(
            g.tensors[i].size_bytes, want_ts,
            "{:?} 256 params (one super-block) byte size",
            dt,
        );
    }
    // Multi-block scaling: 512 params of Q6_K = 2 super-blocks = 2*210 = 420.
    // A mutant dividing by the wrong block size (e.g. 32) would give
    // (512*210)/32 = 3360, way off.
    let buf2 = build_gguf(&[("q6k.512", &[512], GgufDType::Q6K as u32, 0)]);
    let g2 = read_gguf(Cursor::new(buf2)).expect("read gguf 2");
    assert_eq!(
        g2.tensors[0].size_bytes, 420,
        "Q6_K 512 params = 2 super-blocks"
    );
}

#[test]
fn gguf_data_start_is_32_byte_aligned_after_tensor_infos() {
    // data_start = (pos_after_tensor_infos + 31) & !31.  We build a stream
    // whose tensor-info section ends at a NON-32-aligned position and verify
    // data_start rounds UP to the next 32-boundary (not down, not unchanged).
    let buf = build_gguf(&[("w", &[16], GgufDType::F32 as u32, 0)]);
    // The raw tensor-info section length here is some non-multiple-of-32;
    // data_start must be the next multiple of 32 ≥ that length.
    let info_end = buf.len() as u64; // reader position after consuming all infos
    let expected_data_start = (info_end + 31) & !31;
    assert!(
        expected_data_start != info_end,
        "test setup error: tensor-info end is already 32-aligned ({info_end}); \
         pick a layout that is not",
    );
    let g = read_gguf(Cursor::new(buf)).expect("read gguf");
    assert_eq!(
        g.data_start, expected_data_start,
        "data_start 32-byte round-up"
    );
    assert_eq!(g.data_start % 32, 0, "data_start must be 32-aligned");
    assert!(g.data_start >= info_end, "data_start must not round DOWN");
}

#[test]
fn gguf_read_tensor_bytes_uses_data_start_plus_offset() {
    // A tensor's bytes are read from `data_start + info.offset`. We place a
    // known 4-byte payload at that exact position and round-trip it.
    let mut buf = build_gguf(&[("w", &[1], GgufDType::F32 as u32, 0)]);
    let g_before = read_gguf(Cursor::new(buf.clone())).expect("read gguf");
    let data_start = g_before.data_start as usize;
    // Pad up to data_start, then write a 4-byte F32 payload (0x40490FDB ≈ π).
    buf.resize(data_start, 0xAA); // fill alignment gap with non-zero garbage
    let payload = [0xDBu8, 0x0F, 0x49, 0x40]; // π LE
    buf.extend_from_slice(&payload);

    let g = read_gguf(Cursor::new(buf.clone())).expect("read gguf 2");
    let info = &g.tensors[0];
    assert_eq!(info.size_bytes, 4, "F32 1 param = 4 bytes");
    let mut cur = Cursor::new(buf);
    let bytes = grim_format::gguf::read_tensor_bytes(&mut cur, &g, info).expect("read tensor");
    assert_eq!(
        bytes, payload,
        "tensor payload round-trip via data_start+offset"
    );
    // Sanity: reading from the wrong offset (data_start-1) would give garbage.
    let _ = cur.seek(SeekFrom::Start((data_start - 1) as u64));
    let mut wrong = vec![0u8; 4];
    let _ = cur.read_exact(&mut wrong);
    assert_ne!(
        wrong, payload,
        "offset sanity: one byte earlier is NOT the payload"
    );
}

// ===========================================================================
// dtype mapping — especially the silent I16/I32/I64 → F32 promotion.
// ===========================================================================

#[test]
fn map_gguf_dtype_to_storage_integer_kinds_promote_to_f32() {
    // I16/I32/I64 → DType::F32 (NOT an integer storage). This is a deliberate
    // but surprising promotion; a mutant returning a native-int storage would
    // silently change downstream dtype handling. Pin it.
    for dt in [GgufDType::I16, GgufDType::I32, GgufDType::I64] {
        let mapped = map_gguf_dtype_to_storage(dt);
        assert_eq!(mapped, DType::F32, "{:?} must map to F32 storage", dt);
    }
    // I8 is the exception: native U8 storage (embeddings / token ids).
    let i8 = map_gguf_dtype_to_storage(GgufDType::I8);
    assert_eq!(i8.arith, ArithType::U8, "I8 arith = U8");
    assert!(matches!(i8.storage, Storage::Native), "I8 storage = Native");
}

#[test]
fn map_gguf_dtype_to_storage_k_quants_route_to_correct_scheme() {
    // Each GGUF K-quant must route to its matching grim KQuantScheme. A mutant
    // swapping schemes (e.g. Q4K→Q5K) would silently pick the wrong dequant
    // codebook downstream.
    let cases: &[(GgufDType, KQuantScheme)] = &[
        (GgufDType::Q2K, KQuantScheme::Q2K),
        (GgufDType::Q3K, KQuantScheme::Q3K),
        (GgufDType::Q4K, KQuantScheme::Q4K),
        (GgufDType::Q5K, KQuantScheme::Q5K),
        (GgufDType::Q6K, KQuantScheme::Q6K),
        (GgufDType::Q8K, KQuantScheme::Q80),
        (GgufDType::IQ4_NL, KQuantScheme::IQ4NL),
    ];
    for &(dt, scheme) in cases {
        let mapped = map_gguf_dtype_to_storage(dt);
        match mapped.storage {
            Storage::KQuant(got) => assert_eq!(
                got, scheme,
                "{:?} must route to {:?}, got {:?}",
                dt, scheme, got,
            ),
            other => panic!("{:?} storage must be KQuant, got {:?}", dt, other),
        }
        assert_eq!(mapped.arith, ArithType::F32, "{:?} arith must be F32", dt);
    }
}

#[test]
fn map_gguf_dtype_to_grim_bpw_matches_bitwidth() {
    // The bpw provenance tag must equal the quantization bitwidth.
    let cases: &[(GgufDType, u32)] = &[
        (GgufDType::F32, 0), // F32 → None, encoded as 0 in this test harness
        (GgufDType::F16, 16),
        (GgufDType::I8, 8),
        (GgufDType::Q2K, 2),
        (GgufDType::Q3K, 3),
        (GgufDType::Q4K, 4),
        (GgufDType::Q5K, 5),
        (GgufDType::Q6K, 6),
        (GgufDType::Q8K, 8),
        (GgufDType::IQ4_NL, 4),
    ];
    for &(dt, want_bpw) in cases {
        let (_dtype, bpw) = map_gguf_dtype_to_grim(dt);
        match bpw {
            Some(b) => assert_eq!(b, want_bpw, "{:?} bpw", dt),
            None => assert_eq!(want_bpw, 0, "{:?} should be None (encoded 0)", dt),
        }
    }
}
