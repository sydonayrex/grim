//! Payload-geometry validation and companion overrides for GGUF dtype tags.
//!
//! A GGUF stores tensors back to back, so each tensor's byte length is the gap
//! to the next offset. When an exporter tags packed data with a float dtype
//! (e.g. 4-bit data tagged `F64`), the canonical `F64 -> F32` mapping would read
//! the packed bytes as floats *and* allocate the float-sized buffer. These tests
//! pin the refusal and the escape hatch.

use grim_format::gguf::{
    dtype_contradictions, expected_tensor_bytes, read_gguf, GgufDType,
};
use grim_format::tprov::GgufProvider;
use grim_tensor::TensorProvider;
use std::io::{Cursor, Write};

fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn push_tensor(buf: &mut Vec<u8>, name: &str, dims: &[u64], dtype: GgufDType, offset: u64) {
    push_string(buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&(dtype as u32).to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// Full builder with explicit per-tensor payload sizes.
fn build_gguf_with_payloads(
    tensors: &[(&str, Vec<u64>, GgufDType, usize)],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&grim_format::gguf::GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&grim_format::gguf::GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    let mut off = 0u64;
    for (name, dims, dtype, nbytes) in tensors {
        push_tensor(&mut buf, name, dims, *dtype, off);
        off += *nbytes as u64;
    }
    // Real GGUF aligns the tensor data section (general.alignment, default 32);
    // the parser rounds data_start up to it, so the fixture must pad to match or
    // every derived span is short.
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    // Payload: distinctive filler so slices are distinguishable.
    for (_, _, _, nbytes) in tensors {
        buf.extend(std::iter::repeat(0xA5u8).take(*nbytes));
    }
    buf
}

fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("grim-gguf-geometry-{}-{}", std::process::id(), name));
    std::fs::write(&p, bytes).expect("write temp gguf");
    p
}

#[test]
fn expected_bytes_matches_gguf_rule() {
    // F64: 1 element per block, 8 bytes per block.
    assert_eq!(expected_tensor_bytes(GgufDType::F64, &[64]), Some(512));
    // BF16: 2 bytes per element.
    assert_eq!(expected_tensor_bytes(GgufDType::BF16, &[8, 4]), Some(64));
    // Q4_0 and IQ4_NL are both 0.5625 B/elem — the ambiguity that forces a
    // refusal instead of an automatic guess.
    let q40 = expected_tensor_bytes(GgufDType::Q4_0, &[64]).unwrap() as f64 / 64.0;
    let iq = expected_tensor_bytes(GgufDType::IQ4_NL, &[64]).unwrap() as f64 / 64.0;
    assert!((q40 - 0.5625).abs() < 1e-9, "Q4_0 density {q40}");
    assert!((iq - 0.5625).abs() < 1e-9, "IQ4_NL density {iq}");
}

#[test]
fn contradiction_is_detected_and_candidates_reported() {
    // One tensor tagged F64 whose payload is only 0.5625 B/elem (4-bit packed).
    let bytes = build_gguf_with_payloads(&[(
        "blk.0.ffn_gate_exps.weight",
        vec![64, 64],
        GgufDType::F64,
        64 * 64 * 9 / 16,
    )]);
    let g = read_gguf(Cursor::new(bytes.clone())).expect("parse");
    let data_len = 64 * 64 * 9 / 16;
    let contradictions = dtype_contradictions(&g.tensors, data_len as u64);
    assert_eq!(contradictions.len(), 1, "expected exactly one contradiction");
    let c = &contradictions[0];
    assert_eq!(c.name, "blk.0.ffn_gate_exps.weight");
    assert_eq!(c.declared, GgufDType::F64);
    assert_eq!(c.expected, 64 * 64 * 8);
    assert_eq!(c.actual, 64 * 64 * 9 / 16);
    // The candidate set must contain the 4-bit formats, and must NOT contain F64.
    let names: Vec<_> = c.candidates.iter().map(|d| format!("{d:?}")).collect();
    assert!(names.iter().any(|n| n == "Q4_0"), "candidates: {names:?}");
    assert!(
        names.iter().any(|n| n == "IQ4_NL"),
        "candidates: {names:?}"
    );
    assert!(!names.iter().any(|n| n == "F64"), "candidates: {names:?}");
}

#[test]
fn honest_file_has_no_contradictions() {
    // Tags that match the payload: a clean file must not be flagged.
    let bytes = build_gguf_with_payloads(&[
        ("blk.0.attn_q_a.weight", vec![8, 8], GgufDType::BF16, 8 * 8 * 2),
        ("blk.0.norm.weight", vec![8], GgufDType::F32, 8 * 4),
    ]);
    let g = read_gguf(Cursor::new(bytes)).expect("parse");
    let data_len = 8 * 8 * 2 + 8 * 4;
    assert!(dtype_contradictions(&g.tensors, data_len as u64).is_empty());
}

#[test]
fn provider_refuses_a_mislabeled_file() {
    let bytes = build_gguf_with_payloads(&[(
        "blk.0.ffn_gate_exps.weight",
        vec![64, 64],
        GgufDType::F64,
        64 * 64 * 9 / 16,
    )]);
    let p = write_temp("refuse", &bytes);
    let msg = match GgufProvider::open(p.to_str().unwrap()) {
        Ok(_) => panic!("a mislabeled GGUF must be refused, not reinterpreted"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("dtype/payload mismatch"), "msg: {msg}");
    assert!(msg.contains("ffn_gate_exps"), "msg: {msg}");
    // The message must name a way out.
    assert!(msg.contains("quant_overrides"), "msg: {msg}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn companion_override_declares_the_true_dtype() {
    let bytes = build_gguf_with_payloads(&[
        (
            "blk.0.ffn_gate_exps.weight",
            vec![64, 64],
            GgufDType::F64,
            64 * 64 * 9 / 16,
        ),
        ("blk.0.attn_q_a.weight", vec![8, 8], GgufDType::BF16, 8 * 8 * 2),
    ]);
    let p = write_temp("override", &bytes);
    let companion = {
        let mut c = p.clone();
        c.set_extension("json");
        let mut f = std::fs::File::create(&c).expect("create companion");
        f.write_all(
            br#"{"quant_overrides":[
                 {"tensor_name":"blk.*.ffn_gate_exps.weight","override_dtype":"IQ4_NL","effective_bpw":4}
               ]}"#,
        )
        .expect("write companion");
        c
    };

    // With the declaration the file opens, and the tensor is read as IQ4_NL.
    let provider = GgufProvider::open(p.to_str().unwrap())
        .expect("companion override should satisfy the geometry check");
    let meta = provider
        .meta("blk.0.ffn_gate_exps.weight")
        .expect("meta for overridden tensor");
    assert!(
        matches!(
            meta.dtype.storage,
            grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::IQ4NL)
        ),
        "IQ4_NL should map to the KQuant(IQ4NL) storage, got {:?}",
        meta.dtype
    );
    // The payload length is preserved (not widened to F32).
    let raw = provider
        .get_packed("blk.0.ffn_gate_exps.weight")
        .expect("read overridden tensor");
    assert_eq!(raw.bytes.len(), 64 * 64 * 9 / 16);

    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(&companion);
}

#[test]
fn override_only_covers_the_tensor_it_names() {
    // A partial override must NOT let the file through: the untagged tensor is
    // still contradictory, so the loader must keep refusing.
    let bytes = build_gguf_with_payloads(&[
        (
            "blk.0.ffn_gate_exps.weight",
            vec![64, 64],
            GgufDType::F64,
            64 * 64 * 9 / 16,
        ),
        (
            "blk.0.ffn_up_exps.weight",
            vec![64, 64],
            GgufDType::F64,
            64 * 64 * 9 / 16,
        ),
    ]);
    let p = write_temp("partial", &bytes);
    let companion = {
        let mut c = p.clone();
        c.set_extension("json");
        std::fs::write(
            &c,
            br#"{"quant_overrides":[
                 {"tensor_name":"blk.0.ffn_gate_exps.weight","override_dtype":"IQ4_NL"}
               ]}"#,
        )
        .expect("write companion");
        c
    };
    let msg = match GgufProvider::open(p.to_str().unwrap()) {
        Ok(_) => panic!("an un-overridden contradiction must still refuse"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("ffn_up_exps"), "{msg}");
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(&companion);
}
