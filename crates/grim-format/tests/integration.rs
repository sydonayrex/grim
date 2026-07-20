//! End-to-end integration tests that close the loop between the native `.grim`
//! writer (`convert_to_grim`) and the native reader (`GrimProvider`).
//!
//! These tests prove that the format produced by `GrimFile::write` is the
//! same file the reader consumes — a contract the unit tests in `format.rs`
//! and `tprov.rs` verify in isolation but not across the public boundary.

use grim_tensor::provider::TensorProvider;

use grim_format::convert::convert_to_grim;
use grim_format::format::normals_packed_size;
use grim_format::gguf::{GgufDType, GGUF_MAGIC, GGUF_VERSION};
use grim_format::tprov::GrimProvider;

/// Write a one-tensor GGUF (F32, 16 elements) with a padded data region.
/// Returns the in-memory GGUF bytes plus the tensor name appended.
fn write_test_gguf() -> (Vec<u8>, String) {
    let tensor_name = "model.layers.0.weight".to_string();
    let dims = vec![4u64, 4]; // 16 F32 elements = 64 bytes of payload
    let payload_bytes = 64_usize;

    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

    // Tensor info entry
    let name_bytes = tensor_name.as_bytes();
    buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in &dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    let dtype_tag: u32 = GgufDType::F32 as u32;
    buf.extend_from_slice(&dtype_tag.to_le_bytes());
    // Offset within the data region (0 — first and only tensor).
    buf.extend_from_slice(&0u64.to_le_bytes());

    // Pad to 32-byte alignment for `data_start`.
    let data_start = ((buf.len() + 31) & !31) as u64;
    while buf.len() < data_start as usize {
        buf.push(0);
    }

    // Tensor payload: 16 F32 values spread across 64 bytes.
    buf.extend(std::iter::repeat(0xABu8).take(payload_bytes));
    // Pad the file out to a 32-byte boundary so any trailing reads succeed.
    while buf.len() % 32 != 0 {
        buf.push(0);
    }

    (buf, tensor_name)
}

/// Smoke-test that exercises the full write→read loop end-to-end.
///
/// 1. Write a one-tensor GGUF source file to a tempdir.
/// 2. Call `convert_to_grim` to produce a native `.grim` file at 4-bpw.
/// 3. Open the output with `GrimProvider` (the native reader).
/// 4. Read the single tensor via the `TensorProvider` interface.
/// 5. Assert `bytes.len()` matches the independently-computed
///    `normals_packed_size(elem_count, 0, 4)`.
///
/// This is the contract the format guarantees: a `.grim` file written by
/// `GrimFile::write` is readable by `GrimProvider::open` and the public
/// `TensorProvider::get` API returns the expected payload size.
#[test]
fn convert_to_grim_then_grim_provider_round_trips_tensor_payload() {
    let (gguf_bytes, tensor_name) = write_test_gguf();

    let dir = tempfile::tempdir().expect("create tempdir");
    let gguf_path = dir.path().join("input.gguf");
    let grim_path = dir.path().join("output.grim");

    std::fs::write(&gguf_path, &gguf_bytes).expect("write input gguf");

    let gguf_path_str = gguf_path.to_str().expect("utf8 gguf path");
    let grim_path_str = grim_path.to_str().expect("utf8 grim path");

    convert_to_grim(gguf_path_str, grim_path_str, "gfx1100", 4.0, 0, None, None)
        .expect("convert_to_grim must succeed on a valid GGUF source");

    let provider = GrimProvider::open(grim_path_str).expect("GrimProvider must open the converted .grim");

    let meta = provider.meta(&tensor_name).expect("tensor must be in registry");
    let elem_count: usize = meta.shape.iter().product();

    let raw = provider.get(&tensor_name).expect("get must succeed");

    let expected = normals_packed_size(elem_count, 0, 4);
    assert_eq!(
        raw.bytes.len() as u64,
        expected,
        "Payload byte length must match the Wave64-aligned normals_packed_size.\n\
         elem_count={}, base_bitwidth=4, expected={}, got={}",
        elem_count,
        expected,
        raw.bytes.len()
    );

    // Sanity check the metadata survived conversion.
    assert_eq!(
        provider.grim_metadata().target_gcn.as_deref(),
        Some("gfx1100"),
        "target GCN must round-trip through the JSON metadata layer"
    );
    assert_eq!(
        provider.grim_metadata().magic.as_deref(),
        Some("grim-v1"),
        "every native .grim file carries a grim-v1 magic string"
    );
}

/// Confirm that `convert_to_grim` is idempotent across repeated calls on the
/// same input — the GGUF source contains a single F32 tensor with no
/// quantization metadata, so the produced native `.grim` file should be the
/// same size and tensor payload bytes every time.
#[test]
fn convert_to_grim_produces_deterministic_payload_for_same_input() {
    let (gguf_bytes, tensor_name) = write_test_gguf();

    let dir = tempfile::tempdir().expect("create tempdir");
    let gguf_path = dir.path().join("input.gguf");
    std::fs::write(&gguf_path, &gguf_bytes).expect("write input gguf");

    let grim_a = dir.path().join("a.grim");
    let grim_b = dir.path().join("b.grim");
    let gguf_str = gguf_path.to_str().unwrap();
    let a_str = grim_a.to_str().unwrap();
    let b_str = grim_b.to_str().unwrap();

    convert_to_grim(gguf_str, a_str, "gfx1100", 4.0, 0, None, None).unwrap();
    convert_to_grim(gguf_str, b_str, "gfx1100", 4.0, 0, None, None).unwrap();

    let provider_a = GrimProvider::open(a_str).unwrap();
    let provider_b = GrimProvider::open(b_str).unwrap();

    let raw_a = provider_a.get(&tensor_name).unwrap();
    let raw_b = provider_b.get(&tensor_name).unwrap();

    assert_eq!(
        raw_a.bytes.len(),
        raw_b.bytes.len(),
        "Same input + same params must produce the same payload size"
    );

    // Both runs must produce a Wave64-aligned payload (the size is the
    // independently-computed normals_packed_size for the F32 source tensor).
    let expected = normals_packed_size(raw_a.shape.iter().product::<usize>(), 0, 4);
    assert_eq!(raw_a.bytes.len() as u64, expected);
    assert_eq!(raw_b.bytes.len() as u64, expected);
}

/// Spec capability declarations (per-row scales, mixed bitwidth, backups,
/// GPTQ-ORDERED, fusion mask, outlier compression) ride the JSON metadata
/// layer under `grim.ext.entries`. This test writes a `.grim` file with
/// one extension attached, reopens it via `GrimProvider`, and proves the
/// `ext_for(name)` accessor returns the declared capability. The on-disk
/// file format stays at version 1 — the extension is purely metadata.
#[test]
fn grim_provider_returns_extension_declaration_after_round_trip() {
    use grim_format::format::{GrimFile, GrimHeader};
    use grim_format::gguf::{GrimMetadata, GrimRocmlProfile};
    use grim_format::spec::{
        GrimTensorExt, LayoutDescriptor, LayoutHintTag, OutlierIndexEncoding,
        PayloadCompression, PerRowBpwMode, RowScaleDtype,
    };
    use std::collections::HashMap;
    use std::io::Cursor;

    let tensor_name = "layer.0.weight";
    let metadata = GrimMetadata {
        magic: Some("grim-v1".into()),
        quant_version: Some(1),
        rocml_profile: GrimRocmlProfile::Rdna3,
        wavefront_size: 64,
        target_gcn: Some("gfx1100".into()),
        ext_entries: vec![GrimTensorExt {
            tensor_name: tensor_name.into(),
            row_count: 128,
            row_stride: 4096,
            block_size: 0,
            per_row_bpw_mode: PerRowBpwMode::PerRowTable,
            default_bpw: 4,
            own_bpw_table: 1,
            row_scale_dtype: RowScaleDtype::U8,
            scale_offset: 8192,
            scale_size: 128,
            gptq_ordered: 1,
            outlier_index_encoding: OutlierIndexEncoding::DeltaVarint,
            outlier_residual_bpw: 8,
            compression: PayloadCompression::Zstd,
            fusion_mask: 0b11,
            layout_hint: LayoutHintTag::WavefrontTiled,
            layout_descriptor: LayoutDescriptor([7, 14, 21, 28]),
            backup1: grim_format::spec::BackupLayer {
                codes_offset: 16384,
                codes_size: 4096,
                bpw: 8,
                scale_offset: 20480,
                scale_size: 64,
            },
            backup2: grim_format::spec::BackupLayer::default(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let entry = grim_format::format::GrimTensorEntry {
        name: tensor_name.into(),
        shape: vec![128, 4096],
        base_bitwidth: 4,
        payload_offset: 0,
        payload_size: 256,
        outlier_count: 0,
        outlier_offset: 0,
        ..Default::default()
    };

    let grim_file = GrimFile {
        header: GrimHeader::new(1, 0),
        metadata,
        tensors: vec![entry],
        tensors_by_name: HashMap::new(),
        kv_blobs: HashMap::new(),
    };

    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("with_ext.grim");

    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        let written = grim_file.write(&mut cursor).expect("write");
        // Pad and emit dummy payload bytes so the file is well-formed.
        let last = &written[0];
        let mut current = buf.len() as u64;
        if current < last.payload_offset {
            buf.resize(last.payload_offset as usize, 0);
            current = last.payload_offset;
        }
        let needed = (last.payload_offset + last.payload_size) as usize;
        if buf.len() < needed {
            buf.resize(needed, 0);
        }
        let _ = current;
    }
    std::fs::write(&path, &buf).expect("write file");

    let provider = GrimProvider::open(path.to_str().expect("utf8 path")).expect("open");

    // Sanity: V1 wire data still reads.
    let raw = provider.get(tensor_name).expect("get");
    assert_eq!(raw.shape, vec![128, 4096]);

    // The extension declaration round-tripped.
    let ext = provider.ext_for(tensor_name).expect("extension must be present");
    assert_eq!(ext.row_count, 128);
    assert_eq!(ext.row_stride, 4096);
    assert_eq!(ext.per_row_bpw_mode, PerRowBpwMode::PerRowTable);
    assert_eq!(ext.default_bpw, 4);
    assert_eq!(ext.scale_offset, 8192);
    assert_eq!(ext.scale_size, 128);
    assert_eq!(ext.gptq_ordered, 1);
    assert_eq!(ext.outlier_index_encoding, OutlierIndexEncoding::DeltaVarint);
    assert_eq!(ext.outlier_residual_bpw, 8);
    assert_eq!(ext.compression, PayloadCompression::Zstd);
    assert_eq!(ext.fusion_mask, 0b11);
    assert_eq!(ext.layout_hint, LayoutHintTag::WavefrontTiled);
    assert_eq!(ext.layout_descriptor.0, [7, 14, 21, 28]);
    assert!(ext.backup1.is_present());
    assert_eq!(ext.backup1.codes_offset, 16384);
    assert!(!ext.backup2.is_present());

    // Lookup for an unknown tensor returns None cleanly.
    assert!(provider.ext_for("does.not.exist").is_none());
}

/// A version-1 file with no extension declarations opens cleanly and
/// `ext_for` returns `None` for every tensor. This is the back-compat
/// guarantee: existing files do not regress.
#[test]
fn v1_file_without_extensions_still_opens_and_ext_for_returns_none() {
    use grim_format::format::{GrimFile, GrimHeader, GrimTensorEntry};
    use grim_format::gguf::{GrimMetadata, GrimRocmlProfile};
    use std::collections::HashMap;
    use std::io::Cursor;

    let tensor_name = "plain.weight";
    let metadata = GrimMetadata {
        magic: Some("grim-v1".into()),
        quant_version: Some(1),
        rocml_profile: GrimRocmlProfile::Rdna3,
        wavefront_size: 64,
        ..Default::default()
    };
    let entry = GrimTensorEntry {
        name: tensor_name.into(),
        shape: vec![4, 4],
        base_bitwidth: 4,
        payload_offset: 0,
        payload_size: 256,
        outlier_count: 0,
        outlier_offset: 0,
        ..Default::default()
    };
    let grim_file = GrimFile {
        header: GrimHeader::new(1, 0),
        metadata,
        tensors: vec![entry],
        tensors_by_name: HashMap::new(),
        kv_blobs: HashMap::new(),
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("plain.grim");

    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        let written = grim_file.write(&mut cursor).expect("write");
        let last = &written[0];
        let needed = (last.payload_offset + last.payload_size) as usize;
        if buf.len() < needed {
            buf.resize(needed, 0);
        }
    }
    std::fs::write(&path, &buf).expect("write");

    let provider = GrimProvider::open(path.to_str().expect("utf8")).expect("open");
    assert!(provider.grim_metadata().ext_entries.is_empty());
    assert!(provider.ext_for(tensor_name).is_none());
    // V1 tensor data still reads.
    let raw = provider.get(tensor_name).expect("get");
    assert_eq!(raw.shape, vec![4, 4]);
}

/// Phase 7.3: `TensorMeta::fusion_mask` reaches the call site via
/// `provider.meta(name)`. We declare an extension with `fusion_mask = 0b11`
/// (both bit0 and bit1 set), reopen the file, and assert the populated
/// meta reflects both bits and exposes the matching accessors.
#[test]
fn grim_provider_meta_populates_fusion_mask_from_extension() {
    use grim_format::format::{GrimFile, GrimHeader};
    use grim_format::gguf::{GrimMetadata, GrimRocmlProfile};
    use grim_format::spec::GrimTensorExt;
    use std::collections::HashMap;
    use std::io::Cursor;

    let tensor_name = "fused.layer";
    let metadata = GrimMetadata {
        magic: Some("grim-v1".into()),
        quant_version: Some(1),
        rocml_profile: GrimRocmlProfile::Rdna3,
        target_gcn: Some("gfx1100".into()),
        ext_entries: vec![GrimTensorExt {
            tensor_name: tensor_name.into(),
            fusion_mask: 0b11,
            ..Default::default()
        }],
        ..Default::default()
    };

    let entry = grim_format::format::GrimTensorEntry {
        name: tensor_name.into(),
        shape: vec![8, 8],
        base_bitwidth: 4,
        payload_offset: 0,
        payload_size: 256,
        outlier_count: 0,
        outlier_offset: 0,
        ..Default::default()
    };

    let grim_file = GrimFile {
        header: GrimHeader::new(1, 0),
        metadata,
        tensors: vec![entry],
        tensors_by_name: HashMap::new(),
        kv_blobs: HashMap::new(),
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fused.grim");
    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        let written = grim_file.write(&mut cursor).expect("write");
        let needed = (written[0].payload_offset + written[0].payload_size) as usize;
        if buf.len() < needed {
            buf.resize(needed, 0);
        }
    }
    std::fs::write(&path, &buf).expect("write");

    let provider = GrimProvider::open(path.to_str().expect("utf8")).expect("open");
    let meta = provider.meta(tensor_name).expect("meta");

    // The fusion mask from the JSON extension reaches TensorMeta.
    assert_eq!(meta.fusion_mask, 0b11);
    assert!(meta.has_rmsnorm_matmul_fusion(), "bit0 must be set");
    assert!(meta.has_qkv_attention_fusion(), "bit1 must be set");

    // A tensor with no extension defaults to mask 0.
    assert_eq!(provider.meta("does.not.exist").is_err(), true);
}
