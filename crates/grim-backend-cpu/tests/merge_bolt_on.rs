//! Integration test: `merge_bolt_on` bakes an adapter into the primary stream
//! and the resulting file dequantizes (via `dequant_row`) exactly to the
//! pre-merge effective weights plus `ΔW = (scale / rank)·B@A`.
//!
//! Lives in `grim-backend-cpu` (not `grim-format`) so the test can call the
//! real CPU decoder directly without the cyclic dev-dependency that would
//! otherwise produce two distinct `grim_format` crate units.

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom};

use grim_backend_cpu::dequant_row;
use grim_format::bolt_on::{attach_bolt_on, merge_bolt_on};
use grim_format::format::{GrimFile, GrimHeader, GrimTensorEntry, read_outliers_with_encoding};
use grim_format::gguf::{GrimMetadata, GrimRocmlProfile};
use grim_format::spec::{BackupLayer, GrimTensorExt};
use grim_tensor::shape::Shape;

const TENSOR: &str = "layer.0.weight";
const ROWS: usize = 32;
const COLS: usize = 128;
const BPW: u8 = 4;
const TENSOR_ROWS: u64 = ROWS as u64;

fn build_base_file() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.grim");

    let metadata = GrimMetadata {
        magic: Some("grim-v1".into()),
        quant_version: Some(1),
        rocml_profile: GrimRocmlProfile::Rdna3,
        wavefront_size: 64,
        target_gcn: Some("gfx1100".into()),
        ext_entries: vec![GrimTensorExt {
            tensor_name: TENSOR.into(),
            row_count: ROWS as u64,
            row_stride: COLS as u64,
            default_bpw: BPW,
            scale_size: TENSOR_ROWS,
            scale_offset: 256,
            backup2: BackupLayer {
                codes_offset: 512,
                codes_size: 256,
                bpw: 2,
                scale_offset: 768,
                scale_size: 256,
            },
            ..Default::default()
        }],
        ..Default::default()
    };

    let entry = GrimTensorEntry {
        name: TENSOR.into(),
        shape: vec![ROWS, COLS],
        base_bitwidth: BPW,
        payload_offset: 0,
        payload_size: 1024,
        ..Default::default()
    };

    let grim_file = GrimFile {
        header: GrimHeader::new(1, 0),
        metadata,
        tensors: vec![entry],
        tensors_by_name: HashMap::new(),
        kv_blobs: HashMap::new(),
        wave: grim_format::format::WaveSize::W64,
    };

    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        let written = grim_file.write(&mut cursor).expect("write base");
        let needed = (written[0].payload_offset + written[0].payload_size) as usize;
        if buf.len() < needed {
            buf.resize(needed + 512, 0);
        }
    }
    std::fs::write(&path, &buf).expect("write base file");
    (dir, path)
}

/// Read every row of `TENSOR` through the real `dequant_row` decoder.
fn read_rows(path: &std::path::Path) -> Vec<Vec<f32>> {
    let mut file = std::fs::File::open(path).expect("open");
    let grim = GrimFile::read(&mut file).expect("read grim");
    let idx = grim
        .tensors_by_name
        .get(TENSOR)
        .copied()
        .expect("tensor by name");
    let entry = &grim.tensors[idx];
    let ext = grim.metadata.get_tensor_ext(TENSOR).expect("tensor ext");

    let mut payload = vec![0u8; entry.payload_size as usize];
    file.seek(SeekFrom::Start(entry.payload_offset)).expect("seek");
    file.read_exact(&mut payload).expect("read payload");

    let scales: Vec<u8> = if ext.scale_size > 0 {
        let s = ext.scale_offset as usize;
        let e = s + ext.scale_size as usize;
        payload[s..e].to_vec()
    } else {
        Vec::new()
    };

    let outliers = read_outliers_with_encoding(&mut file, entry, ext.outlier_index_encoding)
        .expect("read outliers");
    let outlier_pairs: Vec<(u32, f32)> =
        outliers.into_iter().map(|o| (o.index, o.value)).collect();

    (0..ext.row_count as usize)
        .map(|r| {
            dequant_row(
                r,
                ext.row_stride as usize,
                &payload,
                &scales,
                ext.default_bpw,
                Some(ext),
                &outlier_pairs,
            )
        })
        .collect()
}

#[test]
fn merge_bakes_adapter_and_matches_reference() {
    let (_dir, path) = build_base_file();

    // Attach a first adapter so backup2 holds a real residual.
    let a1 = grim_backend_cpu::cpu_tensor(
        vec![0.05f32; 2 * COLS],
        Shape::new(vec![2, COLS]),
    );
    let b1 = grim_backend_cpu::cpu_tensor(
        vec![0.05f32; ROWS * 2],
        Shape::new(vec![ROWS, 2]),
    );
    attach_bolt_on(&path, TENSOR, &a1, &b1, 1.0).expect("attach");

    let pre = read_rows(&path);

    // Snapshot ext metadata post-attach.
    let mut file = std::fs::File::open(&path).expect("open");
    let grim = GrimFile::read(&mut file).expect("read");
    let ext = grim.metadata.get_tensor_ext(TENSOR).expect("ext");
    assert!(ext.backup2.is_present(), "backup2 should be populated");

    // Merge a distinct adapter: rank 3, scale 0.5.
    let rank = 3usize;
    let merge_scale = 0.5f32;
    let am: Vec<f32> = vec![0.03f32; rank * COLS];
    let bm: Vec<f32> = vec![0.02f32; ROWS * rank];
    let a2 = grim_backend_cpu::cpu_tensor(am.clone(), Shape::new(vec![rank, COLS]));
    let b2 = grim_backend_cpu::cpu_tensor(bm.clone(), Shape::new(vec![ROWS, rank]));
    merge_bolt_on(&path, TENSOR, &a2, &b2, merge_scale).expect("merge");

    // After merge, backup slots must be freed and gptq ordering native.
    let mut file = std::fs::File::open(&path).expect("open");
    let grim = GrimFile::read(&mut file).expect("read");
    let ext = grim.metadata.get_tensor_ext(TENSOR).expect("ext");
    assert_eq!(ext.gptq_ordered, 0);
    assert!(!ext.backup1.is_present(), "backup1 should be cleared");
    assert!(!ext.backup2.is_present(), "backup2 should be cleared");

    // Expected = pre-merge effective + (scale / rank) * Bm @ Am.
    let adj = merge_scale / rank as f32;
    let mut expected = pre.clone();
    for o in 0..ROWS {
        for i in 0..COLS {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += bm[o * rank + r] * am[r * COLS + i];
            }
            expected[o][i] += adj * sum;
        }
    }

    let post = read_rows(&path);
    let tol = 0.2f32;
    for o in 0..ROWS {
        for i in 0..COLS {
            let diff = (post[o][i] - expected[o][i]).abs();
            assert!(
                diff <= tol,
                "row {o} col {i}: got {} expected {} diff {}",
                post[o][i],
                expected[o][i],
                diff
            );
        }
    }
}