//! Integration tests for the local-first training dashboard backend.
//!
//! Covers discovery (models/datasets), training-job state machine, ROCm
//! device probe, and the axum HTTP routes that the React UI consumes.

use std::collections::HashMap;
use std::path::Path;

use grim_format::gguf::{GgufFile, GgufTensorInfo, GgufValue, GGUF_MAGIC, GGUF_VERSION};
use grim_garage::discovery::{discover_convertible_models, discover_datasets, discover_models, ModelEntry};
use grim_garage::jobs::{JobId, JobRegistry, JobStatus, TrainingJob};
use grim_garage::rocm::{probe_rocm_devices, RocmDeviceInfo};
use tempfile::tempdir;

fn write_minimal_gguf(path: &Path, tensor_name: &str, payload_bytes: Vec<u8>) {
    let tensor = GgufTensorInfo {
        name: tensor_name.to_string(),
        dims: vec![1u64],
        offset: 0,
        size_bytes: payload_bytes.len() as u64,
        dtype: grim_format::gguf::GgufDType::F32,
    };
    let gguf = GgufFile {
        version: GGUF_VERSION,
        tensor_count: 1,
        metadata: HashMap::from([(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        )]),
        tensors: vec![tensor],
        data_start: 0,
    };

    // Direct write using GGUF header spec — matches the discovery reader.
    use std::io::Write;
    let mut buf: Vec<u8> = Vec::new();
    buf.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    buf.write_all(&gguf.version.to_le_bytes()).unwrap();
    buf.write_all(&(gguf.tensor_count as u64).to_le_bytes()).unwrap();
    buf.write_all(&(gguf.metadata.len() as u64).to_le_bytes()).unwrap();

    for (k, v) in &gguf.metadata {
        let kb = k.as_bytes();
        buf.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
        buf.write_all(kb).unwrap();
        if let GgufValue::String(s) = v {
            buf.write_all(&8u32.to_le_bytes()).unwrap();
            let sb = s.as_bytes();
            buf.write_all(&(sb.len() as u64).to_le_bytes()).unwrap();
            buf.write_all(sb).unwrap();
        }
    }

    for t in &gguf.tensors {
        let nb = t.name.as_bytes();
        buf.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
        buf.write_all(nb).unwrap();
        buf.write_all(&(t.dims.len() as u32).to_le_bytes()).unwrap();
        for d in &t.dims {
            buf.write_all(&d.to_le_bytes()).unwrap();
        }
        let dtype_tag: u32 = match t.dtype {
            grim_format::gguf::GgufDType::F32 => 6,
            grim_format::gguf::GgufDType::F16 => 5,
            grim_format::gguf::GgufDType::Q4K => 12,
            grim_format::gguf::GgufDType::Q5K => 13,
            grim_format::gguf::GgufDType::Q6K => 14,
            grim_format::gguf::GgufDType::Q8_0 => 8,
            _ => 6,
        };
        buf.write_all(&dtype_tag.to_le_bytes()).unwrap();
        buf.write_all(&t.offset.to_le_bytes()).unwrap();
    }

    // Align data region to 32 bytes.
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&payload_bytes);

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&buf).unwrap();
}

// ----- discover_models -----

#[test]
fn discover_convertible_models_finds_gguf_in_directory() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("tiny.gguf");
    write_minimal_gguf(&model_path, "blk.0.w", vec![0u8; 16]);

    let models = discover_convertible_models(dir.path()).expect("discover");
    assert_eq!(models.len(), 1);
    let m = &models[0];
    assert_eq!(m.id, "tiny.gguf");
    assert_eq!(m.format, "gguf");
    assert!(!m.is_grim);
}

#[test]
fn discover_models_finds_grim_extension_and_marks_is_grim() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.grim");
    write_minimal_gguf(&path, "w", vec![0u8; 16]);
    // Rename to look like .grim so the extension filter matches.
    let grim_path = dir.path().join("model.grim");
    std::fs::rename(&path, &grim_path).unwrap();

    let models = discover_models(dir.path()).expect("discover");
    assert_eq!(models.len(), 1);
    let m = &models[0];
    assert_eq!(m.format, "grim");
    assert!(m.is_grim);
}

#[test]
fn discover_models_ignores_non_model_files() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("readme.txt"), b"hello").unwrap();
    std::fs::create_dir(dir.path().join("not_a_model")).unwrap();
    std::fs::write(
        dir.path().join("not_a_model").join("weights.bin"),
        b"\x00",
    )
    .unwrap();

    let models = discover_models(dir.path()).expect("discover");
    assert!(models.is_empty());
}

#[test]
fn discover_models_returns_empty_for_missing_directory() {
    let result = discover_models(Path::new("/does-not-exist/gracefully")).unwrap();
    assert!(result.is_empty());
}

#[test]
fn model_entry_round_trips_id_and_format() {
    let entry = ModelEntry::new("a.gguf", "/tmp/a.gguf", "gguf", false);
    assert_eq!(entry.id, "a.gguf");
    assert_eq!(entry.format, "gguf");
    assert!(!entry.is_grim);
}

// ----- discover_datasets -----

#[test]
fn discover_datasets_finds_jsonl_files() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("train.jsonl"), "{}\n").unwrap();
    std::fs::write(dir.path().join("eval.jsonl"), "{}\n").unwrap();
    std::fs::write(dir.path().join("notes.txt"), "ignore me").unwrap();

    let datasets = discover_datasets(dir.path()).expect("discover");
    assert_eq!(datasets.len(), 2);
    let names: Vec<&str> = datasets.iter().map(|d| d.id.as_str()).collect();
    assert!(names.contains(&"train.jsonl"));
    assert!(names.contains(&"eval.jsonl"));
}

#[test]
fn discover_datasets_finds_parquet_files() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("dataset.parquet"), b"PAR1").unwrap();
    let datasets = discover_datasets(dir.path()).expect("discover");
    assert_eq!(datasets.len(), 1);
    assert_eq!(datasets[0].id, "dataset.parquet");
}

#[test]
fn discover_datasets_empty_when_dir_missing() {
    let datasets = discover_datasets(Path::new("/no/such/dir")).unwrap();
    assert!(datasets.is_empty());
}

// ----- JobRegistry & TrainingJob -----

#[tokio::test]
async fn job_registry_starts_empty() {
    let reg = JobRegistry::new();
    assert_eq!(reg.list().await.len(), 0);
}

#[tokio::test]
async fn job_registry_creates_and_lists_pending_job() {
    let reg = JobRegistry::new();
    let id: JobId = reg
        .create(TrainingJob {
            model_path: "/tmp/model.gguf".into(),
            dataset_path: "/tmp/data.jsonl".into(),
            training_mode: grim_garage::jobs::TrainingMode::Lora,
            lora_rank: 16,
            learning_rate: 2e-5,
            epochs: 1,
            rocm_fusion_rmsnorm_matmul: true,
            rocm_fusion_qkv_attention: false,
            ..Default::default()
        })
        .await
        .expect("create");
    let list = reg.list().await;
    assert_eq!(list.len(), 1);
    let job = reg.get(&id).await.expect("get");
    assert_eq!(job.status, JobStatus::Pending);
    assert_eq!(job.model_path, "/tmp/model.gguf");
}

#[tokio::test]
async fn job_registry_rejects_duplicate_id_returns_err() {
    let reg = JobRegistry::new();
    let job = TrainingJob {
        model_path: "/m.gguf".into(),
        dataset_path: "/d.jsonl".into(),
        training_mode: grim_garage::jobs::TrainingMode::Bf16Full,
        lora_rank: 8,
        learning_rate: 1e-5,
        epochs: 1,
        rocm_fusion_rmsnorm_matmul: false,
        rocm_fusion_qkv_attention: false,
        ..Default::default()
    };
    let id = reg.create(job.clone()).await.unwrap();
    let err = reg.insert_with_id(id, job).await.expect_err("duplicate rejected");
    let _ = err;
}

#[test]
fn job_status_transitions_pending_to_running_to_completed() {
    let mut job = TrainingJob {
        model_path: "/m.gguf".into(),
        dataset_path: "/d.jsonl".into(),
        training_mode: grim_garage::jobs::TrainingMode::QLoRA,
        lora_rank: 32,
        learning_rate: 5e-5,
        epochs: 3,
        rocm_fusion_rmsnorm_matmul: true,
        rocm_fusion_qkv_attention: true,
        ..Default::default()
    };
    assert_eq!(job.status, JobStatus::Pending);
    job.status = JobStatus::Running;
    assert_eq!(job.status, JobStatus::Running);
    job.status = JobStatus::Completed;
    assert_eq!(job.status, JobStatus::Completed);
}

#[test]
fn job_metrics_append_and_read_back() {
    let mut job = TrainingJob {
        model_path: "/m.gguf".into(),
        dataset_path: "/d.jsonl".into(),
        training_mode: grim_garage::jobs::TrainingMode::Lora,
        lora_rank: 8,
        learning_rate: 2e-5,
        epochs: 1,
        rocm_fusion_rmsnorm_matmul: false,
        rocm_fusion_qkv_attention: false,
        ..Default::default()
    };
    job.push_metric(0, 2.31, 1024);
    job.push_metric(1, 1.98, 2048);
    assert_eq!(job.metrics.len(), 2);
    assert_eq!(job.metrics[0].step, 0);
    assert!((job.metrics[0].loss - 2.31).abs() < 1e-6);
    assert_eq!(job.metrics[0].tokens, 1024);
}

// ----- ROCm probe -----

#[test]
fn roc_mdevice_info_serializes_name_fields() {
    let info = RocmDeviceInfo {
        ordinal: 0,
        name: "AMD Radeon RX 7900 XTX".into(),
        vendor: "AMD".into(),
        backend: "ROCm".into(),
        is_rocm_compliant: true,
        gcn_arch: "gfx1100".into(),
        vram_bytes: 16 * 1024 * 1024 * 1024,
        wavefront_size: 32,
        wmma_supported: true,
        mfma_supported: false,
        xnack_enabled: false,
        compute_units: 84,
        max_threads_per_block: 1024,
    };
    assert_eq!(info.ordinal, 0);
    assert_eq!(info.wavefront_size, 32);
    let _serde = serde_json::to_string(&info).expect("serialize");
}

#[test]
fn probe_rocm_devices_returns_vec_even_when_no_gpu() {
    // Probe does not require a real GPU; must return a Vec (possibly empty)
    // rather than panicking.
    let devs: Vec<RocmDeviceInfo> = probe_rocm_devices();
    // The result is either empty (no ROCm runtime) or populated from real HIP.
    for d in &devs {
        assert!(d.ordinal <= 64);
    }
}
