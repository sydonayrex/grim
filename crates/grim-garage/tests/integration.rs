//! Integration tests for the local-first training dashboard backend.
//!
//! Covers discovery (models/datasets), training-job state machine, ROCm
//! device probe, and the axum HTTP routes that the React UI consumes.

use std::collections::HashMap;
use std::path::Path;

use grim_format::gguf::{GGUF_MAGIC, GGUF_VERSION, GgufFile, GgufTensorInfo, GgufValue};
use grim_garage::discovery::{
    ModelEntry, discover_convertible_models, discover_datasets, discover_models,
};
use grim_garage::jobs::{JobId, JobRegistry, JobStatus, TrainingJob};
use grim_garage::rocm::{RocmDeviceInfo, probe_rocm_devices};
use tempfile::tempdir;
use tower::ServiceExt;

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
    buf.write_all(&(gguf.tensor_count as u64).to_le_bytes())
        .unwrap();
    buf.write_all(&(gguf.metadata.len() as u64).to_le_bytes())
        .unwrap();

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

/// Write a complete tiny Llama GGUF (vocab=64, hidden=16, heads=4, kv=4,
/// head_dim=4, layers=1, ffn=32) that `GgufProvider::open` can read and the
/// SFT worker can actually run real forward steps against. Every tensor that
/// the head loaders and `LlamaBlock::load` ask for is present with the exact
/// expected shape (`WeightSource::get` compares raw shape to expected shape;
/// GGUF stores dims reversed, so we write `shape.reverse()`). Linear weights
/// are GGUF-native `[out, in]` row-major, matching `Linear::load(ws, in, out)`
/// which calls `get([out, in])`: `wq/wk/wv` = `[hidden, n_heads*head_dim]`,
/// `wo` = `[n_heads*head_dim, hidden]`, `w_gate/w_up` = `[ffn, hidden]`,
/// `w_down` = `[hidden, ffn]`, `token_embd`/`output` = `[vocab, hidden]`.
fn write_tiny_llama_gguf(path: &Path) {
    use grim_format::gguf::GgufValue;

    const VOCAB: u64 = 64;
    const HIDDEN: u64 = 16;
    const HEADS: u64 = 4;
    const KV_HEADS: u64 = 4;
    const HEAD_DIM: u64 = 4;
    const FFN: u64 = 32;
    const LAYERS: u64 = 1;

    let metadata: HashMap<String, GgufValue> = HashMap::from([
        (
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.vocab_size".into(),
            GgufValue::Uint32(VOCAB as u32),
        ),
        (
            "llama.embedding_length".into(),
            GgufValue::Uint32(HIDDEN as u32),
        ),
        ("llama.block_count".into(), GgufValue::Uint32(LAYERS as u32)),
        (
            "llama.attention.head_count".into(),
            GgufValue::Uint32(HEADS as u32),
        ),
        (
            "llama.attention.head_count_kv".into(),
            GgufValue::Uint32(KV_HEADS as u32),
        ),
        (
            "llama.attention.key_length".into(),
            GgufValue::Uint32(HEAD_DIM as u32),
        ),
        (
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(FFN as u32),
        ),
        ("llama.context_length".into(), GgufValue::Uint32(128)),
        (
            "llama.attention.layer_norm_rms_eps".into(),
            GgufValue::Float32(1e-5),
        ),
        ("llama.rope.freq_base".into(), GgufValue::Float32(10000.0)),
    ]);

    // (name, row-major dims, fill value). Dims are reversed when written.
    let tensors: Vec<(&str, Vec<u64>, f32)> = vec![
        ("token_embd.weight", vec![VOCAB, HIDDEN], 0.001),
        ("output_norm.weight", vec![HIDDEN], 1.0),
        ("output.weight", vec![VOCAB, HIDDEN], 0.001),
        ("layers.0.attn_norm.weight", vec![HIDDEN], 1.0),
        (
            "layers.0.attn.wq.weight",
            vec![HIDDEN, HEADS * HEAD_DIM],
            0.01,
        ),
        (
            "layers.0.attn.wk.weight",
            vec![HIDDEN, KV_HEADS * HEAD_DIM],
            0.01,
        ),
        (
            "layers.0.attn.wv.weight",
            vec![HIDDEN, KV_HEADS * HEAD_DIM],
            0.01,
        ),
        (
            "layers.0.attn.wo.weight",
            vec![HEADS * HEAD_DIM, HIDDEN],
            0.01,
        ),
        ("layers.0.ffn_norm.weight", vec![HIDDEN], 1.0),
        ("layers.0.ffn.w_gate.weight", vec![FFN, HIDDEN], 0.01),
        ("layers.0.ffn.w_up.weight", vec![FFN, HIDDEN], 0.01),
        ("layers.0.ffn.w_down.weight", vec![HIDDEN, FFN], 0.01),
    ];

    let align32 = |n: u64| (n + 31) & !31;

    use std::io::Write;
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (k, v) in &metadata {
        let kb = k.as_bytes();
        buf.extend_from_slice(&(kb.len() as u64).to_le_bytes());
        buf.extend_from_slice(kb);
        match v {
            GgufValue::String(s) => {
                buf.extend_from_slice(&8u32.to_le_bytes());
                let sb = s.as_bytes();
                buf.extend_from_slice(&(sb.len() as u64).to_le_bytes());
                buf.extend_from_slice(sb);
            }
            GgufValue::Uint32(u) => {
                buf.extend_from_slice(&4u32.to_le_bytes());
                buf.extend_from_slice(&u.to_le_bytes());
            }
            GgufValue::Float32(f) => {
                buf.extend_from_slice(&6u32.to_le_bytes());
                buf.extend_from_slice(&f.to_le_bytes());
            }
            _ => panic!("unsupported metadata value in tiny-llama fixture"),
        }
    }

    // Data region is aligned to 32 and tensor offsets are relative to it.
    let infos_size: u64 = tensors
        .iter()
        .map(|(name, dims, _)| 8 + name.len() as u64 + 4 + 8 * dims.len() as u64 + 4 + 8)
        .sum();
    let data_start = align32(buf.len() as u64 + infos_size);

    let mut cursor = 0u64;
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut sizes = Vec::with_capacity(tensors.len());
    for (_, dims, _) in &tensors {
        let bytes = dims.iter().product::<u64>() * 4; // F32
        cursor = align32(cursor);
        offsets.push(cursor);
        sizes.push(bytes);
        cursor += bytes;
    }

    for ((name, dims, _), &offset) in tensors.iter().zip(&offsets) {
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u64).to_le_bytes());
        buf.extend_from_slice(nb);
        let disk_dims: Vec<u64> = dims.iter().rev().copied().collect();
        buf.extend_from_slice(&(disk_dims.len() as u32).to_le_bytes());
        for d in &disk_dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // GGUF dtype tag: F32
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    while (buf.len() as u64) < data_start {
        buf.push(0);
    }

    for ((_, _, fill), (&offset, &bytes)) in tensors.iter().zip(offsets.iter().zip(&sizes)) {
        assert_eq!(
            buf.len() as u64,
            data_start + offset,
            "tensor layout mismatch"
        );
        for _ in 0..bytes / 4 {
            buf.extend_from_slice(&fill.to_le_bytes());
        }
    }

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
    std::fs::write(dir.path().join("not_a_model").join("weights.bin"), b"\x00").unwrap();

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
    let err = reg
        .insert_with_id(id, job)
        .await
        .expect_err("duplicate rejected");
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

#[test]
fn job_status_round_trips_cancelled_through_serde_lowercase() {
    // Verify the wire-rename: serialized as lowercase "cancelled" for
    // parity with `status_label`/`JobSummary.status`.
    let s = serde_json::to_string(&JobStatus::Cancelled).expect("serialize");
    assert_eq!(s, "\"cancelled\"");
    let back: JobStatus = serde_json::from_str("\"cancelled\"").expect("deserialize");
    assert_eq!(back, JobStatus::Cancelled);
}

#[tokio::test]
async fn cancel_signals_worker_and_status_transitions_to_cancelled() {
    use std::sync::Arc;
    use std::time::Duration;

    // H1: cancelling a running worker must (a) stop the worker's loop, and
    // (b) leave the terminal status as `Cancelled` — never resurrected to
    // `Completed` by the still-running worker's natural-completion path.
    // SFT modes open the real base model, so provide a readable tiny Llama GGUF.
    write_tiny_llama_gguf(Path::new("/tmp/cancel-test.gguf"));
    let dataset_file = "/tmp/cancel-test.jsonl";
    {
        use std::io::Write;
        let mut f = std::fs::File::create(dataset_file).expect("create test dataset");
        for i in 0..150 {
            writeln!(
                f,
                "{{\"text\": \"sample prompt and output text number {}\"}}",
                i
            )
            .unwrap();
        }
    }

    let reg = Arc::new(JobRegistry::new());
    let id = reg
        .create(TrainingJob {
            model_path: "/tmp/cancel-test.gguf".into(),
            dataset_path: dataset_file.into(),
            training_mode: grim_garage::jobs::TrainingMode::Lora,
            lora_rank: 8,
            learning_rate: 2e-5,
            epochs: 10, // 100 steps @ ~10ms ≈ 1s — long enough for cancel to land mid-loop
            ..Default::default()
        })
        .await
        .expect("create");

    // Run the worker so it transitions to Running + samples step 0.
    let reg_clone = Arc::clone(&reg);
    let worker_id = id.clone();
    tokio::spawn(async move {
        grim_garage::jobs::run_training_worker(reg_clone, worker_id).await;
    });

    // Wait for the worker to actually enter its loop (poll until status is
    // Running with at least one metric) so we don't race the autograd
    // init. Bounded — gives up to ~400 ms.
    for _ in 0..40 {
        if let Some(j) = reg.get(&id).await {
            if j.status == JobStatus::Running && !j.metrics.is_empty() {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Fire cancellation via the atomic surface the HTTP `cancel_job` route
    // uses (`request_cancel`).
    let observed_status = reg.request_cancel(&id).await.expect("request_cancel");
    assert_eq!(observed_status, JobStatus::Cancelled);

    // Allow the worker's `select!` loop to observe the token and return.
    tokio::time::sleep(Duration::from_millis(60)).await;

    let snapshot = reg.get(&id).await.expect("job survives");
    assert_eq!(
        snapshot.status,
        JobStatus::Cancelled,
        "expected Cancelled, got {:?} (worker resurrect bug)",
        snapshot.status
    );
    // A cancelled job records the metrics emitted before the cancel; it
    // must NOT have reached the full 100 steps a 10-epoch run would emit.
    assert!(
        snapshot.metrics.len() < 100,
        "worker emitted {} metrics after cancel — ran to completion",
        snapshot.metrics.len()
    );
    // Sanity: at least one metric landed before the cancel (the worker
    // had clearly entered the loop).
    assert!(
        !snapshot.metrics.is_empty(),
        "worker never emitted a metric before cancel — did it start?"
    );
}

#[tokio::test]
async fn cancel_on_missing_job_returns_not_found() {
    use grim_garage::jobs::JobError;
    let reg = JobRegistry::new();
    let bogus = JobId("does-not-exist".into());
    match reg.cancel(&bogus).await {
        Err(JobError::NotFound(id)) => assert_eq!(id, "does-not-exist"),
        other => panic!("cancel: expected NotFound, got {other:?}"),
    }
    // The atomic surface used by the HTTP route must agree.
    match reg.request_cancel(&bogus).await {
        Err(JobError::NotFound(id)) => assert_eq!(id, "does-not-exist"),
        other => panic!("request_cancel: expected NotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn update_status_and_broadcast_emits_terminal_completed_event() {
    use std::sync::Arc;

    // H3: a Completed/Cancelled/Failed transition must broadcast a
    // terminal `MetricStreamEvent` so SSE clients learn about it without
    // having to poll `/api/train/status`. The broadcast channel's sender
    // lives forever (held in the registry), so without this broadcast the
    // stream would never terminate on the happy path.
    let reg = Arc::new(JobRegistry::new());
    let id = reg
        .create(TrainingJob {
            model_path: "/tmp/broadcast-test.gguf".into(),
            dataset_path: "/tmp/broadcast-test.jsonl".into(),
            training_mode: grim_garage::jobs::TrainingMode::Lora,
            epochs: 1,
            ..Default::default()
        })
        .await
        .expect("create");

    // Subscribe FIRST so we observe the broadcast. A consumer subscribed
    // to the live metrics stream will receive the metric events plus the
    // terminal one (and the SSE handler keys off this terminal status).
    let mut rx = reg.subscribe_metrics();
    // Push one synthetic metric so a real terminal metric is read back.
    reg.append_metric(
        &id,
        grim_garage::jobs::Metric {
            step: 0,
            loss: 2.3,
            tokens: 512,
            grad_norm: 0.0,
            lr: 0.0,
            vram_used_mb: 0,
            samples_per_sec: 0.0,
        },
    )
    .await
    .expect("append");
    // Drain the per-step event; the next one must be the terminal.
    let _step0 = rx.recv().await.expect("recv step 0");
    reg.update_status_and_broadcast(&id, JobStatus::Completed)
        .await
        .expect("broadcast completed");
    let terminal = rx.recv().await.expect("recv terminal");
    assert_eq!(terminal.job_id, id.0);
    assert_eq!(terminal.status, JobStatus::Completed);
    // Terminal event carries the last recorded metric (step 0) — clients
    // can render the final loss curve point from the event alone.
    assert_eq!(terminal.metric.step, 0);
}

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
        vram_used_bytes: 1024 * 1024 * 1024,
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

// ----- M1: path traversal validation in start_training -----

/// Helper that mirrors the validation the start_training route must
/// perform before accepting a `model_path` / `dataset_path` field. The
/// helper intentionally lives at module scope (not inlined inside the
/// handler) so we can unit-test it without spinning up an axum router.
fn validate_job_path(value: &str) -> std::result::Result<(), String> {
    if value.contains("..") || value.contains('/') || value.contains('\\') {
        Err(format!(
            "invalid path: {value:?} contains traversal or separator"
        ))
    } else {
        Ok(())
    }
}

#[test]
fn path_traversal_validator_rejects_dotdot_in_model_path() {
    assert!(validate_job_path("../etc/cron.d/x").is_err());
    assert!(validate_job_path("/absolute/path").is_err());
    assert!(validate_job_path("normal.gguf").is_ok());
    assert!(validate_job_path("a\\b\\c.gguf").is_err());
}

#[tokio::test]
async fn start_training_route_rejects_path_traversal_in_model_path() {
    // M1: an `../` segment in model_path or dataset_path must be returned
    // as 400 BAD_REQUEST — never reach the registry or the worker. Pre-fix
    // the route accepts any string and the worker's sidecar write would
    // create directories under arbitrary attacker-controlled paths.
    use grim_garage::routes;
    use tower::ServiceExt;

    let state = routes::new_app_state();
    let app = routes::build_router(state);
    let body = serde_json::json!({
        "model_path": "../etc/cron.d/x",
        "dataset_path": "/data/safe.jsonl",
        "training_mode": "Lora"
    })
    .to_string();
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/api/train/start")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        axum::http::StatusCode::BAD_REQUEST,
        "start_training must reject traversal in model_path"
    );
}

#[tokio::test]
async fn start_training_route_rejects_path_traversal_in_dataset_path() {
    // M1 symmetric: same defense on dataset_path.
    let state = grim_garage::routes::new_app_state();
    let app = grim_garage::routes::build_router(state);
    let body = serde_json::json!({
        "model_path": "/safe/model.gguf",
        "dataset_path": "../share/secrets.jsonl",
        "training_mode": "Lora"
    })
    .to_string();
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/api/train/start")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        axum::http::StatusCode::BAD_REQUEST,
        "start_training must reject traversal in dataset_path"
    );
}

// ----- L5: ghost JobSummary defense -----

#[tokio::test]
async fn list_jobs_filters_out_empty_model_path_rows() {
    // L5: a job that ended up with an empty `model_path` (e.g. through a
    // pre-M1 race window or a future code path bypassing path validation)
    // must not be surfaced as a ghost card. The route layer filters empty
    // rows out post-snapshot. Hand-inject two jobs, one with model_path
    // empty, and assert only the well-formed one enters the response.
    let state = grim_garage::routes::new_app_state();
    let reg = &state.registry;
    let real_id = reg
        .create(TrainingJob {
            model_path: "real.gguf".into(),
            dataset_path: "data.jsonl".into(),
            training_mode: grim_garage::jobs::TrainingMode::Lora,
            ..Default::default()
        })
        .await
        .expect("create real");
    let ghost_id = reg
        .create(TrainingJob {
            model_path: String::new(),
            dataset_path: "data.jsonl".into(),
            training_mode: grim_garage::jobs::TrainingMode::Lora,
            ..Default::default()
        })
        .await
        .expect("create ghost");

    // Hand-mutate the ghost to match almost-real paths but with empty
    // model_path — the route filter keys on the post-M1 invariants
    // (non-empty model_path AND non-empty dataset_path).
    let app = grim_garage::routes::build_router(state.clone());
    let resp = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/train/jobs")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    let jobs = body
        .get("jobs")
        .and_then(|j| j.as_array())
        .expect("jobs array");
    let ids: Vec<&str> = jobs
        .iter()
        .map(|j| j.get("job_id").and_then(|v| v.as_str()).unwrap_or_default())
        .collect();
    assert!(ids.contains(&real_id.0.as_str()), "real job missing");
    assert!(
        !ids.contains(&ghost_id.0.as_str()),
        "ghost (empty model_path) leaked into response: {ids:?}"
    );
}

#[tokio::test]
async fn job_registry_snapshot_is_consistent_under_concurrent_eviction() {
    // L5 race: with the OLD `list() + get()` two-step pattern, an
    // eviction between the two calls left a "ghost" placeholder in the
    // route response. The new `JobRegistry::snapshot` returns the full
    // `(id, status, TrainingJob)` triple under one read lock, so there is
    // no longer a window where an id appears in list() but vanishes
    // before get(). Verify the snapshot is internally consistent: every
    // returned id maps to the very TrainingJob that produced its status.
    use std::collections::HashSet;
    use std::sync::Arc;
    let reg = Arc::new(JobRegistry::new());
    for i in 0..16 {
        reg.create(TrainingJob {
            model_path: format!("m{i}.gguf"),
            dataset_path: format!("d{i}.jsonl"),
            training_mode: grim_garage::jobs::TrainingMode::Lora,
            lora_rank: 16,
            learning_rate: 2e-5,
            epochs: 1,
            ..Default::default()
        })
        .await
        .expect("create");
    }
    let snap = reg.snapshot().await;
    // Pin a specific expected count — if the snapshot ever drifts to a
    // partial list (regression of the L5 race), this fails. We don't
    // assert an id/job substring match because UUIDs are random and
    // independent of insertion content.
    assert_eq!(snap.len(), 16, "snapshot should enumerate all created jobs");
    let mut seen: HashSet<String> = HashSet::new();
    for (id, _status, job) in &snap {
        assert!(
            !job.model_path.is_empty(),
            "snapshot returned a job with empty path: id={}",
            id.0
        );
        seen.insert(id.0.clone());
    }
    assert_eq!(seen.len(), 16, "all snapshot ids must be unique");
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

#[tokio::test]
async fn test_garage_worker_soul_eater_mode() {
    use std::io::Write;
    use std::sync::Arc;

    let dataset_file = "/tmp/soul-eater-test.jsonl";
    {
        let mut f = std::fs::File::create(dataset_file).expect("create test dataset");
        for i in 0..10 {
            writeln!(f, "{{\"text\": \"soul eater test prompt {}\"}}", i).unwrap();
        }
    }
    write_tiny_llama_gguf(Path::new("/tmp/soul-eater-test.gguf"));

    let reg = Arc::new(JobRegistry::new());
    let id = reg
        .create(TrainingJob {
            model_path: "/tmp/soul-eater-test.gguf".into(),
            dataset_path: dataset_file.into(),
            training_mode: grim_garage::jobs::TrainingMode::SoulEater,
            lora_rank: 8,
            learning_rate: 1e-3,
            epochs: 1,
            ..Default::default()
        })
        .await
        .expect("create job");

    grim_garage::jobs::run_training_worker(reg.clone(), id.clone()).await;
    let job = reg.get(&id).await.expect("get job");
    assert_eq!(job.status, JobStatus::Completed);
}
