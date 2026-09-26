//! `SplitGgufProvider` resolution tests.
//!
//! Covers the three GGUF packaging shapes that actually occur in the wild:
//! monolithic (no `split.*` keys), dual shard, and N-shard (3..6+), plus the
//! two ways a naive directory scan goes wrong: a non-conforming sibling file
//! sitting next to the model (a projector shard, a different architecture), and
//! a companion set whose union does not match `split.tensors.count`.
//!
//! Synthetic shards are used rather than real checkpoints so the matrix is cheap
//! and deterministic. The real 2-shard Qwen3.8 pair is covered by
//! `gguf_split_loader_real_checkpoint_test` (ignored unless the model is present).

use std::io::Write;
use std::path::{Path, PathBuf};

use grim_format::gguf::{GGUF_MAGIC, GGUF_VERSION, GgufDType, GgufValue};
use grim_format::tprov::SplitGgufProvider;
use grim_tensor::provider::TensorProvider;

// ---- GGUF byte-stream writer -------------------------------------------

fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn push_value(buf: &mut Vec<u8>, v: &GgufValue) {
    match v {
        GgufValue::Uint32(x) => {
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        GgufValue::Float32(x) => {
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&x.to_le_bytes());
        }
        GgufValue::Bool(x) => {
            buf.extend_from_slice(&7u32.to_le_bytes());
            buf.extend_from_slice(&u8::from(*x).to_le_bytes());
        }
        GgufValue::String(s) => {
            buf.extend_from_slice(&8u32.to_le_bytes());
            push_string(buf, s);
        }
        GgufValue::Array(items) => {
            let elem_tag = match items.first() {
                Some(GgufValue::Uint32(_)) => 4u32,
                Some(GgufValue::Bool(_)) => 7u32,
                Some(GgufValue::String(_)) => 8u32,
                _ => panic!("unsupported array element type: {items:?}"),
            };
            buf.extend_from_slice(&9u32.to_le_bytes());
            buf.extend_from_slice(&elem_tag.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                match item {
                    GgufValue::Uint32(x) => buf.extend_from_slice(&x.to_le_bytes()),
                    GgufValue::Bool(x) => buf.extend_from_slice(&u8::from(*x).to_le_bytes()),
                    GgufValue::String(s) => push_string(buf, s),
                    other => panic!("unsupported array element {other:?}"),
                }
            }
        }
        other => panic!("unsupported metadata value: {other:?}"),
    }
}

/// One synthetic tensor: a name, F32 dims, and a byte payload big enough to
/// satisfy `GgufProvider`'s size accounting.
struct SynthTensor {
    name: String,
    dims: Vec<u64>,
}

impl SynthTensor {
    fn new(name: &str, dims: &[u64]) -> Self {
        Self {
            name: name.to_string(),
            dims: dims.to_vec(),
        }
    }

    fn payload_bytes(&self) -> usize {
        let params: u64 = self.dims.iter().product();
        (params * 4) as usize
    }
}

/// Serialize a GGUF v3 file with the given metadata and F32 tensor records.
///
/// Real (zero-filled) payloads are written so `GgufProvider`'s dtype-vs-payload-size
/// guard accepts the file; the shapes are kept tiny so that costs nothing.
fn write_gguf(path: &Path, metadata: &[(String, GgufValue)], tensors: &[SynthTensor]) {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        push_string(&mut buf, k);
        push_value(&mut buf, v);
    }

    // Tensor-info section: name, rank, dims, dtype tag, offset into data section.
    let mut offset: u64 = 0;
    for t in tensors {
        push_string(&mut buf, &t.name);
        buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&(GgufDType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        offset += t.payload_bytes() as u64;
    }

    // Data section starts 32-byte aligned past the header.
    let aligned = (buf.len() + 31) & !31;
    buf.resize(aligned, 0);
    for t in tensors {
        buf.resize(buf.len() + t.payload_bytes(), 0);
    }

    let mut f = std::fs::File::create(path).expect("create gguf file");
    f.write_all(&buf).expect("write gguf file");
}

/// Temp directory that lives for the duration of the test.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let base = std::env::temp_dir().join(format!("grim_split_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create scratch dir");
        Self(base)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn write(
        &self,
        name: &str,
        metadata: &[(String, GgufValue)],
        tensors: &[SynthTensor],
    ) -> PathBuf {
        let p = self.path(name);
        write_gguf(&p, metadata, tensors);
        p
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn meta_arch(arch: &str) -> (String, GgufValue) {
    (
        "general.architecture".to_string(),
        GgufValue::String(arch.into()),
    )
}

fn meta_split(no: u32, count: u32, tensors: u32) -> Vec<(String, GgufValue)> {
    vec![
        meta_arch("qwen4exp"),
        ("split.no".to_string(), GgufValue::Uint32(no)),
        ("split.count".to_string(), GgufValue::Uint32(count)),
        (
            "split.tensors.count".to_string(),
            GgufValue::Uint32(tensors),
        ),
    ]
}

fn path_str(p: &Path) -> String {
    p.to_str().expect("utf-8 path").to_string()
}

// ---- Monolithic --------------------------------------------------------

#[test]
fn monolithic_gguf_resolves_all_tensors_from_one_shard() {
    let dir = Scratch::new("mono");
    let tensors = vec![
        SynthTensor::new("blk.0.attn_qkv.weight", &[16, 8]),
        SynthTensor::new("blk.0.ffn_gate_inp.weight", &[8, 8]),
        SynthTensor::new("per_layer_token_embd.weight", &[64, 8]),
    ];
    // No split.* keys at all — the single-file download shape.
    let p = dir.write("monolithic.gguf", &[meta_arch("qwen4exp")], &tensors);

    let provider = SplitGgufProvider::open(&path_str(&p)).expect("monolithic must open");
    assert_eq!(provider.shard_count(), 1, "monolithic file is one shard");
    assert_eq!(provider.total_expected_tensors(), 3);
    assert_eq!(provider.tensor_names().len(), 3);

    for t in &tensors {
        let m = provider
            .meta(&t.name)
            .unwrap_or_else(|e| panic!("tensor {} must resolve: {e}", t.name));
        // GGUF writes dims in file order; `TensorMeta.shape` follows GGML's
        // column-major convention and is the reverse.
        let want: Vec<usize> = t.dims.iter().rev().map(|d| *d as usize).collect();
        assert_eq!(m.shape, want, "dims for {}", t.name);
    }
}

#[test]
fn monolithic_gguf_ignores_neighbouring_unrelated_files() {
    let dir = Scratch::new("mono_sibling");
    let tensors = vec![SynthTensor::new("blk.0.ssm_out.weight", &[8, 16])];
    let p = dir.write("monolithic.gguf", &[meta_arch("qwen4exp")], &tensors);
    // A different architecture sitting in the same directory must not merge.
    dir.write(
        "projector.gguf",
        &[meta_arch("clip")],
        &[SynthTensor::new("mm.0.weight", &[8, 8])],
    );

    let provider = SplitGgufProvider::open(&path_str(&p)).expect("must open");
    assert_eq!(provider.shard_count(), 1);
    assert!(
        provider.meta("mm.0.weight").is_err(),
        "clip shard must be excluded"
    );
    assert!(provider.meta("blk.0.ssm_out.weight").is_ok());
}

// ---- Two shards --------------------------------------------------------

#[test]
fn two_shard_model_resolves_backbone_and_embedding_tensors() {
    let dir = Scratch::new("two");
    let main = dir.write(
        "main.gguf",
        &meta_split(0, 2, 5),
        &[
            SynthTensor::new("blk.0.attn_qkv.weight", &[16, 8]),
            SynthTensor::new("blk.0.hc_attn_norm.weight", &[4]),
            SynthTensor::new("output.weight", &[32, 8]),
        ],
    );
    dir.write(
        "emb.gguf",
        &meta_split(1, 2, 5),
        &[
            SynthTensor::new("per_layer_token_embd.weight", &[64, 8]),
            SynthTensor::new("token_embd.weight", &[32, 8]),
        ],
    );

    let provider = SplitGgufProvider::open(&path_str(&main)).expect("2-shard must open");
    assert_eq!(provider.shard_count(), 2);
    assert_eq!(provider.total_expected_tensors(), 5);
    assert_eq!(provider.tensor_names().len(), 5);
    assert!(provider.meta("blk.0.attn_qkv.weight").is_ok());
    assert!(provider.meta("per_layer_token_embd.weight").is_ok());
}

#[test]
fn two_shard_model_rejects_missing_companion_shard() {
    let dir = Scratch::new("two_missing");
    // Declares 2 shards but the companion was never downloaded.
    let main = dir.write(
        "main.gguf",
        &meta_split(0, 2, 3),
        &[SynthTensor::new("blk.0.attn_qkv.weight", &[16, 8])],
    );

    let err = SplitGgufProvider::open(&path_str(&main))
        .expect_err("a declared-but-absent companion shard must be a hard error");
    let msg = format!("{err}");
    assert!(
        msg.contains("expected 2 shards"),
        "error should name the shard shortfall, got: {msg}"
    );
}

#[test]
fn two_shard_model_rejects_tensor_count_mismatch() {
    let dir = Scratch::new("two_miscount");
    // Header claims 9 tensors across the split; the two files supply 3.
    let main = dir.write(
        "main.gguf",
        &meta_split(0, 2, 9),
        &[SynthTensor::new("blk.0.attn_qkv.weight", &[16, 8])],
    );
    dir.write(
        "emb.gguf",
        &meta_split(1, 2, 9),
        &[
            SynthTensor::new("per_layer_token_embd.weight", &[64, 8]),
            SynthTensor::new("token_embd.weight", &[32, 8]),
        ],
    );

    let err = SplitGgufProvider::open(&path_str(&main))
        .expect_err("union must match split.tensors.count");
    let msg = format!("{err}");
    assert!(
        msg.contains("expected 9 tensors"),
        "error should name the tensor shortfall, got: {msg}"
    );
}

// ---- N shards (3..6+) --------------------------------------------------

#[test]
fn six_shard_model_resolves_every_split_in_order() {
    let dir = Scratch::new("six");
    let n = 6u32;
    let total = n; // one tensor per shard
    let mut main = None;
    for i in 0..n {
        let p = dir.write(
            &format!("part{i}.gguf"),
            &meta_split(i, n, total),
            &[SynthTensor::new(
                &format!("blk.{i}.ssm_out.weight"),
                &[8, 16],
            )],
        );
        if i == 0 {
            main = Some(p);
        }
    }

    let provider = SplitGgufProvider::open(&path_str(&main.unwrap())).expect("6-shard must open");
    assert_eq!(provider.shard_count(), 6);
    assert_eq!(provider.tensor_names().len(), 6);
    for i in 0..n as usize {
        assert!(
            provider.meta(&format!("blk.{i}.ssm_out.weight")).is_ok(),
            "shard {i} tensor must resolve"
        );
    }
}

#[test]
fn n_shard_model_opens_from_a_non_zero_shard_index() {
    // The user may point at shard 3 of 6. Resolution must still find 0..5.
    let dir = Scratch::new("nonzero");
    let n = 6u32;
    for i in 0..n {
        dir.write(
            &format!("part{i}.gguf"),
            &meta_split(i, n, n),
            &[SynthTensor::new(
                &format!("blk.{i}.ssm_out.weight"),
                &[8, 16],
            )],
        );
    }

    let provider =
        SplitGgufProvider::open(&path_str(&dir.path("part3.gguf"))).expect("non-zero shard entry");
    assert_eq!(provider.shard_count(), 6);
    assert_eq!(provider.tensor_names().len(), 6);
    assert!(provider.meta("blk.0.ssm_out.weight").is_ok());
}

#[test]
fn n_shard_model_rejects_a_gap_in_the_split_index_range() {
    let dir = Scratch::new("gap");
    let n = 4u32;
    for i in [0u32, 1, 3] {
        // shard 2 intentionally absent
        dir.write(
            &format!("part{i}.gguf"),
            &meta_split(i, n, n),
            &[SynthTensor::new(
                &format!("blk.{i}.ssm_out.weight"),
                &[8, 16],
            )],
        );
    }

    let err = SplitGgufProvider::open(&path_str(&dir.path("part0.gguf")))
        .expect_err("a hole in the split index range must be a hard error");
    let msg = format!("{err}");
    assert!(
        msg.contains("shards") || msg.contains("split index"),
        "error should describe the split shortfall, got: {msg}"
    );
}

#[test]
fn extra_architecture_matched_sibling_beyond_split_count_is_rejected() {
    // A second model of the SAME architecture in the directory must not be
    // silently absorbed: split.count bounds how many shards belong to this model.
    let dir = Scratch::new("extra_same_arch");
    let main = dir.write(
        "main.gguf",
        &meta_split(0, 2, 2),
        &[
            SynthTensor::new("blk.0.attn_qkv.weight", &[16, 8]),
            SynthTensor::new("blk.0.hc_attn_norm.weight", &[4]),
        ],
    );
    dir.write(
        "emb.gguf",
        &meta_split(1, 2, 2),
        &[SynthTensor::new("per_layer_token_embd.weight", &[64, 8])],
    );
    // A *different* model, same architecture, also numbered shard 0/1.
    dir.write(
        "other.gguf",
        &meta_split(0, 2, 2),
        &[SynthTensor::new("totally_other.weight", &[4, 4])],
    );

    // Either the stray file is refused (tensor-count mismatch) or it is not
    // merged at all. What must never happen is a silent success with the
    // stranger's tensor visible under this model's namespace.
    match SplitGgufProvider::open(&path_str(&main)) {
        Ok(p) => {
            assert!(
                p.meta("totally_other.weight").is_err(),
                "an unrelated same-arch file must not leak into this model's namespace"
            );
        }
        Err(_) => { /* rejecting the ambiguous directory is also acceptable */ }
    }
}

// ---- Real checkpoint (opt-in) ------------------------------------------

/// The genuine 2-shard Qwen3.8 pair, when it is on disk.
#[test]
fn gguf_split_loader_real_checkpoint() {
    let main = "models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf";
    if !Path::new(main).exists() {
        eprintln!("skipping: {main} not present");
        return;
    }
    let provider = SplitGgufProvider::open(main).expect("real 2-shard model must open");
    assert_eq!(provider.architecture(), Some("qwen4exp"));
    assert_eq!(provider.shard_count(), 2);
    assert_eq!(provider.total_expected_tensors(), 1224);
    assert_eq!(provider.tensor_names().len(), 1224);
    assert!(provider.meta("blk.0.attn_qkv.weight").is_ok());
    assert!(provider.meta("per_layer_token_embd.weight").is_ok());
    assert!(provider.meta("token_embd.weight").is_ok());
    // The CLIP projector shard in the same directory must not be merged.
    assert!(
        !provider.tensor_names().iter().any(|n| n.starts_with("mm.")),
        "mmproj shard leaked into the language-model namespace"
    );
}
