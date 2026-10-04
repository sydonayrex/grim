//! Loader tests for the Qwen3-VL clip tower.
//!
//! These run against an in-memory provider, not the 927 MB checkpoint, so the
//! tensor-name contract and the load-time gates are exercised on every CI run.
//! The real file is covered by `qwen3vl_clip_real`, which is `#[ignore]`d and
//! run explicitly - a gate that only runs on a machine that happens to have the
//! checkpoint is not a gate.
//!
//! The failure this file exists to prevent: every weight in the tower loading as
//! zeros. That passes every shape check (the shapes come from the file's own
//! header), so shape agreement is not evidence the bytes arrived.

use std::collections::HashMap;

use grim_models_vision::qwen3vl_clip::{assert_non_degenerate, Qwen3VlClipConfig};
use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

struct MemProvider {
    tensors: HashMap<String, (Vec<u8>, Vec<usize>)>,
}

impl TensorProvider for MemProvider {
    fn get(&self, name: &str) -> Result<RawTensor, grim_tensor::error::Error> {
        let (bytes, shape) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing: {name}")))?;
        Ok(RawTensor {
            bytes,
            shape,
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta, grim_tensor::error::Error> {
        let (_, shape) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing: {name}")))?;
        Ok(TensorMeta {
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
            shape,
            fusion_mask: 0,
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// A tiny but structurally complete tower: 2 blocks, hidden 8, 2 heads of
/// dim 4, patch 4, merge 2. Small enough to load in a unit test, and every
/// shape relationship the real checkpoint has is preserved.
fn tiny_config() -> Qwen3VlClipConfig {
    Qwen3VlClipConfig {
        image_size: 16,
        patch_size: 4,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 2,
        num_heads: 2,
        projection_dim: 12,
        spatial_merge_size: 2,
        layer_norm_eps: 1e-6,
        use_gelu: true,
        in_channels: 3,
        merge_area: 4,
        image_mean: vec![0.5, 0.5, 0.5],
        image_std: vec![0.5, 0.5, 0.5],
        deepstack_layers: vec![],
    }
}

/// Non-degenerate filler: a deterministic ramp, so variance is non-zero and a
/// real load is distinguishable from a zero fill.
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 17) as f32 * 0.03125 - 0.25).collect()
}

fn tiny_provider(cfg: &Qwen3VlClipConfig) -> MemProvider {
    let e = cfg.embedding_length;
    let f = cfg.feed_forward_length;
    let mut m: HashMap<String, (Vec<u8>, Vec<usize>)> = HashMap::new();
    let mut put = |name: String, vals: Vec<f32>, shape: Vec<usize>| {
        m.insert(name, (f32_bytes(&vals), shape));
    };

    let patch_in = cfg.in_channels * cfg.patch_size * cfg.patch_size;
    // The two patch kernels are byte-distinct in the real checkpoint, so make
    // them distinct here too: a loader that picked one instead of summing them
    // would still "load 334 tensors" and be wrong.
    put(
        "v.patch_embd.weight".into(),
        ramp(patch_in * e),
        vec![cfg.patch_size, cfg.patch_size, cfg.in_channels, e],
    );
    let second = ramp(patch_in * e)
        .iter()
        .rev()
        .copied()
        .collect::<Vec<f32>>();
    put(
        "v.patch_embd.weight.1".into(),
        second,
        vec![cfg.patch_size, cfg.patch_size, cfg.in_channels, e],
    );
    put("v.patch_embd.bias".into(), ramp(e), vec![e]);
    put(
        "v.position_embd.weight".into(),
        ramp(e * cfg.n_patches()),
        vec![e, cfg.n_patches()],
    );
    put("v.post_ln.weight".into(), ramp(e), vec![e]);
    put("v.post_ln.bias".into(), ramp(e), vec![e]);

    for i in 0..cfg.block_count {
        let p = format!("v.blk.{i}");
        put(
            format!("{p}.attn_qkv.weight"),
            ramp(e * 3 * e),
            vec![e, 3 * e],
        );
        put(format!("{p}.attn_qkv.bias"), ramp(3 * e), vec![3 * e]);
        put(format!("{p}.attn_out.weight"), ramp(e * e), vec![e, e]);
        put(format!("{p}.attn_out.bias"), ramp(e), vec![e]);
        put(format!("{p}.ffn_up.weight"), ramp(e * f), vec![e, f]);
        put(format!("{p}.ffn_up.bias"), ramp(f), vec![f]);
        put(format!("{p}.ffn_down.weight"), ramp(f * e), vec![f, e]);
        put(format!("{p}.ffn_down.bias"), ramp(e), vec![e]);
        put(format!("{p}.ln1.weight"), ramp(e), vec![e]);
        put(format!("{p}.ln1.bias"), ramp(e), vec![e]);
        put(format!("{p}.ln2.weight"), ramp(e), vec![e]);
        put(format!("{p}.ln2.bias"), ramp(e), vec![e]);
    }

    let mi = cfg.merger_in();
    put("mm.0.weight".into(), ramp(mi * mi), vec![mi, mi]);
    put("mm.0.bias".into(), ramp(mi), vec![mi]);
    put(
        "mm.2.weight".into(),
        ramp(mi * cfg.projection_dim),
        vec![mi, cfg.projection_dim],
    );
    put(
        "mm.2.bias".into(),
        ramp(cfg.projection_dim),
        vec![cfg.projection_dim],
    );
    MemProvider { tensors: m }
}

#[test]
fn loads_every_block_and_reports_the_count() {
    let cfg = tiny_config();
    let p = tiny_provider(&cfg);
    let clip = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&p, cfg.clone())
        .expect("tiny tower loads");
    assert_eq!(clip.blocks.len(), cfg.block_count);
    assert_eq!(clip.tensors_loaded, p.tensor_names().len());
}

#[test]
fn missing_second_patch_kernel_is_a_named_error_not_a_panic() {
    let cfg = tiny_config();
    let mut p = tiny_provider(&cfg);
    p.tensors.remove("v.patch_embd.weight.1");
    let err = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&p, cfg)
        .err()
        .expect("a missing kernel must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("v.patch_embd.weight.1"),
        "the error must name the tensor the operator needs to fix: {msg}"
    );
}

#[test]
fn the_two_patch_kernels_are_summed_not_chosen() {
    let cfg = tiny_config();
    let p = tiny_provider(&cfg);
    let clip = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&p, cfg.clone())
        .expect("tiny tower loads");
    let e = cfg.embedding_length;
    let patch_in = cfg.in_channels * cfg.patch_size * cfg.patch_size;
    // Reconstruct what the sum must equal from the provider's own bytes.
    let w0 = f32_from(&p, "v.patch_embd.weight");
    let w1 = f32_from(&p, "v.patch_embd.weight.1");
    for i in 0..(patch_in * e).min(64) {
        let want = w0[i] + w1[i];
        assert!(
            (clip.patch_kernel_summed[i] - want).abs() < 1e-6,
            "kernel[{i}]: got {}, want {want} (the sum of both convs)",
            clip.patch_kernel_summed[i]
        );
    }
    // A loader that used only kernel A would satisfy the first check at i=0
    // for a symmetric ramp but not across the span, so also assert the total
    // differs from either kernel alone.
    let sum: f32 = clip.patch_kernel_summed.iter().sum();
    let only0: f32 = w0.iter().sum();
    let only1: f32 = w1.iter().sum();
    assert_ne!(sum, only0, "must not be kernel A alone");
    assert_ne!(sum, only1, "must not be kernel B alone");
}

fn f32_from(p: &MemProvider, name: &str) -> Vec<f32> {
    let (bytes, _) = &p.tensors[name];
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn an_all_zero_tensor_is_refused_by_the_degeneracy_gate() {
    // This is the failure mode the gate exists for: shapes match, bytes are
    // all zero, and nothing else in the load complains.
    let zeros = vec![0.0f32; 128];
    assert!(assert_non_degenerate("fake.zero", &zeros).is_err());
    assert!(assert_non_degenerate("fake.ok", &ramp(128)).is_ok());
}

#[test]
fn a_zero_weight_matrix_fails_the_load_rather_than_producing_zeros() {
    let cfg = tiny_config();
    let mut p = tiny_provider(&cfg);
    // Zero only mm.0.weight, with a byte length that matches its declared
    // element count. A wrong byte length would be caught by the decode, which
    // is a different (and also good) failure - the point of this test is the
    // variance gate, so the fixture must otherwise be valid.
    let elems = cfg.merger_in() * cfg.merger_in();
    let shape = p.tensors["mm.0.weight"].1.clone();
    p.tensors
        .insert("mm.0.weight".into(), (f32_bytes(&vec![0.0; elems]), shape));
    let err = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&p, cfg)
        .err()
        .expect("an all-zero merger weight must fail the load");
    let msg = err.to_string();
    assert!(
        msg.contains("mm.0.weight") && msg.contains("degenerate"),
        "the error must name the tensor and the reason: {msg}"
    );
}

/// Every tensor the loader variance-gates, zeroed one at a time.
///
/// A single zeroed tensor only proves that ONE gate works; the rest could be
/// deleted and this test would still pass. Mutation testing showed exactly
/// that: dropping the mm.2 / position / patch-bias / block gates survived while
/// only mm.0 was exercised. This walks the whole gated set instead.
#[test]
fn every_gated_tensor_is_caught_when_zeroed() {
    let cfg = tiny_config();
    let base = tiny_provider(&cfg);
    let e = cfg.embedding_length;
    let f = cfg.feed_forward_length;
    let patch_elems = cfg.in_channels * cfg.patch_size * cfg.patch_size * e;

    // (tensor name, expected element count)
    let gated: Vec<(&str, usize)> = vec![
        ("v.patch_embd.weight", patch_elems),
        ("v.patch_embd.weight.1", patch_elems),
        ("v.patch_embd.bias", e),
        ("v.position_embd.weight", e * cfg.n_patches()),
        ("mm.0.weight", cfg.merger_in() * cfg.merger_in()),
        ("mm.2.weight", cfg.merger_in() * cfg.projection_dim),
        ("v.blk.0.attn_qkv.weight", e * 3 * e),
        ("v.blk.0.attn_out.weight", e * e),
        ("v.blk.0.ffn_up.weight", e * f),
        ("v.blk.0.ffn_down.weight", f * e),
        ("v.blk.1.attn_qkv.weight", e * 3 * e),
        ("v.blk.1.ffn_down.weight", f * e),
    ];

    for (name, elems) in gated {
        // Confirm the fixture is sized as the gate expects before zeroing it,
        // so a failure below is about the gate and not a malformed fixture.
        assert_eq!(
            base.tensors[name].1.iter().product::<usize>(),
            elems,
            "fixture shape for {name}"
        );
        let mut p = MemProvider {
            tensors: base.tensors.clone(),
        };
        let shape = p.tensors[name].1.clone();
        p.tensors
            .insert(name.to_string(), (f32_bytes(&vec![0.0; elems]), shape));

        let err = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&p, cfg.clone())
            .err()
            .unwrap_or_else(|| panic!("zeroing {name} must fail the load"));
        let msg = err.to_string();
        assert!(
            msg.contains("degenerate"),
            "zeroing {name} must be refused as degenerate, got: {msg}"
        );
    }
}

/// A position table whose entry count disagrees with the declared geometry is
/// refused by name.
///
/// This is the reachable form of the "is the position grid square" question:
/// `n_patches` is `per_side^2` by construction so a config-derived grid is
/// always square, but the FILE's table can still carry a different count, and
/// the forward pass indexes it as `embedding_length * n_patches`.
#[test]
fn position_table_count_must_match_the_declared_geometry() {
    let cfg = tiny_config();
    let mut p = tiny_provider(&cfg);
    // The fixture's table has 16 entries (4x4). Give it a prime count instead,
    // with a byte length that matches, so the shape check passes and the
    // disagreement is purely geometric.
    let odd = 15;
    let shape = vec![cfg.embedding_length, odd];
    p.tensors.insert(
        "v.position_embd.weight".into(),
        (f32_bytes(&ramp(cfg.embedding_length * odd)), shape),
    );
    let err = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&p, cfg)
        .err()
        .expect("a position table that disagrees with the geometry must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("v.position_embd.weight"),
        "the error must name the tensor whose count is wrong: {msg}"
    );
}

/// An unsupported configuration is refused before a single tensor is read.
///
/// DeepStack changes the projector output width, so loading the weights and only
/// then refusing would burn a 927 MB read to arrive at the same conclusion.
#[test]
fn unsupported_config_is_refused_before_any_tensor_read() {
    let cfg = tiny_config();
    let mut deepstack = cfg.clone();
    deepstack.deepstack_layers = vec![8, 16, 24];
    // A provider with nothing in it at all: if the load touches a tensor before
    // the support gate, the error names a missing tensor instead of DeepStack.
    let empty = MemProvider {
        tensors: HashMap::new(),
    };
    let err = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&empty, deepstack)
        .err()
        .expect("a deepstack checkpoint must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("DeepStack") && !msg.contains("missing"),
        "refusal must come from the support gate, not from a missing tensor: {msg}"
    );
}

#[test]
fn merger_output_width_must_match_the_declared_projection_dim() {
    let cfg = tiny_config();
    let p = tiny_provider(&cfg);
    let clip = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&p, cfg.clone())
        .expect("tiny tower loads");
    assert_eq!(clip.merger_out_cols, cfg.projection_dim);
    assert_eq!(clip.merger_in_cols, cfg.merger_in());
}

#[test]
fn merged_token_count_matches_the_merge_arithmetic() {
    let cfg = tiny_config();
    let p = tiny_provider(&cfg);
    let clip = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&p, cfg.clone())
        .expect("tiny tower loads");
    assert_eq!(clip.merged_tokens, cfg.merged_tokens());
    assert_eq!(clip.merged_tokens, (16 / 4) * (16 / 4) / 4);
}

/// The real checkpoint, loaded end to end. `#[ignore]`d so CI stays fast and
/// so its absence is a skip, never a silent pass.
#[test]
#[ignore = "requires models/qwen38-27b/mmproj-F16.gguf"]
fn qwen3vl_clip_real() {
    use grim_format::tprov::GgufProvider;
    // `cargo test` runs with CWD set to the crate directory, so walk up to the
    // workspace root rather than assuming it. A silently-skipped real-file test
    // is indistinguishable from a passing one unless the skip is loud.
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut path: Option<std::path::PathBuf> = None;
    for _ in 0..4 {
        let candidate = dir.join("models/qwen38-27b/mmproj-F16.gguf");
        if candidate.exists() {
            path = Some(candidate);
            break;
        }
        dir = dir.join("..");
    }
    let Some(path) = path else {
        panic!(
            "mmproj-F16.gguf not found walking up from {}; this test must either \
             find the checkpoint or fail, not pass by absence",
            env!("CARGO_MANIFEST_DIR")
        );
    };
    println!("real mmproj: {}", path.display());
    let provider = GgufProvider::open(path.to_str().expect("utf-8 path")).expect("mmproj opens");
    let cfg = Qwen3VlClipConfig::from_provider(&provider).expect("geometry parses");
    cfg.ensure_supported()
        .expect("no deepstack in this checkpoint");
    let clip = grim_models_vision::qwen3vl_clip::Qwen3VlClip::load(&provider, cfg.clone())
        .expect("real tower loads");

    assert_eq!(clip.blocks.len(), 27, "block_count from metadata");
    assert_eq!(clip.merged_tokens, 576, "2304 patches / 4");
    assert_eq!(
        clip.merger_out_cols, 5120,
        "matches Qwen3.8-27B hidden size"
    );

    // The gate's own report. If this prints var=0, the mmap/slice path is
    // wrong and nothing downstream can be trusted.
    let n = clip.patch_kernel_summed.len();
    let mean = clip.patch_kernel_summed.iter().sum::<f32>() / n as f32;
    let var = clip
        .patch_kernel_summed
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f32>()
        / n as f32;
    println!(
        "loaded {} tensors, patch_kernel_summed n={n} var={var}",
        clip.tensors_loaded
    );
    assert_eq!(n, 16 * 16 * 3 * 1152, "patch kernel element count");
    assert!(var > 1e-12, "real weights must not be all zero (var={var})");
}
