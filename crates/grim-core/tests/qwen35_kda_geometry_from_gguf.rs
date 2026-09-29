//! Is the KDA geometry the model runs with actually taken from the GGUF, or do
//! the literals in the Qwen35 loader arms quietly decide it?
//!
//! `model_loader.rs` builds the Qwen35 config as
//! `hparams.ssm_*.unwrap_or(<literal>)` — for `ssm_dt_rank` the literal is 48,
//! which is the 27B's value. That is safe ONLY if `hparams` is genuinely
//! populated from the checkpoint, and nothing in the type system enforces it:
//! a renamed metadata key would leave `None`, the 27B geometry would be handed
//! to a 9B, and the forward would run at the wrong shape while still emitting
//! plausible tokens.
//!
//! The 9B and the 27B genuinely differ on this field (32 vs 48), so a test
//! that asserted 48 would pass on one model and bless the other's corruption.
//! This asserts the geometry is file-derived AND cross-checks each derived
//! dimension against the tensor that actually consumes it, so a substituted
//! literal fails on a width rather than producing a wrong-shaped forward.
//!
//! CPU-only: it reads the GGUF header and metadata, and touches no device.

use grim_core::architecture::ModelArchitecture;
use grim_core::hyperparams::{HyperparameterExtractor, MetadataLookup};
use grim_format::tprov::GgufProvider;

/// `GgufProvider` is not itself a `MetadataLookup`; the loader wraps it in
/// `GgufMetadataLookup`, and this mirrors that wrapper exactly (same pattern as
/// `qwen4exp_config_metadata_test.rs`). Using the loader's real adapter is the
/// point: a test that read metadata through a different path could disagree
/// with production for reasons that have nothing to do with geometry.
struct GgufLookup<'a>(&'a GgufProvider);

impl MetadataLookup for GgufLookup<'_> {
    fn get_str(&self, key: &str) -> Option<String> {
        self.0.metadata(key)?.as_str().map(|s| s.to_string())
    }
    fn get_u32(&self, key: &str) -> Option<u32> {
        self.0.metadata(key)?.as_u32()
    }
    fn get_f32(&self, key: &str) -> Option<f32> {
        self.0.metadata(key)?.as_f32()
    }
    fn get_array_len(&self, key: &str) -> Option<usize> {
        self.0.metadata(key)?.as_array().map(|a| a.len())
    }
    fn get_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        self.0.metadata(key)?.as_u32_array()
    }
    fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        self.0.metadata(key)?.as_i32_array()
    }
    fn get_u64(&self, key: &str) -> Option<u64> {
        self.0.metadata(key)?.as_u32().map(|u| u as u64)
    }
    fn get_u64_array(&self, key: &str) -> Option<Vec<u64>> {
        self.0
            .metadata(key)?
            .as_u32_array()
            .map(|v| v.into_iter().map(|u| u as u64).collect())
    }
}

fn find_checkpoint(rel: &str) -> Option<String> {
    if let Ok(p) = std::env::var("GRIM_CHECKPOINT") {
        return Some(p);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    for p in [
        format!("{home}/.grim/models/{rel}"),
        format!("/drive/bigfast/grim/models/qwen35-9b/{rel}"),
    ] {
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    eprintln!("[SKIP] no checkpoint for {rel} (set GRIM_CHECKPOINT)");
    None
}

/// `tensors()` yields `(name, info)` pairs, not bare infos.
fn tensor_shape(provider: &GgufProvider, suffix: &str) -> Vec<usize> {
    provider
        .tensors()
        .iter()
        .find(|(name, _)| name.ends_with(suffix))
        .unwrap_or_else(|| panic!("no tensor ending in {suffix}"))
        .1
        .shape()
        .to_vec()
}

/// Assert the loader's KDA geometry is file-derived and self-consistent with
/// the tensors that consume it.
fn check(path: &str) {
    let provider = GgufProvider::open(path).expect("open gguf");
    let arch = ModelArchitecture::Qwen35;
    let hp = HyperparameterExtractor::extract(arch, &GgufLookup(&provider));

    // Each of these must be Some. A None means `unwrap_or(<literal>)` in
    // model_loader.rs fires and the 27B geometry reaches a 9B (or vice versa).
    let n_v_heads = hp
        .ssm_dt_rank
        .expect("ssm_dt_rank None -> loader uses literal 48");
    let n_k_heads = hp
        .ssm_n_group
        .expect("ssm_n_group None -> loader uses literal 16");
    let head_dim = hp
        .ssm_d_state
        .expect("ssm_d_state None -> loader uses literal 128");
    let taps = hp
        .ssm_d_conv
        .expect("ssm_d_conv None -> loader uses literal 4");
    let interval = hp
        .full_attention_interval
        .expect("full_attention_interval None -> loader uses literal 4");

    // Exactly the derivations in gated_delta_net_forward_d2d / the host ref.
    let key_dim = n_k_heads * head_dim;
    let value_dim = n_v_heads * head_dim;
    let conv_dim = 2 * key_dim + value_dim;

    println!("== {path}");
    println!("   n_v_heads (ssm_dt_rank)  = {n_v_heads}");
    println!("   n_k_heads (ssm_n_group)  = {n_k_heads}");
    println!("   head_dim  (ssm_d_state)  = {head_dim}");
    println!("   d_conv                   = {taps}");
    println!("   full_attention_interval  = {interval}");
    println!("   derived key_dim={key_dim} value_dim={value_dim} conv_dim={conv_dim}");

    // `ssm_conv1d.weight` is [conv_dim, d_conv]. A substituted value-head count
    // makes conv_dim disagree with the file, and the short conv would read the
    // wrong channels — the same class as the [q|k|v] layout defect.
    let conv = tensor_shape(&provider, "ssm_conv1d.weight");
    println!("   ssm_conv1d.weight        = {conv:?}");
    assert_eq!(
        conv[0], conv_dim,
        "derived conv_dim != checkpoint conv width"
    );
    assert_eq!(conv[1], taps, "derived d_conv != checkpoint conv taps");

    // `attn_qkv` out-width on a recurrent layer is the same conv_dim.
    let qkv = tensor_shape(&provider, "attn_qkv.weight");
    println!("   attn_qkv.weight          = {qkv:?}");
    assert_eq!(
        qkv[0], conv_dim,
        "derived conv_dim != attn_qkv out-width: the recurrent stream would be read wrong"
    );

    // `ssm_norm` is the per-HEAD weight: head_dim wide, NOT value_dim. This is
    // the invariant the D2D guard enforces after the width fix; asserting it
    // here ties it to the real file so it cannot regress.
    let norm = tensor_shape(&provider, "ssm_norm.weight");
    println!("   ssm_norm.weight          = {norm:?}");
    assert_eq!(
        norm[0], head_dim,
        "ssm_norm must be head_dim wide, not value_dim — this is the D2D guard's requirement"
    );

    // `ssm_dt.bias` / `ssm_a` are per-VALUE-HEAD, so n_v_heads wide. This is
    // the check that actually catches a substituted literal: a 27B's 48
    // against a 9B's 32-wide bias is a hard mismatch.
    let dt_bias = tensor_shape(&provider, "ssm_dt.bias");
    let ssm_a = tensor_shape(&provider, "ssm_a");
    println!("   ssm_dt.bias              = {dt_bias:?}");
    println!("   ssm_a                    = {ssm_a:?}");
    assert_eq!(
        dt_bias[0], n_v_heads,
        "ssm_dt.bias width disagrees with ssm_dt_rank — the loader is using a literal"
    );
    assert_eq!(
        ssm_a[0], n_v_heads,
        "ssm_a width disagrees with ssm_dt_rank"
    );

    // Cross-check attention layer geometry against full-attention tensors (e.g. blk.3)
    let num_heads = hp.num_heads;
    let num_kv_heads = hp.num_kv_heads;
    let attn_head_dim = hp.head_dim;
    let q_dim = num_heads * attn_head_dim;
    let kv_dim = num_kv_heads * attn_head_dim;

    println!("   attn num_heads           = {num_heads}");
    println!("   attn num_kv_heads        = {num_kv_heads}");
    println!("   attn head_dim            = {attn_head_dim}");
    println!("   derived q_dim={q_dim} kv_dim={kv_dim}");

    // attn_q on attention layers is fused [Q | gate] at 2 * q_dim
    let attn_q = tensor_shape(&provider, "blk.3.attn_q.weight");
    println!("   blk.3.attn_q.weight      = {attn_q:?}");
    assert_eq!(
        attn_q[0], 2 * q_dim,
        "blk.3.attn_q width != 2 * q_dim"
    );

    // attn_k, attn_v are kv_dim wide
    let attn_k = tensor_shape(&provider, "blk.3.attn_k.weight");
    let attn_v = tensor_shape(&provider, "blk.3.attn_v.weight");
    println!("   blk.3.attn_k.weight      = {attn_k:?}");
    println!("   blk.3.attn_v.weight      = {attn_v:?}");
    assert_eq!(attn_k[0], kv_dim, "blk.3.attn_k width != kv_dim");
    assert_eq!(attn_v[0], kv_dim, "blk.3.attn_v width != kv_dim");

    // attn_output in-width is q_dim
    let attn_out = tensor_shape(&provider, "blk.3.attn_output.weight");
    println!("   blk.3.attn_output.weight = {attn_out:?}");
    assert_eq!(attn_out[1], q_dim, "blk.3.attn_output in-width != q_dim");

    // attn_q_norm and attn_k_norm are head_dim wide
    let q_norm = tensor_shape(&provider, "blk.3.attn_q_norm.weight");
    let k_norm = tensor_shape(&provider, "blk.3.attn_k_norm.weight");
    println!("   blk.3.attn_q_norm.weight = {q_norm:?}");
    println!("   blk.3.attn_k_norm.weight = {k_norm:?}");
    assert_eq!(q_norm[0], attn_head_dim, "blk.3.attn_q_norm width != attn_head_dim");
    assert_eq!(k_norm[0], attn_head_dim, "blk.3.attn_k_norm width != attn_head_dim");

    println!("   OK: all KDA and attention dims file-derived and cross-checked\n");
}

#[test]
fn kda_geometry_is_file_derived_and_cross_checked() {
    for rel in ["Qwen3.5-9B-Q4_K_M.gguf", "Qwen3.8-27B-Q4_K_M.gguf"] {
        if let Some(p) = find_checkpoint(rel) {
            check(&p);
        }
    }
}

/// The two supported sizes genuinely disagree on value-head count. That
/// disagreement is the whole reason the literal fallback is dangerous, so
/// assert it explicitly rather than let the premise rot unnoticed.
#[test]
fn the_two_sizes_really_do_differ_on_value_head_count() {
    let mut seen = Vec::new();
    for rel in ["Qwen3.5-9B-Q4_K_M.gguf", "Qwen3.8-27B-Q4_K_M.gguf"] {
        let Some(p) = find_checkpoint(rel) else {
            continue;
        };
        let provider = GgufProvider::open(&p).expect("open gguf");
        let hp =
            HyperparameterExtractor::extract(ModelArchitecture::Qwen35, &GgufLookup(&provider));
        seen.push((rel, hp.ssm_dt_rank.expect("ssm_dt_rank")));
    }
    if seen.len() < 2 {
        eprintln!(
            "[SKIP] only {} checkpoint(s) present; nothing to compare",
            seen.len()
        );
        return;
    }
    println!("value-head counts across sizes: {seen:?}");
    assert_ne!(
        seen[0].1, seen[1].1,
        "expected the 9B and 27B to differ on ssm_dt_rank; if they no longer do, \
         this guard no longer tests what it claims to"
    );
}
