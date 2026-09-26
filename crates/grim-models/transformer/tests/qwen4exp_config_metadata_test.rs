//! `Qwen38FlashNextConfig` must be derived from the checkpoint, not guessed.
//!
//! The previous loader hardcoded several fields and got them wrong for the
//! released GSQ-RCO file: `layer_types` was empty, `ngram_dim` was 512 instead
//! of the real 160-wide PLE embedding, `mrope_section` dropped the trailing 0,
//! `partial_rotary_factor` was forced to 1.0, and the SSM geometry was inferred
//! from the wrong quantity. Every one of those yields a model that loads,
//! emits finite logits, and is numerically wrong — so this test asserts the
//! values against the real GGUF header.

use grim_core::hyperparams::MetadataLookup;
use grim_models_transformer::qwen38_flash_next::Qwen38FlashNextConfig;

/// A `MetadataLookup` over the parsed real checkpoint.
///
/// The provider is leaked deliberately: this test opens a 47 GB file's index
/// exactly once, and a `&'static` keeps the borrow out of the way.
fn lookup_real() -> Option<GgufLookup> {
    // `cargo test` runs with CWD set to the package root, not the workspace
    // root, so resolve the model relative to the workspace explicitly. A
    // relative "models/..." path silently skipped this test, which is exactly
    // the failure mode it exists to prevent.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf");
    if !path.exists() {
        return None;
    }
    let path = path.to_str().expect("utf-8 path");
    let provider = grim_format::tprov::GgufProvider::open(path).expect("open checkpoint");
    Some(GgufLookup(Box::leak(Box::new(provider))))
}

struct GgufLookup(&'static grim_format::tprov::GgufProvider);

impl MetadataLookup for GgufLookup {
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

#[test]
fn config_from_real_checkpoint_matches_its_header() {
    let Some(m) = lookup_real() else {
        eprintln!("skipping: checkpoint not present");
        return;
    };
    let cfg = Qwen38FlashNextConfig::from_qwen4exp_metadata(&m);

    // Scalar hyperparameters, verbatim from the parsed header.
    assert_eq!(cfg.num_layers, 48, "block_count");
    assert_eq!(cfg.hidden_size, 2560, "embedding_length");
    assert_eq!(cfg.num_heads, 24, "attention.head_count");
    assert_eq!(cfg.num_kv_heads, 2, "attention.head_count_kv");
    assert_eq!(cfg.head_dim, 256, "attention.key_length");
    assert_eq!(cfg.value_head_dim, 256, "attention.value_length");
    assert_eq!(cfg.num_experts, 512, "expert_count");
    assert_eq!(cfg.num_experts_per_tok, 10, "expert_used_count");
    assert_eq!(cfg.intermediate_size, 640, "expert_feed_forward_length");
    assert_eq!(
        cfg.shared_expert_intermediate_size,
        Some(640),
        "expert_shared_feed_forward_length"
    );
    assert_eq!(cfg.hc_count, 4, "hyper_connection.count");
    assert_eq!(cfg.hc_lowrank, 320, "hyper_connection.low_rank");
    assert_eq!(cfg.indexer_top_k, 2048, "attention.indexer.top_k");
    assert_eq!(cfg.max_seq_len, 262_144, "context_length");

    // eps is 1e-6, not the 1e-5 the old default carried.
    assert!(
        (cfg.rms_norm_eps - 1.0e-6).abs() < 1e-12,
        "rms_norm_eps was {}",
        cfg.rms_norm_eps
    );
    assert!(
        (cfg.rope_theta - 10_000_000.0).abs() < 1.0,
        "rope.freq_base"
    );

    // RoPE is partial: 64 of 256 head dims rotate -> 0.25, not 1.0.
    assert!(
        (cfg.partial_rotary_factor - 0.25).abs() < 1e-6,
        "partial_rotary_factor was {} (64/256 = 0.25)",
        cfg.partial_rotary_factor
    );

    // mrope sections keep the trailing 0.
    assert_eq!(
        cfg.mrope_section,
        [11, 11, 10, 0],
        "rope.dimension_sections"
    );

    // SSM geometry.
    assert_eq!(cfg.ssm_d_state, 128, "ssm.state_size");
    assert_eq!(cfg.ssm_n_group, 16, "ssm.group_count");
    assert_eq!(cfg.ssm_d_inner, 6144, "ssm.inner_size");
    assert_eq!(cfg.ssm_dt_rank, 48, "ssm.time_step_rank");
    assert_eq!(cfg.linear_conv_kernel_dim, 4, "ssm.conv_kernel");
    // keys = group_count * state_size = 2048; values = inner_size = 6144
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert_eq!(cfg.linear_num_value_heads, 48);
    assert_eq!(
        cfg.linear_num_key_heads * cfg.linear_key_head_dim,
        2048,
        "GDN key width"
    );
    assert_eq!(
        cfg.linear_num_value_heads * cfg.linear_value_head_dim,
        6144,
        "GDN value width"
    );

    // PLE.
    assert_eq!(cfg.ple_layer_ids, vec![1], "ple.layers");
    assert_eq!(cfg.ngram_size, 3, "ple.ngram_size");
    assert_eq!(cfg.ple_conv_kernel_size, 4, "ple.conv_kernel");
    assert_eq!(
        cfg.ngram_dim,
        Some(160),
        "PLE table width is embedding_length_per_layer_input=160, not 512"
    );

    // Layer schedule: 3 GDN then 1 QSA, repeating; 12 full-attention layers.
    assert_eq!(cfg.layer_types.len(), 48, "one entry per layer");
    assert_eq!(
        cfg.layer_types
            .iter()
            .filter(|t| *t == "full_attention")
            .count(),
        12,
        "every 4th layer is full attention"
    );
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let expect = if (i + 1) % 4 == 0 {
            "full_attention"
        } else {
            "linear_attention"
        };
        assert_eq!(t, expect, "layer {i} kind");
    }

    // Vocabulary from the tokenizer, not a constant.
    assert_eq!(cfg.vocab_size, 248_320, "tokenizer.ggml.tokens length");
}

#[test]
fn absent_metadata_falls_back_to_documented_defaults() {
    struct Empty;
    impl MetadataLookup for Empty {
        fn get_str(&self, _: &str) -> Option<String> {
            None
        }
        fn get_u32(&self, _: &str) -> Option<u32> {
            None
        }
        fn get_f32(&self, _: &str) -> Option<f32> {
            None
        }
    }
    let cfg = Qwen38FlashNextConfig::from_qwen4exp_metadata(&Empty);
    let d = Qwen38FlashNextConfig::default();
    assert_eq!(cfg.num_layers, d.num_layers);
    assert_eq!(cfg.num_experts, d.num_experts);
    assert_eq!(cfg.indexer_top_k, d.indexer_top_k);
    assert_eq!(cfg.mrope_section, d.mrope_section);
    // layer_types still gets built from the interval fallback, so it is usable.
    assert_eq!(cfg.layer_types.len(), 48);
}

#[test]
fn dt_rank_and_value_head_count_are_independent_fields() {
    // They coincide at 48 for the released checkpoint, which is exactly why the
    // old loader could conflate them and still appear to work. Assert the
    // config keeps them separate so a different geometry is representable.
    let d = Qwen38FlashNextConfig::default();
    let _ = d;
    // Compile-time: the fields exist and are distinct types/usages.
    let cfg = Qwen38FlashNextConfig::default();
    assert_eq!(cfg.ssm_dt_rank, 48);
    assert_eq!(cfg.linear_num_value_heads, 48);
    // And `ssm_alpha` is sized by the rank in the loader; a config where they
    // differ must still be constructible.
    let mut c = Qwen38FlashNextConfig::default();
    c.ssm_dt_rank = 7;
    c.linear_num_value_heads = 11;
    assert_eq!(c.ssm_dt_rank, 7);
    assert_eq!(c.linear_num_value_heads, 11);
}

/// A `MetadataLookup` carrying only `block_count`, so every other field takes
/// its documented fallback. Covers the default paths the real checkpoint never
/// exercises — the ones a differently-shaped file would hit.
struct Partial {
    layers: u32,
}
impl MetadataLookup for Partial {
    fn get_str(&self, _: &str) -> Option<String> {
        None
    }
    fn get_u32(&self, key: &str) -> Option<u32> {
        if key == "qwen4exp.block_count" {
            Some(self.layers)
        } else {
            None
        }
    }
    fn get_f32(&self, _: &str) -> Option<f32> {
        None
    }
}

#[test]
fn fallback_rope_factor_is_64_over_256_not_unity() {
    let cfg = Qwen38FlashNextConfig::from_qwen4exp_metadata(&Partial { layers: 48 });
    // Defaults: head_dim 256, rope dim 64 -> a quarter rotates. Forcing 1.0
    // would rotate the whole head and silently destroy positional encoding.
    assert!(
        (cfg.partial_rotary_factor - 0.25).abs() < 1e-6,
        "fallback partial_rotary_factor was {}",
        cfg.partial_rotary_factor
    );
}

#[test]
fn fallback_mrope_section_keeps_four_slots() {
    let cfg = Qwen38FlashNextConfig::from_qwen4exp_metadata(&Partial { layers: 48 });
    assert_eq!(
        cfg.mrope_section,
        [11, 11, 10, 0],
        "the trailing 0 is the unrotated tail and must be present"
    );
}

#[test]
fn fallback_ngram_dim_is_the_ple_table_width() {
    let cfg = Qwen38FlashNextConfig::from_qwen4exp_metadata(&Partial { layers: 48 });
    assert_eq!(
        cfg.ngram_dim,
        Some(160),
        "PLE embedding width defaults to 160, not the model hidden size"
    );
}

#[test]
fn fallback_layer_schedule_follows_the_interval() {
    let cfg = Qwen38FlashNextConfig::from_qwen4exp_metadata(&Partial { layers: 12 });
    assert_eq!(cfg.layer_types.len(), 12, "one entry per declared layer");
    assert_eq!(
        cfg.layer_types
            .iter()
            .filter(|t| *t == "full_attention")
            .count(),
        3,
        "12 layers at interval 4 has 3 full-attention layers"
    );
    assert_eq!(cfg.layer_types[3], "full_attention");
    assert_eq!(cfg.layer_types[0], "linear_attention");
}
