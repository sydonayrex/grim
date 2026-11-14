use grim_core::sampler::{
    SamplingParams, gen_sampling_from_kv, gen_sampling_parse, gen_sidecar_path,
};
use std::collections::HashMap;

/// Qwen3.8-Flash-Next's generation_config.json verbatim (`tests/gen_config_test.cpp:44-52`).
const QWEN_GEN_CONFIG: &str = r#"{
    "bos_token_id": 248044,
    "do_sample": true,
    "eos_token_id": [248046, 248044],
    "pad_token_id": 248044,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95
}"#;

/// Unlimited-OCR's generation_config.json verbatim (`tests/gen_config_test.cpp:213-220`).
const UNLIMITED_OCR_GEN_CONFIG: &str = r#"{
    "bos_token_id": 0,
    "eos_token_id": 1,
    "no_repeat_ngram_size": 35,
    "no_repeat_ngram_window_size": 128,
    "max_new_tokens": 32768,
    "use_cache": true
}"#;

#[test]
fn radiance_oracle_gen_config_reads_qwen_and_stacks_partial_layers() {
    let g = gen_sampling_parse(QWEN_GEN_CONFIG).expect("parse Qwen generation_config.json");
    assert_eq!(g.temp, Some(1.0));
    assert_eq!(g.top_k, Some(20));
    assert!((g.top_p.unwrap() - 0.95).abs() < 1e-6);
    assert_eq!(g.min_p, None);

    let sp = g.onto(SamplingParams::default());
    assert!((sp.temperature - 1.0).abs() < 1e-6);
    assert_eq!(sp.top_k, 20);
    assert!((sp.top_p - 0.95).abs() < 1e-6);

    // Partial sidecar overlay claims only what it states (`gen_config_claims_only_what_a_source_states`)
    let topk_only = gen_sampling_parse(r#"{"top_k": 40}"#).expect("topk_only");
    let mut base = SamplingParams::default();
    base.temperature = 0.6;
    base.top_p = 0.8;
    let stacked = topk_only.onto(base);
    assert!((stacked.temperature - 0.6).abs() < 1e-6);
    assert!((stacked.top_p - 0.8).abs() < 1e-6);
    assert_eq!(stacked.top_k, 40);

    // Config stating only token IDs claims nothing
    assert!(gen_sampling_parse(r#"{"bos_token_id": 1}"#).is_none());
}

#[test]
fn radiance_oracle_do_sample_false_forces_greedy_and_ignores_other_fields() {
    let g = gen_sampling_parse(
        r#"{"do_sample": false, "temperature": 0.7, "top_k": 20, "no_repeat_ngram_size": 35}"#,
    )
    .expect("do_sample false");
    assert_eq!(g.temp, Some(0.0));
    assert_eq!(g.top_k, None);
    assert_eq!(g.ngram_ban_n, None);
    let sp = g.onto(SamplingParams::default());
    assert_eq!(sp.temperature, 0.0);

    // top_k: -1 translates to 0 (off)
    let tk_neg = gen_sampling_parse(r#"{"top_k": -1}"#).expect("top_k -1");
    assert_eq!(tk_neg.top_k, Some(0));
}

#[test]
fn radiance_oracle_drops_out_of_range_field_and_keeps_valid_siblings() {
    // temperature 9.0 is out of [0, 5], so dropped; top_p and top_k survive
    let g = gen_sampling_parse(r#"{"temperature": 9.0, "top_p": 0.95, "top_k": 20}"#)
        .expect("partial valid");
    assert_eq!(g.temp, None);
    assert!((g.top_p.unwrap() - 0.95).abs() < 1e-6);
    assert_eq!(g.top_k, Some(20));

    // top_p is half-open (0.0, 1.0]: 0.0 is rejected, 1.0 is accepted
    assert!(gen_sampling_parse(r#"{"top_p": 0.0}"#).is_none());
    let one = gen_sampling_parse(r#"{"top_p": 1.0}"#).expect("top_p 1.0");
    assert_eq!(one.top_p, Some(1.0));

    // Malformed JSON or array root returns None without panicking
    assert!(gen_sampling_parse("{not json").is_none());
    assert!(gen_sampling_parse("[1, 2, 3]").is_none());
    assert!(gen_sampling_parse("").is_none());
}

#[test]
fn radiance_oracle_ngram_ban_suppress_tokens_kv_and_sidecar_path() {
    let ocr = gen_sampling_parse(UNLIMITED_OCR_GEN_CONFIG).expect("Unlimited-OCR");
    assert_eq!(ocr.ngram_ban_n, Some(35));
    assert_eq!(ocr.ngram_window, Some(128));
    assert_eq!(ocr.temp, None);

    // Gemma-4 suppress_tokens array and space-separated KV table
    let gem4 = gen_sampling_parse(r#"{"temperature": 1.0, "suppress_tokens": [258883, 258882]}"#)
        .expect("Gemma-4 suppress_tokens");
    assert_eq!(gem4.suppress_tokens, vec![258883, 258882]);

    let mut kv = HashMap::new();
    kv.insert("do_sample", "1");
    kv.insert("temperature", "1");
    kv.insert("top_k", "20");
    kv.insert("top_p", "0.95");
    kv.insert("suppress_tokens", "258883 258882");
    let from_kv = gen_sampling_from_kv(|k| kv.get(k).copied()).expect("from_kv");
    assert_eq!(from_kv.temp, Some(1.0));
    assert_eq!(from_kv.top_k, Some(20));
    assert_eq!(from_kv.suppress_tokens, vec![258883, 258882]);

    // Sidecar path derivation (`tests/gen_config_test.cpp:201-207`)
    assert_eq!(
        gen_sidecar_path("/m/q38-flashnext-unc-w4-mtp.grim"),
        "/m/q38-flashnext-unc-w4-mtp.generation.json"
    );
    assert_eq!(gen_sidecar_path("model.gguf"), "model.generation.json");
    assert_eq!(
        gen_sidecar_path("/a.b/model"),
        "/a.b/model.generation.json"
    );
    assert_eq!(
        gen_sidecar_path("/a.b/model.grim"),
        "/a.b/model.generation.json"
    );
}
