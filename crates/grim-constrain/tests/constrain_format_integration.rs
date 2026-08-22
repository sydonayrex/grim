//! Cross-crate integration tests for `grim-constrain`.
//!
//! Validates:
//! - JSON Schema compiler compilation and structural property validation
//! - Pushdown automaton (JsonState) incremental validation & token masking
//! - ConstrainedSampler end-to-end token sampling with vocabulary FSM masking
//! - TokenMaskCache memoization across identical parser states

use grim_backend_cpu::cpu_tensor;
use grim_constrain::{
    ConstrainedSampler, Constraint, JsonState, TokenMaskCache, apply_mask, compile_json_schema,
};
use grim_core::sampler::{GreedySampler, Sampler};
use grim_tensor::Shape;
use serde_json::json;
use std::sync::Arc;

#[test]
fn test_compile_json_schema_valid_and_invalid() {
    // 1. Valid JSON Schema with nested properties
    let valid_schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean"}
        },
        "required": ["name", "age"]
    });
    let constraint_res = compile_json_schema(valid_schema);
    assert!(
        constraint_res.is_ok(),
        "Valid schema must compile successfully"
    );

    // 2. Invalid schema (unsupported type)
    let invalid_schema = json!({
        "type": "unsupported_data_type"
    });
    let invalid_res = compile_json_schema(invalid_schema);
    assert!(
        invalid_res.is_err(),
        "Invalid schema type must fail compilation"
    );
}

#[test]
fn test_token_mask_cache_and_apply_mask() {
    let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mask = vec![true, false, true, false, true]; // tokens 1 and 3 are invalid

    apply_mask(&mut logits, &mask);
    assert_eq!(logits[0], 1.0);
    assert_eq!(logits[1], f32::NEG_INFINITY);
    assert_eq!(logits[2], 3.0);
    assert_eq!(logits[3], f32::NEG_INFINITY);
    assert_eq!(logits[4], 5.0);

    // Verify TokenMaskCache insert and retrieval via mask_for
    let mut cache = TokenMaskCache::new();
    let state = JsonState::default();
    let vocab = vec!["{".to_string(), "foo".to_string()];
    let computed_mask = cache.mask_for(state.clone(), &vocab);
    assert_eq!(computed_mask.len(), 2);
    assert_eq!(cache.len(), 1);

    // Second retrieval should hit cache
    let cached_mask = cache.mask_for(state, &vocab);
    assert_eq!(cached_mask.as_ref(), computed_mask.as_ref());
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_constrained_sampler_json_object_enforcement() {
    // Vocab: 0="{" 1="\"key\"" 2=":" 3="\"val\"" 4="}" 5="invalid_token"
    let vocab: Arc<[String]> = Arc::new([
        "{".to_string(),
        "\"key\"".to_string(),
        ":".to_string(),
        "\"val\"".to_string(),
        "}".to_string(),
        "invalid_garbage".to_string(),
    ]);

    let greedy = Arc::new(GreedySampler {
        repeat_penalty: Some(1.0),
    });
    let sampler = ConstrainedSampler::new(greedy, Constraint::json_object()).with_vocab(vocab);

    // Initial state: tokens 0, 1, 3 are valid value starts; token 5 is invalid garbage with highest raw logit (10.0)
    let initial_logits = cpu_tensor(
        vec![5.0f32, 1.0, 2.0, 3.0, 4.0, 10.0],
        Shape::new(vec![1, 6]),
    );

    let sampled = sampler.sample(&initial_logits, &[]).unwrap();
    // ConstrainedSampler must mask invalid token 5 (-inf) so token 0 ("{", logit 5.0) is selected
    assert_eq!(sampled, 0, "Sampler must force valid opening brace");
}

/// F9 regression: the bounded-backtracking regex subset must accept/reject
/// exactly the language of the pattern, including character classes, ranges,
/// and `{m,n}` quantifiers — not the old substring heuristic.
#[test]
fn test_f9_regex_subset_engine() {
    use grim_constrain::schema::validate_pattern;

    // Anchored repetition
    assert!(validate_pattern("^[A-Z]{3}$", "ABC"));
    assert!(!validate_pattern("^[A-Z]{3}$", "AB"));
    assert!(!validate_pattern("^[A-Z]{3}$", "ABCD"));

    // Character classes and ranges
    assert!(validate_pattern(r"^\d{4}-\d{2}$", "2026-08"));
    assert!(!validate_pattern(r"^\d{4}-\d{2}$", "2026-8"));
    assert!(validate_pattern(r"^[^0-9]+$", "abc"));
    assert!(!validate_pattern(r"^[^0-9]+$", "a1c"));

    // Unanchored substring match still works
    assert!(validate_pattern("foo", "foobar"));
    assert!(!validate_pattern("foo", "bar"));
}

/// F9 regression: schema-driven lookahead jump-forward must deterministically
/// emit the forced continuation when the schema admits exactly one path
/// (single required key, single enum value).
#[test]
fn test_f9_schema_lookahead_jump_forward() {
    use grim_constrain::schema::compile_json_schema;

    // Single required key → after `{`, the only valid continuation is the key.
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "name": {"type": "string"} },
        "required": ["name"]
    });
    let comp = compile_json_schema(schema).unwrap();
    assert_eq!(
        comp.lookahead_jump_forward("{").as_deref(),
        Some("\"name\": "),
        "forced key name must be jump-forwarded"
    );

    // Single enum value → the whole value is forced.
    let schema = serde_json::json!({"type": "string", "enum": ["x"]});
    let comp = compile_json_schema(schema).unwrap();
    assert_eq!(
        comp.lookahead_jump_forward("").as_deref(),
        Some("\"x\""),
        "single-enum value must be jump-forwarded"
    );
}
