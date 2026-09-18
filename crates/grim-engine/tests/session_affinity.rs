//! WI-HYBRID Layer 2 (session identity): slot affinity + block pinning.
//!
//! Contract (session-continuity-design2.md):
//! - a session-tagged request gets its own session-scoped decode-graph slot
//!   key (`{model}#s{hash}`), reused across turns of the same session;
//! - its KV blocks are pinned so the radix LRU cannot reclaim the cached
//!   prefix between turns under other traffic's pressure;
//! - a request WITHOUT the tag degrades to exactly Layer 1.5 behavior (no
//!   session-scoped slot, no pins) — the acceptance criterion.
//!
//! Deterministic: `Llama::random` is fixed-seed; outputs compare bit-for-bit
//! on CPU.

use grim_core::model::CausalLm;
use grim_engine::{Engine, EngineConfig};
use grim_models_transformer::{Llama, LlamaConfig};
use grim_scheduler::Request;
use grim_tensor::Device;

fn small_llama() -> Box<dyn CausalLm> {
    Box::new(Llama::random(
        Device::Cpu,
        LlamaConfig {
            vocab_size: 256,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
            yarn: None,
            max_seq_len: 256,
        },
    ))
}

fn fresh_engine() -> Engine {
    let mut engine = Engine::new(EngineConfig::default());
    engine.register_model("small", small_llama());
    engine
}

fn run_session_request(
    engine: &mut Engine,
    id: u64,
    prompt: Vec<u32>,
    session: Option<&str>,
    ticks: usize,
) -> Vec<f32> {
    engine
        .enqueue_request(Request {
            id,
            prompt_tokens: prompt.len(),
            input_ids: Some(prompt),
            max_new_tokens: 4,
            session: session.map(str::to_string),
            ..Default::default()
        })
        .expect("enqueue");
    let mut logits = Vec::new();
    for _ in 0..ticks {
        engine.tick().expect("tick");
        if let Some(outcome) = engine.last_outcome(id) {
            if let Some(l) = &outcome.logits {
                logits = l.to_vec_f32().unwrap();
            }
        }
    }
    logits
}

fn session_slot_keys(engine: &Engine, model: &str) -> Vec<String> {
    let prefix = format!("{model}#s");
    let mut keys: Vec<String> = engine
        .session_slot_last_use
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .cloned()
        .collect();
    keys.sort();
    keys
}

/// Slot affinity: two turns of the SAME session map to ONE session-scoped
/// slot key; a different session gets a different key; a session-less request
/// gets no session-scoped key at all (plain Layer 1.5 slot).
#[test]
fn session_slot_affinity_groups_turns_and_splits_sessions() {
    let prompt: Vec<u32> = (0..40).map(|i| 3 + (i % 191)).collect();
    let mut engine = fresh_engine();

    run_session_request(&mut engine, 1, prompt.clone(), Some("alpha"), 3);
    let keys_after_a1 = session_slot_keys(&engine, "small");
    assert_eq!(keys_after_a1.len(), 1, "one session-scoped key after turn 1");

    run_session_request(&mut engine, 2, prompt.clone(), Some("alpha"), 3);
    let keys_after_a2 = session_slot_keys(&engine, "small");
    assert_eq!(
        keys_after_a2, keys_after_a1,
        "same session must reuse the same session-scoped slot"
    );

    run_session_request(&mut engine, 3, prompt.clone(), Some("beta"), 3);
    let keys_after_b = session_slot_keys(&engine, "small");
    assert_eq!(
        keys_after_b.len(),
        2,
        "a different session must get a different session-scoped slot"
    );

    // Session-less request: no new session-scoped key, no growth.
    let before = session_slot_keys(&engine, "small").len();
    run_session_request(&mut engine, 4, prompt, None, 3);
    assert_eq!(
        session_slot_keys(&engine, "small").len(),
        before,
        "session-less request must not create a session-scoped slot"
    );
}

/// Retention: a session-tagged request pins its KV blocks (radix-LRU cannot
/// reclaim them); a session-less request leaves them unpinned — the exact
/// Layer 1.5 behavior.
#[test]
fn session_tag_pins_blocks_and_absence_does_not() {
    let prompt: Vec<u32> = (0..40).map(|i| 9 + (i % 137)).collect();

    // Session-tagged: after prefill the request's blocks must be pinned.
    let mut tagged = fresh_engine();
    run_session_request(&mut tagged, 1, prompt.clone(), Some("alpha"), 2);
    {
        let pool = tagged.block_pool.lock().unwrap_or_else(|e| e.into_inner());
        let pinned: Vec<usize> = (0..pool.capacity()).filter(|&b| pool.is_block_pinned(b)).collect();
        assert!(
            !pinned.is_empty(),
            "session-tagged request must pin its blocks"
        );
    }

    // Session-less: identical request shape, zero pins (pure Layer 1.5).
    let mut untagged = fresh_engine();
    run_session_request(&mut untagged, 1, prompt.clone(), None, 2);
    {
        let pool = untagged.block_pool.lock().unwrap_or_else(|e| e.into_inner());
        let pinned: Vec<usize> = (0..pool.capacity()).filter(|&b| pool.is_block_pinned(b)).collect();
        assert!(
            pinned.is_empty(),
            "session-less request must not pin anything, got {pinned:?}"
        );
    }
}

/// Acceptance criterion: dropping the `session` field produces byte-identical
/// behavior to Layer 1.5 alone — same prompt, same everything, only the tag
/// differs, and the logits must not differ by a single bit.
#[test]
fn dropping_session_field_is_byte_identical_to_layer15() {
    let prompt: Vec<u32> = (0..40).map(|i| 3 + (i % 191)).collect();

    let mut with_tag = fresh_engine();
    let tagged_logits = run_session_request(&mut with_tag, 1, prompt.clone(), Some("alpha"), 3);

    let mut without_tag = fresh_engine();
    let plain_logits = run_session_request(&mut without_tag, 1, prompt, None, 3);

    assert_eq!(tagged_logits, plain_logits, "session tag must not change output");
}

/// Two different sessions on the same model must not observe each other's
/// outputs either — same guarantee as the no-session case.
#[test]
fn two_sessions_produce_their_own_outputs() {
    let prompt: Vec<u32> = (0..40).map(|i| 3 + (i % 191)).collect();

    let mut single = fresh_engine();
    let lone_logits = run_session_request(&mut single, 1, prompt.clone(), Some("solo"), 3);

    let mut shared = fresh_engine();
    let alpha_logits = run_session_request(&mut shared, 1, prompt.clone(), Some("alpha"), 3);
    let beta_logits = run_session_request(&mut shared, 2, prompt, Some("beta"), 3);

    assert_eq!(alpha_logits, lone_logits, "session alpha must match the lone run");
    assert_eq!(beta_logits, lone_logits, "session beta must match the lone run");
}
