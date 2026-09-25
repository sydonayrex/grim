//! Layer 1.5 (session-continuity design): radix prefix cache CONSUMPTION.
//!
//! The engine must (a) skip re-prefill of a matched shared prefix, (b) feed
//! the seeded prefix to the decode attention path, and (c) produce output
//! identical to a cold engine that never saw a radix hit. Deterministic:
//! `Llama::random` is fixed-seed, so logits compare bit-for-bit on CPU.

use grim_core::model::CausalLm;
use grim_engine::{Engine, EngineConfig};
use grim_models_transformer::{Llama, LlamaConfig};
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

fn run_request(engine: &mut Engine, id: u64, prompt: Vec<u32>, extra_ticks: usize) -> Vec<f32> {
    let prompt_len = prompt.len();
    engine
        .enqueue_request(grim_scheduler::Request {
            id,
            prompt_tokens: prompt_len,
            input_ids: Some(prompt),
            max_new_tokens: 4,
            ..Default::default()
        })
        .expect("enqueue");
    // Drain prefill, then a few decode ticks; capture final logits.
    let mut logits: Vec<f32> = Vec::new();
    for _ in 0..extra_ticks {
        engine.tick().expect("tick");
        if let Some(outcome) = engine.last_outcome(id) {
            if let Some(l) = &outcome.logits {
                logits = l.to_vec_f32().unwrap();
            }
        }
    }
    logits
}

/// Two sequential requests sharing a 32-token (2-block) prefix: the second
/// must skip exactly those blocks in prefill and still land byte-identical
/// logits to a cold engine.
#[test]
fn second_request_with_shared_prefix_skips_prefill_and_matches_cold_output() {
    let prefix: Vec<u32> = (0..32).map(|i| 3 + (i % 191)).collect();
    let suffix_a: Vec<u32> = (0..16).map(|i| 100 + (i % 89)).collect();
    let suffix_b: Vec<u32> = (0..16).map(|i| 200 + (i % 53)).collect();
    let mut prompt_a = prefix.clone();
    prompt_a.extend_from_slice(&suffix_a);
    let mut prompt_b = prefix.clone();
    prompt_b.extend_from_slice(&suffix_b);

    // Cold reference: B alone on a fresh engine.
    let mut cold = fresh_engine();
    let cold_logits = run_request(&mut cold, 1, prompt_b.clone(), 3);
    let (lookups_cold, hits_cold, tokens_cold) = cold.radix_cache_telemetry();
    assert!(
        hits_cold == 0 && tokens_cold == 0,
        "cold engine has no hits"
    );
    let _ = lookups_cold;

    // Warm engine: A registers the shared prefix, B must hit it.
    let mut warm = fresh_engine();
    run_request(&mut warm, 1, prompt_a, 3);
    warm.finish_request(1);
    let (l1, h1, t1) = warm.radix_cache_telemetry();
    assert_eq!((l1, h1, t1), (1, 0, 0), "A populates the tree, no hits yet");

    let warm_logits = run_request(&mut warm, 2, prompt_b.clone(), 3);
    let (_l2, h2, t2) = warm.radix_cache_telemetry();
    assert_eq!(h2, 1, "request B must consume the radix prefix");
    assert_eq!(
        t2, 32,
        "exactly the 32 shared tokens (2 blocks) must be reused, got {t2}"
    );

    assert_eq!(
        warm_logits.len(),
        cold_logits.len(),
        "logit shapes must match"
    );
    assert_eq!(
        warm_logits, cold_logits,
        "radix-reused output must match a cold engine bit-for-bit"
    );
    warm.finish_request(2);
}

/// A mid-conversation edit keeps the unedited prefix reusable: B diverges
/// after token 16, so exactly one block (16 tokens) is reused — not the old
/// session-slot all-or-nothing behavior.
#[test]
fn mid_conversation_edit_reuses_only_the_shared_blocks() {
    let base: Vec<u32> = (0..48).map(|i| 5 + (i % 97)).collect();
    let mut edited = base.clone();
    for (i, t) in edited.iter_mut().enumerate().skip(16) {
        *t = 250 - (i as u32 % 100);
    }

    let mut cold = fresh_engine();
    let cold_logits = run_request(&mut cold, 1, edited.clone(), 3);

    let mut warm = fresh_engine();
    run_request(&mut warm, 1, base, 3);
    warm.finish_request(1);
    let warm_logits = run_request(&mut warm, 2, edited.clone(), 3);

    let (_, hits, tokens) = warm.radix_cache_telemetry();
    assert_eq!(hits, 1);
    assert_eq!(tokens, 16, "only the untouched first block is reused");
    assert_eq!(warm_logits, cold_logits);
    warm.finish_request(2);
}

/// Releasing requests must return the tree to refcount-0 cached state:
/// eviction eligibility survives the finish, the pool does not leak blocks,
/// and a third request still hits the cached prefix.
#[test]
fn finished_requests_leave_cached_prefix_reusable_and_unpinned() {
    let prompt: Vec<u32> = (0..32).map(|i| 11 + (i % 71)).collect();

    let mut engine = fresh_engine();
    run_request(&mut engine, 1, prompt.clone(), 2);
    engine.finish_request(1);
    run_request(&mut engine, 2, prompt.clone(), 2);
    engine.finish_request(2);
    let (_, hits, tokens) = engine.radix_cache_telemetry();
    // The 32-token prompt fully matches the cached prefix, but the last block
    // is always recomputed so the prefill pass yields logits for the first
    // decode token — the reuse count is one block (16 tokens), not two.
    assert_eq!((hits, tokens), (1, 16));

    // The whole prompt was cached; no live request holds anything. Refcounts
    // are all zero (removal happens inside finish_request's remove_prefix and
    // the session rollback), so a pool scan finds no live references.
    let pool = engine.block_pool.lock().unwrap_or_else(|e| e.into_inner());
    let pids: Vec<usize> = (0..pool.capacity()).collect();
    for pid in pids {
        assert!(
            pool.block_ref_count(pid) == 0,
            "block {pid} still referenced after both requests finished"
        );
    }
    drop(pool);

    // A third request still benefits from the cached prefix.
    run_request(&mut engine, 3, prompt, 2);
    engine.finish_request(3);
    let (_, hits, _) = engine.radix_cache_telemetry();
    assert_eq!(
        hits, 2,
        "cached prefix survives unreferenced between requests"
    );
}

/// Radix seeding must not engage for a request with synthetic (absent) input
/// ids — hashing synthetic ids could alias a real cached prompt.
#[test]
fn requests_without_token_ids_never_seed() {
    let prefix: Vec<u32> = (0..32).map(|i| 41 + (i % 61)).collect();
    let mut engine = fresh_engine();
    run_request(&mut engine, 1, prefix.clone(), 2);
    engine.finish_request(1);

    engine
        .enqueue_request(grim_scheduler::Request {
            id: 2,
            prompt_tokens: 48,
            input_ids: None, // engine synthesizes 0..48
            max_new_tokens: 2,
            ..Default::default()
        })
        .unwrap();
    engine.tick().unwrap();
    engine.tick().unwrap();
    let (_, hits, tokens) = engine.radix_cache_telemetry();
    assert_eq!((hits, tokens), (0, 0), "synthetic ids must not seed");
    engine.finish_request(2);
}
