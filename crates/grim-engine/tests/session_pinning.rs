//! Layer 2 (session identity) integration: session-tagged requests pin their
//! radix-tree blocks against idle-time eviction, while untagged requests get
//! exactly Layer 1.5 behavior. Both paths must produce bit-identical outputs
//! to a no-sharing baseline.

use grim_core::model::CausalLm;
use grim_engine::{Engine, EngineConfig};
use grim_models_transformer::{Llama, LlamaConfig};
use grim_tensor::Device;

/// GRIM_SESSION_PIN_SECS is read per call (no cache), but the env itself is
/// process-global — the pin tests serialize on this lock and set it
/// explicitly so they cannot poison each other.
static PIN_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
            max_seq_len: 4096,
        },
    ))
}

fn engine_with_pool(blocks: usize) -> Engine {
    let mut engine = Engine::new(EngineConfig {
        block_pool_capacity: blocks,
        ..EngineConfig::default()
    });
    engine.register_model("small", small_llama());
    engine
}

fn run(engine: &mut Engine, id: u64, prompt: Vec<u32>, session: Option<&str>) -> Vec<f32> {
    let n = prompt.len();
    engine
        .enqueue_request(grim_scheduler::Request {
            id,
            prompt_tokens: n,
            input_ids: Some(prompt),
            max_new_tokens: 4,
            session: session.map(str::to_string),
            ..Default::default()
        })
        .expect("enqueue");
    let mut logits = Vec::new();
    for _ in 0..2 {
        engine.tick().expect("tick");
        let (l, h, t) = engine.radix_cache_telemetry();
        eprintln!("[pin-dbg] id {id} tick: lookups={l} hits={h} reused={t}");
        if let Some(o) = engine.last_outcome(id) {
            if let Some(l) = &o.logits {
                logits = l.to_vec_f32().unwrap();
            }
        }
    }
    engine.finish_request(id);
    logits
}

/// Pinned session blocks must survive unrelated traffic that would otherwise
/// evict the cached prefix. Pool of 12 blocks holds one 32-token prefix (2
/// blocks) plus work; after the pinned session retires, other requests cycle
/// the pool AND the pinned blocks remain matched.
#[test]
fn session_tag_pins_prefix_against_idle_eviction() {
    let _env = PIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: serialized by PIN_ENV_LOCK.
    unsafe { std::env::remove_var("GRIM_SESSION_PIN_SECS") };
    let mut engine = engine_with_pool(8);

    // Pinned session turn.
    let prefix: Vec<u32> = (0..32).map(|i| 5 + (i % 61)).collect();
    run(&mut engine, 1, prefix.clone(), Some("sess-a"));

    let pinned_count = {
        let pool = engine.block_pool.lock().unwrap_or_else(|e| e.into_inner());
        (0..8).filter(|&b| pool.is_block_pinned(b)).count()
    };
    assert_eq!(pinned_count, 2, "both prefix blocks of sess-a stay pinned");

    // Churn the pool with unrelated traffic (fits within the pool only if eviction is not blocked).
    for i in 0..12u64 {
        let prompt: Vec<u32> = (0..32).map(|x| 17 + i as u32 * 7 + x).collect();
        run(&mut engine, 1000 + i, prompt, None);
    }

    // The pinned prefix still matches for the session: a follow-up turn with
    // the same 32-token prefix + new tail hits the pinned cache.
    let mut follow_up = prefix.clone();
    follow_up.extend_from_slice(&[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]);
    run(&mut engine, 2, follow_up, Some("sess-a"));
    let (_, hits, tokens) = engine.radix_cache_telemetry();
    assert!(hits >= 1, "later session turn must hit the pinned prefix");
    assert!(
        tokens >= 16,
        "at least one pinned block reused, got {tokens}"
    );
}

/// Session tagging must NOT change outputs vs anonymous Layer 1.5 reuse —
/// the plan's invariant: dropping the field degrades to plain radix behavior,
/// never a different answer.
#[test]
fn session_telemetry_does_not_change_output() {
    let prompt: Vec<u32> = (0..48).map(|i| 7 + (i % 87)).collect();

    let mut e1 = engine_with_pool(64);
    let anon = {
        let _ = run(&mut e1, 1, prompt.clone(), None);
        run(&mut e1, 2, prompt.clone(), None)
    };
    let mut e2 = engine_with_pool(64);
    let tagged = {
        let _ = run(&mut e2, 1, prompt.clone(), Some("s"));
        run(&mut e2, 2, prompt.clone(), Some("s"))
    };
    let mut e3 = engine_with_pool(64);
    let cold = run(&mut e3, 1, prompt.clone(), None);

    assert_eq!(anon, tagged, "session tag must not change outputs");
    assert_eq!(tagged, cold, "warm session must match cold-engine output");
}

/// An expired pin falls back to plain LRU: with pin secs = 0 the pin is dead
/// on arrival, the next request may evict the prefix, and the engine neither
/// panics nor corrupts.
#[test]
fn expired_pin_is_plain_lru() {
    let _env = PIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: serialized by PIN_ENV_LOCK; no concurrent reader in this binary.
    unsafe { std::env::set_var("GRIM_SESSION_PIN_SECS", "0") };
    let mut engine = engine_with_pool(8);
    let prefix: Vec<u32> = (0..32).map(|i| 3 + (i % 53)).collect();
    run(&mut engine, 1, prefix.clone(), Some("sess-b"));
    {
        let pool = engine.block_pool.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            (0..8).all(|b| !pool.is_block_pinned(b)),
            "sec=0 pins expire immediately"
        );
    }
    // Churn hard enough to force eviction of the prefix (pool holds 8 blocks).
    for i in 0..8u64 {
        let p: Vec<u32> = (0..32).map(|x| 100 + i as u32 * 3 + x).collect();
        run(&mut engine, 10 + i, p, None);
    }
    assert!(engine.tick().is_ok());
}
