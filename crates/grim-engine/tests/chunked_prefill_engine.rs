//! F9 follow-on gate: the ENGINE must honor the scheduler's chunked-prefill
//! budget — each prompt token runs through the model EXACTLY once, with its
//! true position.
//!
//! Pre-fix behavior: `drive_prefill_inner` ignored `consumed_tokens` and
//! re-ran the FULL prompt on every pass. Because models append KV
//! sequentially while placing RoPE by the positions tensor, pass 2 appended
//! the whole prompt's KV a second time (session pos 120 → 240 → 360 …) —
//! duplicated context and corrupt outputs for any request chunked under
//! pressure, not just wasted compute.

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

#[test]
fn chunked_prefill_processes_each_token_exactly_once() {
    // backlog (120) > max_batched_tokens (100) keeps pressure_active on
    // every pass, so the 120-token prompt drains in 50/50/20 chunks.
    // `tick()` re-applies the self-tuner's knobs every pass, so pin them
    // there (floor = ceiling = initial), not on the scheduler.
    let cfg = EngineConfig {
        max_batched_tokens: 100,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(cfg);
    engine.register_model("small", small_llama());
    engine.scheduler.chunked_prefill_size = 50; // first pass, before the tuner applies
    {
        use grim_scheduler::self_tuning::{KnobKind, KnobTuner};
        let ttft = engine.self_tuning_controller.chunked_prefill_size.target;
        engine.self_tuning_controller.chunked_prefill_size =
            KnobTuner::new_fixed(KnobKind::ChunkedPrefillSize, ttft, 50.0, 50.0, 50.0, 0.0);
        engine.self_tuning_controller.max_batched_tokens =
            KnobTuner::new_fixed(KnobKind::MaxBatchedTokens, ttft, 100.0, 100.0, 100.0, 0.0);
    }

    let prompt: Vec<u32> = (0..120).map(|i| 7 + (i % 211)).collect();
    engine
        .enqueue_request(grim_scheduler::Request {
            id: 1,
            prompt_tokens: 120,
            input_ids: Some(prompt),
            max_new_tokens: 4,
            ..Default::default()
        })
        .expect("enqueue");

    let pos = |e: &Engine| e.sessions.get(&1).map(|s| s.current_pos()).unwrap_or(0);

    // Pass 1: chunk of 50. Pre-fix this was already 120 (full re-prefill).
    let out = engine.tick().expect("tick 1");
    assert_eq!(out.prefill_ids, vec![1]);
    assert_eq!(pos(&engine), 50, "pass 1 must prefill exactly 50 tokens");
    assert_eq!(engine.prefill_progress.get(&1), Some(&50));

    // Pass 2: next 50 → cumulative 100 (the F9 scheduler accumulation,
    // now mirrored by actual model execution).
    engine.tick().expect("tick 2");
    assert_eq!(pos(&engine), 100, "pass 2 must prefill tokens [50, 100)");
    assert_eq!(engine.prefill_progress.get(&1), Some(&100));

    // Pass 3: final 20 → exactly 120 total. The pre-fix engine landed at
    // 360 here (120 + 120 + 120) with triplicated KV.
    engine.tick().expect("tick 3");
    assert_eq!(
        pos(&engine),
        120,
        "three chunked passes must prefill each of the 120 tokens exactly once"
    );
    assert_eq!(engine.prefill_progress.get(&1), Some(&120));
    assert!(
        !engine.scheduler.waiting.iter().any(|r| r.id == 1),
        "fully-consumed request must not return to waiting"
    );

    // A fourth tick budgets no new prefill. With the scheduler's running
    // dedup there is exactly ONE entry for request 1, so exactly ONE
    // decode step runs: 120 prompt tokens + 1 generated = 121.
    let out4 = engine.tick().expect("tick 4");
    assert!(out4.prefill_ids.is_empty());
    assert_eq!(out4.decode_ids, vec![1]);
    assert_eq!(
        pos(&engine),
        121,
        "one decode step per tick after full prefill (running-copy dedup)"
    );
}
