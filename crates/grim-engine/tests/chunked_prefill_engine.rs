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

/// Pin the self-tuner knobs so `tick()` cannot move them mid-scenario.
/// `tick()` re-applies tuned params every pass, so the scheduler fields
/// alone would be overwritten.
fn pin_knobs(engine: &mut Engine, chunk: f64, mbt: f64) {
    use grim_scheduler::self_tuning::{KnobKind, KnobTuner};
    let ttft = engine.self_tuning_controller.chunked_prefill_size.target;
    engine.self_tuning_controller.chunked_prefill_size =
        KnobTuner::new_fixed(KnobKind::ChunkedPrefillSize, ttft, chunk, chunk, chunk, 0.0);
    engine.self_tuning_controller.max_batched_tokens =
        KnobTuner::new_fixed(KnobKind::MaxBatchedTokens, ttft, mbt, mbt, mbt, 0.0);
    engine.scheduler.chunked_prefill_size = chunk as usize;
    engine.scheduler.max_batched_tokens = mbt as usize;
}

/// Scheduler duplicate-entry audit for one tick: request 1 may hold at most
/// one `running` copy and at most one prefill / decode slot. More than one
/// copy is the P4 bug class (re-prefill of already-consumed tokens, or
/// preemption swapping one copy while a twin keeps running).
fn assert_single_copy(engine: &Engine, out: &grim_scheduler::SchedulerOutput, tick: &str) {
    assert_eq!(
        engine
            .scheduler
            .running
            .iter()
            .filter(|r| r.id == 1)
            .count(),
        1,
        "{tick}: exactly one running copy for request 1"
    );
    assert!(
        out.prefill_ids.iter().filter(|&&id| id == 1).count() <= 1,
        "{tick}: request 1 prefills at most once per tick"
    );
    assert!(
        out.decode_ids.iter().filter(|&&id| id == 1).count() <= 1,
        "{tick}: request 1 decodes at most once per tick"
    );
}

/// P4: preempted + resumed + chunked remainder.
///
/// A chunked request preempted mid-prefill must resume at its TRUE offset —
/// never re-running consumed tokens (exact-once) and never leaving a stale
/// swapped twin behind. Scenario on a 120-token prompt with chunk 50:
///
/// - tick 1: `[0, 50)` (pos 50, progress 50);
/// - tick 2 (budget squeezed to 60): scheduler preempts request 1
///   (`preempted_ids == [1]`, `consumed = 50` copy parked in `swapped`)
///   while the waiting remainder still drains `[50, 100)` (pos 100);
/// - tick 3 (budget restored): the stale swapped copy is dropped as
///   already-tracked and the remainder resumes `[100, 120)` (pos 120 —
///   a resume-from-zero bug would land at 170 or re-append KV);
/// - tick 4: exactly one decode step (pos 121).
#[test]
fn chunked_prefill_preempt_resume_processes_remainder_exactly_once() {
    let cfg = EngineConfig {
        max_batched_tokens: 100,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(cfg);
    engine.register_model("small", small_llama());
    pin_knobs(&mut engine, 50.0, 100.0);

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
    let progress =
        |e: &Engine| e.prefill_progress.get(&1).copied().unwrap_or(usize::MAX);

    // Tick 1: first chunk [0, 50). No preemption (nothing running yet).
    let out = engine.tick().expect("tick 1");
    assert!(out.preempted_ids.is_empty());
    assert_eq!(out.prefill_ids, vec![1]);
    assert_single_copy(&engine, &out, "tick 1");
    assert_eq!(pos(&engine), 50);
    assert_eq!(progress(&engine), 50);

    // Tick 2: squeeze the budget below the 70-token remainder. The running
    // copy (50 consumed, 70 left) exceeds the budget, so request 1 is
    // preempted — yet the waiting remainder is still admitted and drains
    // [50, 100) in the same pass.
    pin_knobs(&mut engine, 50.0, 60.0);
    let out = engine.tick().expect("tick 2");
    assert_eq!(out.preempted_ids, vec![1], "tick 2 must preempt request 1");
    assert_eq!(out.prefill_ids, vec![1]);
    assert_single_copy(&engine, &out, "tick 2");
    assert_eq!(pos(&engine), 100, "tick 2 must continue at offset 50, not restart");
    assert_eq!(progress(&engine), 100);

    // Tick 3: pressure lifts. The stale swapped copy (consumed = 50) must be
    // dropped as already-tracked — not re-admitted — and the remainder
    // resumes [100, 120): pos lands at exactly 120.
    pin_knobs(&mut engine, 50.0, 1000.0);
    let out = engine.tick().expect("tick 3");
    assert!(out.preempted_ids.is_empty());
    assert_eq!(out.prefill_ids, vec![1]);
    assert_single_copy(&engine, &out, "tick 3");
    assert!(
        engine.scheduler.swapped.iter().all(|r| r.id != 1),
        "stale swapped copy of request 1 must be gone after resume"
    );
    assert_eq!(
        pos(&engine),
        120,
        "remainder must resume at offset 100 (a restart would exceed 120)"
    );
    assert_eq!(progress(&engine), 120);
    assert!(
        !engine.scheduler.waiting.iter().any(|r| r.id == 1),
        "fully-consumed request must not return to waiting"
    );

    // Tick 4: no prefill left — exactly one decode step, pos 121.
    let out = engine.tick().expect("tick 4");
    assert!(out.prefill_ids.is_empty());
    assert_eq!(out.decode_ids, vec![1]);
    assert_single_copy(&engine, &out, "tick 4");
    assert_eq!(pos(&engine), 121);
}
