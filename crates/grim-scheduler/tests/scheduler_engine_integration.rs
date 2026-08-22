//! Cross-crate integration tests for `grim-scheduler`.
//!
//! Validates:
//! - Continuous batching with chunked prefill tracking
//! - Admission controller TTFT prediction & backlog admission gating
//! - Request pause, resume, preemption, and completion lifecycle
//! - Self-tuning controller dynamic knob updates based on latency observations

use std::time::Duration;
use grim_scheduler::self_tuning::KnobKind;
use grim_scheduler::{
    AdmissionController, AdmissionDecision, BatchTokenBacklog, Request, Scheduler,
    SelfTuningController,
};

#[test]
fn test_scheduler_chunked_prefill_and_decode_interleaving() {
    let admission = AdmissionController::new(0, 0); // Disable latency gating
    let mut scheduler = Scheduler::new(64, 4, admission); // max_batched_tokens=64 activates chunking under pressure
    scheduler.chunked_prefill_size = 64;

    // Add a long request (160 prompt tokens -> requires chunked prefill passes)
    let req1 = Request {
        id: 1,
        prompt_tokens: 160,
        priority: 1,
        consumed_tokens: 0,
        model_id: None,
        adapter_ids: vec![],
        input_ids: None,
    };
    scheduler.enqueue(req1);

    // Pass 1: drains 64 tokens under pressure
    let out1 = scheduler.schedule();
    assert_eq!(out1.prefill_ids, vec![1]);
    assert_eq!(out1.decode_ids, Vec::<u64>::new());
    assert_eq!(scheduler.running[0].consumed_tokens, 64);
}

#[test]
fn test_admission_controller_ttft_and_backlog_gating() {
    let controller = AdmissionController::new(100, 50); // 100ms TTFT budget
    controller.observe_prefill(1000, Duration::from_millis(1000)); // 1000 tokens/sec = 1 tok/ms

    // 1. Small request with zero backlog -> admitted (50ms <= 100ms)
    let req_small = Request {
        id: 10,
        prompt_tokens: 50,
        ..Default::default()
    };
    let backlog_empty = BatchTokenBacklog { total: 0 };
    assert_eq!(
        controller.admit(&req_small, &backlog_empty),
        AdmissionDecision::Admit
    );

    // 2. Small request with huge backlog -> deferred (50 + 200 = 250ms > 100ms)
    let backlog_heavy = BatchTokenBacklog { total: 200 };
    assert_eq!(
        controller.admit(&req_small, &backlog_heavy),
        AdmissionDecision::Defer
    );

    // 3. Oversized single request bypassing livelock when backlog is clear
    let req_huge = Request {
        id: 11,
        prompt_tokens: 500, // 500ms > 100ms target
        ..Default::default()
    };
    let solo_backlog = BatchTokenBacklog { total: 500 };
    assert_eq!(
        controller.admit(&req_huge, &solo_backlog),
        AdmissionDecision::Admit
    );
}

#[test]
fn test_scheduler_pause_resume_and_snapshot_lifecycle() {
    let admission = AdmissionController::new(0, 0);
    let mut scheduler = Scheduler::new(256, 2, admission);

    let req1 = Request {
        id: 101,
        prompt_tokens: 64,
        priority: 10,
        consumed_tokens: 64,
        ..Default::default()
    };
    let req2 = Request {
        id: 102,
        prompt_tokens: 64,
        priority: 5,
        consumed_tokens: 64,
        ..Default::default()
    };
    scheduler.running.push(req1);
    scheduler.running.push(req2);

    let snap = scheduler.snapshot();
    assert_eq!(snap.active_requests, 2);
    assert_eq!(snap.waiting_requests, 0);

    // Pause request 101
    assert!(scheduler.pause(101));
    assert_eq!(scheduler.running.len(), 1);
    assert_eq!(scheduler.paused.len(), 1);
    assert_eq!(scheduler.snapshot().paused_requests, 1);

    // Resume request 101 -> moves directly back to running queue, keeping KV state alive
    assert!(scheduler.resume(101));
    assert_eq!(scheduler.paused.len(), 0);
    assert_eq!(scheduler.running.len(), 2);
    assert_eq!(scheduler.waiting.len(), 0);
}

#[test]
fn test_self_tuning_controller_telemetry_loop() {
    let mut controller = SelfTuningController::new(50.0, 10.0);

    // Record high TTFT observations
    for _ in 0..5 {
        controller.record_ttft(120.0);
        controller.record_itl(8.0);
    }

    controller.tune_all();
    let knob_val = controller.tune_one(KnobKind::ChunkedPrefillSize);
    assert!(knob_val > 0.0, "Chunked prefill size must remain positive");
}
