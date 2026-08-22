//! Cross-crate integration tests for `grim-speculative`.
//!
//! Validates:
//! - SpeculativeCausalLm plain fallback construction and telemetry reporting
//! - ConfidenceScheduler throughput & load-aware verification length selection
//! - DSpark parallel drafting via TinyDraftBackbone and MarkovHead refinement
//! - Mamba state checkpointing and rollback upon speculative candidate rejection

use grim_backend_cpu::cpu_tensor;
use grim_core::session::Inner;
use grim_core::CausalLm;
use grim_models_transformer::{Llama, LlamaConfig};
use grim_speculative::{
    ConfidenceScheduler, DraftBackbone, MambaSpeculativeEngine, MarkovHead,
    SpeculationConfig, SpeculativeCausalLm, Strategy, ThroughputProfile,
    TinyDraftBackbone, UniformMarkovHead,
};
use grim_tensor::{Device, Shape};

#[test]
fn test_speculative_causal_lm_plain_forward_and_telemetry() {
    let llama_cfg = LlamaConfig {
        vocab_size: 1000,
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        num_layers: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 256,
        partial_rotary_factor: 1.0,
        yarn: None,
    };
    let target = Box::new(Llama::random(Device::Cpu, llama_cfg));
    let spec_model = SpeculativeCausalLm::plain(target);

    assert_eq!(spec_model.strategy(), Strategy::Plain);
    let telem = spec_model.telemetry();
    assert_eq!(telem.strategy, "plain");
    assert_eq!(telem.steps_observed, 0);

    let mut session = Inner::new(Device::Cpu);
    let input = cpu_tensor(vec![10.0f32, 20.0f32], Shape::new(vec![1, 2]));
    let positions = cpu_tensor(vec![0.0f32, 1.0f32], Shape::new(vec![1, 2]));
    let logits = spec_model.forward(&mut session, &input, &positions, &[]).unwrap();
    assert_eq!(logits.shape().dims(), &[2, 1000]);
}

#[test]
fn test_confidence_scheduler_dynamic_block_length_and_adaptation() {
    let profile = ThroughputProfile {
        verify_ms_per_token: 1.0,
        accepted_tokens_per_sec: 100.0,
    };
    let spec_cfg = SpeculationConfig {
        block_len: 5,
        min_verify_len: 1,
        confidence_floor: 0.05,
    };
    let mut scheduler = ConfidenceScheduler::new(profile, spec_cfg);

    let draft_block = grim_speculative::draft_backbone::DraftBlock {
        tokens: vec![101, 102, 103, 104, 105],
        base_logits: cpu_tensor(vec![0.0f32; 5 * 100], Shape::new(vec![5, 100])),
        confidence: vec![0.95, 0.90, 0.85, 0.70, 0.40],
    };

    // 1. Idle GPU utilization (0.0) -> high verify length
    let verify_idle = scheduler.choose_verify_len(&draft_block, 0.0, 0);
    assert!(verify_idle >= 3, "Idle GPU should verify deep prefix");

    // 2. Fully saturated GPU (1.0) under extreme load -> minimal verify length
    let verify_busy = scheduler.choose_verify_len(&draft_block, 1.0, 1000);
    assert_eq!(verify_busy, 1, "Saturated GPU should clamp to minimum verify len");

    // 3. Record acceptance telemetry and verify adaptation trigger
    for _ in 0..15 {
        scheduler.record_acceptance(1, 5); // 20% acceptance rate < 30% threshold
    }
    assert!(scheduler.should_adapt_draft(), "Drift below acceptance floor should trigger adaptation");
}

#[test]
fn test_tiny_draft_backbone_and_uniform_markov_head() {
    let draft = TinyDraftBackbone::new(1000, 64, 4, 42);
    let markov = UniformMarkovHead::new(1000, 4, 42);

    let mut session = Inner::new(Device::Cpu);
    let context = cpu_tensor(vec![0.5f32; 64], Shape::new(vec![1, 64]));

    let draft_block = draft.draft_block(&mut session, &context, 4).unwrap();
    assert_eq!(draft_block.len(), 4);
    assert_eq!(draft_block.confidence.len(), 4);

    let biased_logits = markov.bias(&[10, 20], &draft_block.base_logits).unwrap();
    assert_eq!(biased_logits.shape().dims(), &[4, 1000]);
}

#[test]
fn test_mamba_speculative_rollback_on_rejection() {
    let mut engine = MambaSpeculativeEngine::new(64, 16, 4);

    // Record step states along speculative proposal
    engine.record_state(0, &[1.0; 16], &[0.1; 4]);
    engine.record_state(1, &[2.0; 16], &[0.2; 4]);
    engine.record_state(2, &[3.0; 16], &[0.3; 4]);

    // Target rejected token at step 2 -> rollback to step 1 state
    let rolled = engine.rollback_to(1).unwrap();
    assert_eq!(rolled.step, 1);
    assert_eq!(rolled.ssm_state[0], 2.0);
    assert_eq!(rolled.conv_state[0], 0.2);
}
