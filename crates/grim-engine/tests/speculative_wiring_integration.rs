//! Integration test for speculative decoding registration with EAGLE3 and drafters.

use grim_engine::Engine;
use grim_models_transformer::{Eagle3, Eagle3Config, Llama, LlamaConfig};
use grim_tensor::Device;
use std::sync::Arc;

#[test]
fn test_engine_speculative_eagle3_registration() {
    let mut engine = Engine::new(grim_engine::EngineConfig::default());

    // 1. Create mock base model
    let base_cfg = LlamaConfig {
        vocab_size: 100,
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

    let base_lm = Llama::random(Device::Cpu, base_cfg.clone());

    // 2. Create mock EAGLE3 model
    let eagle3_cfg = Eagle3Config {
        vocab_size: 100,
        hidden_size: 64,
        target_hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        num_layers: 1,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 256,
        num_target_fusion_layers: 3,
    };

    let eagle3 = Eagle3::random(Device::Cpu, eagle3_cfg);

    // 3. Register EAGLE3 model with Engine
    engine.register_eagle3_model("test-model", Box::new(base_lm), Arc::new(eagle3));

    assert!(engine.has_model("test-model"));
}

#[test]
fn test_engine_decode_sampling_interaction_pipeline() {
    let mut engine = Engine::new(grim_engine::EngineConfig {
        block_pool_capacity: 64,
        num_kv_heads: 2,
        head_dim: 16,
        ..Default::default()
    });

    let vocab_size = 128usize;
    let base_cfg = LlamaConfig {
        vocab_size,
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

    let base_lm = Llama::random(Device::Cpu, base_cfg);
    engine.register_model("decode-model", Box::new(base_lm));

    let req_id = 42u64;
    let prompt_tokens = vec![10u32, 25u32, 40u32];
    let request = grim_scheduler::Request {
        id: req_id,
        model_id: Some("decode-model".to_string()),
        prompt_tokens: prompt_tokens.len(),
        max_new_tokens: 4,
        priority: 0,
        consumed_tokens: 0,
        adapter_ids: Vec::new(),
        input_ids: Some(prompt_tokens.clone()),
    };

    engine.enqueue_request(request).expect("enqueue request");

    // 1. First tick drives prefill
    let sched_out = engine.tick().expect("prefill tick");
    assert!(sched_out.prefill_ids.contains(&req_id));

    let outcome = engine.last_outcome(req_id).expect("prefill outcome");
    let logits_arc = outcome.logits.as_ref().expect("logits must be present");
    let full_logits = logits_arc.to_vec_f32().expect("logits to vec");
    assert_eq!(full_logits.len(), prompt_tokens.len() * vocab_size);
    // Extract last token logits: [vocab_size]
    let mut logits_vec = full_logits[(prompt_tokens.len() - 1) * vocab_size..].to_vec();
    assert_eq!(logits_vec.len(), vocab_size);

    // 2. Decode steps driving sampling -> engine.record_generated_token -> next decode forward
    for step in 0..3 {
        // Run greedy sampling on logits
        let mut best_tok = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &val) in logits_vec.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_tok = i as u32;
            }
        }

        // Record sampled token for next decode iteration
        engine.record_generated_token(req_id, best_tok);

        let decode_out = engine.tick().expect("decode tick");
        assert!(
            decode_out.decode_ids.contains(&req_id),
            "step {step}: request must be scheduled for decode"
        );

        let dec_outcome = engine.last_outcome(req_id).expect("decode outcome");
        assert!(
            dec_outcome.logits.is_some(),
            "step {step}: decode step must yield logits"
        );
        let dec_logits = dec_outcome.logits.as_ref().unwrap().to_vec_f32().unwrap();
        assert_eq!(dec_logits.len(), vocab_size);
        logits_vec = dec_logits;
    }
}
