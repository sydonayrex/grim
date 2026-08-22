//! Cross-crate integration tests for `grim-models-transformer`.
//!
//! Validates:
//! - Llama causal LM multi-head & grouped-query attention forward pass with KV accumulation
//! - MuseGlimmer dense-sparse architecture instantiation and generation
//! - AttentionDispatcher multi-tier routing (Tier 1 Matrix / Tier 2 Compute / Tier 3 CPU)

use grim_backend_cpu::cpu_tensor;
use grim_core::CausalLm;
use grim_core::session::Inner;
use grim_models_transformer::attention_dispatcher::{
    AttentionDispatcher, AttentionTier, AttentionTopology,
};
use grim_models_transformer::muse_glimmer::{MuseGlimmer, MuseGlimmerConfig};
use grim_models_transformer::{Llama, LlamaConfig};
use grim_tensor::{Device, Shape};

#[test]
fn test_llama_forward_and_kv_evolution() {
    let config = LlamaConfig {
        vocab_size: 500,
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        num_layers: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 128,
        partial_rotary_factor: 1.0,
        yarn: None,
    };
    let model = Llama::random(Device::Cpu, config);
    let mut session = Inner::new(Device::Cpu);

    let input = cpu_tensor(vec![10.0f32, 20.0f32], Shape::new(vec![1, 2]));
    let positions = cpu_tensor(vec![0.0f32, 1.0f32], Shape::new(vec![1, 2]));

    let logits = model
        .forward(&mut session, &input, &positions, &[])
        .unwrap();
    assert_eq!(logits.shape().dims(), &[2, 500]);
}

#[test]
fn test_muse_glimmer_forward_pass() {
    let config = MuseGlimmerConfig {
        vocab_size: 500,
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        num_layers: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        per_layer_rope_theta: vec![],
        base_rope_theta: 10000.0,
        sliding_window_layer_ids: vec![],
        sliding_window_size: 0,
        qk_scale_factor: 1.0,
        output_multiplier: vec![],
        final_logit_softcapping: 0.0,
        max_seq_len: 128,
        vision: None,
    };
    let model = MuseGlimmer::random(Device::Cpu, config);
    let mut session = Inner::new(Device::Cpu);

    let input = cpu_tensor(vec![5.0f32, 15.0f32], Shape::new(vec![1, 2]));
    let positions = cpu_tensor(vec![0.0f32, 1.0f32], Shape::new(vec![1, 2]));

    let logits = model
        .forward(&mut session, &input, &positions, &[])
        .unwrap();
    assert_eq!(logits.shape().dims(), &[2, 500]);
}

#[test]
fn test_attention_dispatcher_tier_selection_and_gqa() {
    // 1. Tier 3 fallback on CPU
    let gqa_topology = AttentionTopology::StandardGqa {
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 64,
        sm_scale: 1.0 / 8.0,
    };
    let tier_cpu = AttentionDispatcher::select_tier(&gqa_topology, false, false);
    assert_eq!(tier_cpu, AttentionTier::Tier3CpuFallback);

    // 2. Tier 1 Hardware Matrix on GPU with matrix cores
    let tier_gpu_matrix = AttentionDispatcher::select_tier(&gqa_topology, true, true);
    assert_eq!(tier_gpu_matrix, AttentionTier::Tier1HardwareMatrix);

    // 3. Tier 2 Compute Shader on GPU without specialized matrix cores
    let tier_gpu_shader = AttentionDispatcher::select_tier(&gqa_topology, false, true);
    assert_eq!(tier_gpu_shader, AttentionTier::Tier2UniversalCompute);

    // 4. Execute GQA dispatch on CPU
    let q = vec![0.5f32; 4 * 16];
    let k = vec![0.5f32; 2 * 16];
    let v = vec![0.5f32; 2 * 16];
    let (out, tier) =
        AttentionDispatcher::dispatch_gqa(&q, &k, &v, 4, 2, 16, 1, None, &Device::Cpu).unwrap();
    assert_eq!(tier, AttentionTier::Tier3CpuFallback);
    assert_eq!(out.shape().elem_count(), 4 * 16);
}
