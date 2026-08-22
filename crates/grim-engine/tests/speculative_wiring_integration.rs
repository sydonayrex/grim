//! Integration test for speculative decoding registration with EAGLE3 and drafters.

use grim_engine::Engine;
use grim_models_transformer::{Eagle3, Eagle3Config, Llama, LlamaConfig};
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{Device, Shape};
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

    let base_lm = Llama {
        cfg: base_cfg.clone(),
        device: Device::Cpu,
        tok_embeddings: Embedding {
            weight: grim_backend_cpu::cpu_tensor(vec![0.01; 100 * 64], Shape::new(vec![100, 64])),
        },
        layers: Vec::new(),
        moe_blocks: Vec::new(),
        norm: RmsNorm::new(
            grim_backend_cpu::cpu_tensor(vec![1.0; 64], Shape::new(vec![64])),
            1e-5,
        ),
        output: Linear::from_tensor(
            grim_backend_cpu::cpu_tensor(vec![0.01; 100 * 64], Shape::new(vec![100, 64])),
            None,
        ),
    };

    // 2. Create mock EAGLE3 model
    let eagle3_cfg = Eagle3Config {
        vocab_size: 100,
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        num_layers: 1,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 256,
    };

    let eagle3 = Eagle3 {
        cfg: eagle3_cfg,
        device: Device::Cpu,
        inner: Llama {
            cfg: base_cfg,
            device: Device::Cpu,
            tok_embeddings: Embedding {
                weight: grim_backend_cpu::cpu_tensor(
                    vec![0.01; 100 * 64],
                    Shape::new(vec![100, 64]),
                ),
            },
            layers: Vec::new(),
            moe_blocks: Vec::new(),
            norm: RmsNorm::new(
                grim_backend_cpu::cpu_tensor(vec![1.0; 64], Shape::new(vec![64])),
                1e-5,
            ),
            output: Linear::from_tensor(
                grim_backend_cpu::cpu_tensor(vec![0.01; 100 * 64], Shape::new(vec![100, 64])),
                None,
            ),
        },
    };

    // 3. Register EAGLE3 model with Engine
    engine.register_eagle3_model("test-model", Box::new(base_lm), Arc::new(eagle3));

    assert!(engine.has_model("test-model"));
}
