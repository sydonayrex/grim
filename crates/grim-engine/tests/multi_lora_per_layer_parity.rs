//! Multi-LoRA per-layer parity test (Task 10.1 Step 1).
//!
//! Validates parity between:
//! - Arm A: Base weights + runtime per-layer multi-LoRA dispatch on `blk.0.attn_q` and `blk.0.attn_v`
//! - Arm B: Base weights with `W' = W + (alpha / r) * B * A` pre-merged into `q_proj` / `v_proj`.
//!
//! Asserts max abs logit difference < 1e-3.

use grim_core::model::CausalLm;
use grim_core::session::Inner;
use grim_models_transformer::{Llama, LlamaConfig};
use grim_tensor::{Device, Shape};

#[test]
fn test_multi_lora_per_layer_parity() {
    let cfg = LlamaConfig {
        vocab_size: 128,
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        num_layers: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 128,
        ..Default::default()
    };

    let dev = Device::Cpu;
    let base_model = Llama::random(dev.clone(), cfg.clone());
    let mut merged_model = Llama::random(dev.clone(), cfg.clone());

    // Rank-8 LoRA adapter on blk.0.attn_q and blk.0.attn_v
    let rank = 8usize;
    let in_dim = cfg.hidden_size; // 64
    let q_out_dim = cfg.num_heads * cfg.head_dim; // 64
    let v_out_dim = cfg.num_kv_heads * cfg.head_dim; // 32
    let alpha = 16.0f32;
    let scaling = alpha / (rank as f32);

    // Known non-zero deterministic weights for A and B
    let q_a: Vec<f32> = (0..rank * in_dim)
        .map(|i| (i as f32 * 0.17).sin() * 0.05)
        .collect();
    let q_b: Vec<f32> = (0..q_out_dim * rank)
        .map(|i| (i as f32 * 0.23).cos() * 0.05)
        .collect();

    let v_a: Vec<f32> = (0..rank * in_dim)
        .map(|i| (i as f32 * 0.31).sin() * 0.05)
        .collect();
    let v_b: Vec<f32> = (0..v_out_dim * rank)
        .map(|i| (i as f32 * 0.19).cos() * 0.05)
        .collect();

    // Arm B: Pre-merge delta = scaling * (B @ A) into merged_model.layers[0].wq and wv
    let merge_weights = |base_w: &mut [f32], b: &[f32], a: &[f32], out_dim: usize, in_dim: usize, r: usize, scale: f32| {
        for o in 0..out_dim {
            for i in 0..in_dim {
                let mut sum = 0.0f32;
                for k in 0..r {
                    sum += b[o * r + k] * a[k * in_dim + i];
                }
                base_w[o * in_dim + i] += scale * sum;
            }
        }
    };

    let mut wq_data = merged_model.layers[0].wq.weight().to_vec_f32().expect("wq vec");
    merge_weights(&mut wq_data, &q_b, &q_a, q_out_dim, in_dim, rank, scaling);
    merged_model.layers[0].wq = grim_nn::ColumnParallelLinear::new(
        grim_nn::Linear::from_tensor(
            grim_backend_cpu::cpu_tensor(wq_data, Shape::new(vec![q_out_dim, in_dim])),
            None,
        ),
        grim_nn::TensorParallelConfig::default(),
    );

    let mut wv_data = merged_model.layers[0].wv.weight().to_vec_f32().expect("wv vec");
    merge_weights(&mut wv_data, &v_b, &v_a, v_out_dim, in_dim, rank, scaling);
    merged_model.layers[0].wv = grim_nn::ColumnParallelLinear::new(
        grim_nn::Linear::from_tensor(
            grim_backend_cpu::cpu_tensor(wv_data, Shape::new(vec![v_out_dim, in_dim])),
            None,
        ),
        grim_nn::TensorParallelConfig::default(),
    );

    // Run forward pass on Arm B (merged)
    let prompt_tokens = vec![1u32, 5, 20, 42];
    let seq_len = prompt_tokens.len();
    let ids = grim_backend_cpu::cpu_tensor(
        prompt_tokens.iter().map(|&x| x as f32).collect(),
        Shape::new(vec![seq_len]),
    );
    let pos = grim_backend_cpu::cpu_tensor(
        (0..seq_len).map(|x| x as f32).collect(),
        Shape::new(vec![seq_len]),
    );

    let mut sess_b = Inner::new(dev.clone());
    let logits_b = merged_model
        .forward(&mut sess_b, &ids, &pos, &[])
        .expect("arm B forward")
        .to_vec_f32()
        .expect("logits B vec");

    // Arm A: Base weights with runtime adapter handles targeted at layer 0 attn_q and attn_v
    let adapter_q = grim_core::model::AdapterHandle::with_target(
        1,
        grim_backend_cpu::cpu_tensor(q_a, Shape::new(vec![rank, in_dim])),
        grim_backend_cpu::cpu_tensor(q_b, Shape::new(vec![q_out_dim, rank])),
        alpha,
        0,
        grim_core::model::TargetModule::AttnQ,
    );
    let adapter_v = grim_core::model::AdapterHandle::with_target(
        2,
        grim_backend_cpu::cpu_tensor(v_a, Shape::new(vec![rank, in_dim])),
        grim_backend_cpu::cpu_tensor(v_b, Shape::new(vec![v_out_dim, rank])),
        alpha,
        0,
        grim_core::model::TargetModule::AttnV,
    );

    let mut sess_a = Inner::new(dev.clone());
    let logits_a = base_model
        .forward(&mut sess_a, &ids, &pos, &[adapter_q, adapter_v])
        .expect("arm A forward")
        .to_vec_f32()
        .expect("logits A vec");

    let mut max_diff = 0.0f32;
    for (&a, &b) in logits_a.iter().zip(logits_b.iter()) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(
        max_diff < 1e-3,
        "multi-LoRA per-layer parity failure: max_diff = {max_diff} >= 1e-3"
    );
}
