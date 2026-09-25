//! Regression coverage for LFM2 blocks that combine attention and MoE FFN.

use grim_backend_cpu::cpu_tensor;
use grim_models_transformer::lfm2::{Lfm2AttentionMode, Lfm2Block};
use grim_models_transformer::shared_moe::CharonCache;
use grim_nn::{Linear, RmsNorm};
use grim_tensor::Shape;

fn linear(data: Vec<f32>, shape: Shape) -> Linear {
    Linear::from_tensor(cpu_tensor(data, shape), None)
}

fn norm(dim: usize) -> RmsNorm {
    RmsNorm {
        weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
        eps: 1e-5,
    }
}

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
        })
        .collect()
}

fn block(is_moe: bool) -> Lfm2Block {
    let hidden = 32usize;
    let head_dim = 8usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let inter = 64usize;
    let gate_data = values(inter * hidden, 55);
    let up_data = values(inter * hidden, 66);
    let down_data = values(hidden * inter, 77);
    let mut down_expert_data = vec![0.0; inter * hidden];
    for out_idx in 0..hidden {
        for in_idx in 0..inter {
            down_expert_data[in_idx * hidden + out_idx] = down_data[out_idx * inter + in_idx];
        }
    }

    Lfm2Block {
        index: 0,
        attn_norm: norm(hidden),
        wq: Some(linear(
            values(heads * head_dim * hidden, 11),
            Shape::new(vec![heads * head_dim, hidden]),
        )),
        wk: Some(linear(
            values(kv_heads * head_dim * hidden, 22),
            Shape::new(vec![kv_heads * head_dim, hidden]),
        )),
        wv: Some(linear(
            values(kv_heads * head_dim * hidden, 33),
            Shape::new(vec![kv_heads * head_dim, hidden]),
        )),
        wo: Some(linear(
            values(hidden * heads * head_dim, 44),
            Shape::new(vec![hidden, heads * head_dim]),
        )),
        attn_q_norm: Some(norm(head_dim)),
        attn_k_norm: Some(norm(head_dim)),
        wqkv_codes: None,
        wqkv_exps: None,
        gamma_q: None,
        gamma_k: None,
        w_gate_up_q80_fused: None,
        shortconv_in_proj: None,
        shortconv_conv: None,
        shortconv_conv_vec: None,
        shortconv_out_proj: None,
        ffn_norm: norm(hidden),
        ffn_gate: linear(gate_data.clone(), Shape::new(vec![inter, hidden])),
        ffn_up: linear(up_data.clone(), Shape::new(vec![inter, hidden])),
        ffn_down: linear(down_data, Shape::new(vec![hidden, inter])),
        ffn_gate_inp: if is_moe {
            Some(linear(values(hidden, 88), Shape::new(vec![1, hidden])))
        } else {
            None
        },
        ffn_gate_exps: if is_moe {
            Some(cpu_tensor(gate_data, Shape::new(vec![1, inter, hidden])))
        } else {
            None
        },
        ffn_up_exps: if is_moe {
            Some(cpu_tensor(up_data, Shape::new(vec![1, inter, hidden])))
        } else {
            None
        },
        ffn_down_exps: if is_moe {
            Some(cpu_tensor(
                down_expert_data,
                Shape::new(vec![inter, hidden, 1]),
            ))
        } else {
            None
        },
        ffn_exp_probs_b: None,
        is_moe,
        n_expert: if is_moe { 1 } else { 0 },
        n_expert_used: if is_moe { 1 } else { 0 },
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim,
        rope_theta: 10000.0,
        eps: 1e-5,
        charon_cache: CharonCache::new(),
        attention_mode: Lfm2AttentionMode::Softmax,
        gdl_gates: grim_models_transformer::gla::gdl_gate_defaults(head_dim),
        gdl_b_proj: None,
        gdl_w_proj: None,
        gdl_f_proj: None,
        gdl_fused_qkv_gates: None,
    }
}

#[test]
fn non_recurrent_moe_block_runs_attention_before_moe_ffn() -> Result<(), Box<dyn std::error::Error>>
{
    let hidden = 32usize;
    let input = cpu_tensor(values(2 * hidden, 99), Shape::new(vec![2, hidden]));

    let mut dense_cache = None;
    let dense = block(false)
        .forward(&input, &mut dense_cache)?
        .to_vec_f32()?;

    let mut moe_cache = None;
    let moe = block(true).forward(&input, &mut moe_cache)?.to_vec_f32()?;

    assert_eq!(dense.len(), moe.len());
    for (index, (expected, actual)) in dense.iter().zip(&moe).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-4,
            "element {index}: dense attention+FFN={expected}, MoE block={actual}"
        );
    }
    Ok(())
}
