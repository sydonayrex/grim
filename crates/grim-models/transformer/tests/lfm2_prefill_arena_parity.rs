//! A5 (PLAN-reduce-d2h-h2d / WI-X2-PREFILL-ARENA) parity: the eager
//! prefill-arena path (RoPE/KV device-resident, zero D2H, attention via
//! `fused_or_scalar_attention_arena_device`) must produce the same layer
//! output as the legacy host path (rope D2H, host KV mirrors, run on the
//! CPU device — who-dat P1-5 removed the legacy path from ROCm), across a
//! multi-step
//! forward where the second call's cache_offset comes from `dev_pos`.
//!
//! RUN: GRIM_GPU_TEST=1 cargo test -p grim-models-transformer \
//!   --test lfm2_prefill_arena_parity

use std::panic;
use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_models_transformer::lfm2::Lfm2Block;
use grim_models_transformer::shared_moe::CharonCache;
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

fn gpu() -> Option<(RocmDevice, usize)> {
    if !grim_backend_rocm::gpu_test_enabled() {
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        return None;
    }
    let dev = panic::catch_unwind(|| RocmDevice::try_new(0).unwrap()).ok()?;
    Some((dev, 0))
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
        })
        .collect()
}

fn rocm_tensor(
    dev: &RocmDevice,
    ordinal: usize,
    data: Vec<f32>,
    shape: Shape,
) -> Tensor {
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        Device::Rocm(ordinal),
    )
}

fn test_linear(dev: &RocmDevice, ordinal: usize, out_dim: usize, in_dim: usize, seed: u64) -> Linear {
    let w = rocm_tensor(dev, ordinal, rand_vec(out_dim * in_dim, seed), Shape::new(vec![out_dim, in_dim]));
    Linear { weight: w.clone(), bias: None, w_t: w, quant_format: None }
}

fn test_norm(dev: &RocmDevice, ordinal: usize, n: usize) -> RmsNorm {
    RmsNorm {
        weight: rocm_tensor(dev, ordinal, vec![1.0; n], Shape::new(vec![n])),
        eps: 1e-5,
    }
}

fn attention_block(dev: &RocmDevice, ordinal: usize) -> Lfm2Block {
    let hidden = 32usize;
    let hd = 8usize;
    let nh = 2usize;
    let nkv = 1usize;
    let inter = 64usize;
    let n_q = nh * hd;
    let n_kv = nkv * hd;
    Lfm2Block {
        attn_norm: test_norm(dev, ordinal, hidden),
        wq: Some(test_linear(dev, ordinal, n_q, hidden, 11)),
        wk: Some(test_linear(dev, ordinal, n_kv, hidden, 22)),
        wv: Some(test_linear(dev, ordinal, n_kv, hidden, 33)),
        wo: Some(test_linear(dev, ordinal, hidden, n_q, 44)),
        attn_q_norm: Some(test_norm(dev, ordinal, hd)),
        attn_k_norm: Some(test_norm(dev, ordinal, hd)),
        wqkv_codes: None,
        wqkv_exps: None,
        gamma_q: None,
        gamma_k: None,
        w_gate_up_q80_fused: None,
        shortconv_in_proj: None,
        shortconv_conv: None,
        shortconv_conv_vec: None,
        shortconv_out_proj: None,
        ffn_norm: test_norm(dev, ordinal, hidden),
        ffn_gate: test_linear(dev, ordinal, inter, hidden, 55),
        ffn_up: test_linear(dev, ordinal, inter, hidden, 66),
        ffn_down: test_linear(dev, ordinal, hidden, inter, 77),
        ffn_gate_inp: None,
        ffn_gate_exps: None,
        ffn_up_exps: None,
        ffn_down_exps: None,
        ffn_exp_probs_b: None,
        is_moe: false,
        n_expert: 0,
        n_expert_used: 0,
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        rope_theta: 10000.0,
        eps: 1e-5,
        charon_cache: CharonCache::new(),
    }
}

// Reference: the SAME weights built on the CPU device run the legacy host
// path (rope D2H, host KV mirrors) — since who-dat P1-5 removed the legacy
// path from ROCm, the CPU block is the only remaining reference for it.
fn cpu_block() -> Lfm2Block {
    let hidden = 32usize;
    let hd = 8usize;
    let nh = 2usize;
    let nkv = 1usize;
    let inter = 64usize;
    let n_q = nh * hd;
    let n_kv = nkv * hd;
    let cpu_lin = |out: usize, inp: usize, seed: u64| -> Linear {
        Linear {
            weight: grim_backend_cpu::cpu_tensor(rand_vec(out * inp, seed), Shape::new(vec![out, inp])),
            bias: None,
            w_t: grim_backend_cpu::cpu_tensor(
                rand_vec(out * inp, seed),
                Shape::new(vec![out, inp]),
            ),
            quant_format: None,
        }
    };
    let cpu_norm = |n: usize| -> RmsNorm {
        RmsNorm {
            weight: grim_backend_cpu::cpu_tensor(vec![1.0; n], Shape::new(vec![n])),
            eps: 1e-5,
        }
    };
    Lfm2Block {
        attn_norm: cpu_norm(hidden),
        wq: Some(cpu_lin(n_q, hidden, 11)),
        wk: Some(cpu_lin(n_kv, hidden, 22)),
        wv: Some(cpu_lin(n_kv, hidden, 33)),
        wo: Some(cpu_lin(hidden, n_q, 44)),
        attn_q_norm: Some(cpu_norm(hd)),
        attn_k_norm: Some(cpu_norm(hd)),
        wqkv_codes: None,
        wqkv_exps: None,
        gamma_q: None,
        gamma_k: None,
        w_gate_up_q80_fused: None,
        shortconv_in_proj: None,
        shortconv_conv: None,
        shortconv_conv_vec: None,
        shortconv_out_proj: None,
        ffn_norm: cpu_norm(hidden),
        ffn_gate: cpu_lin(inter, hidden, 55),
        ffn_up: cpu_lin(inter, hidden, 66),
        ffn_down: cpu_lin(hidden, inter, 77),
        ffn_gate_inp: None,
        ffn_gate_exps: None,
        ffn_up_exps: None,
        ffn_down_exps: None,
        ffn_exp_probs_b: None,
        is_moe: false,
        n_expert: 0,
        n_expert_used: 0,
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        rope_theta: 10000.0,
        eps: 1e-5,
        charon_cache: CharonCache::new(),
    }
}

#[test]
fn prefill_arena_matches_host_path() {
    let Some((dev, ordinal)) = gpu() else {
        eprintln!("[SKIP] requires GRIM_GPU_TEST=1 + ROCm device");
        return;
    };
    let _ = &dev;
    // Target the arena branch: the device decode path (GRIM_DECODE_GRAPH
    // default-on) handles single-token steps itself and would bypass it.
    // SAFETY: test process; no concurrent reader of this env var.
    unsafe {
        std::env::set_var("GRIM_DECODE_GRAPH", "0");
    }

    let hidden = 32usize;
    // 2-step prefill then two 1-token steps: the later calls prove the arena
    // path's dev_pos-derived cache_offset agrees with the host path's
    // mirror-length-derived offset.
    let x_all: Vec<f32> = rand_vec(4 * hidden, 7);
    let mut out_host = Vec::new();
    let mut out_arena = Vec::new();

    let mut cache_host = None;
    {
        let block = cpu_block();
        let mut cursor = 0usize;
        for n in [2usize, 1, 1] {
            let x = grim_backend_cpu::cpu_tensor(
                x_all[cursor..cursor + n * hidden].to_vec(),
                Shape::new(vec![n, hidden]),
            );
            cursor += n * hidden;
            out_host.extend(block.forward(&x, &mut cache_host).unwrap().to_vec_f32().unwrap());
        }
    }

    let mut cache_arena = None;
    {
        let block = attention_block(&dev, ordinal);
        let mut cursor = 0usize;
        for n in [2usize, 1, 1] {
            let x = rocm_tensor(
                &dev,
                ordinal,
                x_all[cursor..cursor + n * hidden].to_vec(),
                Shape::new(vec![n, hidden]),
            );
            cursor += n * hidden;
            out_arena.extend(block.forward(&x, &mut cache_arena).unwrap().to_vec_f32().unwrap());
        }
    }

    // Arena mode must leave the host mirrors empty and track rows in dev_pos.
    match cache_arena.as_ref() {
        Some(grim_models_transformer::lfm2::Lfm2LayerCache::Attention { k, dev_pos, .. }) => {
            assert!(k.is_empty(), "arena path must not extend the host K mirror");
            assert_eq!(*dev_pos, 4, "dev_pos must track 4 arena rows after 2+1+1 steps");
        }
        _ => panic!("expected attention cache"),
    }
    // CPU host path must have mirrored 4 rows.
    match cache_host.as_ref() {
        Some(grim_models_transformer::lfm2::Lfm2LayerCache::Attention { k, .. }) => {
            assert_eq!(k.len(), 4 * 8, "host path must mirror 4 rows of head_dim 8");
        }
        _ => panic!("expected attention cache"),
    }

    assert_eq!(out_host.len(), out_arena.len());
    let mut worst = 0.0f32;
    for (i, (a, b)) in out_host.iter().zip(&out_arena).enumerate() {
        worst = worst.max((a - b).abs());
        assert!((a - b).abs() < 1e-3, "elem {i}: host {a} vs arena {b}");
    }
    eprintln!("A5 prefill-arena parity ok (worst |Δ| = {worst:.3e})");
}
