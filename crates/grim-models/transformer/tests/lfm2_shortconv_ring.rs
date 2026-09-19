//! S1 (PLAN-kernel-fusion): the ShortConv device ring buffer (device-resident
//! conv state updated in place) must evolve the state byte-for-reason vs the
//! host-loop reference, across 3 successive decode steps on ROCm.
//!
//! Gated: GRIM_GPU_TEST=1 + ROCm device present (house rule).

use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::device::util::gpu_test_enabled;
use grim_models_transformer::lfm2::{Lfm2Block, Lfm2LayerCache};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

fn tensor(dev: &RocmDevice, ordinal: usize, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        Device::Rocm(ordinal),
    )
}

fn cpu_t(data: Vec<f32>, shape: Shape) -> Tensor {
    grim_backend_cpu::cpu_tensor(data, shape)
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) % 2000) as f32 - 1000.0) / 1000.0 * 0.4
        })
        .collect()
}

fn lin_rocm(dev: &RocmDevice, ordinal: usize, w: Vec<f32>, out_dim: usize, in_dim: usize) -> Linear {
    Linear::from_tensor(tensor(dev, ordinal, w, Shape::new(vec![out_dim, in_dim])), None)
}

fn norm_rocm(dev: &RocmDevice, ordinal: usize, dim: usize) -> RmsNorm {
    RmsNorm {
        weight: tensor(dev, ordinal, vec![1.0f32; dim], Shape::new(vec![dim])),
        eps: 1e-5,
    }
}

fn shortconv_block(dev: &RocmDevice, ordinal: usize, hidden: usize, l_cache: usize) -> Lfm2Block {
    Lfm2Block {
        attn_norm: norm_rocm(dev, ordinal, hidden),
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        attn_q_norm: None,
        attn_k_norm: None,
        wqkv_codes: None,
        wqkv_exps: None,
        gamma_q: None,
        gamma_k: None,
        w_gate_up_q80_fused: None,
        shortconv_in_proj: Some(lin_rocm(dev, ordinal, rand_vec(3 * hidden * hidden, 1), 3 * hidden, hidden)),
        shortconv_conv: Some(tensor(
            dev,
            ordinal,
            rand_vec(hidden * l_cache, 2),
            Shape::new(vec![hidden, 1, l_cache]),
        )),
        shortconv_conv_vec: Some(rand_vec(hidden * l_cache, 2)),
        shortconv_out_proj: Some(lin_rocm(dev, ordinal, rand_vec(hidden * hidden, 3), hidden, hidden)),
        ffn_norm: norm_rocm(dev, ordinal, hidden),
        ffn_gate: lin_rocm(dev, ordinal, rand_vec(hidden * hidden, 4), hidden, hidden),
        ffn_up: lin_rocm(dev, ordinal, rand_vec(hidden * hidden, 5), hidden, hidden),
        ffn_down: lin_rocm(dev, ordinal, rand_vec(hidden * hidden, 6), hidden, hidden),
        ffn_gate_inp: None,
        ffn_gate_exps: None,
        ffn_up_exps: None,
        ffn_down_exps: None,
        ffn_exp_probs_b: None,
        is_moe: false,
        n_expert: 0,
        n_expert_used: 1,
        charon_cache: grim_models_transformer::shared_moe::CharonCache::new(),
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        rope_theta: 10000.0,
        eps: 1e-5,
    }
}

#[test]
fn shortconv_device_ring_matches_host_reference() {
    if !gpu_test_enabled() {
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        eprintln!("skip: set GRIM_GPU_TEST=1");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let dev = RocmDevice::shared(0);
    let hidden = 64usize;
    let l_cache = 3usize;
    let steps = 5usize;
    let x_data = rand_vec(steps * hidden, 42);

    // GPU path: 5 single-token calls; the device ring holds state between them.
    let block_gpu = shortconv_block(&dev, 0, hidden, l_cache);
    let mut cache_gpu: Option<Lfm2LayerCache> = None;
    let mut out_gpu = Vec::new();
    for t in 0..steps {
        let x = tensor(
            &dev,
            0,
            x_data[t * hidden..(t + 1) * hidden].to_vec(),
            Shape::new(vec![1, hidden]),
        );
        out_gpu.extend(
            block_gpu
                .forward(&x, &mut cache_gpu)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
        );
    }

    // CPU reference: same weights pulled back to host into a CPU block.
    // ShortConv reference math = `full_ref` in
    // `shortconv_decode_matches_prefill` (host loop from block weights).
    let block_cpu = Lfm2Block {
        attn_norm: RmsNorm {
            weight: cpu_t(vec![1.0f32; hidden], Shape::new(vec![hidden])),
            eps: 1e-5,
        },
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        attn_q_norm: None,
        attn_k_norm: None,
        wqkv_codes: None,
        wqkv_exps: None,
        gamma_q: None,
        gamma_k: None,
        w_gate_up_q80_fused: None,
        shortconv_in_proj: Some(Linear::from_tensor(
            cpu_t(rand_vec(3 * hidden * hidden, 1), Shape::new(vec![3 * hidden, hidden])),
            None,
        )),
        shortconv_conv: Some(cpu_t(
            rand_vec(hidden * l_cache, 2),
            Shape::new(vec![hidden, 1, l_cache]),
        )),
        shortconv_conv_vec: Some(rand_vec(hidden * l_cache, 2)),
        shortconv_out_proj: Some(Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 3), Shape::new(vec![hidden, hidden])),
            None,
        )),
        ffn_norm: RmsNorm {
            weight: cpu_t(vec![1.0f32; hidden], Shape::new(vec![hidden])),
            eps: 1e-5,
        },
        ffn_gate: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 4), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_up: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 5), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_down: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 6), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_gate_inp: None,
        ffn_gate_exps: None,
        ffn_up_exps: None,
        ffn_down_exps: None,
        ffn_exp_probs_b: None,
        is_moe: false,
        n_expert: 0,
        n_expert_used: 1,
        charon_cache: grim_models_transformer::shared_moe::CharonCache::new(),
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        rope_theta: 10000.0,
        eps: 1e-5,
    };
    let mut cache_cpu: Option<Lfm2LayerCache> = None;
    let mut out_cpu = Vec::new();
    for t in 0..steps {
        let x = cpu_t(
            x_data[t * hidden..(t + 1) * hidden].to_vec(),
            Shape::new(vec![1, hidden]),
        );
        out_cpu.extend(
            block_cpu
                .forward(&x, &mut cache_cpu)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
        );
    }

    assert_eq!(out_gpu.len(), out_cpu.len());
    let max_diff = out_gpu
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "device ring vs host loop diverged: max_diff={max_diff}"
    );

    // The device ring must now exist (i.e. isn't lazily re-uploaded per step).
    match &cache_gpu {
        Some(Lfm2LayerCache::ShortConv { dev, .. }) => {
            assert!(dev.is_some(), "device ring must be allocated on ROCm");
        }
        _ => panic!("expected ShortConv cache"),
    }
}

#[test]
fn test_shortconv_prefill_then_decode_matches_reference() {
    if !gpu_test_enabled() {
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        eprintln!("skip: set GRIM_GPU_TEST=1");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let dev = RocmDevice::shared(0);
    let hidden = 64usize;
    let l_cache = 3usize;
    let prefill_len = 4usize;
    let decode_steps = 3usize;
    let total_steps = prefill_len + decode_steps;
    let x_data = rand_vec(total_steps * hidden, 101);

    // GPU path: 1 multi-token prefill call (steps=4) followed by 3 single-token decode calls (steps=1).
    let block_gpu = shortconv_block(&dev, 0, hidden, l_cache);
    let mut cache_gpu: Option<Lfm2LayerCache> = None;
    let mut out_gpu = Vec::new();

    // 1. Prefill
    let x_prefill = tensor(
        &dev,
        0,
        x_data[..prefill_len * hidden].to_vec(),
        Shape::new(vec![prefill_len, hidden]),
    );
    out_gpu.extend(
        block_gpu
            .forward(&x_prefill, &mut cache_gpu)
            .unwrap()
            .to_vec_f32()
            .unwrap(),
    );

    // 2. Decode steps
    for t in 0..decode_steps {
        let step = prefill_len + t;
        let x_tok = tensor(
            &dev,
            0,
            x_data[step * hidden..(step + 1) * hidden].to_vec(),
            Shape::new(vec![1, hidden]),
        );
        out_gpu.extend(
            block_gpu
                .forward(&x_tok, &mut cache_gpu)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
        );
    }

    // CPU reference: 1 full prefill call over all total_steps tokens.
    let block_cpu = Lfm2Block {
        attn_norm: RmsNorm {
            weight: cpu_t(vec![1.0f32; hidden], Shape::new(vec![hidden])),
            eps: 1e-5,
        },
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        attn_q_norm: None,
        attn_k_norm: None,
        wqkv_codes: None,
        wqkv_exps: None,
        gamma_q: None,
        gamma_k: None,
        w_gate_up_q80_fused: None,
        shortconv_in_proj: Some(Linear::from_tensor(
            cpu_t(rand_vec(3 * hidden * hidden, 1), Shape::new(vec![3 * hidden, hidden])),
            None,
        )),
        shortconv_conv: Some(cpu_t(
            rand_vec(hidden * l_cache, 2),
            Shape::new(vec![hidden, 1, l_cache]),
        )),
        shortconv_conv_vec: Some(rand_vec(hidden * l_cache, 2)),
        shortconv_out_proj: Some(Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 3), Shape::new(vec![hidden, hidden])),
            None,
        )),
        ffn_norm: RmsNorm {
            weight: cpu_t(vec![1.0f32; hidden], Shape::new(vec![hidden])),
            eps: 1e-5,
        },
        ffn_gate: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 4), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_up: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 5), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_down: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 6), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_gate_inp: None,
        ffn_gate_exps: None,
        ffn_up_exps: None,
        ffn_down_exps: None,
        ffn_exp_probs_b: None,
        is_moe: false,
        n_expert: 0,
        n_expert_used: 1,
        charon_cache: grim_models_transformer::shared_moe::CharonCache::new(),
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        rope_theta: 10000.0,
        eps: 1e-5,
    };
    let mut cache_cpu: Option<Lfm2LayerCache> = None;
    let x_all_cpu = cpu_t(x_data, Shape::new(vec![total_steps, hidden]));
    let out_cpu = block_cpu
        .forward(&x_all_cpu, &mut cache_cpu)
        .unwrap()
        .to_vec_f32()
        .unwrap();

    assert_eq!(out_gpu.len(), out_cpu.len());
    let max_diff = out_gpu
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "prefill-then-decode diverged from reference: max_diff={max_diff}"
    );
}

// ===========================================================================
// Regression (A1/S1 follow-up): the host mirror must stay fresh after device
// decode steps even now that the GRIM_DECODE_GRAPH opt-in gate is gone.
// `Clone for Lfm2LayerCache` copies ONLY the host ring (device ring starts
// fresh), so a stale host mirror would silently fork state on clone.
// =========================================================================

#[test]
fn shortconv_host_mirror_fresh_after_device_steps_clone_regression() {
    if !gpu_test_enabled() {
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
        return;
    }
    let dev = RocmDevice::shared(0);
    let hidden = 16usize;
    let l_cache = 3usize;

    let block = shortconv_block(&dev, 0, hidden, l_cache);
    let mut cache = Some(Lfm2LayerCache::ShortConv {
        host: vec![0.0f32; hidden * (l_cache - 1)],
        dev: None,
    });

    // 4 device-path decode steps (steps == 1 → shortconv_step_device).
    let mut next_input = vec![0.5f32; hidden];
    for _t in 0..4 {
        let x = tensor(
            &dev,
            0,
            next_input.clone(),
            Shape::new(vec![1, hidden]),
        );
        let y = block.forward(&x, &mut cache).unwrap();
        next_input = y.to_vec_f32().unwrap()[..hidden].to_vec();
    }

    // The host ring must hold the last kc-1 bx rows — all non-zero (inputs
    // and weights are non-degenerate), proving the mirror was re-synced.
    match cache.as_ref().unwrap() {
        Lfm2LayerCache::ShortConv { host, dev: ring } => {
            assert!(ring.is_some(), "device ring must exist on ROCm");
            assert!(
                host.iter().any(|&v| v != 0.0),
                "host mirror stale after device steps: clone would fork state"
            );
        }
        _ => panic!("expected ShortConv cache"),
    }

    // Behavioral check: continuing from the clone must match continuing from
    // the original (a stale mirror makes the clone's first step diverge).
    let cloned = cache.as_ref().unwrap().clone();
    let mut cache_a = Some(match cloned {
        Lfm2LayerCache::ShortConv { host, .. } => Lfm2LayerCache::ShortConv {
            host,
            dev: None,
        },
        _ => unreachable!(),
    });
    let mut cache_b = cache;

    let x5 = tensor(&dev, 0, next_input.clone(), Shape::new(vec![1, hidden]));
    let y_a = block.forward(&x5, &mut cache_a).unwrap().to_vec_f32().unwrap();
    let y_b = block.forward(&x5, &mut cache_b).unwrap().to_vec_f32().unwrap();
    let max_diff = y_a
        .iter()
        .zip(y_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "cloned cache diverged from original after device steps: max_diff={max_diff}"
    );
}

// ===========================================================================
// Integration (A1/S1 closure): a recurrent (ShortConv) LFM2 must decode
// IDENTICALLY through the HIP decode graph (seeded conv rings + capture +
// replay) as through the eager path. Regression: graph `sc_state` was never
// seeded from the eager session, so every post-prefill replay ran conv layers
// against a zeroed ring.
//
// Mirrors the production sequence in run.rs `try_graph_decode_step`:
// warmups (unseeded, discarded) -> capture (RECORDS ONLY, does not execute)
// -> eager step for the capture token -> re-seed from session -> replay.
// =========================================================================

use grim_core::model::CausalLm;
use grim_models_transformer::{DecodeGraphModel, Lfm2, Lfm2Config};
use grim_nn::Embedding;

fn conv_lfm2(dev: &RocmDevice, ordinal: usize, n_layers: usize) -> Lfm2 {
    let hidden = 16usize;
    let l_cache = 3usize;
    let inter = 32usize;
    let vocab = 32usize;
    let layers = (0..n_layers)
        .map(|l| {
            let block = shortconv_block(dev, ordinal, hidden, l_cache);
            Lfm2Block {
                shortconv_in_proj: Some(lin_rocm(dev, ordinal, rand_vec(3 * hidden * hidden, (100 + l * 10 + 1) as u64), 3 * hidden, hidden)),
                shortconv_conv: Some(tensor(dev, ordinal, rand_vec(hidden * l_cache, (100 + l * 10 + 2) as u64), Shape::new(vec![hidden, 1, l_cache]))),
                shortconv_conv_vec: Some(rand_vec(hidden * l_cache, (100 + l * 10 + 2) as u64)),
                shortconv_out_proj: Some(lin_rocm(dev, ordinal, rand_vec(hidden * hidden, (100 + l * 10 + 3) as u64), hidden, hidden)),
                ffn_gate: lin_rocm(dev, ordinal, rand_vec(inter * hidden, (100 + l * 10 + 4) as u64), inter, hidden),
                ffn_up: lin_rocm(dev, ordinal, rand_vec(inter * hidden, (100 + l * 10 + 5) as u64), inter, hidden),
                ffn_down: lin_rocm(dev, ordinal, rand_vec(inter * hidden, (100 + l * 10 + 6) as u64), hidden, inter),
                ..block
            }
        })
        .collect();
    Lfm2 {
        cfg: Lfm2Config {
            vocab_size: vocab,
            hidden_size: hidden,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden,
            num_layers: n_layers,
            intermediate_size: inter,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            n_shortconv_l_cache: l_cache,
            is_recr: vec![true; n_layers],
            n_layer_dense_lead: 0,
            n_expert: 0,
            n_expert_used: 0,
            n_ff_exp: 0,
            n_embd_out: 0,
            mxfp4_qkv_attention: false,
        },
        device: Device::Rocm(ordinal),
        tok_embeddings: Embedding {
            weight: tensor(dev, ordinal, rand_vec(vocab * hidden, 900u64), Shape::new(vec![vocab, hidden])),
        },
        layers,
        norm: norm_rocm(dev, ordinal, hidden),
        output: lin_rocm(dev, ordinal, rand_vec(vocab * hidden, 901u64), vocab, hidden),
        dense_2_out: None,
        dense_2_out_bias: None,
    }
}

fn run_step(
    dev: &RocmDevice,
    model: &Lfm2,
    session: &mut Box<dyn grim_core::session::SessionT>,
    token: u32,
) -> Vec<f32> {
    let x = tensor(dev, 0, vec![token as f32], Shape::new(vec![1]));
    let pos = tensor(dev, 0, vec![0.0], Shape::new(vec![1]));
    let out = CausalLm::forward(model, session.as_mut(), &x, &pos, &[]).unwrap();
    out.to_vec_f32().unwrap()
}

#[test]
fn shortconv_lfm2_graph_replay_matches_eager_after_prefill() {
    if !gpu_test_enabled() {
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
        return;
    }
    let dev = RocmDevice::shared(0);

    // Two identical conv-only models: one decoded through the graph, one eager.
    let model_graph = conv_lfm2(&dev, 0, 2);
    let model_eager = conv_lfm2(&dev, 0, 2);

    let prompt = [5u32, 9, 17, 3];
    let capture_tok = 21u32;
    let replay_toks = [7u32, 12, 30];

    // ---- Graph path: eager prefill, warmups, capture, re-seed, replays ----
    let mut sess_g = CausalLm::new_session(&model_graph);
    for &t in &prompt {
        let _ = run_step(&dev, &model_graph, &mut sess_g, t);
    }

    let mut graph = DecodeGraphModel::get_or_create_decode_graph(&model_graph, 16, 1).unwrap();
    // Warmups run unseeded; the seed below discards their state writes.
    let _ = DecodeGraphModel::forward_capture(&model_graph, &mut graph, capture_tok);
    let _ = DecodeGraphModel::forward_capture(&model_graph, &mut graph, capture_tok);
    graph.begin_capture().unwrap();
    DecodeGraphModel::forward_capture(&model_graph, &mut graph, capture_tok).unwrap();
    graph.end_capture().unwrap();
    assert!(graph.is_captured);

    // Production: the capture-step token runs EAGERLY on the session (its
    // logits produce this step's token), then the graph is re-seeded from the
    // session so the first replay continues exactly where the session is.
    let _ = run_step(&dev, &model_graph, &mut sess_g, capture_tok);
    let srcs = DecodeGraphModel::eager_kv_seed_sources(
        &model_graph,
        sess_g.as_ref(),
        (prompt.len() + 1) as u32,
    )
    .unwrap();
    graph.buffers.seed_kv_arena_from_eager(&dev, &srcs).unwrap();
    let conv_seeds = DecodeGraphModel::eager_conv_seed_rings(&model_graph, sess_g.as_ref()).unwrap();
    assert_eq!(conv_seeds.len(), 2, "per-layer conv seeds");
    assert!(conv_seeds.iter().all(|s| s.is_some()), "conv layers must be seedable");
    graph.buffers.seed_conv_rings(&conv_seeds).unwrap();

    let mut logits_graph = Vec::new();
    for &t in &replay_toks {
        DecodeGraphModel::forward_replay(&model_graph, &mut graph, t).unwrap();
        dev.synchronize();
        logits_graph.extend(graph.read_logits_f32().unwrap());
    }

    // ---- Eager reference: identical token sequence, all eager ----
    let mut sess_e = CausalLm::new_session(&model_eager);
    for &t in &prompt {
        let _ = run_step(&dev, &model_eager, &mut sess_e, t);
    }
    let _ = run_step(&dev, &model_eager, &mut sess_e, capture_tok);
    let mut logits_eager = Vec::new();
    for &t in &replay_toks {
        logits_eager.extend(run_step(&dev, &model_eager, &mut sess_e, t));
    }

    assert_eq!(logits_graph.len(), logits_eager.len());
    let max_diff = logits_graph
        .iter()
        .zip(logits_eager.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "graph replay diverged from eager on conv model: max_diff={max_diff}"
    );
}
