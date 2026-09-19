//! A5 Phase 2 (WI-X2-PREFILL-ARENA): `eager_kv_seed_sources` + 
//! `seed_kv_arena_from_eager` move the eager device KV arenas into the graph
//! arenas bit-exactly, and fail closed when arenas are missing.
//!
//! Gated on `gpu_test_enabled()` (GRIM_GPU_TEST=1) + ROCm device. Uses a tiny
//! dense-only random LFM2 (no model files) driven through `CausalLm::forward`
//! for a 6-token prefill.

use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::device::util::gpu_test_enabled;
use grim_core::model::CausalLm;
use grim_models_transformer::lfm2::{Lfm2, Lfm2Block, Lfm2Config, Lfm2LayerCache};
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{BackendStorage, CoreTensorOps, DType, Device, Shape, Tensor};

fn rocm_tensor(dev: &RocmDevice, ordinal: usize, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        Device::Rocm(ordinal),
    )
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

fn test_linear(dev: &RocmDevice, ordinal: usize, out_dim: usize, in_dim: usize, seed: u64) -> Linear {
    let w = rocm_tensor(dev, ordinal, rand_vec(out_dim * in_dim, seed), Shape::new(vec![out_dim, in_dim]));
    Linear { weight: w.clone(), bias: None, w_t: w, quant_format: None }
}

fn test_norm(dev: &RocmDevice, ordinal: usize, dim: usize) -> RmsNorm {
    RmsNorm { weight: rocm_tensor(dev, ordinal, vec![1.0f32; dim], Shape::new(vec![dim])), eps: 1e-5 }
}

fn attention_block(
    dev: &RocmDevice,
    ordinal: usize,
    hidden: usize,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    inter: usize,
) -> Lfm2Block {
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
        n_expert_used: 1,
        charon_cache: grim_models_transformer::shared_moe::CharonCache::new(),
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: n_q / hd,
        num_kv_heads: n_kv / hd,
        head_dim: hd,
        rope_theta: 10000.0,
        eps: 1e-5,
    }
}

fn tiny_dense_lfm2(dev: &RocmDevice, ordinal: usize, n_layers: usize) -> Lfm2 {
    let hidden = 32usize;
    let hd = 8usize;
    let nh = 2usize;
    let nkv = 1usize;
    let inter = 64usize;
    let vocab = 32usize;
    let layers = (0..n_layers)
        .map(|_| attention_block(dev, ordinal, hidden, nh * hd, nkv * hd, hd, inter))
        .collect();
    let tok_w = rocm_tensor(dev, ordinal, rand_vec(vocab * hidden, 99), Shape::new(vec![vocab, hidden]));
    Lfm2 {
        cfg: Lfm2Config {
            vocab_size: vocab,
            hidden_size: hidden,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            num_layers: n_layers,
            intermediate_size: inter,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            n_shortconv_l_cache: 0,
            is_recr: vec![false; n_layers],
            n_layer_dense_lead: 0,
            n_expert: 0,
            n_expert_used: 0,
            n_ff_exp: 0,
            n_embd_out: 0,
            mxfp4_qkv_attention: false,
        },
        device: Device::Rocm(ordinal),
        tok_embeddings: Embedding { weight: tok_w.clone() },
        layers,
        norm: test_norm(dev, ordinal, hidden),
        output: Linear { weight: tok_w.clone(), bias: None, w_t: tok_w, quant_format: None },
        dense_2_out: None,
        dense_2_out_bias: None,
    }
}

fn gpu_dev() -> Option<std::sync::Arc<RocmDevice>> {
    if !gpu_test_enabled() {
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        eprintln!("skip: set GRIM_GPU_TEST=1 for GPU seed test");
        return None;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return None;
    }
    Some(RocmDevice::shared(0))
}

#[test]
fn kv_seed_moves_eager_arenas_bit_exact() {
    let Some(dev) = gpu_dev() else { return };
    let model = tiny_dense_lfm2(&dev, 0, 2);
    let mut session = model.new_session();

    // 6-token prefill through the real forward (populates device KV arenas).
    let prompt: Vec<u32> = vec![1, 5, 9, 13, 17, 21];
    let input = rocm_tensor(
        &dev,
        0,
        prompt.iter().map(|&t| t as f32).collect(),
        Shape::new(vec![prompt.len()]),
    );
    let pos = rocm_tensor(&dev, 0, vec![0.0f32; prompt.len()], Shape::new(vec![prompt.len()]));
    model.forward(session.as_mut(), &input, &pos, &[]).expect("prefill");

    let caches = session
        .model_state()
        .and_then(|s| s.downcast_ref::<Vec<Option<Lfm2LayerCache>>>())
        .expect("session caches");
    assert_eq!(caches.len(), 2);

    // Export + fail-closed cases (no GPU needed for the error paths, but the
    // happy path below needs arenas present).
    let srcs = model.eager_kv_seed_sources(caches, prompt.len() as u32).expect("export");
    assert_eq!(srcs.len(), 2);
    assert!(srcs.iter().all(|s| s.is_some()), "dense layers must export");
    assert_eq!(srcs[0].as_ref().unwrap().kv_stride, 8); // nkv(1) * hd(8)
    assert_eq!(srcs[0].as_ref().unwrap().prefill_len, 6);

    // Fresh session (all-None caches) with nonzero rows must fail closed.
    let fresh: Vec<Option<Lfm2LayerCache>> = vec![None, None];
    assert!(model.eager_kv_seed_sources(&fresh, 6).is_err());
    // Zero rows is a no-op export.
    let empty = model.eager_kv_seed_sources(&fresh, 0).expect("empty export");
    assert!(empty.iter().all(|s| s.is_none()));
    // Length mismatch fails closed.
    assert!(model.eager_kv_seed_sources(&fresh[..1], 6).is_err());

    // Seed a fresh graph pool and compare rows bit-exactly vs eager arenas.
    let mut graph = model.get_or_create_decode_graph(64, 1).expect("graph alloc");
    let ordinal = 0usize;
    let rdev = RocmDevice::shared(ordinal);
    graph.buffers.seed_kv_arena_from_eager(&rdev, &srcs).expect("seed");
    assert_eq!(graph.buffers.current_pos, 6);

    for (layer, cache) in caches.iter().enumerate() {
        let eager_k = match cache {
            Some(Lfm2LayerCache::Attention { k_dev, .. }) => k_dev.as_deref().expect("k_dev"),
            _ => panic!("layer {layer}: expected attention cache"),
        };
        let eager_v = match cache {
            Some(Lfm2LayerCache::Attention { v_dev, .. }) => v_dev.as_deref().expect("v_dev"),
            _ => panic!("layer {layer}: expected attention cache"),
        };
        let stride = 8usize;
        let n = 6 * stride;
        let ek: Vec<f32> = eager_k.storage().to_cpu_vec_f32().expect("eager k read");
        let ev: Vec<f32> = eager_v.storage().to_cpu_vec_f32().expect("eager v read");
        let sk: Vec<f32> = graph.buffers.k_arena[layer].to_cpu_vec_f32().expect("seeded k read");
        let sv: Vec<f32> = graph.buffers.v_arena[layer].to_cpu_vec_f32().expect("seeded v read");
        assert_eq!(&ek[..n], &sk[..n], "layer {layer}: seeded K != eager K");
        assert_eq!(&ev[..n], &sv[..n], "layer {layer}: seeded V != eager V");
        // Seeded rows must be non-trivial (guards vacuous all-zero pass).
        assert!(ek[..n].iter().any(|&x| x != 0.0), "layer {layer}: eager K all zero?");
    }
}
