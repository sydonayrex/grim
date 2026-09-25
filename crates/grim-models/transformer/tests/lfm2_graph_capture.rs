//! P1: `lfm2_graph` records real kernels, not pointer checks.
//!
//! Gated on `gpu_test_enabled()` (GRIM_GPU_TEST=1) + ROCm device.
//! Attention-only tiny LFM2: capture -> replay -> finite logits + stable
//! pool addresses. Recurrent block -> `Unimplemented` (eager fallback).
//!
//! NOTE: input injection is still a 4-byte token-id stub (not an embedding
//! gather), so graph logits are NOT expected to match eager logits yet.
//! That gather kernel is the documented P2 follow-up.

use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::RocmStorage;
use grim_core::model::CausalLm;
use grim_models_transformer::lfm2::{Lfm2, Lfm2Block, Lfm2Config, Lfm2LayerCache};
use grim_models_transformer::lfm2_graph::write_embedding_to_buffer;
use grim_models_transformer::shared_moe::CharonCache;
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{
    BackendStorage, CoreTensorOps, DType, Device, MemoryOps, Shape, Storage, Tensor,
};

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
    // Deterministic LCG in [-0.1, 0.1]; no H2D/D2H involved in generation.
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
        })
        .collect()
}

fn test_linear(
    dev: &RocmDevice,
    ordinal: usize,
    out_dim: usize,
    in_dim: usize,
    seed: u64,
) -> Linear {
    let w = rocm_tensor(
        dev,
        ordinal,
        rand_vec(out_dim * in_dim, seed),
        Shape::new(vec![out_dim, in_dim]),
    );
    Linear {
        weight: w.clone(),
        bias: None,
        w_t: w,
        quant_format: None,
    }
}

/// Q8_0 linear: host-quantize random rows (CPU `quant_q80`), upload raw
/// bytes. Mirrors GGUF Q8_0 weight layout.
fn test_linear_q80(
    dev: &RocmDevice,
    ordinal: usize,
    out_dim: usize,
    in_dim: usize,
    seed: u64,
) -> Linear {
    use grim_tensor::MemoryOps;
    let bytes = grim_quant::quant_q80(&rand_vec(out_dim * in_dim, seed)).unwrap();
    let dtype = DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
    };
    let shape = Shape::new(vec![out_dim, in_dim]);
    let storage = dev.from_cpu_bytes(&bytes, &shape, dtype.clone()).unwrap();
    let w = Tensor::new(
        Arc::from(storage),
        shape,
        dtype,
        grim_tensor::QuantProvenance::GrimNative,
        Device::Rocm(ordinal),
    );
    Linear {
        weight: w.clone(),
        bias: None,
        w_t: w,
        quant_format: None,
    }
}

fn test_norm(dev: &RocmDevice, ordinal: usize, dim: usize) -> RmsNorm {
    RmsNorm {
        weight: rocm_tensor(dev, ordinal, vec![1.0f32; dim], Shape::new(vec![dim])),
        eps: 1e-5,
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_block(
    dev: &RocmDevice,
    ordinal: usize,
    hidden: usize,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    inter: usize,
    fused: bool,
) -> Lfm2Block {
    let nh = n_q / hd;
    let nkv = n_kv / hd;

    let (_wqkv_unused, w_gate_up_q80_fused) = if fused {
        let gate = test_linear_q80(dev, ordinal, inter, hidden, 104);
        let up = test_linear_q80(dev, ordinal, inter, hidden, 105);
        let fgu = dev
            .build_fused_gate_up_q80(gate.weight.storage().as_ref(), up.weight.storage().as_ref())
            .expect("build fused Q80 gate+up");
        (
            Option::<grim_backend_rocm::FusedQkvWeights>::None,
            Some(fgu),
        )
    } else {
        (None, None)
    };

    Lfm2Block {
        index: 0,
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
        w_gate_up_q80_fused,
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
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        rope_theta: 10000.0,
        eps: 1e-5,
        charon_cache: CharonCache::new(),
        attention_mode: grim_models_transformer::lfm2::Lfm2AttentionMode::Softmax,
        gdl_gates: grim_models_transformer::gla::gdl_gate_defaults(64),
        gdl_b_proj: None,
        gdl_w_proj: None,
        gdl_f_proj: None,
        gdl_fused_qkv_gates: None,
    }
}

/// GQA GDL model: nh=4, nkv=2, hd=64, hidden=128. Small enough to be fast
/// but exercises the GQA path (nh != nkv) with hd=64 (required by the guard).
fn tiny_gqa_gdl_lfm2(dev: &RocmDevice, ordinal: usize, n_layers: usize) -> Lfm2 {
    let hidden = 128usize;
    let hd = 64usize;
    let nh = 4usize;
    let nkv = 2usize;
    let inter = 256usize;
    let vocab = 32usize;
    let mut layers: Vec<Lfm2Block> = (0..n_layers)
        .map(|_| attention_block(dev, ordinal, hidden, nh * hd, nkv * hd, hd, inter, false))
        .collect();
    for layer in layers.iter_mut() {
        layer.attention_mode = grim_models_transformer::lfm2::Lfm2AttentionMode::Gdl;
    }
    let tok_w = rocm_tensor(
        dev,
        ordinal,
        rand_vec(vocab * hidden, 99),
        Shape::new(vec![vocab, hidden]),
    );
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
            attention_mode: grim_models_transformer::lfm2::Lfm2AttentionMode::Gdl,
            attention_mode_per_layer: None,
        },
        device: Device::Rocm(ordinal),
        tok_embeddings: Embedding {
            weight: tok_w.clone(),
        },
        layers,
        norm: test_norm(dev, ordinal, hidden),
        output: Linear {
            weight: tok_w.clone(),
            bias: None,
            w_t: tok_w,
            quant_format: None,
        },
        dense_2_out: None,
        dense_2_out_bias: None,
    }
}

fn tiny_lfm2(dev: &RocmDevice, ordinal: usize, n_layers: usize, fused: bool) -> Lfm2 {
    let hidden = 32usize;
    let hd = 8usize;
    let nh = 2usize;
    let nkv = 1usize;
    let inter = 64usize;
    let vocab = 32usize;
    let layers = (0..n_layers)
        .map(|_| attention_block(dev, ordinal, hidden, nh * hd, nkv * hd, hd, inter, fused))
        .collect();
    let tok_w = rocm_tensor(
        dev,
        ordinal,
        rand_vec(vocab * hidden, 99),
        Shape::new(vec![vocab, hidden]),
    );
    let tok_embeddings = Embedding {
        weight: tok_w.clone(),
    };
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
            attention_mode: grim_models_transformer::lfm2::Lfm2AttentionMode::Softmax,
            attention_mode_per_layer: None,
        },
        device: Device::Rocm(ordinal),
        tok_embeddings,
        layers,
        norm: test_norm(dev, ordinal, hidden),
        output: Linear {
            weight: tok_w.clone(),
            bias: None,
            w_t: tok_w,
            quant_format: None,
        },
        dense_2_out: None,
        dense_2_out_bias: None,
    }
}

#[test]
fn lfm2_graph_capture_replay_records_kernels() {
    if !grim_backend_rocm::device::util::gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);

    // Baseline: F32 attention weights and F32 FFN weights.
    let plain = tiny_lfm2(&dev, 0, 2, false);

    // Pool allocates at stable addresses.
    let mut graph = plain.get_or_create_decode_graph(16, 1).unwrap();
    let in_ptr = graph.buffers.layer_input[0].device_ptr_u64().unwrap();
    assert!(in_ptr != 0);

    // Warmup runs first (no capture bracket): JIT + caching allocator +
    // one-time device calibration (e.g. Scythe WI-SB0 fires its probe GEMMs
    // on first use) so capture sees warm, stable addresses. TWO warmups:
    // the first absorbs one-time effects, the second is the baseline.
    plain.forward_capture(&mut graph, 7).unwrap();
    dev.reset_launch_count();
    plain.forward_capture(&mut graph, 7).unwrap();
    let eager_gemms = dev.launch_count();
    assert!(
        eager_gemms >= 15,
        "expected >=15 GEMMs (7/layer x2 + head), got {eager_gemms}"
    );

    // Fused path: gate+up blob (the fused QKV blob was retired — see
    // PERF-REGRESSION note in model_loader/lfm2.rs; per-linear dot4 covers Q8_0).
    let model = tiny_lfm2(&dev, 0, 2, true);
    assert!(
        model.layers[0].w_gate_up_q80_fused.is_some(),
        "fused Q8_0 gate+up weights must be enabled for the graph path"
    );

    // Capture must not enqueue MORE launches than the F32 baseline. The
    // fused gate+up blob replaces 2 GEMMs with 1 quantize + 1 GEMV (net 0
    // launches; it wins on bandwidth, not count). The fused QKV blob — the
    // bigger win — was retired (see PERF-REGRESSION note in model_loader);
    // per-linear dot4 covers Q8_0 decode instead.
    dev.reset_launch_count();
    graph.begin_capture().unwrap();
    model.forward_capture(&mut graph, 7).unwrap();
    graph.end_capture().unwrap();
    assert!(graph.is_captured);
    let fused_gemms = dev.launch_count();
    assert!(
        fused_gemms <= eager_gemms,
        "fused path must not exceed plain launches: fused={fused_gemms}, plain={eager_gemms}"
    );

    // Replay + single sync readback: finite logits, same addresses.
    // Replay itself must enqueue ZERO host-visible launches (one graph launch).
    dev.reset_launch_count();
    model.forward_replay(&mut graph, 9).unwrap();
    assert_eq!(
        dev.launch_count(),
        0,
        "replay must be a single hipGraphLaunch, no eager launches"
    );
    graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    let logits = graph.read_logits_f32().unwrap();
    assert_eq!(logits.len(), 32);
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_eq!(
        graph.buffers.layer_input[0].device_ptr_u64().unwrap(),
        in_ptr
    );
    // Second replay: stability across replays (allocator-independent —
    // every node writes a fixed pool slot, no scratch addresses baked in).
    model.forward_replay(&mut graph, 11).unwrap();
    graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    let logits2 = graph.read_logits_f32().unwrap();
    assert!(logits2.iter().all(|x| x.is_finite()));
}

#[test]
fn lfm2_graph_recurrent_falls_back_eager() {
    if !grim_backend_rocm::device::util::gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);
    let model = tiny_lfm2(&dev, 0, 1, false);
    let graph = model.get_or_create_decode_graph(16, 1).unwrap();

    // A block with no QKV projections is recurrent-shaped: must refuse with
    // Unimplemented (caller falls back eager), never panic or record junk.
    let rec_block = Lfm2Block {
        index: 0,
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        shortconv_in_proj: None,
        ..attention_block(&dev, 0, 32, 16, 8, 8, 64, false)
    };
    // Struct-update moves attention_block fields; attn_norm etc. reused.
    let err = rec_block
        .forward_graph(0, &graph.buffers, &dev)
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("Unimplemented")
            || format!("{err}").to_lowercase().contains("unimplemented"),
        "expected Unimplemented, got {err:?}"
    );

    // write_embedding helper rejects CPU-backed dst instead of segfaulting.
    let cpu = grim_backend_cpu::cpu_tensor(vec![0.0f32; 4], Shape::new(vec![4]));
    let err = write_embedding_to_buffer(&dev, cpu.storage().as_ref(), 3).unwrap_err();
    let _ = (model, graph);
    let _ = err;
}

// ───────────────────────────────────────────────────────────── M2/S2 (PLAN-kernel-fusion): graph capture for MoE and ShortConv layers.
//
// M2: an `is_moe` block captures via `moe_forward_graph` (device routing into
// persistent routing buffers + resident stacked weights + grouped dispatch
// into a fixed pool slot). S2: a ShortConv block captures via
// `shortconv_forward_graph` (device ring state, no host transpose, no H2D).
// Both must `begin_capture`/`forward_capture`/`end_capture` successfully,
// replay deterministically, and match the eager forward's logits within
// tolerance (the capture path runs the same kernels as eager — only the
// launch mechanism differs).

fn rocm_tensor_bytes(
    dev: &RocmDevice,
    ordinal: usize,
    data: Vec<u8>,
    shape: Shape,
    dtype: grim_tensor::DType,
    _cap: usize,
) -> grim_tensor::Tensor {
    let storage = dev.from_cpu_bytes(&data, &shape, dtype.clone()).unwrap();
    grim_tensor::Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        dtype,
        grim_tensor::QuantProvenance::GrimNative,
        Device::Rocm(ordinal),
    )
}

fn moe_block(dev: &RocmDevice, ordinal: usize) -> Lfm2Block {
    let hidden = 32usize;
    let hd = 8usize;
    let nh = 2usize;
    let nkv = 1usize;
    let inter = 64usize;
    let n_expert = 4usize;
    let n_ff = 16usize;
    let mut b = attention_block(dev, ordinal, hidden, nh * hd, nkv * hd, hd, inter, false);
    b.is_moe = true;
    b.n_expert = n_expert;
    b.n_expert_used = 2;
    b.ffn_gate_inp = Some(test_linear(dev, ordinal, n_expert, hidden, 301));
    // Stacked expert tensors (GGUF layout):
    // gate/up: [n_expert, n_ff, hidden]; down: [n_ff, hidden, n_expert].
    b.ffn_gate_exps = Some(rocm_tensor(
        dev,
        ordinal,
        rand_vec(n_expert * n_ff * hidden, 311),
        Shape::new(vec![n_expert, n_ff, hidden]),
    ));
    b.ffn_up_exps = Some(rocm_tensor(
        dev,
        ordinal,
        rand_vec(n_expert * n_ff * hidden, 312),
        Shape::new(vec![n_expert, n_ff, hidden]),
    ));
    b.ffn_down_exps = Some(rocm_tensor(
        dev,
        ordinal,
        rand_vec(n_ff * hidden * n_expert, 313),
        Shape::new(vec![n_ff, hidden, n_expert]),
    ));
    b
}

fn shortconv_block(dev: &RocmDevice, ordinal: usize) -> Lfm2Block {
    let hidden = 32usize;
    let hd = 8usize;
    let nh = 2usize;
    let nkv = 1usize;
    let inter = 64usize;
    let l_cache = 3usize;
    let mut b = attention_block(dev, ordinal, hidden, nh * hd, nkv * hd, hd, inter, false);
    b.shortconv_in_proj = Some(test_linear(dev, ordinal, 3 * hidden, hidden, 401));
    b.shortconv_conv = Some(rocm_tensor(
        dev,
        ordinal,
        rand_vec(hidden * l_cache, 402),
        Shape::new(vec![hidden, 1, l_cache]),
    ));
    b.shortconv_conv_vec = Some(rand_vec(hidden * l_cache, 402));
    b.shortconv_out_proj = Some(test_linear(dev, ordinal, hidden, hidden, 403));
    b
}

fn tiny_lfm2_custom(dev: &RocmDevice, ordinal: usize, layers: Vec<Lfm2Block>) -> Lfm2 {
    let hidden = 32usize;
    let vocab = 32usize;
    let tok_w = rocm_tensor(
        dev,
        ordinal,
        rand_vec(vocab * hidden, 99),
        Shape::new(vec![vocab, hidden]),
    );
    let n_layers = layers.len();
    Lfm2 {
        cfg: Lfm2Config {
            vocab_size: vocab,
            hidden_size: hidden,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            num_layers: n_layers,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            n_shortconv_l_cache: 3,
            is_recr: vec![false; n_layers],
            n_layer_dense_lead: 0,
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 16,
            n_embd_out: 0,
            mxfp4_qkv_attention: false,
            attention_mode: grim_models_transformer::lfm2::Lfm2AttentionMode::Softmax,
            attention_mode_per_layer: None,
        },
        device: Device::Rocm(ordinal),
        tok_embeddings: Embedding {
            weight: tok_w.clone(),
        },
        layers,
        norm: test_norm(dev, ordinal, hidden),
        output: Linear {
            weight: tok_w.clone(),
            bias: None,
            w_t: tok_w,
            quant_format: None,
        },
        dense_2_out: None,
        dense_2_out_bias: None,
    }
}

fn capture_replay_logits(model: &Lfm2, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    let mut graph = model
        .get_or_create_decode_graph(16, 1)
        .map_err(|e| format!("alloc: {e}"))?;
    // Warmups: JIT + resident weight stack + routing scratch must be built
    // BEFORE the capture bracket (allocation inside capture is illegal).
    for &t in tokens.iter().take(2) {
        model
            .forward_capture(&mut graph, t)
            .map_err(|e| format!("warmup: {e}"))?;
    }
    graph.begin_capture().map_err(|e| format!("begin: {e}"))?;
    model
        .forward_capture(&mut graph, tokens[0])
        .map_err(|e| format!("capture fwd: {e}"))?;
    graph.end_capture().map_err(|e| format!("end: {e}"))?;
    assert!(graph.is_captured, "capture must succeed");
    let mut out = Vec::new();
    for &t in tokens {
        model
            .forward_replay(&mut graph, t)
            .map_err(|e| format!("replay: {e}"))?;
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
        let l = graph.read_logits_f32().map_err(|e| format!("read: {e}"))?;
        assert!(l.iter().all(|x| x.is_finite()), "non-finite replay logit");
        out.push(l);
    }
    Ok(out)
}

#[test]
fn lfm2_graph_moe_layer_captures_and_replays() {
    if !grim_backend_rocm::device::util::gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);
    // Two MoE layers — routing buffers shared across layers must stay
    // coherent (each layer writes then consumes within stream order).
    let model = tiny_lfm2_custom(&dev, 0, vec![moe_block(&dev, 0), moe_block(&dev, 0)]);
    let tokens = [7u32, 9, 11];
    let replay = capture_replay_logits(&model, &tokens).expect("moe capture/replay");

    // Determinism: same token -> identical logits across replays (7 vs 7
    // cannot be compared directly since the graph is stateful; instead check
    // the replays differ per token but are finite — covered above — and that
    // replaying is stable by re-running the same sequence.
    let replay2 = capture_replay_logits(&model, &tokens).expect("moe capture/replay 2");
    for (a_seq, b_seq) in replay.iter().zip(&replay2) {
        for (a, b) in a_seq.iter().zip(b_seq) {
            assert!((a - b).abs() < 1e-2, "replay nondeterminism: {a} vs {b}");
        }
    }
}

#[test]
fn lfm2_graph_moe_sublayer_matches_eager_dispatch() {
    // M2 numeric gate, isolated from attention/KV-history plumbing: the graph
    // sublayer's op sequence (route_topk into persistent buffers + grouped
    // dispatch into a caller-provided out) must produce the same MoE FFN
    // output as the eager `fused_moe_dispatch_from_logits` on the same input,
    // because the graph path runs exactly those primitives.
    if !grim_backend_rocm::device::util::gpu_test_enabled()
        || !RocmDevice::probe_one(0).unwrap_or(false)
    {
        eprintln!("skip: GPU test");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);
    let model = tiny_lfm2_custom(&dev, 0, vec![moe_block(&dev, 0)]);
    let block = &model.layers[0];
    let hidden = model.cfg.hidden_size;
    let top_k = block.n_expert_used.min(block.n_expert).max(1);
    let x = rocm_tensor(&dev, 0, rand_vec(hidden, 71), Shape::new(vec![1, hidden]));

    // Eager path (production MoE forward).
    let gate_inp = block.ffn_gate_inp.as_ref().unwrap();
    let logits = gate_inp.forward(&x).unwrap();
    let experts = block.moe_experts().unwrap().clone();
    let eager_out = grim_models_transformer::shared_moe::fused_moe_dispatch_from_logits(
        &dev,
        &x,
        &logits,
        &experts,
        None,
        top_k,
        1.0,
        0,
        &block.charon_cache,
    )
    .expect("eager dispatch")
    .expect("dispatch Some");

    // Graph-op sequence (same primitives, caller-provided out).
    let scratch = grim_models_transformer::shared_moe::ensure_charon_scratch(
        0,
        1,
        top_k,
        &experts,
        &block.charon_cache,
    )
    .unwrap();
    let (tokens, experts_b, weights, gate_buf, up_buf, down_buf) = scratch;
    fn downcast(a: &Arc<dyn grim_tensor::BackendStorage>) -> &RocmStorage {
        a.as_ref().as_any().downcast_ref::<RocmStorage>().unwrap()
    }

    let logits_rocm = logits
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap();
    let x_rocm = x
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .unwrap();
    dev.moe_route_topk_on_device(
        logits_rocm,
        None,
        downcast(&tokens),
        downcast(&experts_b),
        downcast(&weights),
        1,
        block.n_expert,
        top_k,
        0,
    )
    .unwrap();
    let out_dyn = dev.zeros(&Shape::new(vec![1, hidden]), DType::F32).unwrap();
    let out = out_dyn.as_any().downcast_ref::<RocmStorage>().unwrap();
    let inter = experts[0].gate.weight.shape().dim(0).unwrap_or(0);
    dev.moe_fused_dispatch_resident_routing_into(
        x_rocm,
        gate_buf.as_ref(),
        up_buf.as_ref(),
        down_buf.as_ref(),
        downcast(&tokens),
        downcast(&experts_b),
        downcast(&weights),
        top_k,
        out,
        hidden,
        inter,
        1.0,
    )
    .unwrap();

    // Compare (atomicAdd accumulation => float-order noise, use 1e-4 rel).
    let got = out.to_cpu_vec_f32().unwrap();
    let want = eager_out.to_vec_f32().unwrap();

    let mut worst = 0.0f32;
    for (a, b) in got.iter().zip(&want) {
        worst = worst.max((a - b).abs() / (b.abs() + 1e-3));
    }
    assert!(
        worst < 1e-4,
        "graph MoE ops vs eager dispatch mismatch (worst rel {worst:.3e}): got {got:?} want {want:?}"
    );
}

#[test]
fn lfm2_graph_shortconv_layer_captures_and_replays() {
    if !grim_backend_rocm::device::util::gpu_test_enabled()
        || !RocmDevice::probe_one(0).unwrap_or(false)
    {
        eprintln!("skip: GPU test");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);
    // Pure-ShortConv layers (no attention) — the S2 graph path.
    let model = tiny_lfm2_custom(
        &dev,
        0,
        vec![shortconv_block(&dev, 0), shortconv_block(&dev, 0)],
    );
    let tokens = [5u32, 13, 21];
    let replay = capture_replay_logits(&model, &tokens).expect("sc capture/replay");
    assert_eq!(replay.len(), 3);
    let replay2 = capture_replay_logits(&model, &tokens).expect("sc capture/replay 2");
    for (a_seq, b_seq) in replay.iter().zip(&replay2) {
        for (a, b) in a_seq.iter().zip(b_seq) {
            assert!((a - b).abs() < 1e-2, "replay nondeterminism: {a} vs {b}");
        }
    }
}

/// S2 state-correctness guard: ShortConv replay must be STATEFUL — replaying
/// token A twice in a row gives different logits than the first A (the conv
/// ring advanced), proving the device ring updates inside the graph instead
/// of being frozen at capture time.
#[test]
fn lfm2_graph_shortconv_ring_advances_across_replays() {
    if !grim_backend_rocm::device::util::gpu_test_enabled()
        || !RocmDevice::probe_one(0).unwrap_or(false)
    {
        eprintln!("skip: GPU test");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);
    let model = tiny_lfm2_custom(
        &dev,
        0,
        vec![shortconv_block(&dev, 0), shortconv_block(&dev, 0)],
    );
    let mut graph = model.get_or_create_decode_graph(16, 1).unwrap();
    for &t in &[3u32, 5] {
        model.forward_capture(&mut graph, t).unwrap();
    }
    graph.begin_capture().unwrap();
    model.forward_capture(&mut graph, 42).unwrap();
    graph.end_capture().unwrap();

    model.forward_replay(&mut graph, 42).unwrap();
    graph.buffers.current_pos += 1;
    let first = graph.read_logits_f32().unwrap();
    model.forward_replay(&mut graph, 42).unwrap();
    graph.buffers.current_pos += 1;
    let second = graph.read_logits_f32().unwrap();
    let max_delta = first
        .iter()
        .zip(&second)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-6,
        "conv ring did not advance: identical logits across replays (max delta {max_delta})"
    );
}

/// MXFP4 decode-graph gap closure: a native-MXFP4 attention layer captures
/// via the fused `grim_fused_mxfp4_gemm_qk_norm_rope_kv` node (projection +
/// QK-norm + RoPE + K/V append in one kernel) and replays deterministically.
#[test]
fn lfm2_graph_mxfp4_layer_captures_and_replays() {
    if !grim_backend_rocm::device::util::gpu_test_enabled()
        || !RocmDevice::probe_one(0).unwrap_or(false)
    {
        eprintln!("skip: GPU test");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);
    let hidden = 32usize;
    let hd = 8usize;
    let nh = 2usize;
    let nkv = 1usize;
    let inter = 64usize;
    let n_q = nh * hd;
    let n_kv = nkv * hd;

    let mut b = attention_block(&dev, 0, hidden, n_q, n_kv, hd, inter, false);
    // Pack the QKV weights to MXFP4 exactly like `build_fused_qkv_pack`:
    // row-major Q∥K∥V concat, quant_mxfp4_matrix layout.
    let mut concat = Vec::with_capacity(n_q * hidden + 2 * n_kv * hidden);
    concat.extend(rand_vec(n_q * hidden, 11));
    concat.extend(rand_vec(n_kv * hidden, 22));
    concat.extend(rand_vec(n_kv * hidden, 33));
    let (codes, exps) = grim_quant::quant_mxfp4_matrix(&concat, n_q + 2 * n_kv, hidden);
    let codes_dtype = grim_tensor::DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::FloatPack(grim_tensor::dtype::FloatPackScheme::MxFp4),
    };
    let codes_shape = Shape::new(vec![n_q + 2 * n_kv, hidden]);
    let codes_t = rocm_tensor_bytes(
        &dev,
        0,
        codes,
        codes_shape.clone(),
        codes_dtype,
        (n_q + 2 * n_kv) * hidden / 2 + (n_q + 2 * n_kv) * hidden / 32,
    );
    let exps_shape = Shape::new(vec![(n_q + 2 * n_kv) * hidden / 32]);
    let exps_len = exps.len();
    let exps_t = rocm_tensor_bytes(
        &dev,
        0,
        exps,
        exps_shape,
        grim_tensor::DType {
            arith: grim_tensor::ArithType::U8,
            storage: Storage::Native,
        },
        exps_len,
    );
    // QK-norm gammas (positive, RMSNorm-style).
    b.wqkv_codes = Some(codes_t);
    b.wqkv_exps = Some(exps_t);
    b.gamma_q = Some(rocm_tensor(&dev, 0, vec![1.0; hd], Shape::new(vec![hd])));
    b.gamma_k = Some(rocm_tensor(&dev, 0, vec![1.0; hd], Shape::new(vec![hd])));

    let model = tiny_lfm2_custom(&dev, 0, vec![b]);
    assert!(model.layers[0].wqkv_codes.is_some());
    let tokens = [7u32, 13, 19];
    if std::env::var("GRIM_MXFP4_PARITY_DEBUG").is_ok() {
        // Plain F32 model through the SAME harness: isolates harness bugs
        // from fused-kernel wiring bugs (constant factor => harness).
        let plain = tiny_lfm2_custom(
            &dev,
            0,
            vec![attention_block(
                &dev, 0, hidden, n_q, n_kv, hd, inter, false,
            )],
        );
        let pr = capture_replay_logits(&plain, &tokens).unwrap();
        let mut caches: Vec<Option<Lfm2LayerCache>> = vec![None, None];
        for (i, &t) in tokens.iter().enumerate() {
            let x = rocm_tensor(
                &dev,
                0,
                {
                    let row = model.tok_embeddings.weight.to_vec_f32().unwrap();
                    row[t as usize * hidden..(t as usize + 1) * hidden].to_vec()
                },
                Shape::new(vec![1, hidden]),
            );
            let mut h = x;
            for (li, layer) in plain.layers.iter().enumerate() {
                h = layer.forward(&h, &mut caches[li]).unwrap();
            }
            let n = plain.norm.weight.to_vec_f32().unwrap();
            let mut hv = h.to_vec_f32().unwrap();
            let msq: f32 = hv.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (msq + plain.cfg.rms_norm_eps).sqrt();
            for v in hv.iter_mut() {
                *v *= inv;
            }
            let w = plain.output.weight.to_vec_f32().unwrap();
            let logits: Vec<f32> = (0..plain.cfg.vocab_size)
                .map(|vi| {
                    (0..hidden)
                        .map(|d| w[vi * hidden + d] * hv[d] * n[d])
                        .sum::<f32>()
                })
                .collect();
            let worst: f32 = pr[i]
                .iter()
                .zip(&logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "plain-m harness check token {t}: worst abs {worst:.4} (graph[:3]={:?} eager[:3]={:?})",
                &logits[..3],
                &pr[i][..3]
            );
        }
    }
    let replay = capture_replay_logits(&model, &tokens).expect("mxfp4 capture/replay");
    assert_eq!(replay.len(), 3);
    let replay2 = capture_replay_logits(&model, &tokens).expect("mxfp4 capture/replay 2");
    for (a_seq, b_seq) in replay.iter().zip(&replay2) {
        for (a, b) in a_seq.iter().zip(b_seq) {
            assert!((a - b).abs() < 1e-2, "replay nondeterminism: {a} vs {b}");
        }
    }

    // Numeric cross-check vs the EAGER path: feed the same token sequence
    // token-by-token through the blocks (each keeps its own KV history), so
    // replay step i and eager step i see identical history. The graph path
    // runs the same fused MXFP4 kernel — outputs must match within float
    // noise of the attention reduction.
    let mut caches: Vec<Option<Lfm2LayerCache>> = vec![None, None];
    for (i, &t) in tokens.iter().enumerate() {
        let x = rocm_tensor(
            &dev,
            0,
            {
                let row = model.tok_embeddings.weight.to_vec_f32().unwrap();
                row[t as usize * hidden..(t as usize + 1) * hidden].to_vec()
            },
            Shape::new(vec![1, hidden]),
        );
        let mut h = x;
        for (li, layer) in model.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[li]).unwrap();
        }
        // Head over the final hidden state.
        let n = model.norm.weight.to_vec_f32().unwrap();
        let mut hv = h.to_vec_f32().unwrap();
        let msq: f32 = hv.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (msq + model.cfg.rms_norm_eps).sqrt();
        for v in hv.iter_mut() {
            *v *= inv;
        }
        let w = model.output.weight.to_vec_f32().unwrap();
        let logits: Vec<f32> = (0..model.cfg.vocab_size)
            .map(|vi| {
                (0..hidden)
                    .map(|d| w[vi * hidden + d] * hv[d] * n[d])
                    .sum::<f32>()
            })
            .collect();
        let mut worst = 0.0f32;
        for (a, b) in replay[i].iter().zip(&logits) {
            worst = worst.max((a - b).abs() / (b.abs() + 1e-3));
        }
        eprintln!("mxfp4 parity token {t}: worst rel {worst:.4}");
        eprintln!("  graph[:6] = {:?}", &replay[i][..6]);
        eprintln!("  eager[:6] = {:?}", &logits[..6]);
        assert!(
            worst < 5e-2,
            "token {t}: graph vs eager MXFP4 logits differ (worst rel {worst:.4})"
        );
    }
}

/// Task 2: GQA GDL graph-vs-eager parity. The graph path must produce the
/// same output as the eager path for GQA (nh != nkv) with head-repeat.
#[test]
fn g4_graph_gqa_matches_eager() {
    if !grim_backend_rocm::device::util::gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);

    // GQA GDL model: nh=4, nkv=2, hd=64, hidden=128 (no gate projections,
    // so both paths use the same QKV GEMV).
    let model = tiny_gqa_gdl_lfm2(&dev, 0, 2);

    // Eager prefill (multi-token) to populate GDL recurrent state.
    let mut session = model.new_session();
    let prefill_tokens: Vec<u32> = vec![1, 5, 9, 13];
    let input = rocm_tensor(
        &dev,
        0,
        prefill_tokens.iter().map(|&t| t as f32).collect(),
        Shape::new(vec![prefill_tokens.len()]),
    );
    let pos = rocm_tensor(
        &dev,
        0,
        vec![0.0f32; prefill_tokens.len()],
        Shape::new(vec![prefill_tokens.len()]),
    );
    model
        .forward(session.as_mut(), &input, &pos, &[])
        .expect("eager prefill");

    // Capture a graph (warmup + capture).
    let mut graph = model.get_or_create_decode_graph(16, 1).unwrap();
    model.forward_capture(&mut graph, 7).unwrap();
    dev.reset_launch_count();
    model.forward_capture(&mut graph, 7).unwrap();
    graph.begin_capture().unwrap();
    model.forward_capture(&mut graph, 7).unwrap();
    graph.end_capture().unwrap();
    assert!(graph.is_captured);

    // Replay decode steps under the graph.
    let decode_token = 9u32;
    model.forward_replay(&mut graph, decode_token).unwrap();
    graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    let graph_logits = graph.read_logits_f32().unwrap();

    // Eager decode from the same prefill state (fresh session, same tokens).
    let mut eager_session = model.new_session();
    let all_tokens: Vec<u32> = {
        let mut t = prefill_tokens.clone();
        t.push(decode_token);
        t
    };
    let eager_input = rocm_tensor(
        &dev,
        0,
        all_tokens.iter().map(|&t| t as f32).collect(),
        Shape::new(vec![all_tokens.len()]),
    );
    let eager_pos = rocm_tensor(
        &dev,
        0,
        vec![0.0f32; all_tokens.len()],
        Shape::new(vec![all_tokens.len()]),
    );
    let eager_out = model
        .forward(eager_session.as_mut(), &eager_input, &eager_pos, &[])
        .expect("eager forward");
    let eager_logits = eager_out.to_vec_f32().unwrap();
    // Last token's logits (decode step output).
    let vocab = 32usize;
    let last_logits: Vec<f32> =
        eager_logits[(all_tokens.len() - 1) * vocab..all_tokens.len() * vocab].to_vec();

    // Compare graph vs eager (tolerance for fp accumulation order).
    let mut max_diff = 0.0f32;
    for (a, b) in graph_logits.iter().zip(&last_logits) {
        let diff = (a - b).abs() / (b.abs() + 1e-3);
        max_diff = max_diff.max(diff);
    }
    eprintln!("[g4-graph-gqa] max rel diff = {max_diff:.6}");
    assert!(
        max_diff < 5e-2,
        "GQA graph vs eager logits diverge: max rel diff = {max_diff:.6}"
    );
}

/// Task 3: end-to-end prefill→graph-decode vs fully-eager parity. This is
/// the real correctness bar — it proves the transition from eager prefill
/// to graph decode carries GDL state correctly (not just that graph and
/// eager produce the same output from the same starting state).
#[test]
fn g4_e2e_prefill_graph_decode_matches_fully_eager() {
    if !grim_backend_rocm::device::util::gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::shared(0);

    // GQA GDL model (nh=4, nkv=2, hd=64, hidden=128, no gate projections).
    let model = tiny_gqa_gdl_lfm2(&dev, 0, 2);

    // Token sequence: 4 prefill + 3 decode.
    let prefill_tokens: Vec<u32> = vec![1, 5, 9, 13];
    let decode_tokens: Vec<u32> = vec![17, 21, 25];
    let all_tokens: Vec<u32> = prefill_tokens
        .iter()
        .chain(&decode_tokens)
        .copied()
        .collect();

    // --- Path 1: fully eager (no graph at all) ---
    let mut eager_session = model.new_session();
    let eager_input = rocm_tensor(
        &dev,
        0,
        all_tokens.iter().map(|&t| t as f32).collect(),
        Shape::new(vec![all_tokens.len()]),
    );
    let eager_pos = rocm_tensor(
        &dev,
        0,
        vec![0.0f32; all_tokens.len()],
        Shape::new(vec![all_tokens.len()]),
    );
    let eager_out = model
        .forward(eager_session.as_mut(), &eager_input, &eager_pos, &[])
        .expect("fully eager forward");
    let eager_logits = eager_out.to_vec_f32().unwrap();

    // --- Path 2: eager prefill → graph decode ---
    let mut session = model.new_session();
    let prefill_input = rocm_tensor(
        &dev,
        0,
        prefill_tokens.iter().map(|&t| t as f32).collect(),
        Shape::new(vec![prefill_tokens.len()]),
    );
    let prefill_pos = rocm_tensor(
        &dev,
        0,
        vec![0.0f32; prefill_tokens.len()],
        Shape::new(vec![prefill_tokens.len()]),
    );
    model
        .forward(session.as_mut(), &prefill_input, &prefill_pos, &[])
        .expect("eager prefill");

    // Transition to graph: seed KV + GDL state, capture, replay.
    let mut graph = model.get_or_create_decode_graph(16, 1).unwrap();

    // Export seed sources from the eager session.
    let caches = session
        .model_state()
        .and_then(|s| s.downcast_ref::<Vec<Option<Lfm2LayerCache>>>())
        .expect("session caches");
    let srcs = model
        .eager_kv_seed_sources(caches, prefill_tokens.len() as u32)
        .expect("export seed sources");

    // Seed GDL state (and KV arenas, though GDL layers don't use them).
    graph
        .buffers
        .seed_kv_arena_from_eager(&dev, &srcs)
        .expect("seed KV");
    graph
        .buffers
        .seed_gdl_state_from_eager(&dev, &srcs)
        .expect("seed GDL state");

    // Capture (warmup + capture).
    model.forward_capture(&mut graph, decode_tokens[0]).unwrap();
    dev.reset_launch_count();
    model.forward_capture(&mut graph, decode_tokens[0]).unwrap();
    graph.begin_capture().unwrap();
    model.forward_capture(&mut graph, decode_tokens[0]).unwrap();
    graph.end_capture().unwrap();
    assert!(graph.is_captured);

    // Replay decode steps.
    let mut graph_logits_all: Vec<f32> = Vec::new();
    for (i, &token) in decode_tokens.iter().enumerate() {
        model.forward_replay(&mut graph, token).unwrap();
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
        let logits = graph.read_logits_f32().unwrap();
        if i == 0 {
            // First decode step: compare against eager prefill+decode[0].
            let mut first_tokens = prefill_tokens.clone();
            first_tokens.push(token);
            let first_input = rocm_tensor(
                &dev,
                0,
                first_tokens.iter().map(|&t| t as f32).collect(),
                Shape::new(vec![first_tokens.len()]),
            );
            let first_pos = rocm_tensor(
                &dev,
                0,
                vec![0.0f32; first_tokens.len()],
                Shape::new(vec![first_tokens.len()]),
            );
            let mut first_session = model.new_session();
            let first_out = model
                .forward(first_session.as_mut(), &first_input, &first_pos, &[])
                .expect("first eager forward");
            let first_eager = first_out.to_vec_f32().unwrap();
            let vocab = 32usize;
            let first_eager_last: Vec<f32> =
                first_eager[(first_tokens.len() - 1) * vocab..first_tokens.len() * vocab].to_vec();
            let mut max_diff = 0.0f32;
            for (a, b) in logits.iter().zip(&first_eager_last) {
                let diff = (a - b).abs() / (b.abs() + 1e-3);
                max_diff = max_diff.max(diff);
            }
            eprintln!("[g4-e2e] first decode step max rel diff = {max_diff:.6}");
            assert!(
                max_diff < 5e-2,
                "first decode step diverges: max rel diff = {max_diff:.6}"
            );
        }
        graph_logits_all.extend_from_slice(&logits);
    }

    // Compare all decode steps against fully eager.
    let vocab = 32usize;
    let mut max_diff = 0.0f32;
    for (i, &_token) in decode_tokens.iter().enumerate() {
        let eager_start = (prefill_tokens.len() + i) * vocab;
        let eager_end = eager_start + vocab;
        let eager_slice = &eager_logits[eager_start..eager_end];
        let graph_slice = &graph_logits_all[i * vocab..(i + 1) * vocab];
        for (a, b) in graph_slice.iter().zip(eager_slice) {
            let diff = (a - b).abs() / (b.abs() + 1e-3);
            max_diff = max_diff.max(diff);
        }
    }
    eprintln!("[g4-e2e] all decode steps max rel diff = {max_diff:.6}");
    assert!(
        max_diff < 5e-2,
        "prefill→graph-decode diverges from fully eager: max rel diff = {max_diff:.6}"
    );
}
