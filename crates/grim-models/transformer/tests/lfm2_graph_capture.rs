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
use grim_models_transformer::lfm2::{Lfm2, Lfm2Block, Lfm2Config};
use grim_models_transformer::lfm2_graph::write_embedding_to_buffer;
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

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
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
        })
        .collect()
}

fn test_linear(dev: &RocmDevice, ordinal: usize, out_dim: usize, in_dim: usize, seed: u64) -> Linear {
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

    let (wqkv_q80_fused, w_gate_up_q80_fused) = if fused {
        let q = test_linear_q80(dev, ordinal, n_q, hidden, 101);
        let k = test_linear_q80(dev, ordinal, n_kv, hidden, 102);
        let v = test_linear_q80(dev, ordinal, n_kv, hidden, 103);
        let gate = test_linear_q80(dev, ordinal, inter, hidden, 104);
        let up = test_linear_q80(dev, ordinal, inter, hidden, 105);
        let fq = dev
            .build_fused_qkv_q80(
                q.weight.storage().as_ref(),
                k.weight.storage().as_ref(),
                v.weight.storage().as_ref(),
            )
            .expect("build fused Q80 QKV");
        let fgu = dev
            .build_fused_gate_up_q80(
                gate.weight.storage().as_ref(),
                up.weight.storage().as_ref(),
            )
            .expect("build fused Q80 gate+up");
        (Some(fq), Some(fgu))
    } else {
        (None, None)
    };

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
        wqkv_q80_fused,
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
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        rope_theta: 10000.0,
        eps: 1e-5,
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
            expert_weights_scale: 0.0,
            expert_gating_func: 0,
            n_swa: 0,
            swa_type: 0,
            n_embd_out: 0,
            mxfp4_qkv_attention: false,
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
    if RocmDevice::probe_one(0).unwrap_or(false) == false {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let dev = RocmDevice::shared(0);

    // Baseline: F32 attention weights and F32 FFN weights.
    let plain = tiny_lfm2(&dev, 0, 2, false);

    // Pool allocates at stable addresses.
    let mut graph = plain.get_or_create_decode_graph(16, 1).unwrap();
    let in_ptr = graph.buffers.layer_input[0]
        .device_ptr_u64()
        .unwrap();
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

    // Fused path: Q8_0 QKV and gate+up blobs.
    let model = tiny_lfm2(&dev, 0, 2, true);
    assert!(
        model.layers[0].wqkv_q80_fused.is_some(),
        "fused Q8_0 QKV weights must be enabled for the graph path"
    );
    assert!(
        model.layers[0].w_gate_up_q80_fused.is_some(),
        "fused Q8_0 gate+up weights must be enabled for the graph path"
    );

    // Capture must enqueue fewer GEMM launches than the F32 baseline:
    // one fused QKV dot4 replaces 3 GEMMs, one fused gate+up replaces 2.
    dev.reset_launch_count();
    graph.begin_capture().unwrap();
    model.forward_capture(&mut graph, 7).unwrap();
    graph.end_capture().unwrap();
    assert!(graph.is_captured);
    let fused_gemms = dev.launch_count();
    assert!(
        fused_gemms < eager_gemms,
        "fused path should have fewer GEMMs than plain path: fused={fused_gemms}, plain={eager_gemms}"
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
    if RocmDevice::probe_one(0).unwrap_or(false) == false {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let dev = RocmDevice::shared(0);
    let model = tiny_lfm2(&dev, 0, 1, false);
    let graph = model.get_or_create_decode_graph(16, 1).unwrap();

    // A block with no QKV projections is recurrent-shaped: must refuse with
    // Unimplemented (caller falls back eager), never panic or record junk.
    let rec_block = Lfm2Block {
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
