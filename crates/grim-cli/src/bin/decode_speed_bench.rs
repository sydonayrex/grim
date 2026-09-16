//! Decode-speed benchmark: eager vs graph vs graph + GPU sampler.
//!
//! Builds a tiny in-memory LFM2 (no GGUF file needed) and measures wall-clock
//! ms/token for three paths. Run under rocprof to compare kernel launch counts:
//!
//!   rocprof --hip-trace --stats target/release/decode_speed_bench 2>&1
//!
//! Three paths:
//!   eager   — full model.forward() loop (per token) + D2H logits
//!   graph   — capture once, replay per token, D2H read_logits_f32
//!   graph+  — capture once, replay per token, GPU sampler (zero D2H)

use std::sync::Arc;
use std::time::Instant;

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::SamplingOps;
use grim_core::model::CausalLm;
use grim_models_transformer::lfm2::{Lfm2, Lfm2Block, Lfm2Config};
use grim_models_transformer::shared_moe::CharonCache;
use grim_nn::{Embedding, Linear, RmsNorm};
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

// ---- tiny model helpers (mirror the integration tests) ------------------------

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
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
        })
        .collect()
}

fn test_linear(dev: &RocmDevice, ordinal: usize, out: usize, inn: usize, seed: u64) -> Linear {
    let w = rocm_tensor(dev, ordinal, rand_vec(out * inn, seed), Shape::new(vec![out, inn]));
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
) -> Lfm2Block {
    let nh = n_q / hd;
    let nkv = n_kv / hd;
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
        wqkv_q80_fused: None,
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
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        rope_theta: 10000.0,
        eps: 1e-5,
        charon_cache: CharonCache::new(),
    }
}

fn tiny_lfm2(dev: &RocmDevice, ordinal: usize, n_layers: usize) -> Lfm2 {
    // Realistic small-LFM2-like dims. The eager device path crashes with very
    // small sizes (hidden < ~64) due to a pre-existing rocBLAS/autotune minimum
    // dimension issue unrelated to graph capture, so we stay above that floor.
    let hidden = 128usize;
    let hd = 16usize;
    let nh = 4usize;
    let nkv = 2usize;
    let inter = 256usize;
    let vocab = 256usize;
    let layers: Vec<Lfm2Block> = (0..n_layers)
        .map(|_| attention_block(dev, ordinal, hidden, nh * hd, nkv * hd, hd, inter))
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

fn main() {
    let ordinal = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0usize);
    let n_steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    if !RocmDevice::probe_one(ordinal).unwrap_or(false) {
        eprintln!("no ROCm ordinal {ordinal}");
        std::process::exit(2);
    }
    let dev = RocmDevice::shared(ordinal);

    println!(
        "decode_speed_bench  ordinal={ordinal} steps={n_steps}  (tiny in-memory LFM2)"
    );

    for &n_layers in &[1usize, 2, 4] {
        let model = tiny_lfm2(&dev, ordinal, n_layers);

        // ------- eager path: full forward() loop + D2H -------
        // Run eager WITHOUT GRIM_DECODE_GRAPH and with GRIM_ROPE_DEV_BASE=0 so
        // the non-device eager path is used. The device eager path
        // (lfm2.rs:883-894) has a latent cache-unwrap bug (assumes cache is
        // Some before its lazy-init) that is pre-existing and unrelated to
        // graph capture. The non-device path (lfm2.rs:1179) inits cache
        // correctly. This is the fair baseline the graph path replaces.
        unsafe {
            std::env::remove_var("GRIM_DECODE_GRAPH");
            std::env::set_var("GRIM_ROPE_DEV_BASE", "0");
        }
        // Warmup (JIT + allocator).
        {
            let mut session = model.new_session();
            for t in 0..5u32 {
                let input = rocm_tensor(&dev, ordinal, vec![t as f32], Shape::new(vec![1]));
                let pos = rocm_tensor(&dev, ordinal, vec![t as f32], Shape::new(vec![1]));
                let _ = model.forward(&mut *session, &input, &pos, &[]);
            }
        }
        dev.synchronize();
        dev.reset_launch_count();
        let t0 = Instant::now();
        for t in 0..n_steps as u32 {
            // Fresh session per iteration: reusing one across eager forwards
            // corrupts the layer cache state and faults the GPU.
            let mut session = model.new_session();
            let input = rocm_tensor(&dev, ordinal, vec![t as f32], Shape::new(vec![1]));
            let pos = rocm_tensor(&dev, ordinal, vec![t as f32], Shape::new(vec![1]));
            let _ = model.forward(&mut *session, &input, &pos, &[]);
            dev.synchronize();
        }
        dev.synchronize();
        let eager_us = t0.elapsed().as_micros();
        let eager_launches = dev.launch_count();

        // ------- graph path: capture once, replay per token + D2H -------
        let mut graph = model.get_or_create_decode_graph(256, 1).unwrap();
        model.forward_capture(&mut graph, 0).unwrap(); // warmup (JIT)
        graph.begin_capture().unwrap();
        model.forward_capture(&mut graph, 0).unwrap();
        graph.end_capture().unwrap();
        graph.buffers.current_pos = 1;

        for t in 0..5u32 {
            model.forward_replay(&mut graph, t).unwrap();
        }
        dev.synchronize();
        dev.reset_launch_count();
        let t0 = Instant::now();
        for t in 0..n_steps as u32 {
            model.forward_replay(&mut graph, t).unwrap();
            let _ = graph.read_logits_f32().unwrap();
        }
        dev.synchronize();
        let graph_us = t0.elapsed().as_micros();
        let graph_launches = dev.launch_count();

        // ------- graph+GPU-sampler: replay + device-side sample, zero D2H -------
        dev.reset_launch_count();
        let t0 = Instant::now();
        let mut sampled = 0u64;
        for t in 0..n_steps as u32 {
            model.forward_replay(&mut graph, t).unwrap();
            let logits_storage = graph.logits_device_storage();
            let tok = dev
                .sample_on_device(logits_storage, 0.0, 1.0, 0, t as u64)
                .unwrap();
            sampled += tok as u64;
        }
        dev.synchronize();
        let graph_gpu_us = t0.elapsed().as_micros();
        let graph_gpu_launches = dev.launch_count();

        let fmt = |us: u128| {
            let ms_tok = us as f64 / n_steps as f64 / 1000.0;
            let tok_s = 1e6 / (us as f64 / n_steps as f64);
            (ms_tok, tok_s)
        };
        let (e_ms, e_s) = fmt(eager_us);
        let (g_ms, g_s) = fmt(graph_us);
        let (gg_ms, gg_s) = fmt(graph_gpu_us);

        println!(
            "\nlayers={n_layers}\n\
             path          ms/tok      tok/s       launches/tok\n\
             eager         {:9.3}  {:10.1}  {:.1}\n\
             graph         {:9.3}  {:10.1}  {:.1}\n\
             graph+gpusp   {:9.3}  {:10.1}  {:.1}\n",
            e_ms,
            e_s,
            eager_launches as f64 / n_steps as f64,
            g_ms,
            g_s,
            graph_launches as f64 / n_steps as f64,
            gg_ms,
            gg_s,
            graph_gpu_launches as f64 / n_steps as f64,
        );
        println!(
            "  speedup graph/eager = {:.2}x   graph+gpusp/eager = {:.2}x   sampled_checksum={}",
            e_ms / g_ms,
            e_ms / gg_ms,
            sampled
        );
    }
}
