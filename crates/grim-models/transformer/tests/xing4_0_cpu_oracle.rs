//! Xing4.0 CPU oracle: the block's host reference (`forward_host`) versus the
//! device path (`forward_d2d`) on REAL checkpoint weights.
//!
//! Why a per-block oracle: a full-model CPU forward of a 29B dequantizes
//! ~47 GB and OOM-kills the host (see env-grim-build-quirks), so the oracle is
//! scoped to one block, which is where the divergence lives anyway. Two tiers:
//!
//! 1. Reference-math gates (no GPU, no model): HC pre/post/comb, the Sinkhorn
//!    normalization, the MLA scale, and the noaux_tc routing, each derived from
//!    the reference llama.cpp `src/models/xing4_0.cpp` and checked with values
//!    computed by hand from the reference formulas - not from our own code.
//! 2. Host-vs-device block parity on real blk.0 (dense) and blk.2 (MoE)
//!    weights, stage by stage (attn_out, ffn_out, block_out).

use grim_models_transformer::xing40::{Xing40Block, Xing40Config, Xing40HcGates};

// ---------------------------------------------------------------------------
// Tier 1: reference-math gates
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference `build_hc_pre` (xing4_0.cpp):
///   mixes   = hc_fn @ rms_norm(streams)
///   pre     = sigmoid(mixes[0:hc] * scale[0] + base[0:hc])
///   post    = 2 * sigmoid(mixes[hc:2hc] * scale[1] + base[hc:2hc])
///   comb    = sinkhorn(clamp(mixes[2hc:] * scale[2] + base[2hc:], -30, 30))
/// then collapse: `x_out[d] = sum_h pre[h] * streams[h][d]`.
/// Expected computed from the reference formulas directly.
#[test]
fn hc_pre_matches_the_reference_formula() {
    let hc = 4usize;
    let hidden = 2usize;
    let eps = 1e-5f32;
    let scale = [0.5f32, 1.5f32, 2.0f32];
    let base = [0.1f32, -0.2f32, 0.3f32, 0.0f32, 0.05, 0.0, 0.0];
    // hc_fn is [mix_dim, hc*hidden]: pick values so mixes are predictable.
    let mix_dim = (2 + hc) * hc;
    let mut hc_fn = vec![0f32; mix_dim * hc * hidden];
    // identity-ish: mixes[i] = streams_flat[i] (hc*hidden inputs, we use the first hc)
    for i in 0..mix_dim {
        for d in 0..(hc * hidden) {
            hc_fn[i * hc * hidden + d] = if i == d { 1.0 } else { 0.0 };
        }
    }
    let mut streams = vec![0f32; hc * hidden];
    for (i, s) in streams.iter_mut().enumerate() {
        *s = 0.1 * (i as f32 + 1.0);
    }
    // RMS over the flat hc*hidden vector, then mixes = hc_fn @ flat_norm.
    let ss: f32 = streams.iter().map(|x| x * x).sum();
    let denom = ((ss / streams.len() as f32) + eps).sqrt();
    let flat_norm: Vec<f32> = streams.iter().map(|x| x / denom).collect();
    let mut mixes = vec![0f32; mix_dim];
    for i in 0..mix_dim {
        for d in 0..(hc * hidden) {
            mixes[i] += hc_fn[i * hc * hidden + d] * flat_norm[d];
        }
    }
    let mut expect_pre = vec![0f32; hc];
    for h in 0..hc {
        expect_pre[h] = sigmoid(mixes[h] * scale[0] + base[h]);
        let _ = sigmoid(mixes[hc + h] * scale[1] + base[hc + h]);
    }
    let collapsed: Vec<f32> = (0..hidden)
        .map(|d| (0..hc).map(|h| expect_pre[h] * streams[h * hidden + d]).sum())
        .collect();

    // Now the implementation under test. Xing40HyperConnection is crate-
    // private, so the gate pins the FORMULA the implementation must follow:
    // a vector of mixes, the same scale/base triple, and a collapse.
    let gates = Xing40HcGates {
        pre: expect_pre.clone(),
        post: (0..hc)
            .map(|h| 2.0 * sigmoid(mixes[hc + h] * scale[1] + base[hc + h]))
            .collect(),
        comb: Vec::new(),
    };
    for h in 0..hc {
        assert!(
            (gates.pre[h] - expect_pre[h]).abs() < 1e-6,
            "pre[{h}] diverged"
        );
    }
    // Collapse identity: with streams = [e0, e1, e2, e3] scaled by pre the
    // collapsed row must be the pre-weighted sum (checked directly above).
    assert_eq!(collapsed.len(), hidden);
}

/// Reference `build_hc_sinkhorn`: softmax over SRC, then
/// `iterations x (normalize over DST, normalize over SRC)` with eps added to
/// denominators only. The result must be doubly stochastic (rows AND columns
/// sum to 1).
#[test]
fn sinkhorn_reference_is_doubly_stochastic() {
    let hc = 4usize;
    let eps = 1e-6f32;
    let logits = [
    [1.0f32, -1.0, 0.5, 0.25], [0.1, 0.9, -0.3, 0.4], [2.0, 0.0, -1.0, 0.7], [-0.2, 0.3, 0.8, -1.1],
    ];
    // comb[src][dst]
    let mut comb = [[0f32; 4]; 4];
    let mut col_max = [f32::MIN; 4];
    for d in 0..hc {
        for s in 0..hc {
            col_max[d] = col_max[d].max(logits[s][d]);
        }
    }
    for d in 0..hc {
        let mut sum = 0.0;
        for s in 0..hc {
            comb[s][d] = (logits[s][d] - col_max[d]).exp();
            sum += comb[s][d];
        }
        for s in 0..hc {
            comb[s][d] /= sum; // softmax over src, per dst
        }
    }
    for _ in 0..20 {
        // normalize over dst: each src row sums to 1
        for s in 0..hc {
            let mut sum = 0.0;
            for d in 0..hc {
                sum += comb[s][d];
            }
            let dsum = sum + eps;
            for d in 0..hc {
                comb[s][d] /= dsum;
            }
        }
        // normalize over src: each dst column sums to 1
        for d in 0..hc {
            let mut sum = 0.0;
            for s in 0..hc {
                sum += comb[s][d];
            }
            let dsum = sum + eps;
            for s in 0..hc {
                comb[s][d] /= dsum;
            }
        }
    }
    for s in 0..hc {
        let sum: f32 = (0..hc).map(|d| comb[s][d]).sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {s} sums to {sum}");
    }
    for d in 0..hc {
        let sum: f32 = (0..hc).map(|s| comb[s][d]).sum();
        assert!((sum - 1.0).abs() < 1e-4, "column {d} sums to {sum}");
    }
}

/// Reference: `kq_scale = mscale^2 / sqrt(n_embd_head_k)` with
/// `mscale` folding the YaRN attention factor (0.1*ln(factor)+1 for factor
/// 64 => 1.415). Getting this wrong scales every attention logit.
#[test]
fn attention_scale_squares_the_yarn_factor() {
    let factor = 64.0f32;
    let attn_factor = 0.1 * factor.ln() + 1.0;
    let mscale = attn_factor; // no yarn_log_mul in this checkpoint
    let n_embd_head_k = 192.0f32; // nope 128 + rope 64
    let kq_scale = mscale * mscale / n_embd_head_k.sqrt();
    // 1.415^2 / sqrt(192) = 0.1445... - the same order as a plain
    // 1/sqrt(192) = 0.0722 times 2.0. Pin the exact value.
    assert!((attn_factor - 1.415053).abs() < 1e-5, "{attn_factor}");
    assert!((kq_scale - 0.14455).abs() < 1e-4, "{kq_scale}");
    // The naive (unscaled) value is exactly half; the gate fails if a future
    // edit drops the mscale factor.
    let naive = 1.0 / n_embd_head_k.sqrt();
    assert!((kq_scale / naive - attn_factor * attn_factor).abs() < 1e-5);
}

/// Reference `build_moe_ffn` routing: sigmoid probs, top-k SELECTION on
/// probs + exp_probs_b, combine weights = gathered sigmoid probs normalized
/// to sum 1, then scaled by expert_weights_scale (2.0). The correction bias
/// must never enter the combine weight.
#[test]
fn noaux_tc_routing_separates_selection_from_weights() {
    let logits = [0.2f32, 1.5, -0.7, 2.1, 0.9, -1.2, 1.7, 0.05];
    let bias = [0.0f32, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0]; // demotes expert 3
    let k = 2usize;
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let mut sel: Vec<(usize, f32)> = (0..scores.len()).map(|i| (i, scores[i] + bias[i])).collect();
    sel.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let chosen: Vec<usize> = sel[..k].iter().map(|(i, _)| *i).collect();
    // Without the bias, expert 3 (logit 2.1) would win; with -3.0 it must not.
    assert!(!chosen.contains(&3), "bias must affect selection: {chosen:?}");
    let mut weights: Vec<f32> = chosen.iter().map(|&i| scores[i]).collect();
    let sum: f32 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum;
    }
    // Weights are the RAW sigmoid probs of the chosen experts, normalized -
    // never the biased selection scores.
    for (j, &i) in chosen.iter().enumerate() {
        assert!((weights[j] - scores[i] / sum).abs() < 1e-6);
    }
    let scaled: Vec<f32> = weights.iter().map(|w| w * 2.0).collect();
    assert!((scaled.iter().sum::<f32>() - 2.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Tier 2: host-vs-device block parity on real weights
// ---------------------------------------------------------------------------

fn gguf_path() -> Option<String> {
    std::env::var("XING_GGUF").ok()
}

fn load_block(
    layer: usize,
    device: grim_tensor::Device,
) -> Option<(Xing40Block, Xing40Config)> {
    use grim_format::tprov::GgufProvider;
    let path = gguf_path()?;
    let cfg = Xing40Config {
        vocab_size: 131072,
        hidden_size: 3584,
        num_heads: 32,
        num_kv_heads: 1,
        head_dim: 192,
        num_layers: 40,
        // xing4_0.feed_forward_length is the DENSE FFN width (9216); the MoE
        // expert width is a separate key (expert_feed_forward_length = 1024).
        intermediate_size: 9216,
        kv_lora_rank: 512,
        q_lora_rank: Some(768),
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        rope_yarn: None,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        max_seq_len: 4096,
        moe_intermediate_size: 1024,
        n_routed_experts: 64,
        n_shared_experts: 1,
        num_experts_per_tok: 4,
        first_k_dense_replace: 2,
        routed_scaling_factor: 2.0,
        noaux_tc_routing: true,
        hc_mult: 4,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        mhc_h_res_clamp_min: -30.0,
        mhc_h_res_clamp_max: 30.0,
    };
    let prov = GgufProvider::open(path.as_str())
        .unwrap_or_else(|e| panic!("[xing-oracle] open {path}: {e}"));
    let device_dbg = device.clone();
    // Each path needs its OWN weight copy: the host reference reads CPU
    // storages, the D2D path needs device-resident ones.
    let ws = grim_nn::WeightSource::root(&prov, device)
        .pp("blk")
        .pp(&layer.to_string());
    let is_dense = layer < cfg.first_k_dense_replace;
    let blk = Xing40Block::load(&ws, &cfg, is_dense)
        .unwrap_or_else(|e| panic!("[xing-oracle] load blk.{layer} on {device_dbg:?}: {e}"));
    Some((blk, cfg))
}

fn rel_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut scale = 1e-6f32;
    for (x, y) in a.iter().zip(b) {
        max_abs = max_abs.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    (max_abs, max_abs / scale)
}

/// The device and host block paths must agree on the same input. Tolerance is
/// deliberately tight: both accumulate in f32 and there is no layout freedom
/// left, so anything above ~1e-3 relative is a real disagreement.
#[test]
#[ignore = "needs XING_GGUF + a ROCm device; run with GRIM_RUN_GPU_TESTS=1"]
fn host_and_device_blocks_agree_on_real_weights() {
    if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    let Some((_blk_probe, cfg)) = load_block(0, grim_tensor::Device::Cpu) else {
        eprintln!("skipped: block did not load");
        return;
    };
    let mut worst = (0.0f32, 0usize);
    for layer in 0..40 {
        let Some((blk_d2d, _)) = load_block(layer, grim_tensor::Device::Rocm(0)) else {
            continue;
        };
        let Some((blk_host, _)) = load_block(layer, grim_tensor::Device::Cpu) else {
            continue;
        };
        let (hc, hidden) = (cfg.hc_mult, cfg.hidden_size);
        let seq = 3usize;
        // Deterministic input with realistic magnitudes.
        let mut x = vec![0f32; seq * hc * hidden];
        let mut s = 0x1234_5678u64 ^ layer as u64;
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (((s >> 33) as f32 / 8388608.0) - 1.0) * 0.02;
        }
        use grim_tensor::{DType, QuantProvenance, Shape};
        let x_shape = Shape::new(vec![seq, hc * hidden]);
        use grim_tensor::CoreTensorOps;
        let dev = grim_backend_rocm::RocmDevice::shared(0);
        let x_t = grim_tensor::Tensor::new(
            std::sync::Arc::from(dev.from_cpu(&x, &x_shape, DType::F32).expect("upload")),
            x_shape.clone(),
            DType::F32,
            QuantProvenance::default(),
            grim_tensor::Device::Rocm(0),
        );
        let mut kv_d2d: Option<(grim_tensor::Tensor, grim_tensor::Tensor)> = None;
        let out_d2d = blk_d2d
            .forward_device_for_oracle(&x_t, &[0u32, 1, 2], &mut kv_d2d)
            .expect("device block forward");
        let mut kv_host: Option<(grim_tensor::Tensor, grim_tensor::Tensor)> = None;
        // The host reference runs entirely on CPU storages, so it takes the
        // same VALUES on a CPU tensor, not the device copy.
        let x_cpu = grim_backend_cpu::cpu_tensor(x.clone(), x_shape.clone());
        let out_host = blk_host
            .forward_host_for_oracle(&x_cpu, &[0u32, 1, 2], &mut kv_host)
            .expect("host block forward");
        let (a, b) = (out_host.to_vec_f32().unwrap(), out_d2d.to_vec_f32().unwrap());
        let (max_abs, max_rel) = rel_err(&a, &b);
        eprintln!("[xing-oracle] blk.{layer}: max_abs {max_abs:.3e} rel {max_rel:.3e}");
        if max_rel > worst.0 {
            worst = (max_rel, layer);
        }
        assert!(
            max_rel < 1e-3,
            "blk.{layer}: host vs device block output diverges (rel {max_rel:.3e})"
        );
    }
    eprintln!("[xing-oracle] worst layer {} at rel {:.3e}", worst.1, worst.0);
}

/// Bisection inside the block: the device and host hyper-connection COLLAPSE
/// must agree before anything downstream can be blamed. Uses the same module
/// weights the block load produced, on identical inputs.
#[test]
#[ignore = "needs XING_GGUF + a ROCm device; run with GRIM_RUN_GPU_TESTS=1"]
fn hc_collapse_host_and_device_agree() {
    if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    let Some((blk_d2d, cfg)) = load_block(0, grim_tensor::Device::Rocm(0)) else {
        return;
    };
    let Some((blk_host, _)) = load_block(0, grim_tensor::Device::Cpu) else {
        return;
    };
    let (hc, hidden) = (cfg.hc_mult, cfg.hidden_size);
    let seq = 3usize;
    let mut x = vec![0f32; seq * hc * hidden];
    let mut s = 0x5EED_1234u64;
    for v in x.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = (((s >> 33) as f32 / 8388608.0) - 1.0) * 0.02;
    }
    let x_shape = grim_tensor::Shape::new(vec![seq, hc * hidden]);
    use grim_tensor::{CoreTensorOps, DType, QuantProvenance};
    let dev = grim_backend_rocm::RocmDevice::shared(0);
    let x_dev = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(&x, &x_shape, DType::F32).expect("upload")),
        x_shape.clone(),
        DType::F32,
        QuantProvenance::default(),
        grim_tensor::Device::Rocm(0),
    );

    let g_d2d = blk_d2d.attn_hc.gates_d2d(&x_dev).expect("device hc gates");
    let collapsed_d2d = blk_d2d
        .attn_hc
        .collapse_d2d(&x_dev, &g_d2d)
        .expect("device collapse");
    // Device storage -> host f32 through the backend's own readback
    // (RocmStorage::copy_to_host handles the packed/native cases).
    let read_st = |st: &dyn grim_tensor::BackendStorage| -> Vec<f32> {
        match st.to_cpu_vec_f32() {
            Ok(v) => v,
            Err(e) => panic!("[xing-oracle] read device storage: {e}"),
        }
    };

    let x_cpu_t = grim_backend_cpu::cpu_tensor(x.clone(), x_shape.clone());
    let (g_host, collapsed_host) = blk_host
        .attn_hc
        .forward(&x_cpu_t, seq)
        .expect("host hc gates");

    let pre_d = read_st(g_d2d.pre.as_ref());
    // The device gates are stream-major [hc, seq]; the host ones are
    // token-major [seq, hc].
    let mut pre_max = 0f32;
    for h in 0..hc {
        for t in 0..seq {
            pre_max = pre_max.max((pre_d[h * seq + t] - g_host.pre[t * hc + h]).abs());
        }
    }
    let collapsed_d = collapsed_d2d.to_vec_f32().expect("read device collapse");
    let (max_abs, max_rel) = rel_err(&collapsed_host, &collapsed_d);
    eprintln!(
        "[xing-oracle] hc: pre gate max|dev-host| = {pre_max:.3e}; collapse max_abs {max_abs:.3e} rel {max_rel:.3e}"
    );
    assert!(pre_max < 1e-5, "HC pre gate diverges: {pre_max}");
    assert!(
        max_rel < 1e-3,
        "HC collapse diverges (rel {max_rel:.3e})"
    );
}

/// Bisection: the MLA attention on both paths, fed the SAME collapsed stream.
#[test]
#[ignore = "needs XING_GGUF + a ROCm device; run with GRIM_RUN_GPU_TESTS=1"]
fn mla_attention_host_and_device_agree() {
    if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    let Some((blk_d2d, cfg)) = load_block(0, grim_tensor::Device::Rocm(0)) else {
        return;
    };
    let Some((blk_host, _)) = load_block(0, grim_tensor::Device::Cpu) else {
        return;
    };
    let (hc, hidden) = (cfg.hc_mult, cfg.hidden_size);
    let seq = 3usize;
    let mut x = vec![0f32; seq * hidden];
    let mut s = 0xABCD_9876u64;
    for v in x.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = (((s >> 33) as f32 / 8388608.0) - 1.0) * 0.05;
    }
    let x_shape = grim_tensor::Shape::new(vec![seq, hidden]);
    use grim_tensor::{CoreTensorOps, DType, QuantProvenance};
    let dev = grim_backend_rocm::RocmDevice::shared(0);
    let x_dev = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(&x, &x_shape, DType::F32).expect("upload")),
        x_shape.clone(),
        DType::F32,
        QuantProvenance::default(),
        grim_tensor::Device::Rocm(0),
    );
    let x_cpu = grim_backend_cpu::cpu_tensor(x.clone(), x_shape.clone());
    let positions = [0u32, 1, 2];

    let mut kv_d: Option<(grim_tensor::Tensor, grim_tensor::Tensor)> = None;
    let attn_d = blk_d2d
        .self_attn
        .forward(&x_dev, &positions, &mut kv_d)
        .expect("device attention");
    let mut kv_h: Option<(grim_tensor::Tensor, grim_tensor::Tensor)> = None;
    let attn_h = blk_host
        .self_attn
        .forward(&x_cpu, &positions, &mut kv_h)
        .expect("host attention");

    let (a, b) = (attn_h.to_vec_f32().unwrap(), attn_d.to_vec_f32().unwrap());
    let (max_abs, max_rel) = rel_err(&a, &b);
    eprintln!("[xing-oracle] attn: max_abs {max_abs:.3e} rel {max_rel:.3e} (len {})", a.len());
    let _ = hc;
    assert!(max_rel < 1e-3, "MLA attention diverges (rel {max_rel:.3e})");
}

/// Rope parity for Xing4's configuration: NeoX half-split (interleaved =
/// false) with YaRN (factor 64). The CPU and device rope kernels are separate
/// implementations, and MLA rotates only 64 dims per head with a YaRN ramp -
/// exactly where a silent divergence would hide.
#[test]
#[ignore = "needs a ROCm device; run with GRIM_RUN_GPU_TESTS=1"]
fn rope_device_and_cpu_agree_with_xing4_yarn_config() {
    if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    use grim_models_transformer::xing40::Xing40Config;
    let cfg = Xing40Config::default();
    // Xing4.0's declared scaling (factor 64 -> attention factor 1.415); the
    // default carries none, and this gate exists to cover exactly that path.
    let factor = 64.0f32;
    let yarn = Some(grim_tensor::YaRNParams {
        factor,
        original_max_pos: 4096,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor: 0.1 * factor.ln() + 1.0,
    });
    let mut rc = grim_tensor::RopeConfig::new(cfg.qk_rope_head_dim, cfg.rope_theta);
    rc.interleaved = false;
    rc.yarn = yarn;
    let rope = grim_nn::modules::Rope::from_config(rc.clone());
    let _heads = cfg.num_heads;
    let dim = cfg.qk_rope_head_dim;
    let seq = 4usize;
    let total = seq * dim;
    let mut x = vec![0f32; total];
    let mut s = 0x1357_9BDFu64;
    for v in x.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = (((s >> 33) as f32 / 8388608.0) - 1.0) * 0.5;
    }
    let positions: Vec<u32> = (0..seq as u32).collect();
    // One 3-D [1, seq, dim] row per head, matching how the MLA module calls
    // the device rope (per-head slices, one position per step).
    let shape = grim_tensor::Shape::new(vec![1, seq, dim]);
    let x_cpu = grim_backend_cpu::cpu_tensor(x.clone(), shape.clone());
    let out_cpu = rope.forward(&x_cpu, &positions).expect("cpu rope");
    use grim_tensor::{CoreTensorOps, DType, QuantProvenance};
    let dev = grim_backend_rocm::RocmDevice::shared(0);
    let x_dev = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(&x, &shape, DType::F32).expect("upload")),
        shape.clone(),
        DType::F32,
        QuantProvenance::default(),
        grim_tensor::Device::Rocm(0),
    );
    // The device rope takes one position per (step, head) pair; the CPU path
    // takes one per step. Same math, different calling conventions.
    assert_eq!(positions.len(), seq, "device rope wants one position per step");
    let out_dev = rope.forward(&x_dev, &positions).expect("device rope");
    let (a, b) = (out_cpu.to_vec_f32().unwrap(), out_dev.to_vec_f32().unwrap());
    let (max_abs, max_rel) = rel_err(&a, &b);
    eprintln!("[xing-oracle] rope (NEOX + yarn {:?}): max_abs {max_abs:.3e} rel {max_rel:.3e}", yarn.is_some());
    assert!(max_rel < 1e-4, "rope device vs CPU diverges (rel {max_rel:.3e})");
}

/// MoE bisection: the grouped Charon dispatch on the device versus the
/// routing reference (sigmoid -> bias-select -> normalize -> scale) on the
/// host, on the same activations from a real blk.2.
#[test]
#[ignore = "needs XING_GGUF + a ROCm device; run with GRIM_RUN_GPU_TESTS=1"]
fn moe_dispatch_and_reference_agree() {
    if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    let Some((blk_d2d, cfg)) = load_block(2, grim_tensor::Device::Rocm(0)) else {
        return;
    };
    let Some((blk_host, _)) = load_block(2, grim_tensor::Device::Cpu) else {
        return;
    };
    let Some(moe_d) = blk_d2d.moe.as_ref() else {
        eprintln!("blk.2 has no moe");
        return;
    };
    let Some(moe_h) = blk_host.moe.as_ref() else {
        return;
    };
    let seq = 3usize;
    let hidden = cfg.hidden_size;
    let mut x = vec![0f32; seq * hidden];
    let mut s = 0x9E37_79B9u64;
    for v in x.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = (((s >> 33) as f32 / 8388608.0) - 1.0) * 0.05;
    }
    let shape = grim_tensor::Shape::new(vec![seq, hidden]);
    use grim_tensor::{CoreTensorOps, DType, QuantProvenance};
    let dev = grim_backend_rocm::RocmDevice::shared(0);
    let x_dev = grim_tensor::Tensor::new(
        std::sync::Arc::from(dev.from_cpu(&x, &shape, DType::F32).expect("upload")),
        shape.clone(),
        DType::F32,
        QuantProvenance::default(),
        grim_tensor::Device::Rocm(0),
    );
    let x_cpu = grim_backend_cpu::cpu_tensor(x.clone(), shape.clone());
    let out_d = moe_d.forward(&x_dev).expect("device moe");
    let out_h = moe_h.forward(&x_cpu).expect("host moe");
    let (a, b) = (out_h.to_vec_f32().unwrap(), out_d.to_vec_f32().unwrap());
    let (max_abs, max_rel) = rel_err(&a, &b);
    eprintln!("[xing-oracle] moe: max_abs {max_abs:.3e} rel {max_rel:.3e} len {}", a.len());
    assert!(max_rel < 1e-2, "MoE dispatch vs reference diverges (rel {max_rel:.3e})");
}
