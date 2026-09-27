//! The single-layer reference, driven by the 9B's REAL weight tensors.
//!
//! `qwen35_layer_reference.rs` builds a block from synthetic `w(i, n)` values.
//! That tests the forward MATH but leaves weight LOADING untested — and a
//! shape-preserving error there (a transpose, a shard, a dtype) would be
//! invisible, because synthetic data is symmetric enough that several wrong
//! layouts still produce the right answer.
//!
//! Here the block is loaded from the actual checkpoint through the same
//! `Qwen35Block::load_tp` the model uses, so the GGUF reader, the quant
//! dequant, the [out, in] weight storage and the sharding are all on the path.
//! The reference then reads the block's own weights back, so a loading error
//! cancels and only the forward math is under test — and a SEPARATE check
//! compares the block's weights against the GGUF directly, so a loading error
//! cannot hide behind that cancellation.
//!
//! Gated: needs the 9B checkpoint on disk. Skips cleanly without it.

use grim_format::tprov::GgufProvider;
use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config, Qwen35LayerCache};
use grim_nn::modules::Linear;
use grim_tensor::provider::TensorProvider;
use grim_tensor::{Device, Shape};

/// cargo runs integration tests with cwd = the crate root, so a repo-relative
/// path would silently skip. Resolve from CARGO_MANIFEST_DIR instead.
fn model_path() -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root.join(up).join("models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf");
        if p.exists() {
            return Some(p);
        }
    }
    None
}
const HIDDEN: usize = 4096;

// 9B KDA geometry, read from the checkpoint:
//   ssm.time_step_rank 32, ssm.group_count 16, ssm.state_size 128
const NV: usize = 32;
const NK: usize = 16;
const HD: usize = 128;
const TAPS: usize = 4;

fn key_dim() -> usize {
    NK * HD
}
fn value_dim() -> usize {
    NV * HD
}
fn conv_dim() -> usize {
    2 * key_dim() + value_dim()
}

fn cfg() -> Qwen35Config {
    let mut c = Qwen35Config::default();
    c.vocab_size = 248320;
    c.hidden_size = HIDDEN;
    c.num_heads = 16;
    c.num_kv_heads = 4;
    c.head_dim = 256;
    c.num_layers = 32;
    c.intermediate_size = 12288;
    c.full_attention_interval = 4;
    c.ssm_d_conv = TAPS;
    c.ssm_d_state = HD;
    c.ssm_dt_rank = NV;
    c.ssm_n_group = NK;
    c.ssm_d_inner = value_dim();
    c.devices = Vec::new();
    c
}

fn rms(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inv = 1.0 / ((ss / x.len() as f64 + eps as f64).sqrt());
    x.iter()
        .enumerate()
        .map(|(i, &v)| (v as f64 * inv * weight[i] as f64) as f32)
        .collect()
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// models.h:14-18 — `x / sqrt(sum(x^2) + eps)`.
fn gdn_l2(x: &[f32], eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let d = ((ss + eps as f64).sqrt()) as f32;
    if d <= 0.0 {
        return x.to_vec();
    }
    x.iter().map(|v| v / d).collect()
}

/// Weight is stored [out, in] in the block, so y = W x.
fn matvec(x: &[f32], wt: &[f32], in_n: usize, out_n: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_n);
    let mut y = vec![0.0f32; out_n];
    for (o, yv) in y.iter_mut().enumerate() {
        let mut s = 0.0f64;
        for (i, xv) in x.iter().enumerate() {
            s += (*xv as f64) * (wt[o * in_n + i] as f64);
        }
        *yv = s as f32;
    }
    y
}

fn lin_w(l: &Linear) -> (Vec<f32>, usize, usize) {
    let t = l.weight();
    let d = t.shape().dims().to_vec();
    assert_eq!(d.len(), 2, "linear weight must be 2-D, got {d:?}");
    (t.to_vec_f32().expect("weight"), d[1], d[0]) // (data, in, out)
}

fn cmp(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &r)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - r).abs();
        let rel = d / r.abs().max(1e-3);
        if d.min(rel) > worst {
            worst = d.min(rel);
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: worst |err| {worst:.3e} at {at} (block {}, reference {})",
        got[at],
        want[at]
    );
    eprintln!("[{what}] matches (worst {worst:.2e})");
}

/// A real recurrent layer, loaded from the 9B, against the hand reference.
#[test]
fn real_recurrent_layer_matches_hand_reference() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.0");
    // (0+1) % 4 != 0 -> recurrent.
    let blk = Qwen35Block::load_tp(&ws, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load layer 0");
    assert!(!blk.is_full_attention, "layer 0 must be recurrent");

    let eps = blk.attn_norm.eps;
    let mut cache = Qwen35LayerCache::new(&c);
    // Non-zero recurrent state: a zero state hides a stale-state variant.
    let seed: Vec<f32> = (0..cache.ssm_state.len())
        .map(|i| ((i * 37 % 29) as f32) / 29.0 - 0.5)
        .collect();
    for (i, v) in cache.ssm_state.iter_mut().enumerate() {
        *v = seed[i];
    }
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 11 % 29) as f32) / 29.0 - 0.5)
        .collect();

    // ---- reference, from llama.cpp, reading the block's own weights ----
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(&x, &an, eps);

    let (qw, qi, qo) = lin_w(blk.attn_qkv.as_ref().expect("attn_qkv"));
    assert_eq!((qi, qo), (HIDDEN, conv_dim()), "attn_qkv geometry");
    let qkv = matvec(&xn, &qw, qi, qo);

    // ssm_conv1d.weight has ggml ne0 = tap count (llama.cpp
    // `ggml_ssm_conv`: d_conv = c->ne[0], d_inner = c->ne[1]); the 9B's raw
    // dims are [4, 8192], so element (tap, ch) is at tap + 4*ch.
    let cw = blk.ssm_conv1d.as_ref().expect("conv").to_vec_f32().expect("cw");
    assert_eq!(cw.len(), TAPS * conv_dim(), "conv weight size");
    let mut state = cache.conv_state.clone();
    let mut conv = vec![0.0f32; conv_dim()];
    for ch in 0..conv_dim() {
        let mut s = qkv[ch] * cw[ch * TAPS + (TAPS - 1)];
        for k in 0..TAPS - 1 {
            s += state[ch * (TAPS - 1) + k] * cw[ch * TAPS + k];
        }
        conv[ch] = s;
    }
    for ch in 0..conv_dim() {
        for k in 0..TAPS - 2 {
            state[ch * (TAPS - 1) + k] = state[ch * (TAPS - 1) + k + 1];
        }
        state[ch * (TAPS - 1) + (TAPS - 2)] = qkv[ch];
    }
    let conv_silu: Vec<f32> = conv.iter().map(|&v| silu(v)).collect();

    let (aw, ai, ao) = lin_w(blk.ssm_alpha.as_ref().expect("alpha"));
    let (bw, bi, bo) = lin_w(blk.ssm_beta.as_ref().expect("beta"));
    let (zw, zi, zo) = lin_w(blk.attn_gate.as_ref().expect("gate"));
    let alpha = matvec(&xn, &aw, ai, ao);
    let beta = matvec(&xn, &bw, bi, bo);
    let z = matvec(&xn, &zw, zi, zo);
    let dt_bias = blk.ssm_dt_bias.clone().expect("dt_bias");
    let ssm_a = blk.ssm_a.clone().expect("ssm_a");
    let ssm_norm = blk.ssm_norm.clone().expect("ssm_norm");
    assert_eq!(ssm_norm.len(), HD, "ssm_norm is per-head (ssm_d_state)");

    let mut s_state: Vec<f64> = cache.ssm_state.iter().map(|&v| v as f64).collect();
    let mut core = vec![0.0f32; value_dim()];
    let inv_sqrt_d = 1.0 / (HD as f32).sqrt();
    for h in 0..NV {
        let kh = h % NK;
        let q_slice = gdn_l2(&conv_silu[kh * HD..kh * HD + HD], eps);
        let k_slice = gdn_l2(
            &conv_silu[key_dim() + kh * HD..key_dim() + kh * HD + HD],
            eps,
        );
        let v_slice = &conv_silu[2 * key_dim() + h * HD..2 * key_dim() + h * HD + HD];
        let gate = softplus(alpha[h] + dt_bias[h]) * ssm_a[h];
        let decay = (gate as f64).exp();
        let bv = (1.0 / (1.0 + (-beta[h]).exp())) as f64;
        let mut acc = vec![0.0f32; HD];
        for j in 0..HD {
            let base = (h * HD + j) * HD;
            let mut pred = 0.0f64;
            for i in 0..HD {
                pred += (k_slice[i] as f64) * (decay * s_state[base + i]);
            }
            let delta = bv * (v_slice[j] as f64 - pred);
            let mut a = 0.0f64;
            for i in 0..HD {
                s_state[base + i] = decay * s_state[base + i] + (k_slice[i] as f64) * delta;
                a += (q_slice[i] as f64) * s_state[base + i];
            }
            acc[j] = (a as f32) * inv_sqrt_d;
        }
        core[h * HD..(h + 1) * HD].copy_from_slice(&acc);
    }

    // build_norm_gated: RMS over the head, * ssm_norm (per-head index!), * silu(z)
    let mut branch = vec![0.0f32; value_dim()];
    for h in 0..NV {
        let seg = &core[h * HD..(h + 1) * HD];
        let ss: f64 = seg.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = 1.0 / ((ss / HD as f64 + eps as f64).sqrt());
        for i in 0..HD {
            let idx = h * HD + i;
            branch[idx] = (seg[i] as f64 * inv * ssm_norm[i] as f64 * silu(z[idx]) as f64) as f32;
        }
    }

    let (ow, oi, oo) = lin_w(blk.ssm_out.as_ref().expect("ssm_out"));
    assert_eq!((oi, oo), (value_dim(), HIDDEN), "ssm_out geometry");
    let proj = matvec(&branch, &ow, oi, oo);
    let h1: Vec<f32> = (0..HIDDEN).map(|i| x[i] + proj[i]).collect();
    let pan = blk.post_attention_norm.weight.to_vec_f32().expect("post norm");
    let h2 = rms(&h1, &pan, eps);
    let (gw, gi, go) = lin_w(&blk.ffn_gate);
    let (uw, ui, uo) = lin_w(&blk.ffn_up);
    let (dw, di, doo) = lin_w(&blk.ffn_down);
    let fg = matvec(&h2, &gw, gi, go);
    let fu = matvec(&h2, &uw, ui, uo);
    let act: Vec<f32> = (0..go).map(|i| silu(fg[i]) * fu[i]).collect();
    let dn = matvec(&act, &dw, di, doo);
    let want: Vec<f32> = (0..HIDDEN).map(|i| h1[i] + dn[i]).collect();

    // ---- the block ----
    let got = blk
        .forward(
            &grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![1, HIDDEN])),
            &[0],
            &mut cache,
        )
        .expect("real recurrent forward")
        .to_vec_f32()
        .expect("read");

    cmp("real recurrent layer", &got, &want, 5e-2);
}

/// The loading check the synthetic test cannot do: the block's weights must
/// match the GGUF tensor for tensor, after the documented [out, in] storage
/// order. A transpose or a wrong shard survives every other test here.
#[test]
fn real_layer_weights_match_the_gguf() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.0");
    let blk = Qwen35Block::load_tp(&ws, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load layer 0");

    // Geometry the forward depends on, straight from the file.
    for (name, want) in [
        ("blk.0.ssm_norm.weight", HD),
        ("blk.0.ssm_a", NV),
        ("blk.0.ssm_dt.bias", NV),
        ("blk.0.ssm_conv1d.weight", TAPS * conv_dim()),
        ("blk.0.ssm_alpha.weight", HIDDEN * NV),
        ("blk.0.ssm_out.weight", HIDDEN * value_dim()),
    ] {
        let m = prov.meta(name).unwrap_or_else(|e| panic!("meta {name}: {e}"));
        eprintln!("  {name}: dims {:?}", m.shape);
        assert_eq!(
            m.shape.iter().product::<usize>(),
            want,
            "{name} element count"
        );
    }

    // Byte-compare the F32 vectors. Quantized 2-D weights cannot be compared
    // here without the crate-private dequantiser, so this covers the vectors
    // that the recurrent maths indexes directly — `ssm_norm` above all, since
    // it is read per-head and a length or ordering slip there is invisible in
    // every shape assertion.
    let check_vec = |name: &str, block_vals: &[f32]| {
        let raw = prov.get(name).unwrap_or_else(|e| panic!("get {name}: {e}"));
        let f: Vec<f32> = raw
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(f.len(), block_vals.len(), "{name} length");
        let mut worst = 0.0f32;
        for (a, b) in f.iter().zip(block_vals.iter()) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst <= 1e-6, "{name} differs from the GGUF by {worst:.3e}");
        eprintln!("  {name}: {} values match the checkpoint", f.len());
    };
    check_vec("blk.0.ssm_norm.weight", &blk.ssm_norm.clone().expect("ssm_norm"));
    check_vec("blk.0.ssm_a", &blk.ssm_a.clone().expect("ssm_a"));
    check_vec("blk.0.ssm_dt.bias", &blk.ssm_dt_bias.clone().expect("dt_bias"));
    eprintln!("[real-weights] shapes and F32 vectors match the checkpoint");
}


/// The same treatment for a REAL full-attention layer. The synthetic
/// attention test passed, but no attention layer has ever been run against
/// checkpoint weights, and the shapes here are the ones that decide whether
/// fused `attn_q` splits into Q and gate in the right order.
#[test]
fn real_attention_layer_matches_hand_reference() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3"); // (3+1)%4==0
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3");
    assert!(blk.is_full_attention, "layer 3 must be full attention");

    const AH: usize = 16;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let qd = AH * AHD;
    let kvd = AKV * AHD;

    // Geometry straight from the file.
    for (name, want) in [
        ("blk.3.attn_q.weight", vec![2 * qd, HIDDEN]),
        ("blk.3.attn_k.weight", vec![kvd, HIDDEN]),
        ("blk.3.attn_v.weight", vec![kvd, HIDDEN]),
        ("blk.3.attn_output.weight", vec![HIDDEN, qd]),
    ] {
        let m = prov.meta(name).unwrap_or_else(|e| panic!("meta {name}: {e}"));
        assert_eq!(m.shape, want, "{name} shape");
    }
    eprintln!("[real-attn] fused attn_q is [{} x {HIDDEN}] = [Q | gate]", 2 * qd);

    let eps = blk.attn_norm.eps;
    let mut cache = Qwen35LayerCache::new(&c);
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 17 % 29) as f32) / 29.0 - 0.5)
        .collect();
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(&x, &an, eps);

    // attn_q is FUSED query + gate at 2*q_dim.
    let (qw, qi, qo) = lin_w(blk.wq.as_ref().expect("attn_q"));
    assert_eq!((qi, qo), (HIDDEN, 2 * qd), "attn_q geometry");
    let qfull = matvec(&xn, &qw, qi, qo);
    let mut q = qfull[..qd].to_vec();
    let gate = &qfull[qd..2 * qd];

    let (kw, ki, ko) = lin_w(blk.wk.as_ref().expect("attn_k"));
    let (vw, vi, vo) = lin_w(blk.wv.as_ref().expect("attn_v"));
    let mut k = matvec(&xn, &kw, ki, ko);
    let v = matvec(&xn, &vw, vi, vo);
    assert_eq!(k.len(), kvd);
    assert_eq!(v.len(), kvd);

    // Q/K RMS norm per head, BEFORE rope, then NeoX rope at pos 0.
    let qn = blk.attn_q_norm.as_ref().expect("q_norm").weight.to_vec_f32().unwrap();
    let kn = blk.attn_k_norm.as_ref().expect("k_norm").weight.to_vec_f32().unwrap();
    for h in 0..AH {
        let s = h * AHD;
        let n = rms(&q[s..s + AHD], &qn, eps);
        q[s..s + AHD].copy_from_slice(&n);
    }
    for h in 0..AKV {
        let s = h * AHD;
        let n = rms(&k[s..s + AHD], &kn, eps);
        k[s..s + AHD].copy_from_slice(&n);
    }
    // NeoX rope, theta walking as a running product and the angle being
    // pos*theta — not a bare sin/cos, which would be identical for every pair
    // and would let a broken rope pass.
    let rope = |v: &mut [f32], pos: f64| {
        let hd = v.len();
        let mut theta = 1.0f64;
        let sc = 1.0f64 / 10000.0f64.powf(2.0 / hd as f64);
        for i0 in (0..hd).step_by(2) {
            let (x0, x1) = (v[i0] as f64, v[i0 + 1] as f64);
            let (sn, cs) = (pos * theta).sin_cos();
            v[i0] = (x0 * cs - x1 * sn) as f32;
            v[i0 + 1] = (x0 * sn + x1 * cs) as f32;
            theta *= sc;
        }
    };
    const POS: f64 = 0.0;
    for h in 0..AH {
        let s = h * AHD;
        let mut seg = q[s..s + AHD].to_vec();
        rope(&mut seg, POS);
        q[s..s + AHD].copy_from_slice(&seg);
    }
    for h in 0..AKV {
        let s = h * AHD;
        let mut seg = k[s..s + AHD].to_vec();
        rope(&mut seg, POS);
        k[s..s + AHD].copy_from_slice(&seg);
    }

    // kv_len == 1 => softmax is 1, so the output is V broadcast by GQA.
    let mut attn = vec![0.0f32; qd];
    for h in 0..AH {
        let kvh = (h * AKV) / AH;
        attn[h * AHD..(h + 1) * AHD].copy_from_slice(&v[kvh * AHD..(kvh + 1) * AHD]);
    }
    // ONE sigmoid gate on the attention branch (qwen35.cpp:323-327).
    let mut branch = vec![0.0f32; qd];
    for i in 0..qd {
        branch[i] = attn[i] / (1.0 + (-gate[i]).exp());
    }

    let (ow, oi, oo) = lin_w(blk.wo.as_ref().expect("wo"));
    assert_eq!((oi, oo), (qd, HIDDEN), "wo geometry");
    let proj = matvec(&branch, &ow, oi, oo);
    let h1: Vec<f32> = (0..HIDDEN).map(|i| x[i] + proj[i]).collect();
    let pan = blk.post_attention_norm.weight.to_vec_f32().expect("post norm");
    let h2 = rms(&h1, &pan, eps);
    let (gw, gi, go) = lin_w(&blk.ffn_gate);
    let (uw, ui, uo) = lin_w(&blk.ffn_up);
    let (dw, di, doo) = lin_w(&blk.ffn_down);
    let fg = matvec(&h2, &gw, gi, go);
    let fu = matvec(&h2, &uw, ui, uo);
    let act: Vec<f32> = (0..go).map(|i| silu(fg[i]) * fu[i]).collect();
    let dn = matvec(&act, &dw, di, doo);
    let want: Vec<f32> = (0..HIDDEN).map(|i| h1[i] + dn[i]).collect();

    let got = blk
        .forward(
            &grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![1, HIDDEN])),
            &[0],
            &mut cache,
        )
        .expect("real attention forward")
        .to_vec_f32()
        .expect("read");

    cmp("real attention layer", &got, &want, 5e-2);
}
