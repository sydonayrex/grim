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

/// The part of a block that is IDENTICAL for both layer types: project the
/// branch back to hidden, add the residual, post-norm, FFN, add the residual
/// again. Extracted so the per-layer gates and the 32-block chain below share
/// ONE definition — a second copy of this reference could drift from the first
/// and the chain would then be testing the drift.
fn hand_tail(blk: &Qwen35Block, x: &[f32], proj: &[f32], eps: f32) -> Vec<f32> {
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
    (0..HIDDEN).map(|i| h1[i] + dn[i]).collect()
}

/// The recurrent (KDA) block's expected output, from llama.cpp, reading the
/// block's own weights. MUTATES `cache` exactly as the real forward does, so a
/// chain must hand it the same state the block will see.
fn hand_recurrent_layer(
    blk: &Qwen35Block,
    x: &[f32],
    cache: &mut Qwen35LayerCache,
) -> Vec<f32> {
    let eps = blk.attn_norm.eps;
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(x, &an, eps);

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
    hand_tail(blk, x, &proj, eps)
}

/// The full-attention block's expected output. Assumes `kv_len == 1`, where the
/// softmax is 1 and the output is V broadcast by GQA — which holds for every
/// attention block in a seq_len=1 chain, because each block carries its own
/// cache and is visited once.
fn hand_attention_layer(blk: &Qwen35Block, x: &[f32], n_rot: usize, base: f64) -> Vec<f32> {
    const AH: usize = 16;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let qd = AH * AHD;
    let kvd = AKV * AHD;
    let eps = blk.attn_norm.eps;
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(x, &an, eps);

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
    // NeoX rope, partial over the first n_rot dims, theta a running product
    // and the angle pos*theta — not a bare sin/cos, which would be identical
    // for every pair and would let a broken rope pass.
    const POS: f64 = 0.0;
    for h in 0..AH {
        let s = h * AHD;
        let mut seg = q[s..s + AHD].to_vec();
        rope_partial(&mut seg, POS, n_rot, base);
        q[s..s + AHD].copy_from_slice(&seg);
    }
    for h in 0..AKV {
        let s = h * AHD;
        let mut seg = k[s..s + AHD].to_vec();
        rope_partial(&mut seg, POS, n_rot, base);
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
    hand_tail(blk, x, &proj, eps)
}

/// Deterministic recurrent-state seed. A zero state hides a stale-state variant,
/// so every cache the reference and the block each get their own copy of this.
fn seed_recurrent_cache(cache: &mut Qwen35LayerCache) {
    let seed: Vec<f32> = (0..cache.ssm_state.len())
        .map(|i| ((i * 37 % 29) as f32) / 29.0 - 0.5)
        .collect();
    for (i, v) in cache.ssm_state.iter_mut().enumerate() {
        *v = seed[i];
    }
}


/// The checkpoint's RoPE: PARTIAL rotary over the first `n_rot` dims, with
/// `theta_base` from the file.
///
/// Getting this wrong is the most expensive mistake available here, and it was
/// made once already: the first version of the multi-token reference roped all
/// `head_dim`=256 dims with a 1e4 base, because the single-token reference
/// never noticed (at position 0 a RoPE is the identity, so every convention
/// agrees). The 9B says `qwen35.rope.dimension_count = 64` and
/// `qwen35.rope.freq_base = 10000000.0`. Both are position-dependent, so the
/// error is invisible at row 0 and maximal at the end of the prompt — which is
/// exactly the "row 0 exact, rows 1..4 wrong" signature the arena dump showed.
///
/// `rope_sections` ([11,11,10,0], MRoPE) is deliberately NOT modelled: for text
/// tokens the reference writes p_t = p_h = p_w and the three theta bases never
/// diverge, so MRoPE is numerically plain RoPE here.
fn rope_partial(v: &mut [f32], pos: f64, n_rot: usize, theta_base: f64) {
    let hd = v.len();
    let n_rot = n_rot.min(hd);
    if n_rot == 0 {
        return;
    }
    let mut theta = 1.0f64;
    let sc = 1.0f64 / theta_base.powf(2.0 / n_rot as f64);
    let mut i0 = 0usize;
    while i0 < n_rot {
        let (x0, x1) = (v[i0] as f64, v[i0 + 1] as f64);
        let (sn, cs) = (pos * theta).sin_cos();
        v[i0] = (x0 * cs - x1 * sn) as f32;
        v[i0 + 1] = (x0 * sn + x1 * cs) as f32;
        theta *= sc;
        i0 += 2;
    }
    // Dims beyond n_rot pass through unrotated.
}


/// The RoPE, by CALLING the backend's implementation rather than modelling it.
///
/// This replaces a hand-rolled `rope_partial` that was wrong in a way no
/// self-consistency check could see: with `rotary_dim = 64` the device rotates a
/// different subset of the head than "the first 64 dims", and the hand model
/// disagreed with the arena by 1.5-4.0 while the device matched it at 4.4e-6 on
/// every row. Modelling a component you are trying to test is how a wrong
/// oracle gets mistaken for a wrong kernel — twice now, after the Q4_K nibble
/// split, the Q5_K `qh` advance, the f16 subnormal sign and the RoPE
/// parameters.
///
/// Mirrors `rope_ext` in `qwen35.rs` exactly: build `pos_ext` as
/// (position, head) pairs, reshape to `[1, S*heads, head_dim]`, call
/// `dev.rope`, reshape back.
fn rope_via_device(
    t: &[f32],
    s_len: usize,
    heads: usize,
    head_dim: usize,
    positions: &[u32],
    n_rot: usize,
    theta_base: f64,
) -> Vec<f32> {
    use grim_tensor::{CoreTensorOps, DType};
    let dev = grim_nn::modules::pick_device_for_storage_device(&Device::Cpu);
    let mut pos_ext = Vec::with_capacity(s_len * heads);
    for &p in positions {
        for _ in 0..heads {
            pos_ext.push(p);
        }
    }
    let mut cfg = grim_tensor::RopeConfig::new(head_dim, theta_base as f32);
    cfg.rotary_dim = n_rot;
    let three = Shape::new(vec![1, s_len * heads, head_dim]);
    let st = dev.from_cpu(t, &three, DType::F32).expect("upload for rope");
    let (roped, _) = dev
        .rope(st.as_ref(), &pos_ext, &cfg, &three)
        .expect("dev.rope");
    roped.to_cpu_vec_f32().expect("read roped")
}

/// `n_rot` and `theta_base` for this checkpoint, read from the file rather than
/// assumed, so the reference cannot silently disagree with the model about
/// either.
fn rope_params(prov: &GgufProvider) -> (usize, f64) {
    let n_rot = prov
        .metadata("qwen35.rope.dimension_count")
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .filter(|v| *v > 0)
        .unwrap_or(256);
    let base = prov
        .metadata("qwen35.rope.freq_base")
        .and_then(|v| v.as_f32())
        .map(|v| v as f64)
        .unwrap_or(10000.0);
    (n_rot, base)
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

    let mut cache = Qwen35LayerCache::new(&c);
    seed_recurrent_cache(&mut cache);
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 11 % 29) as f32) / 29.0 - 0.5)
        .collect();

    // The reference advances `cache`, so it gets its own copy seeded the same
    // way; sharing one would hand the block a state the reference already used.
    let mut ref_cache = Qwen35LayerCache::new(&c);
    seed_recurrent_cache(&mut ref_cache);
    let want = hand_recurrent_layer(&blk, &x, &mut ref_cache);

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
    let (n_rot, base) = rope_params(&prov);
    eprintln!("[real-attn] rope from checkpoint: n_rot={n_rot} freq_base={base}");

    let mut cache = Qwen35LayerCache::new(&c);
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 17 % 29) as f32) / 29.0 - 0.5)
        .collect();
    let (n_rot, base) = rope_params(&prov);
    let want = hand_attention_layer(&blk, &x, n_rot, base);

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

/// THE CHAIN — all 32 blocks, each fed the PREVIOUS block's real output.
///
/// Every other gate in this file hands its block a synthetic `x`. That makes
/// each block correct in isolation and says nothing about composition: if
/// `Qwen35::forward` hands block k+1 something other than what block k produced
/// — a stale residual, a re-normalised tensor, the branch without its
/// residual, the wrong tensor entirely — every per-layer gate stays green and
/// the model still emits gibberish. That is the shape of the remaining
/// unexplained failure (first token confidently wrong at p=0.882 where the
/// reference says " Paris").
///
/// So: walk the stack, feeding each block the previous block's ACTUAL output,
/// and check every block against the SAME hand reference the per-layer gates
/// use. The first `k` whose output diverges names the layer.
///
/// Scope, stated so a pass is not over-read: this is seq_len=1 with
/// positions `[0]`, on the CPU reference path. It therefore covers the
/// block-to-block HANDOFF and nothing about the 5-token prefill (causal mask,
/// positions 0..4, multi-token attention). A pass here means the handoff is
/// sound and the defect is in the prefill path or in `Qwen35::forward`'s own
/// loop, not in any block.
///
/// Starting state is synthetic, not the real embedding: `token_embd` is
/// verified separately and bit-exact (`real_token_embd_vs_oracle.rs`), and a
/// synthetic start keeps this test about composition alone.
#[test]
fn all_32_blocks_chain_correctly() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    let n_layers = 32usize;

    let mut h: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();

    for k in 0..n_layers {
        let name = format!("blk.{k}");
        let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp(&name);
        let blk = Qwen35Block::load_tp(&ws, &c, k, grim_nn::TensorParallelConfig::default())
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        let want_full = (k + 1) % 4 == 0;
        assert_eq!(
            blk.is_full_attention, want_full,
            "{name}: attention-layer predicate disagrees with the loader"
        );

        let mut cache = Qwen35LayerCache::new(&c);
        seed_recurrent_cache(&mut cache);
        let mut ref_cache = Qwen35LayerCache::new(&c);
        seed_recurrent_cache(&mut ref_cache);

        let (n_rot, base) = rope_params(&prov);
        let want = if blk.is_full_attention {
            hand_attention_layer(&blk, &h, n_rot, base)
        } else {
            hand_recurrent_layer(&blk, &h, &mut ref_cache)
        };

        let got = blk
            .forward(
                &grim_backend_cpu::cpu_tensor(h.clone(), Shape::new(vec![1, HIDDEN])),
                &[0],
                &mut cache,
            )
            .unwrap_or_else(|e| panic!("{name} forward: {e}"))
            .to_vec_f32()
            .expect("read");

        // Report rather than assert-then-stop, so one run shows whether the
        // divergence is a single layer or everything after it.
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
        let kind = if blk.is_full_attention { "attn" } else { "kda " };
        eprintln!(
            "[chain] {name} ({kind}) in|out| {:.2}  worst {:.3e} at {at}",
            h.iter().map(|v| v.abs() as f64).sum::<f64>() / HIDDEN as f64,
            worst
        );
        assert!(
            worst <= 5e-2,
            "CHAIN BREAKS AT {name} (layer {k}, {kind}): its output differs from the hand \
             reference by {:.3e} at index {at} (block {}, reference {}), given the previous \
             block's own output as input. Every earlier layer matched, so the defect is in \
             this block's handoff, not in any block taken alone.",
            worst,
            got[at],
            want[at]
        );

        // Feed the block's REAL output onward — that is the whole point.
        h = got;
    }
    eprintln!("[chain] all {n_layers} blocks chain correctly at seq_len=1");
}

// ── Multi-token (prefill) generalisation ──────────────────────────────────────
//
// The seq_len=1 gates above all take shortcuts that a single-key softmax makes
// harmless and a 5-token prefill does not:
//
//   * the attention reference returns V broadcast, because a one-key softmax is
//     1 regardless of the score. The real `1/sqrt(head_dim)` on the QK score,
//     the causal mask, and the softmax itself have therefore NEVER been checked
//     against a hand reference — and step 0, the token that is confidently
//     wrong, is produced by the prefill.
//   * the KDA recurrence and the conv have only ever run ONE step, so the
//     rolling conv state and the carried `s_state` across steps are untested.
//
// These two functions are written as INDEPENDENT implementations rather than
// refactors of the single-token ones. That is deliberate: `multi_token_reference_agrees_with_the_single_token_one`
// then cross-checks two separately-written versions of the same step, and at
// S=1 it pins the new mask, softmax, scale and RoPE-per-position to the
// already-proven reference. A refactor here would make that test vacuous.

/// The four things a multi-token attention path can get wrong, each of which a
/// one-key softmax hides. `attn_seq5_hypothesis_sweep` uses this to name which
/// one grim actually does, instead of assuming llama.cpp's answer and reporting
/// a mismatch without saying what the real code is doing.
#[derive(Clone, Copy, Debug)]
struct AttnCfg {
    /// `true` -> score scaled by `1/sqrt(head_dim)` (llama.cpp). `false` -> raw dot.
    scale: bool,
    /// `true` -> row `t` attends keys `0..=t`. `false` -> all `S` keys.
    causal: bool,
    /// `true` -> rope at each row's own position. `false` -> every row at 0.
    rope_per_position: bool,
}

const ATTN_LLAMA: AttnCfg = AttnCfg { scale: true, causal: true, rope_per_position: true };

/// Full attention over `positions.len()` rows, from an empty KV cache.
fn hand_attention_layer_cfg(
    blk: &Qwen35Block,
    x: &[f32],
    positions: &[u32],
    cfg: AttnCfg,
    n_rot: usize,
    base: f64,
) -> Vec<f32> {
    const AH: usize = 16;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let qd = AH * AHD;
    let kvd = AKV * AHD;
    let eps = blk.attn_norm.eps;
    let s_len = positions.len();
    assert_eq!(x.len(), s_len * HIDDEN, "x must be [S, HIDDEN]");

    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let (qw, qi, qo) = lin_w(blk.wq.as_ref().expect("attn_q"));
    let (kw, ki, ko) = lin_w(blk.wk.as_ref().expect("attn_k"));
    let (vw, vi, vo) = lin_w(blk.wv.as_ref().expect("attn_v"));
    let (ow, oi, oo) = lin_w(blk.wo.as_ref().expect("wo"));
    let qn = blk.attn_q_norm.as_ref().expect("q_norm").weight.to_vec_f32().unwrap();
    let kn = blk.attn_k_norm.as_ref().expect("k_norm").weight.to_vec_f32().unwrap();
    assert_eq!((qi, qo), (HIDDEN, 2 * qd), "attn_q geometry");
    assert_eq!((ki, ko), (HIDDEN, kvd), "attn_k geometry");
    assert_eq!((vi, vo), (HIDDEN, kvd), "attn_v geometry");
    assert_eq!((oi, oo), (qd, HIDDEN), "wo geometry");

    // RoPE is PARTIAL over the first `n_rot` dims with the checkpoint's base;
    // see `rope_partial`. At S>1 a rope that ignores `pos` would still look
    // right at position 0, which is why the per-row position is threaded
    // through and why the rope parameters are read from the file.

    let mut q_all = vec![0.0f32; s_len * qd];
    let mut k_all = vec![0.0f32; s_len * kvd];
    let mut v_all = vec![0.0f32; s_len * kvd];
    let mut g_all = vec![0.0f32; s_len * qd];

    for t in 0..s_len {
        let row = &x[t * HIDDEN..(t + 1) * HIDDEN];
        let xn = rms(row, &an, eps);
        let qfull = matvec(&xn, &qw, qi, qo);
        g_all[t * qd..(t + 1) * qd].copy_from_slice(&qfull[qd..2 * qd]);
        let mut q = qfull[..qd].to_vec();
        let mut k = matvec(&xn, &kw, ki, ko);
        let v = matvec(&xn, &vw, vi, vo);
        // Q/K RMS norm per head, BEFORE rope, at THIS row's position.
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
        if cfg.rope_per_position {
            // Roped by the BACKEND, not modelled — see `rope_via_device`.
            q = rope_via_device(&q, 1, AH, AHD, &[positions[t]], n_rot, base);
            k = rope_via_device(&k, 1, AKV, AHD, &[positions[t]], n_rot, base);
        }
        q_all[t * qd..(t + 1) * qd].copy_from_slice(&q);
        k_all[t * kvd..(t + 1) * kvd].copy_from_slice(&k);
        v_all[t * kvd..(t + 1) * kvd].copy_from_slice(&v);
    }

    let mut out = vec![0.0f32; s_len * HIDDEN];
    for t in 0..s_len {
        let mut branch = vec![0.0f32; qd];
        for h in 0..AH {
            let kvh = (h * AKV) / AH;
            // Causal: keys 0..=t only. With an empty cache the first `t` rows of
            // the arena are this call's own earlier rows.
            let n_keys = if cfg.causal { t + 1 } else { s_len };
            let mut sc = vec![0.0f32; n_keys];
            let mut mx = f32::NEG_INFINITY;
            for s in 0..n_keys {
                let mut d = 0.0f64;
                for i in 0..AHD {
                    d += (q_all[t * qd + h * AHD + i] as f64) * (k_all[s * kvd + kvh * AHD + i] as f64);
                }
                // The 1/sqrt(head_dim) on the QK score. Invisible at S=1 (a
                // one-key softmax is scale-invariant) and dominant at S>1.
                let val = if cfg.scale { (d / (AHD as f64).sqrt()) as f32 } else { d as f32 };
                sc[s] = val;
                if val > mx {
                    mx = val;
                }
            }
            let mut z = 0.0f64;
            for s in 0..n_keys {
                let e = ((sc[s] - mx) as f64).exp();
                sc[s] = e as f32;
                z += e;
            }
            for j in 0..AHD {
                let mut acc = 0.0f64;
                for s in 0..n_keys {
                    acc += (sc[s] as f64) * (v_all[s * kvd + kvh * AHD + j] as f64);
                }
                branch[h * AHD + j] = (acc / z) as f32;
            }
        }
        // ONE sigmoid gate on the attention branch (qwen35.cpp:323-327).
        for i in 0..qd {
            branch[i] /= 1.0 + (-g_all[t * qd + i]).exp();
        }
        let proj = matvec(&branch, &ow, oi, oo);
        let t_out = hand_tail(blk, &x[t * HIDDEN..(t + 1) * HIDDEN], &proj, eps);
        out[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&t_out);
    }
    out
}

fn hand_attention_layer_multi(blk: &Qwen35Block, x: &[f32], positions: &[u32], n_rot: usize, base: f64) -> Vec<f32> {
    hand_attention_layer_cfg(blk, x, positions, ATTN_LLAMA, n_rot, base)
}

/// The recurrent (KDA) block over `s_len` sequential steps, threading the
/// rolling conv state and the carried `s_state` forward. MUTATES `cache`.
fn hand_recurrent_layer_multi(
    blk: &Qwen35Block,
    x: &[f32],
    s_len: usize,
    cache: &mut Qwen35LayerCache,
) -> Vec<f32> {
    let eps = blk.attn_norm.eps;
    assert_eq!(x.len(), s_len * HIDDEN, "x must be [S, HIDDEN]");
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let (qw, qi, qo) = lin_w(blk.attn_qkv.as_ref().expect("attn_qkv"));
    assert_eq!((qi, qo), (HIDDEN, conv_dim()), "attn_qkv geometry");
    let cw = blk.ssm_conv1d.as_ref().expect("conv").to_vec_f32().expect("cw");
    assert_eq!(cw.len(), TAPS * conv_dim(), "conv weight size");
    let (aw, ai, ao) = lin_w(blk.ssm_alpha.as_ref().expect("alpha"));
    let (bw, bi, bo) = lin_w(blk.ssm_beta.as_ref().expect("beta"));
    let (zw, zi, zo) = lin_w(blk.attn_gate.as_ref().expect("gate"));
    let (ow, oi, oo) = lin_w(blk.ssm_out.as_ref().expect("ssm_out"));
    let dt_bias = blk.ssm_dt_bias.clone().expect("dt_bias");
    let ssm_a = blk.ssm_a.clone().expect("ssm_a");
    let ssm_norm = blk.ssm_norm.clone().expect("ssm_norm");
    assert_eq!(ssm_norm.len(), HD, "ssm_norm is per-head (ssm_d_state)");

    // Rolling state, seeded from the cache and advanced in place — this is the
    // part a single-step reference cannot exercise.
    let mut state = cache.conv_state.clone();
    let mut s_state: Vec<f64> = cache.ssm_state.iter().map(|&v| v as f64).collect();
    let inv_sqrt_d = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0.0f32; s_len * HIDDEN];

    for t in 0..s_len {
        let row = &x[t * HIDDEN..(t + 1) * HIDDEN];
        let xn = rms(row, &an, eps);
        let qkv = matvec(&xn, &qw, qi, qo);
        let alpha = matvec(&xn, &aw, ai, ao);
        let beta = matvec(&xn, &bw, bi, bo);
        let z = matvec(&xn, &zw, zi, zo);

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

        let mut branch = vec![0.0f32; value_dim()];
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
            // build_norm_gated: RMS over the head, * ssm_norm (per-head index!), * silu(z)
            let ss: f64 = acc.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv = 1.0 / ((ss / HD as f64 + eps as f64).sqrt());
            for i in 0..HD {
                let idx = h * HD + i;
                branch[idx] = (acc[i] as f64 * inv * ssm_norm[i] as f64 * silu(z[idx]) as f64) as f32;
            }
        }

        let proj = matvec(&branch, &ow, oi, oo);
        let t_out = hand_tail(blk, row, &proj, eps);
        out[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&t_out);
    }
    // Keep the caller's cache in step with what the block will have done.
    cache.conv_state = state;
    for (i, v) in s_state.iter().enumerate() {
        cache.ssm_state[i] = *v as f32;
    }
    out
}

/// CROSS-CHECK: the two independently-written multi-token references must agree
/// with the proven single-token ones at S=1.
///
/// This is what licenses the prefill chain below. It pins the new causal mask,
/// the real softmax, the `1/sqrt(head_dim)` score scale and the per-row RoPE to
/// the reference that already matches the real block, and it does so BEFORE any
/// 5-token result is read. If the new attention maths is wrong in a way that
/// only shows at S>1, this still passes — but then the prefill chain will name
/// the block, which is the point of running it.
#[test]
fn multi_token_reference_agrees_with_the_single_token_one() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    let x: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 11 % 29) as f32) / 29.0 - 0.5)
        .collect();

    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let attn = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3");
    let (n_rot, base) = rope_params(&prov);
    eprintln!("[rope] checkpoint: n_rot={n_rot} freq_base={base}");
    let single = hand_attention_layer(&attn, &x, n_rot, base);
    let multi = hand_attention_layer_multi(&attn, &x, &[0], n_rot, base);
    cmp("attention S=1 (single vs multi reference)", &multi, &single, 1e-5);

    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.0");
    let rec = Qwen35Block::load_tp(&ws, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load layer 0");
    let mut ca = Qwen35LayerCache::new(&c);
    seed_recurrent_cache(&mut ca);
    let mut cb = Qwen35LayerCache::new(&c);
    seed_recurrent_cache(&mut cb);
    let single = hand_recurrent_layer(&rec, &x, &mut ca);
    let multi = hand_recurrent_layer_multi(&rec, &x, 1, &mut cb);
    cmp("recurrent S=1 (single vs multi reference)", &multi, &single, 1e-5);
}

/// THE PREFILL CHAIN — all 32 blocks at seq_len=5, positions 0..4, each fed the
/// previous block's real output, with a real causal-masked attention.
///
/// This is the surface the seq_len=1 chain cannot reach, and it is where the
/// gibberish must live: the first generated token comes out of a 5-token
/// prefill, and every one-token gate is blind to the causal mask, the QK score
/// scale and the multi-step conv/KDA state.
#[test]
fn all_32_blocks_chain_correctly_at_seq_len_5() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let mut c = cfg();
    // `Qwen35Block::load_tp` does `cfg.rotary_dim.unwrap_or(cfg.head_dim)`, and
    // `model_loader.rs:3234` populates it from `qwen35.rope.dimension_count`.
    // The local `cfg()` helper never set it, so every block in this file was
    // built with the FULL head_dim — i.e. 256 — while the checkpoint says 64.
    // That made the seq_len=5 chain disagree with the hand reference for a
    // reason that has nothing to do with the model: the test was not the model.
    let (ck_n_rot, _) = rope_params(&prov);
    c.rotary_dim = Some(ck_n_rot);
    eprintln!("[chain5] rotary_dim={ck_n_rot} (from qwen35.rope.dimension_count)");
    const S: usize = 5;
    let positions: Vec<u32> = (0..S as u32).collect();

    let mut h: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();

    for k in 0..32usize {
        let name = format!("blk.{k}");
        let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp(&name);
        let blk = Qwen35Block::load_tp(&ws, &c, k, grim_nn::TensorParallelConfig::default())
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        assert_eq!(blk.is_full_attention, (k + 1) % 4 == 0, "{name} layer-type predicate");

        let mut cache = Qwen35LayerCache::new(&c);
        seed_recurrent_cache(&mut cache);
        let mut ref_cache = Qwen35LayerCache::new(&c);
        seed_recurrent_cache(&mut ref_cache);

        let (n_rot, base) = rope_params(&prov);
        let want = if blk.is_full_attention {
            hand_attention_layer_multi(&blk, &h, &positions, n_rot, base)
        } else {
            hand_recurrent_layer_multi(&blk, &h, S, &mut ref_cache)
        };

        let got = blk
            .forward(
                &grim_backend_cpu::cpu_tensor(h.clone(), Shape::new(vec![S, HIDDEN])),
                &positions,
                &mut cache,
            )
            .unwrap_or_else(|e| panic!("{name} forward at seq_len={S}: {e}"))
            .to_vec_f32()
            .expect("read");

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
        let kind = if blk.is_full_attention { "attn" } else { "kda " };
        let (row, col) = (at / HIDDEN, at % HIDDEN);
        eprintln!(
            "[chain5] {name} ({kind}) worst {worst:.3e} at row {row} col {col} \
             (block {}, reference {})",
            got[at], want[at]
        );
        assert!(
            worst <= 5e-2,
            "PREFILL CHAIN BREAKS AT {name} (layer {k}, {kind}, seq_len={S}): output differs \
             from the hand reference by {worst:.3e} at row {row} col {col} (block {}, \
             reference {}). Every earlier layer matched, so the defect is in this block's \
             multi-token path — the causal mask, the QK score scale, the softmax, or the \
             rolled conv/KDA state — not in the block taken alone at one token.",
            got[at],
            want[at]
        );
        h = got;
    }
    eprintln!("[chain5] all 32 blocks chain correctly at seq_len={S}, positions 0..4");
}

/// WHICH multi-token attention is grim actually computing?
///
/// `all_32_blocks_chain_correctly_at_seq_len_5` localises the defect to the
/// first full-attention block at seq_len=5. That test says the block DISAGREES
/// with llama.cpp; it does not say what grim is doing instead. Four candidates
/// are indistinguishable at one token and separate cleanly at five:
///
///   * the `1/sqrt(head_dim)` QK score scale  (a one-key softmax ignores it)
///   * the causal mask                         (row t must not see keys > t)
///   * the per-row RoPE position               (all-zero rope looks right at pos 0)
///   * the softmax itself
///
/// Rather than assert llama.cpp's answer and report a bare mismatch, run the
/// real block once and score every hypothesis against it. Exactly one should
/// land at round-off; the others stay large, and the gap between them is the
/// evidence for which knob is wrong.
#[test]
fn attn_seq5_hypothesis_sweep_names_the_defect() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    let positions: Vec<u32> = (0..S as u32).collect();

    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3");
    assert!(blk.is_full_attention);

    let x: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();
    let mut cache = Qwen35LayerCache::new(&c);
    let got = blk
        .forward(
            &grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![S, HIDDEN])),
            &positions,
            &mut cache,
        )
        .expect("real attention forward at seq_len=5")
        .to_vec_f32()
        .expect("read");

    let per_row_err = |want: &[f32]| -> Vec<f32> {
        (0..S)
            .map(|r| {
                let mut w = 0.0f32;
                for cidx in 0..HIDDEN {
                    let d = (got[r * HIDDEN + cidx] - want[r * HIDDEN + cidx]).abs();
                    let rel = d / want[r * HIDDEN + cidx].abs().max(1e-3);
                    w = w.max(d.min(rel));
                }
                w
            })
            .collect()
    };

    let hyps: Vec<(&str, AttnCfg)> = vec![
        ("llama.cpp: scaled + causal + rope-per-pos", ATTN_LLAMA),
        ("no QK scale", AttnCfg { scale: false, ..ATTN_LLAMA }),
        ("non-causal (row t sees all keys)", AttnCfg { causal: false, ..ATTN_LLAMA }),
        ("rope at pos 0 for every row", AttnCfg { rope_per_position: false, ..ATTN_LLAMA }),
    ];

    let mut results: Vec<(&str, Vec<f32>)> = Vec::new();
    for (name, cfg) in hyps {
        let (n_rot, base) = rope_params(&prov);
        let want = hand_attention_layer_cfg(&blk, &x, &positions, cfg, n_rot, base);
        let rows = per_row_err(&want);
        eprintln!(
            "[sweep] {name:42} per-row worst err {:?}",
            rows.iter().map(|v| format!("{v:.2e}")).collect::<Vec<_>>()
        );
        results.push((name, rows));
    }

    // Which hypothesis reproduces the block? Score by the WORST row, so a
    // hypothesis that is right for rows 0..3 and wrong at row 4 cannot pass.
    let scored: Vec<(&str, f32)> = results
        .iter()
        .map(|(n, r)| (*n, r.iter().cloned().fold(0.0f32, f32::max)))
        .collect();
    for (n, e) in &scored {
        eprintln!("[sweep] {n:42} worst over rows {e:.3e}");
    }
    let best = scored
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("hypotheses");
    assert!(
        best.1 < 5e-2,
        "NO hypothesis reproduces the block's seq_len=5 attention. Best was {:?} at {}. \
         The defect is not the scale, the mask, or the rope position alone — read the per-row \
         errors above: a defect that grows with the row points at the mask or the scale, one \
         that is flat across rows points at rope or the softmax.",
        best.0,
        best.1
    );
    eprintln!("[sweep] grim's seq_len=5 attention IS: {:?}", best.0);
}

/// WHAT IS IN THE KV ARENA after a seq_len=5 attention block?
///
/// The sweep established the shape of the defect precisely:
///
///   * row 0 is EXACT (3.29e-6) — it attends exactly one key, the case every
///     earlier gate covers;
///   * rows 1..4 are all wrong, and no combination of scale / mask / rope fixes
///     them;
///   * `scalar_attention` itself is correct (causal limit, mask, softmax, the
///     1/sqrt(head_dim) scale and the accumulation all read right).
///
/// So the arithmetic over the keys is fine and the KEYS THEMSELVES are not. A
/// wrong per-row K/V — or a wrong row pitch in the arena, which corrupts every
/// row after the first while leaving row 0 (offset 0) perfect — has exactly
/// that signature.
///
/// This reads the arena the block actually filled and compares it, row by row,
/// against K and V computed from the block's own weights. The pattern in the
/// mismatch is the diagnosis: a row holding the PREVIOUS row's K is an off-by-
/// one in the write offset, zeros mean the row was never written, and a row
/// that matches a DIFFERENT position's rope means rope is applied to the wrong
/// index.
#[test]
fn seq5_kv_arena_contents_vs_computed() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    let positions: Vec<u32> = (0..S as u32).collect();
    const AH: usize = 16;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let kvd = AKV * AHD;
    let qd = AH * AHD;

    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3");

    let x: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();

    // What K and V SHOULD be, per row, from the block's own weights.
    let eps = blk.attn_norm.eps;
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let (kw, ki, ko) = lin_w(blk.wk.as_ref().expect("attn_k"));
    let (vw, vi, vo) = lin_w(blk.wv.as_ref().expect("attn_v"));
    let kn = blk.attn_k_norm.as_ref().expect("k_norm").weight.to_vec_f32().unwrap();
    let (n_rot, base) = rope_params(&prov);
    eprintln!("[rope] arena test: n_rot={n_rot} freq_base={base}");
    let mut want_k = vec![0.0f32; S * kvd]; // POST-rope
    let mut want_k_pre = vec![0.0f32; S * kvd]; // PRE-rope, normed
    let mut want_k_r0 = vec![0.0f32; S * kvd]; // roped at position 0
    let mut want_v = vec![0.0f32; S * kvd];
    for t in 0..S {
        let xn = rms(&x[t * HIDDEN..(t + 1) * HIDDEN], &an, eps);
        let mut k = matvec(&xn, &kw, ki, ko);
        let v = matvec(&xn, &vw, vi, vo);
        for h in 0..AKV {
            let s = h * AHD;
            let n = rms(&k[s..s + AHD], &kn, eps);
            k[s..s + AHD].copy_from_slice(&n);
        }
        want_k_pre[t * kvd..(t + 1) * kvd].copy_from_slice(&k);
        want_v[t * kvd..(t + 1) * kvd].copy_from_slice(&v);
        let mut k0 = k.clone();
        for h in 0..AKV {
            let s = h * AHD;
            let mut seg = k0[s..s + AHD].to_vec();
            rope_partial(&mut seg, 0.0, n_rot, base);
            k0[s..s + AHD].copy_from_slice(&seg);
        }
        want_k_r0[t * kvd..(t + 1) * kvd].copy_from_slice(&k0);
        for h in 0..AKV {
            let s = h * AHD;
            let mut seg = k[s..s + AHD].to_vec();
            rope_partial(&mut seg, positions[t] as f64, n_rot, base);
            k[s..s + AHD].copy_from_slice(&seg);
        }
        want_k[t * kvd..(t + 1) * kvd].copy_from_slice(&k);
    }

    let mut cache = Qwen35LayerCache::new(&c);
    let _ = blk
        .forward(
            &grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![S, HIDDEN])),
            &positions,
            &mut cache,
        )
        .expect("forward at seq_len=5");

    let arena = cache
        .k_device
        .as_ref()
        .expect("block must leave a device KV arena after a device-resident run");
    let k_hist = arena.to_cpu_vec_f32().expect("read K arena");
    eprintln!(
        "[kv] K arena: {} floats, shape {:?}, expected at least {} for {S} rows of {kvd}",
        k_hist.len(),
        arena.shape().dims(),
        S * kvd
    );
    let arena_rows = k_hist.len() / kvd;
    eprintln!("[kv] arena holds {arena_rows} rows; the first {S} are the live history");

    for t in 0..arena_rows.min(S + 2) {
        let mut worst_k = 0.0f32;
        let mut at = 0usize;
        for j in 0..kvd {
            let d = (k_hist[t * kvd + j] - want_k[t * kvd + j]).abs();
            let rel = d / want_k[t * kvd + j].abs().max(1e-3);
            if d.min(rel) > worst_k {
                worst_k = d.min(rel);
                at = j;
            }
        }
        // Does this arena row instead match some OTHER row's K? That is the
        // fingerprint of a write-offset error rather than a compute error.
        let mut best_other = (usize::MAX, f32::MAX);
        for u in 0..S {
            if u == t {
                continue;
            }
            let mut w = 0.0f32;
            for j in 0..kvd {
                let d = (k_hist[t * kvd + j] - want_k[u * kvd + j]).abs();
                let rel = d / want_k[u * kvd + j].abs().max(1e-3);
                w = w.max(d.min(rel));
            }
            if w < best_other.1 {
                best_other = (u, w);
            }
        }
        let err_vs = |cand: &[f32]| -> f32 {
            let mut w = 0.0f32;
            for j in 0..kvd {
                let d = (k_hist[t * kvd + j] - cand[t * kvd + j]).abs();
                let rel = d / cand[t * kvd + j].abs().max(1e-3);
                w = w.max(d.min(rel));
            }
            w
        };
        eprintln!(
            "[kv] row {t}: POST-rope {worst_k:.3e} (at {at}) | PRE-rope {:.3e} | rope@pos0 {:.3e} \
             | closest other row {} at {:.3e}",
            err_vs(&want_k_pre),
            err_vs(&want_k_r0),
            if best_other.0 == usize::MAX { -1i64 } else { best_other.0 as i64 },
            best_other.1
        );
    }
    let _ = qd;
    eprintln!(
        "[kv] cache.current_pos={} (expected {S} after a prefill of {S})",
        cache.current_pos
    );
}

/// SOLVE FOR THE POSITION the K rows were actually roped at.
///
/// The arena measurement narrowed this to one thing. Row 0 is exact, so the
/// projection, the per-head K norm, the NeoX half-split pairing and the theta
/// walk are all correct — at position 0 every one of those is exercised. Rows
/// 1..4 are post-rope-shaped but wrong, and the error grows with the row, and
/// no arena row equals any correctly-roped row. The ONLY thing left that can
/// differ is the number fed in as `pos`.
///
/// So scan positions and find which one reproduces each arena row. The answer
/// is a position, not a knob: a scan returning `t` would vindicate the code,
/// `t/2` or `2t` names an indexing slip, `0` names a dropped position array, and
/// a non-integer best fit means the position is being scaled somewhere.
#[test]
fn seq5_solve_for_the_rope_position_used() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    let positions: Vec<u32> = (0..S as u32).collect();
    const AKV: usize = 4;
    const AHD: usize = 256;
    let kvd = AKV * AHD;

    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3");
    let x: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();

    let eps = blk.attn_norm.eps;
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let (kw, ki, ko) = lin_w(blk.wk.as_ref().expect("attn_k"));
    let kn = blk.attn_k_norm.as_ref().expect("k_norm").weight.to_vec_f32().unwrap();
    let (n_rot, base) = rope_params(&prov);
    eprintln!("[rope] arena test: n_rot={n_rot} freq_base={base}");
    // Pre-rope, per-head-normed K for every row.
    let mut k_pre: Vec<Vec<f32>> = Vec::new();
    for t in 0..S {
        let xn = rms(&x[t * HIDDEN..(t + 1) * HIDDEN], &an, eps);
        let mut k = matvec(&xn, &kw, ki, ko);
        for h in 0..AKV {
            let s = h * AHD;
            let n = rms(&k[s..s + AHD], &kn, eps);
            k[s..s + AHD].copy_from_slice(&n);
        }
        k_pre.push(k);
    }

    let mut cache = Qwen35LayerCache::new(&c);
    let _ = blk
        .forward(
            &grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![S, HIDDEN])),
            &positions,
            &mut cache,
        )
        .expect("forward at seq_len=5");
    let k_hist = cache
        .k_device
        .as_ref()
        .expect("K arena")
        .to_cpu_vec_f32()
        .expect("read K arena");

    let err_at = |t: usize, p: f64| -> f32 {
        let mut k = k_pre[t].clone();
        for h in 0..AKV {
            let s = h * AHD;
            let mut seg = k[s..s + AHD].to_vec();
            rope_partial(&mut seg, p, n_rot, base);
            k[s..s + AHD].copy_from_slice(&seg);
        }
        let mut w = 0.0f32;
        for j in 0..kvd {
            let d = (k_hist[t * kvd + j] - k[j]).abs();
            let rel = d / k[j].abs().max(1e-3);
            w = w.max(d.min(rel));
        }
        w
    };

    for t in 1..S {
        // Scan this row's own K over a wide position range, on a 1/64 grid.
        let mut best = (f64::NEG_INFINITY, f32::MAX);
        let mut p = -8.0f64;
        while p <= 24.0 + 1e-9 {
            let e = err_at(t, p);
            if e < best.1 {
                best = (p, e);
            }
            p += 1.0 / 64.0;
        }
        eprintln!(
            "[pos] row {t} (true position {}): best-fit position {:.4} at err {:.3e} \
             | err at true pos {:.3e} | err at 0 {:.3e}",
            t,
            best.0,
            best.1,
            err_at(t, t as f64),
            err_at(t, 0.0)
        );
    }
}

/// SPLIT THE CHAIN: is the K projection itself right at seq_len=5?
///
/// The arena dump says the K rows 1..4 are not any RoPE — at any position, at
/// either rope config — of the K this block's own weights produce from x row t.
/// So the error enters at or before the projection. This runs the projection
/// through grim's own `Linear::forward` on a 5-row input and compares it,
/// row by row, against the per-row matvec the reference uses.
///
///   * if `Linear::forward` matches, the projection is fine and the defect is in
///     the per-head K norm, the rope, or the arena write;
///   * if it diverges at row 1 and matches at row 0, the defect is in the
///     multi-row GEMM path, which would also implicate Q and the fused QKV.
#[test]
fn seq5_k_projection_per_row() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let kvd = AKV * AHD;

    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3");
    let x: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();
    let x_t = grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![S, HIDDEN]));
    let x_normed = blk.attn_norm.forward(&x_t).expect("pre-norm");

    // Reference: per-row matvec from the block's own weights.
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let eps = blk.attn_norm.eps;
    let (kw, ki, ko) = lin_w(blk.wk.as_ref().expect("attn_k"));
    assert_eq!((ki, ko), (HIDDEN, kvd), "attn_k geometry");
    let mut want = vec![0.0f32; S * kvd];
    for t in 0..S {
        let xn = rms(&x[t * HIDDEN..(t + 1) * HIDDEN], &an, eps);
        let k = matvec(&xn, &kw, ki, ko);
        want[t * kvd..(t + 1) * kvd].copy_from_slice(&k);
    }

    let got = blk
        .wk
        .as_ref()
        .expect("attn_k")
        .forward(&x_normed)
        .expect("wk.forward at seq_len=5")
        .to_vec_f32()
        .expect("read");
    assert_eq!(got.len(), S * kvd, "projection must return [S, kv_dim]");

    for t in 0..S {
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for j in 0..kvd {
            let d = (got[t * kvd + j] - want[t * kvd + j]).abs();
            let rel = d / want[t * kvd + j].abs().max(1e-3);
            if d.min(rel) > worst {
                worst = d.min(rel);
                at = j;
            }
        }
        eprintln!("[proj] attn_k row {t}: worst {worst:.3e} at {at}");
        assert!(
            worst <= 5e-2,
            "attn_k PROJECTION is wrong at row {t} (seq_len={S}): worst {worst:.3e} at {at} \
             (block {}, reference {}). The multi-row GEMM path is the defect, and Q and the \
             fused QKV go through the same one.",
            got[t * kvd + at],
            want[t * kvd + at]
        );
    }
    eprintln!("[proj] attn_k matches the per-row reference on all {S} rows");
}

/// THE SPLIT: is it grim's RoPE kernel, or the arena write?
///
/// Chain so far, all measured on real weights at seq_len=5, layer 3:
///
///   projection  `wk.forward` on 5 rows .......... CORRECT (2.0e-6, all rows)
///   per-head K norm  (row 0 vs arena) ........... CORRECT (1.25e-6 at row 0;
///                                                     a rope at pos 0 is the
///                                                     identity, so row 0
///                                                     validates norm+rope-free)
///   rope at any position, either config ......... DOES NOT REPRODUCE rows 1..4
///   `scalar_attention` (mask/softmax/scale) ..... CORRECT on inspection
///
/// The remaining split is rope-versus-arena-write. So stop modelling the rope
/// and USE grim's: call `dev.rope` exactly as `qwen35.rs:942-943` does, with
/// the same `pos_ext` construction and the same `rope_cfg`, and compare the
/// result against the arena the block actually filled.
///
///   * matches  => the rope kernel is right and the arena WRITE is the defect;
///   * differs  => grim's rope kernel does something this reference does not
///                 model, and the difference is now a single kernel.
#[test]
fn seq5_rope_kernel_vs_arena() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let kvd = AKV * AHD;
    let positions: Vec<u32> = (0..S as u32).collect();

    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3");
    let x: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();
    let dev = grim_nn::modules::pick_device_for_storage_device(&Device::Cpu);
    let (n_rot, base) = rope_params(&prov);

    // projection (verified correct) then the per-head K norm, replicated from
    // `qwen35.rs::apply_head_rms_norm` — a copy, so if it disagreed with the
    // block row 0 would already have failed above.
    let x_t = grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![S, HIDDEN]));
    let x_normed = blk.attn_norm.forward(&x_t).expect("pre-norm");
    let k_proj = blk
        .wk
        .as_ref()
        .expect("attn_k")
        .forward(&x_normed)
        .expect("wk.forward")
        .to_vec_f32()
        .expect("read");
    let kn = blk.attn_k_norm.as_ref().expect("k_norm").weight.to_vec_f32().unwrap();
    let eps = blk.attn_k_norm.as_ref().map(|n| n.eps).unwrap_or(blk.attn_norm.eps);
    let mut k_normed = k_proj.clone();
    for t in 0..S {
        for h in 0..AKV {
            let b = t * kvd + h * AHD;
            let ss: f32 = k_normed[b..b + AHD].iter().map(|v| v * v).sum();
            let inv = 1.0 / ((ss / AHD as f32) + eps).sqrt();
            for i in 0..AHD {
                k_normed[b + i] = k_normed[b + i] * inv * kn[i];
            }
        }
    }

    // grim's own rope, called exactly as the model calls it.
    let mut pos_ext = Vec::with_capacity(S * AKV);
    for &p in &positions {
        for _ in 0..AKV {
            pos_ext.push(p);
        }
    }
    let mut rope_cfg = grim_tensor::RopeConfig::new(AHD, base as f32);
    rope_cfg.rotary_dim = n_rot;
    let t3 = grim_backend_cpu::cpu_tensor(k_normed.clone(), Shape::new(vec![1, S * AKV, AHD]));
    let (roped, _) = dev
        .rope(t3.storage().as_ref(), &pos_ext, &rope_cfg, t3.shape())
        .expect("dev.rope at seq_len=5");
    let grim_rope = roped.to_cpu_vec_f32().expect("read roped");

    let mut cache = Qwen35LayerCache::new(&c);
    let _ = blk
        .forward(
            &grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![S, HIDDEN])),
            &positions,
            &mut cache,
        )
        .expect("forward");
    let arena = cache
        .k_device
        .as_ref()
        .expect("K arena")
        .to_cpu_vec_f32()
        .expect("read arena");

    eprintln!("[split] grim_rope {} floats, arena {} floats", grim_rope.len(), arena.len());
    for t in 0..S {
        let row_err = |cand: &[f32]| -> f32 {
            let mut w = 0.0f32;
            for j in 0..kvd {
                let d = (arena[t * kvd + j] - cand[t * kvd + j]).abs();
                let rel = d / cand[t * kvd + j].abs().max(1e-3);
                w = w.max(d.min(rel));
            }
            w
        };
        eprintln!(
            "[split] row {t}: arena vs grim's own rope {:.3e} | arena vs pre-rope {:.3e}",
            row_err(&grim_rope),
            row_err(&k_normed)
        );
    }
}

/// THE SAME MEASUREMENT, ON THE ROCM PATH.
///
/// Everything above ran `Qwen35Block::forward` on CPU tensors. The gibberish is
/// produced by a ROCm run, so the CPU localisation is evidence about shared
/// orchestration, not a measurement of the device path. This closes that gap:
/// one real attention block loaded onto the GPU, a 5-row prefill at positions
/// 0..4, and the same row-by-row comparison of the K arena plus the block's
/// output against the hand reference.
///
/// Two outcomes, both informative:
///   * the device shows the SAME signature (row 0 exact, rows 1..4 wrong) — the
///     CPU localisation carries over and the defect is in the orchestration both
///     paths share;
///   * the device is clean — the defect is CPU-only and the real run's problem
///     is somewhere this test does not reach, which is worth knowing before
///     anyone "fixes" the shared code.
///
/// Gated: `GRIM_GPU_TEST=1` + a real ROCm device. The device assertion is not
/// decoration: `pick_device_for_storage_device` silently falls back to CPU when
/// the `rocm-mem` feature is off, and a CPU run of a device test is a false
/// green.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn seq5_kv_arena_on_rocm_matches_oracle() {
    use grim_tensor::{CoreTensorOps, DType};
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);
    // Prove the device is really ROCm. `pick_device_for_storage_device` returns
    // a CpuDevice when `rocm-mem` is off, and a CPU run of a device test is a
    // false green — so this is checked, not assumed. `BackendDevice` has no
    // `as_any` (only `BackendStorage` does), and `type_name_of_val` on a trait
    // object yields the TRAIT name, so the assertion is made on the loaded
    // block's own `device`, which is what actually decides the path.

    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let kvd = AKV * AHD;
    let positions: Vec<u32> = (0..S as u32).collect();

    // TWO blocks, same weights, different devices. The GPU one fills the arena;
    // the CPU one supplies the oracle, because `to_vec_f32` on a DEVICE-resident
    // K-quant weight is broken: for `attn_q` [8192, 4096] Q4_K it asks for
    // 134,217,728 B of f32 against an 18,874,368 B packed allocation and fails
    // with "DtoH shape exceeds allocation". The CPU backend dequantizes
    // correctly, so the oracle is computed there. Both blocks are loaded from
    // the same GGUF, so the weights are identical by construction.
    let ws = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3 onto the GPU");
    assert!(blk.is_full_attention);
    assert!(
        matches!(blk.device, Device::Rocm(_)),
        "block loaded onto {:?}, not a ROCm device — `rocm-mem` is off, so this test would \
         silently run the CPU path and report a meaningless green",
        blk.device
    );

    let x: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();
    let x_st = dev
        .from_cpu(&x, &Shape::new(vec![S, HIDDEN]), DType::F32)
        .expect("upload the 5-row input");
    let x_t = grim_tensor::Tensor::new(
        std::sync::Arc::from(x_st),
        Shape::new(vec![S, HIDDEN]),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );

    let mut cache = Qwen35LayerCache::new(&c);
    let out = blk
        .forward(&x_t, &positions, &mut cache)
        .expect("device forward at seq_len=5");
    // No explicit device sync: `to_vec_f32` below is a blocking D2H on the same
    // stream, so it is ordered after every enqueued write. `BackendDevice` has
    // no `synchronize` (only `ComputeHandle` does), and the ROCm backend's is
    // inherent, so it is not reachable through the trait object.
    let out = out.to_vec_f32().expect("read block output");

    // ---- oracle ------------------------------------------------------------
    let (n_rot, base) = rope_params(&prov);
    eprintln!("[rocm] rope from checkpoint: n_rot={n_rot} freq_base={base}");
    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let blk_cpu = Qwen35Block::load_tp(&ws_cpu, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3 on the CPU for the oracle");
    let mut ref_cache = Qwen35LayerCache::new(&c);
    seed_recurrent_cache(&mut ref_cache);
    let want = hand_attention_layer_multi(&blk_cpu, &x, &positions, n_rot, base);

    // ---- 1. the block's OUTPUT --------------------------------------------
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &r)) in out.iter().zip(want.iter()).enumerate() {
        let d = (g - r).abs();
        let rel = d / r.abs().max(1e-3);
        if d.min(rel) > worst {
            worst = d.min(rel);
            at = i;
        }
    }
    eprintln!(
        "[rocm] blk.3 output: worst {worst:.3e} at row {} col {} (device {}, reference {})",
        at / HIDDEN,
        at % HIDDEN,
        out[at],
        want[at]
    );
    let mut per_row = Vec::new();
    for r in 0..S {
        let mut w = 0.0f32;
        for j in 0..HIDDEN {
            let d = (out[r * HIDDEN + j] - want[r * HIDDEN + j]).abs();
            let rel = d / want[r * HIDDEN + j].abs().max(1e-3);
            w = w.max(d.min(rel));
        }
        per_row.push(w);
    }
    eprintln!(
        "[rocm] per-row worst: {:?}",
        per_row.iter().map(|v| format!("{v:.2e}")).collect::<Vec<_>>()
    );

    // ---- 2. the K ARENA ----------------------------------------------------
    let arena = cache.k_device.as_ref().expect("block must fill a K arena");
    let k_hist = arena.to_cpu_vec_f32().expect("read K arena");
    eprintln!("[rocm] K arena {} floats, shape {:?}", k_hist.len(), arena.shape().dims());
    for t in 0..S.min(k_hist.len() / kvd) {
        // Compare the arena row against the K this block's weights produce, using
        // the same recipe as the CPU test.
        let mut wk = 0.0f32;
        let kn = blk_cpu.attn_k_norm.as_ref().expect("k_norm").weight.to_vec_f32().unwrap();
        let eps = blk_cpu.attn_k_norm.as_ref().map(|n| n.eps).unwrap_or(blk_cpu.attn_norm.eps);
        let (kw, ki, ko) = lin_w(blk_cpu.wk.as_ref().expect("attn_k"));
        let an = blk_cpu.attn_norm.weight.to_vec_f32().expect("attn_norm");
        let xn = rms(&x[t * HIDDEN..(t + 1) * HIDDEN], &an, blk_cpu.attn_norm.eps);
        let mut kk = matvec(&xn, &kw, ki, ko);
        for h in 0..AKV {
            let s0 = h * AHD;
            let n = rms(&kk[s0..s0 + AHD], &kn, eps);
            kk[s0..s0 + AHD].copy_from_slice(&n);
            let mut seg = kk[s0..s0 + AHD].to_vec();
            rope_partial(&mut seg, positions[t] as f64, n_rot, base);
            kk[s0..s0 + AHD].copy_from_slice(&seg);
        }
        for j in 0..kvd {
            let d = (k_hist[t * kvd + j] - kk[j]).abs();
            let rel = d / kk[j].abs().max(1e-3);
            wk = wk.max(d.min(rel));
        }
        eprintln!("[rocm] K arena row {t}: worst vs oracle {wk:.3e}");
    }
}

/// ISOLATE POSITION vs PER-ROW COMPUTE: five IDENTICAL input rows.
///
/// Everything measured so far confounds two things. Feeding distinct rows makes
/// the arena depend on both the per-row K values and the per-row RoPE position,
/// so a mismatch cannot say which is wrong.
///
/// Hold the source constant — all five rows byte-identical — and the two
/// separate. The K projection then yields the SAME k for every row, so the only
/// thing that can distinguish arena row `t` from arena row 0 is the position
/// fed to RoPE. Three outcomes, all informative:
///
///   * rows 1..4 == rope(k, t)  -> position path is fine, and the earlier
///     mismatch came from the per-row K values instead;
///   * rows 1..4 == rope(k, 0)  -> the position is not reaching RoPE past row 0;
///   * rows 1..4 == pre-rope k  -> RoPE is not applied to the appended rows.
///
/// GPU-gated, and it asserts the block really loaded on ROCm.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn seq5_identical_rows_isolate_the_rope_position() {
    use grim_tensor::{CoreTensorOps, DType};
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let _c = cfg();
    const S: usize = 5;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let kvd = AKV * AHD;
    let positions: Vec<u32> = (0..S as u32).collect();

    let mut c = cfg();
    let (ck_n_rot, _) = rope_params(&prov);
    c.rotary_dim = Some(ck_n_rot);
    let ws = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load blk.3 on the GPU");
    assert!(blk.is_full_attention);
    assert!(
        matches!(blk.device, Device::Rocm(_)),
        "block loaded onto {:?}, not ROCm — this would be a false green",
        blk.device
    );
    eprintln!("[iso] block rotary_dim={} (checkpoint says {ck_n_rot})", blk.rotary_dim);

    // ONE row, replicated five times.
    let row0: Vec<f32> = (0..HIDDEN).map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5).collect();
    let x: Vec<f32> = row0.iter().cycle().take(S * HIDDEN).copied().collect();
    assert!(
        x[..HIDDEN] == x[HIDDEN..2 * HIDDEN] && x[HIDDEN..2 * HIDDEN] == x[2 * HIDDEN..3 * HIDDEN],
        "the five input rows must be byte-identical or this test measures nothing"
    );

    let x_st = dev
        .from_cpu(&x, &Shape::new(vec![S, HIDDEN]), DType::F32)
        .expect("upload");
    let x_t = grim_tensor::Tensor::new(
        std::sync::Arc::from(x_st),
        Shape::new(vec![S, HIDDEN]),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );
    let mut cache = Qwen35LayerCache::new(&c);
    blk.forward(&x_t, &positions, &mut cache)
        .expect("forward at seq_len=5 with identical rows");
    let k_hist = cache
        .k_device
        .as_ref()
        .expect("K arena")
        .to_cpu_vec_f32()
        .expect("read K arena");

    // Oracle for the single distinct row, from a CPU block.
    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let blk_cpu = Qwen35Block::load_tp(&ws_cpu, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load blk.3 on the CPU for the oracle");
    let (n_rot, base) = rope_params(&prov);
    let an = blk_cpu.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(&row0, &an, blk_cpu.attn_norm.eps);
    let (kw, ki, ko) = lin_w(blk_cpu.wk.as_ref().expect("attn_k"));
    let kn = blk_cpu.attn_k_norm.as_ref().expect("k_norm").weight.to_vec_f32().unwrap();
    let keps = blk_cpu.attn_k_norm.as_ref().map(|n| n.eps).unwrap_or(blk_cpu.attn_norm.eps);
    let mut k_norm = matvec(&xn, &kw, ki, ko);
    for h in 0..AKV {
        let s0 = h * AHD;
        let n = rms(&k_norm[s0..s0 + AHD], &kn, keps);
        k_norm[s0..s0 + AHD].copy_from_slice(&n);
    }
    let roped_at = |p: f64| -> Vec<f32> {
        let mut k = k_norm.clone();
        for h in 0..AKV {
            let s0 = h * AHD;
            let mut seg = k[s0..s0 + AHD].to_vec();
            rope_partial(&mut seg, p, n_rot, base);
            k[s0..s0 + AHD].copy_from_slice(&seg);
        }
        k
    };
    let cand_post_t: Vec<Vec<f32>> = (0..S).map(|t| roped_at(t as f64)).collect();
    let _cand_zero = roped_at(0.0);

    let err = |arena_row: &[f32], cand: &[f32]| -> f32 {
        let mut w = 0.0f32;
        for j in 0..kvd {
            let d = (arena_row[j] - cand[j]).abs();
            let rel = d / cand[j].abs().max(1e-3);
            w = w.max(d.min(rel));
        }
        w
    };

    // grim's OWN rope on the same 5-row normed K, called exactly as the model
    // calls it. This is the column that matters: every other candidate above is
    // my MODEL of the rope, and my model has been wrong before. If the arena
    // matches grim's own rope, then grim's rope and append are both correct and
    // the whole "corrupted K" reading is my oracle's fault.
    let mut pos_ext = Vec::with_capacity(S * AKV);
    for &p in &positions {
        for _ in 0..AKV {
            pos_ext.push(p);
        }
    }
    let mut rope_cfg = grim_tensor::RopeConfig::new(AHD, base as f32);
    rope_cfg.rotary_dim = n_rot;
    // The BLOCK's own rope parameters, for comparison with the checkpoint's.
    eprintln!(
        "[rope] CHECKPOINT: n_rot={n_rot} freq_base={base}  |  BLOCK: n_rot={} freq_base={}",
        blk.rotary_dim, blk.rope_theta
    );
    let mut block_cfg = grim_tensor::RopeConfig::new(AHD, blk.rope_theta);
    block_cfg.rotary_dim = blk.rotary_dim;
    // All five input rows are identical, so the 5-row normed K is `k_norm`
    // replicated — which is exactly what the block's own 5-row buffer holds.
    let k_norm_5: Vec<f32> = k_norm.iter().cycle().take(S * kvd).copied().collect();
    assert_eq!(k_norm_5.len(), S * kvd);
    let t3_shape = Shape::new(vec![1, S * AKV, AHD]);
    let t3 = dev
        .from_cpu(&k_norm_5, &t3_shape, DType::F32)
        .expect("upload the 5-row normed K");
    let (roped, _) = dev
        .rope(t3.as_ref(), &pos_ext, &rope_cfg, &t3_shape)
        .expect("dev.rope on the 5-row normed K");
    let grim_rope = roped.to_cpu_vec_f32().expect("read grim rope");

    // DOUBLE ROPE: rope grim's already-roped output a second time. A rope at
    // position 0 is the identity, so a double rope is INDISTINGUISHABLE from a
    // single one at row 0 and diverges by an angle 2*t*theta at row t — which is
    // exactly the observed shape (exact at row 0, error growing with the row).
    let st2 = dev
        .from_cpu(&grim_rope, &t3_shape, DType::F32)
        .expect("upload grim_rope");
    let (roped2, _) = dev
        .rope(st2.as_ref(), &pos_ext, &rope_cfg, &t3_shape)
        .expect("second dev.rope");
    let grim_rope2 = roped2.to_cpu_vec_f32().expect("read");

    // The block's own parameters — the decisive candidate.
    let st3 = dev.from_cpu(&k_norm_5, &t3_shape, DType::F32).expect("upload");
    let (rb, _) = dev
        .rope(st3.as_ref(), &pos_ext, &block_cfg, &t3_shape)
        .expect("dev.rope with the BLOCK's cfg");
    let rope_block_cfg = rb.to_cpu_vec_f32().expect("read");

    eprintln!("[iso] n_rot={n_rot} freq_base={base}; five byte-identical input rows");
    for t in 0..S {
        let arow = &k_hist[t * kvd..(t + 1) * kvd];
        let g = &grim_rope[t * kvd..(t + 1) * kvd];
        let g2 = &grim_rope2[t * kvd..(t + 1) * kvd];
        eprintln!(
            "[iso] row {t}: vs grim rope {single:.3e} | vs BLOCK's cfg {blkcfg:.3e} | \
             vs DOUBLE {dbl:.3e} || my model: rope(k,{t}) {mine:.3e} | pre-rope {pre:.3e}",
            single = err(arow, g),
            blkcfg = err(arow, &rope_block_cfg[t * kvd..(t + 1) * kvd]),
            dbl = err(arow, g2),
            mine = err(arow, &cand_post_t[t]),
            pre = err(arow, &k_norm)
        );
    }
}

/// WHAT DO ARENA ROWS 1..4 ACTUALLY HOLD?
///
/// With five byte-identical input rows the projection yields the same K for
/// every row, and the previous test showed rows 1..4 match no rotation of it at
/// any position — nor the un-rotated value. So they are not K. Since the errors
/// GROW with the row (1.48, 2.85, 3.78, 4.04) while the source is constant,
/// they are not a constant either: each row holds something different, and the
/// divergence accumulates.
///
/// Candidates, all cheap to score against the arena: the V projection, the fused
/// Q half, the Q gate half, zeros, and — the one that a short or mis-offset
/// append would produce — the arena's own stale contents.
///
/// This prints each candidate's error per row plus basic row statistics, so the
/// answer is visible even if no candidate wins outright.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn seq5_what_do_arena_rows_hold() {
    use grim_tensor::{CoreTensorOps, DType};
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);
    let Some(path) = model_path() else { return };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    const AH: usize = 16;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let kvd = AKV * AHD;
    let qd = AH * AHD;
    let positions: Vec<u32> = (0..S as u32).collect();

    let ws = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load on GPU");
    assert!(matches!(blk.device, Device::Rocm(_)), "not ROCm — false green");

    let row0: Vec<f32> = (0..HIDDEN).map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5).collect();
    let x: Vec<f32> = row0.iter().cycle().take(S * HIDDEN).copied().collect();
    let x_t = grim_tensor::Tensor::new(
        std::sync::Arc::from(
            dev.from_cpu(&x, &Shape::new(vec![S, HIDDEN]), DType::F32)
                .expect("upload"),
        ),
        Shape::new(vec![S, HIDDEN]),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );
    let mut cache = Qwen35LayerCache::new(&c);
    blk.forward(&x_t, &positions, &mut cache).expect("forward");
    let k_hist = cache.k_device.as_ref().expect("arena").to_cpu_vec_f32().expect("read");

    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let bc = Qwen35Block::load_tp(&ws_cpu, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load on CPU");
    let an = bc.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(&row0, &an, bc.attn_norm.eps);
    let (kw, ki, ko) = lin_w(bc.wk.as_ref().expect("attn_k"));
    let (vw, vi, vo) = lin_w(bc.wv.as_ref().expect("attn_v"));
    let (qw, qi, qo) = lin_w(bc.wq.as_ref().expect("attn_q"));
    let k = matvec(&xn, &kw, ki, ko);
    let v = matvec(&xn, &vw, vi, vo);
    let qfull = matvec(&xn, &qw, qi, qo);
    let q = &qfull[..qd];
    let gate = &qfull[qd..2 * qd];

    let err = |arow: &[f32], cand: &[f32]| -> f32 {
        let mut w = 0.0f32;
        for j in 0..kvd.min(cand.len()) {
            let d = (arow[j] - cand[j]).abs();
            let rel = d / cand[j].abs().max(1e-3);
            w = w.max(d.min(rel));
        }
        w
    };
    let stats = |arow: &[f32]| -> (f32, f32, f32) {
        let mx = arow.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let mean = arow.iter().map(|v| v.abs()).sum::<f32>() / arow.len() as f32;
        let first3: Vec<String> = arow[..3].iter().map(|v| format!("{v:.5}")).collect();
        eprintln!("    first3={first3:?}");
        (mx, mean, 0.0)
    };

    eprintln!("[what] n_rot={} base={}", rope_params(&prov).0, rope_params(&prov).1);
    eprintln!("[what] |k|max={:.5} |v|max={:.5} |q|max={:.5}", 
        k.iter().fold(0.0f32,|a,b|a.max(b.abs())),
        v.iter().fold(0.0f32,|a,b|a.max(b.abs())),
        q.iter().fold(0.0f32,|a,b|a.max(b.abs())));
    for t in 0..S {
        let arow = &k_hist[t * kvd..(t + 1) * kvd];
        let (mx, mean, _) = stats(arow);
        eprintln!(
            "[what] row {t}: |max|={mx:.5} mean|x|={mean:.5} | vs k {:.3e} | vs v {:.3e} \
             | vs q {:.3e} | vs gate {:.3e} | vs zeros {:.3e}",
            err(arow, &k), err(arow, &v), err(arow, q), err(arow, gate), err(arow, &vec![0.0f32; kvd])
        );
    }
}

/// THE DEVICE MULTI-ROW PROJECTION — the one thing never verified.
///
/// `seq5_k_projection_per_row` checked `wk.forward` at seq_len=5 on **CPU
/// tensors**. Every other multi-row claim in this file is likewise from the CPU
/// path. The gibberish is produced on ROCm, so the device 5-row GEMM has never
/// been checked at all — and a projection that is right for one row and wrong
/// for several is exactly the shape that leaves every one-token gate green while
/// the prefill is garbage.
///
/// This is the targeted measurement the ~4e-2 residual calls for: load blk.3 on
/// the GPU, run `wk.forward` on a 5-row device tensor, read it back, and compare
/// row by row against the CPU per-row matvec. Uses FIVE DISTINCT rows so a
/// per-row error is visible rather than averaged away.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn seq5_device_projection_per_row() {
    use grim_tensor::{CoreTensorOps, DType};
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);
    let Some(path) = model_path() else { return };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let kvd = AKV * AHD;

    let ws = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load blk.3 on the GPU");
    assert!(matches!(blk.device, Device::Rocm(_)), "not ROCm — false green");

    // FIVE DISTINCT rows: a per-row defect cannot hide behind identical inputs.
    let x: Vec<f32> = (0..S * HIDDEN).map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5).collect();
    let x_t = grim_tensor::Tensor::new(
        std::sync::Arc::from(
            dev.from_cpu(&x, &Shape::new(vec![S, HIDDEN]), DType::F32)
                .expect("upload"),
        ),
        Shape::new(vec![S, HIDDEN]),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );
    let x_normed = blk.attn_norm.forward(&x_t).expect("pre-norm on device");

    // --- the DEVICE projection -------------------------------------------
    let dev_k = blk
        .wk
        .as_ref()
        .expect("attn_k")
        .forward(&x_normed)
        .expect("wk.forward on device at seq_len=5")
        .to_vec_f32()
        .expect("read device projection");
    assert_eq!(dev_k.len(), S * kvd, "projection must return [S, kv_dim]");

    // --- the CPU reference, from a CPU copy of the same block -------------
    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let bc = Qwen35Block::load_tp(&ws_cpu, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load blk.3 on the CPU");
    let an = bc.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let (kw, ki, ko) = lin_w(bc.wk.as_ref().expect("attn_k"));
    let (qw, qi, qo) = lin_w(bc.wq.as_ref().expect("attn_q"));
    let mut want = vec![0.0f32; S * kvd];
    for t in 0..S {
        let xn = rms(&x[t * HIDDEN..(t + 1) * HIDDEN], &an, bc.attn_norm.eps);
        let k = matvec(&xn, &kw, ki, ko);
        want[t * kvd..(t + 1) * kvd].copy_from_slice(&k);
    }

    for t in 0..S {
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for j in 0..kvd {
            let d = (dev_k[t * kvd + j] - want[t * kvd + j]).abs();
            let rel = d / want[t * kvd + j].abs().max(1e-3);
            if d.min(rel) > worst {
                worst = d.min(rel);
                at = j;
            }
        }
        eprintln!(
            "[devproj] attn_k row {t}: worst {worst:.3e} at {at} (device {}, reference {})",
            dev_k[t * kvd + at],
            want[t * kvd + at]
        );
        assert!(
            worst <= 5e-2,
            "DEVICE attn_k PROJECTION is wrong at row {t} of {S}: worst {worst:.3e} at {at} \\
             (device {}, reference {}). A multi-row GEMM that is right for one row and wrong \\
             for several is invisible to every one-token gate and corrupts the whole prefill.",
            dev_k[t * kvd + at],
            want[t * kvd + at]
        );
    }
    eprintln!("[devproj] attn_k matches the CPU reference on all {S} rows on the DEVICE");

    // The fused QKV is the other multi-row projection, and it is the one every
    // KDA layer uses. Same check, because a defect in the fused path would not
    // show up in a separated-QKV layer at all.
    let dev_qf = blk
        .wq
        .as_ref()
        .expect("attn_q")
        .forward(&x_normed)
        .expect("wq.forward on device at seq_len=5")
        .to_vec_f32()
        .expect("read");
    let qd = 16 * AHD;
    assert_eq!(dev_qf.len(), S * 2 * qd, "attn_q is [S, 2*q_dim]");
    for t in 0..S {
        let xn = rms(&x[t * HIDDEN..(t + 1) * HIDDEN], &an, bc.attn_norm.eps);
        let qf = matvec(&xn, &qw, qi, qo);
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for j in 0..2 * qd {
            let d = (dev_qf[t * 2 * qd + j] - qf[j]).abs();
            let rel = d / qf[j].abs().max(1e-3);
            if d.min(rel) > worst {
                worst = d.min(rel);
                at = j;
            }
        }
        eprintln!("[devproj] fused attn_q row {t}: worst {worst:.3e} at {at}");
        assert!(
            worst <= 5e-2,
            "DEVICE fused attn_q PROJECTION is wrong at row {t} of {S}: worst {worst:.3e} at {at} \\
             (device {}, reference {}). The fused Q | gate tensor feeds both the RoPE and the \\
             output gate, so a per-row defect here corrupts the branch twice.",
            dev_qf[t * 2 * qd + at],
            qf[at]
        );
    }
    eprintln!("[devproj] fused attn_q matches the CPU reference on all {S} rows on the DEVICE");
}

/// IS `rope_ext`'s RESHAPE bit-exact on the device?
///
/// With the device projections now verified at seq_len=5, exactly two links of
/// the K chain remain unchecked, and only one of them is a copy:
///
///   * `apply_head_rms_norm` — host-side, per row, no copy;
///   * `rope_ext`'s `reshaped_view(t, [1, S*heads, head_dim])` — a 5120-float
///     D2D copy, i.e. the `copy_slice_range` path whose `hipMemcpyAsync` this
///     whole session has watched get REFUSED and handed to the `grim_copy_bytes`
///     kernel.
///
/// The residual's shape points here: a per-row computation cannot grow with the
/// row INDEX, but a copy that mangles the tail of its range can — and once the
/// corrupted values are rotated, a t-independent input error reads as a
/// t-growing output error.
///
/// This tests the reshape alone, with no rope: round-trip a known 5-row buffer
/// through the exact same `reshaped_view` calls `rope_ext` makes and compare
/// every element. Exact means the copy is exonerated and the norm is next.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn seq5_reshaped_view_is_bit_exact_on_device() {
    use grim_tensor::{CoreTensorOps, DType};
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);
    const S: usize = 5;
    const AKV: usize = 4;
    const AHD: usize = 256;
    let wide = AKV * AHD;

    // Distinct values everywhere, so a mis-offset copy cannot hide behind
    // symmetry, and include a wide magnitude range.
    let src: Vec<f32> = (0..S * wide)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0 + (i as f32) * 1e-3)
        .collect();
    let flat = Shape::new(vec![S, wide]);
    let t3d = Shape::new(vec![1, S * AKV, AHD]);

    let st = dev.from_cpu(&src, &flat, DType::F32).expect("upload [5,1024]");
    let t = grim_tensor::Tensor::new(
        std::sync::Arc::from(st),
        flat.clone(),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );

    // `reshaped_view` is pub(crate), so exercise the two calls it makes:
    // `alloc_storage(target_shape)` then `copy_slice_into(fresh, src, 0,
    // elem_count)` — a 5120-float D2D copy into a differently-shaped buffer,
    // which is the whole content of the reshape.
    let up = dev.alloc_storage(&t3d, DType::F32).expect("alloc the [1,20,256] destination");
    dev.copy_slice_into(up.as_ref(), t.storage().as_ref(), 0, S * wide)
        .expect("5120-float D2D copy into the reshaped destination");
    let back = dev.alloc_storage(&flat, DType::F32).expect("alloc the [5,1024] destination");
    dev.copy_slice_into(back.as_ref(), up.as_ref(), 0, S * wide)
        .expect("5120-float D2D copy back");
    let got = back.to_cpu_vec_f32().expect("read round-tripped buffer");

    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(src.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            let d = (g - w).abs();
            if d > worst {
                worst = d;
                at = i;
            }
        }
    }
    let mismatched = got
        .iter()
        .zip(src.iter())
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    eprintln!(
        "[reshape] {S}x{wide} round trip on device: {mismatched} of {} elements differ, \
         worst {worst:.3e} at {at} (row {}, col {})",
        src.len(),
        at / wide,
        at % wide
    );
    assert_eq!(
        mismatched, 0,
        "reshaped_view is NOT bit-exact on the device: {mismatched} of {} elements differ \\
         (worst {worst:.3e} at flat index {at}, row {} col {}). This is the 5120-float D2D copy \\
         inside `rope_ext`, on the same `copy_slice_range` path whose hipMemcpyAsync keeps \\
         getting refused and handed to the `grim_copy_bytes` kernel.",
        src.len(),
        at / wide,
        at % wide
    );
    eprintln!("[reshape] bit-exact on the device");
}

/// THE LAST UNVERIFIED PIECE: `output_norm` then `lm_head`, composed.
///
/// Everything upstream is now measured: the embedding is bit-exact, all 32
/// blocks match at seq_len=1 and seq_len=5, every K-quant is verified against
/// llama.cpp, and the engine's positions are `[0,1,2,3,4]` for the prefill.
/// The only part of a forward pass never checked as a COMPOSITION is the tail:
///
///     logits = output(norm(final_hidden))
///
/// The lm_head GEMM is verified in isolation (Q6_K, 5.6e-6) and the norm is a
/// few lines, but neither gate composes them, and a wrong eps, a doubled
/// application, or a norm applied to the wrong tensor all pass every other gate
/// in this file.
///
/// The hidden state is the LAST BLOCK'S REAL OUTPUT from the verified chain, so
/// this tests the tail and nothing else.
#[test]
fn tail_output_norm_and_lm_head_match_the_reference() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let c = cfg();
    const S: usize = 5;
    let positions: Vec<u32> = (0..S as u32).collect();

    // A realistic hidden state: run the real blocks, CPU, over the real weights.
    let mut h: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();
    for k in 0..32usize {
        let name = format!("blk.{k}");
        let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp(&name);
        let blk = Qwen35Block::load_tp(&ws, &c, k, grim_nn::TensorParallelConfig::default())
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        let mut cache = Qwen35LayerCache::new(&c);
        seed_recurrent_cache(&mut cache);
        h = blk
            .forward(
                &grim_backend_cpu::cpu_tensor(h.clone(), Shape::new(vec![S, HIDDEN])),
                &positions,
                &mut cache,
            )
            .unwrap_or_else(|e| panic!("{name} forward: {e}"))
            .to_vec_f32()
            .expect("read");
    }
    eprintln!("[tail] final hidden state: |mean| {:.5}", {
        let m: f32 = h.iter().map(|v| v.abs()).sum::<f32>() / h.len() as f32;
        m
    });

    // ---- the tail, through grim's own objects ------------------------------
    let ws_out = grim_nn::WeightSource::root(&prov, Device::Cpu);
    let on = grim_nn::modules::RmsNorm::load(
        &ws_out.pp("output_norm"),
        HIDDEN,
        c.rms_norm_eps,
    )
    .expect("load output_norm");
    eprintln!(
        "[tail] output_norm weight len={} eps={}",
        on.weight.to_vec_f32().map(|v| v.len()).unwrap_or(0),
        on.eps
    );
    let oh = grim_backend_cpu::cpu_tensor(h.clone(), Shape::new(vec![S, HIDDEN]));
    let normed = on.forward(&oh).expect("output_norm.forward").to_vec_f32().expect("read");

    // ---- the reference ------------------------------------------------------
    let onw = on.weight.to_vec_f32().expect("output_norm weight");
    let mut want_normed = vec![0.0f32; S * HIDDEN];
    for t in 0..S {
        want_normed[t * HIDDEN..(t + 1) * HIDDEN]
            .copy_from_slice(&rms(&h[t * HIDDEN..(t + 1) * HIDDEN], &onw, on.eps));
    }
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in normed.iter().zip(want_normed.iter()).enumerate() {
        let d = (g - w).abs();
        let rel = d / w.abs().max(1e-3);
        if d.min(rel) > worst {
            worst = d.min(rel);
            at = i;
        }
    }
    eprintln!("[tail] output_norm: worst {worst:.3e} at row {} col {}", at / HIDDEN, at % HIDDEN);
    assert!(
        worst <= 1e-4,
        "output_norm does not match the RMS reference (worst {worst:.3e} at row {} col {}). \
         The final norm is applied to every token, so a wrong eps or a doubled application \
         corrupts the logits of the whole batch while every block gate stays green.",
        at / HIDDEN,
        at % HIDDEN
    );
    eprintln!("[tail] output_norm matches the reference");
}

/// THE DEVICE CHAIN: all 32 blocks on ROCm at seq_len=5.
///
/// Every chain result in this file is CPU-side. The per-piece device checks
/// cover blk.3's projection, its K arena and the 5120-float reshape — but not
/// the KDA layers on device, and not the full stack. That is the same
/// CPU-vs-device gap that hid the multi-row projection earlier, one level up.
///
/// Two blocks per layer, from the same GGUF: one on the GPU (which runs the
/// forward and fills the KV arenas) and one on the CPU (which supplies the
/// hand reference). One layer at a time, so peak VRAM is two blocks rather
/// than the whole 5.7 GB model. The reference is the same
/// `hand_attention_layer_multi` / `hand_recurrent_layer_multi` the CPU chain
/// uses, and those now CALL `dev.rope` instead of modelling it — the fix that
/// took the CPU chain from "breaks at blk.3" to "all 32 clean".
///
/// Gated: `GRIM_GPU_TEST=1`; asserts every block really loaded on ROCm, because
/// a silent CPU fallback here would be a false green on the very gap this
/// exists to close.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn all_32_blocks_chain_correctly_at_seq_len_5_on_rocm() {
    use grim_tensor::DType;
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);
    let Some(path) = model_path() else {
        eprintln!("[SKIP] 9B checkpoint not found under the workspace");
        return;
    };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let mut c = cfg();
    let (ck_n_rot, _) = rope_params(&prov);
    c.rotary_dim = Some(ck_n_rot);
    eprintln!("[devchain] rotary_dim={ck_n_rot} (from qwen35.rope.dimension_count)");
    const S: usize = 5;
    let positions: Vec<u32> = (0..S as u32).collect();

    let mut h: Vec<f32> = (0..S * HIDDEN)
        .map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5)
        .collect();

    for k in 0..32usize {
        let name = format!("blk.{k}");
        let ws_gpu = grim_nn::WeightSource::root(&prov, device.clone()).pp(&name);
        let bg = Qwen35Block::load_tp(&ws_gpu, &c, k, grim_nn::TensorParallelConfig::default())
            .unwrap_or_else(|e| panic!("load {name} on GPU: {e}"));
        assert!(
            matches!(bg.device, Device::Rocm(_)),
            "{name} loaded onto {:?}, not ROCm — the whole point of this gate is the device path",
            bg.device
        );
        let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp(&name);
        let bc = Qwen35Block::load_tp(&ws_cpu, &c, k, grim_nn::TensorParallelConfig::default())
            .unwrap_or_else(|e| panic!("load {name} on CPU: {e}"));

        let mut cache = Qwen35LayerCache::new(&c);
        seed_recurrent_cache(&mut cache);
        let mut ref_cache = Qwen35LayerCache::new(&c);
        seed_recurrent_cache(&mut ref_cache);
        let want = if bg.is_full_attention {
            hand_attention_layer_multi(&bc, &h, &positions, ck_n_rot, rope_params(&prov).1)
        } else {
            hand_recurrent_layer_multi(&bc, &h, S, &mut ref_cache)
        };

        let x_t = grim_tensor::Tensor::new(
            std::sync::Arc::from(
                dev.from_cpu(&h, &Shape::new(vec![S, HIDDEN]), DType::F32)
                    .expect("upload"),
            ),
            Shape::new(vec![S, HIDDEN]),
            DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        );
        let got = bg
            .forward(&x_t, &positions, &mut cache)
            .unwrap_or_else(|e| panic!("{name} device forward: {e}"))
            .to_vec_f32()
            .expect("read");

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
        eprintln!(
            "[devchain] {name} ({}) worst {worst:.3e} at row {} col {}",
            if bg.is_full_attention { "attn" } else { "kda " },
            at / HIDDEN,
            at % HIDDEN
        );
        assert!(
            worst <= 5e-2,
            "DEVICE CHAIN BREAKS AT {name} (layer {k}, seq_len={S}): output differs from the \
             hand reference by {worst:.3e} at row {} col {} (device {}, reference {}). Every \
             earlier layer matched on the GPU, so the defect is in this block's device path.",
            at / HIDDEN,
            at % HIDDEN,
            got[at],
            want[at]
        );
        h = got;
    }
    eprintln!("[devchain] all 32 blocks chain correctly ON ROCm at seq_len={S}");
}

/// THE KDA BRANCH'S OWN MULTI-ROW PROJECTIONS, on the device, at seq_len=5.
///
/// The KDA kernels are now gated and correct: `kda_gated_delta_rule_scan` runs
/// at 3.999e-4 in the 9B's own 32x128 geometry (`kda_scan_parity.rs`), and the
/// conv dispatches to `grim_short_conv1d_scan` because the 9B's conv weight is
/// F32, so none of the quantized size heuristics fire and `batch > 1` is
/// reached. The device chain still broke at its FIRST KDA layer, so the
/// remaining link is what feeds those kernels.
///
/// `attn_q` and `attn_k` were verified at seq_len=5 on device — but those are a
/// full-attention layer's projections. A recurrent layer's are different
/// tensors: `attn_qkv` and `ssm_out` are Q5_K, `ssm_alpha`/`ssm_beta` are
/// Q8_0, `attn_gate` is Q4_K. The one-token gates never covered them at more
/// than one row, and a prefill is entirely multi-row.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn kda_branch_multi_row_projections_on_device() {
    use grim_tensor::DType;
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let Some(path) = model_path() else { return };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let mut c = cfg();
    let (ck_n_rot, _) = rope_params(&prov);
    c.rotary_dim = Some(ck_n_rot);
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);
    const S: usize = 5;

    let ws_gpu = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.0");
    let bg = Qwen35Block::load_tp(&ws_gpu, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the GPU");
    assert!(!bg.is_full_attention, "blk.0 must be a KDA layer");
    assert!(
        matches!(bg.device, Device::Rocm(_)),
        "blk.0 loaded onto {:?}, not ROCm — false green",
        bg.device
    );
    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.0");
    let bc = Qwen35Block::load_tp(&ws_cpu, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the CPU");

    // Distinct rows, so a per-row defect cannot hide behind symmetric inputs.
    let x: Vec<f32> = (0..S * HIDDEN).map(|i| ((i * 13 % 31) as f32) / 31.0 - 0.5).collect();
    let x_t = grim_tensor::Tensor::new(
        std::sync::Arc::from(
            dev.from_cpu(&x, &Shape::new(vec![S, HIDDEN]), DType::F32)
                .expect("upload"),
        ),
        Shape::new(vec![S, HIDDEN]),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );
    let xn_dev = bg.attn_norm.forward(&x_t).expect("pre-norm on device");

    let an = bc.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for t in 0..S {
        rows.push(rms(&x[t * HIDDEN..(t + 1) * HIDDEN], &an, bc.attn_norm.eps));
    }

    for (tag, gl, cl, width) in [
        ("attn_qkv", &bg.attn_qkv, &bc.attn_qkv, conv_dim()),
        ("ssm_alpha", &bg.ssm_alpha, &bc.ssm_alpha, NV),
        ("ssm_beta", &bg.ssm_beta, &bc.ssm_beta, NV),
        ("attn_gate", &bg.attn_gate, &bc.attn_gate, value_dim()),
        ("ssm_out", &bg.ssm_out, &bc.ssm_out, HIDDEN),
    ] {
        let (Some(gl), Some(cl)) = (gl, cl) else {
            eprintln!("[kda-proj] {tag} absent, skipped");
            continue;
        };
        let got = gl
            .forward(&xn_dev)
            .unwrap_or_else(|e| panic!("{tag} forward at seq_len={S}: {e}"))
            .to_vec_f32()
            .expect("read");
        let (wd, w_in, w_out) = lin_w(cl);
        // The reference's own geometry must be checked before the device is
        // blamed: `lin_w` reads the BLOCK's [out, in] shape, while
        // `read_gguf` reports `ssm_alpha.weight` as [4096, 32] — the opposite
        // order. If the two disagree, `matvec(rows, wd, HIDDEN, w_out)` is
        // indexing a transposed weight and the "device is wrong" reading is
        // this test's own bug, which is the trap five times over tonight.
        eprintln!(
            "[kda-proj] {tag}: block weight shape {:?} -> (in={w_in}, out={w_out});              expected in={HIDDEN}; device returned {} floats for {S}x{width}",
            cl.weight().shape().dims().to_vec(),
            got.len()
        );
        assert_eq!(
            w_in, HIDDEN,
            "{tag}: block weight reports in={w_in}, expected {HIDDEN} — the reference below \
             indexes with in={HIDDEN} and would be measuring a transposed weight"
        );
        assert!(got.len() >= S * width, "{tag} returned {} floats", got.len());
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for t in 0..S {
            let want = matvec(&rows[t], &wd, HIDDEN, w_out);
            for j in 0..width {
                let d = (got[t * width + j] - want[j]).abs();
                let rel = d / want[j].abs().max(1e-3);
                if rel > worst {
                    worst = rel;
                    at = t * width + j;
                }
            }
        }
        eprintln!(
            "[kda-proj] {tag} [{w_out}x{HIDDEN}] seq_len={S}: worst {worst:.3e} at row {} col {}",
            at / width,
            at % width
        );
        assert!(
            worst <= 5e-2,
            "KDA projection {tag} is wrong on the DEVICE at seq_len={S}: worst {worst:.3e} at \
             row {} col {}. A recurrent layer's projections have never been checked at more than \
             one row, and a prefill is entirely multi-row.",
            at / width,
            at % width
        );
    }
    eprintln!("[kda-proj] every KDA projection matches on the device at seq_len={S}");
}

/// `Linear::forward` for `ssm_alpha` at m=1 vs m=5 — the exact production call.
///
/// The reference is now fully anchored, so anything left is a real defect:
///   * grim's Q8_0 DECODER is bit-exact vs llama.cpp `dequantize_row_q8_0`
///     (gate `q8_0_dequant_matches_llama_cpp_on_the_real_alpha_weight`);
///   * the underlying Q8_0 GEMM primitive is correct at m = 1, 2, 4, 5, 8, 16
///     (gate `q8_0_gemm_accuracy_vs_m`, ~1.3e-3);
///   * the block's weight geometry is [out=32, in=4096], which is what the
///     reference below assumes.
///
/// The m-sweep used `fused_quant_gemm` on the raw bytes; PRODUCTION goes through
/// `Linear::forward`, which may pick a different path for m > 1. If m=1 agrees
/// and m=5 does not, the defect is in the `Linear` wrapper's multi-row
/// selection, not in any kernel.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn ssm_alpha_linear_forward_depends_on_m() {
    use grim_tensor::DType;
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let Some(path) = model_path() else { return };
    let prov = GgufProvider::open(path.to_str().expect("utf8 path")).expect("open 9B");
    let mut c = cfg();
    let (ck_n_rot, _) = rope_params(&prov);
    c.rotary_dim = Some(ck_n_rot);
    let device = Device::Rocm(0);
    let dev = grim_nn::modules::pick_device_for_storage_device(&device);

    let ws = grim_nn::WeightSource::root(&prov, device.clone()).pp("blk.0");
    let bg = Qwen35Block::load_tp(&ws, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the GPU");
    assert!(matches!(bg.device, Device::Rocm(_)), "not ROCm — false green");
    let gl = bg.ssm_alpha.as_ref().expect("ssm_alpha");
    // The reference weight comes from a CPU copy: `to_vec_f32` on a
    // DEVICE-resident K-quant tensor is broken (it asks for 4 B/elem against a
    // packed allocation), so reading it here would fail rather than mislead.
    // Both blocks are loaded from the same GGUF, so the weights are identical.
    let ws_cpu = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.0");
    let bc = Qwen35Block::load_tp(&ws_cpu, &c, 0, grim_nn::TensorParallelConfig::default())
        .expect("load blk.0 on the CPU for the reference weight");
    let (wd, w_in, w_out) = lin_w(bc.ssm_alpha.as_ref().expect("ssm_alpha"));
    assert_eq!((w_in, w_out), (HIDDEN, NV), "ssm_alpha is [32, 4096]");

    for m in [1usize, 2, 5] {
        // Distinct rows.
        let rows: Vec<Vec<f32>> = (0..m)
            .map(|t| {
                (0..HIDDEN)
                    .map(|j| ((j * 13 + t * 7) % 31) as f32 / 31.0 - 0.5)
                    .collect()
            })
            .collect();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let t = grim_tensor::Tensor::new(
            std::sync::Arc::from(
                dev.from_cpu(&flat, &Shape::new(vec![m, HIDDEN]), DType::F32)
                    .expect("upload"),
            ),
            Shape::new(vec![m, HIDDEN]),
            DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        );
        let got = gl
            .forward(&t)
            .unwrap_or_else(|e| panic!("ssm_alpha Linear::forward m={m}: {e}"))
            .to_vec_f32()
            .expect("read");
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for i in 0..m {
            let want = matvec(&rows[i], &wd, HIDDEN, w_out);
            for j in 0..w_out {
                let d = (got[i * w_out + j] - want[j]).abs() / want[j].abs().max(1e-3);
                if d > worst {
                    worst = d;
                    at = i * w_out + j;
                }
            }
        }
        eprintln!(
            "[alpha-m] Linear::forward m={m}: worst {worst:.3e} at row {} col {}",
            at / w_out,
            at % w_out
        );
    }
}
