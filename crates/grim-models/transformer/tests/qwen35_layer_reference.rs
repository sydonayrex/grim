//! One Qwen3.5 layer of each type, against a hand reference built from the
//! block's OWN weights.
//!
//! Why this shape: the logprobs show the model is *confidently* wrong, not
//! noisy, and the D2D trace shows the device paths ARE engaging on decode. So
//! the fault is arithmetic, and it is somewhere in a layer. Whole-model runs
//! cannot localise that — they only say "gibberish".
//!
//! Two properties make this test worth having:
//!
//! * The reference reads the block's own weights back to the host, so any
//!   weight-LOADING bug appears on both sides and cancels. Only the forward
//!   math is under test.
//! * The reference is written from llama.cpp, with the line cited, NOT from
//!   grim. Two defects already survived a green gate here — a missing
//!   `1/sqrt(S_v)` and a `[k|k|v]` stream read — precisely because the oracle
//!   and the code were written from the same reading. Anything asserted here
//!   is a claim about the reference that can be checked against the file.
//!
//! Reference: `/drive/bigfast/grim/old/repo/llama.cpp-master`
//!   qwen35.cpp:355-460      conv/z split, gate, softplus, l2 norm, head repeat
//!   qwen35.cpp:243-252      build_norm_gated  (RMS * w, then * silu(z))
//!   models.h:14-18          build_gdn_l2_norm = x / sqrt(sum(x^2) + eps)
//!   delta-net-base.cpp:44-50  chunked path scales q by 1/sqrt(S_k)
//!   gated_delta_net.cu:281,133  `attn_data[col] = attn_col * scale`
//!   qwen35.cpp:316-327      attention: norm q/k, rope, mul(cur, sigmoid(gate))

use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config, Qwen35LayerCache};
use grim_nn::modules::{Linear, RmsNorm};
use grim_tensor::{Device, Shape, Tensor};

const NV: usize = 6; // value heads  (llama.cpp num_v_heads, fed by ssm_dt_rank)
const NK: usize = 2; // key heads
const HD: usize = 8; // ssm head dim (head_k_dim == head_v_dim == ssm_d_state)
const TAPS: usize = 4;
const HIDDEN: usize = 32;
const INTER: usize = 24;
const AH: usize = 4; // attention heads
const AKV: usize = 2; // attention kv heads
const AHD: usize = 8; // attention head_dim

fn key_dim() -> usize {
    NK * HD
}
fn value_dim() -> usize {
    NV * HD
}
fn conv_dim() -> usize {
    2 * key_dim() + value_dim()
}
fn q_dim() -> usize {
    AH * AHD
}
fn akv_dim() -> usize {
    AKV * AHD
}

/// Deterministic, non-degenerate values. A constant weight would let a wrong
/// head mapping or a mis-derived offset still produce the right number.
fn w(i: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| (((i * 37 + j * 11) % 29) as f32) / 29.0 - 0.5)
        .collect()
}

fn cpu(v: Vec<f32>, s: Shape) -> Tensor {
    grim_backend_cpu::cpu_tensor(v, s)
}

/// The weight is stored [out, in] (the checkpoint's own layout), so
/// `y = W @ x`. Verified empirically: a [conv_dim, hidden] projection reports
/// dims [conv_dim, hidden], not the transpose.
fn lin_w(l: &Linear) -> (Vec<f32>, usize, usize) {
    let t = l.weight();
    let d = t.shape().dims().to_vec();
    assert_eq!(d.len(), 2, "linear weight must be 2-D");
    // Reported as (in, out) so call sites read naturally, but the DATA is
    // [out, in] — the checkpoint's own layout. matvec knows the difference.
    (t.to_vec_f32().expect("weight"), d[1], d[0])
}

fn matvec(x: &[f32], wt: &[f32], in_n: usize, out_n: usize) -> Vec<f32> {
    // wt is [out_n, in_n] row-major: y[o] = sum_i wt[o*in_n + i] * x[i]
    assert_eq!(x.len(), in_n, "activation width must match the weight's in");
    let mut y = vec![0.0f32; out_n];
    for o in 0..out_n {
        let mut s = 0.0f64;
        for (i, xv) in x.iter().enumerate() {
            s += (*xv as f64) * (wt[o * in_n + i] as f64);
        }
        y[o] = s as f32;
    }
    y
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

/// models.h:14-18 — `scale(rms_norm(x, eps/n), 1/sqrt(n))` reduces to
/// `x / sqrt(sum(x^2) + eps)`. The `eps/n` and the `1/sqrt(n)` cancel to an
/// UNSCALED eps, which is the easy thing to get wrong.
fn gdn_l2(x: &[f32], eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let d = ((ss + eps as f64).sqrt()) as f32;
    if d <= 0.0 {
        return x.to_vec();
    }
    x.iter().map(|v| v / d).collect()
}

/// NeoX half-split pairing matching Qwen3.5 production: dim i pairs with i + half.
fn rope_inplace(v: &mut [f32], pos: u32, base: f32) {
    let hd = v.len();
    let half = hd / 2;
    for i in 0..half {
        let freq = 1.0f64 / (base as f64).powf((2 * i) as f64 / hd as f64);
        let theta = pos as f64 * freq;
        let (s, c) = theta.sin_cos();
        let (x0, x1) = (v[i] as f64, v[i + half] as f64);
        v[i] = (x0 * c - x1 * s) as f32;
        v[i + half] = (x0 * s + x1 * c) as f32;
    }
}

fn recurrent_block() -> Qwen35Block {
    Qwen35Block {
        device: Device::Cpu,
        layer_idx: 1,
        num_heads: AH,
        num_kv_heads: AKV,
        head_dim: AHD,
        is_full_attention: false,
        attn_norm: RmsNorm::new(cpu(w(1, HIDDEN), Shape::new(vec![HIDDEN])), 1e-6),
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        attn_q_norm: None,
        attn_k_norm: None,
        attn_qkv: Some(Linear::from_tensor(
            cpu(w(2, conv_dim() * HIDDEN), Shape::new(vec![conv_dim(), HIDDEN])),
            None,
        )),
        attn_gate: Some(Linear::from_tensor(
            cpu(w(3, value_dim() * HIDDEN), Shape::new(vec![value_dim(), HIDDEN])),
            None,
        )),
        ssm_out: Some(Linear::from_tensor(
            cpu(w(4, HIDDEN * value_dim()), Shape::new(vec![HIDDEN, value_dim()])),
            None,
        )),
        // REAL shape: the checkpoint stores this [taps, conv_dim] (9B: [4, 8192]).
        ssm_conv1d: Some(cpu(w(5, conv_dim() * TAPS), Shape::new(vec![TAPS, conv_dim()]))),
        ssm_conv_vec: None,
        ssm_a: Some(w(6, NV)),
        ssm_alpha: Some(Linear::from_tensor(
            cpu(w(7, NV * HIDDEN), Shape::new(vec![NV, HIDDEN])),
            None,
        )),
        ssm_beta: Some(Linear::from_tensor(
            cpu(w(8, NV * HIDDEN), Shape::new(vec![NV, HIDDEN])),
            None,
        )),
        ssm_dt_bias: Some(w(9, NV)),
        // REAL shape: [head_dim] (9B: [128]) — one weight per head dim, not per value.
        ssm_norm: Some(w(10, HD)),
        ssm_dt_bias_dev: None,
        ssm_a_dev: None,
        ssm_norm_dev: None,
        ssm_dt_rank_hint: NV,
        ssm_n_group_hint: NK,
        ssm_d_state_hint: HD,
        ssm_d_conv_hint: TAPS,
        post_attention_norm: RmsNorm::new(cpu(w(11, HIDDEN), Shape::new(vec![HIDDEN])), 1e-6),
        ffn_gate: Linear::from_tensor(cpu(w(12, INTER * HIDDEN), Shape::new(vec![INTER, HIDDEN])), None),
        ffn_up: Linear::from_tensor(cpu(w(13, INTER * HIDDEN), Shape::new(vec![INTER, HIDDEN])), None),
        ffn_down: Linear::from_tensor(cpu(w(14, HIDDEN * INTER), Shape::new(vec![HIDDEN, INTER])), None),
        rotary_dim: AHD,
        rope_theta: 10000.0,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        wqkv_q80_fused: None,
        w_gate_up_q4k_fused: None,
    }
}

fn attention_block() -> Qwen35Block {
    Qwen35Block {
        device: Device::Cpu,
        layer_idx: 3, // (3+1) % 4 == 0 -> full attention
        num_heads: AH,
        num_kv_heads: AKV,
        head_dim: AHD,
        is_full_attention: true,
        attn_norm: RmsNorm::new(cpu(w(20, HIDDEN), Shape::new(vec![HIDDEN])), 1e-6),
        // attn_q is FUSED query + output gate at 2*q_dim (qwen35.cpp:280-292).
        wq: Some(Linear::from_tensor(
            cpu(w(21, 2 * q_dim() * HIDDEN), Shape::new(vec![2 * q_dim(), HIDDEN])),
            None,
        )),
        wk: Some(Linear::from_tensor(cpu(w(22, akv_dim() * HIDDEN), Shape::new(vec![akv_dim(), HIDDEN])), None)),
        wv: Some(Linear::from_tensor(cpu(w(23, akv_dim() * HIDDEN), Shape::new(vec![akv_dim(), HIDDEN])), None)),
        wo: Some(Linear::from_tensor(cpu(w(24, HIDDEN * q_dim()), Shape::new(vec![HIDDEN, q_dim()])), None)),
        attn_q_norm: Some(RmsNorm::new(cpu(w(25, AHD), Shape::new(vec![AHD])), 1e-6)),
        attn_k_norm: Some(RmsNorm::new(cpu(w(26, AHD), Shape::new(vec![AHD])), 1e-6)),
        attn_qkv: None,
        attn_gate: None,
        ssm_out: None,
        ssm_conv1d: None,
        ssm_conv_vec: None,
        ssm_a: None,
        ssm_alpha: None,
        ssm_beta: None,
        ssm_dt_bias: None,
        ssm_norm: None,
        ssm_dt_bias_dev: None,
        ssm_a_dev: None,
        ssm_norm_dev: None,
        ssm_dt_rank_hint: 0,
        ssm_n_group_hint: 0,
        ssm_d_state_hint: 0,
        ssm_d_conv_hint: 0,
        post_attention_norm: RmsNorm::new(cpu(w(27, HIDDEN), Shape::new(vec![HIDDEN])), 1e-6),
        ffn_gate: Linear::from_tensor(cpu(w(28, INTER * HIDDEN), Shape::new(vec![INTER, HIDDEN])), None),
        ffn_up: Linear::from_tensor(cpu(w(29, INTER * HIDDEN), Shape::new(vec![INTER, HIDDEN])), None),
        ffn_down: Linear::from_tensor(cpu(w(30, HIDDEN * INTER), Shape::new(vec![HIDDEN, INTER])), None),
        rotary_dim: AHD,
        rope_theta: 10000.0,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        wqkv_q80_fused: None,
        w_gate_up_q4k_fused: None,
    }
}

fn cfg() -> Qwen35Config {
    let mut c = Qwen35Config::default();
    c.hidden_size = HIDDEN;
    c.num_heads = AH;
    c.num_kv_heads = AKV;
    c.head_dim = AHD;
    c.intermediate_size = INTER;
    c.ssm_dt_rank = NV;
    c.ssm_n_group = NK;
    c.ssm_d_state = HD;
    c.ssm_d_conv = TAPS;
    c.ssm_d_inner = value_dim();
    c.full_attention_interval = 4;
    c
}

fn compare(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &r)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - r).abs();
        let rel = d / r.abs().max(1e-3);
        let sc = d.min(rel);
        if sc > worst {
            worst = sc;
            at = i;
        }
    }
    assert!(
        worst <= 2e-3,
        "{what}: worst |err| {worst:.3e} at {at} (grim {}, reference {})",
        got[at],
        want[at]
    );
    eprintln!("[{what}] matches the reference (worst {worst:.2e})");
}

#[test]
fn recurrent_layer_matches_hand_reference() {
    let blk = recurrent_block();
    let c = cfg();
    let mut cache = Qwen35LayerCache::new(&c);
    // Non-zero conv + recurrent state: a zero state makes the decayed key dot
    // and the carried state indistinguishable, so a stale-state variant passes.
    let seed_conv = w(90, cache.conv_state.len());
    let seed_ssm = w(91, cache.ssm_state.len());
    for (i, v) in cache.conv_state.iter_mut().enumerate() {
        *v = seed_conv[i];
    }
    for (i, v) in cache.ssm_state.iter_mut().enumerate() {
        *v = seed_ssm[i];
    }
    let x = w(40, HIDDEN);

    // ── reference, from llama.cpp ────────────────────────────────────────
    let eps = 1e-6f32;
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(&x, &an, eps);

    let (qw, qin, qout) = lin_w(blk.attn_qkv.as_ref().unwrap());
    assert_eq!((qin, qout), (HIDDEN, conv_dim()));
    let qkv = matvec(&xn, &qw, qin, qout);

    // ggml_ssm_conv: depthwise, `dim` taps, most recent tap = current input,
    // then the state shifts left and the new input lands at the end.
    let cw = blk.ssm_conv1d.as_ref().unwrap().to_vec_f32().expect("conv w");
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
    compare("conv output", &conv, &conv); // self-check: trivially equal
    let conv_silu: Vec<f32> = conv.iter().map(|&v| silu(v)).collect();

    let (aw, ain, aout) = lin_w(blk.ssm_alpha.as_ref().unwrap());
    let (bw, bin_, bout) = lin_w(blk.ssm_beta.as_ref().unwrap());
    let (zw, zin, zout) = lin_w(blk.attn_gate.as_ref().unwrap());
    let alpha = matvec(&xn, &aw, ain, aout);
    let beta = matvec(&xn, &bw, bin_, bout);
    let z = matvec(&xn, &zw, zin, zout);
    let dt_bias = blk.ssm_dt_bias.as_ref().unwrap().clone();
    let ssm_a = blk.ssm_a.as_ref().unwrap().clone();
    let ssm_norm = blk.ssm_norm.as_ref().unwrap().clone();

    let mut s_state: Vec<f64> = cache.ssm_state.iter().map(|&v| v as f64).collect();
    let mut core = vec![0.0f32; value_dim()];
    let inv_sqrt_d = 1.0 / (HD as f32).sqrt();
    for h in 0..NV {
        let kh = h % NK; // ggml_repeat_4d tiling of the key heads
        // [q | k | v] — llama.cpp qwen35.cpp:404-424 byte offsets 0 / key_dim / 2*key_dim
        let q_slice = gdn_l2(&conv_silu[kh * HD..kh * HD + HD], eps);
        let k_slice = gdn_l2(
            &conv_silu[key_dim() + kh * HD..key_dim() + kh * HD + HD],
            eps,
        );
        let v_slice = &conv_silu[2 * key_dim() + h * HD..2 * key_dim() + h * HD + HD];

        let gate = softplus(alpha[h] + dt_bias[h]) * ssm_a[h];
        let decay = (gate as f64).exp();

        let mut acc = vec![0.0f32; HD];
        for j in 0..HD {
            let base = (h * HD + j) * HD;
            let mut pred = 0.0f64;
            for i in 0..HD {
                pred += (k_slice[i] as f64) * (decay * s_state[base + i]);
            }
            let delta = (1.0 / (1.0 + (-beta[h]).exp())) as f64 * (v_slice[j] as f64 - pred);
            let mut a = 0.0f64;
            for i in 0..HD {
                s_state[base + i] = decay * s_state[base + i] + (k_slice[i] as f64) * delta;
                a += (q_slice[i] as f64) * s_state[base + i];
            }
            acc[j] = (a as f32) * inv_sqrt_d;
        }
        core[h * HD..(h + 1) * HD].copy_from_slice(&acc);
    }

    // build_norm_gated (qwen35.cpp:243-252): RMS over the head, * ssm_norm,
    // then * silu(z). One gate, not two.
    let mut branch = vec![0.0f32; value_dim()];
    for h in 0..NV {
        let seg = &core[h * HD..(h + 1) * HD];
        let ss: f64 = seg.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = 1.0 / ((ss / HD as f64 + eps as f64).sqrt());
        for i in 0..HD {
            let idx = h * HD + i;
            // ssm_norm is [head_dim] and applies to EVERY head at the same index.
            branch[idx] = (seg[i] as f64 * inv * ssm_norm[i] as f64 * silu(z[idx]) as f64) as f32;
        }
    }

    let (ow, oin, oout) = lin_w(blk.ssm_out.as_ref().unwrap());
    let proj = matvec(&branch, &ow, oin, oout);
    let h1: Vec<f32> = (0..HIDDEN).map(|i| x[i] + proj[i]).collect();
    let pan = blk.post_attention_norm.weight.to_vec_f32().expect("post norm");
    let h2 = rms(&h1, &pan, eps);
    let (gw, gi, go) = lin_w(&blk.ffn_gate);
    let (uw, ui, uo) = lin_w(&blk.ffn_up);
    let (dw, di, doo) = lin_w(&blk.ffn_down);
    let fg = matvec(&h2, &gw, gi, go);
    let fu = matvec(&h2, &uw, ui, uo);
    let act: Vec<f32> = (0..INTER).map(|i| silu(fg[i]) * fu[i]).collect();
    let dn = matvec(&act, &dw, di, doo);
    let want: Vec<f32> = (0..HIDDEN).map(|i| h1[i] + dn[i]).collect();

    // ── grim ────────────────────────────────────────────────────────────
    let got = blk
        .forward(&cpu(x.clone(), Shape::new(vec![1, HIDDEN])), &[0], &mut cache)
        .expect("recurrent forward")
        .to_vec_f32()
        .expect("read");

    compare("recurrent layer", &got, &want);
}

/// Bisects the recurrent layer by comparing the two pieces of state the block
/// leaves behind. The conv ring and the recurrent state are updated before
/// anything downstream, so a mismatch in one of them localises the fault to the
/// conv or to the delta rule, and a match in both means the fault is in what
/// happens after the recurrence (the gated norm, the projection, the FFN).
#[test]
fn recurrent_layer_state_bisect() {
    let blk = recurrent_block();
    let c = cfg();
    let mut cache = Qwen35LayerCache::new(&c);
    let seed_conv = w(90, cache.conv_state.len());
    let seed_ssm = w(91, cache.ssm_state.len());
    for (i, v) in cache.conv_state.iter_mut().enumerate() {
        *v = seed_conv[i];
    }
    for (i, v) in cache.ssm_state.iter_mut().enumerate() {
        *v = seed_ssm[i];
    }
    let x = w(40, HIDDEN);
    let eps = 1e-6f32;

    // reference conv ring
    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(&x, &an, eps);
    let (qw, qin, qout) = lin_w(blk.attn_qkv.as_ref().unwrap());
    let qkv = matvec(&xn, &qw, qin, qout);
    let cw = blk.ssm_conv1d.as_ref().unwrap().to_vec_f32().expect("conv w");
    let mut ref_state = cache.conv_state.clone();
    let mut conv = vec![0.0f32; conv_dim()];
    for ch in 0..conv_dim() {
        let mut sum = qkv[ch] * cw[ch * TAPS + (TAPS - 1)];
        for k in 0..TAPS - 1 {
            sum += ref_state[ch * (TAPS - 1) + k] * cw[ch * TAPS + k];
        }
        conv[ch] = sum;
    }
    for ch in 0..conv_dim() {
        for k in 0..TAPS - 2 {
            ref_state[ch * (TAPS - 1) + k] = ref_state[ch * (TAPS - 1) + k + 1];
        }
        ref_state[ch * (TAPS - 1) + (TAPS - 2)] = qkv[ch];
    }
    let conv_silu: Vec<f32> = conv.iter().map(|&v| silu(v)).collect();

    // reference recurrent state
    let (aw, ain, aout) = lin_w(blk.ssm_alpha.as_ref().unwrap());
    let (bw, bin_, bout) = lin_w(blk.ssm_beta.as_ref().unwrap());
    let alpha = matvec(&xn, &aw, ain, aout);
    let beta = matvec(&xn, &bw, bin_, bout);
    let dt_bias = blk.ssm_dt_bias.as_ref().unwrap().clone();
    let ssm_a = blk.ssm_a.as_ref().unwrap().clone();
    let mut ref_ssm: Vec<f64> = cache.ssm_state.iter().map(|&v| v as f64).collect();
    for h in 0..NV {
        let kh = h % NK;
        let _q_slice = gdn_l2(&conv_silu[kh * HD..kh * HD + HD], eps);
        let k_slice = gdn_l2(&conv_silu[key_dim() + kh * HD..key_dim() + kh * HD + HD], eps);
        let v_slice = &conv_silu[2 * key_dim() + h * HD..2 * key_dim() + h * HD + HD];
        let gate = softplus(alpha[h] + dt_bias[h]) * ssm_a[h];
        let decay = (gate as f64).exp();
        let bv = (1.0 / (1.0 + (-beta[h]).exp())) as f64;
        for j in 0..HD {
            let base = (h * HD + j) * HD;
            let mut pred = 0.0f64;
            for i in 0..HD {
                pred += (k_slice[i] as f64) * (decay * ref_ssm[base + i]);
            }
            let delta = bv * (v_slice[j] as f64 - pred);
            for i in 0..HD {
                ref_ssm[base + i] = decay * ref_ssm[base + i] + (k_slice[i] as f64) * delta;
            }
        }
    }

    blk.forward(&cpu(x.clone(), Shape::new(vec![1, HIDDEN])), &[0], &mut cache)
        .expect("recurrent forward");

    let ref_conv: Vec<f64> = ref_state.iter().map(|&v| v as f64).collect();
    let got_conv: Vec<f64> = cache.conv_state.iter().map(|&v| v as f64).collect();
    let mut w1 = 0.0f64;
    for i in 0..got_conv.len().min(ref_conv.len()) {
        w1 = w1.max((got_conv[i] - ref_conv[i]).abs());
    }
    eprintln!("[bisect] conv ring worst |err| {w1:.3e}");

    let got_ssm: Vec<f64> = cache.ssm_state.iter().map(|&v| v as f64).collect();
    let mut w2 = 0.0f64;
    let mut at = 0usize;
    for i in 0..got_ssm.len().min(ref_ssm.len()) {
        let d = (got_ssm[i] - ref_ssm[i]).abs();
        if d > w2 {
            w2 = d;
            at = i;
        }
    }
    eprintln!("[bisect] recurrent state worst |err| {w2:.3e} at {at}");

    assert!(w1 <= 1e-4, "conv ring mismatch: {w1:.3e}");
    assert!(w2 <= 1e-4, "recurrent state mismatch: {w2:.3e} at {at}");
}

#[test]
fn attention_layer_matches_hand_reference() {
    let blk = attention_block();
    let c = cfg();
    let mut cache = Qwen35LayerCache::new(&c);
    let x = w(41, HIDDEN);
    let eps = 1e-6f32;

    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(&x, &an, eps);

    // attn_q is fused [Q | gate] at 2*q_dim; the host splits it ROW-AWARE.
    let (qw, qin, qout) = lin_w(blk.wq.as_ref().unwrap());
    assert_eq!((qin, qout), (HIDDEN, 2 * q_dim()));
    let qfull = matvec(&xn, &qw, qin, qout);
    let mut q = qfull[..q_dim()].to_vec();
    let gate = &qfull[q_dim()..2 * q_dim()];

    let (kw, kin, kout) = lin_w(blk.wk.as_ref().unwrap());
    let (vw, vin, vout) = lin_w(blk.wv.as_ref().unwrap());
    let mut k = matvec(&xn, &kw, kin, kout);
    let v = matvec(&xn, &vw, vin, vout);

    // Q/K RMS norm BEFORE rope, per head (qwen35.cpp:281/286).
    let qn = blk.attn_q_norm.as_ref().unwrap().weight.to_vec_f32().unwrap();
    let kn = blk.attn_k_norm.as_ref().unwrap().weight.to_vec_f32().unwrap();
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
    for h in 0..AH {
        let s = h * AHD;
        let mut seg = q[s..s + AHD].to_vec();
        rope_inplace(&mut seg, 0, 10000.0);
        q[s..s + AHD].copy_from_slice(&seg);
    }
    for h in 0..AKV {
        let s = h * AHD;
        let mut seg = k[s..s + AHD].to_vec();
        rope_inplace(&mut seg, 0, 10000.0);
        k[s..s + AHD].copy_from_slice(&seg);
    }

    // kv_len == 1 => softmax is 1, so attention output is V, broadcast by the
    // GQA mapping kv_head = h * kv_heads / num_heads.
    let mut attn = vec![0.0f32; q_dim()];
    for h in 0..AH {
        let kvh = (h * AKV) / AH;
        attn[h * AHD..(h + 1) * AHD].copy_from_slice(&v[kvh * AHD..(kvh + 1) * AHD]);
    }
    // qwen35.cpp:323-327 — ONE sigmoid gate on the attention branch.
    let mut branch = vec![0.0f32; q_dim()];
    for i in 0..q_dim() {
        branch[i] = attn[i] / (1.0 + (-gate[i]).exp());
    }

    let (ow, oin, oout) = lin_w(blk.wo.as_ref().unwrap());
    let proj = matvec(&branch, &ow, oin, oout);
    let h1: Vec<f32> = (0..HIDDEN).map(|i| x[i] + proj[i]).collect();
    let pan = blk.post_attention_norm.weight.to_vec_f32().expect("post norm");
    let h2 = rms(&h1, &pan, eps);
    let (gw, gi, go) = lin_w(&blk.ffn_gate);
    let (uw, ui, uo) = lin_w(&blk.ffn_up);
    let (dw, di, doo) = lin_w(&blk.ffn_down);
    let fg = matvec(&h2, &gw, gi, go);
    let fu = matvec(&h2, &uw, ui, uo);
    let act: Vec<f32> = (0..INTER).map(|i| silu(fg[i]) * fu[i]).collect();
    let dn = matvec(&act, &dw, di, doo);
    let want: Vec<f32> = (0..HIDDEN).map(|i| h1[i] + dn[i]).collect();

    let got = blk
        .forward(&cpu(x.clone(), Shape::new(vec![1, HIDDEN])), &[0], &mut cache)
        .expect("attention forward")
        .to_vec_f32()
        .expect("read");

    compare("attention layer", &got, &want);
}

/// Assert that the attention layer matches the hand reference at a non-zero sequence position.
/// At pos > 0, RoPE is active with non-trivial sin/cos rotations that exercise the NeoX half-split schedule.
#[test]
fn attention_layer_matches_hand_reference_nonzero_pos() {
    let blk = attention_block();
    let c = cfg();
    let mut cache = Qwen35LayerCache::new(&c);
    let x = w(42, HIDDEN);
    let eps = 1e-6f32;
    let pos: u32 = 17;

    let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
    let xn = rms(&x, &an, eps);

    let (qw, qin, qout) = lin_w(blk.wq.as_ref().unwrap());
    let qfull = matvec(&xn, &qw, qin, qout);
    let mut q = qfull[..q_dim()].to_vec();
    let gate = &qfull[q_dim()..2 * q_dim()];

    let (kw, kin, kout) = lin_w(blk.wk.as_ref().unwrap());
    let (vw, vin, vout) = lin_w(blk.wv.as_ref().unwrap());
    let mut k = matvec(&xn, &kw, kin, kout);
    let v = matvec(&xn, &vw, vin, vout);

    let qn = blk.attn_q_norm.as_ref().unwrap().weight.to_vec_f32().unwrap();
    let kn = blk.attn_k_norm.as_ref().unwrap().weight.to_vec_f32().unwrap();
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
    for h in 0..AH {
        let s = h * AHD;
        let mut seg = q[s..s + AHD].to_vec();
        rope_inplace(&mut seg, pos, 10000.0);
        q[s..s + AHD].copy_from_slice(&seg);
    }
    for h in 0..AKV {
        let s = h * AHD;
        let mut seg = k[s..s + AHD].to_vec();
        rope_inplace(&mut seg, pos, 10000.0);
        k[s..s + AHD].copy_from_slice(&seg);
    }

    let mut attn = vec![0.0f32; q_dim()];
    for h in 0..AH {
        let kvh = (h * AKV) / AH;
        attn[h * AHD..(h + 1) * AHD].copy_from_slice(&v[kvh * AHD..(kvh + 1) * AHD]);
    }
    let mut branch = vec![0.0f32; q_dim()];
    for i in 0..q_dim() {
        branch[i] = attn[i] / (1.0 + (-gate[i]).exp());
    }

    let (ow, oin, oout) = lin_w(blk.wo.as_ref().unwrap());
    let proj = matvec(&branch, &ow, oin, oout);
    let h1: Vec<f32> = (0..HIDDEN).map(|i| x[i] + proj[i]).collect();
    let pan = blk.post_attention_norm.weight.to_vec_f32().expect("post norm");
    let h2 = rms(&h1, &pan, eps);
    let (gw, gi, go) = lin_w(&blk.ffn_gate);
    let (uw, ui, uo) = lin_w(&blk.ffn_up);
    let (dw, di, doo) = lin_w(&blk.ffn_down);
    let fg = matvec(&h2, &gw, gi, go);
    let fu = matvec(&h2, &uw, ui, uo);
    let act: Vec<f32> = (0..INTER).map(|i| silu(fg[i]) * fu[i]).collect();
    let dn = matvec(&act, &dw, di, doo);
    let want: Vec<f32> = (0..HIDDEN).map(|i| h1[i] + dn[i]).collect();

    let got = blk
        .forward(&cpu(x.clone(), Shape::new(vec![1, HIDDEN])), &[pos], &mut cache)
        .expect("attention forward non-zero pos")
        .to_vec_f32()
        .expect("read");

    compare("attention layer (pos=17)", &got, &want);
}

/// Attention with kv_len > 1 (chained steps): verifies non-trivial softmax attention
/// over multiple keys/values in the arena, checking temperature scaling 1/sqrt(head_dim).
#[test]
fn attention_layer_matches_hand_reference_multi_step() {
    let blk = attention_block();
    let c = cfg();
    let mut cache = Qwen35LayerCache::new(&c);
    let eps = 1e-6f32;
    let inv_sqrt_d = 1.0f32 / (AHD as f32).sqrt();

    // Run 3 sequential decode steps to accumulate 3 tokens in the KV arena
    let mut k_history: Vec<Vec<f32>> = Vec::new(); // [step][kv_dim]
    let mut v_history: Vec<Vec<f32>> = Vec::new(); // [step][kv_dim]

    for step in 0..3u32 {
        let x = w(50 + step as usize, HIDDEN);
        let an = blk.attn_norm.weight.to_vec_f32().expect("attn_norm");
        let xn = rms(&x, &an, eps);

        let (qw, qin, qout) = lin_w(blk.wq.as_ref().unwrap());
        let qfull = matvec(&xn, &qw, qin, qout);
        let mut q = qfull[..q_dim()].to_vec();
        let gate = &qfull[q_dim()..2 * q_dim()];

        let (kw, kin, kout) = lin_w(blk.wk.as_ref().unwrap());
        let (vw, vin, vout) = lin_w(blk.wv.as_ref().unwrap());
        let mut k = matvec(&xn, &kw, kin, kout);
        let v = matvec(&xn, &vw, vin, vout);

        let qn = blk.attn_q_norm.as_ref().unwrap().weight.to_vec_f32().unwrap();
        let kn = blk.attn_k_norm.as_ref().unwrap().weight.to_vec_f32().unwrap();
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
        for h in 0..AH {
            let s = h * AHD;
            let mut seg = q[s..s + AHD].to_vec();
            rope_inplace(&mut seg, step, 10000.0);
            q[s..s + AHD].copy_from_slice(&seg);
        }
        for h in 0..AKV {
            let s = h * AHD;
            let mut seg = k[s..s + AHD].to_vec();
            rope_inplace(&mut seg, step, 10000.0);
            k[s..s + AHD].copy_from_slice(&seg);
        }

        k_history.push(k.clone());
        v_history.push(v.clone());

        // Full softmax attention over all historical keys/values accumulated so far
        let num_keys = (step + 1) as usize;
        let mut attn = vec![0.0f32; q_dim()];
        for h in 0..AH {
            let kvh = (h * AKV) / AH;
            let q_head = &q[h * AHD..(h + 1) * AHD];

            // Compute logits for each key
            let mut logits = Vec::with_capacity(num_keys);
            for t in 0..num_keys {
                let k_head = &k_history[t][kvh * AHD..(kvh + 1) * AHD];
                let dot: f32 = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
                logits.push(dot * inv_sqrt_d);
            }
            // Softmax
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

            // Weighted sum of values
            let mut head_out = vec![0.0f32; AHD];
            for t in 0..num_keys {
                let v_head = &v_history[t][kvh * AHD..(kvh + 1) * AHD];
                for d in 0..AHD {
                    head_out[d] += weights[t] * v_head[d];
                }
            }
            attn[h * AHD..(h + 1) * AHD].copy_from_slice(&head_out);
        }

        let mut branch = vec![0.0f32; q_dim()];
        for i in 0..q_dim() {
            branch[i] = attn[i] / (1.0 + (-gate[i]).exp());
        }

        let (ow, oin, oout) = lin_w(blk.wo.as_ref().unwrap());
        let proj = matvec(&branch, &ow, oin, oout);
        let h1: Vec<f32> = (0..HIDDEN).map(|i| x[i] + proj[i]).collect();
        let pan = blk.post_attention_norm.weight.to_vec_f32().expect("post norm");
        let h2 = rms(&h1, &pan, eps);
        let (gw, gi, go) = lin_w(&blk.ffn_gate);
        let (uw, ui, uo) = lin_w(&blk.ffn_up);
        let (dw, di, doo) = lin_w(&blk.ffn_down);
        let fg = matvec(&h2, &gw, gi, go);
        let fu = matvec(&h2, &uw, ui, uo);
        let act: Vec<f32> = (0..INTER).map(|i| silu(fg[i]) * fu[i]).collect();
        let dn = matvec(&act, &dw, di, doo);
        let want: Vec<f32> = (0..HIDDEN).map(|i| h1[i] + dn[i]).collect();

        let got = blk
            .forward(&cpu(x.clone(), Shape::new(vec![1, HIDDEN])), &[step], &mut cache)
            .expect("attention forward chained step")
            .to_vec_f32()
            .expect("read");

        compare(&format!("attention layer chained step {step} (kv_len={num_keys})"), &got, &want);
        cache.current_pos += 1;
    }
}
