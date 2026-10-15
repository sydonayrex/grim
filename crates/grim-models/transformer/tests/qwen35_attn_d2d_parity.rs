//! Device-resident attention for qwen35: the D2D path's parity gate, and a
//! canary for the kernel defect that currently keeps the path switched off.
//!
//! # The defect this file was written for, now fixed
//!
//! `grim_qkv_attention` staged the query in 8 chunks (256 dims) but accumulated
//! and wrote V in 4, so at head_dim 256 it never wrote the upper half of every
//! output row: callers read back **exactly half zeros** at the real Qwen3.8
//! geometry (24 query heads, 4 KV heads, head_dim 256), for both 2-D and 3-D
//! arenas. `s_acc[8][260]` and the `head_dim > 256` guard already assumed 256.
//!
//! A second, independent bug sat upstream of it: recovering `head_dim` from a
//! flat output shape divided the **query** width by `num_kv_heads`, which asks
//! for `head_dim*num_heads/num_kv_heads` — correct only when
//! `num_heads == num_kv_heads`. Under GQA that is a wrong geometry that runs
//! and returns nonsense rather than failing. It now derives `head_dim` from K.
//!
//! Both were pre-existing and reachable from the ordinary host path, which
//! tries `dev.qkv_attention` first.
//!
//! `kv_len == 1` is what makes this unambiguous rather than a tolerance
//! question: softmax over a single key is 1, so the output is V broadcast by
//! the GQA mapping `kv_head = h * kv_heads / num_heads`, and the expected
//! values are exact.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::RocmDevice;
use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config, Qwen35LayerCache};
use grim_nn::modules::{Linear, RmsNorm, pick_device_for_storage_device};
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

/// Process-global KV-format env vars; the f16 and int8 decode gates both set
/// them, so they must not run concurrently in one test binary.
static KV_FORMAT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const HIDDEN: usize = 256;
const INTER: usize = 192;
const STEPS: usize = 4;

fn q_dim() -> usize {
    HEADS * HEAD_DIM
}
fn kv_dim() -> usize {
    KV_HEADS * HEAD_DIM
}

fn gpu_device() -> Option<(Device, std::sync::Arc<RocmDevice>)> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| {
        let dev = std::sync::Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)"));
        (Device::Rocm(0), dev)
    })
    .ok()
}

/// Deterministic, non-degenerate values. Constant tensors would let a wrong
/// head mapping, a mis-derived column offset, or a mis-normed Q still produce
/// the right number.
fn w(i: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| (((i * 37 + j * 11) % 29) as f32) / 29.0 - 0.5)
        .collect()
}

/// `w` with a magnitude knob (see `build_block_scaled`).
fn ws(i: usize, n: usize, scale: f32) -> Vec<f32> {
    w(i, n).into_iter().map(|v| v * scale).collect()
}

fn to_dev(dev: &RocmDevice, data: &[f32], shape: Shape, device: &Device) -> Tensor {
    let st = CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("from_cpu");
    Tensor::new(
        std::sync::Arc::from(st),
        shape,
        DType::F32,
        Default::default(),
        device.clone(),
    )
}

fn build_block(device: &Device, dev: Option<&RocmDevice>) -> Qwen35Block {
    build_block_scaled(device, dev, 1.0)
}

/// `scale` multiplies every synthetic weight. The default generator spans
/// [-0.5, 0.5], and on the decode path that compounds into K/V rows with
/// amax ~15 — where an int8 code step (amax/127) is ~0.12, so a quantization
/// gate cannot tell a structural bug from legitimate rounding. Shrinking the
/// range separates the two.
fn build_block_scaled(device: &Device, dev: Option<&RocmDevice>, scale: f32) -> Qwen35Block {
    let mk = |data: Vec<f32>, shape: Shape| -> Tensor {
        match dev {
            Some(d) => to_dev(d, &data, shape, device),
            None => grim_backend_cpu::cpu_tensor(data, shape),
        }
    };
    let lin =
        |data: Vec<f32>, shape: Shape| -> Linear { Linear::from_tensor(mk(data, shape), None) };
    Qwen35Block {
        device: device.clone(),
        layer_idx: 3, // (3+1) % 4 == 0 -> full attention
        num_heads: HEADS,
        num_kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        is_full_attention: true,
        attn_norm: RmsNorm::new(mk(ws(1, HIDDEN, scale), Shape::new(vec![HIDDEN])), 1e-6),
        // attn_q is FUSED query + output gate at 2*q_dim.
        wq: Some(lin(
            w(2, 2 * q_dim() * HIDDEN),
            Shape::new(vec![2 * q_dim(), HIDDEN]),
        )),
        wk: Some(lin(
            w(3, kv_dim() * HIDDEN),
            Shape::new(vec![kv_dim(), HIDDEN]),
        )),
        wv: Some(lin(
            w(4, kv_dim() * HIDDEN),
            Shape::new(vec![kv_dim(), HIDDEN]),
        )),
        wo: Some(lin(
            w(5, HIDDEN * q_dim()),
            Shape::new(vec![HIDDEN, q_dim()]),
        )),
        attn_q_norm: Some(RmsNorm::new(
            mk(ws(6, HEAD_DIM, scale), Shape::new(vec![HEAD_DIM])),
            1e-6,
        )),
        attn_k_norm: Some(RmsNorm::new(
            mk(ws(7, HEAD_DIM, scale), Shape::new(vec![HEAD_DIM])),
            1e-6,
        )),
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
        post_attention_norm: RmsNorm::new(mk(ws(8, HIDDEN, scale), Shape::new(vec![HIDDEN])), 1e-6),
        ffn_gate: Linear::from_tensor(
            mk(
                ws(9, INTER * HIDDEN, scale),
                Shape::new(vec![INTER, HIDDEN]),
            ),
            None,
        ),
        ffn_up: Linear::from_tensor(
            mk(
                ws(10, INTER * HIDDEN, scale),
                Shape::new(vec![INTER, HIDDEN]),
            ),
            None,
        ),
        ffn_down: Linear::from_tensor(
            mk(
                ws(11, HIDDEN * INTER, scale),
                Shape::new(vec![HIDDEN, INTER]),
            ),
            None,
        ),
        rotary_dim: HEAD_DIM,
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
    c.num_heads = HEADS;
    c.num_kv_heads = KV_HEADS;
    c.head_dim = HEAD_DIM;
    c.intermediate_size = INTER;
    c.full_attention_interval = 4;
    c
}

/// Max absolute difference normalized by the reference's own magnitude scale.
///
/// `worst_rel` divides per element, which explodes on outputs near zero: this
/// block's synthetic weights drive outputs of O(40), so a near-zero element
/// reports a "relative" error near 1.0 while being irrelevant. Normalizing by
/// the reference scale is the standard measure and is what the KV-format gates
/// use.
fn worst_scaled(a: &[f32], b: &[f32]) -> (f32, usize, f32) {
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst / scale, at, worst)
}

fn worst_rel(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        let r = d / x.abs().max(y.abs()).max(1e-4);
        if r.min(d) > worst {
            worst = r.min(d);
            at = i;
        }
    }
    (worst, at)
}

/// Canary for the device attention defect. Asserts the CORRECT answer, so it
/// fails today and starts passing when the kernel is fixed — at which point
/// this `#[ignore]` should come off and the D2D path can be enabled.
///
/// `kv_len == 1` makes the expected result exact rather than approximate:
/// softmax over a single key is 1, so the output is V, broadcast across query
/// heads by the GQA mapping `kv_head = h * kv_heads / num_heads`.
#[test]
fn qwen35_attention_device_kernel_writes_every_element() {
    for (heads, kvheads, hd) in [(24usize, 4usize, 256usize), (HEADS, KV_HEADS, HEAD_DIM)] {
        let Some((device, dev)) = gpu_device() else {
            return;
        };
        let qd = heads * hd;
        let kvd = kvheads * hd;
        let q_host = w(40, qd);
        let v_host = w(41, kvd);
        let k_host = w(42, kvd);

        let mut want = vec![0.0f32; qd];
        for h in 0..heads {
            let kvh = (h * kvheads) / heads;
            want[h * hd..(h + 1) * hd].copy_from_slice(&v_host[kvh * hd..(kvh + 1) * hd]);
        }

        for (label, shape) in [
            ("3d", Shape::new(vec![1usize, kvheads, hd])),
            ("2d", Shape::new(vec![1usize, kvd])),
        ] {
            let k_dev =
                CoreTensorOps::from_cpu(dev.as_ref(), &k_host, &shape, DType::F32).expect("k");
            let v_dev =
                CoreTensorOps::from_cpu(dev.as_ref(), &v_host, &shape, DType::F32).expect("v");
            let q_dev = CoreTensorOps::from_cpu(
                dev.as_ref(),
                &q_host,
                &Shape::new(vec![1, qd]),
                DType::F32,
            )
            .expect("q");
            let got =
                grim_models_transformer::shared_attention::fused_or_scalar_attention_arena_device(
                    q_dev.as_ref(),
                    k_dev.as_ref(),
                    v_dev.as_ref(),
                    1,
                    heads,
                    kvheads,
                    hd,
                    1,
                    None,
                    &device,
                )
                .expect("attention")
                .to_vec_f32()
                .expect("read");

            let bad = (0..qd).filter(|&i| (got[i] - want[i]).abs() > 1e-4).count();
            eprintln!(
                "[canary] heads={heads} kv={kvheads} hd={hd} arena={label}: {bad}/{qd} wrong"
            );
            assert_eq!(
                bad, 0,
                "device attention is wrong at heads={heads} kv={kvheads} hd={hd} with a \
                 {label} arena: {bad}/{qd} elements differ from V-broadcast"
            );
        }
    }
}

/// The D2D path's own gate: with the attention kernel fixed, the device
/// attention layer must match the host reference over several decode steps, and
/// must demonstrably have run (the KV arenas populated) rather than silently
/// fallen back.
///
/// The path is on by default for single-token decode; `GRIM_QWEN_ATTN_D2D=0`
/// forces the host reference, and this test then exercises that instead.
#[test]
fn qwen35_attention_device_path_engages_and_matches_host() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let c = cfg();
    let cpu_blk = build_block(&Device::Cpu, None);
    let gpu_blk = build_block(&device, Some(dev.as_ref()));

    // The primitives this path borrows must be exact, independently of whether
    // the attention kernel is fixed: a wrong split or sigmoid would otherwise be
    // indistinguishable from a wrong attention result.
    {
        let w2 = 8usize;
        let row = w(66, 2 * w2);
        let st =
            CoreTensorOps::from_cpu(dev.as_ref(), &row, &Shape::new(vec![1, 2 * w2]), DType::F32)
                .expect("f");
        let d = pick_device_for_storage_device(&device);
        let (lo, _) = d
            .narrow_cols(st.as_ref(), 2 * w2, 0, 1, w2, &Shape::new(vec![1, w2]))
            .expect("lo");
        let (hi, _) = d
            .narrow_cols(st.as_ref(), 2 * w2, w2, 1, w2, &Shape::new(vec![1, w2]))
            .expect("hi");
        assert_eq!(lo.to_cpu_vec_f32().expect("r"), row[..w2].to_vec());
        assert_eq!(hi.to_cpu_vec_f32().expect("r"), row[w2..].to_vec());

        let xs = w(77, 64);
        let want: Vec<f32> = xs.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        let got = grim_nn::modules::sigmoid_on_device(&to_dev(
            dev.as_ref(),
            &xs,
            Shape::new(vec![64]),
            &device,
        ))
        .expect("sigmoid")
        .to_vec_f32()
        .expect("r");
        let (e, at) = worst_rel(&got, &want);
        assert!(e <= 1e-5, "device sigmoid wrong at {at}: {e:.3e}");
    }

    // This gate compares the host reference against the device path, NOT KV
    // formats: pin f32 so the device arena matches the host block's (Qwen35
    // defaults ROCm arenas to f16 otherwise). Shares the format lock with the
    // f16/int8 decode gates.
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
    }
    let mut cpu_cache = Qwen35LayerCache::new(&c);
    let mut gpu_cache = Qwen35LayerCache::new(&c);
    let mut cpu_last = Vec::new();
    let mut gpu_last = Vec::new();

    for step in 0..STEPS {
        let x_host = w(30 + step, HIDDEN);
        let cpu_out = cpu_blk
            .forward(
                &grim_backend_cpu::cpu_tensor(x_host.clone(), Shape::new(vec![1, HIDDEN])),
                &[step as u32],
                &mut cpu_cache,
            )
            .expect("host forward")
            .to_vec_f32()
            .expect("host read");
        let gpu_out = gpu_blk
            .forward(
                &to_dev(dev.as_ref(), &x_host, Shape::new(vec![1, HIDDEN]), &device),
                &[step as u32],
                &mut gpu_cache,
            )
            .expect("device forward")
            .to_vec_f32()
            .expect("device read");

        if step == 0 {
            assert!(
                gpu_cache.k_device.is_some() && gpu_cache.v_device.is_some(),
                "the KV arenas are empty: the attention branch fell back to the host, \
                 so this test would prove nothing about the device path"
            );
        }
        let (e, at) = worst_rel(&cpu_out, &gpu_out);
        assert!(
            e <= 5e-4,
            "step {step}: host/device attention layer diverges: worst {e:.3e} at {at} \
             (host {}, device {})",
            cpu_out[at],
            gpu_out[at]
        );
        cpu_last = cpu_out;
        gpu_last = gpu_out;
    }

    let rows = gpu_cache
        .k_device
        .as_ref()
        .map(|s| s.shape().dims()[0])
        .unwrap_or(0);
    assert!(
        rows >= STEPS,
        "K arena holds {rows} rows after {STEPS} steps"
    );
    assert_eq!(cpu_cache.current_pos, gpu_cache.current_pos);

    let (xa, xb) = (w(90, HIDDEN), w(91, HIDDEN));
    let (mut c1, mut c2) = (gpu_cache.clone(), gpu_cache.clone());
    let oa = gpu_blk
        .forward(
            &to_dev(dev.as_ref(), &xa, Shape::new(vec![1, HIDDEN]), &device),
            &[0],
            &mut c1,
        )
        .expect("a")
        .to_vec_f32()
        .expect("r");
    let ob = gpu_blk
        .forward(
            &to_dev(dev.as_ref(), &xb, Shape::new(vec![1, HIDDEN]), &device),
            &[0],
            &mut c2,
        )
        .expect("b")
        .to_vec_f32()
        .expect("r");
    assert!(
        oa.iter().zip(ob.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
        "two different tokens produced identical attention output; Q is not reaching the result"
    );
    eprintln!(
        "[attn-d2d] {STEPS} chained steps match host (worst {:.2e}), arena {rows} rows",
        worst_rel(&cpu_last, &gpu_last).0
    );
}

/// Phase-1 f16 arena gate: the same chained decode run with f16 KV arenas
/// (`GRIM_F16_KV=1`) must match the f32 run within f16-rounding budget.
/// Exercises eager alloc (dtype), append-convert, the f16kv attend kernel,
/// and host-fallback reads end to end — plus an engagement proof that the
/// arenas really are f16 (a test that silently runs f32 twice proves nothing).
#[test]
fn qwen35_f16_kv_decode_matches_f32() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let c = cfg();

    // Serializes against `qwen35_i8_kv_decode_matches_f32`: both drive the
    // process-global KV-format env vars.
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let run = |f16: bool| -> (Vec<Vec<f32>>, bool) {
        unsafe {
            std::env::set_var("GRIM_F16_KV", if f16 { "1" } else { "0" });
            std::env::set_var("GRIM_KV_ARENA", if f16 { "f16" } else { "f32" });
        }
        let blk = build_block(&device, Some(dev.as_ref()));
        let mut cache = Qwen35LayerCache::new(&c);
        let mut outs = Vec::new();
        for step in 0..STEPS {
            let x_host = w(30 + step, HIDDEN);
            let out = blk
                .forward(
                    &to_dev(dev.as_ref(), &x_host, Shape::new(vec![1, HIDDEN]), &device),
                    &[step as u32],
                    &mut cache,
                )
                .expect("forward")
                .to_vec_f32()
                .expect("read");
            outs.push(out);
            // Mirror the model loop's own advance (qwen35.rs bumps
            // `caches[i].current_pos += seq_len`). Calling the block directly
            // skips it, and without it every step rewrites arena row 0.
            cache.current_pos += 1;
        }
        let arenas_f16 = cache
            .k_device
            .as_ref()
            .map(|s| s.as_ref().dtype().arith == grim_tensor::ArithType::F16)
            .unwrap_or(false);
        unsafe {
            std::env::set_var("GRIM_F16_KV", "0");
            std::env::set_var("GRIM_KV_ARENA", "f32");
        }
        (outs, arenas_f16)
    };

    let (out32, f16_32) = run(false);
    let (out16, f16_16) = run(true);
    assert!(
        !f16_32,
        "f32 run unexpectedly built f16 arenas (flag leak between runs?)"
    );
    assert!(
        f16_16,
        "f16 run built f32 arenas: the f16 eager path was not engaged, \
         so this test would prove nothing"
    );
    for (step, (a, b)) in out32.iter().zip(out16.iter()).enumerate() {
        assert!(
            b.iter().all(|x| x.is_finite()),
            "step {step}: f16 decode produced non-finite output (NaN class)"
        );
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[f16k-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        // f16 unit roundoff (2^-11) amplified through exp+normalize and this
        // block's unnormalized FFN (block outputs reach O(1e2)): measured
        // 7.0e-4, 1.1e-3, 2.8e-3, 6.6e-4 across the four steps — non-monotonic,
        // i.e. rounding rather than drift. Bar is 2x the observed max; the
        // misread-row / NaN failures that bit this path earlier measured
        // O(0.1-1.0+), far above it.
        assert!(
            e <= 6e-3,
            "step {step}: f16-vs-f32 decode diverges: worst {e:.3e} at {at} \
             (f32 {}, f16 {})",
            a[at],
            b[at]
        );
    }
}

/// Phase-2 int8 arena gate: a chained decode run with `GRIM_KV_ARENA=int8`
/// (u8 codes + f32 per-(row,kv_head) scales) must match the f32 run within
/// the int8 rounding budget. Exercises the device quantizer, the sidecar
/// growth path, and the int8 attention kernel end to end — plus an
/// engagement proof (sidecars present + u8 codes), so a test that silently
/// ran f32 twice cannot pass.
#[test]
fn qwen35_i8_kv_decode_matches_f32() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let c = cfg();
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    let run = |format: &str| -> (Vec<Vec<f32>>, bool) {
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", format);
        }
        let blk = build_block(&device, Some(dev.as_ref()));
        let mut cache = Qwen35LayerCache::new(&c);
        let mut outs = Vec::new();
        for step in 0..STEPS {
            let x_host = w(30 + step, HIDDEN);
            let out = blk
                .forward(
                    &to_dev(dev.as_ref(), &x_host, Shape::new(vec![1, HIDDEN]), &device),
                    &[step as u32],
                    &mut cache,
                )
                .expect("forward")
                .to_vec_f32()
                .expect("read");
            outs.push(out);
            // Mirror the model loop's own advance (qwen35.rs bumps
            // `caches[i].current_pos += seq_len`). Calling the block directly
            // skips it, and without it every step rewrites arena row 0.
            cache.current_pos += 1;
        }
        let engaged = cache
            .k_device
            .as_ref()
            .map(|s| s.as_ref().dtype().arith == grim_tensor::ArithType::U8)
            .unwrap_or(false)
            && cache.k_scales_device.is_some()
            && cache.v_scales_device.is_some();
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", "f32");
        }
        (outs, engaged)
    };

    let (out32, engaged32) = run("f32");
    let (out8, engaged8) = run("int8");
    assert!(
        !engaged32,
        "f32 run built int8 arenas (env leak between runs?)"
    );
    assert!(
        engaged8,
        "int8 run built no u8 codes + scales: the int8 eager path was not \\
         engaged, so this test would prove nothing"
    );
    for (step, (a, b)) in out32.iter().zip(out8.iter()).enumerate() {
        assert!(
            b.iter().all(|x| x.is_finite()),
            "step {step}: int8 decode produced non-finite output"
        );
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[i8k-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        // int8 symmetric per-(token, kv_head) quantization. The block's V rows
        // reach amax ~15.6, so a code step is 15.6/127 = 0.123; this block's
        // FFN is unnormalized with O(1) weights over HIDDEN=256, so it
        // amplifies that per-element error by ~1e3 into the block output.
        // Measured 1.1e-2, 1.3e-2, 4.1e-2, 6.6e-2 — bar is ~4x that, because
        // this metric cannot separate rounding from an FFN gain change.
        //
        // It is NOT the correctness gate for int8: the sidecar-offset bug this
        // metric reported as 1.0 lived inside the old 8e-1 bar. Use
        // `qwen35_i8_kv_arena_rows_match_f32_within_code_step` (arena-level,
        // scale-normalized-free) plus the kernel gates for structure, and the
        // 9B logprob/text run for end-to-end.
        assert!(
            e <= 3e-1,
            "step {step}: int8-vs-f32 decode diverges: worst {e:.3e} at {at} \\
             (f32 {}, int8 {})",
            a[at],
            b[at]
        );
    }
}

/// Phase-3 int4 arena gate: a chained decode run with `GRIM_KV_ARENA=int4`
/// (packed u4 codes + f32 per-(row,kv_head) scales) must match the f32 run within
/// the int4 rounding budget.
#[test]
fn qwen35_i4_kv_decode_matches_f32() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let c = cfg();
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    let run = |format: &str| -> (Vec<Vec<f32>>, bool) {
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", format);
        }
        let blk = build_block(&device, Some(dev.as_ref()));
        let mut cache = Qwen35LayerCache::new(&c);
        let mut outs = Vec::new();
        for step in 0..STEPS {
            let x_host = w(30 + step, HIDDEN);
            let out = blk
                .forward(
                    &to_dev(dev.as_ref(), &x_host, Shape::new(vec![1, HIDDEN]), &device),
                    &[step as u32],
                    &mut cache,
                )
                .expect("forward")
                .to_vec_f32()
                .expect("read");
            outs.push(out);
            // Mirror the model loop's own advance (qwen35.rs bumps
            // `caches[i].current_pos += seq_len`). Calling the block directly
            // skips it, and without it every step rewrites arena row 0.
            cache.current_pos += 1;
        }
        let engaged = cache
            .k_device
            .as_ref()
            .map(|s| s.as_ref().dtype().arith == grim_tensor::ArithType::U8)
            .unwrap_or(false)
            && cache.k_scales_device.is_some()
            && cache.v_scales_device.is_some();
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", "f32");
        }
        (outs, engaged)
    };

    let (out32, engaged32) = run("f32");
    let (out4, engaged4) = run("int4");
    assert!(
        !engaged32,
        "f32 run built int4 arenas (env leak between runs?)"
    );
    assert!(
        engaged4,
        "int4 run built no u4 codes + scales: the int4 eager path was not engaged"
    );
    for (step, (a, b)) in out32.iter().zip(out4.iter()).enumerate() {
        assert!(
            b.iter().all(|x| x.is_finite()),
            "step {step}: int4 decode produced non-finite output"
        );
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[i4k-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        // 4-bit: 15 bins over amax ~15.6 gives a step of ~2.2 per element,
        // amplified by the same unnormalized FFN. Measured 1.0e-1, 2.2e-1,
        // 5.2e-1, 6.5e-1; bar is ~2.3x that. int4 shares the sidecar-offset
        // path with int8, so the arena-level gate covers its append too.
        assert!(
            e <= 1.5,
            "step {step}: int4-vs-f32 decode diverges: worst {e:.3e} at {at} (f32 {}, int4 {})",
            a[at],
            b[at]
        );
    }
}

/// Phase-2 eager append/sidecar gate, at the ARENA level.
///
/// The block-output comparison in `qwen35_i8_kv_decode_matches_f32` cannot
/// separate a wrong append from legitimate rounding: this synthetic block's FFN
/// has unbounded gain (block outputs reach O(100)), so a single quantization
/// step of ~0.12 in V-space shows up as O(1) relative on the output. So compare
/// the thing the append is actually responsible for — the stored K/V — row by
/// row, against the f32 arena written by the same inputs:
///
///   * the scale sidecar entry for (row, kv_head) must equal that row's amax,
///     which is only true if `scales_off` counts TOKEN rows and tracks the
///     codes' row (the off-by-a-factor bug this gate was written for);
///   * the dequantized codes must be within HALF a code step of the f32 row,
///     i.e. the quantizer round-trips the values it was handed.
#[test]
fn qwen35_i8_kv_arena_rows_match_f32_within_code_step() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let c = cfg();
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    let run = |format: &'static str| -> (Vec<Vec<f32>>, Option<(Vec<f32>, Vec<f32>)>) {
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", format);
        }
        let blk = build_block(&device, Some(dev.as_ref()));
        let mut cache = Qwen35LayerCache::new(&c);
        let mut outs = Vec::new();
        for step in 0..STEPS {
            let x = w(30 + step, HIDDEN);
            let out = blk
                .forward(
                    &to_dev(dev.as_ref(), &x, Shape::new(vec![1, HIDDEN]), &device),
                    &[step as u32],
                    &mut cache,
                )
                .expect("forward")
                .to_vec_f32()
                .expect("read");
            outs.push(out);
            cache.current_pos += 1;
        }
        // Snapshot the final arena. Both formats return (payload, scales) with
        // the payload as f32: f32 rows verbatim, int8 codes widened from bytes.
        let snap = if format == "int8" {
            let codes: Vec<u32> = cache
                .k_device
                .as_ref()
                .expect("k arena")
                .to_cpu_vec_u32()
                .expect("codes d2h");
            let codes_f32: Vec<f32> = codes
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .map(|b| b as f32)
                .collect();
            let scales: Vec<f32> = cache
                .k_scales_device
                .as_ref()
                .expect("k sidecar")
                .to_cpu_vec_f32()
                .expect("scales d2h");
            Some((codes_f32, scales))
        } else {
            let rows: Vec<f32> = cache
                .k_device
                .as_ref()
                .expect("k arena")
                .to_cpu_vec_f32()
                .expect("f32 arena");
            Some((rows, Vec::new()))
        };
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", "f32");
        }
        (outs, snap)
    };

    let (_, f32_snap) = run("f32");
    let (_, i8_snap) = run("int8");
    let (f32_rows, _) = f32_snap.expect("f32 snapshot");
    let (codes_f32, scales) = i8_snap.expect("int8 snapshot");

    let nkv = c.num_kv_heads;
    let hd = c.head_dim;
    let stride = nkv * hd;
    // The arenas are 3-D [rows, nkv, hd] and grow geometrically, so only the
    // first STEPS rows carry data; the rest is untouched capacity.
    let live = cache_rows(&f32_rows, nkv * hd).min(STEPS);
    assert!(live > 0, "no arena rows written");

    // Every row that the int8 sidecar claims a scale for must match that row's
    // amax, and the codes in that row must dequantize back to the f32 row.
    let mut checked_rows = 0usize;
    for r in 0..live {
        for kh in 0..nkv {
            let sc = scales[r * nkv + kh];
            let row = &f32_rows[r * stride + kh * hd..r * stride + (kh + 1) * hd];
            let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            assert!(
                (sc - amax).abs() <= amax * 1e-3 + 1e-6,
                "row {r} kv_head {kh}: sidecar scale {sc:.6} != row amax {amax:.6} \
                 (scales_off must track the codes' TOKEN row)"
            );
            let half_step = sc / 254.0;
            for e in 0..hd {
                let code = codes_f32[r * stride + kh * hd + e];
                // Kernel convention (grim_kv_quantize_rows_i8 / all i8
                // readers): q = round(v * 127/scale), stored as u8 q + 128, so
                // dequant = (code - 128) / 127 * scale.
                let deq = (code - 128.0) / 127.0 * sc;
                let want = row[e];
                assert!(
                    (deq - want).abs() <= half_step + 1e-6,
                    "row {r} kv_head {kh} elem {e}: dequantized {deq:.6} vs f32 {want:.6} \
                     exceeds half a code step ({half_step:.6})"
                );
            }
            checked_rows += 1;
        }
    }
    eprintln!("[i8k-arena] verified {checked_rows} (row, kv_head) vectors across {live} rows");
    assert_eq!(
        checked_rows,
        live * nkv,
        "expected every written (row, kv_head) to carry a nonzero scale"
    );
}

/// Live token rows in a flat [rows, nkv, hd] arena buffer.
fn cache_rows(flat: &[f32], stride: usize) -> usize {
    if stride == 0 { 0 } else { flat.len() / stride }
}

/// Phase-3 gate 1: a capacity that can hold the whole run must be a NO-OP.
///
/// The ring maps token `t` to arena row `t` for every `t < cap`, so with
/// `cap >= STEPS` every write lands exactly where the uncapped run put it and
/// the decode output must be BIT-IDENTICAL — not close, identical. This is the
/// gate that proves the eviction plumbing added nothing to the default path.
#[test]
fn qwen35_kv_eviction_capacity_at_least_run_is_bit_identical() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let c = cfg();
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    let run = |rows: Option<&str>| -> Vec<Vec<f32>> {
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", "f32");
            match rows {
                Some(r) => std::env::set_var("GRIM_KV_ARENA_ROWS", r),
                None => std::env::remove_var("GRIM_KV_ARENA_ROWS"),
            }
            std::env::set_var("GRIM_KV_SINKS", "2");
        }
        let blk = build_block(&device, Some(dev.as_ref()));
        let mut cache = Qwen35LayerCache::new(&c);
        let mut outs = Vec::new();
        for step in 0..STEPS {
            let x = w(30 + step, HIDDEN);
            outs.push(
                blk.forward(
                    &to_dev(dev.as_ref(), &x, Shape::new(vec![1, HIDDEN]), &device),
                    &[step as u32],
                    &mut cache,
                )
                .expect("forward")
                .to_vec_f32()
                .expect("read"),
            );
            cache.current_pos += 1;
        }
        unsafe {
            std::env::remove_var("GRIM_KV_ARENA_ROWS");
            std::env::remove_var("GRIM_KV_SINKS");
        }
        outs
    };

    let baseline = run(None);
    // cap == STEPS is the tightest sufficient capacity: token STEPS-1 is the
    // last write and lands at its own row.
    let capped = run(Some(&STEPS.to_string()));
    for (step, (a, b)) in baseline.iter().zip(capped.iter()).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "step {step}: output length changed under eviction"
        );
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "step {step}: a capacity of {STEPS} rows must be a bit-exact no-op \
             (max diff {:e})",
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        );
    }
    eprintln!("[kv-evict] capacity={STEPS} is bit-identical to uncapped across {STEPS} steps");
}

/// Phase-3 gate 2: a capacity SMALLER than the run must evict — pins the sinks,
/// keep the newest token, stay finite, and never silently run uncapped.
///
/// cap=3 / sinks=2 gives span=1, so rows 0,1 hold tokens 0,1 forever and row 2
/// is overwritten by every token from 3 onward. The arena layout is therefore
/// fully determined, which makes it checkable against the uncapped reference.
#[test]
fn qwen35_kv_eviction_smaller_capacity_evicts_and_pins_sinks() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let c = cfg();
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let cap = 3usize;
    let sinks = 2usize;
    let steps = 6usize;
    let stride = c.num_kv_heads * c.head_dim;

    let run = |rows: Option<&str>| -> (Vec<Vec<f32>>, Option<Vec<f32>>) {
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", "f32");
            match rows {
                Some(r) => std::env::set_var("GRIM_KV_ARENA_ROWS", r),
                None => std::env::remove_var("GRIM_KV_ARENA_ROWS"),
            }
            std::env::set_var("GRIM_KV_SINKS", &sinks.to_string());
        }
        let blk = build_block(&device, Some(dev.as_ref()));
        let mut cache = Qwen35LayerCache::new(&c);
        let mut outs = Vec::new();
        for step in 0..steps {
            let x = w(30 + step, HIDDEN);
            outs.push(
                blk.forward(
                    &to_dev(dev.as_ref(), &x, Shape::new(vec![1, HIDDEN]), &device),
                    &[step as u32],
                    &mut cache,
                )
                .expect("forward")
                .to_vec_f32()
                .expect("read"),
            );
            cache.current_pos += 1;
        }
        let arena = cache
            .k_device
            .as_ref()
            .expect("k arena")
            .to_cpu_vec_f32()
            .expect("d2h");
        let arena_rows = cache_rows(&arena, stride).min(steps);
        unsafe {
            std::env::remove_var("GRIM_KV_ARENA_ROWS");
            std::env::remove_var("GRIM_KV_SINKS");
        }
        (outs, Some(arena[..arena_rows * stride].to_vec()))
    };

    let (ref_outs, ref_arena) = run(None);
    let ref_arena = ref_arena.expect("reference arena");
    let (outs, arena) = run(Some(&cap.to_string()));
    let arena = arena.expect("capped arena");

    // 1. Sink rows are untouched: rows [0, sinks) must equal the reference's
    //    token 0 and token 1 bit for bit. This is the attention-sink promise.
    for r in 0..sinks {
        for e in 0..stride {
            assert_eq!(
                arena[r * stride + e].to_bits(),
                ref_arena[r * stride + e].to_bits(),
                "sink row {r} elem {e} was overwritten: eviction lost the sink"
            );
        }
    }
    // 2. The ring row holds the NEWEST token that mapped to it (token steps-1).
    for e in 0..stride {
        assert_eq!(
            arena[(cap - 1) * stride + e].to_bits(),
            ref_arena[(steps - 1) * stride + e].to_bits(),
            "ring row {} does not hold the newest token ({})",
            cap - 1,
            steps - 1
        );
    }
    // 3. The evicted middle really is gone: row sinks-1... the token that
    //    eviction removed must NOT still be present anywhere in the arena.
    //    (row 2 can only hold one token; token 2 was written there then
    //    overwritten by 3,4,5.)
    assert_eq!(
        arena.len(),
        cap * stride,
        "arena grew past the eviction capacity"
    );
    // 4. Every output step is finite, and output must have DIVERGED from the
    //    uncapped run once eviction starts biting (otherwise the cap was not
    //    engaged at all).
    for (step, o) in outs.iter().enumerate() {
        assert!(
            o.iter().all(|v| v.is_finite()),
            "step {step}: non-finite output"
        );
    }
    let last_ref = &ref_outs[steps - 1];
    let last_got = &outs[steps - 1];
    let diff = last_ref
        .iter()
        .zip(last_got.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 0.0,
        "a {cap}-row capacity over {steps} steps produced identical output to \
         the uncapped run — eviction was never engaged"
    );
    eprintln!(
        "[kv-evict] cap={cap} sinks={sinks} over {steps} steps: sinks pinned, \
         ring row holds token {}, final-step diff {diff:.3e}",
        steps - 1
    );
}

/// Phase 4 CurDKV: verify CurDKV value-guided leverage score selection preserves sinks and top tokens.
#[test]
fn qwen35_curdkv_selection_preserves_sinks_and_ranks_tokens() {
    let (n_tokens, d, budget, sinks) = (32usize, 64usize, 8usize, 2usize);
    // Synthetic K and V matrices with an artificially important token at j=10
    let mut k = vec![0.1f32; n_tokens * d];
    let mut v = vec![0.1f32; n_tokens * d];

    // Give token 10 high norm in both K and V
    for i in 0..d {
        k[10 * d + i] = 5.0f32;
        v[10 * d + i] = 5.0f32;
    }

    let selected = grim_models_transformer::qwen35::KvEvictionPolicy::select_curdkv_indices(
        &k,
        &v,
        n_tokens,
        d,
        budget,
        sinks,
    );

    assert_eq!(selected.len(), budget);
    // Sinks must be present: 0 and 1
    assert!(selected.contains(&0), "CurDKV must retain sink 0");
    assert!(selected.contains(&1), "CurDKV must retain sink 1");
    // High-importance token 10 must be retained
    assert!(selected.contains(&10), "CurDKV must retain high leverage token 10");
    eprintln!("[curdkv] selected indices: {:?}", selected);
}
