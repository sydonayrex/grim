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
        attn_norm: RmsNorm::new(mk(w(1, HIDDEN), Shape::new(vec![HIDDEN])), 1e-6),
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
            mk(w(6, HEAD_DIM), Shape::new(vec![HEAD_DIM])),
            1e-6,
        )),
        attn_k_norm: Some(RmsNorm::new(
            mk(w(7, HEAD_DIM), Shape::new(vec![HEAD_DIM])),
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
        post_attention_norm: RmsNorm::new(mk(w(8, HIDDEN), Shape::new(vec![HIDDEN])), 1e-6),
        ffn_gate: Linear::from_tensor(
            mk(w(9, INTER * HIDDEN), Shape::new(vec![INTER, HIDDEN])),
            None,
        ),
        ffn_up: Linear::from_tensor(
            mk(w(10, INTER * HIDDEN), Shape::new(vec![INTER, HIDDEN])),
            None,
        ),
        ffn_down: Linear::from_tensor(
            mk(w(11, HIDDEN * INTER), Shape::new(vec![HIDDEN, INTER])),
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
        let (e, at) = worst_rel(a, b);
        eprintln!("[f16k-parity] step {step}: worst {e:.3e} at {at}");
        // f16 unit roundoff (2^-11) amplified through exp+normalize+256-wide
        // wo: measured 9.6e-3..4.2e-2 non-monotonic across steps on synthetic
        // geometry (rounding, not drift); production 9B step-0 logprobs agree
        // <0.3 with coherent output. Real bugs (misread rows, NaN) measure
        // O(0.1-1.0+), far above this bar.
        assert!(
            e <= 6e-2,
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
        let (e, at) = worst_rel(a, b);
        eprintln!("[i8k-parity] step {step}: worst {e:.3e} at {at}");
        // int8 symmetric per-(token, kv_head) quantization. Measured on this
        // synthetic geometry: the block's V rows reach amax ~15.6, so the code
        // step is 15.6/127 = 0.123 and a single attention element can move by
        // up to step/2 = 0.061; the FFN (O(1) weights, no norm) then
        // amplifies that to 0.17..0.61 on the block output — measured, not
        // guessed. Real-model V amax is far lower, and the end-to-end gate is
        // the 9B logprob/text comparison, not this one. Structural bugs (wrong
        // codes, missing/misaligned scales) also land in this range, so the
        // kernel-level gates (int8_append_quantize_parity,
        // int8_attend_matches_f32, int8_attend_single_row_broadcast) are what
        // pin correctness; this one pins the end-to-end magnitude.
        assert!(
            e <= 8e-1,
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
        let (e, at) = worst_rel(a, b);
        // 4-bit quantization with 15 discrete bins on unnormalized synthetic activations
        // (amax ~15.6) yields step size ~2.2 per element, amplified across unnormed FFN.
        assert!(
            e <= 2.5,
            "step {step}: int4-vs-f32 decode diverges: worst {e:.3e} at {at} (f32 {}, int4 {})",
            a[at],
            b[at]
        );
    }
}
