//! The KDA decode path must produce the SAME numbers on device as on the host,
//! and must actually take the device path when it can.
//!
//! `gated_delta_net_forward_d2d` returns `Ok(false)` — declining, without
//! touching state — for prefill, for a missing conv weight, for a quantized
//! conv weight, or for any geometry that does not line up. That is the safe
//! default, but it is also a silent one: a block whose `attn_gate` or conv
//! weight is slightly mis-shaped would fall back to the host reference and
//! still produce correct output, forever, with the state never leaving VRAM
//! and none of the transfers removed. So this asserts BOTH directions:
//!
//! * on a GPU-resident block the device state is populated and the host state
//!   stays zero (the paths are alternatives, not a mirror), and
//! * the layer output matches the host reference on the same weights.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::RocmDevice;
use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config, Qwen35LayerCache};
use grim_nn::modules::{Linear, RmsNorm};
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

/// KDA geometry, shrunk in head count but exact in the relationships the op
/// derives its offsets from: conv width = 2*key_dim + value_dim, state square
/// per head, and the 3:1 value:key ratio the reference broadcasts with.
const NV: usize = 6;
const NK: usize = 2;
const HD: usize = 8;
const TAPS: usize = 4;
const HIDDEN: usize = 16;

fn key_dim() -> usize {
    NK * HD
}
fn value_dim() -> usize {
    NV * HD
}
fn conv_dim() -> usize {
    2 * key_dim() + value_dim()
}

fn gpu_device() -> Option<(Device, std::sync::Arc<RocmDevice>)> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    let d = std::panic::catch_unwind(|| {
        let dev = std::sync::Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)"));
        (Device::Rocm(0), dev)
    })
    .ok()?;
    Some(d)
}

/// Deterministic, non-degenerate weights. Constant tensors would let a wrong
/// head mapping or a mis-derived offset still produce the right number.
fn w(i: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| (((i * 37 + j * 11) % 29) as f32) / 29.0 - 0.5)
        .collect()
}

fn to_dev(dev: &RocmDevice, data: &[f32], shape: Shape, device: &Device) -> Tensor {
    let storage = CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("from_cpu");
    Tensor::new(
        std::sync::Arc::from(storage),
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
        layer_idx: 0,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: HD,
        is_full_attention: false,
        attn_norm: RmsNorm::new(mk(w(1, HIDDEN), Shape::new(vec![HIDDEN])), 1e-6),
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        attn_q_norm: None,
        attn_k_norm: None,
        attn_qkv: Some(lin(
            w(2, conv_dim() * HIDDEN),
            Shape::new(vec![conv_dim(), HIDDEN]),
        )),
        // z must be at least value_dim wide; the real model loads this at
        // `q_dim.max(value_dim)` because the tensor serves both layer types.
        attn_gate: Some(lin(
            w(3, value_dim() * HIDDEN),
            Shape::new(vec![value_dim(), HIDDEN]),
        )),
        ssm_out: Some(lin(
            w(4, HIDDEN * value_dim()),
            Shape::new(vec![HIDDEN, value_dim()]),
        )),
        // Present and f32, which is what the device path requires.
        ssm_conv1d: Some(mk(
            w(5, conv_dim() * TAPS),
            Shape::new(vec![TAPS, conv_dim()]),
        )),
        ssm_conv_vec: None,
        ssm_a: Some(w(6, NV)),
        ssm_alpha: Some(lin(w(7, NV * HIDDEN), Shape::new(vec![NV, HIDDEN]))),
        ssm_beta: Some(lin(w(8, NV * HIDDEN), Shape::new(vec![NV, HIDDEN]))),
        ssm_dt_bias: Some(w(9, NV)),
        // The real width, and the one the checkpoint actually stores:
        // `ssm_norm.weight` is `ssm_d_state` (HD) elements, indexed by position
        // WITHIN a head by both host reference and both device kernels. This
        // fixture was `value_dim()` wide, which is 3x too wide for this
        // geometry (NV=6, NK=2) and so let the D2D guard's `ssm_norm.len() <
        // value_dim` requirement pass in the test while failing on every real
        // checkpoint — a survivor that hid the bug for the life of the path.
        ssm_norm: Some(w(10, HD)),
        ssm_dt_rank_hint: NV,
        ssm_n_group_hint: NK,
        ssm_d_state_hint: HD,
        ssm_d_conv_hint: TAPS,
        post_attention_norm: RmsNorm::new(mk(w(11, HIDDEN), Shape::new(vec![HIDDEN])), 1e-6),
        ffn_gate: Linear::from_tensor(
            mk(w(12, HIDDEN * HIDDEN), Shape::new(vec![HIDDEN, HIDDEN])),
            None,
        ),
        ffn_up: Linear::from_tensor(
            mk(w(13, HIDDEN * HIDDEN), Shape::new(vec![HIDDEN, HIDDEN])),
            None,
        ),
        ffn_down: Linear::from_tensor(
            mk(w(14, HIDDEN * HIDDEN), Shape::new(vec![HIDDEN, HIDDEN])),
            None,
        ),
        rotary_dim: HD,
        rope_theta: 10000.0,
        hidden_size: HIDDEN,
        intermediate_size: HIDDEN,
        wqkv_q80_fused: None,
        w_gate_up_q4k_fused: None,
    }
}

fn cfg() -> Qwen35Config {
    let mut c = Qwen35Config::default();
    c.hidden_size = HIDDEN;
    c.ssm_dt_rank = NV;
    c.ssm_n_group = NK;
    c.ssm_d_state = HD;
    c.ssm_d_conv = TAPS;
    c.ssm_d_inner = value_dim();
    c.intermediate_size = HIDDEN;
    c.num_heads = 1;
    c.num_kv_heads = 1;
    c.head_dim = HD;
    c
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn qwen35_kda_device_path_engages_and_matches_host() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let c = cfg();

    let cpu_blk = build_block(&Device::Cpu, None);
    let gpu_blk = build_block(&device, Some(dev.as_ref()));

    let x_host: Vec<f32> = w(20, HIDDEN);
    let x_dev = to_dev(dev.as_ref(), &x_host, Shape::new(vec![1, HIDDEN]), &device);

    let mut cpu_cache = Qwen35LayerCache::new(&c);
    let mut gpu_cache = Qwen35LayerCache::new(&c);

    let cpu_out = cpu_blk
        .forward(
            &grim_backend_cpu::cpu_tensor(x_host.clone(), Shape::new(vec![1, HIDDEN])),
            &[0],
            &mut cpu_cache,
        )
        .expect("host forward")
        .to_vec_f32()
        .expect("host read");

    let gpu_out = gpu_blk
        .forward(&x_dev, &[0], &mut gpu_cache)
        .expect("device forward")
        .to_vec_f32()
        .expect("device read");

    // The device path must have ENGAGED. Without this the parity assert below
    // would pass even if every call silently fell back to the host reference.
    assert!(
        gpu_cache.ssm_state_dev.is_some(),
        "ssm_state_dev is None: the KDA layer fell back to the host path, so this \
         test would prove nothing about the device path"
    );
    assert!(
        gpu_cache.conv_state_dev.is_some(),
        "conv_state_dev is None: the short conv never ran on device"
    );
    assert!(
        gpu_cache.ssm_state.iter().all(|v| *v == 0.0),
        "the host ssm_state must stay untouched when the device path owns the state"
    );
    assert!(
        cpu_cache.ssm_state.iter().any(|v| *v != 0.0),
        "the host reference must still advance its own state"
    );

    // And the two must agree.
    assert_eq!(cpu_out.len(), gpu_out.len(), "output length");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&a, &b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        let diff = (a - b).abs();
        let rel = diff / a.abs().max(b.abs()).max(1e-4);
        let score = diff.min(rel);
        if score > worst {
            worst = score;
            worst_at = i;
        }
    }
    assert!(
        worst <= 2e-3,
        "host/device KDA layer output diverges: worst |err| {worst:.3e} at {worst_at} \
         (host {}, device {})",
        cpu_out[worst_at],
        gpu_out[worst_at]
    );
    eprintln!("[kda-d2d] device path engaged; layer output matches host (worst {worst:.2e})");

    // A second token must advance the DEVICE state, not silently reuse step 0.
    let x2_host: Vec<f32> = w(21, HIDDEN);
    let x2 = to_dev(dev.as_ref(), &x2_host, Shape::new(vec![1, HIDDEN]), &device);
    let gpu_out2 = gpu_blk
        .forward(&x2, &[1], &mut gpu_cache)
        .expect("device forward 2")
        .to_vec_f32()
        .expect("device read 2");
    let cpu_out2 = cpu_blk
        .forward(
            &grim_backend_cpu::cpu_tensor(x2_host, Shape::new(vec![1, HIDDEN])),
            &[1],
            &mut cpu_cache,
        )
        .expect("host forward 2")
        .to_vec_f32()
        .expect("host read 2");

    let differs = gpu_out
        .iter()
        .zip(gpu_out2.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differs,
        "the second decode step returned the first step's output"
    );

    let mut worst2 = 0.0f32;
    let mut worst2_at = 0usize;
    for (i, (&a, &b)) in cpu_out2.iter().zip(gpu_out2.iter()).enumerate() {
        let diff = (a - b).abs();
        let rel = diff / a.abs().max(b.abs()).max(1e-4);
        let score = diff.min(rel);
        if score > worst2 {
            worst2 = score;
            worst2_at = i;
        }
    }
    assert!(
        worst2 <= 2e-3,
        "second step diverges: worst |err| {worst2:.3e} at {worst2_at} \
         (host {}, device {})",
        cpu_out2[worst2_at],
        gpu_out2[worst2_at]
    );
    eprintln!("[kda-d2d] second chained step matches host (worst {worst2:.2e})");
}
