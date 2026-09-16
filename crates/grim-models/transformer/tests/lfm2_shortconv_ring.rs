//! S1 (PLAN-kernel-fusion): the ShortConv device ring buffer (device-resident
//! conv state updated in place) must evolve the state byte-for-reason vs the
//! host-loop reference, across 3 successive decode steps on ROCm.
//!
//! Gated: GRIM_GPU_TEST=1 + ROCm device present (house rule).

use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::device::util::gpu_test_enabled;
use grim_models_transformer::lfm2::{Lfm2Block, Lfm2LayerCache};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

fn tensor(dev: &RocmDevice, ordinal: usize, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        Device::Rocm(ordinal),
    )
}

fn cpu_t(data: Vec<f32>, shape: Shape) -> Tensor {
    grim_backend_cpu::cpu_tensor(data, shape)
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) % 2000) as f32 - 1000.0) / 1000.0 * 0.4
        })
        .collect()
}

fn lin_rocm(dev: &RocmDevice, ordinal: usize, w: Vec<f32>, out_dim: usize, in_dim: usize) -> Linear {
    Linear::from_tensor(tensor(dev, ordinal, w, Shape::new(vec![out_dim, in_dim])), None)
}

fn norm_rocm(dev: &RocmDevice, ordinal: usize, dim: usize) -> RmsNorm {
    RmsNorm {
        weight: tensor(dev, ordinal, vec![1.0f32; dim], Shape::new(vec![dim])),
        eps: 1e-5,
    }
}

fn shortconv_block(dev: &RocmDevice, ordinal: usize, hidden: usize, l_cache: usize) -> Lfm2Block {
    Lfm2Block {
        attn_norm: norm_rocm(dev, ordinal, hidden),
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        attn_q_norm: None,
        attn_k_norm: None,
        wqkv_codes: None,
        wqkv_exps: None,
        gamma_q: None,
        gamma_k: None,
        wqkv_q80_fused: None,
        w_gate_up_q80_fused: None,
        shortconv_in_proj: Some(lin_rocm(dev, ordinal, rand_vec(3 * hidden * hidden, 1), 3 * hidden, hidden)),
        shortconv_conv: Some(tensor(
            dev,
            ordinal,
            rand_vec(hidden * l_cache, 2),
            Shape::new(vec![hidden, 1, l_cache]),
        )),
        shortconv_conv_vec: Some(rand_vec(hidden * l_cache, 2)),
        shortconv_out_proj: Some(lin_rocm(dev, ordinal, rand_vec(hidden * hidden, 3), hidden, hidden)),
        ffn_norm: norm_rocm(dev, ordinal, hidden),
        ffn_gate: lin_rocm(dev, ordinal, rand_vec(hidden * hidden, 4), hidden, hidden),
        ffn_up: lin_rocm(dev, ordinal, rand_vec(hidden * hidden, 5), hidden, hidden),
        ffn_down: lin_rocm(dev, ordinal, rand_vec(hidden * hidden, 6), hidden, hidden),
        ffn_gate_inp: None,
        ffn_gate_exps: None,
        ffn_up_exps: None,
        ffn_down_exps: None,
        ffn_exp_probs_b: None,
        is_moe: false,
        n_expert: 0,
        n_expert_used: 1,
        charon_cache: grim_models_transformer::shared_moe::CharonCache::new(),
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        rope_theta: 10000.0,
        eps: 1e-5,
    }
}

#[test]
fn shortconv_device_ring_matches_host_reference() {
    if !gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1");
        return;
    }
    if RocmDevice::probe_one(0).unwrap_or(false) == false {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let dev = RocmDevice::shared(0);
    let hidden = 64usize;
    let l_cache = 3usize;
    let steps = 5usize;
    let x_data = rand_vec(steps * hidden, 42);

    // GPU path: 5 single-token calls; the device ring holds state between them.
    let block_gpu = shortconv_block(&dev, 0, hidden, l_cache);
    let mut cache_gpu: Option<Lfm2LayerCache> = None;
    let mut out_gpu = Vec::new();
    for t in 0..steps {
        let x = tensor(
            &dev,
            0,
            x_data[t * hidden..(t + 1) * hidden].to_vec(),
            Shape::new(vec![1, hidden]),
        );
        out_gpu.extend(
            block_gpu
                .forward(&x, &mut cache_gpu)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
        );
    }

    // CPU reference: same weights pulled back to host into a CPU block.
    // ShortConv reference math = `full_ref` in
    // `shortconv_decode_matches_prefill` (host loop from block weights).
    let block_cpu = Lfm2Block {
        attn_norm: RmsNorm {
            weight: cpu_t(vec![1.0f32; hidden], Shape::new(vec![hidden])),
            eps: 1e-5,
        },
        wq: None,
        wk: None,
        wv: None,
        wo: None,
        attn_q_norm: None,
        attn_k_norm: None,
        wqkv_codes: None,
        wqkv_exps: None,
        gamma_q: None,
        gamma_k: None,
        wqkv_q80_fused: None,
        w_gate_up_q80_fused: None,
        shortconv_in_proj: Some(Linear::from_tensor(
            cpu_t(rand_vec(3 * hidden * hidden, 1), Shape::new(vec![3 * hidden, hidden])),
            None,
        )),
        shortconv_conv: Some(cpu_t(
            rand_vec(hidden * l_cache, 2),
            Shape::new(vec![hidden, 1, l_cache]),
        )),
        shortconv_conv_vec: Some(rand_vec(hidden * l_cache, 2)),
        shortconv_out_proj: Some(Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 3), Shape::new(vec![hidden, hidden])),
            None,
        )),
        ffn_norm: RmsNorm {
            weight: cpu_t(vec![1.0f32; hidden], Shape::new(vec![hidden])),
            eps: 1e-5,
        },
        ffn_gate: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 4), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_up: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 5), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_down: Linear::from_tensor(
            cpu_t(rand_vec(hidden * hidden, 6), Shape::new(vec![hidden, hidden])),
            None,
        ),
        ffn_gate_inp: None,
        ffn_gate_exps: None,
        ffn_up_exps: None,
        ffn_down_exps: None,
        ffn_exp_probs_b: None,
        is_moe: false,
        n_expert: 0,
        n_expert_used: 1,
        charon_cache: grim_models_transformer::shared_moe::CharonCache::new(),
        moe_experts_cache: std::sync::OnceLock::new(),
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        rope_theta: 10000.0,
        eps: 1e-5,
    };
    let mut cache_cpu: Option<Lfm2LayerCache> = None;
    let mut out_cpu = Vec::new();
    for t in 0..steps {
        let x = cpu_t(
            x_data[t * hidden..(t + 1) * hidden].to_vec(),
            Shape::new(vec![1, hidden]),
        );
        out_cpu.extend(
            block_cpu
                .forward(&x, &mut cache_cpu)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
        );
    }

    assert_eq!(out_gpu.len(), out_cpu.len());
    let max_diff = out_gpu
        .iter()
        .zip(out_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "device ring vs host loop diverged: max_diff={max_diff}"
    );

    // The device ring must now exist (i.e. isn't lazily re-uploaded per step).
    match &cache_gpu {
        Some(Lfm2LayerCache::ShortConv { dev, .. }) => {
            assert!(dev.is_some(), "device ring must be allocated on ROCm");
        }
        _ => panic!("expected ShortConv cache"),
    }
}
