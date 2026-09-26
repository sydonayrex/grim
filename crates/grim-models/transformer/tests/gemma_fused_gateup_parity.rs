//! Gemma dense Q8_0 GateUp fusion parity.
//!
//! This is a device-only regression for the first P1.3b model migration. The
//! fused and split blocks use identical deterministic weights; only the
//! single-token Gate/Up projection path differs.

use std::sync::Arc;

use grim_backend_rocm::{RocmDevice, as_rocm};
use grim_models_transformer::gemma::GemmaBlock;
use grim_nn::{Linear, RmsNorm, Rope};
use grim_tensor::{
    ArithType, CoreTensorOps, DType, Device, QuantFormat, QuantProvenance, Shape, Storage, Tensor,
};

fn f32_tensor(dev: &RocmDevice, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        QuantProvenance::GrimNative,
        Device::Rocm(0),
    )
}

fn q8_tensor(dev: &RocmDevice, data: Vec<f32>, shape: Shape) -> Tensor {
    let raw = f32_tensor(dev, data, shape.clone());
    let (storage, handle) = dev
        .quantize_on_device(raw.storage().as_ref(), QuantFormat::Q8_0)
        .unwrap();
    handle.synchronize().unwrap();
    Tensor::new(
        Arc::from(storage),
        shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
        },
        QuantProvenance::GrimNative,
        Device::Rocm(0),
    )
}

fn linear(dev: &RocmDevice, out_dim: usize, in_dim: usize, seed: u64, q8: bool) -> Linear {
    let data: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (((i as u64 + seed * 17) % 37) as f32 - 18.0) / 100.0)
        .collect();
    let shape = Shape::new(vec![out_dim, in_dim]);
    let weight = if q8 {
        q8_tensor(dev, data, shape)
    } else {
        f32_tensor(dev, data, shape)
    };
    Linear::from_tensor(weight, None)
}

fn make_block(dev: &RocmDevice, fused: bool) -> GemmaBlock {
    let hidden = 64usize;
    let inter = 128usize;
    let gate = linear(dev, inter, hidden, 1, true);
    let up = linear(dev, inter, hidden, 2, true);
    let fused_weights = if fused {
        Some(Arc::new(
            dev.build_fused_gate_up_q80(
                as_rocm(gate.weight.storage().as_ref()).unwrap(),
                as_rocm(up.weight.storage().as_ref()).unwrap(),
            )
            .unwrap(),
        ))
    } else {
        None
    };
    GemmaBlock {
        attn_norm: RmsNorm::new(
            f32_tensor(dev, vec![1.0; hidden], Shape::new(vec![hidden])),
            1e-5,
        ),
        wq: linear(dev, hidden, hidden, 3, false),
        wk: linear(dev, 32, hidden, 4, false),
        wv: linear(dev, 32, hidden, 5, false),
        wo: linear(dev, hidden, hidden, 6, false),
        ffn_norm: RmsNorm::new(
            f32_tensor(dev, vec![1.0; hidden], Shape::new(vec![hidden])),
            1e-5,
        ),
        ffn_gate: gate,
        ffn_up: up,
        ffn_down: linear(dev, hidden, inter, 7, false),
        rope: Rope::new(16, 10000.0),
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        wqkv_q80_fused: None,
        w_gate_up_q80_fused: fused_weights,
    }
}

#[test]
#[ignore]
fn gemma_fused_gateup_matches_split_on_gpu() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        return;
    }
    let dev = RocmDevice::shared(0);
    let split = make_block(&dev, false);
    let fused = make_block(&dev, true);
    let x = f32_tensor(
        &dev,
        (0..64).map(|i| (i as f32 - 32.0) / 64.0).collect(),
        Shape::new(vec![1, 64]),
    );
    let split_out = split
        .forward_kv(&x, &[0], &mut None)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let fused_out = fused
        .forward_kv(&x, &[0], &mut None)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let max_abs = split_out
        .iter()
        .zip(&fused_out)
        .map(|(split, fused)| (split - fused).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 2e-3, "Gemma fused/split max abs error {max_abs}");
}
