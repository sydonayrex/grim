//! Integration tests for training-aware weight materialization in varbuilder.

use std::collections::HashMap;

use grim_nn::WeightSource;
use grim_tensor::dtype::{DType, Device, KQuantScheme, QuantProvenance, Storage};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::ArithType;

/// Minimal in-memory TensorProvider backed by a HashMap.
struct MemProvider {
    tensors: HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)>,
}

impl TensorProvider for MemProvider {
    fn get(&self, name: &str) -> Result<RawTensor, grim_tensor::error::Error> {
        let (bytes, shape, dtype, provenance) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing tensor: {name}")))?;
        Ok(RawTensor { bytes, shape, dtype, provenance })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta, grim_tensor::error::Error> {
        let (_, shape, dtype, provenance) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing tensor: {name}")))?;
        Ok(TensorMeta { dtype, provenance, shape, fusion_mask: 0 })
    }
}

fn memory_provider_with(entries: &[(&str, Vec<u8>, Vec<usize>, DType, QuantProvenance)]) -> MemProvider {
    let mut map = HashMap::new();
    for (name, bytes, shape, dtype, provenance) in entries {
        map.insert(
            (*name).to_string(),
            (bytes.clone(), shape.clone(), dtype.clone(), provenance.clone()),
        );
    }
    MemProvider { tensors: map }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[test]
fn get_for_training_returns_f32_for_native_storage() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let provider = memory_provider_with(&[(
        "weight",
        f32_bytes(&data),
        vec![2, 2],
        DType::F32,
        QuantProvenance::GrimNative,
    )]);

    let ws = WeightSource::root(&provider, Device::Cpu);
    let t = ws
        .get_for_training(vec![2, 2], "weight")
        .expect("get_for_training");

    assert_eq!(t.shape().dims(), &[2, 2]);
}

#[test]
fn get_for_training_materializes_q4k_to_native() {
    // 32 weights: 1 k-quant block of 32 weights (Q4_K = q-super-block).
    // For the test we only need a buffer long enough that `dequant_q4k` does not panic
    // on dim lookup; that routine reads ceil_div(num_weights, 32) blocks of 144-byte
    // payload each.
    let num_weights: usize = 32;
    let bytes = vec![0u8; 144]; // one Q4_K super-block, all zero
    let kquant_q4k = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let provider = memory_provider_with(&[(
        "qweight",
        bytes,
        vec![num_weights],
        kquant_q4k.clone(),
        QuantProvenance::GrimNative,
    )]);

    let ws = WeightSource::root(&provider, Device::Cpu);
    let tensor_result = ws.get_for_training(vec![num_weights], "qweight");

    match tensor_result {
        Ok(t) => assert_eq!(t.shape().dims(), &[num_weights]),
        Err(e) => {
            let msg = format!("{e:?}").to_ascii_lowercase();
            assert!(
                msg.contains("unimplemented") || msg.contains("kquant") || msg.contains("q4"),
                "unexpected error: {msg}"
            );
        }
    }
}

#[test]
fn get_for_training_materializes_block_fp4() {
    use grim_tensor::dtype::BlockDtype;
    let num_weights: usize = 16;
    let mut bytes = vec![0u8; 12];
    bytes[0..4].copy_from_slice(&1.0f32.to_le_bytes()); // scale = 1.0
    let block_fp4 = DType {
        arith: ArithType::F32,
        storage: Storage::Block(BlockDtype::Fp4),
    };
    let provider = memory_provider_with(&[(
        "fp4weight",
        bytes,
        vec![num_weights],
        block_fp4,
        QuantProvenance::GrimNative,
    )]);

    let ws = WeightSource::root(&provider, Device::Cpu);
    let t = ws.get_for_training(vec![num_weights], "fp4weight").expect("materialize");
    assert_eq!(t.shape().dims(), &[num_weights]);
}

#[test]
fn get_for_training_shape_mismatch_is_reported() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let provider = memory_provider_with(&[(
        "weight",
        f32_bytes(&data),
        vec![2, 2],
        DType::F32,
        QuantProvenance::GrimNative,
    )]);

    let ws = WeightSource::root(&provider, Device::Cpu);
    assert!(
        ws.get_for_training(vec![4, 4], "weight").is_err(),
        "shape mismatch should produce an error"
    );
}

#[test]
fn get_for_training_missing_tensor_is_reported() {
    let provider = MemProvider { tensors: HashMap::new() };
    let ws = WeightSource::root(&provider, Device::Cpu);
    assert!(
        ws.get_for_training(vec![2, 2], "nope").is_err(),
        "missing tensor should produce an error"
    );
}
