//! Integration tests for LoRA weight loading and ROCm alignment.

use std::collections::HashMap;

use grim_models_transformer::lora::{align_tensor_for_rocm_gemm, LoRAWeights};
use grim_nn::WeightSource;
use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn align_tensor_for_rocm_gemm_returns_tensor_for_native_f32() {
    let provider = MemProvider {
        tensors: HashMap::from([(
            "t.weight".to_string(),
            (
                f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                vec![2, 3],
                DType::F32,
                QuantProvenance::GrimNative,
            ),
        )]),
    };
    let ws = WeightSource::root(&provider, Device::Cpu);

    let t = ws.get(vec![2, 3], "t.weight").expect("get");
    let aligned = align_tensor_for_rocm_gemm(&t).expect("align");
    assert_eq!(aligned.shape().dims(), &[64, 3]);
}

#[test]
fn load_for_rocm_populates_down_up_alpha() {
    // hidden_size = 4, rank = 2 -> A is [2,4], B is [4,2]
    let provider = MemProvider {
        tensors: HashMap::from([
            (
                "blk.0.lora_A.weight".to_string(),
                (
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                    vec![2, 4],
                    DType::F32,
                    QuantProvenance::GrimNative,
                ),
            ),
            (
                "blk.0.lora_B.weight".to_string(),
                (
                    f32_bytes(&[
                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    ]),
                    vec![4, 2],
                    DType::F32,
                    QuantProvenance::GrimNative,
                ),
            ),
            (
                "blk.0.lora_alpha".to_string(),
                (
                    f32_bytes(&[16.0]),
                    vec![1],
                    DType::F32,
                    QuantProvenance::GrimNative,
                ),
            ),
        ]),
    };
    let ws = WeightSource::root(&provider, Device::Cpu);

    let lora = LoRAWeights::load_for_rocm(&ws, "blk.0", 2, 4).expect("load");
    assert_eq!(lora.down_proj.shape().dims(), &[64, 4]);
    assert_eq!(lora.up_proj.shape().dims(), &[64, 2]);
    // alpha_scale = alpha / rank = 16 / 2 = 8
    assert!((lora.alpha_scale - 8.0).abs() < 1e-6);
}
