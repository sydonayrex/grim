use std::collections::HashMap;

use grim_models_vision::{Vit, VitConfig};
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
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing: {name}")))?;
        Ok(RawTensor {
            bytes,
            shape,
            dtype,
            provenance,
        })
    }
    fn meta(&self, name: &str) -> Result<TensorMeta, grim_tensor::error::Error> {
        let (_, shape, dtype, provenance) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing: {name}")))?;
        Ok(TensorMeta {
            dtype,
            provenance,
            shape,
            fusion_mask: 0,
        })
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

#[test]
fn golden_vit_load_happy_path() {
    let cfg = VitConfig {
        image_size: 32,
        patch_size: 16,
        in_channels: 3,
        hidden_size: 32,
        num_layers: 1,
        num_heads: 4,
        intermediate_size: 64,
        rms_norm_eps: 1e-5,
    };
    let patch_dim = 3 * 16 * 16;
    let num_patches = (32 / 16) * (32 / 16);
    let hidden_size = cfg.hidden_size;
    let mut tensors: HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)> =
        HashMap::new();
    let insert = |map: &mut HashMap<_, _>, name: &str, values: &[f32], shape: Vec<usize>| {
        map.insert(
            name.to_string(),
            (
                f32_bytes(values),
                shape,
                DType::F32,
                QuantProvenance::GrimNative,
            ),
        );
    };

    insert(
        &mut tensors,
        "proj.weight",
        &vec![1.0; cfg.hidden_size * patch_dim],
        vec![cfg.hidden_size, patch_dim],
    );
    insert(
        &mut tensors,
        "proj.bias",
        &vec![2.0; cfg.hidden_size],
        vec![cfg.hidden_size],
    );
    insert(
        &mut tensors,
        "cls_token",
        &vec![3.0; cfg.hidden_size],
        vec![cfg.hidden_size],
    );
    let pos_count = (num_patches + 1) * cfg.hidden_size;
    insert(
        &mut tensors,
        "pos_embed",
        &vec![4.0; pos_count],
        vec![num_patches + 1, cfg.hidden_size],
    );

    let n = cfg.hidden_size;
    insert(
        &mut tensors,
        "blocks.0.attn.q.weight",
        &vec![10.0; n * n],
        vec![n, n],
    );
    insert(
        &mut tensors,
        "blocks.0.attn.k.weight",
        &vec![11.0; n * n],
        vec![n, n],
    );
    insert(
        &mut tensors,
        "blocks.0.attn.v.weight",
        &vec![12.0; n * n],
        vec![n, n],
    );
    insert(
        &mut tensors,
        "blocks.0.attn.o.weight",
        &vec![13.0; n * n],
        vec![n, n],
    );
    insert(
        &mut tensors,
        "blocks.0.attn_norm.weight",
        &vec![14.0; n],
        vec![n],
    );
    // Pre-MLP LayerNorm (P1-34).
    insert(
        &mut tensors,
        "blocks.0.ffn_norm.weight",
        &vec![14.0; n],
        vec![n],
    );
    insert(
        &mut tensors,
        "blocks.0.ffn.0.weight",
        &vec![15.0; cfg.intermediate_size * n],
        vec![cfg.intermediate_size, n],
    );
    insert(
        &mut tensors,
        "blocks.0.ffn.0.bias",
        &vec![16.0; cfg.intermediate_size],
        vec![cfg.intermediate_size],
    );
    insert(
        &mut tensors,
        "blocks.0.ffn.1.weight",
        &vec![17.0; n * cfg.intermediate_size],
        vec![n, cfg.intermediate_size],
    );
    insert(&mut tensors, "blocks.0.ffn.1.bias", &vec![18.0; n], vec![n]);
    insert(&mut tensors, "ln.weight", &vec![19.0; n], vec![n]);

    let provider = MemProvider { tensors };
    let ws = WeightSource::root(&provider, Device::Cpu);
    let model = Vit::load(Device::Cpu, &ws, cfg).expect("Vit::load should succeed");

    assert_eq!(model.patch_proj_w.len(), hidden_size * patch_dim);
    assert_eq!(model.patch_proj_b.len(), hidden_size);
    assert!(model.patch_proj_w.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    assert!(model.patch_proj_b.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    assert!(model.cls_token.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    assert!(model.pos_embed.iter().all(|&v| (v - 4.0).abs() < 1e-6));
    assert!(model.ln.weight.iter().all(|&v| (v - 19.0).abs() < 1e-6));
}

#[test]
fn golden_vit_load_truncated_rejected() {
    let cfg = VitConfig {
        image_size: 32,
        patch_size: 16,
        in_channels: 3,
        hidden_size: 32,
        num_layers: 1,
        num_heads: 4,
        intermediate_size: 64,
        rms_norm_eps: 1e-5,
    };
    let mut tensors: HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)> =
        HashMap::new();
    tensors.insert(
        "proj.weight".to_string(),
        (
            vec![0u8; 16],
            vec![cfg.hidden_size, 3 * 16 * 16],
            DType::F32,
            QuantProvenance::GrimNative,
        ),
    );
    let provider = MemProvider { tensors };
    let ws = WeightSource::root(&provider, Device::Cpu);
    let result = Vit::load(Device::Cpu, &ws, cfg);
    assert!(result.is_err(), "truncated buffer should be rejected");
}
