use std::collections::HashMap;

use grim_models_audio::{Whisper, WhisperConfig};
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
fn golden_whisper_load_happy_path() {
    let cfg = WhisperConfig {
        d_model: 32,
        num_heads: 4,
        ffn_dim: 64,
        vocab_size: 256,
        n_mels: 80,
        num_enc_layers: 1,
        num_dec_layers: 1,
        rms_norm_eps: 1e-5,
        max_audio_len: 3000,
        max_text_len: 448,
    };

    let d = cfg.d_model;
    let ffn = cfg.ffn_dim;
    let nq = d;
    let vs = cfg.vocab_size;
    let nm = cfg.n_mels;

    let mut t: HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)> = HashMap::new();
    let mut ins = |name: &str, vals: &[f32], shape: Vec<usize>| {
        t.insert(
            name.to_string(),
            (
                f32_bytes(vals),
                shape,
                DType::F32,
                QuantProvenance::GrimNative,
            ),
        );
    };

    ins("tok_emb.weight", &vec![1.0; vs * d], vec![vs, d]);
    ins("enc_in_proj.weight", &vec![2.0; nm * d], vec![d, nm]);
    ins("enc_in_proj.bias", &vec![3.0; d], vec![d]);
    ins(
        "encoder.blocks.0.attn.q.weight",
        &vec![10.0; nq * d],
        vec![nq, d],
    );
    ins(
        "encoder.blocks.0.attn.k.weight",
        &vec![11.0; nq * d],
        vec![nq, d],
    );
    ins(
        "encoder.blocks.0.attn.v.weight",
        &vec![12.0; nq * d],
        vec![nq, d],
    );
    ins(
        "encoder.blocks.0.attn.o.weight",
        &vec![13.0; d * nq],
        vec![d, nq],
    );
    ins("encoder.blocks.0.attn_norm.weight", &vec![14.0; d], vec![d]);
    ins("encoder.blocks.0.ffn_norm.weight", &vec![15.0; d], vec![d]);
    ins(
        "encoder.blocks.0.ffn.0.weight",
        &vec![16.0; ffn * d],
        vec![ffn, d],
    );
    ins("encoder.blocks.0.ffn.0.bias", &vec![17.0; ffn], vec![ffn]);
    ins(
        "encoder.blocks.0.ffn.1.weight",
        &vec![18.0; d * ffn],
        vec![d, ffn],
    );
    ins("encoder.blocks.0.ffn.1.bias", &vec![19.0; d], vec![d]);
    ins("encoder.norm.weight", &vec![20.0; d], vec![d]);
    ins(
        "decoder.blocks.0.self_attn.q.weight",
        &vec![30.0; nq * d],
        vec![nq, d],
    );
    ins(
        "decoder.blocks.0.self_attn.k.weight",
        &vec![31.0; nq * d],
        vec![nq, d],
    );
    ins(
        "decoder.blocks.0.self_attn.v.weight",
        &vec![32.0; nq * d],
        vec![nq, d],
    );
    ins(
        "decoder.blocks.0.self_attn.o.weight",
        &vec![33.0; nq * d],
        vec![nq, d],
    );
    ins(
        "decoder.blocks.0.self_attn_norm.weight",
        &vec![34.0; d],
        vec![d],
    );
    ins(
        "decoder.blocks.0.cross_attn.q.weight",
        &vec![35.0; nq * d],
        vec![nq, d],
    );
    ins(
        "decoder.blocks.0.cross_attn.k.weight",
        &vec![36.0; nq * d],
        vec![nq, d],
    );
    ins(
        "decoder.blocks.0.cross_attn.v.weight",
        &vec![37.0; nq * d],
        vec![nq, d],
    );
    ins(
        "decoder.blocks.0.cross_attn.o.weight",
        &vec![38.0; nq * d],
        vec![nq, d],
    );
    ins(
        "decoder.blocks.0.cross_attn_norm.weight",
        &vec![39.0; d],
        vec![d],
    );
    ins(
        "decoder.blocks.0.ffn.0.weight",
        &vec![38.0; ffn * d],
        vec![ffn, d],
    );
    ins("decoder.blocks.0.ffn.0.bias", &vec![39.0; ffn], vec![ffn]);
    ins(
        "decoder.blocks.0.ffn.1.weight",
        &vec![40.0; d * ffn],
        vec![d, ffn],
    );
    ins("decoder.blocks.0.ffn.1.bias", &vec![41.0; d], vec![d]);
    ins("decoder.norm.weight", &vec![42.0; d], vec![d]);
    ins("output.weight", &vec![50.0; vs * d], vec![vs, d]);
    ins("output.bias", &vec![51.0; vs], vec![vs]);

    let provider = MemProvider { tensors: t };
    let ws = WeightSource::root(&provider, Device::Cpu);
    let model = Whisper::load(Device::Cpu, &ws, cfg).expect("Whisper::load should succeed");

    let tok_w = model.tok_emb.weight.to_vec_f32().unwrap();
    assert_eq!(tok_w.len(), vs * d);
    assert!(tok_w.iter().all(|&v| (v - 1.0).abs() < 1e-6));

    let proj_w = model.enc_in_proj.weight().to_vec_f32().unwrap();
    assert_eq!(proj_w.len(), d * nm);
    assert!(proj_w.iter().all(|&v| (v - 2.0).abs() < 1e-6));

    let proj_b = model.enc_in_proj.bias().unwrap().to_vec_f32().unwrap();
    assert!(proj_b.iter().all(|&v| (v - 3.0).abs() < 1e-6));

    let out_w = model.output.weight().to_vec_f32().unwrap();
    assert_eq!(out_w.len(), vs * d);
    assert!(out_w.iter().all(|&v| (v - 50.0).abs() < 1e-6));
}

#[test]
fn golden_whisper_load_truncated_rejected() {
    let cfg = WhisperConfig {
        d_model: 32,
        num_heads: 4,
        ffn_dim: 64,
        vocab_size: 256,
        n_mels: 80,
        num_enc_layers: 1,
        num_dec_layers: 1,
        rms_norm_eps: 1e-5,
        max_audio_len: 3000,
        max_text_len: 448,
    };
    let mut t: HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)> = HashMap::new();
    t.insert(
        "tok_emb.weight".to_string(),
        (
            vec![0u8; 16],
            vec![cfg.vocab_size, cfg.d_model],
            DType::F32,
            QuantProvenance::GrimNative,
        ),
    );
    let provider = MemProvider { tensors: t };
    let ws = WeightSource::root(&provider, Device::Cpu);
    let result = Whisper::load(Device::Cpu, &ws, cfg);
    assert!(result.is_err(), "truncated buffer should be rejected");
}
