//! Tests for Diffusion LoRA (Task 10.2).
//!
//! Validates:
//! 1. Kohya + Diffusers key normalization.
//! 2. Merge-at-load vs runtime dispatched forward pass equivalence on Flux2 MM-DiT.
//! 3. Base model without LoRA differs from adapted models.

use std::collections::HashMap;
use grim_backend_cpu::cpu_tensor;
use grim_models_diffusion::flux2::{Flux2Config, Flux2Transformer2D};
use grim_models_diffusion::lora::{
    normalize_diffusion_lora_key, DiffusionLoraAdapter, LoraKeyKind,
};
use grim_tensor::{Device, Shape};

#[test]
fn test_diffusion_lora_key_normalization() {
    // Kohya UNet key
    let kohya_key = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight";
    let (canon, kind) = normalize_diffusion_lora_key(kohya_key).unwrap();
    assert_eq!(kind, LoraKeyKind::Down);
    assert!(canon.contains("down_blocks") && canon.contains("to_q"));

    // Diffusers Flux key
    let diffusers_flux_key = "transformer.joint_blocks.0.img_attn.qkv.lora_A.weight";
    let (canon, kind) = normalize_diffusion_lora_key(diffusers_flux_key).unwrap();
    assert_eq!(kind, LoraKeyKind::Down);
    assert_eq!(canon, "joint_blocks.0.img_attn.qkv");

    let diffusers_flux_key_b = "transformer.joint_blocks.0.img_attn.qkv.lora_B.weight";
    let (canon_b, kind_b) = normalize_diffusion_lora_key(diffusers_flux_key_b).unwrap();
    assert_eq!(kind_b, LoraKeyKind::Up);
    assert_eq!(canon_b, "joint_blocks.0.img_attn.qkv");

    let diffusers_flux_alpha = "transformer.joint_blocks.0.img_attn.qkv.alpha";
    let (canon_a, kind_a) = normalize_diffusion_lora_key(diffusers_flux_alpha).unwrap();
    assert_eq!(kind_a, LoraKeyKind::Alpha);
    assert_eq!(canon_a, "joint_blocks.0.img_attn.qkv");
}

#[test]
fn test_flux2_lora_merged_vs_dispatched_parity() {
    let dev = Device::Cpu;
    let cfg = Flux2Config {
        in_channels: 16,
        joint_attention_dim: 32,
        num_attention_heads: 2,
        attention_head_dim: 16,
        num_layers: 2,
        num_single_layers: 2,
        mlp_ratio: 2.0,
        axes_dims_rope: vec![8, 8, 8, 8],
        rope_theta: 2000.0,
        timestep_guidance_channels: 16,
    };

    let seed = 424242u64;
    let base_model = Flux2Transformer2D::new_with_seed(dev.clone(), cfg.clone(), seed);
    let mut runtime_model = Flux2Transformer2D::new_with_seed(dev.clone(), cfg.clone(), seed);
    let mut merged_model = Flux2Transformer2D::new_with_seed(dev.clone(), cfg.clone(), seed);

    let hidden_dim = cfg.num_attention_heads * cfg.attention_head_dim; // 32
    let qkv_dim = hidden_dim * 3; // 96
    let rank = 4usize;
    let alpha = 8.0f32;
    let scale = 1.0f32;

    // Create synthetic LoRA weights for joint_blocks.0.img_attn.qkv
    // in_dim = hidden_dim (32), out_dim = qkv_dim (96)
    let a_vals: Vec<f32> = (0..rank * hidden_dim)
        .map(|i| (i as f32 * 0.13).sin() * 0.5)
        .collect();
    let b_vals: Vec<f32> = (0..qkv_dim * rank)
        .map(|i| (i as f32 * 0.29).cos() * 0.5)
        .collect();

    let mut raw_tensors = HashMap::new();
    raw_tensors.insert(
        "transformer.joint_blocks.0.img_attn.qkv.lora_A.weight".into(),
        (vec![rank, hidden_dim], a_vals.clone()),
    );
    raw_tensors.insert(
        "transformer.joint_blocks.0.img_attn.qkv.lora_B.weight".into(),
        (vec![qkv_dim, rank], b_vals.clone()),
    );
    raw_tensors.insert(
        "transformer.joint_blocks.0.img_attn.qkv.alpha".into(),
        (vec![1], vec![alpha]),
    );

    let adapter = DiffusionLoraAdapter::from_raw_tensors(&raw_tensors, scale).unwrap();

    // 1. Attach adapter to runtime model
    runtime_model.lora_adapters.push(adapter.clone());

    // 2. Pre-merge adapter into merged_model.joint_blocks[0].img_qkv
    let mut weights_map = HashMap::new();
    weights_map.insert(
        "joint_blocks.0.img_attn.qkv".into(),
        merged_model.joint_blocks[0].img_qkv.weight().clone(),
    );
    let merged_count = adapter.apply_merged(&mut weights_map, None).unwrap();
    assert_eq!(merged_count, 1);
    let updated_w = weights_map.remove("joint_blocks.0.img_attn.qkv").unwrap();
    merged_model.joint_blocks[0].img_qkv = grim_nn::Linear::from_tensor(
        updated_w,
        merged_model.joint_blocks[0].img_qkv.bias().cloned(),
    );

    // Synthetic inputs
    let img_latents = cpu_tensor(
        (0..4 * cfg.in_channels).map(|i| (i as f32 * 0.07).cos()).collect(),
        Shape::new(vec![4, cfg.in_channels]),
    );
    let txt_latents = cpu_tensor(
        (0..6 * cfg.joint_attention_dim).map(|i| (i as f32 * 0.05).sin()).collect(),
        Shape::new(vec![6, cfg.joint_attention_dim]),
    );
    let timestep = 250.0f32;

    // Run forwards
    let out_base = base_model.forward(&img_latents, &txt_latents, timestep).unwrap();
    let out_runtime = runtime_model.forward(&img_latents, &txt_latents, timestep).unwrap();
    let out_merged = merged_model.forward(&img_latents, &txt_latents, timestep).unwrap();

    let v_base = out_base.to_vec_f32().unwrap();
    let v_runtime = out_runtime.to_vec_f32().unwrap();
    let v_merged = out_merged.to_vec_f32().unwrap();

    // Base vs Adapted must differ
    let mut base_diff = 0.0f32;
    for (&b, &r) in v_base.iter().zip(v_runtime.iter()) {
        base_diff = base_diff.max((b - r).abs());
    }
    assert!(base_diff > 1e-4, "Base model output should differ from LoRA adapted output (got {base_diff})");

    // Runtime vs Merged must match within < 1e-4
    let mut max_diff = 0.0f32;
    for (&r, &m) in v_runtime.iter().zip(v_merged.iter()) {
        max_diff = max_diff.max((r - m).abs());
    }
    assert!(
        max_diff < 1e-4,
        "Runtime dispatched LoRA vs merged LoRA parity failure: max_diff = {max_diff} >= 1e-4"
    );
}
