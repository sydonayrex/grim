use grim_backend_cpu::cpu_tensor;
use grim_models_diffusion::{
    FlowMatchEulerConfig, FlowMatchEulerScheduler, Flux2Config, Flux2Transformer2D, Flux2VAE,
    Flux2VaeConfig,
};
use grim_tensor::{Device, Shape};

#[test]
fn test_flux2_vae_pack_and_unpack() {
    let cfg = Flux2VaeConfig::default();
    let vae = Flux2VAE::random(Device::Cpu, cfg);

    // Create [batch=1, channels=32, height=16, width=16] spatial latents
    let spatial_latents = cpu_tensor(
        (0..32 * 16 * 16).map(|i| (i as f32) * 0.001).collect(),
        Shape::new(vec![1, 32, 16, 16]),
    );

    // Pack 2x2 spatial patches -> [batch=1, seq_len=64, channels=128]
    let packed = vae.pack_latents(&spatial_latents).expect("pack latents");
    assert_eq!(packed.shape().dims(), &[1, 64, 128]);

    // Unpack back to [batch=1, 32, 16, 16]
    let unpacked = vae.unpack_latents(&packed, 8, 8).expect("unpack latents");
    assert_eq!(unpacked.shape().dims(), &[1, 32, 16, 16]);

    let orig_v = spatial_latents.to_vec_f32().unwrap();
    let unpack_v = unpacked.to_vec_f32().unwrap();
    assert_eq!(
        orig_v, unpack_v,
        "Packing and unpacking must be mathematically bijective"
    );
}

#[test]
fn test_flux2_vae_decode_to_rgb() {
    let cfg = Flux2VaeConfig::default();
    let vae = Flux2VAE::random(Device::Cpu, cfg);

    // Latents [1, 32, 8, 8] -> RGB [1, 3, 64, 64]
    let latents = cpu_tensor(vec![0.5f32; 32 * 8 * 8], Shape::new(vec![1, 32, 8, 8]));
    let rgb = vae.decode(&latents).expect("decode to RGB");
    assert_eq!(rgb.shape().dims(), &[1, 3, 64, 64]);
}

#[test]
fn test_flux2_mmdit_forward_step() {
    let cfg = Flux2Config {
        in_channels: 128,
        joint_attention_dim: 256,
        num_attention_heads: 4,
        attention_head_dim: 32,
        num_layers: 2,
        num_single_layers: 2,
        mlp_ratio: 2.0,
        axes_dims_rope: vec![8, 8, 8, 8],
        rope_theta: 2000.0,
        timestep_guidance_channels: 64,
    };
    let model = Flux2Transformer2D::random(Device::Cpu, cfg);

    let img_seq = 16;
    let txt_seq = 8;

    let img_latents = cpu_tensor(vec![0.1f32; img_seq * 128], Shape::new(vec![img_seq, 128]));
    let txt_latents = cpu_tensor(vec![0.2f32; txt_seq * 256], Shape::new(vec![txt_seq, 256]));

    let out = model
        .forward(&img_latents, &txt_latents, 500.0)
        .expect("Flux2 DiT forward pass");
    assert_eq!(out.shape().dims(), &[img_seq, 128]);
}

#[test]
fn test_flow_match_euler_scheduler_loop() {
    let cfg = FlowMatchEulerConfig::default();
    let scheduler = FlowMatchEulerScheduler::new(cfg, 4, 256);

    assert_eq!(scheduler.sigmas.len(), 5);
    assert!(scheduler.sigmas[0] > scheduler.sigmas[4]);

    let mut latents = cpu_tensor(vec![1.0f32; 64 * 128], Shape::new(vec![64, 128]));
    let model_velocity = cpu_tensor(vec![0.1f32; 64 * 128], Shape::new(vec![64, 128]));

    for step in 0..4 {
        latents = scheduler
            .step_euler(&model_velocity, &latents, step)
            .expect("euler step");
    }

    assert_eq!(latents.shape().dims(), &[64, 128]);
}
