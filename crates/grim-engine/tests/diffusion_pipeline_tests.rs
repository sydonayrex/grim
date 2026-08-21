//! End-to-end integration tests for the DiffusionPipeline (Flux 2 MM-DiT + VAE).

use grim_backend_cpu::cpu_tensor;
use grim_engine::pipelines::diffusion::{DiffusionPipeline, DiffusionPipelineConfig};
use grim_models_diffusion::{Flux2Config, Flux2VaeConfig};
use grim_tensor::{Device, Shape};

#[test]
fn test_diffusion_pipeline_synthetic_generation() {
    let dit_cfg = Flux2Config {
        num_layers: 1,
        num_single_layers: 1,
        ..Default::default()
    };
    let vae_cfg = Flux2VaeConfig::default();
    let pipe_cfg = DiffusionPipelineConfig {
        height: 64,
        width: 64,
        num_inference_steps: 2,
        guidance_scale: 1.0,
    };

    let pipe = DiffusionPipeline::new(&dit_cfg, &vae_cfg, pipe_cfg, Device::Cpu).unwrap();

    let prompt_embeds = cpu_tensor(
        vec![0.1f32; 16 * dit_cfg.joint_attention_dim],
        Shape::new(vec![16, dit_cfg.joint_attention_dim]),
    );

    let image = pipe.generate(&prompt_embeds, 42).unwrap();
    assert_eq!(image.shape().dims(), &[1, 3, 64, 64]);
}
