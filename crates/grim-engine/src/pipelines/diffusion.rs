//! End-to-end Image Generation Pipeline (Flux.2 MM-DiT + FlowMatch Euler + Flux2VAE).

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::NoiseScheduler;
use grim_models_diffusion::{
    FlowMatchEulerConfig, FlowMatchEulerScheduler, Flux2Config, Flux2Transformer2D, Flux2VAE,
    Flux2VaeConfig,
};
use grim_tensor::{Device, Shape, Tensor};

/// Configuration for the end-to-end diffusion pipeline.
#[derive(Debug, Clone)]
pub struct DiffusionPipelineConfig {
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
}

impl Default for DiffusionPipelineConfig {
    fn default() -> Self {
        Self {
            height: 512,
            width: 512,
            num_inference_steps: 28,
            guidance_scale: 3.5,
        }
    }
}

/// End-to-end Flux.2 Image Diffusion Pipeline.
pub struct DiffusionPipeline {
    pub transformer: Flux2Transformer2D,
    pub vae: Flux2VAE,
    pub scheduler: FlowMatchEulerScheduler,
    pub config: DiffusionPipelineConfig,
    pub device: Device,
}

impl DiffusionPipeline {
    /// Create a new DiffusionPipeline instance.
    pub fn new(
        transformer_config: &Flux2Config,
        vae_config: &Flux2VaeConfig,
        pipeline_config: DiffusionPipelineConfig,
        device: Device,
    ) -> Result<Self> {
        let transformer = Flux2Transformer2D::random(device.clone(), transformer_config.clone());
        let vae = Flux2VAE::random(device.clone(), vae_config.clone());
        let scheduler_config = FlowMatchEulerConfig::default();
        let scheduler = FlowMatchEulerScheduler::new(
            scheduler_config,
            pipeline_config.num_inference_steps,
            (pipeline_config.height / 16) * (pipeline_config.width / 16),
        );

        Ok(Self {
            transformer,
            vae,
            scheduler,
            config: pipeline_config,
            device,
        })
    }

    /// Generate an image from prompt embeddings.
    pub fn generate(&self, prompt_embeds: &Tensor, seed: u64) -> Result<Tensor> {
        let lat_h = self.config.height / 8;
        let lat_w = self.config.width / 8;
        let patch_h = lat_h / 2;
        let patch_w = lat_w / 2;
        let seq_len = patch_h * patch_w;

        // 1. Initialize random Gaussian noise latent: [seq_len, 128]
        let mut rng = grim_core::rng::SimpleRng::new(seed);
        let mut noise_data = vec![0.0f32; seq_len * 128];
        for val in noise_data.iter_mut() {
            *val = (rng.next_f32() - 0.5) * 2.0;
        }
        let mut latents = cpu_tensor(noise_data, Shape::new(vec![seq_len, 128]));

        // 2. Multi-step Flow-Matching Euler Denoising Loop
        for step in 0..self.scheduler.timesteps.len() {
            let t = self.scheduler.timesteps[step];

            // Predict velocity vector field v_theta
            let v_pred = self.transformer.forward(&latents, prompt_embeds, t)?;

            // Euler integration step
            latents = self.scheduler.step(&v_pred, &latents, step as u32)?;
        }

        // 3. Unpack packed latents to spatial form: [1, 32, lat_h, lat_w]
        let spatial_latents = self.vae.unpack_latents(&latents, patch_h, patch_w)?;

        // 4. Decode spatial latents to pixel values: [1, 3, height, width]
        self.vae.decode(&spatial_latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffusion_pipeline_instantiation() {
        let dit_cfg = Flux2Config {
            in_channels: 128,
            joint_attention_dim: 32,
            num_attention_heads: 2,
            attention_head_dim: 16,
            num_layers: 1,
            num_single_layers: 1,
            mlp_ratio: 2.0,
            axes_dims_rope: vec![4, 4, 4, 4],
            rope_theta: 2000.0,
            timestep_guidance_channels: 32,
        };
        let vae_cfg = Flux2VaeConfig::default();
        let pipe_cfg = DiffusionPipelineConfig {
            height: 64,
            width: 64,
            num_inference_steps: 2,
            guidance_scale: 1.0,
        };

        let pipe = DiffusionPipeline::new(&dit_cfg, &vae_cfg, pipe_cfg, Device::Cpu).unwrap();
        assert_eq!(pipe.config.height, 64);
    }
}
