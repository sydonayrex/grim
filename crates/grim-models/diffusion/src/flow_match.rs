//! FlowMatch Euler Discrete Noise Scheduler for Rectified Flow Diffusion Models.
//!
//! Implements resolution-dependent empirical time-shifting and deterministic Euler ODE steps:
//! $$x_{t - \Delta t} = x_t + (\sigma_{t - \Delta t} - \sigma_t) \cdot v_\theta(x_t, \sigma_t, c)$$

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::NoiseScheduler;
use grim_tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Configuration parameters for FlowMatchEulerDiscreteScheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMatchEulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f32,
    pub use_dynamic_shifting: bool,
    pub base_shift: f32,
    pub max_shift: f32,
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
}

impl Default for FlowMatchEulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 3.0,
            use_dynamic_shifting: true,
            base_shift: 0.5,
            max_shift: 1.15,
            base_image_seq_len: 256,
            max_image_seq_len: 4096,
        }
    }
}

/// FlowMatch discrete Euler noise scheduler with resolution time-shift.
#[derive(Debug, Clone)]
pub struct FlowMatchEulerScheduler {
    pub config: FlowMatchEulerConfig,
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

impl FlowMatchEulerScheduler {
    /// Instantiate a FlowMatch Euler scheduler configured for `num_inference_steps` and image token sequence length.
    pub fn new(
        config: FlowMatchEulerConfig,
        num_inference_steps: usize,
        image_seq_len: usize,
    ) -> Self {
        let num_steps = num_inference_steps.max(1);
        let shift = if config.use_dynamic_shifting {
            calculate_dynamic_time_shift(
                image_seq_len,
                config.base_image_seq_len,
                config.max_image_seq_len,
                config.base_shift,
                config.max_shift,
            )
        } else {
            config.shift
        };

        // Linear sigma grid from 1.0 to 0.0
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        let mut timesteps = Vec::with_capacity(num_steps);

        for i in 0..=num_steps {
            let t = 1.0 - (i as f32) / (num_steps as f32);
            // Time-shift transform: sigma = (shift * t) / (1 + (shift - 1) * t)
            let sigma = if t <= 0.0 {
                0.0
            } else if t >= 1.0 {
                1.0
            } else {
                (shift * t) / (1.0 + (shift - 1.0) * t)
            };
            sigmas.push(sigma);
            if i < num_steps {
                timesteps.push(sigma * 1000.0);
            }
        }

        Self {
            config,
            sigmas,
            timesteps,
        }
    }

    /// Single Euler step advancement along velocity vector field:
    /// `x_prev = latents + (sigma_next - sigma_curr) * model_output`
    pub fn step_euler(
        &self,
        model_output: &Tensor,
        latents: &Tensor,
        step_index: usize,
    ) -> Result<Tensor> {
        if step_index + 1 >= self.sigmas.len() {
            return Err(Error::Config(format!(
                "step_index {} out of bounds for sigmas len {}",
                step_index,
                self.sigmas.len()
            )));
        }

        let sigma_curr = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];
        let dt = sigma_next - sigma_curr;

        let v_vec = model_output.to_vec_f32()?;
        let x_vec = latents.to_vec_f32()?;

        if v_vec.len() != x_vec.len() {
            return Err(Error::Shape(format!(
                "shape mismatch: latents {:?} vs model_output {:?}",
                latents.shape().dims(),
                model_output.shape().dims()
            )));
        }

        let mut next_vec = vec![0.0f32; x_vec.len()];
        for i in 0..x_vec.len() {
            next_vec[i] = x_vec[i] + dt * v_vec[i];
        }

        Ok(cpu_tensor(next_vec, latents.shape().clone()))
    }
}

impl NoiseScheduler for FlowMatchEulerScheduler {
    fn step(&self, model_output: &Tensor, latents: &Tensor, timestep: u32) -> Result<Tensor> {
        // Map integer timestep to closest sigma index
        let step_idx = (timestep as usize).min(self.sigmas.len().saturating_sub(2));
        self.step_euler(model_output, latents, step_idx)
    }
}

/// Compute resolution-dependent time-shift interpolation.
fn calculate_dynamic_time_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / ((max_seq_len as f32) - (base_seq_len as f32)).max(1.0);
    let b = base_shift - m * (base_seq_len as f32);
    let shift = m * (image_seq_len as f32) + b;
    shift.clamp(base_shift, max_shift)
}
