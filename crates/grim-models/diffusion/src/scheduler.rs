//! Noise schedulers for diffusion: DDIM and Euler (deterministic).
//!
//! A noise scheduler owns a step loop. Sampling is a sequence of:
//!   predicted_noise = model.denoise_step(latents, timestep, cond)
//!   next_latents    = scheduler.step(predicted_noise, latents, timestep)

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::NoiseScheduler;
use grim_tensor::Shape;

/// DDIM (deterministic) scheduler. v1 ships eta=0; stochasticity is a
/// follow-on tuning knob.
#[derive(Debug, Clone)]
pub struct DdimScheduler {
    /// Monotonically descending sigma schedule, length = num_steps.
    pub timesteps: Vec<u32>,
    /// Per-step alpha_cumprod, length = len(timesteps).
    pub alphas_cumprod: Vec<f32>,
}

impl DdimScheduler {
    pub fn new(timesteps: Vec<u32>, alphas_cumprod: Vec<f32>) -> Self {
        assert_eq!(timesteps.len(), alphas_cumprod.len());
        Self { timesteps, alphas_cumprod }
    }

    /// Build a linear-schedule DDIM scheduler of `num_steps` steps.
    pub fn linear(num_steps: usize, beta_start: f32, beta_end: f32) -> Self {
        let betas: Vec<f32> = (0..num_steps)
            .map(|i| beta_start + (beta_end - beta_start) * (i as f32) / (num_steps as f32))
            .collect();
        Self::from_betas(betas)
    }

    pub fn from_betas(betas: Vec<f32>) -> Self {
        let mut alphas: Vec<f32> = betas.iter().map(|b| 1.0 - *b).collect();
        let mut cumprod = vec![0.0f32; alphas.len()];
        let mut acc = 1.0f32;
        for (i, a) in alphas.iter().enumerate() {
            acc *= *a;
            cumprod[i] = acc;
        }
        alphas.clear();
        let timesteps: Vec<u32> = (0..cumprod.len() as u32).rev().collect();
        Self { timesteps, alphas_cumprod: cumprod }
    }
}

impl NoiseScheduler for DdimScheduler {
    fn step(&self, model_output: &grim_tensor::Tensor, latents: &grim_tensor::Tensor, timestep: u32) -> Result<grim_tensor::Tensor> {
        let pos = self.timesteps.iter().position(|&t| t == timestep);
        if pos.is_none() {
            return Err(Error::Config(format!("DDIM unknown timestep {timestep}")));
        }
        let lshape = latents.shape().dims().to_vec();
        let mshape = model_output.shape().dims().to_vec();
        if lshape != mshape {
            return Err(Error::Shape(format!(
                "DDIM step: latents {:?} ≠ model_output {:?}",
                lshape, mshape
            )));
        }
        let lat = latents.to_vec_f32()?;
        let noise = model_output.to_vec_f32()?;
        let n = lat.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = lat[i] - noise[i];
        }
        Ok(cpu_tensor(out, Shape::new(lshape)))
    }
}

/// Euler (DDIM-equivalent at eta=0) deterministic scheduler.
#[derive(Debug, Clone)]
pub struct EulerScheduler {
    pub timestep: u32,
    pub sigma_next: f32,
    pub sigma_cur: f32,
}

impl EulerScheduler {
    pub fn new(timestep: u32, sigma_cur: f32, sigma_next: f32) -> Self {
        Self { timestep, sigma_cur, sigma_next }
    }
}

impl NoiseScheduler for EulerScheduler {
    fn step(&self, model_output: &grim_tensor::Tensor, latents: &grim_tensor::Tensor, timestep: u32) -> Result<grim_tensor::Tensor> {
        let _ = timestep;
        let _ = self.timestep;
        let lat_shape = latents.shape().dims().to_vec();
        let lat = latents.to_vec_f32()?;
        let noise = model_output.to_vec_f32()?;
        let n = lat.len();
        let dt = self.sigma_next - self.sigma_cur;
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let dx = (lat[i] - noise[i]) * dt;
            out[i] = lat[i] + dx;
        }
        Ok(cpu_tensor(out, Shape::new(lat_shape)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::Tensor;

    fn tensor_with(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        cpu_tensor(data, Shape::new(shape))
    }

    #[test]
    fn ddim_linear_schedule_basic() {
        let sched = DdimScheduler::linear(10, 0.0001, 0.02);
        assert_eq!(sched.timesteps.len(), 10);
        assert_eq!(sched.alphas_cumprod.len(), 10);
        // alpha_cumprod shrinks monotonically as more noise is added.
        assert!(sched.alphas_cumprod[0] > sched.alphas_cumprod[sched.alphas_cumprod.len() - 1]);
        // timesteps are descending.
        for w in sched.timesteps.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn ddim_unknown_timestep_is_error() {
        let sched = DdimScheduler::linear(4, 0.0001, 0.02);
        let lat = tensor_with(vec![1.0f32; 8], vec![2, 4]);
        let n = tensor_with(vec![0.1f32; 8], vec![2, 4]);
        let err = <DdimScheduler as NoiseScheduler>::step(&sched, &n, &lat, 9999)
            .err()
            .expect("step should fail on unknown timestep");
        match err {
            Error::Config(_) => {}
            other => panic!("expected Config error, got {:?}", other),
        }
    }

    #[test]
    fn euler_step_applies_dt() {
        let sched = EulerScheduler::new(0, 1.0, 0.5);
        let lat = tensor_with(vec![1.0f32; 4], vec![4]);
        let noise = tensor_with(vec![0.5f32; 4], vec![4]);
        let out = <EulerScheduler as NoiseScheduler>::step(&sched, &noise, &lat, 0).unwrap();
        let v = out.to_vec_f32().unwrap();
        for x in &v {
            assert!((*x - 0.75).abs() < 1e-6, "expected 0.75, got {x}");
        }
    }
}
