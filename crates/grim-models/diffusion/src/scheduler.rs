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
        Self {
            timesteps,
            alphas_cumprod,
        }
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
        Self {
            timesteps,
            alphas_cumprod: cumprod,
        }
    }
}

impl NoiseScheduler for DdimScheduler {
    fn step(
        &self,
        model_output: &grim_tensor::Tensor,
        latents: &grim_tensor::Tensor,
        timestep: u32,
    ) -> Result<grim_tensor::Tensor> {
        if !self.timesteps.iter().any(|&t| t == timestep) {
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
        // DDIM (eta = 0) deterministic update from epsilon-prediction:
        //   x0_pred = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
        //   x_{t-1}  = sqrt(alpha_prev) * x0_pred + sqrt(1 - alpha_prev) * eps
        // The final step lands on alpha_prev = 1 (clean image).
        let t = timestep as usize;
        let alpha_t = self.alphas_cumprod[t];
        let alpha_prev = if t == 0 {
            1.0
        } else {
            self.alphas_cumprod[t - 1]
        };
        let sqrt_alpha_t = alpha_t.max(1e-12).sqrt();
        let sqrt_alpha_prev = alpha_prev.max(1e-12).sqrt();
        let sqrt_one_minus_t = (1.0 - alpha_t).max(0.0).sqrt();
        let sqrt_one_minus_prev = (1.0 - alpha_prev).max(0.0).sqrt();
        let lat = latents.to_vec_f32()?;
        let noise = model_output.to_vec_f32()?;
        let n = lat.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let x0_pred = (lat[i] - sqrt_one_minus_t * noise[i]) / sqrt_alpha_t;
            out[i] = sqrt_alpha_prev * x0_pred + sqrt_one_minus_prev * noise[i];
        }
        Ok(cpu_tensor(out, Shape::new(lshape)))
    }
}

/// Euler (deterministic) scheduler on the probability-flow ODE.
///
/// Follows the same linear beta schedule as DDIM, expressed as the
/// per-step noise level `sigma = sqrt(1 - alpha_cumprod)`.
#[derive(Debug, Clone)]
pub struct EulerScheduler {
    /// Descending sigma schedule, length = num_steps.
    pub sigmas: Vec<f32>,
    /// Monotonically descending timesteps, length = num_steps.
    pub timesteps: Vec<u32>,
}

impl EulerScheduler {
    pub fn from_betas(betas: Vec<f32>) -> Self {
        let mut alphas: Vec<f32> = betas.iter().map(|b| 1.0 - *b).collect();
        let mut cumprod = vec![0.0f32; alphas.len()];
        let mut acc = 1.0f32;
        for (i, a) in alphas.iter().enumerate() {
            acc *= *a;
            cumprod[i] = acc;
        }
        let sigmas: Vec<f32> = cumprod.iter().map(|a| (1.0 - a).max(0.0).sqrt()).collect();
        alphas.clear();
        let timesteps: Vec<u32> = (0..cumprod.len() as u32).rev().collect();
        Self { sigmas, timesteps }
    }

    /// Build a linear-schedule Euler scheduler of `num_steps` steps.
    pub fn linear(num_steps: usize, beta_start: f32, beta_end: f32) -> Self {
        let betas: Vec<f32> = (0..num_steps)
            .map(|i| beta_start + (beta_end - beta_start) * (i as f32) / (num_steps as f32))
            .collect();
        Self::from_betas(betas)
    }
}

impl NoiseScheduler for EulerScheduler {
    fn step(
        &self,
        model_output: &grim_tensor::Tensor,
        latents: &grim_tensor::Tensor,
        timestep: u32,
    ) -> Result<grim_tensor::Tensor> {
        let pos = self.timesteps.iter().position(|&t| t == timestep);
        let pos = match pos {
            Some(p) => p,
            None => return Err(Error::Config(format!("Euler unknown timestep {timestep}"))),
        };
        let lat_shape = latents.shape().dims().to_vec();
        let mshape = model_output.shape().dims().to_vec();
        if lat_shape != mshape {
            return Err(Error::Shape(format!(
                "Euler step: latents {:?} ≠ model_output {:?}",
                lat_shape, mshape
            )));
        }
        let sigma_cur = self.sigmas[pos];
        let sigma_next = if pos + 1 < self.sigmas.len() {
            self.sigmas[pos + 1]
        } else {
            0.0
        };
        let lat = latents.to_vec_f32()?;
        let noise = model_output.to_vec_f32()?;
        let n = lat.len();
        let dt = sigma_next - sigma_cur;
        // Euler update on the probability-flow ODE with epsilon-prediction:
        //   x_{t-1} = x_t + (sigma_next - sigma_cur) * (x_t - eps) / sigma_cur
        let denom = if sigma_cur.abs() > 1e-12 {
            sigma_cur
        } else {
            1.0
        };
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let dx = ((lat[i] - noise[i]) / denom) * dt;
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
    fn ddim_step_recovers_clean_x0_from_noise_prediction() {
        // At the final timestep (alpha_prev = 1), a correct epsilon-prediction
        // recovers the clean image: x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t).
        let sched = DdimScheduler::linear(4, 0.0001, 0.02);
        let t = sched.timesteps[sched.timesteps.len() - 1]; // final (smallest) timestep
        let idx = t as usize;
        let alpha_t = sched.alphas_cumprod[idx];
        let s_t = alpha_t.max(1e-12).sqrt();
        let s_1m = (1.0 - alpha_t).max(0.0).sqrt();
        // Construct latents/noise so x0_pred = [2.0, -3.0, 0.5, 7.0].
        let x0: Vec<f32> = vec![2.0, -3.0, 0.5, 7.0];
        let eps: Vec<f32> = vec![0.1, 0.2, -0.3, 0.4];
        let lat: Vec<f32> = x0
            .iter()
            .zip(eps.iter())
            .map(|(x, e)| x * s_t + s_1m * e)
            .collect();
        let out = <DdimScheduler as NoiseScheduler>::step(
            &sched,
            &tensor_with(eps, vec![4]),
            &tensor_with(lat, vec![4]),
            t,
        )
        .unwrap();
        let v = out.to_vec_f32().unwrap();
        for (got, want) in v.iter().zip(x0.iter()) {
            assert!((got - want).abs() < 1e-5, "expected {want}, got {got}");
        }
    }

    #[test]
    fn euler_linear_schedule_basic() {
        let sched = EulerScheduler::linear(10, 0.0001, 0.02);
        assert_eq!(sched.sigmas.len(), 10);
        assert_eq!(sched.timesteps.len(), 10);
        // sigma grows with added noise: descending timestep → larger sigma.
        let s0 = sched.timesteps[0] as usize;
        let sl = sched.timesteps[sched.timesteps.len() - 1] as usize;
        assert!(sched.sigmas[s0] > sched.sigmas[sl]);
        // timesteps are descending.
        for w in sched.timesteps.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn euler_unknown_timestep_is_error() {
        let sched = EulerScheduler::linear(4, 0.0001, 0.02);
        let lat = tensor_with(vec![1.0f32; 8], vec![2, 4]);
        let n = tensor_with(vec![0.1f32; 8], vec![2, 4]);
        let err = <EulerScheduler as NoiseScheduler>::step(&sched, &n, &lat, 9999)
            .err()
            .expect("step should fail on unknown timestep");
        match err {
            Error::Config(_) => {}
            other => panic!("expected Config error, got {:?}", other),
        }
    }

    #[test]
    fn euler_step_applies_dt() {
        // From a 4-step linear schedule, step at the noisiest timestep (pos 0).
        // sigma_cur = sigma[timestep=3], sigma_next = sigma[timestep=2].
        let sched = EulerScheduler::linear(4, 0.0001, 0.02);
        let lat = tensor_with(vec![1.0f32; 4], vec![4]);
        let noise = tensor_with(vec![0.5f32; 4], vec![4]);
        let t0 = sched.timesteps[0];
        let out = <EulerScheduler as NoiseScheduler>::step(&sched, &noise, &lat, t0).unwrap();
        let v = out.to_vec_f32().unwrap();
        // Euler: x + ((x - eps)/sigma_cur) * (sigma_next - sigma_cur).
        let sigma_cur = sched.sigmas[0];
        let sigma_next = sched.sigmas[1];
        let expected = 1.0 + ((1.0 - 0.5) / sigma_cur) * (sigma_next - sigma_cur);
        for x in &v {
            assert!((*x - expected).abs() < 1e-6, "expected {expected}, got {x}");
        }
    }
}
