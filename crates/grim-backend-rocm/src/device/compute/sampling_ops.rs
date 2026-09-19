//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! The trait-required `impl SamplingOps for RocmDevice` block, kept whole.



use grim_tensor::error::{Error, Result};
use grim_tensor::{ BackendStorage, ElementwiseOps, SamplingOps };

use crate::device::roc_device::RocmDevice;
use crate::{ as_rocm };

impl SamplingOps for RocmDevice {
    /// B1 (PLAN-reduce-d2h-h2d): penalty-aware device sampling. Penalty
    /// pre-pass + sample stay on-device (one small H2D for history ids, 4-byte
    /// D2H for the token). `Err` (incl. `Ok(None)` from validation) → caller
    /// falls back to the CPU sampler; never silently samples unpenalized.
    fn sample_on_device_with_penalty(
        &self,
        logits: &dyn BackendStorage,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        seed: u64,
        repeat_penalty: f32,
        history: &[u32],
    ) -> Result<u32> {
        let vocab = logits.shape().dims().last().copied().unwrap_or(0);
        if let Ok(rocm_s) = as_rocm(logits) {
            if let Ok(Some(token)) =
                crate::kernels::device_sampler::sample_logits_on_device_with_penalty(
                    self,
                    rocm_s,
                    vocab,
                    temperature,
                    top_k as i32,
                    top_p,
                    seed,
                    repeat_penalty,
                    history,
                )
            {
                return Ok(token);
            }
        }
        Err(Error::Backend(
            "sample_on_device_with_penalty: device path unavailable".into(),
        ))
    }

    fn sample_on_device(
        &self,
        logits: &dyn BackendStorage,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        seed: u64,
    ) -> Result<u32> {
        let vocab = logits.shape().dims().last().copied().unwrap_or(0);
        // Greedy (temperature == 0): use the fast GPU-resident argmax kernel
        // which avoids the full-vocab D2H copy that the stochastic sampler
        // performs. This is the dominant decode path for temp=0.
        if temperature <= 0.0 {
            return self.argmax(logits);
        }
        if let Ok(rocm_s) = as_rocm(logits) {
            if let Ok(Some(token)) = crate::kernels::device_sampler::sample_logits_on_device(
                self,
                rocm_s,
                vocab,
                temperature,
                top_k as i32,
                top_p,
                seed,
            ) {
                return Ok(token);
            }
        }
        let cpu_logits = logits.to_cpu_vec_f32()?;
        if cpu_logits.is_empty() {
            return Err(Error::Backend("sample_on_device: empty logits".into()));
        }
        let max_logit = cpu_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return Err(Error::Backend(format!(
                "sample_on_device: logits have non-finite maximum ({max_logit})"
            )));
        }
        let inv_t = 1.0 / temperature;
        let mut exp_logits: Vec<f32> = cpu_logits
            .iter()
            .map(|&l| ((l - max_logit) * inv_t).exp())
            .collect();
        let sum: f32 = exp_logits.iter().sum();
        let inv_sum = 1.0 / sum;
        for p in &mut exp_logits {
            *p *= inv_sum;
        }
        let mut indexed: Vec<(usize, f32)> = exp_logits.into_iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 && (top_k as usize) < indexed.len() {
            indexed.truncate(top_k as usize);
        }
        if top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut keep = 0;
            for (_, p) in &indexed {
                cum += p;
                keep += 1;
                if cum >= top_p {
                    break;
                }
            }
            indexed.truncate(keep.max(1));
        }
        let sub_sum: f32 = indexed.iter().map(|(_, p)| p).sum();
        let inv_sub = 1.0 / sub_sum;
        let mut s = seed ^ (seed >> 30);
        s = s.wrapping_mul(0xbf58476d1ce4e5b9);
        s ^= s >> 27;
        s = s.wrapping_mul(0x94d049bb133111eb);
        s ^= s >> 31;
        let u = (s as f64) / (u64::MAX as f64);
        let mut acc = 0.0f64;
        for (idx, p) in indexed {
            acc += (p * inv_sub) as f64;
            if acc >= u {
                return Ok(idx as u32);
            }
        }
        Ok(0)
    }
}
