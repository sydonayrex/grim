//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! The trait-required `impl SamplingOps for RocmDevice` block, kept whole.

use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, ElementwiseOps, SamplingOps};

use crate::as_rocm;
use crate::device::roc_device::RocmDevice;

fn log_device_sampler_fallback(reason: &str) {
    use std::sync::OnceLock;
    static LOGGED: OnceLock<bool> = OnceLock::new();
    if LOGGED.get().is_none() {
        eprintln!("[sampling] device sampler unavailable ({reason}); falling back to CPU");
        LOGGED.set(true).ok();
    }
}

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
            match crate::kernels::device_sampler::sample_logits_on_device(
                self,
                rocm_s,
                vocab,
                temperature,
                top_k as i32,
                top_p,
                seed,
            ) {
                Ok(Some(token)) => return Ok(token),
                Ok(None) => log_device_sampler_fallback("validation rejected the device path"),
                Err(e) => log_device_sampler_fallback(&format!("kernel failed: {e}")),
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

    fn speculative_rejection_sample(
        &self,
        target_probs: &dyn BackendStorage,
        draft_probs: &dyn BackendStorage,
        draft_tokens: &dyn BackendStorage,
        uniform_rands: &dyn BackendStorage,
        accepted_tokens: &dyn BackendStorage,
        accepted_lens: &dyn BackendStorage,
        batch_size: usize,
        num_draft_tokens: usize,
        vocab_size: usize,
    ) -> Result<()> {
        let tp = as_rocm(target_probs)?;
        let dp = as_rocm(draft_probs)?;
        let dt = as_rocm(draft_tokens)?;
        let ur = as_rocm(uniform_rands)?;
        let at = as_rocm(accepted_tokens)?;
        let al = as_rocm(accepted_lens)?;

        self.launch_speculative_rejection_sample(
            tp,
            dp,
            dt,
            ur,
            at,
            al,
            batch_size,
            num_draft_tokens,
            vocab_size,
        )?;
        Ok(())
    }

    fn speculative_tree_verify(
        &self,
        candidate_paths: &dyn BackendStorage,
        target_logits: &dyn BackendStorage,
        path_lens: &dyn BackendStorage,
        best_path_idx: &dyn BackendStorage,
        best_accepted_len: &dyn BackendStorage,
        num_paths: usize,
        max_path_len: usize,
        vocab_size: usize,
    ) -> Result<()> {
        let cp = as_rocm(candidate_paths)?;
        let tl = as_rocm(target_logits)?;
        let pl = as_rocm(path_lens)?;
        let bpi = as_rocm(best_path_idx)?;
        let bal = as_rocm(best_accepted_len)?;

        self.launch_speculative_tree_verify(
            cp,
            tl,
            pl,
            bpi,
            bal,
            num_paths,
            max_path_len,
            vocab_size,
        )?;
        Ok(())
    }
}

impl RocmDevice {
    /// Launch GPU speculative tree verify kernel (Task 10.6).
    pub fn launch_speculative_tree_verify(
        &self,
        candidate_paths: &crate::memory::storage::RocmStorage,
        target_logits: &crate::memory::storage::RocmStorage,
        path_lens: &crate::memory::storage::RocmStorage,
        best_path_idx: &crate::memory::storage::RocmStorage,
        best_accepted_len: &crate::memory::storage::RocmStorage,
        num_paths: usize,
        max_path_len: usize,
        vocab_size: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let cp_ptr = candidate_paths
            .device_ptr
            .ok_or_else(|| Error::Backend("tree_verify: candidate_paths has no device ptr".into()))?;
        let tl_ptr = target_logits
            .device_ptr
            .ok_or_else(|| Error::Backend("tree_verify: target_logits has no device ptr".into()))?;
        let pl_ptr = path_lens
            .device_ptr
            .ok_or_else(|| Error::Backend("tree_verify: path_lens has no device ptr".into()))?;
        let bpi_ptr = best_path_idx
            .device_ptr
            .ok_or_else(|| Error::Backend("tree_verify: best_path_idx has no device ptr".into()))?;
        let bal_ptr = best_accepted_len
            .device_ptr
            .ok_or_else(|| Error::Backend("tree_verify: best_accepted_len has no device ptr".into()))?;

        let grid_dim = crate::HipDim3::new((num_paths as u32).div_ceil(256), 1, 1);
        let block_dim = crate::HipDim3::new(256, 1, 1);

        let mut cpptr = cp_ptr;
        let mut tlptr = tl_ptr;
        let mut plptr = pl_ptr;
        let mut bpiptr = bpi_ptr;
        let mut balptr = bal_ptr;
        let mut np = num_paths as i32;
        let mut mpl = max_path_len as i32;
        let mut vs = vocab_size as i32;

        self.launch_compute_kernel(
            "grim_speculative_tree_verify",
            grid_dim,
            block_dim,
            &mut [
                crate::arg(&mut cpptr),
                crate::arg(&mut tlptr),
                crate::arg(&mut plptr),
                crate::arg(&mut bpiptr),
                crate::arg(&mut balptr),
                crate::arg(&mut np),
                crate::arg(&mut mpl),
                crate::arg(&mut vs),
            ],
        )
    }

    /// Launch GPU batch argmax kernel (Task 10.6).
    pub fn launch_sample_logits_argmax_batch(
        &self,
        logits: &crate::memory::storage::RocmStorage,
        out_tokens: &crate::memory::storage::RocmStorage,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let logits_ptr = logits
            .device_ptr
            .ok_or_else(|| Error::Backend("argmax_batch: logits has no device ptr".into()))?;
        let out_ptr = out_tokens
            .device_ptr
            .ok_or_else(|| Error::Backend("argmax_batch: out_tokens has no device ptr".into()))?;

        let grid_dim = crate::HipDim3::new(batch_size as u32, 1, 1);
        let block_dim = crate::HipDim3::new(256, 1, 1);

        let mut lptr = logits_ptr;
        let mut optr = out_ptr;
        let mut bs = batch_size as i32;
        let mut vs = vocab_size as i32;

        self.launch_compute_kernel(
            "grim_sample_logits_argmax",
            grid_dim,
            block_dim,
            &mut [
                crate::arg(&mut lptr),
                crate::arg(&mut optr),
                crate::arg(&mut bs),
                crate::arg(&mut vs),
            ],
        )
    }
}
