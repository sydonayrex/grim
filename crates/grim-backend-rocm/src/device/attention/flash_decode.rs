//! FlashDecoding launchers, split-count heuristic and block-dim autotune.

use std::ffi::c_void;
use std::sync::Mutex;

use grim_tensor::Shape;
use grim_tensor::error::{Error, Result};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{HipDim3, arg, dev_ptr, dtype_f32, hipStreamSynchronize, hipSuccess};

impl RocmDevice {
    /// Launch FlashDecoding (Split-KV Parallel Attention) across sequence chunks + merge reduction.
    pub fn launch_flash_decode(
        &self,
        q_storage: &RocmStorage,
        k_storage: &RocmStorage,
        v_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        num_splits: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: q has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: k has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: v has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("flash_decode: out has no device ptr".into()))?;

        let num_splits = num_splits.max(1);
        let mid_out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads, head_dim]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mid_max_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mid_sum_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_splits, num_heads]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;

        let mut mid_out_ptr = dev_ptr(&mid_out_storage)?;
        let mut mid_max_ptr = dev_ptr(&mid_max_storage)?;
        let mut mid_sum_ptr = dev_ptr(&mid_sum_storage)?;

        let block_dim = HipDim3::new(head_dim.max(32).next_power_of_two() as u32, 1, 1);
        let grid_stage1 = HipDim3::new(num_heads as u32, num_splits as u32, 1);

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut nh = num_heads as i32;
        let mut nkvh = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut slen = kv_seq_len as i32;
        let mut nsplits = num_splits as i32;
        let mut inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        // Stage 1
        let lds_stage1_bytes = (head_dim + block_dim.x as usize) * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_flash_decode_stage1",
            grid_stage1,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mid_out_ptr),
                arg(&mut mid_max_ptr),
                arg(&mut mid_sum_ptr),
                arg(&mut nh),
                arg(&mut nkvh),
                arg(&mut hd),
                arg(&mut slen),
                arg(&mut nsplits),
                arg(&mut inv_sqrt_d),
            ],
            None,
            lds_stage1_bytes,
        )?;

        // Stage 2
        let grid_stage2 = HipDim3::new(num_heads as u32, 1, 1);
        let mut optr = out_ptr;
        self.launch_compute_kernel(
            "grim_flash_decode_stage2",
            grid_stage2,
            block_dim,
            &mut [
                arg(&mut mid_out_ptr),
                arg(&mut mid_max_ptr),
                arg(&mut mid_sum_ptr),
                arg(&mut optr),
                arg(&mut nh),
                arg(&mut hd),
                arg(&mut nsplits),
            ],
        )
    }

    /// Launch DeepSeek Multi-Head Latent Attention (MLA) Matrix-Absorbed Decode.
    pub fn launch_mla_absorbed_decode(
        &self,
        q_absorbed: &RocmStorage,
        q_rope: &RocmStorage,
        kv_cache: &RocmStorage,
        w_uv: Option<&RocmStorage>,
        out: &RocmStorage,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        seq_len: usize,
        w_uv_offset_words: usize,
        w_uv_head_stride_words: usize,
    ) -> Result<*mut c_void> {
        let q_abs_ptr = q_absorbed.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: q_absorbed has no device ptr".into())
        })?;
        let q_rope_ptr = q_rope.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: q_rope has no device ptr".into())
        })?;
        let kv_ptr = kv_cache.device_ptr.ok_or_else(|| {
            Error::Backend("mla_absorbed_decode: kv_cache has no device ptr".into())
        })?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: out has no device ptr".into()))?;
        let w_uv_ptr = w_uv.and_then(|s| s.device_ptr).unwrap_or(0);
        let has_w_uv = if w_uv.is_some() { 1i32 } else { 0i32 };

        let block_dim = HipDim3::new(256, 1, 1);
        let grid_dim = HipDim3::new(num_heads as u32, 1, 1);

        let mut qabsptr = q_abs_ptr;
        let mut qropeptr = q_rope_ptr;
        let mut kvptr = kv_ptr;
        let mut wuvptr = w_uv_ptr;
        let mut optr = out_ptr;
        let mut nh = num_heads as i32;
        let mut lora_r = kv_lora_rank as i32;
        let mut rope_d = qk_rope_dim as i32;
        let mut v_dim = v_head_dim as i32;
        let mut slen = seq_len as i32;
        let mut inv_sqrt = 1.0f32 / ((kv_lora_rank + qk_rope_dim) as f32).sqrt();
        let mut has_w = has_w_uv;
        let mut w_off = w_uv_offset_words as i32;
        let mut w_stride = w_uv_head_stride_words as i32;

        let lds_bytes = 256 * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_mla_absorbed_decode",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qabsptr),
                arg(&mut qropeptr),
                arg(&mut kvptr),
                arg(&mut wuvptr),
                arg(&mut optr),
                arg(&mut nh),
                arg(&mut lora_r),
                arg(&mut rope_d),
                arg(&mut v_dim),
                arg(&mut slen),
                arg(&mut inv_sqrt),
                arg(&mut has_w),
                arg(&mut w_off),
                arg(&mut w_stride),
            ],
            None,
            lds_bytes,
        )
    }

    /// Minimum KV sequence length to trigger Split-KV FlashDecoding.
    /// Can be overridden via `GRIM_FLASH_DECODE_MIN_KV`.
    /// Defaults to 256 for RDNA3/4 (gfx11/gfx12) and 512 for other architectures.
    pub(crate) fn flash_decode_min_kv(&self) -> usize {
        if let Ok(v) = std::env::var("GRIM_FLASH_DECODE_MIN_KV") {
            if let Ok(parsed) = v.parse::<usize>() {
                return parsed;
            }
        }
        if self.is_rdna34 { 256 } else { 512 }
    }

    /// Split-KV count for FlashDecoding: consult the autotuner (persisted in `.autotune_cache/{gpu_target}.json`) keyed by `(num_heads, head_dim, kv_len)`; on miss return the static heuristic.
    /// With `GRIM_ATTENTION_AUTOTUNE=1` (and outside stream capture), a miss instead benchmarks candidate split counts with real.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_decode_split_count(
        &self,
        q_s: &RocmStorage,
        k_s: &RocmStorage,
        v_s: &RocmStorage,
        out: &RocmStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
    ) -> usize {
        let heuristic = (kv_seq_len / 256).clamp(2, 64);
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let key = crate::autotune::KernelKey {
            kernel: "grim_flash_decode",
            gpu_arch: arch_leak,
            m: num_heads,
            n: head_dim,
            k: kv_seq_len.clamp(1, 1 << 16),
        };
        let Ok(mut tuner) = self.autotuner.lock() else {
            return heuristic;
        };
        if let Some(cfg) = tuner.lookup(key) {
            return cfg.tile_kv.max(1) as usize;
        }
        let tune_enabled = std::env::var("GRIM_ATTENTION_AUTOTUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !tune_enabled || self.active_capture_stream().is_some() {
            return heuristic;
        }

        // Bench candidate splits: real launches on the active stream, timed wall-clock (launch + synchronize).
        // Each candidate runs 3 iterations after 1 warmup; the minimum wins.
        let mut candidates: Vec<usize> = [2usize, 4, 8, 16, 32, 64]
            .into_iter()
            .filter(|&s| s <= kv_seq_len.max(2))
            .collect();
        if !candidates.contains(&heuristic) {
            candidates.push(heuristic);
        }
        let mut best = (heuristic, f64::INFINITY);
        for &splits in &candidates {
            let mut best_ms = f64::INFINITY;
            for iter in 0..4 {
                if let Err(e) = self.launch_flash_decode(
                    q_s,
                    k_s,
                    v_s,
                    out,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_seq_len,
                    splits,
                ) {
                    // A failing candidate (e.g. LDS overflow at high split
                    // counts) is simply not viable; skip it.
                    let _ = e;
                    best_ms = f64::INFINITY;
                    break;
                }
                if iter == 0 {
                    continue; // warmup
                }
                let t = std::time::Instant::now();
                if let Err(e) = self.launch_flash_decode(
                    q_s,
                    k_s,
                    v_s,
                    out,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_seq_len,
                    splits,
                ) {
                    let _ = e;
                    best_ms = f64::INFINITY;
                    break;
                }
                let stream = self.active_stream();
                if unsafe { hipStreamSynchronize(stream) } != hipSuccess {
                    best_ms = f64::INFINITY;
                    break;
                }
                let ms = t.elapsed().as_secs_f64() * 1e3;
                best_ms = best_ms.min(ms);
            }
            if best_ms < best.1 {
                best = (splits, best_ms);
            }
        }
        if best.1.is_finite() {
            let cfg = crate::autotune::AutotuneConfig {
                block_dim: 256,
                tile_kv: best.0 as u32,
                grid_stride: 1,
                cycles_per_invocation: (best.1 * 1e6) as u64,
                spec_gamma: 4,
                spec_acceptance_threshold: 0.6,
                spec_alpha: 0.0,
                split_k: 0,
            };
            let _ = tuner.record(key, cfg);
            let _ = self.save_autotune_cache(std::path::Path::new(&format!(
                ".autotune_cache/{}.json",
                self.gpu_target
            )));
            best.0
        } else {
            heuristic
        }
    }

    /// WI-X5: RECORD side of the attention block-dim autotuner, shared by the dense `grim_qkv_attention` lookup (`qkv_attention`) and the paged `grim_qkv_attention_paged` launch sites.
    /// Callers consult `tuner.lookup(key)` FIRST; this helper runs only on a cache miss so the hot.
    pub(crate) fn autotune_attention_block_dim(
        &self,
        key: crate::autotune::KernelKey,
        fallback_block_dim: u32,
        kv_seq_len: usize,
        min_kv_len: usize,
        mut launch_one: impl FnMut(u32) -> Result<()>,
    ) -> Option<u32> {
        let tune_enabled = std::env::var("GRIM_ATTENTION_AUTOTUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !tune_enabled || self.active_capture_stream().is_some() || kv_seq_len < min_kv_len {
            return None;
        }
        // Double-check under the tuner lock: another thread may have recorded
        // this key between the caller's lookup miss and this sweep.
        if let Ok(tuner) = self.autotuner.lock() {
            if tuner.lookup(key).is_some() {
                return None;
            }
        }
        // At most one sweep attempt per key per process. A failed sweep also
        // lands here — retrying every call would stall the decode loop.
        static SWEPT_KEYS: std::sync::OnceLock<Mutex<std::collections::HashSet<u64>>> =
            std::sync::OnceLock::new();
        let swept = SWEPT_KEYS.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
        let key_hash = seahash::hash(
            format!(
                "{}|{}|{}|{}|{}",
                key.kernel, key.gpu_arch, key.m, key.n, key.k
            )
            .as_bytes(),
        );
        match swept.lock() {
            Ok(mut set) => {
                if !set.insert(key_hash) {
                    return None;
                }
            }
            Err(_) => return None,
        }

        let wf = self.wavefront_size() as u32;
        let candidates: Vec<u32> = [64u32, 128, 256]
            .into_iter()
            .filter(|&d| d % wf == 0 && d / wf <= 8 && d != fallback_block_dim)
            .collect();
        if candidates.is_empty() {
            return None;
        }

        let mut best = (fallback_block_dim, f64::INFINITY);
        for cand in std::iter::once(fallback_block_dim).chain(candidates) {
            let mut best_ms = f64::INFINITY;
            for rep in 0..3 {
                if launch_one(cand).is_err() {
                    // Non-viable candidate (launch/arg error): disqualify.
                    best_ms = f64::INFINITY;
                    break;
                }
                if rep == 0 {
                    // Warmup: drain it so timed reps measure only themselves.
                    let _ = unsafe { hipStreamSynchronize(self.active_stream()) };
                    continue;
                }
                let t0 = std::time::Instant::now();
                if unsafe { hipStreamSynchronize(self.active_stream()) } != hipSuccess {
                    best_ms = f64::INFINITY;
                    break;
                }
                best_ms = best_ms.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            if best_ms < best.1 {
                best = (cand, best_ms);
            }
        }
        if !best.1.is_finite() {
            return None;
        }
        let cfg = crate::autotune::AutotuneConfig {
            block_dim: best.0,
            tile_kv: 64,
            grid_stride: 1,
            cycles_per_invocation: (best.1 * 1e6) as u64,
            spec_gamma: 4,
            spec_acceptance_threshold: 0.6,
            spec_alpha: 0.0,
            split_k: 0,
        };
        if let Ok(mut tuner) = self.autotuner.lock() {
            let _ = tuner.record(key, cfg);
        }
        let _ = self.save_autotune_cache(std::path::Path::new(&format!(
            ".autotune_cache/{}.json",
            self.gpu_target
        )));
        Some(best.0)
    }
}
