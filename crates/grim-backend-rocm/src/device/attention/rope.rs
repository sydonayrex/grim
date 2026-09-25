//! RoPE-family launchers (yarn/base/mRoPE, RMSNorm+RoPE fusion).

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    HipDim3, RocmHandle, arg, as_rocm, dev_ptr, dtype_f32, hipFreeAsync, linear_launch,
    upload_device_buffer,
};

impl RocmDevice {
    /// GPU-side YaRN / partial-rotary RoPE: computes `inv_freq[]` on the host, uploads it once per call, then dispatches `grim_rope_yarn` entirely on-device.
    /// # Contract - `x_s` must have a valid device pointer (caller checks `device_ptr_is_valid`).
    pub(crate) fn rope_launch_yarn(
        &self,
        x_s: &RocmStorage,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dims = out_shape.dims();
        if dims.len() != 3 || dims[2] != cfg.dim {
            return Err(Error::Shape(format!(
                "rope_launch_yarn: expected [B,S,D={}], got {:?}",
                cfg.dim, dims
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let rotary_dim = cfg.rotary_dim.min(d);
        let rotary_half = rotary_dim / 2;
        let yarn = cfg.yarn;

        if positions.len() != s {
            return Err(Error::Shape(
                "rope_launch_yarn: positions length must match seq_len".into(),
            ));
        }

        // Build the YaRN-ramp-corrected inv_freq[] on the host — O(rotary_half) work,
        // negligible vs kernel launch overhead. This avoids storing per-layer buffers.
        let inv_freq: Vec<f32> = (0..rotary_half)
            .map(|i| {
                let freq = 1.0_f32 / cfg.base.powf((2 * i) as f32 / d as f32);
                match yarn {
                    None => freq,
                    Some(y) => {
                        let wavelength = 2.0 * std::f32::consts::PI / freq;
                        let low = y.original_max_pos as f32 / y.beta_slow;
                        let high = y.original_max_pos as f32 / y.beta_fast;
                        if wavelength < high {
                            freq
                        } else if wavelength > low {
                            freq / y.factor
                        } else {
                            let ramp = (y.original_max_pos as f32 / wavelength - y.beta_slow)
                                / (y.beta_fast - y.beta_slow);
                            (1.0 - ramp) * (freq / y.factor) + ramp * freq
                        }
                    }
                }
            })
            .collect();
        let mscale = yarn.map(|y| y.attention_factor).unwrap_or(1.0_f32);

        // Upload positions and inv_freq to device-resident scratch buffers.
        // These are temporary allocations freed after the stream synchronises.
        let mut pos_ptr = upload_device_buffer(self.ordinal, positions)?;
        let mut freq_ptr = upload_device_buffer(self.ordinal, &inv_freq)?;

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut b_i = b as i32;
        let mut s_i = s as i32;
        let mut d_i = d as i32;
        let mut rh_i = rotary_half as i32;
        let mut ms_f = mscale;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };

        // Launch grid covers max(b*s*rotary_half, b*s*copy_len) threads to
        // handle both the rotate pass and the verbatim-copy pass in one kernel launch.
        let copy_len = d - 2 * rotary_half;
        let total = b
            * s
            * rotary_half
                .max(if copy_len > 0 { copy_len } else { 0 })
                .max(1);
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_rope_yarn",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut freq_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut rh_i),
                arg(&mut ms_f),
                arg(&mut inter_i),
            ],
        );

        // Free scratch device buffers stream-ordered (after the kernel's
        // reads); graph-capturable and no host stall.
        unsafe {
            let free_stream = stream
                .as_ref()
                .map(|_| self.active_stream())
                .unwrap_or(std::ptr::null_mut());
            let _ = hipFreeAsync(pos_ptr, free_stream);
            let _ = hipFreeAsync(freq_ptr, free_stream);
        }

        let stream = stream?;

        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    /// Item 3: device-base RoPE writing into a CALLER-PROVIDED output buffer.
    /// No allocation inside — required for HIP graph capture (stable pointers
    /// across replays). Same kernel as `rope_dev_base`, different output target.
    pub fn rope_dev_base_into(
        &self,
        q_storage: &dyn BackendStorage,
        pos_base_dev: &dyn BackendStorage,
        out_storage: &RocmStorage,
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
        num_heads: usize,
        steps: usize,
    ) -> Result<()> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let q_s = as_rocm(q_storage)?;
        let pos_s = as_rocm(pos_base_dev)?;
        if !q_s.device_ptr_is_valid() || !pos_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope_dev_base_into: input lacks a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "rope_dev_base_into expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let expected_s = (num_heads * steps) as i32;
        if s != expected_s {
            return Err(Error::Shape(format!(
                "rope_dev_base_into: out_shape middle dim {s} != num_heads({num_heads})*steps({steps})={expected_s}"
            )));
        }
        let d = dim as i32;
        let half = d / 2;

        let mut out_ptr = dev_ptr(out_storage)?;
        let mut x_ptr = dev_ptr(q_s)?;
        let mut pos_ptr = dev_ptr(pos_s)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };
        let mut heads_i = num_heads as i32;

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        self.launch_compute_kernel(
            "grim_rope_dev_base",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
                arg(&mut heads_i),
            ],
        )?;

        Ok(())
    }

    /// PLAN-decode-throughput-restore: fused QK-norm + device-base RoPE,
    /// written into caller-provided buffers (capture-safe, in place OK).
    /// Normalizes each head row with `gamma`/`eps` (plain RMS over head_dim),
    /// then applies NeoX/interleaved rotation. Requires head_dim == 64
    /// (half == warpSize) — the caller falls back to the split path otherwise.
    pub fn qk_rope_dev_base_into(
        &self,
        x: &dyn BackendStorage,
        pos_base_dev: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
        eps: f32,
        out: &RocmStorage,
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
        num_heads: usize,
        steps: usize,
    ) -> Result<()> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let x_s = as_rocm(x)?;
        let pos_s = as_rocm(pos_base_dev)?;
        if !x_s.device_ptr_is_valid() || !pos_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "qk_rope_dev_base_into: input lacks a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "qk_rope_dev_base_into expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let expected_s = (num_heads * steps) as i32;
        if s != expected_s {
            return Err(Error::Shape(format!(
                "qk_rope_dev_base_into: middle dim {s} != num_heads*steps={expected_s}"
            )));
        }
        let d = dim as i32;
        let half = d / 2;
        if half != 32 {
            return Err(Error::Backend(format!(
                "qk_rope_dev_base_into requires head_dim 64 (half==warpSize), got {d}"
            )));
        }

        let mut out_ptr = dev_ptr(out)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut pos_ptr = dev_ptr(pos_s)?;
        let mut g_ptr = dev_ptr(as_rocm(gamma)?)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };
        let mut heads_i = num_heads as i32;
        let mut eps_f = eps;

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        self.launch_compute_kernel(
            "grim_qk_rope_dev_base",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut g_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
                arg(&mut heads_i),
                arg(&mut eps_f),
            ],
        )?;
        Ok(())
    }

    /// PLAN 4 Task 4: QK-rope + KV-append fusion for the K/V pair — one
    /// launch replaces (qk_rope k, kv_append k, kv_append v). Bit-identical
    /// to the separate launches (pair-identical thread mapping, same
    /// offsets, same f16 conversion). Q rope stays separate (no append);
    /// bump stays after attention (which reads pre-bump total_dev).
    /// Same `half == 32` warpSize constraint as `qk_rope_dev_base_into`;
    /// caller falls back to the split path otherwise. Graph-capture safe
    /// (caller-owned buffers, no H2D/sync/alloc).
    ///
    /// F16 gate replicated from `kernels::qkv_attention::kv_f16_enabled`
    /// (not imported: that module's f16 plumbing stays out of this commit;
    /// this 4-line contract is pinned by `docs/debug-vars.md`).
    pub fn qk_rope_append_kv_into(
        &self,
        k: &dyn BackendStorage,
        pos_base_dev: &dyn BackendStorage,
        gamma_k: &dyn BackendStorage,
        eps: f32,
        v: &dyn BackendStorage,
        k_out: &RocmStorage,
        k_arena: &dyn BackendStorage,
        v_arena: &dyn BackendStorage,
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
        nkv_heads: usize,
        steps: usize,
        kv_stride: usize,
        arena_slot_stride: usize,
    ) -> Result<()> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        // M3 fail-closed (mirrors kv_append): arena dtype must match GRIM_F16_KV.
        fn rope_kv_f16_enabled() -> bool {
            std::env::var("GRIM_F16_KV")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(false)
        }
        for (arena, what) in [(k_arena, "rope_append k"), (v_arena, "rope_append v")] {
            let s = arena
                .as_any()
                .downcast_ref::<RocmStorage>()
                .ok_or_else(|| Error::Backend(format!("{what}: arena must be RocmStorage")))?;
            let want_f16 = rope_kv_f16_enabled();
            let is_f16 = s.dtype.arith == grim_tensor::dtype::ArithType::F16;
            if is_f16 != want_f16 {
                return Err(Error::Backend(format!(
                    "{what}: arena dtype {:?} does not match GRIM_F16_KV={want_f16}",
                    s.dtype.arith,
                )));
            }
        }
        let dim = cfg.dim;
        let base = cfg.base;
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "qk_rope_append_kv expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let expected_s = (nkv_heads * steps) as i32;
        if s != expected_s {
            return Err(Error::Shape(format!(
                "qk_rope_append_kv: middle dim {s} != nkv_heads*steps={expected_s}"
            )));
        }
        let d = dim as i32;
        let half = d / 2;
        if half != 32 {
            return Err(Error::Backend(format!(
                "qk_rope_append_kv requires head_dim 64 (half==warpSize), got {d}"
            )));
        }
        let k_s = as_rocm(k)?;
        let pos_s = as_rocm(pos_base_dev)?;
        let v_s = as_rocm(v)?;
        if !k_s.device_ptr_is_valid() || !pos_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "qk_rope_append_kv: input lacks a valid device pointer".into(),
            ));
        }
        let mut k_ptr = dev_ptr(k_s)?;
        let mut pos_ptr = dev_ptr(pos_s)?;
        let mut g_ptr = dev_ptr(as_rocm(gamma_k)?)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut k_out_ptr = dev_ptr(k_out)?;
        let mut k_arena_ptr = dev_ptr(as_rocm(k_arena)?)?;
        let mut v_arena_ptr = dev_ptr(as_rocm(v_arena)?)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };
        let mut nkv_i = nkv_heads as i32;
        let mut eps_f = eps;
        let mut kv_stride_i = kv_stride as i32;
        let mut steps_i = steps as i32;
        let mut slot_stride_i = arena_slot_stride as i32;
        let mut f16_i = rope_kv_f16_enabled() as i32;

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        self.launch_compute_kernel(
            "grim_qk_rope_append_kv",
            grid,
            block,
            &mut [
                arg(&mut k_ptr),
                arg(&mut pos_ptr),
                arg(&mut g_ptr),
                arg(&mut v_ptr),
                arg(&mut k_out_ptr),
                arg(&mut k_arena_ptr),
                arg(&mut v_arena_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
                arg(&mut nkv_i),
                arg(&mut eps_f),
                arg(&mut kv_stride_i),
                arg(&mut steps_i),
                arg(&mut slot_stride_i),
                arg(&mut f16_i),
            ],
        )?;
        Ok(())
    }

    /// Item 2: device-base RoPE for the decode path. Instead of uploading a
    /// per-layer per-token `positions[]` host vector, the single base position
    /// lives in a device buffer (`pos_base_dev`, one u32) and the kernel derives
    /// each step's position as `base + si` internally. Same fp32 math as `rope`.
    ///
    /// `pos_base_dev` must point to device memory holding one u32 (the absolute
    /// position of step 0); `steps` query positions are rotated by base..base+steps-1.
    /// `q_storage` is the per-head, RoPE-normalized Q/K (already reshaped to
    /// `[1, heads*steps, head_dim]`). Returns the rotated storage.
    pub fn rope_dev_base(
        &self,
        q_storage: &dyn BackendStorage,
        pos_base_dev: &dyn BackendStorage,
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
        num_heads: usize,
        steps: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let q_s = as_rocm(q_storage)?;
        let pos_s = as_rocm(pos_base_dev)?;
        if !q_s.device_ptr_is_valid() || !pos_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope_dev_base: input lacks a valid device pointer".into(),
            ));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "rope_dev_base expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let expected_s = (num_heads * steps) as i32;
        if s != expected_s {
            return Err(Error::Shape(format!(
                "rope_dev_base: out_shape middle dim {s} != num_heads({num_heads})*steps({steps})={expected_s}"
            )));
        }
        let d = dim as i32;
        let half = d / 2;

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(q_s)?;
        let mut pos_ptr = dev_ptr(pos_s)?;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };
        let mut heads_i = num_heads as i32;

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        self.launch_compute_kernel(
            "grim_rope_dev_base",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut pos_ptr),
                arg(&mut out_ptr),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
                arg(&mut heads_i),
            ],
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// SPEED-DOT-OPFUSE (Phase 4a): fused RMSNorm + RoPE for Q/K paths.
    /// Normalizes x across head_dim d with optional norm_weight, then applies RoPE rotation
    /// using positions.
    pub fn rmsnorm_rope(
        &self,
        x_storage: &dyn BackendStorage,
        norm_weight: Option<&dyn BackendStorage>,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
        eps: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dim = cfg.dim;
        let base = cfg.base;
        let x_s = as_rocm(x_storage)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rmsnorm_rope: x lacks valid device ptr".into(),
            ));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || out_dims[2] != dim {
            return Err(Error::Shape(format!(
                "rmsnorm_rope expects (B,S,D={}), got {:?}",
                dim, out_dims
            )));
        }
        let b = out_dims[0] as i32;
        let s = out_dims[1] as i32;
        let d = dim as i32;
        let half = d / 2;
        if positions.len() != s as usize {
            return Err(Error::Shape(
                "rmsnorm_rope: positions length must match seq_len".into(),
            ));
        }

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = match norm_weight {
            Some(w) => dev_ptr(as_rocm(w)?)?,
            None => 0u64,
        };
        let mut pos_ptr = upload_device_buffer(self.ordinal, positions)?;
        let mut eps_f = eps;
        let mut b_i = b;
        let mut s_i = s;
        let mut d_i = d;
        let mut half_i = half;
        let mut base_f = base;
        let mut inter_i = if cfg.interleaved { 1 } else { 0 };

        let total = (b * s * half) as usize;
        let (grid, block) = linear_launch(total);

        self.launch_compute_kernel(
            "grim_rmsnorm_rope",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut pos_ptr),
                arg(&mut out_ptr),
                arg(&mut eps_f),
                arg(&mut b_i),
                arg(&mut s_i),
                arg(&mut d_i),
                arg(&mut half_i),
                arg(&mut base_f),
                arg(&mut inter_i),
            ],
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch LFM2-style fused QKV projection: MXFP4 GEMM (x @ W_qkv) followed by per-head QK-Norm + RoPE (YaRN-aware).
    /// The GEMM result is staged in a scratch buffer (or `out_all` if provided) and consumed.
    pub fn launch_fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        x_storage: &RocmStorage,
        gamma_q_storage: &RocmStorage,
        gamma_k_storage: &RocmStorage,
        w_codes_storage: &RocmStorage,
        w_exps_storage: &RocmStorage,
        q_out_storage: Option<&RocmStorage>,
        k_cache_storage: Option<&RocmStorage>,
        v_cache_storage: Option<&RocmStorage>,
        out_all_storage: Option<&RocmStorage>,
        positions_storage: Option<&RocmStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq_storage: Option<&RocmStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
        rope_interleaved: bool,
    ) -> Result<*mut c_void> {
        let gamma_q_ptr = gamma_q_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gamma_q has no device ptr".into())
        })?;
        let gamma_k_ptr = gamma_k_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gamma_k has no device ptr".into())
        })?;

        let n_q = num_q_heads * head_dim;
        let n_k = num_kv_heads * head_dim;
        let n_total = n_q + 2 * n_k;

        // Stage the raw QKV GEMM output. Reuse `out_all` if supplied; otherwise
        // allocate a transient scratch buffer freed after the stream syncs.
        let scratch = if out_all_storage.is_some() {
            None
        } else {
            Some(RocmStorage::alloc_gpu(
                &Shape::from_slice(&[m, n_total]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?)
        };
        let gemm_storage: &RocmStorage = match (out_all_storage, &scratch) {
            (Some(o), _) => o,
            (None, Some(s)) => s,
            (None, None) => unreachable!(),
        };
        let gemm_ptr = gemm_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: gemm buffer has no device ptr".into())
        })?;

        // Phase 1: MXFP4 GEMM -> gemm_out (C = x @ W_qkv)
        self.launch_mxfp4_gemm_tiled(
            x_storage,
            w_codes_storage.device_ptr_u64().ok_or_else(|| {
                Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: codes ptr".into())
            })?,
            w_exps_storage.device_ptr_u64().ok_or_else(|| {
                Error::Backend("fused_mxfp4_gemm_qk_norm_rope_kv: exps ptr".into())
            })?,
            gemm_storage,
            m,
            n_total,
            k,
        )?;

        // Phase 2: per-head QK-Norm + RoPE -> q_out / k_cache / v_cache
        let q_out_ptr = q_out_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let k_cache_ptr = k_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let v_cache_ptr = v_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let positions_ptr = positions_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let inv_freq_ptr = inv_freq_storage.and_then(|s| s.device_ptr).unwrap_or(0);

        let total = m * (num_q_heads + 2 * num_kv_heads);
        let (grid, block) = linear_launch(total);

        let mut gemmptr = gemm_ptr;
        let mut gqptr = gamma_q_ptr;
        let mut gkptr = gamma_k_ptr;
        let mut posptr = positions_ptr;
        let mut qptr = q_out_ptr;
        let mut kptr = k_cache_ptr;
        let mut vptr = v_cache_ptr;
        let mut mm = m as i32;
        let mut nq = num_q_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut theta = rope_theta;
        let mut invfreqptr = inv_freq_ptr;
        let mut mscale_val = mscale;
        let mut eps_val = eps;
        let mut max_seq = max_seq_len as i32;
        let mut rope_ilv = rope_interleaved as i32;

        let stream = self.launch_compute_kernel(
            "grim_qk_norm_rope",
            grid,
            block,
            &mut [
                arg(&mut gemmptr),
                arg(&mut gqptr),
                arg(&mut gkptr),
                arg(&mut posptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut theta),
                arg(&mut invfreqptr),
                arg(&mut mscale_val),
                arg(&mut eps_val),
                arg(&mut max_seq),
                arg(&mut rope_ilv),
            ],
        )?;

        // The transient scratch buffer returns to the caching allocator when `scratch` drops at scope exit.
        // The previous code additionally hipFree'd the pointer by hand - a double free that also.
        drop(scratch);

        Ok(stream)
    }

    /// Launch Multimodal 3D Rotary Position Embedding (M-RoPE) for Q and K tensors.
    pub fn launch_mrope_qk(
        &self,
        q_storage: &RocmStorage,
        k_storage: &RocmStorage,
        positions_storage: &RocmStorage,
        num_tokens: usize,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        section_t: usize,
        section_h: usize,
        section_w: usize,
        rope_theta: f32,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: q has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: k has no device ptr".into()))?;
        let pos_ptr = positions_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mrope_qk: positions has no device ptr".into()))?;

        let block_dim = HipDim3::new((rotary_dim / 2) as u32, 1, 1);
        let grid_dim = HipDim3::new(num_tokens as u32, (num_q_heads + num_k_heads) as u32, 1);

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut posptr = pos_ptr;
        let mut nt = num_tokens as i32;
        let mut nqh = num_q_heads as i32;
        let mut nkh = num_k_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut st = section_t as i32;
        let mut sh = section_h as i32;
        let mut sw = section_w as i32;
        let mut theta = rope_theta;

        self.launch_compute_kernel(
            "grim_mrope_qk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut posptr),
                arg(&mut nt),
                arg(&mut nqh),
                arg(&mut nkh),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut st),
                arg(&mut sh),
                arg(&mut sw),
                arg(&mut theta),
            ],
        )
    }
}
