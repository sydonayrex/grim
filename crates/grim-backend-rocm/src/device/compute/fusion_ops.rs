//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! The trait-required `impl FusionOps for RocmDevice` block, kept whole.

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, FusionOps, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{arg, as_rocm, dev_ptr, dtype_f32, linear_launch, RocmHandle};

impl FusionOps for RocmDevice {
    fn fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma_q: &dyn BackendStorage,
        gamma_k: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        q_out: Option<&dyn BackendStorage>,
        k_cache: Option<&dyn BackendStorage>,
        v_cache: Option<&dyn BackendStorage>,
        out_all: Option<&dyn BackendStorage>,
        positions: Option<&dyn BackendStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq: Option<&dyn BackendStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
        rope_interleaved: bool,
    ) -> Result<Box<dyn ComputeHandle>> {
        self.fused_mxfp4_gemm_qk_norm_rope_kv(
            x,
            gamma_q,
            gamma_k,
            w_codes,
            w_exps,
            q_out,
            k_cache,
            v_cache,
            out_all,
            positions,
            m,
            k,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            inv_freq,
            mscale,
            eps,
            max_seq_len,
            rope_interleaved,
        )
    }

    /// Broadcast 1-D bias tensor `[out_dim]` into 2-D storage `[batch, out_dim]` via `grim_broadcast_bias`.
    fn broadcast_bias(
        &self,
        bias: &dyn BackendStorage,
        batch: usize,
        out_dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let b_s = as_rocm(bias)?;
        if !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "broadcast_bias: bias lacks a valid device pointer".into(),
            ));
        }
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut b_ptr = dev_ptr(b_s)?;
        let mut batch_i = batch as i32;
        let mut out_dim_i = out_dim as i32;
        let total = batch * out_dim;
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_broadcast_bias",
            grid,
            block,
            &mut [
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut batch_i),
                arg(&mut out_dim_i),
            ],
        )?;

        let _ = stream; // no post-launch sync: output consumed via stream order

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// In-place scale+bias epilogue on a `[batch, out_dim]` GEMM output via `grim_scale_bias_epilogue`.
    /// Plain rocBLAS has no epilogue-fusion API, so this standalone kernel is the required post-GEMM step.
    fn scale_bias_epilogue(
        &self,
        out: &dyn BackendStorage,
        a_scale: Option<&dyn BackendStorage>,
        b_scale: Option<&dyn BackendStorage>,
        bias: Option<&dyn BackendStorage>,
        batch: usize,
        out_dim: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let o_s = as_rocm(out)?;
        if !o_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "scale_bias_epilogue: out lacks a valid device pointer".into(),
            ));
        }
        // `_a_s` / `_b_s` / `_bt_s` hold borrows that keep the underlying storage allocations alive
        // until after the kernel launch; only the raw pointers are forwarded into the kernel args.
        let (_a_s, a_ptr): (Option<&dyn BackendStorage>, Option<*mut c_void>) = match a_scale {
            Some(s) => {
                let s = as_rocm(s)?;
                (Some(s), Some(dev_ptr(s)? as *mut c_void))
            }
            None => (None, None),
        };
        let (_b_s, b_ptr): (Option<&dyn BackendStorage>, Option<*mut c_void>) = match b_scale {
            Some(s) => {
                let s = as_rocm(s)?;
                (Some(s), Some(dev_ptr(s)? as *mut c_void))
            }
            None => (None, None),
        };
        let (_bt_s, b_ptr2): (Option<&dyn BackendStorage>, Option<*mut c_void>) = match bias {
            Some(s) => {
                let s = as_rocm(s)?;
                (Some(s), Some(dev_ptr(s)? as *mut c_void))
            }
            None => (None, None),
        };

        let mut out_ptr = dev_ptr(o_s)?;
        let mut a_p = a_ptr.unwrap_or(std::ptr::null_mut());
        let mut b_p = b_ptr.unwrap_or(std::ptr::null_mut());
        let mut bpt = b_ptr2.unwrap_or(std::ptr::null_mut());
        let mut batch_i = batch as i32;
        let mut out_dim_i = out_dim as i32;
        let total = batch * out_dim;
        let (grid, block) = linear_launch(total);

        let stream = self.launch_compute_kernel(
            "grim_scale_bias_epilogue",
            grid,
            block,
            &mut [
                arg(&mut out_ptr),
                arg(&mut a_p),
                arg(&mut b_p),
                arg(&mut bpt),
                arg(&mut batch_i),
                arg(&mut out_dim_i),
            ],
        )?;

        let _ = stream; // no post-launch sync: output consumed via stream order

        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }
}
