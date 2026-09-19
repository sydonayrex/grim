//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! MXFP4/NVFP4 tiled + split-K GEMMs and fused RMSNorm+MXFP4 GEMM.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};
use grim_tensor::{ Shape };

use crate::device::roc_device::{ RocmDevice };
use crate::memory::storage::RocmStorage;
use crate::{ HipDim3, arg, dtype_f32 };



impl RocmDevice {
    /// Launch the JIT compiled tiled MXFP4 GEMM kernel.
    /// `b_codes_ptr` / `b_exps_ptr` are raw device pointers: either standalone storages or interior pointers into a.
    pub fn launch_mxfp4_gemm_tiled(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // The kernels read activations as float4 and codes as uint4; both
        // require K to be a multiple of 32 (one MXFP4 micro-block).
        if k % 32 != 0 {
            return Err(Error::Backend(format!(
                "mxfp4_gemm_tiled: K must be a multiple of 32, got {k}"
            )));
        }
        // Skinny-M decode: a plain (n/16, m/16) grid leaves most CUs idle (e.g.
        // m=1, n=4096 -> 16 CTAs on a 28+ CU part).
        if m <= 8 && k >= 2048 {
            return self.launch_mxfp4_gemm_splitk(
                a_storage,
                b_codes_ptr,
                b_exps_ptr,
                out_storage,
                m,
                n,
                k,
            );
        }
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_tiled: a has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_tiled: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_x = n.div_ceil(16) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_mxfp4_gemm_tiled",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Split-K MXFP4 GEMM for skinny-M decode (M <= 8): slice K across CUs, reduce the partials deterministically.
    /// Kept as two launches so the result is bit-stable across runs (no float atomics).
    pub(crate) fn launch_mxfp4_gemm_splitk(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: a has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: out has no device ptr".into()))?;

        let num_splits: u32 = if k >= 8192 {
            8
        } else if k >= 4096 {
            4
        } else {
            2
        };
        let partials = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[num_splits as usize, m, n]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let partials_ptr = partials
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: partials alloc failed".into()))?;

        const SPLITK_BLOCK: usize = 64;
        let grid_dim = HipDim3::new((n.div_ceil(SPLITK_BLOCK)) as u32, m as u32, num_splits);
        let block_dim = HipDim3::new(SPLITK_BLOCK as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut pptr = partials_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut splits = num_splits as i32;

        let stream = self.launch_compute_kernel(
            "grim_mxfp4_gemm_splitk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut pptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut splits),
            ],
        )?;

        const REDUCE_BLOCK: usize = 256;
        let total = m * n;
        let reduce_grid = HipDim3::new((total.div_ceil(REDUCE_BLOCK)) as u32, 1, 1);
        let reduce_block = HipDim3::new(REDUCE_BLOCK as u32, 1, 1);

        let mut optr = out_ptr;
        let _ = self.launch_compute_kernel(
            "grim_mxfp4_splitk_reduce",
            reduce_grid,
            reduce_block,
            &mut [
                arg(&mut pptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut splits),
            ],
        )?;

        // `partials` drops to the pool here; same-stream reuse is ordered
        // after both kernels above.
        Ok(stream)
    }

    /// Fused NVFP4 GEMV with cooperative Wave reduction in LDS (for decode batch M <= 4).
    pub fn launch_nvfp4_gemv(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemv: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemv: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(256, 1, 1);
        let grid_dim = HipDim3::new(n as u32, m as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_nvfp4_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Tiled NVFP4 GEMM for prefill batch (M > 4).
    pub fn launch_nvfp4_gemm_tiled(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemm_tiled: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemm_tiled: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemm_tiled: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_nvfp4_gemm_tiled",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled backward MXFP4 GEMM kernel (dA = dY @ B^T).
    pub(crate) fn launch_mxfp4_backward_gemm(
        &self,
        dy_storage: &RocmStorage,
        b_codes_storage: &RocmStorage,
        b_exps_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_backward_gemm: dy has no device ptr".into()))?;
        let b_codes_ptr = b_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("mxfp4_backward_gemm: b_codes has no device ptr".into())
        })?;
        let b_exps_ptr = b_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("mxfp4_backward_gemm: b_exps has no device ptr".into())
        })?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_backward_gemm: dx has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_x = k.div_ceil(16) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);

        let mut dyptr = dy_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_mxfp4_backward_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the fused RMSNorm + MXFP4 GEMM kernel (e.g. for MLP projections).
    pub fn launch_fused_rmsnorm_mxfp4_gemm(
        &self,
        x_storage: &RocmStorage,
        gamma_storage: &RocmStorage,
        w_codes_storage: &RocmStorage,
        w_exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        eps: f32,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: x has no device ptr".into())
        })?;
        let gamma_ptr = gamma_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: gamma has no device ptr".into())
        })?;
        let w_codes_ptr = w_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: w_codes has no device ptr".into())
        })?;
        let w_exps_ptr = w_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: w_exps has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: out has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(64, 1, 1);
        let grid_dim = HipDim3::new(m as u32, n.div_ceil(64) as u32, 1);

        let mut xptr = x_ptr;
        let mut gammaptr = gamma_ptr;
        let mut wcodesptr = w_codes_ptr;
        let mut wexpsptr = w_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut eps_val = eps;

        self.launch_compute_kernel_with_solution(
            "grim_fused_rmsnorm_mxfp4_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut gammaptr),
                arg(&mut wcodesptr),
                arg(&mut wexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut eps_val),
            ],
            None,
            64 * std::mem::size_of::<f32>(),
        )
    }

    /// Launch the fused RMSNorm + MXFP4 GEMM + RoPE + direct KV cache scatter kernel.
    pub fn launch_fused_rmsnorm_mxfp4_gemm_rope_kv(
        &self,
        x_storage: &RocmStorage,
        gamma_storage: &RocmStorage,
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
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: x has no device ptr".into())
        })?;
        let gamma_ptr = gamma_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: gamma has no device ptr".into())
        })?;
        let w_codes_ptr = w_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: w_codes has no device ptr".into())
        })?;
        let w_exps_ptr = w_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: w_exps has no device ptr".into())
        })?;

        let q_out_ptr = q_out_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let k_cache_ptr = k_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let v_cache_ptr = v_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let out_all_ptr = out_all_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let positions_ptr = positions_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let inv_freq_ptr = inv_freq_storage.and_then(|s| s.device_ptr).unwrap_or(0);

        let n_total = (num_q_heads + 2 * num_kv_heads) * head_dim;
        let block_dim = HipDim3::new(64, 1, 1);
        let grid_dim = HipDim3::new(m as u32, n_total.div_ceil(64) as u32, 1);

        let mut xptr = x_ptr;
        let mut gammaptr = gamma_ptr;
        let mut wcodesptr = w_codes_ptr;
        let mut wexpsptr = w_exps_ptr;
        let mut qptr = q_out_ptr;
        let mut kptr = k_cache_ptr;
        let mut vptr = v_cache_ptr;
        let mut allptr = out_all_ptr;
        let mut posptr = positions_ptr;
        let mut mm = m as i32;
        let mut kk = k as i32;
        let mut nq = num_q_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut theta = rope_theta;
        let mut invfreqptr = inv_freq_ptr;
        let mut mscale_val = mscale;
        let mut eps_val = eps;
        let mut max_seq = max_seq_len as i32;

        self.launch_compute_kernel_with_solution(
            "grim_fused_rmsnorm_mxfp4_gemm_rope_kv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut gammaptr),
                arg(&mut wcodesptr),
                arg(&mut wexpsptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut allptr),
                arg(&mut posptr),
                arg(&mut mm),
                arg(&mut kk),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut theta),
                arg(&mut invfreqptr),
                arg(&mut mscale_val),
                arg(&mut eps_val),
                arg(&mut max_seq),
            ],
            None,
            64 * std::mem::size_of::<f32>(),
        )
    }
}
