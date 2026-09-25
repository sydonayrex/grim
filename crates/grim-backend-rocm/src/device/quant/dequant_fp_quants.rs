//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! FP-family (fp8/mxfp4/mxfp8/nvfp4/f16) fused dequant GEMMs + elementwise dequant GEMM.

use std::ffi::c_void;

use grim_tensor::dtype::ArithType;
use grim_tensor::error::{Error, Result};

use crate::device::gemm_tuning::lookup_solution_index;
use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{HipDim3, arg};

impl RocmDevice {
    /// Launch the JIT compiled fused dequantization GEMM kernel for [see: `b_storage`, `Storage::ResidualPacked`]
    pub(crate) fn launch_fused_dequant_gemm_f16(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        b_scales_ptr: *const c_void,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        default_bpw: u8,
        outlier_count: usize,
        outlier_indices_ptr: *const c_void,
        outlier_values_ptr: *const c_void,
        backup_bpw: u8,
        backup_codes_offset: usize,
        backup_scale_offset: usize,
        backup2_bpw: u8,
        backup2_codes_offset: usize,
        backup2_scale_offset: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "fused_dequant_gemm: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut bsptr = b_scales_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        let stride_a = k; // A[M, K]
        let stride_c = n; // C[M, N]
        let mut sa = stride_a as i32;
        let mut sc = stride_c as i32;

        let mut bpw_val = default_bpw as i32;
        let mut out_cnt = outlier_count as i32;
        let mut out_idx_ptr = outlier_indices_ptr;
        let mut out_val_ptr = outlier_values_ptr;

        let mut b_bpw = backup_bpw as i32;
        let mut b_codes_off = backup_codes_offset as i32;
        let mut b_scale_off = backup_scale_offset as i32;
        let mut b2_bpw = backup2_bpw as i32;
        let mut b2_codes_off = backup2_codes_offset as i32;
        let mut b2_scale_off = backup2_scale_offset as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_fused_dequant_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut bsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sc),
                arg(&mut bpw_val),
                arg(&mut out_cnt),
                arg(&mut out_idx_ptr),
                arg(&mut out_val_ptr),
                arg(&mut b_bpw),
                arg(&mut b_codes_off),
                arg(&mut b_scale_off),
                arg(&mut b2_bpw),
                arg(&mut b2_codes_off),
                arg(&mut b2_scale_off),
            ],
            Some(solution_index),
            0,
        )
    }

    /// Launch the Charon fused MoE dispatch kernel (`rocm_kernel_plan.md` WI-A).
    /// Single sortless launch: each block reads its (token, expert) pair from the uploaded routing arrays.
    pub(crate) fn launch_fused_dequant_backward_gemm_f16(
        &self,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        b_scales_ptr: *const c_void,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        default_bpw: u8,
        outlier_count: usize,
        outlier_indices_ptr: *const c_void,
        outlier_values_ptr: *const c_void,
        backup_bpw: u8,
        backup_codes_offset: usize,
        backup_scale_offset: usize,
        backup2_bpw: u8,
        backup2_codes_offset: usize,
        backup2_scale_offset: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: dY has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: B has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        // Grid covers M*K output elements (one thread per element of dX[M,K]).
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_backward: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "fused_dequant_backward: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut bsptr = b_scales_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        // dY is [M, N] row-major → stride_dy = N
        let mut sdy = n as i32;
        let mut sdx = k as i32;

        let mut bpw_val = default_bpw as i32;
        let mut out_cnt = outlier_count as i32;
        let mut out_idx_ptr = outlier_indices_ptr;
        let mut out_val_ptr = outlier_values_ptr;

        let mut b_bpw = backup_bpw as i32;
        let mut b_codes_off = backup_codes_offset as i32;
        let mut b_scale_off = backup_scale_offset as i32;

        let mut b2_bpw = backup2_bpw as i32;
        let mut b2_codes_off = backup2_codes_offset as i32;
        let mut b2_scale_off = backup2_scale_offset as i32;

        // STE: grad_scale = 1.0 for pure identity (straight-through estimator).
        // The quantize→dequantize step receives zero gradient - the upstream gradient flows straight through to the.
        let mut grad_scale: f32 = 1.0;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut bsptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sdy),
                arg(&mut sdx),
                arg(&mut bpw_val),
                arg(&mut out_cnt),
                arg(&mut out_idx_ptr),
                arg(&mut out_val_ptr),
                arg(&mut b_bpw),
                arg(&mut b_codes_off),
                arg(&mut b_scale_off),
                arg(&mut b2_bpw),
                arg(&mut b2_codes_off),
                arg(&mut b2_scale_off),
                arg(&mut grad_scale),
            ],
        )
    }

    /// Standalone FP8 dequant: convert FP8 E4M3 bytes to F32.
    pub(crate) fn launch_dequant_fp8(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_fp8: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_fp8: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_fp8: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_fp8",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_w)],
        )
    }

    /// Standalone MXFP4 dequant: decompress MXFP4 codes + shared exponents to F32.
    pub(crate) fn launch_dequant_mxfp4(
        &self,
        codes_storage: &RocmStorage,
        exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let codes_ptr = codes_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: codes has no device ptr".into()))?;
        let exps_ptr = exps_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: exps has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_mxfp4: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut codes = codes_ptr;
        let mut exps = exps_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_mxfp4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut codes),
                arg(&mut exps),
                arg(&mut out),
                arg(&mut n_w),
            ],
        )
    }

    /// Standalone MXFP8 dequant: decompress MXFP8 codes + shared exponents to F32.
    pub(crate) fn launch_dequant_mxfp8(
        &self,
        codes_storage: &RocmStorage,
        exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let codes_ptr = codes_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: codes has no device ptr".into()))?;
        let exps_ptr = exps_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: exps has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_mxfp8: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut codes = codes_ptr;
        let mut exps = exps_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_mxfp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut codes),
                arg(&mut exps),
                arg(&mut out),
                arg(&mut n_w),
            ],
        )
    }

    /// Standalone NVFP4 dequant: decompress NVFP4 codes + interleaved scales to F32.
    pub(crate) fn launch_dequant_nvfp4(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_nvfp4: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_nvfp4: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_nvfp4: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_nvfp4",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_w)],
        )
    }

    /// Launch the JIT compiled FP8 fused dequantization matmul kernel (Raven Tier).
    pub(crate) fn launch_fused_dequant_gemm_fp8(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: a has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_fp8: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_gemm_fp8: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_fp8",
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

    /// Launch the JIT compiled FP8 fused dequantization backward matmul kernel (Raven Tier).
    pub(crate) fn launch_fused_dequant_backward_gemm_fp8(
        &self,
        dy_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: dY has no device ptr".into())
        })?;
        let b_ptr = b_fp8_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: B has no device ptr".into())
        })?;
        let dx_ptr = dx_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: dX has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_backward_fp8: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_backward_fp8: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_fp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled MXFP4 fused dequantization matmul kernel (Jay Tier).
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_mxfp4(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: a has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_mxfp4: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_gemm_mxfp4: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_mxfp4",
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

    /// Launch the JIT compiled MXFP8 fused dequantization matmul kernel (Magpie Tier).
    pub(crate) fn launch_fused_dequant_gemm_mxfp8(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        b_exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: a has no device ptr".into())
        })?;
        let b_fp8_ptr = b_fp8_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: b_fp8 has no device ptr".into())
        })?;
        let b_exps_ptr = b_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: b_exps has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_mxfp8: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_gemm_mxfp8: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bfp8ptr = b_fp8_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_mxfp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bfp8ptr),
                arg(&mut bexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Elementwise dequant-GEMM launcher shared by the compressed-tensors
    /// W8A8 kernels (256-thread blocks over M*N outputs).
    pub(crate) fn launch_elementwise_dequant_gemm(
        &self,
        entry: &str,
        a: &RocmStorage,
        blob: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        extra_args: &mut [*mut c_void],
    ) -> Result<()> {
        let mut aptr = a
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm: a has no device ptr".into()))?
            as *mut c_void;
        let mut bptr = blob
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm: blob has no device ptr".into()))?
            as *mut c_void;
        let mut optr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm: out has no device ptr".into()))?
            as *mut c_void;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let grid_x = (m * n).div_ceil(256) as u32;
        let mut args: Vec<*mut c_void> = vec![
            arg(&mut aptr),
            arg(&mut bptr),
            arg(&mut optr),
            arg(&mut mm),
            arg(&mut nn),
            arg(&mut kk),
        ];
        args.extend_from_slice(extra_args);
        self.launch_compute_kernel(
            entry,
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut args,
        )?;
        Ok(())
    }

    /// Elementwise dequant-GEMM backward launcher: dX[M, K] = dY[M, N] @ deq(B)[N, K].
    /// Same 256-thread blocks, but grid covers M*K outputs (the dX dimension).
    pub(crate) fn launch_elementwise_dequant_gemm_backward(
        &self,
        entry: &str,
        dy: &RocmStorage,
        blob: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        extra_args: &mut [*mut c_void],
    ) -> Result<()> {
        let mut dyptr = dy
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm backward: dY has no device ptr".into()))?
            as *mut c_void;
        let mut bptr = blob
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm backward: blob has no device ptr".into()))?
            as *mut c_void;
        let mut dxptr = dx
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm backward: dX has no device ptr".into()))?
            as *mut c_void;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let grid_x = (m * k).div_ceil(256) as u32;
        let mut args: Vec<*mut c_void> = vec![
            arg(&mut dyptr),
            arg(&mut bptr),
            arg(&mut dxptr),
            arg(&mut mm),
            arg(&mut nn),
            arg(&mut kk),
        ];
        args.extend_from_slice(extra_args);
        self.launch_compute_kernel(
            entry,
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut args,
        )?;
        Ok(())
    }

    /// Launch the gfx1200 MFMA FP8 fused dequant GEMM kernel. [see: `should_use_wmma_path`, `rocm_device_props::gfx_level >= 12`]
    pub(crate) fn launch_fused_dequant_gemm_fp8_mfma(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: a has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: B_fp8 has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fp8_mfma: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("fp8_mfma: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_fp8_mfma",
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

    /// Launch the gfx1200 MFMA FP8 backward kernel.
    pub(crate) fn launch_fused_dequant_backward_gemm_fp8_mfma(
        &self,
        dy_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: dY has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: B_fp8 has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("fp8_mfma_bwd: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_fp8_mfma",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }
}
