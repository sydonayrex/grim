//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! GEMM launchers: rocBLAS/decode paths, WMMA (incl. fused dequant-GEMM), FP8 RDNA4, matmul ops.

use std::ffi::c_void;

use std::sync::atomic::Ordering;
use std::sync::OnceLock;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};
use std::sync::Arc;

use crate::device::gemm_tuning::{lookup_gemm_config_for_shape, lookup_solution_index};
use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    arg, arith_to_compute_dtype, arith_to_rocblas_dtype, check_hip, rocblas_gemm_ex,
    rocblas_gemm_strided_batched_ex, rocblas_set_stream, rocblas_sgemm, rocblas_status_success,
    select_gemm_algo, HipDim3, RocblasInt, RocblasOperation, RocmHandle, ROCBLAS_GEMM_FLAGS_NONE,
};

impl RocmDevice {
    /// WI 2.4.4-2c — dispatch `grim_decode_gemm_f16` and return the [see: `launch_compute_kernel`, `DecodeGemmConfig::enabled`]
    #[allow(dead_code)]
    pub(crate) fn launch_decode_gemm_f16(
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
            .ok_or_else(|| Error::Backend("decode_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("decode_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("decode_gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems = m * n;
        let grid_x = (total_elems.div_ceil(BLOCK_SIZE)) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        // Row-major strides in fp16 elements (not bytes).
        let stride_a = k; // A[M, K]
        let stride_b = k; // B[N, K] natural weight (SPEED-ROC-16 contract)
        let stride_c = n; // C[M, N]
        let mut sa = stride_a as i32;
        let mut sb = stride_b as i32;
        let mut sc = stride_c as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_decode_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(solution_index),
            0,
        )
    }

    /// Launch rocBLAS F16 GEMM directly on active stream (used for decode dispatch and graph capture).
    pub(crate) fn launch_rocblas_gemm_f16(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        use crate::device::rocblas::*;
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let handle = self.get_rocblas_handle()?;
        let stream = self.active_stream();
        unsafe {
            let _ = rocblas_set_stream(handle, stream);
        }
        let alpha: f32 = 1.0f32;
        let beta: f32 = 0.0f32;
        let a_ptr_void = a_storage.device_ptr_checked()? as *const c_void;
        let b_ptr_void = b_storage.device_ptr_checked()? as *const c_void;
        let out_ptr_void = out_storage.device_ptr_checked()? as *mut c_void;
        let alpha_ptr = &alpha as *const f32 as *const c_void;
        let beta_ptr = &beta as *const f32 as *const c_void;
        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        unsafe {
            // SPEED-ROC-16: C = A @ B^T, B stored (N, K). transA=Trans, transB=NoTrans;
            // m=N, n=M, k=K, A=b_ptr (lda=K), B=a_ptr (ldb=K), C=D=out (ldc/ldd=N).
            let mut status = rocblas_gemm_ex(
                handle,
                RocblasOperation::Transpose,
                RocblasOperation::None,
                n as RocblasInt,
                m as RocblasInt,
                k as RocblasInt,
                alpha_ptr,
                b_ptr_void,
                rocblas_datatype::f16_r,
                k as RocblasInt,
                a_ptr_void,
                rocblas_datatype::f16_r,
                k as RocblasInt,
                beta_ptr,
                out_ptr_void,
                rocblas_datatype::f16_r,
                n as RocblasInt,
                out_ptr_void,
                rocblas_datatype::f16_r,
                n as RocblasInt,
                rocblas_datatype::f32_r,
                select_gemm_algo(solution_index),
                solution_index as RocblasInt,
                ROCBLAS_GEMM_FLAGS_NONE,
            );
            if status != rocblas_status_success && solution_index != 0 {
                // Tuned solution_index may not exist on this arch or kernel configuration (status 11: invalid value);
                // fall back to default standard rocBLAS algorithm.
                status = rocblas_gemm_ex(
                    handle,
                    RocblasOperation::Transpose,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    rocblas_datatype::f16_r,
                    k as RocblasInt,
                    a_ptr_void,
                    rocblas_datatype::f16_r,
                    k as RocblasInt,
                    beta_ptr,
                    out_ptr_void,
                    rocblas_datatype::f16_r,
                    n as RocblasInt,
                    out_ptr_void,
                    rocblas_datatype::f16_r,
                    n as RocblasInt,
                    rocblas_datatype::f32_r,
                    rocblas_gemm_algo::standard,
                    0 as RocblasInt,
                    ROCBLAS_GEMM_FLAGS_NONE,
                );
            }
            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas_gemm_ex failed with code {status}"
                )));
            }
        }
        Ok(stream)
    }

    /// Enqueues the JIT-compiled WMMA matrix-core GEMM kernel (WI-G).
    pub(crate) fn launch_wmma_gemm(
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
            .ok_or_else(|| Error::Backend("wmma_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_gemm: out has no device ptr".into()))?;

        let is_native_wmma = matches!(
            crate::quantization::gcn_arch(&self.gpu_target),
            crate::quantization::GcnArch::RDNA3
                | crate::quantization::GcnArch::RDNA4
                | crate::quantization::GcnArch::UDNA
        );

        let (grid_dim, block_dim) = if is_native_wmma {
            // Native rocWMMA path: 16x16 tile per block, 1 wavefront (32 threads for W32).
            let grid_x = n.div_ceil(16) as u32;
            let grid_y = m.div_ceil(16) as u32;
            (HipDim3::new(grid_x, grid_y, 1), HipDim3::new(32, 1, 1))
        } else {
            // Scalar fallback path: 1D grid of 256 threads.
            const BLOCK_SIZE: usize = 256;
            let total_elems = m * n;
            let grid_x = (total_elems.div_ceil(BLOCK_SIZE)) as u32;
            (
                HipDim3::new(grid_x, 1, 1),
                HipDim3::new(BLOCK_SIZE as u32, 1, 1),
            )
        };

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let stride_a = k; // A[M, K]
        let stride_b = n; // B[K, N]
        let stride_c = n; // C[M, N]
        let mut sa = stride_a as i32;
        let mut sb = stride_b as i32;
        let mut sc = stride_c as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_wmma_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(solution_index),
            0,
        )
    }

    /// Enqueues the pre-transposed / column-major B WMMA kernel (fastest path).
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_gemm_b_transposed(
        &self,
        a_storage: &RocmStorage,
        b_col_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_gemm_b_transposed: a has no device ptr".into()))?;
        let b_ptr = b_col_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_gemm_b_transposed: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("wmma_gemm_b_transposed: out has no device ptr".into())
        })?;

        let is_rdna4 = matches!(
            crate::quantization::gcn_arch(&self.gpu_target),
            crate::quantization::GcnArch::RDNA4 | crate::quantization::GcnArch::UDNA
        );

        // Architectural policy: Multi-wave workgroup (RDNA4) pays off when M >= 16 (reusing larger
        // activation matrices in LDS). For decode shapes (M < 16), single-wave R3 avoids barrier stalls.
        if is_rdna4 && m >= 16 {
            return self.launch_wmma_gemm_b_transposed_rdna4(
                a_storage,
                b_col_storage,
                out_storage,
                m,
                n,
                k,
            );
        }

        let is_native_wmma = matches!(
            crate::quantization::gcn_arch(&self.gpu_target),
            crate::quantization::GcnArch::RDNA3
                | crate::quantization::GcnArch::RDNA4
                | crate::quantization::GcnArch::UDNA
        );

        let (grid_dim, block_dim) = if is_native_wmma {
            let grid_x = n.div_ceil(32) as u32;
            let grid_y = m.div_ceil(16) as u32;
            (HipDim3::new(grid_x, grid_y, 1), HipDim3::new(32, 1, 1))
        } else {
            const BLOCK_SIZE: usize = 256;
            let total_elems = m * n;
            let grid_x = (total_elems.div_ceil(BLOCK_SIZE)) as u32;
            (
                HipDim3::new(grid_x, 1, 1),
                HipDim3::new(BLOCK_SIZE as u32, 1, 1),
            )
        };

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let stride_a = k; // A[M, K]
        let stride_b_col = k; // B_col[N, K], leading dimension is K
        let stride_c = n; // C[M, N]
        let mut sa = stride_a as i32;
        let mut sb = stride_b_col as i32;
        let mut sc = stride_c as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_wmma_gemm_b_transposed",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(solution_index),
            0,
        )
    }

    /// Enqueues the RDNA4-specific multi-wave (16x64 tile per block) WMMA kernel.
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_gemm_b_transposed_rdna4(
        &self,
        a_storage: &RocmStorage,
        b_col_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("wmma_gemm_b_transposed_rdna4: a has no device ptr".into())
        })?;
        let b_ptr = b_col_storage.device_ptr.ok_or_else(|| {
            Error::Backend("wmma_gemm_b_transposed_rdna4: b has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("wmma_gemm_b_transposed_rdna4: out has no device ptr".into())
        })?;

        // 4 tiles of 16 along N = 64 elements of N per workgroup (2 waves: 64 threads).
        let grid_x = n.div_ceil(64) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(64, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let stride_a = k;
        let stride_b_col = k;
        let stride_c = n;
        let mut sa = stride_a as i32;
        let mut sb = stride_b_col as i32;
        let mut sc = stride_c as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_wmma_gemm_b_transposed_rdna4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sb),
                arg(&mut sc),
            ],
            Some(solution_index),
            0,
        )
    }

    /// SPEED-ROC: WMMA fused-dequant Q8_0 GEMM launcher (RDNA3/4).
    /// Computes C[M,N] = A[M,K] @ B^T where B is Q8_0 packed.
    /// Uses rocWMMA 16×16×16 tensor-core tiles with inline Q8_0 dequantization.
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_q8_0(
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
            .ok_or_else(|| Error::Backend("wmma_q80_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_q80_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_q80_gemm: out has no device ptr".into()))?;

        // 16×16 output tile per block, 1 wavefront (32 threads, wave32).
        let grid_x = n.div_ceil(64) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(128, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_wmma_fused_dequant_q8_0",
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

    /// SPEED-ROC: FP16-input Q8_0 WMMA launcher — reads activations as
    /// `_Float16` directly (no per-element cast), halving A-read bandwidth.
    /// Kernel: `grim_wmma_fused_dequant_q8_0_fp16`.
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_q8_0_fp16(
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
            .ok_or_else(|| Error::Backend("wmma_q80_gemm_fp16: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_q80_gemm_fp16: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_q80_gemm_fp16: out has no device ptr".into()))?;
        let grid_x = n.div_ceil(64) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(128, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_wmma_fused_dequant_q8_0_fp16",
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

    /// SPEED-ROC: WMMA fused-dequant Q4_K GEMM launcher (RDNA3/4).
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_q4k(
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
            .ok_or_else(|| Error::Backend("wmma_q4k_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_q4k_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_q4k_gemm: out has no device ptr".into()))?;

        let grid_x = n.div_ceil(64) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(128, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_wmma_fused_dequant_q4k",
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

    /// SPEED-ROC: WMMA fused-dequant Q5_K GEMM launcher (RDNA3/4).
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_q5k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_q5k", a, b, out, m, n, k)
    }

    /// SPEED-ROC: WMMA fused-dequant Q2_K GEMM launcher (RDNA3/4).
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_q2k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_q2k", a, b, out, m, n, k)
    }

    /// SPEED-ROC: WMMA fused-dequant Q3_K GEMM launcher (RDNA3/4).
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_q3k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_q3k", a, b, out, m, n, k)
    }

    /// SPEED-ROC: WMMA fused-dequant Q6_K GEMM launcher (RDNA3/4).
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_q6k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_q6k", a, b, out, m, n, k)
    }

    /// SPEED-ROC: WMMA fused-dequant IQ-family GEMM launchers (RDNA3/4).
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_iq2xxs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_iq2xxs", a, b, out, m, n, k)
    }

    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_iq2xs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_iq2xs", a, b, out, m, n, k)
    }

    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_iq2s(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_iq2s", a, b, out, m, n, k)
    }

    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_iq3xxs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_iq3xxs", a, b, out, m, n, k)
    }

    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_iq3s(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_iq3s", a, b, out, m, n, k)
    }

    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_iq4nl(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_iq4nl", a, b, out, m, n, k)
    }

    #[allow(dead_code)]
    pub(crate) fn launch_wmma_fused_dequant_iq4xs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_fused_dequant_iq4xs", a, b, out, m, n, k)
    }

    /// SPEED-ROC: FP8 E4M3 WMMA GEMM launcher (RDNA4 only, 383 TFLOPS).
    ///
    /// Deliberately NOT routed through `launch_wmma_fused_dequant_quant`. That
    /// shared body launches the fused-dequant kernels, which use a 64-wide N
    /// tile; `grim_wmma_gemm_fp8_e4m3` packs **two 16-wide** N tiles per block
    /// (`tile_col_base = blockIdx.x * 2`, `col_base = tile_col_base * 16`, i.e.
    /// `blockIdx.x * 32`). Launching it with `ceil(N/64)` covers only half the
    /// output columns and silently leaves the rest unwritten.
    ///
    /// Block is 32, not 128: `mma_sync` and `store_matrix_sync` are per-wave
    /// in rocwmma, so extra waves recompute the same tile and race on the same
    /// shared `c_out`. The `256` in the kernel's store loop is the size of the
    /// 16x16 output tile, not the block width.
    /// GreyRaven (WS-E E6) layout probe: one 16x16x32 sparse FP8 SWMMAC with
    /// every operand forwarded verbatim from device buffers.
    ///
    /// Deliberately does no fragment packing. The A/B lane-to-element mapping
    /// and the sparsity-index correspondence are exactly what E6 has to
    /// discover, so any layout helper applied here would encode a guess into the
    /// kernel -- and a wrong guess still returns finite, plausible numbers. With
    /// the packing left to the host, a wrong hypothesis is a test failure rather
    /// than a plausible-looking kernel.
    ///
    /// Compiles as a **standalone module** rather than through the aggregate
    /// JIT source. The probe shares no code with the rest of the tree, so
    /// making it depend on a ~30k-line aggregate means a compile error in any
    /// unrelated kernel blocks it -- which is exactly what happened while this
    /// was being written. A module holding only the probe is also easier to
    /// reason about when the result is surprising: nothing else is in the binary
    /// that could be responsible.
    pub fn launch_grey_raven_probe(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        sidx: u32,
        c_out: &RocmStorage,
    ) -> Result<*mut c_void> {
        let ap = a
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_probe: a has no device ptr".into()))?;
        let bp = b
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_probe: b has no device ptr".into()))?;
        let cp = c_out
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_probe: c has no device ptr".into()))?;
        let mut aptr = ap;
        let mut bptr = bp;
        let mut cptr = cp;
        let mut s = sidx;
        self.launch_from_source(
            crate::kernels::grey_raven::PROBE_SOURCE,
            "grim_grey_raven_probe",
            HipDim3::new(1, 1, 1),
            HipDim3::new(32, 1, 1),
            &mut [arg(&mut aptr), arg(&mut bptr), arg(&mut s), arg(&mut cptr)],
        )
    }

    /// GreyRaven B-prologue: quantize X into fragment-ordered FP8 once per
    /// (M-tile, K-window), so the main kernel loads plain 16 B per lane with
    /// no gather, no encode, and no bounds checks in its hot loop.
    ///
    /// Without this, every N-tile re-gathers and re-encodes the same B
    /// window (256x redundant on square shapes) -- measured as the entire
    /// performance gap to dense. With it, the prologue runs once per
    /// (M-tile, window) into cached scratch, stream-ordered before the GEMM
    /// on the same stream (the WhiteRaven act-prologue discipline).
    pub fn launch_grey_raven_b_prologue(
        &self,
        x: &RocmStorage,
        b_frag: &RocmStorage,
        m: usize,
        k: usize,
        m_tiles: usize,
        n_windows: usize,
    ) -> Result<*mut c_void> {
        let want_bytes = m_tiles
            .checked_mul(n_windows)
            .and_then(|t| t.checked_mul(512))
            .ok_or_else(|| Error::Backend("grey_b_prologue: frag geometry overflows".into()))?;
        // Scratch comes from a grow-only cache: it may be LARGER than asked
        // (reuse), never smaller. Validate the floor; the kernel addresses
        // the first want_bytes and ignores the rest.
        if b_frag.bytes < want_bytes {
            return Err(Error::Backend(format!(
                "grey_b_prologue: scratch is {} bytes, need at least {} ({} M-tiles x {} windows x 512)",
                b_frag.bytes, want_bytes, m_tiles, n_windows
            )));
        }
        let xptr = x
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_b_prologue: x has no device ptr".into()))?;
        let bptr = b_frag
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_b_prologue: out has no device ptr".into()))?;
        let total = want_bytes as u64;
        let grid_dim = HipDim3::new(total.div_ceil(256) as u32, 1, 1);
        let block_dim = HipDim3::new(256, 1, 1);
        let mut xx = xptr;
        let mut bb = bptr;
        let mut mm = m as i32;
        let mut kk = k as i32;
        let mut mt = m_tiles as i32;
        let mut nw = n_windows as i32;
        self.launch_compute_kernel(
            "grim_grey_raven_b_prologue",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xx),
                arg(&mut bb),
                arg(&mut mm),
                arg(&mut kk),
                arg(&mut mt),
                arg(&mut nw),
            ],
        )
    }

    /// GreyRaven production GEMM over HW-order tiled weights.
    ///
    /// Unlike the probe above this runs from the JIT aggregate (it needs
    /// `float_to_fp8_e4m3_hip` from quant_standalone for the in-register B
    /// gather), so the entry is `grim_grey_raven_gemm` via
    /// `launch_compute_kernel`, which also records the route counter the
    /// journey tests assert on.
    ///
    /// Contract (refused, not clamped): `blob.bytes == n_tiles*n_windows*260`
    /// with `n_tiles = ceil(n/16)`, `n_windows = ceil(k/32)`. A short blob
    /// would fault the frag loads; an over-long one means the caller paired
    /// the wrong geometry with the payload. Both name exact sizes.
    pub fn launch_grey_raven_gemm(
        &self,
        w_blob: &RocmStorage,
        b_frag: &RocmStorage,
        y_out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        n_tiles: usize,
        n_windows: usize,
    ) -> Result<*mut c_void> {
        let want_bytes = n_tiles
            .checked_mul(n_windows)
            .and_then(|t| t.checked_mul(260))
            .ok_or_else(|| Error::Backend("grey_raven_gemm: tile geometry overflows".into()))?;
        if w_blob.bytes != want_bytes {
            return Err(Error::Backend(format!(
                "grey_raven_gemm: HW blob is {} bytes, need {} ({} N-tiles x {} K-windows x 260) -- truncated upload or wrong tensor",
                w_blob.bytes, want_bytes, n_tiles, n_windows
            )));
        }
        let wptr = w_blob
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_gemm: w has no device ptr".into()))?;
        let bptr = b_frag
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_gemm: b_frag has no device ptr (run the prologue first)".into()))?;
        let yptr = y_out
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_gemm: y has no device ptr".into()))?;
        let grid_dim = HipDim3::new(n_tiles as u32, m.div_ceil(16) as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let m_tiles = m.div_ceil(16);
        let mut w = wptr;
        let mut bb = bptr;
        let mut yy = yptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut nt = n_tiles as i32;
        let mut nw = n_windows as i32;
        let mut mt = m_tiles as i32;
        self.launch_compute_kernel(
            "grim_grey_raven_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut w),
                arg(&mut bb),
                arg(&mut yy),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut nt),
                arg(&mut nw),
                arg(&mut mt),
            ],
        )
    }

    pub fn launch_wmma_gemm_fp8_e4m3_for_ab(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        const TILE_M: usize = 16;
        const TILE_N: usize = 16;
        /// N tiles handled per block; must match `blockIdx.x * 2` in the kernel.
        const N_TILES_PER_BLOCK: usize = 2;
        let n_per_block = TILE_N * N_TILES_PER_BLOCK;

        let a_ptr = a
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_fp8_e4m3: a has no device ptr".into()))?;
        let b_ptr = b
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_fp8_e4m3: b has no device ptr".into()))?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_fp8_e4m3: out has no device ptr".into()))?;

        // rocwmma walks K in 16-element steps with no tail handling.
        if k % 16 != 0 {
            return Err(Error::Backend(format!(
                "wmma_fp8_e4m3: K={k} must be divisible by 16"
            )));
        }

        // A and B must be padded up to whole 16-row tiles. The kernel's
        // `load_matrix_sync` calls fetch a full 16x16 fragment and are not masked
        // against M/N -- only the epilogue store is. So an operand with
        // m % 16 != 0 (or n % 16 != 0) makes the kernel read up to 15*k bytes
        // past the end of the allocation. Those bytes are discarded by the
        // masked store, so the result is still correct, but the read itself is
        // out of bounds: it silently returns neighbouring memory when the
        // allocation abuts other live buffers, and faults with "Page not
        // present" when it abuts an unmapped page. Reject the unpadded case here
        // rather than leaving a landmine that depends on heap layout.
        let tile = 16usize;
        if a.bytes() % (tile * k) != 0 || a.bytes() == 0 {
            return Err(Error::Backend(format!(
                "wmma_fp8_e4m3: A must be padded to whole {tile}-row tiles (got {} bytes, k={k})",
                a.bytes()
            )));
        }
        if b.bytes() % (tile * k) != 0 || b.bytes() == 0 {
            return Err(Error::Backend(format!(
                "wmma_fp8_e4m3: B must be padded to whole {tile}-row tiles (got {} bytes, k={k})",
                b.bytes()
            )));
        }

        let grid_dim = HipDim3::new(
            (n as u32).div_ceil(n_per_block as u32),
            (m as u32).div_ceil(TILE_M as u32),
            1,
        );
        let block_dim = HipDim3::new(32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_wmma_gemm_fp8_e4m3",
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

    /// Cached per-device activation scratch for the WhiteRaven blocked leg.
    ///
    /// Sized on first use and reused thereafter, so the eager dispatch pays no
    /// allocation per call -- measured at ~200us/call when it allocated a fresh
    /// buffer and did a D2H readback, which was more than the kernel saved.
    /// Keyed by byte size and grown (never shrunk) when a wider activation
    /// shows up, so a decode-then-prefill sequence settles to one allocation.
    ///
    /// Reuse is stream-ordered: the prologue and the GEMM that reads the result
    /// are enqueued back to back on this device's stream, so no second kernel
    /// can observe a half-written scratch.
    pub fn white_raven_act_scratch(&self, bytes: usize) -> Result<Arc<RocmStorage>> {
        static SCRATCH: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashMap<usize, Arc<RocmStorage>>>,
        > = std::sync::OnceLock::new();
        let cache = SCRATCH.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = guard.get(&bytes) {
            return Ok(s.clone());
        }
        // Grow-only: reuse the largest entry that already fits rather than
        // allocating a second buffer for a smaller shape.
        if let Some((&sz, s)) = guard
            .iter()
            .filter(|(_, s)| s.bytes() >= bytes)
            .max_by_key(|(_, s)| s.bytes())
        {
            let _ = sz;
            return Ok(s.clone());
        }
        let st = RocmStorage::alloc_gpu_with_bytes(
            &Shape::new(vec![bytes]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            bytes,
            &self.allocator,
            self.ordinal,
        )?;
        let st = Arc::new(st);
        guard.insert(bytes, st.clone());
        Ok(st)
    }

    /// Stage already-fp8 activation codes into the padded scratch: copy the
    /// `m` real rows and zero the tail rows the fragment load will still read.
    /// Stage already-fp8 activation codes into the padded scratch.
    ///
    /// A thin alias for [`Self::launch_quant_fp8_pad16`] with the dtype decided
    /// by the caller: the kernel zeroes the pad rows in the same launch, which
    /// is not optional -- a stale tail row is accumulated into the output as a
    /// phantom activation, and at decode (m=1) that is 15 rows of whatever the
    /// previous call left behind.
    pub fn launch_fp8_pad_rows_into(
        &self,
        act_u8: &RocmStorage,
        scratch: &RocmStorage,
        m: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_quant_fp8_pad16(act_u8, scratch, m, k)
    }

    /// Act prologue for [`Self::launch_wmma_gemm_fp8_e4m3_blocked`]: quantize an
    /// F32 activation `[m, k]` to FP8 E4M3 codes into `act_fp8_pad`
    /// (`16*ceil(m/16) * k` bytes, pad rows +0.0).
    ///
    /// Exists so graph capture has a capture-safe path: the eager dispatch's
    /// host conversion does a D2H readback (sync) and an allocation, both of
    /// which poison or stall a captured graph. This writes into caller-owned
    /// scratch, so replay re-quantizes fresh activations on device.
    pub fn launch_quant_fp8_pad16(
        &self,
        act: &RocmStorage,
        act_fp8_pad: &RocmStorage,
        m: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = act
            .device_ptr
            .ok_or_else(|| Error::Backend("quant_fp8_pad16: act has no device ptr".into()))?;
        let o_ptr = act_fp8_pad
            .device_ptr
            .ok_or_else(|| Error::Backend("quant_fp8_pad16: out has no device ptr".into()))?;

        // An already-fp8 act skips the conversion but still needs its pad rows,
        // which is why this is a flag on one entry point rather than two
        // kernels: the pad rows are zeroed by the same launch either way.
        let src_is_fp8 = match act.dtype().arith {
            ArithType::F32 => 0,
            ArithType::U8 => 1,
            other => {
                return Err(Error::Backend(format!(
                    "quant_fp8_pad16: act must be f32 or u8 codes, got {other:?}"
                )))
            }
        };
        if src_is_fp8 == 1 && act.bytes() < m * k {
            return Err(Error::Backend(format!(
                "quant_fp8_pad16: u8 act has {} bytes, need {m}*{k}",
                act.bytes()
            )));
        }

        let a_rows = m.div_ceil(16) * 16;
        if act_fp8_pad.bytes < a_rows * k {
            return Err(Error::Backend(format!(
                "quant_fp8_pad16: scratch {}B < {needed}B (16*ceil({m}/16)*{k})",
                act_fp8_pad.bytes,
                needed = a_rows * k
            )));
        }

        // Grid covers the PADDED element count, not m*k. The kernel zeroes the
        // tail rows itself; sizing the grid to m*k would leave 15 of 16 rows
        // holding whatever the previous call wrote.
        let (grid, block) = crate::device::util::linear_launch(a_rows * k);
        let mut x_ptr = x_ptr;
        let mut o_ptr = o_ptr;
        let mut mm = m as i32;
        let mut kk = k as i32;
        let mut is_fp8 = src_is_fp8;
        self.launch_compute_kernel(
            "grim_quant_fp8_pad16",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut o_ptr),
                arg(&mut mm),
                arg(&mut kk),
                arg(&mut is_fp8),
            ],
        )
    }

    /// Capture-safe WhiteRaven blocked decode leg: `launch_quant_fp8_pad16`
    /// into caller-owned scratch, then the blocked WMMA GEMM. Both launches
    /// read pre-allocated pointers, so this is safe inside `hipStreamBeginCapture`.
    pub fn linear_decode_blocked_into(
        &self,
        a: &RocmStorage,
        w: &RocmStorage,
        out: &RocmStorage,
        act_fp8_pad: &RocmStorage,
    ) -> Result<*mut c_void> {
        let k = a.shape().dims().last().copied().unwrap_or(0);
        let m = a.shape().elem_count().checked_div(k.max(1)).unwrap_or(0);
        if m == 0 || k == 0 {
            return Err(Error::Unimplemented(
                "linear_decode_blocked_into: empty activation".into(),
            ));
        }
        let n = if m == 1 {
            out.shape().elem_count()
        } else {
            out.shape().elem_count() / m
        };
        if w.shape().elem_count() % k.max(1) != 0 || w.shape().elem_count() / k.max(1) != n {
            return Err(Error::ShapeMismatch {
                expected: vec![1, n],
                got: w.shape().dims().to_vec(),
            });
        }
        if k % 16 != 0 || n % 16 != 0 {
            return Err(Error::Backend(format!(
                "WhiteRaven blocked requires k % 16 == 0 and n % 16 == 0; got k={k} n={n}"
            )));
        }
        if a.dtype().arith != ArithType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "linear_decode_blocked_into: act must be F32 (the prologue quantizes it); got {:?}",
                a.dtype().arith
            )));
        }
        self.launch_quant_fp8_pad16(a, act_fp8_pad, m, k)?;
        self.launch_wmma_gemm_fp8_e4m3_blocked(act_fp8_pad, w, out, m, n, k)
    }

    /// Blocked-B twin of [`Self::launch_wmma_gemm_fp8_e4m3_for_ab`]: B arrives
    /// in 16x16-blocked order (`grim_quant::block_fp8_16x16`,
    /// [`grim_quant::FP8_BLOCK16_ENCODER_VERSION`]) so the kernel's B fragment
    /// loads are contiguous 256B tiles instead of K-strided gathers. Same grid,
    /// same A contract (padded to whole 16-row tiles), same masked epilogue.
    /// `b` must hold exactly `n * k` blocked bytes with `n % 16 == 0`.
    pub fn launch_wmma_gemm_fp8_e4m3_blocked(
        &self,
        a: &RocmStorage,
        b_blocked: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        const TILE_M: usize = 16;
        const N_TILES_PER_BLOCK: usize = 2;
        let n_per_block = TILE_M * N_TILES_PER_BLOCK;

        let a_ptr = a
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_fp8_e4m3_blocked: a has no device ptr".into()))?;
        let b_ptr = b_blocked
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_fp8_e4m3_blocked: b has no device ptr".into()))?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("wmma_fp8_e4m3_blocked: out has no device ptr".into()))?;

        if k % 16 != 0 {
            return Err(Error::Backend(format!(
                "wmma_fp8_e4m3_blocked: K={k} must be divisible by 16"
            )));
        }
        if n % 16 != 0 {
            return Err(Error::Backend(format!(
                "wmma_fp8_e4m3_blocked: N={n} must be divisible by 16 for 16x16 blocks"
            )));
        }
        // Same A-pad contract as the row-major entry: unmasked fragment loads
        // read whole 16-row tiles.
        let tile = 16usize;
        if a.bytes() % (tile * k) != 0 || a.bytes() == 0 {
            return Err(Error::Backend(format!(
                "wmma_fp8_e4m3_blocked: A must be padded to whole {tile}-row tiles (got {} bytes, k={k})",
                a.bytes()
            )));
        }
        // Blocked B is a permutation, not a compression: exactly n*k bytes.
        if b_blocked.bytes() != n * k {
            return Err(Error::Backend(format!(
                "wmma_fp8_e4m3_blocked: blocked B must hold exactly n*k={} bytes, got {}",
                n * k,
                b_blocked.bytes()
            )));
        }

        let grid_dim = HipDim3::new(
            (n as u32).div_ceil(n_per_block as u32),
            (m as u32).div_ceil(TILE_M as u32),
            1,
        );
        let block_dim = HipDim3::new(32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_wmma_gemm_fp8_e4m3_blocked",
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

    /// Shared launcher body for all WMMA fused-dequant quant GEMM kernels.
    fn launch_wmma_fused_dequant_quant(
        &self,
        name: &str,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{name}: a has no device ptr")))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{name}: b has no device ptr")))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{name}: out has no device ptr")))?;

        let grid_x = n.div_ceil(64) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(128, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            name,
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

    /// Launch the standalone FP8 GEMM kernel (gfx1200+ native MFMA,
    #[allow(dead_code)]
    pub(crate) fn launch_fp8_gemm_rdna4(
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
            .ok_or_else(|| Error::Backend("fp8_gemm_rdna4: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_gemm_rdna4: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_gemm_rdna4: out has no device ptr".into()))?;

        // 16×16 tiling: one thread per output element, tile = 16 threads
        const TILE: usize = 16;
        let grid_x = (n.div_ceil(TILE)) as u32;
        let grid_y = (m.div_ceil(TILE)) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(TILE as u32, TILE as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fp8_gemm_rdna4",
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

    /// Op-tagged GEMM. `op` drives the `ShapeClass` via `ShapeClass::from_op`: `LmHead` selects the TLOLog tile arm (wide block_n
    /// for the vocab-dominated output column); everything else bins by M as before (from_op(Other, m) == from_m(m)).
    pub(crate) fn matmul_op(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
        op: crate::autotune::GemmOp,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // For matmul on GPU, both inputs must be RocmStorage (or we need to copy them to the device first)
        let a_storage = match a.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return Err(Error::Backend("matmul: input a is not RocmStorage".into())),
        };
        // Allocate output GPU storage with the actual input precision, then run
        // the shared _into core (zero duplicated dispatch logic).
        let dtype_out = DType {
            arith: a_storage.dtype.arith,
            storage: DTypeStorage::Native,
        };
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_out, &self.allocator, self.ordinal)?;
        let handle = self.matmul_op_into(a, b, &out_storage, op)?;
        Ok((Box::new(out_storage), handle))
    }

    /// Launch the opt-in native-F32 BLASLt prefill candidate.
    ///
    /// Model tensors use `A:[M,K]` and `B:[N,K]` row-major storage and the
    /// dispatcher contract is `C = A @ B^T`. The physical layout is handled
    /// without a host round trip: transposing `A` into a device-owned `[K,M]`
    /// row-major buffer makes its bytes canonical column-major `[M,K]`, while
    /// the existing `[N,K]` row-major bytes are already canonical column-major
    /// `[K,N]` for `B`. The result is staged in a device-owned canonical
    /// buffer and converted back into the caller's row-major output.
    fn launch_blaslt_prefill_into(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let stream = self.active_stream();
        if stream.is_null() {
            return Err(Error::Backend(
                "BLASLt prefill requires a non-null active stream".into(),
            ));
        }
        let mut scratch_guard = self
            .blaslt_scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(scratch) = scratch_guard.as_ref() {
            if scratch.in_flight && scratch.stream != stream {
                check_hip("hipStreamWaitEvent(BLASLt scratch)", unsafe {
                    crate::hipStreamWaitEvent(stream, scratch.completion, 0)
                })?;
            }
        }
        let needs_new = scratch_guard
            .as_ref()
            .map(|scratch| !scratch.fits(m, n, k))
            .unwrap_or(true);
        if needs_new {
            *scratch_guard = Some(crate::device::roc_device::BlasLtScratch::new(
                self, m, n, k,
            )?);
        }

        let launch_result = (|| {
            let scratch = scratch_guard
                .as_ref()
                .ok_or_else(|| Error::Backend("BLASLt scratch disappeared".into()))?;
            self.transpose_f32_2d_into(a_storage, &scratch.a_col, m, k)?;
            let a_ptr = scratch.a_col.device_ptr_checked()? as *const c_void;
            let b_ptr = b_storage.device_ptr_checked()? as *const c_void;
            let d_ptr = scratch.d_col.device_ptr_checked()? as *mut c_void;
            crate::device::blaslt::matmul_col_major_f32(stream, a_ptr, b_ptr, d_ptr, m, n, k)
                .map_err(Error::Backend)?;
            crate::device::blaslt::launch_col_major_to_row_major(
                self,
                &scratch.d_col,
                out_storage,
                m,
                n,
                m,
            )
            .map_err(Error::Backend)?;
            Ok::<(), Error>(())
        })();

        if let Err(error) = launch_result {
            // A failed BLASLt call may still have queued work. Drain before
            // allowing the persistent scratch to be reused or dropped.
            self.synchronize();
            if let Some(scratch) = scratch_guard.as_mut() {
                scratch.in_flight = false;
            }
            return Err(error);
        }

        let scratch = scratch_guard
            .as_mut()
            .ok_or_else(|| Error::Backend("BLASLt scratch disappeared after launch".into()))?;
        if let Err(error) = check_hip("hipEventRecord(BLASLt scratch)", unsafe {
            crate::hipEventRecord(scratch.completion, stream)
        }) {
            self.synchronize();
            scratch.in_flight = false;
            return Err(error);
        }
        scratch.in_flight = true;
        scratch.stream = stream;
        drop(scratch_guard);
        // The two staging/conversion kernels increment the generic counter;
        // account for the external BLASLt GEMM itself as one GEMM launch.
        self.launch_counter.fetch_add(1, Ordering::Relaxed);
        // BLASLt is a vendor library, so there is no HIP entry-point name to
        // record. It gets a stable synthetic route instead, because *where a
        // dispatch went* is the question the counter exists to answer: a T2
        // journey test asserting `route(grim_wmma_gemm_fp8_e4m3) > 0` can also
        // assert `route(BLASLT_ROUTE) == 0` and thereby prove the kernel under
        // test ran rather than silently falling back to cuBLASLt.
        crate::device::compute::kernel_infra::record_kernel_route(BLASLT_ROUTE);
        Ok(Box::new(RocmHandle::new(Some(stream))))
    }

    /// `C = A @ B^T` writing into CALLER-PROVIDED `out` — no allocation inside.
    /// Required for HIP graph capture (stable pointers across replays); same
    /// dispatch as [`Self::matmul_op`] (scythe route, split-K, dot paths,
    /// WMMA, rocBLAS). `out` must hold `m*n` F32 elems for `a:[M,K]`,
    /// `b:[N,K]`.
    pub fn matmul_op_into(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &RocmStorage,
        op: crate::autotune::GemmOp,
    ) -> Result<Box<dyn ComputeHandle>> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: matmul");

        // For matmul on GPU, both inputs must be RocmStorage (or we need to copy them to the device first)
        let a_storage = match a.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return Err(Error::Backend("matmul: input a is not RocmStorage".into())),
        };

        let b_storage = match b.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return Err(Error::Backend("matmul: input b is not RocmStorage".into())),
        };

        if !a_storage.device_ptr_is_valid() || !b_storage.device_ptr_is_valid() {
            return Err(Error::Backend(
                "matmul: inputs must have valid GPU device pointers".into(),
            ));
        }

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(Error::Shape("matmul expects inputs with rank >= 2".into()));
        }

        // SPEED-ROC-16: a is [M, K], b is the natural weight [N, K]; matmul computes
        // C = A @ B^T, so the shared dim is the trailing K of both operands.
        let k = a_dims[a_dims.len() - 1];
        let m = a.shape().elem_count() / k;

        let k2 = b_dims[b_dims.len() - 1];
        let n = b.shape().elem_count() / k2;

        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }

        if out.shape().elem_count() != m * n {
            return Err(Error::Shape(format!(
                "matmul_op_into: out holds {} elems, need {}x{}={}",
                out.shape().elem_count(),
                m,
                n,
                m * n
            )));
        }

        // P1-3 context discipline: rocBLAS executes against the CALLING THREAD's current HIP device, not the handle's construction device.
        // `try_new` is context-neutral (restores the caller's device on return), so on a multi-GPU box the.
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        // Output precision follows the actual input precision.
        let dtype_out = DType {
            arith: a_storage.dtype.arith,
            storage: DTypeStorage::Native,
        };
        if !out.device_ptr_is_valid() {
            return Err(Error::Backend(
                "matmul_op_into: out lacks a valid GPU device pointer".into(),
            ));
        }
        let out_storage: &RocmStorage = out;

        // Opt-in native-F32 prefill experiment. The gate is deliberately
        // outside the default path: Q8/Q4 model weights and all decode shapes
        // continue through their measured dot4/rocBLAS dispatchers.
        let native_f32 = |storage: &RocmStorage| {
            storage.dtype.arith == ArithType::F32
                && matches!(storage.dtype.storage, DTypeStorage::Native)
                && storage.device_ptr_is_valid()
        };
        if crate::device::blaslt::blaslt_prefill_enabled()
            && self.active_capture_stream().is_none()
            && native_f32(a_storage)
            && native_f32(b_storage)
            && native_f32(out_storage)
            && dtype_out.arith == ArithType::F32
        {
            static PROBE: OnceLock<crate::device::blaslt::BlasLtProbe> = OnceLock::new();
            let probe = PROBE.get_or_init(crate::device::blaslt::probe_blaslt);
            if matches!(
                crate::device::blaslt::select_blaslt_candidate(&probe, m, n, k, true),
                crate::device::blaslt::BlasLtSelection::Eligible
            ) {
                return self.launch_blaslt_prefill_into(a_storage, b_storage, out_storage, m, n, k);
            }
            eprintln!(
                "[blaslt-prefill] candidate not eligible for {m}x{n}x{k}: {}",
                probe.describe()
            );
        }

        // WI-SB6 production routing: GRIM_SCYTHE_RING=1 rides F32 GEMMs (the dense-layer op of every decode step) through the ScytheRing persistent dispatch wave instead of the rocBLAS direct path.
        // Benchmark-gated, never default - see device::scythe_route.
        if dtype_out.arith == ArithType::F32 && crate::device::scythe_route::ring_routing_enabled()
        {
            let stream = crate::device::scythe_route::route_gemm(
                self,
                self.active_stream(),
                a_storage,
                b_storage,
                out_storage,
                m,
                n,
                k,
            )?;
            self.launch_counter.fetch_add(1, Ordering::Relaxed);
            // BLASLt has no HIP entry name; recorded synthetically so a
            // fallback to it is detectable rather than invisible. See the
            // sibling site and BLASLT_ROUTE.
            crate::device::compute::kernel_infra::record_kernel_route(BLASLT_ROUTE);
            let compute_handle = Box::new(RocmHandle::new(Some(stream)));
            return Ok(compute_handle);
        }

        // Shape-indexed GEMM dispatch lookup (Tensile-inspired layout resolution).
        // Op-identity classifier: LmHead -> TLOLog tile arm; everything else bins by m.
        let shape_class = crate::autotune::ShapeClass::from_op(op, m);
        // SPEED-ROC-3: prefer the offline-tuned split_k when the exact
        // (entry, arch, M, N, K) was tuned via examples/tune_gemm.rs; the
        // static heuristic table remains the fallback.
        let entry: &'static str = match shape_class {
            crate::autotune::ShapeClass::TLOLog => "grim_lm_head",
            crate::autotune::ShapeClass::Prefill => "grim_prefill_gemm",
            crate::autotune::ShapeClass::Decode => "grim_decode_gemm",
        };
        let tuned_split_k = self.lookup_tuned_gemm_split_k(entry, m, n, k);
        let tile_config = match tuned_split_k {
            Some(split_k) => {
                let mut cfg =
                    lookup_gemm_config_for_shape(m, n, k, self.props.wavefront_size, shape_class);
                cfg.split_k = split_k;
                cfg
            }
            None => lookup_gemm_config_for_shape(m, n, k, self.props.wavefront_size, shape_class),
        };
        // Offline-tuned solution_index per (M,N,K) for FP32. Falls back to 0 for [see: `examples/tune_gemm.rs`]
        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, dtype_out.arith);
        // WI 2.4.3 — split_k clamp gate.
        let split_k_effective: u32 = {
            let split_k_enabled = self
                .split_k_config
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .enabled;
            if split_k_enabled
                && tile_config.split_k > 1
                && (k % tile_config.split_k as usize == 0)
                && (m > 1 || k > 8192)
            {
                tile_config.split_k
            } else {
                1
            }
        };

        // ─── WI 2.4.4-2 — decode GEMM dispatch (opt-in, F16-only, m ≤ 8) ───── [see: `ck_gemm.cpp`, `grim_decode_gemm_f16`]
        {
            // Lock-free read via AtomicBool shadow — avoids Mutex acquisition on every matmul.
            // [see: `decode_gemm_enabled`, `set_decode_gemm_enabled`]
            if self.decode_gemm_enabled.load(Ordering::Relaxed)
                && dtype_out.arith == ArithType::F16
                && m <= 8
            {
                // SPEED-ROC-4: opt-in (GRIM_CAPTURE_GRAPH) capture/replay of the
                // decode GEMM. Replay is pointer-bound via DecodeGraphKey, so a
                // recycled buffer re-captures instead of replaying stale memory;
                // any capture failure falls back to the direct launch.
                if self.graph_capture_enabled() {
                    let key = crate::graph_capture::DecodeGraphKey {
                        batch: m as u32,
                        seq_len: 1,
                        kv_seq_len: 1,
                        head_dim: k as u32,
                        num_heads: 1,
                        num_kv_heads: 1,
                        fused_dequant: false,
                        a_ptr: 0,
                        b_ptr: 0,
                        out_ptr: 0,
                    };
                    match self.decode_graph_capture_and_replay(
                        key,
                        a_storage,
                        b_storage,
                        out_storage,
                        m,
                        n,
                        k,
                    ) {
                        Ok(_) => {
                            self.launch_counter.fetch_add(1, Ordering::Relaxed);
                            crate::device::compute::kernel_infra::record_kernel_route(entry);
                            let compute_handle =
                                Box::new(RocmHandle::new(Some(self.active_stream())));
                            return Ok(compute_handle);
                        }
                        Err(_) => {
                            // fall through to the direct launch below
                        }
                    }
                }
                // SPEED-ROC-16: B is now [N, K] row-major — exactly what R3 expects.
                // WMMA-T R3 wins the bake-off at 453µs vs 805µs rocBLAS for decode shapes.
                // The R3 single-wave tile mis-computes on RDNA4 wave32 when N or K is
                // not 16-aligned (observed as parity failures at e.g. 8x64x8); irregular
                // decode shapes take the scalar `grim_decode_gemm_f16` kernel instead.
                if n % 16 == 0 && k % 16 == 0 {
                    let stream = self.launch_wmma_gemm_b_transposed(
                        a_storage,
                        b_storage,
                        out_storage,
                        m,
                        n,
                        k,
                    )?;
                    let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                    return Ok(compute_handle);
                }
                let stream =
                    self.launch_decode_gemm_f16(a_storage, b_storage, out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok(compute_handle);
            }
        }

        if split_k_effective > 1 {
            let k_part = k / split_k_effective as usize;
            let partials_shape = Shape::from_slice(&[split_k_effective as usize, m, n]);
            let partials_storage = RocmStorage::alloc_gpu(
                &partials_shape,
                dtype_out.clone(),
                &self.allocator,
                self.ordinal,
            )?;

            let handle = self.get_rocblas_handle()?;
            // Bind the rocBLAS handle to the active stream so GEMM executes on the correct stream and the returned ComputeHandle synchronizes correctly.
            // [P0-17 fix: previously missing - caused sync-lie and split-K race.]
            let _ = unsafe { rocblas_set_stream(handle, self.active_stream()) };
            let alpha: f32 = 1.0f32;
            let beta: f32 = 0.0f32;

            let a_ptr_void = a_storage.device_ptr_checked()? as *const c_void;
            let b_ptr_void = b_storage.device_ptr_checked()? as *const c_void;
            let partials_ptr_void = partials_storage.device_ptr_checked()? as *mut c_void;

            let status = unsafe {
                let a_type = arith_to_rocblas_dtype(a_storage.dtype.arith);
                let b_type = arith_to_rocblas_dtype(b_storage.dtype.arith);
                let out_type = arith_to_rocblas_dtype(dtype_out.arith);
                let compute_type = arith_to_compute_dtype(dtype_out.arith);
                let alpha_ptr = &alpha as *const f32 as *const c_void;
                let beta_ptr = &beta as *const f32 as *const c_void;

                // SPEED-ROC-16: C = A @ B^T, B=[N,K], A=[M,K], split along K.
                // Mirror non-split-K rocBLAS layout: pass b_ptr as rocBLAS A with Transpose,
                // a_ptr as rocBLAS B with None. m=N, n=M, k=k_part.
                // lda=k (B full row stride), stride_a=k_part (advance k_part cols per batch).
                // ldb=k (A full row stride), stride_b=k_part (advance k_part cols per batch).
                rocblas_gemm_strided_batched_ex(
                    handle,
                    RocblasOperation::Transpose,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k_part as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    b_type,
                    k as RocblasInt,
                    k_part as i64,
                    a_ptr_void,
                    a_type,
                    k as RocblasInt,
                    k_part as i64,
                    beta_ptr,
                    partials_ptr_void,
                    out_type,
                    n as RocblasInt,
                    (m * n) as i64,
                    partials_ptr_void,
                    out_type,
                    n as RocblasInt,
                    (m * n) as i64,
                    split_k_effective as RocblasInt,
                    compute_type,
                    select_gemm_algo(solution_index),
                    solution_index,
                    ROCBLAS_GEMM_FLAGS_NONE,
                )
            };

            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas_gemm_strided_batched_ex failed with status {status}"
                )));
            }
            self.launch_counter.fetch_add(1, Ordering::Relaxed);
            crate::device::compute::kernel_infra::record_kernel_route(entry);

            // Sum up the partials along the batch dimension using the hand-written reduction kernel
            let stream = self.launch_split_k_reduction(
                &partials_storage,
                out_storage,
                m,
                n,
                split_k_effective,
            )?;
            let compute_handle = Box::new(RocmHandle::new(Some(stream)));
            return Ok(compute_handle);
        }
        #[cfg(feature = "rocm-profile")]
        println!(
            "[RocmDevice] GEMM Dispatch: Shape ({}, {}, {}) resolved to autotune tile config {:?} on Wavefront {:?}, solution_index={}",
            m, n, k, tile_config, self.props.wavefront_size, solution_index
        );

        // ─── Phase 4.5d: BF16 decode GEMV via V_DOT2_F32_BF16 (RDNA3/4 dot12-insts, m <= 4)
        {
            let is_rdna3_or_newer = matches!(
                crate::quantization::gcn_arch(&self.gpu_target),
                crate::quantization::GcnArch::RDNA3
                    | crate::quantization::GcnArch::RDNA4
                    | crate::quantization::GcnArch::UDNA
            );
            let dot_disabled = matches!(
                std::env::var("GRIM_DOT_GEMV").as_deref(),
                Ok("0" | "false" | "off")
            );
            if is_rdna3_or_newer
                && !dot_disabled
                && dtype_out.arith == ArithType::BF16
                && m <= 4
                && k % 32 == 0
            {
                let stream =
                    self.launch_dot2_bf16_gemv(a_storage, b_storage, out_storage, m, n, k)?;
                self.launch_counter.fetch_add(1, Ordering::Relaxed);
                crate::device::compute::kernel_infra::record_kernel_route(entry);
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok(compute_handle);
            }
        }

        // ─── RDNA3/4 + F16: route all M through b_transposed launcher ─────
        // launch_wmma_gemm_b_transposed internally dispatches:
        //   RDNA4/UDNA + m >= 16 → R4 multi-wave (16x64 tile, 2 waves)
        //   everything else      → R3 single-wave (16x32 tile, 1 wave)
        // RDNA3 always stays on R3 (the internal is_rdna4 guard blocks R4).
        // Decode shapes (m <= 8) already returned above; prefill lands here.
        {
            let is_rdna3_or_newer = matches!(
                crate::quantization::gcn_arch(&self.gpu_target),
                crate::quantization::GcnArch::RDNA3
                    | crate::quantization::GcnArch::RDNA4
                    | crate::quantization::GcnArch::UDNA
            );
            // R3 single-wave tile requires N/K 16-aligned: its boundary-B
            // staging mis-computes on RDNA4 wave32 when N % 16 != 0 (observed
            // as seed-dependent parity failures at e.g. 8x64x8). Non-aligned
            // shapes fall through to rocBLAS, which is exact.
            let tile_safe = n % 16 == 0 && k % 16 == 0;
            if is_rdna3_or_newer && tile_safe && dtype_out.arith == ArithType::F16 {
                let stream =
                    self.launch_wmma_gemm_b_transposed(a_storage, b_storage, out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok(compute_handle);
            }
        }

        // ─── WI-G — WMMA GEMM dispatch (opt-in, F16-only) ─────
        // SPEED-ROC-16 widened the trait contract to B = [N, K]; the
        // `grim_wmma_gemm` kernel still assumes the pre-widening B = [K, N]
        // layout, so it is layout-incompatible with this trait's matmul on
        // RDNA3/4 (where the [N, K]-aware b_transposed path above owns F16).
        // Only reachable there when the b_transposed path declined the shape.
        {
            let rdna34_f16_owned = matches!(
                crate::quantization::gcn_arch(&self.gpu_target),
                crate::quantization::GcnArch::RDNA3
                    | crate::quantization::GcnArch::RDNA4
                    | crate::quantization::GcnArch::UDNA
            ) && dtype_out.arith == ArithType::F16;
            if self.should_use_wmma_path(None, dtype_out.arith) && !rdna34_f16_owned {
                let stream = self.launch_wmma_gemm(a_storage, b_storage, out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok(compute_handle);
            }
        }

        // Get rocBLAS handle and execute sgemm. If handle is null (due to memory error fallback),
        // execute using WMMA HIP GEMM kernel directly.
        let handle = match self.get_rocblas_handle() {
            Ok(h) if !h.0.is_null() => h,
            _ => {
                let stream = self.launch_wmma_gemm(a_storage, b_storage, out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok(compute_handle);
            }
        };
        let _ = unsafe { rocblas_set_stream(handle, self.active_stream()) };

        let alpha: f32 = 1.0f32;
        let beta: f32 = 0.0f32;

        let a_ptr_void = a_storage.device_ptr_checked()? as *const c_void;
        let b_ptr_void = b_storage.device_ptr_checked()? as *const c_void;
        let out_ptr_void = out_storage.device_ptr_checked()? as *mut c_void;

        // In ROCm/rocBLAS (column-major), row-major C[M,N] = A[M,K] @ B[K,N] is

        let use_gemm_ex = cfg!(feature = "rocm-aiter")
            || self.gpu_target == "gfx90a"
            || self.gpu_target == "gfx942";

        unsafe {
            let status = if use_gemm_ex
                || dtype_out.arith == ArithType::F16
                || dtype_out.arith == ArithType::BF16
            {
                let a_type = arith_to_rocblas_dtype(a_storage.dtype.arith);
                let b_type = arith_to_rocblas_dtype(b_storage.dtype.arith);
                let out_type = arith_to_rocblas_dtype(dtype_out.arith);
                let compute_type = arith_to_compute_dtype(dtype_out.arith);
                let alpha_ptr = &alpha as *const f32 as *const c_void;
                let beta_ptr = &beta as *const f32 as *const c_void;
                // SPEED-ROC-16: C = A @ B^T, B stored (N, K). Row-major data is read by
                // rocBLAS as its transpose; to get C_colmajor[N,M] = B @ A^T we pass
                // b_ptr with transA=Trans (→ op(A)=[N,K]) and a_ptr with transB=NoTrans
                // (→ op(B)=[K,M]=A^T): m=N, n=M, k=K, lda=K, ldb=K, ldc=N.
                rocblas_gemm_ex(
                    handle,
                    RocblasOperation::Transpose,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    b_type,
                    k as RocblasInt,
                    a_ptr_void,
                    a_type,
                    k as RocblasInt,
                    beta_ptr,
                    out_ptr_void,
                    out_type,
                    n as RocblasInt,
                    out_ptr_void,
                    out_type,
                    n as RocblasInt,
                    compute_type,
                    select_gemm_algo(solution_index),
                    solution_index as RocblasInt,
                    ROCBLAS_GEMM_FLAGS_NONE,
                )
            } else {
                rocblas_sgemm(
                    handle,
                    RocblasOperation::Transpose,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    &alpha,
                    b_ptr_void as *const f32,
                    k as RocblasInt,
                    a_ptr_void as *const f32,
                    k as RocblasInt,
                    &beta,
                    out_ptr_void as *mut f32,
                    n as RocblasInt,
                )
            };

            if status != rocblas_status_success {
                // If rocBLAS matmul returns an error (e.g. status 1 = invalid handle),
                // fall back seamlessly to WMMA HIP GEMM kernel.
                let stream = self.launch_wmma_gemm(a_storage, b_storage, out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok(compute_handle);
            }
            self.launch_counter.fetch_add(1, Ordering::Relaxed);
            crate::device::compute::kernel_infra::record_kernel_route(entry);
        };

        let compute_handle = Box::new(RocmHandle::new(Some(self.active_stream())));
        Ok(compute_handle)
    }

    /// `C = A @ B^T` (SPEED-ROC-16 contract) writing into CALLER-PROVIDED
    /// `out` — no allocation inside. Mirrors the [`CoreTensorOps::matmul`]
    /// dispatch (FP32-GEMV fast path, then [`Self::matmul_op_into`]).
    /// Required for HIP graph capture: `out` is a stable pool address.
    pub fn matmul_into(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<Box<dyn ComputeHandle>> {
        let out_dims = out.shape().dims();
        let m = out_dims[..out_dims.len().saturating_sub(1)]
            .iter()
            .product::<usize>()
            .max(1);
        let n = out_dims.last().copied().unwrap_or(0);
        let k = a.shape().dims().last().copied().unwrap_or(0);
        if m == 1 && n >= 2048 {
            if let (Some(a_s), Some(b_s)) = (
                a.as_any().downcast_ref::<RocmStorage>(),
                b.as_any().downcast_ref::<RocmStorage>(),
            ) {
                let a_is_f32 = a_s.dtype().arith == ArithType::F32
                    && matches!(a_s.dtype().storage, crate::DTypeStorage::Native);
                let b_is_f32 = b_s.dtype().arith == ArithType::F32
                    && matches!(b_s.dtype().storage, crate::DTypeStorage::Native);
                if a_is_f32 && b_is_f32 && a_s.device_ptr_is_valid() && b_s.device_ptr_is_valid() {
                    let stream = self.launch_fp32_gemv(a_s, b_s, out, m, n, k)?;
                    return Ok(Box::new(RocmHandle::new(Some(stream))));
                }
            }
        }
        self.matmul_op_into(a, b, out, crate::autotune::GemmOp::Other)
    }

    /// Decode-only linear `y = a @ w^T` (m==1) writing into CALLER-PROVIDED
    /// `out` ([1, N] pool slot) plus caller `act_q81` scratch — no allocation
    /// inside. Mirrors the eager decode dispatch exactly so graph parity
    /// holds (same kernels eager would pick):
    /// - F32 weights -> [`Self::matmul_into`] (rocBLAS / FP32-GEMV fast path).
    /// - Q80 weights -> quantize + sudot4 `dot4_q80_q81` (dot4 arch,
    ///   `k_aligned>=32`), else WMMA fused-dequant.
    /// - Q4K/Q5K/Q6K/Q2K/Q3K weights -> quantize + matching sudot4 dot
    ///   (dot4 arch, `k%256==0`), else WMMA fused-dequant.
    /// - Anything else (IQ schemes, F16/BF16 acts, legacy/env-gated paths that
    ///   need their own scratch) -> `Unimplemented`, caller falls back eager.
    /// - `GRIM_DOT_GEMV=0` honored (forces WMMA legs).
    pub fn linear_decode_into(
        &self,
        a: &dyn BackendStorage,
        w: &RocmStorage,
        out: &RocmStorage,
        act_q81: &RocmStorage,
    ) -> Result<Box<dyn ComputeHandle>> {
        use grim_tensor::KQuantScheme;
        let a_dims = a.shape().dims();
        let k = a_dims.last().copied().unwrap_or(0);
        let m = a.shape().elem_count().checked_div(k.max(1)).unwrap_or(0);
        if m == 0 {
            return Err(Error::Unimplemented(
                "linear_decode_into: empty activation".into(),
            ));
        }
        // P3: batch>1 decode rides the same kernels (fp32/dot4 GEMVs index
        // rows via grid.y); all weight/output shapes are per-slot.
        let out_total = out.shape().elem_count();
        if m > 1 && out_total % m != 0 {
            return Err(Error::ShapeMismatch {
                expected: vec![m, out_total / m.max(1)],
                got: out.shape().dims().to_vec(),
            });
        }
        let n = if m == 1 { out_total } else { out_total / m };
        if w.shape().elem_count() % k.max(1) != 0 {
            return Err(Error::Shape(format!(
                "linear_decode_into: weight elems {} not a multiple of k={k}",
                w.shape().elem_count()
            )));
        }
        let w_n = w.shape().elem_count() / k.max(1);
        if w_n != n {
            return Err(Error::ShapeMismatch {
                expected: vec![1, w_n],
                got: vec![1, n],
            });
        }
        let need_q81 = m * (k / 32) * 36;
        let dot_disabled = matches!(
            std::env::var("GRIM_DOT_GEMV").as_deref(),
            Ok("0" | "false" | "off")
        );
        match &w.dtype().storage {
            DTypeStorage::Native => {
                // Same dispatch eager's Linear uses (fp32-GEMV fast path for
                // n>=2048, rocBLAS below) so both decode paths share one
                // primitive per weight — hc_fn is [24, 14336] and takes
                // rocBLAS on BOTH paths through this route. (GRIM_F32_GEMV=0
                // used to force this leg; it is now the only leg.)
                self.matmul_into(a, w, out)?;
                Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
            }
            DTypeStorage::KQuant(KQuantScheme::Q80) => {
                if act_q81.bytes < need_q81 {
                    return Err(Error::Backend(format!(
                        "linear_decode_into: act scratch {}B < {need_q81}B",
                        act_q81.bytes
                    )));
                }
                let k_aligned = k - (k % 32);
                if self.is_dot4_arch && !dot_disabled && k_aligned >= 32 {
                    let a_rocm = a.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
                        Error::Backend("linear_decode_into: a not RocmStorage".into())
                    })?;
                    self.launch_dot4_q80_f32act_gemv(a_rocm, w, out, m, n, k_aligned)?;
                    Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
                } else {
                    self.launch_wmma_fused_dequant_q8_0(
                        a.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
                            Error::Backend("linear_decode_into: a not RocmStorage".into())
                        })?,
                        w,
                        out,
                        m,
                        n,
                        k,
                    )?;
                    Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
                }
            }
            DTypeStorage::KQuant(
                scheme @ (KQuantScheme::Q4K
                | KQuantScheme::Q5K
                | KQuantScheme::Q6K
                | KQuantScheme::Q2K
                | KQuantScheme::Q3K
                | KQuantScheme::IQ3S
                | KQuantScheme::IQ4NL
                | KQuantScheme::IQ4XS
                | KQuantScheme::IQ2S
                | KQuantScheme::IQ2XS
                | KQuantScheme::IQ2XXS
                | KQuantScheme::IQ3XXS),
            ) => {
                if act_q81.bytes < need_q81 {
                    return Err(Error::Backend(format!(
                        "linear_decode_into: act scratch {}B < {need_q81}B",
                        act_q81.bytes
                    )));
                }
                let a_s = a.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
                    Error::Backend("linear_decode_into: a not RocmStorage".into())
                })?;
                if !(self.is_dot4_arch && !dot_disabled && k % 256 == 0) {
                    // No dot4 GEMV on this arch (or k not 256-aligned): the
                    // WMMA fused-dequant leg is the capture-safe fallback —
                    // the same launch shape the Q8_0 arm uses.
                    self.launch_quantize_q8_1(a_s, act_q81, m, k)?;
                    match scheme {
                        KQuantScheme::Q4K => {
                            self.launch_wmma_fused_dequant_q4k(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::Q5K => {
                            self.launch_wmma_fused_dequant_q5k(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::Q6K => {
                            self.launch_wmma_fused_dequant_q6k(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::Q3K => {
                            self.launch_wmma_fused_dequant_q3k(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::Q2K => {
                            self.launch_wmma_fused_dequant_q2k(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::IQ3S => {
                            self.launch_wmma_fused_dequant_iq3s(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::IQ4NL => {
                            self.launch_wmma_fused_dequant_iq4nl(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::IQ4XS => {
                            self.launch_wmma_fused_dequant_iq4xs(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::IQ2S => {
                            self.launch_wmma_fused_dequant_iq2s(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::IQ2XS => {
                            self.launch_wmma_fused_dequant_iq2xs(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::IQ2XXS => {
                            self.launch_wmma_fused_dequant_iq2xxs(act_q81, w, out, m, n, k)?;
                        }
                        KQuantScheme::IQ3XXS => {
                            self.launch_wmma_fused_dequant_iq3xxs(act_q81, w, out, m, n, k)?;
                        }
                        other => {
                            return Err(Error::Backend(format!(
                                "linear_decode_into: no WMMA leg for K-quant {other:?}"
                            )))
                        }
                    }
                    return Ok(Box::new(RocmHandle::new(Some(self.active_stream()))));
                }
                // GRIM_DECODE_W4A4=1: q4k weights are requantized once, per
                // tensor, to the WhiteCrow u4 group-128 layout and decode rides
                // the native v_dot8 GEMV. Probe-measured 435 GB/s vs 23 GB/s
                // for the q4k dot4 kernel (tests/dot4_gemv_floor_probe.rs).
                // Conversion is cached per weight device pointer; it D2Hs the
                // packed bytes on first use, so it must NOT run under graph
                // capture (hipMemcpy sync + alloc poison capture) — eager path
                // only, env-gated off by default.
                static W4A4_DECODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let w4a4_decode = *W4A4_DECODE.get_or_init(|| {
                    matches!(
                        std::env::var("GRIM_DECODE_W4A4").as_deref(),
                        Ok("1" | "true")
                    )
                });
                if matches!(
                    scheme,
                    KQuantScheme::Q4K | KQuantScheme::Q5K | KQuantScheme::Q6K
                ) && w4a4_decode
                    && k % 128 == 0
                {
                    // KDA-FIX: the cache key was the RAW DEVICE POINTER. The
                    // caching allocator recycles blocks — a freed weight
                    // storage's pointer can be handed to a DIFFERENT tensor,
                    // and the cache then serves that tensor the first one's
                    // converted weights (silent wrong weights). Key on
                    // (pointer, bytes, shape) — three recycled-pointer
                    // collisions in a row with matching bytes AND shape are
                    // not a realistic hazard, and the true fix (keyed on the
                    // allocator generation) needs an allocator API this does
                    // not have. Better still: weights are static for the
                    // process lifetime, so entries never evict.
                    static CONVERTED: std::sync::OnceLock<
                        std::sync::Mutex<
                            std::collections::HashMap<
                                (usize, usize, [usize; 2]),
                                std::sync::Arc<WhiteCrowDecodedWeights>,
                            >,
                        >,
                    > = std::sync::OnceLock::new();
                    let cache = CONVERTED
                        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
                    let key = (
                        w.device_ptr.map(|p| p as usize).unwrap_or(0),
                        w.bytes,
                        [n, k],
                    );
                    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
                    let conv = match guard.get(&key) {
                        Some(c) => c.clone(),
                        None => {
                            let c = std::sync::Arc::new(
                                self.requant_kquant_to_whitecrow(w, n, k, *scheme)?,
                            );
                            guard.insert(key, c.clone());
                            c
                        }
                    };
                    drop(guard);
                    self.launch_w4a4_ostquant_gemv(
                        a_s,
                        conv.qweight_rocm()?,
                        conv.scales_rocm()?,
                        conv.zeros_rocm()?,
                        out,
                        m,
                        n,
                        k,
                    )?;
                    return Ok(Box::new(RocmHandle::new(Some(self.active_stream()))));
                }
                // Exact leg: dequantize the weight inline and keep the
                // activation in F32. Shares its switch with eager's
                // `quantized_matmul` so the two decode paths cannot pick
                // different GEMVs - see `exact_decode_gemv`.
                if crate::device::quant::exact_decode_gemv() {
                    match scheme {
                        KQuantScheme::Q4K => {
                            self.launch_fused_dequant_gemm_q4k(a_s, w, out, m, n, k)?;
                            return Ok(Box::new(RocmHandle::new(Some(self.active_stream()))));
                        }
                        KQuantScheme::IQ3S => {
                            self.launch_fused_dequant_gemm_iq3s(a_s, w, out, m, n, k)?;
                            return Ok(Box::new(RocmHandle::new(Some(self.active_stream()))));
                        }
                        _ => {}
                    }
                }
                self.launch_quantize_q8_1(a_s, act_q81, m, k)?;
                match scheme {
                    KQuantScheme::Q4K => self.launch_dot4_q4k_q81_gemv(act_q81, w, out, m, n, k)?,
                    KQuantScheme::Q5K => self.launch_dot4_q5k_q81_gemv(act_q81, w, out, m, n, k)?,
                    // Q6_K's dot4 GEMV is not bit-exact at production shapes:
                    // the Xing4.0-29B lm_head oracle (real output.weight,
                    // k=3584) measures rel 3.2e-3 / max_abs 1.5 at m=1, while
                    // the fused-dequant path agrees with the host dequant at
                    // 3.9e-7. The k=256 arch probes cannot see this (the scale
                    // skew grows with k), so Q6_K skips the dot4 leg here.
                    // Same treatment as the IQ3_S fused WMMA GEMM, and gated
                    // on the same GRIM_Q6K_DOT4 escape hatch as the sibling
                    // dispatch in device/quant/mod.rs — two sites, one switch,
                    // so a capture run and an eager run cannot disagree about
                    // which Q6_K path is trusted.
                    KQuantScheme::Q6K => {
                        if matches!(
                            std::env::var("GRIM_Q6K_DOT4").as_deref(),
                            Ok("1" | "true" | "on")
                        ) {
                            self.launch_dot4_q6k_q81_gemv(act_q81, w, out, m, n, k)?
                        } else {
                            self.launch_fused_dequant_gemm_q6k(a_s, w, out, m, n, k)?
                        }
                    }
                    KQuantScheme::Q2K => self.launch_dot4_q2k_q81_gemv(act_q81, w, out, m, n, k)?,
                    // IQ3_S has NO dot4 GEMV, and it must never reach the
                    // catch-all below: IQ3_S and Q3_K share the 110 B / 256
                    // block size but not the layout, so decoding IQ3_S bytes
                    // with the Q3_K kernel silently yields garbage (measured:
                    // every Xing4.0 projection came out zero, because the whole
                    // model is IQ3_S). The fused-dequant leg reads A as f32 and
                    // dequantizes inline, so it needs no act_q81 either.
                    KQuantScheme::IQ3S => self.launch_fused_dequant_gemm_iq3s(a_s, w, out, m, n, k)?,
                    _ => self.launch_dot4_q3k_q81_gemv(act_q81, w, out, m, n, k)?,
                };
                Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
            }
            // WhiteRaven blocked FP8 (16x16-blocked E4M3 B). The EAGER leg
            // only: it quantizes A through the host (D2H readback plus an
            // allocation), which is correct but not fast and is exactly what a
            // HIP graph capture cannot contain. Graph decode must call
            // `linear_decode_blocked_into` instead, which quantizes A on device
            // into caller-owned scratch. Refusing under capture rather than
            // silently doing it keeps the failure at the call site instead of
            // as a poisoned graph.
            DTypeStorage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16) => {
                if crate::decode_graph_buffers::capture_fp8_pad_scratch().is_some() {
                    return Err(Error::Backend(
                        "linear_decode_into: WhiteRaven blocked weight reached the eager leg \
                         during graph capture; use linear_decode_blocked_into (host act \
                         quantization allocates and syncs, which poisons capture)"
                            .into(),
                    ));
                }
                let a_s = a.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
                    Error::Backend("linear_decode_into: a not RocmStorage".into())
                })?;
                if k % 16 != 0 || n % 16 != 0 {
                    return Err(Error::Backend(format!(
                        "WhiteRaven blocked requires k % 16 == 0 and n % 16 == 0; got k={k} n={n}"
                    )));
                }
                // Either f32 acts (quantize here) or u8 codes (pad rows only).
                let a_rows = m.div_ceil(16) * 16;
                let mut padded = vec![0u8; a_rows * k];
                if a.dtype().arith == ArithType::U8 {
                    let raw = a_s
                        .copy_to_host()
                        .map_err(|e| Error::Backend(format!("WhiteRaven act readback: {e}")))?;
                    if raw.len() < m * k {
                        return Err(Error::Backend(format!(
                            "WhiteRaven u8 act needs {0} codes, got {1}",
                            m * k,
                            raw.len()
                        )));
                    }
                    for r in 0..m {
                        padded[r * k..(r + 1) * k].copy_from_slice(&raw[r * k..(r + 1) * k]);
                    }
                } else {
                    let raw = a
                        .to_cpu_vec_f32()
                        .map_err(|e| Error::Backend(format!("WhiteRaven act readback: {e}")))?;
                    if raw.len() < m * k {
                        return Err(Error::Backend(format!(
                            "WhiteRaven act needs {0} values, got {1}",
                            m * k,
                            raw.len()
                        )));
                    }
                    for (i, v) in raw.iter().take(m * k).enumerate() {
                        padded[i] = grim_quant::f32_to_fp8_e4m3(*v);
                    }
                }
                let converted = RocmStorage::copy_from_host_raw_bytes(
                    &padded,
                    &Shape::new(vec![a_rows, k]),
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::Native,
                    },
                    &self.allocator,
                    self.ordinal,
                )?;
                self.launch_wmma_gemm_fp8_e4m3_blocked(&converted, w, out, m, n, k)?;
                Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
            }
            _ => Err(Error::Unimplemented(format!(
                "linear_decode_into: unsupported weight dtype for graph decode: {:?}",
                w.dtype()
            ))),
        }
    }

    /// Public hook for the engine layer to tag the lm_head / logit-projection GEMM, so the dispatch layer classifies it as `ShapeClass::TLOLog` (op-identity) and selects the distinct wide-N tile regardless of M.
    /// This is the jit-mgpu.md §4.2 dispatch-layer tag.
    pub fn matmul_lm_head(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, crate::autotune::GemmOp::LmHead)
    }
}

/// Synthetic route name for a GEMM dispatched to the vendor BLASLt library
/// rather than to a named HIP compute kernel.
///
/// Recorded so a journey test can assert the kernel under test actually ran.
/// Without it a silent fallback to cuBLASLt would be invisible: the per-device
/// total launch counter would still have moved, so "my kernel ran" could not be
/// distinguished from "something ran".
pub const BLASLT_ROUTE: &str = "blaslt_external";

/// One q4k weight tensor requantized to the WhiteCrow u4 group-128 layout.
/// The three storages must outlive every decode launch that reads them, so
/// they live in the per-device conversion cache for the process lifetime.
pub struct WhiteCrowDecodedWeights {
    pub qweight: Box<dyn grim_tensor::backend::BackendStorage>,
    pub scales: Box<dyn grim_tensor::backend::BackendStorage>,
    pub zeros: Box<dyn grim_tensor::backend::BackendStorage>,
}

impl WhiteCrowDecodedWeights {
    pub fn qweight_rocm(&self) -> Result<&RocmStorage> {
        self.qweight
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("requant: qweight not RocmStorage".into()))
    }
    pub fn scales_rocm(&self) -> Result<&RocmStorage> {
        self.scales
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("requant: scales not RocmStorage".into()))
    }
    pub fn zeros_rocm(&self) -> Result<&RocmStorage> {
        self.zeros
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("requant: zeros not RocmStorage".into()))
    }
}

impl RocmDevice {
    /// Dequantize a packed Q4_K weight tensor on the host and re-encode it as
    /// unsigned-4-bit group-128 OSTQuant (WhiteCrow). One-time cost per tensor
    /// (~a few ms for a 9B FFN matrix); see GRIM_DECODE_W4A4 in
    /// `linear_decode_into`.
    pub fn requant_kquant_to_whitecrow(
        &self,
        w: &RocmStorage,
        n: usize,
        k: usize,
        scheme: grim_tensor::KQuantScheme,
    ) -> Result<WhiteCrowDecodedWeights> {
        let packed = w.copy_to_host()?;

        // On-disk cache (GRIM_OSTQUANT_CACHE_DIR, default ~/.cache/grim/ostquant;
        // GRIM_OSTQUANT_CACHE=0 disables). The key mixes the encoder version,
        // the geometry and a hash of the PACKED source bytes, so a stale
        // converter can never be served for a different weight.
        let disk_enabled = !matches!(
            std::env::var("GRIM_OSTQUANT_CACHE").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        );
        let cache_key: Option<u64> = disk_enabled.then(|| {
            let mut h = seahash::hash(&packed);
            h ^= (n as u64) << 1;
            h ^= (k as u64) << 17;
            h ^= (scheme as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            h ^= (grim_quant::OSTQUANT_ENCODER_VERSION as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            h
        });
        let cache_dir = || {
            std::env::var("GRIM_OSTQUANT_CACHE_DIR")
                .ok()
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    let base = std::env::var("XDG_CACHE_HOME")
                        .ok()
                        .map(std::path::PathBuf::from)
                        .or_else(|| {
                            std::env::var("HOME")
                                .ok()
                                .map(|h| std::path::PathBuf::from(h).join(".cache"))
                        })?;
                    Some(base.join("grim").join("ostquant"))
                })
        };
        if let (Some(key), Some(dir)) = (cache_key, cache_dir()) {
            let path = dir.join(format!("{key:016x}.wc"));
            // Header: magic, encoder version, n, k, three lengths.
            if let Ok(meta) = std::fs::read(&path) {
                if meta.len() >= 28 {
                    let rd = |o: usize| {
                        u32::from_le_bytes([meta[o], meta[o + 1], meta[o + 2], meta[o + 3]])
                            as usize
                    };
                    if rd(0) == 0x57_43_00_01
                        && rd(4) == grim_quant::OSTQUANT_ENCODER_VERSION as usize
                        && rd(8) == n
                        && rd(12) == k
                        && meta.len() == 28 + rd(16) + rd(20) + rd(24)
                    {
                        let (lq, ls, lz) = (rd(16), rd(20), rd(24));
                        let qw = meta[28..28 + lq].to_vec();
                        let sc = meta[28 + lq..28 + lq + ls].to_vec();
                        let zr = meta[28 + lq + ls..28 + lq + ls + lz].to_vec();
                        return self.whitecrow_from_host_bytes(qw, sc, zr, n, k);
                    }
                }
            }
        }

        // Convert row-block-parallel: both stages are row-separable (a Q4_K
        // super-block and an OSTQuant group-128 never cross a row, and
        // linear_decode_into already requires k % 128 == 0, k % 256 == 0), and
        // every output buffer is row-major, so chunking rows and concatenating
        // in order is byte-identical to the serial path.
        let (block_bytes, per_256): (usize, usize) = match scheme {
            grim_tensor::KQuantScheme::Q4K => (144, 256),
            grim_tensor::KQuantScheme::Q5K => (176, 256),
            grim_tensor::KQuantScheme::Q6K => (210, 256),
            grim_tensor::KQuantScheme::Q3K => (110, 256),
            // IQ3_S: same 256/110 super-block geometry as Q3_K.
            grim_tensor::KQuantScheme::IQ3S => (110, 256),
            grim_tensor::KQuantScheme::Q2K => (84, 256),
            // IQ4_NL: 18 B per 32 weights => 8 blocks x 18 B per 256.
            grim_tensor::KQuantScheme::IQ4NL => (144, 256),
            _ => {
                return Err(Error::Backend(format!(
                    "requant_kquant_to_whitecrow: unsupported scheme {scheme:?}"
                )))
            }
        };
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(8)
            .max(1);
        let row_bytes = (k / per_256) * block_bytes;
        let (qw, sc, zr) = if threads > 1 && n >= 2 && k % per_256 == 0 {
            let rows_per = n.div_ceil(threads);
            let chunks: Vec<(usize, usize)> = (0..n)
                .step_by(rows_per)
                .map(|r0| (r0, (r0 + rows_per).min(n)))
                .collect();
            let parts: Vec<Result<(Vec<u8>, Vec<u8>, Vec<u8>)>> = std::thread::scope(|scope| {
                let handles: Vec<_> = chunks
                    .iter()
                    .map(|&(r0, r1)| {
                        let packed = &packed;
                        let scheme = scheme;
                        scope.spawn(move || {
                            let slice = &packed[r0 * row_bytes..r1 * row_bytes];
                            let rows = r1 - r0;
                            let dq = match scheme {
                                grim_tensor::KQuantScheme::Q4K => {
                                    grim_quant::dequant_q4k(slice, rows * k)?
                                }
                                grim_tensor::KQuantScheme::Q5K => {
                                    grim_quant::dequant_q5k(slice, rows * k)?
                                }
                                grim_tensor::KQuantScheme::Q6K => {
                                    grim_quant::dequant_q6k(slice, rows * k)?
                                }
                                grim_tensor::KQuantScheme::Q3K => {
                                    grim_quant::dequant_q3k(slice, rows * k)?
                                }
                                grim_tensor::KQuantScheme::IQ3S => {
                                    grim_quant::dequant_iq3s(slice, rows * k)?
                                }
                                grim_tensor::KQuantScheme::Q2K => {
                                    grim_quant::dequant_q2k(slice, rows * k)?
                                }
                                grim_tensor::KQuantScheme::IQ4NL => {
                                    grim_quant::dequant_iq4nl(slice, rows * k)?
                                }
                                _ => {
                                    return Err(Error::Backend(format!(
                                        "requant_kquant_to_whitecrow: no dequant for {scheme:?}"
                                    )))
                                }
                            };
                            grim_quant::quant_ostquant_w4_group128(&dq, rows, k)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            Err(Error::Backend("requant worker panicked".into()))
                        })
                    })
                    .collect()
            });
            let mut qw_all = Vec::new();
            let mut sc_all = Vec::new();
            let mut zr_all = Vec::new();
            for part in parts {
                let (a, b, c) = part?;
                qw_all.extend_from_slice(&a);
                sc_all.extend_from_slice(&b);
                zr_all.extend_from_slice(&c);
            }
            (qw_all, sc_all, zr_all)
        } else {
            let dequantized = match scheme {
                grim_tensor::KQuantScheme::Q4K => grim_quant::dequant_q4k(&packed, n * k)?,
                grim_tensor::KQuantScheme::Q5K => grim_quant::dequant_q5k(&packed, n * k)?,
                grim_tensor::KQuantScheme::Q6K => grim_quant::dequant_q6k(&packed, n * k)?,
                grim_tensor::KQuantScheme::Q3K => grim_quant::dequant_q3k(&packed, n * k)?,
                grim_tensor::KQuantScheme::IQ3S => grim_quant::dequant_iq3s(&packed, n * k)?,
                grim_tensor::KQuantScheme::Q2K => grim_quant::dequant_q2k(&packed, n * k)?,
                grim_tensor::KQuantScheme::IQ4NL => grim_quant::dequant_iq4nl(&packed, n * k)?,
                _ => {
                    return Err(Error::Backend(format!(
                        "requant_kquant_to_whitecrow: no dequant for {scheme:?}"
                    )))
                }
            };
            grim_quant::quant_ostquant_w4_group128(&dequantized, n, k)?
        };
        // Persist for the next run (best-effort; a failed write only costs
        // the conversion next time).
        if let (Some(key), Some(dir)) = (cache_key, cache_dir()) {
            let mut blob: Vec<u8> = Vec::with_capacity(28 + qw.len() + sc.len() + zr.len());
            let push_u = |v: &mut Vec<u8>, x: usize| {
                v.extend_from_slice(&(x as u32).to_le_bytes());
            };
            push_u(&mut blob, 0x57_43_00_01);
            push_u(&mut blob, grim_quant::OSTQUANT_ENCODER_VERSION as usize);
            push_u(&mut blob, n);
            push_u(&mut blob, k);
            push_u(&mut blob, qw.len());
            push_u(&mut blob, sc.len());
            push_u(&mut blob, zr.len());
            blob.extend_from_slice(&qw);
            blob.extend_from_slice(&sc);
            blob.extend_from_slice(&zr);
            let tmp = dir.join(format!("{key:016x}.wc.tmp"));
            if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&tmp, &blob).is_ok() {
                let _ = std::fs::rename(&tmp, dir.join(format!("{key:016x}.wc")));
            }
        }
        self.whitecrow_from_host_bytes(qw, sc, zr, n, k)
    }

    /// Upload already-converted WhiteCrow bytes (qweight as u32 words,
    /// group-128 bf16 scales, u8 zeros).
    fn whitecrow_from_host_bytes(
        &self,
        qw: Vec<u8>,
        sc: Vec<u8>,
        zr: Vec<u8>,
        n: usize,
        k: usize,
    ) -> Result<WhiteCrowDecodedWeights> {
        use grim_tensor::{BackendDevice, MemoryOps};
        let bf16_dtype = DType {
            arith: ArithType::BF16,
            storage: DTypeStorage::Native,
        };
        // qweight is u32 WORDS ([N, K/8] words, 4 B each) — the W4A4 kernel
        // indexes it as `const unsigned int*`. Uploading as U8 truncated the
        // buffer to a quarter and the kernel read garbage.
        let qweight = MemoryOps::from_cpu_bytes(
            self as &dyn BackendDevice,
            &qw,
            &Shape::new(vec![n, k / 8]),
            DType {
                arith: ArithType::U32,
                storage: DTypeStorage::Native,
            },
        )?;
        let scales = MemoryOps::from_cpu_bytes(
            self as &dyn BackendDevice,
            &sc,
            &Shape::new(vec![n, k / 128]),
            bf16_dtype,
        )?;
        let zeros = MemoryOps::from_cpu_bytes(
            self as &dyn BackendDevice,
            &zr,
            &Shape::new(vec![n, k / 128]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        Ok(WhiteCrowDecodedWeights {
            qweight,
            scales,
            zeros,
        })
    }
    /// As [`Self::launch_grey_raven_probe`], but with a **per-lane** sparsity index
    /// buffer of 32 u32s, which is the form the ISA actually defines.
    pub fn launch_grey_raven_probe_lane_idx(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        sidx: &RocmStorage,
        c_out: &RocmStorage,
    ) -> Result<*mut c_void> {
        let ap = a
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_probe: a has no device ptr".into()))?;
        let bp = b
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_probe: b has no device ptr".into()))?;
        let sp = sidx
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_probe: sidx has no device ptr".into()))?;
        let cp = c_out
            .device_ptr
            .ok_or_else(|| Error::Backend("grey_raven_probe: c has no device ptr".into()))?;
        let (mut aptr, mut bptr, mut sptr, mut cptr) = (ap, bp, sp, cp);
        self.launch_from_source(
            crate::kernels::grey_raven::PROBE_LANE_IDX_SOURCE,
            "grim_grey_raven_probe_lane_idx",
            HipDim3::new(1, 1, 1),
            HipDim3::new(32, 1, 1),
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sptr),
                arg(&mut cptr),
            ],
        )
    }
}

/// Capture-safe staging primitives: a byte memset and a device-to-device copy
/// issued on the device's active stream.
///
/// Both are needed by paths that stage into a caller-owned scratch (the FP8
/// padded-activation pad rows, for one) and both must be stream-ordered and
/// allocation-free, which rules out the host round-trip they replace.
impl RocmDevice {
    /// Zero (or fill) `bytes` at `buf` on the active stream.
    pub fn launch_memset_u8(
        &self,
        buf: &RocmStorage,
        value: u8,
        bytes: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let ptr = buf.device_ptr.ok_or_else(|| {
            crate::Error::Backend("launch_memset_u8: buffer has no device ptr".into())
        })? as *mut std::ffi::c_void;
        if bytes == 0 {
            return Ok(ptr);
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let stream = self.active_stream();
        crate::check_hip(
            "hipMemsetD8Async",
            unsafe { crate::device::handles::hipMemsetD8Async(ptr, value, bytes, stream) },
        )?;
        Ok(ptr)
    }

    /// Copy `bytes` from `src` to `dst + offset_bytes` on the active stream.
    pub fn launch_memcpy_d2d(
        &self,
        dst: &RocmStorage,
        src: &RocmStorage,
        offset_bytes: usize,
        bytes: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let d = dst.device_ptr.ok_or_else(|| {
            crate::Error::Backend("launch_memcpy_d2d: dst has no device ptr".into())
        })? as *mut std::ffi::c_void;
        let s = src.device_ptr.ok_or_else(|| {
            crate::Error::Backend("launch_memcpy_d2d: src has no device ptr".into())
        })? as *const std::ffi::c_void;
        if bytes == 0 {
            return Ok(d);
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let stream = self.active_stream();
        crate::check_hip(
            "hipMemcpyDtoDAsync",
            unsafe {
                crate::device::handles::hipMemcpyDtoDAsync(
                    (d as usize + offset_bytes) as *mut std::ffi::c_void,
                    s,
                    bytes,
                    stream,
                )
            },
        )?;
        Ok(d)
    }
}
