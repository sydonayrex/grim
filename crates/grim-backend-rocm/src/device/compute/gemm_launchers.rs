//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! GEMM launchers: rocBLAS/decode paths, WMMA (incl. fused dequant-GEMM), FP8 RDNA4, matmul ops.

use std::ffi::c_void;

use std::sync::atomic::Ordering;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::gemm_tuning::{lookup_gemm_config_for_shape, lookup_solution_index};
use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    arg, arith_to_compute_dtype, arith_to_rocblas_dtype, rocblas_gemm_ex,
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
    #[allow(dead_code)]
    pub(crate) fn launch_wmma_gemm_fp8_e4m3(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_wmma_fused_dequant_quant("grim_wmma_gemm_fp8_e4m3", a, b, out, m, n, k)
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
        let f32_gemv_disabled = matches!(
            std::env::var("GRIM_F32_GEMV").as_deref(),
            Ok("0" | "false" | "off")
        );
        match &w.dtype().storage {
            DTypeStorage::Native => {
                if !f32_gemv_disabled {
                    let stream = self.launch_f32_gemv_into(a, w, out, n, k)?;
                    return Ok(Box::new(RocmHandle::new(Some(stream))));
                }
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
                | KQuantScheme::Q3K),
            ) => {
                if act_q81.bytes < need_q81 {
                    return Err(Error::Backend(format!(
                        "linear_decode_into: act scratch {}B < {need_q81}B",
                        act_q81.bytes
                    )));
                }
                if !(self.is_dot4_arch && !dot_disabled && k % 256 == 0) {
                    return Err(Error::Unimplemented(
                        "linear_decode_into: K-quant needs dot4 arch + k%256==0 (else WMMA leg needs its own scratch)".into(),
                    ));
                }
                let a_s = a.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
                    Error::Backend("linear_decode_into: a not RocmStorage".into())
                })?;
                self.launch_quantize_q8_1(a_s, act_q81, m, k)?;
                match scheme {
                    KQuantScheme::Q4K => self.launch_dot4_q4k_q81_gemv(act_q81, w, out, m, n, k)?,
                    KQuantScheme::Q5K => self.launch_dot4_q5k_q81_gemv(act_q81, w, out, m, n, k)?,
                    KQuantScheme::Q6K => self.launch_dot4_q6k_q81_gemv(act_q81, w, out, m, n, k)?,
                    KQuantScheme::Q2K => self.launch_dot4_q2k_q81_gemv(act_q81, w, out, m, n, k)?,
                    _ => self.launch_dot4_q3k_q81_gemv(act_q81, w, out, m, n, k)?,
                };
                Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
            }
            _ => Err(Error::Unimplemented(
                "linear_decode_into: unsupported weight dtype for graph decode".into(),
            )),
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
