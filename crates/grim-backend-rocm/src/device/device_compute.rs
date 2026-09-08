//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.

use std::ffi::c_void;

use std::sync::atomic::Ordering;

use grim_tensor::backend::{ComputeHandle, ReadyHandle};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{
    AutogradOps, BackendStorage, CoreTensorOps, ElementwiseOps, FusionOps, OptimizerOps,
    SamplingOps, Shape,
};

use crate::device::gemm_tuning::{
    lookup_gemm_config, lookup_gemm_config_for_shape, lookup_solution_index,
};
use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    HipDim3, QkvAttentionFusionConfig, QuantMode, ROCBLAS_GEMM_FLAGS_NONE,
    RmsNormMatMulFusionConfig, RocblasInt, RocblasOperation, RocmHandle, arg,
    arith_to_compute_dtype, arith_to_rocblas_dtype, as_rocm, check_hip, dev_ptr, dtype_f32,
    hipFree, hipFreeAsync, hipMemAdvise, hipMemsetAsync, hipModuleGetFunction,
    hipModuleLaunchKernel, hipModuleLoad, hipModuleUnload, hipSuccess, jit_compile_hsaco,
    linear_launch, rocblas_gemm_ex, rocblas_gemm_strided_batched_ex, rocblas_set_stream,
    rocblas_sgemm, rocblas_status_success, select_gemm_algo, upload_device_buffer,
    warp_rows_launch,
};

impl CoreTensorOps for RocmDevice {
    /// Audit B5: delegate to the device-resident `grim_transpose_2d_f32` HIP kernel via the existing inherent helper - the tensor
    /// never leaves GPU memory (the helper synchronizes the kernel launch internally, so the returned handle is trivially ready).
    fn transpose_2d(
        &self,
        x: &dyn BackendStorage,
        rows: usize,
        cols: usize,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out = self.transpose_f32_2d(x, rows, cols)?;
        Ok((out, Box::new(ReadyHandle)))
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: zeros");

        // `hipMemset` zeroes bytes, which is only correct when the dtype's zero
        let storage = RocmStorage::alloc_gpu(shape, dtype.clone(), &self.allocator, self.ordinal)?;

        if !storage.device_ptr_is_valid() {
            return Err(Error::Backend("Invalid device pointer after alloc".into()));
        }

        let dev_ptr_void = storage.device_ptr_checked()? as *mut c_void;

        // If a graph-capture session is active, record an async memset on the capture stream; otherwise enqueue async on the
        // active stream so zeroing stays stream-ordered instead of blocking the host (the old default-stream hipMemset was a device-wide serialization point).
        let res = match self.active_capture_stream() {
            Some(capture_stream) => unsafe {
                hipMemsetAsync(dev_ptr_void, 0, storage.bytes, capture_stream)
            },
            None => unsafe { hipMemsetAsync(dev_ptr_void, 0, storage.bytes, self.active_stream()) },
        };

        if res != hipSuccess {
            // Free on failure
            if let Some(ptr) = storage.device_ptr {
                let ptr_void = ptr as *mut c_void;
                unsafe {
                    _ = hipFree(ptr_void);
                }
            }
            return Err(Error::Backend(format!(
                "hipMemset for zeros failed with error code {}",
                res
            )));
        }

        Ok(Box::new(storage))
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: from_cpu");

        RocmStorage::copy_from_host(data, shape, dtype, &self.allocator, self.ordinal)
            .map(|s| Box::new(s) as Box<dyn BackendStorage>)
    }

    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, crate::autotune::GemmOp::Other)
    }

    fn matmul_with_solution(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
        solution_index: i32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: matmul_with_solution");

        // P1-3: this is the plain `matmul` dispatch path - rocBLAS executes on the calling thread's current device, which after the context-neutral `try_new` is typically ordinal 0 on multi-GPU boxes.
        // Pin before any alloc or rocBLAS call or the GEMM launches cross-device and silently writes.
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        // For matmul on GPU, both inputs must be RocmStorage (or we need to copy them to the device first)
        let a_storage = match a.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => {
                return Err(Error::Backend(
                    "matmul_with_solution: input a is not RocmStorage".into(),
                ));
            }
        };

        let b_storage = match b.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => {
                return Err(Error::Backend(
                    "matmul_with_solution: input b is not RocmStorage".into(),
                ));
            }
        };

        if !a_storage.device_ptr_is_valid() || !b_storage.device_ptr_is_valid() {
            return Err(Error::Backend(
                "matmul_with_solution: inputs must have valid GPU device pointers".into(),
            ));
        }

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(Error::Shape(
                "matmul_with_solution expects 2-D inputs".into(),
            ));
        }

        let (m, k) = (a_dims[0], a_dims[1]);
        let (k2, n) = (b_dims[0], b_dims[1]);

        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }

        if out_shape.dims() != [m, n] {
            return Err(Error::Shape(format!(
                "expected out [{m},{n}], got {:?}",
                out_shape.dims()
            )));
        }

        // Allocate output GPU storage with the actual input precision
        let dtype_out = DType {
            arith: a_storage.dtype.arith,
            storage: DTypeStorage::Native,
        };
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_out.clone(), &self.allocator, self.ordinal)?;

        // Shape-indexed GEMM dispatch lookup (Tensile-inspired layout resolution)
        let _tile_config = lookup_gemm_config(m, n, k, self.props.wavefront_size);
        #[cfg(feature = "rocm-profile")]
        println!(
            "[RocmDevice] GEMM Dispatch: Shape ({}, {}, {}) resolved to autotune tile config {:?} on Wavefront {:?}",
            m, n, k, _tile_config, self.props.wavefront_size
        );

        // Get rocBLAS handle and execute gemm_ex with the provided solution_index
        let handle = self.get_rocblas_handle()?;
        // Bind the rocBLAS handle to the active stream so GEMM executes on the correct stream and the returned ComputeHandle synchronizes correctly.
        // [P0-17 fix: previously missing - caused sync-lie and split-K race.]
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
                rocblas_gemm_ex(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    b_type,
                    n as RocblasInt,
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
                    // Wire `lookup_solution_index` to `algo` so rocBLAS actually [see: `select_gemm_algo(0)`, `standard`]
                    select_gemm_algo(solution_index),
                    solution_index as RocblasInt,
                    ROCBLAS_GEMM_FLAGS_NONE,
                )
            } else {
                rocblas_sgemm(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    &alpha,
                    b_ptr_void as *const f32,
                    n as RocblasInt,
                    a_ptr_void as *const f32,
                    k as RocblasInt,
                    &beta,
                    out_ptr_void as *mut f32,
                    n as RocblasInt,
                )
            };

            if status != rocblas_status_success {
                return Err(Error::Backend(format!(
                    "rocblas matmul_with_solution execution failed with error status {}",
                    status
                )));
            }
        };

        let compute_handle = Box::new(RocmHandle::new(Some(self.active_stream())));
        Ok((Box::new(out_storage), compute_handle))
    }

    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        if !a_s.device_ptr_is_valid() || !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "add: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;

        let mut out_ptr = dev_ptr(&storage)?;
        let mut a_ptr = dev_ptr(a_s)?;
        let mut b_ptr = dev_ptr(b_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_add",
            grid,
            block,
            &mut [
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        if !a_s.device_ptr_is_valid() || !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "mul: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut a_ptr = dev_ptr(a_s)?;
        let mut b_ptr = dev_ptr(b_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_mul",
            grid,
            block,
            &mut [
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let gate_s = as_rocm(gate)?;
        let up_s = as_rocm(up)?;
        if !gate_s.device_ptr_is_valid() || !up_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_mul: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut gate_ptr = dev_ptr(gate_s)?;
        let mut up_ptr = dev_ptr(up_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_silu_mul",
            grid,
            block,
            &mut [
                arg(&mut gate_ptr),
                arg(&mut up_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let w_s = as_rocm(weight)?;
        if !x_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rms_norm: inputs lack a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        if x_dims.is_empty() {
            return Err(Error::Shape("rms_norm: empty input".into()));
        }
        let row_len = out
            .dims()
            .last()
            .copied()
            .ok_or_else(|| Error::Shape("empty tensor dims".into()))?;
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
        // grim_rms_norm is warp-per-row (32 lanes reduce with shuffles).
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_rms_norm",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut out_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "softmax: input lacks a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        if x_dims.is_empty() {
            return Err(Error::Shape("softmax: empty input".into()));
        }
        let row_len = x_dims
            .last()
            .copied()
            .ok_or_else(|| Error::Shape("empty tensor dims".into()))?;
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut row_len_i = row_len as i32;
        let mut total_i = total as i32;
        // grim_softmax is warp-per-row (32 lanes reduce with shuffles).
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_softmax",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut out_ptr),
                arg(&mut row_len_i),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let w_s = match as_rocm(weight) {
            Ok(s) => s,
            Err(_) => {
                return Err(Error::Backend(
                    "embedding: weight is not RocmStorage".into(),
                ));
            }
        };
        if !w_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "embedding: weight lacks a valid device pointer".into(),
            ));
        }
        let out_dims = out.dims();
        if out_dims.len() < 2 {
            return Err(Error::Shape("embedding: out must be [n, dim]".into()));
        }
        let n = out_dims[0];
        let dim = out_dims[1];
        if n != indices.len() {
            return Err(Error::Shape(format!(
                "embedding: indices len {} != out leading dim {}",
                indices.len(),
                n
            )));
        }

        // materialize() already dequantizes Q8_0 to F32 before returning
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut idx_ptr = upload_device_buffer(self.ordinal, indices)?;
        let mut dim_i = dim as i32;
        let mut total_i = total as i32;
        let (grid, block) = linear_launch(total);
        let stream = self.launch_compute_kernel(
            "grim_embedding",
            grid,
            block,
            &mut [
                arg(&mut w_ptr),
                arg(&mut out_ptr),
                arg(&mut idx_ptr),
                arg(&mut dim_i),
                arg(&mut total_i),
            ],
        )?;
        // The fused kernel reads idx_ptr from the GPU.
        // Free stream-ordered so the release happens after the kernel's reads; this is also graph-capturable (the.
        unsafe {
            let free_stream = stream
                .as_ref()
                .map(|_| self.active_stream())
                .unwrap_or(std::ptr::null_mut());
            let _ = hipFreeAsync(idx_ptr, free_stream);
        }
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn advise(
        &self,
        storage: &dyn BackendStorage,
        advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
        #[cfg(feature = "rocm-profile")]
        println!("[rocprofiler-sdk] Begin marker span: advise");

        let rocm_storage = storage
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("advise: storage is not RocmStorage".into()))?;

        let dev_ptr = match rocm_storage.device_ptr {
            Some(ptr) => ptr as *const c_void,
            None => return Ok(()), // Unallocated or CPU-side: no-op
        };

        // Correctness Gate: Probe XNACK. If disabled, pageable unified memory migrations fail - and there is nothing useful to substitute: the
        // old fallback issued a whole-tensor self-copy on the null stream, which was a no-op for data but a device-wide serialization point.
        if !self.props.xnack_enabled {
            return Ok(());
        }

        let raw_advice = match advice {
            grim_tensor::MemAdvice::ReadMostly => {
                crate::device::handles::HIP_MEM_ADVISE_SET_READ_MOSTLY
            }
            grim_tensor::MemAdvice::PreferredLocation { device_id: _ } => {
                crate::device::handles::HIP_MEM_ADVISE_SET_PREFERRED_LOCATION
            }
            grim_tensor::MemAdvice::AccessedBy { device_id: _ } => {
                crate::device::handles::HIP_MEM_ADVISE_SET_ACCESSED_BY
            }
            grim_tensor::MemAdvice::CoarseGrain => {
                crate::device::handles::HIP_MEM_ADVISE_SET_COARSE_GRAIN
            }
            grim_tensor::MemAdvice::FineGrain => {
                crate::device::handles::HIP_MEM_ADVISE_UNSET_COARSE_GRAIN
            }
            // OS-level hints (madvise) are ignored on the GPU memory space
            _ => return Ok(()),
        };

        unsafe {
            check_hip(
                "hipMemAdvise",
                hipMemAdvise(dev_ptr, rocm_storage.bytes, raw_advice, self.ordinal as i32),
            )?;
        }
        Ok(())
    }
}

impl ElementwiseOps for RocmDevice {
    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "mul_scalar: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let mut s = scalar;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_mul_scalar",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut s), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn add_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "add_scalar: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let mut s = scalar;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_add_scalar",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut s), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn sub_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.add_scalar(x, -scalar, out)
    }

    fn div_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.mul_scalar(x, 1.0 / scalar, out)
    }

    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "sqrt: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_sqrt",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn recip(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "recip: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_recip",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn sub(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let neg_b = self.mul_scalar(b, -1.0, out)?;
        self.add(a, neg_b.0.as_ref(), out)
    }

    fn reduce_sum(&self, x: &dyn BackendStorage) -> Result<f32> {
        let v = x.to_cpu_vec_f32()?;
        if v.is_empty() {
            return Err(Error::Backend("reduce_sum: empty tensor".into()));
        }
        Ok(v.iter().sum())
    }

    fn reduce_max(&self, x: &dyn BackendStorage) -> Result<f32> {
        let v = x.to_cpu_vec_f32()?;
        v.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| Error::Backend("reduce_max: empty tensor".into()))
    }

    fn argmax(&self, x: &dyn BackendStorage) -> Result<u32> {
        let v = x.to_cpu_vec_f32()?;
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| Error::Backend("argmax: empty tensor".into()))
    }
}

impl SamplingOps for RocmDevice {
    fn sample_on_device(
        &self,
        logits: &dyn BackendStorage,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        seed: u64,
    ) -> Result<u32> {
        let vocab = logits.shape().dims().last().copied().unwrap_or(0);
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
        if temperature <= 0.0 {
            return cpu_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| Error::Backend("sample_on_device: empty logits".into()));
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

impl AutogradOps for RocmDevice {
    /// SwiGLU backward: `(df, de) = silu_mul_backward(e, g, dw)`.
    /// `df` = gradient w.r.t. `g` (up), `de` = gradient w.r.t. `e` (gate).
    fn silu_mul_backward(
        &self,
        e: &dyn BackendStorage,
        g: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let e_s = as_rocm(e)?;
        let g_s = as_rocm(g)?;
        let dw_s = as_rocm(dw)?;
        if !e_s.device_ptr_is_valid() || !g_s.device_ptr_is_valid() || !dw_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_mul_backward: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out_shape.elem_count();
        let df_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let de_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut e_ptr = dev_ptr(e_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut dw_ptr = dev_ptr(dw_s)?;
        let mut df_ptr = dev_ptr(&df_storage)?;
        let mut de_ptr = dev_ptr(&de_storage)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_silu_mul_backward",
            grid,
            block,
            &mut [
                arg(&mut e_ptr),
                arg(&mut g_ptr),
                arg(&mut dw_ptr),
                arg(&mut df_ptr),
                arg(&mut de_ptr),
                arg(&mut n),
            ],
        )?;
        Ok((
            Box::new(df_storage),
            Box::new(de_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn rmsnorm_backward(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        out_grad: &dyn BackendStorage,
        eps: f32,
        x_shape: &Shape,
        w_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let x_s = as_rocm(x)?;
        let w_s = as_rocm(weight)?;
        let g_s = as_rocm(out_grad)?;
        if !x_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid() || !g_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rmsnorm_backward: missing device pointer".into(),
            ));
        }
        let row_len = *w_shape.dims().last().unwrap_or(&1);
        let total = x_shape.elem_count();
        let dx_storage =
            RocmStorage::alloc_gpu(x_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let dw_storage =
            RocmStorage::alloc_gpu(w_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut dx_ptr = dev_ptr(&dx_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;

        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_rmsnorm_backward",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut g_ptr),
                arg(&mut dx_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(dx_storage),
            Box::new(dw_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn rope_backward(
        &self,
        out_grad: &dyn BackendStorage,
        cos: &dyn BackendStorage,
        sin: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = as_rocm(out_grad)?;
        let c_s = as_rocm(cos)?;
        let s_s = as_rocm(sin)?;
        if !g_s.device_ptr_is_valid() || !c_s.device_ptr_is_valid() || !s_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rope_backward: missing device pointer".into(),
            ));
        }
        let half_dim = cos.shape().elem_count();
        let head_dim = half_dim * 2;
        let total_tokens = out_shape.elem_count() / head_dim.max(1);
        let dx_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut c_ptr = dev_ptr(c_s)?;
        let mut s_ptr = dev_ptr(s_s)?;
        let mut dx_ptr = dev_ptr(&dx_storage)?;
        let mut half_dim_i = half_dim as i32;
        let mut total_tokens_i = total_tokens as i32;

        let total_pairs = (total_tokens * head_dim) / 2;
        let (grid, block) = linear_launch(total_pairs);
        self.launch_compute_kernel(
            "grim_rope_backward",
            grid,
            block,
            &mut [
                arg(&mut g_ptr),
                arg(&mut c_ptr),
                arg(&mut s_ptr),
                arg(&mut dx_ptr),
                arg(&mut half_dim_i),
                arg(&mut total_tokens_i),
            ],
        )?;
        Ok((
            Box::new(dx_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn softmax_backward(
        &self,
        out_grad: &dyn BackendStorage,
        softmax_out: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = as_rocm(out_grad)?;
        let s_s = as_rocm(softmax_out)?;
        if !g_s.device_ptr_is_valid() || !s_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "softmax_backward: missing device pointer".into(),
            ));
        }
        let row_len = *out_shape.dims().last().unwrap_or(&1);
        let total = out_shape.elem_count();
        let dx_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut s_ptr = dev_ptr(s_s)?;
        let mut dx_ptr = dev_ptr(&dx_storage)?;
        let mut row_len_i = row_len as i32;
        let mut total_i = total as i32;

        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_softmax_backward",
            grid,
            block,
            &mut [
                arg(&mut g_ptr),
                arg(&mut s_ptr),
                arg(&mut dx_ptr),
                arg(&mut row_len_i),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(dx_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// P3 (4th fused backward kernel): scatter-add embedding gradient on device - `dweight[token_ids[t], :] += out_grad[t, :]`.
    /// Token ids are uploaded as a small U32 buffer; dweight is zero-filled first, then atomically.
    fn embedding_backward(
        &self,
        out_grad: &dyn BackendStorage,
        token_ids: &[u32],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g_s = as_rocm(out_grad)?;
        if !g_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "embedding_backward: missing device pointer".into(),
            ));
        }
        let num_tokens = token_ids.len();
        if num_tokens == 0 || hidden_dim == 0 || vocab_size == 0 {
            return Err(Error::Shape(
                "embedding_backward: empty vocab/hidden/tokens".into(),
            ));
        }

        let dw_shape = Shape::new(vec![vocab_size, hidden_dim]);
        let dw_storage =
            RocmStorage::alloc_gpu(&dw_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let ids_shape = Shape::new(vec![num_tokens]);
        let ids_bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        let ids_storage = RocmStorage::copy_from_host_raw_bytes(
            &ids_bytes,
            &ids_shape,
            DType::U32,
            &self.allocator,
            self.ordinal,
        )?;

        let mut g_ptr = dev_ptr(g_s)?;
        let mut ids_ptr = dev_ptr(&ids_storage)?;
        let mut dw_ptr = dev_ptr(&dw_storage)?;
        let mut dw_total_i = (vocab_size * hidden_dim) as i32;
        let mut num_tokens_i = num_tokens as i32;
        let mut hidden_dim_i = hidden_dim as i32;
        let mut vocab_size_i = vocab_size as i32;

        // 1) zero-fill dweight.
        let (grid, block) = linear_launch(vocab_size * hidden_dim);
        self.launch_compute_kernel(
            "grim_zero_f32",
            grid,
            block,
            &mut [arg(&mut dw_ptr), arg(&mut dw_total_i)],
        )?;
        // 2) atomic scatter-add.
        let (grid, block) = linear_launch(num_tokens * hidden_dim);
        self.launch_compute_kernel(
            "grim_embedding_backward",
            grid,
            block,
            &mut [
                arg(&mut g_ptr),
                arg(&mut ids_ptr),
                arg(&mut dw_ptr),
                arg(&mut num_tokens_i),
                arg(&mut hidden_dim_i),
                arg(&mut vocab_size_i),
            ],
        )?;
        Ok((
            Box::new(dw_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }
}

impl RocmDevice {
    /// SPEED-ROC-10: multi-tensor (foreach) fused AdamW — one launch for the
    /// whole parameter list, amortizing per-tensor kernel-launch overhead
    /// (dominant when there are many small tensors, e.g. LoRA or shard splits).
    /// All tensors must be F32 device storages; lengths come from each storage.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_adamw_step_foreach(
        &self,
        params: &[&dyn BackendStorage],
        grads: &[&dyn BackendStorage],
        moments_m: &[&dyn BackendStorage],
        moments_v: &[&dyn BackendStorage],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<Box<dyn ComputeHandle>> {
        if params.len() != grads.len()
            || params.len() != moments_m.len()
            || params.len() != moments_v.len()
        {
            return Err(Error::Backend(
                "fused_adamw_step_foreach: p/g/m/v list length mismatch".into(),
            ));
        }
        if params.is_empty() {
            return Ok(Box::new(RocmHandle::new(Some(self.active_stream()))));
        }

        // Flatten to raw device pointers + element-count prefix sums.
        let mut p_ptrs = Vec::with_capacity(params.len());
        let mut g_ptrs = Vec::with_capacity(grads.len());
        let mut m_ptrs = Vec::with_capacity(moments_m.len());
        let mut v_ptrs = Vec::with_capacity(moments_v.len());
        let mut offsets = Vec::with_capacity(params.len() + 1);
        offsets.push(0i32);
        let mut total: i64 = 0;
        for (i, p) in params.iter().enumerate() {
            let p_s = as_rocm(*p)?;
            let g_s = as_rocm(grads[i])?;
            let m_s = as_rocm(moments_m[i])?;
            let v_s = as_rocm(moments_v[i])?;
            if !p_s.device_ptr_is_valid()
                || !g_s.device_ptr_is_valid()
                || !m_s.device_ptr_is_valid()
                || !v_s.device_ptr_is_valid()
            {
                return Err(Error::Backend(
                    "fused_adamw_step_foreach: tensor {i} lacks a valid device pointer".into(),
                ));
            }
            p_ptrs.push(dev_ptr(p_s)? as *mut f32);
            g_ptrs.push(dev_ptr(g_s)? as *const f32);
            m_ptrs.push(dev_ptr(m_s)? as *mut f32);
            v_ptrs.push(dev_ptr(v_s)? as *mut f32);
            total += p_s.shape.elem_count() as i64;
            offsets.push(total as i32);
        }
        if total > i32::MAX as i64 {
            return Err(Error::Backend(
                "fused_adamw_step_foreach: concatenated parameter count overflows i32".into(),
            ));
        }

        // Upload the pointer/index arrays; freed stream-ordered after launch.
        let p_dev = crate::upload_device_buffer(self.ordinal, &p_ptrs)?;
        let g_dev = crate::upload_device_buffer(self.ordinal, &g_ptrs)?;
        let m_dev = crate::upload_device_buffer(self.ordinal, &m_ptrs)?;
        let v_dev = crate::upload_device_buffer(self.ordinal, &v_ptrs)?;
        let off_dev = crate::upload_device_buffer(self.ordinal, &offsets)?;

        let mut p_list = p_dev;
        let mut g_list = g_dev;
        let mut m_list = m_dev;
        let mut v_list = v_dev;
        let mut offs = off_dev;
        let mut n_tensors = params.len() as i32;
        let mut total_i = total;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut eps = eps;
        let mut weight_decay = weight_decay;
        let mut bc1 = bc1;
        let mut bc2 = bc2;

        // The kernel grid-strides over the concatenated buffer, so a mild
        // oversubscription suffices regardless of the parameter count.
        let (grid, block_dim) = linear_launch(total as usize / 4 + 1);

        let stream = self.launch_compute_kernel(
            "grim_fused_adamw_step_foreach",
            grid,
            block_dim,
            &mut [
                arg(&mut p_list),
                arg(&mut g_list),
                arg(&mut m_list),
                arg(&mut v_list),
                arg(&mut offs),
                arg(&mut n_tensors),
                arg(&mut total_i),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut eps),
                arg(&mut weight_decay),
                arg(&mut bc1),
                arg(&mut bc2),
            ],
        )?;

        // Stream-ordered free of the transient index/pointer arrays.
        if self.active_capture_stream().is_none() {
            unsafe {
                let free_stream = self.active_stream();
                let _ = crate::hipFreeAsync(p_list, free_stream);
                let _ = crate::hipFreeAsync(g_list, free_stream);
                let _ = crate::hipFreeAsync(m_list, free_stream);
                let _ = crate::hipFreeAsync(v_list, free_stream);
                let _ = crate::hipFreeAsync(offs, free_stream);
            }
        }
        Ok(Box::new(RocmHandle::new(Some(stream))))
    }
}

impl OptimizerOps for RocmDevice {
    fn fused_adamw_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        m: &dyn BackendStorage,
        v: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let m_s = as_rocm(m)?;
        let v_s = as_rocm(v)?;
        if !p_s.device_ptr_is_valid()
            || !g_s.device_ptr_is_valid()
            || !m_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_adamw_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut m_ptr = dev_ptr(m_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut eps = eps;
        let mut weight_decay = weight_decay;
        let mut bc1 = bc1;
        let mut bc2 = bc2;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_adamw_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut m_ptr),
                arg(&mut v_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut eps),
                arg(&mut weight_decay),
                arg(&mut bc1),
                arg(&mut bc2),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    fn fused_lion_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        exp_avg: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let exp_s = as_rocm(exp_avg)?;
        if !p_s.device_ptr_is_valid() || !g_s.device_ptr_is_valid() || !exp_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_lion_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut exp_ptr = dev_ptr(exp_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut weight_decay = weight_decay;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_lion_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut exp_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut weight_decay),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    fn fused_madam_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        m: &dyn BackendStorage,
        v: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        gamma: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let m_s = as_rocm(m)?;
        let v_s = as_rocm(v)?;
        if !p_s.device_ptr_is_valid()
            || !g_s.device_ptr_is_valid()
            || !m_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_madam_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut m_ptr = dev_ptr(m_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut eps = eps;
        let mut gamma = gamma;
        let mut weight_decay = weight_decay;
        let mut bc1 = bc1;
        let mut bc2 = bc2;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_madam_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut m_ptr),
                arg(&mut v_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut eps),
                arg(&mut gamma),
                arg(&mut weight_decay),
                arg(&mut bc1),
                arg(&mut bc2),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }
}

impl RocmDevice {
    /// WI 2.4.4-2c — dispatch `grim_decode_gemm_f16` and return the [see: `launch_compute_kernel`, `DecodeGemmConfig::enabled`]
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
        let stride_b = n; // B[K, N]
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
    #[allow(dead_code)]
    pub(crate) fn launch_madam_update_f32(
        &self,
        dx_storage: &RocmStorage,
        weight_storage: &RocmStorage,
        scale_storage: Option<&RocmStorage>,
        m_buffer: &RocmStorage,
        v_buffer: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: i32,
    ) -> Result<*mut c_void> {
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: dX has no device ptr".into()))?;
        let w_ptr = weight_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: weight has no device ptr".into()))?;
        let m_ptr = m_buffer
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: m_buffer has no device ptr".into()))?;
        let v_ptr = v_buffer
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: v_buffer has no device ptr".into()))?;
        let scale_ptr: *const std::ffi::c_void = scale_storage
            .and_then(|s| s.device_ptr)
            .map(|p| p as *const std::ffi::c_void)
            .unwrap_or(std::ptr::null());

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("madam_update: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "madam_update: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;

        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dxptr = dx_ptr;
        let mut wptr = w_ptr;
        let mut sptr = scale_ptr;
        let mut mptr = m_ptr;
        let mut vptr = v_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut lr_f = lr;
        let mut b1 = beta1;
        let mut b2 = beta2;
        let mut ep = eps;
        let mut stp = step;

        self.launch_compute_kernel(
            "grim_madam_update_f32",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dxptr),
                arg(&mut wptr),
                arg(&mut sptr),
                arg(&mut mptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut lr_f),
                arg(&mut b1),
                arg(&mut b2),
                arg(&mut ep),
                arg(&mut stp),
            ],
        )?;
        Ok(std::ptr::null_mut())
    }

    /// Launch the JIT compiled SplitK reduction kernel (WI-D).
    pub(crate) fn launch_split_k_reduction(
        &self,
        partials_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        split_k: u32,
    ) -> Result<*mut c_void> {
        let partials_ptr = partials_storage.device_ptr.ok_or_else(|| {
            Error::Backend("split_k_reduction: partials has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("split_k_reduction: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems = m * n;
        let grid_x = (total_elems.div_ceil(BLOCK_SIZE)) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut p_ptr = partials_ptr;
        let mut o_ptr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut sk = split_k as i32;

        // The reduction entry point must match the partials' element type: the historical f16-only kernel silently corrupted every
        // F32/BF16 split-K GEMM (f32 partials read as _Float16, f16 bits written back into the f32 output buffer).
        let entry = match partials_storage.dtype.arith {
            ArithType::F32 => "grim_split_k_reduction_f32",
            ArithType::BF16 => "grim_split_k_reduction_bf16",
            _ => "grim_split_k_reduction",
        };
        self.launch_compute_kernel(
            entry,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut p_ptr),
                arg(&mut o_ptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut sk),
            ],
        )
    }

    /// JIT-compile or query the cache, then launch the specified kernel on a [see: `entry`, `module_cache`]
    pub(crate) fn launch_compute_kernel(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
    ) -> Result<*mut c_void> {
        self.launch_compute_kernel_with_solution(entry, grid, block, args, None, 0)
    }

    /// JIT compile source or fetch cached binary.
    /// When a `HardwareSpec` is supplied, the cache key incorporates the hardware fingerprint (wavefront/lds/cu/mp/threads) via `JitCacheKey::from_spec`,.
    pub fn jit_compile_or_cache(
        &self,
        source: &str,
        entry: &str,
        spec: Option<&crate::device::hardware_spec::HardwareSpec>,
    ) -> Result<(std::path::PathBuf, String)> {
        if std::env::var("GRIM_ALLOC_TRACE").is_ok() {
            eprintln!("[jit-trace] compiling entry={}", entry);
        }
        let hash = seahash::hash(source.as_bytes());
        let cache_key = if let Some(spec) = spec {
            crate::kernels::jit_cache::JitCacheKey::from_spec(entry, &self.gpu_target, spec, hash)
                .to_key_string()
        } else {
            format!("grim_{}_{}_{:016x}", entry, self.gpu_target, hash)
        };

        if let Some((cached_path, cached_lowered)) = self.hsaco_cache.get_cached_kernel(&cache_key)
        {
            if std::env::var_os("GRIM_RING_DIAG").is_some() {
                eprintln!(
                    "[prov] {} entry={} DISK-HIT key={}",
                    self.ordinal, entry, cache_key
                );
            }
            Ok((cached_path, cached_lowered))
        } else {
            if std::env::var_os("GRIM_RING_DIAG").is_some() {
                eprintln!(
                    "[prov] {} entry={} FRESH-COMPILE key={}",
                    self.ordinal, entry, cache_key
                );
            }
            let (code, lowered) = jit_compile_hsaco(source, entry, &self.gpu_target)?;
            let p = self
                .hsaco_cache
                .cache_kernel(&cache_key, source, &code, &lowered)?;
            Ok((p, lowered.to_string()))
        }
    }

    /// Benchmark kernel execution time in milliseconds using HIP events.
    /// Loads the module, resolves the entry, launches once on the device stream bracketed by start/stop.
    pub fn time_kernel_ms(
        &self,
        hsaco: &std::path::Path,
        lowered: &str,
        dims: crate::kernels::tile_picker::ShapeDims,
        cand: &crate::kernels::tile_picker::TileConfig,
    ) -> f64 {
        use crate::device::handles::{
            hipEventCreate, hipEventDestroy, hipEventElapsedTime, hipEventRecord,
            hipEventSynchronize, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoad,
            hipModuleUnload,
        };
        // P1-3: module load, events and the launch all bind to the calling thread's current device -
        // pin to the owning ordinal so autotune timing runs on the device it is tuning for.
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut start_event: *mut c_void = std::ptr::null_mut();
        let mut stop_event: *mut c_void = std::ptr::null_mut();

        unsafe {
            if hipEventCreate(&mut start_event) != hipSuccess {
                return 0.5;
            }
            if hipEventCreate(&mut stop_event) != hipSuccess {
                let _ = hipEventDestroy(start_event);
                return 0.5;
            }

            let _ = hipEventRecord(start_event, std::ptr::null_mut());

            let grid = HipDim3::new(
                dims.m.div_ceil(cand.grid_stride_m),
                dims.n.div_ceil(cand.grid_stride_n),
                1,
            );
            let block = HipDim3::new(cand.threads, 1, 1);

            let path_c = match std::ffi::CString::new(hsaco.to_str().unwrap_or("")) {
                Ok(c) => c,
                Err(_) => {
                    let _ = hipEventDestroy(start_event);
                    let _ = hipEventDestroy(stop_event);
                    return 0.5;
                }
            };
            let entry_c = match std::ffi::CString::new(lowered) {
                Ok(c) => c,
                Err(_) => {
                    let _ = hipEventDestroy(start_event);
                    let _ = hipEventDestroy(stop_event);
                    return 0.5;
                }
            };

            let mut module: *mut c_void = std::ptr::null_mut();
            if hipModuleLoad(&mut module, path_c.as_ptr()) == hipSuccess {
                let mut func: *mut c_void = std::ptr::null_mut();
                if hipModuleGetFunction(&mut func, module, entry_c.as_ptr()) == hipSuccess {
                    let mut dummy_args: [*mut c_void; 0] = [];
                    let _ = hipModuleLaunchKernel(
                        func,
                        grid.x,
                        grid.y,
                        grid.z,
                        block.x,
                        block.y,
                        block.z,
                        0,
                        std::ptr::null_mut(),
                        dummy_args.as_mut_ptr(),
                        std::ptr::null_mut(),
                    );
                }
                let _ = hipModuleUnload(module);
            }

            let _ = hipEventRecord(stop_event, std::ptr::null_mut());
            let _ = hipEventSynchronize(stop_event);

            let mut elapsed_ms: f32 = 0.0;
            let status = hipEventElapsedTime(&mut elapsed_ms, start_event, stop_event);

            let _ = hipEventDestroy(start_event);
            let _ = hipEventDestroy(stop_event);

            if status == hipSuccess && elapsed_ms > 0.0 {
                elapsed_ms as f64
            } else {
                0.5
            }
        }
    }

    /// Store empirically discovered winning tile configuration into the autotuner cache.
    /// `winner_ms` is the measured GPU time of the winning candidate; persisted as `cycles_per_invocation` (ns-scale u64).
    pub(crate) fn intern_str(&self, s: &str) -> &'static str {
        if let Ok(mut set) = self.str_interner.lock() {
            if let Some(existing) = set.get(s) {
                return existing;
            }
            let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
            set.insert(leaked);
            leaked
        } else {
            // Poisoned interner: fall back to a one-shot leak (correctness
            // over boundedness).
            Box::leak(s.to_string().into_boxed_str())
        }
    }

    pub fn store_tune_cache(
        &self,
        entry: &str,
        _spec: &crate::device::hardware_spec::HardwareSpec,
        dims: crate::kernels::tile_picker::ShapeDims,
        winner: &crate::kernels::tile_picker::TileConfig,
        winner_ms: f64,
    ) {
        let mut autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
        // &'static str keys via the interner — one leak per unique
        // (entry, arch) pair, not per call.
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let entry_leak: &'static str = self.intern_str(entry);
        let key = crate::autotune::KernelKey {
            kernel: entry_leak,
            gpu_arch: arch_leak,
            m: dims.m as usize,
            n: dims.n as usize,
            k: dims.k as usize,
        };
        let config = crate::autotune::AutotuneConfig {
            block_dim: winner.threads,
            tile_kv: winner.block_k,
            grid_stride: winner.grid_stride_m,
            cycles_per_invocation: (winner_ms * 1e6) as u64,
            spec_gamma: 4,
            spec_acceptance_threshold: 0.6,
            spec_alpha: 0.0,
            split_k: winner.split_k,
        };
        let _ = autotuner.record(key, config);
    }

    /// SPEED-ROC-3: read-only lookup of the persisted GEMM autotune table for
    /// the canonical workload entries (`grim_decode_gemm`, `grim_prefill_gemm`,
    /// `grim_lm_head`). Unlike `get_or_tune_tiles`, this NEVER triggers the
    /// FCP search — on a miss the caller falls back to the static heuristic
    /// table. Only fields the rocBLAS dispatch consumes (split_k) are honored;
    /// older tables without a recorded `split_k` (0) are ignored.
    pub(crate) fn lookup_tuned_gemm_split_k(
        &self,
        entry: &'static str,
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<u32> {
        let autotuner = self.autotuner.lock().ok()?;
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let key = crate::autotune::KernelKey {
            kernel: entry,
            gpu_arch: arch_leak,
            m,
            n,
            k,
        };
        let split_k = autotuner.lookup(key)?.split_k;
        (split_k > 1).then_some(split_k)
    }

    /// Persist the in-memory autotune cache to a JSON file at `path`.
    pub fn save_autotune_cache(&self, path: &std::path::Path) -> Result<()> {
        let autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
        autotuner.save_to_file(path)
    }

    /// Return a fresh HardwareSpec snapshot describing this device.
    pub fn hardware_spec(&self) -> crate::device::hardware_spec::HardwareSpec {
        crate::device::hardware_spec::HardwareSpec::from(self)
    }

    /// Read-through tile-cache lookup. On a hit, maps the stored `AutotuneConfig` back to a `TileConfig`.
    pub fn get_or_tune_tiles(
        &self,
        entry: &str,
        spec: &crate::device::hardware_spec::HardwareSpec,
        dims: crate::kernels::tile_picker::ShapeDims,
        shape_class: crate::autotune::ShapeClass,
    ) -> crate::kernels::tile_picker::TileConfig {
        // Same interning as store_tune_cache — one leak per unique (entry, arch).
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let entry_leak: &'static str = self.intern_str(entry);
        let key = crate::autotune::KernelKey {
            kernel: entry_leak,
            gpu_arch: arch_leak,
            m: dims.m as usize,
            n: dims.n as usize,
            k: dims.k as usize,
        };

        // 1. Hot path: in-memory table hit.
        {
            let autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(cfg) = autotuner.lookup(key) {
                return crate::kernels::tile_picker::TileConfig {
                    block_m: 0,
                    block_n: 0,
                    block_k: cfg.tile_kv,
                    split_k: 1,
                    grid_stride_m: cfg.grid_stride,
                    grid_stride_n: cfg.grid_stride,
                    lds_double_buffer: 64 * 1024
                        >= 2 * (2
                            * (cfg.tile_kv * (spec.wavefront_size.max(16))
                                + cfg.tile_kv * (spec.wavefront_size.max(16))
                                + (spec.wavefront_size.max(16)) * (spec.wavefront_size.max(16)))),
                    use_wmma: spec.gcn_arch.starts_with("gfx11")
                        || spec.gcn_arch.starts_with("gfx12"),
                    use_mfma: spec.gcn_arch.starts_with("gfx12")
                        || spec.gcn_arch.starts_with("gfx9"),
                    threads: cfg.block_dim,
                }
                .with_block_geometry(spec, shape_class);
            }
        }

        // 2. Cold path: empirical FCP search. Self-persists, so the next call hits step 1.
        crate::kernels::tile_picker::fcp_fallback_tile_search(self, spec, entry, dims, shape_class)
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

        let k = a_dims[a_dims.len() - 1];
        let m = a.shape().elem_count() / k;

        let n = b_dims[b_dims.len() - 1];
        let k2 = b.shape().elem_count() / n;

        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }

        if out_shape.elem_count() != m * n {
            return Err(Error::Shape(format!(
                "expected out elem_count {}, got {:?}",
                m * n,
                out_shape.dims()
            )));
        }

        // P1-3 context discipline: rocBLAS executes against the CALLING THREAD's current HIP device, not the handle's construction device.
        // `try_new` is context-neutral (restores the caller's device on return), so on a multi-GPU box the.
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        // Allocate output GPU storage with the actual input precision
        let dtype_out = DType {
            arith: a_storage.dtype.arith,
            storage: DTypeStorage::Native,
        };
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_out.clone(), &self.allocator, self.ordinal)?;

        // WI-SB6 production routing: GRIM_SCYTHE_RING=1 rides F32 GEMMs (the dense-layer op of every decode step) through the ScytheRing persistent dispatch wave instead of the rocBLAS direct path.
        // Benchmark-gated, never default - see device::scythe_route.
        if dtype_out.arith == ArithType::F32 && crate::device::scythe_route::ring_routing_enabled()
        {
            let stream = crate::device::scythe_route::route_gemm(
                self,
                self.active_stream(),
                a_storage,
                b_storage,
                &out_storage,
                m,
                n,
                k,
            )?;
            self.launch_counter.fetch_add(1, Ordering::SeqCst);
            let compute_handle = Box::new(RocmHandle::new(Some(stream)));
            return Ok((Box::new(out_storage), compute_handle));
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

                rocblas_gemm_strided_batched_ex(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k_part as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    b_type,
                    n as RocblasInt,
                    (k_part * n) as i64,
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
            self.launch_counter.fetch_add(1, Ordering::SeqCst);

            // Sum up the partials along the batch dimension using the hand-written reduction kernel
            let stream = self.launch_split_k_reduction(
                &partials_storage,
                &out_storage,
                m,
                n,
                split_k_effective,
            )?;
            let compute_handle = Box::new(RocmHandle::new(Some(stream)));
            return Ok((Box::new(out_storage), compute_handle));
        }
        #[cfg(feature = "rocm-profile")]
        println!(
            "[RocmDevice] GEMM Dispatch: Shape ({}, {}, {}) resolved to autotune tile config {:?} on Wavefront {:?}, solution_index={}",
            m, n, k, tile_config, self.props.wavefront_size, solution_index
        );

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
                        &out_storage,
                        m,
                        n,
                        k,
                    ) {
                        Ok(_) => {
                            self.launch_counter.fetch_add(1, Ordering::SeqCst);
                            let compute_handle =
                                Box::new(RocmHandle::new(Some(self.active_stream())));
                            return Ok((Box::new(out_storage), compute_handle));
                        }
                        Err(_) => {
                            // fall through to the direct launch below
                        }
                    }
                }
                // WI 2.4.4-2(a) — thread the *real* enqueued stream into the [see: `launch_compute_kernel`, `hipModuleLaunchKernel`]
                let stream =
                    self.launch_decode_gemm_f16(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
        }

        // ─── WI-G — WMMA GEMM dispatch (opt-in, F16-only) ─────
        {
            if self.should_use_wmma_path(None, dtype_out.arith) {
                let stream = self.launch_wmma_gemm(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
        }

        // Get rocBLAS handle and execute sgemm. If handle is null (due to memory error fallback),
        // execute using WMMA HIP GEMM kernel directly.
        let handle = match self.get_rocblas_handle() {
            Ok(h) if !h.0.is_null() => h,
            _ => {
                let stream = self.launch_wmma_gemm(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
        };

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
                rocblas_gemm_ex(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    alpha_ptr,
                    b_ptr_void,
                    b_type,
                    n as RocblasInt,
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
                    // Wire `lookup_solution_index` to `algo` so rocBLAS actually [see: `select_gemm_algo(0)`, `standard`]
                    select_gemm_algo(solution_index),
                    solution_index as RocblasInt,
                    ROCBLAS_GEMM_FLAGS_NONE,
                )
            } else {
                rocblas_sgemm(
                    handle,
                    RocblasOperation::None,
                    RocblasOperation::None,
                    n as RocblasInt,
                    m as RocblasInt,
                    k as RocblasInt,
                    &alpha,
                    b_ptr_void as *const f32,
                    n as RocblasInt,
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
                let stream = self.launch_wmma_gemm(a_storage, b_storage, &out_storage, m, n, k)?;
                let compute_handle = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_storage), compute_handle));
            }
            self.launch_counter.fetch_add(1, Ordering::SeqCst);
        };

        let compute_handle = Box::new(RocmHandle::new(Some(self.active_stream())));
        Ok((Box::new(out_storage), compute_handle))
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

    /// WI-F1 - Fused QKV projection GEMM. `qkv_weight` must be the load-time concatenation of the per-layer Q/K/V projection weights along the
    /// output dim - row-major `[hidden, q_dim + k_dim + v_dim]`, built once at model load via [`crate::fusion::concat_qkv_weights`] (never per forward pass).
    pub fn fused_qkv_proj(
        &self,
        x: &dyn BackendStorage,
        qkv_weight: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_dims = x.shape().dims();
        let w_dims = qkv_weight.shape().dims();
        if x_dims.len() < 2 || w_dims.len() < 2 {
            return Err(Error::Shape(
                "fused_qkv_proj expects rank >= 2 inputs".into(),
            ));
        }
        let hidden = x_dims[x_dims.len() - 1];
        let qkv_dim = w_dims[w_dims.len() - 1];
        let k2 = qkv_weight.shape().elem_count() / qkv_dim;
        if hidden != k2 {
            return Err(Error::ShapeMismatch {
                expected: x_dims.to_vec(),
                got: w_dims.to_vec(),
            });
        }
        let tokens = x.shape().elem_count() / hidden;
        if out_shape.elem_count() != tokens * qkv_dim {
            return Err(Error::Shape(format!(
                "expected out elem_count {}, got {:?}",
                tokens * qkv_dim,
                out_shape.dims()
            )));
        }
        self.matmul_op(x, qkv_weight, out_shape, crate::autotune::GemmOp::Attention)
    }

    /// WI-F2 - Fused attention output projection.
    /// Runs the same fused QKV attention kernel (`grim_qkv_attention`) with the O-projection applied in the kernel.
    pub fn fused_attn_o_proj(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        o_proj: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let q_dims = q.shape().dims();
        let o_dims = out_shape.dims();
        if q_dims.len() != 3 || o_dims.len() != 2 {
            return Err(Error::Shape(
                "fused_attn_o_proj expects q [seq, heads, head_dim] and out [seq, o_dim]".into(),
            ));
        }
        let seq_len = q_dims[0];
        let num_heads = q_dims[1];
        let head_dim = q_dims[2];
        let o_dim = o_dims[1];
        if o_dims[0] != seq_len {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: out rows {} must equal seq_len {seq_len}",
                o_dims[0]
            )));
        }
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || o_dim == 0 {
            return Err(Error::Shape(
                "fused_attn_o_proj: zero-sized heads / head_dim / o_dim".into(),
            ));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )));
        }
        if head_dim > 256 {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj supports head_dim <= 256 (got {head_dim})"
            )));
        }
        let o_s = as_rocm(o_proj)?;
        if o_proj.shape().elem_count() != num_heads * head_dim * o_dim {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: o_proj must be [num_heads*head_dim, o_dim] = {} elems (got {})",
                num_heads * head_dim * o_dim,
                o_proj.shape().elem_count()
            )));
        }
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !o_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_attn_o_proj: inputs lack a valid device pointer".into(),
            ));
        }

        let config = QkvAttentionFusionConfig {
            enabled: true,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len: seq_len,
            wavefront_size: self.props.wavefront_size as u32,
            quant_mode: QuantMode::Fp32,
        };
        let launch = config.hip_launch_params();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;

        // atomicAdd accumulation across heads requires a zeroed output;
        // async memset on the active stream keeps it stream-ordered.
        let res = unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                storage.bytes,
                self.active_stream(),
            )
        };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "fused_attn_o_proj: hipMemsetAsync failed with status {res}"
            )));
        }

        let q_ptr = dev_ptr(q_s)?;
        let k_ptr = dev_ptr(k_s)?;
        let v_ptr = dev_ptr(v_s)?;
        let o_proj_ptr = dev_ptr(o_s)?;

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut max_ptr: u64 = 0;
        let mut sum_ptr: u64 = 0;
        let mut nh = num_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sl = seq_len as i32;
        let mut ksl = kv_seq_len as i32;
        let mut co = cache_offset as i32;
        let mut isd: f32 = 1.0 / (head_dim as f32).sqrt();
        let mut wlo: i32 = 0;
        let mut oproj_ptr = o_proj_ptr;
        let mut odim = o_dim as i32;
        let mut fuseo: i32 = 1;
        let mut alibi_ptr: u64 = 0;
        let mut has_alibi: i32 = 0;

        let stream = self.launch_compute_kernel(
            "grim_qkv_attention",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut max_ptr),
                arg(&mut sum_ptr),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut wlo),
                arg(&mut oproj_ptr),
                arg(&mut odim),
                arg(&mut fuseo),
                arg(&mut alibi_ptr),
                arg(&mut has_alibi),
            ],
        )?;
        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd, wlo,
            oproj_ptr, odim, fuseo, alibi_ptr, has_alibi,
        );
        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    /// WI-M2/M3: stamp (self_dev, ctx_dev) for the drift gates and print the launch trace - called while this
    /// device's P1-3 guard is held, so the recorded `ctx_dev` is the context the kernel actually launches under.
    fn stamp_launch_post_pin(&self, trace_on: bool, entry: &str, grid: HipDim3) {
        if !(trace_on || cfg!(test)) {
            return;
        }
        let mut cur_dev: i32 = -1;
        unsafe {
            crate::device::handles::hipGetDevice(&mut cur_dev);
        }
        #[cfg(test)]
        crate::device::util::stamp_launch_context(self.ordinal as i32, cur_dev);
        if trace_on {
            eprintln!(
                "[launch-trace] self_dev={} ctx_dev={} {} grid=({},{},{})",
                self.ordinal, cur_dev, entry, grid.x, grid.y, grid.z
            );
        }
    }

    pub(crate) fn launch_compute_kernel_with_solution(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
        solution_index: Option<i32>,
        shared_mem_bytes: usize,
    ) -> Result<*mut c_void> {
        // Fast path: a previously resolved hipFunction for this (entry, grid-shape, solution_index) launches directly - no source rebuild, no seahash, no CString, no module-cache walk.
        // Same solution_index is required because different indices map to different on-disk hsaco files (cache_key includes.
        let trace_on = std::env::var("GRIM_ALLOC_TRACE").is_ok();
        if std::env::var("GRIM_ALLOC_TRACE").is_ok() {
            eprintln!("[launch-done] {}", entry);
        }
        // SPEED-ROC-12: intern once per unique entry (one leaked &str) — the
        // old `entry.to_string()` heap-allocated on every launch lookup.
        let fast_key = (self.intern_str(entry), grid.x, grid.y, solution_index);
        let cached_func: Option<*mut c_void> = self
            .resolved_kernel_cache
            .lock()
            .ok()
            .and_then(|c| c.get(&fast_key).copied());
        if let Some(func) = cached_func {
            if !func.is_null() {
                // P1-3 discipline: the launching thread's HIP context may be parked on another device (profiler probes, multi-device loaders).
                // Pin THIS device or the kernel executes against foreign pointers - observed as GPU page.
                let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
                self.stamp_launch_post_pin(trace_on, entry, grid);
                let stream = self.active_stream();
                let args_ptr = args.as_mut_ptr();
                check_hip("hipModuleLaunchKernel (cached)", unsafe {
                    hipModuleLaunchKernel(
                        func,
                        grid.x,
                        grid.y,
                        grid.z,
                        block.x,
                        block.y,
                        block.z,
                        shared_mem_bytes as u32,
                        stream,
                        args_ptr,
                        std::ptr::null_mut(),
                    )
                })?;
                drop(_dev_guard);
                return Ok(stream);
            }
        }

        // Build the kernel source. Under `jit-hw-adaptive`, inject hardware-specific #defines (wavefront/LDS/CU +
        // tile geometry) via `compute_kernel_source_with_spec` and route the compile through the fingerprinted `jit_compile_or_cache`.
        #[cfg(feature = "jit-hw-adaptive")]
        let (path, lowered_name, cache_key) = {
            let spec = self.hardware_spec();
            // `launch_compute_kernel` is a generic launcher (no GEMM M/N/K in its signature), so infer a coarse (m, n) from the grid dims; the per-op TLOLog tagging is handled at the `matmul_op` layer, not here.
            // K is unknown to the generic launcher; use a conservative default - split-K is derived.
            let (m_val, n_val) = if grid.y > 1 { (grid.x, grid.y) } else { (1, 1) };
            let shape_class = crate::autotune::ShapeClass::from_m(m_val as usize);
            let dims = crate::kernels::tile_picker::ShapeDims::new(m_val, n_val, 64);
            let kernel_source = crate::kernels::source_asm::compute_kernel_source_with_spec(
                &spec,
                entry,
                shape_class,
                dims,
                0,
                1,
                None,
            );
            let (p, lowered) = self.jit_compile_or_cache(&kernel_source, entry, Some(&spec))?;
            let mut key = format!(
                "grim_{}_{}_{:016x}",
                entry,
                self.gpu_target,
                seahash::hash(kernel_source.as_bytes())
            );
            if let Some(sol) = solution_index {
                key = format!("{}_sol{}", key, sol);
            }
            (p, lowered, key)
        };

        #[cfg(not(feature = "jit-hw-adaptive"))]
        let (path, lowered_name, cache_key) = {
            let kernel_source = crate::kernels::source_asm::compute_kernel_source();
            let hash = seahash::hash(kernel_source.as_bytes());
            let base_key = format!("grim_{}_{}_{:016x}", entry, self.gpu_target, hash);
            let cache_key = if let Some(sol) = solution_index {
                format!("{}_sol{}", base_key, sol)
            } else {
                base_key
            };
            let (path, lowered_name) = if let Some((cached_path, cached_lowered)) =
                self.hsaco_cache.get_cached_kernel(&cache_key)
            {
                (cached_path, cached_lowered)
            } else {
                let (code, lowered) = jit_compile_hsaco(&kernel_source, entry, &self.gpu_target)?;
                let p =
                    self.hsaco_cache
                        .cache_kernel(&cache_key, &kernel_source, &code, &lowered)?;
                (p, lowered)
            };
            (path, lowered_name, cache_key)
        };

        let path_c = std::ffi::CString::new(path.to_str().unwrap_or(""))
            .map_err(|e| Error::Backend(format!("hsaco path CString: {}", e)))?;
        let entry_c = std::ffi::CString::new(lowered_name.as_str())
            .map_err(|e| Error::Backend(format!("entry CString: {}", e)))?;

        // Load the HIP module once per unique kernel; reuse the cached module + Pin the current device to self.ordinal before loading: the JIT pipeline queries CapabilityProfiler which sweeps every device and can leave the thread on a foreign ordinal.
        // Loading a gfx1201 hsaco on gfx1200 yields HIP error 209 (no binary for device).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        self.stamp_launch_post_pin(trace_on, entry, grid);
        let mut module_cache = self.module_cache.lock().unwrap_or_else(|e| e.into_inner());
        let (_module, func) = if let Some(cached) = module_cache.get(&cache_key) {
            let (m, f) = *cached;
            if let Ok(mut fast) = self.resolved_kernel_cache.lock() {
                fast.insert(fast_key, f);
            }
            (m, f)
        } else {
            let mut module: *mut c_void = std::ptr::null_mut();
            let load_res = unsafe { hipModuleLoad(&mut module, path_c.as_ptr()) };
            if load_res != hipSuccess {
                return Err(Error::Backend(format!(
                    "hipModuleLoad failed: {load_res} (entry={entry}, path={}, gpu_target={})",
                    path.display(),
                    self.gpu_target
                )));
            }
            let mut func: *mut c_void = std::ptr::null_mut();
            let res = unsafe { hipModuleGetFunction(&mut func, module, entry_c.as_ptr()) };
            if res != hipSuccess {
                unsafe {
                    hipModuleUnload(module);
                }
                return Err(Error::Backend(format!(
                    "hipModuleGetFunction failed: {}",
                    res
                )));
            }
            self.module_load_count.fetch_add(1, Ordering::SeqCst);
            module_cache.insert(cache_key, (module, func));
            if let Ok(mut fast) = self.resolved_kernel_cache.lock() {
                fast.insert(fast_key, func);
            }
            (module, func)
        };
        drop(module_cache);

        let stream = self.active_stream();

        let args_ptr = args.as_mut_ptr();
        check_hip("hipModuleLaunchKernel", unsafe {
            hipModuleLaunchKernel(
                func,
                grid.x,
                grid.y,
                grid.z,
                block.x,
                block.y,
                block.z,
                shared_mem_bytes as u32,
                stream,
                args_ptr,
                std::ptr::null_mut(),
            )
        })?;
        self.launch_counter.fetch_add(1, Ordering::SeqCst);
        Ok(stream)
    }

    /// Dispatch a fused RMSNorm + MatMul operation onto the GPU.
    pub fn rmsnorm_matmul(
        &self,
        x: &dyn BackendStorage,
        w_norm: &dyn BackendStorage,
        weight_mat: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let w_norm_s = as_rocm(w_norm)?;
        let w_mat_s = as_rocm(weight_mat)?;
        if !x_s.device_ptr_is_valid()
            || !w_norm_s.device_ptr_is_valid()
            || !w_mat_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "rmsnorm_matmul: inputs lack a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        let w_mat_dims = weight_mat.shape().dims();
        if x_dims.len() < 2 || w_mat_dims.len() < 2 {
            return Err(Error::Shape(
                "rmsnorm_matmul expects rank >= 2 inputs".into(),
            ));
        }
        let k = x_dims[x_dims.len() - 1];
        let m = x.shape().elem_count() / k;
        let n = w_mat_dims[w_mat_dims.len() - 1];
        let k2 = weight_mat.shape().elem_count() / n;
        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: x_dims.to_vec(),
                got: w_mat_dims.to_vec(),
            });
        }
        if out_shape.elem_count() != m * n {
            return Err(Error::Shape(format!(
                "expected out elem_count {}, got {:?}",
                m * n,
                out_shape.dims()
            )));
        }

        let config = RmsNormMatMulFusionConfig {
            hidden_size: k,
            intermediate_size: n,
            wavefront_size: self.props.wavefront_size as u32,
            lds_size: 65536,
        };
        let launch = config.hip_launch_params();

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_norm_ptr = dev_ptr(w_norm_s)?;
        let mut w_mat_ptr = dev_ptr(w_mat_s)?;
        let mut m_i = m as i32;
        let mut n_i = n as i32;
        let mut k_i = k as i32;
        let mut eps_f = eps;

        self.launch_compute_kernel(
            "grim_rmsnorm_matmul",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_norm_ptr),
                arg(&mut w_mat_ptr),
                arg(&mut out_ptr),
                arg(&mut m_i),
                arg(&mut n_i),
                arg(&mut k_i),
                arg(&mut eps_f),
            ],
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// High-level fused RMSNorm + MXFP4 GEMM (e.g. for MLP gate/up/down projections).
    pub fn fused_rmsnorm_mxfp4_gemm(
        &self,
        x: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        m: usize,
        n: usize,
        k: usize,
        eps: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let gamma_s = as_rocm(gamma)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;

        let out_shape = Shape::new(vec![m, n]);
        let out_storage =
            RocmStorage::alloc_gpu(&out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        self.launch_fused_rmsnorm_mxfp4_gemm(
            x_s,
            gamma_s,
            w_codes_s,
            w_exps_s,
            &out_storage,
            m,
            n,
            k,
            eps,
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// High-level fused RMSNorm + MXFP4 GEMM + RoPE + direct KV cache scatter.
    pub fn fused_rmsnorm_mxfp4_gemm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
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
    ) -> Result<Box<dyn ComputeHandle>> {
        let x_s = as_rocm(x)?;
        let gamma_s = as_rocm(gamma)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;

        let q_out_s = match q_out {
            Some(q) => Some(as_rocm(q)?),
            None => None,
        };
        let k_cache_s = match k_cache {
            Some(k) => Some(as_rocm(k)?),
            None => None,
        };
        let v_cache_s = match v_cache {
            Some(v) => Some(as_rocm(v)?),
            None => None,
        };
        let out_all_s = match out_all {
            Some(a) => Some(as_rocm(a)?),
            None => None,
        };
        let positions_s = match positions {
            Some(p) => Some(as_rocm(p)?),
            None => None,
        };
        let inv_freq_s = match inv_freq {
            Some(f) => Some(as_rocm(f)?),
            None => None,
        };

        self.launch_fused_rmsnorm_mxfp4_gemm_rope_kv(
            x_s,
            gamma_s,
            w_codes_s,
            w_exps_s,
            q_out_s,
            k_cache_s,
            v_cache_s,
            out_all_s,
            positions_s,
            m,
            k,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            inv_freq_s,
            mscale,
            eps,
            max_seq_len,
        )?;

        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// LFM2-style fused QKV projection: MXFP4 GEMM (C = x @ W_qkv) followed by per-head QK-Norm + RoPE (YaRN-aware).
    /// Mirrors `fused_rmsnorm_mxfp4_gemm_rope_kv` but applies the normalization *after* the projection (QK-norm) instead of before it, matching.
    pub fn fused_mxfp4_gemm_qk_norm_rope_kv(
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
    ) -> Result<Box<dyn ComputeHandle>> {
        let x_s = as_rocm(x)?;
        let gamma_q_s = as_rocm(gamma_q)?;
        let gamma_k_s = as_rocm(gamma_k)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;
        let q_out_s = q_out.map(|q| as_rocm(q)).transpose()?;
        let k_cache_s = k_cache.map(|k| as_rocm(k)).transpose()?;
        let v_cache_s = v_cache.map(|v| as_rocm(v)).transpose()?;
        let out_all_s = out_all.map(|a| as_rocm(a)).transpose()?;
        let positions_s = positions.map(|p| as_rocm(p)).transpose()?;
        let inv_freq_s = inv_freq.map(|f| as_rocm(f)).transpose()?;

        self.launch_fused_mxfp4_gemm_qk_norm_rope_kv(
            x_s,
            gamma_q_s,
            gamma_k_s,
            w_codes_s,
            w_exps_s,
            q_out_s,
            k_cache_s,
            v_cache_s,
            out_all_s,
            positions_s,
            m,
            k,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            inv_freq_s,
            mscale,
            eps,
            max_seq_len,
        )?;

        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// Fused Add + RMSNorm kernel.
    /// Computes `y = x + residual` and `norm_out = RMSNorm(y, weight, eps)` in a single HIP kernel pass.
    pub fn fused_add_rms_norm(
        &self,
        x: &dyn BackendStorage,
        residual: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let x_s = as_rocm(x)?;
        let res_s = as_rocm(residual)?;
        let w_s = as_rocm(weight)?;
        if !x_s.device_ptr_is_valid() || !res_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_add_rms_norm: inputs lack a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        if x_dims.is_empty() {
            return Err(Error::Shape("fused_add_rms_norm: empty input".into()));
        }
        let row_len = x_dims
            .last()
            .copied()
            .ok_or_else(|| Error::Shape("empty tensor dims".into()))?;
        let total = out_shape.elem_count();
        let y_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let norm_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut res_ptr = dev_ptr(res_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut y_out_ptr = dev_ptr(&y_storage)?;
        let mut norm_out_ptr = dev_ptr(&norm_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
        // grim_add_rms_norm is warp-per-row (32 lanes reduce with shuffles).
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_add_rms_norm",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut res_ptr),
                arg(&mut w_ptr),
                arg(&mut y_out_ptr),
                arg(&mut norm_out_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(y_storage),
            Box::new(norm_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch the Design-A on-device linear cross-entropy forward pass.
    pub fn fused_linear_cross_entropy_forward(
        &self,
        hidden: &dyn BackendStorage,
        lm_head: &dyn BackendStorage,
        targets: &dyn BackendStorage,
        v_tile_size: i32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let h = as_rocm(hidden)?;
        let w = as_rocm(lm_head)?;
        let t = as_rocm(targets)?;
        if !h.device_ptr_is_valid() || !w.device_ptr_is_valid() || !t.device_ptr_is_valid() {
            return Err(Error::Backend(
                "fused_linear_ce: invalid input pointer".into(),
            ));
        }
        let hd = hidden.shape().dims();
        let wd = lm_head.shape().dims();
        let td = targets.shape().dims();
        if hd.len() != 2 || wd.len() != 2 || td.len() != 1 || td[0] != hd[0] || wd[1] != hd[1] {
            return Err(Error::Shape(
                "fused_linear_ce: incompatible input shapes".into(),
            ));
        }
        if v_tile_size <= 0 {
            return Err(Error::Backend(
                "fused_linear_ce: v_tile_size must be positive".into(),
            ));
        }
        let batch = hd[0];
        let loss = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let lse = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mut hp = dev_ptr(h)?;
        let mut wp = dev_ptr(w)?;
        let mut tp = dev_ptr(t)?;
        let mut lp = dev_ptr(&loss)?;
        let mut ep = dev_ptr(&lse)?;
        let mut k = hd[1] as i32;
        let mut v = wd[0] as i32;
        let mut tile = v_tile_size;
        let mut b = batch as i32;
        let block = crate::HipDim3 { x: 256, y: 1, z: 1 };
        let grid = crate::HipDim3 {
            x: batch as u32,
            y: 1,
            z: 1,
        };
        self.launch_compute_kernel(
            "grim_fused_linear_ce_forward",
            grid,
            block,
            &mut [
                arg(&mut hp),
                arg(&mut wp),
                arg(&mut tp),
                arg(&mut lp),
                arg(&mut ep),
                arg(&mut k),
                arg(&mut v),
                arg(&mut tile),
                arg(&mut b),
            ],
        )?;
        Ok((
            Box::new(loss),
            Box::new(lse),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch the Design-A on-device linear cross-entropy backward pass.
    pub fn fused_linear_cross_entropy_backward(
        &self,
        hidden: &dyn BackendStorage,
        lm_head: &dyn BackendStorage,
        targets: &dyn BackendStorage,
        lse: &dyn BackendStorage,
        v_tile_size: i32,
        inv_batch: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let h = as_rocm(hidden)?;
        let w = as_rocm(lm_head)?;
        let t = as_rocm(targets)?;
        let e = as_rocm(lse)?;
        if !h.device_ptr_is_valid()
            || !w.device_ptr_is_valid()
            || !t.device_ptr_is_valid()
            || !e.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_linear_ce: invalid input pointer".into(),
            ));
        }
        let hd = hidden.shape().dims();
        let wd = lm_head.shape().dims();
        if hd.len() != 2 || wd.len() != 2 || wd[1] != hd[1] || targets.shape().elem_count() != hd[0]
        {
            return Err(Error::Shape(
                "fused_linear_ce: incompatible input shapes".into(),
            ));
        }
        if v_tile_size <= 0 {
            return Err(Error::Backend(
                "fused_linear_ce: v_tile_size must be positive".into(),
            ));
        }
        let batch = hd[0];
        let grad =
            RocmStorage::alloc_gpu(hidden.shape(), dtype_f32(), &self.allocator, self.ordinal)?;
        let mut hp = dev_ptr(h)?;
        let mut wp = dev_ptr(w)?;
        let mut tp = dev_ptr(t)?;
        let mut ep = dev_ptr(e)?;
        let mut gp = dev_ptr(&grad)?;
        let mut k = hd[1] as i32;
        let mut v = wd[0] as i32;
        let mut tile = v_tile_size;
        let mut inv = inv_batch;
        let mut b = batch as i32;
        let block = crate::HipDim3 { x: 256, y: 1, z: 1 };
        let grid = crate::HipDim3 {
            x: batch as u32,
            y: 1,
            z: 1,
        };
        self.launch_compute_kernel(
            "grim_fused_linear_ce_backward",
            grid,
            block,
            &mut [
                arg(&mut hp),
                arg(&mut wp),
                arg(&mut tp),
                arg(&mut ep),
                arg(&mut gp),
                arg(&mut k),
                arg(&mut v),
                arg(&mut tile),
                arg(&mut inv),
                arg(&mut b),
            ],
        )?;
        Ok((
            Box::new(grad),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Cross-entropy loss + softmax gradient, host-staged. Returns `(avg_loss, grad)`.
    pub fn cross_entropy_gpu(
        &self,
        logits: &dyn BackendStorage,
        targets: &[usize],
        label_smoothing: Option<f32>,
    ) -> Result<(f32, Box<dyn BackendStorage>)> {
        let l_s = as_rocm(logits)?;
        if !l_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "cross_entropy_gpu: logits lack a valid device pointer".into(),
            ));
        }

        let dims = logits.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Shape(
                "cross_entropy_gpu: logits must be 2-D [batch_size, vocab_size]".into(),
            ));
        }
        let batch_size = dims[0];
        let vocab_size = dims[1];
        if batch_size == 0 {
            return Err(Error::Backend(
                "cross_entropy_gpu: batch_size must be > 0".into(),
            ));
        }
        if targets.len() != batch_size {
            return Err(Error::Shape(format!(
                "cross_entropy_gpu: targets len {} != batch_size {}",
                targets.len(),
                batch_size
            )));
        }
        let smooth = label_smoothing.unwrap_or(0.0).clamp(0.0, 1.0);
        let uniform = smooth / (vocab_size as f32);
        let confident = 1.0 - smooth;

        let logits_vec = logits.to_cpu_vec_f32()?;
        if logits_vec.len() < batch_size * vocab_size {
            return Err(Error::Backend(format!(
                "cross_entropy_gpu: logits length {} < batch_size * vocab_size {}",
                logits_vec.len(),
                batch_size * vocab_size
            )));
        }

        let mut grad_vec = vec![0.0f32; batch_size * vocab_size];
        let mut total_loss = 0.0f32;
        let inv_batch = 1.0 / (batch_size as f32);

        for (b, &target_token) in targets.iter().enumerate() {
            if target_token >= vocab_size {
                return Err(Error::Backend(format!(
                    "cross_entropy_gpu: target token {} out of bounds for vocab_size {}",
                    target_token, vocab_size
                )));
            }

            let row_start = b * vocab_size;
            let row_logits = &logits_vec[row_start..row_start + vocab_size];

            // Max trick for numerical stability.
            let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut exp_logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let exp_val = (row_logits[v] - max_logit).exp();
                exp_logits[v] = exp_val;
                sum_exp += exp_val;
            }
            let log_sum_exp = max_logit + sum_exp.ln();

            // Cross-entropy with optional label smoothing: loss = -sum_v q(v) * log_softmax(v) where log_softmax(v) = row_logits[v] - log_sum_exp, and the target distribution is q(target) = confident, q(other) = uniform.
            // This collapses to: confident * (log_sum_exp - logit_target) + uniform * (vocab_size * log_sum_exp -.
            let log_target = log_sum_exp - row_logits[target_token];
            let sum_logits: f32 = row_logits.iter().sum();
            let smooth_loss = uniform * ((vocab_size as f32) * log_sum_exp - sum_logits);
            total_loss += confident * log_target + smooth_loss;

            // Gradient dL/dLogits = (softmax - q) / batch_size.
            for v in 0..vocab_size {
                let prob = exp_logits[v] / sum_exp;
                let target_q = if v == target_token {
                    confident + uniform
                } else {
                    uniform
                };
                grad_vec[row_start + v] = (prob - target_q) * inv_batch;
            }
        }

        let avg_loss = total_loss * inv_batch;
        let grad_shape = logits.shape().clone();
        let grad_storage = RocmStorage::copy_from_host(
            &grad_vec,
            &grad_shape,
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        Ok((avg_loss, Box::new(grad_storage) as Box<dyn BackendStorage>))
    }
}
