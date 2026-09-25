//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! The trait-required `impl CoreTensorOps for RocmDevice` block, kept whole.

use std::ffi::c_void;

use grim_tensor::backend::{ComputeHandle, ReadyHandle};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, Shape};

use crate::device::gemm_tuning::lookup_gemm_config;
use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    arg, arith_to_compute_dtype, arith_to_rocblas_dtype, as_rocm, check_hip, dev_ptr, dtype_f32,
    hipFree, hipFreeAsync, hipMemAdvise, hipMemsetAsync, hipSuccess, linear_launch,
    rocblas_gemm_ex, rocblas_set_stream, rocblas_sgemm, rocblas_status_success, select_gemm_algo,
    upload_device_buffer, warp_rows_launch, RocblasInt, RocblasOperation, RocmHandle,
    ROCBLAS_GEMM_FLAGS_NONE,
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
        // SPEED-FP32-GEMV: For FP32 weights with M=1 (decode) and large N,
        // use the custom FP32 GEMV kernel instead of rocBLAS. rocBLAS is
        // pathologically slow at M=1 GEMV due to workspace allocation and
        // teardown per call. The custom kernel is ~3x faster.
        let dims = out_shape.dims();
        let m = dims[..dims.len().saturating_sub(1)]
            .iter()
            .product::<usize>()
            .max(1);
        let n = dims.last().copied().unwrap_or(0);
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
                    let out_storage = RocmStorage::alloc_gpu(
                        out_shape,
                        grim_tensor::DType {
                            arith: ArithType::F32,
                            storage: crate::DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?;
                    let stream = self.launch_fp32_gemv(a_s, b_s, &out_storage, m, n, k)?;
                    return Ok((
                        Box::new(out_storage),
                        Box::new(RocmHandle::new(Some(stream))),
                    ));
                }
            }
        }
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
        // SPEED-ROC-16: `b` is the natural weight (N, K); matmul computes C = A @ B^T.
        let (n, k2) = (b_dims[0], b_dims[1]);

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
        // View-compatible: accept RocmStorage AND RocmStorageView (zero-copy
        // slices, e.g. the fused gate+up output) via the trait device_ptr.
        let gate_ptr_dyn = crate::device::util::dev_ptr_dyn(gate)?;
        let up_ptr_dyn = crate::device::util::dev_ptr_dyn(up)?;
        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut gate_ptr = gate_ptr_dyn;
        let mut up_ptr = up_ptr_dyn;
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
        // Accept both RocmStorage and RocmStorageView (byte-offset views) via the
        // BackendStorage trait — no downcast, so views flow through untouched.
        let mut x_ptr = crate::device::util::dev_ptr_dyn(x)?;
        let mut w_ptr = crate::device::util::dev_ptr_dyn(weight)?;
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
        // Reject an out-of-range token BEFORE launching. The kernel indexes
        // `weight[indices[i] * dim + j]` with no bounds check, so a bad id
        // either faults or returns another token's row. This is cheap: it is
        // one host-side pass over the (small) index vector.
        {
            let dims = w_s.shape().dims();
            if dims.len() != 2 {
                return Err(Error::Shape(format!(
                    "embedding: weight must be 2-D, got {dims:?}"
                )));
            }
            let rows = dims[0];
            if let Some(bad) = indices.iter().find(|&&t| t as usize >= rows) {
                return Err(Error::Backend(format!(
                    "embedding: token id {bad} is out of range for a {rows}-row table"
                )));
            }
        }
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

    /// Q4_K embedding gather: rows stay packed in VRAM and are dequantized on
    /// read, so a 248320 x 5120 table costs 715 MB instead of 5.09 GB of f32
    /// (and no 5 GB transient host buffer during load).
    fn embedding_q4k(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
        dim: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Q4_K geometry: 256 elements per 144-byte super-block.
        const QK_BLOCK: usize = 256;
        const QK_BLOCK_BYTES: usize = 144;

        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let w_s = as_rocm(weight)
            .map_err(|_| Error::Backend("embedding_q4k: weight is not RocmStorage".into()))?;
        if !w_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "embedding_q4k: weight lacks a valid device pointer".into(),
            ));
        }
        let out_dims = out.dims();
        if out_dims.len() != 2 {
            return Err(Error::Shape("embedding_q4k: out must be [n, dim]".into()));
        }
        if out_dims[1] != dim {
            return Err(Error::Shape(format!(
                "embedding_q4k: out width {} != dim {dim}",
                out_dims[1]
            )));
        }
        if out_dims[0] != indices.len() {
            return Err(Error::Shape(format!(
                "embedding_q4k: indices len {} != out leading dim {}",
                indices.len(),
                out_dims[0]
            )));
        }
        // The kernel's address math is `(row/256)*144`, which is only exact when
        // every row is a whole number of super-blocks. Reject rather than
        // silently reading misaligned bytes.
        if dim == 0 || dim % QK_BLOCK != 0 {
            return Err(Error::Shape(format!(
                "embedding_q4k: dim {dim} must be a non-zero multiple of the \
                 Q4_K super-block size {QK_BLOCK}"
            )));
        }
        // Derive the row count from the packed byte length so an out-of-range
        // token is caught before the kernel indexes off the end.
        let packed_bytes = w_s.shape().dims()[0];
        let expected = (dim / QK_BLOCK) * QK_BLOCK_BYTES;
        if expected == 0 || packed_bytes % expected != 0 {
            return Err(Error::Shape(format!(
                "embedding_q4k: packed table of {packed_bytes} B is not a whole \
                 number of {expected}-B rows"
            )));
        }
        let rows = packed_bytes / expected;
        if let Some(bad) = indices.iter().find(|&&t| t as usize >= rows) {
            return Err(Error::Backend(format!(
                "embedding_q4k: token id {bad} is out of range for a {rows}-row table"
            )));
        }

        let total = out.elem_count();
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut idx_ptr = upload_device_buffer(self.ordinal, indices)?;
        let mut dim_i = dim as i32;
        let mut total_i = total as i32;
        let (grid, block) = linear_launch(total);
        let stream = self.launch_compute_kernel(
            "grim_embedding_q4k",
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
