//! Core tensor computation, GEMM, elementwise, and autograd operations for `CudaDevice`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{
    AutogradOps, BackendStorage, CoreTensorOps, ElementwiseOps, FusionOps, OptimizerOps,
    SamplingOps, Shape,
};

use crate::autotune::GemmOp;
use crate::device::cuda_device::CudaDevice;
use crate::device::handles::{
    CUBLAS_OP_N, CUBLAS_STATUS_SUCCESS, CUfunction, CudaHandle, cuLaunchKernel,
    cuModuleGetFunction, cublasSgemm_v2, cudaDeviceSynchronize, cudaFree, cudaMalloc, cudaMemcpy,
    cudaMemcpyHostToDevice, cudaSetDevice, cudaSuccess,
};
use crate::device::jit_cache::compile_and_load_kernel;
use crate::memory::storage::CudaStorage;

impl CudaDevice {
    pub fn matmul_op(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
        op: Option<GemmOp>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("matmul a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("matmul b is not CudaStorage".into()))?;

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(Error::Shape("matmul expects 2-D inputs".into()));
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
        if out_shape.dims() != &[m, n] {
            return Err(Error::Shape(format!(
                "expected out [{m},{n}], got {out_shape:?}"
            )));
        }

        let dtype_out = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::Native,
        };
        if a_storage.dtype != DType::F32 || b_storage.dtype != DType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "matmul: CUDA backend only supports F32 inputs (a={:?}, b={:?})",
                a_storage.dtype, b_storage.dtype
            )));
        }
        let out_storage = CudaStorage::alloc_gpu(out_shape, dtype_out, self.ordinal)?;

        unsafe {
            let _ = cudaSetDevice(self.ordinal as i32);
        }

        let handle_guard = self.cublas_handle.lock().unwrap_or_else(|e| e.into_inner());
        let handle = handle_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("cuBLAS handle not initialized".into()))?
            .0;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("matmul: A storage has no valid device pointer".into()))?
            as *const c_void;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("matmul: B storage has no valid device pointer".into()))?
            as *const c_void;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("matmul: out storage has no valid device pointer".into())
        })? as *mut c_void;

        unsafe {
            // SPEED-ROC-16: C = A @ B^T, B stored (N, K). transB=T transposes B's
            // (N, K) layout to (K, N) for the multiply; lda=K, ldb=K, ldc=M.
            let status = cublasSgemm_v2(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a_ptr as *const f32,
                k as i32,
                b_ptr as *const f32,
                k as i32,
                &beta,
                out_ptr as *mut f32,
                m as i32,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(Error::Backend(format!(
                    "cublasSgemm_v2 failed with status {}",
                    status
                )));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(true)),
        });

        let effective_op = op.unwrap_or(GemmOp::Other);
        let tile_cfg = self.gemm_tile_config(m, n, k, effective_op);
        if !tile_cfg.is_valid(&self.caps) {
            tracing::warn!(
                target: "grim_cuda",
                m = m,
                n = n,
                k = k,
                tile = ?tile_cfg,
                op = ?op,
                "CUDA matmul: tile config exceeds device resource limits"
            );
        }
        tracing::debug!(
            target: "grim_cuda",
            m = m,
            n = n,
            k = k,
            tile = ?tile_cfg,
            op = ?op,
            "CUDA matmul: tile config logged"
        );

        Ok((Box::new(out_storage), compute_handle))
    }

    /// `ShapeClass::TLOLog` (op-identity) regardless of M, so the wide-N tile is selected
    /// instead of relying on the n>=16384 dimension heuristic in the trait `matmul`.
    pub fn matmul_lm_head(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, Some(GemmOp::LmHead))
    }

    /// Fused Add + RMSNorm: `y_out = x + residual`, `norm_out = rms_norm(y_out, w, eps)`.
    /// Returns `(y_out, res_out, compute_handle)`.
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
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("fused_add_rms_norm x is not CudaStorage".into()))?;
        let res_storage = residual
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("fused_add_rms_norm residual is not CudaStorage".into())
            })?;
        let w_storage = weight
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("fused_add_rms_norm weight is not CudaStorage".into()))?;
        Self::ensure_f32_input("fused_add_rms_norm x", x_storage)?;
        Self::ensure_f32_input("fused_add_rms_norm residual", res_storage)?;
        Self::ensure_f32_input("fused_add_rms_norm weight", w_storage)?;

        let dtype_out = DType::F32;
        let y_storage = CudaStorage::alloc_gpu(out_shape, dtype_out.clone(), self.ordinal)?;
        let norm_storage = CudaStorage::alloc_gpu(out_shape, dtype_out, self.ordinal)?;

        let total = out_shape.elem_count();
        let row_len = out_shape.dims()[out_shape.dims().len() - 1];

        let mut x_ptr = Self::dev_ptr_or_err("fused_add_rms_norm x", x_storage)?;
        let mut res_ptr = Self::dev_ptr_or_err("fused_add_rms_norm residual", res_storage)?;
        let mut w_ptr = Self::dev_ptr_or_err("fused_add_rms_norm weight", w_storage)?;
        let mut y_ptr = Self::dev_ptr_or_err("fused_add_rms_norm y_out", &y_storage)?;
        let mut norm_ptr = Self::dev_ptr_or_err("fused_add_rms_norm norm_out", &norm_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_val = eps;
        let mut total_i = total as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut res_ptr as *mut *mut c_void as *mut c_void,
            &mut w_ptr as *mut *mut c_void as *mut c_void,
            &mut y_ptr as *mut *mut c_void as *mut c_void,
            &mut norm_ptr as *mut *mut c_void as *mut c_void,
            &mut row_len_i as *mut i32 as *mut c_void,
            &mut eps_val as *mut f32 as *mut c_void,
            &mut total_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_add_rms_norm", &mut args, total)?;
        Ok((Box::new(y_storage), Box::new(norm_storage), handle))
    }
}

impl CoreTensorOps for CudaDevice {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        if dtype != DType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "zeros: CUDA backend only supports F32 (got {dtype:?})"
            )));
        }
        let storage = CudaStorage::alloc_gpu(shape, dtype, self.ordinal)?;
        let dev_ptr = storage
            .device_ptr
            .ok_or_else(|| Error::Backend("zeros: device_ptr was null after alloc_gpu".into()))?
            as *mut c_void;

        let zeros_host = vec![0.0f32; shape.elem_count()];
        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                zeros_host.as_ptr() as *const c_void,
                storage.bytes,
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemcpy failed to initialize zeros with error {}",
                res
            )));
        }

        Ok(Box::new(storage))
    }

    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.matmul_op(a, b, out_shape, None)
    }

    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("add a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("add b is not CudaStorage".into()))?;
        Self::ensure_f32_input("add a", a_storage)?;
        Self::ensure_f32_input("add b", b_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let mut a_ptr = Self::dev_ptr_or_err("add a", a_storage)?;
        let mut b_ptr = Self::dev_ptr_or_err("add b", b_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("add out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut a_ptr as *mut *mut c_void as *mut c_void,
            &mut b_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_add", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }

    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mul a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mul b is not CudaStorage".into()))?;
        Self::ensure_f32_input("mul a", a_storage)?;
        Self::ensure_f32_input("mul b", b_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let mut a_ptr = Self::dev_ptr_or_err("mul a", a_storage)?;
        let mut b_ptr = Self::dev_ptr_or_err("mul b", b_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("mul out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut a_ptr as *mut *mut c_void as *mut c_void,
            &mut b_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_mul", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }

    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let gate_storage = gate
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul gate is not CudaStorage".into()))?;
        let up_storage = up
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul up is not CudaStorage".into()))?;
        Self::ensure_f32_input("silu_mul gate", gate_storage)?;
        Self::ensure_f32_input("silu_mul up", up_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let n = out.elem_count();

        let mut gate_ptr = Self::dev_ptr_or_err("silu_mul gate", gate_storage)?;
        let mut up_ptr = Self::dev_ptr_or_err("silu_mul up", up_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("silu_mul out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut gate_ptr as *mut *mut c_void as *mut c_void,
            &mut up_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_silu_mul", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }

    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("rms_norm x is not CudaStorage".into()))?;
        let w_storage = w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("rms_norm w is not CudaStorage".into()))?;
        Self::ensure_f32_input("rms_norm x", x_storage)?;
        Self::ensure_f32_input("rms_norm w", w_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let total = out.elem_count();
        let row_len = out.dims()[out.dims().len() - 1];

        let mut x_ptr = Self::dev_ptr_or_err("rms_norm x", x_storage)?;
        let mut w_ptr = Self::dev_ptr_or_err("rms_norm w", w_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("rms_norm out", &out_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_val = eps;
        let mut total_i = total as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut w_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut row_len_i as *mut i32 as *mut c_void,
            &mut eps_val as *mut f32 as *mut c_void,
            &mut total_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_rms_norm", &mut args, total)?;
        Ok((Box::new(out_storage), handle))
    }

    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("softmax x is not CudaStorage".into()))?;
        Self::ensure_f32_input("softmax x", x_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let total = out.elem_count();
        let last_dim = out.dims()[out.dims().len() - 1];

        let mut x_ptr = Self::dev_ptr_or_err("softmax x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("softmax out", &out_storage)?;
        let mut last_dim_i = last_dim as i32;
        let mut total_i = total as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut last_dim_i as *mut i32 as *mut c_void,
            &mut total_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_softmax", &mut args, total)?;
        Ok((Box::new(out_storage), handle))
    }

    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let weight_storage = weight
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("embedding weight is not CudaStorage".into()))?;
        Self::ensure_f32_input("embedding weight", weight_storage)?;

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let num_indices = indices.len();
        let embedding_dim = out.dims()[out.dims().len() - 1];

        // Allocate, upload, run, sync, free indices. Error paths free before returning.
        let mut dev_indices_ptr: *mut c_void = std::ptr::null_mut();
        let size_indices = num_indices * 4;
        unsafe {
            let res = cudaMalloc(&mut dev_indices_ptr, size_indices);
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaMalloc for indices failed: {res}"
                )));
            }
            let res = cudaMemcpy(
                dev_indices_ptr,
                indices.as_ptr() as *const c_void,
                size_indices,
                cudaMemcpyHostToDevice,
            );
            if res != cudaSuccess {
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!(
                    "cudaMemcpy for indices failed: {res}"
                )));
            }
        }

        // Embedding takes dev_indices_ptr and uses num_indices * embedding_dim threads,
        // so it can't use launch_rank1_kernel directly.
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_embedding")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!("cuModuleGetFunction failed: {res}")));
            }

            let mut w_ptr = Self::dev_ptr_or_err("embedding weight", weight_storage)?;
            let mut indices_ptr = dev_indices_ptr;
            let mut out_ptr = Self::dev_ptr_or_err("embedding out", &out_storage)?;
            let mut emb_dim_i = embedding_dim as i32;
            let mut num_idx_i = num_indices as i32;

            let mut args = [
                &mut w_ptr as *mut *mut c_void as *mut c_void,
                &mut indices_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut emb_dim_i as *mut i32 as *mut c_void,
                &mut num_idx_i as *mut i32 as *mut c_void,
            ];

            let block_size: usize = 256;
            let total_threads = num_indices * embedding_dim;
            let grid_size = (total_threads + block_size - 1) / block_size;

            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                block_size as u32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                let _ = cudaFree(dev_indices_ptr);
                return Err(Error::Backend(format!(
                    "cuLaunchKernel failed: {launch_res}"
                )));
            }
        }

        // Sync so staging buffer is safe to free.
        unsafe {
            let _ = cudaDeviceSynchronize();
            let _ = cudaFree(dev_indices_ptr);
        }
        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(true)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = CudaStorage::copy_from_host(data, shape, dtype, self.ordinal)?;
        Ok(Box::new(storage))
    }

    fn advise(
        &self,
        _storage: &dyn BackendStorage,
        _advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
        Ok(())
    }
}

impl ElementwiseOps for CudaDevice {
    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mul_scalar x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let n = out_shape.elem_count();

        let mut x_ptr = Self::dev_ptr_or_err("mul_scalar x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("mul_scalar out", &out_storage)?;
        let mut s_val = scalar;
        let mut n_i = n as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut s_val as *mut f32 as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_mul_scalar", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }

    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sqrt x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let n = out_shape.elem_count();

        let mut x_ptr = Self::dev_ptr_or_err("sqrt x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("sqrt out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_sqrt", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }

    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("recip x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let n = out_shape.elem_count();

        let mut x_ptr = Self::dev_ptr_or_err("recip x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("recip out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_recip", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
    }

    fn sub(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sub a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sub b is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let n = out_shape.elem_count();

        let mut a_ptr = Self::dev_ptr_or_err("sub a", a_storage)?;
        let mut b_ptr = Self::dev_ptr_or_err("sub b", b_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("sub out", &out_storage)?;
        let mut n_i = n as i32;
        let mut args = [
            &mut a_ptr as *mut *mut c_void as *mut c_void,
            &mut b_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];
        let handle = self.launch_rank1_kernel("grim_sub", &mut args, n)?;
        Ok((Box::new(out_storage), handle))
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

impl SamplingOps for CudaDevice {}

impl FusionOps for CudaDevice {
    /// Override the trait default with the real fused `grim_add_rms_norm` PTX kernel.
    fn fused_add_rms_norm(
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
        CudaDevice::fused_add_rms_norm(self, x, residual, weight, eps, out_shape)
    }
}

impl AutogradOps for CudaDevice {
    /// Fused (non-fused first cut) dequantized matmul backward on CUDA.
    /// Computes `dX[M, K] = dY[M, N] @ B_dequant^T` where `B` is a quantized, CUDA-resident weight.
    fn silu_mul_backward(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let gate_s = gate
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul_backward: gate is not CudaStorage".into()))?;
        let up_s = up
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul_backward: up is not CudaStorage".into()))?;
        let dw_s = dw
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("silu_mul_backward: dw is not CudaStorage".into()))?;

        Self::ensure_f32_input("silu_mul_backward gate", gate_s)?;
        Self::ensure_f32_input("silu_mul_backward up", up_s)?;
        Self::ensure_f32_input("silu_mul_backward dw", dw_s)?;

        if out_shape.dims() != gate_s.shape().dims() {
            return Err(Error::Shape(format!(
                "silu_mul_backward: out_shape must match gate shape, got {:?} vs {:?}",
                out_shape.dims(),
                gate_s.shape().dims()
            )));
        }

        let n = out_shape.elem_count();
        let mut gate_ptr = Self::dev_ptr_or_err("silu_mul_backward gate", gate_s)?;
        let mut up_ptr = Self::dev_ptr_or_err("silu_mul_backward up", up_s)?;
        let mut dw_ptr = Self::dev_ptr_or_err("silu_mul_backward dw", dw_s)?;

        let df_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let de_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let mut df_ptr = Self::dev_ptr_or_err("silu_mul_backward df", &df_storage)?;
        let mut de_ptr = Self::dev_ptr_or_err("silu_mul_backward de", &de_storage)?;

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut f: CUfunction = std::ptr::null_mut();
        let func_name = std::ffi::CString::new("grim_silu_mul_backward")
            .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
        let res = unsafe {
            cuModuleGetFunction(
                &mut f as *mut *mut c_void as *mut CUfunction,
                module,
                func_name.as_ptr(),
            )
        };
        if res != 0 || f.is_null() {
            return Err(Error::Backend(format!(
                "cuModuleGetFunction(grim_silu_mul_backward) failed: {res}"
            )));
        }

        let mut n_i = n as i32;
        let mut args = [
            &mut gate_ptr as *mut *mut c_void as *mut c_void,
            &mut up_ptr as *mut *mut c_void as *mut c_void,
            &mut dw_ptr as *mut *mut c_void as *mut c_void,
            &mut df_ptr as *mut *mut c_void as *mut c_void,
            &mut de_ptr as *mut *mut c_void as *mut c_void,
            &mut n_i as *mut i32 as *mut c_void,
        ];

        let handle = self.launch_rank1_kernel("silu_mul_backward", &mut args, n)?;
        Ok((Box::new(df_storage), Box::new(de_storage), handle))
    }
}

impl OptimizerOps for CudaDevice {}
