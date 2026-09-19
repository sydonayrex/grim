//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! The trait-required `impl AutogradOps for RocmDevice` block, kept whole.



use grim_tensor::backend::{ ComputeHandle };
use grim_tensor::dtype::{ DType };
use grim_tensor::error::{Error, Result};
use grim_tensor::{ AutogradOps, BackendStorage, Shape };

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{ RocmHandle, arg, as_rocm, dev_ptr, dtype_f32, linear_launch, warp_rows_launch };

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
