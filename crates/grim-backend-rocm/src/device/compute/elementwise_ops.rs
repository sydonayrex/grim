//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! The trait-required `impl ElementwiseOps for RocmDevice` block, kept whole.

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::ArithType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, ElementwiseOps, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::device::util::ROCM_COMPUTE_BLOCK;
use crate::{arg, as_rocm, dev_ptr, dtype_f32, linear_launch, RocmHandle};

impl ElementwiseOps for RocmDevice {
    fn row_scale(
        &self,
        x: &dyn BackendStorage,
        scale: &dyn BackendStorage,
        rows: usize,
        cols: usize,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let s_s = as_rocm(scale)?;
        if !x_s.device_ptr_is_valid() || !s_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "row_scale: input lacks a valid device pointer".into(),
            ));
        }
        if x_s.shape().elem_count() != rows * cols {
            return Err(Error::Shape(format!(
                "row_scale: x holds {} elements, expected {rows}x{cols}",
                x_s.shape().elem_count()
            )));
        }
        if s_s.shape().elem_count() < rows {
            return Err(Error::Shape(format!(
                "row_scale: scale holds {} elements, expected {rows}",
                s_s.shape().elem_count()
            )));
        }
        let total = rows * cols;
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut s_ptr = dev_ptr(s_s)?;
        let mut r = rows as i32;
        let mut c = cols as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_row_scale",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut s_ptr),
                arg(&mut out_ptr),
                arg(&mut r),
                arg(&mut c),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn narrow_rows(
        &self,
        x: &dyn BackendStorage,
        start_row: usize,
        rows: usize,
        cols: usize,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "narrow_rows: input lacks a valid device pointer".into(),
            ));
        }
        let total = x_s.shape().elem_count();
        if (start_row + rows) * cols > total {
            return Err(Error::Shape(format!(
                "narrow_rows: rows [{start_row}, {}) of {cols} exceed {total} elements",
                start_row + rows
            )));
        }
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut src_ptr = dev_ptr(x_s)?;
        let mut dst_ptr = dev_ptr(&storage)?;
        let mut start = start_row as i32;
        let mut r = rows as i32;
        let mut c = cols as i32;
        let (grid, block) = (crate::HipDim3::new(rows as u32, 1, 1), crate::HipDim3::new(ROCM_COMPUTE_BLOCK as u32, 1, 1));
        self.launch_compute_kernel(
            "grim_row_copy",
            grid,
            block,
            &mut [
                arg(&mut src_ptr),
                arg(&mut dst_ptr),
                arg(&mut start),
                arg(&mut r),
                arg(&mut c),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn write_rows(
        &self,
        dst: &mut dyn BackendStorage,
        start_row: usize,
        src: &dyn BackendStorage,
        rows: usize,
        cols: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let dst_s = as_rocm(dst)?;
        let src_s = as_rocm(src)?;
        if !dst_s.device_ptr_is_valid() || !src_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "write_rows: an input lacks a valid device pointer".into(),
            ));
        }
        if src_s.shape().elem_count() != rows * cols {
            return Err(Error::Shape(format!(
                "write_rows: src holds {} elements, expected {rows}x{cols}",
                src_s.shape().elem_count()
            )));
        }
        if (start_row + rows) * cols > dst_s.shape().elem_count() {
            return Err(Error::Shape(format!(
                "write_rows: rows [{start_row}, {}) of {cols} exceed {} destination elements",
                start_row + rows,
                dst_s.shape().elem_count()
            )));
        }
        let mut src_ptr = dev_ptr(src_s)?;
        let mut dst_ptr = dev_ptr(dst_s)?;
        let mut start = start_row as i32;
        let mut r = rows as i32;
        let mut c = cols as i32;
        let (grid, block) = (crate::HipDim3::new(rows as u32, 1, 1), crate::HipDim3::new(ROCM_COMPUTE_BLOCK as u32, 1, 1));
        self.launch_compute_kernel(
            "grim_row_copy_into",
            grid,
            block,
            &mut [
                arg(&mut src_ptr),
                arg(&mut dst_ptr),
                arg(&mut start),
                arg(&mut r),
                arg(&mut c),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    fn narrow_cols(
        &self,
        x: &dyn BackendStorage,
        total_cols: usize,
        start_col: usize,
        rows: usize,
        cols: usize,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "narrow_cols: input lacks a valid device pointer".into(),
            ));
        }
        if start_col + cols > total_cols || rows * total_cols > x_s.shape().elem_count() {
            return Err(Error::Shape(format!(
                "narrow_cols: cols [{start_col}, {}) of {total_cols} / {rows} rows exceed {} elements",
                start_col + cols,
                x_s.shape().elem_count()
            )));
        }
        let storage = RocmStorage::alloc_gpu(out, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut src_ptr = dev_ptr(x_s)?;
        let mut dst_ptr = dev_ptr(&storage)?;
        let mut start = start_col as i32;
        let mut r = rows as i32;
        let mut sc = total_cols as i32;
        let mut c = cols as i32;
        self.launch_compute_kernel(
            "grim_col_copy",
            crate::HipDim3::new(rows as u32, 1, 1),
            crate::HipDim3::new(ROCM_COMPUTE_BLOCK, 1, 1),
            &mut [
                arg(&mut src_ptr),
                arg(&mut dst_ptr),
                arg(&mut start),
                arg(&mut r),
                arg(&mut sc),
                arg(&mut c),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    fn write_cols(
        &self,
        dst: &mut dyn BackendStorage,
        total_cols: usize,
        start_col: usize,
        src: &dyn BackendStorage,
        rows: usize,
        cols: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let dst_s = as_rocm(dst)?;
        let src_s = as_rocm(src)?;
        if !dst_s.device_ptr_is_valid() || !src_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "write_cols: an input lacks a valid device pointer".into(),
            ));
        }
        if src_s.shape().elem_count() != rows * cols {
            return Err(Error::Shape(format!(
                "write_cols: src holds {} elements, expected {rows}x{cols}",
                src_s.shape().elem_count()
            )));
        }
        if start_col + cols > total_cols || rows * total_cols > dst_s.shape().elem_count() {
            return Err(Error::Shape(format!(
                "write_cols: cols [{start_col}, {}) of {total_cols} / {rows} rows exceed {} destination elements",
                start_col + cols,
                dst_s.shape().elem_count()
            )));
        }
        let mut src_ptr = dev_ptr(src_s)?;
        let mut dst_ptr = dev_ptr(dst_s)?;
        let mut start = start_col as i32;
        let mut r = rows as i32;
        let mut dc = total_cols as i32;
        let mut c = cols as i32;
        self.launch_compute_kernel(
            "grim_col_copy_into",
            crate::HipDim3::new(rows as u32, 1, 1),
            crate::HipDim3::new(ROCM_COMPUTE_BLOCK, 1, 1),
            &mut [
                arg(&mut src_ptr),
                arg(&mut dst_ptr),
                arg(&mut start),
                arg(&mut r),
                arg(&mut dc),
                arg(&mut c),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

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
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        if !a_s.device_ptr_is_valid() || !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "sub: inputs lack a valid device pointer".into(),
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
            "grim_sub",
            grid,
            block,
            &mut [
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        let handle = RocmHandle::new(None);
        Ok((Box::new(storage), Box::new(handle)))
    }

    fn reduce_sum(&self, x: &dyn BackendStorage) -> Result<f32> {
        if let Ok(rocm_s) = as_rocm(x) {
            if rocm_s.device_ptr_is_valid() && rocm_s.dtype().arith == ArithType::F32 {
                return self.gpu_reduce_sum(rocm_s);
            }
        }
        let v = x.to_cpu_vec_f32()?;
        if v.is_empty() {
            return Err(Error::Backend("reduce_sum: empty tensor".into()));
        }
        Ok(v.iter().sum())
    }

    fn reduce_max(&self, x: &dyn BackendStorage) -> Result<f32> {
        if let Ok(rocm_s) = as_rocm(x) {
            if rocm_s.device_ptr_is_valid() && rocm_s.dtype().arith == ArithType::F32 {
                return self.gpu_reduce_max(rocm_s);
            }
        }
        let v = x.to_cpu_vec_f32()?;
        v.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| Error::Backend("reduce_max: empty tensor".into()))
    }

    fn argmax(&self, x: &dyn BackendStorage) -> Result<u32> {
        if let Ok(rocm_s) = as_rocm(x) {
            if rocm_s.device_ptr_is_valid() && rocm_s.dtype().arith == ArithType::F32 {
                return self.gpu_argmax(rocm_s);
            }
        }
        let v = x.to_cpu_vec_f32()?;
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| Error::Backend("argmax: empty tensor".into()))
    }
}
