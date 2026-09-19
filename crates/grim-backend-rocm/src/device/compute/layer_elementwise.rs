//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! Bare-impl elementwise/normalization ops (gelu, axpy, norms, silu-mul).

use std::ffi::c_void;


use grim_tensor::backend::{ ComputeHandle };
use grim_tensor::error::{Error, Result};
use grim_tensor::{ BackendStorage, Shape };

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{ RocmHandle, arg, as_rocm, dev_ptr, dtype_f32, linear_launch, warp_rows_launch };

impl RocmDevice {
    /// S2 (PLAN-kernel-fusion): elementwise `out = a * b` into
    /// CALLER-PROVIDED `out` — no allocation inside. Graph-capture-safe
    /// companion to [`Self::mul`] (same `grim_mul` kernel).
    pub fn mul_into(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<()> {
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        if !a_s.device_ptr_is_valid() || !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "mul_into: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.shape().elem_count();
        let mut out_ptr = dev_ptr(out)?;
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
        Ok(())
    }

    /// Elementwise `out = silu(x)` into CALLER-PROVIDED `out` — no allocation inside.
    pub fn silu_into(
        &self,
        x: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<()> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_into: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.shape().elem_count();
        let mut out_ptr = dev_ptr(out)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_silu",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok(())
    }

    /// Elementwise `out = sigmoid(x)` into CALLER-PROVIDED `out` — no allocation inside.
    pub fn sigmoid_into(
        &self,
        x: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<()> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "sigmoid_into: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.shape().elem_count();
        let mut out_ptr = dev_ptr(out)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_sigmoid",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok(())
    }

    /// Elementwise `out = gelu(x)` into CALLER-PROVIDED `out` — no allocation inside.
    pub fn gelu_into(
        &self,
        x: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<()> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "gelu_into: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.shape().elem_count();
        let mut out_ptr = dev_ptr(out)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_gelu",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok(())
    }

    /// Elementwise GeLU activation returning newly allocated storage.
    pub fn gelu(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<Box<dyn BackendStorage>> {
        let out = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.gelu_into(x, &out)?;
        Ok(Box::new(out))
    }

    /// Elementwise GeLU-tanh gated multiplication `out = (gelu_tanh(gate)) * up` into CALLER-PROVIDED `out`.
    pub fn gelu_tanh_mul_into(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<()> {
        let gate_s = as_rocm(gate)?;
        let up_s = as_rocm(up)?;
        if !gate_s.device_ptr_is_valid() || !up_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "gelu_tanh_mul_into: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.shape().elem_count();
        if gate.shape().elem_count() != total || up.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "gelu_tanh_mul_into: elem mismatch gate={} up={} out={total}",
                gate.shape().elem_count(),
                up.shape().elem_count()
            )));
        }
        if !out.device_ptr_is_valid() {
            return Err(Error::Backend(
                "gelu_tanh_mul_into: out lacks a valid device pointer".into(),
            ));
        }
        let mut out_ptr = dev_ptr(out)?;
        let mut gate_ptr = dev_ptr(gate_s)?;
        let mut up_ptr = dev_ptr(up_s)?;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_gelu_tanh_mul",
            grid,
            block,
            &mut [
                arg(&mut gate_ptr),
                arg(&mut up_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok(())
    }

    /// Elementwise GeLU-tanh gated multiplication returning newly allocated storage.
    pub fn gelu_tanh_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<Box<dyn BackendStorage>> {
        let out = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.gelu_tanh_mul_into(gate, up, &out)?;
        Ok(Box::new(out))
    }

    /// Elementwise `out = cap * tanh(x / cap)` into CALLER-PROVIDED `out`.
    pub fn tanh_softcap_into(
        &self,
        x: &dyn BackendStorage,
        cap: f32,
        out: &RocmStorage,
    ) -> Result<()> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "tanh_softcap_into: input lacks a valid device pointer".into(),
            ));
        }
        let total = out.shape().elem_count();
        let mut out_ptr = dev_ptr(out)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut cap_f = cap;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_tanh_softcap",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut cap_f),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok(())
    }

    /// `out = a + s * b` writing into CALLER-PROVIDED `out`.
    pub fn axpy_into(
        &self,
        a: &dyn BackendStorage,
        s: f32,
        b: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<()> {
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        if !a_s.device_ptr_is_valid() || !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "axpy_into: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.shape().elem_count();
        if a.shape().elem_count() != total || b.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "axpy_into: elem mismatch a={} b={} out={total}",
                a.shape().elem_count(),
                b.shape().elem_count()
            )));
        }
        if !out.device_ptr_is_valid() {
            return Err(Error::Backend(
                "axpy_into: out lacks a valid device pointer".into(),
            ));
        }
        let mut out_ptr = dev_ptr(out)?;
        let mut a_ptr = dev_ptr(a_s)?;
        let mut b_ptr = dev_ptr(b_s)?;
        let mut s_f = s;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_axpy",
            grid,
            block,
            &mut [
                arg(&mut a_ptr),
                arg(&mut s_f),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut n),
            ],
        )?;
        Ok(())
    }

    /// `out = a + s * b` returning a newly allocated storage.
    pub fn axpy(
        &self,
        a: &dyn BackendStorage,
        s: f32,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<Box<dyn BackendStorage>> {
        let out = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.axpy_into(a, s, b, &out)?;
        Ok(Box::new(out))
    }

    /// `y = rms_norm(x, weight)` writing into CALLER-PROVIDED `out` — no
    /// allocation inside. Same kernel as [`CoreTensorOps::rms_norm`].
    /// `out_shape` carries row semantics explicitly (like
    /// `rope_dev_base_into`): it may reshape flat storage, so QK-norm over
    /// `[nh, hd]` rows can target a `[1, n]` slot and vice versa, provided
    /// element counts agree. In-place (`out` aliases `x`) is safe: the
    /// kernel reduces each row fully before storing it.
    pub fn rms_norm_into(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &RocmStorage,
        out_shape: &Shape,
    ) -> Result<Box<dyn ComputeHandle>> {
        use crate::device::util::dev_ptr_dyn;
        let mut x_ptr = dev_ptr_dyn(x)?;
        let mut w_ptr = dev_ptr_dyn(weight)?;
        let row_len = out_shape.dims().last().copied().ok_or_else(|| {
            Error::Shape("rms_norm_into: empty out dims".into())
        })?;
        let total = out_shape.elem_count();
        if x.shape().elem_count() != total || out.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "rms_norm_into: elem mismatch x={} out={} shape={total}",
                x.shape().elem_count(),
                out.shape().elem_count()
            )));
        }
        if !out.device_ptr_is_valid() {
            return Err(Error::Backend(
                "rms_norm_into: out lacks a valid device pointer".into(),
            ));
        }
        let mut out_ptr = dev_ptr(out)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
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
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// `y = (x - mean) / sqrt(var + eps) * w + b` per row (mean/variance
    /// LayerNorm, unlike RMS norm). `bias` may be `None` (kernel receives a
    /// NULL pointer). Writes into CALLER-PROVIDED `out`; in-place (`out`
    /// aliases `x`) is safe: each element is read once before the write pass.
    /// Capture-safe: pure kernel launches, no allocation, no sync.
    pub fn layer_norm_into(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        bias: Option<&dyn BackendStorage>,
        eps: f32,
        out: &RocmStorage,
        out_shape: &Shape,
    ) -> Result<Box<dyn ComputeHandle>> {
        use crate::device::util::dev_ptr_dyn;
        let mut x_ptr = dev_ptr_dyn(x)?;
        let mut w_ptr = dev_ptr_dyn(weight)?;
        let mut b_ptr: *mut c_void = match bias {
            Some(b) => dev_ptr_dyn(b)?,
            None => std::ptr::null_mut(),
        };
        let row_len = out_shape
            .dims()
            .last()
            .copied()
            .ok_or_else(|| Error::Shape("layer_norm_into: empty out dims".into()))?;
        let total = out_shape.elem_count();
        if x.shape().elem_count() != total || out.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "layer_norm_into: elem mismatch x={} out={} shape={total}",
                x.shape().elem_count(),
                out.shape().elem_count()
            )));
        }
        if !out.device_ptr_is_valid() {
            return Err(Error::Backend(
                "layer_norm_into: out lacks a valid device pointer".into(),
            ));
        }
        let mut out_ptr = dev_ptr(out)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_layer_norm",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut b_ptr),
                arg(&mut out_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// `y = a + b` writing into CALLER-PROVIDED `out` — no allocation inside.
    /// Same kernel as [`CoreTensorOps::add`]. In-place (`out` aliases an
    /// input) is safe: strictly per-element.
    pub fn add_into(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<Box<dyn ComputeHandle>> {
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        if !a_s.device_ptr_is_valid() || !b_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "add_into: inputs lack a valid device pointer".into(),
            ));
        }
        let total = out.shape().elem_count();
        if a.shape().elem_count() != total || b.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "add_into: elem mismatch a={} b={} out={total}",
                a.shape().elem_count(),
                b.shape().elem_count()
            )));
        }
        if !out.device_ptr_is_valid() {
            return Err(Error::Backend(
                "add_into: out lacks a valid device pointer".into(),
            ));
        }
        let mut out_ptr = dev_ptr(out)?;
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
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// `y = silu(gate) * up` writing into CALLER-PROVIDED `out` — no
    /// allocation inside. Same kernel as [`CoreTensorOps::silu_mul`].
    pub fn silu_mul_into(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &RocmStorage,
    ) -> Result<Box<dyn ComputeHandle>> {
        let gate_ptr_dyn = crate::device::util::dev_ptr_dyn(gate)?;
        let up_ptr_dyn = crate::device::util::dev_ptr_dyn(up)?;
        let total = out.shape().elem_count();
        if gate.shape().elem_count() != total || up.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "silu_mul_into: elem mismatch gate={} up={} out={total}",
                gate.shape().elem_count(),
                up.shape().elem_count()
            )));
        }
        if !out.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_mul_into: out lacks a valid device pointer".into(),
            ));
        }
        let mut out_ptr = dev_ptr(out)?;
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
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }
}
