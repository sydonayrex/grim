//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! Bare-impl elementwise/normalization ops (gelu, axpy, norms, silu-mul).

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::device::util::dev_ptr_dyn;
use crate::{
    arg, as_rocm, dev_ptr, dtype_f32, linear_launch, warp_rows_launch, RocmHandle,
};

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

    /// Broadcast a per-channel vector `in` [dk] across heads -> `out` [nh, dk].
    /// Used by the on-device GDL-2 gate path so the shared erase/write/decay
    /// gates reach the fused kernel without a host readback.
    pub fn broadcast_heads(
        &self,
        in_: &dyn BackendStorage,
        out: &RocmStorage,
        dk: usize,
        nh: usize,
    ) -> Result<()> {
        let in_s = as_rocm(in_)?;
        if !in_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "broadcast_heads: input lacks a valid device pointer".into(),
            ));
        }
        if dk == 0 || nh == 0 || dk > 1024 {
            return Err(Error::Shape(format!(
                "broadcast_heads: require 1 <= dk <= 1024 and nh > 0, got dk={dk}, nh={nh}"
            )));
        }
        let total = dk
            .checked_mul(nh)
            .ok_or_else(|| Error::Shape("broadcast_heads: dk * nh overflow".into()))?;
        if in_s.shape().elem_count() != dk || out.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "broadcast_heads: expected input [{dk}] and output [{nh}, {dk}], got {} and {} elements",
                in_s.shape().elem_count(),
                out.shape().elem_count(),
            )));
        }
        let mut dk_i = i32::try_from(dk)
            .map_err(|_| Error::Shape(format!("broadcast_heads: dk={dk} exceeds i32")))?;
        let mut nh_i = i32::try_from(nh)
            .map_err(|_| Error::Shape(format!("broadcast_heads: nh={nh} exceeds i32")))?;
        let mut out_ptr = dev_ptr(out)?;
        let mut in_ptr = dev_ptr(in_s)?;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_broadcast_heads",
            grid,
            block,
            &mut [
                arg(&mut in_ptr),
                arg(&mut out_ptr),
                arg(&mut dk_i),
                arg(&mut nh_i),
            ],
        )?;
        Ok(())
    }

    /// GQA head-repeat: expand K/V from `[nkv, hd]` to `[nh, hd]` by repeating
    /// each KV head `kv_group` times (`kv_h = h / kv_group`). One launch handles
    /// both K and V. `nh` must equal `nkv * kv_group`.
    pub fn head_repeat(
        &self,
        k_in: &dyn BackendStorage,
        v_in: &dyn BackendStorage,
        k_out: &RocmStorage,
        v_out: &RocmStorage,
        nkv: usize,
        nh: usize,
        kv_group: usize,
        hd: usize,
    ) -> Result<()> {
        let k_in_s = as_rocm(k_in)?;
        let v_in_s = as_rocm(v_in)?;
        if !k_in_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "head_repeat: k_in lacks a valid device pointer".into(),
            ));
        }
        if !v_in_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "head_repeat: v_in lacks a valid device pointer".into(),
            ));
        }
        if hd == 0 || nh == 0 || nkv == 0 || kv_group == 0 {
            return Err(Error::Shape(format!(
                "head_repeat: require hd>0, nh>0, nkv>0, kv_group>0 (got hd={hd}, nh={nh}, nkv={nkv}, kv_group={kv_group})"
            )));
        }
        if nh != nkv * kv_group {
            return Err(Error::Shape(format!(
                "head_repeat: nh must equal nkv * kv_group (got nh={nh}, nkv={nkv}, kv_group={kv_group})"
            )));
        }
        let total = hd
            .checked_mul(nh)
            .ok_or_else(|| Error::Shape("head_repeat: hd * nh overflow".into()))?;
        if k_in_s.shape().elem_count() != nkv * hd {
            return Err(Error::Shape(format!(
                "head_repeat: expected k_in [{nkv}, {hd}], got {} elements",
                k_in_s.shape().elem_count()
            )));
        }
        if v_in_s.shape().elem_count() != nkv * hd {
            return Err(Error::Shape(format!(
                "head_repeat: expected v_in [{nkv}, {hd}], got {} elements",
                v_in_s.shape().elem_count()
            )));
        }
        if k_out.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "head_repeat: expected k_out [{nh}, {hd}], got {} elements",
                k_out.shape().elem_count()
            )));
        }
        if v_out.shape().elem_count() != total {
            return Err(Error::Shape(format!(
                "head_repeat: expected v_out [{nh}, {hd}], got {} elements",
                v_out.shape().elem_count()
            )));
        }
        let mut nkv_i = i32::try_from(nkv)
            .map_err(|_| Error::Shape(format!("head_repeat: nkv={nkv} exceeds i32")))?;
        let mut nh_i = i32::try_from(nh)
            .map_err(|_| Error::Shape(format!("head_repeat: nh={nh} exceeds i32")))?;
        let mut kv_group_i = i32::try_from(kv_group)
            .map_err(|_| Error::Shape(format!("head_repeat: kv_group={kv_group} exceeds i32")))?;
        let mut hd_i = i32::try_from(hd)
            .map_err(|_| Error::Shape(format!("head_repeat: hd={hd} exceeds i32")))?;
        let mut k_in_ptr = dev_ptr(k_in_s)?;
        let mut v_in_ptr = dev_ptr(v_in_s)?;
        let mut k_out_ptr = dev_ptr(k_out)?;
        let mut v_out_ptr = dev_ptr(v_out)?;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_head_repeat",
            grid,
            block,
            &mut [
                arg(&mut k_in_ptr),
                arg(&mut v_in_ptr),
                arg(&mut k_out_ptr),
                arg(&mut v_out_ptr),
                arg(&mut nkv_i),
                arg(&mut nh_i),
                arg(&mut kv_group_i),
                arg(&mut hd_i),
            ],
        )?;
        Ok(())
    }

    /// Elementwise `out = silu(x)` into CALLER-PROVIDED `out` — no allocation inside.
    pub fn silu_into(&self, x: &dyn BackendStorage, out: &RocmStorage) -> Result<()> {
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
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok(())
    }

    /// Elementwise `out = sigmoid(x)` into CALLER-PROVIDED `out` — no allocation inside.
    pub fn sigmoid_into(&self, x: &dyn BackendStorage, out: &RocmStorage) -> Result<()> {
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
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut n)],
        )?;
        Ok(())
    }

    /// Elementwise SiLU activation returning newly allocated storage.
    pub fn silu(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<Box<dyn BackendStorage>> {
        let out = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.silu_into(x, &out)?;
        Ok(Box::new(out))
    }

    /// Elementwise Sigmoid activation returning newly allocated storage.
    pub fn sigmoid(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<Box<dyn BackendStorage>> {
        let out = RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.sigmoid_into(x, &out)?;
        Ok(Box::new(out))
    }

    /// Elementwise `out = gelu(x)` into CALLER-PROVIDED `out` — no allocation inside.
    pub fn gelu_into(&self, x: &dyn BackendStorage, out: &RocmStorage) -> Result<()> {
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
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut n)],
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
        let mut gate_ptr = dev_ptr_dyn(gate)?;
        let mut up_ptr = dev_ptr_dyn(up)?;
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
    /// Fused residual-add + RMS norm writing into caller-provided buffers.
    /// Decode-graph hot path: replaces (add_into + rms_norm_into) pairs with
    /// ONE launch. `norm_out` may alias `residual` (the kernel reads residual
    /// only in pass 1 and only reads the sum in pass 2).
    pub fn fused_add_rms_norm_into(
        &self,
        x: &dyn BackendStorage,
        residual: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        sum_out: &RocmStorage,
        norm_out: &RocmStorage,
        out_shape: &Shape,
    ) -> Result<Box<dyn ComputeHandle>> {
        use crate::device::util::dev_ptr_dyn;
        let mut x_ptr = dev_ptr_dyn(x)?;
        let mut r_ptr = dev_ptr_dyn(residual)?;
        let mut w_ptr = dev_ptr_dyn(weight)?;
        let row_len = out_shape
            .dims()
            .last()
            .copied()
            .ok_or_else(|| Error::Shape("fused_add_rms_norm_into: empty out dims".into()))?;
        let total = out_shape.elem_count();
        if x.shape().elem_count() != total
            || residual.shape().elem_count() != total
            || sum_out.shape().elem_count() != total
            || norm_out.shape().elem_count() != total
        {
            return Err(Error::Shape(format!(
                "fused_add_rms_norm_into: elem mismatch x={} r={} sum={} norm={} shape={total}",
                x.shape().elem_count(),
                residual.shape().elem_count(),
                sum_out.shape().elem_count(),
                norm_out.shape().elem_count()
            )));
        }
        if !sum_out.device_ptr_is_valid() || !norm_out.device_ptr_is_valid() {
            return Err(Error::Backend(
                "fused_add_rms_norm_into: outputs lack valid device pointers".into(),
            ));
        }
        let mut y_ptr = dev_ptr(sum_out)?;
        let mut n_ptr = dev_ptr(norm_out)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_add_rms_norm",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut r_ptr),
                arg(&mut w_ptr),
                arg(&mut y_ptr),
                arg(&mut n_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// GRAVE Phase 1: elementwise f32 -> f16 conversion with element offsets
    /// (prefill writes into the f16 KV arena). Graph-capture safe.
    pub fn convert_f32_to_f16_into(
        &self,
        src: &dyn BackendStorage,
        src_off: usize,
        dst: &RocmStorage,
        dst_off: usize,
        n: usize,
    ) -> Result<()> {
        use crate::device::util::dev_ptr_dyn;
        let mut src_ptr = dev_ptr_dyn(src)?;
        let mut dst_ptr = dev_ptr(dst)?;
        let mut src_off_i = src_off as i32;
        let mut dst_off_i = dst_off as i32;
        let mut n_i = n as i32;
        let (grid, block) = linear_launch(n);
        self.launch_compute_kernel(
            "grim_f32_to_f16",
            grid,
            block,
            &mut [
                arg(&mut src_ptr),
                arg(&mut dst_ptr),
                arg(&mut src_off_i),
                arg(&mut dst_off_i),
                arg(&mut n_i),
            ],
        )?;
        Ok(())
    }

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
        let row_len = out_shape
            .dims()
            .last()
            .copied()
            .ok_or_else(|| Error::Shape("rms_norm_into: empty out dims".into()))?;
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

/// Xing4.0 MHC gate tensors produced by one `grim_mhc_gates` launch.
pub struct MhcGateTensors {
    pub pre: Box<dyn BackendStorage>,
    pub post: Box<dyn BackendStorage>,
    pub comb: Box<dyn BackendStorage>,
}

impl RocmDevice {
    /// Fused Xing4.0 hyper-connection gate math, one block per token.
    ///
    /// `proj` is `[seq, (2 + hc) * hc]` (pre | post | comb), `base` is `[mix]`,
    /// `scale` is `[3]`. Emits `pre` `[hc, seq]`, `post` `[hc, seq]`, and the
    /// Sinkhorn-normalized `comb` `[hc * hc, seq]` — all stream-major with the
    /// token index last, so downstream per-stream weight vectors are contiguous
    /// row slices. All device-resident.
    ///
    /// This exists so the gate math never round-trips to the host: the Sinkhorn
    /// iterations need per-token row/column reductions over `hc * hc` values,
    /// which no dim-wise reduction primitive exposes, and the projection is
    /// only 24 floats per token.
    pub fn mhc_gates_into(
        &self,
        proj: &dyn BackendStorage,
        base: &dyn BackendStorage,
        scale: &dyn BackendStorage,
        seq: usize,
        hc: usize,
        iters: usize,
        hc_eps: f32,
        clamp_min: f32,
        clamp_max: f32,
    ) -> Result<MhcGateTensors> {
        if seq == 0 {
            return Err(Error::Backend("mhc_gates_into: empty sequence".into()));
        }
        if hc == 0 || hc > 8 {
            return Err(Error::Backend(format!(
                "mhc_gates_into: hc must be 1..=8, got {hc}"
            )));
        }
        let proj_s = as_rocm(proj)?;
        let base_s = as_rocm(base)?;
        let scale_s = as_rocm(scale)?;
        if !proj_s.device_ptr_is_valid()
            || !base_s.device_ptr_is_valid()
            || !scale_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "mhc_gates_into: an input lacks a valid device pointer".into(),
            ));
        }
        let mix = (2 + hc) * hc;
        if proj_s.shape().elem_count() != seq * mix {
            return Err(Error::Shape(format!(
                "mhc_gates_into: proj holds {} elements, expected {seq}x{mix}",
                proj_s.shape().elem_count()
            )));
        }

        // Stream-major / token-last so each per-stream weight vector is a
        // contiguous row slice downstream (see grim_mhc_gates).
        let pre_shape = Shape::new(vec![hc, seq]);
        let comb_shape = Shape::new(vec![hc * hc, seq]);
        let pre = RocmStorage::alloc_gpu(&pre_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let post = RocmStorage::alloc_gpu(&pre_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let comb = RocmStorage::alloc_gpu(&comb_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        let mut proj_ptr = dev_ptr(proj_s)?;
        let mut base_ptr = dev_ptr(base_s)?;
        let mut scale_ptr = dev_ptr(scale_s)?;
        let mut pre_ptr = dev_ptr(&pre)?;
        let mut post_ptr = dev_ptr(&post)?;
        let mut comb_ptr = dev_ptr(&comb)?;
        let mut s = seq as i32;
        let mut h = hc as i32;
        let mut it = iters as i32;
        let mut eps = hc_eps;
        let mut cmin = clamp_min;
        let mut cmax = clamp_max;

        // One block per token; 64 threads keeps the shared-memory max reduction
        // in two wave32 waves on RDNA.
        self.launch_compute_kernel(
            "grim_mhc_gates",
            crate::HipDim3::new(seq as u32, 1, 1),
            crate::HipDim3::new(64, 1, 1),
            &mut [
                arg(&mut proj_ptr),
                arg(&mut base_ptr),
                arg(&mut scale_ptr),
                arg(&mut pre_ptr),
                arg(&mut post_ptr),
                arg(&mut comb_ptr),
                arg(&mut s),
                arg(&mut h),
                arg(&mut it),
                arg(&mut eps),
                arg(&mut cmin),
                arg(&mut cmax),
            ],
        )?;

        Ok(MhcGateTensors {
            pre: Box::new(pre),
            post: Box::new(post),
            comb: Box::new(comb),
        })
    }
}
