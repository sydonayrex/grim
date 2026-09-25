//! GRAVE Phase 4 — fused GDN-2 launcher (one launch per GDL layer).
//!
//! Wraps `kernels::gla_kernels::GLA_KERNEL_SOURCE` (`grim_gla_state_update_output`)
//! in the same capture-safe shape as `fused_ops.rs`: the `_into` variant takes
//! caller-provided `out` + in-place `state` and allocates nothing, so a
//! surrounding `begin/end_capture` bracket records a real node.
//!
//! Dims contract (host-validated, perimeter defense): `dk == 64`, `dv == 64`,
//! `heads >= 1`. Other shapes return a clean backend error — never a silent
//! wrong-math launch (the kernel early-returns on `dv > 64` as backstop).

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::kernels::gla_kernels::{gla_fused_enabled, gla_state_fp8_enabled};
use crate::memory::storage::RocmStorage;
use crate::{arg, as_rocm, dev_ptr, dtype_f32, HipDim3, RocmHandle};

/// All device pointers for one fused GDN-2 launch. Layouts: q/k/alpha/b
/// `[batch, heads, dk]`, v/w/norm_w `[batch, heads, dv]`, gate
/// `[batch, heads]`, state `[batch, heads, dk, dv]` (in-place),
/// out `[batch, heads, dv]`.
pub struct GlaLaunchArgs<'a> {
    pub q: &'a RocmStorage,
    pub k: &'a RocmStorage,
    pub v: &'a RocmStorage,
    pub alpha: &'a RocmStorage,
    pub b_gate: &'a RocmStorage,
    pub w_gate: &'a RocmStorage,
    pub norm_w: &'a RocmStorage,
    pub out_gate: &'a RocmStorage,
    pub state: &'a RocmStorage,
    pub out: &'a RocmStorage,
    pub heads: usize,
    pub batch: usize,
    pub dk: usize,
    pub dv: usize,
    pub eps: f32,
}

impl RocmDevice {
    /// Fused GDN-2 step writing into caller `out`, updating `state` in place.
    /// Returns the stream pointer for capture chaining (mirrors
    /// `launch_fused_qkv_dot4_into`).
    pub fn launch_gla_state_update_output_into(&self, a: &GlaLaunchArgs) -> Result<*mut c_void> {
        if !gla_fused_enabled() {
            return Err(Error::Backend(
                "launch_gla: fused path disabled via GRIM_GLA_FUSED=0 (split path not yet implemented)".into(),
            ));
        }
        if gla_state_fp8_enabled() {
            return Err(Error::Backend(
                "launch_gla: GRIM_GLA_STATE_FP8=1 needs the FP8 parity re-run (plan §Phase 4.5)"
                    .into(),
            ));
        }
        // Perimeter defense: exact dims, valid pointers, LDS-resident shape.
        if a.dk != 64 || a.dv != 64 {
            return Err(Error::Backend(format!(
                "launch_gla: need dk==64 && dv==64 (LDS fast path), got dk={} dv={}",
                a.dk, a.dv
            )));
        }
        if a.heads == 0 || a.batch == 0 {
            return Err(Error::Backend(
                "launch_gla: heads and batch must be >= 1".into(),
            ));
        }
        let n = a.batch;
        for (name, s, want) in [
            ("q", a.q, n * a.heads * a.dk),
            ("k", a.k, n * a.heads * a.dk),
            ("alpha", a.alpha, n * a.heads * a.dk),
            ("b_gate", a.b_gate, n * a.heads * a.dk),
            ("v", a.v, n * a.heads * a.dv),
            ("w_gate", a.w_gate, n * a.heads * a.dv),
            ("norm_w", a.norm_w, n * a.heads * a.dv),
            ("out_gate", a.out_gate, n * a.heads),
            ("state", a.state, n * a.heads * a.dk * a.dv),
            ("out", a.out, n * a.heads * a.dv),
        ] {
            if !s.device_ptr_is_valid() {
                return Err(Error::Backend(format!(
                    "launch_gla: {name} lacks a valid device pointer"
                )));
            }
            if s.shape().elem_count() != want {
                return Err(Error::Backend(format!(
                    "launch_gla: {name} holds {} elems, need {want}",
                    s.shape().elem_count()
                )));
            }
        }
        let mut q_ptr = dev_ptr(a.q)?;
        let mut k_ptr = dev_ptr(a.k)?;
        let mut v_ptr = dev_ptr(a.v)?;
        let mut alpha_ptr = dev_ptr(a.alpha)?;
        let mut b_ptr = dev_ptr(a.b_gate)?;
        let mut w_ptr = dev_ptr(a.w_gate)?;
        let mut norm_ptr = dev_ptr(a.norm_w)?;
        let mut gate_ptr = dev_ptr(a.out_gate)?;
        let mut state_ptr = dev_ptr(a.state)?;
        let mut out_ptr = dev_ptr(a.out)?;
        let mut dk_i = a.dk as i32;
        let mut dv_i = a.dv as i32;
        let mut eps_f = a.eps;
        // Grid (heads, batch): one 64-thread block (2× wave32) per
        // (batch, head) slot — matches __launch_bounds__(64).
        let grid = HipDim3 {
            x: a.heads as u32,
            y: a.batch as u32,
            z: 1,
        };
        let block = HipDim3 { x: 64, y: 1, z: 1 };
        self.launch_compute_kernel(
            "grim_gla_state_update_output",
            grid,
            block,
            &mut [
                arg(&mut q_ptr),
                arg(&mut k_ptr),
                arg(&mut v_ptr),
                arg(&mut alpha_ptr),
                arg(&mut b_ptr),
                arg(&mut w_ptr),
                arg(&mut norm_ptr),
                arg(&mut gate_ptr),
                arg(&mut state_ptr),
                arg(&mut out_ptr),
                arg(&mut dk_i),
                arg(&mut dv_i),
                arg(&mut eps_f),
            ],
        )?;
        Ok(self.active_stream())
    }

    /// Allocating wrapper (eager path): allocates `out [batch, heads, dv]` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gla_state_update_output(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        alpha: &dyn BackendStorage,
        b_gate: &dyn BackendStorage,
        w_gate: &dyn BackendStorage,
        norm_w: &dyn BackendStorage,
        out_gate: &dyn BackendStorage,
        state: &RocmStorage,
        heads: usize,
        batch: usize,
        dk: usize,
        dv: usize,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_gla_state_update_output_into(&GlaLaunchArgs {
            q: as_rocm(q)?,
            k: as_rocm(k)?,
            v: as_rocm(v)?,
            alpha: as_rocm(alpha)?,
            b_gate: as_rocm(b_gate)?,
            w_gate: as_rocm(w_gate)?,
            norm_w: as_rocm(norm_w)?,
            out_gate: as_rocm(out_gate)?,
            state,
            out: &out_storage,
            heads,
            batch,
            dk,
            dv,
            eps,
        })?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }
}
