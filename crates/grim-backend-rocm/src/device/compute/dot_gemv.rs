//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! Dot-product GEMV launchers (dot4/dot2/dot8), fp32 GEMV, and activation quantization.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};
use grim_tensor::BackendStorage;

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{arg, HipDim3};

impl RocmDevice {
    /// Launch grim_quantize_q8_1 activation quantizer.
    pub fn launch_quantize_q8_1(
        &self,
        src: &RocmStorage,
        dst: &RocmStorage,
        m: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let src_ptr = src
            .device_ptr
            .ok_or_else(|| Error::Backend("quantize_q8_1: src has no device ptr".into()))?;
        let dst_ptr = dst
            .device_ptr
            .ok_or_else(|| Error::Backend("quantize_q8_1: dst has no device ptr".into()))?;
        let n_q_blocks = k / 32;
        let total_blocks = (n_q_blocks * m) as u32;
        let grid_dim = HipDim3::new(total_blocks, 1, 1);
        let block_dim = HipDim3::new(256, 1, 1);
        let mut sptr = src_ptr;
        let mut dptr = dst_ptr;
        let mut kk = k as i32;
        let mut mm = m as i32;
        if std::env::var("GRIM_TRACE_FUSED_QKV").is_ok() {
            eprintln!(
                "[trace] quantize_q8_1 src={:#x} dst={:#x} k={k} m={m}",
                { src_ptr },
                { dst_ptr }
            );
        }
        let handle = self.launch_compute_kernel(
            "grim_quantize_q8_1",
            grid_dim,
            block_dim,
            &mut [arg(&mut sptr), arg(&mut dptr), arg(&mut kk), arg(&mut mm)],
        )?;

        // P1-1 (PLAN-improve-grim-perf): real dependency edge for the
        // quantize -> dot4 GEMV producer/consumer pair, replacing the old
        // `if !is_rdna34 { synchronize() }` arch-gated guess. The event is
        // recorded on the quantize launch's stream; the consuming GEMV's
        // stream waits on it before its own launch. Correct under eager AND
        // capture (the wait becomes a DAG edge), on every arch, and removes
        // the host-side stall entirely.
        let ev = self.q81_quant_event();
        if !ev.is_null() {
            // SAFETY: ev created via hipEventCreate; handle is a live stream.
            unsafe { crate::hipEventRecord(ev, handle) };
        }
        Ok(handle)
    }

    /// P1-1: lazily-created, per-device event used to order the
    /// quantize_q8_1 producer against its dot4 GEMV consumer. Reused across
    /// launches (hipEventRecord re-arms a recorded event).
    fn q81_quant_event(&self) -> *mut c_void {
        let mut guard = self.q81_event.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let mut new_ev: *mut c_void = std::ptr::null_mut();
            let res = unsafe { crate::hipEventCreate(&mut new_ev) };
            if res != 0 {
                return std::ptr::null_mut();
            }
            *guard = Some(new_ev);
        }
        guard.unwrap_or(std::ptr::null_mut())
    }

    /// SPEED-DOT: Q8_0 x Q8_1 GEMV via V_DOT4_I32_IU8 (RDNA3/4).
    pub fn launch_dot4_q80_q81_gemv(
        &self,
        act_q81: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // P1-1: wait for the quantize producer's event (no-op once complete).
        let ev = self.q81_quant_event();
        if !ev.is_null() {
            // SAFETY: ev is a live event; active_stream is the device's stream.
            unsafe { crate::hipStreamWaitEvent(self.active_stream(), ev, 0) };
        }
        if std::env::var("GRIM_TRACE_FUSED_QKV").is_ok() {
            eprintln!(
                "[trace] dot4_q80_q81 act={:p} w={:p} out={:p} n={n} k={k}",
                act_q81.device_ptr.unwrap_or(0) as *const std::ffi::c_void,
                b_storage.device_ptr.unwrap_or(0) as *const std::ffi::c_void,
                out_storage.device_ptr.unwrap_or(0) as *const std::ffi::c_void
            );
        }
        let a_ptr = act_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_q81_gemv: act_q81 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_q81_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_q81_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q80_q81_gemv",
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

    /// PLAN 2: Fused activation quantize + dot4 Q8_0 GEMV directly from f32 activations.
    pub fn launch_dot4_q80_f32act_gemv(
        &self,
        act_f32: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_f32
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_f32act_gemv: act_f32 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_f32act_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_f32act_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q80_f32act_gemv",
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

    /// PLAN 4 Task 3: norm-fused single-projection dot4 GEMV directly from
    /// the residual stream. Bit-identical to separate rms_norm_into +
    /// f32act launches. Graph-capture safe (caller-owned buffers only).
    pub fn launch_dot4_q80_norm_f32act_gemv_into(
        &self,
        res: &RocmStorage,
        gamma: &RocmStorage,
        eps: f32,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if k % 32 != 0 {
            return Err(Error::Backend(format!(
                "dot4_q80_norm_f32act: K must be 32-aligned (k={k})"
            )));
        }
        let res_ptr = res
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_norm_f32act: res has no device ptr".into()))?;
        let gamma_ptr = gamma
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_norm_f32act: gamma has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_norm_f32act: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_norm_f32act: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let (mut resptr, mut gammaptr, mut epsv) = (res_ptr, gamma_ptr, eps);
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q80_norm_f32act_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut resptr),
                arg(&mut gammaptr),
                arg(&mut epsv),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )?;
        Ok(())
    }

    /// Graph fusion: residual add + RMSNorm + Q8_0 Gate/Up GEMV + SiLU.
    pub fn fused_add_rms_norm_gate_up_silu_dot4_into(
        &self,
        base: &RocmStorage,
        attn: &RocmStorage,
        gamma: &RocmStorage,
        eps: f32,
        wg: &RocmStorage,
        wu: &RocmStorage,
        residual_out: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if k % 32 != 0 {
            return Err(Error::Backend(format!(
                "fused add+rms+gateup: K must be 32-aligned (k={k})"
            )));
        }
        let base_ptr = base
            .device_ptr
            .ok_or_else(|| Error::Backend("fused add+rms+gateup: base has no device ptr".into()))?;
        let attn_ptr = attn
            .device_ptr
            .ok_or_else(|| Error::Backend("fused add+rms+gateup: attn has no device ptr".into()))?;
        let gamma_ptr = gamma.device_ptr.ok_or_else(|| {
            Error::Backend("fused add+rms+gateup: gamma has no device ptr".into())
        })?;
        let wg_ptr = wg
            .device_ptr
            .ok_or_else(|| Error::Backend("fused add+rms+gateup: gate has no device ptr".into()))?;
        let wu_ptr = wu
            .device_ptr
            .ok_or_else(|| Error::Backend("fused add+rms+gateup: up has no device ptr".into()))?;
        let residual_ptr = residual_out.device_ptr.ok_or_else(|| {
            Error::Backend("fused add+rms+gateup: residual output has no device ptr".into())
        })?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("fused add+rms+gateup: output has no device ptr".into()))?;
        let grid_dim = HipDim3::new((n as u32).div_ceil(4), m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let (mut baseptr, mut attnptr, mut gammaptr) = (base_ptr, attn_ptr, gamma_ptr);
        let (mut wgptr, mut wuptr) = (wg_ptr, wu_ptr);
        let (mut residualptr, mut outptr) = (residual_ptr, out_ptr);
        let mut epsv = eps;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_add_rms_norm_gate_up_silu_q80_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut baseptr),
                arg(&mut attnptr),
                arg(&mut gammaptr),
                arg(&mut epsv),
                arg(&mut wgptr),
                arg(&mut wuptr),
                arg(&mut residualptr),
                arg(&mut outptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )?;
        Ok(())
    }

    /// One-wave 16-output tile experiment for residual-add Q8_0 GEMV.
    pub fn launch_dot4_q80_f32act_add_gemv_tile16(
        &self,
        act_f32: &RocmStorage,
        b_storage: &RocmStorage,
        residual: Option<&RocmStorage>,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        if k % 32 != 0 {
            return Err(Error::Backend(format!(
                "dot4_q80_f32act_add_tile16: K must be 32-aligned (k={k})"
            )));
        }
        let a_ptr = act_f32.device_ptr.ok_or_else(|| {
            Error::Backend("dot4_q80_f32act_add_tile16: act_f32 has no device ptr".into())
        })?;
        let b_ptr = b_storage.device_ptr.ok_or_else(|| {
            Error::Backend("dot4_q80_f32act_add_tile16: weight has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("dot4_q80_f32act_add_tile16: out has no device ptr".into())
        })?;
        let residual_ptr = residual.and_then(|r| r.device_ptr).unwrap_or(0);
        let grid_dim = HipDim3::new((n as u32).div_ceil(16), m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let (mut aptr, mut bptr, mut resptr, mut optr) = (a_ptr, b_ptr, residual_ptr, out_ptr);
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q80_f32act_add_gemv_tile16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut resptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Phase D.2: Fused activation quantize + dot4 Q8_0 GEMV with optional residual add epilogue directly from f32 activations.
    /// When residual is Some, computes C = residual + A * B. Supports in-place addition when residual is out_storage.
    pub fn launch_dot4_q80_f32act_add_gemv(
        &self,
        act_f32: &RocmStorage,
        b_storage: &RocmStorage,
        residual: Option<&RocmStorage>,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        if matches!(
            std::env::var("GRIM_DOT4_TILE16").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        ) {
            return self.launch_dot4_q80_f32act_add_gemv_tile16(
                act_f32,
                b_storage,
                residual,
                out_storage,
                m,
                n,
                k,
            );
        }
        let a_ptr = act_f32
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_f32act_add_gemv: act_f32 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_f32act_add_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q80_f32act_add_gemv: out has no device ptr".into()))?;
        let res_ptr = match residual {
            Some(r) => r
                .device_ptr
                .ok_or_else(|| Error::Backend("dot4_q80_f32act_add_gemv: residual has no device ptr".into()))?,
            None => 0,
        };
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut resptr = res_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q80_f32act_add_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut resptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Phase D.2 wrapper: fused activation quantize + QKV dot4 GEMV in a single kernel.
    /// Takes f32 activation input directly, bypassing standalone launch_quantize_q8_1.
    pub fn fused_qkv_dot4_into(
        &self,
        a: &dyn BackendStorage,
        wq: &RocmStorage,
        wk: &RocmStorage,
        wv: &RocmStorage,
        q_out: &RocmStorage,
        k_out: &RocmStorage,
        v_out: &RocmStorage,
        n_q: usize,
        n_kv: usize,
        k: usize,
        act_q81: &RocmStorage,
    ) -> Result<()> {
        let a_rocm = a
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("fused_qkv_dot4_into: a not RocmStorage".into()))?;
        let m = a.shape().elem_count() / k.max(1);
        let dot_fused_ok = k != 0
            && k % 32 == 0
            && self.supports_dot4()
            && !matches!(
                std::env::var("GRIM_DOT_GEMV").as_deref(),
                Ok("0" | "false" | "off")
            );
        if dot_fused_ok {
            self.launch_dot4_qkv_q80_f32act_gemv_into(
                a_rocm, wq, wk, wv, q_out, k_out, v_out, m, n_q, n_kv, k,
            )
        } else {
            self.launch_quantize_q8_1(a_rocm, act_q81, m, k)?;
            self.launch_dot4_qkv_q80_gemv_into(
                act_q81, wq, wk, wv, q_out, k_out, v_out, m, n_q, n_kv, k,
            )
        }
    }

    /// Phase D.2: one dot4 GEMV launch covering the three QKV projections directly from f32 activation.
    pub fn launch_dot4_qkv_q80_f32act_gemv_into(
        &self,
        act_f32: &RocmStorage,
        wq: &RocmStorage,
        wk: &RocmStorage,
        wv: &RocmStorage,
        q_out: &RocmStorage,
        k_out: &RocmStorage,
        v_out: &RocmStorage,
        m: usize,
        n_q: usize,
        n_kv: usize,
        k: usize,
    ) -> Result<()> {
        if n_q % 4 != 0 || n_kv % 4 != 0 {
            return Err(Error::Backend(format!(
                "dot4_qkv_f32act: sections must be 4-aligned (n_q={n_q}, n_kv={n_kv})"
            )));
        }
        let a_ptr = act_f32
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_f32act: act_f32 has no device ptr".into()))?;
        let wq_ptr = wq
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_f32act: wq has no device ptr".into()))?;
        let wk_ptr = wk
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_f32act: wk has no device ptr".into()))?;
        let wv_ptr = wv
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_f32act: wv has no device ptr".into()))?;
        let q_ptr = q_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_f32act: q out has no device ptr".into()))?;
        let k_ptr = k_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_f32act: k out has no device ptr".into()))?;
        let v_ptr = v_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_f32act: v out has no device ptr".into()))?;
        let grid_x = ((n_q + 2 * n_kv) as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let (mut aptr, mut wqptr, mut wkptr, mut wvptr) = (a_ptr, wq_ptr, wk_ptr, wv_ptr);
        let (mut qptr, mut kptr, mut vptr) = (q_ptr, k_ptr, v_ptr);
        let mut mm = m as i32;
        let mut nq = n_q as i32;
        let mut nkv = n_kv as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_qkv_q80_f32act_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut wqptr),
                arg(&mut wkptr),
                arg(&mut wvptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut kk),
            ],
        )?;
        Ok(())
    }

    /// PLAN 4 (DukeNukem): norm-fused QKV — rms_norm prologue + quantize +
    /// QKV dot4 GEMV in one launch. Bit-identical to separate rms_norm_into
    /// + f32act launches (same traversal, formula, quantizer). Graph-capture
    /// safe (caller-owned buffers, no H2D/sync/alloc).
    pub fn launch_dot4_qkv_q80_norm_f32act_gemv_into(
        &self,
        res: &RocmStorage,
        gamma: &RocmStorage,
        eps: f32,
        wq: &RocmStorage,
        wk: &RocmStorage,
        wv: &RocmStorage,
        q_out: &RocmStorage,
        k_out: &RocmStorage,
        v_out: &RocmStorage,
        m: usize,
        n_q: usize,
        n_kv: usize,
        k: usize,
    ) -> Result<()> {
        if n_q % 4 != 0 || n_kv % 4 != 0 {
            return Err(Error::Backend(format!(
                "dot4_qkv_norm_f32act: sections must be 4-aligned (n_q={n_q}, n_kv={n_kv})"
            )));
        }
        if k % 32 != 0 {
            return Err(Error::Backend(format!(
                "dot4_qkv_norm_f32act: KD must be 32-aligned (k={k})"
            )));
        }
        let res_ptr = res
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_norm_f32act: res has no device ptr".into()))?;
        let gamma_ptr = gamma
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_norm_f32act: gamma has no device ptr".into()))?;
        let wq_ptr = wq
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_norm_f32act: wq has no device ptr".into()))?;
        let wk_ptr = wk
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_norm_f32act: wk has no device ptr".into()))?;
        let wv_ptr = wv
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_norm_f32act: wv has no device ptr".into()))?;
        let q_ptr = q_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_norm_f32act: q out has no device ptr".into()))?;
        let k_ptr = k_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_norm_f32act: k out has no device ptr".into()))?;
        let v_ptr = v_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv_norm_f32act: v out has no device ptr".into()))?;
        let grid_x = ((n_q + 2 * n_kv) as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let (mut resptr, mut gammaptr, mut epsv) = (res_ptr, gamma_ptr, eps);
        let (mut wqptr, mut wkptr, mut wvptr) = (wq_ptr, wk_ptr, wv_ptr);
        let (mut qptr, mut kptr, mut vptr) = (q_ptr, k_ptr, v_ptr);
        let mut mm = m as i32;
        let mut nq = n_q as i32;
        let mut nkv = n_kv as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_qkv_q80_norm_f32act_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut resptr),
                arg(&mut gammaptr),
                arg(&mut epsv),
                arg(&mut wqptr),
                arg(&mut wkptr),
                arg(&mut wvptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut kk),
            ],
        )?;
        Ok(())
    }

    /// PLAN 4 wrapper: norm-fused QKV dot4 directly from the residual stream.
    /// Caller skips the standalone `rms_norm_into`; predicates mirror
    /// `fused_qkv_dot4_into` so kill-switches (`GRIM_DOT_GEMV`, arch,
    /// alignment, Q80 weights) keep working — verified at the call site.
    pub fn fused_qkv_dot4_norm_into(
        &self,
        res: &RocmStorage,
        gamma: &RocmStorage,
        eps: f32,
        wq: &RocmStorage,
        wk: &RocmStorage,
        wv: &RocmStorage,
        q_out: &RocmStorage,
        k_out: &RocmStorage,
        v_out: &RocmStorage,
        n_q: usize,
        n_kv: usize,
        k: usize,
    ) -> Result<()> {
        self.launch_dot4_qkv_q80_norm_f32act_gemv_into(
            res,
            gamma,
            eps,
            wq,
            wk,
            wv,
            q_out,
            k_out,
            v_out,
            res.shape().elem_count() / k.max(1),
            n_q,
            n_kv,
            k,
        )
    }

    /// Phase D.2 wrapper: fused activation quantize + gate/up GEMV with SiLU epilogue directly from f32 activations.
    /// Takes f32 activation input directly, bypassing standalone launch_quantize_q8_1.
    pub fn fused_gate_up_silu_dot4_into(
        &self,
        a: &dyn BackendStorage,
        wg: &RocmStorage,
        wu: &RocmStorage,
        out: &RocmStorage,
        n: usize,
        k: usize,
        act_q81: &RocmStorage,
    ) -> Result<()> {
        let a_rocm = a
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("fused_gate_up_silu: a not RocmStorage".into()))?;
        let m = a.shape().elem_count() / k.max(1);
        let dot_fused_ok = k != 0
            && k % 32 == 0
            && self.supports_dot4()
            && !matches!(
                std::env::var("GRIM_DOT_GEMV").as_deref(),
                Ok("0" | "false" | "off")
            );
        if dot_fused_ok {
            let use_prequant = matches!(
                std::env::var("GRIM_DOT4_PREQUANT").as_deref(),
                Ok("1") | Ok("true") | Ok("on")
            );
            if use_prequant {
                self.launch_quantize_q8_1(a_rocm, act_q81, m, k)?;
                self.launch_dot4_gate_up_silu_q80_gemv_into(act_q81, wg, wu, out, m, n, k)
            } else {
                let use_256 = matches!(
                    std::env::var("GRIM_DOT4_256").as_deref(),
                    Ok("1") | Ok("true") | Ok("on")
                );
                if use_256 {
                    self.launch_dot4_gate_up_silu_q80_f32act_gemv_256_into(
                        a_rocm, wg, wu, out, m, n, k,
                    )
                } else {
                    self.launch_dot4_gate_up_silu_q80_f32act_gemv_into(a_rocm, wg, wu, out, m, n, k)
                }
            }
        } else {
            self.launch_quantize_q8_1(a_rocm, act_q81, m, k)?;
            self.launch_dot4_gate_up_silu_q80_gemv_into(act_q81, wg, wu, out, m, n, k)
        }
    }

    /// Phase D.2: fused activation quantize + gate/up GEMV + SiLU directly from f32 activation.
    pub fn launch_dot4_gate_up_silu_q80_f32act_gemv_into(
        &self,
        act_f32: &RocmStorage,
        wg: &RocmStorage,
        wu: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let a_ptr = act_f32
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_gate_up_silu_f32act: act_f32 has no device ptr".into()))?;
        let wg_ptr = wg
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_gate_up_silu_f32act: wg has no device ptr".into()))?;
        let wu_ptr = wu
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_gate_up_silu_f32act: wu has no device ptr".into()))?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_gate_up_silu_f32act: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let (mut aptr, mut wgptr, mut wuptr, mut optr) = (a_ptr, wg_ptr, wu_ptr, out_ptr);
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_gate_up_silu_q80_f32act_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut wgptr),
                arg(&mut wuptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )?;
        Ok(())
    }

    /// Eight-wave RDNA4 experiment: 256 threads per workgroup with a
    /// cross-wave LDS reduction. Opt-in through `GRIM_DOT4_256=1`.
    pub fn launch_dot4_gate_up_silu_q80_f32act_gemv_256_into(
        &self,
        act_f32: &RocmStorage,
        wg: &RocmStorage,
        wu: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let a_ptr = act_f32.device_ptr.ok_or_else(|| {
            Error::Backend("dot4_gate_up_silu_256: act_f32 has no device ptr".into())
        })?;
        let wg_ptr = wg.device_ptr.ok_or_else(|| {
            Error::Backend("dot4_gate_up_silu_256: gate has no device ptr".into())
        })?;
        let wu_ptr = wu.device_ptr.ok_or_else(|| {
            Error::Backend("dot4_gate_up_silu_256: up has no device ptr".into())
        })?;
        let out_ptr = out.device_ptr.ok_or_else(|| {
            Error::Backend("dot4_gate_up_silu_256: out has no device ptr".into())
        })?;
        let grid_dim = HipDim3::new((n as u32).div_ceil(4), m as u32, 1);
        let block_dim = HipDim3::new(256, 1, 1);
        let (mut aptr, mut wgptr, mut wuptr, mut optr) = (a_ptr, wg_ptr, wu_ptr, out_ptr);
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_gate_up_silu_q80_f32act_gemv_256",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut wgptr),
                arg(&mut wuptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )?;
        Ok(())
    }

    /// PLAN-kernel-launch-reduction Phase B: one dot4 GEMV launch covering the
    /// three QKV projections. Sections must be 4-aligned so no 4-col group
    /// spans two weight blobs. Graph-capture safe (caller-owned outputs).
    pub fn launch_dot4_qkv_q80_gemv_into(
        &self,
        act_q81: &RocmStorage,
        wq: &RocmStorage,
        wk: &RocmStorage,
        wv: &RocmStorage,
        q_out: &RocmStorage,
        k_out: &RocmStorage,
        v_out: &RocmStorage,
        m: usize,
        n_q: usize,
        n_kv: usize,
        k: usize,
    ) -> Result<()> {
        if n_q % 4 != 0 || n_kv % 4 != 0 {
            return Err(Error::Backend(format!(
                "dot4_qkv: sections must be 4-aligned (n_q={n_q}, n_kv={n_kv})"
            )));
        }
        let a_ptr = act_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv: act_q81 has no device ptr".into()))?;
        let wq_ptr = wq
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv: wq has no device ptr".into()))?;
        let wk_ptr = wk
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv: wk has no device ptr".into()))?;
        let wv_ptr = wv
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv: wv has no device ptr".into()))?;
        let q_ptr = q_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv: q out has no device ptr".into()))?;
        let k_ptr = k_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv: k out has no device ptr".into()))?;
        let v_ptr = v_out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_qkv: v out has no device ptr".into()))?;
        let grid_x = ((n_q + 2 * n_kv) as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let (mut aptr, mut wqptr, mut wkptr, mut wvptr) = (a_ptr, wq_ptr, wk_ptr, wv_ptr);
        let (mut qptr, mut kptr, mut vptr) = (q_ptr, k_ptr, v_ptr);
        let mut mm = m as i32;
        let mut nq = n_q as i32;
        let mut nkv = n_kv as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_qkv_q80_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut wqptr),
                arg(&mut wkptr),
                arg(&mut wvptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut kk),
            ],
        )?;
        Ok(())
    }

    /// PLAN-kernel-launch-reduction Phase A: fused gate+up GEMV + SiLU —
    /// one launch replaces (gate GEMV, up GEMV, silu_mul). Graph-capture safe.
    pub fn launch_dot4_gate_up_silu_q80_gemv_into(
        &self,
        act_q81: &RocmStorage,
        wg: &RocmStorage,
        wu: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let a_ptr = act_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_gate_up_silu: act_q81 has no device ptr".into()))?;
        let wg_ptr = wg
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_gate_up_silu: wg has no device ptr".into()))?;
        let wu_ptr = wu
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_gate_up_silu: wu has no device ptr".into()))?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_gate_up_silu: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let (mut aptr, mut wgptr, mut wuptr, mut optr) = (a_ptr, wg_ptr, wu_ptr, out_ptr);
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_gate_up_silu_q80_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut wgptr),
                arg(&mut wuptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )?;
        Ok(())
    }

    /// Phase 4.5c: FP8 E4M3 GEMV via V_DOT4_F32_FP8_FP8 (RDNA4 dot11-insts).
    /// A is f32 [M,K] (quantized to E4M3 in-kernel), B is E4M3 column-major [N,K].
    pub(crate) fn launch_dot4_fp8_gemv(
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
            .ok_or_else(|| Error::Backend("dot4_fp8_gemv: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_fp8_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_fp8_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_fp8_gemv",
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

    /// Phase 4.5f: Q2_K x Q8_1 GEMV via sudot4 + two-dot decomposition (RDNA3/4).
    pub(crate) fn launch_dot4_q2k_q81_gemv(
        &self,
        act_q81: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q2k_q81_gemv: act_q81 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q2k_q81_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q2k_q81_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q2k_q81_gemv",
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

    /// Phase 4.5f: Q3_K x Q8_1 GEMV via sudot4 + two-dot + sign correction (RDNA3/4).
    pub(crate) fn launch_dot4_q3k_q81_gemv(
        &self,
        act_q81: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q3k_q81_gemv: act_q81 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q3k_q81_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q3k_q81_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q3k_q81_gemv",
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

    /// SPEED-DOT: Q4_K x Q8_1 GEMV via V_DOT4_I32_IU8 (RDNA3/4).
    pub(crate) fn launch_dot4_q4k_q81_gemv(
        &self,
        act_q81: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q4k_q81_gemv: act_q81 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q4k_q81_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q4k_q81_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q4k_q81_gemv",
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

    /// SPEED-DOT: Q5_K x Q8_1 GEMV via V_DOT4_I32_IU8 (RDNA3/4).
    pub(crate) fn launch_dot4_q5k_q81_gemv(
        &self,
        act_q81: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q5k_q81_gemv: act_q81 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q5k_q81_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q5k_q81_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q5k_q81_gemv",
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

    /// SPEED-DOT: Q6_K x Q8_1 GEMV via V_DOT4_I32_IU8 (RDNA3/4).
    pub(crate) fn launch_dot4_q6k_q81_gemv(
        &self,
        act_q81: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q6k_q81_gemv: act_q81 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q6k_q81_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot4_q6k_q81_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot4_q6k_q81_gemv",
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

    /// SPEED-FP32-GEMV: custom FP32 matrix-vector multiply for the output
    /// projection (lm_head). Replaces rocBLAS which is pathologically slow at M=1.
    pub(crate) fn launch_fp32_gemv(
        &self,
        act: &dyn BackendStorage,
        weight: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act
            .device_ptr()
            .ok_or_else(|| Error::Backend("fp32_gemv: act has no device ptr".into()))?;
        let w_ptr = weight
            .device_ptr
            .ok_or_else(|| Error::Backend("fp32_gemv: weight has no device ptr".into()))?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("fp32_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut wptr = w_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_fp32_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut wptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// twinkie-zombieland P1: launch dedicated F32 M=1 GEMV into pool slot.
    pub fn launch_f32_gemv_into(
        &self,
        act: &dyn BackendStorage,
        weight: &RocmStorage,
        out: &RocmStorage,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // m = batch rows of `act` (shape [m, k]); out is [m, n].
        let m = act.shape().elem_count() / k.max(1);
        self.launch_fp32_gemv(act, weight, out, m.max(1), n, k)
    }

    /// Phase 4.5b: Launch grim_quantize_u4_group128 activation quantizer (RDNA4 gfx1200/gfx1201).
    pub fn launch_quantize_u4_group128(
        &self,
        src: &RocmStorage,
        dst_codes: &RocmStorage,
        dst_scales: &RocmStorage,
        dst_sums: &RocmStorage,
        m: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let src_ptr = src
            .device_ptr
            .ok_or_else(|| Error::Backend("quantize_u4_group128: src has no device ptr".into()))?;
        let dst_codes_ptr = dst_codes.device_ptr.ok_or_else(|| {
            Error::Backend("quantize_u4_group128: dst_codes has no device ptr".into())
        })?;
        let dst_scales_ptr = dst_scales.device_ptr.ok_or_else(|| {
            Error::Backend("quantize_u4_group128: dst_scales has no device ptr".into())
        })?;
        let dst_sums_ptr = dst_sums.device_ptr.ok_or_else(|| {
            Error::Backend("quantize_u4_group128: dst_sums has no device ptr".into())
        })?;

        if k % 128 != 0 {
            return Err(Error::Backend(format!(
                "quantize_u4_group128: K={k} must be a multiple of 128"
            )));
        }
        let n_groups = k / 128;
        let total_blocks = (n_groups * m) as u32;
        let grid_dim = HipDim3::new(total_blocks, 1, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut sptr = src_ptr;
        let mut dc_ptr = dst_codes_ptr;
        let mut ds_ptr = dst_scales_ptr;
        let mut dsum_ptr = dst_sums_ptr;
        let mut kk = k as i32;
        let mut mm = m as i32;
        self.launch_compute_kernel(
            "grim_quantize_u4_group128",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut sptr),
                arg(&mut dc_ptr),
                arg(&mut ds_ptr),
                arg(&mut dsum_ptr),
                arg(&mut kk),
                arg(&mut mm),
            ],
        )
    }

    /// Phase 4.5b: W4A4 sudot8 GEMV via V_DOT8_I32_IU4 (RDNA4 gfx1200/gfx1201).
    pub fn launch_dot8_w4a4_gemv(
        &self,
        a_codes: &RocmStorage,
        a_scales: &RocmStorage,
        a_sums: &RocmStorage,
        b_qweight: &RocmStorage,
        b_scales: &RocmStorage,
        b_zeros: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_codes_ptr = a_codes
            .device_ptr
            .ok_or_else(|| Error::Backend("dot8_w4a4_gemv: a_codes has no device ptr".into()))?;
        let a_scales_ptr = a_scales
            .device_ptr
            .ok_or_else(|| Error::Backend("dot8_w4a4_gemv: a_scales has no device ptr".into()))?;
        let a_sums_ptr = a_sums
            .device_ptr
            .ok_or_else(|| Error::Backend("dot8_w4a4_gemv: a_sums has no device ptr".into()))?;
        let b_qw_ptr = b_qweight
            .device_ptr
            .ok_or_else(|| Error::Backend("dot8_w4a4_gemv: b_qweight has no device ptr".into()))?;
        let b_sc_ptr = b_scales
            .device_ptr
            .ok_or_else(|| Error::Backend("dot8_w4a4_gemv: b_scales has no device ptr".into()))?;
        let b_zr_ptr = b_zeros
            .device_ptr
            .ok_or_else(|| Error::Backend("dot8_w4a4_gemv: b_zeros has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot8_w4a4_gemv: out has no device ptr".into()))?;

        if k % 128 != 0 {
            return Err(Error::Backend(format!(
                "dot8_w4a4_gemv: K={k} must be divisible by 128"
            )));
        }

        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);

        let mut a_c = a_codes_ptr;
        let mut a_s = a_scales_ptr;
        let mut a_sum = a_sums_ptr;
        let mut b_qw = b_qw_ptr;
        let mut b_sc = b_sc_ptr;
        let mut b_zr = b_zr_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_dot8_w4a4_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_c),
                arg(&mut a_s),
                arg(&mut a_sum),
                arg(&mut b_qw),
                arg(&mut b_sc),
                arg(&mut b_zr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// SPEED-DOT: Q8_0 GEMV via `V_DOT2_F32_f16` at M=1 (RDNA3/4).
    /// utilization vs WMMA's 1/16 at m=1. Requires K % 32 == 0 (Q8_0 layout).
    /// Activations arrive as packed fp16 (exact same precision as the WMMA
    /// path's internal cast); dot4_i32_i8 is UNSIGNED-only on gfx12, so the
    /// signed fp16 dot is the vector-dot primitive for Q8_0 codes.
    pub(crate) fn launch_dot2_q80_gemv(
        &self,
        act_f16: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_f16
            .device_ptr
            .ok_or_else(|| Error::Backend("dot2_q80_gemv: act_f16 has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot2_q80_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot2_q80_gemv: out has no device ptr".into()))?;
        let grid_dim = HipDim3::new(n as u32, 1, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot2_q80_gemv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Phase 4.5d: BF16 × BF16 GEMV via V_DOT2_F32_BF16 (RDNA3/4 dot12-insts).
    /// A is BF16 [M, K], B is row-major/transposed BF16 weights [N, K].
    pub(crate) fn launch_dot2_bf16_gemv(
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
            .ok_or_else(|| Error::Backend("dot2_bf16_gemv: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot2_bf16_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dot2_bf16_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_dot2_bf16_gemv",
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
}
