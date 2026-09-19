//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! Dot-product GEMV launchers (dot4/dot2/dot8), fp32 GEMV, and activation quantization.

use std::ffi::c_void;


use grim_tensor::error::{Error, Result};
use grim_tensor::{ BackendStorage };

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{ HipDim3, arg };

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
            eprintln!("[trace] quantize_q8_1 src={:#x} dst={:#x} k={k} m={m}", { src_ptr }, { dst_ptr });
        }
        let handle = self.launch_compute_kernel(
            "grim_quantize_q8_1",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut sptr),
                arg(&mut dptr),
                arg(&mut kk),
                arg(&mut mm),
            ],
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
        let mut guard = self
            .q81_event
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
    pub(crate) fn launch_dot4_q80_q81_gemv(
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
            eprintln!("[trace] dot4_q80_q81 act={:p} w={:p} out={:p} n={n} k={k}",
                act_q81.device_ptr.unwrap_or(0) as *const std::ffi::c_void,
                b_storage.device_ptr.unwrap_or(0) as *const std::ffi::c_void,
                out_storage.device_ptr.unwrap_or(0) as *const std::ffi::c_void);
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
        &self, act_q81: &RocmStorage, b_storage: &RocmStorage,
        out_storage: &RocmStorage, m: usize, n: usize, k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_q81.device_ptr
            .ok_or_else(|| Error::Backend("dot4_q2k_q81_gemv: act_q81 has no device ptr".into()))?;
        let b_ptr = b_storage.device_ptr
            .ok_or_else(|| Error::Backend("dot4_q2k_q81_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr
            .ok_or_else(|| Error::Backend("dot4_q2k_q81_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr; let mut bptr = b_ptr; let mut optr = out_ptr;
        let mut mm = m as i32; let mut nn = n as i32; let mut kk = k as i32;
        self.launch_compute_kernel("grim_dot4_q2k_q81_gemv", grid_dim, block_dim,
            &mut [arg(&mut aptr), arg(&mut bptr), arg(&mut optr), arg(&mut mm), arg(&mut nn), arg(&mut kk)])
    }

    /// Phase 4.5f: Q3_K x Q8_1 GEMV via sudot4 + two-dot + sign correction (RDNA3/4).
    pub(crate) fn launch_dot4_q3k_q81_gemv(
        &self, act_q81: &RocmStorage, b_storage: &RocmStorage,
        out_storage: &RocmStorage, m: usize, n: usize, k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = act_q81.device_ptr
            .ok_or_else(|| Error::Backend("dot4_q3k_q81_gemv: act_q81 has no device ptr".into()))?;
        let b_ptr = b_storage.device_ptr
            .ok_or_else(|| Error::Backend("dot4_q3k_q81_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr
            .ok_or_else(|| Error::Backend("dot4_q3k_q81_gemv: out has no device ptr".into()))?;
        let grid_x = (n as u32).div_ceil(4);
        let grid_dim = HipDim3::new(grid_x, m as u32, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut aptr = a_ptr; let mut bptr = b_ptr; let mut optr = out_ptr;
        let mut mm = m as i32; let mut nn = n as i32; let mut kk = k as i32;
        self.launch_compute_kernel("grim_dot4_q3k_q81_gemv", grid_dim, block_dim,
            &mut [arg(&mut aptr), arg(&mut bptr), arg(&mut optr), arg(&mut mm), arg(&mut nn), arg(&mut kk)])
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
        let dst_codes_ptr = dst_codes
            .device_ptr
            .ok_or_else(|| Error::Backend("quantize_u4_group128: dst_codes has no device ptr".into()))?;
        let dst_scales_ptr = dst_scales
            .device_ptr
            .ok_or_else(|| Error::Backend("quantize_u4_group128: dst_scales has no device ptr".into()))?;
        let dst_sums_ptr = dst_sums
            .device_ptr
            .ok_or_else(|| Error::Backend("quantize_u4_group128: dst_sums has no device ptr".into()))?;

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
