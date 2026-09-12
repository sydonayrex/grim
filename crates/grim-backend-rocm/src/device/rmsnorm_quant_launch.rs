//! SPEED-DOT-OPFUSE launcher: fused RMSNorm + int8 q8_1 quantization.
//!
//! Own file on purpose — the dot-GEMV dispatch (device_quant.rs /
//! device_compute.rs) is under concurrent development; this keeps the fusion
//! launcher collision-free. Output is the packed q8_1 layout that
//! `grim_dot4_q80_q81_gemv` reads as A_q81, so a decode path can replace the
//! norm + quantize launch pair with this single call before the GEMV.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{arg, HipDim3};

impl RocmDevice {
    /// Fused weighted-RMSNorm + int8 q8_1 quantization in one launch
    /// (decode-shaped: single row, grid (1,1,1), block (32,1,1)).
    ///
    /// Writes `out_q81` as packed 36-byte blocks (fp16 `d` LE, fp16 `sum` LE,
    /// 32 int8 codes) — bit-identical layout to `grim_quantize_q8_1`, so
    /// `grim_dot4_q80_q81_gemv` consumes it unchanged as A_q81.
    /// `norm_cache` is an optional fp32 buffer receiving the normalized row
    /// (pass a storage with `.len() == 0`-free dummy or the real row; a null
    /// device pointer disables the write).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rmsnorm_quant_i8(
        &self,
        x: &RocmStorage,
        weight: &RocmStorage,
        eps: f32,
        k: usize,
        out_q81: &RocmStorage,
        norm_cache: Option<&RocmStorage>,
    ) -> Result<*mut c_void> {
        let x_ptr = x
            .device_ptr
            .ok_or_else(|| Error::Backend("rmsnorm_quant_i8: x has no device ptr".into()))?;
        let w_ptr = weight
            .device_ptr
            .ok_or_else(|| Error::Backend("rmsnorm_quant_i8: weight has no device ptr".into()))?;
        let out_ptr = out_q81
            .device_ptr
            .ok_or_else(|| Error::Backend("rmsnorm_quant_i8: out_q81 has no device ptr".into()))?;
        let cache_ptr = norm_cache
            .and_then(|s| s.device_ptr)
            .unwrap_or(0);

        let grid_dim = HipDim3::new(1, 1, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut xptr = x_ptr;
        let mut wptr = w_ptr;
        let mut ep = eps;
        let mut kk = k as i32;
        let mut optr = out_ptr;
        let mut cptr = cache_ptr;
        self.launch_compute_kernel(
            "grim_rmsnorm_quant_i8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut wptr),
                arg(&mut ep),
                arg(&mut kk),
                arg(&mut optr),
                arg(&mut cptr),
            ],
        )
    }
}
