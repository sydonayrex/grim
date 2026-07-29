//! Stub for FP8 GEMM on RDNA4 (gfx1150+).
//!
//! Intentionally not implemented yet. When implemented, this kernel will
//! provide a RocmDevice dispatchable op that multiplies two FP16 matrices
//! using the BF16→FP8 fast-path path through RDNA4 matrix cores.
//!
//! The plan requires `train_step()` to call `gemm_rgba8_16x16()` from this
//! module when `quant_mode == QuantMode::Fp8Native && arch >= "gfx1150"`.
//! Until the kernel source is written, `gemm_rgba8_16x16()` panics.

use crate::{BackendDevice, DeviceTensor, Shape};

/// Multiply two FP16 matrices using RDNA4 BF16→FP8 fast path.
///
/// **Panics** with "[fp8_gemm_rdna4] stub: not yet implemented".
/// Replace this with the actual GEMM kernel dispatch when the HIPRTC
/// source is ready.
pub fn gemm_rgba8_16x16(
    _device: &dyn BackendDevice,
    _a: DeviceTensor,
    _b: DeviceTensor,
    _c: &mut DeviceTensor,
    _m: usize,
    _n: usize,
    _k: usize,
) {
    panic!("[fp8_gemm_rdna4] stub: not yet implemented");
}
