//! Stub for FP8 GEMM on RDNA4 (gfx1150+).
//!
//! Intentionally not implemented yet. When implemented, this kernel will
//! provide a RocmDevice dispatchable op that multiplies two FP16 matrices
//! using the BF16→FP8 fast-path path through RDNA4 matrix cores.
//!
//! The plan requires `train_step()` to call `gemm_rgba8_16x16()` from this
//! module when `quant_mode == QuantMode::Fp8Native && arch >= "gfx1150"`.
//! Until the kernel source is written, `gemm_rgba8_16x16()` panics.

use grim_tensor::BackendDevice;

/// Multiply two FP16 matrices using RDNA4 BF16→FP8 fast path.
///
/// Returns `Err(Error::Unimplemented)` because the HIPRTC JIT kernel for RDNA4 FP8
/// matrix cores is not yet implemented.
pub fn gemm_rgba8_16x16(
    _device: &dyn BackendDevice,
    _a: &dyn grim_tensor::backend::BackendStorage,
    _b: &dyn grim_tensor::backend::BackendStorage,
    _c: &mut dyn grim_tensor::backend::BackendStorage,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<(), grim_tensor::error::Error> {
    Err(grim_tensor::error::Error::Unimplemented(
        "[fp8_gemm_rdna4] fp8 gemm kernel on RDNA4 is not yet implemented".into(),
    ))
}

