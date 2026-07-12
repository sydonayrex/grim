//! rocBLAS FFI surface: handle type, enums (operation / datatype / algo),
//! flag typedef, status constants, and the raw rocBLAS function
//! declarations (`rocblas_create_handle`, `rocblas_set_stream`,
//! `rocblas_sgemm`, `rocblas_gemm_ex`, `rocblas_gemm_strided_batched_ex`).
//! Also hosts the small `ArithType -> rocblas_datatype` mapping
//! helpers that convert Grim's dtype vocabulary to rocBLAS'.
//!
//! All of this is consumed by `RocmDevice::matmul` and friends in
//! lib.rs (and by `device::gemm_tuning::lookup_solution_index`).
//! Re-exported at the crate root so callers see no API change.
//!
//! Skill attribution:
//! - `rust-ffi` — the SAFETY rationale for the `unsafe impl Send` /
//!   Sync lines on `RoclabsHandle` is the same as on `RocmHandle`:
//!   opaque platform resources, library owns the lock.
//! - `rocm-quantization-inference` — `rocblas_datatype` /
//!   `rocblas_gemm_algo` are the schema that Intel hipBLASLt /
//!   rocBLAS GEMM dispatch reads at runtime; wrong discriminants
//!   silently zero outputs (rocblas_status_invalid_value).
//! - `rust-gpu-discipline` §3 — explicit `Result` mapping for
//!   `rocblas_status`; no silent fallback.

use std::ffi::c_void;

use grim_tensor::{ArithType, Error, Result};

/// rocBLAS status code (every function returns one).
pub type Rocblstatus = i32;

/// Success return code for rocBLAS FFI calls.
pub const rocblas_status_success: Rocblstatus = 0;

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RocblasOperation {
    None = 111,                  // rocblas_operation_none
    Transpose = 112,             // rocblas_operation_transpose
    ConjugateTranspose = 113,    // rocblas_operation_conjugate_transpose
}

/// rocBLAS dim type (m/n/k/lda/ldb/etc.).
pub type RocblasInt = i32;

/// Opaque rocBLAS handle. rocBLAS handles are thread-safe per the lib docs.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct RoclabsHandle(pub *mut c_void);

unsafe impl Send for RoclabsHandle {}
unsafe impl Sync for RoclabsHandle {}

#[link(name = "rocblas", kind = "dylib")]
unsafe extern "C" {
    pub fn rocblas_create_handle(handle: *mut RoclabsHandle) -> Rocblstatus;
    pub fn rocblas_destroy_handle(handle: RoclabsHandle) -> Rocblstatus;
    pub fn rocblas_set_stream(handle: RoclabsHandle, stream: *mut c_void) -> Rocblstatus;

    /// Special-case FP32 GEMM (16 args). Used by the FP32-only
    /// paths of `RocmDevice::matmul` (the legacy dispatch before
    /// Item 0's `gemm_ex` was wired in).
    pub fn rocblas_sgemm(
        handle: RoclabsHandle,
        trans_a: RocblasOperation,
        trans_b: RocblasOperation,
        m: RocblasInt,
        n: RocblasInt,
        k: RocblasInt,
        alpha: *const f32,
        a: *const f32,
        lda: RocblasInt,
        b: *const f32,
        ldb: RocblasInt,
        beta: *const f32,
        c: *mut f32,
        ldc: RocblasInt,
    ) -> Rocblstatus;
}

/// rocBLAS data types. Discriminants match the official rocBLAS
/// `rocblas_datatype` enum (see rocblas/rocblas-types.h). Passing the
/// wrong integer here silently yields rocblas_status_invalid_value
/// and zeroes the output.
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum rocblas_datatype {
    f16_r = 150,
    f32_r = 151,
    f64_r = 152,
    f16_c = 153,
    f32_c = 154,
    f64_c = 155,
    i8_r = 160,
    u8_r = 161,
    i32_r = 162,
    u32_r = 163,
    i8_c = 164,
    u8_c = 165,
    i32_c = 166,
    u32_c = 167,
    bf16_r = 168,
    bf16_c = 169,
    invalid = 255,
}

/// GEMM algorithm selector (rocblas_gemm_algo).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum rocblas_gemm_algo {
    standard = 0x0,
    solution_index = 0x1,
}

/// GEMM control flags (rocblas_gemm_flags). Bitmask; 0x0 = none.
pub type rocblas_gemm_flags = u32;
pub const ROCBLAS_GEMM_FLAGS_NONE: rocblas_gemm_flags = 0x0;

unsafe extern "C" {
    // gemm_ex — signature matches rocBLAS exactly (29 args). Used by
    // RocmDevice::matmul (Item 0 / Item 7 of the spec).
    pub fn rocblas_gemm_ex(
        handle: RoclabsHandle,
        trans_a: RocblasOperation,
        trans_b: RocblasOperation,
        m: RocblasInt,
        n: RocblasInt,
        k: RocblasInt,
        alpha: *const c_void,
        a: *const c_void,
        a_type: rocblas_datatype,
        lda: RocblasInt,
        b: *const c_void,
        b_type: rocblas_datatype,
        ldb: RocblasInt,
        beta: *const c_void,
        c: *const c_void,
        c_type: rocblas_datatype,
        ldc: RocblasInt,
        d: *mut c_void,
        d_type: rocblas_datatype,
        ldd: RocblasInt,
        compute_type: rocblas_datatype,
        algo: rocblas_gemm_algo,
        solution_index: RocblasInt,
        flags: rocblas_gemm_flags,
    ) -> Rocblstatus;

    // gemm_strided_batched_ex — 29 args, batch_count inserted before
    // compute_type, stride_a..stride_d inserted after each lda/ldb/ldc/ldd.
    // rocblas_stride is int64_t.
    pub fn rocblas_gemm_strided_batched_ex(
        handle: RoclabsHandle,
        trans_a: RocblasOperation,
        trans_b: RocblasOperation,
        m: RocblasInt,
        n: RocblasInt,
        k: RocblasInt,
        alpha: *const c_void,
        a: *const c_void,
        a_type: rocblas_datatype,
        lda: RocblasInt,
        stride_a: i64,
        b: *const c_void,
        b_type: rocblas_datatype,
        ldb: RocblasInt,
        stride_b: i64,
        beta: *const c_void,
        c: *const c_void,
        c_type: rocblas_datatype,
        ldc: RocblasInt,
        stride_c: i64,
        d: *mut c_void,
        d_type: rocblas_datatype,
        ldd: RocblasInt,
        stride_d: i64,
        batch_count: RocblasInt,
        compute_type: rocblas_datatype,
        algo: rocblas_gemm_algo,
        solution_index: RocblasInt,
        flags: rocblas_gemm_flags,
    ) -> Rocblstatus;
}

/// Maps Grim `ArithType` to the corresponding `rocblas_datatype` enum
/// value. Falls back to `f32_r` for unknown or unsupported types.
pub fn arith_to_rocblas_dtype(arith: ArithType) -> rocblas_datatype {
    match arith {
        ArithType::F32 => rocblas_datatype::f32_r,
        ArithType::F16 => rocblas_datatype::f16_r,
        ArithType::BF16 => rocblas_datatype::bf16_r,
        ArithType::I64 | ArithType::U32 => rocblas_datatype::i32_r,
        ArithType::U8 => rocblas_datatype::u8_r,
    }
}

/// Maps Grim `ArithType` to the rocBLAS compute (accumulation) datatype.
/// Mixed-precision GEMMs accumulate in FP32 regardless of the input
/// precision (FP16/BF16 -> FP32) for numerical stability.
pub fn arith_to_compute_dtype(_arith: ArithType) -> rocblas_datatype {
    rocblas_datatype::f32_r
}

/// Wrap a rocBLAS status code into the crate `Result` type.
pub fn status_to_result(status: Rocblstatus, op: &'static str) -> Result<()> {
    if status == rocblas_status_success {
        Ok(())
    } else {
        Err(Error::Backend(format!("rocBLAS {op} failed with status {status}")))
    }
}

#[cfg(test)]
mod rocblas_self_tests {
    use super::*;

    #[test]
    fn arith_f32_maps_to_f32_r() {
        assert_eq!(arith_to_rocblas_dtype(ArithType::F32), rocblas_datatype::f32_r);
    }

    #[test]
    fn arith_f16_maps_to_f16_r() {
        assert_eq!(arith_to_rocblas_dtype(ArithType::F16), rocblas_datatype::f16_r);
    }

    #[test]
    fn arith_bf16_maps_to_bf16_r() {
        assert_eq!(arith_to_rocblas_dtype(ArithType::BF16), rocblas_datatype::bf16_r);
    }

    #[test]
    fn arith_u8_maps_to_u8_r() {
        assert_eq!(arith_to_rocblas_dtype(ArithType::U8), rocblas_datatype::u8_r);
    }

    #[test]
    fn rocblas_operation_none_value_is_111() {
        assert_eq!(RocblasOperation::None as i32, 111);
    }

    #[test]
    fn status_zero_is_success() {
        assert!(status_to_result(0, "test").is_ok());
    }

    #[test]
    fn status_nonzero_lifts_to_err() {
        assert!(status_to_result(-1, "test").is_err());
    }
}
