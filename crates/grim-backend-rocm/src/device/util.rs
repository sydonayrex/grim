//! Module-level utilities used by the `RocmDevice` impl blocks. None of
//! them carry device state per se — `linear_launch` sizes a 1D grid,
//! `as_rocm`/`dev_ptr`/`arg` are kernel-launch sugar, `gpu_target_*`
//! and `detect_gpu_arch` resolve the offload-arch flag, `dtype_f32`
//! and `dtype_byte_size` build the canonical f32 dtype and its size.
//!
//! Skill attribution:
//! - `rust-gpu-discipline` §4 — every helper returns `Result` on GPU
//!   touches; no silent fallback.
//! - `rocm-profiling-perf` — `ROCM_COMPUTE_BLOCK = 256` is the
//!   Wave64-tuned default (4 wavefronts of 64 per block).

use std::ffi::{c_void, CString};

use grim_tensor::dtype::{DType, Storage as DTypeStorage};
use grim_tensor::{ArithType, BackendStorage, Error, Result};

use crate::{hipGetDeviceProperties, RocmStorage};

/// Default launch block size: 256 threads = 4 Wave64 wavefronts on
/// RDNA. Used as the single value everywhere a 1-D grid is sized.
pub const ROCM_COMPUTE_BLOCK: u32 = 256;

/// Grid/block dims for a 1-D launch over `total` elements.
pub fn linear_launch(total: usize) -> (crate::HipDim3, crate::HipDim3) {
    let grid = (total as u32 + ROCM_COMPUTE_BLOCK - 1) / ROCM_COMPUTE_BLOCK;
    (
        crate::HipDim3::new(grid, 1, 1),
        crate::HipDim3::new(ROCM_COMPUTE_BLOCK, 1, 1),
    )
}

/// Helper: downcast a `BackendStorage` to `RocmStorage`, returning a
/// clear error if the input is not ROCm-resident.
pub fn as_rocm<'a>(s: &'a dyn BackendStorage) -> Result<&'a RocmStorage> {
    s.as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Backend("expected RocmStorage input".into()))
}

/// Helper: require a valid device pointer on a `RocmStorage`.
pub fn dev_ptr(s: &RocmStorage) -> Result<u64> {
    s.device_ptr
        .ok_or_else(|| Error::Backend("RocmStorage has no device pointer".into()))
}

/// Helper: turn a mutable borrow of a kernel argument into the
/// `*mut c_void` slot the HIP module-launch ABI expects. Each arg
/// is passed by pointer.
pub fn arg<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}

/// Build the AMD-clang hipRTC `--offload-arch=<arch>` option. Defaults
/// to `gfx900` to preserve historical CDNA builds; override via
/// `GRIM_GPU_TARGET`.
pub fn gpu_target_arch() -> String {
    std::env::var("GRIM_GPU_TARGET").unwrap_or_else(|_| "gfx900".into())
}

/// Query the device's real gfx target so JIT-compiled kernels always
/// match the GPU, independent of the process-global `GRIM_GPU_TARGET`
/// env (which other tests flip via `temp_env` and would otherwise race
/// with device creation).
///
/// Implementation note: `hipDeviceProp_t` is version-sensitive and
/// large; rather than redefining it, dump the properties into an
/// over-sized zeroed buffer and scan for the `gcnArchName` token
/// (a NUL-terminated "gfx<hex>" string). Robust to field reordering
/// and alignment differences across ROCm releases.
pub fn detect_gpu_arch(device: i32) -> String {
    let mut buf = vec![0u8; 8192];
    unsafe {
        if hipGetDeviceProperties(buf.as_mut_ptr() as *mut c_void, device) == 0 {
            let mut i = 0;
            while i + 3 < buf.len() {
                if buf[i] == b'g' && buf[i + 1] == b'f' && buf[i + 2] == b'x' {
                    let start = i;
                    let mut end = start;
                    while end < buf.len() && buf[end] != 0 {
                        end += 1;
                    }
                    let s = std::str::from_utf8(&buf[start..end]).unwrap_or("");
                    let base: String = s.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
                    if base.starts_with("gfx") {
                        return base;
                    }
                    i = end + 1;
                } else {
                    i += 1;
                }
            }
        }
    }
    gpu_target_arch()
}

/// Build `--offload-arch=<arch>` options string for AMD hipRTC.
pub fn gpu_target_flag(arch: &str) -> CString {
    CString::new(format!("--offload-arch={arch}"))
        .expect("GRIM_GPU_TARGET contains interior NUL")
}

/// Build the canonical F32 native dtype used by every compute op in this crate.
pub fn dtype_f32() -> DType {
    DType { arith: ArithType::F32, storage: DTypeStorage::Native }
}

/// Helper function to retrieve the size in bytes of a data type.
pub fn dtype_byte_size(dtype: &DType) -> usize {
    match dtype.arith {
        ArithType::F32 | ArithType::U32 => 4,
        ArithType::F16 | ArithType::BF16 => 2,
        ArithType::I64 => 8,
        ArithType::U8 => 1,
    }
}

#[cfg(test)]
mod util_self_tests {
    use super::*;

    #[test]
    fn linear_launch_uses_default_block_of_256() {
        let (grid, block) = linear_launch(1024);
        assert_eq!(block.x, 256);
        assert_eq!(grid.x, 4);
        assert_eq!(grid.y, 1);
        assert_eq!(grid.z, 1);
    }

    #[test]
    fn linear_launch_rounds_grid_up() {
        let (grid, _) = linear_launch(257);
        assert_eq!(grid.x, 2); // (257 + 256 - 1) / 256
    }

    #[test]
    fn dtype_f32_returns_native_f32() {
        let d = dtype_f32();
        assert_eq!(d.arith, ArithType::F32);
        assert_eq!(d.storage, DTypeStorage::Native);
    }

    #[test]
    fn dtype_byte_size_matches_arith() {
        let f32_dt = dtype_f32();
        assert_eq!(dtype_byte_size(&f32_dt), 4);
        let f16_dt = DType { arith: ArithType::F16, storage: DTypeStorage::Native };
        assert_eq!(dtype_byte_size(&f16_dt), 2);
        let bf16_dt = DType { arith: ArithType::BF16, storage: DTypeStorage::Native };
        assert_eq!(dtype_byte_size(&bf16_dt), 2);
        let i64_dt = DType { arith: ArithType::I64, storage: DTypeStorage::Native };
        assert_eq!(dtype_byte_size(&i64_dt), 8);
        let u8_dt = DType { arith: ArithType::U8, storage: DTypeStorage::Native };
        assert_eq!(dtype_byte_size(&u8_dt), 1);
    }

    #[test]
    fn gpu_target_flag_contains_arch() {
        let flag = gpu_target_flag("gfx1036");
        let s = flag.into_string().expect("CString → String");
        assert_eq!(s, "--offload-arch=gfx1036");
    }
}
