//! Module-level utilities used by the `RocmDevice` impl blocks. None of [see: `linear_launch`, `as_rocm`, `dev_ptr`, `arg`]

use std::ffi::{CString, c_void};

use grim_tensor::dtype::{DType, Storage as DTypeStorage};
use grim_tensor::{ArithType, BackendStorage, Error, Result};

use crate::{RocmStorage, hipGetDeviceProperties};

/// Default launch block size for 1-D elementwise launches (rotary, scale-bias,
/// copy): 256 threads. These are latency-bound elementwise ops where more
/// threads improve occupancy without register-pressure concerns.
/// On RDNA2 (gfx1036, Wave32): 256 = 8 Wave32 wavefronts.
/// On CDNA (gfx9xx, Wave64): 256 = 4 Wave64 wavefronts.
/// Fused attention kernels launch 128 threads on Wave32 (fusion.rs:78,
/// roc_device.rs:8145) and derive num_waves from blockDim.x at runtime.
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

/// Helper: turn a mutable borrow of a kernel argument into the [see: `*mut c_void`]
pub fn arg<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}

/// Build the AMD-clang hipRTC `--offload-arch=<arch>` option. Defaults [see: `gfx900`, `GRIM_GPU_TARGET`]
pub fn gpu_target_arch() -> String {
    std::env::var("GRIM_GPU_TARGET").unwrap_or_else(|_| "gfx900".into())
}

/// Query the device's real gfx target so JIT-compiled kernels always [see: `GRIM_GPU_TARGET`, `temp_env`, `hipDeviceProp_t`, `gcnArchName`]
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
                    let base: String = s
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric())
                        .collect();
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
    CString::new(format!("--offload-arch={arch}")).expect("GRIM_GPU_TARGET contains interior NUL")
}

/// True for CDNA-class targets (gfx9xx, MI-series), where Matrix-FMA (MFMA)
/// is Wave64-native. RDNA (gfx10/11/12) uses Wave32 and Wave32-only WMMA,
/// so forcing Wave64 there faults at runtime.
fn is_cdna(arch: &str) -> bool {
    arch.starts_with("gfx9")
}

/// Build compiler options list for AMD hipRTC based on detected hardware target `arch`. [see: `gfx103x`, `gfx11xx`, `gfx12xx`, `gfx9xx`]
///
/// Injects the ROCm include directory (`-I`) so that JIT-compiled HIP
/// kernels can `#include` third-party headers like `<rocwmma/rocwmma.hpp>`.
/// Without this, hipRTC has no header search path for ROCm's own includes
/// and compilation fails with "file not found" on `rocwmma`, `rccl`, etc.
pub fn hiprtc_options_for_arch(arch: &str) -> Vec<CString> {
    let mut opts = vec![
        // rocWMMA 2.x targets C++17 (`inline constexpr`, nested namespace
        // definitions, `namespace X::Y`), and its headers are pulled in by
        // kernels on gfx11/gfx12 targets. Other HIP kernels in this crate
        // are a strict subset of C++17, so --std=c++17 is safe for all.
        CString::new("--std=c++17").unwrap(),
    ];
    if is_cdna(arch) {
        // CDNA / MFMA is Wave64-native: do NOT force a wave size, let hipRTC
        // pick the 64-wide wavefront the Matrix-FMA path expects.
    } else {
        // RDNA2/3/4 (incl. gfx1036): these are Wave32-native and WMMA is
        // Wave32-only. We do NOT push `-mwavefrontsize32` here: hipRTC
        // (unlike offline clang) rejects that flag with "unknown argument",
        // which blocks JIT compilation on gfx1036 (confirmed via
        // hiprtcCompileProgram status 6). hipRTC derives the wave size from
        // `--offload-arch=<gfx>` automatically, so the flag is unnecessary
        // and harmful.
    }
    opts.push(gpu_target_flag(arch));
    // HIPRTC does not search the ROCm include tree by default. Add the
    // discovered include directory so `<rocwmma/rocwmma.hpp>` and friends
    // resolve at JIT-compile time. If discovery fails we proceed without
    // the flag (kernels that don't need ROCm headers still compile).
    if let Some(include_dir) = crate::rocm_detect::rocm_include_dir() {
        let inc_flag = format!("-I{}", include_dir.display());
        if let Ok(c) = CString::new(inc_flag) {
            opts.push(c);
        }
    }
    opts
}

/// Build the canonical F32 native dtype used by every compute op in this crate.
pub fn dtype_f32() -> DType {
    DType {
        arith: ArithType::F32,
        storage: DTypeStorage::Native,
    }
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
        let f16_dt = DType {
            arith: ArithType::F16,
            storage: DTypeStorage::Native,
        };
        assert_eq!(dtype_byte_size(&f16_dt), 2);
        let bf16_dt = DType {
            arith: ArithType::BF16,
            storage: DTypeStorage::Native,
        };
        assert_eq!(dtype_byte_size(&bf16_dt), 2);
        let i64_dt = DType {
            arith: ArithType::I64,
            storage: DTypeStorage::Native,
        };
        assert_eq!(dtype_byte_size(&i64_dt), 8);
        let u8_dt = DType {
            arith: ArithType::U8,
            storage: DTypeStorage::Native,
        };
        assert_eq!(dtype_byte_size(&u8_dt), 1);
    }

    #[test]
    fn gpu_target_flag_contains_arch() {
        let flag = gpu_target_flag("gfx1036");
        let s = flag.into_string().expect("CString → String");
        assert_eq!(s, "--offload-arch=gfx1036");
    }

    #[test]
    fn rdna_does_not_pass_rejected_wavefront_flag() {
        let opts: Vec<String> = hiprtc_options_for_arch("gfx1036")
            .into_iter()
            .map(|c| c.into_string().unwrap())
            .collect();
        // hipRTC rejects `-mwavefrontsize32` with "unknown argument"
        // (confirmed: hiprtcCompileProgram status 6 on ROCm 7.2 / gfx1036).
        // The flag is unnecessary: hipRTC derives wave size from the
        // `--offload-arch=gfx1036` target automatically.
        assert!(
            !opts.iter().any(|o| o == "-mwavefrontsize32"),
            "RDNA must not pass -mwavefrontsize32 to hipRTC (rejected): {opts:?}"
        );
        assert!(
            !opts.iter().any(|o| o == "-mwavefrontsize64"),
            "RDNA must never force Wave64: {opts:?}"
        );
    }

    #[test]
    fn cdna_uses_native_wave_size() {
        let opts: Vec<String> = hiprtc_options_for_arch("gfx90a")
            .into_iter()
            .map(|c| c.into_string().unwrap())
            .collect();
        assert!(
            !opts.iter().any(|o| o.starts_with("-mwavefrontsize")),
            "CDNA (gfx90a) must leave wave size to native MFMA: {opts:?}"
        );
    }
}
