//! Module-level utilities used by the `RocmDevice` impl blocks. None of [see: `linear_launch`, `as_rocm`, `dev_ptr`, `arg`]

use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
use std::sync::atomic::AtomicI32;

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

/// Grid/block dims for warp-per-row kernels (`grim_rms_norm`,
/// `grim_add_rms_norm`, `grim_softmax`): 256-thread blocks = 8 warps, each
/// warp owning one row with `__shfl_xor` reductions.
pub fn warp_rows_launch(rows: usize) -> (crate::HipDim3, crate::HipDim3) {
    const WARPS_PER_BLOCK: usize = (ROCM_COMPUTE_BLOCK as usize) / 32;
    let grid = ((rows.max(1) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK) as u32;
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

/// Canonical GPU test check. Defaults to `GRIM_GPU_TEST=1`, but also recognizes
/// legacy aliases `GRIM_RUN_GPU_TESTS=1` and `GRIM_RUN_GPU_TEST=1`.
pub fn gpu_test_enabled() -> bool {
    let check = |var: &str| {
        std::env::var(var)
            .map(|val| val == "1" || val.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    };
    check("GRIM_GPU_TEST") || check("GRIM_RUN_GPU_TESTS") || check("GRIM_RUN_GPU_TEST")
}

/// RAII guard that switches the calling thread to `ordinal` and restores the
/// previous current device on drop. Probes that call `hipSetDevice` on
/// multi-GPU boxes must use this: leaving the thread on a foreign device makes
/// subsequent `hipModuleLoad` calls bind the module to the wrong context,
/// which surfaces as `hipErrorNoBinaryForGPU` (209) at kernel-launch time.
pub struct DeviceGuard {
    prev: i32,
}

impl DeviceGuard {
    pub fn set(ordinal: i32) -> Self {
        let mut prev: i32 = 0;
        unsafe {
            let _ = crate::device::handles::hipGetDevice(&mut prev);
            let _ = crate::device::handles::hipSetDevice(ordinal);
        }
        emit_ctx_trace("DeviceGuard", ordinal, prev);
        Self { prev }
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = crate::device::handles::hipSetDevice(self.prev);
        }
    }
}

// ── Context-drift watch (gguf_multigpu_context_plan.md WI-M1/M2) ────────────

/// Process-wide latch set by the engine for the duration of a prefill pass.
/// While it is up, ANY context switch to a non-zero ordinal is traced with a
/// forced backtrace under `GRIM_ALLOC_TRACE`, so the setter that flips the
/// main thread's HIP context mid-forward (the ctx_dev=2 page-fault producer)
/// is named in the log even when it does not go through `DeviceGuard`.
static PREFILL_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// Mark the beginning (`true`) or end (`false`) of a prefill pass. Called by
/// the engine around `drive_prefill`; cheap enough to leave permanently wired.
pub fn set_prefill_in_flight(on: bool) {
    PREFILL_IN_FLIGHT.store(on, Ordering::SeqCst);
}

/// Current state of the drift-watch latch.
pub fn prefill_in_flight() -> bool {
    PREFILL_IN_FLIGHT.load(Ordering::SeqCst)
}

fn alloc_trace_enabled() -> bool {
    std::env::var("GRIM_ALLOC_TRACE").is_ok()
}

/// Emit one `[ctx-trace]` line plus a forced backtrace when `target` is a
/// context switch worth naming: the legacy TEMP-DIAG case (anything parking
/// on ordinal 2) and — new in WI-M2 — any switch to a non-zero device while
/// the prefill latch is up. The thread id is recorded next to `prev` so
/// cross-thread flips are obvious in the log.
///
/// Hot-path cost: the overwhelmingly common case (latch down, target != 2)
/// costs ONE atomic load and returns before any environment lookup. The
/// env::var check deliberately runs only when a trace would actually be
/// emitted; guard calls sit on per-op paths where an unconditional getenv
/// measurably perturbs timing-sensitive fused pipelines.
fn emit_ctx_trace(site: &str, target: i32, prev: i32) {
    let latch_up = prefill_in_flight();
    if target != 2 && !(latch_up && target != 0) {
        return;
    }
    if !alloc_trace_enabled() {
        return;
    }
    eprintln!(
        "[ctx-trace] {site} set({target}) prev={prev} tid={:?} prefill_latch={latch_up}",
        std::thread::current().id()
    );
    eprintln!("{}", std::backtrace::Backtrace::force_capture());
}

/// The sanctioned raw-context setter. Every `hipSetDevice` call outside
/// `DeviceGuard` (i.e. the two legitimate unguarded callers: the
/// `RocmDevice::try_new` construction path and `peer_access.rs`'s
/// save-restore pair) must route through here so the [ctx-trace] drift watch
/// sees it. Returns the raw HIP status like the FFI it wraps.
pub fn raw_set_device(ordinal: i32) -> crate::HipErrorT {
    let mut prev: i32 = 0;
    unsafe {
        let _ = crate::device::handles::hipGetDevice(&mut prev);
    }
    let status = unsafe { crate::device::handles::hipSetDevice(ordinal) };
    emit_ctx_trace("raw_set_device", ordinal, prev);
    status
}

/// Test-only launch-seam stamps used by the WI-M3 context-drift gates:
/// every kernel launch records `(self_dev, ctx_dev)` so a test can assert
/// the launching thread was not parked on a foreign device.
#[cfg(test)]
static LAUNCH_SELF_STAMP: AtomicI32 = AtomicI32::new(-1);
#[cfg(test)]
static LAUNCH_CTX_STAMP: AtomicI32 = AtomicI32::new(-1);

#[cfg(test)]
pub(crate) fn stamp_launch_context(self_dev: i32, ctx_dev: i32) {
    LAUNCH_SELF_STAMP.store(self_dev, Ordering::Relaxed);
    LAUNCH_CTX_STAMP.store(ctx_dev, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn last_launch_context() -> (i32, i32) {
    (
        LAUNCH_SELF_STAMP.load(Ordering::Relaxed),
        LAUNCH_CTX_STAMP.load(Ordering::Relaxed),
    )
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
