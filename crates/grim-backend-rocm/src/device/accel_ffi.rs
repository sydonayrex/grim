//! grim-sonnet F9 (MIOpen) + F11 (RCCL) — real Rust FFI bindings. [see: `rust-ffi`, `libloading`, `.so`, `.rocm-3`]

use std::ffi::c_void;

use grim_tensor::Error;
use libloading::{Library, Symbol};

// ---------------------------------------------------------------------------
// F9 — MIOpen (dynamic load; no link-time .so required)
// ---------------------------------------------------------------------------

/// MIOpen status code (every function returns one).
pub type MiopenStatus = i32;
#[allow(non_upper_case_globals)]
pub const miopen_status_success: MiopenStatus = 0;

/// Opaque MIOpen handle (mirrors `#[repr(transparent)]` newtype pattern).
pub type MiopenHandle = *mut c_void;

/// Handle to a dlopen'd MIOpen library. Created once per process; `Library` [see: `probe`, `Symbol`]
pub struct MiopenLib {
    lib: Library,
}

impl MiopenLib {
    /// Probe the MIOpen SONAME chain. Returns `Err` (not a panic) if the
    pub fn load() -> Result<Self, Error> {
        // Probe newest SONAME first; fall back. Matches ZLUDA's ROCm
        let lib = unsafe { Library::new("libMIOpen.so.1") }
            .or_else(|_| unsafe { Library::new("libMIOpen.so") })
            .map_err(|e| Error::Backend(format!("MIOpen not loadable: {e}")))?;
        Ok(Self { lib })
    }

    /// Create + immediately destroy a handle to prove the C ABI resolves and [see: `Err`]
    pub fn probe(&self) -> Result<(), Error> {
        type CreateFn = unsafe extern "C" fn(*mut *mut c_void) -> MiopenStatus;
        type DestroyFn = unsafe extern "C" fn(*mut c_void) -> MiopenStatus;
        // SAFETY: `lib.get` resolves a symbol name we control; the returned [see: `Symbol`, `self.lib`]
        let create: Symbol<'_, CreateFn> = unsafe {
            self.lib
                .get(b"miopenCreate\0")
                .map_err(|e| Error::Backend(format!("MIOpen miopenCreate missing: {e}")))?
        };
        let destroy: Symbol<'_, DestroyFn> = unsafe {
            self.lib
                .get(b"miopenDestroy\0")
                .map_err(|e| Error::Backend(format!("MIOpen miopenDestroy missing: {e}")))?
        };
        let mut handle: MiopenHandle = std::ptr::null_mut();
        // SAFETY: `handle` is a local with stable address; `create` writes one [see: `self.lib`]
        let status = unsafe { create(&mut handle as *mut MiopenHandle) };
        if status != miopen_status_success {
            return Err(Error::Backend(format!(
                "MIOpen miopenCreate failed: {status}"
            )));
        }
        let status = unsafe { destroy(handle) };
        if status != miopen_status_success {
            return Err(Error::Backend(format!(
                "MIOpen miopenDestroy failed: {status}"
            )));
        }
        Ok(())
    }
}

/// One-shot MIOpen availability probe: load the lib and cycle a handle. [see: `accel_features::miopen_conv_dispatch`]
pub fn miopen_probe() -> Result<(), Error> {
    MiopenLib::load()?.probe()
}

// F11 — RCCL FFI moved to its own module `rccl` (ownership + `Drop` for
// communicators live there). Nothing RCCL-related remains in this file.

// F8 — Composable Kernel (ck_tile) GEMM used to live here as a C FFI [see: `grim_ck_gemm_f16`, `src/device/ck_gemm.cpp`]
#[cfg(test)]
mod self_tests {
    use super::miopen_probe;

    // F9 — MIOpen: probe must ERROR (not panic) here because no real
    #[test]
    fn f9_miopen_absent_errors_cleanly() {
        let r = miopen_probe();
        assert!(r.is_err(), "MIOpen must error cleanly when .so is absent");
    }
}
