//! Free-standing helpers used by `RocmDevice::launch_compute_kernel` and a few
//! other impl methods. Kept module-level (not methods) because they don't need
//! self state — moving them out of `lib.rs` cuts a chunk of lines off the giant
//! per-device impl.
//!
//! Skill attribution:
//! - `rust-gpu-discipline` §4 — every helper returns `Result` on GPU touches; no
//!   silent fallback to a CPU path.
//! - `rocm-profiling-perf` — `upload_device_buffer` is in the per-call hot path
//!   (it allocates + uploads a scratch every call); pooling it is what Phase-3
//!   §3.1 replaces.

use std::ffi::{c_void, CString};
use std::sync::Arc;

use grim_tensor::error::{Error, Result};

use crate::{
    hipFree, hipMalloc, hipMemcpy, hipMemcpyAsync, hipStreamCreate,
    hipStreamDestroy, hipStreamSynchronize, gpu_target_flag,
    hiprtcCompileProgram, hiprtcCreateProgram, hiprtcDestroyProgram,
    hiprtcGetCode, hiprtcGetCodeSize, hiprtcGetProgramLog, hiprtcGetProgramLogSize,
    hipSuccess, HipErrorT, HipMemcpyKind, HiprtcProgram,
};

/// Memory copy that handles XNACK automatically.
/// Falls back to async copy with stream when XNACK is available.
pub fn memcpy_with_xnack_fallback(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: HipMemcpyKind,
    device_ordinal: usize,
) -> HipErrorT {
    if crate::probe_xnack(device_ordinal) {
        unsafe {
            let mut stream: *mut c_void = std::ptr::null_mut();
            let status = hipStreamCreate(&mut stream);
            if status != hipSuccess {
                return hipMemcpy(dst, src, count, kind);
            }
            let status = hipMemcpyAsync(dst, src, count, kind, stream);
            let _ = hipStreamSynchronize(stream);
            let _ = hipStreamDestroy(stream);
            status
        }
    } else {
        unsafe { hipMemcpy(dst, src, count, kind) }
    }
}

/// JIT compile HIP source to .hsaco binary.
///
/// Wraps `hiprtcCreateProgram` / `hiprtcCompileProgram` / `hiprtcGetCode`.
/// On compile failure, returns the program log via `hiprtcGetProgramLog` so the
/// caller can see which `--offload-arch`-mismatch tripped. Caller owns the
/// returned `Vec<u8>` (an `.hsaco` blob).
pub fn jit_compile_hsaco(source: &str, entry_name: &str, arch: &str) -> Result<Vec<u8>> {
    let mut prog: HiprtcProgram = std::ptr::null_mut();
    let source_cstr = CString::new(source)
        .map_err(|e| Error::Backend(format!("CString conversion failed: {}", e)))?;
    let name_cstr = CString::new(entry_name)
        .map_err(|e| Error::Backend(format!("CString conversion failed: {}", e)))?;

    unsafe {
        let status = hiprtcCreateProgram(
            &mut prog,
            source_cstr.as_ptr(),
            name_cstr.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if status != hipSuccess {
            return Err(Error::Backend(format!("hiprtcCreateProgram failed: {}", status)));
        }

        let options_c = vec![
            CString::new("--std=c++14").unwrap(),
            gpu_target_flag(arch),
        ];
        let options_ptrs: Vec<*const i8> = options_c.iter().map(|c| c.as_ptr()).collect();

        let status = hiprtcCompileProgram(prog, options_ptrs.len() as i32, options_ptrs.as_ptr());

        if status != hipSuccess {
            let mut log_size: usize = 0;
            let _ = hiprtcGetProgramLogSize(prog, &mut log_size);
            let mut log: Vec<u8> = vec![0u8; log_size.max(1)];
            let _ = hiprtcGetProgramLog(prog, log.as_mut_ptr() as *mut i8);
            let log_string = String::from_utf8_lossy(&log);
            let _ = hiprtcDestroyProgram(&mut prog);
            return Err(Error::Backend(format!(
                "hiprtcCompileProgram failed (status {}): {}",
                status, log_string
            )));
        }

        let mut code_size: usize = 0;
        let status = hiprtcGetCodeSize(prog, &mut code_size);
        if status != hipSuccess {
            let _ = hiprtcDestroyProgram(&mut prog);
            return Err(Error::Backend(format!("hiprtcGetCodeSize failed: {}", status)));
        }

        let mut code_bytes = vec![0u8; code_size];
        let status = hiprtcGetCode(prog, code_bytes.as_mut_ptr() as *mut i8);
        if status != hipSuccess {
            let _ = hiprtcDestroyProgram(&mut prog);
            return Err(Error::Backend(format!("hiprtcGetCode failed: {}", status)));
        }

        let _ = hiprtcDestroyProgram(&mut prog);

        Ok(code_bytes)
    }
}

/// Allocate a device-side scratch buffer, copy `data` into it, and return the
/// raw device pointer. Caller is responsible for `hipFree` on the returned ptr.
///
/// Used by the path-RHS compute ops (matmul_batched pre-packing, etc.) where a
/// pooled-scratch BPool would otherwise force a fresh allocation. Phase-3 §3.1
/// `upload_to_scratch` is the spec-blessed successor that uses the pool.
pub fn upload_device_buffer<T: Copy>(data: &[T]) -> Result<*mut c_void> {
    let bytes = data.len() * std::mem::size_of::<T>();
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let res = unsafe { hipMalloc(&mut ptr, bytes) };
    if res != hipSuccess {
        return Err(Error::Backend(format!("hipMalloc (scratch) failed: {}", res)));
    }
    if !data.is_empty() {
        let res = unsafe {
            hipMemcpy(
                ptr,
                data.as_ptr() as *const c_void,
                bytes,
                HipMemcpyKind::HostToDevice,
            )
        };
        if res != hipSuccess {
            unsafe { hipFree(ptr); }
            return Err(Error::Backend(format!("hipMemcpy (scratch) failed: {}", res)));
        }
    }
    Ok(ptr)
}

// Suppress 'unused' warning for Arc import when only used inside a cfg-gated path.
#[allow(dead_code)]
fn _arc_pinned(_x: Arc<()>) {}
