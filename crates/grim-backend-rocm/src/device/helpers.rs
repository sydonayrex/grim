//! Free-standing helpers used by `RocmDevice::launch_compute_kernel` and a few [see: `lib.rs`, `rust-gpu-discipline`]

use std::ffi::{CString, c_void};
use std::sync::Arc;

use grim_tensor::error::{Error, Result};

use crate::{
    HipErrorT, HipMemcpyKind, HiprtcProgram, hipFree, hipMalloc, hipMallocManaged, hipMemcpy,
    hipMemcpyAsync, hipStreamCreate, hipStreamDestroy, hipStreamSynchronize, hipSuccess,
    hiprtcAddNameExpression, hiprtcCompileProgram, hiprtcCreateProgram, hiprtcDestroyProgram,
    hiprtcGetCode, hiprtcGetCodeSize, hiprtcGetLoweredName,
    hiprtcGetProgramLog, hiprtcGetProgramLogSize,
};

/// Convert a raw `HipErrorT` into `Result<()>`. [see: `hipMalloc`, `hipStreamSynchronize`]
#[inline]
pub fn check_hip(label: &str, res: HipErrorT) -> Result<()> {
    if res != hipSuccess {
        Err(Error::Backend(format!("{} failed: {}", label, res)))
    } else {
        Ok(())
    }
}

/// Memory copy that handles XNACK automatically.
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

/// JIT compile HIP source to .hsaco binary, returning the compiled code and
/// the *lowered* (possibly C++-mangled) kernel name that `hipModuleGetFunction`
/// requires. Some `__global__` kernels (e.g. `grim_moe_fused_grouped_fp8`) are
/// emitted mangled by hipRTC even under `extern "C"`, so callers must use the
/// lowered name, not the plain entry name, to look the function up. [see:
/// `hiprtcAddNameExpression`, `hiprtcGetLoweredName`, `hipModuleGetFunction`]
pub fn jit_compile_hsaco(source: &str, entry_name: &str, arch: &str) -> Result<(Vec<u8>, String)> {
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
            return Err(Error::Backend(format!(
                "hiprtcCreateProgram failed: {}",
                status
            )));
        }

        // Register the entry name as a name expression so we can later resolve
        // its (possibly mangled) lowered name for hipModuleGetFunction.
        let _ = hiprtcAddNameExpression(prog, name_cstr.as_ptr());

        let options_c = crate::device::util::hiprtc_options_for_arch(arch);
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
            return Err(Error::Backend(format!(
                "hiprtcGetCodeSize failed: {}",
                status
            )));
        }

        let mut code_bytes = vec![0u8; code_size];
        let status = hiprtcGetCode(prog, code_bytes.as_mut_ptr() as *mut i8);
        if status != hipSuccess {
            let _ = hiprtcDestroyProgram(&mut prog);
            return Err(Error::Backend(format!("hiprtcGetCode failed: {}", status)));
        }

        // Resolve the lowered (possibly mangled) name. If hipRTC didn't mangle
        // the entry (most kernels), this returns the plain name unchanged.
        let mut lowered_ptr: *const i8 = std::ptr::null();
        let lowered_name = if hiprtcGetLoweredName(prog, name_cstr.as_ptr(), &mut lowered_ptr)
            == hipSuccess
            && !lowered_ptr.is_null()
        {
            std::ffi::CStr::from_ptr(lowered_ptr)
                .to_string_lossy()
                .into_owned()
        } else {
            entry_name.to_string()
        };

        let _ = hiprtcDestroyProgram(&mut prog);

        Ok((code_bytes, lowered_name))
    }
}

/// Allocate a device-side scratch buffer, copy `data` into it, and return the [see: `hipFree`, `upload_to_scratch`]
pub fn upload_device_buffer<T: Copy>(data: &[T]) -> Result<*mut c_void> {
    let bytes = data.len() * std::mem::size_of::<T>();
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let mut res = unsafe { hipMalloc(&mut ptr, bytes) };
    if res != hipSuccess {
        unsafe {
            crate::hipDeviceSynchronize();
        }
        res = unsafe { hipMalloc(&mut ptr, bytes) };
    }
    if res != hipSuccess {
        // Scratch uploads are transient activation/auxiliary buffers. If
        // ordinary VRAM is exhausted, managed memory keeps the operation
        // viable and lets HIP migrate the pages used by the kernel.
        res = unsafe { hipMallocManaged(&mut ptr, bytes, 1) };
    }
    if res != hipSuccess {
        return Err(Error::Backend(format!(
            "hipMalloc (scratch) failed: {}",
            res
        )));
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
            unsafe {
                hipFree(ptr);
            }
            return Err(Error::Backend(format!(
                "hipMemcpy (scratch) failed: {}",
                res
            )));
        }
    }
    Ok(ptr)
}

// Suppress 'unused' warning for Arc import when only used inside a cfg-gated path.
#[allow(dead_code)]
fn _arc_pinned(_x: Arc<()>) {}
