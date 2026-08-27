//! Free-standing helpers used by `RocmDevice::launch_compute_kernel` and a few [see: `lib.rs`, `rust-gpu-discipline`]

use std::ffi::{CString, c_void};
use std::sync::Arc;

use grim_tensor::error::{Error, Result};

use crate::{
    HipErrorT, HipMemcpyKind, HiprtcProgram, hipFree, hipMalloc, hipMallocManaged, hipMemcpy,
    hipMemcpyAsync, hipStreamCreate, hipStreamDestroy, hipStreamSynchronize, hipSuccess,
    hiprtcAddNameExpression, hiprtcCompileProgram, hiprtcCreateProgram, hiprtcDestroyProgram,
    hiprtcGetCode, hiprtcGetCodeSize, hiprtcGetLoweredName, hiprtcGetProgramLog,
    hiprtcGetProgramLogSize,
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

///
/// Memory copy that handles XNACK automatically.
/// WI-SB6 control-plane primitive: async u32-sized copy on an EXPLICIT
/// non-blocking stream followed by synchronizing ONLY that stream. Used by
/// the resident ring so head/stop/tail traffic is never ordered behind the
/// eternally-polling worker kernel.
pub fn hip_stream_synchronize_after_copy(
    dst_dev: *mut std::ffi::c_void,
    src: *mut std::ffi::c_void,
    bytes: usize,
    kind: crate::device::handles::HipMemcpyKind,
    ordinal: usize,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    let _guard = crate::device::util::DeviceGuard::set(ordinal as i32);
    let rc = unsafe { crate::device::handles::hipMemcpyAsync(dst_dev, src, bytes, kind, stream) };
    if rc != 0 {
        return Err(Error::Backend(format!(
            "control async copy failed: hip status {rc}"
        )));
    }
    let rs = unsafe { crate::device::handles::hipStreamSynchronize(stream) };
    if rs != 0 {
        return Err(Error::Backend(format!(
            "control stream sync failed: hip status {rs}"
        )));
    }
    Ok(())
}

/// Synchronize one explicit stream (WI-SB6 shutdown join).
pub fn hip_stream_synchronize(stream: *mut std::ffi::c_void) -> Result<()> {
    let rs = unsafe { crate::device::handles::hipStreamSynchronize(stream) };
    if rs != 0 {
        return Err(Error::Backend(format!(
            "stream sync failed: hip status {rs}"
        )));
    }
    Ok(())
}

pub fn memcpy_with_xnack_fallback(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: HipMemcpyKind,
    device_ordinal: usize,
) -> HipErrorT {
    // WI-M1 context discipline: both the sync and the async+sync-stream copy
    // execute against the calling thread's current device context. Pin the
    // owning ordinal or a drifted thread copies through foreign mappings.
    let _guard = crate::device::util::DeviceGuard::set(device_ordinal as i32);
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
        let cache_key = crate::device::jit_cache::compute_cache_key(arch, source, &options_c);

        if let Some(cached_bytes) = crate::device::jit_cache::load_cached_code_object(&cache_key) {
            return Ok((cached_bytes, entry_name.to_string()));
        }

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

        // Store freshly compiled HSA code object in persistent disk cache.
        let _ = crate::device::jit_cache::store_cached_code_object(&cache_key, &code_bytes);

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
/// Allocate a device-side scratch buffer, copy `data` into it, and return the
/// raw pointer. `ordinal` is the device the buffer must live on: WI-M1 pins
/// the calling thread's context to it for the malloc + H2D copy so a drifted
/// thread cannot land the scratch on another device.
pub fn upload_device_buffer<T: Copy>(ordinal: usize, data: &[T]) -> Result<*mut c_void> {
    let _guard = crate::device::util::DeviceGuard::set(ordinal as i32);
    let bytes = std::mem::size_of_val(data);
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
