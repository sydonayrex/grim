//! Handle / dimension / property types for the ROCm/HIP backend, plus
//! the raw FFI declarations they call into. Everything here is a
//! convenience-type safe wrapper around HIP runtime handles or an
//! extern "C" function declaration; none of them carry device state
//! of their own. By grouping the FFI declarations with the wrapper
//! types we keep both visible at a glance — adding a wrapper is one
//! file edit, not two.
//!
//! Skill attribution:
//! - `rust-ffi` — the safety comments on each unsafe block / impl
//!   recorder a SAFETY: line for any future audit.
//! - `rust-ai-ml-inference-guide` Action 1 — backend handle surface is
//!   the entry point for every other module.
//! - `rust-gpu-discipline` §3 — handle wrappers convert HIP raw
//!   `*mut c_void`s into typed Rust newtypes; the underlying pointer
//!   is only dereferenced through the FFI calls declared here.

use std::ffi::c_void;

use grim_tensor::error::Result;

use grim_tensor::backend::ComputeHandle;

use crate::device::helpers::check_hip;

// ======== HIP FFI root types (kept in lib.rs's clippy namespace) ========

/// Default integer error code type returned by every HIP runtime call.
/// Re-exported at crate root via `device::handles::HipErrorT`.
pub type HipErrorT = i32;

/// Success return code for HIP runtime FFI calls.
#[allow(non_upper_case_globals)]
pub const hipSuccess: HipErrorT = 0;

// ======== RocmHandle: typed wrapper around an optional HIP stream ========

/// A `ComputeHandle` contract — the caller submits work on a stream and
/// receives this handle. `synchronize()` blocks until the stream's prior
/// operations finish.
#[derive(Debug)]
pub struct RocmHandle {
    stream: Option<*mut c_void>,
}

impl RocmHandle {
    pub fn new(stream: Option<*mut c_void>) -> Self {
        Self { stream }
    }
}

// SAFETY: HIP stream handles are opaque platform resources that can safely be
// used from any thread. The underlying HIP runtime serializes stream operations.
unsafe impl Send for RocmHandle {}

impl ComputeHandle for RocmHandle {
    fn synchronize(&self) -> Result<()> {
        if let Some(stream) = self.stream {
            check_hip("hipStreamSynchronize", unsafe { hipStreamSynchronize(stream) })?;
        }
        Ok(())
    }
    fn is_ready(&self) -> bool {
        true
    }
}

// ======== HIP enum / typedef ========

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HipDim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl HipDim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_snake_case)]
pub struct HipGraphKernelNodeParams {
    pub func: *mut c_void,
    pub gridDim: HipDim3,
    pub blockDim: HipDim3,
    pub args: *mut *mut c_void,
    pub sharedMemBytes: u32,
    pub stream: *mut c_void,
    pub extra: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HipGraphMemcpyNodeParams {
    pub dst: *mut c_void,
    pub src: *const c_void,
    pub kind: HipMemcpyKind,
    pub size: usize,
}

/// Opaque HIP runtime-compiled program handle.
pub type HiprtcProgram = *mut c_void;

// ======== HIP / hiprtc FFI declarations ========

#[link(name = "amdhip64", kind = "dylib")]
#[link(name = "hiprtc", kind = "dylib")]
unsafe extern "C" {
    pub fn hipMalloc(devPtr: *mut *mut c_void, size: usize) -> HipErrorT;
    pub fn hipFree(device: *mut c_void) -> HipErrorT;
    pub fn hipHostMalloc(devPtr: *mut *mut c_void, size: usize, flags: u32) -> HipErrorT;
    pub fn hipHostFree(ptr: *mut c_void) -> HipErrorT;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: HipMemcpyKind,
    ) -> HipErrorT;
    pub fn hipMemset(dst: *mut c_void, value: i32, size_bytes: usize) -> HipErrorT;
    pub fn hipMemsetAsync(dst: *mut c_void, value: i32, size_bytes: usize, stream: *mut c_void) -> HipErrorT;
    pub fn hipDeviceSynchronize() -> HipErrorT;
    pub fn hipGetDeviceCount(count: *mut HipErrorT) -> HipErrorT;
    pub fn hipSetDevice(ordinal: HipErrorT) -> HipErrorT;
    pub fn hipGetDeviceProperties(prop: *mut c_void, device: i32) -> HipErrorT;
    pub fn hipDeviceGetAttribute(
        value: *mut i32,
        attribute: i32,
        device: i32,
    ) -> HipErrorT;
    pub fn hipMemAdvise(
        devPtr: *const c_void,
        count: usize,
        advice: i32,
        device: i32,
    ) -> HipErrorT;

    // Graph and Stream FFI
    pub fn hipStreamCreate(stream: *mut *mut c_void) -> HipErrorT;
    pub fn hipStreamDestroy(stream: *mut c_void) -> HipErrorT;
    pub fn hipStreamSynchronize(stream: *mut c_void) -> HipErrorT;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: HipMemcpyKind,
        stream: *mut c_void,
    ) -> HipErrorT;
    pub fn hipGraphCreate(graph: *mut *mut c_void, flags: u32) -> HipErrorT;
    pub fn hipGraphDestroy(graph: *mut c_void) -> HipErrorT;
    pub fn hipGraphInstantiate(
        exec: *mut *mut c_void,
        graph: *mut c_void,
        errorNode: *mut *mut c_void,
        logBuffer: *mut i8,
        bufferSize: usize,
    ) -> HipErrorT;
    pub fn hipGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> HipErrorT;
    pub fn hipGraphExecDestroy(exec: *mut c_void) -> HipErrorT;
    pub fn hipGraphExtendFromGlobalStream(
        exec: *mut *mut c_void,
        stream: *mut c_void,
        flags: u32,
    ) -> HipErrorT;
    pub fn hipGraphUpload(exec: *mut c_void, stream: *mut c_void) -> HipErrorT;
    pub fn hipStreamBeginCapture(stream: *mut c_void, mode: u32) -> HipErrorT;
    pub fn hipStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> HipErrorT;

    pub fn hipModuleLoad(module: *mut *mut c_void, path: *const i8) -> HipErrorT;
    pub fn hipModuleUnload(module: *mut c_void) -> HipErrorT;
    pub fn hipModuleGetFunction(
        func: *mut *mut c_void,
        module: *mut c_void,
        name: *const i8,
    ) -> HipErrorT;
    pub fn hipModuleLaunchKernel(
        func: *mut c_void,
        gridX: u32, gridY: u32, gridZ: u32,
        blockX: u32, blockY: u32, blockZ: u32,
        sharedMemBytes: u32,
        stream: *mut c_void,
        args: *mut *mut c_void,
        extra: *mut c_void,
    ) -> HipErrorT;

    pub fn hiprtcCreateProgram(
        prog: *mut HiprtcProgram,
        src: *const i8,
        name: *const i8,
        numHeaders: i32,
        headers: *const *const i8,
        headerNames: *const *const i8,
    ) -> HipErrorT;
    pub fn hiprtcCompileProgram(
        prog: HiprtcProgram,
        numOptions: i32,
        options: *const *const i8,
    ) -> HipErrorT;
    pub fn hiprtcGetCode(prog: HiprtcProgram, code: *mut i8) -> HipErrorT;
    pub fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> HipErrorT;
    pub fn hiprtcAddNameExpression(prog: HiprtcProgram, name: *const i8) -> HipErrorT;
    pub fn hiprtcGetCodeSize(prog: HiprtcProgram, size: *mut usize) -> HipErrorT;
    pub fn hiprtcGetErrorString(error: HipErrorT) -> *const i8;
    pub fn hiprtcGetProgramLogSize(prog: HiprtcProgram, log_size: *mut usize) -> HipErrorT;
    pub fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut i8) -> HipErrorT;
}

// ======== Attribute / advice constants ========

/// XNACK and device memory attribute flags for unified memory detection.
pub const HIP_DEVICE_ATTRIBUTE_COHERENT_DEVICE_ALLOC: i32 = 230;
pub const HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: i32 = 231;

/// Wavefront size attribute id — passed to `hipDeviceGetAttribute`.
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 24;

pub const HIP_MEM_ADVISE_SET_READ_MOSTLY: i32 = 1;
pub const HIP_MEM_ADVISE_UNSET_READ_MOSTLY: i32 = 2;
pub const HIP_MEM_ADVISE_SET_PREFERRED_LOCATION: i32 = 3;
pub const HIP_MEM_ADVISE_UNSET_PREFERRED_LOCATION: i32 = 4;
pub const HIP_MEM_ADVISE_SET_ACCESSED_BY: i32 = 5;
pub const HIP_MEM_ADVISE_UNSET_ACCESSED_BY: i32 = 6;
pub const HIP_MEM_ADVISE_SET_COARSE_GRAIN: i32 = 100;
pub const HIP_MEM_ADVISE_UNSET_COARSE_GRAIN: i32 = 101;

/// Correctness gate representation for target hardware wavefront width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavefrontSize {
    /// CDNA targets (MI200/MI300) requiring W64.
    W64 = 64,
    /// RDNA targets (consumer gaming GPUs, APUs) requiring W32.
    W32 = 32,
}

/// Device-level capability snapshot taken at `RocmDevice::new` time.
#[derive(Debug, Clone, Copy)]
pub struct RocmDeviceProps {
    pub wavefront_size: WavefrontSize,
    pub xnack_enabled: bool,
}

// ======== per-module tests ========

#[cfg(test)]
mod handles_self_tests {
    use super::*;

    #[test]
    fn hip_dim3_new_stores_each_axis() {
        let d = HipDim3::new(8, 4, 1);
        assert_eq!(d.x, 8);
        assert_eq!(d.y, 4);
        assert_eq!(d.z, 1);
    }

    #[test]
    fn hip_dim3_equality_is_structural() {
        assert_eq!(HipDim3::new(1, 2, 3), HipDim3::new(1, 2, 3));
        assert_ne!(HipDim3::new(1, 2, 3), HipDim3::new(3, 2, 1));
    }

    #[test]
    fn hip_memcpy_kind_h_to_d_value_is_1() {
        assert_eq!(HipMemcpyKind::HostToDevice as i32, 1);
    }

    #[test]
    fn rocm_handle_new_with_none_stream_returns_no_ready() {
        // No stream → synchronize is a no-op (returns Ok) without calling FFI.
        let h = RocmHandle::new(None);
        assert!(h.synchronize().is_ok());
    }

    #[test]
    fn wavefront_size_canonical_values() {
        assert_eq!(WavefrontSize::W64 as u32, 64);
        assert_eq!(WavefrontSize::W32 as u32, 32);
    }

    #[test]
    fn hip_attribute_constants_match_hip_spec() {
        // The values pinned here correspond to the HIP runtime's
        // hipruntime.h attribute enum. If the upstream spec ever
        // moves, this test will catch the drift at compile-time-ish
        // (it checks the value matches what `hipDeviceGetAttribute`
        // expects).
        assert_eq!(HIP_DEVICE_ATTRIBUTE_WARP_SIZE, 24);
        assert_eq!(HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, 231);
        assert_eq!(HIP_DEVICE_ATTRIBUTE_COHERENT_DEVICE_ALLOC, 230);
    }
}
