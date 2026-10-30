//! Handle / dimension / property types for the ROCm/HIP backend, plus [see: `rust-ffi`, `rust-ai-ml-inference-guide`]

#![allow(non_snake_case)]

use std::ffi::c_void;

use grim_tensor::error::Result;

use grim_tensor::backend::ComputeHandle;

use crate::device::helpers::check_hip;

// ======== HIP FFI root types (kept in lib.rs's clippy namespace) ========

/// Default integer error code type returned by every HIP runtime call. [see: `device::handles::HipErrorT`]
pub type HipErrorT = i32;

/// Success return code for HIP runtime FFI calls.
#[allow(non_upper_case_globals)]
pub const hipSuccess: HipErrorT = 0;

/// `hipErrorNotReady` — the return of a query that found work still in flight.
///
/// HIP's own value (600). Used to tell "not finished yet" apart from "the
/// query itself failed".
#[allow(non_upper_case_globals)]
pub const hipErrorNotReady: HipErrorT = 600;

// ======== RocmHandle: typed wrapper around an optional HIP stream ========

/// A `ComputeHandle` contract — the caller submits work on a stream and [see: `synchronize()`]
#[derive(Debug)]
pub struct RocmHandle {
    stream: Option<*mut c_void>,
}

impl RocmHandle {
    pub fn new(stream: Option<*mut c_void>) -> Self {
        Self { stream }
    }
}

// SAFETY: `RocmHandle` wraps an optional HIP stream pointer.
// HIP stream handles are opaque platform resources that are valid process-wide on the owning device.
unsafe impl Send for RocmHandle {}

impl ComputeHandle for RocmHandle {
    fn synchronize(&self) -> Result<()> {
        if let Some(stream) = self.stream {
            crate::device::util::report_rocm_progress("hipStreamSynchronize:in-flight");
            check_hip("hipStreamSynchronize", unsafe {
                hipStreamSynchronize(stream)
            })?;
            crate::device::util::report_rocm_progress("hipStreamSynchronize:done");
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

// ======== HIP / hiprtc FFI (grim-hip-sys adapter layer) ========
//
// Every HIP declaration lives in the vendored `grim-hip-sys` crate
// (bindgen output per HIP patch — see crates/grim-hip-sys/PROVENANCE.md).
// The fns below are thin unsafe adapters keeping grim's loose
// `c_void`/`i32` calling convention: they cast to the typed bindgen
// signatures and cast `hipError_t` (c_uint) back to `HipErrorT` (i32).
// No HIP symbol is hand-declared here except
// `hipGraphExtendFromGlobalStream`, which the ROCm 7.16 headers do not
// expose but the runtime provides (grim-specific graph-extension API).
use std::os::raw::c_char;

use grim_hip_sys as ffi;

pub use ffi::hipDeviceProp_tR0600;

/// Mirrors HIP's `hipPointerAttribute_t`; only the device ordinal is
/// material to our probes. Layout asserted against the bindgen type in
/// `handles_self_tests`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipPointerAttribute {
    pub memory_type: i32,
    pub device: i32,
    pub device_pointer: *mut std::ffi::c_void,
    pub host_pointer: *mut std::ffi::c_void,
    pub is_managed: i32,
    pub allocation_flags: u32,
}

// ---- memory ----
pub unsafe fn hipMalloc(devPtr: *mut *mut c_void, size: usize) -> HipErrorT {
    unsafe { ffi::hipMalloc(devPtr, size) as HipErrorT }
}
pub unsafe fn hipMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: u32) -> HipErrorT {
    unsafe { ffi::hipMallocManaged(devPtr, size, flags) as HipErrorT }
}
pub unsafe fn hipFree(device: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipFree(device) as HipErrorT }
}
/// Enqueue an async free on `stream`; the buffer is released once all prior work on that stream has completed.
/// Available from ROCm 5.4+.
pub unsafe fn hipFreeAsync(device: *mut c_void, stream: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipFreeAsync(device, stream as _) as HipErrorT }
}
pub unsafe fn hipHostMalloc(devPtr: *mut *mut c_void, size: usize, flags: u32) -> HipErrorT {
    unsafe { ffi::hipHostMalloc(devPtr, size, flags) as HipErrorT }
}
pub unsafe fn hipHostFree(ptr: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipHostFree(ptr) as HipErrorT }
}
pub unsafe fn hipHostRegister(hostPtr: *mut c_void, size: usize, flags: u32) -> HipErrorT {
    unsafe { ffi::hipHostRegister(hostPtr, size, flags) as HipErrorT }
}
pub unsafe fn hipHostUnregister(hostPtr: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipHostUnregister(hostPtr) as HipErrorT }
}
pub unsafe fn hipMemset(dst: *mut c_void, value: i32, size_bytes: usize) -> HipErrorT {
    unsafe { ffi::hipMemset(dst, value, size_bytes) as HipErrorT }
}
pub unsafe fn hipMemsetAsync(
    dst: *mut c_void,
    value: i32,
    size_bytes: usize,
    stream: *mut c_void,
) -> HipErrorT {
    unsafe { ffi::hipMemsetAsync(dst, value, size_bytes, stream as _) as HipErrorT }
}

// ---- sync / error ----
/// Returns the most recent asynchronous error reported on any API call
/// (including kernel launches), or success. Does not block.
pub unsafe fn hipGetLastError() -> HipErrorT {
    unsafe { ffi::hipGetLastError() as HipErrorT }
}
/// Blocks the calling thread until all previously enqueued work on the
/// current device's streams has completed.
///
/// # Safety
/// Enqueued kernels must not dereference freed or unmapped memory; this
/// call surfaces such faults on the host.
#[inline]
pub unsafe fn hipDeviceSynchronize() -> HipErrorT {
    if std::env::var("GRIM_HIP_TRACE").is_ok() {
        eprintln!(
            "[hipDeviceSync-trace]\n{:?}",
            std::backtrace::Backtrace::capture()
        );
    }
    unsafe { ffi::hipDeviceSynchronize() as HipErrorT }
}

// ---- device ----
pub unsafe fn hipGetDeviceCount(count: *mut HipErrorT) -> HipErrorT {
    unsafe { ffi::hipGetDeviceCount(count) as HipErrorT }
}
pub unsafe fn hipSetDevice(ordinal: HipErrorT) -> HipErrorT {
    unsafe { ffi::hipSetDevice(ordinal) as HipErrorT }
}
pub unsafe fn hipDeviceGetPCIBusId(pciBusId: *mut c_char, len: i32, device: i32) -> HipErrorT {
    unsafe { ffi::hipDeviceGetPCIBusId(pciBusId, len, device) as HipErrorT }
}
/// The 7.16 headers expose only the versioned `hipGetDevicePropertiesR0600`;
/// the unversioned symbol resolves to the same ABI. Callers pass an 8 KiB
/// scratch buffer, well above `size_of::<hipDeviceProp_tR0600>()`.
pub unsafe fn hipGetDeviceProperties(prop: *mut c_void, device: i32) -> HipErrorT {
    unsafe { ffi::hipGetDevicePropertiesR0600(prop as *mut _, device) as HipErrorT }
}
pub unsafe fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> HipErrorT {
    unsafe { ffi::hipMemGetInfo(free, total) as HipErrorT }
}
pub unsafe fn hipDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> HipErrorT {
    unsafe { ffi::hipDeviceGetAttribute(value, attribute as _, device) as HipErrorT }
}
pub unsafe fn hipPointerGetAttributes(
    attributes: *mut HipPointerAttribute,
    ptr: *const std::ffi::c_void,
) -> HipErrorT {
    unsafe { ffi::hipPointerGetAttributes(attributes as *mut _, ptr) as HipErrorT }
}
pub unsafe fn hipMemAdvise(devPtr: *const c_void, count: usize, advice: i32, device: i32)
-> HipErrorT {
    unsafe { ffi::hipMemAdvise(devPtr, count, advice as _, device) as HipErrorT }
}

// ---- streams ----
pub unsafe fn hipStreamCreate(stream: *mut *mut c_void) -> HipErrorT {
    unsafe { ffi::hipStreamCreate(stream as *mut _) as HipErrorT }
}
/// Create a stream with flags and priority. grim's historical declaration
/// bound the 3-arg signature to the `hipStreamCreateWithFlags` symbol, which
/// is 2-arg in the headers — a latent ABI mismatch. The 3-arg API is
/// `hipStreamCreateWithPriority`; that is what we call now.
pub unsafe fn hipStreamCreateWithFlags(
    stream: *mut *mut c_void,
    flags: u32,
    priority: i32,
) -> HipErrorT {
    unsafe { ffi::hipStreamCreateWithPriority(stream as *mut _, flags, priority) as HipErrorT }
}
pub unsafe fn hipStreamDestroy(stream: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipStreamDestroy(stream as _) as HipErrorT }
}
pub unsafe fn hipStreamSynchronize(stream: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipStreamSynchronize(stream as _) as HipErrorT }
}
/// Non-blocking query: `hipSuccess` when every operation on the stream has
/// completed, `hipErrorNotReady` while any is still in flight.
///
/// This is the OBSERVABLE form of "has the kernel left": a bounded poll
/// can tell the difference between a kernel that finished and one that is
/// still spinning, without blocking the host forever.
pub unsafe fn hipStreamQuery(stream: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipStreamQuery(stream as _) as HipErrorT }
}
/// Blocking host-to-device or device-to-host copy of `count` bytes.
///
/// # Safety
/// Must run under a device context pinned to the ordinal owning both
/// pointers; `dst`/`src` must be valid for `count` bytes.
#[inline]
pub unsafe fn hipMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: HipMemcpyKind,
) -> HipErrorT {
    if crate::device::util::is_debug_enabled() || std::env::var("GRIM_HIP_TRACE").is_ok() {
        eprintln!(
            "[GRIM_DEBUG hipMemcpy] kind={:?} count={} dst={:?} src={:?}",
            kind, count, dst, src
        );
    }
    unsafe { ffi::hipMemcpy(dst, src, count, kind as _) as HipErrorT }
}
/// Asynchronous memory copy on a specific stream.
///
/// # Safety
/// Caller must ensure pointers are valid for `count` bytes on the device/host.
#[inline]
pub unsafe fn hipMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: HipMemcpyKind,
    stream: *mut c_void,
) -> HipErrorT {
    if crate::device::util::is_debug_enabled() || std::env::var("GRIM_HIP_TRACE").is_ok() {
        eprintln!(
            "[GRIM_DEBUG hipMemcpyAsync] kind={:?} count={} stream={:?} dst={:?} src={:?}",
            kind, count, stream, dst, src
        );
    }
    crate::device::util::report_rocm_progress("hipMemcpyAsync:in-flight");
    let ret = unsafe { ffi::hipMemcpyAsync(dst, src, count, kind as _, stream as _) as HipErrorT };
    crate::device::util::report_rocm_progress("hipMemcpyAsync:done");
    ret
}
/// Asynchronous byte memset on a specific stream (capture-safe: no sync, no
/// allocation, issued on the stream that is recording).
#[inline]
pub unsafe fn hipMemsetD8Async(
    devPtr: *mut c_void,
    value: u8,
    count: usize,
    stream: *mut c_void,
) -> HipErrorT {
    unsafe { ffi::hipMemsetD8Async(devPtr, value, count, stream as _) as HipErrorT }
}
/// Asynchronous device-to-device copy on a specific stream (capture-safe).
#[inline]
pub unsafe fn hipMemcpyDtoDAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    stream: *mut c_void,
) -> HipErrorT {
    unsafe { ffi::hipMemcpyDtoDAsync(dst, src as _, count, stream as _) as HipErrorT }
}
pub unsafe fn hipMemPrefetchAsync(
    devPtr: *const c_void,
    count: usize,
    dstDevice: i32,
    stream: *mut c_void,
) -> HipErrorT {
    unsafe { ffi::hipMemPrefetchAsync(devPtr, count, dstDevice, stream as _) as HipErrorT }
}
/// Cross-device async copy (peer_access / rccl callers).
pub unsafe fn hipMemcpyPeerAsync(
    dst: *mut c_void,
    dstDevice: i32,
    src: *const c_void,
    srcDevice: i32,
    count: usize,
    stream: *mut c_void,
) -> HipErrorT {
    unsafe { ffi::hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream as _) as HipErrorT }
}

// ---- graphs ----
pub unsafe fn hipGraphCreate(graph: *mut *mut c_void, flags: u32) -> HipErrorT {
    unsafe { ffi::hipGraphCreate(graph as *mut _, flags) as HipErrorT }
}
pub unsafe fn hipGraphDestroy(graph: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipGraphDestroy(graph as _) as HipErrorT }
}
pub unsafe fn hipGraphInstantiate(
    exec: *mut *mut c_void,
    graph: *mut c_void,
    errorNode: *mut *mut c_void,
    logBuffer: *mut i8,
    bufferSize: usize,
) -> HipErrorT {
    unsafe {
        ffi::hipGraphInstantiate(
            exec as *mut _,
            graph as _,
            errorNode as *mut _,
            logBuffer,
            bufferSize,
        ) as HipErrorT
    }
}
pub unsafe fn hipGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipGraphLaunch(exec as _, stream as _) as HipErrorT }
}
pub unsafe fn hipGraphExecDestroy(exec: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipGraphExecDestroy(exec as _) as HipErrorT }
}
// Not in the ROCm 7.16 headers (bindgen emits nothing for it) but present in
// the runtime; the one hand-declared symbol left in this crate.
#[allow(non_snake_case)]
unsafe extern "C" {
    pub fn hipGraphExtendFromGlobalStream(
        exec: *mut *mut c_void,
        stream: *mut c_void,
        flags: u32,
    ) -> HipErrorT;
}
pub unsafe fn hipGraphUpload(exec: *mut c_void, stream: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipGraphUpload(exec as _, stream as _) as HipErrorT }
}
pub unsafe fn hipStreamBeginCapture(stream: *mut c_void, mode: u32) -> HipErrorT {
    unsafe { ffi::hipStreamBeginCapture(stream as _, mode as _) as HipErrorT }
}
pub unsafe fn hipStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> HipErrorT {
    unsafe { ffi::hipStreamEndCapture(stream as _, graph as *mut _) as HipErrorT }
}
// Graph kernel node parameter update (for scalar updates between replays)
pub unsafe fn hipGraphExecKernelNodeSetParams(
    graphExec: *mut c_void,
    node: *mut c_void,
    nodeParams: *const c_void,
) -> HipErrorT {
    unsafe {
        ffi::hipGraphExecKernelNodeSetParams(graphExec as _, node as _, nodeParams as *const _)
            as HipErrorT
    }
}
pub unsafe fn hipGraphAddKernelNode(
    pGraphNode: *mut *mut c_void,
    graph: *mut c_void,
    pDependencies: *const c_void,
    numDependencies: usize,
    pNodeParams: *const c_void,
) -> HipErrorT {
    unsafe {
        ffi::hipGraphAddKernelNode(
            pGraphNode as *mut _,
            graph as _,
            pDependencies as *const _,
            numDependencies,
            pNodeParams as *const _,
        ) as HipErrorT
    }
}

/// Per-node parameters of a captured graph's kernel nodes (precheck: what
/// the decode step actually launches). Returns (node, func) pairs; callers
/// reverse-map `func` through the device's resolved-kernel cache to names.
pub unsafe fn hipGraphGetKernelNodes(
    graph: *mut c_void,
) -> Vec<(*mut c_void, *mut c_void)> {
    let mut n: usize = 0;
    assert_eq!(
        unsafe { ffi::hipGraphGetNodes(graph as _, std::ptr::null_mut(), &mut n) } as u32,
        0,
        "hipGraphGetNodes(count)"
    );
    let mut nodes = vec![std::ptr::null_mut(); n];
    assert_eq!(
        unsafe { ffi::hipGraphGetNodes(graph as _, nodes.as_mut_ptr(), &mut n) } as u32,
        0,
        "hipGraphGetNodes"
    );
    nodes.truncate(n);
    let mut out = Vec::new();
    for node in nodes {
        let mut params: ffi::hipKernelNodeParams = unsafe { std::mem::zeroed() };
        // Memcpy/memset nodes return hipErrorInvalidValue here; skip them.
        if unsafe { ffi::hipGraphKernelNodeGetParams(node, &mut params) } as u32 == 0 {
            out.push((node as *mut c_void, params.func));
        }
    }
    out
}

/// Number of nodes in a captured graph (precheck: launches per replay).
pub unsafe fn hipGraphGetNodes(
    graph: *mut c_void,
    nodes: *mut *mut c_void,
    numNodes: *mut usize,
) -> HipErrorT {
    unsafe { ffi::hipGraphGetNodes(graph as _, nodes as *mut _, numNodes) as HipErrorT }
}

// ---- events (autotune benchmarking + cross-stream deps) ----
pub unsafe fn hipEventCreate(event: *mut *mut c_void) -> HipErrorT {
    unsafe { ffi::hipEventCreate(event as *mut _) as HipErrorT }
}
pub unsafe fn hipEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> HipErrorT {
    unsafe { ffi::hipEventCreateWithFlags(event as *mut _, flags) as HipErrorT }
}
pub unsafe fn hipEventDestroy(event: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipEventDestroy(event as _) as HipErrorT }
}
pub unsafe fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipEventRecord(event as _, stream as _) as HipErrorT }
}
pub unsafe fn hipEventSynchronize(event: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipEventSynchronize(event as _) as HipErrorT }
}
/// Cross-stream dependency: block `stream` until `event` (recorded on another stream) completes.
/// Enables upload-on-transfer-stream to overlap with compute-on-active-stream (SPEED-ROC-1 decode-step overlap).
pub unsafe fn hipStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> HipErrorT {
    unsafe { ffi::hipStreamWaitEvent(stream as _, event as _, flags) as HipErrorT }
}
pub unsafe fn hipEventElapsedTime(ms: *mut f32, start: *mut c_void, stop: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipEventElapsedTime(ms, start as _, stop as _) as HipErrorT }
}

// ---- device/version + modules ----
pub unsafe fn hipGetDevice(device: *mut i32) -> HipErrorT {
    unsafe { ffi::hipGetDevice(device) as HipErrorT }
}
pub unsafe fn hipRuntimeGetVersion(version: *mut i32) -> HipErrorT {
    unsafe { ffi::hipRuntimeGetVersion(version) as HipErrorT }
}
pub unsafe fn hipModuleLoad(module: *mut *mut c_void, path: *const i8) -> HipErrorT {
    unsafe { ffi::hipModuleLoad(module as *mut _, path) as HipErrorT }
}
pub unsafe fn hipModuleUnload(module: *mut c_void) -> HipErrorT {
    unsafe { ffi::hipModuleUnload(module as _) as HipErrorT }
}
pub unsafe fn hipModuleGetFunction(
    func: *mut *mut c_void,
    module: *mut c_void,
    name: *const i8,
) -> HipErrorT {
    unsafe { ffi::hipModuleGetFunction(func as *mut _, module as _, name) as HipErrorT }
}
pub unsafe fn hipModuleLaunchKernel(
    func: *mut c_void,
    gridX: u32,
    gridY: u32,
    gridZ: u32,
    blockX: u32,
    blockY: u32,
    blockZ: u32,
    sharedMemBytes: u32,
    stream: *mut c_void,
    args: *mut *mut c_void,
    extra: *mut c_void,
) -> HipErrorT {
    unsafe {
        ffi::hipModuleLaunchKernel(
            func as _,
            gridX,
            gridY,
            gridZ,
            blockX,
            blockY,
            blockZ,
            sharedMemBytes,
            stream as _,
            args,
            extra as *mut *mut c_void,
        ) as HipErrorT
    }
}
pub unsafe fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    numBlocks: *mut i32,
    func: *mut c_void,
    blockSize: i32,
    dynamicSMemSize: usize,
) -> HipErrorT {
    unsafe {
        ffi::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            numBlocks,
            func as _,
            blockSize,
            dynamicSMemSize,
        ) as HipErrorT
    }
}

// ---- peer access ----
pub unsafe fn hipDeviceCanAccessPeer(can_access: *mut i32, device: i32, peer_device: i32) -> HipErrorT {
    unsafe { ffi::hipDeviceCanAccessPeer(can_access, device, peer_device) as HipErrorT }
}
pub unsafe fn hipDeviceEnablePeerAccess(peer_device: i32, flags: u32) -> HipErrorT {
    unsafe { ffi::hipDeviceEnablePeerAccess(peer_device, flags) as HipErrorT }
}
pub unsafe fn hipDeviceDisablePeerAccess(peer_device: i32) -> HipErrorT {
    unsafe { ffi::hipDeviceDisablePeerAccess(peer_device) as HipErrorT }
}

// ---- hiprtc (own shared object; linked by build.rs alongside amdhip64) ----
pub unsafe fn hiprtcCreateProgram(
    prog: *mut HiprtcProgram,
    src: *const i8,
    name: *const i8,
    numHeaders: i32,
    headers: *const *const i8,
    headerNames: *const *const i8,
) -> HipErrorT {
    unsafe {
        ffi::hiprtcCreateProgram(prog as *mut _, src, name, numHeaders, headers, headerNames)
            as HipErrorT
    }
}
pub unsafe fn hiprtcCompileProgram(
    prog: HiprtcProgram,
    numOptions: i32,
    options: *const *const i8,
) -> HipErrorT {
    unsafe { ffi::hiprtcCompileProgram(prog as _, numOptions, options) as HipErrorT }
}
pub unsafe fn hiprtcGetCode(prog: HiprtcProgram, code: *mut i8) -> HipErrorT {
    unsafe { ffi::hiprtcGetCode(prog as _, code) as HipErrorT }
}
pub unsafe fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> HipErrorT {
    unsafe { ffi::hiprtcDestroyProgram(prog as *mut _) as HipErrorT }
}
pub unsafe fn hiprtcAddNameExpression(prog: HiprtcProgram, name: *const i8) -> HipErrorT {
    unsafe { ffi::hiprtcAddNameExpression(prog as _, name) as HipErrorT }
}
pub unsafe fn hiprtcGetLoweredName(
    prog: HiprtcProgram,
    name: *const i8,
    lowered_name: *mut *const i8,
) -> HipErrorT {
    unsafe { ffi::hiprtcGetLoweredName(prog as _, name, lowered_name) as HipErrorT }
}
pub unsafe fn hiprtcGetCodeSize(prog: HiprtcProgram, size: *mut usize) -> HipErrorT {
    unsafe { ffi::hiprtcGetCodeSize(prog as _, size) as HipErrorT }
}
pub unsafe fn hiprtcVersion(major: *mut i32, minor: *mut i32) -> HipErrorT {
    unsafe { ffi::hiprtcVersion(major, minor) as HipErrorT }
}
pub unsafe fn hiprtcGetErrorString(error: HipErrorT) -> *const i8 {
    unsafe { ffi::hiprtcGetErrorString(error as _) }
}
pub unsafe fn hiprtcGetProgramLogSize(prog: HiprtcProgram, log_size: *mut usize) -> HipErrorT {
    unsafe { ffi::hiprtcGetProgramLogSize(prog as _, log_size) as HipErrorT }
}
pub unsafe fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut i8) -> HipErrorT {
    unsafe { ffi::hiprtcGetProgramLog(prog as _, log) as HipErrorT }
}

// ======== Attribute / advice constants ======== Numbers verified against ROCm 7.x `hip_runtime_api.h` (`hipDeviceAttribute_t`) and probed live on gfx1201/gfx1200: the previous values here were CUDA
// numbering, so every query landed on a different attribute (WARP_SIZE=24 read ManagedMemory; MULTIPROCESSOR_COUNT=16 read Integrated; MAX_THREADS_PER_BLOCK=1 read AccessPolicyMaxWindowSize; PAGEABLE_MEMORY_ACCESS=231 did not exist and always errored).

/// XNACK and device memory attribute flags for unified memory detection.
/// NOTE: `hipDeviceAttributeCoherentDeviceAlloc` has no backing attribute in current ROCm headers; the number is kept for.
pub const HIP_DEVICE_ATTRIBUTE_COHERENT_DEVICE_ALLOC: i32 = 230;
pub const HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: i32 = 65;

/// Wavefront size attribute id — passed to `hipDeviceGetAttribute`.
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 87;

/// Maximum threads per block attribute id — passed to `hipDeviceGetAttribute`.
pub const HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 56;

/// Multiprocessor (workgroup processor) count attribute id. RDNA parts report
/// WGP count here — half the shader-CU figure marketing uses.
pub const HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 63;

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
        // The values pinned here correspond to ROCm 7.x `hipDeviceAttribute_t`
        // (verified by parsing the installed header and probing live).
        assert_eq!(HIP_DEVICE_ATTRIBUTE_WARP_SIZE, 87);
        assert_eq!(HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, 65);
        assert_eq!(HIP_DEVICE_ATTRIBUTE_COHERENT_DEVICE_ALLOC, 230);
    }

    #[test]
    fn hip_pointer_attribute_layout_matches_ffi() {
        // hipPointerGetAttributes casts grim's struct to the bindgen type;
        // the cast is only sound if the layouts agree field-for-field.
        assert_eq!(
            std::mem::size_of::<HipPointerAttribute>(),
            std::mem::size_of::<grim_hip_sys::hipPointerAttribute_t>()
        );
        assert_eq!(
            std::mem::offset_of!(HipPointerAttribute, device),
            std::mem::offset_of!(grim_hip_sys::hipPointerAttribute_t, device)
        );
        assert_eq!(
            std::mem::offset_of!(HipPointerAttribute, host_pointer),
            std::mem::offset_of!(grim_hip_sys::hipPointerAttribute_t, hostPointer)
        );
    }
}
