//! Minimal reproducer: is a HIP graph that contains kernels launched on TWO
//! different devices capturable and replayable on this ROCm build?
//!
//! Mirrors exactly what the multi-GPU decode graph does:
//!   - enable peer access both ways,
//!   - begin capture on device 0's stream,
//!   - raise the capture gate on BOTH devices pointing at device 0's stream,
//!   - launch a kernel on device 1's buffers while current device is 1,
//!   - end capture / instantiate / launch on device 0,
//!   - read device 1's output back.
//!
//! Prints the result so the caller can see whether the kernel actually ran on
//! device 1 (dst == 2.0) or the process died with a page fault.

use grim_backend_rocm::{BackendStorage, MemoryOps};
use grim_backend_rocm::{
    DeviceGuard, RocmDevice, RocmStorage, hipGraphDestroy, hipGraphExecDestroy,
    hipGraphInstantiate, hipGraphLaunch, hipStreamBeginCapture, hipStreamEndCapture,
};
use grim_tensor::{CoreTensorOps, DType, Shape};

fn alloc(dev: &RocmDevice, n: usize) -> RocmStorage {
    RocmStorage::alloc_gpu(
        &Shape::new(vec![n]),
        DType::F32,
        &dev.allocator_handle(),
        dev.ordinal(),
    )
    .expect("alloc")
}

fn fill(dev: &RocmDevice, dst: &RocmStorage, v: f32, n: usize) {
    let data = vec![v; n];
    let src = dev
        .from_cpu(&data, &Shape::new(vec![n]), DType::F32)
        .expect("from_cpu");
    dev.copy_slice_into(dst, src.as_ref(), 0, n).expect("copy");
    dev.synchronize();
}

fn main() {
    let n = 1024usize;
    let d0 = RocmDevice::try_new(0).expect("dev0");
    let d1 = RocmDevice::try_new(1).expect("dev1");
    let _ = grim_backend_rocm::peer_access::enable_peer_access(0, 1);
    let _ = grim_backend_rocm::peer_access::enable_peer_access(1, 0);

    // Buffers live on device 1, exactly like a layer assigned to GPU 1.
    let a = alloc(&d1, n);
    let b = alloc(&d1, n);
    let out = alloc(&d1, n);
    fill(&d1, &a, 1.0, n);
    fill(&d1, &b, 1.0, n);
    fill(&d1, &out, 0.0, n);

    let stream0 = d0.active_stream();
    {
        let _g = DeviceGuard::set(0);
        let r = unsafe { hipStreamBeginCapture(stream0, 2) };
        assert_eq!(r, grim_backend_rocm::hipSuccess, "begin_capture: {r}");
    }
    d0.set_capture_gate(true, Some(stream0));
    d1.set_capture_gate(true, Some(stream0));
    {
        // Kernel for the device-1 layer: launched with current device = 1.
        let _g = DeviceGuard::set(1);
        d1.add_into(&a, &b, &out).expect("add_into on dev1");
    }
    d0.set_capture_gate(false, None);
    d1.set_capture_gate(false, None);

    let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
    {
        let _g = DeviceGuard::set(0);
        let r = unsafe { hipStreamEndCapture(stream0, &mut graph) };
        assert_eq!(r, grim_backend_rocm::hipSuccess, "end_capture: {r}");
    }
    let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
    {
        let _g = DeviceGuard::set(0);
        let r = unsafe {
            hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(r, grim_backend_rocm::hipSuccess, "instantiate: {r}");
    }
    // Reset out so a replay that does nothing is visible.
    fill(&d1, &out, 0.0, n);
    {
        let _g = DeviceGuard::set(0);
        let r = unsafe { hipGraphLaunch(exec, stream0) };
        assert_eq!(r, grim_backend_rocm::hipSuccess, "graph launch: {r}");
    }
    d0.synchronize();
    d1.synchronize();

    let got = out.to_cpu_vec_f32().expect("readback");
    println!(
        "[p2p-graph] out[0] = {} (expect 2.0 if the device-1 kernel replayed)",
        got[0]
    );
    unsafe {
        let _ = hipGraphExecDestroy(exec);
        let _ = hipGraphDestroy(graph);
    }
}
