//! D3 design probe: may a HIP graph captured on device 1 contain a PEER COPY
//! (hipMemcpyPeerAsync with a device-0 source)? The Phase-A handoff is a
//! per-token peer copy + event record + cross-stream wait OUTSIDE the graphs;
//! if a peer-source memcpy NODE is capturable and replays correctly, the whole
//! handoff collapses into the consumer's graph and the per-token event chain
//! disappears. This is the one thing that decides D3's implementation.
//!
//! `p2p_graph_probe.rs` proves a cross-device KERNEL launch faults; a memcpy
//! node is a different node type, so it needs its own answer.
//!
//! Prints "peer-memcpy node: OK" (and verifies the bytes) or the failure.

use grim_backend_rocm::{
    BackendStorage, MemoryOps, hipGraphDestroy, hipGraphExecDestroy, hipGraphInstantiate,
    hipGraphLaunch, hipStreamBeginCapture, hipStreamEndCapture,
};
use grim_backend_rocm::{DeviceGuard, RocmDevice, RocmStorage};
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

fn main() {
    let n = 4096usize;
    let d0 = RocmDevice::try_new(0).expect("dev0");
    let d1 = RocmDevice::try_new(1).expect("dev1");
    let _ = grim_backend_rocm::peer_access::enable_peer_access(0, 1);
    let _ = grim_backend_rocm::peer_access::enable_peer_access(1, 0);

    // Producer on device 0, consumer on device 1 — the split's real shape.
    let src = alloc(&d0, n);
    let dst = alloc(&d1, n);
    let want: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    let host = d0
        .from_cpu(&want, &Shape::new(vec![n]), DType::F32)
        .expect("host");
    d0.copy_slice_into(&src, host.as_ref(), 0, n)
        .expect("fill src");
    let zeros = vec![0.0f32; n];
    let z = d1
        .from_cpu(&zeros, &Shape::new(vec![n]), DType::F32)
        .expect("zeros");
    d1.copy_slice_into(&dst, z.as_ref(), 0, n)
        .expect("zero dst");
    d0.synchronize();
    d1.synchronize();

    // Capture on DEVICE 1 (the consumer), issuing the peer copy from device 0.
    let stream1 = d1.active_stream();
    {
        let _g = DeviceGuard::set(1);
        let r = unsafe { hipStreamBeginCapture(stream1, 2) };
        assert_eq!(r, grim_backend_rocm::hipSuccess, "begin_capture: {r}");
        let sp = src.device_ptr_u64().expect("src ptr");
        let dp = dst.device_ptr_u64().expect("dst ptr");
        let r = grim_backend_rocm::peer_access::p2p_memcpy_async(
            dp as *mut std::ffi::c_void,
            1,
            sp as *const std::ffi::c_void,
            0,
            n * 4,
            stream1,
        );
        match r {
            Ok(()) => println!("peer copy accepted inside capture"),
            Err(e) => {
                println!("peer copy REJECTED inside capture: {e}");
                let mut g: *mut std::ffi::c_void = std::ptr::null_mut();
                let _ = unsafe { hipStreamEndCapture(stream1, &mut g) };
                if !g.is_null() {
                    unsafe {
                        let _ = hipGraphDestroy(g);
                    }
                }
                return;
            }
        }
        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { hipStreamEndCapture(stream1, &mut graph) };
        if r != grim_backend_rocm::hipSuccess {
            println!("peer-memcpy node: END CAPTURE FAILED ({r})");
            return;
        }
        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe {
            hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if r != grim_backend_rocm::hipSuccess {
            println!("peer-memcpy node: INSTANTIATE FAILED ({r})");
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
            return;
        }
        // Poison the destination so a no-op replay cannot pass.
        let poison = vec![7.0f32; n];
        let p = d1
            .from_cpu(&poison, &Shape::new(vec![n]), DType::F32)
            .expect("poison");
        d1.copy_slice_into(&dst, p.as_ref(), 0, n)
            .expect("poison dst");
        d1.synchronize();

        let r = unsafe { hipGraphLaunch(exec, stream1) };
        if r != grim_backend_rocm::hipSuccess {
            println!("peer-memcpy node: LAUNCH FAILED ({r})");
            unsafe {
                let _ = hipGraphExecDestroy(exec);
                let _ = hipGraphDestroy(graph);
            }
            return;
        }
        let _ = unsafe { grim_backend_rocm::hipDeviceSynchronize() };
        unsafe {
            let _ = hipGraphExecDestroy(exec);
            let _ = hipGraphDestroy(graph);
        }
    }
    let got = dst.to_cpu_vec_f32().expect("readback");
    if got == want {
        println!("peer-memcpy node: OK (bytes verified, {n} elements)");
    } else {
        let bad = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
        println!(
            "peer-memcpy node: REPLAYED WRONG at {bad}: {} vs {}",
            got[bad], want[bad]
        );
    }
}
