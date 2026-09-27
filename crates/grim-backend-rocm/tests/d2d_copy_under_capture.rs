//! D2D `copy_slice_into` **during HIP graph capture**.
//!
//! `d2d_copy_rank_repro` proves the exact failing pair from the 9B run —
//! `[1,16,256]` -> `[1,4096]`, 16384 bytes — copies and verifies on a clean
//! device in 0.04 s. So the refusal is state-dependent, and the one piece of
//! state that differs between that test and the run is capture: the decode
//! graph walks the layer's `forward` with an open capture bracket, and
//! `reshaped_view` allocates its destination and copies INSIDE that bracket.
//!
//! Two sub-cases, because they differ in one variable:
//!   * both buffers pre-allocated outside capture — is the copy itself refused?
//!   * destination allocated inside capture — this is what `reshaped_view`
//!     actually does, and `hipMalloc` during capture is a separate hazard.
//!
//! The test REFUSES to be vacuously green: it asserts capture is enabled
//! before asserting anything, so running it without `GRIM_CAPTURE_GRAPH=1` is
//! a failure, not a skip.
//!
//! Gated: `GRIM_GPU_TEST=1 GRIM_CAPTURE_GRAPH=1`.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, MemoryOps, Shape};

const COUNT: usize = 4096;
const SRC_DIMS: [usize; 3] = [1, 16, 256];
const DST_DIMS: [usize; 2] = [1, 4096];

fn payload() -> Vec<f32> {
    (0..COUNT).map(|i| (i as f32) * 0.5 - 7.25).collect()
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1 GRIM_CAPTURE_GRAPH=1"]
fn d2d_copy_succeeds_inside_graph_capture() {
    assert!(
        grim_backend_rocm::gpu_test_enabled(),
        "set GRIM_GPU_TEST=1"
    );
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    assert!(
        dev.graph_capture_enabled(),
        "GRIM_CAPTURE_GRAPH must be set in the ENVIRONMENT of this process: capture is \
         read once at RocmDevice construction, so setting it here would prove nothing \
         about the copy. Run: GRIM_GPU_TEST=1 GRIM_CAPTURE_GRAPH=1 cargo test -- --ignored"
    );

    let want = payload();

    // ---- sub-case A: both buffers exist before the capture opens -----------
    let src_a = dev
        .from_cpu(&want, &Shape::new(SRC_DIMS.to_vec()), DType::F32)
        .expect("upload src");
    let dst_a = dev
        .alloc_storage(&Shape::new(DST_DIMS.to_vec()), DType::F32)
        .expect("alloc dst");

    dev.begin_graph_capture("d2d_probe_a").expect("begin capture A");
    let copy_a = dev.copy_slice_into(dst_a.as_ref(), src_a.as_ref(), 0, COUNT);
    // End the capture even on failure, or the stream stays mid-capture and every
    // later HIP call in this process inherits the state under test.
    dev.end_graph_capture("d2d_probe_a").expect("end capture A");

    copy_a.unwrap_or_else(|e| {
        panic!(
            "D2D copy REFUSED inside graph capture: {e}\n\
             Both buffers were allocated BEFORE begin_graph_capture, so this isolates the \
             copy itself: hipMemcpyAsync on a capturing stream is what returned the refusal."
        )
    });

    dev.synchronize();
    let got_a = dst_a.to_cpu_vec_f32().expect("read back A");
    for (i, (&g, &w)) in got_a.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "sub-case A: element {i} is {g}, want {w}");
    }
    eprintln!("[d2d-capture] sub-case A (pre-allocated both sides): OK, bytes verified");
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1 GRIM_CAPTURE_GRAPH=1"]
fn d2d_copy_with_destination_allocated_inside_capture_succeeds() {
    assert!(
        grim_backend_rocm::gpu_test_enabled(),
        "set GRIM_GPU_TEST=1"
    );
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    assert!(
        dev.graph_capture_enabled(),
        "GRIM_CAPTURE_GRAPH must be set in the ENVIRONMENT of this process"
    );

    let want = payload();
    let src = dev
        .from_cpu(&want, &Shape::new(SRC_DIMS.to_vec()), DType::F32)
        .expect("upload src");

    // This is `reshaped_view`'s exact sequence: alloc_storage, then copy.
    dev.begin_graph_capture("d2d_probe_b").expect("begin capture B");
    let alloc = dev.alloc_storage(&Shape::new(DST_DIMS.to_vec()), DType::F32);
    let dst = match alloc {
        Ok(d) => d,
        Err(e) => {
            let _ = dev.end_graph_capture("d2d_probe_b");
            panic!("alloc_storage REFUSED inside graph capture: {e}");
        }
    };
    let copy = dev.copy_slice_into(dst.as_ref(), src.as_ref(), 0, COUNT);
    dev.end_graph_capture("d2d_probe_b").expect("end capture B");

    copy.unwrap_or_else(|e| {
        panic!(
            "D2D copy REFUSED with the destination allocated inside capture: {e}\n\
             This is the `reshaped_view` fast path verbatim."
        )
    });

    dev.synchronize();
    let got = dst.to_cpu_vec_f32().expect("read back B");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "sub-case B: element {i} is {g}, want {w}");
    }
    eprintln!("[d2d-capture] sub-case B (alloc dst inside capture): OK, bytes verified");
}
