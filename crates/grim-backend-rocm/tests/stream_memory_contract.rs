//! Explicit-stream staging contract for the ROCm decode path.
//!
//! Verifies that a pinned H2D staging write can be handed from one device
//! stream to another with an event, and that the destination observes the
//! second value rather than a stale or racing copy.

use std::ptr::null_mut;

use grim_backend_rocm::{
    as_rocm, gpu_test_enabled, hipEventCreate, hipEventDestroy, hipEventRecord,
    hipStreamSynchronize, hipStreamWaitEvent, BackendStorage, RocmDevice,
};
use grim_tensor::{CoreTensorOps, DType, Shape};

#[test]
#[ignore]
fn explicit_stream_h2d_event_handoff_preserves_latest_value() {
    if !gpu_test_enabled() {
        eprintln!("skipping: set GRIM_GPU_TEST=1");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skipping: no ROCm ordinal 0");
        return;
    }

    let dev = RocmDevice::shared(0);
    let producer = dev
        .get_stream_from_pool(1)
        .expect("ROCm stream pool must contain slot 1");
    let consumer = dev
        .get_stream_from_pool(0)
        .expect("ROCm stream pool must contain slot 0");
    assert_ne!(producer, consumer, "handoff test needs distinct streams");

    let storage = CoreTensorOps::from_cpu(&dev, &[0.0f32; 8], &Shape::new(vec![8]), DType::F32)
        .expect("upload destination");
    let destination = as_rocm(storage.as_ref()).expect("ROCm destination");

    destination
        .write_host_f32_async(&[1.0f32; 8], producer)
        .expect("producer H2D staging");

    let mut handoff = null_mut();
    assert_eq!(
        unsafe { hipEventCreate(&mut handoff) },
        grim_backend_rocm::hipSuccess,
        "event creation"
    );
    assert_eq!(
        unsafe { hipEventRecord(handoff, producer) },
        grim_backend_rocm::hipSuccess,
        "event record"
    );
    assert_eq!(
        unsafe { hipStreamWaitEvent(consumer, handoff, 0) },
        grim_backend_rocm::hipSuccess,
        "event wait"
    );

    destination
        .write_host_f32_async(&[2.0f32; 8], consumer)
        .expect("consumer H2D staging");
    assert_eq!(
        unsafe { hipStreamSynchronize(consumer) },
        grim_backend_rocm::hipSuccess,
        "consumer synchronization"
    );

    let got = destination.to_cpu_vec_f32().expect("D2H readback");
    assert_eq!(got, vec![2.0f32; 8]);
    assert_eq!(
        unsafe { hipEventDestroy(handoff) },
        grim_backend_rocm::hipSuccess,
        "event destruction"
    );
}
