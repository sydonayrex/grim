//! Integration test for HugePagePinnedBuffer (2MB Linux HugePage Host-Pinned Allocator).

use std::ffi::c_void;
use std::panic;
use grim_backend_rocm::{
    check_hip, hipMemcpy, HipMemcpyKind, HugePagePinnedBuffer, RocmDevice, RocmStorage
};
use grim_tensor::backend::BackendDevice;
use grim_tensor::dtype::DType;
use grim_tensor::shape::Shape;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

#[test]
fn test_hugepage_pinned_buffer_allocation_and_readwrite() {
    let size_bytes = 4 * 1024 * 1024; // 4MB (2 hugepages)
    let mut buf = HugePagePinnedBuffer::new(size_bytes).expect("HugePagePinnedBuffer allocation failed");

    assert!(buf.size() >= size_bytes, "Allocated size must be >= requested");
    assert_eq!(buf.size() % (2 * 1024 * 1024), 0, "Buffer size must be 2MB aligned");

    let slice = buf.as_mut_slice();
    for (i, byte) in slice.iter_mut().take(1024).enumerate() {
        *byte = (i % 255) as u8;
    }

    let read_slice = buf.as_slice();
    for (i, byte) in read_slice.iter().take(1024).enumerate() {
        assert_eq!(*byte, (i % 255) as u8);
    }
}

#[test]
fn test_hugepage_pinned_buffer_gpu_dma_roundtrip() {
    let Some(dev) = gpu_device() else {
        eprintln!("GRIM_RUN_GPU_TESTS unset or no ROCm device; skipping DMA roundtrip test");
        return;
    };

    let count = 512 * 1024; // 512k floats = 2MB
    let size_bytes = count * std::mem::size_of::<f32>();
    let mut host_buf = HugePagePinnedBuffer::new(size_bytes).expect("allocate hugepage pinned buffer");

    // Write source data into hugepage host buffer
    let host_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(host_buf.as_mut_ptr() as *mut f32, count)
    };
    for (i, val) in host_f32.iter_mut().enumerate() {
        *val = (i as f32 * 0.05).sin();
    }

    let shape = Shape::new(vec![count]);
    let d_storage_box = dev.alloc_storage(&shape, DType::F32).expect("alloc GPU storage");
    let d_storage = d_storage_box
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("downcast to RocmStorage");

    let dev_ptr = d_storage.device_ptr_checked().expect("valid device pointer") as *mut c_void;

    // 1. DMA H2D from hugepage buffer into GPU storage
    check_hip("hipMemcpy H2D", unsafe {
        hipMemcpy(
            dev_ptr,
            host_buf.as_ptr() as *const c_void,
            size_bytes,
            HipMemcpyKind::HostToDevice,
        )
    }).expect("H2D transfer failed");

    // 2. Clear host buffer to verify D2H overwrite
    for val in host_f32.iter_mut() {
        *val = 0.0;
    }

    // 3. DMA D2H from GPU storage back into hugepage buffer
    check_hip("hipMemcpy D2H", unsafe {
        hipMemcpy(
            host_buf.as_mut_ptr() as *mut c_void,
            dev_ptr,
            size_bytes,
            HipMemcpyKind::DeviceToHost,
        )
    }).expect("D2H transfer failed");

    // 4. Verify byte-exact equality
    for (i, &val) in host_f32.iter().enumerate() {
        let expected = (i as f32 * 0.05).sin();
        assert_eq!(val, expected, "Mismatch at index {i}");
    }
}
