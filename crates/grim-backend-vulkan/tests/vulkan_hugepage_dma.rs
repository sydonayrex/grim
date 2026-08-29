//! Integration test for VulkanHugePageBuffer.

use grim_backend_vulkan::{VulkanDevice, VulkanHugePageBuffer};
use grim_tensor::backend::BackendDevice;
use grim_tensor::dtype::DType;
use grim_tensor::shape::Shape;

#[test]
fn test_vulkan_hugepage_buffer_allocation_and_dma_roundtrip() {
    let size_bytes = 4 * 1024 * 1024; // 4MB
    let mut host_buf = VulkanHugePageBuffer::new(size_bytes).expect("allocate hugepage buffer");

    assert!(host_buf.size() >= size_bytes);
    assert_eq!(host_buf.size() % (2 * 1024 * 1024), 0);

    let count = 512 * 1024; // 512K floats = 2MB
    let host_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(host_buf.as_mut_ptr() as *mut f32, count)
    };
    for (i, val) in host_f32.iter_mut().enumerate() {
        *val = ((i as f32 + 1.0) * 0.05).sin();
    }

    let devices = VulkanDevice::probe().unwrap();
    if devices.is_empty() {
        eprintln!("Vulkan device uninitialized/unavailable; skipping GPU DMA test");
        return;
    }
    let vk_dev = &devices[0];

    let shape = Shape::new(vec![count]);
    let d_storage = vk_dev.from_cpu(host_f32, &shape, DType::F32).unwrap();

    let d2h_vec = d_storage.to_cpu_vec_f32().unwrap();
    assert_eq!(d2h_vec.len(), count);

    for (i, (&actual, &expected)) in d2h_vec.iter().zip(host_f32.iter()).enumerate() {
        assert_eq!(actual, expected, "Mismatch at index {i}");
    }
}
