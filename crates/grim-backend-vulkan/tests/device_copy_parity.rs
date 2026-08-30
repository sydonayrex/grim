//! Test: Vulkan device-side copy_slice_into via vkCmdCopyBuffer.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test device_copy_parity`.

use grim_tensor::backend::MemoryOps;
use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
fn device_copy_slice_is_device_side() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    let src_shape = Shape::new(vec![64]);
    let dst_shape = Shape::new(vec![128]);
    let src = dev.from_cpu(&vec![42.0f32; 64], &src_shape, DType::F32).unwrap();
    let dst = dev.from_cpu(&vec![0.0f32; 128], &dst_shape, DType::F32).unwrap();

    MemoryOps::copy_slice_into(&dev, &*dst, &*src, 16, 64).unwrap();

    let dst_v = dst.to_cpu_vec_f32().unwrap();
    // Bytes 0-15 should still be 0
    for i in 0..16 {
        assert_eq!(dst_v[i], 0.0, "prefix byte {} should be 0", i);
    }
    // Bytes 16-79 should be 42.0
    for i in 16..80 {
        assert!(
            (dst_v[i] - 42.0).abs() < 1e-6,
            "copied byte {} should be 42.0",
            i
        );
    }
    // Bytes 80-127 should still be 0
    for i in 80..128 {
        assert_eq!(dst_v[i], 0.0, "suffix byte {} should be 0", i);
    }
}

#[test]
fn device_copy_with_add_kernel_roundtrip() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    let shape = Shape::new(vec![256]);
    let a = dev.from_cpu(&vec![1.0f32; 256], &shape, DType::F32).unwrap();
    let b = dev.from_cpu(&vec![2.0f32; 256], &shape, DType::F32).unwrap();

    // Compute a + b into a fresh buffer
    let (sum, _) = CoreTensorOps::add(&dev, &*a, &*b, &shape).unwrap();

    // Allocate a zeroed dst and copy the sum into it at offset 0
    let dst = dev.from_cpu(&vec![0.0f32; 256], &shape, DType::F32).unwrap();
    MemoryOps::copy_slice_into(&dev, &*dst, &*sum, 0, 256).unwrap();

    let dst_v = dst.to_cpu_vec_f32().unwrap();
    for i in 0..256 {
        assert!(
            (dst_v[i] - 3.0).abs() < 1e-6,
            "dst[{}]: {} != 3.0",
            i,
            dst_v[i]
        );
    }
}
