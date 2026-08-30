//! Test: Vulkan ring-allreduce scaffolding + single-GPU accumulation.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test ring_allreduce_parity`.

use grim_tensor::backend::CollectiveOps;
use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::collective::VkCommunicator;
use grim_backend_vulkan::VulkanDevice;

#[test]
fn ring_allreduce_sums_across_inputs_single_gpu() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    let shape = Shape::new(vec![256]);
    let a = dev.from_cpu(&vec![1.0f32; 256], &shape, DType::F32).unwrap();
    let b = dev.from_cpu(&vec![2.0f32; 256], &shape, DType::F32).unwrap();

    let (result, _handle) = CollectiveOps::all_reduce(&dev, &[&*a, &*b], "sum").unwrap();
    let v = result.to_cpu_vec_f32().unwrap();
    for i in 0..256 {
        assert!(
            (v[i] - 3.0).abs() < 1e-5,
            "idx {}: {} != 3.0",
            i,
            v[i]
        );
    }
}

#[test]
fn communicator_multi_gpu_returns_honest_error() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    // Attach a multi-GPU communicator (world_size=2) and verify all_reduce
    // returns an honest error rather than silently falling back to single-GPU.
    let comm = VkCommunicator::new(2, 0).unwrap();
    let dev = VulkanDevice::new().with_communicator(comm);
    let shape = Shape::new(vec![256]);
    let a = dev.from_cpu(&vec![1.0f32; 256], &shape, DType::F32).unwrap();
    let b = dev.from_cpu(&vec![2.0f32; 256], &shape, DType::F32).unwrap();

    let result = CollectiveOps::all_reduce(&dev, &[&*a, &*b], "sum");
    assert!(
        result.is_err(),
        "multi-GPU all_reduce should return an error (P2P transport not wired)"
    );
    let err_msg = match result {
        Err(e) => format!("{}", e),
        Ok(_) => unreachable!(),
    };
    assert!(
        err_msg.contains("P2P") || err_msg.contains("multi-GPU"),
        "error should mention P2P/multi-GPU, got: {}",
        err_msg
    );
}

#[test]
fn vk_communicator_validates_topology() {
    // world_size must be >= 1
    assert!(VkCommunicator::new(0, 0).is_err());
    // rank must be < world_size
    assert!(VkCommunicator::new(2, 2).is_err());
    assert!(VkCommunicator::new(2, 5).is_err());
    // valid topology
    assert!(VkCommunicator::new(2, 0).is_ok());
    assert!(VkCommunicator::new(2, 1).is_ok());
}
