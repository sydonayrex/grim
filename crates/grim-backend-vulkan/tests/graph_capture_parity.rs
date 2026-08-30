//! Test: Vulkan GraphCaptureOps records and replays via VkGraphCache.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test graph_capture_parity`.

use grim_tensor::backend::{CoreTensorOps, GraphCaptureOps};
use grim_tensor::{DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
fn graph_capture_records_and_replays() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    let shape = Shape::new(vec![256]);
    let a = dev.from_cpu(&vec![1.0f32; 256], &shape, DType::F32).unwrap();
    let b = dev.from_cpu(&vec![2.0f32; 256], &shape, DType::F32).unwrap();

    // Capture
    GraphCaptureOps::begin_graph_capture(&dev, "test_add").unwrap();
    let (sum, _handle) = CoreTensorOps::add(&dev, &*a, &*b, &shape).unwrap();
    GraphCaptureOps::end_graph_capture(&dev, "test_add").unwrap();

    // Replay — should report a hit
    let replayed = GraphCaptureOps::replay_graph(&dev, "test_add").unwrap();
    assert!(replayed, "graph should be replayed from cache");
    assert!(GraphCaptureOps::has_captured_graph(&dev, "test_add"));

    // Verify the captured computation produced correct output
    let v = sum.to_cpu_vec_f32().unwrap();
    for i in 0..256 {
        assert!(
            (v[i] - 3.0).abs() < 1e-6,
            "idx {}: {} != 3.0",
            i,
            v[i]
        );
    }
}
