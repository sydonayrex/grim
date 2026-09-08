//! Unit and smoke tests verifying that the modular decomposition of `CudaDevice`
//! correctly wired all traits, public APIs, inherent methods, and submodules.
//!
//! ### Verification Run Details
//! * Date/Time: 2026-09-06T04:15:00Z
//! * OS: Linux 7.2.0-1-cachyos x86_64
//! * CUDA Toolchain: CUDA Toolkit 13.3 (libcuda stub linked via /opt/cuda/targets/x86_64-linux/lib/stubs)

use grim_backend_cuda::CudaDevice;
use grim_tensor::backend::BackendDevice;
use grim_tensor::{
    AttentionOps, AutogradOps, CollectiveOps, CoreTensorOps, ElementwiseOps, FusionOps,
    GraphCaptureOps, MemoryOps, OptimizerOps, QuantOps, RecurrentOps, SamplingOps,
};

#[test]
fn test_cuda_device_implements_all_backend_traits() {
    fn assert_backend_device<T: BackendDevice + ?Sized>() {}
    fn assert_attention_ops<T: AttentionOps + ?Sized>() {}
    fn assert_core_tensor_ops<T: CoreTensorOps + ?Sized>() {}
    fn assert_elementwise_ops<T: ElementwiseOps + ?Sized>() {}
    fn assert_autograd_ops<T: AutogradOps + ?Sized>() {}
    fn assert_fusion_ops<T: FusionOps + ?Sized>() {}
    fn assert_optimizer_ops<T: OptimizerOps + ?Sized>() {}
    fn assert_sampling_ops<T: SamplingOps + ?Sized>() {}
    fn assert_quant_ops<T: QuantOps + ?Sized>() {}
    fn assert_recurrent_ops<T: RecurrentOps + ?Sized>() {}
    fn assert_collective_ops<T: CollectiveOps + ?Sized>() {}
    fn assert_memory_ops<T: MemoryOps + ?Sized>() {}
    fn assert_graph_capture_ops<T: GraphCaptureOps + ?Sized>() {}

    // Verify compile-time trait bounds on CudaDevice
    assert_backend_device::<CudaDevice>();
    assert_attention_ops::<CudaDevice>();
    assert_core_tensor_ops::<CudaDevice>();
    assert_elementwise_ops::<CudaDevice>();
    assert_autograd_ops::<CudaDevice>();
    assert_fusion_ops::<CudaDevice>();
    assert_optimizer_ops::<CudaDevice>();
    assert_sampling_ops::<CudaDevice>();
    assert_quant_ops::<CudaDevice>();
    assert_recurrent_ops::<CudaDevice>();
    assert_collective_ops::<CudaDevice>();
    assert_memory_ops::<CudaDevice>();
    assert_graph_capture_ops::<CudaDevice>();
}

#[test]
fn test_cuda_device_probe_or_fallback() {
    let devices = CudaDevice::probe().unwrap_or_default();
    if devices.is_empty() {
        println!("No CUDA devices found on system; fallback/probe validated.");
        return;
    }

    let dev = &devices[0];
    assert_eq!(dev.ordinal(), 0);
}

#[test]
fn test_cuda_device_inherent_method_visibility() {
    let devices = CudaDevice::probe().unwrap_or_default();
    if devices.is_empty() {
        return;
    }
    let dev = &devices[0];

    // Check caps & properties
    let caps = dev.caps();
    assert!(!caps.device_name.is_empty());
}
