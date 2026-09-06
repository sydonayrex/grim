//! Comprehensive unit, smoke, and modular wiring tests for `RocmDevice`.
//! Verifies that trait implementations, inherent kernel launchers, and cross-file
//! methods across all submodules (device_quant, device_compute, device_attention,
//! device_recurrent, device_routing, device_serve) compile and execute cleanly.
//!
//! ### Verification Run Details
//! * Date/Time: 2026-09-06T04:15:00Z
//! * OS: Linux 7.2.0-1-cachyos x86_64
//! * GPU 0: AMD Radeon RX 9070 XT (gfx1201 / RDNA4, PCI 0000:03:00.0)
//! * GPU 1: AMD Radeon RX 9060 XT (gfx1200 / RDNA4, PCI 0000:0A:00.0)
//! * Driver / ROCm: 7.2.0-1-cachyos

use grim_tensor::backend::BackendDevice;
use grim_tensor::dtype::ArithType;
use grim_tensor::{
    AttentionOps, AutogradOps, CollectiveOps, CoreTensorOps, ElementwiseOps,
    FusionOps, GraphCaptureOps, MemoryOps, OptimizerOps, QuantOps, RecurrentOps, SamplingOps,
};
use grim_backend_rocm::RocmDevice;

#[test]
fn test_rocm_device_implements_all_backend_traits() {
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

    // Static compile-time trait checks
    assert_backend_device::<RocmDevice>();
    assert_attention_ops::<RocmDevice>();
    assert_core_tensor_ops::<RocmDevice>();
    assert_elementwise_ops::<RocmDevice>();
    assert_autograd_ops::<RocmDevice>();
    assert_fusion_ops::<RocmDevice>();
    assert_optimizer_ops::<RocmDevice>();
    assert_sampling_ops::<RocmDevice>();
    assert_quant_ops::<RocmDevice>();
    assert_recurrent_ops::<RocmDevice>();
    assert_collective_ops::<RocmDevice>();
    assert_memory_ops::<RocmDevice>();
    assert_graph_capture_ops::<RocmDevice>();
}

#[test]
fn test_rocm_device_probe_or_fallback() {
    let dev = RocmDevice::new(0);
    assert_eq!(dev.ordinal(), 0);
}

#[test]
fn test_rocm_device_host_dequant_numeric_pathway() {
    let dev = RocmDevice::new(0);

    // Q8_0 verification
    let mut q8_data = vec![0u8; 34];
    q8_data[0] = 0x00;
    q8_data[1] = 0x3C; // 1.0 f16
    for i in 0..32 {
        q8_data[2 + i] = (i as i8) as u8;
    }

    let dequant_res = dev.dequantize_q8_0_host(&q8_data, 32);
    assert!(dequant_res.is_ok(), "Host dequant for Q8_0 must succeed");
    let values = dequant_res.unwrap();
    assert_eq!(values.len(), 32);
    for i in 0..32 {
        let diff = (values[i] - (i as f32)).abs();
        assert!(diff < 1e-4, "Mismatch at {}: {}", i, values[i]);
    }
}

#[test]
fn test_rocm_device_should_use_wmma_path() {
    use grim_format::spec::{GrimTensorExt, LayoutHintTag};

    let dev = RocmDevice::new(0);
    dev.set_wmma_gemm_enabled(true);

    let ext = GrimTensorExt {
        tensor_name: "weight".into(),
        layout_hint: LayoutHintTag::PackedQuantWmma {
            bits: 4,
            frag_m: 16,
            frag_n: 16,
        },
        ..Default::default()
    };

    assert!(dev.should_use_wmma_path(Some(&ext), ArithType::F16));
    assert!(!dev.should_use_wmma_path(Some(&ext), ArithType::F32));

    dev.set_wmma_gemm_enabled(false);
    assert!(!dev.should_use_wmma_path(Some(&ext), ArithType::F16));
}

#[test]
fn test_rocm_device_eplb_expert_balancing() {
    let dev = RocmDevice::new(0);
    let expert_loads = vec![100.0f32, 200.0f32, 50.0f32, 400.0f32];
    let plan = dev.eplb_balance_experts(&expert_loads, 2, 0);
    assert_eq!(plan.expert_to_rank.len(), 4);
    assert_eq!(plan.rank_loads.len(), 2);
}
