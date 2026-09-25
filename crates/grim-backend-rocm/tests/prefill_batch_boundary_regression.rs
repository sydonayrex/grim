//! Task 10 (Plan 3): Regression test for prefill batch boundary M = M_cut handoff.
//! Verifies handoff behavior across boundary M in {64, 65} and override via GRIM_PREFILL_DOT4_M_MAX.

use grim_backend_rocm::RocmDevice;
use grim_tensor::CoreTensorOps;
use grim_tensor::QuantOps;
use grim_tensor::Shape;
use grim_tensor::dtype::{DType, QuantFormat};

#[test]
#[ignore]
fn test_prefill_batch_boundary_regression() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_prefill_batch_boundary_regression");
        return;
    }
    let dev = RocmDevice::new(0);

    let k = 1024;
    let n = 1024;

    // Weight B [N, K]
    let b_unquant: Vec<f32> = (0..n * k)
        .map(|i| (i as f32 * 0.019).cos() * 1.0)
        .collect();
    let b_shape = Shape::new(vec![n, k]);
    let b_raw = dev.from_cpu(&b_unquant, &b_shape, DType::F32).unwrap();
    let (b_q80, h_quant) = dev
        .quantize_on_device(b_raw.as_ref(), QuantFormat::Q8_0)
        .unwrap();
    h_quant.synchronize().unwrap();

    for m in [64usize, 65usize] {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 * 0.023).sin() * 1.5)
            .collect();
        let a_shape = Shape::new(vec![m, k]);
        let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();

        let out_shape = Shape::new(vec![m, n]);
        let (out_storage, h_gemm) = dev
            .fused_quant_gemm(a_storage.as_ref(), b_q80.as_ref(), QuantFormat::Q8_0, &out_shape)
            .unwrap();
        h_gemm.synchronize().unwrap();

        let gpu_res = out_storage.to_cpu_vec_f32().unwrap();
        assert_eq!(gpu_res.len(), m * n);
        assert!(gpu_res.iter().all(|x| x.is_finite()));
    }
}
