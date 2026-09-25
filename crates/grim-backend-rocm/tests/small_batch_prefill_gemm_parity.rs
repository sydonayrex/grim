//! Task 9 (Plan 3): Unit test for small-batch prefill GEMM parity.
//! Compares batched M in {2, 16, 27, 32, 64} across multiple K in {256, 1024}
//! and irregular N in {127, 513, 1024, 4608} against CPU unquantized GEMM reference.

use grim_backend_rocm::RocmDevice;
use grim_tensor::CoreTensorOps;
use grim_tensor::QuantOps;
use grim_tensor::Shape;
use grim_tensor::dtype::{DType, QuantFormat};

#[test]
#[ignore]
fn test_small_batch_prefill_gemm_parity() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_small_batch_prefill_gemm_parity");
        return;
    }
    let dev = RocmDevice::new(0);

    let test_cases: Vec<(usize, usize, usize)> = vec![
        (2, 127, 256),
        (16, 513, 256),
        (27, 1024, 1024),
        (32, 1024, 1024),
        (64, 4608, 1024),
    ];

    for (m, n, k) in test_cases {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 * 0.013).sin() * 2.0)
            .collect();
        let b_unquant: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.017).cos() * 1.5)
            .collect();

        // CPU unquantized reference: C = A * B^T -> [M, N]
        let mut cpu_ref = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[row * k + p] * b_unquant[col * k + p];
                }
                cpu_ref[row * n + col] = sum;
            }
        }

        let a_shape = Shape::new(vec![m, k]);
        let a_storage = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();

        let b_shape = Shape::new(vec![n, k]);
        let b_raw = dev.from_cpu(&b_unquant, &b_shape, DType::F32).unwrap();
        let (b_q80, h_quant) = dev
            .quantize_on_device(b_raw.as_ref(), QuantFormat::Q8_0)
            .unwrap();
        h_quant.synchronize().unwrap();

        let out_shape = Shape::new(vec![m, n]);
        let (out_storage, h_gemm) = dev
            .fused_quant_gemm(a_storage.as_ref(), b_q80.as_ref(), QuantFormat::Q8_0, &out_shape)
            .unwrap();
        h_gemm.synchronize().unwrap();

        let gpu_res = out_storage.to_cpu_vec_f32().unwrap();
        assert_eq!(gpu_res.len(), m * n);

        let mut max_abs_diff = 0.0f32;
        let mut max_ref = 0.0f32;
        for i in 0..m * n {
            let diff = (gpu_res[i] - cpu_ref[i]).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            if cpu_ref[i].abs() > max_ref {
                max_ref = cpu_ref[i].abs();
            }
        }
        let rel_err = max_abs_diff / max_ref.max(1.0);
        eprintln!(
            "[test_small_batch_prefill_gemm_parity] M={m} N={n} K={k} max_diff={max_abs_diff:.4} rel_err={rel_err:.4}"
        );
        assert!(
            rel_err < 0.05,
            "M={m} N={n} K={k} relative error {rel_err:.4} exceeds 5% (max_diff={max_abs_diff}, max_ref={max_ref})"
        );
    }
}
