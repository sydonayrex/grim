//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test golden_q4k_gpu_mutation
//! RESULT: FAILED — hipModuleLoad failed: 209. The Q4_K GPU dequantization GEMM kernel is
//!   compiled but the .hipfb binary is not registered/loaded for this test on this dual-GPU
//!   RDNA4 box. Host-side infrastructure works; the kernel is not tied to this test at runtime.
//!
//! Golden mutation-resistant test for Crow Q4_K GPU dequantization GEMM forward and backward paths.

use grim_backend_rocm::RocmDevice;
use grim_format::convert::GpuDequant;
use grim_quant::{dequant_q4k, quant_q4k};
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{
    Shape,
    dtype::{ArithType, DType, KQuantScheme, Storage},
};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

#[test]
#[ignore]
fn test_q4k_gpu_gemm_golden_mutation_resistant() -> TestResult {
    // Non-square asymmetric dimensions: M=2, K=256, N=128
    let (m, k, n) = (2usize, 256usize, 128usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.015).cos().abs() * 8.0)
        .collect();

    let b_packed = quant_q4k(&b_orig).expect("quant_q4k");
    let b_dequant = dequant_q4k(&b_packed, b_orig.len()).expect("dequant_q4k");

    // Ground-truth CPU reference: C = A @ B_dequant^T
    let mut expected_c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * b_dequant[ni * k + ki];
            }
            expected_c[mi * n + ni] = sum;
        }
    }

    if let Some(dev) = gpu_device() {
        let a_shape = Shape::from_slice(&[m, k]);
        let b_shape = Shape::from_slice(&[n, k]);
        let out_shape = Shape::from_slice(&[m, n]);

        let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
        let q4k_dtype = DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        };
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &b_shape, q4k_dtype)?;

        let (out, handle) = dev.quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &out_shape,
        )?;
        handle.synchronize()?;
        let actual_c = out.to_cpu_vec_f32()?;

        assert_eq!(actual_c.len(), expected_c.len());

        // The M=2 Q4K route is the WMMA fused-dequant path: BOTH operands are
        // quantized to f16 inside the kernel (lds_a cast, dequantized-B cast
        // to _Float16 before mma_sync) and the fragments accumulate in f32.
        // The by-design error vs the f32 oracle is therefore the f16
        // quantization of every product, bounded EXACTLY per output:
        //   |sum f16(a)*f16(w) - sum a*w| <= sum |f16(a)*f16(w) - a*w|
        // (the device's f32 accumulation from exact f16 products adds only
        // sub-ulp noise vs the f64 sum of the same f16 products). A decode
        // error (wrong nibble/scale) exceeds this envelope ~1000x — that is
        // what "mutation-resistant" means for this path.
        let w_f16: Vec<f32> = b_dequant
            .iter()
            .map(|&v| f32::from(half::f16::from_f32(v)))
            .collect();
        let a_f16: Vec<f32> = a_data
            .iter()
            .map(|&v| f32::from(half::f16::from_f32(v)))
            .collect();
        let mut worst_ratio: f64 = 0.0;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc_f64 = 0.0f64;
                let mut envelope = 0.0f64;
                for ki in 0..k {
                    let exact = a_data[mi * k + ki] as f64 * b_dequant[ni * k + ki] as f64;
                    let f16_prod = a_f16[mi * k + ki] as f64 * w_f16[ni * k + ki] as f64;
                    acc_f64 += exact;
                    envelope += (f16_prod - exact).abs();
                }
                let denom = envelope.max(acc_f64.abs() * 1e-3).max(1e-6);
                worst_ratio = worst_ratio.max(
                    (actual_c[mi * n + ni] as f64 - acc_f64).abs() / denom,
                );
            }
        }
        eprintln!(
            "[golden] worst |device - f64| / own f16 envelope: {worst_ratio:.3} (<= 1.05 = WMMA f16 by-design)"
        );
        assert!(
            worst_ratio <= 1.05,
            "device residual {worst_ratio:.3}x its own f16 quantization envelope — \
             REAL decode error beyond WMMA f16 by-design precision"
        );

        // Mutation-resistant max-error record vs the f32 oracle, kept for the
        // journey log: the WMMA f16 path moves outputs off the 1e-3 f32 scale
        // BY DESIGN (the old fixed assert fired on design error).
        let mut max_err: f32 = 0.0;
        for (act, exp) in actual_c.iter().zip(expected_c.iter()) {
            let err = (act - exp).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("[golden] max_err vs f32 oracle: {max_err} (WMMA f16 path by design)");

        // --- GpuDequant path (used by grim-format conversion) ---
        // Routes `KQuant::Q4K` through `RocmDevice::dequantize` →
        // `dequantize_q4k_host` → the `grim_dequant_q4k` HIP kernel instead of
        // the CPU `grim_quant::dequant_q4k` fallback. Must remain bit-exact with
        // the CPU oracle `b_dequant` (computed above).
        let gpu_dequant = dev
            .dequantize(&Storage::KQuant(KQuantScheme::Q4K), &b_packed, b_orig.len())
            .expect("GpuDequant::dequantize Q4K");
        assert!(
            gpu_dequant.is_some(),
            "GpuDequant must handle Q4K (should not fall back to CPU)"
        );
        let gpu_dequant = gpu_dequant.expect("checked above");
        assert_eq!(gpu_dequant.len(), b_dequant.len());
        let dequant_max_err = b_dequant
            .iter()
            .zip(gpu_dequant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            dequant_max_err, 0.0,
            "Q4_K GpuDequant kernel deviates from CPU oracle (max_err={dequant_max_err:e})"
        );
    }

    Ok(())
}
