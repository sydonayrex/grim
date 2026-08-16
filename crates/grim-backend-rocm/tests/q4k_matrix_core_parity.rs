//! Parity test for Q4_K matrix-core (WMMA/MFMA) fused GEMM kernel (WI-B).
//!
//! Verifies that when `Q4kGemmConfig` is enabled, the backend compiles and launches
//! `grim_fused_dequant_gemm_q4k_wmma` / `grim_fused_dequant_gemm_q4k_mfma`, producing
//! outputs equivalent to the scalar fallback path within F16 numerical tolerance.
//!
//! TODO(gpu-verify): Execute this test on a physical RDNA3/4 (7900 XTX / gfx110x / gfx1200)
//! or CDNA2/3 (MI200/MI300 / gfx90a / gfx942) host with `GRIM_RUN_GPU_TESTS=1`.
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test q4k_matrix_core_parity
//! RESULT: inputs are uploaded to the ROCm device via `BackendDevice::from_cpu` /
//!   `from_cpu_bytes` (with `quant_q4k` packing) before `quantized_matmul`, which
//!   requires `RocmStorage`. The scalar (decode-gemm-off) and matrix-core
//!   (decode-gemm-on) paths are compared for parity within 1e-3.

use grim_backend_rocm::RocmDevice;
use grim_quant::quant_q4k;
use grim_tensor::{
    ArithType, BackendDevice, DType, KQuantScheme, Shape, Storage,
};
use std::panic;

/// Build a RocmDevice if the GPU test environment is enabled.
fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    match panic::catch_unwind(|| {
        RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm")
    }) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

/// Compute max absolute difference between two float slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let diff = (a[i] - b[i]).abs();
        if diff > max {
            max = diff;
        }
    }
    max
}

#[test]
fn test_q4k_matrix_core_vs_scalar_parity() {
    let dev = match gpu_device() {
        Some(d) => d,
        None => {
            eprintln!("[SKIP] test_q4k_matrix_core_vs_scalar_parity requires GRIM_RUN_GPU_TESTS=1");
            return;
        }
    };

    // Test matrix dimensions: M=16, N=16, K=256 (1 Q4_K block per row)
    let (m, n, k) = (16, 16, 256);

    // Input A (f32 activations) and B (f32 reference weights).
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.015).cos().abs() * 8.0)
        .collect();

    // Pack B to Q4_K on the host (matches golden_q4k_gpu_mutation).
    let b_packed = quant_q4k(&b_orig).expect("quant_q4k");

    let a_shape = Shape::new(vec![m, k]);
    let b_shape = Shape::new(vec![k, n]);
    let out_shape = Shape::new(vec![m, n]);

    // Upload to the ROCm device — `quantized_matmul` requires `RocmStorage`,
    // not `CpuStorage` (the previous version passed CPU tensors and hit
    // "matmul: input a is not RocmStorage").
    let a_dev = BackendDevice::from_cpu(&dev, &a_data, &a_shape, DType::F32)
        .expect("upload A to device");
    let q4k_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let b_dev = BackendDevice::from_cpu_bytes(&dev, &b_packed, &b_shape, q4k_dtype)
        .expect("upload packed B to device");

    // 1. Run with matrix-core config disabled (scalar path)
    dev.set_decode_gemm_enabled(false);
    let (out_scalar_storage, handle_scalar) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &out_shape,
        )
        .expect("scalar quantized_matmul should succeed");
    handle_scalar.synchronize().expect("sync failed");
    let out_scalar = out_scalar_storage
        .to_cpu_vec_f32()
        .expect("to_cpu_vec_f32 failed");

    // 2. Run with matrix-core config enabled (WMMA/MFMA tiled path)
    dev.set_decode_gemm_enabled(true);
    let (out_tiled_storage, handle_tiled) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &out_shape,
        )
        .expect("tiled quantized_matmul should succeed");
    handle_tiled.synchronize().expect("sync failed");
    let out_tiled = out_tiled_storage
        .to_cpu_vec_f32()
        .expect("to_cpu_vec_f32 failed");

    // 3. Assert parity within F16 tolerance
    let diff = max_abs_diff(&out_scalar, &out_tiled);
    assert!(
        diff < 1e-3,
        "Q4_K matrix-core output diff {} exceeds threshold 1e-3",
        diff
    );
}
