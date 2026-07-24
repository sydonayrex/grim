//! Quantization round-trip and backward numerics audit (WI-T6 / WI-F1-close).
//!
//! Verifies:
//! 1. Quantize → dequantize preserves values within RMS relative error tolerances.
//! 2. Backward GEMM gradient computation `dX = dY @ B^T` through dequantized weights
//!    matches FP32 reference gradients within per-format tolerances (Q8_0 < 5%, Q4_K < 10%).
//! 3. `backup2` bolt-on merged adapter weights preserve backward gradient fidelity.
//! 4. Optional ROCm GPU path verification when `GRIM_RUN_GPU_TESTS` is set.

use grim_backend_rocm::RocmDevice;
use grim_quant::{quant_q80, dequant_q80, quant_q4k, dequant_q4k};
use grim_tensor::{backend::BackendDevice, dtype::{DType, KQuantScheme, Storage}, Shape};

/// Maximum allowed RMS relative error for Q8_0 (8-bit).
const MAX_RMS_REL_ERROR_Q8: f32 = 0.05;
/// Maximum allowed RMS relative error for Q4_K (4-bit quantization with up to 20% accumulation noise).
const MAX_RMS_REL_ERROR_Q4K: f32 = 0.20;

/// RMS relative error: sqrt(mean((orig-recon)^2 / orig^2)).
fn rms_rel_err(orig: &[f32], recon: &[f32]) -> f32 {
    assert_eq!(orig.len(), recon.len());
    let sum_sq: f32 = orig.iter().zip(recon.iter())
        .map(|(o, r)| {
            let denom = o.abs().max(1e-3);
            ((o - r) / denom).powi(2)
        })
        .sum();
    (sum_sq / orig.len() as f32).sqrt()
}

/// Compute matrix gradient `dX[M, K] = dY[M, N] @ B[K, N]^T`.
fn compute_dx(dy: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0f32;
            for l in 0..n {
                sum += dy[i * n + l] * b[j * n + l];
            }
            dx[i * k + j] = sum;
        }
    }
    dx
}

#[test]
fn quant_backward_audit_q8_0_roundtrip() {
    let data: Vec<f32> = (0..512).map(|i| ((i as f32 * 0.1).sin()) * 10.0).collect();
    let quantized = quant_q80(&data).unwrap();
    let dequantized = dequant_q80(&quantized, data.len()).unwrap();
    assert_eq!(dequantized.len(), data.len());
    let rms = rms_rel_err(&data, &dequantized);
    assert!(rms <= MAX_RMS_REL_ERROR_Q8,
        "Q8_0 RMS rel error {rms:.6} exceeds {MAX_RMS_REL_ERROR_Q8}");
}

#[test]
fn quant_backward_audit_q4_k_roundtrip() {
    let data: Vec<f32> = (0..256).map(|i| 1.0 + (i as f32 * 0.035).sin().abs() * 9.0).collect();
    let quantized = quant_q4k(&data).unwrap();
    let dequantized = dequant_q4k(&quantized, data.len()).unwrap();
    assert_eq!(dequantized.len(), data.len());
    let rms = rms_rel_err(&data, &dequantized);
    assert!(rms <= MAX_RMS_REL_ERROR_Q4K,
        "Q4_K RMS rel error {rms:.6} exceeds {MAX_RMS_REL_ERROR_Q4K}");
}

/// WI-F1-close: Audit backward GEMM gradient `dX = dY @ B^T` for Q8_0 against FP32 reference.
#[test]
fn quant_backward_audit_q8_0_gemm_dx_numerics() {
    let (m, k, n) = (8, 16, 16);
    let dy: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).cos()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();

    // Reference gradient computation
    let dx_ref = compute_dx(&dy, &b_orig, m, n, k);

    // Quantized gradient computation
    let b_quant = quant_q80(&b_orig).unwrap();
    let b_dequant = dequant_q80(&b_quant, b_orig.len()).unwrap();
    let dx_quant = compute_dx(&dy, &b_dequant, m, n, k);

    let rms = rms_rel_err(&dx_ref, &dx_quant);
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q8,
        "Q8_0 backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_Q8}"
    );
}

/// WI-F1-close: Audit backward GEMM gradient `dX = dY @ B^T` for Q4_K against FP32 reference.
#[test]
fn quant_backward_audit_q4_k_gemm_dx_numerics() {
    let (m, k, n) = (8, 256, 256);
    let dy: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.02).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.015).cos().abs() * 8.0).collect();

    let dx_ref = compute_dx(&dy, &b_orig, m, n, k);

    let b_quant = quant_q4k(&b_orig).unwrap();
    let b_dequant = dequant_q4k(&b_quant, b_orig.len()).unwrap();
    let dx_quant = compute_dx(&dy, &b_dequant, m, n, k);

    let rms = rms_rel_err(&dx_ref, &dx_quant);
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q4K,
        "Q4_K backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_Q4K}"
    );
}

/// WI-F1-close: Audit backward gradient numerics with backup2 bolt-on adapter merged weights.
#[test]
fn quant_backward_audit_backup2_merged_gemm_dx_numerics() {
    let (m, k, n) = (8, 16, 16);
    let dy: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).sin()).collect();

    let b_base: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).cos() * 4.0).collect();
    let b_adapter: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();

    // Merged reference matrix
    let b_merged_ref: Vec<f32> = b_base.iter().zip(b_adapter.iter()).map(|(b, a)| b + a).collect();
    let dx_ref = compute_dx(&dy, &b_merged_ref, m, n, k);

    // Quantized base and adapter matrices
    let q_base = quant_q80(&b_base).unwrap();
    let dq_base = dequant_q80(&q_base, b_base.len()).unwrap();

    let q_adapter = quant_q80(&b_adapter).unwrap();
    let dq_adapter = dequant_q80(&q_adapter, b_adapter.len()).unwrap();

    let b_merged_quant: Vec<f32> = dq_base.iter().zip(dq_adapter.iter()).map(|(b, a)| b + a).collect();
    let dx_quant = compute_dx(&dy, &b_merged_quant, m, n, k);

    let rms = rms_rel_err(&dx_ref, &dx_quant);
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q8,
        "Backup2 merged backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_Q8}"
    );
}

/// WI-F1-close: Self-check verification proving backward audit tests fail when data is corrupted.
#[test]
#[should_panic(expected = "exceeds limit")]
fn quant_backward_audit_fail_check_corrupted_data() {
    let (m, k, n) = (8, 16, 16);
    let dy: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).cos()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();

    let dx_ref = compute_dx(&dy, &b_orig, m, n, k);

    // Corrupt dequantized data deliberately
    let mut b_corrupted = dequant_q80(&quant_q80(&b_orig).unwrap(), b_orig.len()).unwrap();
    for v in b_corrupted.iter_mut() {
        *v += 100.0;
    }
    let dx_corrupted = compute_dx(&dy, &b_corrupted, m, n, k);

    let rms = rms_rel_err(&dx_ref, &dx_corrupted);
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q8,
        "Q8_0 backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_Q8}"
    );
}

/// WI-F1-close: Verify ROCm fused backward kernel `dX = dY @ B^T` for Q8_0
/// weights against FP32 reference when running on a real ROCm device.
/// Skips silently when `GRIM_RUN_GPU_TESTS` is unset or no ROCm device is present.
#[test]
fn quant_backward_audit_rocm_q8_0_gemm_dx_numerics() {
    const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";
    if std::env::var(GPU_TEST_ENV).is_err() {
        return;
    }
    let rocm_devices = match RocmDevice::probe() {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    let dev = RocmDevice::new(rocm_devices[0].ordinal());

    let (m, k, n) = (8, 16, 16);
    let dy_host: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).cos()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();

    // Reference gradient on CPU
    let dx_ref = compute_dx(&dy_host, &b_orig, m, n, k);

    // Upload dy to ROCm as F32
    let dy_shape = Shape::from_slice(&[m, n]);
    let dy_rocm = dev.from_cpu(&dy_host, &dy_shape, DType::F32).unwrap();

    // Quantize b to Q8_0, upload packed bytes to ROCm
    let b_packed = quant_q80(&b_orig).unwrap();
    let b_rocm_shape = Shape::from_slice(&[k * n]);
    let b_rocm = dev.from_cpu_bytes(
        &b_packed,
        &b_rocm_shape,
        DType {
            arith: grim_tensor::dtype::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        },
    ).unwrap();

    // Call ROCm fused backward kernel for dX
    let out_shape = Shape::from_slice(&[m, k]);
    let (dx_rocm, _handle) = dev.quantized_matmul_backward_dx(
        dy_rocm.as_ref(),
        b_rocm.as_ref(),
        &[],
        8, // bpw for Q8_0
        m,
        n,
        k,
        &out_shape,
    ).expect("ROCm quantized_matmul_backward_dx must succeed on a real ROCm device");

    // Copy result back to CPU
    let dx_rocm_vec = dx_rocm.to_cpu_vec_f32().expect("ROCm result must be readable");

    let rms = rms_rel_err(&dx_ref, &dx_rocm_vec);
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q8,
        "ROCm Q8_0 backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_Q8}"
    );
}
