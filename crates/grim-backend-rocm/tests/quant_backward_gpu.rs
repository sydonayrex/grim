//! GPU backward gradient numerics for quantized matmuls (A.5 coverage gap).
//!
//! Verifies the ROCm fused backward kernel `dX = dY @ B^T` for Q8_0 weights
//! against an FP32 CPU reference.  This test was originally in
//! `grim-quant/tests/quant_backward_audit.rs` but was gated off with
//! `#[cfg(any())]` after `grim-quant` dropped the `rocm` feature to break a
//! dependency cycle.  Moved here so it can actually compile and run.
//!
//! Run with:
//!   GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test quant_backward_gpu -- --ignored --nocapture

use grim_quant::quant_q80;
use grim_tensor::{
    CoreTensorOps, MemoryOps, QuantOps, QuantizedMatmulBackwardResiduals, Shape,
    dtype::{ArithType, DType, KQuantScheme, Storage},
};

/// Maximum allowed RMS relative error for Q8_0 (8-bit).
const MAX_RMS_REL_ERROR_Q8: f32 = 0.05;

/// RMS relative error: sqrt(mean((orig-recon)^2 / orig^2)).
fn rms_rel_err(orig: &[f32], recon: &[f32]) -> f32 {
    assert_eq!(orig.len(), recon.len());
    let sum_sq: f32 = orig
        .iter()
        .zip(recon.iter())
        .map(|(o, r)| {
            let denom = o.abs().max(1e-3);
            ((o - r) / denom).powi(2)
        })
        .sum();
    (sum_sq / orig.len() as f32).sqrt()
}

/// Compute matrix gradient `dX[M, K] = dY[M, N] @ B[K, N]^T` on CPU.
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
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn quant_backward_rocm_q8_0_gemm_dx_numerics() {
    let rocm_devices = match grim_backend_rocm::RocmDevice::probe() {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    let dev = grim_backend_rocm::RocmDevice::try_new(rocm_devices[0].ordinal())
        .expect("RocmDevice::try_new should succeed for probed device");

    let (m, k, n) = (8, 32, 32);
    let dy_host: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).cos()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();

    let dy_shape = Shape::from_slice(&[m, n]);
    let dy_rocm = dev.from_cpu(&dy_host, &dy_shape, DType::F32).unwrap();

    let mut b_trans = vec![0.0f32; k * n];
    for j in 0..k {
        for l in 0..n {
            b_trans[l * k + j] = b_orig[j * n + l];
        }
    }
    let b_packed = quant_q80(&b_trans).unwrap();
    let b_rocm_shape = Shape::from_slice(&[k * n]);

    // Dequantize b to FP32 for exact CPU reference gradient comparison
    let b_dequant = grim_quant::dequant_q80(&b_packed, k * n).unwrap();
    let mut b_dequant_untrans = vec![0.0f32; k * n];
    for l in 0..n {
        for j in 0..k {
            b_dequant_untrans[j * n + l] = b_dequant[l * k + j];
        }
    }

    // Reference gradient on CPU using dequantized weights
    let dx_ref = compute_dx(&dy_host, &b_dequant_untrans, m, n, k);

    let b_rocm = dev
        .from_cpu_bytes(
            &b_packed,
            &b_rocm_shape,
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::Q80),
            },
        )
        .unwrap();

    // Call ROCm fused backward kernel for dX
    let out_shape = Shape::from_slice(&[m, k]);
    let residuals = QuantizedMatmulBackwardResiduals::default();
    let (dx_rocm, _handle) = dev
        .quantized_matmul_backward_dx(
            dy_rocm.as_ref(),
            b_rocm.as_ref(),
            &[],
            8, // bpw for Q8_0
            m,
            n,
            k,
            &out_shape,
            Some(&residuals),
        )
        .expect("ROCm quantized_matmul_backward_dx must succeed on a real ROCm device");

    // Copy result back to CPU
    let dx_rocm_vec = dx_rocm
        .to_cpu_vec_f32()
        .expect("ROCm result must be readable");

    let rms = rms_rel_err(&dx_ref, &dx_rocm_vec);
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q8,
        "ROCm Q8_0 backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_Q8}"
    );
}
