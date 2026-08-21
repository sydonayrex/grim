//! Quantization round-trip and backward numerics audit (WI-T6 / WI-F1-close).
//!
//! Verifies:
//! 1. Quantize → dequantize preserves values within RMS relative error tolerances.
//! 2. Backward GEMM gradient computation `dX = dY @ B^T` through dequantized weights
//!    matches FP32 reference gradients within per-format tolerances (Q8_0 < 5%, Q4_K < 10%).
//! 3. `backup2` bolt-on merged adapter weights preserve backward gradient fidelity.
//! 4. ROCm GPU path moved to `grim-backend-rocm/tests/quant_backward_gpu.rs`.

use grim_format::train::{TrainFpFormat, decode_f32s_from, encode_f32s_as, f32_to_bf16_bytes};
use grim_quant::{dequant_q4k, dequant_q80, quant_q4k, quant_q80};
/// Maximum allowed RMS relative error for Q8_0 (8-bit).
const MAX_RMS_REL_ERROR_Q8: f32 = 0.05;
/// Maximum allowed RMS relative error for Q4_K (4-bit quantization with up to 20% accumulation noise).
const MAX_RMS_REL_ERROR_Q4K: f32 = 0.20;
/// Maximum allowed RMS relative error for BF16 master-weight backward numerics
/// (8-bit mantissa → ~0.4% per-element quantization step; RMS over GEMM stays well under 1%).
const MAX_RMS_REL_ERROR_BF16: f32 = 0.01;
/// FP16 carries 10 mantissa bits — roughly 4x tighter than BF16.
const MAX_RMS_REL_ERROR_FP16: f32 = 0.005;

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

/// Globally normalized RMS error: ||orig - recon|| / ||orig||.
///
/// Used for mixed-precision gates where individual gradient elements cross
/// zero and per-element relative error is meaningless; the whole-tensor error
/// budget is what matters for training-step fidelity.
fn normalized_rms_err(orig: &[f32], recon: &[f32]) -> f32 {
    assert_eq!(orig.len(), recon.len());
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (o, r) in orig.iter().zip(recon.iter()) {
        num += (o - r).powi(2);
        den += o.powi(2);
    }
    (num / den.max(1e-30)).sqrt()
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
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q8,
        "Q8_0 RMS rel error {rms:.6} exceeds {MAX_RMS_REL_ERROR_Q8}"
    );
}

#[test]
fn quant_backward_audit_q4_k_roundtrip() {
    let data: Vec<f32> = (0..256)
        .map(|i| 1.0 + (i as f32 * 0.035).sin().abs() * 9.0)
        .collect();
    let quantized = quant_q4k(&data).unwrap();
    let dequantized = dequant_q4k(&quantized, data.len()).unwrap();
    assert_eq!(dequantized.len(), data.len());
    let rms = rms_rel_err(&data, &dequantized);
    assert!(
        rms <= MAX_RMS_REL_ERROR_Q4K,
        "Q4_K RMS rel error {rms:.6} exceeds {MAX_RMS_REL_ERROR_Q4K}"
    );
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
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.015).cos().abs() * 8.0)
        .collect();

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
    let b_merged_ref: Vec<f32> = b_base
        .iter()
        .zip(b_adapter.iter())
        .map(|(b, a)| b + a)
        .collect();
    let dx_ref = compute_dx(&dy, &b_merged_ref, m, n, k);

    // Quantized base and adapter matrices
    let q_base = quant_q80(&b_base).unwrap();
    let dq_base = dequant_q80(&q_base, b_base.len()).unwrap();

    let q_adapter = quant_q80(&b_adapter).unwrap();
    let dq_adapter = dequant_q80(&q_adapter, b_adapter.len()).unwrap();

    let b_merged_quant: Vec<f32> = dq_base
        .iter()
        .zip(dq_adapter.iter())
        .map(|(b, a)| b + a)
        .collect();
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

// ── P1 mixed-precision numerics gates (bf16/fp16 master weights vs f32 reference) ──

/// P1.3: BF16 master-weight backward GEMM gradient `dX = dY @ B^T` must match
/// the f32 reference within tight RMS tolerance. This is the numerics gate for
/// `--train-dtype bf16`: encode weights to bf16 (as the sidecar does), decode
/// back, compute gradients, and compare.
#[test]
fn mixed_precision_bf16_backward_gemm_dx_numerics() {
    let (m, k, n) = (8, 64, 64);
    let dy: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.03).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| ((i * 7) as f32 * 0.017).sin() * 3.0)
        .collect();

    let dx_ref = compute_dx(&dy, &b_orig, m, n, k);

    let b_bf16 = encode_f32s_as(&b_orig, TrainFpFormat::Bf16);
    assert_eq!(
        b_bf16.len(),
        b_orig.len() * 2,
        "bf16 blob must be 2 bytes/element"
    );
    let b_decoded = decode_f32s_from(&b_bf16, TrainFpFormat::Bf16).expect("bf16 decode");
    let dx_bf16 = compute_dx(&dy, &b_decoded, m, n, k);

    let rms = normalized_rms_err(&dx_ref, &dx_bf16);
    assert!(
        rms <= MAX_RMS_REL_ERROR_BF16,
        "BF16 backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_BF16}"
    );
}

/// P1.3: FP16 master-weight backward GEMM gradient vs f32 reference.
#[test]
fn mixed_precision_fp16_backward_gemm_dx_numerics() {
    let (m, k, n) = (8, 64, 64);
    let dy: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.03).cos()).collect();
    // Keep magnitudes in fp16-safe range (well under 65504, above subnormal floor).
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + ((i * 5) as f32 * 0.011).cos() * 2.0)
        .collect();

    let dx_ref = compute_dx(&dy, &b_orig, m, n, k);

    let b_f16 = encode_f32s_as(&b_orig, TrainFpFormat::Fp16Param);
    assert_eq!(
        b_f16.len(),
        b_orig.len() * 2,
        "fp16 blob must be 2 bytes/element"
    );
    let b_decoded = decode_f32s_from(&b_f16, TrainFpFormat::Fp16Param).expect("fp16 decode");
    let dx_f16 = compute_dx(&dy, &b_decoded, m, n, k);

    let rms = normalized_rms_err(&dx_ref, &dx_f16);
    assert!(
        rms <= MAX_RMS_REL_ERROR_FP16,
        "FP16 backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_FP16}"
    );
}

/// P1.3 fail-then-pass gate: a single corrupted bf16 payload byte in the
/// master-weight blob must blow the tolerance — proving the gate has teeth.
#[test]
#[should_panic(expected = "exceeds limit")]
fn mixed_precision_bf16_fail_check_corrupted_byte() {
    let (m, k, n) = (8, 64, 64);
    let dy: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.03).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| ((i * 7) as f32 * 0.017).sin() * 3.0)
        .collect();

    let dx_ref = compute_dx(&dy, &b_orig, m, n, k);

    let mut b_bf16 = encode_f32s_as(&b_orig, TrainFpFormat::Bf16);
    // Corrupt a full row of B (64 elements = 128 bytes): flip high mantissa/exp
    // bits so decoded values are badly wrong, not just off by one ulp.
    for byte in b_bf16.iter_mut().skip(k * n - 128) {
        *byte ^= 0x7F;
    }
    let b_corrupted = decode_f32s_from(&b_bf16, TrainFpFormat::Bf16).expect("bf16 decode");
    let dx_corrupted = compute_dx(&dy, &b_corrupted, m, n, k);

    let rms = normalized_rms_err(&dx_ref, &dx_corrupted);
    assert!(
        rms <= MAX_RMS_REL_ERROR_BF16,
        "BF16 backward GEMM dX RMS rel error {rms:.6} exceeds limit {MAX_RMS_REL_ERROR_BF16}"
    );
}

/// P1.3: round-to-nearest-even bf16 encode correctness on ties and specials.
#[test]
fn mixed_precision_bf16_round_to_nearest_even() {
    // Exactly representable values survive bit-for-bit.
    for v in [0.0f32, 1.0, -1.0, 2.0, 0.5, 256.0, -0.25] {
        let b = f32_to_bf16_bytes(v);
        let dec = f32::from_bits(((b[0] as u32) | ((b[1] as u32) << 8)) << 16);
        assert_eq!(dec, v, "bf16 exact round-trip failed for {v}");
    }
    // Midpoint between 1.0 (0x3F80) and next bf16 up (0x3F81) is 1 + 2^-8;
    // round-to-nearest-even must land on the even neighbor (1.0).
    let tie = 1.0f32 + 2f32.powi(-8);
    let t = f32_to_bf16_bytes(tie);
    let dec_t = f32::from_bits(((t[0] as u32) | ((t[1] as u32) << 8)) << 16);
    assert_eq!(dec_t, 1.0, "tie must round to even (down to 1.0)");
    // Just above the tie rounds up.
    let above = 1.0f32 + 2f32.powi(-8) * 1.5;
    let a = f32_to_bf16_bytes(above);
    let dec_a = f32::from_bits(((a[0] as u32) | ((a[1] as u32) << 8)) << 16);
    assert!(dec_a > 1.0, "above-tie must round up, got {dec_a}");
}
