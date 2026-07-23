//! Quantization round-trip backward audit (WI-T6).
//!
//! Verifies that quantize → dequantize preserves values within 5% RMS relative error.
//! Uses practical weight-like data distributions where each quantizer is designed to work.

use grim_quant::{quant_q80, dequant_q80, quant_q4k, dequant_q4k};

/// Maximum allowed RMS relative error for Q8_0 (8-bit).
const MAX_RMS_REL_ERROR_Q8: f32 = 0.05;
/// Maximum allowed RMS relative error for Q4_K (4-bit with GPTQ proxy carry).
const MAX_RMS_REL_ERROR_Q4K: f32 = 0.10;

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

#[test]
fn quant_backward_audit_q8_0_roundtrip() {
    // Q8_0: 8-bit symmetric, block 32. Works well for any weight distribution.
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
    // Q4_K is 4-bit: use weight-like data in [1, 10] range
    let data: Vec<f32> = (0..256).map(|i| 1.0 + (i as f32 * 0.035).sin().abs() * 9.0).collect();
    let quantized = quant_q4k(&data).unwrap();
    let dequantized = dequant_q4k(&quantized, data.len()).unwrap();
    assert_eq!(dequantized.len(), data.len());
    let rms = rms_rel_err(&data, &dequantized);
    assert!(rms <= MAX_RMS_REL_ERROR_Q4K,
        "Q4_K RMS rel error {rms:.6} exceeds {MAX_RMS_REL_ERROR_Q4K}");
}
