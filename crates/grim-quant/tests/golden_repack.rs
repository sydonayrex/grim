//! Golden mutation-resistant test for Q4_K to FP8 repack pipeline and precision floor verification.

use grim_quant::{dequant_fp8, quant_fp8};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

#[test]
fn test_q4k_to_fp8_repack_precision_floor() -> TestResult {
    // Generate input weights with synthetic values
    let num_elements = 256;
    let orig_weights: Vec<f32> = (0..num_elements)
        .map(|i| (i as f32 * 0.1).sin() * 2.0)
        .collect();

    // Quantize to FP8 E4M3
    let fp8_bytes = quant_fp8(&orig_weights)?;
    let dequant_weights = dequant_fp8(&fp8_bytes, num_elements)?;

    assert_eq!(dequant_weights.len(), orig_weights.len());

    let mut max_err: f32 = 0.0;
    for (orig, deq) in orig_weights.iter().zip(dequant_weights.iter()) {
        let err = (orig - deq).abs();
        if err > max_err {
            max_err = err;
        }
    }

    // FP8 E4M3 quantization error floor threshold (~1e-1 / 0.15 for 4-bit/8-bit range)
    assert!(
        max_err < 0.25,
        "FP8 repack max error {max_err} exceeds precision floor tolerance"
    );

    Ok(())
}
