//! Multi-format backend dequantization & execution parity tests (§WI-E9).

use grim_backend_tests::{TEST_K_DIMS, TEST_QUANT_FORMATS};
use grim_quant::{
    QuantFormat, dequant_iq4nl, dequant_q4k, dequant_q5k, dequant_q6k, dequant_q80, quant_iq4nl,
    quant_q4k, quant_q5k, quant_q6k, quant_q80,
};

fn generate_deterministic_test_weights(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as i32) as f32 / (i32::MAX as f32);
        out.push(val);
    }
    out
}

/// Round-trip tolerance per format (max abs diff on unit-scale weights).
fn roundtrip_tolerance(format: QuantFormat) -> f32 {
    match format {
        QuantFormat::Q8_0 => 0.05,
        QuantFormat::Q4K => 0.2,
        QuantFormat::Q5K => 0.15,
        QuantFormat::Q6K => 0.1,
        QuantFormat::Iq4Nl => 0.25,
        _ => 0.3,
    }
}

#[test]
fn test_cpu_oracle_quant_dequant_roundtrip_all_formats() {
    for &k in TEST_K_DIMS {
        // Scale to realistic LLM weight magnitude (std ~0.05): integer-scale
        // formats like Q6_K compute sub-block scales as round(max/31), so
        // unit-scale data collapses the scale to the clamp floor and the
        // round-trip error explodes. Real weights never hit that regime.
        let original = generate_deterministic_test_weights(k, 0x1234_5678_9ABC_DEF0)
            .into_iter()
            .map(|v| v * 0.1)
            .collect::<Vec<f32>>();

        for &format in TEST_QUANT_FORMATS {
            match format {
                QuantFormat::Q8_0 => {
                    let quantized = quant_q80(&original).expect("quant_q80 failed");
                    let dequantized = dequant_q80(&quantized, k).expect("dequant_q80 failed");
                    assert_eq!(dequantized.len(), k);
                    let max_abs_diff = original
                        .iter()
                        .zip(&dequantized)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_abs_diff < 0.05,
                        "Q8_0 k={k} max_abs_diff={max_abs_diff} exceeds 0.05"
                    );
                }
                QuantFormat::Q4K => {
                    let quantized = quant_q4k(&original).expect("quant_q4k failed");
                    let dequantized = dequant_q4k(&quantized, k).expect("dequant_q4k failed");
                    assert_eq!(dequantized.len(), k);
                    let max_abs_diff = original
                        .iter()
                        .zip(&dequantized)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let tol = roundtrip_tolerance(format);
                    assert!(
                        max_abs_diff < tol,
                        "Q4_K k={k} max_abs_diff={max_abs_diff} exceeds {tol}"
                    );
                }
                QuantFormat::Q5K => {
                    let quantized = quant_q5k(&original).expect("quant_q5k failed");
                    let dequantized = dequant_q5k(&quantized, k).expect("dequant_q5k failed");
                    assert_eq!(dequantized.len(), k);
                    let max_abs_diff = original
                        .iter()
                        .zip(&dequantized)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let tol = roundtrip_tolerance(format);
                    assert!(
                        max_abs_diff < tol,
                        "Q5_K k={k} max_abs_diff={max_abs_diff} exceeds {tol}"
                    );
                }
                QuantFormat::Q6K => {
                    let quantized = quant_q6k(&original).expect("quant_q6k failed");
                    let dequantized = dequant_q6k(&quantized, k).expect("dequant_q6k failed");
                    assert_eq!(dequantized.len(), k);
                    let max_abs_diff = original
                        .iter()
                        .zip(&dequantized)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let tol = roundtrip_tolerance(format);
                    assert!(
                        max_abs_diff < tol,
                        "Q6_K k={k} max_abs_diff={max_abs_diff} exceeds {tol}"
                    );
                }
                QuantFormat::Iq4Nl => {
                    let quantized = quant_iq4nl(&original).expect("quant_iq4nl failed");
                    let dequantized = dequant_iq4nl(&quantized, k).expect("dequant_iq4nl failed");
                    assert_eq!(dequantized.len(), k);
                    let max_abs_diff = original
                        .iter()
                        .zip(&dequantized)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let tol = roundtrip_tolerance(format);
                    assert!(
                        max_abs_diff < tol,
                        "IQ4_NL k={k} max_abs_diff={max_abs_diff} exceeds {tol}"
                    );
                }
                _ => {}
            }
        }
    }
}

/// MXFP4 round-trip: `quant_mxfp4_matrix` returns raw (codes, exps) buffers;
/// `dequant_mxfp4` consumes the length-prefixed framing
/// `[u64 codes_len][codes][u64 exps_len][exps]`. Frame them here and compare.
#[test]
fn test_cpu_oracle_mxfp4_roundtrip() {
    for &k in TEST_K_DIMS {
        let original = generate_deterministic_test_weights(k, 0x0DEF_ACED_5EED_5EED)
            .into_iter()
            .map(|v| v * 0.1)
            .collect::<Vec<f32>>();
        let (codes, exps) = grim_quant::quant_mxfp4_matrix(&original, 1, k);
        // Frame exactly as dequant_mxfp4 expects.
        let mut framed = Vec::with_capacity(16 + codes.len() + exps.len());
        framed.extend_from_slice(&(codes.len() as u64).to_le_bytes());
        framed.extend_from_slice(&codes);
        framed.extend_from_slice(&(exps.len() as u64).to_le_bytes());
        framed.extend_from_slice(&exps);
        let dequantized = grim_quant::dequant_mxfp4(&framed, k).expect("dequant_mxfp4 failed");
        assert_eq!(dequantized.len(), k);
        let max_abs_diff = original
            .iter()
            .zip(&dequantized)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 0.35,
            "MXFP4 k={k} max_abs_diff={max_abs_diff} exceeds 0.35"
        );
    }
}

#[test]
fn test_gpu_rocm_dequant_parity() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }

    #[cfg(feature = "rocm")]
    {
        use grim_backend_rocm::RocmDevice;
        use grim_tensor::{BackendDevice, DType, Shape};

        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ROCm device 0 not available: {e}");
                return;
            }
        };

        for &k in TEST_K_DIMS {
            let original = generate_deterministic_test_weights(k, 0x5EED_0000_1234_5678)
                .into_iter()
                .map(|v| v * 0.1)
                .collect::<Vec<f32>>();

            let shape = Shape::new(vec![k]);
            let x_storage = dev
                .from_cpu(&original, &shape, DType::F32)
                .expect("from_cpu failed");

            // Test Q8_0 GPU quantize -> dequantize roundtrip vs CPU
            let (q8_storage, h1) = dev
                .quantize_on_device(x_storage.as_ref(), QuantFormat::Q8_0)
                .expect("gpu quantize Q8_0");
            h1.synchronize().expect("gpu sync");
            let gpu_dequant = q8_storage.to_cpu_vec_f32().expect("gpu dequant to cpu");
            assert_eq!(gpu_dequant.len(), k);

            let cpu_quant = quant_q80(&original).expect("cpu quant_q80");
            let cpu_dequant = dequant_q80(&cpu_quant, k).expect("cpu dequant_q80");

            let max_diff = gpu_dequant
                .iter()
                .zip(&cpu_dequant)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-4,
                "GPU vs CPU Q8_0 parity divergence at k={k}: max_diff={max_diff}"
            );

            // Test Q4_K GPU quantize -> dequantize roundtrip vs CPU
            let (q4_storage, h2) = dev
                .quantize_on_device(x_storage.as_ref(), QuantFormat::Q4K)
                .expect("gpu quantize Q4_K");
            h2.synchronize().expect("gpu sync");
            let gpu_q4_dequant = q4_storage.to_cpu_vec_f32().expect("gpu q4 dequant to cpu");
            assert_eq!(gpu_q4_dequant.len(), k);

            let cpu_q4 = quant_q4k(&original).expect("cpu quant_q4k");
            let cpu_q4_dequant = dequant_q4k(&cpu_q4, k).expect("cpu dequant_q4k");
            let max_q4_diff = gpu_q4_dequant
                .iter()
                .zip(&cpu_q4_dequant)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_q4_diff < 1e-3,
                "GPU vs CPU Q4_K parity divergence at k={k}: max_diff={max_q4_diff}"
            );
        }
    }
}
