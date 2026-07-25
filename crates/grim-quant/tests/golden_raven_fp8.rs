use grim_quant::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};

#[test]
fn test_raven_fp8_dequant_repack_golden_mutation_resistant() {
    // Hand-build a vector of known floats
    let input_floats = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0];
    
    // Repack / Quantize to FP8 E4M3 bytes
    let fp8_bytes: Vec<u8> = input_floats.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();

    // Dequantize back to F32
    let dequant_floats: Vec<f32> = fp8_bytes.iter().map(|&b| fp8_e4m3_to_f32(b)).collect();

    // Assert precision floor tolerance at 1e-2
    for (i, (&orig, &deq)) in input_floats.iter().zip(dequant_floats.iter()).enumerate() {
        let diff = (orig - deq).abs();
        assert!(
            diff <= 1e-2,
            "Raven FP8 repack precision mismatch at index {i}: original {orig}, dequantized {deq}, diff {diff}"
        );
    }
}
