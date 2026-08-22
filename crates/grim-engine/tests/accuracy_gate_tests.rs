//! Integration tests for the 18-format Accuracy & Perplexity Regression Suite.

use grim_quant::accuracy_gate::{AccuracyGate, AccuracyVerdict, compute_cross_entropy_ppl};
use grim_tensor::dtype::QuantFormat;

#[test]
fn test_accuracy_gate_all_standard_formats() {
    let gate = AccuracyGate::new();

    // Synthetic golden oracle activation vector
    let oracle: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();

    // 1. FP8 candidate (near-exact)
    let candidate_fp8: Vec<f32> = oracle.iter().map(|&x| x + 0.0005).collect();
    let verdict = gate
        .verify(QuantFormat::Fp8, &oracle, &candidate_fp8)
        .unwrap();
    assert!(
        matches!(verdict, AccuracyVerdict::Pass { .. }),
        "FP8 must pass: {:?}",
        verdict
    );

    // 2. Q8_0 candidate (high-precision)
    let candidate_q8: Vec<f32> = oracle.iter().map(|&x| x + 0.002).collect();
    let verdict = gate
        .verify(QuantFormat::Q8_0, &oracle, &candidate_q8)
        .unwrap();
    assert!(
        matches!(verdict, AccuracyVerdict::Pass { .. }),
        "Q8_0 must pass: {:?}",
        verdict
    );

    // 3. FP4 candidate (micro-scaled 4-bit)
    let candidate_mxfp4: Vec<f32> = oracle.iter().map(|&x| x + 0.015).collect();
    let verdict = gate
        .verify(QuantFormat::Fp4, &oracle, &candidate_mxfp4)
        .unwrap();
    assert!(
        matches!(verdict, AccuracyVerdict::Pass { .. }),
        "FP4 must pass: {:?}",
        verdict
    );

    // 4. Q4_K candidate (standard GGUF 4-bit)
    let candidate_q4k: Vec<f32> = oracle.iter().map(|&x| x + 0.02).collect();
    let verdict = gate
        .verify(QuantFormat::Q4K, &oracle, &candidate_q4k)
        .unwrap();
    assert!(
        matches!(verdict, AccuracyVerdict::Pass { .. }),
        "Q4K must pass: {:?}",
        verdict
    );

    // 5. IQ2_XXS candidate (extreme vector quantized)
    let candidate_iq2: Vec<f32> = oracle.iter().map(|&x| x + 0.08).collect();
    let verdict = gate
        .verify(QuantFormat::Iq2Xxs, &oracle, &candidate_iq2)
        .unwrap();
    assert!(
        matches!(verdict, AccuracyVerdict::Pass { .. }),
        "IQ2_XXS must pass: {:?}",
        verdict
    );
}

#[test]
fn test_cross_entropy_ppl_monotonicity() {
    let vocab_size = 100;
    let seq_len = 16;
    let targets: Vec<u32> = (0..seq_len as u32)
        .map(|i| i % (vocab_size as u32))
        .collect();

    // Confident correct logits -> low PPL
    let mut good_logits = vec![0.0f32; seq_len * vocab_size];
    for t in 0..seq_len {
        good_logits[t * vocab_size + targets[t] as usize] = 10.0;
    }
    let good_ppl = compute_cross_entropy_ppl(&good_logits, &targets, vocab_size);

    // Flat uniform logits -> high PPL (equal to vocab_size)
    let uniform_logits = vec![0.0f32; seq_len * vocab_size];
    let uniform_ppl = compute_cross_entropy_ppl(&uniform_logits, &targets, vocab_size);

    assert!(
        good_ppl < uniform_ppl,
        "Good logits must achieve lower PPL: {} < {}",
        good_ppl,
        uniform_ppl
    );
    assert!(
        (uniform_ppl - vocab_size as f64).abs() < 1e-3,
        "Uniform logits PPL must equal vocab size"
    );
}
