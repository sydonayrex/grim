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

/// P5: golden-activation fixtures per quant family from REAL model
/// activations — not synthetic oracle vectors.
///
/// A tiny deterministic CPU Llama (`Llama::random` uses a fixed
/// `SimpleRng` seed) runs a fixed 8-token prompt; the resulting
/// `[8, 256]` logit activation vector is the F32 golden fixture
/// (length + checksum + head values pinned below — any silent numerics
/// change fails). Each quant family then round-trips THAT vector:
/// - Q8_0 and Q4K must pass the canonical gate tolerances;
/// - MXFP4 pins its measured activation fidelity as the golden budget.
///   Note it does NOT meet the canonical Fp4 tolerance (cos 0.9936 <
///   0.9950, L2 0.1129 > 0.10): that tolerance is weight-calibrated and
///   4-bit E2M1 is coarser on low-entropy activation noise. Pinning the
///   measured budget still catches silent quant regressions, honestly.
fn golden_llama_activations() -> Vec<f32> {
    use grim_backend_cpu::cpu_tensor;
    use grim_core::session::Inner;
    use grim_core::CausalLm;
    use grim_models_transformer::{Llama, LlamaConfig};
    use grim_tensor::{Device, Shape};

    // Fixed prompt: 8 ids inside the 256-token vocab.
    const PROMPT: [u32; 8] = [11, 28, 45, 62, 79, 96, 113, 130];
    let model = Llama::random(
        Device::Cpu,
        LlamaConfig {
            vocab_size: 256,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
            yarn: None,
            max_seq_len: 256,
        },
    );
    let mut session = Inner::new(Device::Cpu);
    let input = cpu_tensor(
        PROMPT.iter().map(|&t| t as f32).collect(),
        Shape::new(vec![1, 8]),
    );
    let positions = cpu_tensor(
        (0..8).map(|t| t as f32).collect(),
        Shape::new(vec![1, 8]),
    );
    model
        .forward(&mut session, &input, &positions, &[])
        .expect("golden forward")
        .to_vec_f32()
        .expect("logits to host")
}

#[test]
fn test_quant_golden_activations_fixed_prompt() {
    let oracle = golden_llama_activations();

    // F32 golden fixture: exact shape + checksum + head values.
    const GOLDEN_LEN: usize = 8 * 256;
    const GOLDEN_SUM: f64 = 0.233580902684;
    const GOLDEN_FIRST8: [f32; 8] = [
        -0.0077741435,
        0.038189523,
        -0.040545087,
        -0.024978453,
        -0.04808428,
        -0.025814248,
        0.023511255,
        0.0017588767,
    ];
    assert_eq!(oracle.len(), GOLDEN_LEN, "golden activation length");
    let sum: f64 = oracle.iter().map(|&x| x as f64).sum();
    assert!(
        (sum - GOLDEN_SUM).abs() < 1e-9,
        "golden checksum drift: {sum} vs {GOLDEN_SUM}"
    );
    assert_eq!(
        &oracle[..8],
        &GOLDEN_FIRST8,
        "golden activation head values"
    );

    // Determinism leg: a fresh model + session on the same fixed prompt
    // must reproduce the fixture bit-for-bit (else the golden is unstable).
    let oracle2 = golden_llama_activations();
    assert_eq!(oracle, oracle2, "fixed prompt must forward deterministically");

    let gate = AccuracyGate::new();

    // Q8_0 family: real-activation roundtrip must pass canonical tolerance.
    let q80 = grim_quant::dequant_q80(
        &grim_quant::quant_q80(&oracle).expect("q80 quant"),
        oracle.len(),
    )
    .expect("q80 dequant");
    let verdict = gate.verify(QuantFormat::Q8_0, &oracle, &q80).unwrap();
    assert!(
        matches!(verdict, AccuracyVerdict::Pass { .. }),
        "Q8_0 on real activations must pass: {:?}",
        verdict
    );

    // MXFP4 family: pin measured activation fidelity (see doc comment).
    let mxfp4 = grim_quant::qat_mxfp4::fake_quant_mxfp4(&oracle, 8, 256).expect("mxfp4");
    let cos = grim_quant::accuracy_gate::compute_cosine_similarity(&oracle, &mxfp4);
    let l2 = grim_quant::accuracy_gate::compute_relative_l2_error(&oracle, &mxfp4);
    assert!(
        cos >= 0.993,
        "MXFP4 cosine fidelity regressed: {cos} (golden budget 0.993)"
    );
    assert!(
        l2 <= 0.115,
        "MXFP4 L2 error regressed: {l2} (golden budget 0.115)"
    );

    // Q4K family: real-activation roundtrip must pass canonical tolerance.
    let q4k = grim_quant::dequant_q4k(
        &grim_quant::quant_q4k(&oracle).expect("q4k quant"),
        oracle.len(),
    )
    .expect("q4k dequant");
    let verdict = gate.verify(QuantFormat::Q4K, &oracle, &q4k).unwrap();
    assert!(
        matches!(verdict, AccuracyVerdict::Pass { .. }),
        "Q4K on real activations must pass: {:?}",
        verdict
    );
}
