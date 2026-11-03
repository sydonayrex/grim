//! Task 10.6 gate: the engine's ROCm speculative verifier must agree with the
//! CPU verify loop for identical logits + uniforms — the wiring that makes
//! `grim_speculative_rejection_sample` a production entry point (per the plan's
//! Rust-entry-point rule).

use grim_engine::RocmSpeculativeVerifier;
use grim_speculative::SpeculativeVerifier;
use grim_tensor::{CoreTensorOps, DType, Shape, Tensor};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_enabled() -> bool {
    std::env::var("GRIM_GPU_TEST").map(|v| v == "1").unwrap_or(false)
}

/// CPU reference: the exact math the wrapper's fallback loop uses.
fn cpu_verify(
    target_logits: &[f32],
    draft_logits: &[f32],
    context_len: usize,
    vocab_size: usize,
    draft_tokens: &[u32],
    uniforms: &[f32],
) -> usize {
    let verify_len = draft_tokens.len();
    let mut accepted = 0usize;
    for i in 0..verify_len {
        let draft_tok = draft_tokens[i] as usize;
        let t_row = &target_logits[(context_len + i) * vocab_size..(context_len + i + 1) * vocab_size];
        let d_row = &draft_logits[i * vocab_size..(i + 1) * vocab_size];
        let softmax_at = |row: &[f32]| {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|l| (l - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            exp[draft_tok] / sum
        };
        let p_target = softmax_at(t_row);
        let p_draft = softmax_at(d_row);
        let p_accept = if p_draft > 1e-10 {
            (p_target / p_draft).min(1.0)
        } else {
            0.0
        };
        if uniforms[i] < p_accept {
            accepted += 1;
        } else {
            break;
        }
    }
    accepted
}

#[test]
#[ignore]
fn gpu_speculative_verify_matches_cpu() -> TestResult {
    if !gpu_enabled() {
        return Ok(());
    }
    let dev = grim_backend_rocm::device::roc_device::RocmDevice::try_new(0)?;

    let vocab = 1024usize;
    let k = 4usize;
    let context_len = 1usize;

    // Deterministic logits: target rows ctx..ctx+K, draft rows 0..K.
    let mut target = Vec::with_capacity((context_len + k) * vocab);
    let mut state = 0x243F6A8885A308D3u64;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    for _ in 0..(context_len + k) * vocab {
        target.push(rand() * 10.0 - 5.0);
    }
    let mut draft = Vec::with_capacity(k * vocab);
    for _ in 0..k * vocab {
        draft.push(rand() * 10.0 - 5.0);
    }
    let draft_tokens: Vec<u32> = (0..k).map(|i| ((i * 271 + 13) % vocab) as u32).collect();
    let uniforms: Vec<f32> = (0..=k).map(|_| rand()).collect();

    let expected = cpu_verify(&target, &draft, context_len, vocab, &draft_tokens, &uniforms);

    // Discriminating cases beyond the random draw: (a) target strongly favors
    // every draft token -> all accepted; (b) rejects at the last position;
    // (c) rejects at the first position.
    let mut case_a = target.clone();
    for i in 0..k {
        let row = &mut case_a[(context_len + i) * vocab..];
        row[draft_tokens[i] as usize] = 100.0;
    }
    let mut case_b = target.clone();
    for i in 0..k {
        let row = &mut case_b[(context_len + i) * vocab..];
        row[draft_tokens[i] as usize] = if i + 1 == k { -100.0 } else { 100.0 };
    }
    let mut case_c = target.clone();
    {
        let row = &mut case_c[(context_len) * vocab..];
        row[draft_tokens[0] as usize] = -100.0;
    }
    let expected_cases = [
        cpu_verify(&case_a, &draft, context_len, vocab, &draft_tokens, &uniforms),
        cpu_verify(&case_b, &draft, context_len, vocab, &draft_tokens, &uniforms),
        cpu_verify(&case_c, &draft, context_len, vocab, &draft_tokens, &uniforms),
    ];
    assert!(
        expected_cases.iter().any(|&e| e > 0),
        "test construction degenerate: no forced-accept case — GPU/CPU agreement would be vacuous"
    );
    assert!(
        expected_cases[1] < k && expected_cases[2] == 0,
        "forced-reject cases did not reject on CPU oracle: {expected_cases:?}"
    );

    // Upload target rows 0..ctx+K (the verifier narrows rows [ctx, ctx+K)).
    let target_dev = CoreTensorOps::from_cpu(
        &dev,
        &target,
        &Shape::new(vec![context_len + k, vocab]),
        DType::F32,
    )?;
    let draft_dev = CoreTensorOps::from_cpu(
        &dev,
        &draft,
        &Shape::new(vec![k, vocab]),
        DType::F32,
    )?;
    let target_t = Tensor::new(
        std::sync::Arc::from(target_dev),
        Shape::new(vec![context_len + k, vocab]),
        DType::F32,
        grim_tensor::dtype::QuantProvenance::default(),
        grim_tensor::Device::Rocm(0),
    );
    let draft_t = Tensor::new(
        std::sync::Arc::from(draft_dev),
        Shape::new(vec![k, vocab]),
        DType::F32,
        grim_tensor::dtype::QuantProvenance::default(),
        grim_tensor::Device::Rocm(0),
    );

    let verifier = RocmSpeculativeVerifier;
    let accepted = verifier.verify(
        &target_t,
        &draft_t,
        context_len,
        vocab,
        &draft_tokens,
        &uniforms,
    )?;
    assert_eq!(
        accepted, expected,
        "GPU verify accepted-count diverges from CPU loop (uniforms identical)"
    );

    // Run each forced case through the GPU verifier.
    for (name, logits, expected_c) in [
        ("all-accept", case_a, expected_cases[0]),
        ("reject-last", case_b, expected_cases[1]),
        ("reject-first", case_c, expected_cases[2]),
    ] {
        let case_dev = CoreTensorOps::from_cpu(
            &dev,
            &logits,
            &Shape::new(vec![context_len + k, vocab]),
            DType::F32,
        )?;
        let case_t = Tensor::new(
            std::sync::Arc::from(case_dev),
            Shape::new(vec![context_len + k, vocab]),
            DType::F32,
            grim_tensor::dtype::QuantProvenance::default(),
            grim_tensor::Device::Rocm(0),
        );
        let got = verifier.verify(&case_t, &draft_t, context_len, vocab, &draft_tokens, &uniforms)?;
        assert_eq!(
            got, expected_c,
            "GPU verify diverges from CPU on case {name}: got={got}, cpu={expected_c}"
        );
        println!("gpu speculative verify parity [{name}]: accepted={got} (CPU={expected_c})");
    }
    println!("gpu speculative verify parity [random]: accepted={accepted} (CPU={expected})");
    Ok(())
}
