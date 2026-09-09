//! WI-X3 done-check: distributional equivalence of the GPU stochastic sampler
//! against the analytic multinomial over the same logits. Env-gated per house
//! rule — run on real hardware with `GRIM_RUN_GPU_TESTS=1`.

use grim_backend_rocm::{
    CoreTensorOps, PinnedLogitsBuf, RocmDevice, Shape, as_rocm, sample_logits_on_device_at,
};
use grim_tensor::DType;

#[test]
fn gpu_stochastic_sampler_matches_multinomial_distribution() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ROCm device 0 not available: {e}");
            return;
        }
    };

    // Bimodal distribution over 64 bins so a broken sampler (argmax-only,
    // uniform, off-by-one window) cannot pass by accident.
    let vocab = 64usize;
    let logits: Vec<f32> = (0..vocab)
        .map(|i| {
            let x = i as f32;
            ((x - 12.0).powi(2) / -18.0).exp() + 0.7 * ((x - 44.0).powi(2) / -60.0).exp()
        })
        .collect();

    let shape = Shape::new(vec![vocab]);
    let storage = dev
        .from_cpu(&logits, &shape, DType::F32)
        .expect("upload logits");
    let rocm_st = as_rocm(storage.as_ref()).expect("rocm storage");

    // Analytic reference probabilities (temperature 1, no filters).
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let probs: Vec<f64> = logits
        .iter()
        .map(|&l| ((l - max).exp() / denom) as f64)
        .collect();

    let draws = 100_000u32;
    let mut counts = vec![0u64; vocab];
    for pos in 0..draws {
        let tok = sample_logits_on_device_at(
            &dev,
            rocm_st,
            vocab,
            1.0,
            0,   // top_k off
            1.0, // top_p off
            0x5EED_1234,
            pos,
        )
        .expect("device sample")
        .expect("in-bounds vocab must sample");
        assert!((tok as usize) < vocab, "sampled token {tok} out of range");
        counts[tok as usize] += 1;
    }

    // Pearson chi-square against the analytic multinomial. Bins with expected
    // count < 5 are excluded (standard validity rule). df ≈ bins - 1 ≈ 40;
    // the 99.99% critical value is ~73, so 80 gives headroom without being
    // loose enough to admit a unimodal-only or argmax-only sampler.
    let mut chi2 = 0.0f64;
    let mut bins = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        let e = probs[i] * draws as f64;
        if e < 5.0 {
            continue;
        }
        chi2 += (c as f64 - e).powi(2) / e;
        bins += 1;
    }
    assert!(
        bins > 20,
        "too many low-probability bins excluded to judge fit (bins={bins})"
    );
    assert!(
        chi2 < 80.0,
        "GPU sampler distribution diverges from multinomial: chi2={chi2} over {bins} bins"
    );

    // Sanity: the argmax bin must be the most-drawn one.
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let top_drawn = counts
        .iter()
        .enumerate()
        .max_by_key(|entry| *entry.1)
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(argmax, top_drawn, "modal bin must dominate the draws");
}

// ── Parity gate: GPU greedy (T=0) must exactly match CPU argmax ───────────
// Fixed logits, fixed seed. If this fails the kernel is argmax-broken.
#[test]
fn gpu_greedy_sampler_exact_parity_with_cpu_argmax() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("ROCm device 0 not available: {e}"); return; }
    };

    let vocab = 32768usize;
    // Sharp spike at index 12345: any working greedy sampler must return that.
    let mut logits = vec![0.0f32; vocab];
    logits[12345] = 100.0;

    let shape = Shape::new(vec![vocab]);
    let storage = dev.from_cpu(&logits, &shape, DType::F32).expect("upload");
    let rocm_st = as_rocm(storage.as_ref()).expect("rocm storage");

    // CPU argmax.
    let cpu_argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;

    // GPU greedy (temperature = 0).
    let gpu_tok = sample_logits_on_device_at(&dev, rocm_st, vocab, 0.0, 0, 1.0, 0xDEAD_BEEF, 0)
        .expect("device sample")
        .expect("must sample");

    assert_eq!(
        gpu_tok, cpu_argmax,
        "GPU greedy token {gpu_tok} != CPU argmax {cpu_argmax}"
    );
}

// ── Throughput gate: device-sample >= 2x faster than CPU vec-alloc path ──
// Measures per-step wall time over STEPS decode steps at vocab=32768.
// The device path keeps logits on GPU and D2H's only 4 bytes.
// The CPU path calls to_cpu_vec_f32() (Vec<f32> alloc + full D2H) per step.
#[test]
fn gpu_sampler_decode_throughput_gate() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("ROCm device 0 not available: {e}"); return; }
    };

    const VOCAB: usize = 32768;
    const STEPS: usize = 256;
    const SEED: u64 = 0x5EED_CAFE_1234_5678;

    // Fixed logits uploaded once; reused every step (mirrors decode: same weights, different step).
    let logits: Vec<f32> = (0..VOCAB)
        .map(|i| ((i as f32 - VOCAB as f32 / 2.0) / 256.0).tanh())
        .collect();
    let shape = Shape::new(vec![VOCAB]);
    let storage = dev.from_cpu(&logits, &shape, DType::F32).expect("upload logits");
    let rocm_st = as_rocm(storage.as_ref()).expect("rocm storage");

    // ── device-sample path ───────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let mut device_tokens = Vec::with_capacity(STEPS);
    for step in 0..STEPS as u32 {
        let tok = sample_logits_on_device_at(&dev, rocm_st, VOCAB, 0.8, 40, 0.9, SEED, step)
            .expect("device sample")
            .expect("must sample");
        device_tokens.push(tok);
    }
    let device_us = t0.elapsed().as_micros() as f64;

    // ── CPU vec-alloc path (mimics to_cpu_vec_f32 + CPU softmax + sort) ─
    let t1 = std::time::Instant::now();
    let mut cpu_tokens = Vec::with_capacity(STEPS);
    for step in 0..STEPS as u32 {
        // Pull full logits to host (this is the expensive part).
        let host = storage.to_cpu_vec_f32().expect("D2H");
        // CPU greedy (argmax) so the timing comparison is fair — the bottleneck
        // is the D2H transfer, not the sampling arithmetic.
        let tok = host
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        // XOR with step to prevent the loop from being optimised away.
        cpu_tokens.push(tok ^ step);
    }
    let cpu_us = t1.elapsed().as_micros() as f64;

    let device_us_per_step = device_us / STEPS as f64;
    let cpu_us_per_step = cpu_us / STEPS as f64;
    eprintln!(
        "[throughput_gate] device {device_us_per_step:.1} µs/step  \
         cpu_d2h {cpu_us_per_step:.1} µs/step  \
         speedup {:.2}×",
        cpu_us_per_step / device_us_per_step
    );

    assert!(
        device_us_per_step * 2.0 < cpu_us_per_step,
        "device sampler ({device_us_per_step:.1} µs/step) must be ≥2× faster \
         than CPU D2H fallback ({cpu_us_per_step:.1} µs/step)"
    );

    // Sanity: all device tokens in range.
    for (i, &tok) in device_tokens.iter().enumerate() {
        assert!(
            (tok as usize) < VOCAB,
            "step {i}: device token {tok} out of range [0, {VOCAB})"
        );
    }
}

// ── PinnedLogitsBuf round-trip: D2H via pinned buf matches to_cpu_vec_f32 ─
#[test]
fn pinned_logits_buf_round_trip_matches_standard_d2h() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("ROCm device 0 not available: {e}"); return; }
    };

    const VOCAB: usize = 4096;
    let logits: Vec<f32> = (0..VOCAB).map(|i| i as f32 * 0.001 - 2.0).collect();
    let shape = Shape::new(vec![VOCAB]);
    let storage = dev.from_cpu(&logits, &shape, DType::F32).expect("upload");
    let rocm_st = as_rocm(storage.as_ref()).expect("rocm storage");

    // Reference: standard blocking D2H.
    let reference = storage.to_cpu_vec_f32().expect("reference D2H");

    // Under test: PinnedLogitsBuf.
    let mut pinned = PinnedLogitsBuf::alloc(VOCAB).expect("alloc pinned buf");

    // Read twice (ping-pong) to exercise both slots.
    for _round in 0..2 {
        let got = pinned
            .read_logits_to_pinned(&dev, rocm_st, VOCAB)
            .expect("pinned D2H");

        assert_eq!(got.len(), VOCAB, "slice length mismatch");
        for (i, (&r, &g)) in reference.iter().zip(got.iter()).enumerate() {
            assert!(
                (r - g).abs() < 1e-6,
                "mismatch at index {i}: reference {r} vs pinned {g}"
            );
        }
    }
}

