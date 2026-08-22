//! WI-X3 done-check: distributional equivalence of the GPU stochastic sampler
//! against the analytic multinomial over the same logits. Env-gated per house
//! rule — run on real hardware with `GRIM_RUN_GPU_TESTS=1`.

use grim_backend_rocm::{BackendDevice, RocmDevice, Shape, as_rocm, sample_logits_on_device_at};
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
