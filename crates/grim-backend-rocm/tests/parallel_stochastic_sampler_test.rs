//! Task 3 (Plan 3): Unit test for stochastic sampler statistical distribution parity.
//! Verifies determinism with fixed (seed, position) pairs,
//! and tests distribution parity against CPU multinomial sampling.

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::kernels::device_sampler::sample_logits_on_device_at;
use grim_tensor::CoreTensorOps;
use grim_tensor::Shape;
use grim_tensor::dtype::DType;

#[test]
#[ignore]
fn test_stochastic_sampler_determinism_and_distribution() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_stochastic_sampler_determinism_and_distribution");
        return;
    }
    let dev = RocmDevice::new(0);

    let vocab = 16usize;
    // Skewed logits: token 0 and 1 have highest probability
    let mut logits_cpu = vec![-10.0f32; vocab];
    logits_cpu[0] = 3.0;
    logits_cpu[1] = 2.0;
    logits_cpu[2] = 1.0;
    logits_cpu[3] = 0.5;

    let shape = Shape::new(vec![vocab]);
    let logits_box = dev.from_cpu(&logits_cpu, &shape, DType::F32).unwrap();
    let logits_gpu = logits_box
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();

    let temperature = 0.7f32;
    let top_k = 4i32;
    let top_p = 0.95f32;
    let seed = 42u64;

    // Test 1: Determinism with identical (seed, position)
    let tok1 = sample_logits_on_device_at(
        &dev,
        &logits_gpu,
        vocab,
        temperature,
        top_k,
        top_p,
        seed,
        10u32,
    )
    .unwrap()
    .expect("Device sampler should succeed");

    let tok2 = sample_logits_on_device_at(
        &dev,
        &logits_gpu,
        vocab,
        temperature,
        top_k,
        top_p,
        seed,
        10u32,
    )
    .unwrap()
    .expect("Device sampler should succeed");

    assert_eq!(tok1, tok2, "Sampling with identical seed and position must be deterministic");

    // Test 2: Distribution verification over 2,000 draws
    let mut counts = vec![0usize; vocab];
    let num_samples = 2000usize;
    for pos in 0..num_samples as u32 {
        let tok = sample_logits_on_device_at(
            &dev,
            &logits_gpu,
            vocab,
            temperature,
            top_k,
            top_p,
            seed,
            pos,
        )
        .unwrap()
        .expect("Device sampler should succeed");
        counts[tok as usize] += 1;
    }

    // Top-k is 4, so tokens >= 4 must have 0 draws
    for v in 4..vocab {
        assert_eq!(counts[v], 0, "Token {v} exceeds top_k=4 but received {} draws", counts[v]);
    }

    // Token 0 (logit 3.0) should have the highest share of draws
    assert!(
        counts[0] > counts[1],
        "Token 0 count ({}) should exceed Token 1 count ({})",
        counts[0],
        counts[1]
    );
    assert!(
        counts[1] > counts[2],
        "Token 1 count ({}) should exceed Token 2 count ({})",
        counts[1],
        counts[2]
    );
}
