//! Task 4 (Plan 3): Regression test for sampler stream synchronization and edge cases.
//! Validates clean execution and correct outputs when top_k = 0, top_k >= vocab, and top_p = 1.0.

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::kernels::device_sampler::sample_logits_on_device_at;
use grim_tensor::CoreTensorOps;
use grim_tensor::Shape;
use grim_tensor::dtype::DType;

#[test]
#[ignore]
fn test_sampler_edge_cases_regression() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_sampler_edge_cases_regression");
        return;
    }
    let dev = RocmDevice::new(0);

    let vocab = 32usize;
    let mut logits_cpu = vec![0.0f32; vocab];
    for (i, v) in logits_cpu.iter_mut().enumerate() {
        *v = (i as f32) * 0.1;
    }

    let shape = Shape::new(vec![vocab]);
    let logits_box = dev.from_cpu(&logits_cpu, &shape, DType::F32).unwrap();
    let logits_gpu = logits_box
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();

    let temperature = 0.8f32;
    let seed = 12345u64;

    // Edge Case 1: top_k = 0 (unmasked top-k path)
    let tok_k0 = sample_logits_on_device_at(
        &dev,
        logits_gpu,
        vocab,
        temperature,
        0,
        0.9f32,
        seed,
        1,
    )
    .unwrap();
    assert!(tok_k0.is_some());
    assert!((tok_k0.unwrap() as usize) < vocab);

    // Edge Case 2: top_k >= vocab (all tokens kept in top-k)
    let tok_k_huge = sample_logits_on_device_at(
        &dev,
        logits_gpu,
        vocab,
        temperature,
        (vocab + 100) as i32,
        0.9f32,
        seed,
        2,
    )
    .unwrap();
    assert!(tok_k_huge.is_some());
    assert!((tok_k_huge.unwrap() as usize) < vocab);

    // Edge Case 3: top_p = 1.0 (unmasked top-p path)
    let tok_p1 = sample_logits_on_device_at(
        &dev,
        logits_gpu,
        vocab,
        temperature,
        10,
        1.0f32,
        seed,
        3,
    )
    .unwrap();
    assert!(tok_p1.is_some());
    assert!((tok_p1.unwrap() as usize) < vocab);
}
