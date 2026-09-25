//! Task 5 (Plan 3): Integration test for stochastic sampling from CLI / engine.
//! Gated by gpu_test_enabled() + #[ignore].

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::kernels::device_sampler::sample_logits_on_device_at;
use grim_tensor::CoreTensorOps;
use grim_tensor::Shape;
use grim_tensor::dtype::DType;

#[test]
#[ignore]
fn test_stochastic_sampler_integration() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("ROCm device tests disabled: skipping test_stochastic_sampler_integration");
        return;
    }
    let dev = RocmDevice::new(0);

    let vocab = 1000usize;
    let mut logits_cpu = vec![0.0f32; vocab];
    for (i, v) in logits_cpu.iter_mut().enumerate() {
        *v = ((i as f32) * 0.05).sin();
    }
    logits_cpu[42] = 15.0; // dominant token

    let shape = Shape::new(vec![vocab]);
    let logits_box = dev.from_cpu(&logits_cpu, &shape, DType::F32).unwrap();
    let logits_gpu = logits_box
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();

    let tok = sample_logits_on_device_at(
        &dev,
        logits_gpu,
        vocab,
        0.7f32,
        50,
        0.9f32,
        12345u64,
        0,
    )
    .unwrap()
    .expect("Sampling should produce a token");

    assert_eq!(tok, 42, "Dominant logit token 42 must be selected under low-entropy sampling");
}
