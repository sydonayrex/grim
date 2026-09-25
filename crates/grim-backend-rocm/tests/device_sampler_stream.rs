//! G1 (PLAN-kernel-fusion) regression: the explicit-stream sampler entry
//! must produce correct results when bound to a NON-default stream. Before
//! G1 the post-replay sampler relied on `graph.stream == default_stream`
//! (both pool[0]) by coincidence; this test launches on pool slot 1 via
//! `sample_logits_on_device_with_penalty_at_stream` so a future pool split
//! cannot pass by accident.
//!
//! Gated: GRIM_GPU_TEST=1 + ROCm device.

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::device::util::gpu_test_enabled;
use grim_backend_rocm::{as_rocm, sample_logits_on_device_with_penalty_at_stream};
use grim_tensor::{CoreTensorOps, DType, Shape};

#[test]
#[ignore]
fn stream_bound_sampler_works_off_default_stream() {
    if !gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return;
    }
    let dev = RocmDevice::shared(0);
    let stream = dev
        .get_stream_from_pool(1)
        .expect("stream pool slot 1 must exist");
    assert!(
        !stream.is_null(),
        "pool slot 1 must be a distinct, non-null stream"
    );

    // Deterministic logits where the answer differs WITH vs WITHOUT penalty:
    // vocab 8; history {2} (twice — dedup contract); base[2] highest so the
    // penalty pushes the greedy argmax to a different token.
    let vocab = 8usize;
    let mut logits = vec![0.0f32; vocab];
    logits[2] = 10.0; // would win unpenalized
    logits[5] = 7.0; // wins once 2 is penalized (10/1.1 < 7? no) — make harder:
    logits[5] = 9.95; // 9.95 barely under 10 → penalty 1.1 flips winner to 5
    let shape = Shape::new(vec![1, vocab]);
    let st = dev.from_cpu(&logits, &shape, DType::F32).expect("upload");
    let r = as_rocm(st.as_ref()).expect("rocm");

    let tok = sample_logits_on_device_with_penalty_at_stream(
        &dev,
        r,
        vocab,
        0.0, // greedy
        0,
        1.0,
        0x1234,
        0,
        1.1,
        &[2, 2, 5usize as u32][..2], // history: token 2 only
        stream,
    )
    .expect("stream sampler")
    .expect("Some");
    // Without penalty: argmax = 2; with penalty 10/1.1 ≈ 9.09 < 9.95 → 5.
    assert_eq!(tok, 5, "penalty must flip winner on explicit stream");
    eprintln!("[g1-test] explicit-stream (pool[1]) sample: token={tok} correct");
}
