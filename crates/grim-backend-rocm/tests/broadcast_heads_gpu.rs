//! Phase 1 G4b gate: per-channel gate logits must replicate exactly into the
//! per-head buffers consumed by the fused GDL kernel.

use grim_backend_rocm::{RocmDevice, as_rocm, gpu_test_enabled};
use grim_tensor::{CoreTensorOps, DType, Shape};

#[test]
#[ignore = "GPU-only G4b Phase 1 parity; run with GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test broadcast_heads_gpu -- --ignored"]
fn broadcast_heads_matches_cpu() {
    assert!(
        gpu_test_enabled(),
        "broadcast_heads parity requires GRIM_RUN_GPU_TESTS=1"
    );
    let _guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::try_new(0).expect("broadcast_heads parity requires a ROCm GPU");
    const DK: usize = 64;
    const NH: usize = 16;
    let input: Vec<f32> = (0..DK)
        .map(|i| ((i as f32 * 0.37).sin() - 0.25) * 3.0)
        .collect();
    let in_storage = dev
        .from_cpu(&input, &Shape::new(vec![DK]), DType::F32)
        .expect("upload input");
    let out_storage = dev
        .from_cpu(&vec![0.0; DK * NH], &Shape::new(vec![NH, DK]), DType::F32)
        .expect("allocate output");

    dev.broadcast_heads(
        in_storage.as_ref(),
        as_rocm(out_storage.as_ref()).expect("ROCm output"),
        DK,
        NH,
    )
    .expect("broadcast heads");
    dev.synchronize();

    let got = out_storage.to_cpu_vec_f32().expect("download output");
    let expected: Vec<f32> = (0..NH).flat_map(|_| input.iter().copied()).collect();
    assert_eq!(got, expected, "broadcast must be an exact replication");
}
