//! GQA head-repeat parity: K/V expanded from [nkv, hd] to [nh, hd] must
//! match the repeat-interleave mapping kv_h = h / kv_group (adjacent query
//! heads share a KV head, per the host loop's `kv_h = h / kv_group`).

use grim_backend_rocm::{RocmDevice, as_rocm, gpu_test_enabled};
use grim_tensor::{CoreTensorOps, DType, Shape};

#[test]
#[ignore = "GPU-only GQA head-repeat parity; run with GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test head_repeat_gpu -- --ignored"]
fn head_repeat_matches_cpu() {
    assert!(
        gpu_test_enabled(),
        "head_repeat parity requires GRIM_RUN_GPU_TESTS=1"
    );
    let _guard = grim_backend_rocm::device::util::gpu_test_lock();
    let dev = RocmDevice::try_new(0).expect("head_repeat parity requires a ROCm GPU");

    const NKV: usize = 8;
    const NH: usize = 16;
    const HD: usize = 64;
    const KV_GROUP: usize = NH / NKV;

    // Random K and V data (deterministic pattern, no RNG dependency).
    let k_data: Vec<f32> = (0..NKV * HD)
        .map(|i| ((i as f32 * 0.37).sin() - 0.25) * 3.0)
        .collect();
    let v_data: Vec<f32> = (0..NKV * HD)
        .map(|i| ((i as f32 * 0.53).cos() + 0.1) * 2.0)
        .collect();

    let k_storage = dev
        .from_cpu(&k_data, &Shape::new(vec![NKV, HD]), DType::F32)
        .expect("upload K");
    let v_storage = dev
        .from_cpu(&v_data, &Shape::new(vec![NKV, HD]), DType::F32)
        .expect("upload V");
    let k_out_storage = dev
        .from_cpu(&vec![0.0; NH * HD], &Shape::new(vec![NH, HD]), DType::F32)
        .expect("allocate K output");
    let v_out_storage = dev
        .from_cpu(&vec![0.0; NH * HD], &Shape::new(vec![NH, HD]), DType::F32)
        .expect("allocate V output");

    dev.head_repeat(
        k_storage.as_ref(),
        v_storage.as_ref(),
        as_rocm(k_out_storage.as_ref()).expect("ROCm K output"),
        as_rocm(v_out_storage.as_ref()).expect("ROCm V output"),
        NKV,
        NH,
        KV_GROUP,
        HD,
    )
    .expect("head repeat");
    dev.synchronize();

    let k_got = k_out_storage.to_cpu_vec_f32().expect("download K");
    let v_got = v_out_storage.to_cpu_vec_f32().expect("download V");

    // Verify: K_expanded[h * HD + d] = K[(h / KV_GROUP) * HD + d]
    for h in 0..NH {
        let kv_h = h / KV_GROUP;
        for d in 0..HD {
            let expected_k = k_data[kv_h * HD + d];
            let got_k = k_got[h * HD + d];
            assert!(
                (got_k - expected_k).abs() < 1e-6,
                "K mismatch at h={h}, d={d}: got {got_k}, expected {expected_k}"
            );

            let expected_v = v_data[kv_h * HD + d];
            let got_v = v_got[h * HD + d];
            assert!(
                (got_v - expected_v).abs() < 1e-6,
                "V mismatch at h={h}, d={d}: got {got_v}, expected {expected_v}"
            );
        }
    }
}
