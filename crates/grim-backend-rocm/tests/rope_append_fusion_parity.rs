//! PLAN 4 Task 4: `qk_rope_append_kv` (QK-rope + K-append + V-append in one
//! launch) must be bit-identical to the separate launches: same rotated K
//! rows, same arena bytes across successive mid-arena positions.
//!
//! Gated: GRIM_GPU_TEST=1 (canonical `gpu_test_enabled`) + ROCm device.

use grim_backend_rocm::{RocmDevice, as_rocm, gpu_test_enabled};
use grim_tensor::{CoreTensorOps, DType, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn f32_tensor(
    dev: &RocmDevice,
    data: &[f32],
    shape: &Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev, data, shape, DType::F32).unwrap()
}

fn u32_tensor(dev: &RocmDevice, v: u32) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev, &[f32::from_bits(v)], &Shape::new(vec![1]), DType::U32)
        .unwrap()
}

fn rand_f32(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((seed + i * 7) % 113) as f32 / 113.0 - 0.5)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore]
fn rope_append_kv_matches_separate_rope_plus_appends() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let _guard = grim_backend_rocm::device::util::gpu_test_lock();

    // Shape-faithful: 2 KV heads, head_dim 64 (half == warpSize, the fused
    // path's documented constraint), batch 1.
    let (nkv, hd) = (2usize, 64usize);
    let (steps, batch) = (1usize, 1usize);
    let kv_stride = nkv * hd;
    let max_ctx = 8usize;
    let slot_stride = max_ctx * kv_stride;
    let eps = 1e-5f32;
    let mut rope_cfg = grim_tensor::RopeConfig::new(hd, 10000.0f32);
    rope_cfg.interleaved = false;
    let k_shape = Shape::new(vec![batch, nkv * steps, hd]);
    let arena_shape = Shape::new(vec![slot_stride]);

    let gamma: Vec<f32> = rand_f32(hd, 31).iter().map(|g| g + 1.0).collect();
    let gamma_t = f32_tensor(&dev, &gamma, &Shape::new(vec![hd]));

    // Accumulate 3 steps at mid-arena positions 3,4,5: exercises offset math
    // beyond position zero. Each step compares fresh arenas on both paths.
    for (step, pos) in [3u32, 4u32, 5u32].iter().enumerate() {
        let k_data = rand_f32(batch * nkv * steps * hd, 40 + step);
        let v_data = rand_f32(batch * nkv * steps * hd, 50 + step);
        let k_t = f32_tensor(&dev, &k_data, &k_shape);
        let v_t = f32_tensor(&dev, &v_data, &k_shape);
        let pos_t = u32_tensor(&dev, *pos);

        // Reference path: qk_rope(k) then two separate appends.
        let k_rot = f32_tensor(
            &dev,
            &vec![0.0f32; batch * nkv * steps * hd],
            &k_shape,
        );
        let arena_k_ref = f32_tensor(&dev, &vec![0.0f32; slot_stride], &arena_shape);
        let arena_v_ref = f32_tensor(&dev, &vec![0.0f32; slot_stride], &arena_shape);
        dev.qk_rope_dev_base_into(
            k_t.as_ref(),
            pos_t.as_ref(),
            gamma_t.as_ref(),
            eps,
            as_rocm(k_rot.as_ref()).unwrap(),
            &rope_cfg,
            &k_shape,
            nkv,
            steps,
        )
        .unwrap();
        grim_backend_rocm::launch_kv_append_batch(
            &dev,
            arena_k_ref.as_ref(),
            k_rot.as_ref(),
            pos_t.as_ref(),
            kv_stride,
            steps,
            batch,
            slot_stride,
        )
        .unwrap();
        grim_backend_rocm::launch_kv_append_batch(
            &dev,
            arena_v_ref.as_ref(),
            v_t.as_ref(),
            pos_t.as_ref(),
            kv_stride,
            steps,
            batch,
            slot_stride,
        )
        .unwrap();

        // Candidate: single fused launch into its own arenas.
        let k_rot2 = f32_tensor(
            &dev,
            &vec![0.0f32; batch * nkv * steps * hd],
            &k_shape,
        );
        let arena_k_new = f32_tensor(&dev, &vec![0.0f32; slot_stride], &arena_shape);
        let arena_v_new = f32_tensor(&dev, &vec![0.0f32; slot_stride], &arena_shape);
        dev.qk_rope_append_kv_into(
            k_t.as_ref(),
            pos_t.as_ref(),
            gamma_t.as_ref(),
            eps,
            v_t.as_ref(),
            as_rocm(k_rot2.as_ref()).unwrap(),
            arena_k_new.as_ref(),
            arena_v_new.as_ref(),
            &rope_cfg,
            &k_shape,
            nkv,
            steps,
            kv_stride,
            slot_stride,
        )
        .unwrap();

        let rk = k_rot.to_cpu_vec_f32().unwrap();
        let nk = k_rot2.to_cpu_vec_f32().unwrap();
        assert_eq!(max_abs_diff(&rk, &nk), 0.0, "step {step}: k rows drifted");
        let rak = arena_k_ref.to_cpu_vec_f32().unwrap();
        let nak = arena_k_new.to_cpu_vec_f32().unwrap();
        assert_eq!(
            max_abs_diff(&rak, &nak),
            0.0,
            "step {step}: k arena drifted"
        );
        let rav = arena_v_ref.to_cpu_vec_f32().unwrap();
        let nav = arena_v_new.to_cpu_vec_f32().unwrap();
        assert_eq!(
            max_abs_diff(&rav, &nav),
            0.0,
            "step {step}: v arena drifted"
        );
    }
}
