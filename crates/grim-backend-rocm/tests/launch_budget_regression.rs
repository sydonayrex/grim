//! PLAN 4 Task 5: launch-budget regression guard for fused launchers.
//!
//! Each fused launcher below must cost EXACTLY ONE kernel launch — that is
//! the entire contract of the Plan 2–4 fusion work (norm/QKV/rope fusions
//! replace N launches with 1). If anyone re-splits a fused path into
//! separate launches, these counts rise and the test fails.
//!
//! Gated: GRIM_GPU_TEST=1 (canonical `gpu_test_enabled`) + ROCm device.

use grim_backend_rocm::{RocmDevice, as_rocm, gpu_test_enabled};
use grim_tensor::{ArithType, CoreTensorOps, DType, MemoryOps, Shape, Storage};

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

fn rand_f32(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((seed + i * 7) % 113) as f32 / 113.0 - 0.5)
        .collect()
}

fn pack_q80(w: &[f32], rows: usize, k: usize) -> Vec<u8> {
    let blocks = k / 32;
    let mut out = vec![0u8; rows * blocks * 34];
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * 34;
            let mut amax = 0.0f32;
            for e in 0..32 {
                amax = amax.max(w[r * k + b * 32 + e].abs());
            }
            let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            let h = half::f16::from_f32(d);
            out[off..off + 2].copy_from_slice(&h.to_le_bytes());
            for e in 0..32 {
                let v = w[r * k + b * 32 + e] / d;
                out[off + 2 + e] = v.round().clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    }
    out
}

fn upload_q80(
    dev: &RocmDevice,
    packed: &[u8],
    rows: usize,
    k: usize,
) -> Box<dyn grim_tensor::BackendStorage> {
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
    };
    MemoryOps::from_cpu_bytes(dev, packed, &Shape::new(vec![rows, k]), q_dtype).unwrap()
}

fn rocm<'a>(s: &'a Box<dyn grim_tensor::BackendStorage>) -> &'a grim_backend_rocm::RocmStorage {
    as_rocm(s.as_ref()).unwrap()
}

#[test]
#[ignore]
fn fused_launchers_cost_exactly_one_launch_each() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let _guard = grim_backend_rocm::device::util::gpu_test_lock();

    // --- norm-fused QKV (Task 2) ---
    let (k, n_q, n_kv, m) = (1024usize, 1024usize, 512usize, 1usize);
    let res = f32_tensor(&dev, &rand_f32(m * k, 61), &Shape::new(vec![m, k]));
    let gamma = f32_tensor(
        &dev,
        &rand_f32(k, 62).iter().map(|g| g + 1.0).collect::<Vec<_>>(),
        &Shape::new(vec![k]),
    );
    let wq = upload_q80(&dev, &pack_q80(&rand_f32(n_q * k, 63), n_q, k), n_q, k);
    let wk = upload_q80(&dev, &pack_q80(&rand_f32(n_kv * k, 64), n_kv, k), n_kv, k);
    let wv = upload_q80(&dev, &pack_q80(&rand_f32(n_kv * k, 65), n_kv, k), n_kv, k);
    let (q, kk, v) = (
        f32_tensor(&dev, &vec![0.0; m * n_q], &Shape::new(vec![m, n_q])),
        f32_tensor(&dev, &vec![0.0; m * n_kv], &Shape::new(vec![m, n_kv])),
        f32_tensor(&dev, &vec![0.0; m * n_kv], &Shape::new(vec![m, n_kv])),
    );
    dev.reset_launch_count();
    dev.fused_qkv_dot4_norm_into(
        rocm(&res),
        rocm(&gamma),
        1e-5,
        rocm(&wq),
        rocm(&wk),
        rocm(&wv),
        rocm(&q),
        rocm(&kk),
        rocm(&v),
        n_q,
        n_kv,
        k,
    )
    .unwrap();
    assert_eq!(
        dev.launch_count(),
        1,
        "norm-fused QKV must be a single launch (was split?)"
    );

    // --- generic norm-fused GEMV (Task 3) ---
    let (n2, k2) = (384usize, 1024usize);
    let res2 = f32_tensor(&dev, &rand_f32(k2, 66), &Shape::new(vec![1, k2]));
    let w2 = upload_q80(&dev, &pack_q80(&rand_f32(n2 * k2, 67), n2, k2), n2, k2);
    let o2 = f32_tensor(&dev, &vec![0.0; n2], &Shape::new(vec![1, n2]));
    dev.reset_launch_count();
    dev.launch_dot4_q80_norm_f32act_gemv_into(
        rocm(&res2),
        rocm(&gamma),
        1e-5,
        rocm(&w2),
        rocm(&o2),
        1,
        n2,
        k2,
    )
    .unwrap();
    assert_eq!(
        dev.launch_count(),
        1,
        "norm-fused GEMV must be a single launch (was split?)"
    );

    // --- rope+append fusion (Task 4) ---
    let (nkv, hd) = (2usize, 64usize);
    let kv_stride = nkv * hd;
    let slot_stride = 8 * kv_stride;
    let k_shape = Shape::new(vec![1, nkv, hd]);
    let k_t = f32_tensor(&dev, &rand_f32(nkv * hd, 68), &k_shape);
    let v_t = f32_tensor(&dev, &rand_f32(nkv * hd, 69), &k_shape);
    let pos_t = CoreTensorOps::from_cpu(
        &dev,
        &[f32::from_bits(3)],
        &Shape::new(vec![1]),
        DType::U32,
    )
    .unwrap();
    let gamma_k = f32_tensor(
        &dev,
        &rand_f32(hd, 70).iter().map(|g| g + 1.0).collect::<Vec<_>>(),
        &Shape::new(vec![hd]),
    );
    let k_rot = f32_tensor(&dev, &vec![0.0; nkv * hd], &k_shape);
    let arena_k = f32_tensor(&dev, &vec![0.0; slot_stride], &Shape::new(vec![slot_stride]));
    let arena_v = f32_tensor(&dev, &vec![0.0; slot_stride], &Shape::new(vec![slot_stride]));
    let mut rope_cfg = grim_tensor::RopeConfig::new(hd, 10000.0f32);
    rope_cfg.interleaved = false;
    dev.reset_launch_count();
    dev.qk_rope_append_kv_into(
        k_t.as_ref(),
        pos_t.as_ref(),
        gamma_k.as_ref(),
        1e-5,
        v_t.as_ref(),
        rocm(&k_rot),
        arena_k.as_ref(),
        arena_v.as_ref(),
        &rope_cfg,
        &k_shape,
        nkv,
        1,
        kv_stride,
        slot_stride,
    )
    .unwrap();
    assert_eq!(
        dev.launch_count(),
        1,
        "rope+append fusion must be a single launch (was split?)"
    );
}
