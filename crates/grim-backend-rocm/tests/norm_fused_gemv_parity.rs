//! PLAN 4 (DukeNukem) Task 2: norm-fused QKV must be bit-identical to the
//! separate `rms_norm_into` + f32act-QKV path (same traversal, formula,
//! quantizer — fp max is order-exact, fp16 scale round-trip, roundf codes).
//!
//! Gated: GRIM_GPU_TEST=1 (canonical `gpu_test_enabled`) + ROCm device.

use grim_backend_rocm::RocmStorage;
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

/// Pack [rows, k] f32 weights into the gguf Q8_0 layout
/// (per 32-block: f16 scale + 32 i8 codes = 34 bytes).
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

fn rocm<'a>(s: &'a Box<dyn grim_tensor::BackendStorage>) -> &'a RocmStorage {
    as_rocm(s.as_ref()).unwrap()
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
fn norm_fused_qkv_matches_separate_norm_path() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let _guard = grim_backend_rocm::device::util::gpu_test_lock();
    let k = 1024usize;
    let (n_q, n_kv) = (1024usize, 512usize);
    let m = 1usize;
    let eps = 1e-5f32;

    let res = rand_f32(m * k, 11);
    let gamma: Vec<f32> = rand_f32(k, 12).iter().map(|g| g + 1.0).collect();
    let res_t = f32_tensor(&dev, &res, &Shape::new(vec![m, k]));
    let gamma_t = f32_tensor(&dev, &gamma, &Shape::new(vec![k]));
    let wq = upload_q80(&dev, &pack_q80(&rand_f32(n_q * k, 2), n_q, k), n_q, k);
    let wk = upload_q80(
        &dev,
        &pack_q80(&rand_f32(n_kv * k, 3), n_kv, k),
        n_kv,
        k,
    );
    let wv = upload_q80(
        &dev,
        &pack_q80(&rand_f32(n_kv * k, 4), n_kv, k),
        n_kv,
        k,
    );

    // Reference: standalone rms_norm_into, then the f32act QKV wrapper.
    let normed = f32_tensor(&dev, &vec![0.0f32; m * k], &Shape::new(vec![m, k]));
    dev.rms_norm_into(
        res_t.as_ref(),
        gamma_t.as_ref(),
        eps,
        rocm(&normed),
        &Shape::new(vec![m, k]),
    )
    .unwrap();
    let (q_ref, k_ref, v_ref) = (
        f32_tensor(&dev, &vec![0.0f32; m * n_q], &Shape::new(vec![m, n_q])),
        f32_tensor(&dev, &vec![0.0f32; m * n_kv], &Shape::new(vec![m, n_kv])),
        f32_tensor(&dev, &vec![0.0f32; m * n_kv], &Shape::new(vec![m, n_kv])),
    );
    dev.fused_qkv_dot4_into(
        normed.as_ref(),
        rocm(&wq),
        rocm(&wk),
        rocm(&wv),
        rocm(&q_ref),
        rocm(&k_ref),
        rocm(&v_ref),
        n_q,
        n_kv,
        k,
        rocm(&f32_tensor(&dev, &vec![0.0f32; 9 * (k / 32)], &Shape::new(vec![(k / 32) * 9]))),
    )
    .unwrap();

    // Candidate: single norm-fused launch straight from the residual.
    let (q_new, k_new, v_new) = (
        f32_tensor(&dev, &vec![0.0f32; m * n_q], &Shape::new(vec![m, n_q])),
        f32_tensor(&dev, &vec![0.0f32; m * n_kv], &Shape::new(vec![m, n_kv])),
        f32_tensor(&dev, &vec![0.0f32; m * n_kv], &Shape::new(vec![m, n_kv])),
    );
    dev.fused_qkv_dot4_norm_into(
        rocm(&res_t),
        rocm(&gamma_t),
        eps,
        rocm(&wq),
        rocm(&wk),
        rocm(&wv),
        rocm(&q_new),
        rocm(&k_new),
        rocm(&v_new),
        n_q,
        n_kv,
        k,
    )
    .unwrap();

    for (name, r, n) in [
        ("q", &q_ref, &q_new),
        ("k", &k_ref, &k_new),
        ("v", &v_ref, &v_new),
    ] {
        let rv = r.to_cpu_vec_f32().unwrap();
        let nv = n.to_cpu_vec_f32().unwrap();
        let d = max_abs_diff(&rv, &nv);
        assert_eq!(d, 0.0, "{name} drifted: max abs diff {d}");
    }
}

/// PLAN 4 Task 3: generic norm-fused single-projection GEMV (shortconv
/// in_proj shape N=384, lm_head-ish N=256, plus N=258 tail coverage) is
/// bit-identical to separate rms_norm_into + f32act GEMV.
#[test]
#[ignore]
fn norm_fused_gemv_matches_separate_norm_path_generic() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let _guard = grim_backend_rocm::device::util::gpu_test_lock();
    let k = 1024usize;
    let eps = 1e-5f32;
    for (tag, n, seed) in [("in_proj", 384usize, 21usize), ("head", 256, 22), ("tail", 258, 23)] {
        let m = 1usize;
        let res = rand_f32(m * k, seed);
        let gamma: Vec<f32> = rand_f32(k, seed + 100).iter().map(|g| g + 1.0).collect();
        let res_t = f32_tensor(&dev, &res, &Shape::new(vec![m, k]));
        let gamma_t = f32_tensor(&dev, &gamma, &Shape::new(vec![k]));
        let w = upload_q80(&dev, &pack_q80(&rand_f32(n * k, seed + 1), n, k), n, k);

        // Reference: standalone norm, then f32act GEMV.
        let normed = f32_tensor(&dev, &vec![0.0f32; m * k], &Shape::new(vec![m, k]));
        dev.rms_norm_into(
            res_t.as_ref(),
            gamma_t.as_ref(),
            eps,
            rocm(&normed),
            &Shape::new(vec![m, k]),
        )
        .unwrap();
        let o_ref = f32_tensor(&dev, &vec![0.0f32; m * n], &Shape::new(vec![m, n]));
        dev.launch_dot4_q80_f32act_gemv(rocm(&normed), rocm(&w), rocm(&o_ref), m, n, k)
            .unwrap();

        // Candidate: single norm-fused launch.
        let o_new = f32_tensor(&dev, &vec![0.0f32; m * n], &Shape::new(vec![m, n]));
        dev.launch_dot4_q80_norm_f32act_gemv_into(
            rocm(&res_t),
            rocm(&gamma_t),
            eps,
            rocm(&w),
            rocm(&o_new),
            m,
            n,
            k,
        )
        .unwrap();

        let rv = o_ref.to_cpu_vec_f32().unwrap();
        let nv = o_new.to_cpu_vec_f32().unwrap();
        let d = max_abs_diff(&rv, &nv);
        assert_eq!(d, 0.0, "{tag} (N={n}) drifted: max abs diff {d}");
    }
}
