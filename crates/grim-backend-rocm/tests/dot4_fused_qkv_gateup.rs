//! PLAN-kernel-launch-reduction Phase A/B regressions:
//! - `grim_dot4_qkv_q80_gemv` (one launch, 3 sections) must match 3 separate
//!   `linear_decode_into` GEMVs.
//! - `grim_dot4_gate_up_silu_q80_gemv` (one launch) must match
//!   gate GEMV + up GEMV + silu_mul.
//!
//! Gated: GRIM_GPU_TEST=1 + ROCm device.

use grim_backend_rocm::RocmStorage;
use grim_backend_rocm::{as_rocm, gpu_test_enabled, RocmDevice};
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
    // Shape [rows, k] with a KQuant(Q80) dtype: the storage byte size is the
    // packed size (rows * (k/32) * 34), while elem_count() = rows*k so
    // `linear_decode_into`'s weight-shape validation passes.
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
    };
    MemoryOps::from_cpu_bytes(dev, packed, &Shape::new(vec![rows, k]), q_dtype).unwrap()
}

fn gemv_ref(
    dev: &RocmDevice,
    a: &Box<dyn grim_tensor::BackendStorage>,
    w: &Box<dyn grim_tensor::BackendStorage>,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let out = f32_tensor(dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    let act = f32_tensor(
        dev,
        &vec![0.0f32; (k / 32) * 9],
        &Shape::new(vec![(k / 32) * 36]),
    );
    dev.linear_decode_into(a.as_ref(), rocm(w), rocm(&out), rocm(&act))
        .unwrap();
    out.to_cpu_vec_f32().unwrap()
}

fn rocm<'a>(s: &'a Box<dyn grim_tensor::BackendStorage>) -> &'a RocmStorage {
    as_rocm(s.as_ref()).unwrap()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn rand_f32(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((seed + i * 7) % 113) as f32 / 113.0 - 0.5)
        .collect()
}

#[test]
#[ignore]
fn fused_qkv_matches_three_gemvs() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let k = 1024usize;
    let (n_q, n_kv) = (1024usize, 512usize);
    let x = rand_f32(k, 1);
    let a = f32_tensor(&dev, &x, &Shape::new(vec![1, k]));
    let wq = upload_q80(&dev, &pack_q80(&rand_f32(n_q * k, 2), n_q, k), n_q, k);
    let wk = upload_q80(&dev, &pack_q80(&rand_f32(n_kv * k, 3), n_kv, k), n_kv, k);
    let wv = upload_q80(&dev, &pack_q80(&rand_f32(n_kv * k, 4), n_kv, k), n_kv, k);

    // Reference: 3 separate GEMVs.
    let q_ref = gemv_ref(&dev, &a, &wq, n_q, k);
    let k_ref = gemv_ref(&dev, &a, &wk, n_kv, k);
    let v_ref = gemv_ref(&dev, &a, &wv, n_kv, k);

    // Fused: one launch, three output slots.
    let q_out = f32_tensor(&dev, &vec![0.0f32; n_q], &Shape::new(vec![n_q]));
    let k_out = f32_tensor(&dev, &vec![0.0f32; n_kv], &Shape::new(vec![n_kv]));
    let v_out = f32_tensor(&dev, &vec![0.0f32; n_kv], &Shape::new(vec![n_kv]));
    let act = f32_tensor(
        &dev,
        &vec![0.0f32; (k / 32) * 9],
        &Shape::new(vec![(k / 32) * 36]),
    );
    dev.fused_qkv_dot4_into(
        a.as_ref(),
        rocm(&wq),
        rocm(&wk),
        rocm(&wv),
        rocm(&q_out),
        rocm(&k_out),
        rocm(&v_out),
        n_q,
        n_kv,
        k,
        rocm(&act),
    )
    .unwrap();

    for (name, out, want) in [
        ("q", q_out.to_cpu_vec_f32().unwrap(), &q_ref),
        ("k", k_out.to_cpu_vec_f32().unwrap(), &k_ref),
        ("v", v_out.to_cpu_vec_f32().unwrap(), &v_ref),
    ] {
        for (i, (g, w)) in out.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-3 + w.abs() * 1e-4,
                "{name}[{i}]: {g} vs {w}"
            );
        }
    }
}

#[test]
#[ignore]
fn fused_gate_up_silu_matches_split() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let k = 1024usize;
    let n = 4608usize;
    let x = rand_f32(k, 5);
    let a = f32_tensor(&dev, &x, &Shape::new(vec![1, k]));
    let wg = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 6), n, k), n, k);
    let wu = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 7), n, k), n, k);

    let g_ref = gemv_ref(&dev, &a, &wg, n, k);
    let u_ref = gemv_ref(&dev, &a, &wu, n, k);

    let out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    let act = f32_tensor(
        &dev,
        &vec![0.0f32; (k / 32) * 9],
        &Shape::new(vec![(k / 32) * 36]),
    );
    dev.fused_gate_up_silu_dot4_into(
        a.as_ref(),
        rocm(&wg),
        rocm(&wu),
        rocm(&out),
        n,
        k,
        rocm(&act),
    )
    .unwrap();

    for (i, (got, (g, u))) in out
        .to_cpu_vec_f32()
        .unwrap()
        .iter()
        .zip(g_ref.iter().zip(&u_ref))
        .enumerate()
    {
        let want = silu(*g) * u;
        assert!(
            (got - want).abs() <= 1e-3 + want.abs() * 1e-4,
            "out[{i}]: {got} vs {want}"
        );
    }
}

#[test]
#[ignore]
fn prequantized_gate_up_path_uses_quantizer_plus_dot4() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    temp_env::with_var("GRIM_DOT4_PREQUANT", Some("1"), || {
        let k = 1024usize;
        let n = 4608usize;
        let a = f32_tensor(&dev, &rand_f32(k, 25), &Shape::new(vec![1, k]));
        let wg = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 26), n, k), n, k);
        let wu = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 27), n, k), n, k);
        let out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
        let act = f32_tensor(
            &dev,
            &vec![0.0f32; (k / 32) * 36],
            &Shape::new(vec![(k / 32) * 36]),
        );
        dev.reset_launch_count();
        dev.fused_gate_up_silu_dot4_into(
            a.as_ref(),
            rocm(&wg),
            rocm(&wu),
            rocm(&out),
            n,
            k,
            rocm(&act),
        )
        .unwrap();
        assert_eq!(
            dev.launch_count(),
            2,
            "prequantized path must launch quantize_q8_1 and dot4 gate/up"
        );
    });
}

#[test]
#[ignore]
fn fused_gate_up_silu_256_matches_split() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    temp_env::with_var("GRIM_DOT4_256", Some("1"), || {
        let k = 1024usize;
        let n = 4608usize;
        let x = rand_f32(k, 15);
        let a = f32_tensor(&dev, &x, &Shape::new(vec![1, k]));
        let wg = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 16), n, k), n, k);
        let wu = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 17), n, k), n, k);

        let g_ref = gemv_ref(&dev, &a, &wg, n, k);
        let u_ref = gemv_ref(&dev, &a, &wu, n, k);
        let out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
        let act = f32_tensor(
            &dev,
            &vec![0.0f32; (k / 32) * 9],
            &Shape::new(vec![(k / 32) * 36]),
        );
        dev.fused_gate_up_silu_dot4_into(
            a.as_ref(),
            rocm(&wg),
            rocm(&wu),
            rocm(&out),
            n,
            k,
            rocm(&act),
        )
        .unwrap();

        for (i, (got, (g, u))) in out
            .to_cpu_vec_f32()
            .unwrap()
            .iter()
            .zip(g_ref.iter().zip(&u_ref))
            .enumerate()
        {
            let want = silu(*g) * u;
            assert!(
                (got - want).abs() <= 1e-3 + want.abs() * 1e-4,
                "256-wave out[{i}]: {got} vs {want}"
            );
        }
    });
}

#[test]
#[ignore]
fn dot4_gate_up_microbench_uses_real_arguments() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let k = 1024usize;
    let n = 4608usize;
    let layers = 16usize;
    let iters = 8usize;
    let a = f32_tensor(&dev, &rand_f32(k, 41), &Shape::new(vec![1, k]));
    let wg_layers: Vec<_> = (0..layers)
        .map(|layer| {
            upload_q80(
                &dev,
                &pack_q80(&rand_f32(n * k, 42 + layer), n, k),
                n,
                k,
            )
        })
        .collect();
    let wu_layers: Vec<_> = (0..layers)
        .map(|layer| {
            upload_q80(
                &dev,
                &pack_q80(&rand_f32(n * k, 100 + layer), n, k),
                n,
                k,
            )
        })
        .collect();
    let out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    let act = f32_tensor(
        &dev,
        &vec![0.0f32; (k / 32) * 36],
        &Shape::new(vec![(k / 32) * 36]),
    );

    for layer in 0..layers {
        dev.fused_gate_up_silu_dot4_into(
            a.as_ref(),
            rocm(&wg_layers[layer]),
            rocm(&wu_layers[layer]),
            rocm(&out),
            n,
            k,
            rocm(&act),
        )
        .unwrap();
    }
    dev.synchronize();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        for layer in 0..layers {
            dev.fused_gate_up_silu_dot4_into(
                a.as_ref(),
                rocm(&wg_layers[layer]),
                rocm(&wu_layers[layer]),
                rocm(&out),
                n,
                k,
                rocm(&act),
            )
            .unwrap();
        }
    }
    dev.synchronize();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let bytes = (layers * iters) as f64 * 2.0 * n as f64 * (k as f64 / 32.0) * 34.0;
    let calls = (layers * iters) as f64;
    let gbps = bytes / (elapsed_ms * 1.0e6);
    eprintln!(
        "[dot4-gateup-microbench] shape=1x{k}x{n} layers={layers} iters={iters} total_ms={elapsed_ms:.3} per_call_us={:.3} weight_GBps={:.2}",
        elapsed_ms * 1000.0 / calls,
        gbps
    );
    assert!(
        out.to_cpu_vec_f32().unwrap().iter().all(|v| v.is_finite()),
        "microbench output must remain finite"
    );
}
