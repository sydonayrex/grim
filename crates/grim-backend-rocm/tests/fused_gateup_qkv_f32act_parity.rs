//! Task 10: Parity test for Phase D.2 direct f32act fused kernels.
//! Compares `launch_dot4_qkv_q80_f32act_gemv_into` and `launch_dot4_gate_up_silu_q80_f32act_gemv_into`
//! against the canonical 2-launch oracle (`launch_quantize_q8_1` + fused GEMV).
//! Tolerance standard: |got - want| <= 1e-5 (scale bit-exactness via (_Float16) cast).
//!
//! Gated: canonical `gpu_test_enabled()`.

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

#[test]
#[ignore]
fn fused_qkv_f32act_matches_quantized_oracle() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    if !dev.supports_dot4() {
        eprintln!("skipping: device does not support dot4");
        return;
    }

    let k = 1024usize;
    let (n_q, n_kv) = (1024usize, 512usize);
    let m = 1usize;
    let x = rand_f32(k, 101);
    let a = f32_tensor(&dev, &x, &Shape::new(vec![m, k]));
    let wq = upload_q80(&dev, &pack_q80(&rand_f32(n_q * k, 102), n_q, k), n_q, k);
    let wk = upload_q80(&dev, &pack_q80(&rand_f32(n_kv * k, 103), n_kv, k), n_kv, k);
    let wv = upload_q80(&dev, &pack_q80(&rand_f32(n_kv * k, 104), n_kv, k), n_kv, k);

    // Oracle: standalone quantize_q8_1 followed by launch_dot4_qkv_q80_gemv_into
    let act_q81 = f32_tensor(
        &dev,
        &vec![0.0f32; (k / 32) * 9],
        &Shape::new(vec![(k / 32) * 36]),
    );
    dev.launch_quantize_q8_1(rocm(&a), rocm(&act_q81), m, k)
        .unwrap();

    let q_ref = f32_tensor(&dev, &vec![0.0f32; n_q], &Shape::new(vec![n_q]));
    let k_ref = f32_tensor(&dev, &vec![0.0f32; n_kv], &Shape::new(vec![n_kv]));
    let v_ref = f32_tensor(&dev, &vec![0.0f32; n_kv], &Shape::new(vec![n_kv]));
    dev.launch_dot4_qkv_q80_gemv_into(
        rocm(&act_q81),
        rocm(&wq),
        rocm(&wk),
        rocm(&wv),
        rocm(&q_ref),
        rocm(&k_ref),
        rocm(&v_ref),
        m,
        n_q,
        n_kv,
        k,
    )
    .unwrap();

    // Subject under test: single-launch launch_dot4_qkv_q80_f32act_gemv_into directly from f32
    let q_got = f32_tensor(&dev, &vec![0.0f32; n_q], &Shape::new(vec![n_q]));
    let k_got = f32_tensor(&dev, &vec![0.0f32; n_kv], &Shape::new(vec![n_kv]));
    let v_got = f32_tensor(&dev, &vec![0.0f32; n_kv], &Shape::new(vec![n_kv]));
    dev.launch_dot4_qkv_q80_f32act_gemv_into(
        rocm(&a),
        rocm(&wq),
        rocm(&wk),
        rocm(&wv),
        rocm(&q_got),
        rocm(&k_got),
        rocm(&v_got),
        m,
        n_q,
        n_kv,
        k,
    )
    .unwrap();

    for (name, got_t, ref_t) in [
        ("q", q_got.to_cpu_vec_f32().unwrap(), q_ref.to_cpu_vec_f32().unwrap()),
        ("k", k_got.to_cpu_vec_f32().unwrap(), k_ref.to_cpu_vec_f32().unwrap()),
        ("v", v_got.to_cpu_vec_f32().unwrap(), v_ref.to_cpu_vec_f32().unwrap()),
    ] {
        for (i, (g, r)) in got_t.iter().zip(ref_t.iter()).enumerate() {
            let diff = (g - r).abs();
            assert!(
                diff <= 1e-4,
                "{name}[{i}]: f32act={g} vs oracle={r}, diff={diff}"
            );
        }
    }
}

#[test]
#[ignore]
fn fused_gate_up_silu_f32act_matches_quantized_oracle() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    if !dev.supports_dot4() {
        eprintln!("skipping: device does not support dot4");
        return;
    }

    let k = 1024usize;
    let n = 4608usize;
    let m = 1usize;
    let x = rand_f32(k, 201);
    let a = f32_tensor(&dev, &x, &Shape::new(vec![m, k]));
    let wg = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 202), n, k), n, k);
    let wu = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 203), n, k), n, k);

    // Oracle: standalone quantize_q8_1 followed by launch_dot4_gate_up_silu_q80_gemv_into
    let act_q81 = f32_tensor(
        &dev,
        &vec![0.0f32; (k / 32) * 9],
        &Shape::new(vec![(k / 32) * 36]),
    );
    dev.launch_quantize_q8_1(rocm(&a), rocm(&act_q81), m, k)
        .unwrap();

    let out_ref = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    dev.launch_dot4_gate_up_silu_q80_gemv_into(
        rocm(&act_q81),
        rocm(&wg),
        rocm(&wu),
        rocm(&out_ref),
        m,
        n,
        k,
    )
    .unwrap();

    // Subject under test: single-launch launch_dot4_gate_up_silu_q80_f32act_gemv_into directly from f32
    let out_got = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    dev.launch_dot4_gate_up_silu_q80_f32act_gemv_into(
        rocm(&a),
        rocm(&wg),
        rocm(&wu),
        rocm(&out_got),
        m,
        n,
        k,
    )
    .unwrap();

    let got_vec = out_got.to_cpu_vec_f32().unwrap();
    let ref_vec = out_ref.to_cpu_vec_f32().unwrap();
    for (i, (g, r)) in got_vec.iter().zip(ref_vec.iter()).enumerate() {
        let diff = (g - r).abs();
        assert!(
            diff <= 1e-4,
            "gateup_silu[{i}]: f32act={g} vs oracle={r}, diff={diff}"
        );
    }
}
