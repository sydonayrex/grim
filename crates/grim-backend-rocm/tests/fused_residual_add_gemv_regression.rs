//! Task 11: Regression test for Phase D.2 fused residual add GEMV.
//! Tests `launch_dot4_q80_f32act_add_gemv` for:
//! 1. Parity against standalone GEMV + elementwise add.
//! 2. In-place aliasing safety when `C == residual`.
//! 3. Null residual (`residual == None`) behaves identically to standalone GEMV.
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
fn fused_residual_rms_gateup_matches_split_path() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    if !dev.supports_dot4() {
        eprintln!("skipping: device does not support dot4");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    let m = 1usize;
    let k = 1024usize;
    let n = 2048usize;
    let eps = 1e-5f32;
    let base = f32_tensor(&dev, &rand_f32(k, 501), &Shape::new(vec![m, k]));
    let attn = f32_tensor(&dev, &rand_f32(k, 502), &Shape::new(vec![m, k]));
    let gamma = f32_tensor(&dev, &vec![1.0; k], &Shape::new(vec![k]));
    let wg = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 503), n, k), n, k);
    let wu = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 504), n, k), n, k);

    let residual_ref = f32_tensor(&dev, &vec![0.0; k], &Shape::new(vec![m, k]));
    let norm_ref = f32_tensor(&dev, &vec![0.0; k], &Shape::new(vec![m, k]));
    let activated_ref = f32_tensor(&dev, &vec![0.0; n], &Shape::new(vec![m, n]));
    let act_q81 = f32_tensor(
        &dev,
        &vec![0.0; (k / 32) * 36],
        &Shape::new(vec![(k / 32) * 36]),
    );
    dev.add_into(base.as_ref(), attn.as_ref(), rocm(&residual_ref))
        .unwrap();
    dev.rms_norm_into(
        residual_ref.as_ref(),
        gamma.as_ref(),
        eps,
        rocm(&norm_ref),
        &Shape::new(vec![m, k]),
    )
    .unwrap();
    dev.fused_gate_up_silu_dot4_into(
        norm_ref.as_ref(),
        rocm(&wg),
        rocm(&wu),
        rocm(&activated_ref),
        n,
        k,
        rocm(&act_q81),
    )
    .unwrap();

    let residual_got = f32_tensor(&dev, &vec![0.0; k], &Shape::new(vec![m, k]));
    let activated_got = f32_tensor(&dev, &vec![0.0; n], &Shape::new(vec![m, n]));
    dev.fused_add_rms_norm_gate_up_silu_dot4_into(
        rocm(&base),
        rocm(&attn),
        rocm(&gamma),
        eps,
        rocm(&wg),
        rocm(&wu),
        rocm(&residual_got),
        rocm(&activated_got),
        m,
        n,
        k,
    )
    .unwrap();

    for (i, (got, want)) in residual_got
        .to_cpu_vec_f32()
        .unwrap()
        .iter()
        .zip(residual_ref.to_cpu_vec_f32().unwrap())
        .enumerate()
    {
        assert!((got - want).abs() < 1e-4, "residual[{i}]: {got} vs {want}");
    }
    for (i, (got, want)) in activated_got
        .to_cpu_vec_f32()
        .unwrap()
        .iter()
        .zip(activated_ref.to_cpu_vec_f32().unwrap())
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-3 + want.abs() * 1e-4,
            "activated[{i}]: {got} vs {want}"
        );
    }
}

#[test]
#[ignore]
fn fused_residual_add_gemv_matches_gemv_plus_add() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    if !dev.supports_dot4() {
        eprintln!("skipping: device does not support dot4");
        return;
    }

    let k = 1024usize;
    let n = 2048usize;
    let m = 1usize;
    let x = rand_f32(k, 301);
    let res_data = rand_f32(n, 302);
    let a = f32_tensor(&dev, &x, &Shape::new(vec![m, k]));
    let w = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 303), n, k), n, k);
    let residual = f32_tensor(&dev, &res_data, &Shape::new(vec![n]));

    // Reference: standalone f32act GEMV + CPU add
    let gemv_out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    dev.launch_dot4_q80_f32act_gemv(rocm(&a), rocm(&w), rocm(&gemv_out), m, n, k)
        .unwrap();
    let gemv_vec = gemv_out.to_cpu_vec_f32().unwrap();
    let want_res: Vec<f32> = gemv_vec.iter().zip(res_data.iter()).map(|(g, r)| g + r).collect();

    // Test 1: Distinct output buffer
    let out_distinct = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    dev.launch_dot4_q80_f32act_add_gemv(
        rocm(&a),
        rocm(&w),
        Some(rocm(&residual)),
        rocm(&out_distinct),
        m,
        n,
        k,
    )
    .unwrap();

    let got_distinct = out_distinct.to_cpu_vec_f32().unwrap();
    for (i, (g, w)) in got_distinct.iter().zip(want_res.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4,
            "distinct[{i}]: got={g} vs want={w}"
        );
    }

    // Test 2: In-place aliasing (C == residual)
    let in_place = f32_tensor(&dev, &res_data, &Shape::new(vec![n]));
    dev.launch_dot4_q80_f32act_add_gemv(
        rocm(&a),
        rocm(&w),
        Some(rocm(&in_place)),
        rocm(&in_place),
        m,
        n,
        k,
    )
    .unwrap();

    let got_inplace = in_place.to_cpu_vec_f32().unwrap();
    for (i, (g, w)) in got_inplace.iter().zip(want_res.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4,
            "inplace[{i}]: got={g} vs want={w}"
        );
    }

    // Test 3: None residual matches standalone GEMV
    let out_none = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    dev.launch_dot4_q80_f32act_add_gemv(
        rocm(&a),
        rocm(&w),
        None,
        rocm(&out_none),
        m,
        n,
        k,
    )
    .unwrap();

    let got_none = out_none.to_cpu_vec_f32().unwrap();
    for (i, (g, w)) in got_none.iter().zip(gemv_vec.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5,
            "none_res[{i}]: got={g} vs gemv={w}"
        );
    }
}

#[test]
#[ignore]
fn fused_residual_add_gemv_tile16_matches_split() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    if !dev.supports_dot4() {
        eprintln!("skipping: device does not support dot4");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    temp_env::with_var("GRIM_DOT4_TILE16", Some("1"), || {
        let k = 1024usize;
        let n = 2048usize;
        let m = 1usize;
        let x = rand_f32(k, 501);
        let res_data = rand_f32(n, 502);
        let a = f32_tensor(&dev, &x, &Shape::new(vec![m, k]));
        let w = upload_q80(&dev, &pack_q80(&rand_f32(n * k, 503), n, k), n, k);
        let residual = f32_tensor(&dev, &res_data, &Shape::new(vec![n]));

        let gemv_out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
        dev.launch_dot4_q80_f32act_gemv(rocm(&a), rocm(&w), rocm(&gemv_out), m, n, k)
            .unwrap();
        let gemv_vec = gemv_out.to_cpu_vec_f32().unwrap();
        let want: Vec<f32> = gemv_vec
            .iter()
            .zip(res_data.iter())
            .map(|(g, r)| g + r)
            .collect();

        let out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
        dev.launch_dot4_q80_f32act_add_gemv(
            rocm(&a),
            rocm(&w),
            Some(rocm(&residual)),
            rocm(&out),
            m,
            n,
            k,
        )
        .unwrap();
        let got = out.to_cpu_vec_f32().unwrap();
        for (i, (got, want)) in got.iter().zip(&want).enumerate() {
            assert!((got - want).abs() < 1e-4, "tile16[{i}]: got={got} vs want={want}");
        }
    });
}

#[test]
#[ignore]
fn dot4_add_microbench_compares_default_and_tile16() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    if !dev.supports_dot4() {
        eprintln!("skipping: device does not support dot4");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let k = 4608usize;
    let n = 1024usize;
    let layers = 16usize;
    let iters = 8usize;
    let activation = f32_tensor(&dev, &rand_f32(k, 601), &Shape::new(vec![1, k]));
    let residual = f32_tensor(&dev, &rand_f32(n, 602), &Shape::new(vec![n]));
    let out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    let weights: Vec<_> = (0..layers)
        .map(|layer| {
            upload_q80(
                &dev,
                &pack_q80(&rand_f32(n * k, 603 + layer), n, k),
                n,
                k,
            )
        })
        .collect();

    let run = |label: &str| {
        for layer in 0..layers {
            dev.launch_dot4_q80_f32act_add_gemv(
                rocm(&activation),
                rocm(&weights[layer]),
                Some(rocm(&residual)),
                rocm(&out),
                1,
                n,
                k,
            )
            .unwrap();
        }
        dev.synchronize();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for layer in 0..layers {
                dev.launch_dot4_q80_f32act_add_gemv(
                    rocm(&activation),
                    rocm(&weights[layer]),
                    Some(rocm(&residual)),
                    rocm(&out),
                    1,
                    n,
                    k,
                )
                .unwrap();
            }
        }
        dev.synchronize();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let calls = (layers * iters) as f64;
        let bytes = calls * n as f64 * (k as f64 / 32.0) * 34.0;
        eprintln!(
            "[dot4-add-microbench] {label} shape=1x{k}x{n} layers={layers} per_call_us={:.3} weight_GBps={:.2}",
            elapsed_ms * 1000.0 / calls,
            bytes / (elapsed_ms * 1.0e6),
        );
    };

    temp_env::with_var("GRIM_DOT4_TILE16", None::<&str>, || run("default"));
    temp_env::with_var("GRIM_DOT4_TILE16", Some("1"), || run("tile16"));
    assert!(out.to_cpu_vec_f32().unwrap().iter().all(|v| v.is_finite()));
}
