//! Q5_K / Q6_K big-tile parity + bench — the prefill formats the Q4_K_M
//! checkpoint uses for FFN/attn_v weights, which the first big-tile pass
//! (Q4_K only) left on the 16-row kernel (11.3 s + 10.4 s of a 33 s 4K
//! prefill; see the 2026-10-10 rocprofv3 breakdown in the session log).
//!
//! Full-C parity against the grim_quant host dequant (the same oracle the
//! 16-row parity gates use), through the DIRECT launchers for contract and
//! masked-partial shapes, then through the PRODUCTION dispatch with route
//! counters asserting the big tile owns large-M.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape, Storage,
};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1");
        return None;
    }
    Some(RocmDevice::try_new(0).unwrap())
}

#[test]
#[ignore]
fn big_q56_parity_and_route() {
    let Some(dev) = gpu_device() else { return; };
    let cases: &[(&str, KQuantScheme, &dyn Fn(&[f32]) -> Vec<u8>, &dyn Fn(&[u8], usize) -> Vec<f32>)] = &[
        ("q5k", KQuantScheme::Q5K, &|d| grim_quant::quant_q5k(d).unwrap(),
         &|b, n| grim_quant::dequant_q5k(b, n).unwrap()),
        ("q6k", KQuantScheme::Q6K, &|d| grim_quant::quant_q6k(d).unwrap(),
         &|b, n| grim_quant::dequant_q6k(b, n).unwrap()),
    ];

    for (fmt, scheme, quant, dequant) in cases {
        // direct-launcher parity: full tile, masked partial, big-k
        for (m, n, k) in [(128usize, 64, 256), (100, 64, 256), (128, 1024, 4096)] {
            let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
            let b_host: Vec<f32> =
                (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
            let packed = quant(&b_host);
            let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32)
                .unwrap();
            let b_dev = MemoryOps::from_cpu_bytes(
                &dev,
                &packed,
                &Shape::new(vec![packed.len()]),
                DType { arith: ArithType::F32, storage: Storage::KQuant(scheme.clone()) },
            )
            .unwrap();
            let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n]), DType::F32).unwrap();
            let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
            let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
            let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();
            match *fmt {
                "q5k" => dev.launch_wmma_big128_q5k_for_ab(a_r, b_r, o_r, m, n, k).unwrap(),
                _ => dev.launch_wmma_big128_q6k_for_ab(a_r, b_r, o_r, m, n, k).unwrap(),
            };
            let got = out.to_cpu_vec_f32().unwrap();
            let ndeq = dequant(&packed, k * n);
            let mut max_diff = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc += a_host[i * k + kk] * ndeq[j * k + kk];
                    }
                    max_diff = max_diff.max((got[i * n + j] - acc).abs());
                }
            }
            let tol = 0.02 * (k as f32).sqrt();
            eprintln!("[big{fmt}-parity] m={m} n={n} k={k} max_diff={max_diff:.6} (tol {tol:.3})");
            assert!(
                max_diff < tol,
                "{fmt} m={m} n={n} k={k}: big tile diverges (max_diff={max_diff})"
            );
        }

        // production dispatch: big tile must own large-M
        for m in [64usize, 1000usize] {
            let a_host: Vec<f32> = (0..m * 4096).map(|i| (i as f32 * 0.05).sin()).collect();
            let b_host: Vec<f32> =
                (0..4096 * 1024).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
            let packed = quant(&b_host);
            let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, 4096]), DType::F32)
                .unwrap();
            let b_dev = MemoryOps::from_cpu_bytes(
                &dev,
                &packed,
                &Shape::new(vec![packed.len()]),
                DType { arith: ArithType::F32, storage: Storage::KQuant(scheme.clone()) },
            )
            .unwrap();
            grim_backend_rocm::reset_kernel_route_counters();
            let fmt_gguf = match *fmt {
                "q5k" => grim_tensor::QuantFormat::Q5K,
                _ => grim_tensor::QuantFormat::Q6K,
            };
            let (_out, handle) = dev
                .quantized_matmul(
                    a_dev.as_ref(),
                    b_dev.as_ref(),
                    &[],
                    fmt_gguf,
                    &Shape::new(vec![m, 1024]),
                )
                .expect("quantized_matmul dispatch");
            handle.synchronize().expect("sync");
            let name = match (fmt.as_ref(), m) {
                ("q5k", mm) if mm >= 128 => "grim_wmma_big128_q5k",
                ("q5k", _) => "grim_wmma_big_q5k",
                ("q6k", mm) if mm >= 128 => "grim_wmma_big128_q6k",
                (_, _) => "grim_wmma_big_q6k",
            };
            let hits = grim_backend_rocm::rocm_kernel_route_counter(name);
            eprintln!("[big{fmt}-route] m={m} {name}={hits}");
            assert!(hits > 0, "{fmt} m={m}: {name} did not serve");
        }
    }
}
