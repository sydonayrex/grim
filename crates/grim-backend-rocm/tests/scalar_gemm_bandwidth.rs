//! Scalar fused-dequant GEMM bandwidth bench (Q4K/Q5K/Q6K).
//!
//! Times `launch_fused_dequant_gemm_q{4,5,6}k` directly at prefill shapes and
//! reports weight-read GB/s. Unlike `dot4_gemv_floor_probe` (which goes through
//! `linear_decode_into`/dot4), this exercises the scalar kernels, so it is the
//! right tool for measuring scalar-kernel changes.
//!
//! RUN: GRIM_RUN_GPU_TESTS=1 HIP_VISIBLE_DEVICES=0 cargo test -p grim-backend-rocm
//!      --test scalar_gemm_bandwidth -- --ignored --nocapture --test-threads=1
//!
//! NOTE: Q4K routes M>=2 to the tiled kernel by default; set GRIM_Q4K_TILED=0
//! to force the scalar path being measured here.

use grim_backend_rocm::{RocmDevice, RocmStorage};
use grim_tensor::{ArithType, CoreTensorOps, DType, KQuantScheme, Shape, Storage};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[scalar-bw] skipping: set GRIM_RUN_GPU_TESTS=1");
        return None;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn pack(scheme: KQuantScheme, n: usize, k: usize) -> Vec<u8> {
    let mut s = 0x12345u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let vals: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    match scheme {
        KQuantScheme::Q4K => grim_quant::quant_q4k(&vals).expect("quant_q4k"),
        KQuantScheme::Q5K => grim_quant::quant_q5k(&vals).expect("quant_q5k"),
        KQuantScheme::Q6K => grim_quant::quant_q6k(&vals).expect("quant_q6k"),
        other => panic!("unsupported scheme {other:?}"),
    }
}

fn block_bytes(scheme: KQuantScheme) -> f64 {
    match scheme {
        KQuantScheme::Q4K => 144.0,
        KQuantScheme::Q5K => 176.0,
        KQuantScheme::Q6K => 210.0,
        _ => unreachable!(),
    }
}

fn bench(dev: &RocmDevice, scheme: KQuantScheme, m: usize, n: usize, k: usize, iters: usize) {
    let alloc = dev.allocator_handle();
    let w_bytes = pack(scheme, n, k);
    let w = grim_tensor::MemoryOps::from_cpu_bytes(
        dev,
        &w_bytes,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(scheme),
        },
    )
    .expect("upload weights");
    let w_rocm = w
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("rocm weights");
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let a_st = CoreTensorOps::from_cpu(dev, &a, &Shape::new(vec![m, k]), DType::F32).expect("A");
    let a_rocm = a_st
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("rocm act");
    let out = RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &alloc, 0).expect("out");

    // Warmup (JIT compile).
    for _ in 0..2 {
        launch(dev, scheme, a_rocm, w_rocm, &out, m, n, k);
    }
    dev.synchronize();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        launch(dev, scheme, a_rocm, w_rocm, &out, m, n, k);
    }
    dev.synchronize();
    let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
    // Weight traffic: every output row re-reads the full bank on the scalar
    // path, so count m banks per launch.
    let gb = m as f64 * n as f64 * (k as f64 / 256.0) * block_bytes(scheme) / 1e9;
    let gbps = gb / (us * 1e-6);
    let name = match scheme {
        KQuantScheme::Q4K => "Q4K",
        KQuantScheme::Q5K => "Q5K",
        KQuantScheme::Q6K => "Q6K",
        _ => "?",
    };
    eprintln!("[scalar-bw] {name} m={m:<3} n={n:<5} k={k:<5} per_launch={us:>9.1} us  weight_GBps={gbps:>7.1}");
}

fn launch(
    dev: &RocmDevice,
    scheme: KQuantScheme,
    a: &RocmStorage,
    w: &RocmStorage,
    out: &RocmStorage,
    m: usize,
    n: usize,
    k: usize,
) {
    match scheme {
        KQuantScheme::Q4K => {
            dev.launch_fused_dequant_gemm_q4k_for_ab(a, w, out, m, n, k)
                .expect("q4k");
        }
        KQuantScheme::Q5K => {
            dev.launch_fused_dequant_gemm_q5k_for_ab(a, w, out, m, n, k)
                .expect("q5k");
        }
        KQuantScheme::Q6K => {
            dev.launch_fused_dequant_gemm_q6k_for_ab(a, w, out, m, n, k)
                .expect("q6k");
        }
        _ => unreachable!(),
    };
}

#[ignore = "device-gated: run with GRIM_RUN_GPU_TESTS=1"]
#[test]
fn scalar_gemm_bandwidth() {
    let Some(dev) = gpu_device() else {
        return;
    };
    for scheme in [KQuantScheme::Q4K, KQuantScheme::Q5K, KQuantScheme::Q6K] {
        for (m, n, k, iters) in [(1usize, 4096usize, 4096usize, 20usize), (8, 4096, 4096, 10)] {
            bench(&dev, scheme, m, n, k, iters);
        }
    }
}
