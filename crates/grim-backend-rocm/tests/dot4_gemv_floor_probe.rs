//! SPEED-PROBE (9B perf plan Step 1): is the dot4 GEMV ~1 ms per-launch floor
//! kernel-structural or bandwidth-honest?
//!
//! rocprof on the 9B (gfx1201) shows ~222 dot4 GEMV launches per decoded token
//! with a near-uniform ~1-3 ms floor regardless of tensor size — ~12.6 GB/s
//! effective weight-read bandwidth on a ~896 GB/s card — while the ONE big
//! launch (lm_head, Q6K, 1.27 GB, grid 62080) sustains ~554 GB/s. Same kernel
//! family, two regimes. This probe pins down which:
//!
//!   a) small tensor (N=16384, grid 4096 blocks) timed per launch — ~1 ms => structural floor;
//!   b) same tensor, 200 back-to-back in one window — per-launch DROPS => queue amortization; FLAT => in-kernel floor;
//!   c) lm_head-scale tensor (N=248320, grid 62080) — the known-good bandwidth ceiling for the SAME kernel;
//!   d) q5k/q6k at the small shape — format dependence of the floor.
//!
//! Everything goes through `linear_decode_into` — the production decode entry —
//! so the numbers include the Q8_1 prequant launch exactly as inference pays
//! them. No production code is changed by this test; it is evidence, not a gate.
//!
//! RUN: GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=1 cargo test -p grim-backend-rocm \
//!      --test dot4_gemv_floor_probe -- --ignored --nocapture

use grim_backend_rocm::memory::allocator::RocmCachingAllocator;
use grim_backend_rocm::RocmStorage;
use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage,
};
use std::sync::Arc;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[floor-probe] skipping: set GRIM_RUN_GPU_TEST=1");
        return None;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

/// Pack an n×k f32 matrix to a k-quant format using grim-quant's own encoders
/// (bit-exact vs llama.cpp), the same bytes a checkpoint stores.
fn pack_qk(scheme: KQuantScheme, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
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

struct Case {
    label: &'static str,
    scheme: KQuantScheme,
    n: usize,
    k: usize,
}

/// Time `iters` launches of the production decode entry; returns
/// (per_launch_us, weight_GBps) over one sync at the end.
fn bench(dev: &RocmDevice, c: &Case, iters: usize, seed: u64) -> (f64, f64) {
    // Same override the parity gate uses: production dispatch on non-RDNA2
    // arches checks this env before the dot4 leg.
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }

    let alloc: Arc<RocmCachingAllocator> = dev.allocator_handle();
    let k = c.k;
    let n = c.n;

    let w_bytes = pack_qk(c.scheme, n, k, seed);
    let w_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(c.scheme),
    };
    // Shape [n, k]: linear_decode_into derives n from elem_count()/k, so the
    // storage must carry the logical element shape, not the byte length.
    let w = MemoryOps::from_cpu_bytes(dev, &w_bytes, &Shape::new(vec![n, k]), w_dtype)
        .expect("upload weights");
    let w_rocm = w
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("rocm weight storage");

    let a: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let a_st = CoreTensorOps::from_cpu(dev, &a, &Shape::new(vec![1, k]), DType::F32)
        .expect("upload act");

    let q81_bytes = (k / 32) * 36;
    let act_q81 = RocmStorage::alloc_gpu(
        &Shape::new(vec![q81_bytes]),
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
        &alloc,
        0,
    )
    .expect("alloc q81 scratch");
    let out = RocmStorage::alloc_gpu(
        &Shape::new(vec![n]),
        DType::F32,
        &alloc,
        0,
    )
    .expect("alloc out");

    // Warmup: JIT compile + allocator priming, untimed.
    for _ in 0..3 {
        dev.linear_decode_into(a_st.as_ref(), w_rocm, &out, &act_q81)
            .expect("linear_decode_into");
    }
    dev.synchronize();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        dev.linear_decode_into(a_st.as_ref(), w_rocm, &out, &act_q81)
            .expect("linear_decode_into");
    }
    dev.synchronize();
    let per_launch_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let weight_gb = n as f64 * (k as f64 / 256.0) * block_bytes(c.scheme) / 1e9;
    let gbps = weight_gb / (per_launch_us * 1e-6);
    (per_launch_us, gbps)
}

fn report(c: &Case, per_launch_us: f64, gbps: f64) {
    eprintln!(
        "[floor-probe] {:<26} scheme={:?} n={:<7} per_launch={:>8.1} us  weight_GBps={:>7.1}",
        c.label,
        c.scheme,
        c.n,
        per_launch_us,
        gbps
    );
}

#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
#[test]
fn dot4_gemv_floor_small_vs_big_launch() {
    let Some(dev) = gpu_device() else {
        return;
    };
    let k = 4096usize;

    // (a) small launch, sparsely issued.
    let small = Case { label: "a_small_q4k", scheme: KQuantScheme::Q4K, n: 16384, k };
    let (us, gbps) = bench(&dev, &small, 20, 41);
    report(&small, us, gbps);

    // (b) same tensor, densely issued — queue amortization vs in-kernel floor.
    let (us10, gbps10) = bench(&dev, &small, 200, 41);
    eprintln!(
        "[floor-probe] b_small_q4k_x200            amortized per_launch={:>8.1} us  weight_GBps={:>7.1}",
        us10, gbps10
    );

    // (c) lm_head-scale launch of the SAME kernel — the bandwidth ceiling.
    let big = Case { label: "c_big_q4k_lmhead_scale", scheme: KQuantScheme::Q4K, n: 248320, k };
    let (us_big, gbps_big) = bench(&dev, &big, 5, 43);
    report(&big, us_big, gbps_big);

    // (d) other formats at the small shape.
    for scheme in [KQuantScheme::Q5K, KQuantScheme::Q6K] {
        let c = Case { label: "d_small", scheme, n: 16384, k };
        let (us, gbps) = bench(&dev, &c, 20, 47);
        report(&c, us, gbps);
    }

    // (e) q8_0 at the small shape — SAME sdot4 inner loop but NO nibble
    // unpacking. If q8_0 reaches multiples of q4k's 23 GB/s, the q4k/q5k/q6k
    // bottleneck is the nibble-unpack ALU chain, not DRAM or launch overhead.
    let q80_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    };
    {
        let n = 16384usize;
        let mut sseed: u64 = 51;
        let mut nxt = move || {
            sseed ^= sseed << 13;
            sseed ^= sseed >> 17;
            sseed ^= sseed << 5;
            sseed
        };
        let vals: Vec<f32> = (0..n * k).map(|_| ((nxt() & 0xffff) as f32 / 32768.0) - 1.0).collect();
        let w_bytes = grim_quant::quant_q80(&vals).expect("quant_q80");
        let alloc = dev.allocator_handle();
        let w = MemoryOps::from_cpu_bytes(&dev, &w_bytes, &Shape::new(vec![n, k]), q80_dtype)
            .expect("upload q80");
        let w_rocm = w.as_any().downcast_ref::<RocmStorage>().expect("rocm");
        let a: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let a_st = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![1, k]), DType::F32).unwrap();
        let q81_bytes = (k / 32) * 36;
        let act_q81 = RocmStorage::alloc_gpu(&Shape::new(vec![q81_bytes]), DType { arith: ArithType::U8, storage: Storage::Native }, &alloc, 0).unwrap();
        let out = RocmStorage::alloc_gpu(&Shape::new(vec![n]), DType::F32, &alloc, 0).unwrap();
        for _ in 0..3 {
            dev.linear_decode_into(a_st.as_ref(), w_rocm, &out, &act_q81).unwrap();
        }
        dev.synchronize();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            dev.linear_decode_into(a_st.as_ref(), w_rocm, &out, &act_q81).unwrap();
        }
        dev.synchronize();
        let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        let gbps = (n as f64) * (k as f64 / 32.0) * 34.0 / 1e9 / (us * 1e-6);
        eprintln!("[floor-probe] e_small_q80_no_unpack        scheme=Q80 n={n:<7} per_launch={us:>8.1} us  weight_GBps={gbps:>7.1}");
    }

    // (f) q4k with GRIM_DOT4_FAST=1 — the word-wide-mask unpack variant.
    // Expected: multiple-x the stock 23 GB/s if the unpack chain was the
    // bottleneck; identical output mapping is gated by dot_gemv_parity.
    {
        unsafe { std::env::set_var("GRIM_DOT4_FAST", "1"); }
        let fast = Case { label: "f_small_q4k_FAST", scheme: KQuantScheme::Q4K, n: 16384, k };
        let (us, gbps) = bench(&dev, &fast, 20, 41);
        report(&fast, us, gbps);
        unsafe { std::env::set_var("GRIM_DOT4_FAST", "0"); }
    }

    // Verdict inputs, not a hard gate: this probe is evidence.
    let small_gb = 16384.0f64 * (k as f64 / 256.0) * 144.0 / 1e9;
    let big_gb = 248320.0f64 * (k as f64 / 256.0) * 144.0 / 1e9;
    eprintln!(
        "[floor-probe] verdict inputs: small moves {small_gb:.3} GB in {us:.0} us; \
         big moves {big_gb:.3} GB in {us_big:.0} us; big is {:.1}x the bytes for {:.1}x the time",
        big_gb / small_gb,
        us_big / us
    );
}
