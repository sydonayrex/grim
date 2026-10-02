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

    // (g) WhiteCrow W4A4 (OSTQuant u4 x u4, native v_dot8) at the small
    // shape: same GEMM as (a) but weights in the 4-bit OSTQuant layout
    // (0.5234 B/elem incl. scales+zeros) and activations quantized to u4 on
    // device. If this sustains q8_0-class GB/s, WhiteCrow is the decode
    // kernel candidate and the q4k dot4 kernel is the problem.
    {
        let n = 16384usize;
        let n_groups = k / 128;
        let words_per_col = k / 8;
        let alloc = dev.allocator_handle();
        let mut sseed: u64 = 61;
        let mut nxt = move || {
            sseed ^= sseed << 13;
            sseed ^= sseed >> 17;
            sseed ^= sseed << 5;
            sseed
        };
        let b_qw: Vec<u32> = (0..n * words_per_col).map(|_| nxt() as u32).collect();
        let b_sc: Vec<u16> = (0..n * n_groups).map(|_| (nxt() & 0x3fff) as u16 | 0x3800).collect(); // bf16 ~[0.5,1.0)
        let b_zr: Vec<u8> = (0..n * n_groups).map(|_| (nxt() & 0x0f) as u8).collect();
        let qw_bytes: Vec<u8> = b_qw.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sc_bytes: Vec<u8> = b_sc.iter().flat_map(|w| w.to_le_bytes()).collect();
        let b_qw_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &qw_bytes, &Shape::new(vec![n, words_per_col]),
            DType { arith: ArithType::U32, storage: Storage::Native }).unwrap();
        let b_sc_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &sc_bytes, &Shape::new(vec![n, n_groups]),
            DType { arith: ArithType::BF16, storage: Storage::Native }).unwrap();
        let b_zr_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &b_zr, &Shape::new(vec![n, n_groups]),
            DType { arith: ArithType::U8, storage: Storage::Native }).unwrap();
        let a: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let a_st = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![1, k]), DType::F32).unwrap();
        let out = RocmStorage::alloc_gpu(&Shape::new(vec![n]), DType::F32, &alloc, 0).unwrap();
        let a_rocm = a_st.as_any().downcast_ref::<RocmStorage>().unwrap();
        let qw_rocm = b_qw_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        let sc_rocm = b_sc_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        let zr_rocm = b_zr_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        // Warmup (JIT + u4 scratch alloc).
        for _ in 0..3 {
            dev.launch_w4a4_ostquant_gemv(a_rocm, qw_rocm, sc_rocm, zr_rocm, &out, 1, n, k).unwrap();
        }
        dev.synchronize();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            dev.launch_w4a4_ostquant_gemv(a_rocm, qw_rocm, sc_rocm, zr_rocm, &out, 1, n, k).unwrap();
        }
        dev.synchronize();
        let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        let gb = n as f64 * k as f64 * (0.5 + 2.0/128.0 + 1.0/128.0) / 1e9;
        let gbps = gb / (us * 1e-6);
        eprintln!("[floor-probe] g_small_W4A4_whitecrow       scheme=U4  n={n:<7} per_launch={us:>8.1} us  weight_GBps={gbps:>7.1}");
    }

    // (h) TreePie GEMV (5.0 bpw, V_DOT2_F32_F16) — packed in-register decode,
    // f16 activations. Weight bytes: 5 x i32 per 32 values = 0.625 B/elem.
    {
        let n = 16384usize;
        let alloc = dev.allocator_handle();
        let mut sseed: u64 = 71;
        let mut nxt = move || {
            sseed ^= sseed << 13;
            sseed ^= sseed >> 17;
            sseed ^= sseed << 5;
            sseed
        };
        let vals: Vec<f32> = (0..n * k).map(|_| ((nxt() & 0xffff) as f32 / 32768.0) - 1.0).collect();
        let packed_u32 = grim_quant::tree_pie::pack_tree_pie(&vals);
        let packed_bytes: Vec<u8> = packed_u32.iter().flat_map(|w| w.to_le_bytes()).collect();
        let b_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &packed_bytes,
            &Shape::new(vec![packed_bytes.len()]),
            DType { arith: ArithType::U32, storage: Storage::Native }).unwrap();
        let a: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let a_f16_bytes: Vec<u8> = a.iter()
            .map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .flatten().collect();
        let a_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &a_f16_bytes,
            &Shape::new(vec![a_f16_bytes.len()]),
            DType { arith: ArithType::F16, storage: Storage::Native }).unwrap();
        let out = RocmStorage::alloc_gpu(&Shape::new(vec![n]), DType::F32, &alloc, 0).unwrap();
        let a_rocm = a_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        let b_rocm = b_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        for _ in 0..3 {
            dev.launch_tree_pie_gemv(a_rocm, b_rocm, &out, n, k).unwrap();
        }
        dev.synchronize();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            dev.launch_tree_pie_gemv(a_rocm, b_rocm, &out, n, k).unwrap();
        }
        dev.synchronize();
        let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        let gb = n as f64 * k as f64 * 0.625 / 1e9;
        let gbps = gb / (us * 1e-6);
        eprintln!("[floor-probe] h_small_TREAPIE_dot2         scheme=TP  n={n:<7} per_launch={us:>8.1} us  weight_GBps={gbps:>7.1}");
    }

    // (i) WhiteRaven FP8 E4M3 WMMA GEMM (RDNA4 V_WMMA_FP8) — 1.0 B/elem.
    // M=1 needs the A operand padded to a 16-row tile (launcher contract);
    // N=16384 is a whole 16-tile. m=1 output row only is stored.
    {
        let n = 16384usize;
        let alloc = dev.allocator_handle();
        let mut sseed: u64 = 73;
        let mut nxt = move || {
            sseed ^= sseed << 13;
            sseed ^= sseed >> 17;
            sseed ^= sseed << 5;
            sseed
        };
        let b_fp8: Vec<u8> = (0..n * k).map(|_| (nxt() & 0xff) as u8).collect();
        let a_row: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let mut a_fp8: Vec<u8> = vec![0u8; 16 * k]; // 16-row pad, row 0 = real
        for (i, &v) in a_row.iter().enumerate() {
            a_fp8[i] = grim_quant::quant_fp8(&[v]).unwrap()[0];
        }
        let a_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &a_fp8, &Shape::new(vec![16, k]),
            DType { arith: ArithType::U8, storage: Storage::Native }).unwrap();
        let b_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &b_fp8, &Shape::new(vec![n, k]),
            DType { arith: ArithType::U8, storage: Storage::Native }).unwrap();
        let out = RocmStorage::alloc_gpu(&Shape::new(vec![1, n]), DType::F32, &alloc, 0).unwrap();
        let a_rocm = a_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        let b_rocm = b_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        for _ in 0..3 {
            dev.launch_wmma_gemm_fp8_e4m3_for_ab(a_rocm, b_rocm, &out, 1, n, k).unwrap();
        }
        dev.synchronize();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            dev.launch_wmma_gemm_fp8_e4m3_for_ab(a_rocm, b_rocm, &out, 1, n, k).unwrap();
        }
        dev.synchronize();
        let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        let gb = n as f64 * k as f64 * 1.0 / 1e9;
        let gbps = gb / (us * 1e-6);
        eprintln!("[floor-probe] i_small_WHITERAVEN_fp8_wmma  scheme=FP8 n={n:<7} per_launch={us:>8.1} us  weight_GBps={gbps:>7.1}");
    }

    // (j) WhiteRaven at M=16 — its native tile. Prefill-relevant: how many
    // weight bytes/sec when the WMMA fragments are full.
    {
        let n = 16384usize;
        let m = 16usize;
        let alloc = dev.allocator_handle();
        let mut sseed: u64 = 79;
        let mut nxt = move || {
            sseed ^= sseed << 13;
            sseed ^= sseed >> 17;
            sseed ^= sseed << 5;
            sseed
        };
        let b_fp8: Vec<u8> = (0..n * k).map(|_| (nxt() & 0xff) as u8).collect();
        let a_fp8: Vec<u8> = (0..m * k).map(|_| (nxt() & 0xff) as u8).collect();
        let a_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &a_fp8, &Shape::new(vec![m, k]),
            DType { arith: ArithType::U8, storage: Storage::Native }).unwrap();
        let b_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &b_fp8, &Shape::new(vec![n, k]),
            DType { arith: ArithType::U8, storage: Storage::Native }).unwrap();
        let out = RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &alloc, 0).unwrap();
        let a_rocm = a_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        let b_rocm = b_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        for _ in 0..3 {
            dev.launch_wmma_gemm_fp8_e4m3_for_ab(a_rocm, b_rocm, &out, m, n, k).unwrap();
        }
        dev.synchronize();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            dev.launch_wmma_gemm_fp8_e4m3_for_ab(a_rocm, b_rocm, &out, m, n, k).unwrap();
        }
        dev.synchronize();
        let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        let gb = n as f64 * k as f64 * 1.0 / 1e9;
        let gbps = gb / (us * 1e-6);
        eprintln!("[floor-probe] j_m16_WHITERAVEN_fp8_wmma    scheme=FP8 n={n:<7} per_launch={us:>8.1} us  weight_GBps={gbps:>7.1}");
    }

    // (k) WhiteRaven-BLOCKED FP8 E4M3 WMMA (16x16-blocked B, ldm=16): same
    // 1.0 B/elem as (i), but each fragment load is one contiguous 256B tile
    // instead of 16 K-strided 16B segments. Same bytes moved, so the GB/s
    // delta against (i) isolates the access pattern. Correctness of the
    // layout is gated by whiteraven_blocked_parity (bit-exact vs (i)).
    {
        let n = 16384usize;
        let alloc = dev.allocator_handle();
        let mut sseed: u64 = 83;
        let mut nxt = move || {
            sseed ^= sseed << 13;
            sseed ^= sseed >> 17;
            sseed ^= sseed << 5;
            sseed
        };
        let b_fp8: Vec<u8> = (0..n * k).map(|_| (nxt() & 0xff) as u8).collect();
        let b_blocked = grim_quant::block_fp8_16x16(&b_fp8, n, k).expect("block_fp8_16x16");
        let a_row: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let mut a_fp8: Vec<u8> = vec![0u8; 16 * k]; // 16-row pad, row 0 = real
        for (i, &v) in a_row.iter().enumerate() {
            a_fp8[i] = grim_quant::quant_fp8(&[v]).unwrap()[0];
        }
        let a_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &a_fp8, &Shape::new(vec![16, k]),
            DType { arith: ArithType::U8, storage: Storage::Native }).unwrap();
        let b_dev = grim_tensor::MemoryOps::from_cpu_bytes(&dev, &b_blocked, &Shape::new(vec![n * k]),
            DType { arith: ArithType::U8, storage: Storage::Native }).unwrap();
        let out = RocmStorage::alloc_gpu(&Shape::new(vec![1, n]), DType::F32, &alloc, 0).unwrap();
        let a_rocm = a_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        let b_rocm = b_dev.as_any().downcast_ref::<RocmStorage>().unwrap();
        for _ in 0..3 {
            dev.launch_wmma_gemm_fp8_e4m3_blocked(a_rocm, b_rocm, &out, 1, n, k).unwrap();
        }
        dev.synchronize();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            dev.launch_wmma_gemm_fp8_e4m3_blocked(a_rocm, b_rocm, &out, 1, n, k).unwrap();
        }
        dev.synchronize();
        let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        let gb = n as f64 * k as f64 * 1.0 / 1e9;
        let gbps = gb / (us * 1e-6);
        eprintln!("[floor-probe] k_blocked_WHITERAVEN_fp8    scheme=FP8b n={n:<7} per_launch={us:>8.1} us  weight_GBps={gbps:>7.1}");
    }

    // Verdict inputs, not a hard gate: this probe is evidence.
    let small_gb = 16384.0f64 * (k as f64 / 256.0) * 144.0 / 1e9;    let big_gb = 248320.0f64 * (k as f64 / 256.0) * 144.0 / 1e9;
    eprintln!(
        "[floor-probe] verdict inputs: small moves {small_gb:.3} GB in {us:.0} us; \
         big moves {big_gb:.3} GB in {us_big:.0} us; big is {:.1}x the bytes for {:.1}x the time",
        big_gb / small_gb,
        us_big / us
    );
}
