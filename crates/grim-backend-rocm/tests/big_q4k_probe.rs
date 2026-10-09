//! Full-C parity probe for the 64x64x32 big-tile Q4_K kernel.
//!
//! HISTORY: this file began life as an LDS dump probe for a dump build of the
//! kernel and ended up finding the real defect: `grim_big_deq_q4k`'s
//! high-nibble path computed `byte >> 4` on a full 32-bit word with NO
//! `& 0x0F` mask, so w >= 32 produced q up to 2^28 and f16 inf on store. The
//! low-nibble path masked, which is why K slice 0 was exact and every later
//! slice was garbage ("exact at one K step, inf from step 2"). The intermediates
//! dump (d/dmin/sc/qs bytes all correct; q decoding to 2^28/2^20/2^12) is what
//! convicted the nibble, after two rounds of stage-dump conclusions that were
//! artifacts of dump placement.
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TESTS=1 HIP_VISIBLE_DEVICES=1 cargo test \
//!   -p grim-backend-rocm --test big_q4k_probe -- --ignored --nocapture
use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage};

fn gpu_device() -> Option<RocmDevice> {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        return None;
    }
    Some(RocmDevice::try_new(0).unwrap())
}

/// Full-C parity through the direct launcher. K must be a multiple of 256
/// (row_bytes = K/256*144; sub-256 K is outside the kernel's contract and is
/// excluded by the production dispatch gate). m=37 exercises the masked
/// partial-tile path.
#[test]
#[ignore]
fn big_q4k_full_c_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TESTS=1");
        return;
    };
    for (m, n, k) in [(64usize, 64, 256), (37, 64, 256), (64, 64, 2048)] {
        let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
        let b_packed = grim_quant::quant_q4k(&b_host).unwrap();
        let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32).unwrap();
        let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q4K) };
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype).unwrap();
        let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n]), DType::F32).unwrap();
        let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
        let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
        let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();
        dev.launch_wmma_big_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
        let got = out.to_cpu_vec_f32().unwrap();

        // Host oracle: dequantize (column-major N x K, blocks of 256 weights)
        // then matmul in f32.
        let ndeq = grim_quant::dequant_q4k(&b_packed, k * n).unwrap();
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
        eprintln!("[big-q4k-parity] m={m} n={n} k={k} max_diff={max_diff:.6}");
        assert!(
            max_diff < 0.35,
            "m={m} n={n} k={k}: big-tile kernel diverges from the host oracle (max_diff={max_diff})"
        );
    }
}

/// Big-tile (64x64) vs 16-row WMMA kernel at prefill shapes. Informational:
/// prints ms per launch over a fixed iteration count on the same tensors.
#[test]
#[ignore]
fn big_q4k_prefill_bench_vs_16row() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TESTS=1");
        return;
    };
    let (m, n, k) = (4056usize, 4096usize, 4096usize);
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
    let b_packed = grim_quant::quant_q4k(&b_host).unwrap();
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q4K) };
    let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype).unwrap();
    let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n]), DType::F32).unwrap();
    let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
    let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
    let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();

    let iters = 10usize;
    dev.launch_wmma_fused_dequant_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        dev.launch_wmma_fused_dequant_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    }
    dev.synchronize();
    eprintln!("[big-q4k-bench] 16row: {:.2} ms/launch at m={m} n={n} k={k}",
              t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);

    dev.launch_wmma_big_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        dev.launch_wmma_big_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    }
    dev.synchronize();
    eprintln!("[big-q4k-bench] big64: {:.2} ms/launch at m={m} n={n} k={k}",
              t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
}

/// Full-C parity for the 128x64 tile through the direct launcher, then the
/// big64-vs-big128 bench at the prefill shape.
#[test]
#[ignore]
fn big128_q4k_parity_and_bench() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TESTS=1");
        return;
    };
    for (m, n, k) in [(128usize, 64, 256), (100, 64, 256), (128, 1024, 4096)] {
        let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
        let b_packed = grim_quant::quant_q4k(&b_host).unwrap();
        let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32).unwrap();
        let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q4K) };
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype).unwrap();
        let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n]), DType::F32).unwrap();
        let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
        let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
        let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();
        dev.launch_wmma_big128_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
        let got = out.to_cpu_vec_f32().unwrap();
        let ndeq = grim_quant::dequant_q4k(&b_packed, k * n).unwrap();
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
        eprintln!("[big128-parity] m={m} n={n} k={k} max_diff={max_diff:.6}");
        // per-16x16-fragment diff map: which quadrants/frags are wrong
        for bi in 0..m / 16 {
            let mut row = String::new();
            for bj in 0..n / 16 {
                let mut md = 0.0f32;
                for i in 0..16 {
                    for j in 0..16 {
                        let (r, c) = (bi * 16 + i, bj * 16 + j);
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            acc += a_host[r * k + kk] * ndeq[c * k + kk];
                        }
                        md = md.max((got[r * n + c] - acc).abs());
                    }
                }
                row.push_str(&format!("{:9.2}", md));
            }
            eprintln!("[big128-map] m={m} fragrow {bi}: {row}");
        }
        // f16-accumulate error grows ~sqrt(k): 0.486 at k=4096 is the same
        // class the 64x64 tile shows at this shape.
        let tol = 0.02 * (k as f32).sqrt();
        assert!(
            max_diff < tol,
            "m={m} n={n} k={k}: 128x64 tile diverges from the host oracle (max_diff={max_diff}, tol={tol})"
        );
    }

    // bench at the production prefill shape
    let (m, n, k) = (4056usize, 4096usize, 4096usize);
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
    let b_packed = grim_quant::quant_q4k(&b_host).unwrap();
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q4K) };
    let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype).unwrap();
    let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n]), DType::F32).unwrap();
    let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
    let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
    let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();
    let iters = 10usize;
    dev.launch_wmma_big_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        dev.launch_wmma_big_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    }
    dev.synchronize();
    eprintln!("[big128-bench] big64: {:.2} ms/launch at m={m} n={n} k={k}",
              t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
    dev.launch_wmma_big128_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        dev.launch_wmma_big128_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    }
    dev.synchronize();
    eprintln!("[big128-bench] big128: {:.2} ms/launch at m={m} n={n} k={k}",
              t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
}
