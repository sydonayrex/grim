//! The blocked kernel is fast. Is the dispatch that reaches it faster than the
//! path it replaces?
//!
//! The bandwidth probe measures `launch_wmma_gemm_fp8_e4m3_blocked` directly
//! and reports 3.5x over the row-major WMMA. That is the wrong comparison for
//! a shipping decision, because the row-major WMMA is not what production runs:
//! at m=1 the fp8 decode arm goes to the dot4 GEMV, and at m>1 to the MFMA
//! fused-dequant GEMM. This measures `quantized_matmul` -- the path a model
//! actually takes -- against both of those.
//!
//! It exists because this exact question was answered wrong once: the kernel
//! was 3.5x faster while the dispatch around it was 2.5x SLOWER, because the
//! arm converted activations through the host (a D2H readback plus a fresh
//! allocation per call, ~200us) to save ~90us of kernel time. Nothing in the
//! kernel-level gates could see that. Hence the bar below is on the dispatch.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, DType, MemoryOps, QuantFormat, QuantOps, Shape, Storage};
use std::panic;
use std::time::Instant;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice")).ok()
}

struct Lcg(u64);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0  // [-1, 1): the *2.0 is load-bearing (one-sided draws make u4/int8 gates vacuous)
    }
}

/// us/call and weight-read GB/s for `iters` `quantized_matmul` calls.
fn bench(
    dev: &RocmDevice,
    w: &Box<dyn grim_tensor::BackendStorage>,
    x: &Box<dyn grim_tensor::BackendStorage>,
    m: usize,
    n: usize,
    k: usize,
    fmt: QuantFormat,
    iters: usize,
) -> TestResult<(f64, f64)> {
    let out_shape = Shape::new(vec![m, n]);
    for _ in 0..3 {
        dev.quantized_matmul(x.as_ref(), w.as_ref(), &[], fmt, &out_shape)?;
    }
    dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..iters {
        dev.quantized_matmul(x.as_ref(), w.as_ref(), &[], fmt, &out_shape)?;
    }
    dev.synchronize();
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    Ok((us, (n * k) as f64 / 1e9 / (us * 1e-6)))
}

struct Case {
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
}

#[test]
fn blocked_dispatch_beats_the_fp8_path_it_replaces() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    // Serialize: GRIM_DOT_GEMV is process-global and the two arms differ by it.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let cases = [
        Case {
            m: 1,
            n: 16384,
            k: 4096,
            iters: 50,
        }, // decode
        Case {
            m: 16,
            n: 16384,
            k: 4096,
            iters: 20,
        }, // prefill, native tile
        Case {
            m: 128,
            n: 16384,
            k: 4096,
            iters: 10,
        }, // prefill, compute-bound
    ];

    for c in cases {
        let Case { m, n, k, iters } = c;
        let mut rng = Lcg(0x5EED ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();
        let codes: Vec<u8> = w.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();

        // Row-major: bare E4M3 codes, n*k bytes, no scale prefix. This is the
        // dispatch contract (tag-669/convert emit bare; the launcher validates
        // 16-row-tile padding on the byte count). The legacy `quant_fp8`
        // 4-byte-prefixed layout is NOT accepted here -- it fails the tile
        // check by construction, which is preferable to silently shifting
        // every code by four.
        let rowmajor = codes.clone();
        let w_row = MemoryOps::from_cpu_bytes(
            &dev,
            &rowmajor,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8),
            },
        )?;
        let blocked = grim_quant::block_fp8_16x16(&codes, n, k)?;
        let w_blk = MemoryOps::from_cpu_bytes(
            &dev,
            &blocked,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::U8,
                storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16),
            },
        )?;
        let x = MemoryOps::from_cpu_bytes(
            &dev,
            &a.iter()
                .flat_map(|v| v.to_le_bytes().to_vec())
                .collect::<Vec<u8>>(),
            &Shape::new(vec![m, k]),
            DType::F32,
        )?;

        // m=1 routes to the dot4 GEMV, m>1 to the MFMA fused-dequant GEMM. The
        // blocked arm is the same kernel at every m.
        unsafe {
            std::env::set_var("GRIM_DOT_GEMV", if m == 1 { "1" } else { "0" });
        }
        let (us_row, gb_row) = bench(&dev, &w_row, &x, m, n, k, QuantFormat::Fp8, iters)?;
        let (us_blk, gb_blk) = bench(&dev, &w_blk, &x, m, n, k, QuantFormat::Fp8Blocked16, iters)?;

        println!("[wr-bench] m={m:<4} n={n} k={k}");
        println!("  row-major fp8 : {us_row:>9.1} us  {gb_row:>7.1} GB/s");
        println!(
            "  blocked  fp8  : {us_blk:>9.1} us  {gb_blk:>7.1} GB/s   ({:.2}x)",
            us_row / us_blk
        );

        // Decode: the dot4 GEMV is already DRAM-saturated, so the blocked WMMA
        // cannot beat it and must not lose badly. Its 16-row tile pads m=1 to 16
        // rows of activation, so parity is the realistic target, not a win.
        if m == 1 {
            assert!(
                us_blk <= us_row * 1.25,
                "blocked decode is {us_blk:.1}us vs {us_row:.1}us for the dot4 GEMV -- \
                 the dispatch must at least keep pace"
            );
        } else {
            // Prefill: the row-major arm is the fused-dequant MFMA GEMM. The
            // blocked layout is what makes this path viable at all.
            assert!(
                us_blk <= us_row * 0.5,
                "blocked prefill m={m} is {us_blk:.1}us vs {us_row:.1}us row-major -- \
                 expected the blocked layout to be a large win at m>1"
            );
        }
    }
    Ok(())
}
