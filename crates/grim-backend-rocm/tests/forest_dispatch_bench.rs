//! ForestRaven dispatch vs the same-class path: Q8_0 int8 GEMV.
//!
//! ForestRaven is a new format with no prior path to beat, so the honest
//! comparison is same-instruction (V_DOT4_I32_IU8), same shapes, both through
//! `quantized_matmul`. ForestRaven's inner loop is simpler (one fp32 scale
//! per row, no per-32-block scale traffic), so parity is the floor and a win
//! is plausible. The gate below enforces "must not lose badly" -- a new
//! format that is slower than the incumbent at the same job is a regression
//! disguised as a feature.

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
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) - 1.0
    }
}

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
    // Weight-read traffic: codes only (scales are negligible at 4n bytes).
    Ok((us, (n * k) as f64 / 1e9 / (us * 1e-6)))
}

#[test]
fn forest_dispatch_keeps_pace_with_q80() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let arch = dev.gcn_arch().to_string();
    if !(arch.starts_with("gfx11") || arch.starts_with("gfx12")) {
        return Ok(());
    }

    for (m, n, k, iters) in [(1usize, 4096usize, 4096usize, 50usize), (16, 4096, 4096, 20)] {
        let mut rng = Lcg(0xF02E57 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

        // Q8_0 baseline: per-32-block fp16 scales, the incumbent int8 path.
        let q80 = grim_quant::quant_q80(&w).map_err(|e| format!("q80: {e}"))?;
        let w_q80 = MemoryOps::from_cpu_bytes(
            &dev,
            &q80,
            &Shape::new(vec![n * k / 32 * 34]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(grim_tensor::KQuantScheme::Q80),
            },
        )?;

        // ForestRaven: per-row fp32 scales, framed blob.
        let (codes, scales) =
            grim_quant::quant_forest_per_channel(&w, n, k).map_err(|e| format!("forest: {e}"))?;
        let mut blob = Vec::with_capacity(16 + codes.len() + scales.len());
        blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&codes);
        blob.extend_from_slice(&(scales.len() as u64).to_le_bytes());
        blob.extend_from_slice(&scales);
        let w_forest = MemoryOps::from_cpu_bytes(
            &dev,
            &blob,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Block(grim_tensor::BlockDtype::Int8PerChannel),
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

        let (us_q80, gb_q80) = bench(&dev, &w_q80, &x, m, n, k, QuantFormat::Q8_0, iters)?;
        let (us_f, gb_f) = bench(
            &dev,
            &w_forest,
            &x,
            m,
            n,
            k,
            QuantFormat::Int8PerChannel,
            iters,
        )?;

        println!("[forest-bench] m={m:<4} n={n} k={k}");
        println!("  q8_0 dot4   : {us_q80:>9.1} us  {gb_q80:>7.1} GB/s");
        println!(
            "  forest dot4 : {us_f:>9.1} us  {gb_f:>7.1} GB/s   ({:.2}x)",
            us_q80 / us_f
        );

        assert!(
            us_f <= us_q80 * 1.5,
            "forest m={m} is {us_f:.1}us vs q8_0 {us_q80:.1}us -- a new int8 \
             format must not be substantially slower than the incumbent"
        );
    }
    Ok(())
}
