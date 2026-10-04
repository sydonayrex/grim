//! GreyRaven dispatch vs the dense path in its instruction class: Raven FP8.
//!
//! Both run 16x16x32 tiles; GreyRaven skips half the MACs (2:4) and reads
//! 4.06 bpw instead of 8. The honest question is whether the pictured win
//! materializes through the real dispatch (frag loads + B transpose-gather
//! + scattered C writes are all overhead dense FP8 does not pay).
//!
//!
//! Measured (RX 9070 XT, gfx1201, n=k=4096, `grey_raven_dispatch_bench`,
//! WITH the B-prologue; pre-prologue numbers were 783/802 us):
//!   m=1 :  68.5 us, 245 GB/s -- 0.76x vs Raven dot4 GEMV (52-63 us).
//!         Near-parity with the purpose-built decode kernel despite
//!         M-padding 15/16 columns: the prologue + frag loads are cheap
//!         enough that the 2:4 halved traffic nearly compensates.
//!   m=16:  60.7 us, 276 GB/s -- 1.16x vs Raven row-major WMMA (70-100 us).
//!         A real win over dense in the same instruction class, from halved
//!         weight traffic (4.06 vs 8 bpw). Larger batches should widen it
//!         (more N-tiles amortize the same B-frag).
//!
//! Earlier this doc recorded 17x over Raven's fused-dequant MFMA arm; that
//! arm turned out to be a scalar fallback (per-element powf dequant, ~13.7ms
//! at m=16) and has since been rewired to the same row-major WMMA Raven now
//! runs. The comparison above is against the fixed baseline.

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

#[test]
fn grey_dispatch_beats_dense_fp8_at_prefill() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    if !dev.gcn_arch().starts_with("gfx12") {
        return Ok(());
    }

    for (m, n, k, iters) in [(1usize, 4096usize, 4096usize, 30usize), (16, 4096, 4096, 10)] {
        let mut rng = Lcg(0x6E67 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32()).collect();

        // Raven dense FP8: bare E4M3 codes, the same-bytes-different-order
        // baseline. m=1 takes the dot4 GEMV, m>1 the MFMA fused-dequant.
        let codes: Vec<u8> = w.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
        let w_raven = MemoryOps::from_cpu_bytes(
            &dev,
            &codes,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8),
            },
        )?;

        // GreyRaven-HW: coupled patterns + HW pack, tag 673.
        let pats = grim_quant::grey_raven::coupled_patterns_2_4(&w, n, k)
            .map_err(|e| format!("couple: {e}"))?;
        let blob = grim_quant::grey_raven::pack_grey_raven_hw(&w, n, k, &pats)
            .map_err(|e| format!("pack: {e}"))?;
        let w_grey = MemoryOps::from_cpu_bytes(
            &dev,
            &blob,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Block(grim_tensor::BlockDtype::Fp8Sparse24Hw),
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

        let (us_r, gb_r) = bench(&dev, &w_raven, &x, m, n, k, QuantFormat::Fp8, iters)?;
        let (us_g, gb_g) = bench(
            &dev,
            &w_grey,
            &x,
            m,
            n,
            k,
            QuantFormat::Fp8Sparse24Hw,
            iters,
        )?;

        println!("[grey-bench] m={m:<4} n={n} k={k}");
        println!("  raven dense  : {us_r:>9.1} us  {gb_r:>7.1} GB/s");
        println!(
            "  grey 2:4     : {us_g:>9.1} us  {gb_g:>7.1} GB/s   ({:.2}x)",
            us_r / us_g
        );

        if m == 1 {
            // Decode: structurally disadvantaged (M-padding + transpose
            // gather for 1 live column). Recorded, not gated.
        } else {
            assert!(
                us_g <= us_r,
                "grey prefill m={m} is {us_g:.1}us vs raven {us_r:.1}us -- \
                 2:4 must at least match dense in its own instruction class"
            );
        }
    }
    Ok(())
}
