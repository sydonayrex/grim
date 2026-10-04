//! GreyRaven dispatch vs the dense path in its instruction class: Raven FP8.
//!
//! Both run 16x16x32 tiles; GreyRaven skips half the MACs (2:4) and reads
//! 4.06 bpw instead of 8. The honest question is whether the pictured win
//! materializes through the real dispatch (frag loads + B transpose-gather
//! + scattered C writes are all overhead dense FP8 does not pay).
//!
//! Two caveats, both recorded rather than hidden:
//!
//! 1. Raven's m=16 path (fused-dequant MFMA) measures ~13.7ms here --
//!    orders of magnitude slower than the arithmetic justifies, which smells
//!    like a separate pre-existing pathology in that arm (host-side work per
//!    call? a missing fast path?). The 17x below is real (same rig, same
//!    shapes) but flatters grey: beating a broken baseline is not the same
//!    as being fast. Grey's own absolute numbers (20.9 GB/s at m=16) are far
//!    from HBM limits for the opposite reason -- see (2).
//!
//! 2. Grey's B fragment (512 B transpose-gather + E4M3 encode per window)
//!    is recomputed per N-tile, but it depends only on (M-tile, window):
//!    256x redundant work on the benchmark shape. A B-prologue kernel
//!    (quantize once per (M-tile, window) into fragment order, consume from
//!    there -- the WhiteRaven act-prologue playbook) removes it. Until then
//!    grey is correct and competitive, not optimal.
//!
//! Gate: must not lose to dense at prefill (the reason 2:4 exists). Decode
//! (m=1) reports without a hard gate -- the M-padding makes it structurally
//! disadvantaged, and the number is recorded so the trade-off is explicit
//! rather than hidden.

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
