//! WhiteCrow dispatch vs the incumbent 4-bit path: Q4_K dot4 GEMV.
//!
//! WhiteCrow (W4A4 OSTQuant u4x4, group-128, `V_DOT8_I32_IU4`) had a kernel, a
//! dispatch arm, and parity -- but no measured numbers. This is that
//! measurement, through `quantized_matmul` (the path a model takes), at
//! decode (m=1) and small prefill (m=16), both F32 activations (the dispatch
//! quantizes acts on-device: u4-group128 for WhiteCrow, Q8_1 for Q4K).
//!
//! DOMAIN RESTRICTION (load-bearing): the kernel computes
//! `d_a*d_b*(iacc - z_b*sum_qa)` -- a B zero-point but NO A zero-point --
//! and the act quantizer clamps to unsigned codes, so NEGATIVE activations
//! decode as zero. The format is correct only for non-negative activations
//! (post-ReLU/SiLU-type outputs); signed inputs (norm outputs) are silently
//! wrong, and the dispatch arm does NOT check. Fixtures below are
//! abs-valued to stay in-domain (same reason dot_gemv_parity uses ABS);
//! the missing sign check is a separate safety gap, not bench scope.
//!
//! Gate: must not lose badly to Q4_K at the same job. A 4-bit format slower
//! than the incumbent 4-bit path is a regression disguised as a feature.
//! Correctness smoke (cosine vs host-dequantized blob, loose: the dispatch
//! quantizes A to u4 while the reference below does not) guards the framing
//! and prologue wiring that kernel-level parity does not cover.

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
        // NOTE: (x / 2^24) - 1.0 with x in [0, 2^24) yields [-1, 0) --
        // one-sided. The * 2.0 below is load-bearing, not decorative:
        // without it every draw is negative, u4 act codes are all zero,
        // and the bench measures 0-vs-0 vacously (caught by the warmed()
        // gate below on the first run).
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    /// Draw with warmup: raw xorshift output is measurably biased for the
    /// first draws out of a small seed (amean=-0.51 observed without this),
    /// and a biased fixture makes this gate vacuous -- the u4 act quantizer
    /// clamps negatives to code 0, so an all-negative activation row yields
    /// exactly 0.0 output and the test passes 0-vs-0. Same trap
    /// dot_gemv_parity documents with its ABS fixture.
    fn warmed(seed: u64) -> impl FnMut() -> f32 {
        let mut rng = Lcg(seed);
        for _ in 0..1024 {
            rng.f32();
        }
        let mut f = move || rng.f32();
        // Sanity-gate the fixture itself: symmetric, both signs present.
        // A biased RNG must fail HERE (naming the fixture), not downstream
        // as a mysterious all-zero output.
        let probe: Vec<f32> = (0..512).map(|_| f()).collect();
        let mean = probe.iter().sum::<f32>() / probe.len() as f32;
        assert!(
            mean.abs() < 0.2,
            "LCG fixture biased (mean {mean:e}): warmup failed, fix the RNG not the test"
        );
        assert!(
            probe.iter().any(|&v| v > 0.1) && probe.iter().any(|&v| v < -0.1),
            "LCG fixture lacks both signs: vacuous for u4 act quant"
        );
        f
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
    // Weight-read traffic in equivalent dense bytes (codes only).
    Ok((us, (n * k) as f64 / 2e9 / (us * 1e-6)))
}

fn read_f32(t: &Box<dyn grim_tensor::BackendStorage>) -> Vec<f32> {
    grim_backend_rocm::as_rocm(t.as_ref())
        .expect("rocm storage")
        .copy_to_host()
        .expect("d2h")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn whitecrow_dispatch_keeps_pace_with_q4k() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    if !dev.gcn_arch().starts_with("gfx12") {
        return Ok(());
    }

    for (m, n, k, iters) in [(1usize, 4096usize, 4096usize, 50usize), (16, 4096, 4096, 20)] {
        // abs-valued: WhiteCrow's unsigned act codes are only valid for
        // non-negative inputs (see header). Symmetric data would bench
        // 0-vs-0 vacuously AND mislead about the format's domain.
        let mut rng = Lcg::warmed(0xC2049 ^ (m as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng().abs()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng().abs() * 0.5).collect();

        // WhiteCrow: group-128 W4A4, length-prefixed triple-stream blob.
        let (qw, sc, zr) =
            grim_quant::quant_ostquant_w4_group128(&w, n, k).map_err(|e| format!("crow: {e}"))?;
        let mut blob = Vec::with_capacity(24 + qw.len() + sc.len() + zr.len());
        blob.extend_from_slice(&(qw.len() as u64).to_le_bytes());
        blob.extend_from_slice(&qw);
        blob.extend_from_slice(&(sc.len() as u64).to_le_bytes());
        blob.extend_from_slice(&sc);
        blob.extend_from_slice(&(zr.len() as u64).to_le_bytes());
        blob.extend_from_slice(&zr);
        let w_crow = MemoryOps::from_cpu_bytes(
            &dev,
            &blob,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::W4A4OstQuant(grim_tensor::dtype::OstQuantConfig {
                    group_size: 128,
                }),
            },
        )?;

        // Q4_K incumbent: per-column super-blocks, the same dispatch shape.
        let row_bytes = (k / 256) * 144;
        let mut bq = vec![0u8; n * row_bytes];
        for col in 0..n {
            let packed = grim_quant::quant_q4k(&w[col * k..(col + 1) * k])
                .map_err(|e| format!("q4k: {e}"))?;
            bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
        }
        let w_q4k = MemoryOps::from_cpu_bytes(
            &dev,
            &bq,
            &Shape::new(vec![n, k]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(grim_tensor::KQuantScheme::Q4K),
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

        // Correctness smoke on the small case only (one extra dispatch, not
        // timed): host-dequantized blob x exact F32 activations. Loose on
        // purpose -- the dispatch quantizes A to u4, the reference does not,
        // so this guards framing/prologue wiring (O(1) bugs), not quantizer
        // equality.
        if m == 1 {
            let out_shape = Shape::new(vec![m, n]);
            let (c_t, _) = dev.quantized_matmul(
                x.as_ref(),
                w_crow.as_ref(),
                &[],
                QuantFormat::W4A4OstQuant,
                &out_shape,
            )?;
            dev.synchronize();
            let got = read_f32(&c_t);
            let wq = grim_quant::dequant_ostquant_w4a4(&qw, &sc, &zr, &[n, k], 128)
                .map_err(|e| format!("deq: {e}"))?;
            let mut dot = 0.0f64;
            let (mut gg, mut xx) = (0.0f64, 0.0f64);
            for col in 0..n {
                let mut x = 0.0f32;
                for j in 0..k {
                    x += a[j] * wq[col * k + j];
                }
                dot += got[col] as f64 * x as f64;
                gg += (got[col] as f64) * (got[col] as f64);
                xx += (x as f64) * (x as f64);
            }
            let cosine = dot / (gg.sqrt() * xx.sqrt()).max(f64::MIN_POSITIVE);

            eprintln!(
                "[crow-bench] smoke cosine={cosine:.6} gg={gg:e} xx={xx:e} got0={:?} want0={:?}",
                &got[..4],
                {
                    let mut v = vec![0.0f32; 4];
                    for col in 0..4 {
                        let mut x = 0.0f32;
                        for j in 0..k {
                            x += a[j] * wq[col * k + j];
                        }
                        v[col] = x;
                    }
                    v
                }
            );
            assert!(
                cosine >= 0.99,
                "crow dispatch smoke failed (cosine {cosine:.4}): framing or act prologue miswired"
            );
        }

        // Q4K's dot4 path is gated on GRIM_DECODE_EXACT_GEMV=0 (default on
        // forces WMMA/scalar, ~5ms here). Peak-vs-peak means forcing it on:
        // comparing crow against a deliberately-crippled baseline would be a
        // hollow win. (The variable is process-global but this binary holds a
        // single test; WhiteRaven's bench does the same per-case.)
        unsafe {
            std::env::set_var("GRIM_DECODE_EXACT_GEMV", "0");
        }
        let (us_q4k, gb_q4k) = bench(&dev, &w_q4k, &x, m, n, k, QuantFormat::Q4K, iters)?;
        let (us_c, gb_c) = bench(
            &dev,
            &w_crow,
            &x,
            m,
            n,
            k,
            QuantFormat::W4A4OstQuant,
            iters,
        )?;

        println!("[crow-bench] m={m:<4} n={n} k={k}");
        println!("  q4k dot4    : {us_q4k:>9.1} us  {gb_q4k:>7.1} GB/s");
        println!(
            "  crow w4a4    : {us_c:>9.1} us  {gb_c:>7.1} GB/s   ({:.2}x)",
            us_q4k / us_c
        );

        assert!(
            us_c <= us_q4k * 1.5,
            "crow m={m} is {us_c:.1}us vs q4k {us_q4k:.1}us -- a new 4-bit \
             format must not be substantially slower than the incumbent"
        );
    }
    Ok(())
}
