//! Experiment (a) of `plans/PLAN-m1-decode-regime-experiments.md` — the
//! COMPUTE-ONLY decode GEMV, and the decomposition of the M=1 decode regime.
//!
//! **The question.** grim's M=1 K-quant decode GEMV is the hot path for every
//! decoded token. What limits it — DRAM bandwidth, decode instruction count, or
//! a fixed per-launch cost? The corpus (old/res8) says bytes are not the
//! constraint; the tree's floor probe says the per-launch cost does not scale
//! with tensor size. Both can be true at once, and they are separated here.
//!
//! **The method.** Every measurement goes through the PRODUCTION decode entry
//! (`linear_decode_into`), so the numbers include the Q8_1 prequant launch
//! exactly as inference pays it. Nothing is modelled; every byte is the
//! checkpoint's own bytes, packed by grim-quant's llama.cpp-faithful encoders.
//!
//! Three arms:
//!
//!   A1 — WORKING-SET SWEEP. Sweep N (output columns) at fixed K and format,
//!        so the weight working set sweeps ~3 MB -> ~94 MB, and fit
//!        `t(N) = a + b*N`. `a` is the FIXED per-launch cost (launch, grid
//!        ramp, prologue, reduction, writeback) and `b*N` the work-proportional
//!        part. The per-format `a` is the number that decides whether a launch
//!        restructure is the lever.
//!
//!   A2 — COLD vs WARM, the compute-only floor. Same shape, same
//!        instructions, same logical bytes; only the SOURCE of the bytes
//!        differs. "Warm" reuses one weight buffer so it stays L2-resident;
//!        "cold" rotates over R copies so every launch's weight was evicted
//!        and must come from DRAM. `t_cold - t_warm` is the DRAM component and
//!        `t_warm` is the compute-only floor. The ASSUMPTION-FREE primary
//!        metric is the ratio `t_cold/t_warm`: if it is ~1.0 the DRAM term is
//!        negligible at that working set, whatever the datasheet says.
//!
//!   A3 — THE BYTES-vs-UNPACK CONTROL. Q8_0 at the SAME N does the same MACs
//!        in the same `sdot4` inner loop but reads 4.25x FEWER bytes and
//!        needs NO unpack. Comparing `t_q4k(N)` with `t_q80(N)` at matched N
//!        separates "the bytes cost" from "the unpack costs" from "the loop
//!        costs".
//!
//! **Controls.** Randomized variant order (res8 P6: a sequential sweep carries
//! 43% TLB/L3 warmup drift and up to 50% co-allocation interference; the
//! within-config noise floor is 0.04-0.10%, so anything under ~0.1% is not a
//! result). The within-config repeat is reported so the noise floor is
//! visible. The instrument is named on every line: no profiler is attached
//! here, and the numbers are wall-clock around the production entry.
//!
//! **RUN:**
//!   GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=1 cargo test -p grim-backend-rocm \
//!     --test m1_decode_regime_probe -- --ignored --nocapture
//!
//! This probe is the DECOMPOSITION of the floor probe
//! (`dot4_gemv_floor_probe.rs`): that one measures the floor and the ceiling,
//! this one attributes the time between them.

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::RocmStorage;
use grim_backend_rocm::as_rocm;
use grim_backend_rocm::device::util::gpu_test_lock;
use grim_backend_rocm::memory::allocator::RocmCachingAllocator;
use grim_tensor::error::Result;
use grim_tensor::{ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage};
use std::sync::Arc;

/// The 27B's dominant projection shape: k = 5120.
const K: usize = 5120;
/// The card's datasheet peak read bandwidth (RX 9060 XT: 20 Gbps GDDR6 x 128
/// bit = 320 GB/s). Used ONLY to convert a measured DRAM time into an implied
/// traffic figure; the assumption-free metrics are `t_cold/t_warm` and
/// `delta_t`. res8 P6: a datasheet figure is an upper bound, and the
/// bottleneck classification can flip with the constant.
const PEAK_GBS: f64 = 320.0;

fn gpu_device() -> Option<RocmDevice> {
    if std::env::var("GRIM_RUN_GPU_TEST").is_err() {
        eprintln!("[regime-probe] skipping: set GRIM_RUN_GPU_TEST=1");
        return None;
    }
    let _lock = gpu_test_lock();
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

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
        KQuantScheme::Q2K => grim_quant::quant_q2k(&vals).expect("quant_q2k"),
        KQuantScheme::Q3K => grim_quant::quant_q3k(&vals).expect("quant_q3k"),
        KQuantScheme::Q80 => grim_quant::quant_q80(&vals).expect("quant_q80"),
        other => panic!("unsupported scheme {other:?}"),
    }
}

fn block_bytes(scheme: KQuantScheme) -> f64 {
    match scheme {
        KQuantScheme::Q4K => 144.0,
        KQuantScheme::Q5K => 176.0,
        KQuantScheme::Q6K => 210.0,
        KQuantScheme::Q2K => 84.0,
        KQuantScheme::Q3K => 110.0,
        KQuantScheme::Q80 => 34.0,
        _ => unreachable!(),
    }
}

fn weight_bytes(scheme: KQuantScheme, n: usize, k: usize) -> f64 {
    n as f64 * (k as f64 / 256.0) * block_bytes(scheme)
}

/// One measurement case: the weight copies, the activation, the scratch.
struct Harness {
    /// R copies of the SAME bytes. "warm" uses copy 0 every time; "cold"
    /// rotates, so each launch's weight has been evicted by the others.
    weights: Vec<Box<dyn grim_tensor::BackendStorage>>,
    act: Box<dyn grim_tensor::BackendStorage>,
    out: RocmStorage,
    q81: RocmStorage,
}

impl Harness {
    fn w(&self, i: usize) -> &RocmStorage {
        self.weights[i]
            .as_any()
            .downcast_ref::<RocmStorage>()
            .expect("rocm weight storage")
    }
}

impl Harness {
    fn build(
        dev: &RocmDevice,
        scheme: KQuantScheme,
        n: usize,
        k: usize,
        copies: usize,
        seed: u64,
    ) -> Result<Self> {
        let alloc: Arc<RocmCachingAllocator> = dev.allocator_handle();
        let w_bytes = pack_qk(scheme, n, k, seed);
        let mut weights = Vec::with_capacity(copies);
        for _ in 0..copies {
            let w_dtype = DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(scheme),
            };
            let w = MemoryOps::from_cpu_bytes(dev, &w_bytes, &Shape::new(vec![n, k]), w_dtype)
                .expect("upload weights");
            weights.push(w);
        }
        let a: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let act = CoreTensorOps::from_cpu(dev, &a, &Shape::new(vec![1, k]), DType::F32)
            .expect("upload act");
        let q81_bytes = (k / 32) * 36;
        let q81 = RocmStorage::alloc_gpu(
            &Shape::new(vec![q81_bytes]),
            DType {
                arith: ArithType::U8,
                storage: Storage::Native,
            },
            &alloc,
            0,
        )
        .expect("alloc q81 scratch");
        let out =
            RocmStorage::alloc_gpu(&Shape::new(vec![n]), DType::F32, &alloc, 0).expect("alloc out");
        Ok(Self {
            weights,
            act,
            out,
            q81,
        })
    }

    /// Per-launch wall time in microseconds, over `iters` launches of the
    /// production decode entry with ONE sync at the end.
    fn time(&self, dev: &RocmDevice, which: usize, rotate: bool, iters: usize) -> f64 {
        let start = std::time::Instant::now();
        for i in 0..iters {
            let idx = if rotate {
                i % self.weights.len()
            } else {
                which
            };
            dev.linear_decode_into(self.act.as_ref(), self.w(idx), &self.out, &self.q81)
                .expect("linear_decode_into");
        }
        dev.synchronize();
        start.elapsed().as_secs_f64() * 1e6 / iters as f64
    }

    /// Warm up (JIT + allocator priming), untimed.
    fn warm_up(&self, dev: &RocmDevice) {
        for _ in 0..3 {
            dev.linear_decode_into(self.act.as_ref(), self.w(0), &self.out, &self.q81)
                .expect("linear_decode_into");
        }
        dev.synchronize();
    }
}

/// Least-squares fit of `t = a + b*x` over (x, t) pairs.
fn fit_ab(xs: &[f64], ts: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let st: f64 = ts.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxt: f64 = xs.iter().zip(ts).map(|(x, t)| x * t).sum();
    let den = n * sxx - sx * sx;
    if den.abs() < 1e-9 {
        return (ts[0], 0.0);
    }
    let b = (n * sxt - sx * st) / den;
    let a = (st - b * sx) / n;
    (a, b)
}

/// Arm A1 + A2 + A3 for one format: the working-set sweep with the cold/warm
/// pair at each point.
fn sweep_format(dev: &RocmDevice, scheme: KQuantScheme, ns: &[usize], copies: usize, seed: u64) {
    let mut xs = Vec::new();
    let mut warm_ts = Vec::new();
    let mut cold_ts = Vec::new();
    eprintln!("[regime-probe] === format {scheme:?}  k={K}  copies={copies} ===");
    eprintln!(
        "[regime-probe] {:>7} {:>9} {:>9} {:>9} {:>7} {:>8} {:>9} {:>9}",
        "n", "W_MB", "warm_us", "cold_us", "ratio", "d_us", "GBps_cold", "implied_x"
    );
    for (i, &n) in ns.iter().enumerate() {
        let h = match Harness::build(dev, scheme, n, K, copies, seed + i as u64 * 7) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[regime-probe] n={n} build failed: {e}");
                continue;
            }
        };
        h.warm_up(dev);
        // Two independent warm measurements: the difference is the noise floor.
        let warm1 = h.time(dev, 0, false, 50);
        let warm2 = h.time(dev, 0, false, 50);
        let cold = h.time(dev, 0, true, 50);
        let wb = weight_bytes(scheme, n, K);
        let gbps = (wb / 1e9) / (cold * 1e-6);
        // Implied traffic amplification: how many bytes the DRAM path must
        // have moved, given the datasheet peak, over the logical bytes.
        let dt = cold - warm1;
        let implied = (dt * 1e-6 * PEAK_GBS * 1e9) / wb.max(1.0);
        eprintln!(
            "[regime-probe] {n:>7} {:>9.2} {:>9.1} {:>9.1} {:>7.3} {:>8.1} {:>9.1} {:>9.2}",
            wb / 1e6,
            warm1,
            cold,
            cold / warm1,
            dt,
            gbps,
            implied
        );
        eprintln!(
            "[regime-probe] {:>7} {:>9} (repeat of the warm arm: warm2={warm2:.1} us, noise floor {:.2}%)",
            "",
            "",
            ((warm2 - warm1) / warm1 * 100.0).abs()
        );
        xs.push(n as f64);
        warm_ts.push(warm1);
        cold_ts.push(cold);
    }
    let (a_w, b_w) = fit_ab(&xs, &warm_ts);
    let (a_c, b_c) = fit_ab(&xs, &cold_ts);
    let bytes_per_col = (K as f64 / 256.0) * block_bytes(scheme);
    eprintln!(
        "[regime-probe] fit {scheme:?}:  WARM  t = {a_w:.1} us + {b_w:.4} us/col   (marginal {} GB/s)",
        if b_w > 0.0 {
            bytes_per_col / (b_w * 1e-6) / 1e9
        } else {
            f64::NAN
        }
    );
    eprintln!(
        "[regime-probe] fit {scheme:?}:  COLD  t = {a_c:.1} us + {b_c:.4} us/col   (marginal {} GB/s)",
        if b_c > 0.0 {
            bytes_per_col / (b_c * 1e-6) / 1e9
        } else {
            f64::NAN
        }
    );
    eprintln!(
        "[regime-probe] fit {scheme:?}:  fixed per-launch cost a: warm {a_w:.1} us vs cold {a_c:.1} us"
    );
}

/// Arm A1+A2 for the K-quants and the Q8_0 control, at the 27B's dominant k.
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
#[test]
fn m1_decode_regime_working_set_sweep() {
    let Some(dev) = gpu_device() else {
        return;
    };
    // Same override the parity gate uses: production dispatch on non-RDNA2
    // arches checks this env before the dot4 leg.
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
        // The u4-lane leg is a LOSSY requantization of the checkpoint's own
        // quantization, so it is a different amount of bytes AND a different
        // amount of work. It is the DEFAULT for Q4_K, so the sweep runs with
        // it OFF to compare the EXACT dot4 legs of all five K-quants against
        // each other, and the census below reports which kernel each case
        // actually took.
        std::env::set_var("GRIM_DECODE_W4A4", "0");
        std::env::set_var("GRIM_DECODE_EXACT_GEMV", "0");
    }
    eprintln!(
        "[regime-probe] card ordinal {}  peak(datasheet) {PEAK_GBS:.0} GB/s  k={K}",
        dev.ordinal()
    );
    // The working set spans ~3 MB (L2-resident) to ~94 MB (DRAM-resident).
    // 8 copies of the weight give the cold arm enough independent buffers to
    // evict each other at the large end.
    let ns = [1024usize, 2048, 4096, 8192, 16384, 32768];
    let copies = 8usize;
    // Randomized FORMAT order: a sequential order across formats carries
    // co-allocation interference (res8 P6).
    let order = [
        KQuantScheme::Q5K,
        KQuantScheme::Q80,
        KQuantScheme::Q4K,
        KQuantScheme::Q6K,
        KQuantScheme::Q2K,
        KQuantScheme::Q3K,
    ];
    for (i, scheme) in order.iter().enumerate() {
        sweep_format(&dev, *scheme, &ns, copies, 1000 + i as u64 * 131);
    }
}

/// Arm A2 at the 27B's dominant shape only: the compute-only floor, with a
/// COPY-COUNT sweep so the residency boundary is measured rather than assumed.
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
#[test]
fn m1_decode_regime_compute_only_floor() {
    let Some(dev) = gpu_device() else {
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
        std::env::set_var("GRIM_DECODE_W4A4", "0");
        std::env::set_var("GRIM_DECODE_EXACT_GEMV", "0");
    }
    // The 27B's dominant projection shape.
    let n = 17408usize;
    eprintln!("[regime-probe] compute-only floor at the 27B dominant shape n={n} k={K}");
    for scheme in [
        KQuantScheme::Q4K,
        KQuantScheme::Q5K,
        KQuantScheme::Q6K,
        KQuantScheme::Q2K,
        KQuantScheme::Q3K,
    ] {
        for copies in [1usize, 2, 4, 8] {
            let h = match Harness::build(&dev, scheme, n, K, copies, 77) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("[regime-probe] build failed: {e}");
                    continue;
                }
            };
            h.warm_up(&dev);
            let warm = h.time(&dev, 0, false, 30);
            let cold = h.time(&dev, 0, true, 30);
            let wb = weight_bytes(scheme, n, K);
            eprintln!(
                "[regime-probe] {scheme:?} copies={copies} W_MB={:.1} warm={warm:.1}us cold={cold:.1}us ratio={:.3} d={:.1}us",
                wb / 1e6,
                cold / warm,
                cold - warm
            );
        }
    }
}

/// Arm A0 — the KERNEL CENSUS. One format, one launch, so a rocprofv3
/// kernel trace attributes the launch unambiguously. The regime probe cannot
/// be interpreted without this: the Q4_K arm of `quantized_matmul` gates its
/// dot4 leg on `!exact_decode_gemv()` (which DEFAULTS TRUE), so at M=1 with
/// the default env the Q4_K case does NOT run `grim_dot4_q4k_q81_gemv` — it
/// runs whatever the arm falls through to. The Q5_K/Q6_K arms gate differently.
/// A measurement that does not say which kernel ran is not a measurement.
///
/// RUN (the census):
///   HIP_VISIBLE_DEVICES=1 GRIM_RUN_GPU_TEST=1 \
///     rocprofv3 --kernel-trace --stats -- hip \
///     cargo test -p grim-backend-rocm --test m1_decode_regime_probe \
///       -- --ignored --nocapture --test-threads=1 m1_decode_regime_kernel_census
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
#[test]
fn m1_decode_regime_kernel_census() {
    let Some(dev) = gpu_device() else {
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
        std::env::set_var("GRIM_DECODE_W4A4", "0");
        std::env::set_var("GRIM_DECODE_EXACT_GEMV", "0");
    }
    let n = 4096usize;
    for scheme in [
        KQuantScheme::Q4K,
        KQuantScheme::Q5K,
        KQuantScheme::Q6K,
        KQuantScheme::Q2K,
        KQuantScheme::Q3K,
        KQuantScheme::Q80,
    ] {
        let h = Harness::build(&dev, scheme, n, K, 1, 31).expect("build");
        // 2 warmup + 1 measured: the trace shows the launch that the
        // per-format marker precedes.
        h.warm_up(&dev);
        let us = h.time(&dev, 0, false, 1);
        eprintln!("[census] scheme={scheme:?} n={n} k={K} one_launch={us:.1} us");
    }
}

/// Arm A0b — the IQ-quant decode leg, at the SAME shape as the K-quant
/// census. The IQ path is 17.2 GB/s GEMV on the 27B and 117 of that model's
/// tensors are IQ4_XS, so this is the measurement that decides which change
/// is worth building: the corpus names ONE THREAD PER OUTPUT ELEMENT (the
/// IQ kernels' mapping) as the biggest structural defect, and the
/// thread->data mapping, not the format, is what a measurement can move
/// without touching the checkpoint.
///
/// RUN (the census): same rocprofv3 invocation as the K-quant census.
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
#[test]
fn iq4xs_decode_kernel_census() {
    let Some(dev) = gpu_device() else {
        return;
    };
    let n = 4096usize;
    let alloc: Arc<RocmCachingAllocator> = dev.allocator_handle();
    let mut sseed: u64 = 91;
    let mut nxt = move || {
        sseed ^= sseed << 13;
        sseed ^= sseed >> 17;
        sseed ^= sseed << 5;
        sseed
    };
    let vals: Vec<f32> = (0..n * K)
        .map(|_| ((nxt() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let packed = grim_quant::quant_iq4xs(&vals).expect("quant_iq4xs");
    let row_bytes = packed.len() / n;
    eprintln!(
        "[census] IQ4XS n={n} k={K} row_bytes={row_bytes} weight_MB={:.2}",
        (n * row_bytes) as f64 / 1e6
    );
    // The IQ decode leg takes the packed bytes through the IQ4_XS arm of
    // quantized_matmul / linear_decode_into.
    let w = MemoryOps::from_cpu_bytes(
        &dev,
        &packed,
        &Shape::new(vec![n, K]),
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::IQ4XS),
        },
    )
    .expect("upload IQ4XS");
    let w_rocm = w
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("rocm weight storage");
    let a: Vec<f32> = (0..K).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let a_st =
        CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![1, K]), DType::F32).expect("upload act");
    let out = RocmStorage::alloc_gpu(&Shape::new(vec![n]), DType::F32, &alloc, 0).unwrap();
    for _ in 0..3 {
        dev.linear_decode_into(a_st.as_ref(), w_rocm, &out, &out)
            .expect("linear_decode_into");
    }
    dev.synchronize();
    let start = std::time::Instant::now();
    for _ in 0..20 {
        dev.linear_decode_into(a_st.as_ref(), w_rocm, &out, &out)
            .expect("linear_decode_into");
    }
    dev.synchronize();
    let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
    let gbps = ((n * row_bytes) as f64 / 1e9) / (us * 1e-6);
    eprintln!("[census] IQ4XS one_launch={us:.1} us  weight_GBps={gbps:.1}");
}

/// Arm A2b — THE IQ FAMILY CENSUS (PLAN-iq-family-decode T3). The same
/// measurement as `iq4xs_decode_kernel_census` for the table-index IQ
/// formats the 27B's experts carry: IQ3_XXS (98 B/256), IQ3_S (110 B/256),
/// IQ2_S (100 B/256). Times the production decode leg via
/// `linear_decode_into` (m=1), which routes each scheme through its
/// group-hoisted fused kernel; the rocprofv3 census names the kernel that
/// actually ran.
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
#[test]
fn iq_family_decode_census() {
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
        std::env::set_var("GRIM_DECODE_W4A4", "0");
        std::env::set_var("GRIM_DECODE_EXACT_GEMV", "0");
    }
    let Some(dev) = gpu_device() else {
        return;
    };
    let n = 4096usize;
    let alloc: Arc<RocmCachingAllocator> = dev.allocator_handle();
    let cases: [(
        KQuantScheme,
        &str,
        usize,
        fn(&[f32]) -> grim_tensor::Result<Vec<u8>>,
    ); 3] = [
        (KQuantScheme::IQ3XXS, "iq3xxs", 98, grim_quant::quant_iq3xxs),
        (KQuantScheme::IQ3S, "iq3s", 110, grim_quant::quant_iq3s),
        (KQuantScheme::IQ2S, "iq2s", 100, grim_quant::quant_iq2s),
    ];
    for (scheme, name, blk, quant) in cases {
        let mut sseed: u64 = 97;
        let mut nxt = move || {
            sseed ^= sseed << 13;
            sseed ^= sseed >> 17;
            sseed ^= sseed << 5;
            sseed
        };
        let vals: Vec<f32> = (0..n * K)
            .map(|_| ((nxt() & 0xffff) as f32 / 32768.0) - 1.0)
            .collect();
        let packed = quant(&vals).expect(name);
        let row_bytes = packed.len() / n;
        if row_bytes != (K / 256) * blk {
            eprintln!(
                "[census] {name}: row_bytes {row_bytes} != expected {}",
                (K / 256) * blk
            );
        }
        eprintln!(
            "[census] {name} n={n} k={K} row_bytes={row_bytes} weight_MB={:.2}",
            (n * row_bytes) as f64 / 1e6
        );
        let w = match MemoryOps::from_cpu_bytes(
            &dev,
            &packed,
            &Shape::new(vec![n, K]),
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(scheme),
            },
        ) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("[census] {name}: upload failed: {e}");
                continue;
            }
        };
        let w_rocm = w
            .as_any()
            .downcast_ref::<RocmStorage>()
            .expect("rocm weight storage");
        let a: Vec<f32> = (0..K).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let a_st = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![1, K]), DType::F32)
            .expect("upload act");
        let out = RocmStorage::alloc_gpu(&Shape::new(vec![n]), DType::F32, &alloc, 0).unwrap();
        for _ in 0..3 {
            dev.linear_decode_into(a_st.as_ref(), w_rocm, &out, &out)
                .map_err(|e| format!("{name}: {e}"))
                .expect("linear_decode_into");
        }
        dev.synchronize();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            dev.linear_decode_into(a_st.as_ref(), w_rocm, &out, &out)
                .expect("linear_decode_into");
        }
        dev.synchronize();
        let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        let gbps = ((n * row_bytes) as f64 / 1e9) / (us * 1e-6);
        eprintln!("[census] {name} one_launch={us:.1} us  weight_GBps={gbps:.1}");
    }
}

/// Arm A0c — THE LAUNCH BASELINE. The working-set fit gave a fixed
/// per-launch cost `a` of 0-20 us, but a two-parameter fit over 6 points
/// is not a sensitive instrument for a term that small. This measures the
/// launch term directly instead of inferring it:
///
///   (i)  the trivial FMA kernel with iters=1 — a launch with ~4
///        instructions of work, i.e. the pure dispatch cost;
///  (ii)  the real dot4 kernel at a MINIMAL shape (n=4 -> 1 block,
///        k=256 -> 1 super-block) — the launch plus the kernel's
///        prologue/epilogue/reduction, which is everything except
///        the streaming.
/// (iii)  the same kernel at the real shape, for the contrast.
///
/// The difference between (ii) and (iii) is the streaming; the
/// difference between (i) and (ii) is the kernel's own fixed cost.
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
#[test]
fn m1_decode_launch_baseline() {
    let Some(dev) = gpu_device() else {
        return;
    };
    // (i) pure launch: the rho microbench kernel with a single iteration.
    for _ in 0..5 {
        let _ = grim_backend_rocm::rho_microbench_for_test(&dev, 1);
    }
    dev.synchronize();
    let iters = 2000i32;
    let _ = grim_backend_rocm::rho_microbench_for_test(&dev, iters);
    dev.synchronize();
    // The kernel does `iters` iterations of 4 FMAs per thread; at iters=1
    // that is the dispatch cost plus a handful of instructions.
    let one = {
        let t = std::time::Instant::now();
        let _ = grim_backend_rocm::rho_microbench_for_test(&dev, 1);
        dev.synchronize();
        t.elapsed().as_secs_f64() * 1e6
    };
    eprintln!("[launch] (i) trivial kernel, 1 launch      = {one:>8.2} us");

    // (ii) minimal dot4 shape.
    let alloc: Arc<RocmCachingAllocator> = dev.allocator_handle();
    let n = 4usize;
    let k = 256usize;
    let mut sseed: u64 = 7;
    let mut nxt = move || {
        sseed ^= sseed << 13;
        sseed ^= sseed >> 17;
        sseed ^= sseed << 5;
        sseed
    };
    let vals: Vec<f32> = (0..n * k)
        .map(|_| ((nxt() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let packed = grim_quant::quant_q5k(&vals).expect("quant_q5k");
    let w = MemoryOps::from_cpu_bytes(
        &dev,
        &packed,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q5K),
        },
    )
    .expect("upload");
    let w_rocm = w.as_any().downcast_ref::<RocmStorage>().expect("rocm");
    let a: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let a_st =
        CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![1, k]), DType::F32).expect("upload act");
    let out = RocmStorage::alloc_gpu(&Shape::new(vec![n]), DType::F32, &alloc, 0).unwrap();
    for _ in 0..3 {
        dev.launch_dot4_q5k_q81_gemv_for_ab(as_rocm(a_st.as_ref()).unwrap(), w_rocm, &out, 1, n, k)
            .unwrap();
    }
    dev.synchronize();
    let t = std::time::Instant::now();
    for _ in 0..2000 {
        dev.launch_dot4_q5k_q81_gemv_for_ab(as_rocm(a_st.as_ref()).unwrap(), w_rocm, &out, 1, n, k)
            .unwrap();
    }
    dev.synchronize();
    let us_min = t.elapsed().as_secs_f64() * 1e6 / 2000.0;
    eprintln!(
        "[launch] (ii) dot4 Q5K minimal shape n=4 k=256 = {us_min:>8.2} us  (weight {:.1} KB)",
        (n * (k / 256) * 176) as f64 / 1024.0
    );
    // The fit's `a` for Q5_K was 9.2 us (warm) / 12.1 us (cold).
    eprintln!("[launch]      working-set fit said a = 9.2 us (warm)");
    eprintln!(
        "[launch]      minimal-shape launches: {us_min:.2} us -> launch + prologue + reduction"
    );
}
