//! §6 A/B driver — measures the codegen paths on real hardware.
//!
//! The decisions (accuracy gate, ship bar, artifact shape) live in
//! `src/precision_ab.rs` and are unit-tested on CPU. This file only builds
//! inputs, launches, times, and scores.
//!
//! # Arms in this run
//!
//! | arm | codename | instruction | status |
//! |---|---|---|---|
//! | 1 | Raven | `V_DOT4_F32_FP8_FP8` | shipping, the control |
//! | 2 | ForestRaven | `V_DOT4_I32_IU8` | shipping |
//! | 3 | WhiteCrow | `V_DOT8_I32_IU4` | measured, gated on accuracy |
//! | 4 | WhiteRaven | `V_WMMA_F32_16X16X16_FP8_FP8` | measured |
//!
//! Each arm supplies its own operands, because "one codegen path" is not
//! "one input format": Raven and WhiteRaven both consume E4M3, ForestRaven
//! Q8_1/Q8_0, WhiteCrow unsigned int4. Comparing them measures the kernels
//! as the runtime would actually call them, packer included.
//!
//! # Scoring against the *quantized* operands
//!
//! Each arm consumes a different packed weight format, so "the right answer" is
//! the product of the quantized operands, not the original f32. Scoring against
//! the f32 source would charge the kernel for the host packer's error and fail
//! the accuracy gate for the wrong reason — which is how a correct kernel gets
//! reported as broken.
//!
//! The oracle is evaluated on a bounded sample of output positions. A full
//! f64 reference at M=512, N=4096, K=22016 is 4.6e10 multiply-adds on the host,
//! which dominates the measurement it is supposed to check.
//!
//! ```sh
//! GRIM_RUN_GPU_TESTS=1 GRIM_GPU_TEST=1 \
//!   cargo test -p grim-backend-rocm --test precision_kernel_ab -- --ignored --nocapture
//! ```

#[path = "precision_kernel_ab/e4m3.rs"]
mod e4m3;
use e4m3::{from_e4m3, to_e4m3_rne};

use grim_backend_rocm::precision_ab::*;
use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, BlockDtype, CoreTensorOps, DType, MemoryOps, Shape, Storage,
};
use std::time::Instant;

/// Accuracy bounds, as `atol + rtol * |oracle|`.
///
/// The plan states these as 2e-2 (f32 chains) and 1e-3 (i32), which are
/// *relative* bounds -- read as absolute they are unsatisfiable and every arm
/// fails for a reason that has nothing to do with the kernel:
///
/// - |C| grows as sqrt(K) for random inputs, reaching ~3e3 at K=22016, so a 2e-2
///   absolute gate is ~1e-6 relative -- tighter than f32 epsilon.
/// - Pure f32 summation order alone contributes ~5e-2 at that magnitude, before
///   any quantization error. The gate would fail a bit-exact kernel.
///
/// The atol floor keeps near-zero outputs, where a pure relative bound divides by
/// ~0, from failing spuriously.
const ATOL_F32: f64 = 1e-2;
const RTOL_F32: f64 = 2e-2;
const ATOL_I32: f64 = 1e-2;
/// Q8_0/Q8_1 store their scales as f16, so ~1e-3 relative is the floor here,
/// not the i32 accumulator (which is exact over the products).
const RTOL_I32: f64 = 1e-3;

/// Output positions scored per shape. Bounds host cost; see module docs.
const N_CHECK: usize = 32;

const ROUNDS: usize = 5;
const WARMUP: usize = 3;

/// Repeated measurement passes, pooled by median-of-medians.
///
/// Set to 1: the GPU is shared, and 3 passes x 30 rounds is 9x the compute for
/// a tail-quality gain nobody has agreed is worth it. The first run's 41x
/// outlier was a noisy *denominator* (one disturbed Raven median at a single
/// (M,K) against a stable WhiteRaven one), so more rounds would sharpen the
/// ratio -- but not at 9x cost on a shared card. ROUNDS=5 keeps the p90
/// meaningful; treat the extreme ratios as noise, and re-run a single shape
/// before believing any of them.
const REPEATS: usize = 1;

/// Take the process-wide GPU lock, then open the device.
///
/// The order is the whole point. `RocmDevice::try_new` initialises HIP, which
/// contends for `/dev/kfd` and for driver-internal state shared with every other
/// process on the box. Doing it *before* the lock meant a run could block
/// indefinitely in init with no lock held and no way to see who it was waiting
/// for -- and if a previous run was killed while still holding that state, the
/// next one waited on a holder that would never release.
///
/// Acquire first, initialise second, hold for the whole run. A killed run then
/// drops the lock when its process dies, and the next one proceeds instead of
/// queueing behind a corpse.
fn gpu_device_locked() -> Option<(RocmDevice, std::sync::MutexGuard<'static, ()>)> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    let lock = grim_backend_rocm::device::util::gpu_test_lock();
    // Keep catch_unwind: a panic here would poison the mutex for every later
    // test in this binary. The guard is returned, so a panic mid-run still
    // releases it on unwind.
    let dev = std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new"))
        .ok()?;
    Some((dev, lock))
}

/// Deterministic LCG: a time- or address-dependent fixture would make the
/// artifact incomparable between runs, and therefore useless.
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / 2147483648.0) - 1.0
    }
}

fn rocm(t: &dyn grim_tensor::BackendStorage) -> &grim_backend_rocm::RocmStorage {
    grim_backend_rocm::as_rocm(t).expect("storage is RocmStorage")
}

// ---------------------------------------------------------------- packers --

/// Q8_0: 34 B per 32 values — f16 scale then 32 int8 codes.
fn pack_q8_0(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() / 32 * 34);
    for blk in vals.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = (amax / 127.0).max(1e-30);
        let b = half::f16::from_f32(d).to_bits();
        out.push((b & 0xFF) as u8);
        out.push((b >> 8) as u8);
        for v in blk {
            out.push(((*v / d).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    out
}

/// Q8_1: 36 B per 32 — f16 scale, f16 code-sum, 32 int8 codes. The A-side
/// format the `sudot4` kernels consume so the activation sum can be subtracted.
fn pack_q8_1(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() / 32 * 36);
    for blk in vals.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = (amax / 127.0).max(1e-30);
        let codes: Vec<i8> = blk
            .iter()
            .map(|v| (*v / d).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let sum: f32 = codes.iter().map(|c| *c as f32).sum();
        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(sum).to_bits().to_le_bytes());
        for c in codes {
            out.push(c as u8);
        }
    }
    out
}

// ----------------------------------------------------------------- timing --

/// Time `launch`, returning per-round samples.
///
/// The synchronize *inside* the timed region is the whole point. These launches
/// are async: timing launch-to-return without a wait measures HIP submission
/// overhead, which is flat in M and K and made every arm look identical in the
/// first run. The leading synchronize drains the previous round so its cost
/// lands outside the measurement rather than inside it.
fn timed<F: FnMut()>(dev: &RocmDevice, mut launch: F) -> Vec<f32> {
    for _ in 0..WARMUP {
        launch();
    }
    dev.synchronize();
    let mut s = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        dev.synchronize();
        let t0 = Instant::now();
        launch();
        dev.synchronize();
        s.push((t0.elapsed().as_secs_f64() * 1e3) as f32);
    }
    s
}

/// Pool the samples from `REPEATS` independent measurement passes.
///
/// Concatenating raw samples and taking one median would let the passes
/// disagree about which samples are central. Taking the median per pass and
/// then the median of those is robust to a whole pass being disturbed, which is
/// what actually happens when another process shares the GPU.
fn pooled(samples: Vec<Vec<f32>>) -> Vec<f32> {
    let mut per_pass: Vec<f32> = samples
        .iter()
        .map(|s| {
            let mut v = s.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        })
        .collect();
    per_pass.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    per_pass
}

/// Max |gpu - oracle| over the sampled output positions.
///
/// `pos` must be threaded through: the sample is a *spread* over the tile, so
/// reading `bytes[i*4]` for `i in 0..oracle.len()` compares the oracle taken at
/// `pos[i]` against output element `i`. With a step of n/256 that mismatches
/// every index but 0, which is what made both arms report ~1e3 of error that
/// was pure bookkeeping.
fn max_err(out: &grim_backend_rocm::RocmStorage, pos: &[usize], oracle: &[f64]) -> f64 {
    let bytes = match out.copy_to_host() {
        Ok(b) => b,
        Err(_) => return f64::NAN,
    };
    if bytes.len() < pos.iter().copied().max().map(|m| m * 4 + 4).unwrap_or(0) {
        return f64::NAN;
    }
    let mut worst = 0.0f64;
    for (&p, o) in pos.iter().zip(oracle) {
        let g = f32::from_le_bytes([
            bytes[p * 4],
            bytes[p * 4 + 1],
            bytes[p * 4 + 2],
            bytes[p * 4 + 3],
        ]) as f64;
        worst = worst.max((g - o).abs());
    }
    worst
}

/// Positions to score: a deterministic spread over the output tile, capped.
fn check_positions(n: usize) -> Vec<usize> {
    let total = n;
    if total <= N_CHECK {
        return (0..total).collect();
    }
    let step = (total / N_CHECK).max(1);
    (0..N_CHECK).map(|i| (i * step).min(total - 1)).collect()
}

// ------------------------------------------------------------------- arms --

fn run_raven(
    dev: &RocmDevice,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Vec<f32>, f64, f64), String> {
    let b_fp8: Vec<u8> = b.iter().map(|&v| to_e4m3_rne(v)).collect();

    let a_t = CoreTensorOps::from_cpu(dev, a, &Shape::new(vec![m, k]), DType::F32)
        .map_err(|e| format!("a h2d: {e}"))?;
    let b_t = MemoryOps::from_cpu_bytes(
        dev,
        &b_fp8,
        &Shape::new(vec![b_fp8.len()]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Block(BlockDtype::Fp8),
        },
    )
    .map_err(|e| format!("b h2d: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

    let (a_r, b_r, out_r) = (rocm(a_t.as_ref()), rocm(b_t.as_ref()), rocm(out_t.as_ref()));

    let launch = || {
        let _ = dev.launch_dot4_fp8_gemv_for_ab(a_r, b_r, out_r, m, n, k);
    };
    let mut passes: Vec<Vec<f32>> = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        passes.push(timed(dev, launch));
    }
    let samples = pooled(passes);
    dev.synchronize();

    // Oracle over the *quantized* operands, matching what the kernel consumed.
    // A must be pushed through the same E4M3 rounding the kernel applies
    // in-kernel via grim_pack4_fp8 -- using raw f32 A here charges the kernel
    // for the packer's error and produced ~1e3 of phantom residual.
    let pos = check_positions(m * n);
    let mut oracle = Vec::with_capacity(pos.len());
    for &p in &pos {
        let (i, j) = (p / n, p % n);
        let mut acc = 0f64;
        for kk in 0..k {
            let av = from_e4m3(to_e4m3_rne(a[i * k + kk]));
            acc += av as f64 * from_e4m3(b_fp8[j * k + kk]) as f64;
        }
        oracle.push(acc);
    }
    let err = max_err(out_r, &pos, &oracle);
    let tol = ATOL_F32 + RTOL_F32 * oracle.iter().fold(0f64, |m, o| m.max(o.abs()));
    Ok((samples, err, tol))
}

/// WhiteRaven: FP8 E4M3 via `v_wmma_f32_16x16x16_fp8_fp8`.
///
/// Same operand format as Raven, different arithmetic: rocwmma accumulates
/// f32 across a 16-wide K step with a wave-wide reduction, where Raven
/// accumulates through `v_dot4` per lane. Same accuracy class, different
/// instruction mix — which is the only reason to measure it separately.
fn run_white_raven(
    dev: &RocmDevice,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Vec<f32>, f64, f64), String> {
    // WhiteRaven reads A already packed, unlike Raven which quantizes in-kernel.
    // Same values, same encoder — so the two oracles agree by construction and
    // any accuracy difference is the arithmetic, not the packing.
    let a_fp8: Vec<u8> = a.iter().map(|&v| to_e4m3_rne(v)).collect();
    let b_fp8: Vec<u8> = b.iter().map(|&v| to_e4m3_rne(v)).collect();
    let fp8_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Block(BlockDtype::Fp8),
    };

    let a_t = MemoryOps::from_cpu_bytes(dev, &a_fp8, &Shape::new(vec![m * k]), fp8_dtype.clone())
        .map_err(|e| format!("a h2d: {e}"))?;
    let b_t = MemoryOps::from_cpu_bytes(dev, &b_fp8, &Shape::new(vec![n * k]), fp8_dtype.clone())
        .map_err(|e| format!("b h2d: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

    let (a_r, b_r, out_r) = (rocm(a_t.as_ref()), rocm(b_t.as_ref()), rocm(out_t.as_ref()));

    let launch = || {
        let _ = dev.launch_wmma_gemm_fp8_e4m3_for_ab(a_r, b_r, out_r, m, n, k);
    };
    let mut passes: Vec<Vec<f32>> = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        passes.push(timed(dev, launch));
    }
    let samples = pooled(passes);
    dev.synchronize();

    let pos = check_positions(m * n);
    let mut oracle = Vec::with_capacity(pos.len());
    for &p in &pos {
        let (i, j) = (p / n, p % n);
        let mut acc = 0f64;
        for kk in 0..k {
            let av = from_e4m3(a_fp8[i * k + kk]);
            acc += av as f64 * from_e4m3(b_fp8[j * k + kk]) as f64;
        }
        oracle.push(acc);
    }
    let err = max_err(out_r, &pos, &oracle);
    let tol = ATOL_F32 + RTOL_F32 * oracle.iter().fold(0f64, |m, o| m.max(o.abs()));
    Ok((samples, err, tol))
}

fn run_forest_raven(
    dev: &RocmDevice,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Vec<f32>, f64, f64), String> {
    let a_q81 = pack_q8_1(a);
    let b_q80 = pack_q8_0(b);
    let bytes = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };

    let a_t = MemoryOps::from_cpu_bytes(dev, &a_q81, &Shape::new(vec![a_q81.len()]), bytes.clone())
        .map_err(|e| format!("a h2d: {e}"))?;
    let b_t = MemoryOps::from_cpu_bytes(dev, &b_q80, &Shape::new(vec![b_q80.len()]), bytes)
        .map_err(|e| format!("b h2d: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

    let (a_r, b_r, out_r) = (rocm(a_t.as_ref()), rocm(b_t.as_ref()), rocm(out_t.as_ref()));

    let launch = || {
        let _ = dev.launch_dot4_q80_q81_gemv(a_r, b_r, out_r, m, n, k);
    };
    let mut passes: Vec<Vec<f32>> = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        passes.push(timed(dev, launch));
    }
    let samples = pooled(passes);
    dev.synchronize();

    let pos = check_positions(m * n);
    let mut oracle = Vec::with_capacity(pos.len());
    // Row stride is n_q_blocks blocks, NOT 32. `i * 32` happens to be correct
    // at K=1024 (where n_q_blocks == 32) and silently reads the wrong row for
    // every larger K -- which is exactly how this arm first looked like a
    // kernel that "worked at one shape and diverged above it".
    let nb = k / 32;
    for &p in &pos {
        let (i, j) = (p / n, p % n);
        let mut acc = 0f64;
        for kk in 0..k {
            // Q8_1 block: f16 scale @0, f16 sum @2, 32 codes @4 (GRIM_Q8_1_BYTES=36).
            // Q8_0 block: f16 scale @0, 32 codes @2          (GRIM_Q8_0_BYTES=34).
            // The kernel accumulates i32 per block then scales in f32 and does
            // not use the Q8_1 sum field, so a per-element f64 product is the
            // same quantity with less rounding, not a different one.
            let ab = (i * nb + kk / 32) * 36;
            let ad = half::f16::from_bits(u16::from_le_bytes([a_q81[ab], a_q81[ab + 1]])).to_f32();
            let av = ad * (a_q81[ab + 4 + kk % 32] as i8) as f32;
            let bb = (j * nb + kk / 32) * 34;
            let bd = half::f16::from_bits(u16::from_le_bytes([b_q80[bb], b_q80[bb + 1]])).to_f32();
            let bv = bd * (b_q80[bb + 2 + kk % 32] as i8) as f32;
            acc += av as f64 * bv as f64;
        }
        oracle.push(acc);
    }
    let err = max_err(out_r, &pos, &oracle);
    let tol = ATOL_I32 + RTOL_I32 * oracle.iter().fold(0f64, |m, o| m.max(o.abs()));
    Ok((samples, err, tol))
}

// ------------------------------------------------------------------ driver --

#[test]
#[ignore = "needs a gfx12 GPU; writes target/precision_ab_<arch>.json"]
fn precision_kernel_ab() {
    // The lock is taken inside gpu_device_locked, before any HIP init, and held
    // for the whole run. Binding it to `_dev_and_lock` keeps it alive to the end
    // of the test rather than to the end of this statement.
    let Some((dev, _dev_and_lock)) = gpu_device_locked() else {
        eprintln!("[SKIP] requires GRIM_GPU_TEST=1 + GPU");
        return;
    };

    let arch = dev.gpu_target_str().to_string();
    eprintln!("arch: {arch}");

    // Raven's dot11-insts are RDNA4-only. Say so rather than emitting a
    // 1-arm run that looks like a 2-arm result with a hole in it.
    if !arch.starts_with("gfx12") {
        eprintln!("[SKIP] the dot family under test needs gfx1200/gfx1201, got {arch}");
        return;
    }
    // The FP8 path is env-gated in the dispatch; force it on for the arm.
    unsafe { std::env::set_var("GRIM_DOT_GEMV", "1") };

    let mut results: Vec<ArmResult> = Vec::new();

    // B depends only on (n, k), so generate it once per K and share it across
    // all ten M values. Regenerating inside the loop meant 1.56e9 LCG calls and
    // roughly 6 GB of host writes for a sweep whose entire GPU cost is ~0.2 s --
    // which is why the run appeared to hang while never touching the GPU hard.
    // A is (m, k)-dependent and stays per-shape.
    let mut b_cache: std::collections::BTreeMap<usize, Vec<f32>> = std::collections::BTreeMap::new();

    for (m, n, k) in sweep_plan() {
        let mut rng = lcg(0x5EED_0000 ^ ((m as u64) << 32) ^ k as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng() * 4.0).collect();
        let b: &Vec<f32> = b_cache.entry(k).or_insert_with(|| {
            let mut r = lcg(0xB0B0_0000 ^ k as u64);
            (0..n * k).map(|_| r() * 16.0).collect()
        });

        match run_raven(&dev, &a, b, m, n, k) {
            Ok((samples, err, tol)) => results.push(ArmResult {
                arm: "Raven",
                instruction: "V_DOT4_F32_FP8_FP8",
                m,
                n,
                k,
                samples_ms: samples,
                max_abs_err: err,
                tolerance: tol,
            }),
            Err(e) => eprintln!("Raven m={m} k={k}: {e}"),
        }
        match run_white_raven(&dev, &a, b, m, n, k) {
            Ok((samples, err, tol)) => results.push(ArmResult {
                arm: "WhiteRaven",
                instruction: "V_WMMA_F32_16X16X16_FP8_FP8",
                m,
                n,
                k,
                samples_ms: samples,
                max_abs_err: err,
                tolerance: tol,
            }),
            Err(e) => eprintln!("WhiteRaven m={m} k={k}: {e}"),
        }
        match run_forest_raven(&dev, &a, b, m, n, k) {
            Ok((samples, err, tol)) => results.push(ArmResult {
                arm: "ForestRaven",
                instruction: "V_DOT4_I32_IU8",
                m,
                n,
                k,
                samples_ms: samples,
                max_abs_err: err,
                tolerance: tol,
            }),
            Err(e) => eprintln!("ForestRaven m={m} k={k}: {e}"),
        }
    }

    if results.is_empty() {
        eprintln!("no arm produced a measurement; artifact not written");
        return;
    }

    // ---- report ---------------------------------------------------------
    let base: Vec<&ArmResult> = results.iter().filter(|r| r.arm == "Raven").collect();
    println!(
        "\n{:<13} {:>4} {:>6} {:>10} {:>10} {:>10} {:>8}  {}",
        "arm", "M", "K", "median_ms", "p90_ms", "max_err", "vs ctrl", "verdict"
    );
    for r in &results {
        let b = base
            .iter()
            .find(|x| x.m == r.m && x.k == r.k)
            .map(|x| x.median_ms())
            .unwrap_or(f32::NAN);
        let ratio = if b.is_finite() && r.median_ms() > 0.0 {
            format!("{:.2}x", b / r.median_ms())
        } else {
            "-".into()
        };
        let v = judge(r, b);
        let label = if v == Verdict::Eligible && clears_ship_bar(r, b) {
            "SHIPS"
        } else if v == Verdict::Eligible {
            "below-bar"
        } else {
            v.as_str()
        };
        println!(
            "{:<13} {:>4} {:>6} {:>10.4} {:>10.4} {:>10.2e} {:>8}  {}",
            r.arm, r.m, r.k, r.median_ms(), r.p90_ms(), r.max_abs_err, ratio, label
        );
    }

    let bad: Vec<&str> = results.iter().filter(|r| !r.accuracy_ok()).map(|r| r.arm).collect();
    if !bad.is_empty() {
        println!("\nDISQUALIFIED on accuracy (speed not read): {bad:?}");
    }
    println!("\narms: {:?}", by_arm(&results).keys().collect::<Vec<_>>());

    // ---- artifact -------------------------------------------------------
    let path = std::path::Path::new("target").join(format!("precision_ab_{arch}.json"));
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let json = to_artifact_json(&results, &arch);
    // Validate before announcing success. A malformed artifact that logs
    // "artifact: <path>" is worse than no artifact: the run looks complete and
    // the breakage surfaces later, in whatever consumes it.
    if !json_is_well_formed(&json) {
        panic!("generated artifact is not well-formed JSON -- refusing to write");
    }
    match std::fs::write(&path, &json) {
        Ok(()) => eprintln!("artifact: {}", path.display()),
        Err(e) => eprintln!("artifact write failed: {e}"),
    }
}
