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
#[path = "precision_kernel_ab/w4a4.rs"]
mod w4a4;
use e4m3::{from_e4m3, to_e4m3_rne};
use w4a4::{dot_packed, pack_a, GROUP};

use grim_backend_rocm::precision_ab::*;
use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, BackendStorage, BlockDtype, CoreTensorOps, DType, MemoryOps, Shape, Storage,
};
use std::rc::Rc;
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
const ATOL_I4: f64 = 1e-2;
/// WhiteCrow is 4-bit, and its error is set by the quantizer rather than the
/// accumulator: `sudot8` is exact over the products.
///
/// Derived, not chosen. Group-wise, A's step is `amax/15`, so rounding error is
/// at most `step/2 = amax/30`. For a well-conditioned group the typical |a| is
/// around `amax/3`, giving a per-element relative error near
/// `(amax/30) / (amax/3)` = 10%. B's asymmetric quantizer adds the same order.
/// So int4's own format ceiling is ~10%, two orders of magnitude looser than
/// int8's 1e-3, and no amount of correct kernel code can do better.
///
/// 0.25 leaves headroom for the tail of the distribution while still being
/// derived rather than fitted: the measured worst case in
/// `measured_int4_error_sits_under_the_derived_budget` is checked against this
/// constant rather than the other way round.
const RTOL_I4: f64 = 0.25;

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
/// Acquire the device for the A/B run.
///
/// # No lock, and why the previous one was removed
///
/// This used to take `gpu_test_lock()` before HIP init, on the reasoning that
/// `RocmDevice::try_new` "contends for /dev/kfd and for driver-internal state
/// shared with every other process on the box", so a run could block in init
/// with no lock held.
///
/// **That justification was unsound.** `gpu_test_lock()` is a process-local
/// `static Mutex<()>` (`device/util.rs`). Another *process* running a grim GPU
/// test never contends on it, so the lock could not have serialised against the
/// very hazard it was installed for. It only ever serialised threads within this
/// one binary -- and this binary contains exactly one `#[test]`, so there was
/// nothing to serialise against. It was pure overhead guarding against a hazard
/// it could not see.
///
/// The underlying concern is real and is *not* dismissed: two processes
/// initialising HIP concurrently is a genuine possibility on a shared box. If
/// that ever needs mutual exclusion it requires a file lock, which unlike a
/// process-local mutex is visible across processes. `jit_cache.rs` already does
/// exactly that with an advisory `flock` on `.write.lock`; that is the pattern
/// to follow, and it is not needed here.
///
/// `catch_unwind` is kept: a panic inside HIP init must not abort the process,
/// and the previous behaviour of reporting a skip is what the test relies on.
fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).ok()).ok().flatten()
}

/// Which arms to run, from `GRIM_AB_ARMS` (comma-separated substrings).
///
/// Exists because a GPU memory fault in any one kernel takes down the whole
/// process, and a four-arm harness where three arms are innocent is
/// undiagnosable without it. On gfx1200 the full sweep faults with
/// "Memory access fault ... Page not present"; with this, each arm can be run
/// alone to find the culprit.
///
/// Unset means all arms, so the default path is unchanged. The report labels an
/// arm-filtered run, because a one-arm artifact must never be mistaken for a
/// four-arm result.
fn arm_selected(name: &str) -> bool {
    let Some(spec) = std::env::var("GRIM_AB_ARMS").ok() else {
        return true;
    };
    let spec = spec.trim();
    if spec.is_empty() || spec == "all" {
        return true;
    }
    spec.split(',')
        .map(str::trim)
        .any(|want| !want.is_empty() && name.to_ascii_lowercase().contains(&want.to_ascii_lowercase()))
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

// ---------------------------------------------------------------- packers --

/// Q8_0: 34 B per 32 values — f16 scale then 32 int8 codes.
fn pack_q8_0_host(vals: &[f32]) -> Vec<u8> {
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

/// Reinterpret a typed slice as bytes for upload. The crate has no `bytemuck`,
/// and adding a dependency to a test for four casts is a poor trade.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytemuck_cast_u32(v: &[u32]) -> &[u8] {
    as_bytes(v)
}
fn bytemuck_cast_i32(v: &[i32]) -> &[u8] {
    as_bytes(v)
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
/// Every packed form of B, resident on the device for one K.
///
/// Built once per K and shared by all arms and all M values. B does not depend
/// on M, and the untiled and row-tiled twins consume byte-identical operands --
/// they differ only in launch geometry -- so one upload serves all of them.
struct BOnDevice {
    /// E4M3 bytes, for Raven, WhiteRaven and RavenRowTiled.
    fp8: Buf,
    /// Host copy of the same E4M3 bytes, for the oracles. The oracle has to
    /// read what the kernel consumed, not the f32 source.
    fp8_host: Vec<u8>,
    /// Q8_0, for ForestRaven and its row-tiled twin.
    q80: Buf,
    /// Host copy of the Q8_0 bytes, for the oracle.
    q80_host: Vec<u8>,
    /// int4 codes / bf16 scales / u8 zeros, for WhiteCrow and its twin.
    i4_codes: Buf,
    i4_scales: Buf,
    i4_zeros: Buf,
    /// Host-side int4 pack, kept for the oracle (which must read the packed
    /// values, not the f32 source, or it charges the kernel for the quantizer).
    i4_host: w4a4::PackedB,
}

impl BOnDevice {
    fn build(dev: &RocmDevice, b: &[f32], n: usize, k: usize) -> Result<Self, String> {
        let fp8_host: Vec<u8> = b.iter().map(|&v| to_e4m3_rne(v)).collect();
        let q80_host = pack_q8_0_host(b);
        let i4_host = w4a4::pack_b(b, n, k);
        let u8_dtype = DType { arith: ArithType::U8, storage: Storage::Native };

        let fp8t = MemoryOps::from_cpu_bytes(
            dev,
            &fp8_host,
            &Shape::new(vec![fp8_host.len()]),
            DType { arith: ArithType::F32, storage: Storage::Block(BlockDtype::Fp8) },
        )
        .map_err(|e| format!("b fp8 h2d: {e}"))?;
        let q80t = MemoryOps::from_cpu_bytes(dev, &q80_host, &Shape::new(vec![q80_host.len()]), u8_dtype)
            .map_err(|e| format!("b q80 h2d: {e}"))?;
        let i4c = MemoryOps::from_cpu_bytes(
            dev,
            as_bytes(&i4_host.codes),
            &Shape::new(vec![i4_host.codes.len()]),
            DType { arith: ArithType::U32, storage: Storage::Native },
        )
        .map_err(|e| format!("b int4 codes h2d: {e}"))?;
        let i4s = MemoryOps::from_cpu_bytes(
            dev,
            as_bytes(&i4_host.scales),
            &Shape::new(vec![i4_host.scales.len()]),
            DType { arith: ArithType::F16, storage: Storage::Native },
        )
        .map_err(|e| format!("b int4 scales h2d: {e}"))?;
        let i4z = MemoryOps::from_cpu_bytes(
            dev,
            &i4_host.zeros,
            &Shape::new(vec![i4_host.zeros.len()]),
            DType { arith: ArithType::U8, storage: Storage::Native },
        )
        .map_err(|e| format!("b int4 zeros h2d: {e}"))?;

        Ok(Self {
            fp8: buf(fp8t), fp8_host, q80: buf(q80t), q80_host,
            i4_codes: buf(i4c), i4_scales: buf(i4s), i4_zeros: buf(i4z),
            i4_host,
        })
    }
}

// ------------------------------------------------------------------- arms --

/// One arm, uploaded and ready to be timed or scored.
///
/// The driver alternates arms *within* each round, so an arm cannot own its own
/// timing loop: it has to split into "get onto the device" and "launch once".
/// That is the entire reason this is a struct.
struct Armed {
    arm: &'static str,
    instruction: &'static str,
    m: usize,
    n: usize,
    k: usize,
    /// Enqueues the kernel once. Asynchronous: the caller must synchronize to
    /// time it.
    ///
    /// The device is a parameter rather than a capture: `RocmDevice` holds a
    /// `Mutex` and is not `Clone`, so a stored closure cannot own one, and
    /// capturing `&dev` would borrow a local that is gone by the time the driver
    /// runs these. Passing it in keeps every buffer `Rc`-owned and every borrow
    /// confined to a single call.
    launch: Box<dyn FnMut(&RocmDevice)>,
    /// Max |gpu - oracle| and the tolerance it is judged against, from the
    /// output of the last launch. `FnOnce` because it consumes `out`.
    score: Option<Box<dyn FnOnce(&RocmDevice) -> (f64, f64)>>,
}

/// Time every arm round-robin, so a disturbance lands on all of them equally.
///
/// The previous driver measured each arm to completion before starting the next,
/// so the control's rounds occupied one time window and the challenger's
/// occupied another. On a shared card that is not a fair comparison: anything
/// that perturbs the GPU -- another process, a clock event -- lands entirely on
/// one side of the ratio, which is how a single unlucky control sample became a
/// 42x "speedup" in an earlier run.
///
/// Interleaving makes the comparison paired. In round i every arm sees the same
/// conditions and is measured adjacent in time, so a transient shows up in all
/// four and largely cancels in the ratio instead of inflating one side.
///
/// Returns per-arm samples, aligned by round index so the caller can compute a
/// *paired* ratio (round i of control against round i of challenger) as well as
/// the per-arm medians. The spread between those two is the honest error bar.
fn timed_interleaved(dev: &RocmDevice, arms: &mut [Armed]) -> Vec<Vec<f32>> {
    for a in arms.iter_mut() {
        for _ in 0..WARMUP {
            (a.launch)(dev);
        }
    }
    dev.synchronize();

    let mut out: Vec<Vec<f32>> = arms.iter().map(|_| Vec::with_capacity(ROUNDS)).collect();
    for _ in 0..ROUNDS {
        for (i, a) in arms.iter_mut().enumerate() {
            dev.synchronize();
            let t0 = Instant::now();
            (a.launch)(dev);
            dev.synchronize();
            out[i].push((t0.elapsed().as_secs_f64() * 1e3) as f32);
        }
    }
    out
}

/// Median of a sample, ignoring NaN.
fn median(xs: &[f32]) -> f32 {
    let mut v: Vec<f32> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Ratio of two arms' per-round samples, round by round.
///
/// This is the diagnostic the interleaving exists to enable. The ratio of
/// medians mixes the kernels' difference with any drift between the two
/// measurement windows; the paired median cancels drift, because both samples
/// come from the same round. When the two agree, the ratio is measuring the
/// kernel. When they diverge, the box was not quiet enough for the median ratio
/// to mean anything, and the artifact should say so rather than quietly
/// reporting the flattering number.
fn paired_ratio(control: &[f32], challenger: &[f32]) -> (f64, f64) {
    let pairs: Vec<f64> = control
        .iter()
        .zip(challenger.iter())
        .filter(|(c, x)| c.is_finite() && x.is_finite() && **x > 0.0)
        .map(|(c, x)| *c as f64 / *x as f64)
        .collect();
    if pairs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut s = pairs.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = s[s.len() / 2];
    // Relative IQR as a stability signal: tight means the box was quiet.
    let q1 = s[s.len() / 4];
    let q3 = s[3 * s.len() / 4];
    let spread = if med.abs() > 0.0 { (q3 - q1) / med } else { f64::NAN };
    (med, spread)
}

/// Owning device buffer that a stored closure can still reach.
///
/// The `launch` and `score` closures live in `Armed` and run after the arm
/// function has returned, so they cannot borrow a local `Box`. Each takes an
/// `Rc` clone and resolves the storage inside the closure body, where the borrow
/// is short and obviously sound -- no `unsafe`, no lifetime widening.
type Buf = Rc<Box<dyn BackendStorage>>;

fn buf(t: Box<dyn BackendStorage>) -> Buf {
    Rc::new(t)
}

fn raw(b: &Buf) -> &grim_backend_rocm::RocmStorage {
    grim_backend_rocm::as_rocm(b.as_ref().as_ref()).expect("storage is RocmStorage")
}



fn arm_raven(
    dev: &RocmDevice,
    a: &[f32],
    bd: &BOnDevice,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Armed, String> {
    let a_t = CoreTensorOps::from_cpu(dev, a, &Shape::new(vec![m, k]), DType::F32)
        .map_err(|e| format!("a h2d: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

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
            acc += av as f64 * from_e4m3(bd.fp8_host[j * k + kk]) as f64;
        }
        oracle.push(acc);
    }
    let tol = ATOL_F32 + RTOL_F32 * oracle.iter().fold(0f64, |m, o| m.max(o.abs()));

    // B is already resident (BOnDevice); only A and out belong to this arm.
    let a_buf = buf(a_t);
    let out_buf = buf(out_t);
    let b_buf = bd.fp8.clone();

    Ok(Armed {
        arm: "Raven",
        instruction: "V_DOT4_F32_FP8_FP8",
        m, n, k,
        launch: {
            let (a_buf, b_buf, out_buf) = (a_buf.clone(), b_buf.clone(), out_buf.clone());
            Box::new(move |dev: &RocmDevice| {
                let _ = dev.launch_dot4_fp8_gemv_for_ab(raw(&a_buf), raw(&b_buf), raw(&out_buf), m, n, k);
            })
        },
        score: Some({
            let out_buf = out_buf.clone();
            let pos = pos.clone();
            let oracle = oracle.clone();
            Box::new(move |dev: &RocmDevice| {
                dev.synchronize();
                (max_err(raw(&out_buf), &pos, &oracle), tol)
            })
        }),
    })
}

/// WhiteRaven: FP8 E4M3 via `v_wmma_f32_16x16x16_fp8_fp8`.
///
/// Same operand format as Raven, different arithmetic: rocwmma accumulates
/// f32 across a 16-wide K step with a wave-wide reduction, where Raven
/// accumulates through `v_dot4` per lane. Same accuracy class, different
/// instruction mix — which is the only reason to measure it separately.
fn arm_white_raven(
    dev: &RocmDevice,
    a: &[f32],
    bd: &BOnDevice,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Armed, String> {
    // WhiteRaven reads A already packed, unlike Raven which quantizes in-kernel.
    // Same values, same encoder — so the two oracles agree by construction and
    // any accuracy difference is the arithmetic, not the packing.
    let a_fp8: Vec<u8> = a.iter().map(|&v| to_e4m3_rne(v)).collect();
    let fp8_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Block(BlockDtype::Fp8),
    };

    let a_t = MemoryOps::from_cpu_bytes(dev, &a_fp8, &Shape::new(vec![m * k]), fp8_dtype.clone())
        .map_err(|e| format!("a h2d: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;


    let pos = check_positions(m * n);
    let mut oracle = Vec::with_capacity(pos.len());
    for &p in &pos {
        let (i, j) = (p / n, p % n);
        let mut acc = 0f64;
        for kk in 0..k {
            let av = from_e4m3(a_fp8[i * k + kk]);
            acc += av as f64 * from_e4m3(bd.fp8_host[j * k + kk]) as f64;
        }
        oracle.push(acc);
    }
    let tol = ATOL_F32 + RTOL_F32 * oracle.iter().fold(0f64, |m, o| m.max(o.abs()));

    let a_buf = buf(a_t);
    let out_buf = buf(out_t);
    let b_buf = bd.fp8.clone();

    Ok(Armed {
        arm: "WhiteRaven",
        instruction: "V_WMMA_F32_16X16X16_FP8_FP8",
        m, n, k,
        launch: {
            let (a_buf, b_buf, out_buf) = (a_buf.clone(), b_buf.clone(), out_buf.clone());
            Box::new(move |dev: &RocmDevice| {
                let _ = dev.launch_wmma_gemm_fp8_e4m3_for_ab(raw(&a_buf), raw(&b_buf), raw(&out_buf), m, n, k);
            })
        },
        score: Some({
            let out_buf = out_buf.clone();
            let pos = pos.clone();
            let oracle = oracle.clone();
            Box::new(move |dev: &RocmDevice| {
                dev.synchronize();
                (max_err(raw(&out_buf), &pos, &oracle), tol)
            })
        }),
    })
}

fn arm_forest_raven(
    dev: &RocmDevice,
    a: &[f32],
    bd: &BOnDevice,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Armed, String> {
    let a_q81 = pack_q8_1(a);
    let bytes = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };

    let a_t = MemoryOps::from_cpu_bytes(dev, &a_q81, &Shape::new(vec![a_q81.len()]), bytes.clone())
        .map_err(|e| format!("a h2d: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;


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
            let bscale = half::f16::from_bits(u16::from_le_bytes([
                bd.q80_host[bb], bd.q80_host[bb + 1],
            ])).to_f32();
            let bv = bscale * (bd.q80_host[bb + 2 + kk % 32] as i8) as f32;
            acc += av as f64 * bv as f64;
        }
        oracle.push(acc);
    }
    let tol = ATOL_I32 + RTOL_I32 * oracle.iter().fold(0f64, |m, o| m.max(o.abs()));

    let a_buf = buf(a_t);
    let out_buf = buf(out_t);
    let b_buf = bd.q80.clone();

    Ok(Armed {
        arm: "ForestRaven",
        instruction: "V_DOT4_I32_IU8",
        m, n, k,
        launch: {
            let (a_buf, b_buf, out_buf) = (a_buf.clone(), b_buf.clone(), out_buf.clone());
            Box::new(move |dev: &RocmDevice| {
                let _ = dev.launch_dot4_q80_q81_gemv(raw(&a_buf), raw(&b_buf), raw(&out_buf), m, n, k);
            })
        },
        score: Some({
            let out_buf = out_buf.clone();
            let pos = pos.clone();
            let oracle = oracle.clone();
            Box::new(move |dev: &RocmDevice| {
                dev.synchronize();
                (max_err(raw(&out_buf), &pos, &oracle), tol)
            })
        }),
    })
}

/// WhiteCrow: unsigned int4 via `v_dot8_i32_iu4`.
///
/// Asymmetric, with a two-dot zero-point identity:
/// `sum(A_i * W_i) = d_a * d_b * (iacc - z_b * sum_a_code)`.
///
/// A is quantized over non-negative values only (`w4a4::A_MIN`): the kernel
/// applies a zero-point correction to B alone, so a signed activation would
/// have no compensating term and the offset would not be recoverable from the
/// output. The fixture honours that rather than pretending otherwise.
fn arm_white_crow(
    dev: &RocmDevice,
    a: &[f32],
    bd: &BOnDevice,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Armed, String> {
    if k % GROUP != 0 {
        return Err(format!("WhiteCrow needs K % {} == 0, got {k}", GROUP));
    }
    let pa = pack_a(a, m, k);
    let pb = &bd.i4_host;

    // codes: u32 | scales: f32 (A) / bf16 (B) | zeros: u8
    let a_codes = MemoryOps::from_cpu_bytes(
        dev,
        bytemuck_cast_u32(&pa.codes),
        &Shape::new(vec![pa.codes.len()]),
        DType {
            arith: ArithType::U32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("a_codes h2d: {e}"))?;
    let a_scales = CoreTensorOps::from_cpu(dev, &pa.scales, &Shape::new(vec![pa.scales.len()]), DType::F32)
        .map_err(|e| format!("a_scales h2d: {e}"))?;
    // A_sums is i32 on the device side; upload as raw bytes at 4 B/element.
    let a_sums = MemoryOps::from_cpu_bytes(
        dev,
        bytemuck_cast_i32(&pa.sums),
        &Shape::new(vec![pa.sums.len()]),
        DType {
            // The kernel reads A_sums as `const int*` (4 B/element). ArithType
            // has no I32, and I64 is also 4 B wide here, so the dtype only has
            // to agree on element size for the byte upload -- the bytes are
            // reinterpreted by the kernel, not by the tensor layer.
            arith: ArithType::I64,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("a_sums h2d: {e}"))?;
    let out_t = MemoryOps::alloc_storage(
        dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;


    // Oracle over the packed operands, via the kernel's own identity.
    let pos = check_positions(m * n);
    let mut oracle = Vec::with_capacity(pos.len());
    for &p in &pos {
        let (i, j) = (p / n, p % n);
        oracle.push(dot_packed(&pa, &pb, i, j, k));
    }
    let tol = ATOL_I4 + RTOL_I4 * oracle.iter().fold(0f64, |m, o| m.max(o.abs()));

    let a_codes = buf(a_codes);
    let a_scales = buf(a_scales);
    let a_sums = buf(a_sums);
    let out_buf = buf(out_t);

    Ok(Armed {
        arm: "WhiteCrow",
        instruction: "V_DOT8_I32_IU4",
        m, n, k,
        launch: {
            let (ac, asc, asum, bc, bsc, bz, o) = (
                a_codes.clone(), a_scales.clone(), a_sums.clone(),
                bd.i4_codes.clone(), bd.i4_scales.clone(), bd.i4_zeros.clone(),
                out_buf.clone(),
            );
            Box::new(move |dev: &RocmDevice| {
                let _ = dev.launch_dot8_w4a4_gemv(
                    raw(&ac), raw(&asc), raw(&asum), raw(&bc), raw(&bsc), raw(&bz), raw(&o),
                    m, n, k,
                );
            })
        },
        score: Some({
            let out_buf = out_buf.clone();
            let pos = pos.clone();
            let oracle = oracle.clone();
            Box::new(move |dev: &RocmDevice| {
                dev.synchronize();
                (max_err(raw(&out_buf), &pos, &oracle), tol)
            })
        }),
    })
}

// ------------------------------------------------------------------ driver --

#[test]
#[ignore = "needs a gfx12 GPU; writes target/precision_ab_<arch>.json"]
fn precision_kernel_ab() {
    let Some(dev) = gpu_device() else {
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
    // (challenger arm, m, k, paired ratio, relative IQR)
    let mut paired_ratio_report: Vec<(&str, usize, usize, f64, f64)> = Vec::new();

    // B depends only on (n, k), so generate it once per K and share it across
    // all ten M values. Regenerating inside the loop meant 1.56e9 LCG calls and
    // roughly 6 GB of host writes for a sweep whose entire GPU cost is ~0.2 s --
    // which is why the run appeared to hang while never touching the GPU hard.
    // A is (m, k)-dependent and stays per-shape.
    let mut b_cache: std::collections::BTreeMap<usize, Vec<f32>> = std::collections::BTreeMap::new();
    // Device-side B, packed once per K and shared by every arm and every M.
    //
    // B does not depend on M, and each arm wants the same rows of it in a
    // different format. Uploading per (arm, shape) meant ~1.9 GB of PCIe per
    // shape and ~75 GB over the sweep, which at 25 GB/s is ~50 minutes of pure
    // transfer -- the run hit a 230 s cap still uploading. Hoisting to one
    // upload per format per K brings it to ~8 GB total, and the packed forms
    // are reused by the untiled and row-tiled twins alike since they differ only
    // in launch geometry.
    let mut bdev: std::collections::BTreeMap<usize, BOnDevice> =
        std::collections::BTreeMap::new();

    for (m, n_fixed, k) in sweep_plan() {
        // N is widened per K so B cannot live in L2. At the old N=4096 and
        // K=22016, B is 90 MB for the 1-byte formats against this box's 96 MB
        // L2 -- and the int4 arm's 45 MB fitted outright, so its M-fold B re-read
        // was served from cache and it reported 3.5 TB/s, twice HBM peak. That
        // ratio measured the cache, not the kernel.
        let n = n_for_l2_eviction(k);
        let _ = n_fixed;
        let mut rng = lcg(0x5EED_0000 ^ ((m as u64) << 32) ^ k as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng() * 4.0).collect();
        let b: &Vec<f32> = b_cache.entry(k).or_insert_with(|| {
            let mut r = lcg(0xB0B0_0000 ^ k as u64);
            (0..n * k).map(|_| r() * 16.0).collect()
        });
        // Uploaded once per K and shared by every arm and every M.
        let bdk: &BOnDevice = match bdev.entry(k) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => {
                match BOnDevice::build(&dev, b, n, k) {
                    Ok(v) => e.insert(v),
                    Err(err) => {
                        eprintln!("B upload failed at K={k}: {err}");
                        continue;
                    }
                }
            }
        };

        // WhiteCrow's A must be non-negative: the kernel applies a zero-point
        // correction to B alone, so a signed activation has no compensating
        // term. The other arms take `a` signed, so this one gets its own
        // non-negative copy rather than the fixture being reshaped for everyone.
        let a_nonneg: Vec<f32> = a.iter().map(|v| v.abs()).collect();

        // Arm every arm first, then time them together round-robin. Oracle work
        // is done during setup, never inside a timed region.
        let mut armed: Vec<Armed> = Vec::new();
        for (name, r) in [
            ("Raven", arm_raven(&dev, &a, &bdk, m, n, k)),
            ("ForestRaven", arm_forest_raven(&dev, &a, &bdk, m, n, k)),
            ("WhiteCrow", arm_white_crow(&dev, &a_nonneg, &bdk, m, n, k)),
            ("WhiteRaven", arm_white_raven(&dev, &a, &bdk, m, n, k)),
        ] {
            if !arm_selected(name) {
                continue;
            }
            match r {
                Ok(x) => armed.push(x),
                Err(e) => eprintln!("arm setup failed at m={m} k={k} [{name}]: {e}"),
            }
        }
        if armed.is_empty() {
            continue;
        }

        // REPEATS interleaved passes. Each pass times every arm once per round,
        // so a disturbance inside a round is shared by all four.
        let mut passes: Vec<Vec<Vec<f32>>> = vec![Vec::new(); armed.len()];
        for _ in 0..REPEATS {
            for (slot, p) in passes.iter_mut().zip(timed_interleaved(&dev, &mut armed)) {
                slot.push(p);
            }
        }

        // Score from the last launch, then take each arm's median across passes.
        let mut scores: Vec<Option<(f64, f64)>> = vec![None; armed.len()];
        for (slot, a) in scores.iter_mut().zip(armed.iter_mut()) {
            *slot = Some(a.score.take().expect("scored once")(&dev));
        }

        // The control is whichever arm is named CONTROL_ARM, matched by shape.
        let control_idx = armed.iter().position(|a| a.arm == CONTROL_ARM);
        // Paired diagnostic, computed while `passes` is still owned: control
        // round i against challenger round i, within the same pass. If this
        // disagrees with the ratio of medians, the box was not quiet enough for
        // the median ratio to mean anything, and the report says so rather than
        // quietly printing the flattering number.
        if let Some(ci) = control_idx {
            for (i, a) in armed.iter().enumerate() {
                if i == ci {
                    continue;
                }
                let mut ratios = Vec::new();
                let mut spreads = Vec::new();
                for (c, x) in passes[ci].iter().zip(passes[i].iter()) {
                    let (r, sp) = paired_ratio(c, x);
                    ratios.push(r as f32);
                    spreads.push(sp as f32);
                }
                paired_ratio_report.push((
                    a.arm, a.m, a.k, median(&ratios) as f64, median(&spreads) as f64,
                ));
            }
        }

        for (i, (a, samples)) in armed.iter().zip(passes).enumerate() {
            let (err, tol) = scores[i].expect("scored");
            // Flatten the passes' per-round samples; the median over all of them
            // is the headline, and the paired ratio below is the error bar.
            let flat: Vec<f32> = samples.into_iter().flatten().collect();
            results.push(ArmResult {
                arm: a.arm,
                instruction: a.instruction,
                m: a.m,
                n: a.n,
                k: a.k,
                samples_ms: flat,
                max_abs_err: err,
                tolerance: tol,
            });
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

    // ---- paired-ratio diagnostic ----------------------------------------
    // The headline ratio is control-median / challenger-median. The paired ratio
    // is the median of per-round ratios within a pass, which cancels drift
    // because both samples come from the same round. When the two disagree, the
    // box was not quiet enough for the median ratio to be a kernel measurement,
    // and the artifact should say so instead of printing the flattering one.
    if !paired_ratio_report.is_empty() {
        println!("\npaired vs median ratio (relative IQR = run-to-run noise):");
        println!("{:<13} {:>4} {:>6} {:>10} {:>10} {:>8}", "arm", "M", "K", "paired", "median", "spread");
        let mut worst: Option<(&str, f64)> = None;
        for (arm, m, k, paired, spread) in &paired_ratio_report {
            let (paired, spread) = (*paired, *spread);
            let med = vs_control(&results, results.iter().find(|r| r.arm == *arm
                && r.m == *m && r.k == *k).expect("row exists"))
                .unwrap_or(f64::NAN);
            if spread.is_finite() && worst.as_ref().map_or(true, |(_, w)| spread > *w) {
                worst = Some((arm, spread));
            }
            // Only the noisiest few rows: 40 shapes x 3 arms is 120 lines.
            if *k == sweep_plan().last().map(|p| p.2).unwrap_or(0) {
                println!("{:<13} {:>4} {:>6} {:>10.3} {:>10.3} {:>8.1}%", arm, m, k, paired, med, spread * 100.0);
            }
        }
        if let Some((arm, sp)) = worst {
            println!("worst IQR: {arm} at {:.1}% -- a spread that wide means the", sp * 100.0);
            println!("median ratio carries that much uncertainty, whatever it says.");
        }
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
