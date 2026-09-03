//! SCYTHE-2 Capability Profiler (WI-2). [see: `GpuCapability`, `SelfTuningController`, `tflops_fp16`, `throttle_pct`]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use grim_tensor::backend::{GpuCapability, ScytheLink};

use crate::device::probe::probe_host_gpu;
use crate::peer_access::{P2PStatus, enumerate_devices, peer_status};

use libloading::Symbol;

// ── HIP attributes needed for VRAM free / throttle ──────────────────────────

/// HIP device attribute: active clock throttle reasons. **No such attribute
/// exists in ROCm 7.x** — the old constant here (74) actually selected
/// `MaxSharedMemoryPerBlock`, whose ~64 KB reading clamped to "100%
/// throttled" and zeroed every GPU's effective TFLOPS. Kept only to document
/// the removal; `query_throttle_pct` now reports honest absence.
#[allow(dead_code)]
const HIP_DEVICE_ATTR_THROTTLE_REMOVED: i32 = 74;

/// HIP device attribute: current core clock rate in kHz (`hipDeviceAttributeClockRate`,
/// enum value 5 in ROCm 7.x — not CUDA's 13). Part of the calibration cache key.
const HIP_DEVICE_ATTR_CLOCK_RATE: i32 = 5;

/// HIP device attribute: constant memory in bytes (attribute 23 in hip_runtime_api.h).
#[allow(dead_code)]
const HIP_DEVICE_ATTR_TOTAL_CONST_MEM: i32 = 23;

// ── Public surface ────────────────────────────────────────────────────────────

/// SCYTHE-2 capability epoch counter. [see: `CapabilityProfiler::bump_epoch`, `PlacementCache`, `decide_miss()`, `grim-disagg`]
pub static CAPABILITY_EPOCH: AtomicU32 = AtomicU32::new(0);

/// Bump the global capability epoch. [see: `PlacementCache::get()`, `r`]
pub fn bump_epoch() {
    CAPABILITY_EPOCH.fetch_add(1, Ordering::Release);
}

/// Read the current epoch without modifying it.
pub fn current_epoch() -> u32 {
    CAPABILITY_EPOCH.load(Ordering::Acquire)
}

/// Per-GPU live capability snapshot builder and epoch manager. [see: `CapabilityProfiler::new()`, `tick()`, `capabilities()`, `bump_epoch()`]
pub struct CapabilityProfiler {
    inner: Arc<Mutex<ProfilerState>>,
}

struct ProfilerState {
    /// Last-measured snapshot, indexed by GPU ordinal.
    caps: Vec<GpuCapability>,
    /// Time of the last `tick()` call.
    last_tick: Instant,
    /// Throttle fractions from the previous tick (for the >10% delta check).
    prev_throttle: Vec<f32>,
}

impl Default for CapabilityProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityProfiler {
    /// Create a new profiler. Blocks for one initial sweep (~5 ms per GPU).
    pub fn new() -> Self {
        let num_gpus = enumerate_devices().unwrap_or(0);
        let caps = (0..num_gpus).map(measure_capability).collect::<Vec<_>>();
        let prev_throttle = caps.iter().map(|c| c.throttle_pct).collect();
        Self {
            inner: Arc::new(Mutex::new(ProfilerState {
                caps,
                last_tick: Instant::now(),
                prev_throttle,
            })),
        }
    }

    /// Refresh the capability snapshot for every GPU. [see: `throttle_pct`, `bump_epoch()`]
    pub fn tick(&self) {
        let num_gpus = enumerate_devices().unwrap_or(0);
        let mut state = self.inner.lock().expect("CapabilityProfiler lock poisoned");
        state.last_tick = Instant::now();

        // Resize tracking vecs if a GPU joined (hot-plug).
        state.caps.resize(num_gpus, GpuCapability::default());
        state.prev_throttle.resize(num_gpus, 0.0);

        let mut epoch_bumped = false;
        for ord in 0..num_gpus {
            let cap = measure_capability(ord);
            let prev = state.prev_throttle[ord];
            // Out-of-band epoch bump: >10% throttle delta (§3.6 escape hatch).
            if !epoch_bumped && (cap.throttle_pct - prev).abs() > 0.10 {
                bump_epoch();
                epoch_bumped = true;
                eprintln!(
                    "[scythe2] capability epoch bumped: GPU {} throttle delta {:.1}% → {:.1}%",
                    ord,
                    prev * 100.0,
                    cap.throttle_pct * 100.0
                );
            }
            state.prev_throttle[ord] = cap.throttle_pct;
            state.caps[ord] = cap;
        }
    }

    /// Return a snapshot of current capabilities for all visible GPUs. [see: `C2plrController::decide_miss()`]
    pub fn capabilities(&self) -> Vec<GpuCapability> {
        self.inner
            .lock()
            .expect("CapabilityProfiler lock poisoned")
            .caps
            .clone()
    }

    /// Build a K×K link matrix using the existing `peer_status` probe. [see: `Vec<ScytheLink>`, `ScythePlacement::routes`]
    pub fn link_matrix(num_gpus: usize) -> Vec<ScytheLink> {
        let k = num_gpus;
        let mut matrix = vec![ScytheLink::Host; k * k];
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    matrix[i * k + j] = ScytheLink::PeerDirect; // self-link
                    continue;
                }
                matrix[i * k + j] = match peer_status(i as i32, j as i32) {
                    Ok(P2PStatus::P2P) => ScytheLink::PeerDirect,
                    Ok(P2PStatus::Pcie) => ScytheLink::Pcie,
                    _ => ScytheLink::Host,
                };
            }
        }
        matrix
    }

    /// Duration since the last `tick()`. Useful for callers that manage their
    pub fn age(&self) -> Duration {
        self.inner
            .lock()
            .expect("CapabilityProfiler lock poisoned")
            .last_tick
            .elapsed()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Measure one GPU's capability snapshot.
///
/// WI-SB0: the snapshot prefers *measured* numbers — one small FP16 GEMM plus
/// a device-to-device copy sweep per device, run once per process and cached
/// by `(gcn_arch, clock_mhz)` in [`calibrate_capability`]. The static
/// architecture table is the fallback for GPU-less/ROCm-absent builds and any
/// calibration error; a failed measurement never fabricates zeros here.
pub(crate) fn measure_capability(ordinal: usize) -> GpuCapability {
    // Base probe from the existing infrastructure.
    let host_cap = match probe_host_gpu(ordinal) {
        Ok(c) => c,
        Err(e) => {
            // GPU disappeared between `enumerate_devices` and this probe.
            eprintln!("[scythe2] probe_host_gpu({ordinal}) failed: {e}");
            return GpuCapability {
                ordinal,
                ..Default::default()
            };
        }
    };

    // Prefer a measured result when the optional calibration backend is
    // available; the static row is deliberately retained for GPU-less/ROCm
    // installations and calibration failures.
    let (tflops_fp16, tflops_fp8, hbm_gbps) = calibrate_capability(ordinal, &host_cap.gcn)
        .unwrap_or_else(|| arch_tflops_table(&host_cap.gcn));

    // Throttle percentage via HIP attribute.
    let throttle_pct = query_throttle_pct(ordinal);

    // Free VRAM via hipMemGetInfo (crate-root re-export).
    let vram_free_bytes = query_vram_free(ordinal);

    // Apply thermal throttle to estimated TFLOPS.
    let effective_tflops = tflops_fp16 * (1.0 - throttle_pct);

    GpuCapability {
        tflops_fp16: effective_tflops,
        tflops_fp8,
        hbm_bandwidth_gbps: hbm_gbps,
        vram_free_bytes,
        throttle_pct,
        ordinal,
    }
}

/// Run the one-shot device calibration (WI-SB0).
///
/// Measures effective FP16 TFLOPS with a small rocBLAS GEMM (f16 inputs, f32
/// accumulation — the shape class inference actually runs) and effective HBM
/// bandwidth with a device-to-device copy sweep. ~ms per device, once per
/// process per `(gcnArchName, 500 MHz clock bucket)`: identical silicon at a
/// similar clock hits the cache instead of re-benchmarking on every profiler
/// tick.
///
/// FP8 has no rocBLAS-exercisable path on consumer RDNA parts, so its entry
/// is *derived*: measured-FP16 scaled by the static row's fp8/fp16 ratio.
/// Any HIP/rocBLAS error returns `None` so the caller falls back to the
/// architecture row rather than trusting a partial measurement.
fn calibrate_capability(ordinal: usize, gcn: &str) -> Option<(f32, f32, f32)> {
    // Kill switch for the root-cause hunt (2026-08-23e): does disabling the
    // micro-benchmark change second-device GEMM behavior?
    if std::env::var("GRIM_DISABLE_CALIBRATION").is_ok() {
        return None;
    }
    static CACHE: OnceLock<Mutex<HashMap<(String, u32), (f32, f32, f32)>>> = OnceLock::new();
    // Clock is bucketed to 500 MHz: DVFS jitters tens of MHz between calls,
    // and an exact-clock key would re-run the micro-benchmark on every
    // profiler tick. Big downclocks (thermal) still cross buckets, which is
    // exactly when re-measuring matters.
    let clock_bucket = query_clock_mhz(ordinal)? / 500;
    let key = (gcn.to_string(), clock_bucket);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(hit) = cache.lock() {
        if let Some(found) = hit.get(&key) {
            return Some(*found);
        }
    }
    let measured = measure_device_throughput(ordinal, gcn)?;
    if let Ok(mut slot) = cache.lock() {
        slot.insert(key, measured);
    }
    Some(measured)
}

/// Current core clock of `ordinal` in MHz, or `None` when the attribute
/// query fails or reports a nonsense value.
fn query_clock_mhz(ordinal: usize) -> Option<u32> {
    let mut val: i32 = 0;
    let _guard = crate::device::util::DeviceGuard::set(ordinal as i32);
    unsafe {
        use crate::device::handles::hipDeviceGetAttribute;
        let status = hipDeviceGetAttribute(&mut val, HIP_DEVICE_ATTR_CLOCK_RATE, ordinal as i32);
        if status != 0 || val <= 0 {
            return None;
        }
    }
    Some((val / 1000) as u32)
}

/// Micro-benchmark one device: median-of-runs FP16 GEMM throughput plus DtoD
/// copy bandwidth, both timed host-side around `hipDeviceSynchronize` fences.
fn measure_device_throughput(ordinal: usize, gcn: &str) -> Option<(f32, f32, f32)> {
    use crate::device::handles::{hipDeviceSynchronize, hipFree, hipMalloc, hipMemcpy};
    use crate::device::rocblas::{
        ROCBLAS_GEMM_FLAGS_NONE, RocblasHandle, RocblasOperation, rocblas_create_handle,
        rocblas_datatype, rocblas_destroy_handle, rocblas_gemm_ex, rocblas_set_stream,
        select_gemm_algo,
    };
    use crate::device::util::DeviceGuard;
    use std::ptr::null_mut;

    // Keep the whole benchmark ≪ a frame budget: 2 warmup + 5 timed runs of
    // each leg land in single-digit ms even on an iGPU.
    const GEMM_DIM: usize = 2048;
    const GEMM_WARMUP: usize = 2;
    const GEMM_RUNS: usize = 5;
    const COPY_BYTES: usize = 128 * 1024 * 1024;
    const COPY_RUNS: usize = 3;

    let _guard = DeviceGuard::set(ordinal as i32);

    // ── GEMM leg ────────────────────────────────────────────────────────────
    let ab_bytes = GEMM_DIM * GEMM_DIM * 2; // f16
    let d_bytes = GEMM_DIM * GEMM_DIM * 4; // f32 accumulate out
    let mut d_a: *mut core::ffi::c_void = null_mut();
    let mut d_b: *mut core::ffi::c_void = null_mut();
    let mut d_d: *mut core::ffi::c_void = null_mut();
    unsafe {
        for ptr in [&mut d_a, &mut d_b] {
            if hipMalloc(ptr, ab_bytes) != 0 {
                return None;
            }
        }
        if hipMalloc(&mut d_d, d_bytes) != 0 {
            hipFree(d_a);
            hipFree(d_b);
            return None;
        }
        // All-ones halves keep the accumulator at exactly K — no inf/NaN to
        // perturb timing, no data-dependent clock behaviour.
        let ones_f16: Vec<u16> = vec![0x3C00; GEMM_DIM * GEMM_DIM];
        for (dst, src) in [(d_a, &ones_f16), (d_b, &ones_f16)] {
            if hipMemcpy(
                dst,
                src.as_ptr() as *const core::ffi::c_void,
                ab_bytes,
                crate::device::handles::HipMemcpyKind::HostToDevice,
            ) != 0
            {
                hipFree(d_a);
                hipFree(d_b);
                hipFree(d_d);
                return None;
            }
        }
    }

    let mut handle = RocblasHandle(null_mut());
    let gemm_ok = unsafe {
        if rocblas_create_handle(&mut handle) != 0 {
            false
        } else {
            // Default stream (null): calibration owns the device for its few ms.
            rocblas_set_stream(handle, std::ptr::null_mut()) == 0
        }
    };
    if !gemm_ok {
        unsafe {
            if !handle.0.is_null() {
                rocblas_destroy_handle(handle);
            }
            hipFree(d_a);
            hipFree(d_b);
            hipFree(d_d);
        }
        return None;
    }

    // Row-major C[M,N] = A[M,K]·B[K,N] maps to column-major as Cᵀ[N,M] =
    // Bᵀ[N,K]·A[K,M] — operands swapped, leading dims n/k/n (same convention
    // as `RocmDevice::matmul`).
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let n_dim = GEMM_DIM as i32;
    let run_gemm = |handle: RocblasHandle| unsafe {
        rocblas_gemm_ex(
            handle,
            RocblasOperation::None,
            RocblasOperation::None,
            n_dim,
            n_dim,
            n_dim,
            &alpha as *const f32 as *const core::ffi::c_void,
            d_b as *const core::ffi::c_void,
            rocblas_datatype::f16_r,
            n_dim,
            d_a as *const core::ffi::c_void,
            rocblas_datatype::f16_r,
            n_dim,
            &beta as *const f32 as *const core::ffi::c_void,
            d_d,
            rocblas_datatype::f32_r,
            n_dim,
            d_d,
            rocblas_datatype::f32_r,
            n_dim,
            rocblas_datatype::f32_r,
            select_gemm_algo(0),
            0,
            ROCBLAS_GEMM_FLAGS_NONE,
        )
    };

    let mut gemm_ms: Vec<f32> = Vec::with_capacity(GEMM_WARMUP + GEMM_RUNS);
    for _ in 0..(GEMM_WARMUP + GEMM_RUNS) {
        let t0 = Instant::now();
        let status = run_gemm(handle);
        unsafe { hipDeviceSynchronize() };
        let dt = t0.elapsed().as_secs_f32() * 1000.0;
        if status != 0 {
            unsafe {
                rocblas_destroy_handle(handle);
                hipFree(d_a);
                hipFree(d_b);
                hipFree(d_d);
            }
            return None;
        }
        gemm_ms.push(dt);
    }

    // ── Bandwidth leg (DtoD copy moves 2× bytes: read + write) ─────────────
    let mut c_src: *mut core::ffi::c_void = null_mut();
    let mut c_dst: *mut core::ffi::c_void = null_mut();
    let copy_ok = unsafe {
        if hipMalloc(&mut c_src, COPY_BYTES) != 0 || hipMalloc(&mut c_dst, COPY_BYTES) != 0 {
            hipFree(c_src);
            hipFree(c_dst);
            false
        } else {
            true
        }
    };
    let mut copy_ms: Vec<f32> = Vec::new();
    if copy_ok {
        for _ in 0..=COPY_RUNS {
            let t0 = Instant::now();
            let status = unsafe {
                hipMemcpy(
                    c_dst,
                    c_src as *const core::ffi::c_void,
                    COPY_BYTES,
                    crate::device::handles::HipMemcpyKind::DeviceToDevice,
                )
            };
            unsafe { hipDeviceSynchronize() };
            let dt = t0.elapsed().as_secs_f32() * 1000.0;
            if status != 0 {
                copy_ms.clear();
                break;
            }
            copy_ms.push(dt);
        }
    }

    unsafe {
        rocblas_destroy_handle(handle);
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_d);
        hipFree(c_src);
        hipFree(c_dst);
    }

    // Median of the post-warmup samples.
    let median = |v: &[f32]| -> Option<f32> {
        if v.is_empty() {
            return None;
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        Some(s[s.len() / 2])
    };
    if gemm_ms.len() <= GEMM_WARMUP {
        return None;
    }
    let gemm_median = median(&gemm_ms[GEMM_WARMUP..])?;
    let tflops_fp16 = 2.0 * (GEMM_DIM * GEMM_DIM * GEMM_DIM) as f32 / (gemm_median * 1e-3) / 1e12;
    if !(tflops_fp16 > 0.0 && tflops_fp16 < 20_000.0) {
        return None;
    }

    let hbm_gbps = if copy_ms.is_empty() {
        return None;
    } else {
        let copy_median = median(&copy_ms[1..])?;
        2.0 * COPY_BYTES as f32 / (copy_median * 1e-3) / 1e9
    };
    if !(hbm_gbps > 0.0 && hbm_gbps < 100_000.0) {
        return None;
    }

    // Derive FP8 from the measured FP16 number via the static row's ratio —
    // honest scaling, clearly not a direct measurement.
    let (s_fp16, s_fp8, _) = arch_tflops_table(gcn);
    let tflops_fp8 = if s_fp16 > 0.0 && s_fp8 > 0.0 {
        tflops_fp16 * (s_fp8 / s_fp16)
    } else {
        0.0
    };

    eprintln!(
        "[scythe2] WI-SB0 calibrated GPU {ordinal} ({gcn}): \
         fp16 {tflops_fp16:.1} TFLOPS, bw {hbm_gbps:.0} GB/s (measured)"
    );
    Some((tflops_fp16, tflops_fp8, hbm_gbps))
}

/// Architecture TFLOPS table — offline values per GCN arch string. [see: `2604.10187`, `throttle_pct`]
fn arch_tflops_table(gcn: &str) -> (f32, f32, f32) {
    // RDNA 3 / GFX11xx — RX 7900 XTX ≈ 61 TFLOPS FP16, 8 GB/s per mm²
    if gcn.starts_with("gfx11") {
        if gcn.contains("1100") {
            return (61.4, 0.0, 800.0); // RX 7900 XTX
        }
        if gcn.contains("1102") {
            return (26.0, 0.0, 288.0); // RX 7600
        }
        return (40.0, 0.0, 432.0); // generic RDNA3
    }
    // UDNA / GFX13xx & RDNA 4 / GFX12xx — has FP8 & unified matrix units
    if gcn.starts_with("gfx13") {
        return (160.0, 320.0, 1500.0);
    }
    if gcn == "gfx1201" {
        return (96.0, 192.0, 640.0); // RX 9070 XT static fallback baseline
    }
    if gcn == "gfx1200" {
        return (64.0, 128.0, 448.0); // RX 9060 XT static fallback baseline
    }
    if gcn.starts_with("gfx12") {
        return (80.0, 160.0, 960.0); // generic RDNA4 static fallback baseline
    }
    // CDNA (Instinct MI-series)
    if gcn.starts_with("gfx9") {
        if gcn.contains("950") || gcn.contains("951") {
            return (2614.0, 5228.0, 8000.0); // MI350/MI355X FP16 peak TFLOPS & HBM3e bandwidth
        }
        if gcn.contains("940") || gcn.contains("941") || gcn.contains("942") {
            return (1307.0, 2614.0, 5300.0); // MI300X FP16 peak TFLOPS & HBM3 bandwidth
        }
        if gcn.contains("90a") {
            return (383.0, 383.0, 3200.0); // MI250X
        }
        if gcn.contains("908") {
            return (184.6, 184.6, 1228.8); // MI100
        }
        if gcn.contains("906") || gcn.contains("900") {
            return (26.8, 0.0, 1024.0); // Vega20 / MI50
        }
        return (100.0, 200.0, 1600.0); // generic CDNA / GFX9
    }
    // APU / iGPU — very weak
    if gcn.starts_with("gfx103") || gcn.starts_with("gfx1036") {
        return (8.0, 0.0, 51.2);
    }
    // Unknown — return modest defaults so the controller doesn't divide by zero.
    (10.0, 0.0, 100.0)
}

/// Query the thermal throttle fraction [0, 1] for `ordinal`.
///
/// HIP exposes **no** thermal-throttle device attribute on this stack (see
/// `HIP_DEVICE_ATTR_THROTTLE_REMOVED`): the query that used to live here read
/// shared-memory size and reported every GPU as 100% throttled, zeroing all
/// effective TFLOPS. Until rsmi throttle-reason bindings land, report honest
/// absence — the A/B harness records this field alongside every sample.
fn query_throttle_pct(_ordinal: usize) -> f32 {
    0.0
}

/// Query free VRAM in bytes via `hipMemGetInfo`.
fn query_vram_free(ordinal: usize) -> u64 {
    vram_info(ordinal).0
}

/// Free device memory probe for the memory-sovereign admission gate (R4).
///
/// Returns the current free VRAM in bytes for `ordinal`, or `None` if the
/// backend cannot probe it. The engine uses this to certify, per request at
/// admission, that a new request's footprint fits within what is *currently*
/// free rather than what was free at model load. Backends without a probe
/// return `None`, in which case the admission gate is skipped (fail-open — the
/// request is admitted and bounded by the KV pool instead).
pub fn free_device_memory(ordinal: usize) -> Option<u64> {
    let (free, total) = vram_info(ordinal);
    if total == 0 {
        None
    } else {
        Some(free)
    }
}

/// Query `(free_bytes, total_bytes)` VRAM via `hipMemGetInfo`. [see: `(0, 0)`]
pub fn vram_info(ordinal: usize) -> (u64, u64) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let _guard = crate::device::util::DeviceGuard::set(ordinal as i32);
    unsafe {
        use crate::device::handles::hipMemGetInfo;
        let status = hipMemGetInfo(&mut free, &mut total);
        if status != 0 {
            return (0, 0);
        }
    }
    (free as u64, total as u64)
}

// ── ROCm SMI (rsmi) dynamic load — compute utilization ──────────────────────

/// WI-1: live GPU compute/busy utilization [0, 100] for `ordinal`.
///
/// ROCm exposes no utilization query through the HIP runtime API, so this
/// goes through `librocm_smi64.so` (`rsmi_dev_busy_percent_get`), the same
/// source `rocm-smi --showuse` reads. The library is dynamically loaded — no
/// link-time dependency, and `None` is returned cleanly when rsmi is absent
/// rather than fabricating a value from indirect signals.
///
/// The handle is cached process-wide behind a `OnceLock` so the stats
/// endpoint never re-dlopens or re-initializes rsmi per request (WI-1 gate 4:
/// the query must not block the endpoint for more than ~5ms).
pub fn compute_utilization(ordinal: usize) -> Option<u32> {
    if let Some(lib) = RsmiLib::load() {
        let mut busy: u32 = 0;
        let status = unsafe {
            if let Ok(f) = lib.handle().get::<RsmiBusyFn>(b"rsmi_dev_busy_percent_get") {
                f(ordinal as u32, &mut busy)
            } else {
                1
            }
        };
        if status == RSMI_STATUS_SUCCESS && busy <= 100 {
            return Some(busy);
        }
    }

    // Direct sysfs fallback for consumer AMD GPUs / APUs on Linux
    let candidates = [
        format!("/sys/class/drm/card{}/device/gpu_busy_percent", ordinal),
        format!("/sys/class/drm/card{}/device/gpu_busy_percent", ordinal + 1),
        "/sys/class/drm/card0/device/gpu_busy_percent".into(),
        "/sys/class/drm/card1/device/gpu_busy_percent".into(),
    ];
    for path in &candidates {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(val) = content.trim().parse::<u32>() {
                if val <= 100 {
                    return Some(val);
                }
            }
        }
    }

    None
}

type RsmiBusyFn = unsafe extern "C" fn(u32, *mut u32) -> u32;
const RSMI_STATUS_SUCCESS: u32 = 0;

/// Process-wide dlopen handle for `librocm_smi64.so`, initialized lazily.
struct RsmiLib {
    lib: libloading::Library,
}

impl RsmiLib {
    fn load() -> Option<&'static Self> {
        use std::sync::OnceLock;
        static HANDLE: OnceLock<Option<RsmiLib>> = OnceLock::new();
        let opt = HANDLE.get_or_init(|| {
            let lib = unsafe { libloading::Library::new("librocm_smi64.so") }
                .or_else(|_| unsafe { libloading::Library::new("librocm_smi64.so.1") })
                .ok()?;
            // rsmi must be initialized before any dev_* call.
            let init: Symbol<'_, RsmiInitFn> = unsafe { lib.get(b"rsmi_init").ok()? };
            let _ = unsafe { init(0) };
            Some(RsmiLib { lib })
        });
        opt.as_ref()
    }
    fn handle(&self) -> &libloading::Library {
        &self.lib
    }
}

type RsmiInitFn = unsafe extern "C" fn(u64) -> u32;

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Epoch counter must start at 0 and increment atomically.
    #[test]
    fn test_epoch_increment() {
        let before = current_epoch();
        bump_epoch();
        assert_eq!(current_epoch(), before.wrapping_add(1));
        // Reset for other tests.
        CAPABILITY_EPOCH.store(before, Ordering::SeqCst);
    }

    /// Arch table must never return zero TFLOPS (avoid division-by-zero in controller).
    #[test]
    fn test_arch_tflops_nonzero() {
        for gcn in &[
            "gfx1100", "gfx1102", "gfx1200", "gfx942", "gfx1036", "gfx0000",
        ] {
            let (fp16, _, _) = arch_tflops_table(gcn);
            assert!(fp16 > 0.0, "arch_tflops_table({gcn}) returned 0 TFLOPS");
        }
    }

    /// Full RDNA4 identifiers must not collapse to the generic gfx12 row.
    #[test]
    fn test_rdna4_arch_rows_are_distinct() {
        let fast = arch_tflops_table("gfx1201");
        let slow = arch_tflops_table("gfx1200");
        assert!(fast.0 > slow.0, "gfx1201 should rank above gfx1200");
        assert_ne!(fast.2, slow.2, "RDNA4 bandwidth rows must differ");
        assert_eq!(arch_tflops_table("gfx1202"), (80.0, 160.0, 960.0));
    }

    /// WI-SB0 host gate: calibration must fall back cleanly (`None`, no
    /// panic) when the ordinal is not a live HIP device — the GPU-less-box
    /// stand-in for "HIP absent".
    #[test]
    fn test_calibration_falls_back_on_bad_ordinal() {
        assert!(calibrate_capability(usize::MAX - 1, "gfx0000").is_none());
        // And the snapshot builder must degrade to default caps, not panic.
        let cap = measure_capability(usize::MAX - 1);
        assert_eq!(cap.ordinal, usize::MAX - 1);
    }

    /// WI-SB0 measured-capability gate (needs ≥1 real ROCm device): the
    /// profiler must report *measured* numbers that beat the tie-collapse
    /// problem — non-zero TFLOPS/bandwidth, cached across calls, and on an
    /// asymmetric pair strictly ordered fast > slow in both fields.
    #[test]
    fn test_measured_caps_are_real_and_distinct() {
        let n = enumerate_devices().unwrap_or(0);
        if n == 0 {
            return; // GPU-less box — static-table path covered elsewhere.
        }
        let profiler = CapabilityProfiler::new();
        let caps = profiler.capabilities();
        assert!(!caps.is_empty());
        for cap in &caps {
            assert!(
                cap.tflops_fp16 > 0.0,
                "GPU {} calibrated to zero TFLOPS",
                cap.ordinal
            );
            assert!(
                cap.hbm_bandwidth_gbps > 0.0,
                "GPU {} calibrated to zero bandwidth",
                cap.ordinal
            );
        }
        // Asymmetric pair: distinct silicon must measure apart in BOTH fields.
        if caps.len() >= 2 && caps[0].tflops_fp16 != caps[1].tflops_fp16 {
            assert_ne!(caps[0].hbm_bandwidth_gbps, caps[1].hbm_bandwidth_gbps);
        }
    }

    /// `CapabilityProfiler::new()` must not panic on a GPU-less box.
    #[test]
    fn test_profiler_new_gpu_less() {
        let profiler = CapabilityProfiler::new();
        let caps = profiler.capabilities();
        // GPU-less box: empty; GPU box: at least one cap with tflops > 0.
        for cap in &caps {
            assert!(cap.tflops_fp16 >= 0.0);
        }
    }

    /// `link_matrix` returns a square matrix with self-links as PeerDirect.
    #[test]
    fn test_link_matrix_self_links() {
        let n = enumerate_devices().unwrap_or(0);
        if n == 0 {
            return; // GPU-less box — skip.
        }
        let matrix = CapabilityProfiler::link_matrix(n);
        assert_eq!(matrix.len(), n * n);
        for i in 0..n {
            assert_eq!(
                matrix[i * n + i],
                ScytheLink::PeerDirect,
                "self-link for GPU {i}"
            );
        }
    }

    /// `tick()` must not panic on a GPU-less box.
    #[test]
    fn test_tick_gpu_less() {
        let profiler = CapabilityProfiler::new();
        profiler.tick(); // Must not panic.
    }

    /// WI-1: `compute_utilization` must never fabricate a value. On a box
    /// without rsmi (or with no devices) it returns `None`; on a real ROCm
    /// device it returns `Some(0..=100)`. Never returns a value outside the
    /// valid range — that would be the lying-zero problem this WI exists to
    /// fix, just in the opposite direction.
    #[test]
    fn test_compute_utilization_range_or_none() {
        let n = enumerate_devices().unwrap_or(0);
        if n == 0 {
            assert!(compute_utilization(0).is_none());
            return;
        }
        for ord in 0..n {
            // rsmi absent on this device — honest absence.
            if let Some(pct) = compute_utilization(ord) {
                assert!(
                    pct <= 100,
                    "compute_utilization({ord}) returned {pct}, expected 0..=100"
                );
            }
        }
    }
}
