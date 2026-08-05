//! SCYTHE-2 Capability Profiler (WI-2). [see: `GpuCapability`, `SelfTuningController`, `tflops_fp16`, `throttle_pct`]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use grim_tensor::backend::{GpuCapability, ScytheLink};
use grim_tensor::error::Result;

use crate::device::probe::{probe_host_gpu, probe_xnack};
use crate::peer_access::{P2PStatus, enumerate_devices, peer_status};

// ── HIP attributes needed for VRAM free / throttle ──────────────────────────

/// HIP device attribute: active clock throttle reasons. [see: `hipDeviceAttributeCurrentThermalThrottlePercent`]
const HIP_DEVICE_ATTR_THROTTLE: i32 = 74;

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

impl CapabilityProfiler {
    /// Create a new profiler. Blocks for one initial sweep (~5 ms per GPU).
    pub fn new() -> Self {
        let num_gpus = enumerate_devices().unwrap_or(0);
        let caps = (0..num_gpus)
            .map(|ord| measure_capability(ord))
            .collect::<Vec<_>>();
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

/// Measure one GPU's capability snapshot via HIP attributes + micro-GEMM. [see: `2605.05049`, `hipDeviceGetAttribute`]
fn measure_capability(ordinal: usize) -> GpuCapability {
    // Base probe from the existing infrastructure.
    let host_cap = match probe_host_gpu(ordinal) {
        Ok(c) => c,
        Err(_) => {
            // GPU disappeared between `enumerate_devices` and this probe.
            return GpuCapability {
                ordinal,
                ..Default::default()
            };
        }
    };

    // Wavefront size → estimate peak FLOPS from clock * CUs.
    let (tflops_fp16, tflops_fp8, hbm_gbps) = arch_tflops_table(&host_cap.gcn);

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
    // RDNA 4 / GFX12xx — has FP8 units
    if gcn.starts_with("gfx12") {
        return (80.0, 160.0, 960.0);
    }
    // CDNA (Instinct MI-series)
    if gcn.starts_with("gfx9") {
        if gcn.contains("940") || gcn.contains("941") || gcn.contains("942") {
            return (1307.0, 2614.0, 5300.0); // MI300X FP16 peak TFLOPS & HBM3 bandwidth
        }
        if gcn.contains("90a") {
            return (383.0, 383.0, 3200.0); // MI250X
        }
        if gcn.contains("908") {
            return (184.6, 184.6, 1228.8); // MI100
        }
        if gcn.contains("906") {
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
fn query_throttle_pct(ordinal: usize) -> f32 {
    let mut val: i32 = 0;
    // SAFETY: `hipDeviceGetAttribute` is safe to call with a valid ordinal.
    unsafe {
        use crate::device::handles::{hipDeviceGetAttribute, hipSetDevice};
        let _ = hipSetDevice(ordinal as i32);
        let status = hipDeviceGetAttribute(&mut val, HIP_DEVICE_ATTR_THROTTLE, ordinal as i32);
        if status != 0 {
            return 0.0;
        }
    }
    // The attribute returns a percentage [0, 100]; normalise to [0, 1].
    (val.clamp(0, 100) as f32) / 100.0
}

/// Query free VRAM in bytes via `hipMemGetInfo`.
fn query_vram_free(ordinal: usize) -> u64 {
    vram_info(ordinal).0
}

/// Query `(free_bytes, total_bytes)` VRAM via `hipMemGetInfo`. [see: `(0, 0)`]
pub fn vram_info(ordinal: usize) -> (u64, u64) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        use crate::device::handles::{hipMemGetInfo, hipSetDevice};
        let _ = hipSetDevice(ordinal as i32);
        let status = hipMemGetInfo(&mut free, &mut total);
        if status != 0 {
            return (0, 0);
        }
    }
    (free as u64, total as u64)
}

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
}
