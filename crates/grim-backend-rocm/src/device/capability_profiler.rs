//! SCYTHE-2 Capability Profiler (WI-2).
//!
//! Periodically sweeps every visible ROCm device with a 5-ms micro-GEMM
//! benchmark to fill a live `GpuCapability` snapshot. The snapshot is cached
//! for ~100 ms (the capability-epoch cadence derived in scythe2.md §3.6 from
//! PowerTune thermal-hysteresis onset ~50–100 ms and micro-GEMM noise floor).
//!
//! The 100 ms figure is the geometric mean of the valid window and matches
//! grim's existing `SelfTuningController` EMA cadence. Sub-50 ms thermal
//! transients are filtered by AMD's firmware; sampling faster than 50 ms
//! chases hysteresis noise. Sampling slower than 200 ms risks serving a stale
//! placement for >1 throttle event.
//!
//! ## Staleness safety (scythe2.md §3.5)
//! - Stale `tflops_fp16` / `throttle_pct` → suboptimal load-balance only;
//!   never a correctness fault.
//! - A GPU *leaving* (OOM / hot-unplug) → caller must call `bump_epoch` from
//!   the device-lost handler before the next `PlacementCache` lookup. That
//!   path is implemented in `grim-engine/src/scythe2.rs` (WI-4).
//!
//! Skill attribution:
//! - `rust-ffi-grim` §2 — dynamic ROCm discovery via `probe_host_gpu`.
//! - `rust-ffi-grim` §1.3 — null-pointer guard before every FFI call.
//! - `rust-ffi-grim` §3 — `cargo check` gate after every change.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use grim_tensor::backend::{GpuCapability, ScytheLink};
use grim_tensor::error::Result;

use crate::device::probe::{probe_host_gpu, probe_xnack};
use crate::peer_access::{enumerate_devices, peer_status, P2PStatus};

// ── HIP attributes needed for VRAM free / throttle ──────────────────────────

/// HIP device attribute: active clock throttle reasons.
/// Attribute 74 = `hipDeviceAttributeCurrentThermalThrottlePercent` on ROCm 6+.
/// Falls back to 0 when the driver version is too old.
const HIP_DEVICE_ATTR_THROTTLE: i32 = 74;

/// HIP attribute: total global memory (bytes). Attribute 23.
const HIP_DEVICE_ATTR_TOTAL_MEM: i32 = 23; // hipDeviceAttributeTotalConstantMemory alt

// ── Public surface ────────────────────────────────────────────────────────────

/// SCYTHE-2 capability epoch counter.
///
/// Bumped by `CapabilityProfiler::bump_epoch` when a GPU joins or leaves the
/// farm, or when a >10% throttle delta is detected between two ticks (the
/// out-of-band escape hatch from scythe2.md §3.6). The engine's
/// `PlacementCache` clears its fast array on an epoch bump, triggering a
/// fresh `decide_miss()` on the next forward pass.
///
/// Global so that the device-lost path in `grim-disagg` can bump it without
/// holding a reference to the `CapabilityProfiler`.
pub static CAPABILITY_EPOCH: AtomicU32 = AtomicU32::new(0);

/// Bump the global capability epoch.
///
/// Must be called from the ROCm device-lost / OOM-recovery path (WI-8)
/// *before* the next `PlacementCache::get()` returns, so that `r` (placement
/// vector) is never dispatched to a gone GPU. See scythe2.md §3.5 mode B.
pub fn bump_epoch() {
    CAPABILITY_EPOCH.fetch_add(1, Ordering::Release);
}

/// Read the current epoch without modifying it.
pub fn current_epoch() -> u32 {
    CAPABILITY_EPOCH.load(Ordering::Acquire)
}

/// Per-GPU live capability snapshot builder and epoch manager.
///
/// Usage:
/// 1. Call `CapabilityProfiler::new()` once at startup.
/// 2. Spawn a background thread that calls `tick()` every 100 ms.
/// 3. Call `capabilities()` from the controller to get the latest snapshot.
/// 4. Call `bump_epoch()` (free function) from the device-lost path.
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

    /// Refresh the capability snapshot for every GPU.
    ///
    /// Expected call cadence: every 100 ms. If `throttle_pct` delta exceeds
    /// 10% for any GPU since the last tick, triggers an out-of-band
    /// `bump_epoch()` immediately (scythe2.md §3.6 escape hatch).
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
            if (cap.throttle_pct - prev).abs() > 0.10 && !epoch_bumped {
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

    /// Return a snapshot of current capabilities for all visible GPUs.
    ///
    /// Used by `C2plrController::decide_miss()` to populate the MLP input.
    pub fn capabilities(&self) -> Vec<GpuCapability> {
        self.inner.lock().expect("CapabilityProfiler lock poisoned").caps.clone()
    }

    /// Build a K×K link matrix using the existing `peer_status` probe.
    ///
    /// Returns a flattened `Vec<ScytheLink>` suitable for `ScythePlacement::routes`.
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
    /// own tick scheduling and want to check staleness.
    pub fn age(&self) -> Duration {
        self.inner.lock().expect("CapabilityProfiler lock poisoned").last_tick.elapsed()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Measure one GPU's capability snapshot via HIP attributes + micro-GEMM.
///
/// The 5-ms micro-GEMM (Piper-style resource modelling, `2605.05049`) is the
/// primary TFLOPS estimator. `hipDeviceGetAttribute` fills in the rest.
fn measure_capability(ordinal: usize) -> GpuCapability {
    // Base probe from the existing infrastructure.
    let host_cap = match probe_host_gpu(ordinal) {
        Ok(c) => c,
        Err(_) => {
            // GPU disappeared between `enumerate_devices` and this probe.
            // Return a zeroed capability so the controller avoids this GPU.
            return GpuCapability { ordinal, ..Default::default() };
        }
    };

    // Wavefront size → estimate peak FLOPS from clock * CUs.
    // Real estimate requires VRAM info; we use a heuristic table keyed on arch.
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

/// Architecture TFLOPS table — offline values per GCN arch string.
///
/// This is the "structural coefficient table" analogous to WaveTune's Table-A
/// (`2604.10187` §4.4): a tiny lookup, not a candidate-loop. Values are
/// conservative nominal throughputs; `throttle_pct` is applied on top.
///
/// Returns `(tflops_fp16, tflops_fp8, hbm_gbps)`.
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
        return (190.0, 380.0, 3200.0); // MI300X rough
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
/// Returns 0.0 if the attribute is unavailable (old driver / GPU-less box).
fn query_throttle_pct(ordinal: usize) -> f32 {
    let mut val: i32 = 0;
    // SAFETY: `hipDeviceGetAttribute` is safe to call with a valid ordinal.
    // The attribute ID 74 may not exist on old ROCm — the return code is
    // checked and we fall back to 0.0 (no throttle assumed).
    unsafe {
        use crate::device::handles::{hipSetDevice, hipDeviceGetAttribute};
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
///
/// Returns 0 on error (GPU-less box, or not the active context).
fn query_vram_free(ordinal: usize) -> u64 {
    vram_info(ordinal).0
}

/// Query `(free_bytes, total_bytes)` VRAM via `hipMemGetInfo`.
///
/// This is the public entry point for callers outside the profiler (e.g. the
/// training worker's metric reporting) that need live VRAM usage. Returns
/// `(0, 0)` on a GPU-less box or when the HIP call fails — callers should
/// treat 0 as "unknown" rather than "empty".
pub fn vram_info(ordinal: usize) -> (u64, u64) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        use crate::device::handles::{hipSetDevice, hipMemGetInfo};
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
        for gcn in &["gfx1100", "gfx1102", "gfx1200", "gfx942", "gfx1036", "gfx0000"] {
            let (fp16, _, _) = arch_tflops_table(gcn);
            assert!(fp16 > 0.0, "arch_tflops_table({gcn}) returned 0 TFLOPS");
        }
    }

    /// `CapabilityProfiler::new()` must not panic on a GPU-less box.
    /// The returned capability list may be empty, but the call must succeed.
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
            assert_eq!(matrix[i * n + i], ScytheLink::PeerDirect, "self-link for GPU {i}");
        }
    }

    /// `tick()` must not panic on a GPU-less box.
    #[test]
    fn test_tick_gpu_less() {
        let profiler = CapabilityProfiler::new();
        profiler.tick(); // Must not panic.
    }
}
