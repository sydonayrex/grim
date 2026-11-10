//! ROCm/HIP hybrid MoE decode executor with bandwidth-adaptive CPU-GPU co-execution.
//! When serving frontier MoE models (DeepSeek-V4, Qwen3.6-MoE, GLM-5.2) on consumer/workstation AMD GPUs, the full expert.
//!
//! astrum P3 (gated, default OFF): route-prior/residency/trace plumbing composes
//! `grim_memory::moe_budget` primitives (`ExpertRouteTrace`, `ResidentProfileFreq`).

use std::sync::atomic::{AtomicU32, Ordering};

use grim_memory::moe_budget::{ExpertRouteTrace, ResidentProfileFreq};
use grim_tensor::error::{Error, Result};

/// Mapped pinned memory flag for HIP-graph-compatible CPU/GPU handshake without CUDA/HIP host synchronization stalls.
/// # Layout Contract Annotated `#[repr(C, align(64))]` to fit a standard cache line and prevent false.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct MoeGraphSyncFlag {
    /// Raised to 1 by GPU/stream when activations are ready for CPU worker consumption.
    pub ready_flag: AtomicU32,
    /// Raised to 1 by CPU worker coordinator when partial sums are ready for GPU reduction.
    pub done_flag: AtomicU32,
    /// Layer index currently in flight.
    pub layer_idx: AtomicU32,
    /// Error code if CPU worker encountered an unexpected failure (0 = OK).
    pub error_code: AtomicU32,
}

impl Default for MoeGraphSyncFlag {
    fn default() -> Self {
        Self {
            ready_flag: AtomicU32::new(0),
            done_flag: AtomicU32::new(0),
            layer_idx: AtomicU32::new(0),
            error_code: AtomicU32::new(0),
        }
    }
}

impl MoeGraphSyncFlag {
    /// Create a new initialized synchronization flag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset flags before enqueuing next decode step.
    pub fn reset(&self, layer: usize) {
        self.ready_flag.store(0, Ordering::Release);
        self.done_flag.store(0, Ordering::Release);
        self.layer_idx.store(layer as u32, Ordering::Release);
        self.error_code.store(0, Ordering::Release);
    }

    /// Signal CPU workers that input activations are staged.
    pub fn signal_gpu_ready(&self) {
        self.ready_flag.store(1, Ordering::Release);
    }

    /// Check if GPU has raised the ready signal.
    pub fn is_gpu_ready(&self) -> bool {
        self.ready_flag.load(Ordering::Acquire) == 1
    }

    /// Signal GPU that CPU partial execution is complete.
    pub fn signal_cpu_done(&self) {
        self.done_flag.store(1, Ordering::Release);
    }

    /// Check if CPU workers have finished.
    pub fn is_cpu_done(&self) -> bool {
        self.done_flag.load(Ordering::Acquire) == 1
    }
}

/// Partition plan for an MoE layer decode step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeHybridExecutionPlan {
    /// Layer index being evaluated.
    pub layer_idx: usize,
    /// Experts already resident in GPU LRU cache (Set $\mathcal{H}$).
    pub gpu_resident_experts: Vec<usize>,
    /// Missing experts to DMA stream over PCIe into GPU cache (Set $\mathcal{F}$, size $q^*$).
    pub gpu_fill_experts: Vec<usize>,
    /// Missing experts to compute in-place on CPU from host memory (Set $\mathcal{C}$, size $m - q^*$).
    pub cpu_compute_experts: Vec<usize>,
}

impl MoeHybridExecutionPlan {
    /// Returns total number of active experts evaluated across GPU and CPU.
    pub fn total_active_experts(&self) -> usize {
        self.gpu_resident_experts.len()
            + self.gpu_fill_experts.len()
            + self.cpu_compute_experts.len()
    }

    /// Returns `true` if CPU co-execution is required for this step.
    pub fn has_cpu_work(&self) -> bool {
        !self.cpu_compute_experts.is_empty()
    }

    /// Returns `true` if GPU PCIe transfer is required for missing experts.
    pub fn has_gpu_fills(&self) -> bool {
        !self.gpu_fill_experts.is_empty()
    }
}

/// astrum P3: env-gated residency/trace/route-prior config (default OFF).
/// Composes `grim_memory::moe_budget` primitives without changing default math.
#[derive(Debug, Clone, Default)]
pub struct AstrumMoeResidencyConfig {
    /// `GRIM_ASTRUM_EXPERT_RESIDENT=<N>` slots per layer (0/unset => off).
    pub resident_slots: usize,
    /// `GRIM_ASTRUM_ROUTE_PRIOR=<lambda>` (0/unset => no selection bonus).
    pub route_prior_lambda: f32,
    /// `GRIM_ASTRUM_EXPERT_TRACE=1` enables in-memory trace recording hooks.
    pub trace_enabled: bool,
}

impl AstrumMoeResidencyConfig {
    pub fn from_env() -> Self {
        let resident_slots = std::env::var("GRIM_ASTRUM_EXPERT_RESIDENT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let route_prior_lambda = std::env::var("GRIM_ASTRUM_ROUTE_PRIOR")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);
        let trace_enabled = std::env::var("GRIM_ASTRUM_EXPERT_TRACE")
            .ok()
            .map(|v| {
                let t = v.trim();
                !(t.is_empty() || t == "0" || t.eq_ignore_ascii_case("false") || t.eq_ignore_ascii_case("off"))
            })
            .unwrap_or(false);
        Self {
            resident_slots,
            route_prior_lambda,
            trace_enabled,
        }
    }

    pub fn enabled(&self) -> bool {
        self.resident_slots > 0 || self.route_prior_lambda > 0.0 || self.trace_enabled
    }
}

/// astrum P3: mutable residency/trace state for MoE dispatch planning.
/// Default OFF: when disabled, consults fall back to caller-provided residency and record is a no-op.
#[derive(Debug, Clone, Default)]
pub struct AstrumMoeResidencyState {
    pub config: AstrumMoeResidencyConfig,
    /// Optional frequency-ranked profile built from trace (selection-only residency/prior input).
    pub profile: Option<ResidentProfileFreq>,
    /// Optional activation trace (layer, expert hit counts).
    pub trace: Option<ExpertRouteTrace>,
}

impl AstrumMoeResidencyState {
    pub fn from_env(layers: usize, experts: usize) -> Self {
        let config = AstrumMoeResidencyConfig::from_env();
        let trace = if config.trace_enabled && layers > 0 && experts > 0 {
            Some(ExpertRouteTrace::new(layers, experts))
        } else {
            None
        };
        Self {
            config,
            profile: None,
            trace,
        }
    }

    /// Rebuild frequency-ranked profile from current trace (if any).
    pub fn rebuild_profile_from_trace(&mut self) {
        let n = self.config.resident_slots;
        self.profile = match (&self.trace, n > 0) {
            (Some(t), true) => Some(ResidentProfileFreq::from_trace(t, n)),
            _ => None,
        };
    }

    /// Record routed experts for `layer_idx` into trace (no-op when tracing disabled/unavailable).
    pub fn record_routed(&mut self, layer_idx: usize, routed_experts: &[usize]) {
        if let Some(trace) = self.trace.as_mut() {
            for &e in routed_experts {
                trace.record(layer_idx, e);
            }
        }
    }

    /// Record routed expert ids for `layer_idx` from an iterator (e.g. device-readback or assignment pairs).
    pub fn record_routed_iter<I>(&mut self, layer_idx: usize, experts: I)
    where
        I: IntoIterator<Item = usize>,
    {
        if let Some(trace) = self.trace.as_mut() {
            for e in experts {
                trace.record(layer_idx, e);
            }
        }
    }

    /// Selection-only route bonus for (layer, expert). Returns 0.0 when disabled/profile absent.
    pub fn route_bonus(&self, layer_idx: usize, expert: usize) -> f32 {
        let lambda = self.config.route_prior_lambda;
        if lambda <= 0.0 {
            return 0.0;
        }
        match &self.profile {
            Some(p) => p.route_bonus(layer_idx, expert, self.config.resident_slots.max(1), lambda),
            None => 0.0,
        }
    }

    /// Consult residency: profile-driven when available, else fallback.
    pub fn is_resident_or<F>(&self, layer_idx: usize, expert: usize, fallback: F) -> bool
    where
        F: FnOnce(usize) -> bool,
    {
        if self.config.resident_slots > 0 {
            if let Some(p) = &self.profile {
                return p.is_resident(layer_idx, expert);
            }
        }
        fallback(expert)
    }
}


/// Hybrid MoE execution coordinator for ROCm/HIP devices.
pub struct MoeHybridExecutor {
    /// PCIe transfer bandwidth in MB/s (B_P).
    pub pcie_bandwidth_mbps: f64,
    /// Host CPU expert GEMV bandwidth in MB/s (B_H).
    pub host_bandwidth_mbps: f64,
}

impl MoeHybridExecutor {
    /// Create a new hybrid executor with empirical bandwidth measurements.
    /// # Contract `pcie_bandwidth_mbps` and `host_bandwidth_mbps` must be positive.
    pub fn new(pcie_bandwidth_mbps: f64, host_bandwidth_mbps: f64) -> Self {
        assert!(pcie_bandwidth_mbps > 0.0, "pcie_bandwidth_mbps must be > 0");
        assert!(host_bandwidth_mbps > 0.0, "host_bandwidth_mbps must be > 0");
        Self {
            pcie_bandwidth_mbps,
            host_bandwidth_mbps,
        }
    }

    /// Plan the hybrid execution partition for a token's routed top-k experts.
    /// # Contract `resident_check` returns `true` if expert `e` is currently resident in GPU cache.
    pub fn plan_step<F>(
        &self,
        layer_idx: usize,
        routed_experts: &[usize],
        is_resident: F,
    ) -> MoeHybridExecutionPlan
    where
        F: FnMut(usize) -> bool,
    {
        self.plan_step_inner(layer_idx, routed_experts, is_resident)
    }

    /// astrum P3: plan using a frequency-ranked residency profile when present (gated/default OFF).
    /// Falls back to `is_resident_fallback` for experts/layers not covered by the profile.
    pub fn plan_step_with_profile<F>(
        &self,
        layer_idx: usize,
        routed_experts: &[usize],
        profile: Option<&ResidentProfileFreq>,
        resident_slots: usize,
        mut is_resident_fallback: F,
    ) -> MoeHybridExecutionPlan
    where
        F: FnMut(usize) -> bool,
    {
        let use_profile = profile.is_some() && resident_slots > 0;
        self.plan_step_inner(layer_idx, routed_experts, |e| {
            if use_profile {
                if let Some(p) = profile {
                    return p.is_resident(layer_idx, e);
                }
            }
            is_resident_fallback(e)
        })
    }

    /// astrum P3: plan and optionally record routed experts into trace (default OFF).
    pub fn plan_step_traced<F>(
        &self,
        layer_idx: usize,
        routed_experts: &[usize],
        trace: Option<&mut ExpertRouteTrace>,
        is_resident: F,
    ) -> MoeHybridExecutionPlan
    where
        F: FnMut(usize) -> bool,
    {
        if let Some(t) = trace {
            for &e in routed_experts {
                t.record(layer_idx, e);
            }
        }
        self.plan_step_inner(layer_idx, routed_experts, is_resident)
    }

    /// astrum P3: compose residency/trace state (profile consult + optional recording) in one call.
    pub fn plan_step_with_astrum<F>(
        &self,
        layer_idx: usize,
        routed_experts: &[usize],
        state: Option<&mut AstrumMoeResidencyState>,
        mut is_resident_fallback: F,
    ) -> MoeHybridExecutionPlan
    where
        F: FnMut(usize) -> bool,
    {
        match state {
            Some(st) if st.config.enabled() => {
                st.record_routed(layer_idx, routed_experts);
                // consult profile if available, else fallback
                if st.config.resident_slots > 0 {
                    if let Some(profile) = st.profile.as_ref() {
                        let p = profile;
                        return self.plan_step_inner(layer_idx, routed_experts, |e| {
                            p.is_resident(layer_idx, e)
                        });
                    }
                }
                self.plan_step_inner(layer_idx, routed_experts, |e| is_resident_fallback(e))
            }
            Some(st) => {
                // tracing might be enabled independently? config.enabled covers it, but be safe
                st.record_routed(layer_idx, routed_experts);
                self.plan_step_inner(layer_idx, routed_experts, is_resident_fallback)
            }
            None => self.plan_step_inner(layer_idx, routed_experts, is_resident_fallback),
        }
    }

    fn plan_step_inner<F>(
        &self,
        layer_idx: usize,
        routed_experts: &[usize],
        mut is_resident: F,
    ) -> MoeHybridExecutionPlan
    where
        F: FnMut(usize) -> bool,
    {
        let mut resident = Vec::new();
        let mut misses = Vec::new();

        for &exp in routed_experts {
            if is_resident(exp) {
                if !resident.contains(&exp) {
                    resident.push(exp);
                }
            } else if !misses.contains(&exp) {
                misses.push(exp);
            }
        }

        let m = misses.len();
        if m == 0 {
            return MoeHybridExecutionPlan {
                layer_idx,
                gpu_resident_experts: resident,
                gpu_fill_experts: Vec::new(),
                cpu_compute_experts: Vec::new(),
            };
        }

        // Apply FreeToken q* closed-form policy: q* = round(m * B_P / B_H)
        let ratio = (self.pcie_bandwidth_mbps / self.host_bandwidth_mbps).clamp(0.0, 1.0);
        let q_raw = (m as f64 * ratio).round() as usize;
        let q = q_raw.clamp(1, m);

        let gpu_fills = misses[..q].to_vec();
        let cpu_computes = misses[q..].to_vec();

        MoeHybridExecutionPlan {
            layer_idx,
            gpu_resident_experts: resident,
            gpu_fill_experts: gpu_fills,
            cpu_compute_experts: cpu_computes,
        }
    }

    /// Concurrent execution of hybrid MoE step: CPU workers process Set $\mathcal{C}$ while GPU processes Set $\mathcal{H} \cup \mathcal{F}$, synchronized via atomic flags.
    /// # Contract Launches CPU worker execution, executes GPU closure in parallel, waits for CPU completion.
    pub fn execute_hybrid_step<G>(
        &self,
        plan: &MoeHybridExecutionPlan,
        sync_flag: &MoeGraphSyncFlag,
        cpu_worker_pool: &grim_backend_cpu::PersistentMoeWorkerPool,
        tokens: &[f32],
        routed_indices: &[usize],
        routed_weights: &[f32],
        w_gate: &[Vec<f32>],
        w_up: &[Vec<f32>],
        w_down: &[Vec<f32>],
        num_tokens: usize,
        hidden_dim: usize,
        inter_dim: usize,
        num_experts: usize,
        top_k: usize,
        gpu_exec_fn: G,
    ) -> Result<Vec<f32>>
    where
        G: FnOnce(&[usize]) -> Result<Vec<f32>>,
    {
        sync_flag.reset(plan.layer_idx);

        // Form combined GPU execution set G = H ∪ F
        let mut gpu_active_experts = plan.gpu_resident_experts.clone();
        gpu_active_experts.extend_from_slice(&plan.gpu_fill_experts);

        let has_cpu_work = plan.has_cpu_work();

        std::thread::scope(|s| {
            // 1. Launch CPU worker branch if set C is non-empty
            let cpu_handle = if has_cpu_work {
                sync_flag.signal_gpu_ready();
                let cpu_assigned = &plan.cpu_compute_experts;

                Some(s.spawn(move || {
                    cpu_worker_pool.dispatch_partial(
                        tokens,
                        routed_indices,
                        routed_weights,
                        cpu_assigned,
                        w_gate,
                        w_up,
                        w_down,
                        num_tokens,
                        hidden_dim,
                        inter_dim,
                        num_experts,
                        top_k,
                    )
                }))
            } else {
                None
            };

            // 2. Concurrently execute GPU path
            let mut gpu_out = gpu_exec_fn(&gpu_active_experts)?;

            // 3. Join CPU worker branch and signal sync_flag
            if let Some(handle) = cpu_handle {
                let cpu_out = handle
                    .join()
                    .map_err(|_| Error::Backend("CPU MoE worker thread panicked".to_string()))??;
                sync_flag.signal_cpu_done();
                Self::merge_outputs(&mut gpu_out, &cpu_out)?;
            }

            Ok(gpu_out)
        })
    }

    /// Exact additive merge of GPU partial sums and CPU partial sums ($y = y_{\text{GPU}} + y_{\text{CPU}}$).
    /// # Contract `gpu_out` and `cpu_out` must have identical lengths.
    pub fn merge_outputs(gpu_out: &mut [f32], cpu_out: &[f32]) -> Result<()> {
        if gpu_out.len() != cpu_out.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![gpu_out.len()],
                got: vec![cpu_out.len()],
            });
        }
        for (g, &c) in gpu_out.iter_mut().zip(cpu_out.iter()) {
            *g += c;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flag_handshake_transitions() {
        let flag = MoeGraphSyncFlag::new();
        assert!(!flag.is_gpu_ready());
        assert!(!flag.is_cpu_done());

        flag.signal_gpu_ready();
        assert!(flag.is_gpu_ready());
        assert!(!flag.is_cpu_done());

        flag.signal_cpu_done();
        assert!(flag.is_gpu_ready());
        assert!(flag.is_cpu_done());

        flag.reset(1);
        assert!(!flag.is_gpu_ready());
        assert!(!flag.is_cpu_done());
    }

    #[test]
    fn test_hybrid_plan_partitioning() {
        // PCIe 25 GB/s, Host 100 GB/s -> ratio 0.25
        let executor = MoeHybridExecutor::new(25_000.0, 100_000.0);
        let routed = vec![0, 1, 2, 3, 4];

        // Expert 0 is resident, 1..4 miss (m = 4)
        let plan = executor.plan_step(0, &routed, |e| e == 0);
        assert_eq!(plan.gpu_resident_experts, vec![0]);
        // q* = round(4 * 0.25) = 1
        assert_eq!(plan.gpu_fill_experts, vec![1]);
        // m - q* = 3
        assert_eq!(plan.cpu_compute_experts, vec![2, 3, 4]);
        assert!(plan.has_cpu_work());
        assert!(plan.has_gpu_fills());
        assert_eq!(plan.total_active_experts(), 5);
    }

    #[test]
    fn test_merge_outputs_exact_sum() {
        let mut gpu_out = vec![1.0, 2.0, 3.0, 4.0];
        let cpu_out = vec![0.5, 0.5, 0.5, 0.5];
        MoeHybridExecutor::merge_outputs(&mut gpu_out, &cpu_out).unwrap();
        assert_eq!(gpu_out, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn astrum_p3_trace_records_and_profile_consults_residency() {
        let executor = MoeHybridExecutor::new(25_000.0, 50_000.0);
        let mut trace = ExpertRouteTrace::new(2, 8);
        // route: expert 3,7 hot on layer 0 (strict frequency ranking)
        for _ in 0..40 {
            executor.plan_step_traced(0, &[3, 7, 1], Some(&mut trace), |_| false);
        }
        for _ in 0..30 {
            executor.plan_step_traced(0, &[3], Some(&mut trace), |_| false);
        }
        for _ in 0..10 {
            executor.plan_step_traced(0, &[7], Some(&mut trace), |_| false);
        }
        assert!(trace.count(0, 3) > trace.count(0, 7));
        assert!(trace.count(0, 7) > trace.count(0, 1));
        let profile = ResidentProfileFreq::from_trace(&trace, 2);
        assert!(profile.is_resident(0, 3));
        assert!(profile.is_resident(0, 7));
        assert!(!profile.is_resident(0, 1));
        let plan = executor.plan_step_with_profile(0, &[3, 7, 1], Some(&profile), 2, |_| false);
        // 3,7 resident, 1 miss -> split by q*
        assert!(plan.gpu_resident_experts.contains(&3));
        assert!(plan.gpu_resident_experts.contains(&7));
        assert_eq!(plan.total_active_experts(), 3);
    }

    #[test]
    fn astrum_p3_route_bonus_gated() {
        let mut st = AstrumMoeResidencyState::default();
        assert_eq!(st.route_bonus(0, 0), 0.0);
        st.config.route_prior_lambda = 2.0;
        st.config.resident_slots = 2;
        // no profile => 0
        assert_eq!(st.route_bonus(0, 0), 0.0);
        let mut t = ExpertRouteTrace::new(1, 4);
        for _ in 0..10 { t.record(0, 1); }
        st.trace = Some(t);
        st.rebuild_profile_from_trace();
        assert!(st.route_bonus(0, 1) > 0.0);
        assert_eq!(st.route_bonus(0, 2), 0.0);
    }
}
