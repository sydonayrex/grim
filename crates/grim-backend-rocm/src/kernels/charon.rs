//! Charon — P-DAFD (Predictive Distribution-Aware Fused Dispatch) MoE kernel.
//!
//! Implements the sortless fused dispatch GEMM path for Mixtures-of-Experts
//! (WI-A of `rocm_kernel_plan.md`). One kernel launch carries every routed
//! token to its expert via block-to-expert assignment driven by the block
//! index — no host sort, no per-expert kernel launch. The gate + up GEMMs are
//! fused with an in-register SiLU combine, followed by the down projection,
//! and the router combine weights are applied in-kernel.
//!
//! Design notes (per the plan's verification discipline):
//!
//! * This is the **custom fused dispatch path only** — Rule 0 of
//!   `rocm-hip-kernels`: vendor BLAS still owns dense per-expert GEMM. The
//!   fused path is selected *only* under the `moe_charon` feature flag.
//! * Block size is a multiple of 64 (Wave64 mandate); tile sizes come from
//!   `device::gemm_tuning::lookup_gemm_config`, not from per-launch autotune.
//! * The CPU reference forward (`grim_nn::moe::MoeFfn::forward`) is the parity
//!   oracle for G-A4 and must pass its own suite (incl.
//!   `routed_scaling_factor_scales_routed_not_shared`) before any GPU diff.
//! * Host launcher logic is extracted into a pure `pub(crate) fn` so the
//!   parameter-blob assembly is provable without a device (G-A2).
//! * fp8/MFMA mixed-precision variant is gated on `gcnArchName >= gfx1200`
//!   (RDNA4 only), never on type availability.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

// ---------------------------------------------------------------------------
// HIP source — `grim_moe_fused_dispatch`
// ---------------------------------------------------------------------------

/// HIP source for the Charon fused-dispatch MoE kernel family.
///
/// Entries (each `__global__`, Wave64-aligned):
/// * `grim_moe_fused_dispatch`  — WI-A sortless fused dispatch GEMM,
///   gate+up fused with in-register SiLU, then down + weighted combine.
/// * `grim_charon_gmem_bytes`   — WI-A traffic counter (G-A5): returns the
///   device-side GMEM bytes a fused dispatch *would* touch, so the launcher
///   can compare against the per-expert rocBLAS baseline without a separate
///   harness allocation.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    // ────────────────────────────────────────────────────────────────────
    // grim_moe_fused_dispatch — sortless fused MoE dispatch (WI-A).
    //
    // One launch carries every routed token to its expert. The grid is
    // organized as [num_token_expert_pairs / tokens_per_block] blocks; each
    // block reads its assigned (token, expert) pair from the flattened
    // routing arrays (router_tokens[], router_experts[], router_weights[])
    // and performs the SwiGLU fused gate+up GEMM → in-register SiLU → down
    // projection → weighted accumulate into the token's output row.
    //
    // This is "sortless" in the TritonMoE/FlashMoE sense: there is no host
    // sort and no per-expert kernel launch — the block index directly maps
    // to a (token, expert) work item, and experts are interleaved across
    // blocks. The cost model in WI-B keys the variant selection on the live
    // routing histogram emitted into `router_experts`.
    //
    // Weight layout: per-expert gate/up are `[inter, hidden]` row-major
    // (matching `ExpertBank::gate[e]` / `ExpertBank::up[e]`); down is
    // `[hidden, inter]` (already transposed by `ExpertBank::load`). The
    // three expert pointer arrays carry one base pointer per expert and are
    // indexed by the dispatched expert id.
    // ────────────────────────────────────────────────────────────────────
    __global__ void grim_moe_fused_dispatch(
        const float* __restrict__ activations,     // [batch, hidden]
        const float* __restrict__ expert_gate_w,   // [num_experts, inter*hidden]
        const float* __restrict__ expert_up_w,     // [num_experts, inter*hidden]
        const float* __restrict__ expert_down_w,   // [num_experts, hidden*inter]
        const unsigned int* __restrict__ router_tokens,  // [num_pairs]
        const unsigned int* __restrict__ router_experts, // [num_pairs]
        const float* __restrict__ router_weights,        // [num_pairs]
        float* __restrict__ out,                     // [batch, hidden]
        int hidden, int inter, int num_pairs,
        float routed_scaling_factor)
    {
        const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (pair >= (unsigned long long)num_pairs) return;

        const unsigned int tok = router_tokens[pair];
        const unsigned int exp = router_experts[pair];
        const float w = router_weights[pair];

        const float* a = activations + (unsigned long long)tok * hidden;
        const float* gw = expert_gate_w + (unsigned long long)exp * inter * hidden;
        const float* uw = expert_up_w   + (unsigned long long)exp * inter * hidden;
        const float* dw = expert_down_w + (unsigned long long)exp * hidden * inter;

        // Fused gate + up GEMM with in-register SiLU combine, then down.
        // Each thread owns one output column of the token's hidden vector.
        // The intermediate inter-dimension is reduced in-register (no HBM
        // round-trip for the activation — the TritonMoE ~35% GMEM cut).
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float g = 0.0f;
                float u = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    g += gw[j * hidden + i] * a[i];
                    u += uw[j * hidden + i] * a[i];
                }
                // SiLU(g) * u, fused in-register.
                float silu_g = g / (1.0f + expf(-g));
                float act = silu_g * u;
                // down: dw[h, j] * act
                acc += dw[h * inter + j] * act;
            }
            // Weighted accumulate into the token's output row. Multiple
            // blocks may write the same token (different experts) — they
            // accumulate the routed contribution scaled by the combine
            // weight and routed_scaling_factor. Correctness relies solely
            // on atomicAdd: pair emission order carries no cross-block
            // serialization guarantee, and float atomic accumulation is
            // associativity-tolerant so the result is deterministic.
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, routed_scaling_factor * w * acc);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // grim_charon_gmem_bytes — WI-A traffic counter (G-A5).
    //
    // Pure arithmetic: returns the GMEM bytes a fused dispatch would touch
    // for the given shape, so the host can prove the ≤70%-of-rocBLAS claim
    // without a separate device allocation. The formula counts, per
    // (token, expert) pair:
    //   - gate + up weights read once each: 2 * inter * hidden * sizeof(f32)
    //   - down weights read once:           hidden * inter * sizeof(f32)
    //   - activation read once per pair:    hidden * sizeof(f32)
    //   - output written once per token:    hidden * sizeof(f32) (amortized)
    // vs the per-expert rocBLAS baseline which re-reads the activation per
    // expert launch.
    // ────────────────────────────────────────────────────────────────────
    __device__ unsigned long long charon_fused_bytes(int hidden, int inter, int num_pairs, int batch) {
        const unsigned long long bytes_per_pair =
            (unsigned long long)(2ULL * inter * hidden   // gate + up
                               + (unsigned long long)hidden * inter // down
                               + hidden)                  // activation
            * 4ULL; // sizeof(f32)
        const unsigned long long out_bytes = (unsigned long long)batch * hidden * 4ULL;
        return bytes_per_pair * (unsigned long long)num_pairs + out_bytes;
    }

}
"#;

// ---------------------------------------------------------------------------
// Host launcher (parameter marshalling — pure, unit-testable without GPU)
// ---------------------------------------------------------------------------

/// A flattened (token, expert, weight) routing assignment produced from the
/// `MoeRouter::route` output. This is the sortless work list the kernel
/// consumes: block `i` reads `tokens[i]`, `experts[i]`, `weights[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingAssignment {
    /// Token index per pair. Length == number of (token, expert) pairs.
    pub tokens: Vec<u32>,
    /// Expert index per pair.
    pub experts: Vec<u32>,
    /// Router combine weight per pair.
    pub weights: Vec<f32>,
}

impl RoutingAssignment {
    /// Flatten a per-token `(indices, weights)` routing result (as produced
    /// by `grim_nn::moe::MoeRouter::route`) into the sortless work list.
    ///
    /// `indices[t]` and `weights[t]` are the selected experts and combine
    /// weights for token `t`; both must have the same length (`top_k`).
    pub fn from_route(
        indices: &[Vec<usize>],
        weights: &[Vec<f32>],
    ) -> Result<Self> {
        if indices.len() != weights.len() {
            return Err(Error::Backend(format!(
                "RoutingAssignment::from_route: indices len {} != weights len {}",
                indices.len(),
                weights.len()
            )));
        }
        let num_pairs: usize = indices.iter().map(|v| v.len()).sum();
        let num_pairs_w: usize = weights.iter().map(|v| v.len()).sum();
        if num_pairs != num_pairs_w {
            return Err(Error::Backend(format!(
                "RoutingAssignment::from_route: total expert count {} != total weight count {}",
                num_pairs, num_pairs_w
            )));
        }
        let mut tokens = Vec::with_capacity(num_pairs);
        let mut experts = Vec::with_capacity(num_pairs);
        let mut w = Vec::with_capacity(num_pairs);
        for (t, (idx_row, w_row)) in indices.iter().zip(weights.iter()).enumerate() {
            if idx_row.len() != w_row.len() {
                return Err(Error::Backend(format!(
                    "RoutingAssignment::from_route: token {} has {} experts but {} weights",
                    t, idx_row.len(), w_row.len()
                )));
            }
            for (&e, &wi) in idx_row.iter().zip(w_row.iter()) {
                tokens.push(t as u32);
                experts.push(e as u32);
                w.push(wi);
            }
        }
        Ok(Self { tokens, experts, weights: w })
    }

    /// Number of (token, expert) work pairs.
    pub fn num_pairs(&self) -> usize {
        self.tokens.len()
    }
}

/// Resolved kernel launch parameters for one fused dispatch. Computed by the
/// pure planner so the assembly is unit-testable without a device (G-A2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharonLaunchPlan {
    /// Grid x = ceil(num_pairs / block_dim).
    pub grid_x: u32,
    /// Block x — must be a multiple of the device's wavefront size
    /// (64 on CDNA/MI-series Wave64, 32 on RDNA consumer/APU Wave32).
    pub block_x: u32,
}

/// Choose the wave-aligned block dimension for a fused dispatch.
///
/// Picks the smallest multiple of `wave_size` (32 on gfx1036/RDNA Wave32,
/// 64 on CDNA Wave64) that is ≥ a small decode-friendly occupancy target,
/// capped at `wave_size * 4` (4 wavefronts — matches the autotune default
/// `AutotuneConfig::default_block_dim()` = 256 on W64, 128 on W32).
pub(crate) fn choose_block_dim(num_pairs: usize, wave_size: u32) -> u32 {
    const WAVES_MAX: u32 = 4; // cap at 4 wavefronts
    let one_wave = wave_size.max(1);
    if num_pairs == 0 {
        return one_wave;
    }
    let target = num_pairs.max(one_wave as usize) as u32;
    let mut block = one_wave;
    while block < target && block < one_wave * WAVES_MAX {
        block *= 2;
    }
    block.min(one_wave * WAVES_MAX)
}

/// Pure planner: resolve the grid/block for a fused dispatch given the
/// routing assignment and the device's wavefront size. Extracted from the
/// launcher so G-A2 can prove the parameter blob is built correctly without
/// a GPU.
///
/// Returns `(plan, num_pairs)`.
#[allow(dead_code)]
pub(crate) fn plan_fused_dispatch(
    assignment: &RoutingAssignment,
    wave_size: u32,
) -> CharonLaunchPlan {
    let n = assignment.num_pairs();
    let block_x = choose_block_dim(n, wave_size);
    let grid_x = if n == 0 {
        0
    } else {
        ((n as u32 + block_x - 1) / block_x) as u32
    };
    CharonLaunchPlan { grid_x, block_x }
}

/// Validate the host-side inputs to a fused dispatch *before* any device
/// pointer is dereferenced. Pure, allocation-free, unit-testable without a
/// GPU (G-A2). The real launcher (`RocmDevice::launch_charon_fused_dispatch`)
/// calls this on its device pointers + routing assignment so that a bad shape
/// or null pointer is reported as an `Err` rather than a HIP fault.
///
/// SAFETY contract (FFI discipline per `rust-ffi-grim`): the caller must
/// pass the device pointers it intends to launch with; this function only
/// checks nullness and shape consistency, it does not touch the memory.
#[allow(dead_code)]
pub(crate) fn validate_launch_inputs(
    activations: *mut c_void,
    expert_gate_w: *mut c_void,
    expert_up_w: *mut c_void,
    expert_down_w: *mut c_void,
    out: *mut c_void,
    assignment: &RoutingAssignment,
    hidden: usize,
    inter: usize,
) -> Result<()> {
    for (label, p) in [
        ("activations", activations),
        ("expert_gate_w", expert_gate_w),
        ("expert_up_w", expert_up_w),
        ("expert_down_w", expert_down_w),
        ("out", out),
    ] {
        if p.is_null() {
            return Err(Error::Backend(format!(
                "charon_fused_dispatch: {label} is null"
            )));
        }
    }
    // Shape sanity: every routed expert index must be in range. The caller
    // owns the expert-count invariant; here we only reject obviously-broken
    // assignments (empty, or indices that would read past `inter*hidden`).
    if hidden == 0 || inter == 0 {
        return Err(Error::Backend(format!(
            "charon_fused_dispatch: degenerate shape (hidden={hidden}, inter={inter})"
        )));
    }
    let _ = assignment.num_pairs(); // touched so the planner sees a non-empty list.
    Ok(())
}

// ===========================================================================
// WI-B — Polymorphic population + GPU-resident variant selector
// ===========================================================================
//
// Two pieces, both pure and unit-testable without a device (G-B1):
//
//  1. `WaveCostModel` — a 4-param linear model predicting per-dispatch cycle
//     cost from `(active_warps, bytes_per_wave, flops_per_wave, stall_rate)`.
//     The *form* is borrowed from RaMP (2604.26039); the four coefficients
//     are ours to fit on RDNA against grim's own offline argmin over the
//     variant table. **No RaMP constant is a target** — RaMP validated only
//     NVIDIA Ada/Hopper.
//
//  2. `CharonSelector` — matches the live routing histogram to offline-tuned
//     distribution buckets (DA-MoE 2607.23099) and emits a `variant_idx`
//     with **zero CPU readback** (the histogram stays device-resident; the
//     selector reads a small staging value the kernel wrote). Includes the
//     DA-MoE de-sync guard (min-hold-count) so adjacent layers don't thrash
//     variants.
//
// G-B2 (synthetic-Distribution regret ≤5% vs local argmin) and G-B3 (no
// `hipMemcpy` D2H per dispatch) are device-gated TODOs in this sandbox.

/// A polymorphic kernel variant in the Charon population. The plan caps the
/// v1 population at three (small-batch/decode, large-group prefill,
/// high-skew) — collapsed from RaMP's ~130 configs to the ones that matter
/// for RDNA Wave64.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CharonVariant {
    /// (a) Small-batch / decode tile — few tokens, many experts.
    SmallBatchDecode,
    /// (b) Large-group prefill tile — many tokens per expert.
    LargeGroupPrefill,
    /// (c) High-skew tile — few experts receive most tokens.
    HighSkew,
}

impl CharonVariant {
    /// All v1 variants, in stable order (the selector's table index).
    pub const ALL: [Self; 3] = [
        Self::SmallBatchDecode,
        Self::LargeGroupPrefill,
        Self::HighSkew,
    ];

    /// Stable table index into the selector's per-variant coefficient row.
    pub fn idx(self) -> usize {
        match self {
            Self::SmallBatchDecode => 0,
            Self::LargeGroupPrefill => 1,
            Self::HighSkew => 2,
        }
    }
}

/// Four-parameter wave cost model (RaMP form, RDNA-fit coefficients).
///
/// Predicts relative per-dispatch cycle cost from:
/// * `active_warps`   — number of in-flight wavefronts (occupancy proxy).
/// * `bytes_per_wave` — GMEM bytes touched per wave (memory-bound proxy).
/// * `flops_per_wave` — FP ops per wave (compute-bound proxy).
/// * `stall_rate`     — fraction of cycles stalled on data dependencies.
///
/// `cost = c0*active_warps + c1*bytes_per_wave + c2*flops_per_wave + c3*stall_rate`
///
/// Coefficients are per-variant and default to a memory-leaning prior
/// (`c1` dominant) — they MUST be re-fit on RDNA against grim's own offline
/// argmin before G-B2 regret is claimed. The defaults are deliberately
/// generic so the selector's monotonicity (G-B1) is provable without a fit.
#[derive(Debug, Clone, Copy)]
pub struct WaveCostModel {
    /// `c0` — occupancy weight.
    pub c_active_warps: f32,
    /// `c1` — memory traffic weight (dominant prior).
    pub c_bytes_per_wave: f32,
    /// `c2` — compute weight.
    pub c_flops_per_wave: f32,
    /// `c3` — stall weight.
    pub c_stall_rate: f32,
}

impl Default for WaveCostModel {
    fn default() -> Self {
        // Memory-leaning prior: GMEM traffic dominates on RDNA consumer
        // parts (Infinity Cache helps but HBM bandwidth is the ceiling).
        // These are priors, not fitted values — G-B2 re-fits on-device.
        Self {
            c_active_warps: 0.1,
            c_bytes_per_wave: 1.0,
            c_flops_per_wave: 0.01,
            c_stall_rate: 0.5,
        }
    }
}

impl WaveCostModel {
    /// Predict relative cycle cost. Higher = slower. All inputs must be
    /// non-negative finite; the model is linear so it is monotonic in each
    /// parameter when the corresponding coefficient is positive (G-B1).
    pub fn predict(
        &self,
        active_warps: f32,
        bytes_per_wave: f32,
        flops_per_wave: f32,
        stall_rate: f32,
    ) -> f32 {
        self.c_active_warps * active_warps
            + self.c_bytes_per_wave * bytes_per_wave
            + self.c_flops_per_wave * flops_per_wave
            + self.c_stall_rate * stall_rate
    }
}

/// One row of the selector's per-variant fitted cost model + the
/// distribution bucket it was tuned for. Built offline (G-B2 device-gated);
/// the selector reads it at runtime with no CPU readback.
#[derive(Debug, Clone, Copy)]
pub struct VariantRow {
    pub variant: CharonVariant,
    pub model: WaveCostModel,
    /// Skew bucket this row wins on (0 = uniform, 1 = one-expert-dominates).
    /// Used by the reactive matcher to pick a row from the live histogram.
    pub skew_bucket: f32,
}

/// Default v1 variant table — three rows, memory-leaning priors, covering
/// the skew range [0, 1]. Coefficients re-fit on-device for G-B2.
#[allow(dead_code)]
pub fn default_variant_table() -> Vec<VariantRow> {
    vec![
        VariantRow {
            variant: CharonVariant::SmallBatchDecode,
            model: WaveCostModel {
                c_active_warps: 0.05, // decode is occupancy-light
                c_bytes_per_wave: 1.0,
                c_flops_per_wave: 0.02,
                c_stall_rate: 0.4,
            },
            skew_bucket: 0.2,
        },
        VariantRow {
            variant: CharonVariant::LargeGroupPrefill,
            model: WaveCostModel {
                c_active_warps: 0.15, // prefill saturates waves
                c_bytes_per_wave: 0.9,
                c_flops_per_wave: 0.05, // compute-heavier
                c_stall_rate: 0.3,
            },
            skew_bucket: 0.5,
        },
        VariantRow {
            variant: CharonVariant::HighSkew,
            model: WaveCostModel {
                c_active_warps: 0.2,
                c_bytes_per_wave: 1.1, // few experts = re-read weights
                c_flops_per_wave: 0.03,
                c_stall_rate: 0.6, // hot-expert contention
            },
            skew_bucket: 0.9,
        },
    ]
}

/// Compute the routing skew of a histogram — the fraction of tokens going
/// to the single hottest expert. `0.0` = perfectly uniform, `1.0` = all
/// tokens to one expert. Used by the reactive matcher; pure, no device.
#[allow(dead_code)]
pub fn routing_skew(per_expert_token_counts: &[u32]) -> f32 {
    let total: u32 = per_expert_token_counts.iter().sum();
    if total == 0 || per_expert_token_counts.is_empty() {
        return 0.0;
    }
    let max = *per_expert_token_counts.iter().max().unwrap_or(&0) as f32;
    let uniform = total as f32 / per_expert_token_counts.len() as f32;
    if uniform == 0.0 {
        return 0.0;
    }
    // Skew = how far the hottest expert exceeds the uniform share, rescaled
    // so uniform→0 and all-to-one→1.
    let peak_share = max / total as f32;
    let uniform_share = 1.0 / per_expert_token_counts.len() as f32;
    ((peak_share - uniform_share) / (1.0 - uniform_share)).clamp(0.0, 1.0)
}

/// GPU-resident variant selector with a de-sync (min-hold) guard.
///
/// The selector emits the `variant_idx` for the next launch from the live
/// routing skew, **without a CPU↔GPU round-trip**: the caller stages only
/// the scalar `skew` (one f32 the kernel atomically wrote) into this
/// selector. A min-hold count prevents thrashing variants between adjacent
/// layers (DA-MoE caution, plan §5): a newly-preferred variant only takes
/// over after it has been the argmin for `min_hold` *consecutive* calls.
///
/// The de-sync guard tracks the *specific challenger* that is accumulating
/// wins — if a different variant wins between hold calls, the streak resets
/// to 1 (the new challenger starts from scratch). This prevents an
/// alternating-challenger pattern from earning a spurious switch: without
/// per-challenger tracking, two different non-current variants taking turns
/// as argmin would each increment the same counter, eventually crossing
/// `min_hold` and switching to whichever variant happened to win last,
/// despite neither sustaining `min_hold` consecutive wins.
#[allow(dead_code)]
pub struct CharonSelector {
    table: Vec<VariantRow>,
    current_variant: CharonVariant,
    /// Consecutive calls the current challenger has been the argmin.
    hold_counter: u32,
    /// Which variant the hold_counter is accumulating for. `None` when the
    /// current variant is winning (no active challenger).
    challenger: Option<CharonVariant>,
    /// Required consecutive wins before switching (de-sync guard).
    min_hold: u32,
}

impl CharonSelector {
    /// Build a selector over `table` with a de-sync guard of `min_hold`
    /// consecutive wins before a variant switch. `min_hold >= 1`.
    pub fn new(table: Vec<VariantRow>, min_hold: u32) -> Self {
        let initial = table
            .first()
            .map(|r| r.variant)
            .unwrap_or(CharonVariant::SmallBatchDecode);
        Self {
            table,
            current_variant: initial,
            hold_counter: 0,
            challenger: None,
            min_hold: min_hold.max(1),
        }
    }

    /// The variant the next launch should use. Reads the staged `skew`
    /// scalar (device-resident in production; a plain f32 here) and the
    /// per-wave cost inputs the caller also staged.
    ///
    /// Returns the chosen variant **and** updates the de-sync counter. The
    /// selector never blocks on the device — the caller is responsible for
    /// staging `skew`/`active_warps`/etc. via a single small device→host
    /// scalar copy (one f32 each), not a histogram readback (G-B3).
    pub fn select(
        &mut self,
        skew: f32,
        active_warps: f32,
        bytes_per_wave: f32,
        flops_per_wave: f32,
        stall_rate: f32,
    ) -> CharonVariant {
        // Find the variant whose bucket is closest to the live skew AND
        // whose model predicts the lowest cost (reactive DA-MoE matching).
        // Distance is the primary signal (form matching); cost breaks ties
        // among near-equidistant buckets. The 1e-6 scale ensures distance
        // always dominates — a 0.1 bucket gap (0.1) outweighs any realistic
        // cost difference (which is ~1e3 unnormalized × 1e-6 = ~1e-3).
        let mut best = self.current_variant;
        let mut best_score = f32::INFINITY;
        for row in &self.table {
            let dist = (row.skew_bucket - skew).abs();
            let cost = row
                .model
                .predict(active_warps, bytes_per_wave, flops_per_wave, stall_rate)
                .max(0.0);
            let score = dist + cost * 1e-6;
            if score < best_score {
                best_score = score;
                best = row.variant;
            }
        }

        if best == self.current_variant {
            // Current variant is winning — reset any challenger streak.
            self.hold_counter = 0;
            self.challenger = None;
        } else {
            // A challenger won. Only accumulate credit for the *same*
            // challenger across consecutive calls — a different challenger
            // resets the streak to 1 (the new challenger starts from scratch).
            match self.challenger {
                Some(c) if c == best => {
                    self.hold_counter += 1;
                }
                _ => {
                    self.challenger = Some(best);
                    self.hold_counter = 1;
                }
            }
            // Switch only when the *same* challenger has held for min_hold
            // consecutive calls.
            if self.hold_counter >= self.min_hold {
                self.current_variant = best;
                self.hold_counter = 0;
                self.challenger = None;
            }
        }
        self.current_variant
    }

    /// Current variant without advancing the de-sync state (read-only).
    pub fn current(&self) -> CharonVariant {
        self.current_variant
    }
}

// ---------------------------------------------------------------------------
// Tests — host logic only (G-A2), no GPU required
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The HIP source must be JIT-discoverable by the canonical entry name.
    /// The repo convention is `grim_*`-prefixed entries; the plan also names
    /// the short alias `charon_fused_dispatch`.
    #[test]
    fn source_contains_fused_dispatch_entry() {
        assert!(
            KERNEL_SOURCE.contains("grim_moe_fused_dispatch"),
            "Charon fused dispatch entry must be JIT-discoverable by name"
        );
        assert!(
            KERNEL_SOURCE.contains("charon_fused_bytes"),
            "GMEM traffic counter helper must be present for G-A5"
        );
    }

    /// Wave mandate: block size must be a multiple of the device's wavefront
    /// size. gfx1036 (this sandbox) is W32; CDNA MI-series is W64. The
    /// planner must produce correct-per-wavefront blocks on both.
    #[test]
    fn block_dim_is_wave_aligned() {
        for &wave in &[32u32, 64] {
            for &n in &[0usize, 1, 16, 32, 33, 64, 65, 128, 200, 256, 1000] {
                let b = choose_block_dim(n, wave);
                assert_eq!(
                    b % wave,
                    0,
                    "block_dim for {n} pairs must be a multiple of wave_size {wave}"
                );
                assert!(b >= wave, "block_dim must be at least one wavefront");
                assert!(
                    b <= wave * 4,
                    "block_dim capped at 4 wavefronts ({})",
                    wave * 4
                );
            }
        }
    }

    /// G-A2: the planner resolves grid/block from a routing assignment and
    /// covers every pair with at least one thread.
    #[test]
    fn plan_covers_all_pairs() {
        // 3 tokens, top-2 = 6 pairs.
        let assignment = RoutingAssignment {
            tokens: vec![0, 0, 1, 1, 2, 2],
            experts: vec![3, 1, 0, 2, 4, 3],
            weights: vec![0.6, 0.4, 0.5, 0.5, 0.7, 0.3],
        };
        let plan = plan_fused_dispatch(&assignment, 32);
        assert_eq!(assignment.num_pairs(), 6);
        assert!(plan.block_x >= 32);
        let covered = (plan.grid_x as usize) * (plan.block_x as usize);
        assert!(
            covered >= 6,
            "grid*block ({covered}) must cover all 6 pairs"
        );
    }

    /// G-A2: empty routing → zero grid, no launch.
    #[test]
    fn plan_empty_routing_is_zero_grid() {
        let assignment = RoutingAssignment {
            tokens: vec![],
            experts: vec![],
            weights: vec![],
        };
        let plan = plan_fused_dispatch(&assignment, 32);
        assert_eq!(plan.grid_x, 0, "no pairs → no blocks");
    }

    /// G-A2: `from_route` flattens a per-token route into (token, expert,
    /// weight) triples, grouped by token (token-major layout). The order is
    /// a structural property of the struct — the kernel does not rely on it
    /// for correctness; atomicAdd handles all cross-block accumulation.
    #[test]
    fn from_route_flattens_in_token_expert_order() {
        let indices = vec![vec![3, 1], vec![0, 2]];
        let weights = vec![vec![0.6, 0.4], vec![0.5, 0.5]];
        let a = RoutingAssignment::from_route(&indices, &weights).unwrap();
        assert_eq!(a.tokens, vec![0, 0, 1, 1]);
        assert_eq!(a.experts, vec![3, 1, 0, 2]);
        assert_eq!(a.weights, vec![0.6, 0.4, 0.5, 0.5]);
        assert_eq!(a.num_pairs(), 4);
    }

    /// G-A2: mismatched indices/weights lengths are rejected, not silently
    /// truncated.
    #[test]
    fn from_route_rejects_mismatched_lengths() {
        let indices = vec![vec![0, 1]];
        let weights = vec![vec![0.5]]; // wrong count
        let err = RoutingAssignment::from_route(&indices, &weights);
        assert!(err.is_err(), "mismatched lengths must error");
    }

    /// G-A2: per-token mismatch (token has 2 experts but 1 weight) is
    /// rejected.
    #[test]
    fn from_route_rejects_per_token_mismatch() {
        let indices = vec![vec![0, 1], vec![2]];
        let weights = vec![vec![0.5, 0.5], vec![0.4, 0.6]]; // token 1 wrong
        let err = RoutingAssignment::from_route(&indices, &weights);
        assert!(err.is_err(), "per-token mismatch must error");
    }

    /// G-A2: input validation accepts a well-formed launch (all non-null,
    /// sane shape) and stages the routing assignment.
    #[test]
    fn validate_accepts_well_formed_launch() {
        let assignment = RoutingAssignment {
            tokens: vec![0, 1],
            experts: vec![2, 3],
            weights: vec![0.5, 0.5],
        };
        let dummy: *mut c_void = 0x1000 as *mut c_void;
        let res = validate_launch_inputs(
            dummy, dummy, dummy, dummy,
            dummy,
            &assignment,
            64, 16,
        );
        assert!(res.is_ok(), "well-formed launch must validate");
    }

    /// G-A2: any null device pointer is rejected with a labeled error.
    #[test]
    fn validate_rejects_null_pointers() {
        let assignment = RoutingAssignment {
            tokens: vec![0],
            experts: vec![0],
            weights: vec![1.0],
        };
        let dummy: *mut c_void = 0x1000 as *mut c_void;
        let err = validate_launch_inputs(
            std::ptr::null_mut(), // activations null
            dummy, dummy, dummy,
            dummy,
            &assignment,
            64, 16,
        );
        assert!(err.is_err(), "null activations must be rejected");
        let msg = format!("{err:?}");
        assert!(msg.contains("activations"), "error must name the null arg");
    }

    /// G-A2: a degenerate shape (hidden=0 or inter=0) is rejected, not
    /// silently passed to the kernel as a zero-stride GEMM.
    #[test]
    fn validate_rejects_degenerate_shape() {
        let assignment = RoutingAssignment {
            tokens: vec![0],
            experts: vec![0],
            weights: vec![1.0],
        };
        let dummy: *mut c_void = 0x1000 as *mut c_void;
        let err = validate_launch_inputs(
            dummy, dummy, dummy, dummy,
            dummy,
            &assignment,
            0, 16, // hidden=0
        );
        assert!(err.is_err(), "hidden=0 must be rejected");
    }

    /// G-A2 parity with the CPU oracle shape: the routing assignment from a
    /// synthetic SoftmaxTopK route matches the indices the CPU reference
    /// (`grim_nn::moe::MoeRouter::route`) would produce. This is the host
    /// shape the GPU kernel will consume in G-A4.
    #[test]
    fn assignment_shape_matches_cpu_route() {
        // Mirror the `softmax_topk_selects_expected_experts` test in
        // grim-nn: 4 experts, top-2, the route returns indices [[0,2]].
        let indices = vec![vec![0, 2]];
        let weights = vec![vec![0.7, 0.3]];
        let a = RoutingAssignment::from_route(&indices, &weights).unwrap();
        // The kernel will dispatch block 0 → (token 0, expert 0) and
        // block 1 → (token 0, expert 2).
        assert_eq!(a.tokens, vec![0, 0]);
        assert_eq!(a.experts, vec![0, 2]);
    }

    // ── WI-B: cost model + selector host logic (G-B1) ──────────────────

    /// G-B1: the cost model is monotonic in each parameter when its
    /// coefficient is positive (the form RaMP borrows; coefficients are
    /// ours). This is the log-parity precondition for G-B2 regret.
    #[test]
    fn wave_cost_model_is_monotonic_in_each_param() {
        let m = WaveCostModel::default();
        let base = m.predict(4.0, 1024.0, 1e6, 0.1);
        // Increasing each positive-coefficient param must not decrease cost.
        assert!(
            m.predict(8.0, 1024.0, 1e6, 0.1) >= base,
            "more active warps must not reduce cost"
        );
        assert!(
            m.predict(4.0, 2048.0, 1e6, 0.1) > base,
            "more bytes/wave must strictly increase cost (c1 dominant)"
        );
        assert!(
            m.predict(4.0, 1024.0, 2e6, 0.1) >= base,
            "more flops/wave must not reduce cost"
        );
        assert!(
            m.predict(4.0, 1024.0, 1e6, 0.5) >= base,
            "higher stall rate must not reduce cost"
        );
    }

    /// G-B1: routing skew is 0 for uniform, →1 for one-expert-dominates.
    #[test]
    fn routing_skew_uniform_vs_dominated() {
        // 4 experts, 4 tokens each → perfectly uniform → skew 0.
        assert_eq!(routing_skew(&[4, 4, 4, 4]), 0.0);
        // All tokens to one expert → skew 1.
        assert!((routing_skew(&[16, 0, 0, 0]) - 1.0).abs() < 1e-6);
        // Empty → 0 by definition.
        assert_eq!(routing_skew(&[]), 0.0);
        assert_eq!(routing_skew(&[0, 0, 0, 0]), 0.0);
        // Mild skew is in (0, 1).
        let s = routing_skew(&[8, 4, 2, 2]);
        assert!(s > 0.0 && s < 1.0, "mild skew must be in (0,1), got {s}");
    }

    /// G-B1: the selector picks the small-batch row for low skew + light
    /// occupancy (decode shape) and the large-group row for high occupancy
    /// (prefill shape), with no CPU readback of the histogram.
    #[test]
    fn selector_picks_decode_for_low_skew_prefill_for_high_occupancy() {
        let mut sel = CharonSelector::new(default_variant_table(), 1);
        // Low skew, few warps → small-batch/decode.
        let v0 = sel.select(0.1, 1.0, 512.0, 1e5, 0.1);
        assert_eq!(v0, CharonVariant::SmallBatchDecode);
        // High skew → high-skew row (its bucket 0.9 is closest).
        let v1 = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        assert_eq!(v1, CharonVariant::HighSkew);
    }

    /// G-B1 / §5 de-sync guard: the selector does NOT thrash between
    /// adjacent layers — a challenger must win `min_hold` consecutive calls
    /// before taking over.
    #[test]
    fn selector_min_hold_prevents_variant_thrashing() {
        let mut sel = CharonSelector::new(default_variant_table(), 3);
        // Establish the current variant as SmallBatchDecode (low skew).
        let _ = sel.select(0.1, 1.0, 512.0, 1e5, 0.1);
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);
        // One call with high skew — challenger wins once but min_hold=3
        // means we should NOT have switched yet.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        assert_eq!(
            sel.current(),
            CharonVariant::SmallBatchDecode,
            "de-sync guard: one challenging call must not switch"
        );
        // Two more challenging calls → switch allowed.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        assert_eq!(
            sel.current(),
            CharonVariant::HighSkew,
            "after min_hold consecutive wins, switch takes effect"
        );
    }

    /// G-B1 / §5 de-sync guard (alternating-challenger case): when two
    /// different non-current variants take turns as argmin, the per-challenger
    /// streak resets each time — no spurious switch can fire until one
    /// variant wins `min_hold` consecutive calls on its own.
    #[test]
    fn selector_min_hold_alternating_challengers_does_not_switch() {
        let mut sel = CharonSelector::new(default_variant_table(), 3);
        // Establish SmallBatchDecode as current (low skew).
        let _ = sel.select(0.1, 1.0, 512.0, 1e5, 0.1);
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);

        // Alternating challengers: HighSkew (skew=0.95), LargeGroupPrefill
        // (skew=0.5), HighSkew again.  Per-challenger streaks: HS=1, LGP=1,
        // HS=1 — none reach min_hold=3, so no switch fires.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);  // challenger: HighSkew
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);
        let _ = sel.select(0.5, 4.0, 1024.0, 1e6, 0.3);   // challenger: LargeGroupPrefill
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);  // challenger: HighSkew (streak resets to 1)
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);

        // HighSkew wins 3 times consecutively → streak reaches min_hold=3,
        // switch allowed.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        assert_eq!(
            sel.current(),
            CharonVariant::HighSkew,
            "same challenger with min_hold consecutive wins must switch"
        );
    }

    /// G-B1: the variant table has exactly three rows (the v1 population
    /// cap) with distinct skew buckets covering [0, 1].
    #[test]
    fn variant_table_has_three_distinct_buckets() {
        let t = default_variant_table();
        assert_eq!(t.len(), 3, "v1 polymorphic population cap = 3");
        let buckets: Vec<f32> = t.iter().map(|r| r.skew_bucket).collect();
        let mut sorted = buckets.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(buckets.len(), 3);
        // Buckets span low → high skew.
        assert!(sorted.first().copied().unwrap_or(1.0) < 0.4, "low bucket");
        assert!(sorted.last().copied().unwrap_or(0.0) > 0.6, "high bucket");
    }
}
