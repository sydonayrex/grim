//! GRAVE Phase 4 — fused GDN-2 HIP kernel source.
//!
//! `grim_gla_state_update_output`: ONE launch per GDL layer fusing the
//! rank-1 delta state update, the output matvec, AND the gated RMSNorm
//! (vLLM PR #53463 lesson: recurrent+output-norm fused is 10.5x/layer eager;
//! measured against GRAPHED Grim per the vLLM graphs-neutrality caveat).
//!
//! GDN-2 recurrence per head (`gla.rs` oracle, arXiv 2605.22791 Eq. 8–12):
//!   1. decay:  `S̃ = D·S`,  `D = Diag(α)` over key channels
//!   2. read:   `r = S̃ᵀ (b ⊙ k)`
//!   3. write:  `S' = S̃ + k (w ⊙ v − r)ᵀ`
//!   4. output: `o = S'ᵀ q`, then `y = RMSNorm(o) ⊙ σ(gate) ⊙ γ`
//!
//! Launch discipline (rocm-hip skill, RDNA4/gfx1201):
//! * grid = num_heads blocks, block = 64 threads (2× wave32). One block owns
//!   one head's `[64, 64]` state — fully LDS-tiled working set, no grid-wide
//!   barrier (only intra-block `__syncthreads`, which the M7 gate permits).
//! * NO `__shfl` usage at all — sidesteps the hiprtc 64-bit shfl-mask
//!   pitfall the plan's risk notes flag for every other new kernel.
//! * In-place state update; out-of-place normed output. No allocations,
//!   no host sync — capture-safe by construction (same contract as
//!   `qk_rope_dev_base`).

extern crate alloc;

/// Phase 4.5 gate: FP8 GDN-2 state store. Mirrors the `GRIM_F16_KV` pattern.
/// Default off; the launcher returns a clean error (never silent fallback)
// until the FP8 parity re-run lands.
pub fn gla_state_fp8_enabled() -> bool {
    std::env::var("GRIM_GLA_STATE_FP8")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

/// Escape hatch for one release cycle (plan guardrails): `GRIM_GLA_FUSED=0`
/// forces the split state-update/output path.
pub fn gla_fused_enabled() -> bool {
    std::env::var("GRIM_GLA_FUSED")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

/// HIP source for `grim_gla_state_update_output`.
pub const GLA_KERNEL_SOURCE: &str = r#"
extern "C" __global__ __launch_bounds__(64)
void grim_gla_state_update_output(
    const float* __restrict__ q,        // [H, dk] queries
    const float* __restrict__ k,        // [H, dk] keys
    const float* __restrict__ v,        // [H, dv] values
    const float* __restrict__ alpha,    // [H, dk] decay, (0, 1]
    const float* __restrict__ b_gate,   // [H, dk] erase gate, [0, 1]
    const float* __restrict__ w_gate,   // [H, dv] write gate, [0, 1]
    const float* __restrict__ norm_w,   // [H, dv] gated-norm weight gamma
    const float* __restrict__ out_gate, // [H] output-gate scalars (sigmoid applied here)
    float* __restrict__ state,          // [B, H, dk, dv] recurrent state, IN-PLACE
    float* __restrict__ out,            // [B, H, dv] normed gated output
    int dk,                             // key dim (64 fast path; strided loops beyond)
    int dv,                             // value dim (<= 64: LDS-resident r[]/o[])
    float eps                           // RMSNorm epsilon
    // Batching: grid = (H, B, 1); slot = b*H + h. Batch-1 decode is the
    // hot path; batch-4 parity is a plan G4 gate (same math, strided slots).
) {
    const int h = blockIdx.x;
    const int bb = blockIdx.y;
    const int tid = threadIdx.x;   // 0..63, wave32 => 2 waves
    if (dv > 64) return;           // host validates; never silent wrong math
    const size_t slot = (size_t)bb * gridDim.x + h;

    const float* qh = q + slot * dk;
    const float* kh = k + slot * dk;
    const float* vh = v + slot * dv;
    const float* ah = alpha + slot * dk;
    const float* bh = b_gate + slot * dk;
    const float* wh = w_gate + slot * dv;
    const float* gh = norm_w + slot * dv;
    float* sh = state + slot * (size_t)dk * dv;
    float* oh = out + slot * dv;

    __shared__ float s_e[64];   // erased key direction e_i = b_i * k_i
    __shared__ float s_r[64];   // gated read r_j
    __shared__ float s_o[64];   // output accumulator o_j
    __shared__ float s_k[64];   // cached k row for the engrave phase
    __shared__ float s_q[64];   // cached q row for the output phase

    // Phase 1: decay rows in place, publish e_i. Strided for dk > 64.
    for (int i = tid; i < dk; i += 64) {
        float ki = kh[i];
        float ei = bh[i] * ki;
        if (i < 64) { s_e[i] = ei; s_k[i] = ki; s_q[i] = qh[i]; }
        float a = ah[i];
        float* row = sh + (size_t)i * dv;
        for (int j = 0; j < dv; ++j) row[j] = a * row[j];
    }
    __syncthreads();

    // Phase 2: r_j = sum_i e_i * S~_ij. Thread j owns column j.
    for (int j = tid; j < dv; j += 64) {
        float acc = 0.0f;
        for (int i = 0; i < dk && i < 64; ++i) acc += s_e[i] * sh[(size_t)i * dv + j];
        // dk > 64 tail reads e directly (e beyond LDS recomputed cheaply).
        for (int i = 64; i < dk; ++i) acc += bh[i] * kh[i] * sh[(size_t)i * dv + j];
        s_r[j] = acc;
    }
    __syncthreads();

    // Phase 3: engrave S'[i][j] = S~ + k_i * (w_j v_j - r_j). Thread i owns row i.
    for (int i = tid; i < dk; i += 64) {
        float ki = (i < 64) ? s_k[i] : kh[i];
        float* row = sh + (size_t)i * dv;
        for (int j = 0; j < dv; ++j) {
            float delta = wh[j] * vh[j] - s_r[j];
            row[j] = row[j] + ki * delta;
        }
    }
    __syncthreads();

    // Phase 4: o_j = sum_i q_i S'_ij. Thread j owns column j.
    for (int j = tid; j < dv; j += 64) {
        float acc = 0.0f;
        for (int i = 0; i < dk && i < 64; ++i) acc += s_q[i] * sh[(size_t)i * dv + j];
        for (int i = 64; i < dk; ++i) acc += qh[i] * sh[(size_t)i * dv + j];
        s_o[j] = acc;
    }
    __syncthreads();

    // Phase 5: gated RMSNorm in place over s_o, then publish.
    // RMS over dv using 2-wave strided reduction through s_r as scratch.
    float ss = 0.0f;
    for (int j = tid; j < dv; j += 64) ss += s_o[j] * s_o[j];
    s_r[tid] = ss;
    __syncthreads();
    // Single-wave fold (tid < 32 active; wave32 => second wave idle here).
    if (tid < 32) {
        float x = s_r[tid] + s_r[tid + 32];
        s_r[tid] = x;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int j = 0; j < 32; ++j) sum += s_r[j];
        s_r[0] = rsqrtf(sum / (float)dv + eps);
    }
    __syncthreads();
    float inv = s_r[0];
    float gate = 1.0f / (1.0f + expf(-out_gate[slot]));
    for (int j = tid; j < dv; j += 64) {
        oh[j] = s_o[j] * inv * gh[j] * gate;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_kernel_present_with_single_launch_contract() {
        assert!(GLA_KERNEL_SOURCE.contains("grim_gla_state_update_output"));
        // One kernel fuses update + output + gated norm (no split entry points).
        assert!(!GLA_KERNEL_SOURCE.contains("grim_gla_state_update_v1"));
        // M7 gate: no grid-wide sync primitive. Intra-block __syncthreads
        // is the only barrier (permitted: block scope, not grid scope).
        assert!(!GLA_KERNEL_SOURCE.contains("cooperative_groups"));
        assert!(!GLA_KERNEL_SOURCE.contains("grid.sync"));
        assert!(!GLA_KERNEL_SOURCE.contains("this_grid"));
        // Wave32 discipline: no __shfl* (dodges the 64-bit mask pitfall).
        assert!(!GLA_KERNEL_SOURCE.contains("__shfl"));
        // Launch bounds match the 64-thread (2-wave) design.
        assert!(GLA_KERNEL_SOURCE.contains("__launch_bounds__(64)"));
    }

    #[test]
    fn env_gates_default_off_fused_default_on() {
        let keep_fp8 = std::env::var("GRIM_GLA_STATE_FP8").ok();
        let keep_fused = std::env::var("GRIM_GLA_FUSED").ok();
        unsafe {
            std::env::remove_var("GRIM_GLA_STATE_FP8");
            std::env::remove_var("GRIM_GLA_FUSED");
        }
        assert!(!gla_state_fp8_enabled(), "FP8 state must default OFF");
        assert!(gla_fused_enabled(), "fused path must default ON");
        unsafe {
            std::env::set_var("GRIM_GLA_STATE_FP8", "1");
            std::env::set_var("GRIM_GLA_FUSED", "0");
        }
        assert!(gla_state_fp8_enabled());
        assert!(!gla_fused_enabled());
        unsafe {
            std::env::remove_var("GRIM_GLA_STATE_FP8");
            std::env::remove_var("GRIM_GLA_FUSED");
            if let Some(v) = keep_fp8 {
                std::env::set_var("GRIM_GLA_STATE_FP8", v);
            }
            if let Some(v) = keep_fused {
                std::env::set_var("GRIM_GLA_FUSED", v);
            }
        }
    }
}
