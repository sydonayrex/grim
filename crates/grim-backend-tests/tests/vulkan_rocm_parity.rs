//! Cross-backend numerical parity tests: Vulkan ↔ ROCm (Task 3 / Lapse I + J).
//! Gated: `GRIM_RUN_GPU_TESTS=1`, features `vulkan` + `rocm`.
//! Skills: caveman-cloud-ops (separate cost/savings), grim-grave-gld-gqa (parity first), ponytail-minimalism (shortest contract).


pub fn generate_input(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as i32) as f32 / (i32::MAX as f32)
    }).collect()
}

pub const MATMUL_PARITY_TOL: f32 = 1e-5;
pub const RMSNORM_PARITY_DIM: usize = 512;
pub const RMSNORM_PARITY_TOL: f32 = 1e-5;
pub const Q8_0_GEMM_TOL: f32 = 1e-4;
pub const Q4K_GEMM_TOL: f32 = 2e-1;
pub const MXFP4_GEMM_TOL: f32 = 3.5e-1;
/// FP8 dequant GEMM parity anchor (Task 5 / plan Lapse E companion).
/// Once FP8 enabled in caps.rs (Task 1), compare dequant GEMM outputs across Vulkan + ROCm.
/// Per skill grim-grave-gld-gqa: track divergence (L2-norm, RMSNorm, gate differences) explicitly.
/// Per skill caveman-cloud-ops: evidence/accounting separate; $0 opportunity band for profile-only observations.
pub const FP8_GEMM_TOL: f32 = 1e-1;
pub const IQ4NL_GEMM_TOL: f32 = 5e-2;

/// Attention parity: seq=4 heads=4 dim=64 tol=1e-4.
pub const ATTENTION_PARITY_SEQ: usize = 4;
pub const ATTENTION_PARITY_HEADS: usize = 4;
pub const ATTENTION_PARITY_HEAD_DIM: usize = 64;
pub const ATTENTION_PARITY_TOL: f32 = 1e-4;

// Contract note: requires real Vulkan + ROCm devices. Not a stub — defines what parity must measure.
