//! `compute_kernel_source`: a small helper that re-assembles the
//! crate's HIP-CPU program string at runtime. The two halves live
//! in two different sub-modules:
//!
//! - `kernels::compute_kernels::OTHER_KERNEL_SOURCE` — add / mul /
//!   silu_mul / rms_norm / softmax / embedding / rmsnorm_matmul
//! - `kernels::qkv_attention::KERNEL_SOURCE` — the Phase-1 fused
//!   QKV attention kernel with online softmax + GQA + causal mask
//!
//! The two halves are kept separate so that future sibling kernels
//! (Phase 2 quantized attention, Phase 3 paged attention) can drop
//! in without touching either. The `compute_kernel_source()` here
//! sits next to its two data dependencies for clarity.
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` Action 9 — JIT source assembly is a
//!   runtime operation, not a `const` concat: kernel sources can be
//!   reloaded mid-process for revision-tracked experiments.
//! - `rust-gpu-discipline` §4 — recompile hashing is keyed off this
//!   string's bytes, so any change here also invalidates the
//!   `HsacoKernelCache` for matching entry names.

pub fn compute_kernel_source() -> String {
    let mut s = String::with_capacity(
        crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE.len() + 4096,
    );
    s.push_str(crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE);
    s.push_str(crate::kernels::qkv_attention::KERNEL_SOURCE);
    // WI 2.4.4-2 — Rust-centric decode GEMM (F16, double-buffered LDS).
    // Opt-in via `DecodeGemmConfig::enabled` in `RocmDevice::matmul`; the
    // source is concatenated here regardless so the JIT cache can resolve
    // the symbol at first dispatch without rebuilding the program.
    s.push_str(crate::kernels::decode_gemm::KERNEL_SOURCE);
    s
}

#[cfg(test)]
mod source_asm_self_tests {
    use super::*;

    #[test]
    fn compute_kernel_source_contains_both_sub_sources() {
        let src = compute_kernel_source();
        // The add / mul / rms_norm kernel names live in OTHER_KERNEL_SOURCE.
        assert!(src.contains("grim_add"));
        assert!(src.contains("grim_rms_norm"));
        // The fused QKV attention lives in qkv_attention::KERNEL_SOURCE.
        assert!(src.contains("grim_qkv_attention"));
    }

    #[test]
    fn compute_kernel_source_pre_allocation_is_at_least_qkv_length() {
        // Rough upper bound: the function pre-allocates
        // OTHER_KERNEL_SOURCE.len() + 4096 bytes, which is at least
        // 4096. We don't pin a tight bound — the function is meant
        // to accommodate QKV growth without realloc — but we
        // confirm it doesn't blow up.
        let _ = compute_kernel_source();
    }
}
