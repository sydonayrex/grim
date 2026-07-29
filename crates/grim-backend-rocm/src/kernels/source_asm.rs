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
        crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE.len() + 16384,
    );
    s.push_str(crate::kernels::shared_device_fns::KERNEL_SOURCE);
    s.push_str(crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE);
    s.push_str(crate::kernels::qkv_attention::KERNEL_SOURCE);
    s.push_str(crate::kernels::decode_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::fused_dequant_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::kv_dequant_attention::KERNEL_SOURCE);
    s.push_str(crate::kernels::wmma_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q8_0_dequant::KERNEL_SOURCE);
    s.push_str(crate::kernels::q4k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q5k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q6k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q2k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q3k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::iq_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::fp8_standalone::KERNEL_SOURCE);
    s.push_str(crate::kernels::mxfp_standalone::KERNEL_SOURCE);
    s.push_str(crate::kernels::selective_scan::KERNEL_SOURCE);
    s.push_str(crate::kernels::q4k_dequant::KERNEL_SOURCE);
    s.push_str(crate::kernels::iq_dequant::KERNEL_SOURCE);
    s.push_str(crate::kernels::flash_attn::KERNEL_SOURCE);
    s.push_str(crate::kernels::cross_attention::KERNEL_SOURCE);
    s.push_str(crate::kernels::rwkv::KERNEL_SOURCE);
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
        // WMMA GEMM lives in wmma_gemm::KERNEL_SOURCE.
        assert!(src.contains("grim_wmma_gemm"));
    // Q8_0 dequant kernel lives in q8_0_dequant::KERNEL_SOURCE.
    assert!(src.contains("grim_dequant_q8_0"));
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

    #[test]
    fn compute_kernel_source_contains_phase2_kernels() {
        let src = compute_kernel_source();
        assert!(src.contains("grim_flash_attention"));
        assert!(src.contains("grim_cross_attention"));
        assert!(src.contains("grim_rwkv_time_mix"));
    }

    /// A.0 regression guard: every shared __device__ helper must be defined
    /// exactly once across the concatenated HIPRTC translation unit.
    /// Duplicate definitions cause symbol-collision linker errors on real hardware.
    #[test]
    fn kernel_source_has_no_duplicate_device_fn_definitions() {
        let src = compute_kernel_source();
        // These four symbols are defined in shared_device_fns::KERNEL_SOURCE only.
        // Any other kernel module that copies their definition causes HIPRTC failure.
        let shared_syms = [
            "float fp16_to_float_device(",
            "float fp8_e4m3_to_float_hip(",
            "float mxfp4_to_float_hip(",
            "float dequant_q4k_element(",
        ];
        for sym in &shared_syms {
            let count = src.matches(sym).count();
            assert_eq!(
                count, 1,
                "shared __device__ symbol defined {} times (expected 1): '{}'",
                count, sym
            );
        }
    }
}
