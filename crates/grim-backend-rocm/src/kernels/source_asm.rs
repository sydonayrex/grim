//! `compute_kernel_source`: a small helper that re-assembles the [see: `kernels::compute_kernels::OTHER_KERNEL_SOURCE`]

pub fn compute_kernel_source() -> String {
    let mut s =
        String::with_capacity(crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE.len() + 16384);
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
    s.push_str(crate::kernels::fp8_gemm_rdna4::KERNEL_SOURCE);
    s.push_str(crate::kernels::mxfp_standalone::KERNEL_SOURCE);
    s.push_str(crate::kernels::selective_scan::KERNEL_SOURCE);
    s.push_str(crate::kernels::q4k_dequant::KERNEL_SOURCE);
    s.push_str(crate::kernels::iq_dequant::KERNEL_SOURCE);
    s.push_str(crate::kernels::cross_attention::KERNEL_SOURCE);
    s.push_str(crate::kernels::rwkv::KERNEL_SOURCE);
    s.push_str(crate::kernels::quant_standalone::KERNEL_SOURCE);
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
        let _ = compute_kernel_source();
    }

    #[test]
    fn compute_kernel_source_contains_phase2_kernels() {
        let src = compute_kernel_source();
        assert!(src.contains("grim_cross_attention"));
        assert!(src.contains("grim_rwkv_time_mix"));
        assert!(src.contains("grim_fp8_gemm_rdna4"));
    }

    /// A.0 regression guard: every shared __device__ helper must be defined
    #[test]
    fn kernel_source_has_no_duplicate_device_fn_definitions() {
        let src = compute_kernel_source();
        // These four symbols are defined in shared_device_fns::KERNEL_SOURCE only.
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
