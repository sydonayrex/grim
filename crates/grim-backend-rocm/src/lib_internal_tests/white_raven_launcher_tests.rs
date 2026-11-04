//! note: string-only gate retained because grim_wmma_gemm_fp8_e4m3 has no exported launcher; upgrade to launch-and-compare when launcher lands

use crate::kernels::wmma_fp8_gemm::KERNEL_SOURCE;

#[test]
fn white_raven_kernel_names_wmma_gemm_fp8_entry() {
    assert!(KERNEL_SOURCE.contains("grim_wmma_gemm_fp8_e4m3"));
}
