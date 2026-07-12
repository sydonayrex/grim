//! Tests for GEMM algorithm selection under the RED-GREEN-REFACTOR cycle.

use grim_backend_rocm::device::rocblas::{rocblas_gemm_algo, select_gemm_algo};
use grim_tensor::Result;

#[test]
fn test_select_gemm_algo_standard() -> Result<()> {
    let algo = select_gemm_algo(0);
    assert_eq!(algo, rocblas_gemm_algo::standard);
    Ok(())
}

#[test]
fn test_select_gemm_algo_solution_index() -> Result<()> {
    let algo = select_gemm_algo(42);
    assert_eq!(algo, rocblas_gemm_algo::solution_index);
    Ok(())
}

#[test]
fn test_solution_index_for_f16_bf16() -> Result<()> {
    use grim_backend_rocm::lookup_solution_index;
    use grim_tensor::ArithType;

    let f16_idx = lookup_solution_index(1, 4096, 4096, ArithType::F16);
    let bf16_idx = lookup_solution_index(1, 4096, 4096, ArithType::BF16);

    assert_ne!(f16_idx, 0, "F16 must return a tuned solution index");
    assert_ne!(bf16_idx, 0, "BF16 must return a tuned solution index");
    Ok(())
}

