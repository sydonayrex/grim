//! Internal `lib.rs` unit tests, split by concern (Phase 4 of
//! plans/PLAN-monolithic-refactor-execution.md). Each submodule is `#[cfg(test)]`.

#[cfg(test)]
mod device_lifecycle_tests;
#[cfg(test)]
mod layout_tests;
#[cfg(test)]
mod attention_precision_tests;
#[cfg(test)]
mod common;
#[cfg(test)]
mod quant_kernel_compile_tests;
#[cfg(test)]
mod elementwise_norm_tests;
#[cfg(test)]
mod gemm_matmul_tests;
#[cfg(test)]
mod memory_cache_tests;
#[cfg(test)]
mod graph_capture_tests;
#[cfg(test)]
mod native_epilogue_tests;
