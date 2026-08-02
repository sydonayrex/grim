//! CPU reference backend: host buffers, SIMD GEMM, scalar fallback routines.

pub mod dequant_gemm;
pub mod deterministic_rng;
pub mod device;
/// SIMD GEMM kernel (AVX2/SSE on x86_64).
pub mod simd_gemm;
pub mod storage;
/// Strict-mode primitives for [`grim_core::DeterminismMode::Strict`] (§5.8).
pub mod strict_kernels;

pub use dequant_gemm::dequant_row;
pub use deterministic_rng::DeterministicRng;
pub use device::{CpuDevice, add_tensors, cpu_tensor};
pub use simd_gemm::{gemm_f32_lora_fused, gemm_f32_simd};
pub use storage::CpuStorage;
