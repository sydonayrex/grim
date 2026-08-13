//! CPU reference backend: host buffers, SIMD GEMM, scalar fallback routines.

pub mod cache;
pub mod dequant_gemm;
pub mod deterministic_rng;
pub mod device;
pub mod graph_capture;
pub mod hardware_spec;
pub mod moe_dispatch;
/// SIMD GEMM kernel (AVX2/SSE on x86_64).
pub mod simd_gemm;
pub mod storage;
/// Strict-mode primitives for [`grim_core::DeterminismMode::Strict`] (§5.8).
pub mod strict_kernels;
pub mod topology;

pub use cache::CpuCacheKey;
pub use dequant_gemm::dequant_row;
pub use deterministic_rng::DeterministicRng;
pub use device::{CpuDevice, add_tensors, cpu_tensor};
pub use graph_capture::{CpuCapturedGraph, CpuGraphRegistry};
pub use hardware_spec::CpuHardwareSpec;
pub use moe_dispatch::moe_fused_dispatch;
pub use simd_gemm::{gemm_f32_lora_fused, gemm_f32_simd};
pub use storage::CpuStorage;
pub use topology::CpuNumaTopology;
