//! `grim-backend-cpu` — the always-available reference backend for Grim.
//!
//! Implements `BackendDevice` and `BackendStorage` over a host `Vec<f32>`
//! buffer. Operations are synchronous; the returned `ComputeHandle` is
//! always `ReadyHandle`. v1 is scalar + naive GEMM — SIMD specialization
//! (packed_simd / std::simd) is a focus area for a later phase but is
//! structurally isolated so swapping in doesn't require changing model
//! code.

pub mod device;
pub mod storage;
/// CPU implementations of the strict-mode mathematical primitives
/// referenced by [`grim_core::DeterminismMode::Strict`]. Architecture
/// §5.8.
pub mod strict_kernels;
pub mod deterministic_rng;
/// SIMD-accelerated GEMM kernel (AVX2/SSE on x86_64)
pub mod simd_gemm;

pub use device::{cpu_tensor, CpuDevice};
pub use deterministic_rng::DeterministicRng;
pub use storage::CpuStorage;
pub use simd_gemm::{gemm_f32_simd, gemm_f32_lora_fused};
