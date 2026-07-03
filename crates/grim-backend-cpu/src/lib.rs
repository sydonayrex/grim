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

pub use device::{cpu_tensor, CpuDevice};
pub use storage::CpuStorage;
