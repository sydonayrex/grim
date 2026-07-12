//! ROCm device module — the `[device]` grouping the spec's anti-pattern
//! request asks for. Holds:
//!
//! - [`gemm_tuning`] — `lookup_gemm_config` + `lookup_solution_index`:
//!   per-shape GEMM tile selection and offline-tuned rocBLAS solution
//!   indices (Item 7).
//!
//! The `RocmDevice` struct + its impls (`impl RocmDevice`,
//! `impl Drop`, `impl BackendDevice for RocmDevice`) still live in
//! `lib.rs` for now — extracting them is the next modularization
//! step but requires also relocating `RocmStorage::alloc_gpu` /
//! `copy_from_host` (free functions in `lib.rs` that take
//! `&RocmDevice`) so the type-mod trait bounds stay in one place.
//!
//! Skill attribution: see spec §"Skills map" — `rust-ai-ml-inference-guide`
//! Action 1 (host device module), `rocm-hip-kernels` per-`gfx_arch`
//! dispatch table.

pub mod gemm_tuning;
pub mod helpers;
