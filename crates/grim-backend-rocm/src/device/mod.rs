//! ROCm device module — the `[device]` grouping the spec's anti-pattern
//! request asks for. Holds:
//!
//! - [`gemm_tuning`] — `lookup_gemm_config` + `lookup_solution_index`:
//!   per-shape GEMM tile selection and offline-tuned rocBLAS solution
//!   indices (Item 7).
//! - [`helpers`] — module-level helpers used by `RocmDevice` impl
//!   methods (`jit_compile_hsaco`, `memcpy_with_xnack_fallback`,
//!   `upload_device_buffer`).
//! - [`layout`] — KV layout, weight layout, wavefront-tiled layout,
//!   alignment helpers.
//!
//! The `RocmDevice` struct + its impls (`impl RocmDevice`,
//! `impl Drop`, `impl BackendDevice for RocmDevice`) still live in
//! `lib.rs` for now — extracting them is the next modularization
//! step but requires also relocating the impl-method helpers (free
//! functions in `lib.rs` that take `&RocmDevice`) so all the
//! impl-bound types live together in one module.
//!
//! Skill attribution: see spec §"Skills map" — `rust-ai-ml-inference-guide`
//! Action 1 (host device module), `rocm-hip-kernels` per-`gfx_arch`
//! dispatch table.

pub mod gemm_tuning;
pub mod handles;
pub mod helpers;
pub mod layout;
pub mod util;
