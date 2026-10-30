//! Compat shim: re-exports the entire `grim-hip-sys` surface under the
//! `cubecl-hip-sys` name so the upstream `cubecl-hip` crate (which imports
//! `cubecl_hip_sys::…` paths) compiles against the vendored ROCm 7.16
//! bindings. See `grim-hip-sys/PROVENANCE.md`.
pub use grim_hip_sys::*;
