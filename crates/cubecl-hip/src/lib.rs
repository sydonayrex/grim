//! Compat shim: re-exports the entire `grim-cl-hip` surface under the
//! `cubecl-hip` name so the upstream `cubecl` umbrella (`pub use cubecl_hip as hip`)
//! and `grim-backend-rocm` compile against the vendored 0.10.0 source.
//! See `grim-cl-hip/PROVENANCE.md`.
pub use grim_cl_hip::*;
