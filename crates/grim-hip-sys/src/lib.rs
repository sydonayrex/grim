//! Vendored HIP FFI bindings for grim-backend-rocm.
//!
//! Provenance: tracel-ai/cubecl-hip (cubecl-hip-sys 7.14.6085001), MIT OR
//! Apache-2.0 — see LICENSE-APACHE / LICENSE-MIT and PROVENANCE.md.
//! Adapted for grim: static-link default (build.rs links libamdhip64 +
//! libhiprtc via grim's own discovery), runtime dlopen behind the `dynamic`
//! feature, and the committed bindings file is selected by the installed
//! HIP patch (`hipconfig --version`).
//!
//! Raw bindings only — no safe layer (same posture as upstream). Every call
//! in grim-backend-rocm runs under a `DeviceGuard`; do not call these fns
//! directly from outside a pinned device context.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_transmute)]
#![allow(improper_ctypes)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_variables)]

pub mod hipconfig;
pub use hipconfig::*;

#[cfg(feature = "dynamic")]
mod dynamic;
#[cfg(feature = "dynamic")]
pub use dynamic::is_available;
#[cfg(not(feature = "dynamic"))]
/// Static arm: the loader resolved libamdhip64 at process start, so if this
/// crate is running at all the HIP runtime is available.
pub fn is_available() -> bool {
    true
}

mod bindings;
#[allow(unused)]
pub use bindings::*;
