//! RCCL/ROCm lib-dir and include-dir discovery, exposed for unit tests.
//!
//! The implementation lives in `build_rocm_detect.rs` (a `include!`-shared
//! source so `build.rs` and this module cannot drift). See that file for the
//! priority order and the candidate `librccl.so*` names.
//!
//! This module adds the runtime wrappers that `device::util` (and the JIT
//! kernel path) need to discover the ROCm include tree at runtime, so that
//! hipRTC-compiled kernels can `#include <rocwmma/rocwmma.hpp>` and friends.

// `PathBuf` comes from the `include!`'d build_rocm_detect.rs — don't re-import.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_rocm_detect.rs"));

/// Compute the workspace root from the crate manifest dir.
/// Mirrors `build.rs`'s `workspace_root` — the runtime equivalent.
fn workspace_root() -> PathBuf {
    let manifest =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest.clone())
}

/// Discover the ROCm include directory at runtime.
///
/// HIPRTC (used by `device::util::hiprtc_options_for_arch`) does not
/// automatically search the ROCm include tree. This resolves the directory
/// containing `rocwmma/` etc. so we can inject `-I<dir>` into JIT compile
/// options. Returns `None` if no candidate directory is found — callers
/// that need ROCm headers will then surface a clean compile error.
pub fn rocm_include_dir() -> Option<PathBuf> {
    resolve_rocm_include_dir(&workspace_root())
}
