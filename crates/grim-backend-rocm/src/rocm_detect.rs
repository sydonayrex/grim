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

/// Auto-configure `HSA_OVERRIDE_GFX_VERSION` for consumer GPUs / APUs
/// if the variable is not explicitly set in the environment.
pub fn auto_configure_hsa_override(detected_arch: &str) -> Option<&'static str> {
    if std::env::var("HSA_OVERRIDE_GFX_VERSION").is_ok() {
        return None;
    }
    let override_ver = match detected_arch {
        "gfx1031" | "gfx1032" | "gfx1033" | "gfx1034" | "gfx1035" | "gfx1036" => Some("10.3.0"),
        "gfx1101" | "gfx1102" | "gfx1103" => Some("11.0.0"),
        "gfx1150" | "gfx1151" | "gfx1152" => Some("11.5.0"),
        _ => None,
    };
    if let Some(ver) = override_ver {
        unsafe {
            std::env::set_var("HSA_OVERRIDE_GFX_VERSION", ver);
        }
    }
    override_ver
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_configure_hsa_override_known_consumer_targets() {
        unsafe {
            std::env::remove_var("HSA_OVERRIDE_GFX_VERSION");
        }
        assert_eq!(auto_configure_hsa_override("gfx1036"), Some("10.3.0"));
        // Reset HSA_OVERRIDE_GFX_VERSION so the second call sees "not already set".
        unsafe {
            std::env::remove_var("HSA_OVERRIDE_GFX_VERSION");
        }
        assert_eq!(auto_configure_hsa_override("gfx1103"), Some("11.0.0"));
        unsafe {
            std::env::remove_var("HSA_OVERRIDE_GFX_VERSION");
        }
        assert_eq!(auto_configure_hsa_override("gfx1150"), Some("11.5.0"));
        unsafe {
            std::env::remove_var("HSA_OVERRIDE_GFX_VERSION");
        }
        assert_eq!(auto_configure_hsa_override("gfx90a"), None);
    }
}
