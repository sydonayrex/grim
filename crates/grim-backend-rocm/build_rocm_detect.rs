// Shared RCCL/ROCm lib-dir discovery.
//
// Single source of truth, `include!`-ed by both `build.rs` (build crate,
// no crate deps) and `src/rocm_detect.rs` (the lib, for unit tests).
// Keeping the pure logic here means the build script and the testable lib
// cannot drift — no duplicated knowledge (clean-code imperative 11).
//
// Priority order for locating `librccl.so*`:
//   1. `RCCL_PATH`            (RCCL-specific lib dir)
//   2. `ROCM_RCCL_PATH`      (alias)
//   3. `ROCM_PATH` + `/lib`
//   4. `/opt/rocm/lib`
//   5. `/usr/lib/rocm/lib`
//   6. workspace `.rocm-2/lib`, `.rocm-3/lib`, `.rocm-4/lib`
//
// Returns the first dir that actually contains `librccl.so`,
// `librccl.so.1`, or `librccl.so.1.0`.

use std::path::PathBuf;

const RCCL_SO_CANDIDATES: &[&str] = &[
    "librccl.so",
    "librccl.so.1",
    "librccl.so.1.0",
];

/// Candidate ROCm lib-dir roots in priority order. `workspace_root` is the
/// cargo manifest dir's grandparent (crates/grim-backend-rocm -> workspace).
fn candidate_lib_dirs(workspace_root: &std::path::Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let push_env = |dirs: &mut Vec<PathBuf>, key: &str| {
        if let Ok(val) = std::env::var(key) {
            let p = PathBuf::from(val);
            if p.is_dir() {
                dirs.push(p);
            }
        }
    };
    push_env(&mut dirs, "RCCL_PATH");
    push_env(&mut dirs, "ROCM_RCCL_PATH");
    if let Ok(val) = std::env::var("ROCM_PATH") {
        let p = PathBuf::from(val).join("lib");
        if p.is_dir() {
            dirs.push(p);
        }
    }
    for fixed in ["/opt/rocm/lib", "/usr/lib/rocm/lib"] {
        let p = PathBuf::from(fixed);
        if p.is_dir() {
            dirs.push(p);
        }
    }
    for rocm_n in ["rocm-2", "rocm-3", "rocm-4"] {
        let p = workspace_root.join(rocm_n).join("lib");
        if p.is_dir() {
            dirs.push(p);
        }
    }
    dirs
}

/// Resolve the directory containing `librccl.so*`, or `None` if RCCL is
/// not installed at any known location. Never panics.
pub fn resolve_rocm_lib_dir(workspace_root: &std::path::Path) -> Option<PathBuf> {
    for dir in candidate_lib_dirs(workspace_root) {
        for so in RCCL_SO_CANDIDATES {
            if dir.join(so).exists() {
                return Some(dir);
            }
        }
    }
    None
}
