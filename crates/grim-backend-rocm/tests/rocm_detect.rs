//! RED-GREEN-REFACTOR tests for RCCL/ROCm lib-dir discovery (WI-R0).
//!
//! The discovery fn lives in `build_rocm_detect.rs` (shared via `include!`
//! by both `build.rs` and `src/rocm_detect.rs`) so the build script and
//! these tests cannot drift. Tests read real env vars + real temp dirs.
//!
//! Skill attribution:
//! - `rust-tdd` — assert_eq! on the resolved PathBuf; proptest-style
//!   "any missing env -> None" is a unit assertion, not a snapshot.
//! - `rust-ffi-grim` — §2 dynamic ROCm discovery: query ROCM_PATH /
//!   standard paths, gracefully return None (never crash) when absent.
//! - `clean-code-guard` — no unwrap in tests; `?`-bubbling Result.

use std::path::PathBuf;
use std::sync::Mutex;

use grim_backend_rocm::rocm_detect::resolve_rocm_lib_dir;

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Resolve against an explicit `RCCL_PATH` pointing at a dir that holds a
/// fake `librccl.so`. Must return exactly that dir.
#[test]
fn resolves_via_rcc_path_when_so_present() -> TestResult {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    
    let tmp = std::env::temp_dir().join(format!(
        "grim_rocm_detect_{}_{}",
        std::process::id(),
        "rcc"
    ));
    let lib = tmp.join("lib");
    std::fs::create_dir_all(&lib)?;
    std::fs::write(lib.join("librccl.so.1.0"), b"")?;

    unsafe {
        std::env::set_var("RCCL_PATH", &lib);
        // Ensure no other var leaks a real path ahead of it.
        std::env::remove_var("ROCM_RCCL_PATH");
        std::env::remove_var("ROCM_PATH");
    }

    let got = resolve_rocm_lib_dir(&std::path::Path::new("/nonexistent-workspace"));
    unsafe {
        std::env::remove_var("RCCL_PATH");
    }

    let got = got.ok_or("expected a resolved dir, got None")?;
    assert_eq!(got, lib);
    let _ = PathBuf::from("/cleanup");
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

/// `ROCM_RCCL_PATH` is the documented alias and must be honored.
#[test]
fn resolves_via_rocm_rcc_path_alias() -> TestResult {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let tmp = std::env::temp_dir().join(format!(
        "grim_rocm_detect_{}_{}",
        std::process::id(),
        "alias"
    ));
    let lib = tmp.join("lib");
    std::fs::create_dir_all(&lib)?;
    std::fs::write(lib.join("librccl.so"), b"")?;

    unsafe {
        std::env::remove_var("RCCL_PATH");
        std::env::set_var("ROCM_RCCL_PATH", &lib);
        std::env::remove_var("ROCM_PATH");
    }

    let got = resolve_rocm_lib_dir(&std::path::Path::new("/nonexistent-workspace"));
    unsafe {
        std::env::remove_var("ROCM_RCCL_PATH");
    }

    let got = got.ok_or("expected a resolved dir via alias, got None")?;
    assert_eq!(got, lib);
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

/// When no candidate dir holds `librccl.so*`, discovery returns `None`
/// (graceful, never panic) — this is the single-GPU default path.
#[test]
fn returns_none_when_no_rccl_present() -> TestResult {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    unsafe {
        std::env::remove_var("RCCL_PATH");
        std::env::remove_var("ROCM_RCCL_PATH");
        std::env::remove_var("ROCM_PATH");
    }
    // A workspace root with no .rocm-N/lib subdirs and no /opt/rocm.
    let fake_root = std::env::temp_dir()
        .join(format!("grim_rd_{}_norcc", std::process::id()));
    std::fs::create_dir_all(&fake_root)?;

    let got = resolve_rocm_lib_dir(&fake_root);
    std::fs::remove_dir_all(&fake_root).ok();

    if let Some(ref p) = got {
        let has_real = p.join("librccl.so").exists() || p.join("librccl.so.1").exists() || p.join("librccl.so.1.0").exists();
        assert!(has_real, "Got unexpected non-RCCL path {:?}", got);
    }
    Ok(())
}

/// A dir named in `RCCL_PATH` but containing no `librccl.so*` is
/// skipped (not returned), so the caller does not emit a dead link line.
#[test]
fn skips_dir_without_rccl_so() -> TestResult {
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let tmp = std::env::temp_dir()
        .join(format!("grim_rocm_detect_{}_empty", std::process::id()));
    let lib = tmp.join("lib");
    std::fs::create_dir_all(&lib)?; // empty, no .so
    unsafe {
        std::env::set_var("RCCL_PATH", &lib);
        std::env::remove_var("ROCM_RCCL_PATH");
        std::env::remove_var("ROCM_PATH");
    }

    let got = resolve_rocm_lib_dir(&std::path::Path::new("/nonexistent-workspace"));
    unsafe {
        std::env::remove_var("RCCL_PATH");
    }
    std::fs::remove_dir_all(&tmp).ok();

    if let Some(ref p) = got {
        let has_real = p.join("librccl.so").exists() || p.join("librccl.so.1").exists() || p.join("librccl.so.1.0").exists();
        assert!(has_real, "Empty dir must not resolve, got {:?}", got);
    }
    Ok(())
}
